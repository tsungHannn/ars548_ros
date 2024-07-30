from torch import distributed as dist
import numpy as np
import cv2
import math
import tempfile
import os
import pyquaternion
from pyquaternion import Quaternion
import mmcv
from nuscenes.utils.data_classes import Box

DefaultAttribute = {
        'car': 'vehicle.parked',
        'pedestrian': 'pedestrian.moving',
        'trailer': 'vehicle.parked',
        'truck': 'vehicle.parked',
        'bus': 'vehicle.moving',
        'motorcycle': 'cycle.without_rider',
        'construction_vehicle': 'vehicle.parked',
        'bicycle': 'cycle.without_rider',
        'barrier': '',
        'traffic_cone': '',
    }
modality=dict(use_lidar=False,
                      use_camera=True,
                      use_radar=False,
                      use_map=False,
                      use_external=False),

def synchronize():
    """Helper function to synchronize (barrier)
        among all processes when using distributed training"""
    if not dist.is_available():
        return
    if not dist.is_initialized():
        return
    current_world_size = dist.get_world_size()
    if current_world_size == 1:
        return
    dist.barrier()

def get_world_size() -> int:
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()

def all_gather_object(obj):
    world_size = get_world_size()
    if world_size < 2:
        return [obj]
    output = [None for _ in range(world_size)]
    dist.all_gather_object(output, obj)
    return output

def get_rank() -> int:
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


category_map_rope3d = {"car": "Car", "van": "Car", "truck": "Bus", "bus": "Bus", "pedestrian": "Pedestrian", "bicycle": "Cyclist", "trailer": "Cyclist", "motorcycle": "Cyclist"}

def get_cam2lidar():
    denorm = np.array([0.0001480018, -0.9738427, -0.2272234, 7.07289743423])
    Rx = np.array([[1.0, 0.0, 0.0], 
                   [0.0, 0.0, 1.0], 
                   [0.0, -1.0, 0.0]])
    
    Rz = np.array([[0.0, 1.0, 0.0], 
                   [-1.0, 0.0, 0.0],  
                   [0.0, 0.0, 1.0]])
    
    origin_vector = np.array([0, 1, 0])
    target_vector = -1 * np.array([denorm[0], denorm[1], denorm[2]])
    target_vector_norm = target_vector / np.sqrt(target_vector[0]**2 + target_vector[1]**2 + target_vector[2]**2)       
    sita = math.acos(np.inner(target_vector_norm, origin_vector))
    n_vector = np.cross(target_vector_norm, origin_vector) 
    n_vector = n_vector / np.sqrt(n_vector[0]**2 + n_vector[1]**2 + n_vector[2]**2)
    n_vector = n_vector.astype(np.float32)
    cam2lidar, _ = cv2.Rodrigues(n_vector * sita)
    cam2lidar = cam2lidar.astype(np.float32)
    cam2lidar = np.matmul(Rx, cam2lidar)
    cam2lidar = np.matmul(Rz, cam2lidar)
    
    Ax, By, Cz, D = denorm[0], denorm[1], denorm[2], denorm[3]
    mod_area = np.sqrt(np.sum(np.square([Ax, By, Cz])))
    d = abs(D) / mod_area
    Tr_cam2lidar = np.eye(4)
    Tr_cam2lidar[:3, :3] = cam2lidar
    Tr_cam2lidar[:3, 3] = [0, 0, d]
    
    translation = [0, 0, d]
    return cam2lidar, translation, Tr_cam2lidar, denorm

def get_velo2cam():
    _, _, Tr_cam2lidar, _ = get_cam2lidar()
    Tr_velo_to_cam = np.linalg.inv(Tr_cam2lidar) 
    r_velo2cam, t_velo2cam = Tr_velo_to_cam[:3, :3], Tr_velo_to_cam[:3, 3]
    t_velo2cam = t_velo2cam.reshape(3, 1)
    return Tr_velo_to_cam, r_velo2cam, t_velo2cam

def normalize_angle(angle):
    alpha_tan = np.tan(angle)
    alpha_arctan = np.arctan(alpha_tan)
    if np.cos(angle) < 0:
        alpha_arctan = alpha_arctan + math.pi
    return alpha_arctan

def get_camera_3d_8points(obj_size, yaw_lidar, center_lidar, center_in_cam, r_velo2cam, t_velo2cam):
    liadr_r = np.matrix([[math.cos(yaw_lidar), -math.sin(yaw_lidar), 0], [math.sin(yaw_lidar), math.cos(yaw_lidar), 0], [0, 0, 1]])
    l, w, h = obj_size
    corners_3d_lidar = np.matrix(
        [
            [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2],
            [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2],
            [0, 0, 0, 0, h, h, h, h],
        ]
    )
    corners_3d_lidar = liadr_r * corners_3d_lidar + np.matrix(center_lidar).T
    corners_3d_cam = r_velo2cam * corners_3d_lidar + t_velo2cam
    
    x0, z0 = corners_3d_cam[0, 0], corners_3d_cam[2, 0]
    x3, z3 = corners_3d_cam[0, 3], corners_3d_cam[2, 3]
    dx, dz = x0 - x3, z0 - z3
    yaw = math.atan2(-dz, dx)
    alpha = yaw - math.atan2(center_in_cam[0], center_in_cam[2])
    if alpha > math.pi:
        alpha = alpha - 2.0 * math.pi
    if alpha <= (-1 * math.pi):
        alpha = alpha + 2.0 * math.pi
    alpha_arctan = normalize_angle(alpha)
    return alpha_arctan, yaw


def convert_point(point, matrix):
    pos =  matrix @ point
    return pos[0], pos[1], pos[2]


def get_lidar_3d_8points(obj_size, yaw_lidar, center_lidar):
    center_lidar = [float(center_lidar[0]), float(center_lidar[1]), float(center_lidar[2])]
    lidar_r = np.matrix([[math.cos(yaw_lidar), -math.sin(yaw_lidar), 0], [math.sin(yaw_lidar), math.cos(yaw_lidar), 0], [0, 0, 1]])
    l, w, h = obj_size
    center_lidar[2] = center_lidar[2] - h / 2
    corners_3d_lidar = np.matrix(
        [
            [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2],
            [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2],
            [0, 0, 0, 0, h, h, h, h],
        ]
    )
    corners_3d_lidar = lidar_r * corners_3d_lidar + np.matrix(center_lidar).T
    return corners_3d_lidar.T




def bbbox2bbox(box3d, Tr_velo_to_cam, camera_intrinsic, img_size=[1920, 1080]):
    corners_3d = np.array(box3d)
    corners_3d_extend = np.concatenate(
        [corners_3d, np.ones((corners_3d.shape[0], 1), dtype=np.float32)], axis=1) 
    corners_3d_extend = np.matmul(Tr_velo_to_cam, corners_3d_extend.transpose(1, 0))
        
    corners_2d = np.matmul(camera_intrinsic, corners_3d_extend)
    corners_2d = corners_2d[:2] / corners_2d[2]
    box2d = np.array([min(corners_2d[0]), min(corners_2d[1]),
                      max(corners_2d[0]), max(corners_2d[1])])
    
    # [xmin, ymin, xmax, ymax]
    box2d[0] = max(box2d[0], 0.0)
    box2d[1] = max(box2d[1], 0.0)
    box2d[2] = min(box2d[2], img_size[0])
    box2d[3] = min(box2d[3], img_size[1])
    return box2d


def _format_bbox(results, img_metas, jsonfile_prefix=None):
        # nusc_annos = {}
        mapped_class_names =  [
                                'car',
                                'truck',
                                'construction_vehicle',
                                'bus',
                                'trailer',
                                'barrier',
                                'motorcycle',
                                'bicycle',
                                'pedestrian',
                                'traffic_cone',
                            ]

        # print('Start to convert detection format...')
        # for sample_id, det in enumerate(mmcv.track_iter_progress(results)):
        for sample_id, det in enumerate(results):
            boxes, scores, labels = det
            boxes = boxes
            # sample_token = img_metas[sample_id]['token']
            trans = np.array(img_metas[sample_id]['ego2global_translation'])
            rot = Quaternion(img_metas[sample_id]['ego2global_rotation'])
            annos = list()
            for i, box in enumerate(boxes):
                name = mapped_class_names[labels[i]]
                center = box[:3]
                wlh = box[[4, 3, 5]]
                box_yaw = box[6]
                box_vel = box[7:].tolist()
                box_vel.append(0)
                quat = pyquaternion.Quaternion(axis=[0, 0, 1], radians=box_yaw)
                nusc_box = Box(center, wlh, quat, velocity=box_vel)
                nusc_box.rotate(rot)
                nusc_box.translate(trans)
                if np.sqrt(nusc_box.velocity[0]**2 +
                           nusc_box.velocity[1]**2) > 0.2:
                    if name in [
                            'car',
                            'construction_vehicle',
                            'bus',
                            'truck',
                            'trailer',
                    ]:
                        attr = 'vehicle.moving'
                    elif name in ['bicycle', 'motorcycle']:
                        attr = 'cycle.with_rider'
                    else:
                        attr = DefaultAttribute[name]
                else:
                    if name in ['pedestrian']:
                        attr = 'pedestrian.standing'
                    elif name in ['bus']:
                        attr = 'vehicle.stopped'
                    else:
                        attr = DefaultAttribute[name]
                nusc_anno = dict(
                    # sample_token=sample_token,
                    translation=nusc_box.center.tolist(),
                    size=nusc_box.wlh.tolist(),
                    rotation=nusc_box.orientation.elements.tolist(),
                    box_yaw=box_yaw,
                    velocity=nusc_box.velocity[:2],
                    detection_name=name,
                    detection_score=float(scores[i]),
                    attribute_name=attr,
                )
                annos.append(nusc_anno)
            # if sample_token in nusc_annos:
            #     nusc_annos[sample_token].extend(annos)
            # else:
            #     nusc_annos[sample_token] = annos
        # nusc_submissions = {
        #     'meta': modality,
        #     'results': nusc_annos,
        # }
        # mmcv.mkdir_or_exist(jsonfile_prefix)
        # res_path = os.path.join(jsonfile_prefix, 'results_nusc.json')
        # print('Results writes to', res_path)
        # mmcv.dump(nusc_submissions, res_path)

        return annos
        # return res_path

def format_results( results,
                    img_metas,
                    result_names=['img_bbox'],
                    jsonfile_prefix=None,
                    **kwargs):
    assert isinstance(results, list), 'results must be a list'

    if jsonfile_prefix is None:
        tmp_dir = tempfile.TemporaryDirectory()
        jsonfile_prefix = os.path.join(tmp_dir.name, 'results')
    else:
        tmp_dir = None
    result_files = dict()
    for rasult_name in result_names:
        if '2d' in rasult_name:
            continue
        # print(f'\nFormating bboxes of {rasult_name}')
        tmp_file_ = os.path.join(jsonfile_prefix, rasult_name)
        result_files.update({
            rasult_name:
            _format_bbox(results, img_metas, tmp_file_)
        })
    return result_files, tmp_dir


def compute_box_3d_camera(dim, location, rotation_y, denorm):
    c, s = np.cos(rotation_y), np.sin(rotation_y)
    R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)
    l, w, h = dim[2], dim[1], dim[0]
    x_corners = [l/2, l/2, -l/2, -l/2, l/2, l/2, -l/2, -l/2]
    y_corners = [0,0,0,0,-h,-h,-h,-h]
    z_corners = [w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2]

    corners = np.array([x_corners, y_corners, z_corners], dtype=np.float32)
    corners_3d = np.dot(R, corners) 

    denorm = denorm[:3]
    denorm_norm = denorm / np.sqrt(denorm[0]**2 + denorm[1]**2 + denorm[2]**2)
    ori_denorm = np.array([0.0, -1.0, 0.0])
    theta = -1 * math.acos(np.dot(denorm_norm, ori_denorm))
    n_vector = np.cross(denorm, ori_denorm)
    n_vector_norm = n_vector / np.sqrt(n_vector[0]**2 + n_vector[1]**2 + n_vector[2]**2)
    rotation_matrix, j = cv2.Rodrigues(theta * n_vector_norm)
    corners_3d = np.dot(rotation_matrix, corners_3d)
    corners_3d = corners_3d + np.array(location, dtype=np.float32).reshape(3, 1)
    return corners_3d.transpose(1, 0)



def project_to_image(pts_3d, P):
    pts_3d_homo = np.concatenate(
    [pts_3d, np.ones((pts_3d.shape[0], 1), dtype=np.float32)], axis=1)
    pts_2d = np.dot(P, pts_3d_homo.transpose(1, 0)).transpose(1, 0)
    pts_2d = pts_2d[:, :2] / pts_2d[:, 2:]
    return pts_2d


def draw_box_3d(image, corners, c=(0, 255, 0)):
    face_idx = [[0,1,5,4],[1,2,6,5],[2,3,7,6],[3,0,4,7]]
    for ind_f in [3, 2, 1, 0]:
        f = face_idx[ind_f]
        for j in [0, 1, 2, 3]:
            cv2.line(image, (int(corners[f[j], 0]), int(corners[f[j], 1])),
                    (int(corners[f[(j+1)%4], 0]), int(corners[f[(j+1)%4], 1])), c, 2, lineType=cv2.LINE_AA)
        if ind_f == 0:
            cv2.line(image, (int(corners[f[0], 0]), int(corners[f[0], 1])),
                    (int(corners[f[2], 0]), int(corners[f[2], 1])), c, 1, lineType=cv2.LINE_AA)
            cv2.line(image, (int(corners[f[1], 0]), int(corners[f[1], 1])),
                    (int(corners[f[3], 0]), int(corners[f[3], 1])), c, 1, lineType=cv2.LINE_AA)
    return image