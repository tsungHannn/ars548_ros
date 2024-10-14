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
import torch
import csv


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


category_map_rope3d = {"car": "Car", 
                       "van": "Car", 
                       "truck": "Bus", 
                       "bus": "Bus", 
                       "pedestrian": "Pedestrian", 
                       "bicycle": "Cyclist", 
                       "trailer": "Cyclist", 
                       "motorcycle": "Cyclist"}

def get_cam2lidar(denorm_file):
    denorm = load_denorm(denorm_file)
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

def get_velo2cam(denorm_file):
    _, _, Tr_cam2lidar, _ = get_cam2lidar(denorm_file=denorm_file)
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


def write_kitti_in_txt(pred_lines, path_txt):
    wf = open(path_txt, "w")
    for line in pred_lines:
        line_string = " ".join(line) + "\n"
        wf.write(line_string)
    wf.close()

# ========================================================================================
# ========================================================================================
# ========================================================================================
# 讀取 calib, denorm 等資訊


def load_calib(calib_file):
    with open(calib_file, 'r') as csv_file:
        reader = csv.reader(csv_file, delimiter=' ')
        for line, row in enumerate(reader):
            if row[0] == 'P2:':
                P2 = row[1:]
                P2 = [float(i) for i in P2]
                P2 = np.array(P2, dtype=np.float32).reshape(3, 4)
                continue

            # 我加的
            # if row[0] == 'Tr_velo_to_cam:':
            #     velo_to_cam = row[1:]
            #     velo_to_cam = [float(i) for i in velo_to_cam]
            #     velo_to_cam = np.array(velo_to_cam, dtype=np.float32).reshape(4, 4)
            #     continue

    return P2[:3,:3]


def load_denorm(denorm_file):
    with open(denorm_file, 'r') as f:
        lines = f.readlines()
    denorm = np.array([float(item) for item in lines[0].split(' ')])
    return denorm

def get_rot(h):
    return torch.Tensor([
        [np.cos(h), np.sin(h)],
        [-np.sin(h), np.cos(h)],
    ])

def img_transform(resize, resize_dims, crop, flip, rotate):
    ida_rot = torch.eye(2)
    ida_tran = torch.zeros(2)

    # post-homography transformation
    ida_rot *= resize
    ida_tran -= torch.Tensor(crop[:2])
    if flip:
        A = torch.Tensor([[-1, 0], [0, 1]])
        b = torch.Tensor([crop[2] - crop[0], 0])
        ida_rot = A.matmul(ida_rot)
        ida_tran = A.matmul(ida_tran) + b
    A = get_rot(rotate / 180 * np.pi)
    b = torch.Tensor([crop[2] - crop[0], crop[3] - crop[1]]) / 2
    b = A.matmul(-b) + b
    ida_rot = A.matmul(ida_rot)
    ida_tran = A.matmul(ida_tran) + b
    ida_mat = ida_rot.new_zeros(4, 4)
    ida_mat[3, 3] = 1
    ida_mat[2, 2] = 1
    ida_mat[:2, :2] = ida_rot
    ida_mat[:2, 3] = ida_tran
    return ida_mat

def sample_bda_augmentation():
    """Generate bda augmentation values based on bda_config."""
    rotate_bda = 0
    scale_bda = 1.0
    flip_dx = False
    flip_dy = False
    return rotate_bda, scale_bda, flip_dx, flip_dy
    
def sample_ida_augmentation(ida_aug_conf):
    """Generate ida augmentation values based on ida_config."""
    H, W = ida_aug_conf['H'], ida_aug_conf['W']
    fH, fW = ida_aug_conf['final_dim']
    resize = max(fH / H, fW / W)
    resize_dims = (int(W * resize), int(H * resize))
    newW, newH = resize_dims
    crop_h = int(
        (1 - np.mean(ida_aug_conf['bot_pct_lim'])) * newH) - fH
    crop_w = int(max(0, newW - fW) / 2)
    crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
    flip = False
    rotate_ida = 0
    return resize, resize_dims, crop, flip, rotate_ida
    
def equation_plane(points): 
    x1, y1, z1 = points[0, 0], points[0, 1], points[0, 2]
    x2, y2, z2 = points[1, 0], points[1, 1], points[1, 2]
    x3, y3, z3 = points[2, 0], points[2, 1], points[2, 2]
    a1 = x2 - x1
    b1 = y2 - y1
    c1 = z2 - z1
    a2 = x3 - x1
    b2 = y3 - y1
    c2 = z3 - z1
    a = b1 * c2 - b2 * c1
    b = a2 * c1 - a1 * c2
    c = a1 * b2 - b1 * a2
    d = (- a * x1 - b * y1 - c * z1)
    return np.array([a, b, c, d])

def get_denorm(sweepego2sweepsensor):
    ground_points_lidar = np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]])
    ground_points_lidar = np.concatenate((ground_points_lidar, np.ones((ground_points_lidar.shape[0], 1))), axis=1)
    ground_points_cam = np.matmul(sweepego2sweepsensor, ground_points_lidar.T).T
    denorm = -1 * equation_plane(ground_points_cam)
    return denorm


def get_sensor2virtual(denorm):
    origin_vector = np.array([0, 1, 0])    
    target_vector = -1 * np.array([denorm[0], denorm[1], denorm[2]])
    target_vector_norm = target_vector / np.sqrt(target_vector[0]**2 + target_vector[1]**2 + target_vector[2]**2)       
    sita = math.acos(np.inner(target_vector_norm, origin_vector))
    n_vector = np.cross(target_vector_norm, origin_vector) 
    n_vector = n_vector / np.sqrt(n_vector[0]**2 + n_vector[1]**2 + n_vector[2]**2)
    n_vector = n_vector.astype(np.float32)
    rot_mat, _ = cv2.Rodrigues(n_vector * sita)
    rot_mat = rot_mat.astype(np.float32)
    sensor2virtual = np.eye(4)
    sensor2virtual[:3, :3] = rot_mat
    return sensor2virtual.astype(np.float32)


def get_reference_height(denorm):
    ref_height = np.abs(denorm[3]) / np.sqrt(denorm[0]**2 + denorm[1]**2 + denorm[2]**2)
    return ref_height.astype(np.float32)


def get_cam_info(file_root):
    calib_file = os.path.join(file_root, "calib.txt")
    denorm_file = os.path.join(file_root, "denorm.txt")
    info = dict()

    cam_names = ['CAM_FRONT']
    cam_infos = dict()
    for cam_name in cam_names:
        cam_info = dict()
        cam_info['timestamp'] = 1000000
        cam_info['is_key_frame'] = True
        cam_info['height'] = 1080
        cam_info['width'] = 1920
        ego_pose = {"translation": [0.0, 0.0, 0.0], "rotation": [1.0, 0.0, 0.0, 0.0]}
        cam_info['ego_pose'] = ego_pose
        
        camera_intrinsic = load_calib(calib_file)
        # cam2lidar = velo_to_cam[:3, :3]
        # translation = velo_to_cam[:3, 3]
        cam2lidar, translation, Tr_cam2lidar, denorm = get_cam2lidar(denorm_file)
        # _, _, _, denorm = get_cam2lidar(denorm_file)
        
        calibrated_sensor = {"translation": translation, "rotation_matrix": cam2lidar, "camera_intrinsic": camera_intrinsic}
        cam_info['calibrated_sensor'] = calibrated_sensor
        cam_info['denorm'] = denorm
        cam_infos[cam_name] = cam_info

    info['cam_infos'] = cam_infos
    
    return info


def get_mats_dict(cam_info, ida_aug_conf):

    mats_dict = dict()

    assert len(cam_info) > 0


    sweep_sensor2ego_mats = list()
    sweep_intrin_mats = list()
    sweep_ida_mats = list()
    sweep_sensor2sensor_mats = list()
    sweep_sensor2virtual_mats = list()
    sweep_reference_heights = list()


    sensor2ego_mats = list()
    intrin_mats = list()
    ida_mats = list()
    sensor2sensor_mats = list()
    sensor2virtual_mats=list()
    reference_heights = list()
    key_info = cam_info


    cams = ['CAM_FRONT']
    for cam in cams:

        resize, resize_dims, crop, flip, rotate_ida = sample_ida_augmentation(ida_aug_conf)
        if "rotation_matrix" in cam_info[cam]['calibrated_sensor'].keys():
            sweepsensor2sweepego_rot = torch.Tensor(cam_info[cam]['calibrated_sensor']['rotation_matrix'])
        else:
            w, x, y, z = cam_info[cam]['calibrated_sensor']['rotation']
            # sweep sensor to sweep ego
            sweepsensor2sweepego_rot = torch.Tensor(
                Quaternion(w, x, y, z).rotation_matrix)
        sweepsensor2sweepego_tran = torch.Tensor(
            cam_info[cam]['calibrated_sensor']['translation'])
        sweepsensor2sweepego = sweepsensor2sweepego_rot.new_zeros((4, 4))
        sweepsensor2sweepego[3, 3] = 1
        sweepsensor2sweepego[:3, :3] = sweepsensor2sweepego_rot
        sweepsensor2sweepego[:3, -1] = sweepsensor2sweepego_tran
        # sweepsensor2sweepego: 外參

        sweepego2sweepsensor = sweepsensor2sweepego.inverse()
        denorm = get_denorm(sweepego2sweepsensor.numpy())
        # sweep ego to global
        w, x, y, z = cam_info[cam]['ego_pose']['rotation']
        sweepego2global_rot = torch.Tensor(
            Quaternion(w, x, y, z).rotation_matrix)
        sweepego2global_tran = torch.Tensor(
            cam_info[cam]['ego_pose']['translation'])
        sweepego2global = sweepego2global_rot.new_zeros((4, 4))
        sweepego2global[3, 3] = 1
        sweepego2global[:3, :3] = sweepego2global_rot
        sweepego2global[:3, -1] = sweepego2global_tran
        
        intrin_mat = torch.zeros((4, 4))
        intrin_mat[3, 3] = 1
        intrin_mat[:3, :3] = torch.Tensor(
            cam_info[cam]['calibrated_sensor']['camera_intrinsic'])
        sweepego2sweepsensor = sweepsensor2sweepego.inverse()
        

        
        denorm = get_denorm(sweepego2sweepsensor.numpy())

        sweepsensor2sweepego = sweepego2sweepsensor.inverse()

        # global sensor to cur ego
        w, x, y, z = key_info[cam]['ego_pose']['rotation']
        keyego2global_rot = torch.Tensor(
            Quaternion(w, x, y, z).rotation_matrix)
        keyego2global_tran = torch.Tensor(
            key_info[cam]['ego_pose']['translation'])
        keyego2global = keyego2global_rot.new_zeros((4, 4))
        keyego2global[3, 3] = 1
        keyego2global[:3, :3] = keyego2global_rot
        keyego2global[:3, -1] = keyego2global_tran
        global2keyego = keyego2global.inverse()

        # cur ego to sensor
        if "rotation_matrix" in key_info[cam]['calibrated_sensor'].keys():
            keysensor2keyego_rot = torch.Tensor(key_info[cam]['calibrated_sensor']['rotation_matrix'])
        else:
            w, x, y, z = key_info[cam]['calibrated_sensor']['rotation']                    
            keysensor2keyego_rot = torch.Tensor(
                Quaternion(w, x, y, z).rotation_matrix)
        keysensor2keyego_tran = torch.Tensor(
            key_info[cam]['calibrated_sensor']['translation'])
        keysensor2keyego = keysensor2keyego_rot.new_zeros((4, 4))
        keysensor2keyego[3, 3] = 1
        keysensor2keyego[:3, :3] = keysensor2keyego_rot
        keysensor2keyego[:3, -1] = keysensor2keyego_tran
        keyego2keysensor = keysensor2keyego.inverse()
        keysensor2sweepsensor = (
            keyego2keysensor @ global2keyego @ sweepego2global
            @ sweepsensor2sweepego).inverse()
        sweepsensor2keyego = global2keyego @ sweepego2global @ sweepsensor2sweepego
        sensor2virtual = torch.Tensor(get_sensor2virtual(denorm))
        sensor2ego_mats.append(sweepsensor2keyego)
        sensor2sensor_mats.append(keysensor2sweepsensor)
        sensor2virtual_mats.append(sensor2virtual)


        ida_mat = img_transform(
                    resize=resize,
                    resize_dims=resize_dims,
                    crop=crop,
                    flip=flip,
                    rotate=rotate_ida,
                )
        
        ida_mats.append(ida_mat)
        intrin_mats.append(intrin_mat)
        reference_heights.append(get_reference_height(denorm))
        

    sweep_sensor2ego_mats.append(torch.stack(sensor2ego_mats))
    sweep_intrin_mats.append(torch.stack(intrin_mats))
    sweep_ida_mats.append(torch.stack(ida_mats))
    sweep_sensor2sensor_mats.append(torch.stack(sensor2sensor_mats))
    sweep_sensor2virtual_mats.append(torch.stack(sensor2virtual_mats))
    sweep_reference_heights.append(torch.tensor(reference_heights))
        
    ret_list = [
    torch.stack(sweep_sensor2ego_mats).permute(1, 0, 2, 3),
    torch.stack(sweep_intrin_mats).permute(1, 0, 2, 3),
    torch.stack(sweep_ida_mats).permute(1, 0, 2, 3),
    torch.stack(sweep_sensor2sensor_mats).permute(1, 0, 2, 3),
    torch.stack(sweep_sensor2virtual_mats).permute(1, 0, 2, 3),
    torch.stack(sweep_reference_heights).permute(1, 0),
    ]

    rotate_bda, scale_bda, flip_dx, flip_dy = sample_bda_augmentation(
        )
    tempTensor = torch.tensor((), dtype=torch.float32)
    bda_mat = tempTensor.new_zeros(4, 4)
    bda_mat[0, 0] = 1
    bda_mat[1, 1] = 1
    bda_mat[2, 2] = 1
    bda_mat[3, 3] = 1


    
    sensor2ego_mats_batch = list()
    intrin_mats_batch = list()
    ida_mats_batch = list()
    sensor2sensor_mats_batch = list()
    sensor2virtual_mats_batch = list()
    bda_mat_batch = list()
    reference_heights_batch = list()
    
    sensor2ego_mats_batch.append(ret_list[0])
    intrin_mats_batch.append(ret_list[1])
    ida_mats_batch.append(ret_list[2])
    sensor2sensor_mats_batch.append(ret_list[3])
    sensor2virtual_mats_batch.append(ret_list[4])
    bda_mat_batch.append(bda_mat)
    reference_heights_batch.append(ret_list[5])

    mats_dict['sensor2ego_mats'] = torch.stack(sensor2ego_mats_batch)
    mats_dict['intrin_mats'] = torch.stack(intrin_mats_batch)
    mats_dict['ida_mats'] = torch.stack(ida_mats_batch)
    mats_dict['sensor2sensor_mats'] = torch.stack(sensor2sensor_mats_batch)
    mats_dict['sensor2virtual_mats'] = torch.stack(sensor2virtual_mats_batch)
    mats_dict['bda_mat'] = torch.stack(bda_mat_batch)
    mats_dict['reference_heights'] = torch.stack(reference_heights_batch)

    return mats_dict, denorm


