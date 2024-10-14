#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Image, PointCloud2
from std_msgs.msg import String
import sensor_msgs.point_cloud2 as pc2
from cv_bridge import CvBridge, CvBridgeError
import torch
import torchvision.transforms as T
from PIL import Image as PILImage
from BEVHeight_radar.models.bev_height import BEVHeight as BEVHeight_radar
import numpy as np
from mmdet3d.core.bbox.structures.lidar_box3d import LiDARInstance3DBoxes
import message_filters

from visualize_detect import *
# from tracker import get_2d_bounding_box
from mySort import Sort, draw_boxes, get_2d_bounding_box

from visualization_msgs.msg import MarkerArray, Marker
from std_msgs.msg import Header, ColorRGBA
from geometry_msgs.msg import Pose, Point, Vector3, Quaternion

import time
# import collections
# import threading

# range_config 有 r50_102 跟 r50_140 兩種
range_config = "r50_102"

tracking_class = {  "car": 0, 
                    "van": 0, 
                    "truck": 1, 
                    "bus": 1, 
                    "pedestrian": 2, 
                    "bicycle": 3, 
                    "trailer": 3, 
                    "motorcycle": 3}
tracking_class_name = ["Car", "Bus", "Pedestrian", "Cyclist"]

# BEVHeight radar

class BEVHeight_Radar_Ros_Inference:
    def __init__(self, model_path, backbone_conf, backbone_pts_conf, head_conf, img_conf, ida_aug_conf, rda_aug_conf):
        self.bridge = CvBridge()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = BEVHeight_radar(backbone_conf, head_conf, backbone_pts_conf).to(self.device)

        checkpoint = torch.load(model_path, map_location=self.device)['state_dict']
        for key in list(checkpoint.keys()):
            if 'model.' in key:
                checkpoint[key.replace('model.', '')] = checkpoint[key].to(self.device)
                del checkpoint[key]
        self.model.load_state_dict(checkpoint)
        self.model.eval()
        self.img_conf = img_conf
        self.ida_aug_conf = ida_aug_conf
        self.rda_aug_conf = rda_aug_conf
        self.frame_id = 0


        self.cam_info = list()
        self.cam_info = get_cam_info("/john/ncsist/ars548_ros/src/ars548_driver/weights")
        # print("cam_info", self.cam_info)

        self.mats_dict, cal_denorm = get_mats_dict(self.cam_info['cam_infos'], self.ida_aug_conf)
        print(self.mats_dict)

        # self.batch_mats_dict = {}
        # for key, value in self.mats_dict.items():
        #     if isinstance(value, torch.Tensor):  # 检查是否为张量
        #         # 将每个张量扩展为 [4, *value.shape] 的形状
        #         self.batch_mats_dict[key] = value.unsqueeze(0).repeat(4, *([1] * value.dim())).cuda()
        #     else:
        #         # 对于非张量数据结构，也需要批量化处理
        #         raise TypeError(f"Unsupported type for mats_dict entry: {type(value)}")
            
        for key, value in self.mats_dict.items():
            self.mats_dict[key] = value.cuda()

        self.camera_intrinsic = self.cam_info['cam_infos']['CAM_FRONT']['calibrated_sensor']['camera_intrinsic']

        self.denorm = self.cam_info['cam_infos']['CAM_FRONT']['denorm'] # 讀檔案的denorm

        # # 舊內參
        # camera_intrinsic = np.array([

        #         [2033,    0, 1068, 0],
        #         [   0, 2056,  539, 0],
        #         [   0,    0,    1, 0],
        #         [   0,    0,    0, 1]
        #         ], dtype=np.float32)


        rospy.init_node('bev_height_inference', anonymous=True)
        # rospy.Subscriber('/aravis_cam/image_color_row', Image, self.cameraCallback)
        # rospy.Subscriber('/radar_object', PointCloud2, self.radarCallback)
        self.camera_captured = False

        self.camera_input_tensor = ''
        self.camera_cv_image = ''
        self.max_radar_points_pv = 1536

        self.image_pub = rospy.Publisher('/det_image', Image, queue_size=1)
        self.marker_pub = rospy.Publisher('detected_object_marker', MarkerArray, queue_size=10)
        self.radar_hz_pub = rospy.Publisher('/radar_hz', String, queue_size=1)
        self.camera_hz_pub = rospy.Publisher('/camera_hz', String, queue_size=1)
        
        #.... Initialize SORT .... 
        #......................... 
        sort_max_age = 5
        sort_min_hits = 2
        sort_iou_thresh = 0.2
        self.sort_tracker = Sort(max_age=sort_max_age,
                        min_hits=sort_min_hits,
                        iou_threshold=sort_iou_thresh)
        #......................... 

        self.rgb_sub = message_filters.Subscriber('/aravis_cam/image_color_row', Image)
        self.radar_sub = message_filters.Subscriber('/radar_object', PointCloud2)
        self.sync = message_filters.ApproximateTimeSynchronizer([self.rgb_sub, self.radar_sub], queue_size=1, slop=0.1)
        self.sync.registerCallback(self.syncCallback)
        
        print("###"*10)
        print("BEVHeight Radar READY")
        rospy.spin()
    

    def transform_radar_pv(self, points, resize, resize_dims, crop, flip, rotate, radar_idx):
        # points = points[points[:, 2] < self.max_distance_pv, :]
        points = points[points[:, 2] < 58, :] # 過濾太遠的雷達點

        # 根據 resize 參數縮放x和y座標，根據crop對縮放後的座標剪裁
        H, W = resize_dims
        points[:, :2] = points[:, :2] * resize
        points[:, 0] -= crop[0]
        points[:, 1] -= crop[1]

        # flip應該是False
        if flip:
            points[:, 0] = resize_dims[1] - points[:, 0]

        # 座標校正：將x和y坐標平移回到圖像的中心
        points[:, 0] -= W / 2.0
        points[:, 1] -= H / 2.0

        h = rotate / 180 * np.pi
        rot_matrix = [
            [np.cos(h), np.sin(h)],
            [-np.sin(h), np.cos(h)],
        ]
        points[:, :2] = np.matmul(rot_matrix, points[:, :2].T).T

        points[:, 0] += W / 2.0
        points[:, 1] += H / 2.0

        depth_coords = points[:, :2].astype(np.int16)

        # 檢查點是否在圖像範圍內，並根據檢查結果過濾點雲數據
        valid_mask = ((depth_coords[:, 1] < resize_dims[0])
                      & (depth_coords[:, 0] < resize_dims[1])
                      & (depth_coords[:, 1] >= 0)
                      & (depth_coords[:, 0] >= 0))

        points = torch.Tensor(points[valid_mask])

        # if self.remove_z_axis:
        #     points[:, 1] = 1.  # dummy height value

        # 根據radar_idx篩選對應的點雲數據
        points_save = []
        for i in radar_idx:
            points_save.append(points[points[:, 6] == i])
        points = torch.cat(points_save, dim=0)

        # 對反射率(rcs)和速度進行標準化處理
        # mean, std of rcs and speed are from train set
        points[:, 3] = (points[:, 3] - 4.783) / 7.576
        points[:, 4] = (torch.norm(points[:, 4:6], dim=1) - 0.677) / 1.976

        # # 如果在訓練模式下，隨機丟棄一部分點雲數據，以進行數據增強
        # if self.is_train:
        #     drop_idx = np.random.uniform(size=points.shape[0])  # randomly drop points
        #     points = points[drop_idx > self.rda_aug_conf['drop_ratio']]

        # 如果點雲數量超過最大限制，則隨機選擇部分點；如果不足，則填充至最大數量
        num_points, num_feat = points.shape
        if num_points > self.max_radar_points_pv:
            choices = np.random.choice(num_points, self.max_radar_points_pv, replace=False)
            points = points[choices]
        else:
            num_append = self.max_radar_points_pv - num_points
            points = torch.cat([points, -999*torch.ones(num_append, num_feat)], dim=0)

        # 如果點雲數據為空，則添加一個默認值
        if num_points == 0:
            # points[0, :] = points.new_tensor([0.1, 0.1, self.max_distance_pv-1, 0, 0, 0, 0])
            points[0, :] = points.new_tensor([0.1, 0.1, 58-1, 0, 0, 0, 0])

        # 最後將x, y, z軸重新排列為x, z, y順序
        points[..., [0, 1, 2]] = points[..., [0, 2, 1]]  # convert [w, h, d] to [w, d, h]

        return points[..., :5]
    

    def sample_ida_augmentation(self):
        """Generate ida augmentation values based on ida_config."""
        H, W = self.ida_aug_conf['H'], self.ida_aug_conf['W']
        fH, fW = self.ida_aug_conf['final_dim']
        resize = max(fH / H, fW / W)
        resize_dims = (int(W * resize), int(H * resize))
        newW, newH = resize_dims
        crop_h = int(
            (1 - np.mean(self.ida_aug_conf['bot_pct_lim'])) * newH) - fH
        crop_w = int(max(0, newW - fW) / 2)
        crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
        flip = False
        rotate_ida = 0
        return resize, resize_dims, crop, flip, rotate_ida

    def sample_radar_augmentation(self):
        """Generate bda augmentation values based on bda_config."""
        # if self.is_train:
        #     radar_idx = np.random.choice(self.rda_aug_conf['N_sweeps'],
        #                                  self.rda_aug_conf['N_use'],
        #                                  replace=False)
        # else:
        radar_idx = np.arange(self.rda_aug_conf['N_sweeps'])
        return radar_idx



    def syncCallback(self, imgData, radarData):
        try:
            # start_time = time.time()
            # Camera
            cv_image = self.bridge.imgmsg_to_cv2(imgData, 'bgr8')

            # pil_image = PILImage.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
            pil_image = cv_image[:, :, (2,1,0)]
            resize, resize_dims, crop, flip, rotate_ida = self.sample_ida_augmentation()

            # first_time = time.time()
            # 這部分耗費最多時間
            # input_tensor = self.preprocess_image(
            #         img=pil_image,
            #         resize=resize,
            #         resize_dims=resize_dims,
            #         crop=crop,
            #         flip=flip,
            #         rotate=rotate_ida)

            input_tensor = self.preprocess_cv_image(
                    img=pil_image,
                    resize_dims=resize_dims,
                    crop=crop,
                    flip=flip,
                    rotate=rotate_ida)
            input_tensor = input_tensor.to(self.device)

            # second_time = time.time()

            # Radar
            pc_data = pc2.read_points(radarData, field_names=("x", "y", "z", "vx", "vy"), skip_nans=True)
        
            radar_points = list()
            sweep_radar_points = list()

            np_x = []
            np_y = []
            np_z = []
            np_vx = []
            np_vy = []
            np_rcs = []
            np_dummy = []

            for point in pc_data:
                np_x.append(point[0])
                np_y.append(point[1])
                np_z.append(point[2])
                np_vx.append(point[3])
                np_vy.append(point[4])
                np_rcs.append(0.0)
                np_dummy.append(0.0)

            np_x = np.array(np_x, dtype=np.float32)
            np_y = np.array(np_y, dtype=np.float32)
            np_z = np.array(np_z, dtype=np.float32)
            np_vx = np.array(np_vx, dtype=np.float32)
            np_vy = np.array(np_vy, dtype=np.float32)
            np_rcs = np.array(np_rcs, dtype=np.float32)
            np_dummy = np.array(np_dummy, dtype=np.float32)

            points_32 = np.transpose(np.vstack((np_x, np_y, np_z, np_rcs, np_vx, np_vy, np_dummy)))

            # third_time = time.time()

            resize, resize_dims, crop, flip, rotate_ida = self.sample_ida_augmentation()
            radar_idx = self.sample_radar_augmentation()
            radar_point_augmented = self.transform_radar_pv(
                            points_32, resize, self.ida_aug_conf['final_dim'],
                            crop, flip, rotate_ida, radar_idx)
            
            radar_points.append(radar_point_augmented)

            sweep_radar_points.append(torch.stack(radar_points))
            final_radar_points = torch.stack(sweep_radar_points).permute(1, 0, 2, 3)


            final_radar_points = final_radar_points.unsqueeze(0)
            final_radar_points = final_radar_points.to(self.device)

            # forth_time = time.time()

            # 這部分最久
            with torch.no_grad():
                output = self.model(input_tensor, self.mats_dict, final_radar_points)
                self.process_output(output, cv_image)
            
            # end_time = time.time()
            # print("Execution time:\t{:.6f} sec".format(end_time - start_time))
            # print("First time:\t{:.6f} sec".format(first_time-start_time))
            # print("Second time:\t{:.6f} sec".format(second_time-first_time))
            # print("Third time:\t{:.6f} sec".format(third_time-second_time))
            # print("Forth time:\t{:.6f} sec".format(forth_time-third_time))
            # print("Fifth time:\t{:.6f} sec".format(end_time-forth_time))
        except CvBridgeError as e:
            print(e)


    def preprocess_image(self, img, resize, resize_dims, crop, flip, rotate):
        # ====================================
        # adjust image
        img = img.resize(resize_dims)
        img = img.crop(crop)

        img = np.array(img)
        # mmcv.imnormalize
        # assert img.dtype != np.uint8
        img = img.astype(np.float32)
        mean = np.float64(self.img_conf['img_mean'].reshape(1, -1))
        stdinv = 1 / np.float64(self.img_conf['img_std'].reshape(1, -1))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # inplace
        img = cv2.subtract(img, mean)  # inplace
        img = cv2.multiply(img, stdinv)  # inplace

        img = torch.from_numpy(img).permute(2, 0, 1)

        imgs = list()
        sweep_imgs = list()
        imgs_batch = list()

        imgs.append(img)
        sweep_imgs.append(torch.stack(imgs))
        ret_list = [torch.stack(sweep_imgs).permute(1, 0, 2, 3, 4)]
        imgs_batch.append(ret_list[0])

        return torch.stack(imgs_batch)

    
    def preprocess_cv_image(self, img, resize_dims, crop, flip, rotate):
        # ====================================
        # Resize the image
        img = cv2.resize(img, resize_dims)
        # Crop the image
        x, y, w, h = crop
        img = img[y:y+h, x:x+w]

        # Convert image to float32
        img = img.astype(np.float32)

        # Normalize the image
        mean = np.float64(self.img_conf['img_mean'].reshape(1, -1))
        stdinv = 1 / np.float64(self.img_conf['img_std'].reshape(1, -1))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        img = cv2.subtract(img, mean)  # Subtract mean
        img = cv2.multiply(img, stdinv)  # Divide by std

        # Convert to tensor
        img = torch.from_numpy(img).permute(2, 0, 1)

        imgs = list()
        sweep_imgs = list()
        imgs_batch = list()

        imgs.append(img)
        sweep_imgs.append(torch.stack(imgs))
        ret_list = [torch.stack(sweep_imgs).permute(1, 0, 2, 3, 4)]
        imgs_batch.append(ret_list[0])

        return torch.stack(imgs_batch)
    
    # def process_output(self):
    def process_output(self, output, image):
        # print(len(self.output_buffer))
        # while True:
            # if (len(self.output_buffer) > 0):
                # print("PROCESS_OUTPUT")
                # output = self.output_buffer.pop() 
                # image = self.camera_cv_image.pop()

        # Process the model output here (e.g., visualize or publish results)
        img_metas = [dict(
            box_type_3d=LiDARInstance3DBoxes,
            ego2global_translation=np.array([0,0,0]),
            ego2global_rotation=np.array([1,0,0,0]),
        )]
        # print(img_metas)
        results = self.model.get_bboxes(output, img_metas)
        for i in range(len(results)):
            results[i][0] = results[i][0].tensor.detach().cpu().numpy()
            results[i][1] = results[i][1].detach().cpu().numpy()
            results[i][2] = results[i][2].detach().cpu().numpy()
            results[i].append(img_metas[i])


        all_pred_results = list()
        all_img_metas = list()
        for i in range(len(results)):
            all_pred_results.append(results[i][:3])
            all_img_metas.append(results[i][3])
        synchronize()
        # len_dataset = len(self.val_dataloader().dataset)
        len_dataset = 1
        all_pred_results = sum(
            map(list, zip(*all_gather_object(all_pred_results))),
            [])[:len_dataset]
        all_img_metas = sum(map(list, zip(*all_gather_object(all_img_metas))),
                            [])[:len_dataset]
        if get_rank() == 0:
            # src_denorm_file = os.path.join(dair_root, "training/denorm", sample_token + ".txt")
            # src_calib_file = os.path.join(dair_root, "training/calib", sample_token + ".txt")
            # if not os.path.exists(src_denorm_file):
                # src_denorm_file = os.path.join(dair_root, "validation/denorm", sample_token + ".txt")
                # src_calib_file = os.path.join(dair_root, "validation/calib", sample_token + ".txt")
            result_files, tmp_dir = format_results(all_pred_results, all_img_metas)
            Tr_velo_to_cam, r_velo2cam, t_velo2cam = get_velo2cam("/john/ncsist/ars548_ros/src/ars548_driver/weights/denorm.txt")
            # # 舊內參
            # camera_intrinsic = np.array([
            #                             [2033, 0, 1068],
            #                             [0, 2056, 539],
            #                             [0, 0, 1]
            #                         ])

            # # 新內參
            # camera_intrinsic = np.array([
            #         [1804.85904,          0, 1028.16764],
            #         [         0, 1801.73036, 776.526745],
            #         [         0,          0,          1,]])
            camera_intrinsic = self.camera_intrinsic
            
            camera_intrinsic = np.concatenate([camera_intrinsic, np.zeros((camera_intrinsic.shape[0], 1))], axis=1)
            preds = result_files['img_bbox']
            pred_lines = []
            tracking_list = np.empty((0,6))

            marker_array = MarkerArray()
            marker_array.markers.append(Marker(
                header=Header(frame_id="ARS_548", stamp=rospy.Time.now()),
                action=Marker.DELETEALL
            ))
            id = 0

            for pred in preds:
                loc = pred["translation"]
                dim = pred["size"]
                yaw_lidar = pred["box_yaw"]
                detection_score = pred["detection_score"]
                class_name = pred["detection_name"]
                
                w, l, h = dim[0], dim[1], dim[2]
                x, y, z = loc[0], loc[1], loc[2]            
                bottom_center = [x, y, z]
                obj_size = [l, w, h]
                bottom_center_in_cam = r_velo2cam * np.matrix(bottom_center).T + t_velo2cam
                alpha, yaw = get_camera_3d_8points(
                    obj_size, yaw_lidar, bottom_center, bottom_center_in_cam, r_velo2cam, t_velo2cam
                )
                yaw  = 0.5 * np.pi - yaw_lidar

                cam_x, cam_y, cam_z = convert_point(np.array([x, y, z, 1]).T, Tr_velo_to_cam)
                box = get_lidar_3d_8points([w, l, h], yaw_lidar, [x, y, z + h/2])
                box2d = bbbox2bbox(box, Tr_velo_to_cam, camera_intrinsic)
                if detection_score > 0.4 and class_name in category_map_rope3d.keys():
                    i1 = category_map_rope3d[class_name]
                    i2 = str(0)
                    i3 = str(0)
                    i4 = str(round(alpha, 4))
                    i5, i6, i7, i8 = (
                        str(round(box2d[0], 4)),
                        str(round(box2d[1], 4)),
                        str(round(box2d[2], 4)),
                        str(round(box2d[3], 4)),
                    )
                    i9, i11, i10 = str(round(h, 4)), str(round(w, 4)), str(round(l, 4))
                    i12, i13, i14 = str(round(cam_x, 4)), str(round(cam_y, 4)), str(round(cam_z, 4))
                    i15 = str(round(yaw, 4))
                    line = [i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, str(round(detection_score, 4))]
                    pred_lines.append(line) # 這個就是像這種

                    # 這個2d框是模型預測的，不是3d投影到2d上的
                    # image = cv2.rectangle(image, (int(box2d[0]), int(box2d[1])), (int(box2d[2]), int(box2d[3])), (255, 255, 255), 2) # 畫2D框
                    # Car 0 0 1.6138 619.1764 366.0287 722.9627 465.0574 1.1012 1.81 4.2636 -5.9165 -1.9003 44.1509 4.6289 0.7451                

                    COLOR_LIST = {'car':ColorRGBA(r=0.0, g=0.0, b=1.0, a=1.0),
                                'motorcycle':ColorRGBA(r=0.0, g=1.0, b=0.0, a=1.0),
                                'bus':ColorRGBA(r=0.0, g=1.0, b=1.0, a=1.0),
                                'truck':ColorRGBA(r=1.0, g=0.0, b=1.0, a=1.0)
                                }
                    box_point = []
                    box_color = ColorRGBA(r=1.0, g=0.0, b=0.0, a=1.0)
                    if class_name in COLOR_LIST:
                        box_color = COLOR_LIST[class_name]

                    box_marker = Marker(
                        header=Header(frame_id="ARS_548", stamp=rospy.Time.now()),
                        id=id,
                        ns="detected_object",
                        type=Marker.LINE_LIST,
                        action=Marker.ADD,
                        scale=Vector3(x=0.1, y=0.1, z=0.1),
                        color=box_color,
                    )
                    box_marker.points = []
                    for i in range(8):
                        p = Point(x=box[i, 0], y=box[i, 1], z=box[i, 2])
                        box_point.append(p)

                    # 長方體的底面四條線
                    box_marker.points.append(box_point[0])
                    box_marker.points.append(box_point[1])
                    box_marker.points.append(box_point[1])
                    box_marker.points.append(box_point[2])
                    box_marker.points.append(box_point[2])
                    box_marker.points.append(box_point[3])
                    box_marker.points.append(box_point[3])
                    box_marker.points.append(box_point[0])
                    # 往上的四條線
                    box_marker.points.append(box_point[0])
                    box_marker.points.append(box_point[4])
                    box_marker.points.append(box_point[1])
                    box_marker.points.append(box_point[5])
                    box_marker.points.append(box_point[2])
                    box_marker.points.append(box_point[6])
                    box_marker.points.append(box_point[3])
                    box_marker.points.append(box_point[7])
                    # 頂面四條線
                    box_marker.points.append(box_point[4])
                    box_marker.points.append(box_point[5])
                    box_marker.points.append(box_point[5])
                    box_marker.points.append(box_point[6])
                    box_marker.points.append(box_point[6])
                    box_marker.points.append(box_point[7])
                    box_marker.points.append(box_point[7])
                    box_marker.points.append(box_point[4])



                    marker_array.markers.append(box_marker)
                    id += 1

            self.marker_pub.publish(marker_array)

            

            color_map = {"Car":(0, 255, 0), "Bus":(0, 255, 255), "Pedestrian":(255, 255, 0), "Cyclist":(0, 0, 255)}
            for line in pred_lines:
                # line_list = line.split('\n')[0].split(' ')
                object_type = line[0]
                if object_type not in color_map.keys(): 
                    continue
                dim = np.array(line[8:11]).astype(float)
                location = np.array(line[11:14]).astype(float)
                rotation_y = float(line[14])
                
                denorm = self.denorm
                P2 = self.camera_intrinsic.copy()
                P2 = np.insert(P2, 3, [0], axis=1)
                # print(P2)
                # denorm = [1.48001788e-04,-9.73842628e-01,-2.27223369e-01,7.07289670e+00]
                box_3d = compute_box_3d_camera(dim, location, rotation_y, denorm)
                # P2 = [[2.1747119e+03,0.0000000e+00,9.6376355e+02,0.0000000e+00],
                #     [0.0000000e+00,2.3158064e+03,5.7183185e+02,0.0000000e+00],
                #     [0.0000000e+00,0.0000000e+00,1.0000000e+00,0.0000000e+00]]
                
                box_2d = project_to_image(box_3d, P2)   # 2D平面上的8個點，跟預測的2D框不一樣
                x1, y1, x2, y2 = get_2d_bounding_box(box_2d)

                tracking_list = np.vstack((tracking_list, np.array([int(x1), int(y1), int(x2), int(y2), detection_score, tracking_class[class_name]])))
                image = draw_box_3d(image, box_2d, c=color_map[object_type])
            
            results_path = os.path.join("/john/ncsist/ars548_ros/")

            # cv2.imwrite(filename=os.path.join(results_path, "det_png", "cal6_{:06d}".format(self.frame_id)+".jpg"), img=image)
            # cv2.imwrite(filename=os.path.join(results_path, "original_png", "cal6_{:06d}".format(self.frame_id)+".jpg"), img=original_img)
            # write_kitti_in_txt(pred_lines, os.path.join(results_path, "det_txt", "cal6_{:06d}".format(self.frame_id) + ".txt"))


            # Run SORT
            if len(tracking_list):
                tracked_dets = self.sort_tracker.update(tracking_list)
                tracks = self.sort_tracker.getTrackers()

                for track in tracks:
                    if len(tracked_dets)>0:
                        bbox_xyxy = tracked_dets[:,:4]
                        identities = tracked_dets[:, 8]
                        categories = tracked_dets[:, 4]
                        
                        image = draw_boxes(image, bbox_xyxy, identities, categories, tracking_class_name, save_with_object_id=False)
            else:
                tracked_dets = self.sort_tracker.update()



            self.frame_id += 1  

            image_message = self.bridge.cv2_to_imgmsg(image, "bgr8")

            self.image_pub.publish(image_message)

# BEVHeight radar
# ===========================================================================





if __name__ == '__main__':
    

    H = 1536
    W = 2048
    final_dim = (864, 1536)
    img_conf = dict(img_mean=np.array([123.675, 116.28, 103.53]),
                img_std=np.array([58.395, 57.12, 57.375]),
                to_rgb=True,
                final_dim = (864, 1536))


    # 依據不同的model換不同range config
    if range_config == "r50_102":
        backbone_conf = {
        'x_bound': [0, 102.4, 0.8],
        'y_bound': [-51.2, 51.2, 0.8],
        'z_bound': [-5, 3, 8],
        'd_bound': [-1.5, 3.0, 180],
        'final_dim':
        final_dim,
        'output_channels':
        80,
        'downsample_factor':
        16,
        'img_backbone_conf':
        dict(
            type='ResNet',
            depth=50,
            frozen_stages=0,
            out_indices=[0, 1, 2, 3],
            norm_eval=False,
            init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50'),
        ),
        'img_neck_conf':
        dict(
            type='SECONDFPN',
            in_channels=[256, 512, 1024, 2048],
            upsample_strides=[0.25, 0.5, 1, 2],
            out_channels=[128, 128, 128, 128],
        ),
        'height_net_conf':
        dict(in_channels=512, mid_channels=512)
        }
    elif range_config == "r50_140":
        backbone_conf = {
        'x_bound': [0, 140.8, 0.8],
        'y_bound': [-70.4, 70.4, 0.8],
        'z_bound': [-5, 3, 8],
        'd_bound': [-1.5, 3.0, 180],
        'final_dim':
        final_dim,
        'output_channels':
        80,
        'downsample_factor':
        16,
        'img_backbone_conf':
        dict(
            type='ResNet',
            depth=50,
            frozen_stages=0,
            out_indices=[0, 1, 2, 3],
            norm_eval=False,
            init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50'),
        ),
        'img_neck_conf':
        dict(
            type='SECONDFPN',
            in_channels=[256, 512, 1024, 2048],
            upsample_strides=[0.25, 0.5, 1, 2],
            out_channels=[128, 128, 128, 128],
        ),
        'height_net_conf':
        dict(in_channels=512, mid_channels=512)
        }
    else:
        print("Error: range_config not found.")



        
    ida_aug_conf = {
        'final_dim':
        final_dim,
        'H':
        H,
        'W':
        W,
        'bot_pct_lim': (0.0, 0.0),
        'cams': ['CAM_FRONT'],
        'Ncams': 1,
    }

    rda_aug_conf = {
    'N_sweeps': 6,
    'N_use': 5,
    'drop_ratio': 0.1,
    }

    bev_backbone = dict(
        type='ResNet',
        in_channels=80,
        depth=18,
        num_stages=3,
        strides=(1, 2, 2),
        dilations=(1, 1, 1),
        out_indices=[0, 1, 2],
        norm_eval=False,
        base_channels=160,
    )

    bev_neck = dict(type='SECONDFPN',
                    in_channels=[80, 160, 320, 640],
                    upsample_strides=[1, 2, 4, 8],
                    out_channels=[64, 64, 64, 64])

    CLASSES = [
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

    TASKS = [
        dict(num_class=1, class_names=['car']),
        dict(num_class=2, class_names=['truck', 'construction_vehicle']),
        dict(num_class=2, class_names=['bus', 'trailer']),
        dict(num_class=1, class_names=['barrier']),
        dict(num_class=2, class_names=['motorcycle', 'bicycle']),
        dict(num_class=2, class_names=['pedestrian', 'traffic_cone']),
    ]

    common_heads = dict(reg=(2, 2),
                        height=(1, 2),
                        dim=(3, 2),
                        rot=(2, 2),
                        vel=(2, 2))

    if range_config == "r50_102":
        bbox_coder = dict(
            type='CenterPointBBoxCoder',
            post_center_range=[0.0, -61.2, -10.0, 122.4, 61.2, 10.0],
            max_num=500,
            score_threshold=0.1,
            out_size_factor=4,
            voxel_size=[0.2, 0.2, 8],
            pc_range=[0, -51.2, -5, 104.4, 51.2, 3],
            code_size=9,
        )

        train_cfg = dict(
            point_cloud_range=[0, -51.2, -5, 102.4, 51.2, 3],
            grid_size=[512, 512, 1],
            voxel_size=[0.2, 0.2, 8],
            out_size_factor=4,
            dense_reg=1,
            gaussian_overlap=0.1,
            max_objs=500,
            min_radius=2,
            code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 0.5],
        )

        test_cfg = dict(
            post_center_limit_range=[0.0, -61.2, -10.0, 122.4, 61.2, 10.0],
            max_per_img=500,
            max_pool_nms=False,
            min_radius=[4, 12, 10, 1, 0.85, 0.175],
            score_threshold=0.1,
            out_size_factor=4,
            voxel_size=[0.2, 0.2, 8],
            nms_type='circle',
            pre_max_size=1000,
            post_max_size=83,
            nms_thr=0.2,
        )
    elif range_config == "r50_140":
        bbox_coder = dict(
            type='CenterPointBBoxCoder',
            post_center_range=[0.0, -70.4, -10.0, 140.8, 70.4, 10.0],
            max_num=500,
            score_threshold=0.1,
            out_size_factor=4,
            voxel_size=[0.2, 0.2, 8],
            pc_range=[0, -70.4, -5, 140.8, 70.4, 3],
            code_size=9,
        )

        train_cfg = dict(
            point_cloud_range=[0, -70.4, -5, 140.8, 70.4, 3],
            grid_size=[704, 704, 1],
            voxel_size=[0.2, 0.2, 8],
            out_size_factor=4,
            dense_reg=1,
            gaussian_overlap=0.1,
            max_objs=500,
            min_radius=2,
            code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 0.5],
        )

        test_cfg = dict(
            post_center_limit_range=[0.0, -70.4, -10.0, 140.8, 70.4, 10.0],
            max_per_img=500,
            max_pool_nms=False,
            min_radius=[4, 12, 10, 1, 0.85, 0.175],
            score_threshold=0.1,
            out_size_factor=4,
            voxel_size=[0.2, 0.2, 8],
            nms_type='circle',
            pre_max_size=1000,
            post_max_size=83,
            nms_thr=0.2,
        )
    else:
        print("Error: range_config not found.")


    head_conf = {
        'bev_backbone_conf': bev_backbone,
        'bev_neck_conf': bev_neck,
        'tasks': TASKS,
        'common_heads': common_heads,
        'bbox_coder': bbox_coder,
        'train_cfg': train_cfg,
        'test_cfg': test_cfg,
        'in_channels': 256,  # Equal to bev_neck output_channels.
        'loss_cls': dict(type='GaussianFocalLoss', reduction='mean'),
        'loss_bbox': dict(type='L1Loss', reduction='mean', loss_weight=0.25),
        'gaussian_overlap': 0.1,
        'min_radius': 2,
    }


    # Radar Branch
    ################################################
    backbone_pts_conf = {
        'pts_voxel_layer': dict(
            max_num_points=8,
            voxel_size=[8, 0.4, 2],
            point_cloud_range=[0, 2.0, 0, 704, 58.0, 2],
            max_voxels=(768, 1024)
        ),
        'pts_voxel_encoder': dict(
            type='PillarFeatureNet',
            in_channels=5,
            feat_channels=[32, 64],
            with_distance=False,
            with_cluster_center=False,
            with_voxel_center=True,
            voxel_size=[8, 0.4, 2],
            point_cloud_range=[0, 2.0, 0, 704, 58.0, 2],
            norm_cfg=dict(type='BN1d', eps=1e-3, momentum=0.01),
            legacy=True
        ),
        'pts_middle_encoder': dict(
            type='PointPillarsScatter',
            in_channels=64,
            output_shape=(256, 256) # 原本是(140, 88)
        ),
        'pts_backbone': dict(
            type='SECOND',
            in_channels=64,
            out_channels=[64, 128, 256],
            layer_nums=[3, 5, 5],
            layer_strides=[1, 2, 2],
            norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
            conv_cfg=dict(type='Conv2d', bias=True, padding_mode='reflect')
        ),
        'pts_neck': dict(
            type='SECONDFPN',
            in_channels=[64, 128, 256],
            out_channels=[128, 128, 128],
            upsample_strides=[0.5, 1, 2],
            norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
            upsample_cfg=dict(type='deconv', bias=False),
            use_conv_for_no_stride=True
        ),
        'occupancy_init': 0.01,
        'out_channels_pts': 80,
    }
    ################################################

    # Fusion module
    ################################################
    # 還沒用到(應該會用在fusion_module.py)
    fuser_conf = {
        'img_dims': 80,
        'pts_dims': 80,
        'embed_dims': 128,
        'num_layers': 6,
        'num_heads': 4,
        'bev_shape': (128, 128),
    }
    ################################################

    # if range_config == "r50_102":
    #     original_model_path = '/john/ncsist/ars548_ros/src/ars548_driver/weights/BEVHeight_R50_128_102.4_72.45_39_epochs.ckpt' # BEVHeight original
    # elif range_config == "r50_140":
    #     original_model_path = '/john/ncsist/ars548_ros/src/ars548_driver/weights/BEVHeight_R50_128_140.8_79.16_39_epochs.ckpt' # BEVHeight original
    # else:
    #     print("Error: range_config not found.")

    radar_model_path = '/john/ncsist/ars548_ros/src/ars548_driver/weights/after_adjusting.ckpt' # BEVHeight radar

    BEVHeight_Radar_Ros_Inference(radar_model_path, backbone_conf, backbone_pts_conf, head_conf, img_conf, ida_aug_conf, rda_aug_conf)
