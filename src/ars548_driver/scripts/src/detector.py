#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import torch
import torchvision.transforms as transforms
from PIL import Image as PILImage
from BEVHeight_original.models.bev_height import BEVHeight
import numpy as np
from mmdet3d.core.bbox.structures.lidar_box3d import LiDARInstance3DBoxes
from visualize_detect import *




class BEVHeightRosInference:
    def __init__(self, model_path, backbone_conf, head_conf, img_conf):
        self.bridge = CvBridge()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = BEVHeight(backbone_conf, head_conf).to(self.device)

        checkpoint = torch.load(model_path, map_location=self.device)['state_dict']
        for key in list(checkpoint.keys()):
            if 'model.' in key:
                checkpoint[key.replace('model.', '')] = checkpoint[key].to(self.device)
                del checkpoint[key]
        self.model.load_state_dict(checkpoint)
        self.model.eval()
        self.img_conf = img_conf
        

        self.mats_dict = torch.load('/john/ncsist/ars548_ros/src/ars548_driver/weights/mats_dict.pt')
        # sensor2ego_mats = self.mats_dict['sensor2ego_mats']
        # intrin_mats = self.mats_dict['intrin_mats']
        # ida_mats = self.mats_dict['ida_mats']
        # reference_heights = self.mats_dict['reference_heights']
        # sensor2sensor_mats = self.mats_dict['sensor2sensor_mats']
        # sensor2virtual_mats = self.mats_dict['sensor2virtual_mats']
        # bda_mat = self.mats_dict['bda_mat']
        rospy.init_node('bev_height_inference', anonymous=True)
        rospy.Subscriber('/aravis_cam/image_color_row', Image, self.callback)

        self.image_pub = rospy.Publisher('/det_image', Image, queue_size=1)
        
        print("###"*10)
        print("READY")
        rospy.spin()
        
    def callback(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, 'bgr8')

            pil_image = PILImage.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
            pil_image = pil_image.resize(self.img_conf['final_dim'])
            input_tensor = self.preprocess_image(np.array(pil_image))
            input_tensor = input_tensor.to(self.device)

            for key, value in self.mats_dict.items():
                self.mats_dict[key] = value.cuda()

            with torch.no_grad():
                output = self.model(input_tensor, self.mats_dict)
                self.process_output(output, cv_image.copy())
        except CvBridgeError as e:
            print(e)
    
    def preprocess_image(self, image):
        # mmcv.imnormalize
        # assert image.dtype != np.uint8
        image = image.astype(np.float32)
        mean = np.float64(self.img_conf['img_mean'].reshape(1, -1))
        stdinv = 1 / np.float64(self.img_conf['img_std'].reshape(1, -1))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # inplace
        image = cv2.subtract(image, mean)  # inplace
        image = cv2.multiply(image, stdinv)  # inplace

        image = torch.from_numpy(image).permute(2, 0, 1)

        # Transform image to the appropriate format
        # transform = transforms.Compose([
        #     transforms.ToTensor(),
        #     transforms.Resize(self.img_conf['final_dim']),
        #     # transforms.Normalize(mean=self.img_conf['img_mean'], std=self.img_conf['img_std']),
        # ])

        # ret_list = [torch.stack(image), self.mats_dict]
        return image.unsqueeze(0).unsqueeze(0)
    
    def process_output(self, output, image):
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
            Tr_velo_to_cam, r_velo2cam, t_velo2cam = get_velo2cam()
            camera_intrinsic = np.array([
                                        [2033, 0, 1068],
                                        [0, 2056, 539],
                                        [0, 0, 1]
                                    ])
            camera_intrinsic = np.concatenate([camera_intrinsic, np.zeros((camera_intrinsic.shape[0], 1))], axis=1)
            preds = result_files['img_bbox']
            pred_lines = []
            bboxes = []
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

                if detection_score > 0.45 and class_name in category_map_rope3d.keys():
                    print("@@@@@@@@@@@@@@@@@@@@@@")
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
                    # Car 0 0 1.6138 619.1764 366.0287 722.9627 465.0574 1.1012 1.81 4.2636 -5.9165 -1.9003 44.1509 4.6289 0.7451
                    bboxes.append(box)


            color_map = {"Car":(0, 255, 0), "Bus":(0, 255, 255), "Pedestrian":(255, 255, 0), "Cyclist":(0, 0, 255)}
            for line in pred_lines:
                # line_list = line.split('\n')[0].split(' ')
                object_type = line[0]
                if object_type not in color_map.keys(): 
                    continue
                dim = np.array(line[8:11]).astype(float)
                location = np.array(line[11:14]).astype(float)
                rotation_y = float(line[14])
                denorm = [1.48001788e-04,-9.73842628e-01,-2.27223369e-01,7.07289670e+00]
                box_3d = compute_box_3d_camera(dim, location, rotation_y, denorm)
                P2 = [[2.1747119e+03,0.0000000e+00,9.6376355e+02,0.0000000e+00],
                    [0.0000000e+00,2.3158064e+03,5.7183185e+02,0.0000000e+00],
                    [0.0000000e+00,0.0000000e+00,1.0000000e+00,0.0000000e+00]]

                box_2d = project_to_image(box_3d, P2)
                image = draw_box_3d(image, box_2d, c=color_map[object_type])
            
            image_message = self.bridge.cv2_to_imgmsg(image, "bgr8")
            self.image_pub.publish(image_message)

if __name__ == '__main__':
    model_path = '/john/ncsist/ars548_ros/src/ars548_driver/weights/BEVHeight_R50_128_102.4_72.45_39_epochs.ckpt'

    H = 1080
    W = 1920
    final_dim = (864, 1536)
    img_conf = dict(img_mean=np.array([123.675, 116.28, 103.53]),
                img_std=np.array([58.395, 57.12, 57.375]),
                to_rgb=True,
                final_dim = (864, 1536))

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



    BEVHeightRosInference(model_path, backbone_conf, head_conf, img_conf)
