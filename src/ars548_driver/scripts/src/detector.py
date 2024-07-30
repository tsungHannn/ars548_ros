#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import torch
import torchvision.transforms as transforms
from PIL import Image as PILImage
from BEVHeight_original.models.bev_height import BEVHeight
import numpy as np

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
        print("###"*10)
        print("READY")
        rospy.spin()
        
    def callback(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, 'bgr8')
            pil_image = PILImage.fromarray(cv_image)
            input_tensor = self.preprocess_image(pil_image)
            input_tensor = input_tensor.to(self.device)

            for key, value in self.mats_dict.items():
                self.mats_dict[key] = value.cuda()

            with torch.no_grad():
                output = self.model(input_tensor, self.mats_dict)
                self.process_output(output)
        except CvBridgeError as e:
            print(e)
    
    def preprocess_image(self, image):
        # Transform image to the appropriate format
        transform = transforms.Compose([
            transforms.Resize(self.img_conf['final_dim']),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.img_conf['img_mean'], std=self.img_conf['img_std']),
        ])

        # ret_list = [torch.stack(image), self.mats_dict]

        return transform(image).unsqueeze(0).unsqueeze(0)
    
    def process_output(self, output):
        # Process the model output here (e.g., visualize or publish results)
        results = self.model.get_bboxes(output)
        

if __name__ == '__main__':
    model_path = '/john/ncsist/ars548_ros/src/ars548_driver/weights/BEVHeight_R50_128_102.4_72.45_39_epochs.ckpt'

    H = 1080
    W = 1920
    final_dim = (864, 1536)
    img_conf = dict(img_mean=[123.675, 116.28, 103.53],
                img_std=[58.395, 57.12, 57.375],
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
