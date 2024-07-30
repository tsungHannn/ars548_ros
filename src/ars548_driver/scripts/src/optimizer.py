#!/usr/bin/env python3
from collections import defaultdict
from copy import deepcopy

import math
import cv2
import rospy
import numpy as np
import message_filters
from numpy.polynomial import polyutils as pu
from std_msgs.msg import String, Header, ColorRGBA
from sensor_msgs.msg import Image, PointCloud2, PointField
import sensor_msgs.point_cloud2 as pc2
from cv_bridge import CvBridge
# from ars408_msg.msg import RadarPoints
from ars548_messages.msg import ObjectList

# Visualize radar information
from visualization_msgs.msg import MarkerArray, Marker
from geometry_msgs.msg import Pose, Point, Vector3, Quaternion

from ppdeploy.pipeline.datacollector import DataCollector, Result
from ppdeploy.pipeline.pipe_utils import parse_mot_res
from ppdeploy.pptracking.python.mot_sde_infer import SDE_Detector
from ppdeploy.pptracking.python.mot.visualize import plot_tracking_dict
from ppdeploy.pptracking.python.mot.utils import flow_statistic, flow_statistic_multi_class, update_object_info

class TrafficStats():
    def __init__(self):
        self.mean_speed_list = []
        self.mean_occupancy_ratio_list = []
        self.traffic_history = []
    
    def reset(self):
        self.mean_speed_list.clear()
        self.mean_occupancy_ratio_list.clear()

    def add_interval(self, x):
        self.traffic_history.append({**x, 'time': rospy.Time.now()})
    
    def add_record(self, x):
        self.mean_speed_list.extend(x['speed'])
        self.mean_occupancy_ratio_list.append(x['occupancy_ratio'])
    
    def mean_speed(self):
        return np.mean(self.mean_speed_list)

    def mean_occupancy_ratio(self):
        return np.mean(self.mean_occupancy_ratio_list)


class RadarPoint():
    def __init__(self, point_x=0, point_y=0, point_speed=0, point_dist=0):
        self.point_x = point_x
        self.point_y = point_y
        self.point_speed = point_speed
        self.point_dist = point_dist


def calculate_radar_points_averages(points):
    if len(points) == 1:
        return points[0]

    avg = RadarPoint()

    for p in points:
        avg.point_x += p.point_x
        avg.point_y += p.point_y
        avg.point_speed += p.point_speed
        avg.point_dist += p.point_dist
    
    avg.point_x /= len(points)
    avg.point_y /= len(points)
    avg.point_speed /= len(points)
    avg.point_dist /= len(points)

    return avg

class PaddleDetector():
    def __init__(self, use_time_synchronizer=True):
        rospy.init_node('yolov8_detector', anonymous=True)

        base_dir = '/john/ncsist/ars548_ros/src/ars548_driver/'

        self.model_dir = base_dir + 'weights/ppyoloe_plus_crn_l_80e_coco'
        self.tracker_config = base_dir + 'scripts/src/ppdeploy/pipeline/config/tracker_config.yml'
        self.device = 'GPU'
        self.run_mode = 'paddle'
        self.batch_size = 1
        self.skip_frame_num = 1
        self.do_entrance_counting = True
        self.draw_center_traj = False
        self.region_type = 'center-down-17'
        self.secs_interval = 15 # 15 seconds interval
        self.video_fps = 13 # radar rps
        self.do_break_in_counting = False
        self.plot_traffic_on_mot = True

        self.pipeline_res = Result()
        self.collector = DataCollector()

        self.display_radar_image = True
        self.display_radar_rgb_fusion = True
        self.image_width, self.image_height = 2048, 1536
        self.p = np.array([1068, -2033, 0, 0, 539, 0, -2056, 0, 1, 0, 0, 0]).reshape((3, 4))
        self.hard_postprocess = True

        self.bridge = CvBridge()

        if use_time_synchronizer:
            self.sub_image = message_filters.Subscriber('/aravis_cam/image_color', Image)
            self.sub_radar = message_filters.Subscriber('/radar/object_list', ObjectList)
            self.sync = message_filters.ApproximateTimeSynchronizer([self.sub_image, self.sub_radar], 10, 10, reset=True, allow_headerless=True)
            self.only_image = rospy.Subscriber('/aravis_cam/image_color', Image, self.imageCallback)
            self.only_radar = rospy.Subscriber('/radar/point_cloud_object', PointCloud2, self.radarCallback)
            self.radar_objectlist = rospy.Subscriber('/radar/object_list', ObjectList, self.objectListCallback)
            
            self.sync.registerCallback(self.callback)
            
        self.pub_radar_filter = rospy.Publisher('/radar_filter', PointCloud2, queue_size=10)
        self.pub_radar_marker = rospy.Publisher('/radar_marker', MarkerArray, queue_size=1)
        self.pub_mot = rospy.Publisher('/mot_image', Image, queue_size=10)
        self.pub_stats = rospy.Publisher('/traffic_stats', String, queue_size=10)
        if self.display_radar_image:
            self.pub_radar = rospy.Publisher('/radar_image', Image, queue_size=10)

        self.mot_predictor = SDE_Detector(
            model_dir=self.model_dir,
            tracker_config=self.tracker_config,
            device=self.device,
            run_mode=self.run_mode,
            batch_size=self.batch_size,
            skip_frame_num=self.skip_frame_num,
            do_entrance_counting=self.do_entrance_counting,
            region_type=self.region_type,
        )

        self.frame_id = 0

        self.entrance, self.records, self.center_traj = None, None, None
        if self.draw_center_traj:
            self.center_traj = [{}]
        if self.do_entrance_counting:
            if self.region_type == 'horizontal-C':
                self.entrance = [0, self.image_height / 2. * 1.3, self.image_width, self.image_height / 2. * 1.3]
                self.effective_road_ratio = 0.5
            elif self.region_type == 'horizontal-B':
                self.entrance = [0, self.image_height / 2. * 0.6, self.image_width, self.image_height / 2. * 0.6]
                self.effective_road_ratio = 0.5
            elif self.region_type == 'horizontal-A':
                self.entrance = [0, self.image_height / 2. * 1.0, self.image_width, self.image_height / 2. * 1.0]
                self.effective_road_ratio = 0.5
            elif self.region_type == 'tunnel-5':
                self.entrance = [0, self.image_height / 2., self.image_width, self.image_height / 2.]
                self.effective_road_ratio = 0.4
            elif self.region_type == 'tunnel-10':
                self.entrance = [0, self.image_height / 2., self.image_width, self.image_height / 2.]
                self.effective_road_ratio = 0.4
            elif self.region_type == 'tunnel-15':
                self.entrance = [0, self.image_height / 2., self.image_width, self.image_height / 2.]
                self.effective_road_ratio = 0.4
            elif self.region_type == 'center-down-17':
                self.entrance = [0, self.image_height / 2., self.image_width, self.image_height / 2.]
                self.effective_road_ratio = 0.4
            elif self.region_type == 'horizontal':
                self.entrance = [0, self.image_height / 2., self.image_width, self.image_height / 2.]
            elif self.region_type == 'vertical':
                self.entrance = [self.image_width / 2, 0., self.image_width / 2, self.image_height]
            else:
                raise ValueError("region_type:{} unsupported.".format(
                    self.region_type))
        self.reset_mot()
        self.traffic_stats = TrafficStats()

        # Define congestion level thresholds
        self.flow_thresholds = [15, 30, 40]  # Define traffic flow thresholds for levels 1 to 4
        self.speed_thresholds = [20, 30, 40]  # Define mean speed thresholds for levels 1 to 4

        # Optimzer
        self.scale_y_dist = [15, 20, 30, 40, 50, 57]
        self.scale_y = [0] * len(self.scale_y_dist)
        self.optimizer_index = 0
        self.optimizer_count = [0] * len(self.scale_y_dist)
        self.optimizer_step = 3
    
    def reset_mot(self):
        self.id_set = set()
        self.interval_id_set = set()
        self.in_id_list = list()
        self.out_id_list = list()
        self.prev_center = defaultdict()
        self.records = list()
    
    def project_to_image(self, points, projection_matrix):
        num_pts = points.shape[1]
        points = np.vstack((points, np.ones((1, num_pts))))
        points = projection_matrix @ points
        points[:2, :] /= points[2, :]
        return points[:2, :]
    
    def radar_process(self, radar_points):
        radar_points = radar_points.rps
        points_3d = np.empty(shape=(0, 3))
        for p in radar_points:
            points_3d = np.append(points_3d, [[p.distX, p.distY, 1.0]], axis=0)

        points_2d = self.project_to_image(points_3d.transpose(), self.p)

        inds = np.where((points_2d[0, :] < self.image_width*0.9) & (points_2d[0, :] >= self.image_width*0.1)
                & (points_2d[1, :] < self.image_height) & (points_2d[1, :] >= 0)
                & (points_3d[:, 0] > 0)
                )[0]

        points_2d = points_2d[:, inds]
        points_3d = points_3d[inds, :]
        radar_points = [radar_points[i] for i in inds]
        return points_2d, points_3d, radar_points
    
    def map_scene_point(self, point_x, point_y, point_dist, v):
        if self.region_type == 'horizontal-C':
            if point_y < self.image_height / 2:
                scale_x_old_domain = (0, 2047)
                scale_x_new_domain = (0.75, 1.25)

                scale_y_old_domain = (15, 60)
                scale_y_new_domain = (0.8, 1.8)
                # old_point_y = point_y
                scale_y = pu.mapdomain(point_dist, scale_y_old_domain, scale_y_new_domain)
                point_y = int(self.image_height - point_y * scale_y)
                scale_x = pu.mapdomain(point_x, scale_x_old_domain, scale_x_new_domain)
                if scale_x > 1:
                    scale_x = max(scale_x, 1.1)
                else:
                    scale_x = max(0.8, scale_x)
                point_x = int(point_x * scale_x)
        elif self.region_type == 'horizontal-B':
            scale_x_old_domain = (0, 2047)
            scale_x_new_domain = (0.6, 1.4)

            scale_y_old_domain = (15, 60)
            scale_y_new_domain = (2.3, 3.0)
            scale_y = pu.mapdomain(point_dist, scale_y_old_domain, scale_y_new_domain)
            point_y = int(self.image_height - point_y * scale_y)
            scale_x = pu.mapdomain(point_x, scale_x_old_domain, scale_x_new_domain)
            # if scale_x > 1:
            #     scale_x = max(scale_x, 1.1)
            # else:
            #     scale_x = min(0.9, scale_x)
            point_x = int(point_x * scale_x)
        elif self.region_type == 'horizontal-A':
            scale_x_old_domain = (0, 2047)
            scale_x_new_domain = (0.5, 1.4)

            scale_y_old_domain = (10, 60)
            scale_y_new_domain = (1.5, 2.5)
            scale_y = pu.mapdomain(point_dist, scale_y_old_domain, scale_y_new_domain)
            point_y = int(self.image_height - point_y * scale_y)
            scale_x = pu.mapdomain(point_x, scale_x_old_domain, scale_x_new_domain)
            point_x = int(point_x * scale_x)
        elif self.region_type == 'tunnel-5':
            scale_x_old_domain = (0, 2047)
            scale_x_new_domain = (0.5, 1.35)

            scale_y_old_domain = (10, 70)
            scale_y_new_domain = (1.5, 1.8)
            scale_y = pu.mapdomain(point_dist, scale_y_old_domain, scale_y_new_domain)
            if point_dist >= 70 and point_dist <= 120:
                scale_y_old_domain = (70, 120)
                scale_y_new_domain = (1.8, 1.85)
                scale_y = pu.mapdomain(point_dist, scale_y_old_domain, scale_y_new_domain)
            point_y = int(self.image_height - point_y * scale_y)
            scale_x = pu.mapdomain(point_x, scale_x_old_domain, scale_x_new_domain)
            point_x = int(point_x * scale_x)
        elif self.region_type == 'tunnel-10':
            scale_x_old_domain = (0, 2047)
            scale_x_new_domain = (0.5, 1.35)

            scale_y_old_domain = (10, 70)
            scale_y_new_domain = (2.1, 2.6)
            scale_y = pu.mapdomain(point_dist, scale_y_old_domain, scale_y_new_domain)
            point_y = int(self.image_height - point_y * scale_y)
            scale_x = pu.mapdomain(point_x, scale_x_old_domain, scale_x_new_domain)
            point_x = int(point_x * scale_x)
        elif self.region_type == 'tunnel-15':
            scale_x_old_domain = (0, 2047)
            scale_x_new_domain = (0.5, 1.4)

            scale_y_old_domain = (10, 70)
            scale_y_new_domain = (2.8, 3)
            scale_y = pu.mapdomain(point_dist, scale_y_old_domain, scale_y_new_domain)
            point_y = int(self.image_height - point_y * scale_y)
            scale_x = pu.mapdomain(point_x, scale_x_old_domain, scale_x_new_domain)
            point_x = int(point_x * scale_x)
        elif self.region_type == 'center-down-17':
            scale_x_old_domain = (0, 2047)
            scale_x_new_domain = (0.05, 1.65)

            # scale_y_old_domain = [10, 25, 70]
            # scale_y_new_domain = [0.8, 2.2, 3.2]
            # index = max(i for i, num in enumerate(scale_y_old_domain) if num < point_dist)
            # if index < len(scale_y_old_domain) - 1:
            #     scale_y = pu.mapdomain(point_dist, scale_y_old_domain[index:index+2], scale_y_new_domain[index:index+2])
            #     point_y = int(self.image_height - point_y * scale_y)
            scale_x = pu.mapdomain(point_x, scale_x_old_domain, scale_x_new_domain)
            point_x = int(point_x * scale_x)
        
        # index = max(i for i, num in enumerate(self.scale_y_dist) if num < point_dist)
        # if index < len(self.scale_y_dist) - 1:
            # scale_y = pu.mapdomain(point_dist, self.scale_y_dist[index:index+2], self.scale_y[index:index+2])
        point_y = int(self.image_height - point_y * v)

        return point_x, point_y

    def imageCallback(self, image_msg):
        pass
        # print(image_msg.header.stamp)
    def radarCallback(self, objects):
        filtered_points = []
        # Iterate through the points in the PointCloud2 message
        for point in pc2.read_points(objects, field_names=("x", "y", "z", "vx", "vy"), skip_nans=True):
            x, y, z, vx, vy = point
            if (vx**2 + vy**2)**0.5 > 1:
                filtered_points.append([x, y, z, vx, vy])
        header = Header()
        header.stamp = rospy.Time.now()
        header.frame_id = objects.header.frame_id
        fields = [
            PointField('x', 0, PointField.FLOAT32, 1),
            PointField('y', 4, PointField.FLOAT32, 1),
            PointField('z', 8, PointField.FLOAT32, 1),
            PointField('vx', 12, PointField.FLOAT32, 1),
            PointField('vy', 16, PointField.FLOAT32, 1)
        ]

        filtered_cloud = pc2.create_cloud(header, fields, filtered_points)

        self.pub_radar_filter.publish(filtered_cloud)


        # print(radar_msg.timestamp_seconds, radar_msg.timestamp_nanoseconds)
    
    def objectListCallback(self, radarObjectList):
        markers = MarkerArray()

        # # clear previous markers
        markers.markers.append(Marker(
            header=Header(frame_id="base_link", stamp=rospy.Time.now()),
            action=Marker.DELETEALL
        ))


        # # radar points
        id = 0
        for obj in radarObjectList.objectlist_objects:

            speed = math.sqrt(pow(obj.f_dynamics_absvel_x, 2)+pow(obj.f_dynamics_absvel_y,2))
            if speed < 1:
                continue
            # ===============================================================================
            # marker = Marker(
            #         header=Header(frame_id="ARS_548", stamp=rospy.Time.now()),
            #         id=id,
            #         ns="front_center",
            #         type=Marker.CYLINDER,
            #         action=Marker.ADD,
            #         pose=Pose(
            #             position=Point(x=obj.u_position_x, y=obj.u_position_y, z=obj.u_position_z),
            #             orientation=Quaternion(x=0, y=0, z=1)
            #         ),
            #         scale=Vector3(x=1, y=1, z=1.5),
            #         color=ColorRGBA(r=0.0, g=0.0, b=0.9, a=1.0)
            #         )
            # markers.markers.append(marker)
            # dist = math.sqrt(pow(obj.distX, 2)+pow(obj.distY,2))
            width = obj.u_shape_width_edge_mean
            length = obj.u_shape_length_edge_mean
            text_marker = Marker(
                header=Header(frame_id="ARS_548", stamp=rospy.Time.now()),
                id=id,
                ns="front_center_text",
                type=Marker.TEXT_VIEW_FACING,
                action=Marker.ADD,
                # text=f"id: {obj.u_id} cls: {obj.classT}\n{dist:.2f}m\n{speed*3.6:.2f}km/h\n{obj.vrelX:.2f} {obj.vrelY:.2f}",
                text=f"id: {obj.u_id}\n WxL:{width:.2f}x{length:.2f}m\n{speed*3.6:.2f}km/h\n",
                pose=Pose(
                    position=Point(x=obj.u_position_x, y=obj.u_position_y, z=obj.u_position_z),
                    orientation=Quaternion(x=0, y=0, z=1)
                ),
                scale=Vector3(x=2, y=2, z=2),
                color=ColorRGBA(r=0.0, g=1.0, b=0.0, a=1.0),
            )
            markers.markers.append(text_marker)
            id = id + 1
        
        self.pub_radar_marker.publish(markers)

    def callback(self, image_msg, radar_points):
        print("@"*10)
        start_time = rospy.Time.now()
        cv_image = self.bridge.imgmsg_to_cv2(image_msg)

        if self.display_radar_image:
            radar_image = deepcopy(cv_image)

        # object detection & object tracking
        res = self.mot_predictor.predict_image(
            [deepcopy(cv_image)],
            visual=False,
            reuse_det_result=False,
            frame_count=self.frame_id,
            run_benchmark=False)

        mot_res = parse_mot_res(res) # [i, cls_id, score, xmin, ymin, xmin + w, ymin + h]
        
        # """
        new_mos_res = [None if i[1] != 7 else i for i in mot_res['boxes']]
        new_mos_res = [i for i in new_mos_res if i is not None]
        mot_res['boxes'] = new_mos_res
        print(mot_res['boxes'])
        # """
        
        mot_time = rospy.Time.now()

        # radar to image projection
        box_radar_mapping = [[] for i in range(len(mot_res['boxes']))]
        points_2d, points_3d, radar_points = self.radar_process(radar_points)

        # radar and camera late fusion
        for p in range(points_2d.shape[1]):
            if radar_points[p].id != 22:
                continue
            depth = (80 - points_3d[p, 0]) / 80
            length = max(int(80 * depth), 40)
            
            point_x = int(points_2d[0, p])
            point_y = int(points_2d[1, p])

            point_speed = np.sqrt(pow(radar_points[p].vrelX, 2) + pow(radar_points[p].vrelY, 2)) # radar point speed in m/s
            point_dist = np.sqrt(pow(radar_points[p].distX, 2) + pow(radar_points[p].distY, 2))


            #if self.hard_postprocess:
            # print(point_dist, self.scale_y_dist[self.optimizer_index])
            rospy.loginfo_throttle(5, "-"*10)
            rospy.loginfo_throttle(5, self.optimizer_count)
            rospy.loginfo_throttle(5, self.optimizer_index)
            rospy.loginfo_throttle(5, "-"*10)
            if (point_dist - self.scale_y_dist[self.optimizer_index]) > 1 and self.optimizer_count[self.optimizer_index] < self.optimizer_step and len(mot_res['boxes']) == 1:
                print(f'optimizing {self.scale_y_dist[self.optimizer_index]}')
                target_x, target_y = (mot_res['boxes'][0][3]+mot_res['boxes'][0][5])/2, (mot_res['boxes'][0][4]+mot_res['boxes'][0][6])/2
                best_v = 0
                best_error = 1e9
                for v in np.arange(0.1, 4, 0.05):
                    n_point_x, n_point_y = self.map_scene_point(point_x, point_y, point_dist, v)
                    e = math.sqrt(pow(target_x-n_point_x, 2) + pow(target_y-n_point_y,2))
                    if e < best_error:
                        best_error = e
                        best_v = v
                self.scale_y[self.optimizer_index] += best_v
                self.optimizer_count[self.optimizer_index] = self.optimizer_count[self.optimizer_index] + 1

                if self.optimizer_count[self.optimizer_index] >= self.optimizer_step:
                    self.scale_y[self.optimizer_index] /= self.optimizer_step
                    self.optimizer_index = self.optimizer_index+1

                if self.optimizer_index >= len(self.scale_y_dist):
                    print('-'*30)
                    print('result:')
                    print(self.scale_y_dist)
                    print(self.scale_y)
                    exit(0)

            
            if point_dist > 10 and point_dist < 70 and point_speed > 1:
                for i, (box_id, cls_id, score, xmin, ymin, xmax, ymax) in enumerate(mot_res['boxes']):
                    margin = 1e5
                    w, h = xmax - xmin, ymax - ymin
                    if point_x > (xmin - w / margin) and point_x < (xmax + w / margin) \
                        and point_y > (ymin - h / margin) and point_y < (ymax + h / margin):
                        box_radar_mapping[i].append(RadarPoint(point_x, point_y, point_speed, point_dist))
                
                if self.display_radar_image:
                    cv2.putText(radar_image, f"{point_speed * 3.6:.1f} km/h", (point_x, point_y), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2, cv2.LINE_AA)
                    cv2.putText(radar_image, f"{point_dist:.1f} m", (point_x, point_y+50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2, cv2.LINE_AA)
                    # cv2.putText(radar_image, f"{radar_points[p].classT}", (point_x, point_y+100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 4, cv2.LINE_AA)
                    cv2.line(radar_image, (point_x, point_y+length), (point_x, point_y), (0, int(255 * abs(depth)), 50), thickness=6)

        radar_img_msg = self.bridge.cv2_to_imgmsg(radar_image)
        self.pub_radar.publish(radar_img_msg)
        return
        # flow statistic
        statistic = flow_statistic_multi_class(
            (self.frame_id + 1, mot_res['boxes']),
            self.secs_interval,
            self.do_entrance_counting,
            self.do_break_in_counting,
            self.region_type,
            self.video_fps,
            self.entrance,
            self.id_set,
            self.interval_id_set,
            self.in_id_list,
            self.out_id_list,
            self.prev_center,
            self.records,
            ids2names=self.mot_predictor.pred_config.labels)
        records = statistic['records']
        self.pipeline_res.update(mot_res, 'mot')
        self.collector.append(self.frame_id, self.pipeline_res)
        flow_time = rospy.Time.now()

        mot_img = cv_image.copy()

        # plot radar dist and speed on mot_img
        current_frame_mean_speed = []
        current_frame_occupancy = []
        if self.display_radar_rgb_fusion:
            for i, (box_id, cls_id, score, xmin, ymin, xmax, ymax) in enumerate(mot_res['boxes']):
                current_frame_occupancy.append((xmax-xmin)*(ymax-ymin))
                if len(box_radar_mapping[i]):
                    avg = calculate_radar_points_averages(box_radar_mapping[i])
                    current_frame_mean_speed.append(avg.point_speed)
                    # cv2.putText(radar_image, f"{cls} {avg.point_dist:.1f}m {avg.point_speed * 3.6:.1f}km/h", (int(x - w / 2), int(y - h / 2)-15), cv2.FONT_HERSHEY_SIMPLEX, .9, (0, 255, 0), 2, cv2.LINE_AA)
                    # cv2.putText(radar_image, f"{avg.point_dist:.1f}m {avg.point_speed * 3.6:.1f}km/h", (int(x - w / 2), int(y - h / 2)-15), cv2.FONT_HERSHEY_SIMPLEX, .9, (0, 255, 0), 2, cv2.LINE_AA)
                    cv2.putText(mot_img, f"{avg.point_dist:.1f}m {avg.point_speed * 3.6:.1f}km/h", (int(xmin), int(ymin)-6), cv2.FONT_HERSHEY_SIMPLEX, .9, (0, 255, 0), 2, cv2.LINE_AA)
        
        current_frame_occupancy = np.sum(current_frame_occupancy)
        current_frame_occupancy_ratio = min(current_frame_occupancy / (self.image_height * self.image_width * self.effective_road_ratio), 1.0)

        # stats traffic speed and occupancy ratio
        self.traffic_stats.add_record({ 'speed': current_frame_mean_speed, 'occupancy_ratio': current_frame_occupancy_ratio })

        # interval
        if (self.frame_id + 1) % self.video_fps == 0 and (self.frame_id + 1) / self.video_fps % self.secs_interval == 0:
            flow = len(self.in_id_list) + len(self.out_id_list)
            self.in_id_list.clear()
            self.out_id_list.clear()
            self.id_set.clear()
            msg = String()
            speed = self.traffic_stats.mean_speed() * 3.6
            occupancy_ratio = self.traffic_stats.mean_occupancy_ratio()
            self.traffic_stats.reset()
            congestion_level = self.calculate_congestion_level(flow, speed)
            self.traffic_stats.add_interval({'flow': flow, 'speed': speed, 'occupancy_ratio': occupancy_ratio, 'congestion_level': congestion_level})
            msg.data += f'Traffic flow: {flow}, Traffic speed: {speed:.1f} km/h, Congestion level: {congestion_level}, Occupancy ratio: {int(occupancy_ratio*100%100)}%'
            self.pub_stats.publish(msg)

        # visualize mot result
        mot_img = self.visualize_video(mot_img, self.pipeline_res,
                                    self.collector, self.frame_id, self.video_fps,
                                    self.entrance, records, self.center_traj,
                                    False, {})

        if self.plot_traffic_on_mot:
            height, width, _ = mot_img.shape
            new_height = height + 100
            new_img = np.zeros((new_height, width, 3), dtype=np.uint8)
            new_img[new_height-height:new_height,0:width] = mot_img
            if len(self.traffic_stats.traffic_history) != 0:
                last_traffic_stat = self.traffic_stats.traffic_history[-1]
                flow, speed, occupancy_ratio, congestion_level, update_time = list(last_traffic_stat.values())
                last_update_time = rospy.Time.to_sec(rospy.Time.now() - update_time)
                cv2.putText(new_img, f'Traffic flow: {flow}', (20, 30), cv2.FONT_ITALIC, 1.2, (0, 0, 255), 4)
                cv2.putText(new_img, f'Traffic speed: {speed:.1f} km/h', (20, 75), cv2.FONT_ITALIC, 1.2, (0, 0, 255), 4)
                cv2.putText(new_img, f'Congestion level: {congestion_level}', (620, 30), cv2.FONT_ITALIC, 1.2, (0, 0, 255), 4)
                cv2.putText(new_img, f'Occupancy ratio: {int(occupancy_ratio*100%100)}%', (620, 75), cv2.FONT_ITALIC, 1.2, (0, 0, 255), 4)
                cv2.putText(new_img, f'Last updated: {last_update_time:.1f} s', (1120, 75), cv2.FONT_ITALIC, 1.2, (0, 0, 255), 4)
            else:
                cv2.putText(new_img, f'Initiating traffic stats', (20, 30), cv2.FONT_ITALIC, 1.2, (0, 0, 255), 4)
            mot_img = new_img

        mot_img_msg = self.bridge.cv2_to_imgmsg(mot_img)
        self.pub_mot.publish(mot_img_msg)
        if self.display_radar_image:
            radar_img_msg = self.bridge.cv2_to_imgmsg(radar_image)
            self.pub_radar.publish(radar_img_msg)
        self.frame_id = self.frame_id + 1
        vis_time = rospy.Time.now()

        rospy.loginfo_throttle(5, self.records[-1]) # Frame id: <id>, Total count: <len(id_set)>

        try:
            rospy.loginfo_throttle(10, f'mot_time {(mot_time - start_time) / 1e6}ms')
            rospy.loginfo_throttle(10, f'flow_time {(flow_time - mot_time) / 1e6}ms')
            rospy.loginfo_throttle(10, f'vis_time {(vis_time - flow_time) / 1e6}ms')
            rospy.loginfo_throttle(10, f'duration {(vis_time - start_time) / 1e6}ms')
        except:
            pass
    
    def calculate_congestion_level(self, traffic_flow, mean_speed):
        # Determine congestion level based on traffic flow
        if traffic_flow < self.flow_thresholds[0]:
            congestion_level = 1
        elif traffic_flow < self.flow_thresholds[1]:
            congestion_level = 2
        elif traffic_flow < self.flow_thresholds[2]:
            congestion_level = 3
        else:
            congestion_level = 4
    
        # Adjust congestion level based on mean speed
        if mean_speed < self.speed_thresholds[0]:
            congestion_level = max(congestion_level, 4)
        elif mean_speed < self.speed_thresholds[1]:
            congestion_level = max(congestion_level, 3)
        elif mean_speed < self.speed_thresholds[2]:
            congestion_level = max(congestion_level, 2)
        else:
            congestion_level = 1
    
        return congestion_level

    def visualize_video(self,
                        image_rgb,
                        result,
                        collector,
                        frame_id,
                        fps,
                        entrance=None,
                        records=None,
                        center_traj=None,
                        do_illegal_parking_recognition=False,
                        illegal_parking_dict=None,
                        num_classes=80):
        image = image_rgb
        mot_res = deepcopy(result.get('mot'))

        if mot_res is not None and len(mot_res['boxes']) > 0:
            ids = deepcopy(mot_res['boxes'][:, 0])
            scores = deepcopy(mot_res['boxes'][:, 2])
            boxes = deepcopy(mot_res['boxes'][:, 3:])
            boxes[:, 2] = boxes[:, 2] - boxes[:, 0]
            boxes[:, 3] = boxes[:, 3] - boxes[:, 1]
            multi_class_dict = {"num_classes": num_classes, "mot_res": mot_res["boxes"]}
        else:
            boxes = np.zeros([0, 4])
            ids = np.zeros([0])
            scores = np.zeros([0])
            multi_class_dict = {"num_classes": num_classes, "mot_res": []}


        # single class, still need to be defaultdict type for ploting
        num_classes = 1
        online_tlwhs = defaultdict(list)
        online_scores = defaultdict(list)
        online_ids = defaultdict(list)
        online_tlwhs[0] = boxes
        online_scores[0] = scores
        online_ids[0] = ids

        if True or mot_res is not None:
            image = plot_tracking_dict(
                image,
                num_classes,
                online_tlwhs,
                online_ids,
                online_scores,
                frame_id=frame_id,
                fps=fps,
                ids2names=self.mot_predictor.pred_config.labels,
                do_entrance_counting=self.do_entrance_counting,
                do_break_in_counting=self.do_break_in_counting,
                do_illegal_parking_recognition=do_illegal_parking_recognition,
                illegal_parking_dict=illegal_parking_dict,
                entrance=entrance,
                records=records,
                center_traj=center_traj,
                multi_class_dict=multi_class_dict)

        return image

if __name__ == "__main__":
    try:
        PaddleDetector()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass