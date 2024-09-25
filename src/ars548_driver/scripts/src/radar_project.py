#!/usr/bin/env python

import rospy
import cv2
import numpy as np
from sensor_msgs.msg import PointCloud2, Image, PointField
from std_msgs.msg import Header, ColorRGBA
from sensor_msgs import point_cloud2
from cv_bridge import CvBridge, CvBridgeError
import sensor_msgs.point_cloud2 as pc2
import message_filters
from ars548_messages.msg import ObjectList
from visualization_msgs.msg import MarkerArray, Marker
from geometry_msgs.msg import Pose, Point, Vector3, Quaternion
import math

"""
camera matrix
[2033,    0, 1068]
[0   , 2056,  539]
[0   ,    0,    1]
"""
class LidarToImageProjection:
    def __init__(self):
        rospy.init_node('lidar_to_image_projection', anonymous=True)
        
        # 新的內參
        self.camera_matrix = np.array([
            [1804.85904,          0, 1028.16764],
            [         0, 1801.73036, 776.526745],
            [         0,          0,          1]
        ])
        # 原視角
        # self.extrinsics_matrix = np.array([
        #             [ 0.0622288,  -0.996151, -0.0617304,   0.422767],
        #             [-0.0448152,  0.0589988,  -0.997252,    1.46257],
        #             [  0.997055,  0.0648243, -0.0409714,  -0.503814],
        #             [         0,          0,          0,          1]
        # ])
        # cal6.bag
        self.extrinsics_matrix = np.array([
            [-0.0498244,  -0.995333, -0.0826376,  -0.240135],
            [ 0.0091128,  0.0822836,  -0.996568,    3.54319],
            [  0.998716, -0.0504063,  0.0049704,   -0.53661],
            [         0,          0,          0,          1]

        ])

        


        
        # self.extrinsics_matrix = np.array([
        #     [ 0.0622288,  -0.997539,  0.0322897,   0.422767],
        #     [-0.0448152, -0.0351127,  -0.998378,    1.46257],
        #     [  0.997055,  0.0606809, -0.0468901,  -0.503814],
        #     [         0,          0,          0,          1]
        # ])

        # # 原本的內參 + 校正
        # self.camera_matrix = np.array([
        #         [2033,    0, 1068],
        #         [0   , 2056,  539],
        #         [0   ,    0,    1]])
        # self.extrinsics_matrix = np.array([
        #     [0.00130471,  -0.999894,  0.0145059,   0.970467],
        #     [ 0.0693163, -0.0143808,  -0.997491,    1.88237],
        #     [  0.997594, 0.00230699,  0.0692901,    0.53553],
        #     [         0,          0,          0,          1]
        # ])
        self.frameID = 0
        

        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber('/aravis_cam/image_color_row', Image, self.image_callback)
        self.lidar_sub = rospy.Subscriber('/radar_filter', PointCloud2, self.lidar_callback) # 把filter過得雷達點project到RGB上
        self.pub_radar_marker = rospy.Publisher('/radar_marker', MarkerArray, queue_size=1)
        self.pub_range = rospy.Publisher("/radar/range", MarkerArray, queue_size=1)
        
        self.radar_pc = message_filters.Subscriber('/radar/point_cloud_object', PointCloud2)
        self.radar_obj = message_filters.Subscriber('/radar/object_list', ObjectList) # ARS548 object_list
        self.sync = message_filters.ApproximateTimeSynchronizer([self.radar_pc, self.radar_obj], queue_size=10, slop=0.1)
        self.sync.registerCallback(self.radarCallback)

        self.image_pub = rospy.Publisher('/projected_image', Image, queue_size=1)
        # self.pub_radar_filter = rospy.Publisher('/radar_filter', PointCloud2, queue_size=10)

        # 底下兩個都跟原本的雷達資料不同
        # 包含(x, y, z, vx, vy, id)
        self.pub_radar_filter = rospy.Publisher('/radar_filter', PointCloud2, queue_size=10)
        self.pub_radar_object = rospy.Publisher('/radar_object', PointCloud2, queue_size=10)   # 拿來訓練用的

        self.current_image = None
        self.lidar_points = np.empty((0, 3))

        self.drawRadarRange()

    def drawRadarRange(self):
        # radar range
        range_markers = MarkerArray()
        id = 0

        # # ego arrow
        # range_markers.markers.append(
        #     Marker(
        #         header=Header(frame_id="ARS_548", stamp=rospy.Time.now()),
        #         id=0,
        #         ns="ego_arrow",
        #         type=Marker.ARROW,
        #         action=Marker.ADD,
        #         scale=Vector3(x=5, y=0.5, z=0.5),
        #         color=ColorRGBA(r=1.0, b=0.5, g=0.0, a=1.0)
        #     )
        # )

        i = 0
        for t in [[0, 0, 0]]:
            radar_transform = t
            range_marker = Marker(
                header=Header(frame_id="ARS_548", stamp=rospy.Time.now()),
                id=id,
                ns="range_marker_wide",
                type=Marker.LINE_STRIP,
                action=Marker.ADD,
                pose=Pose(
                    position=Point(x=0, y=0, z=0.1),
                    orientation=Quaternion(x=0, y=0, z=0, w=1.0)
                ),
                scale=Vector3(x=0.5, y=0.1, z=0.1),
                color=ColorRGBA(r=0.0, g=0.0, b=1.0, a=1.0)
            )
            id = id + 1
            # p = Point(z=0.5)
            # range_marker.points.append(Point(x=0, y=0, z=0.5))
            # range_marker.points.append(Point(x=1, y=0, z=0.5))
            # range_marker.points.append(Point(x=2, y=0, z=0.5))
            # range_marker.points.append(Point(x=2, y=1, z=0.5))
            # range_marker.points.append(Point(x=2, y=2, z=0.5))
            # range_marker.points.append(Point(x=0, y=0, z=0.5))

            # wide range
            rotate = -40 * math.pi / 180.0 + radar_transform[2]
            range_marker.points.append(Point(
                x=math.cos(rotate) * 70 - math.sin(rotate) * 0,
                y=math.sin(rotate) * 70 + math.cos(rotate) * 0
            ))
            rotate = -46 * math.pi / 180.0 + radar_transform[2]
            range_marker.points.append(Point(
                x=math.cos(rotate) * 35 - math.sin(rotate) * 0,
                y=math.sin(rotate) * 35 + math.cos(rotate) * 0
            ))
            range_marker.points.append(Point(
                x=0 + radar_transform[0],
                y=0 + radar_transform[1]
            ))
            rotate = 46 * math.pi / 180.0 + radar_transform[2]
            range_marker.points.append(Point(
                x=math.cos(rotate) * 35 - math.sin(rotate) * 0,
                y=math.sin(rotate) * 35 + math.cos(rotate) * 0
            ))
            rotate = 40 * math.pi / 180.0 + radar_transform[2]
            range_marker.points.append(Point(
                x=math.cos(rotate) * 70 - math.sin(rotate) * 0,
                y=math.sin(rotate) * 70 + math.cos(rotate) * 0
            ))
            for i in range(40, -41, -5):
                rotate = i * math.pi / 180.0 + radar_transform[2]
                range_marker.points.append(Point(
                    x=math.cos(rotate) * 70 - math.sin(rotate) * 0,
                    y=math.sin(rotate) * 70 + math.cos(rotate) * 0
                ))
            range_markers.markers.append(range_marker)

            # narrow range
            range_marker = Marker(
                header=Header(frame_id="ARS_548", stamp=rospy.Time.now()),
                id=id,
                ns="range_marker_narrow",
                type=Marker.LINE_STRIP,
                action=Marker.ADD,
                pose=Pose(
                    position=Point(x=0, y=0, z=0.1),
                    orientation=Quaternion(x=0, y=0, z=0, w=1.0)
                ),
                scale=Vector3(x=0.5, y=0.1, z=0.1),
                color=ColorRGBA(r=1.0, g=0.0, b=0.0, a=1.0)
            )
            id = id + 1
            
            rotate = 4 * math.pi / 180.0 + radar_transform[2]
            range_marker.points.append(Point(
                x=math.cos(rotate) * 250 - math.sin(rotate) * 0,
                y=math.sin(rotate) * 250 + math.cos(rotate) * 0
            ))
            rotate = 9 * math.pi / 180.0 + radar_transform[2]
            range_marker.points.append(Point(
                x=math.cos(rotate) * 150 - math.sin(rotate) * 0,
                y=math.sin(rotate) * 150 + math.cos(rotate) * 0
            ))
            range_marker.points.append(Point(
                x=0 + radar_transform[0],
                y=0 + radar_transform[1]
            ))
            rotate = -9 * math.pi / 180.0 + radar_transform[2]
            range_marker.points.append(Point(
                x=math.cos(rotate) * 150 - math.sin(rotate) * 0,
                y=math.sin(rotate) * 150 + math.cos(rotate) * 0
            ))
            rotate = -4 * math.pi / 180.0 + radar_transform[2]
            range_marker.points.append(Point(
                x=math.cos(rotate) * 250 - math.sin(rotate) * 0,
                y=math.sin(rotate) * 250 + math.cos(rotate) * 0
            ))
            rotate = 4 * math.pi / 180.0 + radar_transform[2]
            range_marker.points.append(Point(
                x=math.cos(rotate) * 250 - math.sin(rotate) * 0,
                y=math.sin(rotate) * 250 + math.cos(rotate) * 0
            ))
            range_markers.markers.append(range_marker)
        self.pub_range.publish(range_markers)


    def image_callback(self, data):
        try:
            self.current_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            rospy.logerr(e)

    def lidar_callback(self, data):
        self.lidar_points = []
        for point in point_cloud2.read_points(data, skip_nans=True):
            self.lidar_points.append([point[0], point[1], point[2]])
        self.lidar_points = np.array(self.lidar_points)
        self.project_lidar_to_image()


    def radarCallback(self, radar_pc, radar_obj):
        
        radar_point_filter = [] # 只留速度大於1的
        radar_point_object = []
        # print(self.frameID)
        # self.frameID += 1

        markers = MarkerArray()
        # # clear previous markers
        markers.markers.append(Marker(
            header=Header(frame_id="base_link", stamp=rospy.Time.now()),
            action=Marker.DELETEALL
        ))

        # Iterate through the points in the PointCloud2 message
        for obj in radar_obj.objectlist_objects:
            id = obj.u_id
            x = obj.u_position_x
            y = obj.u_position_y
            z = obj.u_position_z
            vx = obj.f_dynamics_absvel_x
            vy = obj.f_dynamics_absvel_y





            classification = ['car', 'truck', 'motorcycle', 'bicycle', 'pedestrian', 'animal', 'hazard', 'unknown']
            prob = [obj.u_classification_car, obj.u_classification_truck, obj.u_classification_motorcycle,
                    obj.u_classification_bicycle, obj.u_classification_pedestrian, obj.u_classification_animal,
                    obj.u_classification_hazard, obj.u_classification_unknown]
            
            max_index = prob.index(max(prob))
            # print(classification[max_index])

            # 畫雷達範圍
            self.drawRadarRange()
            # effective range
            effective_range = True

            effective_range = False
            angle = math.atan2(y, x) * 180 / math.pi
            dist = math.sqrt(x ** 2 + y ** 2)

            if (dist < (10 / math.cos(60 * math.pi / 180)) and abs(angle) < 60) or \
            (dist < 70 and abs(angle) < 40) or \
            (dist < 150 and abs(angle) < 9) or \
            (dist < 250 and abs(angle) < 4):
                effective_range = True
            elif x > 10 and x < 70 * math.cos(40 * math.pi / 180):
                delta_x = (x - 10) / (70 * math.cos(40 * math.pi / 180) - 10)
                delta_y = delta_x * (70 * math.sin(40 * math.pi / 180) - (10 / math.cos(60 * math.pi / 180) * math.sin(60 * math.pi / 180))) + \
                        (10 / math.cos(60 * math.pi / 180) * math.sin(60 * math.pi / 180))
                if abs(y) < delta_y:
                    effective_range = True
            elif x > 150 and x < 250:
                delta_x = (x - 150) / (250 - 150)
                delta_y = delta_x * ((250 / math.cos(4 * math.pi / 180) * math.sin(4 * math.pi / 180)) - \
                                    (150 / math.cos(9 * math.pi / 180) * math.sin(9 * math.pi / 180))) + \
                        (150 / math.cos(9 * math.pi / 180) * math.sin(9 * math.pi / 180))
                if abs(y) < delta_y:
                    effective_range = True
            

            if effective_range == True:
                speed = math.sqrt(pow(obj.f_dynamics_absvel_x, 2)+pow(obj.f_dynamics_absvel_y,2))
                radar_point_object.append([x, y, z, vx, vy, id])

                if speed > 5:
                    radar_point_filter.append([x, y, z, vx, vy, id])

                    # 把資訊印在rviz
                    width = obj.u_shape_width_edge_mean
                    length = obj.u_shape_length_edge_mean
                    text_marker = Marker(
                        header=Header(frame_id="ARS_548", stamp=rospy.Time.now()),
                        id=id,
                        ns="front_center_text",
                        type=Marker.TEXT_VIEW_FACING,
                        action=Marker.ADD,
                        # text=f"id: {obj.u_id} cls: {obj.classT}\n{dist:.2f}m\n{speed*3.6:.2f}km/h\n{obj.vrelX:.2f} {obj.vrelY:.2f}",
                        # text=f"id: {obj.u_id}\n WxL:{width:.2f}x{length:.2f}m\n{speed*3.6:.2f}km/h\nClass:{classification[max_index]}",
                        # text=f"id: {obj.u_id}\n{speed*3.6:.2f}km/h\nClass:{classification[max_index]}",
                        text=f"x:{obj.u_position_x}\ny:{obj.u_position_y}\nz:{obj.u_position_z}",
                        pose=Pose(
                            position=Point(x=obj.u_position_x, y=obj.u_position_y, z=obj.u_position_z),
                            orientation=Quaternion(x=0, y=0, z=1)
                        ),
                        scale=Vector3(x=2, y=2, z=2),
                        color=ColorRGBA(r=0.0, g=1.0, b=0.0, a=1.0),
                    )
                    markers.markers.append(text_marker)

        # for point in pc2.read_points(radar_pc, field_names=("x", "y", "z", "vx", "vy", "id"), skip_nans=True):
        #     x, y, z, vx, vy = point
        #     if (vx**2 + vy**2)**0.5 > 1:
        #         filtered_points_pc.append([x, y, z, vx, vy])

        # # pc
        # header = Header()
        # header.stamp = rospy.Time.now()
        # header.frame_id = radar_pc.header.frame_id
        # fields = [
        #     PointField('x', 0, PointField.FLOAT32, 1),
        #     PointField('y', 4, PointField.FLOAT32, 1),
        #     PointField('z', 8, PointField.FLOAT32, 1),
        #     PointField('vx', 12, PointField.FLOAT32, 1),
        #     PointField('vy', 16, PointField.FLOAT32, 1)
        # ]

        # filtered_cloud = pc2.create_cloud(header, fields, filtered_points_pc)
        # self.pub_radar_filter.publish(filtered_cloud)

        # obj
        header = Header()
        header.stamp = rospy.Time.now()
        header.frame_id = radar_pc.header.frame_id
        fields = [
            PointField('x', 0, PointField.FLOAT32, 1),
            PointField('y', 4, PointField.FLOAT32, 1),
            PointField('z', 8, PointField.FLOAT32, 1),
            PointField('vx', 12, PointField.FLOAT32, 1),
            PointField('vy', 16, PointField.FLOAT32, 1),
            PointField('id', 20, PointField.FLOAT32, 1)
        ]

        point_cloud_filter = pc2.create_cloud(header, fields, radar_point_filter)
        point_cloud_object = pc2.create_cloud(header, fields, radar_point_object)

        self.pub_radar_filter.publish(point_cloud_filter)
        self.pub_radar_object.publish(point_cloud_object)
        self.pub_radar_marker.publish(markers)
        

    def project_lidar_to_image(self):
        if self.current_image is None:
            return

        if self.lidar_points.size == 0:
            try:
                image_message = self.bridge.cv2_to_imgmsg(self.current_image, "bgr8")
                self.image_pub.publish(image_message)
            except CvBridgeError as e:
                rospy.logerr(e)

            return

        lidar_points_homogeneous = np.hstack((self.lidar_points, np.ones((self.lidar_points.shape[0], 1))))
        camera_coordinates = self.extrinsics_matrix @ lidar_points_homogeneous.T
        camera_coordinates = camera_coordinates[:3, :]

        pixel_coordinates = self.camera_matrix @ camera_coordinates
        pixel_coordinates /= pixel_coordinates[2, :]

        pixel_coordinates = pixel_coordinates[:2, :].T

        for point in pixel_coordinates:
            x, y = int(point[0]), int(point[1])
            if 0 <= x < self.current_image.shape[1] and 0 <= y < self.current_image.shape[0]:
                cv2.circle(self.current_image, (x, y), 10, (0, 255, 0), -1)

        try:
            image_message = self.bridge.cv2_to_imgmsg(self.current_image, "bgr8")
            self.image_pub.publish(image_message)
        except CvBridgeError as e:
            rospy.logerr(e)

if __name__ == '__main__':
    try:
        LidarToImageProjection()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
