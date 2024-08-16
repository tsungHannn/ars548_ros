#!/usr/bin/env python

import rospy
import cv2
import numpy as np
from sensor_msgs.msg import PointCloud2, Image, PointField
from std_msgs.msg import Header
from sensor_msgs import point_cloud2
from cv_bridge import CvBridge, CvBridgeError
import sensor_msgs.point_cloud2 as pc2
import message_filters
from ars548_messages.msg import ObjectList

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
        # 看來校正過的參數完全沒有用
        self.camera_matrix = np.array([
            [1804.85904,          0, 1028.16764],
            [         0, 1801.73036, 776.526745],
            [         0,          0,          1]
            ])
        self.extrinsics_matrix = np.array([
                    [ 0.0622288,  -0.996151, -0.0617304,   0.422767],
                    [-0.0448152,  0.0589988,  -0.997252,    1.46257],
                    [  0.997055,  0.0648243, -0.0409714,  -0.503814],
                    [         0,          0,          0,          1]
        ])


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

        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber('/aravis_cam/image_color_row', Image, self.image_callback)
        self.lidar_sub = rospy.Subscriber('/radar_filter', PointCloud2, self.lidar_callback) # 把filter過得雷達點project到RGB上

        self.radar_pc = message_filters.Subscriber('/radar/point_cloud_object', PointCloud2)
        self.radar_obj = message_filters.Subscriber('/radar/object_list', ObjectList) # ARS548 object_list
        self.sync = message_filters.ApproximateTimeSynchronizer([self.radar_pc, self.radar_obj], queue_size=10, slop=0.1)
        self.sync.registerCallback(self.radarCallback)

        self.image_pub = rospy.Publisher('/projected_image', Image, queue_size=1)
        # self.pub_radar_filter = rospy.Publisher('/radar_filter', PointCloud2, queue_size=10)
        self.pub_radar_obj = rospy.Publisher('/radar_filter', PointCloud2, queue_size=10)

        self.current_image = None
        self.lidar_points = np.empty((0, 3))

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
        
        filtered_points_pc = []
        filtered_points_obj = []

        # Iterate through the points in the PointCloud2 message
        for obj in radar_obj.objectlist_objects:
            id = obj.u_id
            x = obj.u_position_x
            y = obj.u_position_y
            z = obj.u_position_z
            vx = obj.f_dynamics_absvel_x
            vy = obj.f_dynamics_absvel_y
            if (vx**2 + vy**2)**0.5 > 1:
                filtered_points_obj.append([x, y, z, vx, vy, id])

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

        filtered_cloud_obj = pc2.create_cloud(header, fields, filtered_points_obj)

        self.pub_radar_obj.publish(filtered_cloud_obj)

        # print(radar_msg.timestamp_seconds, radar_msg.timestamp_nanoseconds)

    def project_lidar_to_image(self):
        if self.current_image is None or self.lidar_points.size == 0:
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
