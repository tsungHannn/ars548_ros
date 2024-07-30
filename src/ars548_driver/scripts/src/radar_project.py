#!/usr/bin/env python

import rospy
import cv2
import numpy as np
from sensor_msgs.msg import PointCloud2, Image
from sensor_msgs import point_cloud2
from cv_bridge import CvBridge, CvBridgeError
"""
camera matrix
[2033,    0, 1068]
[0   , 2056,  539]
[0   ,    0,    1]
"""
class LidarToImageProjection:
    def __init__(self):
        rospy.init_node('lidar_to_image_projection', anonymous=True)

        self.camera_matrix = np.array([
            [2033, 0, 1068],
            [0, 2056, 539],
            [0, 0, 1]
        ])

        self.extrinsics_matrix = np.array([
            [0.00130471,  -0.999894,  0.0145059,   0.970467],
            [ 0.0693163, -0.0143808,  -0.997491,    1.88237],
            [  0.997594, 0.00230699,  0.0692901,    0.53553],
            [         0,          0,          0,          1]
        ])

        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber('/aravis_cam/image_color_row', Image, self.image_callback)
        self.lidar_sub = rospy.Subscriber('/radar_filter', PointCloud2, self.lidar_callback)
        self.image_pub = rospy.Publisher('/projected_image', Image, queue_size=1)

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
                cv2.circle(self.current_image, (x, y), 5, (0, 255, 0), -1)

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
