import rosbag
from rospy import Time
import os


input_bag = '/HDD/ncsist/0919/7.bag'  # 输入rosbag文件路径
output_bag = '/HDD/ncsist/0919/sync_7.bag'  # 输出对齐后的rosbag文件路径

# 读取rosbag并对齐时间戳
def align_timestamps(input_bag_path, output_bag_path):
    with rosbag.Bag(input_bag_path, 'r') as inbag, rosbag.Bag(output_bag_path, 'w') as outbag:

        # 读取所有消息，并分别存储不同话题的消息
        for topic, msg, t in inbag.read_messages():
            if topic == '/aravis_cam/image_color/compressed':

                msg.header.stamp = t
                # cam_msgs.append((msg, t))
            # elif topic == '/radar/object_list':
            #     radar_msgs.append((msg, t))
            # 其他消息直接写入新的bag文件
            # else:
            outbag.write(topic, msg, t)

        # # 开始对齐时间戳
        # cam_idx = 0
        # radar_idx = 0

        # while cam_idx < len(cam_msgs) and radar_idx < len(radar_msgs):
        #     cam_msg, cam_time = cam_msgs[cam_idx]
        #     radar_msg, radar_time = radar_msgs[radar_idx]

        #     # 如果相机的时间戳比雷达的小，增加相机的索引
        #     if cam_time < radar_time:
        #         cam_idx += 1
        #     # 如果雷达的时间戳比相机的小，增加雷达的索引
        #     elif radar_time < cam_time:
        #         radar_idx += 1
        #     else:
        #         # 时间戳接近时，写入新的bag文件，确保时间戳相同
        #         outbag.write('/aravis_cam/image_color/compressed', cam_msg, radar_time)
        #         outbag.write('/radar/object_list', radar_msg, radar_time)
        #         cam_idx += 1
        #         radar_idx += 1

if __name__ == "__main__":
    

    align_timestamps(input_bag, output_bag)
    print(f"对齐后的rosbag已保存到 {output_bag}")
