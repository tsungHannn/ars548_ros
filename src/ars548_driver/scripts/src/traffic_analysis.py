import sensor_msgs.point_cloud2 as pc2
import math
import rospy


# 把每幀的交通狀況都儲存進list中(當前整體平均車速、物件追蹤曾經追蹤到的汽車)
class TrafficAnalyzer:
    def __init__(self):

        # 用於存儲分析結果
        self.vehicle_list = [] # 追蹤到的車子id
        self.speed_list = [] # 平均車速
        # self.timestamps = []
        self.analysis_interval = rospy.Duration(6)  # 6秒
        self.last_analysis_time = None
        self.flow_threshold = [15, 30, 40]
        self.speed_threshold = [40, 30, 20]

    def reset(self):
        self.speed_list.clear()
        self.vehicle_list.clear()


    # 計算當前幀的整體平均速度
    # 只計算速度 > 5 的雷達點
    def calculate_radar_points_averages(self, radarData):
        pc_data = pc2.read_points(radarData, field_names=("x", "y", "z", "vx", "vy", "id"), skip_nans=True)
        average_speed = 0.0
        count = 0
        for point in pc_data:
            speed = math.sqrt(pow(point[3], 2) + pow(point[4], 2))
            
            if speed > 5:
                
                average_speed += speed
                count += 1

        if count == 0:
            self.speed_list.append(0)        
        else:
            average_speed = average_speed / count

        average_speed *= 3.6 # m/s轉成km/h

        self.speed_list.append(average_speed)

        return average_speed


    # input 追蹤結果，之後紀錄在vehicle_list中
    # tracking_result: [id1, id2, id3, ...]
    def record_vehicle(self, tracking_result):
        for id in tracking_result:
            find = False
            
            for tracked in self.vehicle_list:
                if id == tracked:
                    find = True
                    break
            
            if find == False:
                self.vehicle_list.append(id)
        
                
        

    def analyze_traffic_time_interval(self):
        current_time = rospy.Time.now()
        if self.last_analysis_time == None:
            self.last_analysis_time = current_time
            return
        if (current_time - self.last_analysis_time) < self.analysis_interval:
            return

        vehicle_count = len(self.vehicle_list)

        total_speed = 0
        average_speed = 0
        for speed in self.speed_list:
            total_speed += speed

        if len(self.speed_list) != 0:
            average_speed = total_speed / len(self.speed_list)

        # self.vehicle_list.append(vehicle_count)
        # self.speed_list.append(avg_speed)
        # self.timestamps.append(current_time.to_sec())

        # print(self.speed_list)
        # print("Length:", len(self.speed_list))

        score_f = -1
        score_s = -1
        if vehicle_count <= self.flow_threshold[0]:
            score_f = 1
        elif vehicle_count > self.flow_threshold[0] and vehicle_count <= self.flow_threshold[1]:
            score_f = 2
        elif vehicle_count > self.flow_threshold[1] and vehicle_count <= self.flow_threshold[2]:
            score_f = 3
        elif vehicle_count > self.flow_threshold[2]:
            score_f = 4

        if average_speed > self.speed_threshold[0]:
            score_s = 1
        elif average_speed <= self.speed_threshold[0] and average_speed > self.speed_threshold[1]:
            score_s = 2
        elif average_speed <= self.speed_threshold[1] and average_speed > self.speed_threshold[2]:
            score_s = 3
        elif average_speed <= self.speed_threshold[2]:
            score_s = 4
        
        # if score_f == -1:
        #     print("Error: 車流量")
        # if score_s == -1:
        #     print("Error: 平均車速")

        congestion_score = (score_s + score_f) / 2
        congestion_level = -1

        if congestion_score <= 1.5:
            congestion_level = 1
        elif congestion_score > 1.5 and congestion_score <= 2.5:
            congestion_level = 2
        elif congestion_score > 2.5 and congestion_score <= 3.5:
            congestion_level = 3
        elif congestion_score > 3.5:
            congestion_level = 4

        # if congestion_level == -1:
        #     print("Error: 分析結果")
        rospy.loginfo(f"車流量 = {vehicle_count}, 平均速度 = {average_speed:.2f} km/h, 壅塞程度 = {congestion_level}")

        # self.vehicle_data.clear()
        self.reset()
        self.last_analysis_time = current_time


    def analyze_traffic_with_frames(self):


        vehicle_count = len(self.vehicle_list)

        total_speed = 0
        average_speed = 0
        for speed in self.speed_list:
            total_speed += speed

        if len(self.speed_list) != 0:
            average_speed = total_speed / len(self.speed_list)

        # self.vehicle_list.append(vehicle_count)
        # self.speed_list.append(avg_speed)
        # self.timestamps.append(current_time.to_sec())

        # print(self.speed_list)
        # print("Length:", len(self.speed_list))

        score_f = -1
        score_s = -1
        if vehicle_count <= self.flow_threshold[0]:
            score_f = 1
        elif vehicle_count > self.flow_threshold[0] and vehicle_count <= self.flow_threshold[1]:
            score_f = 2
        elif vehicle_count > self.flow_threshold[1] and vehicle_count <= self.flow_threshold[2]:
            score_f = 3
        elif vehicle_count > self.flow_threshold[2]:
            score_f = 4

        if average_speed > self.speed_threshold[0]:
            score_s = 1
        elif average_speed <= self.speed_threshold[0] and average_speed > self.speed_threshold[1]:
            score_s = 2
        elif average_speed <= self.speed_threshold[1] and average_speed > self.speed_threshold[2]:
            score_s = 3
        elif average_speed <= self.speed_threshold[2]:
            score_s = 4
        
        # if score_f == -1:
        #     print("Error: 車流量")
        # if score_s == -1:
        #     print("Error: 平均車速")

        congestion_score = (score_s + score_f) / 2
        congestion_level = -1

        if congestion_score <= 1.5:
            congestion_level = 1
        elif congestion_score > 1.5 and congestion_score <= 2.5:
            congestion_level = 2
        elif congestion_score > 2.5 and congestion_score <= 3.5:
            congestion_level = 3
        elif congestion_score > 3.5:
            congestion_level = 4

        # if congestion_level == -1:
        #     print("Error: 分析結果")
        rospy.loginfo(f"車流量 = {vehicle_count}, 平均速度 = {average_speed:.2f} km/h, 壅塞程度 = {congestion_level}")
        return vehicle_count, average_speed, congestion_level
        # self.vehicle_data.clear()
