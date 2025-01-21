# ARS548

## Radar Connection
設定雷達連線
```bash
sudo ip link add link enp7s0 name radarConnection type vlan id 19
sudo ip addr add 10.13.1.133/24 dev radarConnection
sudo ip link set radarConnection up
sudo ip route replace default via 10.13.1.1 dev radarConnection
```
用完會沒辦法用網路，輸入以下指令即可
```bash
sudo ip route del default
```

## Build ros environment
```
cd ars548_ros
catkin_make
source devel/setup.bash
```
## Build BEVHeight
因為要在ROS中用它，所以要重新建環境，不能用docker。\
https://github.com/ADLab-AutoDrive/BEVHeight \
a. Install pytorch(v1.9.0).\
b. Install mmcv-full==1.4.0 mmdet==2.19.0 mmdet3d==0.18.1.\
c. Install pypcd
```bash
git clone https://github.com/klintan/pypcd.git
cd pypcd
python setup.py install
```
d. Install requirements.
```bash
pip install -r requirements.txt
```
e. Install BEVHeight (gpu required).
```bash
python setup.py develop
```
其中b.那項，不能直接載，要build from source。
- mmcv-full: https://github.com/open-mmlab/mmcv/tree/v1.4.0 \
  選到tag v1.4.0
  ```bash
  git clone https://github.com/open-mmlab/mmcv.git
  cd mmcv
  git checkout v1.4.0
  MMCV_WITH_OPS=1 pip install -e .
  ```
- mmdet: https://github.com/open-mmlab/mmdetection.git \
  選到tag v2.19.0
  ```bash
  git clone https://github.com/open-mmlab/mmdetection.git
  cd mmdetection
  git checkout v2.19.0
  pip install -r requirements/build.txt
  pip install -v -e .  # or "python setup.py develop"
  ```
- mmdet3d: https://github.com/open-mmlab/mmdetection3d.git \
  選到tag v0.18.1
  ```bash
  git clone https://github.com/open-mmlab/mmdetection3d.git
  cd mmdetection3d
  git checkout v0.18.1
  pip install -v -e .  # or "python setup.py develop"
  ```
- mmsegmentation
  ```
  pip install mmsegmentation==0.20.1
  ```
- 可能還要裝mmengine
  ```
  pip install mmengine
  ```
  
最後
```
python ars548_ros/src/ars548_driver/scripts/src/BEVHeight_radar/setup.py develop
```
[Weight下載](https://mailntustedutw-my.sharepoint.com/:u:/g/personal/m11215120_ms_ntust_edu_tw/EfuSXaH-dKVOvtgZQ58bm7kBPrG3F39jfE42-5mvf-PuIg?e=qJKiin)

## 資料錄製
第一個Terminal:
```
roscore
```
第二個Terminal:
```
roslaunch ars548_driver record.launch
```
第三個Terminal: 錄製成rosbag格式
```
rosbag record --duration 30 /aravis_cam/image_color/comporessed /radar/detection_list /radar/direction_velocity /radar/object_list /radar/point_cloud_detection /radar/point_cloud_object /radar/status
```
可用參數：詳見 https://wiki.ros.org/rosbag/Commandline#record

## 播放rosbag
第一個Terminal:
```
roscore
```
第二個Terminal:
```
roslaunch ars548_driver replay.launch
```
第三個Terminal:
```
rosbag play (your_bag_name)
```

## 使用多模態3D物件偵測
第一個Terminal:
```
roscore
```
第二個Terminal:
```
roslaunch ars548_driver detection.launch
```
第三個Terminal:
```
rosbag play (your_bag_name)
```

***
***
***

## 資料集
denorm: 相機座標系的道路平面方程式。可透過把雷達放在道路上，經過一次校正，取得雷達到相機的外參(即相機到道路平面的外參)，之後即可計算denorm。


## 相機內參
約90張棋盤格
```
Camera Matrix:
 [1804.85904, 0, 1028.16764],
 [0, 1801.73036, 776.526745],
 [0, 0, 1]

Distortion Coefficients:
 [[-1.92291452e-01  1.14853550e-01  1.44517360e-04 -9.37258777e-04
  1.19785944e-01]]
```

## Rosbag 轉出 pcd 跟 png
- Tools_RosBag2KITTI: https://github.com/leofansq/Tools_RosBag2KITTI \

在 ROS 環境中
```bash
cd catkin_ws
catkin_make
```
Decode ROSBAG to .png and .pcd, the results are saved in output.
```bash
# 1st terminal for ROS core
roscore
# 2nd terminal for decoding node
./devel/lib/obstacle_detection/map_generate
# 3rd terminal for ROSBAG playing, 0.1 means 0.1 times speed
rosbag play xxx.bag -r 0.1
```
> The actual play speed of ROSBAG is determined by the IO performance. Please adjust the speed to ensure the timestamps are within +/- 50 ms.


## 相機雷達校正 Calibration
- 目前是用 SensorsCalibration: https://github.com/PJLab-ADG/SensorsCalibration \

因為我們是用4D雷達，點雲有xyz，所以要用lidar2camera，不能用radar2camera。\
校正的時候盡量不要點到fx, fy。





\
ARS_548_RDI Driver
================
Introduction
---
This is a ROS 2 driver for the Continental ARS_548 Radar4d created by the Service Robotics Lab from the Pablo de Olavide University (Spain). For the ROS driver, please refer to the "noetic" branch of this repository.

This driver is used to connect with the ARS_548_RDI radar and get all of the data it sends.

First of all, this driver connects with the radar via Ethernet.\
Then it receives all the data the radar sends, translates it, and after that, it sends it in the structure of custom messages for the user to receive and interact with them.

This driver also creates two Point Clouds and a pose array with some of the information gathered and shows it using RViz 2.

About the code:
---
This project consists of two ROS2 packages: ars548_driver and ars548_messages. Below you can find a brief description of each package.

* ### ars548_driver.
    This package contains all of the code used to obtain the data from the radar, decode it and send it for the user to interact with it. 
    * ### ars548_driver.h
        This file contains the class that we will use to connect with the radar, obtain all of the data it sends to us and publish it for later use.

        Firstly it creates the connection with the radar.

        Then, it creates the different publishers that will send the data.

        After that, it enters an infinite loop in wich it filters the data the sensor is sending between three types of message (Status, Object and Detection), translates the data of each of the fields and stores them in an specific Structure for each message and finally it publishes the entire Structure.

    * ### ars548_driver_node.cpp
        This file launches the driver class to execute the project so we can see how it works. 


    * ### ars548_data.h
        In this file we save the different structures we will use to store and send the translated data for later use.

    * ### RadarVisualization.rviz
        This file contains all the configuration for rviz 2 to show the point clouds sent from this driver.

    * ### CMakeLists.txt and package.xml
        This two files configure an define all dependencies it needs to execute the project, including libraries, packages, etc. 
        
        They also define the files and executables this project creates when compiled.

    * ### Configurer.sh
        This file creates a connection to the default radar direction when executed. 

    * ### ars548_launch.xml
        This is the file we will be executing using the following command:
        ```
        ros2 launch ars548_driver ars548_launch.xml
        ```
        Inside this file we create three nodes that will be executed.
        * The first node is the one that executes the code created in this project.\
        This node has three parameters that we can change in case one of them does not correspond to the actual radar value.\
        This parameters are:
            * The radar IP.
            * The connection Port.
            * The Frame ID that we will use to send the messages.

        * The second node opens an RViz 2 window to visualize the data received from the radar.\
        This node uses the **RadarVisualization.rviz** file to configure the window.

        * The third and last node creates an static-transform-publisher to transform the data obtained from the radar to the data shown in RViz 2 (You can also change the arguments so it adapts to your project).
        
* ### ars548_messages
    This package contains all of the structures for the custom messages sent to the user.
    * ### Object.msg
        This file has the data structure for the object message.
    * ### Detection.msg
        This file has the data structure for the detection message.
    * ### Status.msg
        This file has the data structure for the status message.
    * ### ObjectList.msg
        This file has the data structure for the list of object messages. It has an array of messages from the Object.msg file
    * ### DetectionList.msg
        This file has the data structure for the list of detection messages. It has an array of messages from the Detection.msg file.
    * ### The rest of files are the messages to configure the radar. (They are unused in this project)

    
Before Downloading the code:
---
For the code to Work properly you must first do these steps.

- Have ROS2, RViz 2, Tf2 and colcon installed.
- Configure the Network connection with the radar.

In this file, we are working with the **ROS 2 Humble** distribution on Ubuntu 22.04, so all of the ROS2 commands that we will be using are from this particular distribution. In case you use another ROS 2 distribution, this commands may change a bit.

* ### Install ROS 2:
    To install ROS 2 you can follow this tutorial: <https://docs.ros.org/en/humble/Installation.html>

* ### Install RVIZ 2:
    This tool will be used to show the pointClouds that this driver will create to test it.\
    To install RVIZ 2, run this command on your terminal:
    ```
    sudo apt-get install ros-humble-rviz2 
    ```
* ### Install TF 2:
    To install TF 2 and all of its tools, run the next command on your terminal: 
    ```
    sudo apt-get install ros-humble-tf2-ros ros-humble
    ```
* ### Install Colcon:
    This tool is used to build ROS packages.\
    To install Colcon, execute the following command on your computer:
    ```
    sudo apt install python3-colcon-common-extensions
    ```

* ### Configure the Network to connect with the radar:

    This radar sends data using Ethernet Connection.         
    For being able to communicate with the radar and receive the data, you must first configure your network.\
    You can configure your network in two ways:
    
    * #### Configure it manually: 

        To do this you must follow the next steps:

        1. Open your terminal.

        2. Execute `nm-connection-editor`.

        3. Click on the option `Add a new connection` it is represented by a `+`.

        4. On the new Window  titled **Choose a connection Type** select `VLAN`and click on `Create`. ( **CAUTION**: Do not create a Vlan connection inside another Vlan connection)

        5. After selecting `Create`, a new window titled for editting this connection will appear.\
        In this window, you can change the default name into any other name.

        6. In the field **Parent interface**, inside the  `VLAN` tab you must select your phisical interface.

        7. In the field **VLAN id**, select a valid ID for tour connection.

        8. On the `IPv4 Settings` tab, in the **Method** field, select: **Manual** 

        9. On the `Addresses` field, click on **Add** and fill the new address with the next data:
            + `Address` field: **10.13.1.166**
            + `Netmask` field: **24**
            + `Gateway` field: **10.13.1.1**
            ###### (This values are the default data for the radar, in case you have other, change the values so they match yours)

        10. Click on the button labeled `Save`.

        With all this, you have configured your connection with the radar.

    * #### Configure it using the Configurer.sh file:
        This file automatically creates the connection with the radar using it's default values.\
        To execute this file you must go to the folder containing it and execute the following command:

        ```
        ./Configurer.sh
        ```  
        Once you execute the command, the program will ask you if you want to create the vlan connection and after that it will ask you to introduce the parent interface you want to use to create the connection.(It must be your physical interface, otherwise it won't work).\
        After all that it will create the connection with the radar using the default values for the network.
    
    If you configure it manually, you will have to make this process just once. If you do it executing the **Configurer.sh** file, you will have to do it everytime you turn on your computer.

How to execute the driver
---
Once you have installed ROS2, RViz 2, Tf 2, colcon, configured your network and downloaded the project, you can execute this driver.

For executing the driver you should go to the directory in wich you have downloaded this project and execute the next commands:

 ```
    > colcon build --packages-select ars548_driver ars548_messages
    > source install/setup.bash
    > ros2 launch ars548_driver ars548_launch.xml
  ```
The first command is used to build the project.\
The second command is used to source the project.\
The last command is used to launch the project and see the results in Rviz 2.
