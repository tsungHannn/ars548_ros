/**
 * @file ars548_driverNode.cpp 
 */
#include <ros/ros.h>
#include "../include/ars548_driver/ars548_driver.h"
#include <string>

int main(int argc, char *argv[])
{
    ros::init(argc, argv, "ars548_driver_node");
    std::string radarIP;
    std::string frameID;
    int radarPort;
    ROS_INFO_STREAM("Argc:" << argc);
    if (argc == 4){
        radarIP = std::string(argv[1]);
        frameID = std::string(argv[2]);
        radarPort = std::stoi(argv[3]);

    }
    else{
        
        ROS_ERROR("error getting radar parameter(ip, frameID, port)");
        return 1;
    }
    Ars548_Driver ars548_driver(radarIP, radarPort, frameID);
    
    ros::Rate rate(60);
    while(ros::ok())
    {
        ros::spinOnce();
        rate.sleep();
    }
    // ros::spin();
    return 0;
}
