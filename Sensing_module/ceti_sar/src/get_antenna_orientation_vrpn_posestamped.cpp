#include "ros/ros.h"
#include "tf/tf.h"
#include "geometry_msgs/PoseStamped.h"
#include "std_msgs/Bool.h"
#include <chrono>
#include <signal.h>
#include <pwd.h>
#include <time.h>
#include <fstream>
#include <cmath>

bool running = true;
bool get_true_orientation=false;
bool first_sample=true;
bool __Flag_start_motion=false;
bool __Flag_collect_ori=false;
geometry_msgs::PoseStamped groundTruthPose;
double orientation_true;
double prev_ts;
double curr_ts;
double diff_ts=0;
double joint_angle=0.0;
std::vector<std::vector<double>> __robot_ori_gt;
std::vector<std::vector<double>> __robot_ori_joint;
std::vector<double> temp_ori_gt;
std::vector<double> temp_ori_joint;
auto iteration_start = std::chrono::high_resolution_clock::now();
auto starttime = std::chrono::high_resolution_clock::now();
auto endtime = std::chrono::high_resolution_clock::now();

bool validateQuaternion(const tf::Quaternion& quat) {
    return (quat.getW() != 0 || quat.getX() != 0 || quat.getY() != 0 || quat.getZ() != 0);
}


double quaternionToYaw(const tf::Quaternion& q) {
    double yaw = 0.0;

    if (validateQuaternion(q)) {
        tf::Matrix3x3 m(q);

        double roll, pitch;
        m.getRPY(roll, pitch, yaw);
    }

    return yaw;
}


void positionCallback(const geometry_msgs::PoseStamped::ConstPtr& msg) 
{ 
    
    if(__Flag_collect_ori)
    {
        {
            groundTruthPose.pose.position = msg->pose.position;
            groundTruthPose.pose.orientation = msg->pose.orientation;

            tf::Quaternion q(
                    groundTruthPose.pose.orientation.x,
                    groundTruthPose.pose.orientation.y,
                    groundTruthPose.pose.orientation.z,
                    groundTruthPose.pose.orientation.w
            );

            orientation_true = quaternionToYaw(q);

            temp_ori_gt.clear();
            double nsec_timestamp = msg->header.stamp.sec*1e9 + msg->header.stamp.nsec;
            temp_ori_gt.push_back(nsec_timestamp);
            temp_ori_gt.push_back(0);
            temp_ori_gt.push_back(orientation_true);
            temp_ori_gt.push_back( groundTruthPose.pose.position.x);
            temp_ori_gt.push_back( groundTruthPose.pose.position.y);
            temp_ori_gt.push_back( groundTruthPose.pose.position.z);

            //Keep capturing poses and clear each time after AOA calculation.
            __robot_ori_gt.push_back(temp_ori_gt);


        }
    }
}

void CallStartMotion(const std_msgs::Bool::ConstPtr& msg) 
{ 
    __Flag_start_motion = msg->data;
    starttime = std::chrono::high_resolution_clock::now();
}


void CTRL_C(int sig) 
{
    running = false;    
}


void writeTrajToFile(std::vector<std::vector<double>>& ori_vector, 
                     std::string fn)
{
    std::cout.precision(10);
    std::ofstream myfile (fn);
    std::vector<double> temp;

    std::cout << "Trajectory size " << ori_vector.size() << std::endl;
    if (myfile.is_open())
    {
        for(size_t i = 0; i < ori_vector.size(); i++)
        {
            for(int j=0; j< ori_vector[i].size(); j++)
            {
                myfile << std::fixed << ori_vector[i][j] << ",";
            }
            myfile << "\n";
        }
        
    }
    myfile.close();
}



int main(int argc, char **argv)
{
    ros::init(argc, argv, "antenna_rotor", 1); //Option to make the node name anonymous
    ros::NodeHandle nh("~");
    struct passwd *pw = getpwuid(getuid());
    std::string homedir = pw->pw_dir;
    float z_vel = 0.1;
    int track_duration = 15;
    std::string my_vrpn_topic;
    
    nh.getParam("tracking_duration", track_duration);
    nh.getParam("vrpn_topic", my_vrpn_topic);

    float exp_duration=0;
    float iteration_duration = 0;
    bool flip = false;
    bool first = true;
    bool __Flag_started_CSI = false;
    ros::Rate rate(100);    
    geometry_msgs::Twist rotate_right, rotate_left;
    rotate_left.angular.z = -z_vel;
    rotate_right.angular.z = z_vel;
    ros::Subscriber get_ori = nh.subscribe(my_vrpn_topic, 10, positionCallback);
    ros::Subscriber start_rotation = nh.subscribe("/ceti_sar/start_data_logging", 10, CallStartMotion);
    int i = 0;
    int cmd_status = 0;
    double joint_threshold  = 2.36;
    signal(SIGINT, CTRL_C);
    time_t rawtime;
    struct tm * timeinfo;
    char buffer[80];

    float max_y_coordinate = 0, x_coordinate = 0;

    ROS_INFO("Initialized antenna rotation.");

    iteration_start = std::chrono::high_resolution_clock::now();
    starttime = std::chrono::high_resolution_clock::now();
    endtime = std::chrono::high_resolution_clock::now();
    
    while(running)
    {
        ros::spinOnce();
        if(__Flag_start_motion)
        {
	   if(!__Flag_started_CSI)
	   {
		    //cmd_status = system(csi_start_local_cmd.c_str());
		    cmd_status = 1;
	    	if (cmd_status < 0)
	    	{
    			std::cout << "Error: " << strerror(errno) << '\n';
			__Flag_start_motion = false;
			break;
	    	}
	    	else
	    	{
			__Flag_started_CSI = true;
			__Flag_collect_ori = true;
			//sleep(0.5);
  	    	}
 	    }

            endtime = std::chrono::high_resolution_clock::now();
            exp_duration = std::chrono::duration<float, std::milli>(endtime - starttime).count() * 0.001;
            max_y_coordinate = groundTruthPose.pose.position.y > max_y_coordinate ? groundTruthPose.pose.position.y : max_y_coordinate;
            x_coordinate = groundTruthPose.pose.position.y > max_y_coordinate ? groundTruthPose.pose.position.x : x_coordinate;

            ROS_INFO("Duration expired = %f seconds", exp_duration);
            ROS_INFO("Max_y_coordinate = %f meters, corresponding x-coordinate = %f ", max_y_coordinate, x_coordinate);
            // if(abs(orientation_true - 1.57) < 0.01 || abs(orientation_true + 1.57) < 0.01 )
            if(abs(orientation_true - 0) < 0.01 || abs(orientation_true - 1.57) < 0.01 || 
              abs(orientation_true + 1.57) < 0.01 || abs(orientation_true + 3.14) < 0.01 )
            {

                ROS_INFO("Yaw %f, position : %f. %f", orientation_true, groundTruthPose.pose.position.x, groundTruthPose.pose.position.y);
            }

            if(exp_duration > track_duration)
            {
                ROS_INFO("======= Saving displacement data =====");

                ros::spinOnce();
                exp_duration = 0;
                __Flag_start_motion = false;
		        __Flag_started_CSI = false;
                __Flag_collect_ori=false;

                time (&rawtime);
                timeinfo = localtime(&rawtime);
                strftime(buffer,sizeof(buffer),"%Y-%m-%d_%H%M%S",timeinfo);
                std::string time_str(buffer);

                //std::string joint_ori_gt = homedir+"/catkin_ws/src/ceti_sar/data/mocap_vrpn_displacement_"+time_str+".csv";
		std::string joint_ori_gt = homedir+"/mocap_vrpn_displacement.csv";
                std::cout << "Mocap VRPN Orientation file " << std::endl;
                writeTrajToFile(__robot_ori_gt, joint_ori_gt);
                
                __robot_ori_gt.clear();
                __robot_ori_joint.clear();
		                        
                ROS_INFO("======= Completed =====");
            }
        }

        rate.sleep();
    }

    ROS_INFO("Exiting.");
    ros::shutdown();
    return 0;
}
