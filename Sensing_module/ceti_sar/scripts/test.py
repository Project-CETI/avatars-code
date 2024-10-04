#!/usr/bin/env python

import rospy
from geometry_msgs.msg import Twist, Point, Quaternion
from std_msgs.msg import String


class Tester():
    def __init__(self):
        self.val = "empty"
        rospy.init_node('listener', anonymous=True)
        rospy.Subscriber("/chatter", String, self.callback)
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            print(self.val)
            rate.sleep()

    def callback(self,msg):
        self.val = msg.data


if __name__ == '__main__':
    Tester()