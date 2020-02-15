#!/usr/bin/env python2

import numpy as np
import rospy
import os
from sensor_msgs.msg import Image, CompressedImage # Use either
from cv_bridge import CvBridge, CvBridgeError
import cv2

class DataCollect:
    def __init__(self):
        self.images_dir = rospy.get_param("~training_images_dir")
        self.camera_topic_left  = rospy.get_param("~camera_topic_left")
        self.camera_topic_right = rospy.get_param("~camera_topic_right")
        self.camera_sub_left  = rospy.Subscriber(self.camera_topic_left, Image, self.save_images_left)
        self.camera_sub_right = rospy.Subscriber(self.camera_topic_right, Image, self.save_images_right)
        self.bridge = CvBridge() # Instantiate bridge between cv2 and ROS
        self.left_counter = 1
        self.right_counter = 1


    def save_images(self, msg, side):
        time = msg.header.stamp.secs
        print("Received an image at time %s" % time)
        try:
            # Convert your ROS Image message to OpenCV2
            cv2_img = self.bridge.imgmsg_to_cv2(msg, "bgr8")

        except CvBridgeError, e:
            print("error", e)

        else:
            # Save your OpenCV2 image as a jpeg
            counter = self.left_counter if side == "left" else self.right_counter
            image_path = os.path.join(self.images_dir, "%s-%05d.jpeg" % (side, counter))
            cv2.imwrite(image_path, cv2_img) # Write to .jpeg if we are able to
            if side == "left":
                self.left_counter += 1
            else:
                self.right_counter += 1
            rospy.sleep(0.5)


    def save_images_left(self, msg):
        self.save_images(msg, "left")


    def save_images_right(self, msg):
        self.save_images(msg, "right")


if __name__ == "__main__":
    rospy.init_node("data_collect") # Initialize our node
    data_collect = DataCollect()
    rospy.spin()        
