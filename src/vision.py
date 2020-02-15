#!/usr/bin/env python2

import cv2
import numpy as np
import PIL
import os
import sys

import rospy
from sensor_msgs.msg import Image, CompressedImage
from deep_learning.msg import VisionOutput, BBox, Centerline # Import custom message here
from deep_learning.msg import DetectionResult, DetectionResults
from cv_bridge import CvBridge, CvBridgeError

from yolo3.utils import _get_class
import utils
import lane_segmentation
import homography
import cone_segmentation


class Vision:
    # Get parameters, initialize subscribers and publishers, etc.
    def __init__(self):
        self.camera_topic        = rospy.get_param("~camera_topic")
        self.vision_output_topic = rospy.get_param("~vision_output_topic")
        self.labeled_image_topic = rospy.get_param("~labeled_image_topic")
        self.obj_detect_results_topic = rospy.get_param("~obj_detect_results_topic")
        self.classes_path        = rospy.get_param("~classes_path")

        self.class_names = _get_class(self.classes_path)

        self.camera_sub = rospy.Subscriber(self.camera_topic, Image, self.camera_callback)
        # self.camera_sub = rospy.Subscriber(self.camera_topic + "/compressed",
        #    CompressedImage, self.camera_callback, queue_size=1)
        self.obj_detect_results_sub = rospy.Subscriber(self.obj_detect_results_topic,
            DetectionResults, self.obj_detect_results_callback, queue_size=1)

        self.labels_visualizer = utils.LabelsVisualizer(self.class_names)
        self.vision_output_pub = rospy.Publisher(self.vision_output_topic,
            VisionOutput, queue_size=1)
        # self.labeled_image_pub = rospy.Publisher(self.labeled_image_topic, Image, queue_size=1)
        self.labeled_image_pub = rospy.Publisher(self.labeled_image_topic + "/compressed",
            CompressedImage, queue_size=1)

        # bridge is no longer necessary for CompressedImage, consider removing the line below
        self.bridge = CvBridge() # Instantiate bridge between cv2 and ROS

        self.clear_processing_output()
        self.last_obj_detect_results_time = rospy.get_time()
        
        #Threshold parameters for determining whether or not to call controller routines
        self.stop_sign_thr = rospy.get_param("~stop_sign_thr") #Minimum bbox area for stop sign 
        self.yield_sign_thr = rospy.get_param("~yield_sign_thr") #Minimum bbox are for yield sign
        self.traffic_light_thr = rospy.get_param("~traffic_light_thr") #Minimum bbox are for traffic light
        self.pedestrian_sign_thr = rospy.get_param("~pedestrian_sign_thr") #Minimum bbox for pedestrian sign
        

        self.bbox_low_midpoint = None
        #State variable attributes
        self.centerlines = []
        self.are_cones = False
        self.is_disabled_parking = False
        self.is_pedestrian_sign = False
        self.is_pedestrian = False
        self.is_stop_sign = False
        self.is_traffic_light = False
        self.is_green = False
        self.obstacle_distance = -1 #TODO: Initialize to large or small num?
        self.TAcar_distance = -1 #TODO: Initialize to large or small num?
        self.TAcar_loc = [0,0] # I am the TA jkjk
        self.pedestrian_distance = 0 #TODO: Initialize to large or small num?

        self.start_parking_time = 0
        self.is_parking = False
        

    def camera_callback(self, msg):
        
        try:
            # Convert your ROS Image message to OpenCV2
            image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            print "new image at time %.2f" % rospy.get_time()
        except CvBridgeError, e:
            print("error converting imgmsg to cv2: %s" % e)
        
        # image = utils.compressed_imgmsg_to_cv2(msg)
        centerlines_world = lane_segmentation.centerlines(image)

        # for visualizing line in rqt_image_view
        new_lines_world = np.reshape(centerlines_world, (-1, 2)) # reshape to array of endpoints
        new_lines_warped = homography.world_to_orig_image_fn(new_lines_world)
        centerlines_warped = np.reshape(new_lines_warped, (-1, 4))
        line_types = ["other" for _ in centerlines_warped]
        #print('CENTERLINES_WORLD', centerlines_world)
        #print('CENTERLINES_WARPED', centerlines_warped)

        # self.detect_objects(image)
        # TODO: add other processing steps

        '''
        try:
            # publish annotated image for real-time visualization
            labeled_image = image # TODO: CHANGE
            self.labeled_image_pub.publish(self.bridge.cv2_to_imgmsg(labeled_image, "bgr8"))
        except CvBridgeError as e:
            print("error converting cv2 to imgmsg: %s" % e)
        '''


        # CONES
        pcones = cone_segmentation.check_cones(image)
        self.mid_cone_u, self.mid_cone_v = -1, -1
        self.mid_cone_x, self.mid_cone_y = -1, -1
        if pcones is not None:
            size_smallest_cone = pcones["size_smallest_cone"] 
            if size_smallest_cone > 30000:
                print("size_smallest_cone too large %.2f" % size_smallest_cone)
            elif self.is_parking and 6 < (rospy.get_time() - self.start_parking_time) < 8 and pcones["goal_point_smallest"] is not None:
                mid_cone_u, mid_cone_v = pcones["goal_point_smallest"]
                mid_cone_x, mid_cone_y = homography.apply_homography(mid_cone_u, mid_cone_v)
                self.mid_cone_u, self.mid_cone_v = mid_cone_u, mid_cone_v
                self.mid_cone_x, self.mid_cone_y = mid_cone_x, mid_cone_y
                print "use smallest two cones' midpoint instead; x: %.2f, y: %.2f" % (self.mid_cone_x, self.mid_cone_y)
            elif self.is_parking and (rospy.get_time() - self.start_parking_time) >= 8:
                self.is_parking = False
            elif pcones["goal_point_largest_3"] is not None:
                mid_cone_u, mid_cone_v = pcones["goal_point_largest_3"]
                mid_cone_x, mid_cone_y = homography.apply_homography(mid_cone_u, mid_cone_v)
                if mid_cone_x < 2:
                    self.mid_cone_u, self.mid_cone_v = mid_cone_u, mid_cone_v
                    self.mid_cone_x, self.mid_cone_y = mid_cone_x, mid_cone_y
                    print "found 3 cones - midpoint is x: %.2f, y: %.2f" % (self.mid_cone_x, self.mid_cone_y)
                    if not self.is_parking:
                        self.start_parking_time = rospy.get_time()
                        self.is_parking = True 
                        print "start parking at time %.2f" % rospy.get_time()

        if self.labeled_image_pub.get_num_connections() > 0:
            # draw road lines
            labeled_image = utils.draw_lines(image, centerlines_warped, line_types)

            # draw bboxes
            labeled_image = self.labels_visualizer.draw_bboxes(labeled_image,
                self.out_boxes, self.out_scores, self.out_classes)

            # draw obstacle low midpoint
            if self.bbox_low_midpoint is not None:
                u, v = self.bbox_low_midpoint
                u, v = int(u), int(v)
                print("obstacle low midpoint at %d, %d" % (u, v) )
                labeled_image = cv2.circle(labeled_image, (u, v), 10, (0, 0, 255), -1)
            # draw cone midpoint
            if self.mid_cone_u != -1:
                u, v = self.mid_cone_u, self.mid_cone_v
                print("cone midpoint at %d, %d" % (u, v))
                labeled_image = cv2.circle(labeled_image, (u, v), 10, (255, 0, 0), -1)

            msg = utils.cv2_to_compressed_imgmsg(labeled_image)
            self.labeled_image_pub.publish(msg)

        if self.vision_output_pub.get_num_connections() > 0:
            self.centerlines = centerlines_world
            self.publish_vision_message()

        '''
        # clear obj detect results if older than 5 seconds, in case the obj detect node died
        if rospy.get_time() - self.last_obj_detect_results_time > 5:
            self.clear_processing_output()
        '''



    def obj_detect_results_callback(self, msg):
        obj_detect_results = msg.results
        self.out_classes = [result.out_class for result in obj_detect_results]
        self.out_scores = [result.out_score for result in obj_detect_results]
        self.out_boxes = [result.location for result in obj_detect_results]
        self.obstacle_distance = -1 # maybe clear all obj det results here
        self.bbox_low_midpoint = None
        self.is_green = msg.is_green

        for result in obj_detect_results:
            label = self.class_names[result.out_class]
            score = result.out_score
            # bbox from YOLO was switched between u and v coordinates
            v1, u1, v2, u2 = result.location

            # TODO: where to apply homography transform? where to calculate distance?
            bbox_msg = BBox()
            bbox_msg.x1, bbox_msg.y1 = u1, v1
            bbox_msg.x2, bbox_msg.y2 = u2, v2
            bbox_msg.score = score

            if label == "CONE": #TODO: Set to actual class label name
                self.cone_bboxes.append(bbox_msg)

            elif label == "DISABLED_PARKING_SIGN": #TODO: Set to actual class label name
                bbox_area = abs((u2-u1)*(v2-v1)) #Area of our bounding box
                if bbox_area >= self.disabled_parking_sign_thr: #If area exceeds our threshold
                    self.is_disabled_parking = True #Tell our controller we have detected a pedestrian sign

            elif label in ["SOCCER_BALL", "PERSON", "TA_CAR"]:
                u, v = utils.bbox_low_midpoint(bbox_msg)
                x, y = homography.apply_homography(u, v)
                print("person/ball found at %.2f, %.2f" % (u, v))
                if abs(y) < 0.4:
                    print("+++++++++++ side distance %.2f; set as bbox low midpoint" % y)
                    self.bbox_low_midpoint = u, v
                    if label in ["SOCCER_BALL", "PERSON"]:    
                        self.obstacle_distance = min(self.obstacle_distance, x)
                    elif label == "TA_CAR":
                        self.TAcar_distance = min(self.TAcar_distance, x)
                        self.TAcar_loc = [x,y]
                else:
                    print("----------- side distance %.2f; not set")

            elif label == "PEDESTRIAN_SIGN": #TODO: Set to actual class label name
                bbox_area = abs((u2-u1)*(v2-v1)) #Area of our bounding box
                if bbox_area >= self.pedestrian_sign_thr: #If area exceeds our threshold
                    self.is_pedestrian_sign = True #Tell our controller we have detected a pedestrian sign
                
                
            elif label in ["STOP_SIGN","stop sign"]: #TODO: Set to actual class label name
                bbox_area = abs((u2-u1)*(v2-v1)) #Area of our bounding box
                if bbox_area >= self.stop_sign_thr: #If area exceeds our threshold
                    self.is_stop_sign = True #Tell our controller we have detected a stop sign

            elif label in ["traffic light", "parking meter","TRAFFIC_LIGHT"]: #TODO: Set to actual class label names
                bbox_area = abs((u2-u1)*(v2-v1)) #Area of our bounding box
                if bbox_area >= self.traffic_light_thr: #If area exceeds our threshold
                    self.is_traffic_light = True #Tell our controller we have detected a traffic 
                    
        self.publish_vision_message()
        self.last_obj_detect_results_time = rospy.get_time()

    def clear_processing_output(self):
        self.obj_detect_results = []
        self.out_classes = []
        self.out_scores = []
        self.out_boxes = []
        # Get class attributes of our messsage components
        #TODO: When to reset these?
        self.are_cones = False #Boolean for cone detection
        self.is_disabled_parking = False #Boolean for disabled parking sign detection
        self.is_pedestrian_sign = False #Boolean for pedestrian sign detection
        self.is_stop_sign = False #Boolean for stop sign detection
        self.is_traffic_light = False #Boolean for traffic light detection
        self.is_green = False #Boolean for whether or not detected traffic light is green
        self.obstacle_distance = -1 #Distance in meters to obstacle
        self.TAcar_distance = -1 #Distance in pixels to obstacle
        self.mid_cone_u = -1
        self.mid_cone_v = -1
        self.mid_cone_x = -1
        self.mid_cone_y = -1


# =============================================================================
#     def publish_vision_message(self):
#         msg = VisionOutput() # Use custom message
#         ms = []
#         for x1, y1, x2, y2 in self.centerlines:
#             m = Centerline()
#             m.x1 = x1
#             m.y1 = y1
#             m.x2 = x2
#             m.y2 = y2
#             ms.append(m)
#         msg.centerlines = ms # Set centerline to be an array
#         msg.cone_bboxes             = self.cone_bboxes
#         msg.disabled_parking_bboxes = self.disabled_parking_bboxes
#         msg.pedestrian_sign_bboxes  = self.pedestrian_sign_bboxes
#         msg.stop_sign_bboxes        = self.stop_sign_bboxes
#         msg.traffic_light_bboxes    = self.traffic_light_bboxes
#         msg.obstacle_distance       = self.obstacle_distance
#         self.vision_output_pub.publish(msg)
# 
# =============================================================================

    def publish_vision_message(self):
        msg = VisionOutput() #Use custom message definition
        ms = []
        for x1, y1, x2, y2 in self.centerlines:
            m = Centerline()
            m.x1 = x1
            m.y1 = y1
            m.x2 = x2
            m.y2 = y2
            ms.append(m)
        msg.centerlines = ms # Set centerline to be an array
        msg.are_cones               = self.are_cones
        msg.is_disabled_parking     = self.is_disabled_parking
        msg.is_pedestrian_sign      = self.is_pedestrian_sign
        msg.is_pedestrian           = self.is_pedestrian
        msg.is_stop_sign            = self.is_stop_sign
        msg.is_traffic_light        = self.is_traffic_light
        msg.is_green                = self.is_green
        msg.obstacle_distance       = self.obstacle_distance
        msg.TAcar_distance          = self.TAcar_distance
        msg.TAcar_loc               = self.TAcar_loc
        msg.pedestrian_distance     = self.pedestrian_distance
        msg.mid_cone_x              = self.mid_cone_x
        msg.mid_cone_y              = self.mid_cone_y
        self.vision_output_pub.publish(msg)
        

if __name__ == "__main__":
    rospy.init_node("vision")
    vision = Vision()
    rospy.spin()
