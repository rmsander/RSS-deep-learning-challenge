#! /usr/bin/env python

# MIT License

# Copyright (c) 2017-2018 Yongyang Nie

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""
Run a YOLO_v2 style detection model test images.
This ROS node uses the object detector class to run detection.
"""

from object_detector import ObjectDetector
from PIL import Image
import cv2
import numpy as np

#ros
import rospy
from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge, CvBridgeError
from deep_learning.msg import DetectionResult, DetectionResults
from traffic_light_segmentation import check_traffic_light
import utils


class ObjectDetectionNode:

    def __init__(self):
        self.camera_topic = rospy.get_param("/object_detection_node/camera_topic")
        self.obj_detect_results_topic = rospy.get_param("/object_detection_node/obj_detect_results_topic")
        self.obj_detect_viz_topic = rospy.get_param("/object_detection_node/obj_detect_viz_topic")

        rospy.init_node('object_detection')
        self.camera_sub = rospy.Subscriber(self.camera_topic,
            Image, self.image_update_callback, queue_size=1)
        self.current_frame = None
        self.bridge = CvBridge()

        rospy.loginfo("Object Detection Initializing")
        rospy.loginfo("Tiny Yolo V3")

        self.model_path = rospy.get_param("/object_detection_node/model_path")
        self.classes_path = rospy.get_param("/object_detection_node/classes_path")
        self.anchors_path = rospy.get_param("/object_detection_node/anchors_path")
        self.iou_threshold = rospy.get_param("/object_detection_node/iou_threshold")
        self.score_threshold = rospy.get_param("/object_detection_node/score_threshold")
        self.input_size = (416, 416)

        self.detector = ObjectDetector(model_path=self.model_path,
                                       classes_path=self.classes_path,
                                       anchors_path=self.anchors_path,
                                       score_threshold=self.score_threshold,
                                       iou_threshold=self.iou_threshold,
                                       size=self.input_size)

        labels_visualizer = utils.LabelsVisualizer(self.detector.class_names)
        detection_image_pub = rospy.Publisher(self.obj_detect_viz_topic + "/compressed",
            CompressedImage, queue_size=1)
        detection_results_pub = rospy.Publisher(self.obj_detect_results_topic,
            DetectionResults, queue_size=1)

        rate = rospy.Rate(15)

        while not rospy.is_shutdown():
            time = rospy.get_time()
            if self.current_frame is not None:
                print("new image at time %.2f" % time)
                out_boxes, out_scores, out_classes = \
                    self.detector.detect_object(self.current_frame)
                inference_end_time = rospy.get_time()
                print("*** inference time: %.3f" % (inference_end_time - time))

                if detection_image_pub.get_num_connections() > 0:
                    image = labels_visualizer.draw_bboxes(self.current_frame,
                        out_boxes, out_scores, out_classes)
                    print("***   drawing time: %.3f" % (rospy.get_time() - inference_end_time))

                    # img_msg = self.bridge.cv2_to_imgmsg(image, "bgr8")
                    msg = utils.cv2_to_compressed_imgmsg(image)
                    detection_image_pub.publish(msg)

                msg = self.convert_results_to_message(out_boxes, out_scores, out_classes)
                msg.is_green = self.check_traffic_segmentation(out_boxes, out_classes)
                if detection_results_pub.get_num_connections() > 0:
                    detection_results_pub.publish(msg)
            else:
                print("waiting for image at time %.2f" % time)
            rate.sleep()

    def check_traffic_segmentation(self, out_boxes, out_classes):
        labels = [self.detector.class_names[i] for i in out_classes]

        if 'TRAFFIC_LIGHT' not in labels:
            return True

        ind = labels.index('TRAFFIC_LIGHT')
        bbox = out_boxes[ind]

        top, left, bottom, right = bbox
        top = max(0, np.floor(top + 0.5).astype('int32'))
        left = max(0, np.floor(left + 0.5).astype('int32'))
        bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
        right = min(image.size[0], np.floor(right + 0.5).astype('int32'))

        bbox = ((left, top), (right, bottom))
        print '************************** BBOX', bbox

        '''if (x2 - x1) * (y2 - y1) < 13500:
            return True'''

        return not check_traffic_light(self.current_frame, bbox)['detected_traffic_stop']

    def image_update_callback(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            raise e
        # cv_image = utils.compressed_imgmsg_to_cv2(data)
        self.current_frame = cv_image

    @staticmethod
    def convert_results_to_message(out_boxes, out_scores, out_classes):

        msgs = DetectionResults()
        for i in range(len(out_scores)):
            msg = DetectionResult()
            msg.out_class = out_classes[i]
            msg.out_score = out_scores[i]
            msg.location = out_boxes[i, :]
            msgs.results.append(msg)

        return msgs


if __name__ == "__main__":

    try:
        ObjectDetectionNode()
    except rospy.ROSInterruptException:
        pass
