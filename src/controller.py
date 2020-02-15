#!/usr/bin/env python2

import numpy as np
import rospy
from deep_learning.msg import VisionOutput # Import custom message here
from ackermann_msgs.msg import AckermannDriveStamped, AckermannDrive
from goal_controller import GoalController
from homography import apply_homography
import time
import utils

from cone_segmentation import check_cones

class Controller:
    def __init__(self):
        self.speed         = rospy.get_param("~car_speed") # Car speed, in meters/second
        self.safety_buffer = rospy.get_param("~safety_buffer") # Safety buffer, in meters
        self.wheelbase     = rospy.get_param("~wheelbase")
        self.vision_topic  = rospy.get_param("~vision_output_topic")
        self.drive_topic   = rospy.get_param("~drive_topic")
        self.default_lookahead = rospy.get_param("~default_lookahead")
        self.change_lane_lookahead = rospy.get_param("~change_lane_lookahead")
        self.obstacle_thresh = rospy.get_param("~obstacle_thresh")
        self.normal_speed = rospy.get_param("~normal_speed")
        self.cautious_speed = rospy.get_param("~cautious_speed")

        self.goal_controller = GoalController()

        # timer for stop signs
        self.timer = time.time()

        self.current_centerline = (0,0) # slope, y_int
        #self.current_slope = 0
        #self.current_y_intercept = 0
        self.current_theta = np.pi/2
        self.current_rho = 0
        self.is_turning = False
        self.turn_start_time = rospy.get_time()

        self.drive_cmd = self.drive_cmd_msg(0, 0)
        self.line_last_found_time = 0
        self.reuse_drive_cmd_duration = rospy.get_param("~reuse_drive_cmd_duration")
        self.theta_threshold = rospy.get_param("~theta_threshold")
        self.rho_threshold = rospy.get_param("~rho_threshold")
        self.num_frames_turning_threshold = rospy.get_param("~num_frames_turning_threshold")
        self.num_frames_turning_window = rospy.get_param("~num_frames_turning_window")
        self.turn_steering_angle = rospy.get_param("~turn_steering_angle")
        self.turn_duration = rospy.get_param("~turn_duration")
        self.distance_to_horizontal_line_before_turning = rospy.get_param("~distance_to_horizontal_line_before_turning")
        self.straight_duration = 0
        self.num_consecutive_frames_with_horizontal_line = 0
        self.found_horizontal_lines_array = [0] * self.num_frames_turning_window
        self.frame_count = 0
        self.distance_to_pedestrian_thr = rospy.get_param("~distance_to_pedestrian_thr")
        self.follow_TA_max_distance_thr = rospy.get_param("~follow_TA_max_distance_thr")
        self.TA_car_desired_distance = rospy.get_param("~TA_car_desired_distance")
        self.Kp1 = rospy.get_param("~Kp1")
        self.Kp2 = rospy.get_param("~Kp2")

        self.vision_sub    = rospy.Subscriber(self.vision_topic, VisionOutput, self.vision_callback)
        self.drive_pub     = rospy.Publisher(self.drive_topic, AckermannDriveStamped, queue_size=1)


    def vision_callback(self, vision_output):
        # PARKING
        #while self.parking_mode: #While we're in parking mode

        if vision_output.mid_cone_x != -1: #If we detect disabled parking sign
            self.mid_cone_x = vision_output.mid_cone_x
            self.mid_cone_y = vision_output.mid_cone_y
            drive_cmd = self.get_drive_cmd(self.mid_cone_x - 0.5, self.mid_cone_y, self.speed)
            self.drive_pub.publish(drive_cmd)
            print "move towards cone midpoint x: %.2f, y: %.2f" % (self.mid_cone_x, self.mid_cone_y)
            return


        # TURNING
        self.frame_count += 1
        if self.is_turning:
            elapsed_time = rospy.get_time() - self.turn_start_time  
            if elapsed_time < self.straight_duration:
                self.drive_pub.publish(self.drive_cmd_msg(0, self.speed))
                return
            elif elapsed_time < self.straight_duration + self.turn_duration:
                self.drive_pub.publish(self.drive_cmd_msg(self.turn_steering_angle, self.speed))
                return
            else:
                print "TURNING finished"
                # continue using the normal logic
                self.is_turning = False
                # return

        # default drive commands
        drive_cmd_x = 0
        drive_cmd_y = 0
        drive_cmd_v = self.speed
        
        # assume we have a centerline
        # array of lines in real world space, each line has (x1, y1, x2, y2)
        # assume x1 > x2, i.e. x2 is closer to the car
        centerlines_msg = vision_output.centerlines
        centerlines = [(c.x1, c.y1, c.x2, c.y2) for c in centerlines_msg]
        current_line, adjacent_line = self.guess_centerline(centerlines)


        # if obstacle is within thresh to change lanes
        if vision_output.obstacle_distance != -1 and self.obstacle_thresh > vision_output.obstacle_distance:
            print("detecting obstacle, obs dist: %.2f" % vision_output.obstacle_distance)
            # if adjacent lane exists, then move to adjacent lane
            if adjacent_line is not None:
                '''
                self.current_slope, self.current_y_intercept = self.slope_and_intercept(adjacent_line)
                self.current_theta, self.current_rho = utils.theta_rho_line(adjacent_line)
                drive_cmd_x, drive_cmd_y = self.change_lane_lookahead,\
                        self.find_y(self.current_slope, self.current_y_intercept, self.change_lane_lookahead)
                '''
                # TODO: think about how to choose a lookahead point for adjacent line to avoid obstacle
                drive_cmd_x, drive_cmd_y = self.lookahead_point(adjacent_line)
                self.drive_cmd = self.get_drive_cmd(drive_cmd_x, drive_cmd_y, self.speed)
                self.drive_pub.publish(self.drive_cmd)
                self.line_last_found_time = rospy.get_time()
                print("follow adjacent line to avoid obstacle; x: %.2f, y: %.2f" % (drive_cmd_x, drive_cmd_y))
                return
            # if no adjacent lane, then stop
            else:
                print("no adjacent line found; stop to avoid crashing")
                self.drive_pub.publish(self.drive_cmd_msg(0, 0))
                return

        
        # -1 denotes no TA car found
        if vision_output.TAcar_distance > 0 and vision_output.TAcar_distance < self.follow_TA_max_distance_thr:
            x, y = vision_output.TAcar_loc
            self.drive_cmd = self.follow_TA_drive_cmd(x, y)
            self.drive_pub.publish(self.drive_cmd)
            return 

        elif self.timer > time.time():
            #  x coord 0, don't move
            drive_cmd_v = 0

        #Stop at stop sign for 5 seconds
        elif vision_output.is_stop_sign and vision_output.distance_blue_line < 0.1: #We have detected a stop sign 
            drive_cmd_v = 0
            self.timer = time.time() + 5
        
        #Stop when seeing pedestrain handling pedestrians
        elif vision_output.is_pedestrian:
            if vision_output.distance_to_pedestrian < self.distance_to_pedestrian_thr:
                drive_cmd_v = 0

        #Slow down when pedestrian sign exists
        elif vision_output.is_pedestrian_sign: 
            drive_cmd_v = self.cautious_speed

        #Stop when red or yellow traffic light for 5 seconds
        elif vision_output.is_traffic_light and not vision_output.is_green:
            drive_cmd_v = 0
    

        if current_line is None:
            if rospy.get_time() - self.line_last_found_time < self.reuse_drive_cmd_duration:
                print "No line found but reuse drive cmd; speed: %.2f, angle: %.2f"\
                    % (self.drive_cmd.drive.speed, self.drive_cmd.drive.steering_angle)
                self.drive_pub.publish(self.drive_cmd)
                return

            print "No line found; stopping"
            self.drive_cmd = self.drive_cmd_msg(0, 0)
            self.drive_pub.publish(self.drive_cmd)
            return

        '''
        self.current_slope, self.current_y_intercept = self.slope_and_intercept(current_line)
        self.current_theta, self.current_rho = utils.theta_rho_line(current_line)
        drive_cmd_x, drive_cmd_y = self.default_lookahead, self.find_y(self.current_slope,\
                    self.current_y_intercept, self.default_lookahead)
        '''
        drive_cmd_x, drive_cmd_y = self.lookahead_point(current_line)
        self.drive_cmd = self.get_drive_cmd(drive_cmd_x, drive_cmd_y, drive_cmd_v)
        self.drive_pub.publish(self.drive_cmd)
        self.line_last_found_time = rospy.get_time()
        print("follow current line; x: %.2f, y: %.2f" % (drive_cmd_x, drive_cmd_y))


    def points_to_line(p1, p2):
        A = (p1[1] - p2[1])
        B = (p2[0] - p1[0])
        C = (p1[0] * p2[1] - p2[0] * p1[1])
        return A, B, -C

    def intersection(L1, L2):
        D = L1[0] * L2[1] - L1[1] * L2[0]
        Dx = L1[2] * L2[1] - L1[1] * L2[2]
        Dy = L1[0] * L2[2] - L1[2] * L2[0]

        if D != 0:
            x = Dx / D
            y = Dy / D
            return x, y
        else:
            return False

    def find_y(self, slope, y_intercept, x):
        return slope * x + y_intercept

    def get_drive_cmd(self, x, y, v):
        """ P control """
        max_steering_angle = 0.34

        dist = np.sqrt(x**2+y**2)
        th = np.arctan2(y, x)
        
        #print("th: ", th)

        # Proportional Controller
        e = - dist
        #cmd_v = self.Kp1*e - self.Kp3*(math.pi/2 - abs(th))
        #cmd_th = -self.Kp2*th

        # pure pursuit?
        cmd_th = np.arctan2(2 * self.wheelbase * np.sin(th), self.default_lookahead)
        #print ("cmd_th", cmd_th)
        return self.drive_cmd_msg(cmd_th, v)

    def drive_cmd_msg(self, angle, v):
        drive_cmd = AckermannDriveStamped()
        drive_cmd.header.stamp = rospy.Time.now()
        drive_cmd.header.frame_id = "base_link"
        drive_cmd.drive.steering_angle = angle
        drive_cmd.drive.speed = v
        return drive_cmd


    def follow_TA(self, x, y):
        relative_x = x
        relative_y = y
        drive_cmd = AckermannDriveStamped()
        
        #################################
        # Play with this number too
        TA_car_desired_distance = self.TA_car_desired_distance
        
        #We can first get theta and distance for our controller involves:
        dist = np.sqrt(relative_x**2+relative_y**2)
        th = np.arctan(relative_y/relative_x)
        
        #dist = np.sqrt(self.relative_x**2+self.relative_y**2)
        print("th: ", th)

        # proportional control
        e = TA_car_desired_distance - dist
        cmd_v = -self.Kp1*e # - self.Kp3*(math.pi/2 - abs(th))
        cmd_th = self.Kp2*th
        if e > 0:
            cmd_th *= -1

        return self.drive_cmd_msg(cmd_th, cmd_v) #Publish cmd to /drive topic


    def guess_centerline(self, centerlines):
        '''
        slope_error_thresh = 0.5
        slopes_pack = [[ y1-((y2-y1)/(x2-x1))*x1,i]\
                for i, (x1,y1,x2,y2) in enumerate(centerlines)\
                if abs((y2-y1)/(x2-x1) - self.current_slope)<slope_error_thresh]
        slopes_pack.sort(key = lambda x: abs(x[0] - self.current_y_intercept))
        if len(slopes_pack) > 1:
            return centerlines[slopes_pack[0][1]],centerlines[slopes_pack[1][1]] 
        elif len(slopes_pack) > 0:
            return centerlines[slopes_pack[0][1]],None
        else: 
            return None, None
        '''
        lines_theta_rho = [utils.theta_rho_line(line) for line in centerlines]

        # for turning, look at horizontal line ahead
        horizontal_line_index = None
        for i, (theta, rho) in enumerate(lines_theta_rho):
            if abs(theta) < self.theta_threshold and rho < self.rho_threshold:
                found_horizontal_line_index = i
                self.found_horizontal_lines_array[self.frame_count % self.num_frames_turning_window] = 1
                print "found horizontal line %d out of the last %d frames; theta: %.2f, rho: %.2f" % (sum(self.found_horizontal_lines_array), self.num_frames_turning_window, theta, rho)
                if sum(self.found_horizontal_lines_array) > self.num_frames_turning_threshold:
                    print "TURNING about to begin!"
                    self.is_turning = True
                    self.straight_duration = 1. * (rho - self.distance_to_horizontal_line_before_turning) / self.speed
                    print("straight duration: %.2f" % self.straight_duration)
                    self.turn_start_time = rospy.get_time()
                    return centerlines[i], None
 
            else:
                self.found_horizontal_lines_array[self.frame_count % self.num_frames_turning_window] = 0
 
        if len(centerlines) == 1:
            return centerlines[0], None
        elif len(centerlines) == 0:
            return None, None

        indices_sorted = sorted(range(len(lines_theta_rho)),\
            key=lambda i: abs(lines_theta_rho[i][1] - self.current_rho))
        return centerlines[indices_sorted[0]], centerlines[indices_sorted[1]]


    def find_cross_street():
        """find most likely cross street if exists"""
        pass


    def area(self, bbox):
        return abs((bbox.x1 - bbox.x2) * (bbox.y1 - bbox.y2))


    def largest_bbox_area(self, bboxes):
        if len(bboxes) > 0:
            return max(self.area(bbox) for bbox in bboxes)
        return None


    def slope_and_intercept(self, line):
        x1, y1, x2, y2 = line
        if x1 == x2:
            x2 += 1e-6
        slope = (y2 - y1) / (x2 - x1)
        y_intercept = y1 - slope * x1
        return slope, y_intercept


    def two_lines_intersection(self, line1, line2):
        m1, b1 = self.slope_and_intercept(line1)
        m2, b2 = self.slope_and_intercept(line2)
        if m1 == m2:
            print("Bug: two lines are parallel! They should be almost perpendicular")
            return None, None
        x = (b1 - b2) / (m2 - m1)
        y = m1 * x + b1
        return x, y


    def lookahead_point(self, line):
        x1, y1, x2, y2 = line
        # drive_cmd_x, drive_cmd_y = 0.5 * (x1 + x2), 0.5 * (y1 + y2)
        if x1 * x1 + y1 * y1 > x2 * x2 + y2 * y2:
            return x1, y1
        else:
            return x2, y2


if __name__ == "__main__":
    rospy.init_node("controller")
    controller = Controller()
    rospy.spin()
