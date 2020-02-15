#!/usr/bin/env python2

from cone_segmentation.py import check_cones
'''
Assume we have an image passed in, and that we have an instance of the controller class (self).
Assume we have an output msg from vision (vision_msg).
'''


def check_if_parking(self,img):
    pcones = check_cones(img)
    if pcones is not None: #i.e. if we detect a cone
        if pcones["total_num_cone"] >= 3: #i.e. if we detect some or all of the parking cones
            self.parking_mode = True
            self.left_pcone = pcones["left_cone"]
            self.center_pcone = pcone["middle_cone"]
            self.right_pcone = pcone["right_cone"]
    

def parking(self):
    #while self.parking_mode: #While we're in parking mode
    if self.is_disabled_parking: #If we detect disabled parking sign
        disabled_parking_bbox = vision_output.disabled_parking_bbox
        #Now we need to determine which space(s) we can park in
        x_left,y_left = self.left_pcone
        x_middle,y_middle = self.center_pcone
        x_right,y_right = self.right_pcone
                
        ((x1,y1),(x2,y2)) = disabled_parking_bbox #Get coordinates
        bbox_bottom_center_x = (x1+x2)/2
        if bbox_bottom_center_x > x_middle: #Disabled parking sign in right
            self.goal_point_x = (x_middle+x_right)/2
        else:                               #Disabled parking sign in left
            self.goal_point_x = (x_middle+x_left)/2
        
        #Now we need to figure out self.goal_point_y
        '''
        <USE HOMOGRAPHY TRANSFORMS TO FIND Y POINT>
            