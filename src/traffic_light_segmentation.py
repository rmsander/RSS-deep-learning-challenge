import cv2 as cv
import numpy as np
import time
from sklearn.cluster import DBSCAN
import os


import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--images_dir', default='training_images/20190509', help='Image directory')
    parser.add_argument('-b', '--begin_index', type=int, default=541,
        help='Start from the b-th file in images_dir')
    parser.add_argument('-n', '--num_images', type=int, default=20, help='Num images to run through')
    args = parser.parse_args()
    return args

def apply_color_segmentation(frame, lower_color, upper_color):
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    return cv.inRange(hsv, lower_color, upper_color)

def color_transform_white(frame):
    lower_color = np.array([0., 0., 160])
    upper_color = np.array([255., 100., 255])

    orange_mask = apply_color_segmentation(frame, lower_color, upper_color)
    return orange_mask

def filter_area(contour):
    area = cv.contourArea(contour)
    print(area)
    return area > 15. and area < 70.

def crop_with_bbox(frame, bbox):
    bbox_tl = bbox[0]
    bbox_br = bbox[1]

    h, w = bbox_br[1] - bbox_tl[1], bbox_br[0] - bbox_tl[0]

    cropped = frame[int(bbox_tl[1]):int(bbox_br[1]), int(bbox_tl[0]):int(bbox_br[0])]
    cropped = cropped[int(h//7): int(6*h//7), int(w//7):int(6*w//7)]
    return cropped


def check_traffic_light(frame, bbox):
    cropped = crop_with_bbox(frame, bbox)
    white_mask = color_transform_white(cropped)
    cv.imshow('WHITE_MASK', white_mask)

    im2, contours, hierarchy = cv.findContours(white_mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    contours = filter(filter_area, contours)

    #cv.drawContours(cropped, contours, -1, (0, 255, 0), thickness=2)

    if len(contours) <= 0 or len(contours) > 2:
        return {'detected_traffic_stop': False}
        print 'FOUND {} CONTOURS'.format(len(contours))

    detected_red, detected_yellow, detected_green = False, False, False
    traffic_light_h, _ = cropped.shape[:2]

    for contour in contours:
        M = cv.moments(contour)
        centroid_Y = int(M["m01"] / M["m00"])

        if centroid_Y < traffic_light_h / 3:
            detected_red = True
        elif centroid_Y < traffic_light_h / 2:
            detected_yellow = True
        else:
            detected_green = True
    print('RED: {} YELLOW: {} GREEN {}'.format(detected_red, detected_yellow, detected_green))
    return {'detected_traffic_stop': detected_red or detected_yellow}


#frame = cv.imread('deep_learning/training_images/traffic_lights/IMG_20190511_224838.jpg')
#bbox = ((0, 0), (frame.shape[:2][1], frame.shape[:2][0]))
'''
cv.imshow('IMAGE_FRAME', frame)

print(check_traffic_light(cropped))


cv.imshow('IMAGE', cropped)


k = cv.waitKey(0)
if k == 27:
    cv.destroyAllWindows()
cv.destroyAllWindows()
'''
