import cv2 as cv
import numpy as np
import time
import homography
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

def color_transform_orange(frame):
    #white lower
    lower_color = np.array([0, 180, 160])
    upper_color = np.array([15, 255, 255])

    orange_mask = apply_color_segmentation(frame, lower_color, upper_color)
    return orange_mask

def color_transform_blue(frame):
    lower_color = np.array([110, 100, 40])
    upper_color = np.array([130, 255, 255])

    blue_mask = apply_color_segmentation(frame, lower_color, upper_color)
    return blue_mask


def filter_area(contour):
    area = cv.contourArea(contour)
    return area > 400.

def filter_handicapped_sign(contour):
    area = cv.contourArea(contour)
    return area > 1000

def check_handicapped_parking(frame):
    h, w, channels = frame.shape
    chopped_frame = frame[200:,] # chopped top forth of frame off
    blue_mask = color_transform_blue(chopped_frame)
    kernel = np.ones((5,5),np.uint8)

    blue_mask = cv.dilate(blue_mask,kernel,iterations = 1)

    im2, contours, hierarchy = cv.findContours(blue_mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    contours_filtered= filter(filter_handicapped_sign, contours)

    blank_image = np.copy(frame)*0
    #cv.drawContours(chopped_frame, contours_filtered, -1, (0, 255, 0), thickness=cv.FILLED)
    #cv.imshow('BLUE', blue_mask)

    if len(contours_filtered) != 1:
        print 'ERROR: FOUND {} CONTOURS FOR HANDICAP SIGN'.format(len(contours_filtered))
        return None

    M = cv.moments(contours_filtered[0])
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"]+200)

    return cX, cY

def check_cones(frame):
    color_segmented_cones = color_transform_orange(frame)
    im2, contours, hierarchy = cv.findContours(color_segmented_cones, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    contours.sort(key=lambda x: cv.contourArea(x), reverse=True)
    contours = filter(filter_area, contours)
    total_cone_num = len(contours)
    if total_cone_num == 0:
        return None

    cone_points = []

    for contour in contours:
        bbox = cv.boundingRect(contour)
        x1, y1, w, h = bbox
        bottom_center = (x1 + w//2, y1 + h)
        # pt1 = homography.apply_homography(*bottom_center)
        cone_points.append(bottom_center)

    goal_point_largest_3 = None
    goal_point_smallest_2 = None
    goal_point_smallest = None
    size_smallest_cone = None

    if len(cone_points) >= 3:
        front_three = cone_points[:3]
        front_three.sort(key=lambda x: x[0])
        left_cone, middle_cone, right_cone = front_three

        middle_left = ((left_cone[0] + middle_cone[0]) / 2, (left_cone[1] + middle_cone[1]) / 2)
        middle_right = ((middle_cone[0] + right_cone[0]) / 2, (middle_cone[1] + right_cone[1]) / 2)

        _, width, _ = frame.shape
        if middle_cone[0] < 3./5 * width:
            lane = 3
        else:
            lane = 2

        print 'MIDDLE RIGHT {} MIDDLE LEFT {}'.format(middle_right, middle_left)
        handicapped = check_handicapped_parking(frame)
        if handicapped is None:
            if lane == 2:
                goal_point_largest_3 = middle_left
            else:
                goal_point_largest_3 = middle_right
        else:
            if handicapped[0] < middle_cone[0]:
                goal_point_largest_3 = middle_right
            else:
                print 'HANDICAPPED: {} MIDDLE CONE: {}'.format(handicapped, middle_cone)
                goal_point_largest_3 = middle_left

    if len(cone_points) >= 2:
        smallest_first, smallest_second = cone_points[-1], cone_points[-2]
        goal_point_smallest_2 = ((smallest_first[0] + smallest_second[0]) / 2, (smallest_first[1] + smallest_second[1]) / 2)

    if len(cone_points) >= 1:
        goal_point_smallest = cone_points[-1]

    if len(contours) >= 1:
        size_smallest_cone = cv.contourArea(contours[-1])

    '''cv.circle(frame, left_cone, 3, (255, 0, 0), 2)
    cv.circle(frame, middle_cone, 3, (0, 255, 0), 2)
    cv.circle(frame, right_cone, 3, (0, 0, 255), 2)'''

    return {'goal_point_largest_3': goal_point_largest_3, \
    'goal_point_smallest_2': goal_point_smallest_2,
    'goal_point_smallest': goal_point_smallest,
    'size_smallest_cone': size_smallest_cone }



def main():
    args = parse_args()
    img_names = filter(lambda x: x.endswith('jpeg') or x.endswith('jpg') or x.endswith('png'),
        sorted(os.listdir(args.images_dir)))
    img_names = img_names[args.begin_index: args.begin_index + args.num_images]

    for img_name in img_names:
        img_path = os.path.join(args.images_dir, img_name)
        frame = cv.imread(img_path)

        cones = check_cones(frame)
        print(cones)
        cv.imshow(img_name + " cones", frame)


        k = cv.waitKey(0)
        if k == 27:
            cv.destroyAllWindows()
            break
        cv.destroyAllWindows()

if __name__ == '__main__':
    main()
