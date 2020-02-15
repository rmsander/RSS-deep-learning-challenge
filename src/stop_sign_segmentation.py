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

def color_transform_red(frame):
    lower_color = np.array([0.,140., 125])
    upper_color = np.array([10., 255., 190])

    orange_mask = apply_color_segmentation(frame, lower_color, upper_color)
    return orange_mask

def filter_area(contour):
    area = cv.contourArea(contour)
    print(area)
    return area > 10000

def crop_with_bbox(frame, bbox):
    bbox_tl = bbox[0]
    bbox_br = bbox[1]

    h, w = bbox_br[1] - bbox_tl[1], bbox_br[0] - bbox_tl[0]

    cropped = frame[bbox_tl[1]:bbox_br[1], bbox_tl[0]:bbox_br[0]]
    cropped = cropped[h//7: 6*h//7, w//7:6*w//7]
    return cropped


def check_stop_sign(frame):
    red_mask = color_transform_red(frame)

    kernel = np.ones((6,6),np.uint8)

    red_mask = cv.dilate(red_mask,kernel,iterations = 1)

    #cv.imshow('WHITE_MASK', red_mask)

    im2, contours, hierarchy = cv.findContours(red_mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    contours = filter(filter_area, contours)

    if len(contours) >= 1:
        return True
    return False


def main():
    args = parse_args()
    img_names = filter(lambda x: x.endswith('jpeg') or x.endswith('jpg') or x.endswith('png'),
        sorted(os.listdir(args.images_dir)))
    img_names = img_names[args.begin_index: args.begin_index + args.num_images]

    for img_name in img_names:
        img_path = os.path.join(args.images_dir, img_name)
        frame = cv.imread(img_path)

        cones = check_stop_sign(frame)
        print(cones)
        cv.imshow(img_name + " cones", frame)


        k = cv.waitKey(0)
        if k == 27:
            cv.destroyAllWindows()
            break
        cv.destroyAllWindows()
'''
frame = cv.imread('left2-0063.jpeg')
a = check_cones(frame)
print(a)

    #x2, y2 = x1 + w, y1 + h
    #cv.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

#mask = cv.bitwise_and(frame,frame,mask = color_segmented)
#img = do_canny(frame)
#segment = do_segment(img)
#combo = houghTransform(frame, segment)
#tb = time.time()
#print 'TOTAL: {}'.format(tb-ta)
cv.imshow('image', frame)

k = cv.waitKey(0)
if k == 27:
    cv.destroyAllWindows()
cv.destroyAllWindows()
'''

if __name__ == '__main__':
    main()



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
