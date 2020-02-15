from __future__ import division

import argparse
import cv2 as cv
import numpy as np
import os
import time
from sklearn.cluster import DBSCAN
# from IPython import embed
#import utils

import homography

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


def color_transform(frame):
    #white lower
    white_lower_color = np.array([0., 0., 160])
    white_upper_color = np.array([255., 40., 255])

    black_lower_color = np.array([0., 0., 0.])
    black_upper_color = np.array([255., 127., 70.])

    white_mask = apply_color_segmentation(frame, white_lower_color, white_upper_color)

    black_mask = apply_color_segmentation(frame, black_lower_color, black_upper_color)

    kernel = np.ones((5,5), np.uint8)
    #white_mask = cv.dilate(white_mask, kernel, iterations=1)

    # erosion followed by dilation, clean up small dots
    black_mask = cv.morphologyEx(black_mask, cv.MORPH_OPEN, kernel)


    # white_mask = cv.morphologyEx(white_mask, cv.MORPH_OPEN, kernel)
    white_mask = cv.morphologyEx(white_mask, cv.MORPH_CLOSE, kernel)
    # white_mask = cv.dilate(white_mask, kernel, iterations=1)

    return white_mask, black_mask
    '''
    mask = cv.bitwise_or(black_mask,white_mask)
    return mask
    '''


def do_canny(frame):
    # Converts frame to grayscale because we only need the luminance channel for detecting edges - less computationally expensive
    #gray = cv.cvtColor(frame, cv.COLOR_RGB2GRAY)
    # Applies a 5x5 gaussian blur with deviation of 0 to frame - not mandatory since Canny will do this for us
    blur = cv.GaussianBlur(frame, (5, 5), 0)
    # Applies Canny edge detector with minVal of 50 and maxVal of 150
    canny = cv.Canny(blur, 100, 150)
    return canny

def do_segment(frame):
    # Since an image is a multi-directional array containing the relative intensities of each pixel in the image, we can use frame.shape to return a tuple: [number of rows, number of columns, number of channels] of the dimensions of the frame
    # frame.shape[0] give us the number of rows of pixels the frame has. Since height begins from 0 at the top, the y-coordinate of the bottom of the frame is its height
    height = frame.shape[0]
    width = frame.shape[1]
    # Creates a triangular polygon for the mask defined by three (x, y) coordinates
    polygons = np.array([
                            [(0, height), (width, height), (width, height/2), (0, height/2)]
                        ])
    # Creates an image filled with zero intensities with the same dimensions as the frame
    mask = np.zeros_like(frame)
    # Allows the mask to be filled with values of 1 and the other areas to be filled with values of 0
    cv.fillPoly(mask, polygons, 255)
    # A bitwise and operation between the mask and frame keeps only the triangular area of the frame
    segment = cv.bitwise_and(frame, mask)
    return segment

def houghTransform(image, edges):
    rho = 1
    theta = np.pi/90
    threshold = 20
    min_line_length = 100
    max_line_gap = 40

    line_image = np.copy(image)*0 #creating a blank to draw lines on

    # Run Hough on edge detected image
    lines = cv.HoughLinesP(edges, rho, theta, threshold, np.array([]), min_line_length, max_line_gap)
    #print(lines)
    # Iterate over the output "lines" and draw lines on the blank
    # t1 = time.time()
    # lines_sorted = sorted(lines, key=lambda arr: (arr[0][0]-arr[0][2])**2 + (arr[0][1]-arr[0][3])**2, reverse=True)
    # t2 = time.time()
    # print 'SORTING: {}'.format(t2-t1)
    if not lines is None:
        for line in lines:
            for x1,y1,x2,y2 in line:
                cv.line(line_image,(x1,y1),(x2,y2),(0,255,0),1)

    # Create a "color" binary image to combine with line image
    color_edges = np.dstack((edges, edges, edges))

    # Draw the lines on the edge image
    combo = cv.addWeighted(color_edges, 1, line_image, 1, 0)
    combo = cv.addWeighted(image, 1, combo, 1, 0)
    return combo, lines

def filter_area(contour):
    area = cv.contourArea(contour)
    return area < 10000. and area > 200.


def filter_area_warped(contour):
    # maybe do this based on shape too? e.g. not too thin
    area = cv.contourArea(contour)
    x, y, w, h = cv.boundingRect(contour)
    # return area < 400. and area > 40. and max(w, h) < 50
    return area < 400. and area > 50. and max(w, h) < 3 * np.sqrt(area)


def segments_L2_difference(segment1, segment2):
    m_1 = segment1[0]
    b_1 = segment1[1]

    m_2 = segment2[0]
    b_2 = segment2[1]

    return (m_1 - m_2)**2 + (b_1 - b_2)**2

'''
def homography_transform(lines):
    if lines is None:
        print 'CLUSTERS: 0'
        return

    X = []
    d = {}
    for line in lines:
        pt1 = homography.apply_homography(line[0], line[1])
        pt2 = homography.apply_homography(line[2], line[3])

        m = (pt1[1] - pt2[1]) / (pt1[0] - pt2[0])
        b = pt1[1] - m * pt1[0]

        X.append([m, b])
        d[(m, b)] = (pt1, pt2)

    db = DBSCAN(eps=.5, min_samples=5, metric=segments_L2_difference).fit(X)


    #transformed_array = [[homography.apply_homography(i[0][0], i[0][1]), homography.apply_homography(i[0][2], i[0][3])] for i in array]
    labels = db.labels_

    # Number of clusters in labels, ignoring noise if present.
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    print 'CLUSTERS: {}'.format(n_clusters)
'''

def cluster_lines(orig_lines):
    lines = []
    for x1, y1, x2, y2 in orig_lines:
        # make x1 always <= x2
        if x1 > x2:
            x1, y1, x2, y2 = x2, y2, x1, y1
        lines.append((x1, y1, x2, y2))
    lines = np.array(lines)

    X = []
    d = {}
    def dist_function(line_segment1, line_segment2):
        #x11, y11, x12, y12 = line_segment1
        #x21, y21, x22, y22 = line_segment2
        return np.sum((line_segment1 - line_segment2)**2)

    # db = DBSCAN(eps=0.1, min_samples=5, metric=segments_L2_difference).fit(X)
    db = DBSCAN(eps=.2, min_samples=5, metric=dist_function).fit(lines)


    labels = db.labels_

    # Number of clusters in labels, ignoring noise if present.
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    print 'CLUSTERS: {}'.format(n_clusters)
    #print 'LABELS: {}'.format(labels)

    '''
    m_clusters = {k: [] for k in range(n_clusters)}
    b_clusters = {k: [] for k in range(n_clusters)}
    for i in range(len(X)):
        label = labels[i]
        if label != -1:
            m, b = X[i]
            m_clusters[label].append(m)
            b_clusters[label].append(b)

    clustered_lines = []
    for i in range(n_clusters):
        m = np.mean(m_clusters[i])
        b = np.mean(b_clusters[i])
        length = len(m_clusters[i])
        clustered_lines.append((m, b, length))
    '''
    clustered_lines = []

    for i in xrange(n_clusters):
        lines_filtered = lines[labels == i]


        x_min = np.mean(lines_filtered[:,0])
        x_max = np.mean(lines_filtered[:,2])
        y_min = np.mean(lines_filtered[:,1])
        y_max = np.mean(lines_filtered[:,3])

        clustered_lines.append((x_min, y_min, x_max, y_max))

    return clustered_lines


def increase_contrast(img):
        # CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv.createCLAHE(clipLimit=1.5, tileGridSize=(8,8))

    lab = cv.cvtColor(img, cv.COLOR_BGR2LAB)  # convert from BGR to LAB color space
    l, a, b = cv.split(lab)  # split on 3 different channels

    l2 = clahe.apply(l)  # apply CLAHE to the L-channel

    lab = cv.merge((l2,a,b))  # merge channels
    img2 = cv.cvtColor(lab, cv.COLOR_LAB2BGR)  # convert from LAB to BGR
    return img2


def draw_lines_colors(image, lines, colors, thickness):
    for line, color in zip(lines, colors):
        x1, y1, x2, y2 = line
        x1 = int(x1)
        y1 = int(y1)
        x2 = int(x2)
        y2 = int(y2)
        cv.line(image, (x1, y1), (x2, y2), color, thickness)
    return image

def process_frame_with_warped(frame):
    # frame = cv.addWeighted(frame, 2, frame, 0, 0)
    # frame = increase_contrast(frame)
    frame_warped = cv.warpPerspective(frame, homography.image_to_warped_image,
        homography.warped_image_size)
    h, w, _ = frame_warped.shape

    white_mask_warped, black_mask_warped = color_transform(frame_warped)

    # contours detection on black mask
    im2_warped, contours_warped, hierarchy_warped = cv.findContours(white_mask_warped, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    contours_filtered_warped = filter(filter_area_warped, contours_warped)
    line_image_warped = np.zeros((h, w), dtype=np.uint8)
    cv.drawContours(line_image_warped, contours_filtered_warped, -1, (255, 255, 255), thickness=6)

    # line_image_warped_with_rect = np.zeros((h, w), dtype=np.uint8)
    # for contour in contours_filtered_warped:
    #     rect = cv.minAreaRect(contour)
    #     box = cv.boxPoints(rect)
    #     # round float to int
    #     box = np.rint(box).astype(np.int64)
    #     cv.drawContours(line_image_warped_with_rect, [box], 0, (255, 0, 0), 2)

    combo1_warped, lines_warped = houghTransform(frame_warped, line_image_warped)
    if lines_warped is not None:
        lines_warped = np.reshape(lines_warped, (-1, 4))

    # cluster lines

    # transform line coordinates from warped to world
    clustered_lines = []
    lines_world = None
    if lines_warped is not None:
        endpoints_warped = np.reshape(lines_warped, (-1, 2)) # reshape to array of endpoints
        endpoints_world = homography.warped_image_to_world_fn(endpoints_warped)
        lines_world = np.reshape(endpoints_world, (-1, 4))


        clustered_lines = cluster_lines(lines_world)

    new_lines_world = np.reshape(clustered_lines, (-1, 2)) # reshape to array of endpoints
    new_lines_warped = homography.world_to_warped_image_fn(new_lines_world)
    new_lines_warped = np.reshape(new_lines_warped, (-1, 4))
    # print("..............", new_lines_warped)
    colors = [(0, 0, 255) for _ in clustered_lines]
    combo1_warped_with_new_lines = draw_lines_colors(combo1_warped, new_lines_warped, colors, 5)



    # return clustered_lines, [frame_warped, black_mask_warped, line_image_warped, combo1_warped_with_new_lines]
    return clustered_lines, [frame_warped, white_mask_warped, line_image_warped, combo1_warped_with_new_lines]


def centerlines(frame):
    '''
        input: frame (cv2 image)
        output: list of lines (each row is 1D np vector [x1, y1, x2, y2])
    '''
    # TODO: your code here
    centerlines, _ = process_frame_with_warped(frame)
    return centerlines


def stack_four_images(images):
    for i, image in enumerate(images):
        if image.ndim == 2:
            images[i] = cv.cvtColor(image, cv.COLOR_GRAY2BGR)

    left = np.vstack((images[0], images[2]))
    right = np.vstack((images[1], images[3]))
    total = np.hstack((left, right))
    scale = 1.0
    total = cv.resize(total, (0, 0), fx=scale, fy=scale)
    return total


def main():
    args = parse_args()
    img_names = filter(lambda x: x.endswith('jpeg') or x.endswith('jpg') or x.endswith('png'),
        sorted(os.listdir(args.images_dir)))
    img_names = img_names[args.begin_index: args.begin_index + args.num_images]

    for img_name in img_names:
        img_path = os.path.join(args.images_dir, img_name)
        frame = cv.imread(img_path)

        '''
        _, img = color_transform(frame) #do_canny(frame)
        segmented_img = do_segment(img)

        im2, contours, hierarchy = cv.findContours(segmented_img, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        contours_filtered = filter(filter_area, contours)

        line_image = np.copy(frame)*0
        cv.drawContours(line_image, contours_filtered, -1, (255, 255, 255), thickness=cv.FILLED)
        canny_line_image = do_canny(line_image)

        kernel = np.ones((6,6),np.uint8)
        canny_line_image = cv.dilate(canny_line_image,kernel,iterations = 1)

        combo1, lines = houghTransform(frame, canny_line_image)
        transformed_lines = homography_transform(lines)

        #print(transformed_lines)


        first = frame
        second = np.dstack((segmented_img, segmented_img, segmented_img))
        third = line_image
        final = combo1

        total = stack_four_images([first, second, third, final])
        cv.imshow(img_name, total)
        '''

        total = process_frame_with_warped(frame)[1]
        total = stack_four_images(total)
        cv.imshow(img_name + " warped", total)


        k = cv.waitKey(0)
        if k == 27:
            cv.destroyAllWindows()
            break
        cv.destroyAllWindows()
        # except Exception as e:
        #     print(e)


        #total = np.concatenate((frame, segment), axis=0)



'''
#ta = time.time()
frame = cv.imread('test1.jpeg')
white_frame = color_transform(frame)
segment = do_segment(white_frame)
combo = houghTransform(frame, segment)
#mask = cv.bitwise_and(frame,frame,mask = x)
#img = do_canny(frame)
#segment = do_segment(img)
#combo = houghTransform(frame, segment)
#tb = time.time()
#print 'TOTAL: {}'.format(tb-ta)



cv.imshow('test1',combo)
cv.waitKey(10000)
cv.destroyAllWindows()
'''

if __name__ == '__main__':
    main()
