from __future__ import division

import colorsys
import cv2
import numpy as np
import os
import rospy
from sensor_msgs.msg import Image, CompressedImage
from PIL import Image, ImageDraw, ImageFont
from homography import apply_homography


def compressed_imgmsg_to_cv2(msg):
    np_arr = np.fromstring(msg.data, np.uint8)
    return cv2.imdecode(np_arr, cv2.IMREAD_COLOR) 


def cv2_to_compressed_imgmsg(cv_image):
    # Create CompressedImage
    msg = CompressedImage()
    msg.header.stamp = rospy.Time.now()
    msg.format = "jpeg"
    msg.data = np.array(cv2.imencode('.jpg', cv_image)[1]).tostring()
    return msg


def draw_lines_colors(image, lines, colors, thickness):
    for line, color in zip(lines, colors):
        x1, y1, x2, y2 = line
        x1 = int(x1)
        y1 = int(y1)
        x2 = int(x2)
        y2 = int(y2)
        cv2.line(image, (x1, y1), (x2, y2), color, thickness)
    return image


def draw_lines(image, lines, line_types=None, thickness=5):
    if line_types is None:
        colors = [(0, 255, 255)] * len(lines)
    else:
        assert len(lines) == len(line_types),\
            "len(lines) = %d, must be equal to len(line_types) = %d" %\
            (len(lines), len(line_types))
        colors = []
        for line_type in line_types:
            if line_type == 'following':
                colors.append((0, 255, 0))
            elif line_type == 'obstructed':
                colors.append((0, 0, 255))
            elif line_type == 'other':
                colors.append((255, 0, 0))
    return draw_lines_colors(image, lines, colors, thickness)


REPO_ROOT = os.path.dirname(os.path.dirname(__file__))
DEFAULT_FONT_PATH = os.path.join(REPO_ROOT, "YOLOv3/font/FiraMono-Medium.otf")


class LabelsVisualizer:
    def __init__(self, class_names, size=(1280, 720), font_path=DEFAULT_FONT_PATH):
        self.class_names = class_names
        self.font = ImageFont.truetype(font=font_path, size=np.floor(3e-2 * size[1] + 0.5).astype('int32'))
        self.thickness = (size[0] + size[1]) // 300

        # Generate colors for drawing bounding boxes.
        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))
        np.random.seed(10101)  # Fixed seed for consistent colors across runs.
        np.random.shuffle(self.colors)  # Shuffle colors to decorrelate adjacent classes.
        np.random.seed(None)  # Reset seed to default.


    def draw_bboxes(self, image, b_boxes, scores, classes):

        array = np.uint8((image))
        image = Image.fromarray(array)

        # draw the bounding boxes
        for i, c in reversed(list(enumerate(classes))):

            predicted_class = self.class_names[c]
            box = b_boxes[i]
            score = scores[i]

            label = '{} {:.2f}'.format(predicted_class, score)
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, self.font)

            top, left, bottom, right = box
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
            right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
            print(label, (left, top), (right, bottom))

            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            # a good redistributable image drawing library.
            for i in range(self.thickness):
                draw.rectangle([left + i, top + i, right - i, bottom - i], outline=self.colors[c])
            draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)], fill=self.colors[c])
            draw.text(text_origin, label, fill=(0, 0, 0), font=self.font)
            del draw

        image = np.asarray(image)
        return image


def bbox_low_midpoint(bbox):
    u = 0.5 * (bbox.x1 + bbox.x2)
    v = bbox.y2
    return u, v


def image_to_ground(bbox):
    u, v = bbox_low_midpoint(bbox)
    return apply_homography(u, v)

def theta_rho_line(line):
    x1, y1, x2, y2 = line
    line_length = np.linalg.norm((x1 - x2) ** 2 + (y1 - y2) ** 2)
    if line_length == 0:
        return None, None
    # sine = (x2 - x1) / line_length
    # cosine = - (y2 - y1) / line_length
    angle = np.arctan2(x2 - x1, - (y2 - y1))
    dist_from_origin = (x2 * y1 - x1 * y2) / line_length
    if dist_from_origin < 0:
        dist_from_origin *= -1
        angle += np.pi
    angle = (angle + np.pi) % (2 * np.pi) - np.pi
    return angle, dist_from_origin

