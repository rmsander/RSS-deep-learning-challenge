import cv2
import numpy as np

block_size = 0.30 # meters

srcPoints = np.array([
(745, 572),
(713, 460),
(697, 412),
(331, 561),
(455, 456),
(512, 408),
(1182, 578),
(978, 462),
(887, 413)
])

dstPoints = np.array([
(1, 0),
(2, 0),
(3, 0),
(1, 1),
(2, 1),
(3, 1),
(1, -1),
(2, -1),
(3, -1)
])

dstPoints = dstPoints + (0.5, 0)
dstPoints = block_size * dstPoints

h, mask = cv2.findHomography(srcPoints, dstPoints)
# print h


def apply_homography(u, v, h=h):
    uv_point = np.array([u, v, 1]).reshape(3, 1)
    xyz_point = np.matmul(h, uv_point)
    xy_point = xyz_point / xyz_point[2]
    return xy_point.item(0), xy_point.item(1)



world_to_orig_image = np.linalg.inv(h)

def world_to_orig_image_fn(points):
    # TODO: use np.vectorize?
    new_points = []
    for x, y in points:
        u, v = apply_homography(x, y, world_to_orig_image)
        new_points.append((u, v))
    return np.array(new_points)

pts1 = np.float32([[0, 0.3], [0, -0.3], [1.05, 0.3], [1.05, -0.3]])
# pts2 = np.float32([[0, 105], [60, 105], [0, 0], [60, 0]])
# pts2 = np.float32([[70, 400], [130, 400], [70, 295], [130, 295]])
scale = 150.
scale_inverse = 1. / scale
x_to_view, y_to_view = 2, 3
warped_image_size = (int(2 * y_to_view * scale), int(x_to_view * scale))

world_to_warped_image = np.float32([
    [0, -1, y_to_view],
    [-1, 0, x_to_view],
    [0, 0, 1]
    ])
# multiple the first two rows
world_to_warped_image[:2,] *= scale

warped_image_to_world = np.linalg.inv(world_to_warped_image)
# print warped_image_to_world


def world_to_warped_image_fn(points):
    points = - points[:, ::-1]
    return (points + np.float32([y_to_view, x_to_view])) * scale

def warped_image_to_world_fn(points):
    points = - scale_inverse * points[:, ::-1]
    return points + np.float32([x_to_view, y_to_view])
image_to_warped_image = np.matmul(world_to_warped_image, h)


if __name__ == '__main__':
    print apply_homography(745, 572)


    pts_world = np.float32([[0.3, 0.002], [1.05, 0.027]])
    pts_warped = []
    for x, y in pts_world:
        u, v = apply_homography(x, y, world_to_warped_image)
        pts_warped.append((u, v))
    pts_warped = np.float32(pts_warped)
    print "pts_world", pts_world
    print "pts_warped", pts_warped

