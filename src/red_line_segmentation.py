import cv2 as cv
import numpy as np
import homography
import os
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--images_dir', default=r'C:\Users\mitadm\Documents\Ryan\6.141\final_challenge\keras-yolo3\ML_pipeline\red_line', help='Image directory')
    parser.add_argument('-b', '--begin_index', type=int, default=0,
        help='Start from the b-th file in images_dir')
    parser.add_argument('-n', '--num_images', type=int, default=1, help='Num images to run through')
    args = parser.parse_args()
    return args

def apply_color_segmentation(frame, lower_color, upper_color):
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    return cv.inRange(hsv, lower_color, upper_color)


def color_transform_red(frame):
    lower_color = np.array([0, 0, 75])
    upper_color = np.array([10, 255, 150])

    red_mask = apply_color_segmentation(frame, lower_color, upper_color)
    return red_mask

def check_red_line(frame,img_name): #Check to see if a red line(s) exists in a frame
    color_segmented_line = color_transform_red(frame)
    if True: #Try to detect red line
        objective = []
        points = []
        im_space_points = []
        im2, contours, hierarchy = cv.findContours(color_segmented_line, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        bboxes = []
        print(img_name)
        y_max = frame.shape[1]
        for contour in contours: #There may be multiple lines; we should always take the one closest to us
            bbox = cv.boundingRect(contour)
            u, v, w, h = bbox
            if v > y_max/3:
                (x1,y1) = (u,v+h/2)
                (x2,y2) = (u+w,v+h/2)
                xl,yl = homography.apply_homography(x1,y1) #Left end point of red line in world space
                xr,yr = homography.apply_homography(x2,y2) #Right end point of red line in world space
                im_space_points.append((x1,y1,x2,y2))
                bboxes.append((u,v,u+w,v+h))
                objective.append(w*h)
                points.append((xl,yl,xr,yr))
        
        segments_path = r"C:\Users\mitadm\Documents\Ryan\6.141\final_challenge\deep_learning\red_lines/"
        if not os.path.exists(segments_path):
            os.mkdir(segments_path)
        try:
            contour_index = np.argmax(objective)
            contour = contours[contour_index]
            (x1,y1,x2,y2) = bboxes[contour_index]
            cv.imwrite(segments_path+"_segmented_"+img_name,im2)
            im3 = cv.rectangle(frame, (x1,y1),(x2,y2), (0,255,0), 15)
            cv.imwrite(segments_path+"_cropped_"+img_name,im3)
        
            A = (x2-x1)*(y2-y1)
            if A > 5000 and (x2-x1)/(y2-y1) > 5: #Detection
                return points[contour_index],np.min(objective) #Returns xl,yl,xr,yr
            else:
                return None,None
        except:
            return None,None

def main():
    args = parse_args()   
    all_files = os.listdir(args.images_dir)
    filtered_files = []
    for file in all_files:
        if file.endswith(".jpeg") or file.endswith(".jpg") or file.endswith(".png"):
            filtered_files.append(file)
    
    for img_name in filtered_files[args.begin_index:args.begin_index+args.num_images]:
        img_path = os.path.join(args.images_dir, img_name)
        frame = cv.imread(img_path)
        line,dist = check_red_line(frame,img_name)
    cv.destroyAllWindows()        


if __name__ == '__main__':
    main()
