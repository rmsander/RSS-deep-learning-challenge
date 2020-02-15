import sys
import argparse
from yolo import YOLO, detect_video
from PIL import Image
import cv2
import os
import matplotlib.pyplot as plt
from sklearn.metrics import multilabel_confusion_matrix,classification_report
import numpy as np
import seaborn as sn
import pandas as pd

DETECT_IMG = False
ACCURACY_TEST = False

def detect_img(yolo):
    while True:
        img = input('Input image filename:')
        try:
            image = Image.open(img)
        except:
            print('Open Error! Try again!')
            continue
        else:
            r_image = yolo.detect_image(image)
            r_image.show()
            #cv2.imwrite(detected_directory+img,r_image)
    yolo.close_session()
    
def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):
    """
    given a sklearn confusion matrix (cm), make a nice plot

    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix

    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']

    title:        the text to display at the top of the matrix

    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues

    normalize:    If False, plot the raw numbers
                  If True, plot the proportions

    Usage
    -----
    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                              # sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph

    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    """
    import matplotlib.pyplot as plt
    import numpy as np
    import itertools

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.show()
    
def detect_imgs(yolo):
    classes2numbers3 = {"DISABLED_PARKING_SIGN":0,"CONE":1,"YIELD_SIGN":2,
                        "PEDESTRIAN_SIGN":3,"SOCCER_BALL":4,"TA_CAR":5,
                        "TRAFFIC_LIGHT":6,"PERSON":7,"STOP_SIGN":8}
    dir_names = ["disabled_parking_signs/","cones/","yield_sign/",
                 "pedestrian_signs/","soccer_balls/","ta_car/",
                 "traffic_lights/","person/","stop_signs/"]
    custom = "FULL_REPORT"
    
    source_directory = "ML_pipeline/testing_data_report/"
    all_files = os.listdir(source_directory)
    classes2numbers3 = {"DISABLED_PARKING_SIGN":0,"CONE":1,"YIELD_SIGN":2,"PEDESTRIAN_SIGN":3,"SOCCER_BALL":4,"TA_CAR":5,"TRAFFIC_LIGHT":6,"PERSON":7,"STOP_SIGN":8}
    for i in range(len(dir_names)):
        detected_directory = dir_names[i]+"detections_"+custom+"/"
        images_directory = dir_names[i]+"images_"+custom+"/"
        out_file = dir_names[i]+"bboxs_"+custom+".txt"
        try:
            #os.system('powershell.exe mkdir '+source_directory)
            if not os.path.exists(images_directory):
                os.system('powershell.exe mkdir '+images_directory)
            if not os.path.exists(detected_directory):
                os.system('powershell.exe mkdir '+detected_directory)
            if not os.path.exists(dir_names[i]):        
                os.system('powershell.exe mkdir '+dir_names[i])
        except:
            print("Directories already created")
    
    files = []    
    for file in all_files:
        if file.endswith(".jpeg") or file.endswith(".jpg"):
            files.append(file)
    
    j = 5#len(files) #Detect first j files
    classes = np.zeros((j,9))
    k = 0
    files = []
    with open("ML_pipeline/test_report.txt") as f:
        for line in f:
            words = line.split(" ")
            print("Number of words:",len(words))
            if len(words) <= 3:
                files.append(words[0][32:])
    f.close()
    
    for file in files[:j]:
        print("k={}".format(k))
        print(file)
        try:
            image = Image.open(source_directory+file)
        except:
            print('Open Error! Try again!')
            continue
        else:
            r_image,box,class_value = yolo.detect_images(image)
            #r_image.show()
            if class_value is not None:
                for ind_class in class_value:
                    print(class_value)
                    detected_directory = dir_names[classes2numbers3[ind_class]]+"detections_"+custom+"/"
                    images_directory = dir_names[classes2numbers3[ind_class]]+"images_"+custom+"/"
                    out_file = dir_names[classes2numbers3[ind_class]]+"bboxs_"+custom+".txt"
                    r_image.save(detected_directory+file,"JPEG")
                    image.save(images_directory+file,"JPEG")
                    r_image.show()
        if class_value is not None: #i.e. if we detect any objects
            for ind_class in class_value:
                print("INDIVIDUAL CLASS:",ind_class)
                classes[k][classes2numbers3[ind_class]] += 1

        else:
            classes[k] = np.zeros((1,9))
        k += 1


        with open(out_file,"a") as f:
            if box is not None:
                write = ((box[0],box[1]),(box[2],box[3]))
                f.write(file+" "+str(write)+"\n")
        f.close()
    print(classes)
    return classes


def accuracy_reporting(yolo):
    files_to_test = []
    with open("ML_pipeline/train3.txt") as f:
        for line in f:
            words = line.split(" ")
            print(words[0][27:])
            print(words[1])
            files_to_test.append(words[0][27:])
    for i in range(len(files_to_test)):
        pass


def evaluate(classes):
    classes2numbers3 = {"DISABLED_PARKING_SIGN":0,"CONE":1,"YIELD_SIGN":2,
                        "PEDESTRIAN_SIGN":3,"SOCCER_BALL":4,"TA_CAR":5,
                        "TRAFFIC_LIGHT":6,"PERSON":7,"STOP_SIGN":8}
    numbers2classes3 = {list(classes2numbers3.values())[i]:list(classes2numbers3.keys())[i] for i in range(9)}
    k = classes.shape[0]
    labels = np.zeros((k,9))
    files = []
    i = 0
    with open(r"C:\Users\mitadm\Documents\Ryan\6.141\final_challenge\keras-yolo3\ML_pipeline\test_report.txt") as f:
        for line in f:
            if i > k:
                break
            print("i={}".format(i))
            words = line.split(" ")
            if len(words) <= 3:
                print(words[0][33:])
                for j in range(len(words)):
                    if j == 0:
                        continue
                    try:
                        if words[j][-2] == ",":
                            print("Label=",numbers2classes3[int(words[j][-3])])
                            labels[i][int(words[j][-3])] += 1
                    except:
                                #print("Not a word")
                        continue
                    j += 1
            files.append(words[0][27:])
            i += 1

    #print("Labels are:",labels)
    f.close()
    print("LABELS:",labels)
    print("PREDICTED CLASSES:",classes)
    print(CM)
    plot_confusion_matrix(CM, list(classes2numbers3.keys()),
                          title='Confusion matrix', cmap=None, normalize=True)

FLAGS = None

if __name__ == '__main__':
    # class YOLO defines the default value, so suppress any default here
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    '''
    Command line options
    '''
    parser.add_argument(
        '--model', type=str,
        help='path to model weight file, default ' + YOLO.get_defaults("model_path")
    )

    parser.add_argument(
        '--anchors', type=str,
        help='path to anchor definitions, default ' + YOLO.get_defaults("anchors_path")
    )

    parser.add_argument(
        '--classes', type=str,
        help='path to class definitions, default ' + YOLO.get_defaults("classes_path")
    )

    parser.add_argument(
        '--gpu_num', type=int,
        help='Number of GPU to use, default ' + str(YOLO.get_defaults("gpu_num"))
    )

    parser.add_argument(
        '--image', default=False, action="store_true",
        help='Image detection mode, will ignore all positional arguments'
    )
    '''/
    Command line positional arguments -- for video detection mode
    '''
    parser.add_argument(
        "--input", nargs='?', type=str,required=False,default='./path2your_video',
        help = "Video input path"
    )

    parser.add_argument(
        "--output", nargs='?', type=str, default="",
        help = "[Optional] Video output path"
    )

    FLAGS = parser.parse_args()

    if FLAGS.image:
        """
        Image detection mode, disregard any remaining command line arguments
        """
        print("Image detection mode")
        if "input" in FLAGS:
            print(" Ignoring remaining command line arguments: " + FLAGS.input + "," + FLAGS.output)
        if DETECT_IMG:
            detect_img(YOLO(**vars(FLAGS)))
        elif ACCURACY_TEST:
            accuracy_reporting(YOLO(**vars(FLAGS)))
        else:
            classes = detect_imgs(YOLO(**vars(FLAGS)))
            evaluate(classes)
    elif "input" in FLAGS:
        detect_video(YOLO(**vars(FLAGS)), FLAGS.input, FLAGS.output)
    else:
        print("Must specify at least video_input_path.  See usage with --help.")
