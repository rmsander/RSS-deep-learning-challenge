import numpy as np
import os
from shutil import copyfile

files_to_test = []
labels_to_test = []
with open(
        r"C:\Users\mitadm\Documents\Ryan\6.141\final_challenge\keras-yolo3"
        r"\ML_pipeline\train3.txt") as f:
    for line in f:
        words = line.split(" ")
        labels_to_test.append([words[i][-1] for i in range(1, len(words))])
        files_to_test.append(words[0][27:])
f.close()

path_to_data = "ML_pipeline/raw_data_copy3/"
path_training = "ML_pipeline/training_data_report/"
path_testing = "ML_pipeline/testing_data_report/"

if not os.path.exists(path_training):
    os.mkdir(path_training)
if not os.path.exists(path_testing):
    os.mkdir(path_testing)

files_to_test = np.array(files_to_test)
np.random.shuffle(files_to_test)

n = len(files_to_test)
m = int(4 * n / 5)
training_files = []
test_files = []

for i in range(m):  # Make training data
    training_files.append(files_to_test[i])
    copyfile(path_to_data + files_to_test[i], path_training + files_to_test[i])

for j in range(m, n):
    test_files.append(files_to_test[j])
    copyfile(path_to_data + files_to_test[j], path_testing + files_to_test[j])

# =============================================================================
# count = 0 #Score
#     for i in range(len(files_to_test)):
#         for label in labels_to_test:
#             r_image,box,classes = yolo.detect_imgs(files_to_test[i])
#             if classes == labels_to_test[-1]:
#                 count += 1
#     print("Detection accuracy:",count/len(files_to_test))
#     return count/len(files_to_test)
# 
# print(accuracy_reporting())
# =============================================================================
