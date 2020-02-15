def accuracy_reporting(yolo):
    files_to_test = []
    labels_to_test = []
    with open(
            r"C:\Users\mitadm\Documents\Ryan\6.141\final_challenge\keras"
            r"-yolo3\ML_pipeline\train3.txt") as f:
        for line in f:
            words = line.split(" ")
            labels_to_test.append([words[i][-1] for i in range(1, len(words))])
            files_to_test.append(words[0][27:])
    f.close()

    count = 0  # Score
    for i in range(len(files_to_test)):
        for label in labels_to_test:
            r_image, box, classes = yolo.detect_imgs(files_to_test[i])
            if classes == labels_to_test[-1]:
                count += 1
    print("Detection accuracy:", count / len(files_to_test))
    return count / len(files_to_test)
