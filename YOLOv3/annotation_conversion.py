import json
import os

W_SCALE = 1
H_SCALE = 1

export_path = "ML_pipeline/training_data_copy3/vott-json-export/"
raw_data_path = "ML_pipeline/raw_data_copy3/"
training_data_path = "ML_pipeline/training_data_copy3/"

json_data_names = os.listdir(training_data_path)
with open("jsons.txt","w+") as f:
    for i in range(len(json_data_names)):
        f.write(json_data_names[i]+"\n")
f.close()

cwd = os.getcwd()
assets = []
with open("jsons.txt") as f:
    for line in f:
        #print(line)
        line = line[:-1]
        if line.endswith("json"):
            assets.append(line)
print(type(assets[0]))
print(assets[0])
f.close()

#for i in range(len(assets)):
#    with open(training_data_path+assets[i],"r") as f:
#        data = json.load(f)
 #       bboxes.append()
 #   f.close()

#annotations = []
#with open() as f:
#    for line in f:
#       line = line[:-5]
#       annotations.append(line)
#f.close()
#print("ANNOTATIONS",annotations[-1])

image_names = os.listdir(raw_data_path)
# =============================================================================
# with open("images.txt") as f:
#     for line in f:
#         #image_names.append(line)
#         line2 = line[:-6]
#         print(line2)
#         if line2 not in annotations:
#             try:
#                 os.remove(cwd+"\\ML_pipeline\\raw_training_data)copy\\"+line[:-1])
#             except:
#                 "File no longer exists"
#         else:
#             image_names.append(line[:-1])
# =============================================================================

train = open("train.txt","w")
for i in range(len(image_names)):
    print(image_names[i])
    train.write(cwd+"\\cus\\images\\"+image_names[i])
    train.write("\n")
train.close()

#print(assets)
images = []
bboxes = []
names  = []
classes = []
for asset in assets:
    with open(training_data_path+asset) as json_data:
        d = json.load(json_data)
        names.append(d['asset']['name'])
        bboxes.append(d['regions'])
        json_data.close()
print(d['asset'])

classes2numbers = {"TRAFFIC_LIGHT":0,"PERSON":1,"STOP_SIGN":2,"YIELD_SIGN":3,"CONE":4,"PEDESTRIAN_SIGN":5,"DISABLED_PARKING_SIGN":6,"DISABLED_PARKING_SPACE":6}
classes2numbers2 = {"DISABLED_PARKING_SIGN":0,"CONE":1,"YIELD_SIGN":2,"PEDESTRIAN_SIGN":3,"SOCCER_BALL":4,"TA_CAR":5}
classes2numbers3 = {"DISABLED_PARKING_SIGN":0,"CONE":1,"YIELD_SIGN":2,"PEDESTRIAN_SIGN":3,"SOCCER_BALL":4,"TA_CAR":5,"TRAFFIC_LIGHT":6,"PERSON":7,"STOP_SIGN":8}

def format_1():
    for i in range(len(assets)):
        dic = bboxes[i]
        file = open("annotations/"+names[i][:-5]+".txt","w")
        for data in dic:
            #if i == 0:
            label = classes2numbers[data['tags'][0]]
            bbox = data["boundingBox"]
            w = bbox['width']/1280
            h = bbox['height']/720
            l = bbox['left']/1280
            t = bbox['top']/720
            x_cen = l + w/2
            y_cen = t + h/2
            file.write(str(label)+" "+str(x_cen)+" "+str(y_cen)+" "+str(w)+" "+str(h))
        file.close()

try:
    file = open("ML_pipeline/train3.txt","w")
except:
    print("Could not create file")
print("LENGTH OF ASSETS:",len(assets))

for i in range(len(assets)):
    dic = bboxes[i]
    file.write(raw_data_path+names[i]+" ")
    for data in dic:
        #if i == 0:
        label = classes2numbers3[data['tags'][0]]
        bbox = data["boundingBox"]
        w = bbox['width']*W_SCALE
        h = bbox['height']*H_SCALE
        l = bbox['left']*W_SCALE
        t = bbox['top']*H_SCALE
        x_min = l
        y_min = t
        x_max = l+w
        y_max = t+h
        file.write(str(int(x_min))+","+str(int(y_min))+","+str(int(x_max))+","+str(int(y_max))+","+str(label)+" ")
    file.write("\n")
file.close()
    

        