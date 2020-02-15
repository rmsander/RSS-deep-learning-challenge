import pickle
import matplotlib.pyplot as plt

history_1_tiny = pickle.load(
    open("ML_pipeline/pickle_data/stage_1_history_tiny.pickle", "rb"))
history_2_tiny = pickle.load(
    open("ML_pipeline/pickle_data/stage_2_history_tiny.pickle", "rb"))
history_1 = pickle.load(
    open("ML_pipeline/pickle_data/stage_1_history_full.pickle", "rb"))
history_2 = pickle.load(
    open("ML_pipeline/pickle_data/stage_2_history_full.pickle", "rb"))

keys = list(history_1_tiny.keys())
print(keys)

tiny_history = {keys[i]: history_1_tiny[keys[i]] + history_2_tiny[keys[i]] for i
                in range(len(keys))}
full_history = {keys[i]: history_1[keys[i]] + history_2[keys[i]] for i in
                range(len(keys))}

xs = [i for i in range(50)]
for key in keys:
    plt.plot(xs, tiny_history[key], "r", label="tiny-YOLOv3")
    plt.plot(xs, full_history[key], "b", label="YOLOv3")
    if key == "val_class_loss":
        key = "Validation Class Loss"
    elif key == "val_xy_loss":
        key = "Validation xy Loss"
    elif key == "val_wh_loss":
        key = "Validation wh Loss"
    elif key == "val_loss":
        key = "Validation Loss"
    plt.legend(loc="upper right")
    plt.title(key + " vs. Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.savefig("ML_pipeline/pickle_data/" + key + ".png")
    plt.clf()
