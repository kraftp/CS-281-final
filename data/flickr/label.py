import os
import csv
import shutil

def get_images_in_path(rel_path):
    path = os.path.abspath(rel_path)
    files = filter(
        lambda x: os.path.isfile(os.path.join(path, x)),
        os.listdir(path)
    )
    return files

for filedir in ["./train/", "./test/"]:
    files = get_images_in_path(filedir)
    with open(os.path.join(filedir, 'labels.txt'), 'wb') as labels_file:
        labels = csv.writer(labels_file, delimiter=' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for f in files:
            label = 1 if f.split("_")[0] == "dog" else 0
            labels.writerow([f, label])
