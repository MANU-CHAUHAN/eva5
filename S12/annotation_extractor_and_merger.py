from collections import defaultdict
import cv2
import os
import glob
import json
import csv
import pandas as pd

all_files = [os.path.join(os.getcwd(), "worker_dataset", f) for f in
             os.listdir(os.path.join(os.getcwd(), "worker_dataset")) if
             os.path.isfile(os.path.join(os.getcwd(), "worker_dataset", f))]

all_files = all_files + [os.path.join(os.getcwd(), "worker_dataset/new", f) for f in
                         os.listdir(os.path.join(os.getcwd(), "worker_dataset/new")) if
                         os.path.isfile(os.path.join(os.getcwd(), "worker_dataset", f))]

all_files = list(set(all_files))

print(len(all_files))

with open("annotations.json", "r") as f:
    data = json.load(f)

print(len(data))

annotated_data = defaultdict(dict)

for k, v in data.items():
    folder = os.path.join(os.getcwd(), "worker_dataset/new") if v['filename'].startswith("0") else os.path.join(
        os.getcwd(), "worker_dataset")

    full_file_path = os.path.join(folder, v['filename'])
    image_size = cv2.imread(full_file_path).shape
    h, w, c = image_size
    annotated_data[full_file_path] = {"image_h": h, "image_w": w}
    regions = []
    for item in v['regions']:
        shape_dict = item["shape_attributes"]
        regions.append({"class_name": item["region_attributes"]["class_name"],
                        "cx": shape_dict["x"],
                        "cy": shape_dict["y"],
                        "bbox_w": shape_dict["width"],
                        "bbox_h": shape_dict["height"]
                        })

    annotated_data[full_file_path]["regions"] = regions

# for k, v in annotated_data.items():
#     print(k, v)


all_data = []

for k, v in annotated_data.items():
    img_h = v['image_h']
    img_w = v['image_w']
    regions = v['regions']
    for item in regions:
        new_row = [k.split("evaS9\\")[1], img_h, img_w]
        new_row.append(item['class_name'])
        new_row.append(item['cx'])
        new_row.append(item['cy'])
        new_row.append(item['bbox_w'])
        new_row.append(item['bbox_h'])
        all_data.append(new_row)

[print(x) for x in all_data]
print(len(all_data))

with open("calculations.csv", 'w', encoding='utf-8') as f:
    writer = csv.writer(f, delimiter=',')
    columns = ['file', 'image_height', 'image_width', 'class_name', 'cx', 'cy', 'bbox_w', 'bbox_h']
    writer.writerow(columns)
    for row in all_data:
        writer.writerow(row)




