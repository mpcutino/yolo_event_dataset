from os import path

import pandas as pd
import numpy as np
from constants import CLASS_BAG, BAG_NAME, IMG_FOLDER


def get_filename(file, char="/"):
    split = file.split(char)
    if len(split):
        return split[-1]
    return ""

def remove_extension(file):
    split = file.split(".")
    if len(split) > 1:
        return split[-2]
    return ""

def check_missing_annot(class_name, bag_name, data_folder="data_bBox", annot_file="annotate.txt"):
    file_name = path.join(path.join(data_folder, class_name), bag_name)
    file_name = path.join(file_name, IMG_FOLDER)
    file_name = path.join(file_name, annot_file)
    annotation_df = pd.read_csv(file_name, sep=',')

    for _, r in annotation_df.loc[annotation_df["x"]==-1, ["boxImg_name"]].iterrows():
        print(r)

def def_image(img_h, img_w):
    # return np.full((img_h, img_w, 3), (255, 255, 255))
    return np.zeros((img_h, img_w, 3))


if __name__ == "__main__":
    check_missing_annot(CLASS_BAG, BAG_NAME)
