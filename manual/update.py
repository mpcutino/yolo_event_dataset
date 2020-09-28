from os import path, remove
import pandas as pd
import cv2
from cv_bridge import CvBridge

from utils import get_filename, remove_extension
from darknet import draw_bBox
from constants import IMG_FOLDER


def update_yolo_data(class_name, bag_name, labels_file="manual/missing_labels.csv", data_folder="data_bBox", annot_file="annotate.txt"):
    manual_data_df = pd.read_csv(labels_file, header=None)
    manual_data_df.columns = ["class_name", "x", "y", "w", "h", "filename", "img_w", "img_h"]

    file_name = path.join(path.join(data_folder, class_name), bag_name)
    file_name = path.join(file_name, IMG_FOLDER)
    file_name = path.join(file_name, annot_file)
    annotation_df = pd.read_csv(file_name, sep=',')

    sustituir = ["class_name", "proba", "x", "y", "w", "h", "tagger"]
    for _, r in manual_data_df.iterrows():
        new_values = [r.class_name, 1, r.x, r.y, r.w, r.h, "manual"]
        annotation_df.loc[annotation_df['boxImg_name'] == r.filename, sustituir] = new_values

    annotation_df.to_csv(file_name, index=False)


def update_images_bBox(class_name, bag, topic="/dvs/image_raw", data_folder="data_bBox", annot_file="annotate.txt"):
    bag_name = get_filename(bag.filename, char="/")
    bag_name = remove_extension(bag_name)

    save_folder = path.join(path.join(data_folder, class_name), bag_name)
    save_folder = path.join(save_folder, IMG_FOLDER)
    file_name = path.join(save_folder, annot_file)
    df = pd.read_csv(file_name)

    bridge = CvBridge()

    for msg in bag.read_messages(topic):

        cv_image = bridge.imgmsg_to_cv2(msg.message, "rgb8")
        filename = str(msg.message.header.stamp) + ".png"

        cols_of_interest = ["x", "y", "w", "h", "proba", "class_name", "tagger"]
        for _, r in df.loc[df["boxImg_name"] == filename, cols_of_interest].iterrows():
            x, y, w, h, prob, cl, tagger = r
            bbox_color = (0, 255, 0) if tagger == "yolo" else (0, 0, 255)
            draw_bBox(cv_image, (x, y), (x+w, y+h), cl, prob, bbox_color=bbox_color)

        img_name = path.join(save_folder, filename)
        cv2.imwrite(img_name, cv_image)
