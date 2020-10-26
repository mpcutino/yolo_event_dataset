import os

import pandas as pd
import numpy as np
import cv2
import genpy

from constants import EVENT_FOLDER, IMG_FOLDER, TRAIN_FOLDER
from darknet import draw_bBox
from utils import def_image, get_filename, remove_extension


def plot_events_imgs_by_name(class_name, bag_name, img_name, save_folder="data_bBox", img_w=346, img_h=260):
    """
    index is an integer considered the current image
    """
    save_folder = os.path.join(save_folder, class_name)
    save_folder = os.path.join(save_folder, bag_name)
    img_folder = os.path.join(save_folder, IMG_FOLDER)
    ev_folder = os.path.join(save_folder, EVENT_FOLDER)
    annot_file = os.path.join(ev_folder, "annotate_events.txt")

    df = pd.read_csv(annot_file)

    if df.shape[0] == 0: 
        print("No annotation file")
        return
    df = df.loc[df["img_name"] == img_name]
    if df.shape[0] == 0: 
        print("No data recorded for that image ({0})".format(img_name))
        return
    r = df.iloc[0]
    next_class_name, next_box_x, next_box_y, next_box_w, next_box_h = r.class_name, r.box_x, r.box_y, r.box_w, r.box_h

    img = def_image(img_h, img_w)
    # img_path = os.path.join(img_folder, img_name)
    # image = cv2.imread(img_path)
    for _, r in df.loc[df["img_name"] == img_name].iterrows():
        ev_x, ev_y, ev_p = r.x, r.y, r.p
        color = (0, 0, 255) if ev_p > 0 else (255, 0, 0)

        img[ev_y][ev_x] = color
    
    img = np.ascontiguousarray(img, dtype=np.uint8)
    
    start = (next_box_x, next_box_y)
    end = (next_box_x + next_box_w, next_box_y + next_box_h)
    draw_bBox(img, start, end, "")

    img_path = os.path.join(ev_folder, img_name)
    cv2.imwrite(img_path, img)


def plot_events_images(class_name, bag_name, data_folder="data_bBox", overwrite=False):
    folder = os.path.join(os.path.join(data_folder, class_name), bag_name)
    folder = os.path.join(folder, EVENT_FOLDER)

    annot_file = os.path.join(folder, "annotate_events.txt")

    df = pd.read_csv(annot_file)
    if df.shape[0] == 0: 
        print("No annotation file")
        return
    
    images = list(os.listdir(folder))
    count = 0
    for img_name in df.img_name.unique():
        if overwrite or img_name not in images:
            plot_events_imgs_by_name(class_name, bag_name, img_name)
            count += 1
    print(count)


def plot_all_events_at_image_fr(bag, topic="/dvs/events", save_folder="data_bBox",
                                class_bag="chair", img_w=346, img_h=260, img_annotate_file="annotate.txt"):
    
    save_folder = os.path.join(save_folder, class_bag)
    bag_name = get_filename(bag.filename, char="/")
    bag_name = remove_extension(bag_name)
    save_folder = os.path.join(save_folder, bag_name)
    img_annot_file = os.path.join(save_folder, IMG_FOLDER)
    img_annot_file = os.path.join(img_annot_file, img_annotate_file)
    train_folder = os.path.join(save_folder, TRAIN_FOLDER)

    if not os.path.exists(train_folder):
        os.makedirs(train_folder)

    df = pd.read_csv(img_annot_file)
    if df.shape[0] == 0:
        print("Class not present in file")
        return
    df.drop_duplicates(subset='boxImg_name', keep="first", inplace=True)

    previous_data = df.iloc[0]
    count = 0

    events_count = -1
    image = def_image(img_h, img_w)
    for msg in bag.read_messages(topic):
        current_data = df.iloc[count]
        img_ts = genpy.Time.from_sec(current_data.timestamp*1e-9)

        for e in msg.message.events:
            x, y, p, ts = e.x, e.y, e.polarity, e.ts
            while img_ts < ts:
                # we already have all events for consecutive images. Save and reset
                img_name = str(current_data.timestamp) + ".png"
                img_path = os.path.join(train_folder, img_name)
                cv2.imwrite(img_path, image)
                image = def_image(img_h, img_w)
                # esto con un if deberia funcionar porque no debe haber mas de una 
                # imagen entre dos eventos
                count += 1
                if count >= df.shape[0]:
                    return
                tmp = df.iloc[count]
                previous_data = current_data
                current_data = tmp
                img_ts = genpy.Time.from_sec(current_data.timestamp*1e-9)
            # color = (0, 0, 255) if p > 0 else (255, 0, 0)
            color = (255, 255, 255)
            image[y][x] = color
    # save last events
    img_name = str(current_data.timestamp)
    img_path = os.path.join(train_folder, img_name)
    cv2.imwrite(img_path, image)
    image = def_image(img_h, img_w)
