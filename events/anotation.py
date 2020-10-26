import os

import pandas as pd
import genpy

from utils import get_filename, remove_extension
from constants import CLASS_BAG, BAG_NAME, EVENT_FOLDER, IMG_FOLDER


def anotate_events(bag, topic="/dvs/events", save_folder="data_bBox", class_bag="chair",
                   start_indx=0, how_many=-1, img_annotate_file="annotate.txt"):
    
    save_folder = os.path.join(save_folder, class_bag)
    bag_name = get_filename(bag.filename, char="/")
    bag_name = remove_extension(bag_name)
    save_folder = os.path.join(save_folder, bag_name)
    img_annot_file = os.path.join(save_folder, IMG_FOLDER)
    img_annot_file = os.path.join(img_annot_file, img_annotate_file)

    save_folder = os.path.join(save_folder, EVENT_FOLDER)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    df = pd.read_csv(img_annot_file)
    if df.empty: 
        print("Empty file")
        return
    df = df.sort_values(by=['timestamp'])

    previous_data = df.loc[df["timestamp"] == df.iloc[0].timestamp]
    count = 0

    file_data = os.path.join(save_folder, "annotate_events.txt")
    file_data_exist = os.path.exists(file_data)
    with open(file_data, 'a') as fd:
        if not file_data_exist:
            # estamos creando el archivo... escribir cabeceras
            fd.write("timestamp, x, y, p, img_name, class_name, box_x, box_y, box_w, box_h, box_prob, box_tagger\n")

        events_count = -1
        for msg in bag.read_messages(topic):
            current_data = df.iloc[count]
            img_ts = genpy.Time.from_sec(current_data.timestamp*1e-9)
            # use all boxes with the same timestamp
            current_data = df.loc[df["timestamp"] == current_data.timestamp]

            for e in msg.message.events:
                x, y, p, ts = e.x, e.y, e.polarity, e.ts
                while img_ts < ts:
                    count += 1
                    if count >= df.shape[0]:
                        return
                    tmp = df.iloc[count]
                    img_ts = genpy.Time.from_sec(tmp.timestamp*1e-9)
                    tmp = df.loc[df["timestamp"] == tmp.timestamp]
                    previous_data = current_data
                    current_data = tmp
                series_data = is_in_union_box(x, y, previous_data, current_data, class_bag)
                if series_data is not None:
                    # para solo procesar un numero determinado de eventos en cada corrida
                    events_count += 1
                    if events_count < start_indx: continue
                    if how_many > 0 and (start_indx + how_many <= events_count):
                        return
                    fd.write('{0},{1},{2},{3},{4},{5},{6},{7},{8},{9},{10},{11}\n'.format(
                        str(ts),
                        x, y, p,
                        series_data.boxImg_name,
                        series_data.class_name,
                        series_data.x,
                        series_data.y,
                        series_data.w,
                        series_data.h,
                        series_data.proba,
                        series_data.tagger
                    ))


def is_in_union_box(x, y, previous_data, current_data, class_name):
    series_data = is_in_df(x, y, previous_data, class_name)
    if series_data is None:
        series_data = is_in_df(x, y, current_data, class_name)
    return series_data
    # return is_in_box(x, y, previous_data, class_name) or is_in_box(x, y, current_data, class_name)


def is_in_df(x, y, df, class_name):
    for _, data in df.iterrows():
        if is_in_box(x, y, data, class_name):
            return data
    return None


def is_in_box(x, y, data, class_name):
    return data.class_name == class_name and data.x <= x <= (data.x + data.w) and data.y <= y <= (data.y + data.h)
