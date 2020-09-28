import os

import pandas as pd
import genpy

from utils import get_filename, remove_extension
from constants import CLASS_BAG, BAG_NAME, EVENT_FOLDER, IMG_FOLDER


def anotate_events(bag, topic="/dvs/events", save_folder="data_bBox", class_bag="chair", start_indx=0, how_many=-1):
    
    save_folder = os.path.join(save_folder, class_bag)
    bag_name = get_filename(bag.filename, char="/")
    bag_name = remove_extension(bag_name)
    save_folder = os.path.join(save_folder, bag_name)
    img_annot_file = os.path.join(save_folder, IMG_FOLDER)
    img_annot_file = os.path.join(img_annot_file, "annotate.txt")

    save_folder = os.path.join(save_folder, EVENT_FOLDER)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    df = pd.read_csv(img_annot_file)
    if df.empty: 
        print("Empty file")
        return

    previous_data = df.iloc[0]
    count = 0

    file_data = os.path.join(save_folder, "annotate_events.txt")
    file_data_exist = os.path.exists(file_data)
    with open(file_data, 'a') as fd:
        if not file_data_exist:
            # estamos creando el archivo... escribir cabeceras
            fd.write("timestamp,x,y,p,prv_img_name,prv_class_name,prv_box_x,prv_box_y,\
prv_box_w,prv_box_h,prv_proba,prv_tagger,\
next_img_name,next_class_name,next_box_x,next_box_y,next_box_w,\
next_box_h,next_proba,next_tagger\n")

        events_count = -1
        for msg in bag.read_messages(topic):
            current_data = df.iloc[count]
            img_ts = genpy.Time.from_sec(current_data.timestamp*1e-9)

            for e in msg.message.events:
                x, y, p, ts = e.x, e.y, e.polarity, e.ts
                while img_ts < ts:
                    # esto con un if deberia funcionar porque no debe haber mas de una 
                    # imagen entre dos eventos
                    count += 1
                    if count >= df.shape[0]:
                        return
                    tmp = df.iloc[count]
                    previous_data = current_data
                    current_data = tmp
                    img_ts = genpy.Time.from_sec(current_data.timestamp*1e-9)
                if is_in_union_box(x, y, previous_data, current_data, class_bag):
                    # para solo procesar un numero determinado de eventos en cada corrida
                    events_count += 1
                    if events_count < start_indx: continue
                    if how_many > 0  and (start_indx + how_many <= events_count):
                        return
                    fd.write('{0},{1},{2},{3},{4},{5},{6},{7},{8},{9},{10},{11},{12},{13},{14},{15},{16},{17},{18},{19}\n'.format(
                        str(ts),
                        x, y, p,
                        previous_data.boxImg_name,
                        previous_data.class_name,
                        previous_data.x,
                        previous_data.y,
                        previous_data.w,
                        previous_data.h,
                        previous_data.proba,
                        previous_data.tagger,

                        current_data.boxImg_name,
                        current_data.class_name,
                        current_data.x,
                        current_data.y,
                        current_data.w,
                        current_data.h,
                        current_data.proba,
                        current_data.tagger
                    ))


def is_in_union_box(x, y, previous_data, current_data, class_name):
    return is_in_box(x, y, previous_data, class_name) or is_in_box(x, y, current_data, class_name)


def is_in_box(x, y, data, class_name):
    return data.class_name == class_name and data.x <= x <= (data.x + data.w) and data.y <= y <= (data.y + data.h)
