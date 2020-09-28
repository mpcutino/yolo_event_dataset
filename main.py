import os

import rosbag
import pandas as pd

from manual.update import update_yolo_data, update_images_bBox
from constants import CLASS_BAG, BAG_NAME, EVENT_FOLDER, BAG_BASE_PATH
from events.anotation import anotate_events
from events.plot_data import plot_events_imgs_by_name, plot_events_images, plot_all_events_at_image_fr


def get_number_of_annotated_events(class_name, bag_name, data_folder="data_bBox"):
    folder = os.path.join(os.path.join(data_folder, class_name), bag_name)
    folder = os.path.join(folder, EVENT_FOLDER)

    annot_file = os.path.join(folder, "annotate_events.txt")
    if not os.path.exists(annot_file):
        return 0
    with open(annot_file) as fd:
        return len(fd.readlines()) - 1


def fix_csv_unnamed():
    file = 'data_bBox/chair/Record01/events/annotate_events.txt'
    df = pd.read_csv(file, index_col=0)
    df.to_csv(file, index=False)


if __name__ == "__main__":
    class_bag = CLASS_BAG
    bag_name = BAG_NAME
    path = os.path.join(BAG_BASE_PATH, "{0}/{1}.bag".format(class_bag, bag_name))
    bag = rosbag.Bag(path)

    start_indx = get_number_of_annotated_events(class_bag, bag_name)
    print(start_indx)
    ammount = 500000

    update_yolo_data(CLASS_BAG, BAG_NAME)
    # update_images_bBox(class_bag, bag)

    anotate_events(bag, class_bag=class_bag, start_indx=start_indx, how_many=ammount)
    # plot_events_images(class_bag, bag_name)
    # plot_events_imgs_by_name(class_bag, bag_name, "1594373652690415658.png")
    # 1594373551270458649.png
    # plot_all_events_at_image_fr(bag, class_bag=class_bag)
