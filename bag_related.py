import os
import rosbag
import cv2
from cv_bridge import CvBridge

from darknet import get_image_with_bb, load_yolov3
from utils import get_filename, remove_extension
from constants import CLASS_BAG, BAG_NAME, IMG_FOLDER, BAG_BASE_PATH


def save_images(bag, topic="/dvs/image_raw", save_folder="data_bBox", class_bag="chair", start_indx_img=0, how_many=-1):

    bridge = CvBridge()
    save_folder = os.path.join(save_folder, class_bag)
    bag_name = get_filename(bag.filename, char="/")
    bag_name = remove_extension(bag_name)

    save_folder = os.path.join(save_folder, bag_name)
    save_folder = os.path.join(save_folder, IMG_FOLDER)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    
    net = load_yolov3()

    file_data = os.path.join(save_folder, "annotate.txt")
    file_data_exist = os.path.exists(file_data)
    with open(file_data, 'a') as fd:
        if not file_data_exist:
            # estamos creando el archivo... escribir cabeceras
            fd.write("timestamp,boxImg_name,class_name,proba,x,y,w,h,tagger\n")
        count = -1
        for msg in bag.read_messages(topic):
            count += 1
            if count < start_indx_img: continue # esta imagen no es de interes (probablemente ya ha sido analizada)
            if how_many > 0 and (start_indx_img + how_many <= count):
                break # las imagenes de aqui en adelante no interesan, seguramente seran analizadas en otro momento

            cv_image = bridge.imgmsg_to_cv2(msg.message, "rgb8")
            # solo salvar de manera temporal para usar darknet
            tmp_name = "tmp.png"
            cv2.imwrite(tmp_name, cv_image)
            pred, img = get_image_with_bb(net, tmp_name)
            os.remove(tmp_name)

            # timestamp = msg.timestamp
            timestamp = msg.message.header.stamp

            filename = str(timestamp) + ".png"
            img_name = os.path.join(save_folder, filename)
            cv2.imwrite(img_name, img)

            for p in pred:
                class_name, prob, bbox = p
                x, y, w, h = bbox

                fd.write("{0},{1},{2},{3},{4},{5},{6},{7},{8}\n".format(
                    str(timestamp), 
                    filename,
                    class_name,
                    prob,
                    x, y, w, h, "yolo")
                )
            if not len(pred):
                # esta imagen no tiene predicciones
                # poner -1 y revisarla luego
                fd.write("{0},{1},{2},{3},{4},{5},{6},{7},{8}\n".format(
                    str(timestamp),
                    filename,
                    "-1",
                    -1,
                    -1, -1, -1, -1, "yolo")
                )


def get_start_index(class_bag, bag_name, target_folder, save_folder="data_bBox"):
    # en base 0, o sea, el primer indice es 0.
    start_indx = 0
    # listar la carpeta en la que se guardan los archivos y, si tiene elementos, inicializar el indice
    # en el length - 1 (por el archivo annotate.txt)
    data_dolder = os.path.join("data_bBox", class_bag+"/"+bag_name +"/"+target_folder)
    if os.path.exists(data_dolder):
        start_indx = max(len(os.listdir(data_dolder)) - 1, 0)
    return start_indx


if __name__ == "__main__":
    class_bag = CLASS_BAG
    bag_name = BAG_NAME
    path = os.path.join(BAG_BASE_PATH, "{0}/{1}.bag".format(class_bag, bag_name))
    start_indx = get_start_index(class_bag, bag_name, IMG_FOLDER)
    print(start_indx)
    ammount = -1
    #RECORDAR en la siguiente iteracion comenzar en start_index + ammount

    bag = rosbag.Bag(path)
    save_images(bag, class_bag=class_bag, start_indx_img=start_indx, how_many=ammount)
