from os import path, mkdir
from ctypes import *
import math
import random
import cv2


def sample(probs):
    s = sum(probs)
    probs = [a/s for a in probs]
    r = random.uniform(0, 1)
    for i in range(len(probs)):
        r = r - probs[i]
        if r <= 0:
            return i
    return len(probs)-1

def c_array(ctype, values):
    arr = (ctype*len(values))()
    arr[:] = values
    return arr

class BOX(Structure):
    _fields_ = [("x", c_float),
                ("y", c_float),
                ("w", c_float),
                ("h", c_float)]

class DETECTION(Structure):
    _fields_ = [("bbox", BOX),
                ("classes", c_int),
                ("prob", POINTER(c_float)),
                ("mask", POINTER(c_float)),
                ("objectness", c_float),
                ("sort_class", c_int)]


class IMAGE(Structure):
    _fields_ = [("w", c_int),
                ("h", c_int),
                ("c", c_int),
                ("data", POINTER(c_float))]

class METADATA(Structure):
    _fields_ = [("classes", c_int),
                ("names", POINTER(c_char_p))]

    

lib = CDLL("/home/mpcutino/Documents/darknet/libdarknet.so", RTLD_GLOBAL)
#lib = CDLL("libdarknet.so", RTLD_GLOBAL)
lib.network_width.argtypes = [c_void_p]
lib.network_width.restype = c_int
lib.network_height.argtypes = [c_void_p]
lib.network_height.restype = c_int

predict = lib.network_predict
predict.argtypes = [c_void_p, POINTER(c_float)]
predict.restype = POINTER(c_float)

set_gpu = lib.cuda_set_device
set_gpu.argtypes = [c_int]

make_image = lib.make_image
make_image.argtypes = [c_int, c_int, c_int]
make_image.restype = IMAGE

get_network_boxes = lib.get_network_boxes
get_network_boxes.argtypes = [c_void_p, c_int, c_int, c_float, c_float, POINTER(c_int), c_int, POINTER(c_int)]
get_network_boxes.restype = POINTER(DETECTION)

make_network_boxes = lib.make_network_boxes
make_network_boxes.argtypes = [c_void_p]
make_network_boxes.restype = POINTER(DETECTION)

free_detections = lib.free_detections
free_detections.argtypes = [POINTER(DETECTION), c_int]

free_ptrs = lib.free_ptrs
free_ptrs.argtypes = [POINTER(c_void_p), c_int]

network_predict = lib.network_predict
network_predict.argtypes = [c_void_p, POINTER(c_float)]

reset_rnn = lib.reset_rnn
reset_rnn.argtypes = [c_void_p]

load_net = lib.load_network
load_net.argtypes = [c_char_p, c_char_p, c_int]
load_net.restype = c_void_p

do_nms_obj = lib.do_nms_obj
do_nms_obj.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

do_nms_sort = lib.do_nms_sort
do_nms_sort.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

free_image = lib.free_image
free_image.argtypes = [IMAGE]

letterbox_image = lib.letterbox_image
letterbox_image.argtypes = [IMAGE, c_int, c_int]
letterbox_image.restype = IMAGE

load_meta = lib.get_metadata
lib.get_metadata.argtypes = [c_char_p]
lib.get_metadata.restype = METADATA

load_image = lib.load_image_color
load_image.argtypes = [c_char_p, c_int, c_int]
load_image.restype = IMAGE

rgbgr_image = lib.rgbgr_image
rgbgr_image.argtypes = [IMAGE]

predict_image = lib.network_predict_image
predict_image.argtypes = [c_void_p, IMAGE]
predict_image.restype = POINTER(c_float)

def classify(net, meta, im):
    out = predict_image(net, im)
    res = []
    for i in range(meta.classes):
        res.append((meta.names[i], out[i]))
    res = sorted(res, key=lambda x: -x[1])
    return res

def detect(net, meta, image, thresh=.5, hier_thresh=.5, nms=.45):
    im = load_image(image, 0, 0)
    num = c_int(0)
    pnum = pointer(num)
    predict_image(net, im)
    dets = get_network_boxes(net, im.w, im.h, thresh, hier_thresh, None, 0, pnum)
    num = pnum[0]
    if (nms): do_nms_obj(dets, num, meta.classes, nms);

    res = []
    for j in range(num):
        for i in range(meta.classes):
            if dets[j].prob[i] > 0:
                b = dets[j].bbox
                res.append((meta.names[i], dets[j].prob[i], (b.x, b.y, b.w, b.h)))
    res = sorted(res, key=lambda x: -x[1])
    free_image(im)
    free_detections(dets, num)
    return res

def load_yolov3():
    yolo_config_path = "/home/mpcutino/yolo_darknet_ws/src/darknet_ros/darknet_ros/yolo_network_config"
    net_path = "cfg/yolov3.cfg"
    weights_path = "weights/yolov3.weights"

    return load_net(path.join(yolo_config_path, net_path), path.join(yolo_config_path, weights_path), 0)

def predict_image_class_and_BB(net, image_path):
    coco_data_path = "/home/mpcutino/yolo_darknet_ws/src/darknet_ros/darknet/cfg/coco.data"
    meta = load_meta(coco_data_path)
    return detect(net, meta, image_path)

def get_image_with_bb(dark_net, image_path):
    prediction = predict_image_class_and_BB(dark_net, image_path)
    
    img = cv2.imread(image_path)
    transform_pred = []
    for pred in prediction:
        class_name, prob, bbox = pred
        x_min, y_min, w, h = [int(b) for b in bbox]
        w //= 2
        h //= 2
        start = (x_min-w, y_min-h)
        end = (x_min+w, y_min+h)

        draw_bBox(img, start, end, class_name, prob)

        tr_data = class_name, prob, (start[0], start[1], int(bbox[2]), int(bbox[3]))
        transform_pred.append(tr_data)

    return transform_pred, img


def draw_bBox(img, start, end, class_name, prob=None, bbox_color=(0, 255, 0)):
    """
    draw bounding box in image.
    start is a tuple of (x, y) top left pixel
    end is a tuple of bottom right pixel
    """
    cv2.rectangle(img, start, end, bbox_color, 2)

    if class_name and class_name != "":
        text = "{0}: {1:.2f}%".format(class_name, prob) if prob else class_name
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_COMPLEX, 0.5, 2)[0]
        # print((start[0], start[1]-text_size[1]), (start[0]+text_size[0], start[1]))
        cv2.rectangle(img, (start[0], start[1]-text_size[1]), (start[0]+text_size[0], start[1]), bbox_color, cv2.FILLED)
        cv2.putText(img, text, start, cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,0,0), 1)


if __name__ == "__main__":
    image_path = "data/eagle.jpg"

    save_folder="data_bBox"
    img_name = image_path.split("/")[-1]

    if not path.exists(save_folder):
        mkdir(save_folder)
    
    prediction, img = get_image_with_bb(image_path)

    img_name = path.join(save_folder, img_name)
    cv2.imwrite(img_name, img)

    print prediction
