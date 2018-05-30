from keras.optimizers import Adam
from keras import backend as K
from keras.models import load_model
from math import ceil
import numpy as np
from matplotlib import pyplot as plt
from imageio import imread
from models.keras_ssd7 import build_model
from keras_loss_function.keras_ssd_loss import SSDLoss
from keras.preprocessing import image
from keras_layers.keras_layer_AnchorBoxes import AnchorBoxes
from keras_layers.keras_layer_DecodeDetections import DecodeDetections
from keras_layers.keras_layer_DecodeDetections2 import DecodeDetections2

from ssd_encoder_decoder.ssd_input_encoder import SSDInputEncoder
from ssd_encoder_decoder.ssd_output_decoder import decode_detections, decode_detections_fast

from data_generator.object_detection_2d_data_generator import DataGenerator
from data_generator.object_detection_2d_misc_utils import apply_inverse_transforms
from data_generator.data_augmentation_chain_constant_input_size import DataAugmentationConstantInputSize
from data_generator.data_augmentation_chain_original_ssd import SSDDataAugmentation
import cv2
from models.keras_ssd300 import ssd_300
import random
import time
from PIL import Image



img_height = 300 # Height of the input images
img_width = 300 # Width of the input images
img_channels = 3 # Number of color channels of the input images
intensity_mean = 127.5 # Set this to your preference (maybe `None`). The current settings transform the input pixel values to the interval `[-1,1]`.
intensity_range = 127.5 # Set this to your preference (maybe `None`). The current settings transform the input pixel values to the interval `[-1,1]`.
n_classes = 3 # Number of positive classes
scales = [0.08, 0.16, 0.32, 0.64, 0.96] # An explicit list of anchor box scaling factors. If this is passed, it will override `min_scale` and `max_scale`.
aspect_ratios = [0.5, 1.0, 2.0] # The list of aspect ratios for the anchor boxes
two_boxes_for_ar1 = True # Whether or not you want to generate two anchor boxes for aspect ratio 1
steps = None # In case you'd like to set the step sizes for the anchor box grids manually; not recommended
offsets = None # In case you'd like to set the offsets for the anchor box grids manually; not recommended
clip_boxes = False # Whether or not to clip the ground truth and anchor boxes to lie entirely within the image boundaries
variances = [1.0, 1.0, 1.0, 1.0] # The list of variances by which the encoded target coordinates are scaled
normalize_coords = True # Whether or not the model is supposed to use coordinates relative to the image size

key = ''
# 1: Build the Keras model

K.clear_session() # Clear previous models from memory.

# model = build_model(image_size=(img_height, img_width, img_channels),
#                     n_classes=n_classes,
#                     mode='inference',
#                     l2_regularization=0.0005,
#                     scales=scales,
#                     aspect_ratios_global=aspect_ratios,
#                     aspect_ratios_per_layer=None,
#                     two_boxes_for_ar1=two_boxes_for_ar1,
#                     steps=steps,
#                     offsets=offsets,
#                     clip_boxes=clip_boxes,
#                     variances=variances,
#                     normalize_coords=normalize_coords,
#                     subtract_mean=intensity_mean,
#                     divide_by_stddev=intensity_range)

model = ssd_300(image_size=(img_height, img_width, 3),
                n_classes=20,
                mode='inference',
                l2_regularization=0.0005,
                scales=[0.1, 0.2, 0.37, 0.54, 0.71, 0.88, 1.05], # The scales for MS COCO are [0.07, 0.15, 0.33, 0.51, 0.69, 0.87, 1.05]
                aspect_ratios_per_layer=[[1.0, 2.0, 0.5],
                                         [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                                         [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                                         [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                                         [1.0, 2.0, 0.5],
                                         [1.0, 2.0, 0.5]],
                two_boxes_for_ar1=True,
                steps=[8, 16, 32, 64, 100, 300],
                offsets=[0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                clip_boxes=False,
                variances=[0.1, 0.1, 0.2, 0.2],
                normalize_coords=True,
                subtract_mean=[123, 117, 104],
                swap_channels=[2, 1, 0],
                confidence_thresh=0.5,
                iou_threshold=0.45,
                top_k=200,
                nms_max_output_size=400)
# 2: Optional: Load some weights

# model.load_weights('./models/ssd7_weights_epoch-13_loss-0.5756_val_loss-0.8491.h5', by_name=True)
model.load_weights('./models/VGG_VOC0712Plus_SSD_300x300_ft_iter_160000.h5', by_name=True)

# model.load_weights('./models/ssd300_dong', by_name=True)

# orig_images = [] # Store the images here.
 # Store resized versions of the images here.

# We'll only load one image in this example.
# img_path = './dataset/111_resize.jpg'

cap = cv2.VideoCapture('./t4.mp4')
    # vidcap = cv2.VideoCapture('myvid2.mp4')
success, image = cap.read()

while key != 113:
    t1 = time.time()
    input_images = []
    orig_images = [] 
    #.read() have two parameters!!
    success, image = cap.read()

    # imgpil = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
    (h, w) = image.shape[:2]
    center = (w/2, h/2)
    # 将图像旋转180度
    M = cv2.getRotationMatrix2D(center, 90, 1.0)
    image_rotate = cv2.warpAffine(image, M, (w, h))

    orig_images.append(image_rotate)
    # cv2.imshow('video',image)

    resize = cv2.resize(image_rotate, (300, 300))

    # resize = np.rot90(resize)

    input_images.append(resize)


    input_images = np.array(input_images)

    y_pred = model.predict(input_images)
    np.set_printoptions(precision=2, suppress=True, linewidth=180)
    # print(y_pred[0][0], "++++++++++++++++++++++++++++")


    confidence_threshold = 0.5

    y_pred_thresh = [y_pred[k][y_pred[k,:,1] > confidence_threshold] for k in range(y_pred.shape[0])]

    print('obj num: ',len(y_pred_thresh[0]))

    # 使用这个选项来控制输出的设置.

    # print("Predicted boxes:\n")
    # print('   class         conf          xmin           ymin           xmax              ymax')
    # print(y_pred_thresh[0])

    # Display the image and draw the predicted boxes onto it.


    # classes = ['background', 'cone', 'umbrellaman', 'car']
    classes = ['background','Aeroplane', 'Bicycle', 'Bird', 'Boat', 'Bottle',
                   'Bus', 'Car', 'Cat', 'Chair', 'Cow', 'Diningtable',
                   'Dog', 'Horse','Motorbike', 'Person', 'Pottedplant',
                   'Sheep', 'Sofa', 'Train', 'Tvmonitor']
    colors = dict()
    for box in y_pred_thresh[0]:
        # color = colors[int(box[0])]
        cls_id = int(box[0])
        if cls_id not in colors:
            colors[cls_id] = (random.random(), random.random(), random.random())
        xmin = int(box[2] * orig_images[0].shape[1] / img_width)
        ymin = int(box[3] * orig_images[0].shape[0] / img_height)
        xmax = int(box[4] * orig_images[0].shape[1] / img_width)
        ymax = int(box[5] * orig_images[0].shape[0] / img_height)
        # print(box[2])
        if xmin<0:
            xmin = 0
        if ymin<0:
            ymin = 0            
        print(orig_images[0].shape[1])
        print(img_width)
        
        print(xmin,ymin)
        print(xmax,ymax)
        text_top = (xmin, ymin - 20)
        text_bot = (xmin + 280, ymin + 5)
        text_pos = (xmin + 10, ymin)
        label = '{}: {:.2f}'.format(classes[int(box[0])], box[1])
        cv2.rectangle(orig_images[0], (xmin, ymin), (xmax, ymax), color=colors[cls_id],thickness=4)
        # cv2.rectangle(orig_images[0], text_top, text_bot, color=colors[cls_id], -1)
        cv2.putText(orig_images[0], label, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        
    cv2.imshow("success!", orig_images[0])
    # print("OK-----------------------")
    key  = cv2.waitKey(1)
    # count += 1
    t2 = time.time()
    print('fps:',1/(t2-t1))

cap.release()
cv2.destroyAllWindows()



