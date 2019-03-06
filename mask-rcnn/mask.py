# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 13:16:20 2018

@author: Divyanshu
"""

import numpy as np
import random
import time
import cv2
import os
labelsPath = os.path.sep.join(["E:\Computer_Vision_A_Z_Template_Folder\maskrcnn\mask-rcnn\mask-rcnn-coco\object_detection_classes_coco.txt"])
LABELS = open(labelsPath).read().strip().split("\n")
 
# load the set of colors that will be used when visualizing a given
# instance segmentation
colorsPath = os.path.sep.join(["E:\Computer_Vision_A_Z_Template_Folder\maskrcnn\mask-rcnn\mask-rcnn-coco\colors.txt"])
COLORS = open(colorsPath).read().strip().split("\n")
COLORS = [np.array(c.split(",")).astype("int") for c in COLORS]
COLORS = np.array(COLORS, dtype="uint8")
weightsPath = os.path.sep.join(["E:\Computer_Vision_A_Z_Template_Folder\maskrcnn\mask-rcnn\mask-rcnn-coco\frozen_inference_graph.pb"])
configPath = os.path.sep.join(["E:\Computer_Vision_A_Z_Template_Folder\maskrcnn\mask-rcnn\mask-rcnn-coco\maskrcnn.pbtxt"])
print("[INFO] loading Mask R-CNN from disk...")
net = cv2.dnn.readNetFromTensorflow(weightsPath, configPath)
video=cv2.VideoCapture(0)
while True:
            ret,image =video.read()
            
            (H, W) = image.shape[:2]
            blob = cv2.dnn.blobFromImage(image, swapRB=True, crop=False)
            net.setInput(blob)
            start = time.time()
            (boxes, masks) = net.forward(["detection_out_final", "detection_masks"])
            end = time.time()
            print("[INFO] Mask R-CNN took {:.6f} seconds".format(end - start))
            print("[INFO] boxes shape: {}".format(boxes.shape))
            print("[INFO] masks shape: {}".format(masks.shape))
            for i in range(0, boxes.shape[2]):
            	# extract the class ID of the detection along with the confidence
            	# (i.e., probability) associated with the prediction
            	classID = int(boxes[0, 0, i, 1])
            	confidence = boxes[0, 0, i, 2]
             
            	# filter out weak predictions by ensuring the detected probability
            	# is greater than the minimum probability
            	if confidence > 0.5:
                            clone=image.copy()
                            box = boxes[0, 0, i, 3:7] * np.array([W, H, W, H])
                            (startX, startY, endX, endY) = box.astype("int")
                            boxW = endX - startX
                            boxH = endY - startY
                            mask=masks[i,classID]
                            mask = cv2.resize(mask, (boxW, boxH),interpolation=cv2.INTER_NEAREST)
                            mask = (mask > 0.3)
                            roi = clone[startY:endY, startX:endX]
                            cv2.imshow("roi",roi)
                            roi=roi[mask]
                            color=random.choice(COLORS)
                            blended=((0.4*color)+(0.6*roi)).astype("uint8")
                            clone[startY:endY,startX:endX][mask]=blended
                            color=[int(c) for c in color]
                            cv2.rectangle(clone,(startX,startY),(endX,endY),color,2)
                            text = "{}: {:.4f}".format(LABELS[classID], confidence)
                            cv2.putText(clone, text, (startX, startY - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                            # show the output image
                            cv2.imshow("Output", clone)
                            if(cv2.waitKey(1)==ord('q')):
                                break
cv2.destroyAllWindows()
video.release()
                            




        