import torch
import cv2

#define, load and nake inference using the YOLO-NAS model
def detection(frame, yolo_nas):
    model_predictions  = yolo_nas.predict(frame, conf=0.5, fuse_model = False)[0].prediction

    bboxes = model_predictions.bboxes_xyxy # [Num Instances, 4] List of predicted bounding boxes for each object
    poses  = model_predictions.poses       # [Num Instances, Num Joints, 3] list of predicted joints for each detected object (x,y, confidence)
    scores = model_predictions.scores

    return bboxes, poses, scores