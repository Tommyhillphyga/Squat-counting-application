import cv2
import time
import math
import torch
import numpy as np
import super_gradients

from posedetection import detection


class PoseDection():
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.yolo_nas = super_gradients.training.models.get("yolo_nas_pose_m", pretrained_weights="coco_pose").to(self.device)

    def get_detections(self, frame):
        bboxes, poses, scores = detection(frame, self. yolo_nas)
        return bboxes, poses, scores

    def findAngle(self,image,kpts,p1,p2,p3, draw=True):
        overlay = image.copy()
        alpha = 0.4
        cord = [] 

        for kpt in kpts:
             #Get landmarks
            x1, y1 = kpt[p1][:2]
            x2, y2 = kpt[p2][:2]
            x3, y3 = kpt[p3][:2]
            # cord.append((x1, y1))
            # cord.append((x2,y2))
            # cord.append((x3, y3))
        # print(cord)
        # x1, y1 = cord[0]
        # x2, y2 = cord[1]
        # x3, y3 = cord[2]
        #calculate the angles
        angle = math.degrees(math.atan2(y3-y2, x3-x2) - math.atan2(y1-y2, x1-x2))

        if angle < 0:
            angle+=360

        #draw coordinates
        if draw:
            cv2.line(image, (int(x1), int(y1)), (int(x2), int(y2)), (255,255,255), 4)
            cv2.line(image, (int(x3), int(y3)), (int(x2), int(y2)), (255,255,255), 4)
            cv2.circle(image, (int(x1),int(y1)), 5, (255, 255, 255), cv2.FILLED)
            cv2.circle(image, (int(x1),int(y1)), 10, (235, 235, 235), 8)
            cv2.circle(image, (int(x2),int(y2)), 5, (255, 255, 255), cv2.FILLED)
            cv2.circle(image, (int(x2),int(y2)), 10, (235, 235, 235), 8)
            cv2.circle(image, (int(x3),int(y3)), 5, (255, 255, 255), cv2.FILLED)
            cv2.circle(image, (int(x3),int(y3)), 10, (235, 235, 235), 8)
            cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)
            # cv2.putText(image, str(int(angle)), (int(x2) - 50, int(y2) + 50),
            #                 cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)

        return int(angle)
        

def process_uploaded_file(blur, vid_name, enable_gpu, save_video, confidence, c1_text, stframe, use_webcam):
    prev_time = 0
    squat_count = 0
    direction = 0
    output = './inference/output/result.mp4' 
    detector = PoseDection()
    input_file = 0 if use_webcam else vid_name
    input_file = vid_name

    cap = cv2.VideoCapture(input_file)
    while cap.isOpened():
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output, fourcc, fps, (width, height))
      
        for _ in range(length):
            success, frame = cap.read()
            if not success:
                print('webcam failed')
                break
            else:
                bboxes, poses, scores = detector.get_detections(frame)
                angle = detector.findAngle(frame, poses, 11, 13, 15, draw=True)
                percentage = np.interp(angle, (90, 160), (100, 0))

                # check for the squat 
                if percentage == 100:
                    if direction == 0:
                        squat_count += 0.5
                        direction = 1
                if percentage == 0:
                    if direction == 1:
                        squat_count += 0.5
                        direction = 0

                curr_time = time.time()
                fps_ = 1/ (curr_time-prev_time)
                prev_time = curr_time
                stframe.image(frame, channels = 'BGR',use_column_width=True)
                count = int(squat_count)
                c1_text.write(f"<h1 style='text-align: left; color: red; font-size: 70px;'>{count}</h1>", unsafe_allow_html=True)
                if save_video:
                    writer.write(frame)                 
                
        
        cap.release()
        writer.release()