from ast import alias
import cv2
import argparse
import imutils

import numpy as np

from facenet_pytorch import MTCNN

class MTcnn:
    def __init__(self, dev = "cpu") -> None:
        self.model = MTCNN(keep_all=True, device=dev)

    def detect(self, img):
        boxes, probs, landmarks = self.model.detect(img, True)
        boxes = boxes.astype(np.int16)
        landmarks = landmarks.astype(np.int16)

        return boxes, probs, landmarks

    def detect_with_eyes(self, img):
        boxes, probs, landmarks = self.model.detect(img, True)
        boxes = boxes.astype(np.int16)
        landmarks = landmarks.astype(np.int16)

        return boxes, probs, landmarks[:, :2]

def rotate(img, eye_landmarks):
    x_0 = eye_landmarks[0,0]
    y_0 = eye_landmarks[0,1]

    x_1 = eye_landmarks[1,0]
    y_1 = eye_landmarks[1,1]
    
    angle = np.arccos(abs(x_1 - x_0)/np.linalg.norm(eye_landmarks[0] - eye_landmarks[1]))
    angle = np.rad2deg(angle)
    
    return imutils.rotate(img, -angle)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Face detection using MTCNN")
    parser.add_argument("image", action='store')
    parser.add_argument("--align", action='store_true', default=True)
    parser.add_argument("--save", action='store', default=None)

    args = parser.parse_args()
    img = cv2.imread(args.image)
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    model = MTcnn()
 
    boxes, probs, landmarks = model.detect_with_eyes(rgb_img)

    img = cv2.rectangle(img, (boxes[0,0]-5, boxes[0,1]-5), (boxes[0,2]+5, boxes[0,3]+5), color=(255, 0 ,0))

    for mark in landmarks[0]:
        img = cv2.circle(img, mark, 1, (255, 0 , 0), 1)
    
    if args.align:
        img = rotate(img, landmarks[0])

    if args.save:
        cv2.imwrite("./outputs/" + args.save, img)

    cv2.imshow("output", img)
    cv2.waitKey(0)