from sklearn.neighbors import KNeighborsClassifier
import mtcnn
import facenet
import torch

import numpy as np

class FaceRecognizer():
    def __init__(self, pretrained = False, num_neighbors=3) -> None:
        self.knn = KNeighborsClassifier(3)

        # Initializations for MTCNN and Facenet
        self.mtcnn = mtcnn.MTcnn()
        self.facenet = facenet.Facenet(pretrained)

        self.embedings = list()
        self.labels = list()

        self.map = dict()

        self.faces = 0

    def add_face(self, imgs, label):
        # Function to add a face class to the model.
        # The label can be preferrably a string
        # imgs can be tensor or numpy array
        for img in imgs:
            boxes, probs, landmarks = self.mtcnn.detect_with_eyes(img)

            img = img[boxes[0,0]:boxes[0,2], boxes[0,1]:boxes[0,3]]
            
            with torch.no_grad:
                self.embedings.append(self.facenet.forward(img))

        self.map[self.faces] = label

        for i in range(len(imgs)):
            self.labels.append(self.faces)

        self.faces += 1

        self.knn.fit(np.array(self.imgs))

    def predict(self, img):
        # Function to predict a new face
        # img must be a numpy array
        label = self.knn.predict(img)

        return self.map[label]