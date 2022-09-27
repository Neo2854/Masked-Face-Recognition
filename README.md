# Masked-Face-Recognition
Masked Face Recognition using MTCNN and FaceNet

Requires Python>=3.8
Required Packages
  * opencv-python
  * scikit-learn
  * torch
  * imutils
  * facenet-pytorch
  * matplotlib
  * numpy

The required packages can be installed using requirements.txt
```bash
pip install -r requirements.txt
```

# MTCNN
MTCNN paper <a href="https://arxiv.org/pdf/1604.02878.pdf">link</a>.

For detecting face and landmarks in MTCNN
```bash
python mtcnn.py <path_to_img>
```
Example outputs</br></br>
<p align="middle">
 <img src="./outputs/output2.png?raw=true" alt="output2.png">
 <img src="./outputs/output3.png?raw=true" alt="output2.png"></br>
 <img src="./outputs/live_out1.png?raw=true" alt="live_out1.png" width=280 height=340>
 <img src="./outputs/live_out2.png?raw=true" alt="live_out2.png" width=280 height=340>
</p>
</br>

Use the following command to align faces properly based on eye-landmarks
```bash
python mtcnn.py <path_to_img> --align
```
Example outputs</br></br>
<p align="middle">
 <img src="./outputs/aligned.png?raw=true" alt="output2.png">
</p>

# Facenet
Facenet paper <a href="https://arxiv.org/pdf/1503.03832.pdf">link</a>. It is better to go through the paper to get more understanding on architecture and the loss function is used.

<p>The Facenet model can be used with pretrained weights. For testing purposes this repository used <b>RMFRD</b> dataset which can be downloaded <a href="https://drive.google.com/drive/folders/1pBDhnJoz16rVEtW5uLv_3is98og2wyYM?usp=sharing">here</a>.</p>

<p align="middle">
  <img src="./inputs/facenet-arch.png?raw=true" alt="Facenet Architecture" title="Facenet">
</p>
</br>

<p>
 Import the face_recognition_model into your python file and create a <b>FaceRecognizer</b> object. <b>add_face</b> can be called to add some reference images of a person with a label. The predict function can then be called to recognize the face on new data. 
</p>
