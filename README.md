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
 <img src="./outputs/output3.png?raw=true" alt="output2.png">
 <img src="./outputs/live_out1.png?raw=true" alt="live_out1.png">
 <img src="./outputs/live_out2.png?raw=true" alt="live_out2.png">
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
