# Hello Guys,

In this project, I will use a pretrained model to control the dji-tello drone movement using hand gestures implemented with the help of djitellopy API to connect and control the drone movements.


# Frameworks
This project is build on the following frameworks:
1. Mediapipe
2. OpenCv
3. tensorflow (only to import the model and get prediction out of it)
4. NumPy (essential dependency)
5. djitellopy (API to control the dji tello drone)

# Working
The model present in /models/gesture_recognition.h5 is trained to identify following gestures.
|S.No.|Gesture|Label|
|:----|:-|:-|
|1.|Thumbs Up |up|
|2.|Thumbs Down| down|
|3.|Thumb left|left|
|4.|Thumb right|right|
|5.| Palm open|forward|
|6.|Palm open but, middle finger and ring finger are curled inside|backward|
|7.|Index finger pointing in upward direction|flip|
|8.| Middle finger pointing in upward direction|land|

Pretrained model is quite robust. It will correctly identify the hand gestures accurately even hand gestures are significantly deviated. For example, in thumbs up gesture, it will correctly, identify that gesture even if 
    
