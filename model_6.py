import tensorflow as tf
import numpy as np
from keras.models import load_model
import mediapipe as mp
import cv2
import os
import asyncio
import threading
import copy
import time
import djitellopy
from queue import Queue


class events:
    got_image = threading.Event()

    is_running = threading.Event()

    got_lms = threading.Event()




class VideoCaptureThread(threading.Thread):
    def __init__(
            self, 
            video_source, 
            tello_drone = None, 
            height = 480, 
            width = 480, 
            command_thresholds = {'up':0, 'down':0, 'left':0, 'right':0, 'forward':0, 'backward':0, 'flip':0, 'land':0, 'NaN':0},

    ):
        super(VideoCaptureThread, self).__init__()
        self.video_source = video_source
        self.cap = cv2.VideoCapture(video_source)
        self.height = height
        self.width = width
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)

        self.frame = None                                   
        self.running = True
        self.mpDraw = mp.solutions.drawing_utils
        self.lm_obj = None # Stores landmark to plot the hands on frame
        self.lm_present = False # indicates presence of the landmark object
        self.mpHands = mp.solutions.hands # mp hands
        self.labels = 'NaN' # curent prediction label to print on the frame
        self.font = cv2.FONT_HERSHEY_COMPLEX_SMALL # font of the label
        self.landmark_queue = Queue(20)

        self.tello_drone = tello_drone
        self.command_handler = commandHandler(self, tello_drone=tello_drone, command_thresholds=command_thresholds) # command handler to sequentially get the commands
        self.command_handler.start() # start the command handler
        self.drone_label = 'IDLE'
        
        self.gesture_detection_thread = GestureDetectionThread(self,self.command_handler)
        self.gesture_detection_thread.start()

        self.artist = Artist(self, self.gesture_detection_thread)
        self.artist.start()

        self.start()

    def run(self):
        self.running, self.frame = self.cap.read()
        while self.running:
            events.got_image.set() 

            if self.lm_present == True:
                if self.landmark_queue.empty():
                    self.frame = cv2.flip(self.frame,1)
                    self.mpDraw.draw_landmarks(self.frame, self.lm_obj, self.mpHands.HAND_CONNECTIONS)
                    self.frame = cv2.flip(self.frame,1) 
                else:
                    self.lm_obj = self.landmark_queue.get()
                    self.frame = cv2.flip(self.frame,1)
                    self.mpDraw.draw_landmarks(self.frame, self.lm_obj, self.mpHands.HAND_CONNECTIONS)
                    self.frame = cv2.flip(self.frame,1) 
            else:
                self.labels = 'NaN'

            cv2.putText(self.frame,f'Prediction   : {self.labels}'     ,(10,self.height-20), self.font, 1,(255,255,255),1,cv2.LINE_AA)
            cv2.putText(self.frame,f'Drone Status : {self.drone_label}',(10,self.height-40), self.font, 1,(255,255,255),1,cv2.LINE_AA)
            cv2.imshow("Original Frame",self.frame)
                
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.stop()
                break
            self.running, self.frame = self.cap.read()
        self.stop()

    def stop(self):
        self.running = False
        events.is_running.set()
        self.cap.release()
        cv2.destroyAllWindows()

        self.artist.end()
        self.command_handler.end()
        self.gesture_detection_thread.end()

        self.artist.join()
        self.command_handler.join()
        self.gesture_detection_thread.join()


    def put_drone_label(self,label):
        self.drone_label = label
    def put_detection_label(self,label):
        self.labels = label

    def put_landmarks(self,lm_obj = None,NaN = False):
        if NaN == True:
            self.lm_present = False
        else:
            self.landmark_queue.put(copy.copy(lm_obj))
            self.lm_present = True
        



class GestureDetectionThread(threading.Thread):

    def __init__(self,video_capture,command_handler):
        super(GestureDetectionThread, self).__init__()
        self.frame = None
        self.class_names = np.array(['backward', 'down', 'flip', 'forward','land', 'left', 'right', 'up'],dtype=object)
        self.norm_layer = tf.keras.layers.Normalization(axis = 2)
        self.model = load_model('./models/Model_5_120e.h5')

        self.lm_obj = None
        self.lm_present = False 
        self.landmark_queue = Queue(10)
        self.landmark_list = np.zeros((1,21,3))
        self.video_capture = video_capture
        self.direction = 'NaN'
        self.land_thres = 2
        self.running = True

        self.NaN_threshold = 4
        self.cur_NaN_threshold = self.NaN_threshold

        self.command_handler = command_handler


    def run(self):
        while self.running:
            events.got_lms.wait() # until we get landmarks, we wait
            if not self.running:
                break
            self.hand_detection()


    
    def end(self):
        self.running = False
        events.got_lms.set()
        

    def hand_detection(self):
        self.landmark_list[0] = np.array([[landmark.x, landmark.y, landmark.z] for landmark in self.lm_obj.landmark], dtype = np.float64)

        self.norm_layer.adapt(self.landmark_list)
        normalized_landmark = tf.reshape( self.norm_layer(self.landmark_list),shape=(1,63))
        prediction = self.model.predict(normalized_landmark, verbose = 0)
        predicted_idx = np.argmax(prediction)

        self.direction = self.class_names[predicted_idx]
        self.video_capture.put_detection_label(self.direction)
        self.command_handler.send_commands(self.direction)

        events.got_lms.clear()
        return True

    
    def predict_labels(self,lm_obj):
        if lm_obj is None:
            self.command_handler.send_commands('NaN')
            if self.cur_NaN_threshold <= 0:
                self.video_capture.put_landmarks(NaN=True)
                self.video_capture.put_detection_label('NaN')
            else:
                self.cur_NaN_threshold -= 1
        else:
            self.cur_NaN_threshold = self.NaN_threshold # reset the NaN threshold
            self.lm_obj = copy.copy(lm_obj)
            events.got_lms.set() # it starts the hand detection thread




class Artist(threading.Thread):
    def __init__(self,video_capture,detector):
        super(Artist, self).__init__()
        self.video_capture = video_capture
        self.detector = detector
        self.frame = None
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
        self.mpDraw = mp.solutions.drawing_utils
        self.running = True

    def run(self):
        while True:
            events.got_image.wait()  # choke point
            if not self.running:
                break
            self.frame = self.video_capture.frame.copy()
            self.runnable()

    def runnable(self):
        self.frame = cv2.flip(self.frame,1)
        self.frame = cv2.cvtColor(self.frame,cv2.COLOR_BGR2RGB)

        result = self.hands.process(self.frame)

        if result.multi_hand_landmarks:
            for handslms in result.multi_hand_landmarks:
                self.detector.predict_labels(handslms)# Start the detector when you got the landmarks
                self.video_capture.put_landmarks(handslms)
        else:
            self.detector.predict_labels(None)

        events.got_image.clear()

    def end(self):
        self.running = False
        events.got_image.set()
        

class commandHandler(threading.Thread):
    def __init__(
            self, video_capture, 
            tello_drone = None, 
            command_thresholds = {'up':1, 'down':1, 'left':1, 'right':1, 'forward':1, 'backward':1, 'flip':0, 'land':4, 'NaN':8}
    ):
        super(commandHandler, self).__init__()
        self.video_capture = video_capture
        self.tello_drone = tello_drone
        self.commands = Queue(maxsize=10)
        self.last_command = 'None'
        self.next_command = 'None'
        self.flip = False
        self.tello_drone = tello_drone 
        self.flipped_recent = False
        self.running = True

        self.takeoff_event = threading.Event()
        self.landing_event = threading.Event()

        self.got_command = threading.Event()
        self.command_queue = Queue(maxsize = 5)

        self.command_thresholds = command_thresholds # amount of commands received to confirm the given command
        self.current_thresholds = {'up':0, 'down':0, 'left':0, 'right':0, 'forward':0, 'backward':0, 'flip':0, 'land':0, 'NaN':0}

        self.BUSY = False

    def send_commands(self,command):
        if self.BUSY:
            return
        else:
            self.BUSY = True
            if not self.takeoff_event.is_set():
                if command == 'forward' and not self.command_queue.full():
                    # will unbusy here
                    self.command_queue.put('takeoff')
                    self.last_command = 'takeoff'
                    self.takeoff_event.set()
                    self.got_command.set()
                    return
                else:
                    self.BUSY = False
                    return
                
            else: 

                if (command != self.last_command or self.flipped_recent) and not self.command_queue.full() :
                    # will unbusy here
                    if self.current_thresholds[command] >= self.command_thresholds[command]:
                        self.reset_thresholds()
                        self.next_command = 'None'
                        self.command_queue.put(command)
                        self.last_command = command
                        self.got_command.set()
                        self.flipped_recent = False
                        return
                    elif (self.next_command == command or command == 'NaN') and self.next_command != 'None':
                        self.current_thresholds[self.next_command] += 1
                        self.BUSY = False
                        return
                    else:
                        self.reset_thresholds()
                        self.next_command = command
                        self.BUSY = False
                        return
                else:
                    self.BUSY = False
                    return

    
    def end(self):
        self.running = False
        # self.tello_drone.land()
        self.got_command.set()
        self.takeoff_event.set()

    def run(self):
        while True:
            self.got_command.wait()
            self.takeoff_event.wait()  
            if self.running == False:
                break
            temp = self.command_queue.get()

            if temp == 'takeoff':
                self.video_capture.put_drone_label(temp)
                # self.tello_drone.takeoff()
                self.timeout_rountine('Taking off.....',4,'Initializing..... ','Ready')
                
        

            elif temp == 'land':
                self.takeoff_event.clear()
                self.flip = False
                self.video_capture.put_drone_label('Landing.....')
                # self.tello_drone.land()
                self.timeout_rountine('Landing.....',3,'Please Standby... ','IDLE')
                self.takeoff_event.clear()

            elif temp == 'NaN':
                self.video_capture.put_drone_label(temp)
                self.flip = False
                # self.tello_drone.send_rc_control(0,0,0,0)
            elif temp == 'flip':
                self.video_capture.put_drone_label('Command .....')
                self.flip = True
                # self.tello_drone.send_rc_control(0,0,0,0)
            elif self.flip:
                if temp == 'left':
                    self.video_capture.put_drone_label(f"{temp} 'flip'")
                    # self.tello_drone.flip_left()
                elif temp == 'right':
                    self.video_capture.put_drone_label(f"{temp} 'flip'")
                    # self.tello_drone.flip_right()
                elif temp == 'forward':
                    self.video_capture.put_drone_label(f"{temp} 'flip'")
                    # self.tello_drone.flip_forward()
                elif temp == 'backward':
                    self.video_capture.put_drone_label(f"{temp} 'flip'")
                    # self.tello_drone.flip_backward()
                else:
                    self.video_capture.put_drone_label('Invalid Command')

                self.flip = False
                self.flipped_recent = True


            elif not self.flip:
                if temp == 'up':
                    self.video_capture.put_drone_label(temp)
                    # self.tello_drone.send_rc_control(0,0,-20,0)
                    
                elif temp == 'down':
                    self.video_capture.put_drone_label(temp)
                    # self.tello_drone.send_rc_control(0,0,20,0)
                    
                elif temp == 'left':
                    self.video_capture.put_drone_label(temp)
                    # self.tello_drone.send_rc_control(-20,0,0,0)
                    
                elif temp == 'right':
                    self.video_capture.put_drone_label(temp)
                    # self.tello_drone.send_rc_control(20,0,0,0)
                    
                elif temp == 'forward':
                    self.video_capture.put_drone_label(temp)
                    # self.tello_drone.send_rc_control(0,-20,0,0)
                    
                elif temp == 'backward':
                    self.video_capture.put_drone_label(temp)
                    # self.tello_drone.send_rc_control(0,20,0,0)

            self.got_command.clear()
            self.BUSY = False

    def timeout_rountine(self, text, seconds, auxillary_text = '', endingtext = ''):
        self.video_capture.put_drone_label(text)
        time.sleep(0.5)
        for i in range(seconds,0,-1):
            self.video_capture.put_drone_label(f"{auxillary_text} {i}")
            time.sleep(1)
        if endingtext != '':
            self.video_capture.put_drone_label(f"{endingtext}")
            time.sleep(1)
    
    def reset_thresholds(self):
        for i in self.current_thresholds:
            self.current_thresholds[i] = 0
        

class main:
    def __init__(self):
        # print('Waiting fo tello')
        # self.tello = djitellopy.Tello()
        # self.tello.connect()
        # time.sleep(5)
        # print('Tello')
        self.tello = None

        self.video_capture = VideoCaptureThread(
            video_source = 0,
            tello_drone = self.tello,
            command_thresholds = {'up':1, 'down':1, 'left':1, 'right':1, 'forward':1, 'backward':1, 'flip':0, 'land':4, 'NaN':7},
            height= 480,
            width= 480,
        )

        events.is_running.wait()
        self.video_capture.join()
        print('Done')
            

if __name__ == "__main__":
    main()



            


    






