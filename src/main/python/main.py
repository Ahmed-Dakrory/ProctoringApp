# python -m fbs freeze --debug
# fbs freeze
# python 3.6.3
from fbs_runtime.application_context.PyQt5 import ApplicationContext
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5 import uic
import tempfile
from pydub import AudioSegment

import wave
import datetime
import sys
import cv2
import numpy as np
import pyautogui
import requests
import base64
import random
import json
from socketIO_client import SocketIO, BaseNamespace

import threading
import time

from winreg import *
import psutil
import math

import os
import win32api
import winput

################################
# Those for the Devices Checking
import pyaudio
import subprocess
import re
import wmi
import urllib
import ctypes
from ctypes import *
import socket
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

kernel32 = ctypes.WinDLL('kernel32')
user32 = ctypes.WinDLL('user32')



# from imutils import face_utils
from imutils.face_utils import FaceAligner

import tensorflow as tf

import dlib
from imutils import face_utils
from pynput import keyboard,mouse



#######################################
numberOfRunning = 0
for proc in psutil.process_iter():
    if proc.name() == "Proctoring.exe":
        numberOfRunning+=1
        if numberOfRunning >1:
            print("Killed")
            proc.kill()

#######################################################
#folder to store face images
CNN_INPUT_SIZE = 128
ANGLE_THRESHOLD = 0.15
IMAGE_PER_POSE=10
IMAGE_PER_PIC=50 # here number of images for id and hand
FACE_WIDTH = 160


ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001


# if __file__:
#     dir_path = os.path.dirname(__file__)
# elif getattr(sys, 'frozen', False):
#     dir_path = os.path.dirname(sys.executable)
# else:
#     dir_path =os.path.dirname(os.path.realpath(__file__))

if getattr(sys, 'frozen', False):
    # If the application is run as a bundle, the PyInstaller bootloader
    # extends the sys module by a flag frozen=True and sets the app 
    # path into variable _MEIPASS'.
    dir_path = sys._MEIPASS
else:
    dir_path = os.path.dirname(os.path.abspath(__file__))

print("----------------------------------------")
print("----------------------------------------")
print("----------------------------------------")
print(dir_path)
print("----------------------------------------")
print("----------------------------------------")
print("----------------------------------------")
    

#folder to store face images
CNN_INPUT_SIZE = 128
ANGLE_THRESHOLD = 0.15
IMAGE_PER_POSE=5
FACE_WIDTH = 160

# pyinstaller --onefile --add-data assets/deploy.prototxt;assets --add-data assets/model.txt;assets --add-data assets/res10_300x300_ssd_iter_140000.caffemodel;assets --add-data assets/pose_model/saved_model.pb;assets/pose_model --add-data assets/pose_model/variables/variables.data-00000-of-00001;assets/pose_model/variables --add-data assets/pose_model/variables/variables.index;assets/pose_model/variables --add-data shape_predictor_68_face_landmarks.dat;. Camera_Captureall.py

"""Human facial landmark detector based on Convolutional Neural Network."""
token = None
examId = None
class PoseEstimator:
    """Estimate head pose according to the facial landmarks"""

    def __init__(self, img_size=(480, 640)):
        self.size = img_size

        # 3D model points.
        self.model_points = np.array([
            (0.0, 0.0, 0.0),             # Nose tip
            (0.0, -330.0, -65.0),        # Chin
            (-225.0, 170.0, -135.0),     # Left eye left corner
            (225.0, 170.0, -135.0),      # Right eye right corner
            (-150.0, -150.0, -125.0),    # Mouth left corner
            (150.0, -150.0, -125.0)      # Mouth right corner
        ]) / 4.5

        self.model_points_68 = self._get_full_model_points()

        # Camera internals
        self.focal_length = self.size[1]
        self.camera_center = (self.size[1] / 2, self.size[0] / 2)
        self.camera_matrix = np.array(
            [[self.focal_length, 0, self.camera_center[0]],
             [0, self.focal_length, self.camera_center[1]],
             [0, 0, 1]], dtype="double")

        # Assuming no lens distortion
        self.dist_coeefs = np.zeros((4, 1))

        # Rotation vector and translation vector
        self.r_vec = np.array([[0.01891013], [0.08560084], [-3.14392813]])
        self.t_vec = np.array(
            [[-14.97821226], [-10.62040383], [-2053.03596872]])
        # self.r_vec = None
        # self.t_vec = None

    def _get_full_model_points(self, filename=dir_path+'/assets/model.txt'):
        """Get all 68 3D model points from file"""
        raw_value = []
        with open(filename) as file:
            for line in file:
                raw_value.append(line)
        model_points = np.array(raw_value, dtype=np.float32)
        model_points = np.reshape(model_points, (3, -1)).T

        # Transform the model into a front view.
        model_points[:, 2] *= -1

        return model_points

    def show_3d_model(self):
        from matplotlib import pyplot
        from mpl_toolkits.mplot3d import Axes3D
        fig = pyplot.figure()
        ax = Axes3D(fig)

        x = self.model_points_68[:, 0]
        y = self.model_points_68[:, 1]
        z = self.model_points_68[:, 2]

        ax.scatter(x, y, z)
        ax.axis('square')
        pyplot.xlabel('x')
        pyplot.ylabel('y')
        pyplot.show()

    def solve_pose(self, image_points):
        """
        Solve pose from image points
        Return (rotation_vector, translation_vector) as pose.
        """
        assert image_points.shape[0] == self.model_points_68.shape[0], "3D points and 2D points should be of same number."
        (_, rotation_vector, translation_vector) = cv2.solvePnP(
            self.model_points, image_points, self.camera_matrix, self.dist_coeefs)

        # (success, rotation_vector, translation_vector) = cv2.solvePnP(
        #     self.model_points,
        #     image_points,
        #     self.camera_matrix,
        #     self.dist_coeefs,
        #     rvec=self.r_vec,
        #     tvec=self.t_vec,
        #     useExtrinsicGuess=True)
        return (rotation_vector, translation_vector)

    def solve_pose_by_68_points(self, image_points):
        """
        Solve pose from all the 68 image points
        Return (rotation_vector, translation_vector) as pose.
        """

        if self.r_vec is None:
            (_, rotation_vector, translation_vector) = cv2.solvePnP(
                self.model_points_68, image_points, self.camera_matrix, self.dist_coeefs)
            self.r_vec = rotation_vector
            self.t_vec = translation_vector

        (_, rotation_vector, translation_vector) = cv2.solvePnP(
            self.model_points_68,
            image_points,
            self.camera_matrix,
            self.dist_coeefs,
            rvec=self.r_vec,
            tvec=self.t_vec,
            useExtrinsicGuess=True)

        return (rotation_vector, translation_vector)

    def draw_annotation_box(self, image, rotation_vector, translation_vector, color=(255, 255, 255), line_width=2):
        """Draw a 3D box as annotation of pose"""
        point_3d = []
        rear_size = 75
        rear_depth = 0
        point_3d.append((-rear_size, -rear_size, rear_depth))
        point_3d.append((-rear_size, rear_size, rear_depth))
        point_3d.append((rear_size, rear_size, rear_depth))
        point_3d.append((rear_size, -rear_size, rear_depth))
        point_3d.append((-rear_size, -rear_size, rear_depth))

        front_size = 100
        front_depth = 100
        point_3d.append((-front_size, -front_size, front_depth))
        point_3d.append((-front_size, front_size, front_depth))
        point_3d.append((front_size, front_size, front_depth))
        point_3d.append((front_size, -front_size, front_depth))
        point_3d.append((-front_size, -front_size, front_depth))
        point_3d = np.array(point_3d, dtype=np.float).reshape(-1, 3)

        # Map to 2d image points
        (point_2d, _) = cv2.projectPoints(point_3d,
                                          rotation_vector,
                                          translation_vector,
                                          self.camera_matrix,
                                          self.dist_coeefs)
        point_2d = np.int32(point_2d.reshape(-1, 2))

        # Draw all the lines
        cv2.polylines(image, [point_2d], True, color, line_width, cv2.LINE_AA)
        cv2.line(image, tuple(point_2d[1]), tuple(
            point_2d[6]), color, line_width, cv2.LINE_AA)
        cv2.line(image, tuple(point_2d[2]), tuple(
            point_2d[7]), color, line_width, cv2.LINE_AA)
        cv2.line(image, tuple(point_2d[3]), tuple(
            point_2d[8]), color, line_width, cv2.LINE_AA)

    def draw_axis(self, img, R, t):
        points = np.float32(
            [[30, 0, 0], [0, 30, 0], [0, 0, 30], [0, 0, 0]]).reshape(-1, 3)

        axisPoints, _ = cv2.projectPoints(
            points, R, t, self.camera_matrix, self.dist_coeefs)

        img = cv2.line(img, tuple(axisPoints[3].ravel()), tuple(
            axisPoints[0].ravel()), (255, 0, 0), 3)
        img = cv2.line(img, tuple(axisPoints[3].ravel()), tuple(
            axisPoints[1].ravel()), (0, 255, 0), 3)
        img = cv2.line(img, tuple(axisPoints[3].ravel()), tuple(
            axisPoints[2].ravel()), (0, 0, 255), 3)

    def draw_axes(self, img, R, t):
        img	= cv2.drawFrameAxes(img, self.camera_matrix, self.dist_coeefs, R, t, 30)


    def get_pose_marks(self, marks):
        """Get marks ready for pose estimation from 68 marks"""
        pose_marks = []
        pose_marks.append(marks[30])    # Nose tip
        pose_marks.append(marks[8])     # Chin
        pose_marks.append(marks[36])    # Left eye left corner
        pose_marks.append(marks[45])    # Right eye right corner
        pose_marks.append(marks[48])    # Mouth left corner
        pose_marks.append(marks[54])    # Mouth right corner
        return pose_marks
    
class Stabilizer:
    """Using Kalman filter as a point stabilizer."""

    def __init__(self,
                 state_num=4,
                 measure_num=2,
                 cov_process=0.0001,
                 cov_measure=0.1):
        """Initialization"""
        # Currently we only support scalar and point, so check user input first.
        assert state_num == 4 or state_num == 2, "Only scalar and point supported, Check state_num please."

        # Store the parameters.
        self.state_num = state_num
        self.measure_num = measure_num

        # The filter itself.
        self.filter = cv2.KalmanFilter(state_num, measure_num, 0)

        # Store the state.
        self.state = np.zeros((state_num, 1), dtype=np.float32)

        # Store the measurement result.
        self.measurement = np.array((measure_num, 1), np.float32)

        # Store the prediction.
        self.prediction = np.zeros((state_num, 1), np.float32)

        # Kalman parameters setup for scalar.
        if self.measure_num == 1:
            self.filter.transitionMatrix = np.array([[1, 1],
                                                     [0, 1]], np.float32)

            self.filter.measurementMatrix = np.array([[1, 1]], np.float32)

            self.filter.processNoiseCov = np.array([[1, 0],
                                                    [0, 1]], np.float32) * cov_process

            self.filter.measurementNoiseCov = np.array(
                [[1]], np.float32) * cov_measure

        # Kalman parameters setup for point.
        if self.measure_num == 2:
            self.filter.transitionMatrix = np.array([[1, 0, 1, 0],
                                                     [0, 1, 0, 1],
                                                     [0, 0, 1, 0],
                                                     [0, 0, 0, 1]], np.float32)

            self.filter.measurementMatrix = np.array([[1, 0, 0, 0],
                                                      [0, 1, 0, 0]], np.float32)

            self.filter.processNoiseCov = np.array([[1, 0, 0, 0],
                                                    [0, 1, 0, 0],
                                                    [0, 0, 1, 0],
                                                    [0, 0, 0, 1]], np.float32) * cov_process

            self.filter.measurementNoiseCov = np.array([[1, 0],
                                                        [0, 1]], np.float32) * cov_measure

    def update(self, measurement):
        """Update the filter"""
        # Make kalman prediction
        self.prediction = self.filter.predict()

        # Get new measurement
        if self.measure_num == 1:
            self.measurement = np.array([[np.float32(measurement[0])]])
        else:
            self.measurement = np.array([[np.float32(measurement[0])],
                                         [np.float32(measurement[1])]])

        # Correct according to mesurement
        self.filter.correct(self.measurement)

        # Update state value.
        self.state = self.filter.statePost

    def set_q_r(self, cov_process=0.1, cov_measure=0.001):
        """Set new value for processNoiseCov and measurementNoiseCov."""
        if self.measure_num == 1:
            self.filter.processNoiseCov = np.array([[1, 0],
                                                    [0, 1]], np.float32) * cov_process
            self.filter.measurementNoiseCov = np.array(
                [[1]], np.float32) * cov_measure
        else:
            self.filter.processNoiseCov = np.array([[1, 0, 0, 0],
                                                    [0, 1, 0, 0],
                                                    [0, 0, 1, 0],
                                                    [0, 0, 0, 1]], np.float32) * cov_process
            self.filter.measurementNoiseCov = np.array([[1, 0],
                                                        [0, 1]], np.float32) * cov_measure

class FaceDetector:
    """Detect human face from image"""

    def __init__(self,
                 dnn_proto_text=dir_path+'/assets/deploy.prototxt',
                 dnn_model=dir_path+'/assets/res10_300x300_ssd_iter_140000.caffemodel'):
        """Initialization"""
        self.face_net = cv2.dnn.readNetFromCaffe(dnn_proto_text, dnn_model)
        self.detection_result = None

    def get_faceboxes(self, image, threshold=0.5):
        """
        Get the bounding box of faces in image using dnn.
        """
        rows, cols, _ = image.shape

        confidences = []
        faceboxes = []

        self.face_net.setInput(cv2.dnn.blobFromImage(
            image, 1.0, (300, 300), (104.0, 177.0, 123.0), False, False))
        detections = self.face_net.forward()

        for result in detections[0, 0, :, :]:
            confidence = result[2]
            if confidence > threshold:
                x_left_bottom = int(result[3] * cols)
                y_left_bottom = int(result[4] * rows)
                x_right_top = int(result[5] * cols)
                y_right_top = int(result[6] * rows)
                confidences.append(confidence)
                faceboxes.append(
                    [x_left_bottom, y_left_bottom, x_right_top, y_right_top])

        self.detection_result = [faceboxes, confidences]

        return confidences, faceboxes

    def draw_all_result(self, image):
        """Draw the detection result on image"""
        for facebox, conf in self.detection_result:
            cv2.rectangle(image, (facebox[0], facebox[1]),
                          (facebox[2], facebox[3]), (0, 255, 0))
            label = "face: %.4f" % conf
            label_size, base_line = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

            cv2.rectangle(image, (facebox[0], facebox[1] - label_size[1]),
                          (facebox[0] + label_size[0],
                           facebox[1] + base_line),
                          (0, 255, 0), cv2.FILLED)
            cv2.putText(image, label, (facebox[0], facebox[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

class MarkDetector:
    """Facial landmark detector by Convolutional Neural Network"""

    def __init__(self, saved_model=dir_path+'/assets/pose_model'):
        """Initialization"""
        # A face detector is required for mark detection.
        self.face_detector = FaceDetector()

        self.cnn_input_size = 128
        self.marks = None

        # Get a TensorFlow session ready to do landmark detection
        # Load a Tensorflow saved model into memory.
        self.graph = tf.Graph()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(graph=self.graph, config=config)

        # Restore model from the saved_model file, that is exported by
        # TensorFlow estimator.
        tf.saved_model.loader.load(self.sess, ["serve"], saved_model)

    @staticmethod
    def draw_box(image, boxes, box_color=(255, 255, 255)):
        """Draw square boxes on image"""
        for box in boxes:
            cv2.rectangle(image,
                          (box[0], box[1]),
                          (box[2], box[3]), box_color, 3)

    @staticmethod
    def move_box(box, offset):
        """Move the box to direction specified by vector offset"""
        left_x = box[0] + offset[0]
        top_y = box[1] + offset[1]
        right_x = box[2] + offset[0]
        bottom_y = box[3] + offset[1]
        return [left_x, top_y, right_x, bottom_y]

    @staticmethod
    def get_square_box(box):
        """Get a square box out of the given box, by expanding it."""
        left_x = box[0]
        top_y = box[1]
        right_x = box[2]
        bottom_y = box[3]

        box_width = right_x - left_x
        box_height = bottom_y - top_y

        # Check if box is already a square. If not, make it a square.
        diff = box_height - box_width
        delta = int(abs(diff) / 2)

        if diff == 0:                   # Already a square.
            return box
        elif diff > 0:                  # Height > width, a slim box.
            left_x -= delta
            right_x += delta
            if diff % 2 == 1:
                right_x += 1
        else:                           # Width > height, a short box.
            top_y -= delta
            bottom_y += delta
            if diff % 2 == 1:
                bottom_y += 1

        # Make sure box is always square.
        assert ((right_x - left_x) == (bottom_y - top_y)), 'Box is not square.'

        return [left_x, top_y, right_x, bottom_y]

    @staticmethod
    def box_in_image(box, image):
        """Check if the box is in image"""
        rows = image.shape[0]
        cols = image.shape[1]
        return box[0] >= 0 and box[1] >= 0 and box[2] <= cols and box[3] <= rows

    def extract_cnn_facebox(self, image):
        """Extract face area from image."""
        _, raw_boxes = self.face_detector.get_faceboxes(
            image=image, threshold=0.9)

        for box in raw_boxes:
            # Move box down.
            # diff_height_width = (box[3] - box[1]) - (box[2] - box[0])
            offset_y = int(abs((box[3] - box[1]) * 0.1))
            box_moved = self.move_box(box, [0, offset_y])

            # Make box square.
            facebox = self.get_square_box(box_moved)

            if self.box_in_image(facebox, image):
                return facebox

        return None

    def detect_marks(self, image_np):
        """Detect marks from image"""
        # Get result tensor by its name.
        logits_tensor = self.graph.get_tensor_by_name(
            'layer6/final_dense:0')

        # Actual detection.
        predictions = self.sess.run(
            logits_tensor,
            feed_dict={'image_tensor:0': image_np})

        # Convert predictions to landmarks.
        marks = np.array(predictions).flatten()[:136]
        marks = np.reshape(marks, (-1, 2))

        return marks

    @staticmethod
    def draw_marks(image, marks, color=(255, 255, 255)):
        """Draw mark points on image"""
        for mark in marks:
            cv2.circle(image, (int(mark[0]), int(
                mark[1])), 1, color, -1, cv2.LINE_AA)


# keyboard.add_hotkey("alt + f4", lambda: None, suppress =True)
# keyboard.add_hotkey("shift + f10", lambda: None, suppress =True)
# keyboard.add_hotkey("ctrl + c", lambda: None, suppress =True)
# keyboard.add_hotkey("PrtScn", lambda: None, suppress =True)
# keyboard.add_hotkey("alt + PrtScn", lambda: None, suppress =True)
# keyboard.add_hotkey("ctrl + PrtScn", lambda: None, suppress =True)
# keyboard.add_hotkey("ctrl + p", lambda: None, suppress =True)




class ChatNamespace(BaseNamespace):

    def on_connect(self):
        print('connect')

    def on_disconnect(self):
        print('disconnect')

    def on_reconnect():
        print('reconnect')


class GUI(QMainWindow):
    def __init__(self,appctxt):

        

        self.dir_path = dir_path
            # self.dir_path =os.path.dirname(os.path.realpath(__file__))
        # self.dir_path = sys.argv[1:][0] #os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        super(GUI,self).__init__()
        
        ctypes.windll.kernel32.SetThreadExecutionState(
                ES_CONTINUOUS | \
                ES_SYSTEM_REQUIRED)

        uic.loadUi(self.dir_path+'\main.ui', self) # Load the .ui file
        self.setWindowFlags(Qt.WindowStaysOnTopHint | Qt.FramelessWindowHint)
        self.GUIPanel = self
        self.stepNow = 0
        self.appctxt = appctxt
        # User Session Token
        self.token = token
        self.IdFromUploadedImages=None
        self.Username = ''
        # self.IsVerified = None
        self.examId = examId
        self.AllImagesFaces = []
        self.AllImagesHand = []
        self.AllImagesId = []

        self.oldPos = self.pos()

        self.TestName = ''
        self.TestDuration = ''
        self.AllNotAllowed = None
        self.isOnBoard = False
        self.studentId = None

        self.WindowCameraOpened = False


        # Mouse Listener to prevent Right Click
        



        # Adapte Views
        # set Internet gid
        movie = QMovie(self.dir_path+"/imageface/internet.gif")
        self.internetChecking.setMovie(movie)
        movie.start()

        # set Success Devices
        self.bar_cam.setVisible(False)
        self.bar_mouse.setVisible(False)
        self.bar_key.setVisible(False)
        self.bar_speaker.setVisible(False)
        self.bar_micro.setVisible(False)
        self.bar_system.setVisible(False)
        self.bar_hard.setVisible(False)
        self.bar_screen.setVisible(False)

        self.success_cam.setVisible(False)
        self.success_mouse.setVisible(False)
        self.success_key.setVisible(False)
        self.success_speaker.setVisible(False)
        self.success_micro.setVisible(False)
        self.success_hard.setVisible(False)
        self.success_screen.setVisible(False)

        self.predictButton.clicked.connect(self.goNextStep)
        self.exitButton.clicked.connect(self.close)
        self.repeatButton.clicked.connect(self.reloadHandOrIdCap)
        # self.minimizeButton.clicked.connect(lambda: self.ReduceWindowAndMove())
        self.minimizeButton.clicked.connect(lambda: self.showMinimized())
        
        self.username.setText(self.Username)
        # print(self.Username)
        self.testname_label.setText(self.TestName)
        self.testduration_label.setText(str(self.TestDuration) )



        self.thControllersStop = ThreadMouse(self)
        self.thControllersStop.start()
        


        self.show()
        self.goNextStep(True)
        self.exit_code = self.appctxt.app.exec_()      # 2. Invoke appctxt.app.exec_()

   
    def waitmainSocket(self):
        self.socketIO.wait()

    def closeAllFromSocket(self,data):
        print("Data")
        print(data)
        self.EndTheExam()

    def on_click(self,x, y, button, pressed):
        print('{0} at {1}'.format(
            'Pressed' if pressed else 'Released',
            (x, y)))
        if not pressed:
            # Stop listener
            return False


    def ReduceWindowAndMove(self):
        
        headers = {'authorization': "Bearer "+str(self.token)}
        dataNew = {"status": True}
        UrlPostData = 'http://new.tproctor.teqnia-tech.com/api/proctor-app/allow-test-student/'+self.examId
        response = requests.put(UrlPostData,json=dataNew,headers=headers,timeout=1)
        self.message = response.json()['message']
        print(self.message)
        print(response.text)
        if self.message == 'Done':
            self.OpenCameraApp()
            
        else:
            print('Error')

    def OpenLoaderUpload(self):

        
        print('------------------oldkmlkmlk--------------')
        self.thUploadFileCamera = ThreadUploadFileCamera(self,self.th.PathOfFileUploaded,self.IdFromUploadedImages,self.token,self.th)
        self.thUploadFileCamera.changePercentage.connect(self.progressBarValue)
        self.thUploadFileCamera.changeLabel.connect(self.setVideoUploadingLabelProgress)
        self.thUploadFileCamera.startMain.connect(self.openLoader)
        self.thUploadFileCamera.uploadScreen.connect(self.uploadScreenRun)
        self.thUploadFileCamera.start()
        

    def centerWidgetOnScreen(self, widget):
        centerPoint = QScreen.availableGeometry(QApplication.primaryScreen()).center()
        fg = widget.frameGeometry()
        fg.moveCenter(centerPoint)
        widget.move(fg.topLeft())

    @pyqtSlot()
    def afterUploadingStep(self):
        self.LoaderUpload.predictButton.setEnabled(True)
        self.LoaderUpload.predictButton.setStyleSheet("""QPushButton{background-color: #0095ff;
            border-style: outset;
            border-width: 1px;
            border-radius: 5px;
            border-color: #e8e8e8;
            padding: 4px;
            color: #fbfbfb;
            font-size: 15px;
            font-weight: 700;}
            
            QPushButton:hover{background-color: #0095ff;
            border-style: outset;
            border-width: 1px;
            border-radius: 5px;
            border-color: #e8e8e8;
            padding: 4px;
            color: #565050;
            font-size: 15px;
            font-weight: 700;}""")


    @pyqtSlot()
    def uploadScreenRun(self):
        self.thUploadFileScreen = ThreadUploadFileScreen(self,self.th.PathNameOfFileScreen,self.IdFromUploadedImages,self.token)
        self.thUploadFileScreen.changePercentage.connect(self.progressBarValue)
        self.thUploadFileScreen.changeLabel.connect(self.setVideoUploadingLabelProgress)
        self.thUploadFileScreen.finishUploading.connect(self.afterUploadingStep)
        self.thUploadFileScreen.start()

    @pyqtSlot(str)
    def openLoader(self,text):
        print('------------------old--------------')
        self.hide()
        sizeObject = QDesktopWidget().screenGeometry(-1)
        self.LoaderUpload = self
        uic.loadUi(self.dir_path+'\progressUploading.ui', self.LoaderUpload) # Load the .ui file
        
        print('------------------doit--------------')
        self.LoaderUpload.setAttribute(Qt.WA_TranslucentBackground)
        self.LoaderUpload.setWindowFlags(Qt.FramelessWindowHint )
        
        self.LoaderUpload.setFixedWidth(1047)

        self.LoaderUpload.show()
        
        print('------------------new--------------')
        
       
        self.centerWidgetOnScreen(self.LoaderUpload)
        
        print('------------------new--------------')
        self.LoaderUpload.predictButton.clicked.connect(self.close)
        self.LoaderUpload.username.setText('hi, '+self.Username)
        self.LoaderUpload.testname_label.setText(self.TestName)
        print('------------------new33--------------')
        self.LoaderUpload.testduration_label.setText((self.TestDuration) )
        self.LoaderUpload.predictButton.setStyleSheet("""QPushButton{background-color: rgb(190, 188, 188);
        border-style: outset;
        border-width: 1px;
        border-radius: 5px;
        border-color: #e8e8e8;
        padding: 4px;
        color: #fbfbfb;
        font-size: 15px;
        font-weight: 700;}""")
        self.LoaderUpload.predictButton.setEnabled(False)


        print('------------------new-555-------------')
        
        
        #########################################################
        
    @pyqtSlot(str)
    def setVideoUploadingLabelProgress(self,text):
        self.labelUploading.setText(text)


    @pyqtSlot(int)
    def progressBarValue(self, value):
        # HTML TEXT PERCENTAGE
        htmlText = """<p><span style=" font-size:59pt;">{VALUE}</span><span style=" font-size:58pt; vertical-align:super;">%</span></p>"""

        # REPLACE VALUE
        newHtml = htmlText.replace("{VALUE}", str(value))

        self.LoaderUpload.labelPercentage.setText(newHtml)
        # PROGRESSBAR STYLESHEET BASE
        styleSheet = """
        QFrame{
        	border-radius: 150px;
        	background-color: qconicalgradient(cx:0.5, cy:0.5, angle:90, stop:{STOP_1} rgba(255, 0, 127, 0), stop:{STOP_2} rgba(85, 170, 255, 255));
        }
        """

        # GET PROGRESS BAR VALUE, CONVERT TO FLOAT AND INVERT VALUES
        # stop works of 1.000 to 0.000
        progress = (100 - value) / 100.0

        # GET NEW VALUES
        stop_1 = str(progress - 0.001)
        stop_2 = str(progress)

        # SET VALUES TO NEW STYLESHEET
        newStylesheet = styleSheet.replace("{STOP_1}", stop_1).replace("{STOP_2}", stop_2)

        # APPLY STYLESHEET WITH NEW VALUES
        self.LoaderUpload.circularProgress.setStyleSheet(newStylesheet)


    def OpenCameraApp(self):
        self.thCloseAppsThreadRunning=True
        self.thCloseApps = threading.Thread(target=self.while_closeAllBlackList, args=())
        self.thCloseApps.start()

        self.hide()
        sizeObject = QDesktopWidget().screenGeometry(-1)
        self.ex = self
        uic.loadUi(self.dir_path+'\mainCamera.ui', self.ex) # Load the .ui file

        self.ex.setAttribute(Qt.WA_TranslucentBackground)
        self.ex.setWindowFlags(Qt.WindowStaysOnTopHint | Qt.FramelessWindowHint)
        self.IsMinimized = False
        self.ex.minimizeButton.clicked.connect(lambda: self.MinimizeVideo())
        self.ex.EndExamButton.clicked.connect(lambda: self.EndTheExam())
        self.oldPos = self.ex.pos()
        self.ex.setFixedWidth(249)
        self.WindowCameraOpened = True
        self.ex.show()
        

        self.ex.move(sizeObject.width()-250,sizeObject.height()-300)
        self.th = ThreadCameraVideo(self.ex)
        self.th.changePixmap.connect(self.setImageVideo)
        self.th.changeStrLight.connect(self.setLightHint)
        self.th.changeStrTime.connect(self.setTimeForSession)
        self.th.start()
        self.dateTimeStartExam = datetime.datetime.now()
        minutes = int(self.TestDurationInt)
        minutes_added = datetime.timedelta(minutes = minutes)
        self.dateTimeEndExam = self.dateTimeStartExam + minutes_added
    


    def strfdelta(self,tdelta, fmt):
        d = {"days": tdelta.days}
        d["hours"], rem = divmod(tdelta.seconds, 3600)
        d["minutes"], d["seconds"] = divmod(rem, 60)
        return fmt.format(**d)
        
    def EndTheExam(self):
        try:
            # keyboard.unhook_all_hotkeys()
            ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)
            
            self.thControllersStop.listenerKeyBoard.stop()
            self.thControllersStop.listenerMouse.stop()
            
            self.th.ThreadCameraVideoIsRun = False
            self.th.cap.release()
            self.th.out.release()
            self.th.outScreen.release()
            self.th.audio_thread.stop()


            self.thCloseAppsThreadRunning=False
            try:
                self.chat_namespace.emit('ProctorEndExam',{'Ahmed': 'yyy'})
            except:
                pass
            self.OpenLoaderUpload()
            
        except:
            self.close()
        

    def getUnique(self,fileSize):
        t = datetime.datetime.now()
        dateRand = (t-datetime.datetime(1970,1,1)).total_seconds()
        return int(math.floor(random.randint(33333, 999999)) + dateRand + fileSize)


       

    

    def MinimizeVideo(self):
        if self.IsMinimized:
            self.IsMinimized = False
            self.animation1 = QPropertyAnimation(self.ex, b"size")
            self.animation1.setDuration(500) #Default 250ms
            self.animation1.setEndValue(QSize(249,289))
            self.ex.setFixedWidth(249)
            self.animation1.start()
        else:
            self.IsMinimized = True
            self.animation = QPropertyAnimation(self.ex, b"size")
            self.animation.setDuration(500) #Default 250ms
            self.animation.setEndValue(QSize(249,39))
            self.ex.setFixedWidth(249)
            self.animation.start()

        

    def close(self):
        try:
            
            self.thControllersStop.listenerKeyBoard.stop()
            self.thControllersStop.listenerMouse.stop()

            self.thUploadFileCamera.ThreadUploadingFiles = False
            self.thUploadFileScreen.ThreadUploadingFiles = False
            
            
            
            # Delete the files
            if os.path.exists(self.th.PathOfFile):
                os.remove(self.th.PathOfFile)
        
            if os.path.exists(self.th.PathOfFileUploaded):
                os.remove(self.th.PathOfFileUploaded)
            
            if os.path.exists(self.th.PathNameOfFileScreen):
                os.remove(self.th.PathNameOfFileScreen)

            if os.path.exists(self.th.filenameWav):
                os.remove(self.th.filenameWav)

            if os.path.exists(tempfile.gettempdir()+"\\tempsound.wav"):
                os.remove(tempfile.gettempdir()+"\\tempsound.wav")

            
        except:
            pass
        try:
            self.chat_namespace.emit('ProctorEndExam',{'Ahmed': 'yyy'})
            self.socketIO.disconnect()
        except:
            pass

        QCoreApplication.exit(0)
    

    
    @pyqtSlot(str)
    def setLightHint(self, title):
        self.ex.LightNote.setText(title)

    @pyqtSlot(str)
    def setTimeForSession(self, title):
        if(title == 'Accept'):
            
            timeNow = datetime.datetime.now()
            today = datetime.datetime.now()
            today.replace(hour=0, minute=0, second=0, microsecond=0)
            reminingTime = (self.dateTimeEndExam - timeNow)
            self.ex.TimerNote.setText(self.strfdelta(reminingTime, "{hours}:{minutes}:{seconds}"))
            if reminingTime <= datetime.timedelta(seconds=0):
                self.EndTheExam()

    @pyqtSlot()
    def goToErrorPage(self):
        print("Error")
        self.error = QMainWindow()
        self.error.setAttribute(Qt.WA_TranslucentBackground)
        uic.loadUi(self.dir_path+'\error.ui', self.error) # Load the .ui file
        self.error.errorLabel.setText("Error in the internet")
        
        self.error.setWindowFlags(Qt.WindowStaysOnTopHint | Qt.FramelessWindowHint)
        self.error.okError.clicked.connect(self.restart)
        self.error.close.clicked.connect(self.close)
        
        
        try:
            self.hide()
            self.error.show()
        except:
            print("Error")
            
    def goNextStep(self,Move):
        self.predictButton.setStyleSheet("""QPushButton{background-color: rgb(190, 188, 188);
        border-style: outset;
        border-width: 1px;
        border-radius: 5px;
        border-color: #e8e8e8;
        padding: 4px;
        color: #fbfbfb;
        font-size: 15px;
        font-weight: 700;}""")
        self.predictButton.setEnabled(False)
        if Move:
            #Move if the index indicated
            self.pagerWindow.setCurrentIndex(self.stepNow)
        if self.stepNow == 0: # step 0
            print("Checking Internet...")

            
            self.checkInternetThread = ThreadInternetConnection(self)
            self.checkInternetThread.ShowErrorPanel.connect(self.goToErrorPage)
            self.checkInternetThread.GoNextStep.connect(self.goNextStepSlotOutSide)
            self.checkInternetThread.start()
        elif self.stepNow == 1: # step 1

            self.checkCookiesThread = threading.Thread(target=self.checkCookies, args=())
            self.checkCookiesThread.start()
            
            loop = QEventLoop()
            QTimer.singleShot(1000, loop.quit)
            loop.exec_()

        elif self.stepNow == 2: # step 2
            print("Make the analysis")
            try:
                makemodelUrl = 'http://54.74.171.130:8083/makeModelForFaces'
                payload = json.dumps({"token": self.token,
                                    "secretKey": "17iooi1kfb8qq1b",
                                    "privateKey":"160061482862217iooi1kfb8qq1c"})
                headers = {
                    'content-type': "application/json",
                    'cache-control': "no-cache"
                    }

                response = requests.request("POST", makemodelUrl, data=payload, headers=headers,timeout=1)
                print(response.text)
            except:
                pass
            print("Device Checking...")
            # Make a Check for All Devices Thread
            
            self.checkDevicesThread = ThreadDeviceCheckConnection(self)
            self.checkDevicesThread.ShowErrorPanel.connect(self.goToErrorPageWebsite)
            self.checkDevicesThread.openListOfClosingApp.connect(self.openClosingAppMenu)
            self.checkDevicesThread.start()


            
            
        elif self.stepNow == 3: # step 3
            if self.isOnBoard:
                self.predict()
            else:
                self.Recognition()
        elif self.stepNow == 4: # step 3
            if True and (self.isOnBoard):
                self.pagerWindow.setCurrentIndex(self.stepNow)
                self.repeatButton.setVisible(False)
                self.predictHand()
            else:
                self.stepNow +=1
                self.goNextStep(False)
        elif self.stepNow == 5: # step 3
            if True and (self.isOnBoard):
                self.pagerWindow.setCurrentIndex(self.stepNow)
                self.repeatButton.setVisible(False)
                self.predictId()
            else:
                self.stepNow +=1
                self.goNextStep(False)
        elif self.stepNow == 6:
            if (self.isOnBoard):
                SentTheImages = True
                QThread.msleep(1000)
                while SentTheImages:
                    headers = {'authorization': "Bearer "+str(self.token)}
                    if len(self.AllImagesFaces) > 0 and len(self.AllImagesHand) > 0 and len(self.AllImagesId) > 0 :
                        dataNew = {"faceImages":self.AllImagesFaces,
                                    "knuckleImages":self.AllImagesHand,
                                    "idImages":self.AllImagesId,"trial":self.examId}
                    elif len(self.AllImagesFaces) > 0 and len(self.AllImagesId) > 0 :
                        dataNew = {"faceImages":self.AllImagesFaces,
                                    "idImages":self.AllImagesId,"trial":self.examId}
                    elif len(self.AllImagesFaces) > 0 and len(self.AllImagesHand) > 0 :
                        dataNew = {"faceImages":self.AllImagesFaces,
                                    "knuckleImages":self.AllImagesHand,"trial":self.examId}
                    else  :
                        dataNew = {"faceImages":self.AllImagesFaces,"trial":self.examId}
                    # print(dataNew)
                    UrlPostData = 'http://new.tproctor.teqnia-tech.com/api/proctor-app/proctoring-image'
                    response = requests.post(UrlPostData,json=dataNew,headers=headers)
                    try:
                        self.IdFromUploadedImages =self.examId# response.json()['userTestTrial']['id']
                        SentTheImages = False
                    except:
                        print('---------------------------------------------')
                        print(response.text)
            else:
                SentTheImages = True
                QThread.msleep(1000)
                while SentTheImages:
                    headers = {'authorization': "Bearer "+str(self.token)}
                    
                    dataNew = {"faceImages":self.AllImagesFaces,"trial":self.examId}
                    print(dataNew)
                    UrlPostData = 'http://new.tproctor.teqnia-tech.com/api/proctor-app/proctoring-images'
                    response = requests.post(UrlPostData,json=dataNew,headers=headers)
                    try:
                        self.IdFromUploadedImages = self.examId# response.json()['userTestTrial']['id']
                        SentTheImages = False
                    except:
                        print('---------------------------------------------')
                        print(response.text)
            
            self.stepNow +=1
            self.goNextStep(False)
        elif self.stepNow == 7:
            self.predictButton.setEnabled(True)
            self.predictButton.setStyleSheet("""QPushButton{background-color: #0095ff;
            border-style: outset;
            border-width: 1px;
            border-radius: 5px;
            border-color: #e8e8e8;
            padding: 4px;
            color: #fbfbfb;
            font-size: 15px;
            font-weight: 700;}
            
            QPushButton:hover{background-color: #0095ff;
            border-style: outset;
            border-width: 1px;
            border-radius: 5px;
            border-color: #e8e8e8;
            padding: 4px;
            color: #565050;
            font-size: 15px;
            font-weight: 700;}""")
            self.stepNow +=1
            self.goNextStep(False)

        elif self.stepNow == 8:
            self.ReduceWindowAndMove()
        elif self.stepNow == -1:
            self.ReduceWindowAndMove()

    def mousePressEvent(self, event):
        if self.WindowCameraOpened:
            self.oldPos = event.globalPos()
        else:
            self.oldPos = event.globalPos()
            

    def mouseMoveEvent(self, event):
        delta = QPoint (event.globalPos() - self.oldPos)
        if self.WindowCameraOpened:
            self.ex.move(self.ex.x() + delta.x(), self.ex.y() + delta.y())
            self.oldPos = event.globalPos()
        else:
            self.move(self.x() + delta.x(), self.y() + delta.y())
            self.oldPos = event.globalPos()

    def checkCookies(self):
        print("Internet Success...")
        
        
        self.token = token
        self.examId = examId 
        
                

        dataNew = {"token": self.token}
        headers = {'authorization': "Bearer "+str(self.token)}
        UrlPostData = 'http://new.tproctor.teqnia-tech.com/api/proctor-app/me'
        self.TestDurationInt = '0'
        
        print("----------------------------------")
        if self.token != None and self.examId!=None:
            response = requests.get(UrlPostData,headers=headers)
            print(response.json())
            self.Username = response.json()['name']
            # self.IsVerified = response.json()['user']['active']
            self.studentId = response.json()['id']
            self.roomId = response.json()['id']
            print("------------------------------------------------")
            print(self.studentId)
            self.socketIO = SocketIO('http://new.tproctor.teqnia-tech.com', 80,ChatNamespace,params={'id': str(self.roomId)})
            self.chat_namespace = self.socketIO.define(ChatNamespace, '/room')
            self.chat_namespace.on('EndExam', self.closeAllFromSocket)

            print("Socket has been ini")
            socketMainThread = threading.Thread(target=self.waitmainSocket, args=())
            socketMainThread.start()
            print("Socket has been started")

            headers = {'authorization': "Bearer "+str(self.token)}
            dataNew = {"token": self.token}
            UrlPostData = 'http://new.tproctor.teqnia-tech.com/api/proctor-app/test-requirements/'+self.examId
            response = requests.get(UrlPostData,json=dataNew,headers=headers)
            
            print("-----------Request--------------")
            # print(response.json())
            self.TestName = response.json()['test']['name']
            self.isOnBoard = response.json()['test']['onboardingTest']
            # self.isOnBoard = False
            self.TestDuration = str(response.json()['test']['duration']) +' m'
            self.AllNotAllowed = response.json()['testWhiteListApps']
            print('----------------------------------------------')
            print('----------------------------------------------')
            print('----------------------------------------------')
            print(self.AllNotAllowed)
            print('----------------------------------------------')
            print('----------------------------------------------')
            print('----------------------------------------------')

            listAllow = list(['Taskmgr.exe','ClearPassOnGuard.exe','mysqld.exe','httpd.exe','WindowsInternal.ComposableShell.Experiences.TextInput.InputApp.exe','MiPhoneHelper.exe','igfxEM.exe','YourPhone.exe','LWS.exe','jusched.exe','LockApp.exe','fmapp.exe','OneDrive.exe','SecurityHealthHost.exe','dasHost.exe','mbamtray.exe','CameraHelperShell.exe','Music.UI.exe','ApplicationFrameHost.exe','SettingSyncHost.exe','MsMpEng.exe','SASrv.exe','unsecapp.exe','AGMService.exe',
                            'AGSService.exe','CAudioFilterAgent64.exe','igfxHK.exe','sihost.exe','SecurityHealthSystray.exe','sqlceip.exe','MicrosoftEdgeUpdate.exe',
                            'SearchFilterHost.exe','WmiPrvSE.exe','fbs.exe','RtkBtManServ.exe','ETDService.exe','sqlservr.exe','StartMenuExperienceHost.exe','smartscreen.exe',
                            'SearchProtocolHost.exe','dllhost.exe','PanGPA.exe','IEMonitor.exe','ETDCtrl.exe','spoolsv.exe','MBAMService.exe','arubanetsvc.exe','ETDService.exe','PanGPS.exe','TiWorker.exe','TrustedInstaller.exe',
                            'ctfmon.exe','cmd.exe','chrome.exe','SearchUI.exe','RuntimeBroker.exe','jucheck.exe','igfxCUIService.exe','SgrmBorker.exe','dwm.exe','audiodg.exe','Isass.exe','ClearPassAgentController.exe',
                              'ShellExperienceHost.exe','Code.exe','svchost.exe','taskhostw.exe','Video.UI.exe','NVDisplay.Container.exe','MemCompression','smss.exe','fontdrvhost.exe','ClearPassOnGuardAgentService.exe','SgrmBroker.exe',
                              'SecurityHealthService.exe','winlogon.exe','sqlwriter.exe','ETDCtrlHelper.exe','csrss.exe','wininit.exe','services.exe','lsass.exe','PresentationFontCache.exe','NisSrv.exe','SearchIndexer.exe',
                              'fontdrvhost.exe','backgroundTaskHost.exe','conhost.exe','igfxTray.exe',
                              'python.exe','explorer.exe','svchost.exe','Proctoring.exe','System Idle Process','System','Registry','WUDFHost.exe'])
            for item in listAllow:
                prog = {'deleted': False, 'SystemApp': {'serviceName': item }}
                self.AllNotAllowed.append(prog)

            print("----------------------------------")
            # print(self.AllNotAllowed)
            self.TestDurationInt = str(response.json()['test']['duration'])
            
            self.username.setText('hi, '+self.Username)
            print(self.Username)
            self.testname_label.setText(self.TestName)
            self.testduration_label.setText((self.TestDuration) )

            loop = QEventLoop()
            QTimer.singleShot(500, loop.quit)
            loop.exec_()
            self.stepNow +=1
            self.goNextStep(True)
        else:

            self.goToErrorPageWebsite("Please go to the Exam From the Website")


    
        
    

    def restart(self):
        self.show()
        self.error.hide()
        self.stepNow -=1
        self.goNextStep(True)

    

        

    def while_closeAllBlackList(self):
        while self.thCloseAppsThreadRunning:
            self.closeAllBlackList()
            time.sleep(5)

            
    @pyqtSlot()
    def closeAllBlackList(self):
        # PROCNAME = "notepad.exe"

        for proc in psutil.process_iter():
            # check whether the process name matches
            isExist = False
            for program in self.AllNotAllowed:
                if proc.name() == program['SystemApp']['serviceName']:
                    isExist = True

            if not isExist:
                try:
                    # here remove kill 
                    proc.kill()
                    # print("--------------------------------")
                    # print("Kill "+proc.name())
                except:
                    pass

    @pyqtSlot(bool)
    def goNextStepSlotOutSide(self,Accept):
        if Accept:
            self.stepNow +=1
            self.goNextStep(True)
        else:
            self.goNextStep(True)

    @pyqtSlot(str)
    def openClosingAppMenu(self,listString):
        self.hide()
        self.listOfClosingApps = QMainWindow()
        self.listOfClosingApps.setAttribute(Qt.WA_TranslucentBackground)
        uic.loadUi(self.dir_path+'\listOfClosingApps.ui', self.listOfClosingApps) # Load the .ui file
        
        
        self.listOfClosingApps.setWindowFlags(Qt.WindowStaysOnTopHint | Qt.FramelessWindowHint)
        self.listOfClosingApps.okError.clicked.connect(self.closeAppsAndContinue)
        self.listOfClosingApps.close.clicked.connect(self.close)
        

        for proc in psutil.process_iter():
            # check whether the process name matches
            isExist = False
            for program in self.AllNotAllowed:
                if proc.name() == program['SystemApp']['serviceName']:
                    isExist = True

            if not isExist:
                try:
                    i = QListWidgetItem(proc.name())
                    i.setBackground(QColor('#beaed4'))
                    self.listOfClosingApps.listWidget.addItem(i)
                except:
                    pass
        self.listOfClosingApps.show()

    def closeAppsAndContinue(self):
        self.show()
        self.listOfClosingApps.hide()
        self.closeAllBlackList()
        self.stepNow +=1
        self.goNextStep(True)
        

    @pyqtSlot(str)
    def goToErrorPageWebsite(self,statment):
        self.hide()
        self.error = QMainWindow()
        self.error.setAttribute(Qt.WA_TranslucentBackground)
        uic.loadUi(self.dir_path+'\error.ui', self.error) # Load the .ui file
        self.error.errorLabel.setText(statment)
        
        self.error.setWindowFlags(Qt.WindowStaysOnTopHint | Qt.FramelessWindowHint)
        self.error.okError.clicked.connect(self.restart)
        self.error.close.clicked.connect(self.close)
        self.error.show()

    @pyqtSlot(QImage)
    def setImageVideo(self, image):
        source = QPixmap.fromImage(image)
        
        self.ex.CameraVideo.setPixmap(source)

    @pyqtSlot(QImage)
    def setImage(self, image):
        source = QPixmap.fromImage(image)
        output = QPixmap(source.size())
        
        output.fill(Qt.transparent)
        # # create a new QPainter on the output pixmap
        qp = QPainter(output)
        qp.setBrush(QBrush(source))
        qp.setPen(Qt.NoPen)
        qp.drawRoundedRect(output.rect(), 150, 150)
        qp.end()
        self.cameraHolder.setPixmap(output)

    @pyqtSlot(QImage)
    def setImageHandAndId(self, image):
        source = QPixmap.fromImage(image)
        output = QPixmap(source.size())
        
        output.fill(Qt.transparent)
        # # create a new QPainter on the output pixmap
        qp = QPainter(output)
        qp.setBrush(QBrush(source))
        qp.setPen(Qt.NoPen)
        qp.drawRoundedRect(output.rect(), 10, 10)
        qp.end()
        self.cameraHolder_2.setPixmap(output)
    

    @pyqtSlot(bool)
    def setBoolImageFace(self, boolState):
        if boolState == False:
            self.wrongLabel.hide()
        else:
            self.wrongLabel.show()

    @pyqtSlot(str)
    def setCameraPose(self, title):
        self.followPose.setText(title)

    @pyqtSlot(str)
    def setStringValue(self, title):
        self.followPoseHandAndId.setText(title)
        

    @pyqtSlot(str)
    def recognitionFail(self,statment):
        self.hide()
        self.error = QMainWindow()
        self.error.setAttribute(Qt.WA_TranslucentBackground)
        uic.loadUi(self.dir_path+'\error.ui', self.error) # Load the .ui file
        self.error.errorLabel.setText(statment)
        
        self.error.setWindowFlags(Qt.WindowStaysOnTopHint | Qt.FramelessWindowHint)
        self.error.okError.clicked.connect(self.restart)
        self.error.close.clicked.connect(self.close)
        self.error.show()
  

    @pyqtSlot(bool)
    def goCheckingForNext(self,statues):
        # self.stepNow +=1
        # self.goNextStep(statues)
        self.predictButton.setEnabled(True)
        self.predictButton.setStyleSheet("""QPushButton{background-color: #0095ff;
        border-style: outset;
        border-width: 1px;
        border-radius: 5px;
        border-color: #e8e8e8;
        padding: 4px;
        color: #fbfbfb;
        font-size: 15px;
        font-weight: 700;}
        
        QPushButton:hover{background-color: #0095ff;
        border-style: outset;
        border-width: 1px;
        border-radius: 5px;
        border-color: #e8e8e8;
        padding: 4px;
        color: #565050;
        font-size: 15px;
        font-weight: 700;}""")
        self.stepNow +=1
        print("now   ",self.stepNow)

    @pyqtSlot(int)
    def showReloadButtonandSetAction(self, typeOfAction):
        
        print(self.stepNow)
        if typeOfAction == 0:
            # Action for Hand
            self.repeatButton.setVisible(True)
            
            
        elif typeOfAction == 1:
            # Action for Id
            self.repeatButton.setVisible(True)


    def reloadHandOrIdCap(self):
        
        if self.stepNow == 5:
            self.stepNow -= 1
            self.repeatButton.setVisible(False)
            self.AllImagesHand = []
            self.predictHand()
        elif self.stepNow == 6:
            self.stepNow -= 1
            self.repeatButton.setVisible(False)
            self.AllImagesId = []
            self.predictId()
        

    def Recognition(self):
        print("Start Recognition")
        self.Thread_Of_Prediction_Is_Run = True
        self.thCamera = ThreadCameraRecognition(self,self.token,self.examId,self.studentId,self.AllImagesFaces)
        self.thCamera.changePixmap.connect(self.setImage)
        self.thCamera.setPose.connect(self.setCameraPose)
        self.thCamera.setBoolStateFace.connect(self.setBoolImageFace)
        self.thCamera.checkingEnded.connect(self.goCheckingForNext)
        self.thCamera.recognitionFailCamera.connect(self.recognitionFail)
        self.thCamera.start()

    def predict(self):
        print("Start Prediction")
        # self.camera.startCap()
        self.Thread_Of_Prediction_Is_Run = True
        self.thCamera = ThreadCamera(self,self.token,self.examId,self.AllImagesFaces)
        self.thCamera.changePixmap.connect(self.setImage)
        self.thCamera.setPose.connect(self.setCameraPose)
        self.thCamera.setBoolStateFace.connect(self.setBoolImageFace)
        self.thCamera.checkingEnded.connect(self.goCheckingForNext)
        self.thCamera.start()
    


    def predictHand(self):
        print("Start Prediction Hand")
        # self.camera.startCap()
        self.Thread_Of_PredictionHand_Is_Run = True
        self.thCameraHand = ThreadCameraHand(self,self.token,self.examId,self.AllImagesHand)
        self.thCameraHand.changePixmap.connect(self.setImageHandAndId)
        self.thCameraHand.setStringData.connect(self.setStringValue)
        self.thCameraHand.checkingEnded.connect(self.goCheckingForNext)
        self.thCameraHand.reloadCameraShow.connect(self.showReloadButtonandSetAction)
        self.thCameraHand.start()

    def predictId(self):
        print("Start Prediction Id")
        # self.camera.startCap()
        self.Thread_Of_PredictionId_Is_Run = True
        self.thCameraId = ThreadCameraId(self,self.token,self.examId,self.AllImagesId)
        self.thCameraId.changePixmap.connect(self.setImageHandAndId)
        self.thCameraId.setStringData.connect(self.setStringValue)
        self.thCameraId.checkingEnded.connect(self.goCheckingForNext)
        self.thCameraId.reloadCameraShow.connect(self.showReloadButtonandSetAction)
        self.thCameraId.start()

    

class AudioRecorder():
    "Audio class based on pyAudio and Wave"
    def __init__(self, filename="temp_audio.wav", rate=20480, fpb=1024, channels=2):
        self.open = True
        self.rate = rate
        self.frames_per_buffer = fpb
        self.channels = channels
        self.format = pyaudio.paInt16
        self.audio_filename = filename
        self.audio = pyaudio.PyAudio()
        self.stream = self.audio.open(format=self.format,
                                      channels=self.channels,
                                      rate=self.rate,
                                      input=True,
                                      frames_per_buffer = self.frames_per_buffer)
        self.audio_frames = []
        print(self.audio_filename)


    def stop(self):
        "Finishes the audio recording therefore the thread too"
        if self.open:
            self.open = False
            self.stream.stop_stream()
            self.stream.close()
            self.audio.terminate()
            
            print("Finished")
            

    




# Uploading Camera File 
class ThreadUploadFileCamera(QThread):
    # Create the signal
    changePercentage = pyqtSignal(int)
    changeLabel =pyqtSignal(str)
    startMain = pyqtSignal(str)
    uploadScreen =pyqtSignal()

    
    def getUnique(self,fileSize):
        t = datetime.datetime.now()
        dateRand = (t-datetime.datetime(1970,1,1)).total_seconds()
        return int(math.floor(random.randint(33333, 999999)) + dateRand + fileSize)

    
    def __init__(self,window,PathOfFileUploaded,IdFromUploadedImages,token,th):
        super(ThreadUploadFileCamera,self).__init__(window)
        self.filename = PathOfFileUploaded
        self.IdFromUploadedImages = IdFromUploadedImages
        self.ThreadUploadingFiles = True
        self.token = token
        self.th = th
        self.window = window


        

    def upload_fileCamera(self):

        
        self.changeLabel.emit("Processing ...")

        cmd = '"'+dir_path+'\\ffmpeg.exe" -y -ac 2 -channel_layout stereo -i "'+self.th.PathOfFile+'" -i "'+self.th.filenameWav+'" -c:v copy -c:a aac -map 0:v:0 -map 1:a:0 "' +self.th.PathOfFileUploaded+'"'
        subprocess.call(cmd, shell=True)

        chunksize = 1000000
        totalsize = os.path.getsize(self.filename)
        totalChucks = math.ceil(totalsize/chunksize)
        readsofar = 0
        self.IdFromUploadedImages = self.IdFromUploadedImages
        
        url = "http://new.tproctor.teqnia-tech.com/api/proctor-app/proctoring-video/"+str(self.IdFromUploadedImages)+"/STUDENT"
        token = self.token
        i = 0
        uniqueId = self.getUnique(totalsize)
        with open(self.filename, 'rb') as file:
            while self.ThreadUploadingFiles:
                try:
                    data = file.read(chunksize)
                    f = open(tempfile.gettempdir()+"\\"+"fileDownload", "wb")
                    f.write(data)
                    f.close()
                    if not data:
                        sys.stderr.write("\n")
                        break
                    readsofar += len(data)
                    percent = readsofar * 1e2 / totalsize
                    
                    headers = {
                        'Access-Control-Max-Age':'86400',
                        'Access-Control-Allow-Methods': 'POST,OPTIONS' ,
                        'Access-Control-Allow-Headers': 'uploader-chunk-number,uploader-chunks-total,uploader-file-id', 
                        'Access-Control-Allow-Origin':'http://localhost:3000',
                        'authorization': "Bearer "+token,
                        'uploader-file-id': str(uniqueId),
                        'uploader-chunks-total': str(totalChucks),
                        'uploader-chunk-number': str(i)
                        }

                
                    files = {'file': ('fileDownload',open(tempfile.gettempdir()+"\\"+"fileDownload", 'rb'),'application/octet-stream')}
                    isUploaded = False
                    while not isUploaded:
                        # print(self.IdFromUploadedImages)
                        # print('----------------------------')
                        try:
                            r = requests.request('POST',url,files=files,headers=headers, verify=False)
                            print(r.status_code)
                            if r.status_code == 204:
                                print("Done")
                                isUploaded = True
                                i+=1
                        except Exception as exc:
                            print(exc)
                    self.changePercentage.emit(int(percent/2))
                    self.changeLabel.emit("Video "+str(int(percent))+"%")
                    print("\r{percent:3.0f}%".format(percent=percent))
                except:
                    pass

        self.uploadScreen.emit()

    def run(self):
        
        self.startMain.emit("Start")
        QThread.msleep(2000)
        self.upload_fileCamera()






# Uploading Camera Screen 
class ThreadUploadFileScreen(QThread):
    # Create the signal
    changePercentage = pyqtSignal(int)
    changeLabel =pyqtSignal(str)
    finishUploading =pyqtSignal()

    
    def getUnique(self,fileSize):
        t = datetime.datetime.now()
        dateRand = (t-datetime.datetime(1970,1,1)).total_seconds()
        return int(math.floor(random.randint(33333, 999999)) + dateRand + fileSize)

    
    def __init__(self,window,PathOfFileUploaded,IdFromUploadedImages,token):
        super(ThreadUploadFileScreen,self).__init__(window)
        self.filename = PathOfFileUploaded
        self.IdFromUploadedImages = IdFromUploadedImages
        self.ThreadUploadingFiles = True
        self.token = token

    def upload_fileScreen(self):
        chunksize = 1000000
        totalsize = os.path.getsize(self.filename)
        totalChucks = math.ceil(totalsize/chunksize)
        readsofar = 0
        url = "http://new.tproctor.teqnia-tech.com/api/proctor-app/proctoring-video/"+str(self.IdFromUploadedImages)+"/SCREEN"
        token = self.token
        i = 0
        uniqueId = self.getUnique(totalsize)
        with open(self.filename, 'rb') as file:
            while self.ThreadUploadingFiles:
                try:
                    data = file.read(chunksize)
                    f = open(tempfile.gettempdir()+"\\"+"fileDownload", "wb")
                    f.write(data)
                    f.close()
                    if not data:
                        sys.stderr.write("\n")
                        break
                    readsofar += len(data)
                    percent = readsofar * 1e2 / totalsize
                    
                    headers = {
                        'Access-Control-Max-Age':'86400',
                        'Access-Control-Allow-Methods': 'POST,OPTIONS' ,
                        'Access-Control-Allow-Headers': 'uploader-chunk-number,uploader-chunks-total,uploader-file-id', 
                        'Access-Control-Allow-Origin':'http://localhost:3000',
                        'authorization': "Bearer "+token,
                        'uploader-file-id': str(uniqueId),
                        'uploader-chunks-total': str(totalChucks),
                        'uploader-chunk-number': str(i)
                        }

                
                    files = {'file': ("fileDownload",open(tempfile.gettempdir()+'\\'+'fileDownload', 'rb'),'application/octet-stream')}
                    isUploaded = False
                    while not isUploaded:
                        try:
                            r = requests.request('POST',url,files=files,headers=headers, verify=False)
                            print(r.status_code)
                            if r.status_code == 204:
                                print("Done")
                                isUploaded = True
                                i+=1
                        except Exception as exc:
                            print(exc)
                    self.changePercentage.emit(int(percent/2+50))
                    print(self.IdFromUploadedImages)
                    print('----------------------------')
                    self.changeLabel.emit("Screen "+str(int(percent))+"%")
                    print("\r{percent:3.0f}%".format(percent=percent))
                except:
                    pass
        self.finishUploading.emit()
        

    def run(self):
        self.upload_fileScreen()


# Internet Connection as it handle Requests
class ThreadMouse(QThread):
    # Create the signal
    def on_clickMouse(self,x, y, button, pressed):
        if button == mouse.Button.right:
            winput.press_key(winput.VK_ESCAPE)
            time.sleep(0.1)
            winput.press_key(winput.VK_ESCAPE)
            time.sleep(0.001)
            winput.press_key(winput.VK_ESCAPE)
            time.sleep(0.3)
            winput.press_key(winput.VK_ESCAPE)

    def on_pressKeyBoard(self,key):
        try:
            # print('alphanumeric key {0} pressed'.format(
            #     key.char))
            try:
                result = subprocess.check_output('cmd /c echo.|clip', shell=True)
            except:
                pass
        except AttributeError:
            print('special key {0} pressed'.format(
                key))



    def __init__(self, mw, parent=None):
        super().__init__(parent)
        self.threadMouseRun = True
        self.listenerKeyBoard = keyboard.Listener(on_press=self.on_pressKeyBoard)
        self.listenerKeyBoard.start()

        self.listenerMouse = mouse.Listener(on_click=self.on_clickMouse)
        self.listenerMouse.start()

    def run(self):
        print('Thread of Mouse and Keyboard Started')
        # state_left = win32api.GetKeyState(0x01)  # Left button down = 0 or 1. Button up = -127 or -128
        # state_right = win32api.GetKeyState(0x02)  # Right button down = 0 or 1. Button up = -127 or -128

        # while self.threadMouseRun:
        #     b = win32api.GetKeyState(0x02)
            
        #     try:
        #         result = subprocess.check_output('cmd /c echo.|clip', shell=True)
        #     except:
        #         pass
        #     if b != state_right:  # Button state changed
        #         state_right = b
        #         if b < 0:
        #             print('Right Button Pressed')
        #             winput.press_key(winput.VK_ESCAPE)
        #             time.sleep(0.1)
        #             winput.press_key(winput.VK_ESCAPE)
        #             time.sleep(0.001)
        #             winput.press_key(winput.VK_ESCAPE)
        #             time.sleep(0.3)
        #             winput.press_key(winput.VK_ESCAPE)
        #         else:
        #             print('Right Button Released')
        #             # winput.move_mouse(-10, 1)
                    
        #             winput.press_key(winput.VK_ESCAPE)
        #             time.sleep(0.1)
        #             winput.press_key(winput.VK_ESCAPE)
        #             time.sleep(0.001)
        #             winput.press_key(winput.VK_ESCAPE)
        #             time.sleep(0.3)
        #             winput.press_key(winput.VK_ESCAPE)
            





# Internet Connection as it handle Requests
class ThreadInternetConnection(QThread):
    # Create the signal
    ShowErrorPanel = pyqtSignal()
    GoNextStep =pyqtSignal(bool)

    def __init__(self, mw, parent=None):
        super().__init__(parent)
        self.checkInternetTrial = 0

    def checkInternetConnection(self):
        self.checkInternetTrial+=1
        try:
            requests.get('https://www.google.com/', verify=False,timeout=1).status_code
            return True
        except:
            return False

    def run(self):
        while True:
            QThread.msleep(1000)
            if self.checkInternetConnection():
                self.GoNextStep.emit(True)
                break
            else:
                # check internet to 5 times of Seconds
                if self.checkInternetTrial == 5:
                    self.checkInternetTrial = 0
                    self.ShowErrorPanel.emit()
                    break
        
# Internet Connection as it handle Requests
class ThreadDeviceCheckConnection(QThread):
    # Create the signal
    ShowErrorPanel = pyqtSignal(str)
    openListOfClosingApp =pyqtSignal(str)

    
    def __init__(self,window):
        super(ThreadDeviceCheckConnection,self).__init__(window)
        self.WindowPanel = window

   #Check VMWARE
    def checkVmWare(self):
        batcmd='systeminfo /s %computername% | findstr /c:"Model:" /c:"Host Name" /c:"OS Name"'
        result = subprocess.check_output(batcmd, shell=True)
        # print(result)

        if re.search('VirtualBox', str(result), re.IGNORECASE):
            return (True)
        else:
            return (False)

    def checkMicrophone(self):
        winmm= windll.winmm
        if winmm.waveInGetNumDevs()>0:
            return True
        else:
            return False

    def checkSpeaker(self):
        p = pyaudio.PyAudio()

        for i in range(0,10):
            try:
                if p.get_device_info_by_index(i)['maxOutputChannels']>0:
                    return True
            except Exception as e:
                print (e)
                return False

    def CheckDevices(self):
        
        self.checkResult = True
        
        # Check Camera
        cap = cv2.VideoCapture(0) 
        if not (cap is None or not cap.isOpened()):
            self.WindowPanel.bar_cam.setVisible(True)
            self.WindowPanel.success_cam.setVisible(True)
        else:
            self.checkResult = False

        loop = QEventLoop()
        QTimer.singleShot(500, loop.quit)
        loop.exec_()

        # Check Mouse
        wmiService = wmi.WMI()
        self.PointingDevices = wmiService.query("SELECT * FROM Win32_PointingDevice")
        if len(self.PointingDevices)>= 1:
            self.WindowPanel.bar_mouse.setVisible(True)
            self.WindowPanel.success_mouse.setVisible(True)
        else:
            self.checkResult = False
            
        loop = QEventLoop()
        QTimer.singleShot(500, loop.quit)
        loop.exec_()

        # Check Keyboard
        self.keyboards = wmiService.query("SELECT * FROM Win32_Keyboard")
        if len(self.keyboards) >= 1:
            self.WindowPanel.bar_key.setVisible(True)
            self.WindowPanel.success_key.setVisible(True)
        else:
            self.checkResult = False
       
        loop = QEventLoop()
        QTimer.singleShot(500, loop.quit)
        loop.exec_()

        if self.checkSpeaker():
            self.WindowPanel.bar_speaker.setVisible(True)
            self.WindowPanel.success_speaker.setVisible(True)
        else:
            self.checkResult = False
        
        loop = QEventLoop()
        QTimer.singleShot(500, loop.quit)
        loop.exec_()


        if self.checkMicrophone():
            self.WindowPanel.bar_micro.setVisible(True)
            self.WindowPanel.success_micro.setVisible(True)
        else:
            self.checkResult = False
        
        loop = QEventLoop()
        QTimer.singleShot(500, loop.quit)
        loop.exec_()

        # Check Hard
        if True:
            self.WindowPanel.bar_hard.setVisible(True)
            self.WindowPanel.success_hard.setVisible(True)
        else:
            self.checkResult = False
        
        loop = QEventLoop()
        QTimer.singleShot(500, loop.quit)
        loop.exec_()

        # Check Monitor
        if True:
            self.WindowPanel.bar_screen.setVisible(True)
            self.WindowPanel.success_screen.setVisible(True)
        else:
            self.checkResult = False
        
        loop = QEventLoop()
        QTimer.singleShot(500, loop.quit)
        loop.exec_()



        if not self.checkVmWare():
            self.WindowPanel.bar_system.setVisible(True)
        else:
            self.checkResult = False
        
        loop = QEventLoop()
        QTimer.singleShot(1000, loop.quit)
        loop.exec_()



    def run(self):
        self.CheckDevices()
        if self.checkResult:
            # here we will show the list of the closed apps so we will proceed or close the App
            self.openListOfClosingApp.emit("Nothing")
        else:
            self.ShowErrorPanel.emit("Please Check the Last Devices")
        

# Camera For Pose Thread
class ThreadCameraRecognition(QThread):
    changePixmap = pyqtSignal(QImage)
    setPose = pyqtSignal(str)
    setBoolStateFace = pyqtSignal(bool)
    checkingEnded = pyqtSignal(bool)
    recognitionFailCamera= pyqtSignal(str)

    def __init__(self,window,token,examId,studentId,AllImages):
        super(ThreadCameraRecognition,self).__init__(window)
        self.token = token
        self.examId = examId
        self.FinalImage = 5
        self.AllImages = AllImages
        self.studentId = studentId

    def sendImage(self,files,headers):
        try:
            response = requests.post('http://new.tproctor.teqnia-tech.com/api/upload/files',files = files,headers=headers,timeout = 3)
            self.AllImages.append(response.json()['files'][0]['name'])
            print(response.json()['files'][0]['name'])
        except Exception as e:
            print('---------------------------------------')
            print('---------------------------------------')
            print('---------------------------------------')
            print(e)
            pass
        self.FinalImage -= 1

    def saveImage(self,direction,count,image):
        
        # if count == IMAGE_PER_POSE:
        dimOld = (160, 160)
        image = cv2.resize(image, dimOld, interpolation = cv2.INTER_AREA)
        # Save the image to the server with this id
        imencoded = cv2.imencode('.jpg', image)[1]
        fileName = str(direction)+'image.jpg'
        files = {'files': (fileName, imencoded.tostring(), 'image/jpeg', {'Expires': '0'})}
        headers = {'authorization': "Bearer "+str(self.token)}
        sendThread = threading.Thread(target=self.sendImage, args=(files,headers,))
        sendThread.start()
        


    def run(self):
        mark_detector = MarkDetector()
        poses=['verifying']
        file=0
        cap = cv2.VideoCapture(file)
        
        ret, sample_frame = cap.read()
        if ret==False:
            return    
            
        # Introduce pose estimator to solve pose. Get one frame to setup the
        # estimator according to the image size.
        height, width = sample_frame.shape[:2]
        pose_estimator = PoseEstimator(img_size=(height, width))
        
        # Introduce scalar stabilizers for pose.
        pose_stabilizers = [Stabilizer(
            state_num=2,
            measure_num=1,
            cov_process=0.1,
            cov_measure=0.1) for _ in range(6)]
        images_saved_per_pose=0
        number_of_images = 0
        
        shape_predictor = dlib.shape_predictor(dir_path+"/shape_predictor_68_face_landmarks.dat")
        face_aligner = FaceAligner(shape_predictor, desiredFaceWidth=FACE_WIDTH)
        

       
        
        pose_index = 0
        count = 0  
            
        
        images_saved_per_pose=0
        number_of_images = 0
        
        while pose_index<1:
            saveit = False
            # Read frame, crop it, flip it, suits your needs.
            ret, frame = cap.read()
            if ret is False:
                break
            if count % 10 !=0: # skip 10 frames
                count+=1
                continue
            if images_saved_per_pose==IMAGE_PER_POSE:
                pose_index+=1
                images_saved_per_pose=0

            # If frame comes from webcam, flip it so it looks like a mirror.
            frame = cv2.flip(frame, 2)

            
            frame_for_cam=frame.copy()
            original_frame=frame.copy()
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Show the image


            # here responsible for show image
            height, width = frame.shape[:2] 
            # frame =frame[int(height/4):int(3/4*height),int(width/3):int(2/3*width)]
            frame_for_cam =frame_for_cam[int(0):int(7/8*height),int(width/5):int(4/5*width)]
            
            scale_percent = 55 # percent of original size
            widthNew = int(frame_for_cam.shape[1] * scale_percent / 100)
            heightNew = int(frame_for_cam.shape[0] * scale_percent / 100)
            dimOld = (widthNew, heightNew)
            frame_for_cam = cv2.resize(frame_for_cam, dimOld, interpolation = cv2.INTER_AREA)

            rgbImage = cv2.cvtColor(frame_for_cam, cv2.COLOR_BGR2RGB)
            h, w, ch = rgbImage.shape
            bytesPerLine = ch * w

            convertToQtFormat = QImage(rgbImage.data, w, h, bytesPerLine, QImage.Format_RGB888)
            self.changePixmap.emit(convertToQtFormat)
                
            # end of show image

            #
            
            facebox = mark_detector.extract_cnn_facebox(frame)
        
            if facebox is not None:
                # Detect landmarks from image of 128x128.
                x1=max(facebox[0]-0,0)
                x2=min(facebox[2]+0,width)
                y1=max(facebox[1]-0,0)
                y2=min(facebox[3]+0,height)
                
                face = frame[y1: y2,x1:x2]
                face_img = cv2.resize(face, (CNN_INPUT_SIZE, CNN_INPUT_SIZE))
                face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
                
                marks = mark_detector.detect_marks([face_img])
        
                # Convert the marks locations from local CNN to global image.
                marks *= (facebox[2] - facebox[0])
                marks[:, 0] += facebox[0]
                marks[:, 1] += facebox[1]
            
                # Try pose estimation with 68 points.
                pose = pose_estimator.solve_pose_by_68_points(marks)
        
                # Stabilize the pose.
                steady_pose = []
                pose_np = np.array(pose).flatten()
                for value, ps_stb in zip(pose_np, pose_stabilizers):
                    ps_stb.update([value])
                    steady_pose.append(ps_stb.state[0])
                steady_pose = np.reshape(steady_pose, (-1, 3))
        
                if pose_index==0:
                    if abs(steady_pose[0][0])<ANGLE_THRESHOLD and abs(steady_pose[0][1])<ANGLE_THRESHOLD:
                        images_saved_per_pose+=1
                        
                        self.setBoolStateFace.emit(False)
                        saveit = True
                        dimOld = (160, 160)
                        image = cv2.resize(face, dimOld, interpolation = cv2.INTER_AREA)
                        # Save the image to the server with this id
                        imencoded = cv2.imencode('.jpg', image)[1]
                        fileName = 'recognimage.jpg'
                        files = {'image': (fileName, imencoded.tostring(), 'image/jpeg', {'Expires': '0'})}
                        payload = {"secretKey": "17iooi1kfb8qq1b",
                                "privateKey":"160061482862217iooi1kfb8qq1c"}
                        try:
                            urlRecognition = "http://54.74.171.130:8083/faceTheAnalysis"
                            response = requests.request("POST", urlRecognition,files = files,data = payload)
                            print(response.text)
                            print(response.json()['state'])
                            if response.json()['state'] == 'findFace' and str(self.studentId) == response.json()['personId']:
                                pose_index = 0
                                self.setPose.emit('Thank you')
                                cap.release()

                                self.saveImage(poses[pose_index],images_saved_per_pose,face)
                                self.setPose.emit('Verified')
                                self.checkingEnded.emit(True)
                                break
                        except Exception as ex:
                            print(ex)
                            cap.release()
                            self.recognitionFailCamera.emit('There is a problem in the server')
                            break
                    else:
                        self.setBoolStateFace.emit(True)         
                if pose_index>=1:
                    cap.release()
                    self.recognitionFailCamera.emit('You are not Allowed')
                    break

                # frame = cv2.rectangle(frame, (x1,y1), (x2,y2),(255,255,0),2)

            try:
                self.setPose.emit(str(poses[pose_index] +' : '+ str(images_saved_per_pose)+'/'+str(IMAGE_PER_POSE)))
            except:
                pass
            # self.setPose.emit('Look '+str(poses[pose_index]))
             
                
                
                        
                


# Camera For Pose Thread
class ThreadCamera(QThread):
    changePixmap = pyqtSignal(QImage)
    setPose = pyqtSignal(str)
    setBoolStateFace = pyqtSignal(bool)
    checkingEnded = pyqtSignal(bool)

    def __init__(self,window,token,examId,AllImages):
        super(ThreadCamera,self).__init__(window)
        self.token = token
        self.examId = examId
        self.FinalImage = 5
        self.AllImages = AllImages

    def saveImage(self,direction,count,image):
        
        # if count == IMAGE_PER_POSE:
        dimOld = (160, 160)
        image = cv2.resize(image, dimOld, interpolation = cv2.INTER_AREA)
        # Save the image to the server with this id
        imencoded = cv2.imencode('.jpg', image)[1]
        fileName = str(direction)+'image.jpg'
        files = {'files': (fileName, imencoded.tostring(), 'image/jpeg', {'Expires': '0'})}
        headers = {'authorization': "Bearer "+str(self.token)}
        sendThread = threading.Thread(target=self.sendImage, args=(files,headers,))
        sendThread.start()
        
    def sendImage(self,files,headers):
        try:
            response = requests.post('http://new.tproctor.teqnia-tech.com/api/upload/files',files = files,headers=headers,timeout = 3)
            self.AllImages.append(response.json()['files'][0]['name'])
            print(response.json()['files'][0]['name'])
        except Exception as e:
            print('---------------------------------------')
            print('---------------------------------------')
            print('---------------------------------------')
            print(e)
        self.FinalImage -= 1
        


    def run(self):
        mark_detector = MarkDetector()
        poses=['frontal','right','left','up','down']
        file=0
        cap = cv2.VideoCapture(file)
        
        ret, sample_frame = cap.read()
        if ret==False:
            return    
            
        # Introduce pose estimator to solve pose. Get one frame to setup the
        # estimator according to the image size.
        height, width = sample_frame.shape[:2]
        pose_estimator = PoseEstimator(img_size=(height, width))
        
        # Introduce scalar stabilizers for pose.
        pose_stabilizers = [Stabilizer(
            state_num=2,
            measure_num=1,
            cov_process=0.1,
            cov_measure=0.1) for _ in range(6)]
        images_saved_per_pose=0
        number_of_images = 0
        
        shape_predictor = dlib.shape_predictor(dir_path+"/shape_predictor_68_face_landmarks.dat")
        face_aligner = FaceAligner(shape_predictor, desiredFaceWidth=FACE_WIDTH)
        

       
        
        pose_index = 0
        count = 0  
            
        
        images_saved_per_pose=0
        number_of_images = 0
        
        while pose_index<5:
            saveit = False
            # Read frame, crop it, flip it, suits your needs.
            ret, frame = cap.read()
            if ret is False:
                break
            if count % 10 !=0: # skip 10 frames
                count+=1
                continue
            if images_saved_per_pose==IMAGE_PER_POSE:
                pose_index+=1
                images_saved_per_pose=0

            # If frame comes from webcam, flip it so it looks like a mirror.
            frame = cv2.flip(frame, 2)

            
            frame_for_cam=frame.copy()
            original_frame=frame.copy()
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Show the image


            # here responsible for show image
            height, width = frame.shape[:2] 
            # frame =frame[int(height/4):int(3/4*height),int(width/3):int(2/3*width)]
            frame_for_cam =frame_for_cam[int(0):int(7/8*height),int(width/5):int(4/5*width)]
            
            scale_percent = 59 # percent of original size
            widthNew = int(frame_for_cam.shape[1] * scale_percent / 100)
            heightNew = int(frame_for_cam.shape[0] * scale_percent / 100)
            dimOld = (widthNew, heightNew)
            frame_for_cam = cv2.resize(frame_for_cam, dimOld, interpolation = cv2.INTER_AREA)

            rgbImage = cv2.cvtColor(frame_for_cam, cv2.COLOR_BGR2RGB)
            h, w, ch = rgbImage.shape
            bytesPerLine = ch * w

            convertToQtFormat = QImage(rgbImage.data, w, h, bytesPerLine, QImage.Format_RGB888)
            self.changePixmap.emit(convertToQtFormat)
                
            # end of show image

            #
            
            facebox = mark_detector.extract_cnn_facebox(frame)
        
            if facebox is not None:
                # Detect landmarks from image of 128x128.
                x1=max(facebox[0]-0,0)
                x2=min(facebox[2]+0,width)
                y1=max(facebox[1]-0,0)
                y2=min(facebox[3]+0,height)
                
                face = frame[y1: y2,x1:x2]
                face_img = cv2.resize(face, (CNN_INPUT_SIZE, CNN_INPUT_SIZE))
                face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
                
                marks = mark_detector.detect_marks([face_img])
        
                # Convert the marks locations from local CNN to global image.
                marks *= (facebox[2] - facebox[0])
                marks[:, 0] += facebox[0]
                marks[:, 1] += facebox[1]
            
                # Try pose estimation with 68 points.
                pose = pose_estimator.solve_pose_by_68_points(marks)
        
                # Stabilize the pose.
                steady_pose = []
                pose_np = np.array(pose).flatten()
                for value, ps_stb in zip(pose_np, pose_stabilizers):
                    ps_stb.update([value])
                    steady_pose.append(ps_stb.state[0])
                steady_pose = np.reshape(steady_pose, (-1, 3))
        
                if pose_index==0:
                    if abs(steady_pose[0][0])<ANGLE_THRESHOLD and abs(steady_pose[0][1])<ANGLE_THRESHOLD:
                        images_saved_per_pose+=1
                        self.saveImage(poses[pose_index],images_saved_per_pose,face)  
                        self.setBoolStateFace.emit(False)
                        saveit = True
                    else:
                        self.setBoolStateFace.emit(True)         
                if pose_index==1:
                    if steady_pose[0][0]>ANGLE_THRESHOLD:
                        images_saved_per_pose+=1
                        self.saveImage(poses[pose_index],images_saved_per_pose,face)
                        self.setBoolStateFace.emit(False)
                        saveit = True
                    else:
                        self.setBoolStateFace.emit(True)  
                if pose_index==2:
                    if steady_pose[0][0]<-ANGLE_THRESHOLD:
                        images_saved_per_pose+=1
                        self.saveImage(poses[pose_index],images_saved_per_pose,face)
                        self.setBoolStateFace.emit(False)
                        saveit = True
                    else:
                        self.setBoolStateFace.emit(True)  
                if pose_index==3:
                    if steady_pose[0][1]<-ANGLE_THRESHOLD:
                        images_saved_per_pose+=1
                        self.saveImage(poses[pose_index],images_saved_per_pose,face)
                        self.setBoolStateFace.emit(False)
                        saveit = True
                    else:
                        self.setBoolStateFace.emit(True)  
                if pose_index==4:
                    if steady_pose[0][1]>ANGLE_THRESHOLD:
                        images_saved_per_pose+=1
                        self.saveImage(poses[pose_index],images_saved_per_pose,face)
                        self.setBoolStateFace.emit(False)
                        saveit = True
                    else:
                        self.setBoolStateFace.emit(True)  
                # Show preview.
                if pose_index>=5:
                    self.setPose.emit('Thank you')
                    break

                # frame = cv2.rectangle(frame, (x1,y1), (x2,y2),(255,255,0),2)

            try:
                self.setPose.emit('Look '+str(poses[pose_index] +' : '+ str(images_saved_per_pose)+'/'+str(IMAGE_PER_POSE)))
            except:
                pass
            # self.setPose.emit('Look '+str(poses[pose_index]))
             
                
                
                        
                
        cap.release()
        self.setPose.emit('click Next')
        self.checkingEnded.emit(True)


# Camera For Pose Thread
class ThreadCameraHand(QThread):
    changePixmap = pyqtSignal(QImage)
    setStringData = pyqtSignal(str)
    checkingEnded = pyqtSignal(bool)
    reloadCameraShow = pyqtSignal(int)

    def __init__(self,window,token,examId,AllImagesHand):
        super(ThreadCameraHand,self).__init__(window)
        self.token = token
        self.examId = examId
        self.FinalImage = 1
        # 0  for hand
        # 1 for id
        self.CurrentImage = 0
        self.AllImagesHand = AllImagesHand

    def saveImage(self,direction,count,image):
        
        if count == IMAGE_PER_PIC:
            # Save the image to the server with this id
            imencoded = cv2.imencode('.jpg', image)[1]
            fileName = str(direction)+'image.jpg'
            files = {'files': (fileName, imencoded.tostring(), 'image/jpeg', {'Expires': '0'})}
            headers = {'authorization': "Bearer "+str(self.token)}
            sendThread = threading.Thread(target=self.sendImage, args=(files,headers,))
            sendThread.start()
        
    def sendImage(self,files,headers):
        try:
            try:
                response = requests.post('http://new.tproctor.teqnia-tech.com/api/upload/files',files = files,headers=headers,timeout = 3)
                self.AllImagesHand.append(response.json()['files'][0]['name'])
                print(response.json()['files'][0]['name'])
            except Exception as e:
                print('---------------------------------------')
                print('---------------------------------------')
                print('---------------------------------------')
                print(e)
            self.checkingEnded.emit(False)
            self.reloadCameraShow.emit(0)
            print('-------------------------------')
            # print(response.json()['files'][0]['name'])
        except:
            pass
        

    def run(self):
        poses=['Hand']
        cap = cv2.VideoCapture(0)
        
        ret, sample_frame = cap.read()
        if ret==False:
            return    
            
        # Introduce pose estimator to solve pose. Get one frame to setup the
        # estimator according to the image size.
        height, width = sample_frame.shape[:2]
        
        images_saved_per_pose=0
        number_of_images = 0
        
       
        
        pose_index = 0
        count = 0  
            
        
        images_saved_per_pose=0
        number_of_images = 0
        
        while pose_index<2:
            saveit = False
            # Read frame, crop it, flip it, suits your needs.
            ret, frame = cap.read()
            if ret is False:
                break
            if count % 10 !=0: # skip 10 frames
                count+=1
                continue
            if images_saved_per_pose==IMAGE_PER_PIC:
                pose_index+=1
                images_saved_per_pose=0

            # If frame comes from webcam, flip it so it looks like a mirror.
            frame = cv2.flip(frame, 2)

            
            frame_for_cam=frame.copy()
            original_frame=frame.copy()
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Show the image


            # here responsible for show image
            height, width = frame.shape[:2] 
            # frame =frame[int(height/4):int(3/4*height),int(width/3):int(2/3*width)]
            # frame_for_cam =frame_for_cam[int(0):int(7/8*height),int(width/5):int(4/5*width)]
            # frame_for_cam =frame_for_cam[int(0):int(height),0:width]
            
            scale_percent = 55 # percent of original size
            widthNew = int(frame_for_cam.shape[1] * scale_percent / 100)
            heightNew = int(frame_for_cam.shape[0] * scale_percent / 100)
            dimOld = (widthNew, heightNew)
            frame_for_cam = cv2.resize(frame_for_cam, dimOld, interpolation = cv2.INTER_AREA)

            rgbImage = cv2.cvtColor(frame_for_cam, cv2.COLOR_BGR2RGB)
            h, w, ch = rgbImage.shape
            bytesPerLine = ch * w

            convertToQtFormat = QImage(rgbImage.data, w, h, bytesPerLine, QImage.Format_RGB888)
            self.changePixmap.emit(convertToQtFormat)
                
            # end of show image

            #
            
            if pose_index==0:
                images_saved_per_pose+=1
                self.CurrentImage = 0
                self.saveImage(poses[pose_index],images_saved_per_pose,frame)
                saveit = True
                      
            # Show preview.
            if pose_index>=self.FinalImage:
                self.setStringData.emit('Thank you')
                break

            # frame = cv2.rectangle(frame, (x1,y1), (x2,y2),(255,255,0),2)

            self.setStringData.emit('Show Your '+str(poses[pose_index] +' : '+ str(images_saved_per_pose)+'/'+str(IMAGE_PER_PIC)))
            # self.setPose.emit('Look '+str(poses[pose_index]))
             
                
                
                        
                
        cap.release()
        self.setStringData.emit('Prepair Your Id or ReCapture')
    

class ThreadCameraId(QThread):
    changePixmap = pyqtSignal(QImage)
    setStringData = pyqtSignal(str)
    checkingEnded = pyqtSignal(bool)
    reloadCameraShow = pyqtSignal(int)

    def __init__(self,window,token,examId,AllImagesId):
        super(ThreadCameraId,self).__init__(window)
        self.token = token
        self.examId = examId
        self.FinalImage = 1
        # 0  for hand
        # 1 for id
        self.CurrentImage = 0
        self.AllImagesId = AllImagesId

    def saveImage(self,direction,count,image):
        
        if count == IMAGE_PER_PIC:
            # Save the image to the server with this id
            imencoded = cv2.imencode('.jpg', image)[1]
            fileName = str(direction)+'image.jpg'
            files = {'files': (fileName, imencoded.tostring(), 'image/jpeg', {'Expires': '0'})}
            headers = {'authorization': "Bearer "+str(self.token)}
            sendThread = threading.Thread(target=self.sendImage, args=(files,headers,))
            sendThread.start()
        
    def sendImage(self,files,headers):
        try:
            try:
                response = requests.post('http://new.tproctor.teqnia-tech.com/api/upload/files',files = files,headers=headers,timeout = 3)
                self.AllImagesId.append(response.json()['files'][0]['name'])
                print(response.json()['files'][0]['name'])
            except Exception as e:
                print('---------------------------------------')
                print('---------------------------------------')
                print('---------------------------------------')
                print(e)
            self.checkingEnded.emit(False)
            self.reloadCameraShow.emit(1)
            print('-------------------------------')
            # print(response.json()['files'][0]['name'])
        except:
            pass
        

    def run(self):
        poses=['ID']
        cap = cv2.VideoCapture(0)
        
        ret, sample_frame = cap.read()
        if ret==False:
            return    
            
        # Introduce pose estimator to solve pose. Get one frame to setup the
        # estimator according to the image size.
        height, width = sample_frame.shape[:2]
        
        images_saved_per_pose=0
        number_of_images = 0
        
       
        
        pose_index = 0
        count = 0  
            
        
        images_saved_per_pose=0
        number_of_images = 0
        
        while pose_index<2:
            saveit = False
            # Read frame, crop it, flip it, suits your needs.
            ret, frame = cap.read()
            if ret is False:
                break
            if count % 10 !=0: # skip 10 frames
                count+=1
                continue
            if images_saved_per_pose==IMAGE_PER_PIC:
                pose_index+=1
                images_saved_per_pose=0

            # If frame comes from webcam, flip it so it looks like a mirror.
            frame = cv2.flip(frame, 2)

            
            frame_for_cam=frame.copy()
            original_frame=frame.copy()
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Show the image


            # here responsible for show image
            height, width = frame.shape[:2] 
            
            scale_percent = 59 # percent of original size
            widthNew = int(frame_for_cam.shape[1] * scale_percent / 100)
            heightNew = int(frame_for_cam.shape[0] * scale_percent / 100)
            dimOld = (widthNew, heightNew)
            frame_for_cam = cv2.resize(frame_for_cam, dimOld, interpolation = cv2.INTER_AREA)

            rgbImage = cv2.cvtColor(frame_for_cam, cv2.COLOR_BGR2RGB)
            h, w, ch = rgbImage.shape
            bytesPerLine = ch * w

            convertToQtFormat = QImage(rgbImage.data, w, h, bytesPerLine, QImage.Format_RGB888)
            self.changePixmap.emit(convertToQtFormat)
                
            # end of show image

            #
            
            if pose_index==0:
                images_saved_per_pose+=1
                self.CurrentImage = 0
                self.saveImage(poses[pose_index],images_saved_per_pose,frame)
                saveit = True
                      
            # Show preview.
            if pose_index>=self.FinalImage:
                self.setStringData.emit('Thank you')
                break

            # frame = cv2.rectangle(frame, (x1,y1), (x2,y2),(255,255,0),2)

            self.setStringData.emit('Show Your '+str(poses[pose_index] +' : '+ str(images_saved_per_pose)+'/'+str(IMAGE_PER_PIC)))
            # self.setPose.emit('Look '+str(poses[pose_index]))
             
                
                
                        
                
        cap.release()
        self.setStringData.emit('click Next')
    
            
# Main Camera for video exam
Blur_Threshold=125
Dark_Threshold=75

class ThreadCameraVideo(QThread):
    changePixmap = pyqtSignal(QImage)
    changeStrLight = pyqtSignal(str)
    changeStrTime = pyqtSignal(str)
    
    audio_thread = None
    frame_counts = 0
    start_time = time.time()
    ThreadCameraVideoIsRun = True
    

    
    
    def getUnique(self):
        t = datetime.datetime.now()
        dateRand = (t-datetime.datetime(1970,1,1)).total_seconds()
        return int(math.floor(random.randint(33333, 999999)) + dateRand)


    
    def run(self):
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        print("Start Cap....................")
        self.NameOfFile = str(self.getUnique())+'.mp4'
        self.NameOfFileScreen = str(self.getUnique())+'.mp4'
        self.PathOfFile = tempfile.gettempdir()+"\\"+self.NameOfFile
        self.PathOfFileUploaded = tempfile.gettempdir()+"\\"+str(self.getUnique())+'.mp4'
        self.PathNameOfFileScreen = tempfile.gettempdir()+"\\"+self.NameOfFileScreen
        fourcc = cv2.VideoWriter_fourcc(*'H264')
       
        if int(self.cap.get(cv2.CAP_PROP_FPS)) == 0:
            self.fps = 13
        else:
            self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))*0.5
        print("--------------------------------------")
        print("FPS: ",self.fps)
        self.out = cv2.VideoWriter()
        self.out.open(self.PathOfFile, fourcc,  self.fps, (250, 250),True)
        self.filenameWav = tempfile.gettempdir()+"\\"+str(self.getUnique())+".wav"
        self.audio_thread = AudioRecorder(filename=self.filenameWav, rate=int(self.fps*1024), fpb=1024, channels=2)

        
        # display screen resolution, get it from your OS settings
        
        SCREEN_SIZE = pyautogui.size()
        # define the codec
        fourcc2 = cv2.VideoWriter_fourcc(*'H264')
        # create the video write object
        self.outScreen = cv2.VideoWriter()
        self.outScreen.open(self.PathNameOfFileScreen, fourcc2, self.fps, (250,250), True)
        self.audio_thread.stream.start_stream()
        
        count = 0
        self.frame_counts = 1
        self.start_time = time.time()
        audio_frames = []
        if (self.audio_thread.open == True):
            try:
                self.lastdata = self.audio_thread.stream.read(self.audio_thread.frames_per_buffer, exception_on_overflow = False) 
            except:
                self.lastdata = []
            audio_frames.append(self.lastdata)
            waveFile = wave.open(self.filenameWav, 'wb')
            waveFile.setnchannels(self.audio_thread.channels)
            waveFile.setsampwidth(self.audio_thread.audio.get_sample_size(self.audio_thread.format))
            waveFile.setframerate(self.audio_thread.rate)
            waveFile.writeframes(b''.join(audio_frames))
            waveFile.close()
        while self.ThreadCameraVideoIsRun:
            
            imgScreen = pyautogui.screenshot()
            # convert these pixels to a proper numpy array to work with OpenCV
            frameScreen = np.array(imgScreen)
            # convert colors from BGR to RGB
            frameScreen = cv2.cvtColor(frameScreen, cv2.COLOR_BGR2RGB)
            dimOld = (250, 250)
            frameScreen = cv2.resize(frameScreen, dimOld, interpolation = cv2.INTER_AREA)
            # write the frame
            self.outScreen.write(frameScreen)
            # write the audio
            
            # if not self.audio_thread.open:
            #     break
            ret, sample_frame = self.cap.read()
            count+= 1
            if ret:
                self.textLight = ''
                if count % IMAGE_PER_POSE==0:
                    gray = cv2.cvtColor(sample_frame, cv2.COLOR_BGR2GRAY)
                    fm = cv2.Laplacian(gray, cv2.CV_64F).var()
                    
                
                    fm2=np.mean(sample_frame)
                    # print(fm2)
                    if fm2 < Dark_Threshold:
                        self.textLight = "Room too dark"
                        self.changeStrLight.emit(self.textLight)
                    elif fm2 > Dark_Threshold+50:
                        self.textLight="Low the light"
                        self.changeStrLight.emit(self.textLight)
                    else:
                        self.textLight=""
                        self.changeStrLight.emit(self.textLight)

                    

                self.changeStrTime.emit("Accept")
                frame = cv2.flip(sample_frame, 2)
                # width  = self.cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)  # float
                # height = self.cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT) # float
                dim = (250, 250)
                frame = cv2.resize(frame, dim) 
                audio_frames = []
                if (self.audio_thread.open == True):
                    try:
                        data = self.audio_thread.stream.read(self.audio_thread.frames_per_buffer, exception_on_overflow = False) 
                        self.lastdata = data
                    except:
                        pass
                    audio_frames.append(self.lastdata)
                    
                    waveFile = wave.open(tempfile.gettempdir()+"\\tempsound.wav", 'wb')
                    waveFile.setnchannels(self.audio_thread.channels)
                    waveFile.setsampwidth(self.audio_thread.audio.get_sample_size(self.audio_thread.format))
                    waveFile.setframerate(self.audio_thread.rate)
                    waveFile.writeframes(b''.join(audio_frames))
                    waveFile.close()

                    sound1 = AudioSegment.from_wav(self.filenameWav)
                    sound2 = AudioSegment.from_wav(tempfile.gettempdir()+"\\tempsound.wav")

                    combined_sounds = sound1 + sound2
                    # print(len(combined_sounds)) 
                    combined_sounds.export(self.filenameWav, format="wav")
                    if not self.audio_thread.open:
                        break
                self.out.write(frame)
                rgbImage = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgbImage.shape
                bytesPerLine = ch * w
                convertToQtFormat = QImage(rgbImage.data, w, h, bytesPerLine, QImage.Format_RGB888)
                self.changePixmap.emit(convertToQtFormat)
                self.frame_counts += 1

        self.audio_thread.stop()
        self.cap.release()
        self.out.release()
        self.outScreen.release()
   


REG_PATH = r"SOFTWARE\Proctoring"

def set_reg(PATH,name, value):
    try:
        CreateKey(HKEY_CURRENT_USER, PATH)
        registry_key = OpenKey(HKEY_CURRENT_USER, PATH, 0, 
                                       KEY_WRITE)
        SetValueEx(registry_key, name, 0, REG_SZ, value)
        CloseKey(registry_key)
        return True
    except WindowsError as e:
        print(e)
        return False


def get_reg(PATH,name):
    try:
        registry_key = OpenKey(HKEY_CURRENT_USER, PATH, 0,
                                       KEY_READ)
        value, regtype = QueryValueEx(registry_key, name)
        CloseKey(registry_key)
        return value
    except WindowsError:
        return None



# 
# print(get_reg('GUID'))



if __name__ == '__main__':
    
    

    print("-----------------------------------------------")
    print(dir_path)
    #os.path.dirname(os.path.realpath(__file__))
    # print(dir_path)
    set_reg(r"Software\\Classes\\Proctoring\\",'URL Protocol', '')
    set_reg(r"Software\\Classes\\Proctoring\\",'', 'Proctoring')
    # set_reg(r"Software\\Classes\\Proctoring\\",'Path', '\"C:\\Users\\AhmedDakrory\\Desktop\\ProctoringApp\\ProctoringApp\\target\\Proctoring\\"')
    # set_reg(r"Software\\Classes\\Proctoring\\Shell\\Open\\command",'', '\"'+dir_path+'\Proctoring.exe '+dir_path+'\\"')
    # set_reg(r"Software\\Classes\\Proctoring\\Shell\\Open\\command",'', '\"C:\\Users\\AhmedDakrory\\Desktop\\ProctoringApp\\ProctoringApp\\target\\Proctoring\\Proctoring.exe\"  "%C:\\Users\\AhmedDakrory\\Desktop\\ProctoringApp\\ProctoringApp\\target\\Proctoring"')
    set_reg(r"Software\\Classes\\Proctoring\\Shell\\Open\\command",'', '\"'+dir_path+'\\Proctoring.exe\"  "%0" "%1" "%2')
    
    runTheApp = False
    # token = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpZCI6NywiaXNzIjoiQXBwIiwiaWF0IjoxNjA5NDkxMDA5NTQwLCJleHAiOjE2MDk0OTM2MDE1NDB9.3U59e1CSZ2gaoBQtE4RIYDDLTgX05jMMM8UNqCHtc9o' #None
    # examId = '43'
    try:
        argumentData = sys.argv[1]
        token = argumentData.split("@/@")[1]
        examId = argumentData.split("@/@")[2]
        print("------------------------------------------------------------------------")
        print(token)
        print(examId)
        runTheApp = True
    except:
        print("Error")
        pass
    
    if runTheApp:
        appctxt = ApplicationContext()       # 1. Instantiate ApplicationContext
        mainApp = GUI(appctxt)
        sys.exit(mainApp.exit_code)
    