# python -m fbs freeze --debug
# fbs freeze
from fbs_runtime.application_context.PyQt5 import ApplicationContext
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5 import uic

import sys
import cv2
import numpy as np
import requests
import base64

import threading

from winreg import *
import psutil
import math

import os

import cookies as browser_cookie3

################################
# Those for the Devices Checking
import pyaudio
import subprocess
import re
import wmi
import urllib
import ctypes
from ctypes import *

kernel32 = ctypes.WinDLL('kernel32')
user32 = ctypes.WinDLL('user32')


import dlib
from imutils import face_utils

face_landmark_path = 'shape_predictor_68_face_landmarks.dat'

K = [6.5308391993466671e+002, 0.0, 3.1950000000000000e+002,
     0.0, 6.5308391993466671e+002, 2.3950000000000000e+002,
     0.0, 0.0, 1.0]
D = [7.0834633684407095e-002, 6.9140193737175351e-002, 0.0, 0.0, -1.3073460323689292e+000]

cam_matrix = np.array(K).reshape(3, 3).astype(np.float32)
dist_coeffs = np.array(D).reshape(5, 1).astype(np.float32)

object_pts = np.float32([[6.825897, 6.760612, 4.402142],
                         [1.330353, 7.122144, 6.903745],
                         [-1.330353, 7.122144, 6.903745],
                         [-6.825897, 6.760612, 4.402142],
                         [5.311432, 5.485328, 3.987654],
                         [1.789930, 5.393625, 4.413414],
                         [-1.789930, 5.393625, 4.413414],
                         [-5.311432, 5.485328, 3.987654],
                         [2.005628, 1.409845, 6.165652],
                         [-2.005628, 1.409845, 6.165652],
                         [2.774015, -2.080775, 5.048531],
                         [-2.774015, -2.080775, 5.048531],
                         [0.000000, -3.116408, 6.097667],
                         [0.000000, -7.415691, 4.070434]])

reprojectsrc = np.float32([[10.0, 10.0, 10.0],
                           [10.0, 10.0, -10.0],
                           [10.0, -10.0, -10.0],
                           [10.0, -10.0, 10.0],
                           [-10.0, 10.0, 10.0],
                           [-10.0, 10.0, -10.0],
                           [-10.0, -10.0, -10.0],
                           [-10.0, -10.0, 10.0]])

line_pairs = [[0, 1], [1, 2], [2, 3], [3, 0],
              [4, 5], [5, 6], [6, 7], [7, 4],
              [0, 4], [1, 5], [2, 6], [3, 7]]


def get_head_pose(shape):
    image_pts = np.float32([shape[17], shape[21], shape[22], shape[26], shape[36],
                            shape[39], shape[42], shape[45], shape[31], shape[35],
                            shape[48], shape[54], shape[57], shape[8]])

    _, rotation_vec, translation_vec = cv2.solvePnP(object_pts, image_pts, cam_matrix, dist_coeffs)

    reprojectdst, _ = cv2.projectPoints(reprojectsrc, rotation_vec, translation_vec, cam_matrix,
                                        dist_coeffs)

    reprojectdst = tuple(map(tuple, reprojectdst.reshape(8, 2)))

    # calc euler angle
    rotation_mat, _ = cv2.Rodrigues(rotation_vec)
    pose_mat = cv2.hconcat((rotation_mat, translation_vec))
    _, _, _, _, _, _, euler_angle = cv2.decomposeProjectionMatrix(pose_mat)

    return reprojectdst, euler_angle



class GUI(QMainWindow):
    def __init__(self,appctxt):
        try:
            self.dir_path = sys.argv[1:][0] 
        except:
            self.dir_path =os.path.dirname(os.path.realpath(__file__))
        # self.dir_path = sys.argv[1:][0] #os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        super(GUI,self).__init__()
        self.stepNow = 0
        self.checkInternetTrial = 0
        self.appctxt = appctxt
        # User Session Token
        self.token = None
        self.Username = ''
        self.IsVerified = None
        self.examId = None


        self.TestName = ''
        self.TestDuration = ''
        self.AllNotAllowed = None


        uic.loadUi(self.dir_path+'\main.ui', self) # Load the .ui file
        self.setWindowFlags(Qt.FramelessWindowHint)

        
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

        self.success_cam.setVisible(False)
        self.success_mouse.setVisible(False)
        self.success_key.setVisible(False)
        self.success_speaker.setVisible(False)
        self.success_micro.setVisible(False)

        self.predictButton.clicked.connect(self.goNextStep)
        self.exitButton.clicked.connect(self.close)
        # self.minimizeButton.clicked.connect(lambda: self.ReduceWindowAndMove())
        self.minimizeButton.clicked.connect(lambda: self.showMinimized())
        
        self.username.setText(self.Username)
        # print(self.Username)
        self.testname_label.setText(self.TestName)
        self.testduration_label.setText(str(self.TestDuration) )
        self.show()
        self.goNextStep()
        self.exit_code = self.appctxt.app.exec_()      # 2. Invoke appctxt.app.exec_()
        
        
    def ReduceWindowAndMove(self):
        
        headers = {'authorization': "Bearer "+str(self.token)}
        dataNew = {"status": True}
        UrlPostData = 'http://34.243.127.227:3001/api/test/allow-test-student/'+self.examId
        response = requests.put(UrlPostData,json=dataNew,headers=headers)
        self.message = response.json()['message']
        print(self.message)
        print(response.text)
        if self.message == 'Done':
            self.hide()
            sizeObject = QDesktopWidget().screenGeometry(-1)
            self.ex = QMainWindow()
            uic.loadUi(self.dir_path+'\mainCamera.ui', self.ex) # Load the .ui file

            self.ex.setAttribute(Qt.WA_TranslucentBackground)
            self.ex.setWindowFlags(Qt.FramelessWindowHint)


            self.ex.show()
            self.ex.move(sizeObject.width()-250,sizeObject.height()-300)
            th = ThreadCameraVideo(self.ex)
            th.changePixmap.connect(self.setImageVideo)
            th.start()
        else:
            print('Error')


    def close(self):
        QCoreApplication.exit(0)

    def goToErrorPage(self):
        # self.app = QApplication([])
        self.error = QMainWindow()
        self.error.setAttribute(Qt.WA_TranslucentBackground)
        uic.loadUi(self.dir_path+'\error.ui', self.error) # Load the .ui file
        self.error.errorLabel.setText("Error in the internet")
        
        self.error.setWindowFlags(Qt.FramelessWindowHint)
        self.error.okError.clicked.connect(self.restart)
        self.error.show()

    

    def goNextStep(self):
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
        self.pagerWindow.setCurrentIndex(self.stepNow)
        if self.stepNow == 0:
            print("Checking Internet...")

            loop = QEventLoop()
            QTimer.singleShot(1000, loop.quit)
            loop.exec_()
            
            
            # self.checkInternetThread = threading.Thread(target=self.checkInternet, args=())
            # self.checkInternetThread.start()
            self.checkInternet()
            # self.threadpool = QThreadPool()
            # self.threadpool.start(self.checkInternet)
        elif self.stepNow == 1:

            # self.checkCookiesThread = threading.Thread(target=self.checkCookies, args=())
            # self.checkCookiesThread.start()
            
            loop = QEventLoop()
            QTimer.singleShot(1000, loop.quit)
            loop.exec_()

            self.checkCookies()
        elif self.stepNow == 2:
            print("Device Checking...")
            self.CheckDevices()
            if self.checkResult:
                self.stepNow +=1
                self.goNextStep()
            else:
                self.goToErrorPageWebsite("Please Check the Last Devices")
            
        elif self.stepNow == 3:
            self.predict()
        elif self.stepNow == 4:
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
            # self.goNextStep()
            # finished all Steps then we will show a panel to show success or fails

        elif self.stepNow == 5:
            self.ReduceWindowAndMove()
        elif self.stepNow == -1:
            self.ReduceWindowAndMove()


    def checkCookies(self):
        print("Internet Success...")
        
        cookies = browser_cookie3.chrome(domain_name="34.243.127.227")
        print("Get Cookie")
        self.token = None
        for ck in cookies:
            if ck.name == 'token':
                self.token = ck.value

            if ck.name == 'test':
                self.examId = ck.value
                

        dataNew = {"token": self.token}
        UrlPostData = 'http://34.243.127.227:3001/api/user/me'
        
        if self.token != None and self.examId!=None:
            response = requests.post(UrlPostData,json=dataNew)
            self.Username = response.json()['user']['username']
            self.IsVerified = response.json()['user']['active']
            
            headers = {'authorization': "Bearer "+str(self.token)}
            dataNew = {"token": self.token}
            UrlPostData = 'http://34.243.127.227:3001/api/test/test-requirements/'+self.examId
            response = requests.get(UrlPostData,json=dataNew,headers=headers)
            self.TestName = response.json()['test']['name']
            self.TestDuration = str(response.json()['test']['duration']) +' m'
            self.AllNotAllowed = response.json()['testWhiteListApps']
            
            
            self.username.setText('hi, '+self.Username)
            # print(self.Username)
            self.testname_label.setText(self.TestName)
            self.testduration_label.setText((self.TestDuration) )

            loop = QEventLoop()
            QTimer.singleShot(500, loop.quit)
            loop.exec_()
            self.stepNow +=1
            self.goNextStep()
        else:

            self.goToErrorPageWebsite("Please go to the Exam From the Website")


    def goToErrorPageWebsite(self,statment):
        self.hide()
        self.error = QMainWindow()
        self.error.setAttribute(Qt.WA_TranslucentBackground)
        uic.loadUi(self.dir_path+'\error.ui', self.error) # Load the .ui file
        self.error.errorLabel.setText(statment)
        
        self.error.setWindowFlags(Qt.FramelessWindowHint)
        self.error.okError.clicked.connect(self.restart)
        self.error.show()
        
        

    def restart(self):
        self.show()
        self.error.hide()
        self.stepNow
        self.goNextStep()

    def checkInternetConnection(self):
        self.checkInternetTrial+=1
        try:
            requests.get('https://www.google.com/', verify=False,timeout=1).status_code
            return True
        except:
            return False

    def checkInternet(self):
        if self.checkInternetConnection():
            self.stepNow +=1
            self.goNextStep()
        else:
            # check internet to 5 times
            if self.checkInternetTrial < 5:
                self.goNextStep()
            else:
                self.checkInternetTrial = 0
                self.goToErrorPage()
        
        

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
            self.bar_cam.setVisible(True)
            self.success_cam.setVisible(True)
        else:
            self.checkResult = False

        loop = QEventLoop()
        QTimer.singleShot(500, loop.quit)
        loop.exec_()

        # Check Mouse
        wmiService = wmi.WMI()
        self.PointingDevices = wmiService.query("SELECT * FROM Win32_PointingDevice")
        if len(self.PointingDevices)>= 1:
            self.bar_mouse.setVisible(True)
            self.success_mouse.setVisible(True)
        else:
            self.checkResult = False
            
        loop = QEventLoop()
        QTimer.singleShot(500, loop.quit)
        loop.exec_()

        # Check Keyboard
        self.keyboards = wmiService.query("SELECT * FROM Win32_Keyboard")
        if len(self.keyboards) >= 1:
            self.bar_key.setVisible(True)
            self.success_key.setVisible(True)
        else:
            self.checkResult = False
       
        loop = QEventLoop()
        QTimer.singleShot(500, loop.quit)
        loop.exec_()

        if self.checkSpeaker():
            self.bar_speaker.setVisible(True)
            self.success_speaker.setVisible(True)
        else:
            self.checkResult = False
        
        loop = QEventLoop()
        QTimer.singleShot(500, loop.quit)
        loop.exec_()


        if self.checkMicrophone():
            self.bar_micro.setVisible(True)
            self.success_micro.setVisible(True)
        else:
            self.checkResult = False
        
        loop = QEventLoop()
        QTimer.singleShot(500, loop.quit)
        loop.exec_()


        self.closeAllBlackList()

        if not self.checkVmWare():
            self.bar_system.setVisible(True)
        else:
            self.checkResult = False
        
        loop = QEventLoop()
        QTimer.singleShot(1000, loop.quit)
        loop.exec_()

    def closeAllBlackList(self):
        # PROCNAME = "notepad.exe"

        for proc in psutil.process_iter():
            # check whether the process name matches
            # print(proc.name())
            for program in self.AllNotAllowed:
                if proc.name() == program['SystemApp']['serviceName']:
                    proc.kill()

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
        qp.drawRoundedRect(output.rect(), 70, 70)
        qp.end()
        self.cameraHolder.setPixmap(output)
    

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
    def goCheckingForPose(self,statues):
        self.stepNow +=1
        self.goNextStep()

    def predict(self):
        print("Start Prediction")
        # self.camera.startCap()
        self.Thread_Of_Prediction_Is_Run = True
        th = ThreadCamera(self,self.token)
        th.changePixmap.connect(self.setImage)
        th.setPose.connect(self.setCameraPose)
        th.setBoolStateFace.connect(self.setBoolImageFace)
        th.checkingEnded.connect(self.goCheckingForPose)
        th.start()

        

class ThreadCamera(QThread):
    changePixmap = pyqtSignal(QImage)
    setPose = pyqtSignal(str)
    setBoolStateFace = pyqtSignal(bool)
    checkingEnded = pyqtSignal(str)
    # Image path 
    image_path = 'image.jpg'
    def __init__(self,window,token):
        super(ThreadCamera,self).__init__(window)
        self.token = token
        self.FinalImage = 5
        self.AllImages = []

    def saveImage(self,direction,image):
        # Save the image to the server with this id
        imencoded = cv2.imencode('.jpg', image)[1]
        fileName = direction+'image.jpg'
        print(fileName)
        files = {'files': (fileName, imencoded.tostring(), 'image/jpeg', {'Expires': '0'})}
        headers = {'authorization': "Bearer "+str(self.token)}
        sendThread = threading.Thread(target=self.sendImage, args=(files,headers,))
        sendThread.start()
        
    def sendImage(self,files,headers):
        try:
            response = requests.post('http://34.243.127.227:3001/api/upload/files',files = files,headers=headers,timeout = 3)
            self.AllImages.append(response.json()['files'][0]['name'])
            # print(response.json()['files'][0]['name'])
        except:
            pass
        self.FinalImage -= 1
        if self.FinalImage == 0:
            print(self.AllImages)
            headers = {'authorization': "Bearer "+str(self.token)}
            dataNew = {"faceImages":self.AllImages}

            UrlPostData = 'http://34.243.127.227:3001/api/user/proctoring-images'
            response = requests.post(UrlPostData,json=dataNew,headers=headers)
            # print(response.text)
        # print(self.FinalImage)
        # print(self.AllImages)

    def run(self):
        try:
            self.dir_path = sys.argv[1:][0] 
        except:
            self.dir_path =os.path.dirname(os.path.realpath(__file__))
         #os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        cap = cv2.VideoCapture(0)
        
        # face_cascade = cv2.CascadeClassifier(self.dir_path+'/haarcascade_frontalface_alt2.xml')
        
        detector = dlib.get_frontal_face_detector()
        # detector = dlib.cnn_face_detection_model_v1("mmod_human_face_detector.dat")

        predictor = dlib.shape_predictor(self.dir_path+"\\"+face_landmark_path)

        pose = ['front','right','left','up','down']
        pose_index = 0
        count = 1
        takePhotoEvery = 30
        while pose_index < 5:
            ret, sample_frame = cap.read()
            if ret:
                frame = cv2.flip(sample_frame, 2)
                # frame = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
                # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                # faces = face_cascade.detectMultiScale(gray, 1.3, 5)
                face_rects = detector(frame, 0)
                
                if len(face_rects) > 0:
                    for i, d in enumerate(face_rects):
                        frame = cv2.rectangle(frame,(d.left(),d.top()),(d.right(), d.bottom()),(255,0,0),2)
                        # print(abs(d.left() - d.right())*(d.top() - d.bottom()))
                        if abs((d.left() - d.right())*(d.top() - d.bottom()))>46564 and abs((d.left() - d.right())*(d.top() - d.bottom()))<99100:
                            frameWithoutRec = frame
                            shape = predictor(frame, face_rects[0])
                            shape = face_utils.shape_to_np(shape)

                            reprojectdst, euler_angle = get_head_pose(shape)
                            
                            ValueEularRightLeft = 9
                            ValueEularUp = -10
                            poseNow = None
                            if euler_angle[0, 0]<0.5 and euler_angle[0, 0]>ValueEularUp:
                                if euler_angle[1, 0]>ValueEularRightLeft:
                                    cv2.putText(frame, "left", (0, 50), cv2.FONT_HERSHEY_SIMPLEX,
                                                0.75, (30, 80, 0), thickness=2)
                                    poseNow = 'left'
                                    
                                elif euler_angle[1, 0]<-ValueEularRightLeft:
                                    cv2.putText(frame, "right", (0, 50), cv2.FONT_HERSHEY_SIMPLEX,
                                                0.75, (30, 80, 0), thickness=2)
                                    poseNow = "right"
                                elif euler_angle[1, 0]>=-ValueEularRightLeft and euler_angle[1, 0]<=ValueEularRightLeft:
                                    cv2.putText(frame, "front", (0, 50), cv2.FONT_HERSHEY_SIMPLEX,
                                                0.75, (30, 80, 0), thickness=2)
                                    poseNow = "front"
                            elif euler_angle[0, 0]<ValueEularUp:
                                cv2.putText(frame, "up", (0, 50), cv2.FONT_HERSHEY_SIMPLEX,
                                                0.75, (30, 80, 0), thickness=2)
                                poseNow = "up"
                            elif euler_angle[0, 0]>0.5:
                                cv2.putText(frame, "down", (0, 50), cv2.FONT_HERSHEY_SIMPLEX,
                                                0.75, (30, 80, 0), thickness=2)
                                poseNow = "down"
                            
                            self.setPose.emit('Look '+pose[pose_index])
                            if poseNow == pose[pose_index]:
                                print(poseNow)
                                self.setBoolStateFace.emit(False)
                                takePhotoEvery -=1
                                if takePhotoEvery == 0:
                                    takePhotoEvery = 20
                                    
                                    count -=1
                                    self.saveImage(pose[pose_index],frameWithoutRec)
                                    if count == 0:
                                        count = 1
                                        pose_index+=1
                            else:
                                self.setBoolStateFace.emit(True)



                
                # Show the image
                
                
                height, width = frame.shape[:2] 
                # frame =frame[int(height/4):int(3/4*height),int(width/3):int(2/3*width)]
                frame =frame[int(0):int(7/8*height),int(width/5):int(4/5*width)]
                
                scale_percent = 55 # percent of original size
                widthNew = int(frame.shape[1] * scale_percent / 100)
                heightNew = int(frame.shape[0] * scale_percent / 100)
                dimOld = (widthNew, heightNew)
                frame = cv2.resize(frame, dimOld, interpolation = cv2.INTER_AREA)

                rgbImage = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgbImage.shape
                bytesPerLine = ch * w

                convertToQtFormat = QImage(rgbImage.data, w, h, bytesPerLine, QImage.Format_RGB888)
                self.changePixmap.emit(convertToQtFormat)

        cap.release()
        self.setPose.emit('click Next')
        self.checkingEnded.emit('Success')
            
class ThreadCameraVideo(QThread):
    changePixmap = pyqtSignal(QImage)

    

    def run(self):
        cap = cv2.VideoCapture(0)
        
        while True:
            ret, sample_frame = cap.read()
            if ret:
                frame = cv2.flip(sample_frame, 2)
                # frame = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
                
                # # Show the image
                # height, width = frame.shape[:2] 
                dim = (250, 250)
                frame = cv2.resize(frame, dim) 
                rgbImage = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgbImage.shape
                bytesPerLine = ch * w
                convertToQtFormat = QImage(rgbImage.data, w, h, bytesPerLine, QImage.Format_RGB888)
                self.changePixmap.emit(convertToQtFormat)

        cap.release()
   


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
    
    try:
        dir_path = sys.argv[1:][0] 
    except:
        dir_path =os.path.dirname(os.path.realpath(__file__))
    #os.path.dirname(os.path.realpath(__file__))
    # print(dir_path)
    set_reg(r"Software\\Classes\\Proctoring\\",'URL Protocol', '')
    set_reg(r"Software\\Classes\\Proctoring\\",'', 'Proctoring')
    # set_reg(r"Software\\Classes\\Proctoring\\",'Path', '\"C:\\Users\\AhmedDakrory\\Desktop\\ProctoringApp\\ProctoringApp\\target\\Proctoring\\"')
    # set_reg(r"Software\\Classes\\Proctoring\\Shell\\Open\\command",'', '\"'+dir_path+'\Proctoring.exe '+dir_path+'\\"')
    # set_reg(r"Software\\Classes\\Proctoring\\Shell\\Open\\command",'', '\"C:\\Users\\AhmedDakrory\\Desktop\\ProctoringApp\\ProctoringApp\\target\\Proctoring\\Proctoring.exe\"  "%C:\\Users\\AhmedDakrory\\Desktop\\ProctoringApp\\ProctoringApp\\target\\Proctoring"')
    set_reg(r"Software\\Classes\\Proctoring\\Shell\\Open\\command",'', '\"'+dir_path+'\\Proctoring.exe\"  "%'+dir_path+'"')
    

    
    appctxt = ApplicationContext()       # 1. Instantiate ApplicationContext
    mainApp = GUI(appctxt)
    sys.exit(mainApp.exit_code)