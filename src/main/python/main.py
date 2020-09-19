# python -m fbs freeze --debug
# fbs freeze
from fbs_runtime.application_context.PyQt5 import ApplicationContext
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5 import uic
import tempfile

import datetime
import sys
import cv2
import numpy as np
import pyautogui
import requests
import base64
import random

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



# from imutils import face_utils
from imutils.face_utils import FaceAligner

from mark_detector import MarkDetector
from pose_estimator import PoseEstimator
from stabilizer import Stabilizer

import dlib
from imutils import face_utils
import keyboard

#######################################################
#folder to store face images
facepath = 'images'
CNN_INPUT_SIZE = 128
ANGLE_THRESHOLD = 0.15
IMAGE_PER_POSE=10
FACE_WIDTH = 160




mark_detector = MarkDetector() 
try:
    dir_path = sys.argv[1:][0] 
except:
    if getattr(sys, 'frozen', False):
        dir_path = os.path.dirname(sys.executable)
    elif __file__:
        dir_path = os.path.dirname(__file__)
        
shape_predictor = dlib.shape_predictor(dir_path+'/shape_predictor_68_face_landmarks.dat')
face_aligner = FaceAligner(shape_predictor, desiredFaceWidth=FACE_WIDTH)

keyboard.add_hotkey("alt + f4", lambda: None, suppress =True)
keyboard.add_hotkey("ctrl + c", lambda: None, suppress =True)
keyboard.add_hotkey("shift + f10", lambda: None, suppress =True)



class GUI(QMainWindow):
    def __init__(self,appctxt):
        try:
            self.dir_path = sys.argv[1:][0] 
        except:
            if getattr(sys, 'frozen', False):
                self.dir_path = os.path.dirname(sys.executable)
            elif __file__:
                self.dir_path = os.path.dirname(__file__)
            # self.dir_path =os.path.dirname(os.path.realpath(__file__))
        # self.dir_path = sys.argv[1:][0] #os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        super(GUI,self).__init__()
        
        uic.loadUi(self.dir_path+'\main.ui', self) # Load the .ui file
        self.setWindowFlags(Qt.WindowStaysOnTopHint | Qt.FramelessWindowHint)
        self.GUIPanel = self
        self.stepNow = 0
        self.appctxt = appctxt
        # User Session Token
        self.token = None
        self.Username = ''
        self.IsVerified = None
        self.examId = None

        self.oldPos = self.pos()

        self.TestName = ''
        self.TestDuration = ''
        self.AllNotAllowed = None

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
        # self.minimizeButton.clicked.connect(lambda: self.ReduceWindowAndMove())
        self.minimizeButton.clicked.connect(lambda: self.showMinimized())
        
        self.username.setText(self.Username)
        # print(self.Username)
        self.testname_label.setText(self.TestName)
        self.testduration_label.setText(str(self.TestDuration) )
        self.show()
        self.goNextStep()
        self.exit_code = self.appctxt.app.exec_()      # 2. Invoke appctxt.app.exec_()
        


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
        UrlPostData = 'http://34.243.127.227:3001/api/test/allow-test-student/'+self.examId
        response = requests.put(UrlPostData,json=dataNew,headers=headers)
        self.message = response.json()['message']
        print(self.message)
        print(response.text)
        if self.message == 'Done':
            self.OpenCameraApp()
            
        else:
            print('Error')

    def OpenLoaderUpload(self):
        self.hide()
        sizeObject = QDesktopWidget().screenGeometry(-1)
        self.LoaderUpload = self
        uic.loadUi(self.dir_path+'\progressUploading.ui', self.LoaderUpload) # Load the .ui file

        self.LoaderUpload.setAttribute(Qt.WA_TranslucentBackground)
        self.LoaderUpload.setWindowFlags(Qt.FramelessWindowHint )
        
        self.LoaderUpload.setFixedWidth(1047)

        self.LoaderUpload.show()

        
        # sizeObject = QDesktopWidget().screenGeometry(-1)
        # self.LoaderUpload.move(sizeObject.width(),sizeObject.height())

        self.centerWidgetOnScreen(self.LoaderUpload)
        
        self.LoaderUpload.predictButton.clicked.connect(self.close)
        self.LoaderUpload.username.setText('hi, '+self.Username)
        self.LoaderUpload.testname_label.setText(self.TestName)
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
        self.ThreadUploadingFiles = True
        self.uploadThread = threading.Thread(target=self.upload_fileCamera, args=())
        self.uploadThread.start()

    def centerWidgetOnScreen(self, widget):
        centerPoint = QScreen.availableGeometry(QApplication.primaryScreen()).center()
        fg = widget.frameGeometry()
        fg.moveCenter(centerPoint)
        widget.move(fg.topLeft())

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
        self.th.cap.release()
        self.th.out.release()
        self.th.outScreen.release()
        self.OpenLoaderUpload()
        
        # QCoreApplication.exit(0)

    def getUnique(self,fileSize):
        t = datetime.datetime.now()
        dateRand = (t-datetime.datetime(1970,1,1)).total_seconds()
        return int(math.floor(random.randint(33333, 999999)) + dateRand + fileSize)

    def upload_fileCamera(self):
        filename = self.th.PathOfFile
        chunksize = 10000
        totalsize = os.path.getsize(filename)
        totalChucks = math.ceil(totalsize/chunksize)
        readsofar = 0
        url = "http://34.243.127.227:3001/api/upload/video/"+self.examId
        token = self.token
        i = 0
        uniqueId = self.getUnique(totalsize)
        with open(filename, 'rb') as file:
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
                        try:
                            r = requests.request('POST',url,files=files,headers=headers, verify=False)
                            # print(r.text)
                            isUploaded = True
                            i+=1
                        except Exception as exc:
                            print(exc)
                    self.progressBarValue(int(percent/2))
                    print("\r{percent:3.0f}%".format(percent=percent))
                except:
                    pass

        self.upload_fileScreen()
       

    def upload_fileScreen(self):
        filename = self.th.PathNameOfFileScreen
        chunksize = 10000
        totalsize = os.path.getsize(filename)
        totalChucks = math.ceil(totalsize/chunksize)
        readsofar = 0
        url = "http://34.243.127.227:3001/api/upload/video/"+self.examId
        token = self.token
        i = 0
        uniqueId = self.getUnique(totalsize)
        with open(filename, 'rb') as file:
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
                            # print(r.text)
                            isUploaded = True
                            i+=1
                        except Exception as exc:
                            print(exc)
                    self.progressBarValue(int(percent/2+50))
                    print("\r{percent:3.0f}%".format(percent=percent))
                except:
                    pass
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
        self.ThreadUploadingFiles = False
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

    @pyqtSlot()
    def goToErrorPage(self):
        # self.app = QApplication([])
        print("Error")
        self.error = QMainWindow()
        self.error.setAttribute(Qt.WA_TranslucentBackground)
        uic.loadUi(self.dir_path+'\error.ui', self.error) # Load the .ui file
        self.error.errorLabel.setText("Error in the internet")
        
        self.error.setWindowFlags(Qt.WindowStaysOnTopHint | Qt.FramelessWindowHint)
        self.error.okError.clicked.connect(self.restart)
        
        
        try:
            self.hide()
            self.error.show()
        except:
            print("Error")
            
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

            # self.checkCookies()
        elif self.stepNow == 2: # step 2
            print("Device Checking...")
            # Make a Check for All Devices Thread
            
            self.checkDevicesThread = ThreadDeviceCheckConnection(self)
            self.checkDevicesThread.ShowErrorPanel.connect(self.goToErrorPageWebsite)
            self.checkDevicesThread.GoNextStep.connect(self.goNextStepSlotOutSide)
            self.checkDevicesThread.HandleBlackList.connect(self.closeAllBlackList)
            self.checkDevicesThread.start()


            
            
        elif self.stepNow == 3: # step 3
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
        
        cookies = browser_cookie3.chrome(domain_name="34.243.127.227")
        print("Get Cookie")
        self.token = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpZCI6NCwiaXNzIjoiQXBwIiwiaWF0IjoxNTk3NTc1MjE2MTAzLCJleHAiOjE1OTc1Nzc4MDgxMDN9.aprubfcM0eeH1LqyhWGbmnRzpY503AX7eTce8sX0MiA' #None
        self.examId = 'dc5ab342f6a0d3e488bb5d7be33c921c'
        for ck in cookies:
            if ck.name == 'token':
                self.token = ck.value

            if ck.name == 'test':
                self.examId = ck.value
                

        dataNew = {"token": self.token}
        UrlPostData = 'http://34.243.127.227:3001/api/user/me'
        self.TestDurationInt = '0'
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
            listAllow = list(['SettingSyncHost.exe','MsMpEng.exe','SASrv.exe','unsecapp.exe','AGMService.exe',
                            'AGSService.exe','CAudioFilterAgent64.exe','igfxHK.exe','sihost.exe','SecurityHealthSystray.exe',
                            'SearchFilterHost.exe','WmiPrvSE.exe','fbs.exe','RtkBtManServ.exe','ETDService.exe'
                            'SearchProtocolHost.exe','dllhost.exe','PanGPA.exe','IEMonitor.exe','ETDCtrl.exe',
                            'ctfmon.exe','cmd.exe','chrome.exe','SearchUI.exe','RuntimeBroker.exe','jucheck.exe',
                              'ShellExperienceHost.exe','Code.exe','svchost.exe','taskhostw.exe','Video.UI.exe',
                              'SecurityHealthService.exe','winlogon.exe','sqlwriter.exe','ETDCtrlHelper.exe',
                              'fontdrvhost.exe','backgroundTaskHost.exe','conhost.exe','igfxTray.exe',
                              'python.exe','explorer.exe','svchost.exe','Proctoring.exe'])
            for item in listAllow:
                prog = {'deleted': False, 'SystemApp': {'serviceName': item }}
                self.AllNotAllowed.append(prog)

            # print("----------------------------------")
            # print(self.AllNotAllowed)
            self.TestDurationInt = str(response.json()['test']['duration'])
            
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


    
        
    

    def restart(self):
        self.show()
        self.error.hide()
        self.stepNow
        self.goNextStep()

    

        

    
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
                    proc.kill()
                    print("--------------------------------")
                    print("Kill "+proc.name())
                except:
                    pass

    @pyqtSlot(bool)
    def goNextStepSlotOutSide(self,Accept):
        if Accept:
            self.stepNow +=1
            self.goNextStep()
        else:
            self.goNextStep()

    
    @pyqtSlot(str)
    def goToErrorPageWebsite(self,statment):
        self.hide()
        self.error = QMainWindow()
        self.error.setAttribute(Qt.WA_TranslucentBackground)
        uic.loadUi(self.dir_path+'\error.ui', self.error) # Load the .ui file
        self.error.errorLabel.setText(statment)
        
        self.error.setWindowFlags(Qt.WindowStaysOnTopHint | Qt.FramelessWindowHint)
        self.error.okError.clicked.connect(self.restart)
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
    GoNextStep =pyqtSignal(bool)
    HandleBlackList = pyqtSignal()

    
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


        self.HandleBlackList.emit()

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
            self.GoNextStep.emit(True)
        else:
            self.ShowErrorPanel.emit("Please Check the Last Devices")
        

# Camera For Pose Thread
class ThreadCamera(QThread):
    changePixmap = pyqtSignal(QImage)
    setPose = pyqtSignal(str)
    setBoolStateFace = pyqtSignal(bool)
    checkingEnded = pyqtSignal(str)

    def __init__(self,window,token):
        super(ThreadCamera,self).__init__(window)
        self.token = token
        self.FinalImage = 5
        self.AllImages = []

    def saveImage(self,direction,count,image):
        
        if count == IMAGE_PER_POSE:
            # Save the image to the server with this id
            imencoded = cv2.imencode('.jpg', image)[1]
            fileName = str(direction)+'image.jpg'
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
        cap = cv2.VideoCapture(0)
        

        poses=['frontal','right','left','up','down']
       
        
        ret, sample_frame = cap.read()
        pose_index = 0
        count = 0
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
                        self.saveImage(poses[pose_index],images_saved_per_pose,frame)  
                        self.setBoolStateFace.emit(False)
                    else:
                        self.setBoolStateFace.emit(True)         
                if pose_index==1:
                    if steady_pose[0][0]>ANGLE_THRESHOLD:
                        images_saved_per_pose+=1
                        self.saveImage(poses[pose_index],images_saved_per_pose,frame)
                        self.setBoolStateFace.emit(False)
                    else:
                        self.setBoolStateFace.emit(True)  
                if pose_index==2:
                    if steady_pose[0][0]<-ANGLE_THRESHOLD:
                        images_saved_per_pose+=1
                        self.saveImage(poses[pose_index],images_saved_per_pose,frame)
                        self.setBoolStateFace.emit(False)
                    else:
                        self.setBoolStateFace.emit(True)  
                if pose_index==3:
                    if steady_pose[0][1]<-ANGLE_THRESHOLD:
                        images_saved_per_pose+=1
                        self.saveImage(poses[pose_index],images_saved_per_pose,frame)
                        self.setBoolStateFace.emit(False)
                    else:
                        self.setBoolStateFace.emit(True)  
                if pose_index==4:
                    if steady_pose[0][1]>ANGLE_THRESHOLD:
                        images_saved_per_pose+=1
                        self.saveImage(poses[pose_index],images_saved_per_pose,frame)
                        self.setBoolStateFace.emit(False)
                    else:
                        self.setBoolStateFace.emit(True)  
                # Show preview.
                if pose_index>=5:
                    self.setPose.emit('Thank you')
                    break

                # frame = cv2.rectangle(frame, (x1,y1), (x2,y2),(255,255,0),2)

            self.setPose.emit('Look '+str(poses[pose_index] +' : '+ str(images_saved_per_pose)+'/'+str(IMAGE_PER_POSE)))
            # self.setPose.emit('Look '+str(poses[pose_index]))
             
                
                
                        
                
        cap.release()
        self.setPose.emit('click Next')
        self.checkingEnded.emit('Success')
    
    
            
# Main Camera for video exam
Blur_Threshold=125
Dark_Threshold=75

class ThreadCameraVideo(QThread):
    changePixmap = pyqtSignal(QImage)
    changeStrLight = pyqtSignal(str)
    changeStrTime = pyqtSignal(str)

    
    
    def getUnique(self):
        t = datetime.datetime.now()
        dateRand = (t-datetime.datetime(1970,1,1)).total_seconds()
        return int(math.floor(random.randint(33333, 999999)) + dateRand)



    def run(self):
        self.cap = cv2.VideoCapture(0)
        try:
            dir_path = sys.argv[1:][0] 
        except:
            dir_path =os.path.dirname(os.path.realpath(__file__))
        self.NameOfFile = str(self.getUnique())+'.avi'
        self.NameOfFileScreen = str(self.getUnique())+'.avi'
        self.PathOfFile = tempfile.gettempdir()+"\\"+self.NameOfFile
        self.PathNameOfFileScreen = tempfile.gettempdir()+"\\"+self.NameOfFileScreen
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.out = cv2.VideoWriter(self.PathOfFile, fourcc, 20.0, (250,250))
        

        # display screen resolution, get it from your OS settings
        SCREEN_SIZE = (250,250) #pyautogui.size()
        # define the codec
        fourcc2 = cv2.VideoWriter_fourcc(*"XVID")
        # create the video write object
        self.outScreen = cv2.VideoWriter(self.PathNameOfFileScreen, fourcc2, 20.0, SCREEN_SIZE)


        count = 0
        while True:
            
            imgScreen = pyautogui.screenshot()
            # convert these pixels to a proper numpy array to work with OpenCV
            frameScreen = np.array(imgScreen)
            # convert colors from BGR to RGB
            frameScreen = cv2.cvtColor(frameScreen, cv2.COLOR_BGR2RGB)
            dimOld = (250, 250)
            frameScreen = cv2.resize(frameScreen, dimOld, interpolation = cv2.INTER_AREA)
            # write the frame
            self.outScreen.write(frameScreen)
            ret, sample_frame = self.cap.read()
            count+= 1
            if ret:
                self.textLight = ''
                if count % IMAGE_PER_POSE==0:
                    gray = cv2.cvtColor(sample_frame, cv2.COLOR_BGR2GRAY)
                    fm = cv2.Laplacian(gray, cv2.CV_64F).var()
                    
                
                    fm2=np.mean(sample_frame)
                    if fm2 < Dark_Threshold:
                        self.textLight = "Room too dark"
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
                self.out.write(frame)
                rgbImage = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgbImage.shape
                bytesPerLine = ch * w
                convertToQtFormat = QImage(rgbImage.data, w, h, bytesPerLine, QImage.Format_RGB888)
                self.changePixmap.emit(convertToQtFormat)

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
    
    try:
        dir_path = sys.argv[1:][0] 
    except:
        dir_path =os.path.dirname(os.path.realpath(__file__))

    print("-----------------------------------------------")
    print(dir_path)
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