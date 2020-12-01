# python -m fbs freeze --debug
# fbs freeze
from fbs_runtime.application_context.PyQt5 import ApplicationContext
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5 import uic

import sys
import os
if getattr(sys, 'frozen', False):
    # If the application is run as a bundle, the PyInstaller bootloader
    # extends the sys module by a flag frozen=True and sets the app 
    # path into variable _MEIPASS'.
    dir_path = sys._MEIPASS
else:
    dir_path = os.path.dirname(os.path.abspath(__file__))

class GUI(QMainWindow):
    def __init__(self,appctxt):
        
        self.dir_path = dir_path
        super(GUI,self).__init__()
        
        uic.loadUi(self.dir_path+'\progressUploading.ui', self) # Load the .ui file
        self.setWindowFlags(Qt.WindowStaysOnTopHint | Qt.FramelessWindowHint)
        
        self.appctxt = appctxt
        
        self.show()
        self.exit_code = self.appctxt.app.exec_()      # 2. Invoke appctxt.app.exec_()
        



if __name__ == '__main__':
    
    

    
    runTheApp = True
    token = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpZCI6NCwiaXNzIjoiQXBwIiwiaWF0IjoxNjA0NTE0NDUyODUzLCJleHAiOjE2MDQ1MTcwNDQ4NTN9._RWMR0eEkecD8HqEjaDAdWLVNpUq2avg1iG6wu9-yis' #None
    examId = 'dc5ab342f6a0d3e488bb5d7be33c921c'
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
    