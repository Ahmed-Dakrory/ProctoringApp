# from winreg import *
# import psutil

# REG_PATH = r"SOFTWARE\Proctoring"

# def set_reg(PATH,name, value):
#     try:
#         CreateKey(HKEY_CURRENT_USER, PATH)
#         registry_key = OpenKey(HKEY_CURRENT_USER, PATH, 0, 
#                                        KEY_WRITE)
#         SetValueEx(registry_key, name, 0, REG_SZ, value)
#         CloseKey(registry_key)
#         return True
#     except WindowsError as e:
#         print(e)
#         return False


# def get_reg(PATH,name):
#     try:
#         registry_key = OpenKey(HKEY_CURRENT_USER, PATH, 0,
#                                        KEY_READ)
#         value, regtype = QueryValueEx(registry_key, name)
#         CloseKey(registry_key)
#         return value
#     except WindowsError:
#         return None


# # set_reg('GUID', str('asko98dsjsdjk'))
# set_reg(r"Software\\Classes\\Proctoring\\",'URL Protocol', '')
# set_reg(r"Software\\Classes\\Proctoring\\",'', 'URL:Alert Protocol')




# # set_reg(r"Software\\Classes\\Proctoring\\Shell\\Open\\command",'', '\"D:\\AnyDesk.exe\" \"%1\"')
# # print(get_reg('GUID'))

# PROCNAME = "Taskmgr.exe"

# for proc in psutil.process_iter():
#     # check whether the process name matches
#     print(proc.name())
#     if proc.name() == PROCNAME:
#         proc.kill()

################################################################
# import os
# import sys
# import requests  # pip install requests
# import math
# import base64
# import random
# import datetime

# def getUnique(fileSize):
#     t = datetime.datetime.now()
#     dateRand = (t-datetime.datetime(1970,1,1)).total_seconds()
#     return int(math.floor(random.randint(33333, 999999)) + dateRand + fileSize)

# filename = '1100011002.avi'
# chunksize = 100000
# totalsize = os.path.getsize(filename)
# totalChucks = math.ceil(totalsize/chunksize)
# readsofar = 0
# url = "http://34.243.127.227:3001/api/upload/video/b15ef273fc0b1066c8710d4f16c7533b"
# token = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpZCI6MzMsImlzcyI6IkFwcCIsImlhdCI6MTU5OTM0NjA0OTQyMSwiZXhwIjoxNTk5MzQ4NjQxNDIxfQ.3F9nacELyWSGKpG1gkD0m2veNEtd3w_txkxKZQfCK3s'
# i = 0
# uniqueId = getUnique(totalsize)
# with open(filename, 'rb') as file:
#     while True:
#         data = file.read(chunksize)
#         f = open("fileDownload", "wb")
#         f.write(data)
#         f.close()
#         if not data:
#             sys.stderr.write("\n")
#             break
#         readsofar += len(data)
#         percent = readsofar * 1e2 / totalsize
        
#         headers = {
#             'Access-Control-Max-Age':'86400',
#             'Access-Control-Allow-Methods': 'POST,OPTIONS' ,
#             'Access-Control-Allow-Headers': 'uploader-chunk-number,uploader-chunks-total,uploader-file-id', 
#             'Access-Control-Allow-Origin':'http://localhost:3000',
#             'authorization': "Bearer "+token,
#             'uploader-file-id': str(uniqueId),
#             'uploader-chunks-total': str(totalChucks),
#             'uploader-chunk-number': str(i)
#             }

      
#         files = {'file': ('fileDownload',open('fileDownload', 'rb'),'application/octet-stream')}
#         try:
#             r = requests.request('POST',url,files=files,headers=headers, verify=False)
#             # print(r.text)
            
#             i+=1
#         except Exception as exc:
#             print(exc)
        
#         print("\r{percent:3.0f}%".format(percent=percent))
        

# import subprocess
# import re

# POWER_MGMT_RE = re.compile(r'IOPowerManagement.*{(.*)}')

# def display_status():
#     output = subprocess.check_output(
#         'ioreg -w 0 -c IODisplayWrangler -r IODisplayWrangler'.split())
#     status = POWER_MGMT_RE.search(output).group(1)
#     return dict((k[1:-1], v) for (k, v) in (x.split('=') for x in
#                                             status.split(',')))

# print(display_status())

#########################################################################

# import psutil

# listAllow = list(['dllhost.exe','PanGPA.exe','ctfmon.exe','cmd.exe','chrome.exe','SearchUI.exe',
#                               'ShellExperienceHost.exe','Code.exe','svchost.exe',
#                               'fontdrvhost.exe','backgroundTaskHost.exe','conhost.exe',
#                               'python.exe','explorer.exe','svchost.exe','Proctoring.exe'])
# for proc in psutil.process_iter():
#     if proc.name() not in listAllow:
#         try:
#             proc.kill()
#             print(proc.name())
#         except:
#             print("--------------------------------------------")

# import requests

# url="https://file-examples-com.github.io/uploads/2017/04/file_example_MP4_480_1_5MG.mp4"

# response = requests.get(url, stream = True)

# text_file = open("data.mp4","wb")
# i = 0
# for chunk in response.iter_content(chunk_size=1024):
#     i = i + 1024
#     print(i/1024)
#     text_file.write(chunk)

# text_file.close()

# import requests

# url = "http://54.154.79.104:3001/api/user-test-trial/processing-result"


# privateKey = '160061482862217iooi1kfb8qq1c'
# secretKey = '17iooi1kfb8qq1b'
# trial = 127
# data = [{"time":30,"error":["Left Screen "]}]

# payload = {"errors":data,
#             "trial": trial,
#             "secretKey": secretKey,
#             "privateKey":privateKey}

# headers = {
#     'content-type': "application/json"
#     }

# response = requests.request("POST", url, json=payload, headers=headers)
# print(response.status_code)
# print(response.text)



# OpenCV: FFMPEG: format mp4 / MP4 (MPEG-4 Part 14)
# fourcc tag 0x7634706d/'mp4v' codec_id 000C
# fourcc tag 0x31637661/'avc1' codec_id 001B
# fourcc tag 0x33637661/'avc3' codec_id 001B
# fourcc tag 0x31766568/'hev1' codec_id 00AD
# fourcc tag 0x31637668/'hvc1' codec_id 00AD
# fourcc tag 0x7634706d/'mp4v' codec_id 0002
# fourcc tag 0x7634706d/'mp4v' codec_id 0001
# fourcc tag 0x7634706d/'mp4v' codec_id 0007
# fourcc tag 0x7634706d/'mp4v' codec_id 003D
# fourcc tag 0x7634706d/'mp4v' codec_id 0058
# fourcc tag 0x312d6376/'vc-1' codec_id 0046
# fourcc tag 0x63617264/'drac' codec_id 0074
# fourcc tag 0x7634706d/'mp4v' codec_id 00A3
# fourcc tag 0x39307076/'vp09' codec_id 00A7
# fourcc tag 0x31307661/'av01' codec_id 801D
# fourcc tag 0x6134706d/'mp4a' codec_id 15002
# fourcc tag 0x6134706d/'mp4a' codec_id 1502D
# fourcc tag 0x6134706d/'mp4a' codec_id 15001
# fourcc tag 0x6134706d/'mp4a' codec_id 15000
# fourcc tag 0x332d6361/'ac-3' codec_id 15003
# fourcc tag 0x332d6365/'ec-3' codec_id 15028
# fourcc tag 0x6134706d/'mp4a' codec_id 15004
# fourcc tag 0x43614c66/'fLaC' codec_id 1500C
# fourcc tag 0x7375704f/'Opus' codec_id 1503C
# fourcc tag 0x6134706d/'mp4a' codec_id 15005
# fourcc tag 0x6134706d/'mp4a' codec_id 15018
# fourcc tag 0x6134706d/'mp4a' codec_id 15803
# fourcc tag 0x7334706d/'mp4s' codec_id 17000
# fourcc tag 0x67337874/'tx3g' codec_id 17005
# fourcc tag 0x646d7067/'gpmd' codec_id 18807


#!/usr/bin/env python
# -*- coding: utf-8 -*-
# VideoRecorder.py

from __future__ import print_function, division
import numpy as np
import cv2
import pyaudio
import wave
import threading
import time
import subprocess
import os
import sys

class VideoRecorder():  
    "Video class based on openCV"
    if getattr(sys, 'frozen', False):
        # If the application is run as a bundle, the PyInstaller bootloader
        # extends the sys module by a flag frozen=True and sets the app 
        # path into variable _MEIPASS'.
        dir_path = sys._MEIPASS
    else:
        dir_path = os.path.dirname(os.path.abspath(__file__))
    def __init__(self, name=dir_path+"\\temp_video.avi", fourcc="MJPG", sizex=640, sizey=480, camindex=0, fps=30):
        self.open = True
        self.device_index = camindex
        self.fps = fps                  # fps should be the minimum constant rate at which the camera can
        self.fourcc = fourcc            # capture images (with no decrease in speed over time; testing is required)
        self.frameSize = (sizex, sizey) # video formats and sizes also depend and vary according to the camera used
        self.video_filename = name
        self.video_cap = cv2.VideoCapture(self.device_index)
        self.video_writer = cv2.VideoWriter_fourcc(*self.fourcc)
        self.video_out = cv2.VideoWriter(self.video_filename, self.video_writer, self.fps, self.frameSize)
        self.frame_counts = 1
        self.start_time = time.time()

    def record(self):
        "Video starts being recorded"
        # counter = 1
        timer_start = time.time()
        timer_current = 0
        while self.open:
            ret, video_frame = self.video_cap.read()
            if ret:
                self.video_out.write(video_frame)
                # print(str(counter) + " " + str(self.frame_counts) + " frames written " + str(timer_current))
                self.frame_counts += 1
                # counter += 1
                # timer_current = time.time() - timer_start
                # time.sleep(1/self.fps)
                # gray = cv2.cvtColor(video_frame, cv2.COLOR_BGR2GRAY)
                # cv2.imshow('video_frame', gray)
                # cv2.waitKey(1)
            else:
                break

    def stop(self):
        "Finishes the video recording therefore the thread too"
        if self.open:
            self.open=False
            self.video_out.release()
            self.video_cap.release()
            cv2.destroyAllWindows()

    def start(self):
        "Launches the video recording function using a thread"
        video_thread = threading.Thread(target=self.record)
        video_thread.start()

class AudioRecorder():
    "Audio class based on pyAudio and Wave"
    if getattr(sys, 'frozen', False):
        # If the application is run as a bundle, the PyInstaller bootloader
        # extends the sys module by a flag frozen=True and sets the app 
        # path into variable _MEIPASS'.
        dir_path = sys._MEIPASS
    else:
        dir_path = os.path.dirname(os.path.abspath(__file__))

    def __init__(self, filename=dir_path+"\\temp_audio.wav", rate=44100, fpb=1024, channels=2):
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

    def record(self):
        "Audio starts being recorded"
        self.stream.start_stream()
        while self.open:
            data = self.stream.read(self.frames_per_buffer) 
            self.audio_frames.append(data)
            if not self.open:
                break

    def stop(self):
        "Finishes the audio recording therefore the thread too"
        if self.open:
            self.open = False
            self.stream.stop_stream()
            self.stream.close()
            self.audio.terminate()
            waveFile = wave.open(self.audio_filename, 'wb')
            waveFile.setnchannels(self.channels)
            waveFile.setsampwidth(self.audio.get_sample_size(self.format))
            waveFile.setframerate(self.rate)
            waveFile.writeframes(b''.join(self.audio_frames))
            waveFile.close()

    def start(self):
        "Launches the audio recording function using a thread"
        audio_thread = threading.Thread(target=self.record)
        audio_thread.start()

def start_AVrecording(filename="test"):
    global video_thread
    global audio_thread
    video_thread = VideoRecorder()
    audio_thread = AudioRecorder()
    audio_thread.start()
    video_thread.start()
    return filename



def stop_AVrecording(filename="test"):
    audio_thread.stop() 
    frame_counts = video_thread.frame_counts
    elapsed_time = time.time() - video_thread.start_time
    recorded_fps = frame_counts / elapsed_time
    print("total frames " + str(frame_counts))
    print("elapsed time " + str(elapsed_time))
    print("recorded fps " + str(recorded_fps))
    video_thread.stop() 

    # Makes sure the threads have finished
    while threading.active_count() > 1:
        time.sleep(1)

    # Merging audio and video signal
    # if abs(recorded_fps - 6) >= 0.01:    # If the fps rate was higher/lower than expected, re-encode it to the expected
    #     print("Re-encoding")
    #     cmd = "ffmpeg -i temp_video.avi temp_video2.avi"
    #     subprocess.call(cmd, shell=True)
    #     print("Muxing")
    #     cmd = "ffmpeg -y -ac 2 -channel_layout stereo -i temp_audio.wav -i temp_video2.avi -pix_fmt yuv420p " + filename + ".avi"
    #     subprocess.call(cmd, shell=True)
    # else:
    #     print("Normal recording\nMuxing")
    #     cmd = "ffmpeg -y -ac 2 -channel_layout stereo -i temp_audio.wav -i temp_video.avi -pix_fmt yuv420p " + filename + ".avi"
    #     subprocess.call(cmd, shell=True)
    #     print("..")

    print("OKOKOKOKOK")
    if getattr(sys, 'frozen', False):
        # If the application is run as a bundle, the PyInstaller bootloader
        # extends the sys module by a flag frozen=True and sets the app 
        # path into variable _MEIPASS'.
        dir_path = sys._MEIPASS
    else:
        dir_path = os.path.dirname(os.path.abspath(__file__))

    
    print("-----------------------------")
    print(dir_path)
    # import ffmpeg as FFmpeg
    
    # input_video = FFmpeg.input('temp_video.avi')

    # input_audio = FFmpeg.input('temp_audio.wav')

    # dirrrr = dir_path+'\\ffmpeg.exe'
    # print("-----------------------------")
    # print(dirrrr)
    # ff = FFmpeg(executable=dirrrr)
    # ff.concat(input_video, input_audio, v=1, a=1).output('finished_video.mp4').run()
    # cmd = "ffmpeg -i temp_video.avi temp_video2.avi"
    # subprocess.call(cmd, shell=True)
    # print("Muxing")
    # cmd = "ffmpeg -y -ac 2 -channel_layout stereo -i temp_audio.wav -i temp_video2.avi -pix_fmt yuv420p " + "output" + ".avi"
    # subprocess.call(cmd, shell=True)

    cmd = dir_path+"\\ffmpeg.exe -y -ac 2 -channel_layout stereo -i "+dir_path+"\\temp_audio.wav -i "+dir_path+"\\temp_video.avi -pix_fmt yuv420p " +dir_path+"\\output" + ".avi"
    print(cmd)
    subprocess.call(cmd, shell=True)

def file_manager(filename="test"):
    "Required and wanted processing of final files"
    local_path = os.getcwd()
    if getattr(sys, 'frozen', False):
        # If the application is run as a bundle, the PyInstaller bootloader
        # extends the sys module by a flag frozen=True and sets the app 
        # path into variable _MEIPASS'.
        dir_path = sys._MEIPASS
    else:
        dir_path = os.path.dirname(os.path.abspath(__file__))

    if os.path.exists(str(dir_path) + "\\temp_audio.wav"):
        os.remove(str(dir_path) + "\\temp_audio.wav")
    if os.path.exists(str(dir_path) + "\\temp_video.avi"):
        os.remove(str(dir_path) + "\\temp_video.avi")
    if os.path.exists(str(dir_path) + "\\temp_video2.avi"):
        os.remove(str(dir_path) + "\\temp_video2.avi")
    # if os.path.exists(str(local_path) + "/" + filename + ".avi"):
    #     os.remove(str(local_path) + "/" + filename + ".avi")

if __name__ == '__main__':
    start_AVrecording()
    time.sleep(5)
    stop_AVrecording()
    file_manager()