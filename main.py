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

import cv2


cap = cv2.VideoCapture(0)
fourcc = cv2.VideoWriter_fourcc(*'FMP4')
width, height = (
            int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        )
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
out = cv2.VideoWriter()
output_file_name = "output_single.mp4"
out.open(output_file_name, fourcc, fps, (width, height), True)
i = 500
while i>0:
    i=i-1
    ret, sample_frame = cap.read()
    if ret:
        out.write(sample_frame)

cap.release()
out.release()
