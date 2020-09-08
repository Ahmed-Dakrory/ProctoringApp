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
import os
import sys
import requests  # pip install requests
import math
import base64
import random
import datetime

def getUnique(fileSize):
    t = datetime.datetime.now()
    dateRand = (t-datetime.datetime(1970,1,1)).total_seconds()
    return int(math.floor(random.randint(33333, 999999)) + dateRand + fileSize)

filename = '1100011002.avi'
chunksize = 100000
totalsize = os.path.getsize(filename)
totalChucks = math.ceil(totalsize/chunksize)
readsofar = 0
url = "http://34.243.127.227:3001/api/upload/video/b15ef273fc0b1066c8710d4f16c7533b"
token = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpZCI6MzMsImlzcyI6IkFwcCIsImlhdCI6MTU5OTM0NjA0OTQyMSwiZXhwIjoxNTk5MzQ4NjQxNDIxfQ.3F9nacELyWSGKpG1gkD0m2veNEtd3w_txkxKZQfCK3s'
i = 0
uniqueId = getUnique(totalsize)
with open(filename, 'rb') as file:
    while True:
        data = file.read(chunksize)
        f = open("fileDownload", "wb")
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

      
        files = {'file': ('fileDownload',open('fileDownload', 'rb'),'application/octet-stream')}
        try:
            r = requests.request('POST',url,files=files,headers=headers, verify=False)
            # print(r.text)
            
            i+=1
        except Exception as exc:
            print(exc)
        
        print("\r{percent:3.0f}%".format(percent=percent))
        

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