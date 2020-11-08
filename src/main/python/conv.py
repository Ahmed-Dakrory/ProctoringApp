print("Start")

import subprocess
import os
audioCommand = subprocess.Popen("ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 1604994334.wav", shell=False, stdout=subprocess.PIPE)
subprocess_return = audioCommand.stdout.read()

timeOfAudio = str(subprocess_return)[2:len(str(subprocess_return))-5]

videoCommand = subprocess.Popen("ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 1605138330.mp4", shell=False, stdout=subprocess.PIPE)
subprocess_return = videoCommand.stdout.read()

timeOfVideo = str(subprocess_return)[2:len(str(subprocess_return))-5]



print(timeOfAudio," , ",timeOfVideo )
os.system('cmd /k "ffmpeg  -y -ac 2 -channel_layout stereo -i 1605138330.mp4 -i 1604994334.wav -filter_complex "[0:v]setpts=PTS*0.905*'+timeOfAudio+'/'+timeOfVideo+'[v]" -map "[v]" -map "1:a" -vcodec libvpx-vp9  output.mp4"')
# videoCommand = subprocess.Popen('', shell=False, stdout=subprocess.PIPE)
