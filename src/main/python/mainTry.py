import cv2
import pyaudio
import wave
import threading
import time
import subprocess
import os

class VideoRecorder():  

    # Video class based on openCV 
    def __init__(self):

        self.open = True
        self.device_index = 0
        self.fps = 15               # fps should be the minimum constant rate at which the camera can
        self.frameSize = (640,480) # video formats and sizes also depend and vary according to the camera used
        self.video_filename = "temp_video.mp4"
        self.video_cap = cv2.VideoCapture(self.device_index)
        self.video_writer = cv2.VideoWriter_fourcc(*'H264')
        self.video_out = cv2.VideoWriter(self.video_filename, self.video_writer, self.fps, self.frameSize)
        self.frame_counts = 1
        self.start_time = time.time()


    # Video starts being recorded 
    def record(self):


        while(self.open==True):
            ret, video_frame = self.video_cap.read()
            if (ret==True):

                self.video_out.write(video_frame)
                self.frame_counts += 1
                # time.sleep(0.16)
            else:
                break


    # Finishes the video recording therefore the thread too
    def stop(self):

        if self.open==True:

            self.open=False
            self.video_out.release()
            self.video_cap.release()
            cv2.destroyAllWindows()

        else: 
            pass


    # Launches the video recording function using a thread          
    def start(self):
        video_thread = threading.Thread(target=self.record)
        video_thread.start()





class AudioRecorder():


    # Audio class based on pyAudio and Wave
    def __init__(self):

        self.open = True
        self.rate = 44100
        self.frames_per_buffer = 1024
        self.channels = 2
        self.format = pyaudio.paInt16
        self.audio_filename = "temp_audio.wav"
        self.audio = pyaudio.PyAudio()
        self.stream = self.audio.open(format=self.format,
                                      channels=self.channels,
                                      rate=self.rate,
                                      input=True,
                                      frames_per_buffer = self.frames_per_buffer)
        self.audio_frames = []


    # Audio starts being recorded
    def record(self):

        self.stream.start_stream()
        while(self.open == True):
            data = self.stream.read(self.frames_per_buffer) 
            self.audio_frames.append(data)
            if self.open==False:
                break


    # Finishes the audio recording therefore the thread too    
    def stop(self):

        if self.open==True:
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

        pass

    # Launches the audio recording function using a thread
    def start(self):
        audio_thread = threading.Thread(target=self.record)
        audio_thread.start()





def start_AVrecording(filename):

    global video_thread
    global audio_thread

    video_thread = VideoRecorder()
    audio_thread = AudioRecorder()

    audio_thread.start()
    video_thread.start()

    return filename





def stop_AVrecording(filename):
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

    #    Merging audio and video signal
    if abs(recorded_fps - 15) >= 0.01:    # If the fps rate was higher/lower than expected, re-encode it to the expected

        audioCommand = subprocess.Popen('ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 temp_audio.wav', shell=False, stdout=subprocess.PIPE)
        subprocess_return = audioCommand.stdout.read()

        timeOfAudio = str(subprocess_return)[2:len(str(subprocess_return))-5]

        videoCommand = subprocess.Popen('ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 temp_video.mp4', shell=False, stdout=subprocess.PIPE)
        subprocess_return = videoCommand.stdout.read()

        timeOfVideo = str(subprocess_return)[2:len(str(subprocess_return))-5]

        print("Re-encoding")
        # timescale = 15/recorded_fps
        timescale = float(timeOfAudio)/float(timeOfVideo)
        # cmd = 'ffmpeg -i temp_video.mp4 -vf "setpts=15/'+str(recorded_fps)+'*PTS" -vcodec copy temp_video2.mp4'
        cmd = 'ffmpeg -itsscale '+str(timescale)+' -i temp_video.mp4 -codec copy temp_video2.mp4'
        subprocess.call(cmd, shell=True)

        print("Muxing")
        cmd = "ffmpeg -y -ac 2 -channel_layout stereo -i temp_video2.mp4 -i temp_audio.wav -c:v copy -c:a aac -map 0:v:0 -map 1:a:0 " + filename + ".mp4"
        subprocess.call(cmd, shell=True)

    else:

        print("Normal recording\nMuxing")
        cmd = "ffmpeg -y -ac 2 -channel_layout stereo -i temp_video.mp4 -i temp_audio.wav -vcodec libvpx-vp9 " + filename + ".mp4"
        subprocess.call(cmd, shell=True)

        print("..")




# Required and wanted processing of final files
def file_manager(filename):

    local_path = os.getcwd()

    if os.path.exists(str(local_path) + "/temp_audio.wav"):
        os.remove(str(local_path) + "/temp_audio.wav")

    if os.path.exists(str(local_path) + "/temp_video.mp4"):
        os.remove(str(local_path) + "/temp_video.mp4")

    if os.path.exists(str(local_path) + "/temp_video2.mp4"):
        os.remove(str(local_path) + "/temp_video2.mp4")

    # if os.path.exists(str(local_path) + "/" + filename + ".avi"):
    #     os.remove(str(local_path) + "/" + filename + ".avi")


start_AVrecording("ahm3ed")
time.sleep(10)
stop_AVrecording("ahm3ed")
file_manager("ahm3ed")