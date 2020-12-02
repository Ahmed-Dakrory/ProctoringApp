import pyaudio
import soundfile as sf
import wave
from pydub import AudioSegment



audio = pyaudio.PyAudio()
format = pyaudio.paInt16
rate=20480
fpb=1024
channels=2
stream = audio.open(format=format,
                                      channels=channels,
                                      rate=rate,
                                      input=True,
                                      frames_per_buffer = fpb)
stream.start_stream()

i=0
audio_frames = []
data = stream.read(fpb, exception_on_overflow = False)
waveFile = wave.open("ahmed.wav", 'wb')
waveFile.setnchannels(channels)
waveFile.setsampwidth(audio.get_sample_size(format))
waveFile.setframerate(rate)
audio_frames.append(data)
waveFile.writeframes(b''.join(audio_frames))
waveFile.close() 

while i<100:
    i+=1
    audio_frames = []
    data = stream.read(fpb, exception_on_overflow = False)
    waveFile = wave.open("temp.wav", 'wb')
    waveFile.setnchannels(channels)
    waveFile.setsampwidth(audio.get_sample_size(format))
    waveFile.setframerate(rate)
    audio_frames.append(data)
    waveFile.writeframes(b''.join(audio_frames))
    waveFile.close() 

    
    sound1 = AudioSegment.from_wav("ahmed.wav")
    sound2 = AudioSegment.from_wav("temp.wav")

    combined_sounds = sound1 + sound2
    print(len(combined_sounds)) 
    combined_sounds.export("ahmed.wav", format="wav")




stream.stop_stream()
stream.close()
audio.terminate()
