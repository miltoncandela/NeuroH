from pocketsphinx import LiveSpeech, AudioFile
import os

'''
print(os.getcwd())
file_name = "data/HealthA01197730_IntegrativeMedicineVideo.mp4"
for phrase in AudioFile(audio_file=file_name):
    print(phrase)
'''

for phrase in LiveSpeech():
    print(phrase)
