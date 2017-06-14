from pyAudioAnalysis import audioSegmentation as aS
from pydub import AudioSegment
from os import path
        
import os
import sys
import wave
        
modelPath = sys.argv[1]         # Command line argument 1 is the path of the final model.
filePath = sys.argv[2]          # Command line argument 2 is the path of the audio file.


FLAG_CONVERT_SR = False

print >> sys.stderr, 'Processing '+filePath.split("/")[-1]+'...'

try:
    sr = wave.openfp(filePath, 'r').getframerate()
    if sr > 48000:
        FLAG_CONVERT_SR = True
except:
    FLAG_CONVERT_SR =True

if FLAG_CONVERT_SR:
    print >> sys.stderr, "Sampling rate too large, converting to 48K as tmp.wav..."
    os.system('ffmpeg -i ' + filePath + ' -ar 48000 tmp.wav')
    filePath = 'tmp.wav'

[flagsInd, classesAll, acc, CM] = aS.mtFileClassification(filePath, modelPath, 'svm', False)


if FLAG_CONVERT_SR:
    print >> sys.stderr, "Removing tmp.wav"
    os.remove('tmp.wav')


resStr = ''


for chunkRes in flagsInd:
    resStr += str(int(chunkRes))+" "

resStr += '\n'
    
print resStr