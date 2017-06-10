from pyAudioAnalysis import audioSegmentation as aS
from pydub import AudioSegment
from os import path
        
import os
import sys
        
modelPath = sys.argv[1]         # Command line argument 1 is the path of the final model.
filePath = sys.argv[2]          # Command line argument 2 is the path of the audio file.

stepSize = sys.argv[3]

'''
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
'''
[flagsInd, classesAll, acc, CM] = aS.mtFileClassification(filePath, modelPath, 'svm', False)

'''
if FLAG_CONVERT_SR:
    print >> sys.stderr, "Removing tmp.wav"
    os.remove('tmp.wav')
'''

resStr = ''

FLAG_FIRST = True
FLAG_V = None

prevChangeIndex = 0
currentTimeIndex = 0

def changeToMin(blockIndex):
    return "{0:.2f}".format(float(blockIndex)*float(stepSize)/(60.00))

for chunkRes in flagsInd:
    
    if FLAG_FIRST:
        if chunkRes == 0:
            FLAG_V = True
        else:
            FLAG_V = False
        
        FLAG_FIRST = False
        
    else: 
        if FLAG_V and (chunkRes == 1):
            resStr += str(changeToMin(prevChangeIndex)) + ' - ' +str(changeToMin(currentTimeIndex)) + ' : Voice, '
            prevChangeIndex = currentTimeIndex
            
            FLAG_V = False
            
        elif (FLAG_V == False) and (chunkRes == 0):
            resStr += str(changeToMin(prevChangeIndex)) + ' - ' +str(changeToMin(currentTimeIndex)) + ' : Noise, '
            prevChangeIndex = currentTimeIndex
            
            FLAG_V = True
            
        if currentTimeIndex == len(flagsInd) - 1 :
            if FLAG_V:
                resStr += str(changeToMin(prevChangeIndex)) + ' - ' +str(changeToMin(currentTimeIndex + 1)) + ' : Voice'
            else:
                resStr += str(changeToMin(prevChangeIndex)) + ' - ' +str(changeToMin(currentTimeIndex + 1)) + ' : Noise'        
        
    currentTimeIndex += 1
            
resStr += '\n'
    
print resStr