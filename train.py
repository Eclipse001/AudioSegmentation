from pyAudioAnalysis import audioTrainTest as aT
from pyAudioAnalysis import audioSegmentation as aS
from os import path

import os
import sys
import time
import wave

mtw = float(sys.argv[1])        # Command line argument 1 is the mid-term window.
mts = float(sys.argv[2])        # Command line argument 2 is the mid-term step.

vDirPath = "Samples/Voice/"
nDirPath = "Samples/Noise/"

def checkSamplingRate(dirPath):
    
    for fileName in os.listdir(dirPath):
        fullPath = dirPath + fileName
        
        if checkFileProp(fullPath):
            rFileName = str(time.time())+'.wav'
            
            print >> sys.stderr, 'Sampling rate too large, replacing the file with a copy of 48K Sampling rate named: ' + rFileName
            os.system('ffmpeg -i ' + fullPath + ' -ar 48000 ' + dirPath + rFileName)
            
            print >> sys.stderr, 'Removing orginal file...'
            os.remove(fullPath)

def checkFileProp(fullPath):
    FLAG_CONVERT_SR = False
    
    print >> sys.stderr, 'Checking training set file: '+fullPath.split("/")[-1]+'...'
    
    try:
        sr = wave.openfp(fullPath, 'r').getframerate()
        if sr > 48000:
            FLAG_CONVERT_SR = True
    except:
        FLAG_CONVERT_SR =True
    
    return FLAG_CONVERT_SR

'''
checkSamplingRate(vDirPath)
checkSamplingRate(nDirPath)
'''

aT.featureAndTrain([vDirPath, nDirPath], mtw, mts, aT.shortTermWindow, aT.shortTermStep, 'svm', "Models/svm")
