from pyAudioAnalysis import audioTrainTest as aT
from pyAudioAnalysis import audioSegmentation as aS
from os import path
from shutil import copyfile

import os
import sys
import time
import wave

mtw = float(sys.argv[1])        # Command line argument 1 is the mid-term window.
mts = float(sys.argv[2])        # Command line argument 2 is the mid-term step.

orVDirPath = sys.argv[3]         # Command line argument 3 is the folder containing voice samples.
orNDirPath = sys.argv[4]         # Command line argument 4 is the folder containing noise samples.

vDirPath = sys.argv[5]
nDirPath = sys.argv[6]

def removeFile(dirPath):
    for fileName in os.listdir(dirPath):
        os.remove(dirPath+fileName)


def checkSamplingRate(orDirPath, coDirPath):
    
    for fileName in os.listdir(orDirPath):
        orFullPath = orDirPath + fileName
        rFileName = str(time.time())+'.wav'
        
        if checkFileProp(orFullPath):
            print >> sys.stderr, 'Sampling rate too large, changing to 48K while copying to sample folder.' + rFileName +'...'
            os.system('ffmpeg -i ' + orFullPath + ' -ar 48000 ' + coDirPath + rFileName)
        else:
            print >> sys.stderr, 'Copying to the sample folder as ' + rFileName +'...'
            copyfile(orFullPath, coDirPath + rFileName)

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


checkSamplingRate(orVDirPath, vDirPath)
checkSamplingRate(orNDirPath, nDirPath)


aT.featureAndTrain([vDirPath, nDirPath], mtw, mts, aT.shortTermWindow, aT.shortTermStep, 'svm', "Models/svm")

removeFile(vDirPath)
removeFile(nDirPath)