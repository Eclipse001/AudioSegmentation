# AudioSegmentation

- Main objective: Identifies sub-block of the audio which has different type of content than the part of audio adjacent to that block with time stamps. 

# Tool and package used

- Python 2.7 : https://www.python.org/
- pyAudioAnalysis : https://github.com/tyiannak/pyAudioAnalysis
- ffmpeg

Use pip to install dependencies of pyAudioAnalysis:

- pip install numpy matplotlib scipy sklearn hmmlearn simplejson eyed3 pydub

# Pre-setup steps

- Makesure ffmpeg and the dependencies of pyAudioAnalysis are intalled.
- Download this repo.
- Download the pyAudioAnalysis repo from https://github.com/tyiannak/pyAudioAnalysis and replace the empty folder 'pyAudioAnaysis' inside this repo with the downloaded package.

- Put training sample audio files in the "Samples/Voice" and "Samples/Noise" path (which are the relative path of the root path of this repo).


# Run the training script

- python train.py [midtermWindow] [midtermStep] [voiceFolderPath] [noiseFolderPath] [tVoiceFolderPath] [tNoiseFolderPath]

[midtermWindow] and [midtermStep] are the mid-term block size and step size used while training the classifier from the samples. (In seconds)

[voiceFolderPath] and [noiseFolderPath] are the path of the folders contain the orginal training samples, there must be an ending slash in both of these arguments. For example: 'OrginalSamples/Voice' will cause error but 'OrginalSamples/Voice/' won't cause error.

[tVoiceFolderPath] [tNoiseFolderPath] are the path of the folders will contain the training samples that will actually used for training, that is, the copies of content inside [voiceFolderPath] [noiseFolderPath]. This is to prevent any modification of the original files while reducing the audio file's sampling rate. These two path must be exist before running the training script. Also, files inside these two folders will be removed after training. Same as previous arguments, there must be an ending slash in both of these arguments.

Th resulting svm model as well as its related files will be located in the folder named 'Models', the model file name will be 'svm'. 

- The new trained model will replace the previous one.
- The model type is SVM.

# Run the processing script

- python script.py [modelPath] [filePath]

[modelPath] and [filePath] are the path to the trained model file and the path to the audio file to be processed.

The output will be print to stdout, and it will be a list of zeros and ones, which is the segmentation result. (0 stands for voice, 1 stands for noise) Each single number represents the audio chunk of the model's step size in order.

If an 'unknown format" error message pops up during the process, this is probably due to sampling rate of a audio file is higher than 48K is not supported. In this case, try to uncomment all of the commented code and redo the process.


# Run the processing script with fomratted output

- python script2.py [modelPath] [filePath] [stepSize]

Command line arguments are the same with the older version script except for an extra argument [stepSize] which must be the stepSize of the tranied model. Wrong step size in the command line argument will lead to incorrect output.

The output will be print to stdout, its format will be the same as the following sample output:

- 0.00 - 0.35,0.35 - 3.40

The list of time stamps separate by comma stands for the audio block which is not noise.
The float number's time unit is in minutes, for example 0.50 means 30 sec, 1.5 means 1 min and 30 sec.


