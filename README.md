<img src="https://i.imgur.com/kN5MDen.png" height="400" align="right">

# Overview
A machine learning based Text to Speech program with a user friendly GUI. Target audience include Twitch streamers or content creators looking for an open source TTS program. The aim of this software is to make tts synthesis accessible offline (No coding experience, gpu/colab) in a portable exe.

## Features
  * Reads donations from Stream Elements automatically
  * PyQt5 wrapper for dc_tts

## Download link 
A portable executable can be found at the [Releases](https://github.com/lokkelvin2/dc_tts_GUI/releases) page, or directly [here](https://github.com/lokkelvin2/dc_tts_GUI/releases/download/v1.01/gui-v.1.01-windows_x86_64.exe). Download a pretrained model separately (from below) to start playing with text to speech.

Warning: the portable executable runs on CPU which leads to a >10x speed slowdown compared to running it on GPU. I might consider other faster models in the future for CPU inference.

## Pretrained Model 
A pretrained model for dc_tts is available from Kyubyong's repo or directly [here](https://www.dropbox.com/s/1oyipstjxh2n5wo/LJ_logdir.tar?dl=0). Kyubyong also provides pretrained models for 10 different languages from the [CSS10 dataset](https://github.com/Kyubyong/css10/). Of course, you are encouraged to try building your own custom voices to use with this GUI. 

## Todo
- [x] Pygame mixer instead of sounddevice
- [x] PyQt threading
- [x] Package into portable executable (cx_freeze/pyinstaller)
- [ ] pyqt instead of pygame volume control
- [ ] Websockets 
- [ ] Add neural vocoder (Waveglow?) instead of griffin-lim
- [ ] Phoneme support with seq2seq model or espeak
- [ ] Make a tutorial page
- [ ] Add streamlabs support

# Building from source
## Requirements
  * Python >=3.7
  * librosa
  * numpy
  * PyQt5==5.15.0
  * requests
  * tensorflow>=1.13.0,<2.0.0
  * tqdm
  * matplotlib
  * scipy
  * num2words
  * pygame

## To Run
``` 
python gui.py
```


## To train custom voices (transfer learning)
The training steps are slightly modified from kyubyong to fix [#11](https://github.com/Kyubyong/dc_tts/issues/11). The training data is in the format of LJ Speech dataset and the expected folder structure is 
```
.
└── data
    ├── wavs
    │   ├── data1.wav
    │   └── data2.wav
    └── transcript.csv
```
### Steps
1. Use 22050Hz, 16 bit signed PCM wav files. Other formats are untested. 
2. Create a csv transcript in the metadata convention of LJ Speech dataset and save it in the folder structure shown above.
3. Extract the two folders in the pretrained model. Edit hyperparams.py to point to the location of the folders.
3. Run `python prepro.py`
4. Run `python train.py 1` to train Text2Mel 
5. Run `python train.py 2` to train SSRN
And you're done! You can load the model using the GUI to perform synthesis.


## License
* dc_tts: Apache License v2.0

## Notes
  * TTS code by Kyubyong: [https://github.com/Kyubyong/dc_tts](https://github.com/Kyubyong/dc_tts)
  * Partial GUI code from [https://github.com/CorentinJ/Real-Time-Voice-Cloning](https://github.com/CorentinJ/Real-Time-Voice-Cloning) and layout inspired by u/realstreamer's Forsen TTS [https://www.youtube.com/watch?v=kL2tglbcDCo](https://www.youtube.com/watch?v=kL2tglbcDCo)
  