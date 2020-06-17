# -*- coding: utf-8 -*-
# /usr/bin/python2
'''
By kyubyong park. kbpark.linguist@gmail.com.
https://www.github.com/kyubyong/dc_tts
'''

from __future__ import print_function

import os

from hyperparams import Hyperparams as hp
import numpy as np
import tensorflow as tf
#from train import Graph
from utils import spectrogram2wav
import data_load
from scipy.io.wavfile import write
from tqdm import tqdm
import time
import sounddevice as sd
import pygame

def synthesize(input_text,model_path,g,
               process_callback=None):
    # Load text data
    if input_text: # use user input
        L = data_load.load_data_text(input_text)
    else: # use txt file
        L = data_load.load_data("synthesize")
    ## Load graph
    #g = Graph(mode="synthesize")
    #print("Graph loaded")
    
    # Pick model
    #if model_path==None:
        ## TODO use filepath from browser
        #model_path = hp.logdir
    start = time.time()
    print(model_path)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # Restore parameters
        var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'Text2Mel')
        saver1 = tf.train.Saver(var_list=var_list)
        saver1.restore(sess, tf.train.latest_checkpoint(model_path + "-1"))
        print("Text2Mel Restored!")

        var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'SSRN') + \
                   tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'gs')
        saver2 = tf.train.Saver(var_list=var_list)
        saver2.restore(sess, tf.train.latest_checkpoint(model_path + "-2"))
        print("SSRN Restored!")

        # Feed Forward
        ## mel
        Y = np.zeros((len(L), hp.max_T, hp.n_mels), np.float32)
        prev_max_attentions = np.zeros((len(L),), np.int32)
        for j in tqdm(range(hp.max_T)):
            _gs, _Y, _max_attentions, _alignments = \
                sess.run([g.global_step, g.Y, g.max_attentions, g.alignments],
                         {g.L: L,
                          g.mels: Y,
                          g.prev_max_attentions: prev_max_attentions})
            Y[:, j, :] = _Y[:, j, :]
            prev_max_attentions = _max_attentions[:, j]
            if process_callback:
                if j % 10 == 0:
                    elapsed = time.time() - start
                    process_callback(j,elapsed)
                    

        # Get magnitude
        Z = sess.run(g.Z, {g.Y: Y})

        # Generate wav files
        #if not os.path.exists(hp.sampledir): os.makedirs(hp.sampledir)
        output = []
        for i, mag in enumerate(Z):
            print("Working on file", i+1)
            wav = spectrogram2wav(mag)
            wav = (wav/np.amax(wav) * 32767).astype(np.int16)
            output.append(wav)
            #print('writing file')
            #write(hp.sampledir + "/{}.wav".format(i+1), hp.sr, wav)
        if process_callback:
            elapsed = time.time() - start
            process_callback(hp.max_T,elapsed)
        outwav = np.concatenate(output)
        return outwav
            #for wav in output:

            #sd.play(wav,samplerate=hp.sr,blocking=True)
            # TODO Make playback non-blocking, add stream
            #sd.play(wav,samplerate=hp.sr)
        #    curr_wav = pygame.mixer.Sound(wav)
        #    curr_wav.play()
            #my_data = (my_data * 32767).astype(np.int16)

    
if __name__ == '__main__':
    synthesize()
    print("Done")


