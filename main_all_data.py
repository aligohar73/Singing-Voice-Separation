
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 27 16:36:38 2019

@author: Ali Gohar
"""

#!/usr/bin/env python
# -*- coding: utf-8 -*-

# imports
from __future__ import print_function
import numpy as np
from scipy import signal
import tf_transform as tf
from ex5_Part_2_utils import mask_calc, visualize_audio, evaluate_results, \
    print_evaluation_results, load_audio, save_audio

from keras.layers import TimeDistributed, GRU, Activation, Input
from keras.models import Sequential, Model
from keras.layers import Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D, Dropout
from keras.utils import to_categorical
from keras.losses import categorical_crossentropy
from keras.optimizers import SGD
from keras.applications.vgg16 import VGG16
import os


def get_model_3(input_shape, model_case):
    """Creates the RNN model 

    :param input_shape: The input shape to the model
    :type input_shape: (int, int)
    :param output_dimensions: The output dimensionality of the two layers
    :type output_dimensions: (int, int) | list[int]
    :param model_case: The case (e.g. source prediction)
    :type model_case: str
    :return: The model
    :rtype: keras.models.Sequential
    """
    model = Sequential()
    model.add(GRU(16,dropout=0.25,input_shape=input_shape, return_sequences=True)) 
    model.add(GRU(16,dropout=0.25, return_sequences=True))
    model.add(GRU(16,dropout=0.25, return_sequences=True))
    model.add(TimeDistributed(Dense((input_shape[-1]))))
    model.add(Dense(100))
    model.add(Dense(513, activation = 'relu'))
    #out = Activation('sigmoid')(model)
    # TODO: Implement a 2-4 layer GRU-RNN based model
    # TODO: Use FNN to reduce the dimension if necessary
    # TODO: Use proper activation based on the range of the spectrogram/mask magnitude
    model.summary() 
#    _print_informative_message(
#        model=model,
#        model_case=model_case
#    )

    return model

Train_data_path = './DSD100/DSD100/Sources/Dev'
Test_data_path = './DSD100/DSD100/Mixtures/Test'
all_train_other = np.zeros((1,2))
all_train_vocal = np.zeros((1,2))
all_test_mixture = np.zeros((1,2))
#test_data_path = 
audio_vocal = 'vocals.wav'
audio_bass = 'bass.wav'
audio_drums = 'drums.wav'
audio_others = 'other.wav'
audio_mixture = 'mixture.wav'

#load audio files for training
for filename in os.listdir(Train_data_path):
    audio_files1 = os.path.join(Train_data_path,filename,audio_vocal)
    audio_files2 = os.path.join(Train_data_path,filename,audio_bass)
    audio_files3 = os.path.join(Train_data_path,filename,audio_drums)
    audio_files4 = os.path.join(Train_data_path,filename,audio_others)
    vocals, fs = load_audio(audio_files1)
    bass, fs = load_audio(audio_files2)
    drums, fs = load_audio(audio_files3)
    other, fs = load_audio(audio_files4)
    all_other = (drums + bass + other)
    all_train_other = np.concatenate((all_train_other, all_other),0)
    all_train_vocal = np.concatenate((all_train_vocal,vocals),0)

#load mixture audio files for testing
for filename in os.listdir(Test_data_path):
    audio_files = os.path.join(Train_data_path,filename,audio_mixture)
    mixture, fs = load_audio(audio_mixture)
    all_test_mixture = np.concatenate((all_test_mixture, mixture),0)

all_train_mixture = all_train_other + all_train_vocal 

##converet into one channel
all_train_other = all_train_other.mean(1)
all_train_vocal = all_train_vocal.mean(1)
all_train_mixture = all_train_mixture.mean(1)
all_test_mixture = all_test_mixture.mean(1)

 # STFT parameters
win_size = 1024 
fft_size = 1024
hop = 512
windowing_func = signal.hamming(win_size)

# STFT of the music
train_other_mag, mix_phase = tf.stft(all_train_other, windowing_func, fft_size, hop)

# STFT of the mixture
train_mixture_mag, mix_phase = tf.stft(all_train_mixture, windowing_func, fft_size, hop)
test_mixture_mag, mix_phase = tf.stft(all_test_mixture,  windowing_func, fft_size, hop)

# Compute the magnitudes of vocal sources
train_vocals_mag, vocals_phase = tf.stft(all_train_vocal, windowing_func, fft_size, hop)

#exponent = 0.7
#smoothness = 1
#model_case_2 = 'mask prediction'
#
## Estimate the mask using IRM equation
#calculated_mask = mask_calc(vocals_mag, all_other_mag, exponent, smoothness)
#
##split Data into train and test
#data_split = int(len(mixture_mag) * 0.5)
#train_calcu_mask = calculated_mask[0:data_split]
#test_calcu_mask = calculated_mask[data_split:(data_split*2)]
#train_mix_mag = mixture_mag[0:data_split]
#test_mix_mag =  mixture_mag[data_split:(data_split *2)]
#
## Mask calculate
#train_vocals_mag_irm = train_calcu_mask * train_mix_mag
#train_vocals_irm = tf.i_stft(train_vocals_mag_irm, mix_phase, win_size, hop)
#save_audio('{} using irm.wav'.format(model_case_2), train_vocals_irm, fs)
#
#model_2 = get_model_3(
#    input_shape=train_mix_mag.shape,
#    model_case=model_case_2
#)
#model_2.compile(optimizer = 'adam',loss='binary_crossentropy' )
#model_2.fit(
#    x=train_mix_mag.reshape((1, ) + train_mix_mag.shape),
#    y=train_calcu_mask.reshape((1, ) + train_calcu_mask.shape),
#    epochs=50
#)
#predicted_magn_1 = model_2.predict(x=test_mix_mag.reshape((1, ) + test_mix_mag.shape))
#
#predicted_mask_fnn = model_2.predict(x=test_mix_mag.reshape((1, ) + test_mix_mag.shape))
#predicted_magn_2 = test_mix_mag * predicted_mask_fnn.squeeze(0)
#predicted_audio_2 = tf.i_stft(predicted_magn_2, mix_phase, win_size, hop)
#save_audio('test_audio.wav', predicted_audio_2, fs)
###save files
##save_audio('trai.wavn', all_other, fs)
#sources = np.zeros((2, data_split), dtype=np.float32)
#sources[0, :] = vocals[data_split:(data_split*2)]
#sources[1, :] = all_other[data_split:(data_split*2)] 
#
#sdr, sir, sar = evaluate_results(
#        targeted=sources,
#        predicted=predicted_audio_2,
#        mixture=mixture[data_split:(data_split*2)],
#        signal_length=data_split
#    )
#
#print_evaluation_results(
#    sdr=sdr, sir=sir, sar=sar, model_case=model_case_2
#)
#
