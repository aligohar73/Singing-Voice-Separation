# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 11:12:23 2019

@author: Ali Gohar
"""
import wave
import librosa
import scipy.io.wavfile as wav
import numpy as np
from scipy.fftpack import fft


def load_audio(_audio_file):
    _fs , _y = wav.read(_audio_file)
    if _y.dtype == np.int16:
        _y = _y / 32768.0  # short int to float
    return _y, _fs

def extract_feature(_audio_file):

    nfft = 1024
    win_len = nfft
    hop_len = win_len / 2
    window = np.hamming(win_len)
#    
    _y, _fs = load_audio(_audio_file)
    _nb_frames = int(np.floor((len(_y)- win_len)/hop_len))
    _fft_spec = np.zeros((_nb_frames,int(1+hop_len)))
    
    frame_count = 0
    for i in range(_nb_frames):
        y_win = _y[int(i*hop_len):int(i*hop_len+win_len)] * window
        _fft_spec[frame_count,:] = np.abs(fft(y_win)[:int(1+hop_len)]) **2
        frame_count += 1
    _fft_spec1 = np.abs(librosa.stft(_y,n_fft = nfft, hop_length= int(hop_len), window = 'hamming'))
    _fft_spec2, n_fft1 = librosa.core.spectrum._spectrogram(y=_y, n_fft=nfft, hop_length=int(nfft/2), power=1)
    return _fft_spec, _fft_spec1, _fft_spec2, n_fft1


if __name__ == '__main__':
    audio_file = 'ex1.wav'
    y, fs = load_audio(audio_file)
    open_audio = wave.open(audio_file)
    number_of_channels = open_audio.getnchannels()
    fft_feat, fft_feat1, fft_feat2, n_fft = extract_feature(audio_file)
    print('resting')