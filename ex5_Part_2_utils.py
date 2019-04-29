#!/usr/bin/env python
# -*- coding: utf-8 -*-

# imports
from __future__ import print_function

import numpy as np

import scipy.io.wavfile as wav

import matplotlib.pyplot as plt
import matplotlib.colors as colors

from bsseval import bss_eval_sources


eps = np.finfo(np.float32).tiny

__docformat__ = 'reStructuredText'
__all__ = [
    'mask_calc',
    'visualize_audio',
    'evaluate_results',
    'print_evaluation_results',
    'load_audio',
    'save_audio'
]


def mask_calc(source_spectrogram, mixture_spectrogram, exponent, smoothness):
    """Calculates the IRM.

    :param source_spectrogram: The source spectrogram
    :type source_spectrogram: numpy.ndarray
    :param mixture_spectrogram: The mixture spectrogram
    :type mixture_spectrogram: numpy.ndarray
    :param exponent: The exponent
    :type exponent: float
    :param smoothness: The smoothness factor
    :type smoothness: float
    :return: The mask
    :rtype: numpy.ndarray
    """
    numerator = source_spectrogram ** exponent
    denominator = source_spectrogram ** exponent + mixture_spectrogram ** exponent
    mask = numerator/(eps + denominator)
    return mask**smoothness


def visualize_audio(audio_data):
    """Plots the audio data of the mixture, singing voice, and\
    rest of music.

    :param audio_data: The audio data of the mixture, singing voice, and\
                       rest of music.
    :type audio_data: list[numpy.ndarray]
    """
    plot_common_kwargs = {'aspect': 'auto', 'origin': 'lower'}
    titles = ['Mixture', 'Singing voice', 'Musical accompaniment']

    plt.figure()

    for i in range(len(audio_data)):
        plt.subplot2grid((5, 1), (i, 0), colspan=2)
        plt.imshow(
            audio_data[i].T,
            norm=colors.LogNorm(
                vmin=audio_data[i].min() + 0.01,
                vmax=audio_data[i].max()
            ),
            **plot_common_kwargs
        )
        plt.title(titles[i])
        plt.ylabel('Frequency channels')
    plt.show()


def evaluate_results(targeted, predicted, mixture, signal_length):
    """Calculates the source separation metrics.

    :param targeted: The targeted sources
    :type targeted: numpy.ndarray
    :param predicted: The predicted sources
    :type predicted: numpy.ndarray
    :param mixture: The mixture
    :type mixture: numpy.ndarray
    :param signal_length: The signal's length
    :type signal_length: int
    :return: The metrics
    :rtype: (numpy.ndarray, numpy.ndarray, numpy.ndarray)
    """
    predicted_sources = np.zeros((2, signal_length), dtype=np.float32)
    predicted_sources[0, :] = predicted[0:signal_length]
    predicted_sources[1, :] = mixture - predicted[0:signal_length]
    return bss_eval_sources(targeted, predicted_sources)[:-1]


def print_evaluation_results(sdr, sir, sar, model_case):
    """Prints the evaluation results to the command line.

    :param sdr: The SDR.
    :type sdr: numpy.ndarray
    :param sir: The SIR
    :type sir: numpy.ndarray
    :param sar: The SAR
    :type sar: numpy.ndarray
    :param model_case: The case of the model
    :type model_case: str
    """
    str_1 = '\nThe model that does {model_case} had: '.format(
        model_case=model_case)
    str_2 = 'Mean SDR: {sdr:5.2f} | Mean SIR: {sir:5.2f} | Mean SAR: {sar:5.2f}'.format(
        sdr=sdr.mean(), sir=sir.mean(), sar=sar.mean()
    )
    head_sep = '*' * (len(str_1) + len(str_2))

    print(str_1, end='')
    print(str_2)
    print(head_sep)
    print('')


def load_audio(audio_filename):
    """Loads audio file.

    :param audio_filename:
    :return: The audio samples and the sample rate
    :rtype: (numpy.ndarray, int)
    """
    _fs, _y = wav.read(audio_filename)
    if _y.dtype == np.int16:
        _y = _y / 32768.0  # short int to float
    return _y, _fs


def save_audio(file_name, audio_data, fs):
    """Saves to HD the audio data of the mixture.
    
    :param file_name: Name of the audio file
    :type file_name: str
    :param audio_data: The audio data of the mixture.
    :type audio_data: numpy.ndarray
    :param fs: The sampling frequency
    :type fs: int
    """
    wav.write(file_name, fs, audio_data)

# EOF
