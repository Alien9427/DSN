import numpy as np
from scipy import fftpack
import time

def gen_spectrogram_2(slc, win):
    slc_w = slc[0].shape[0]
    win_size = int(slc_w * win)
    hamming_win = np.hamming(win_size)
    hamming_win_2d = np.sqrt(np.outer(hamming_win, hamming_win))

    spectrogram = np.zeros([32, 32, 32, 32], dtype=complex)

    slc_fft = fftpack.fftshift(fftpack.fft2(slc))
    for i in range(int(slc_w/4),int(slc_w*3/4)):
        for j in range(int(slc_w/4),int(slc_w*3/4)):
            spectrogram[:,:,i-int(slc_w/4),j-int(slc_w/4)] = fftpack.ifftn(np.pad(hamming_win_2d * slc_fft[i-int(win_size/2):i+int(win_size/2), j-int(win_size/2):j+int(win_size/2)], int((32-win_size)/2) ,'constant'))

    return spectrogram
