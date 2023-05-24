import numpy as np
from matplotlib import pyplot as plt
import matplotlib
matplotlib.use('tkagg')
np.fft
SAMPLE_RATE = 100
DURATION = 1

def sine_wave(freq, sample_rate, duration):
    x = np.linspace(0, duration, sample_rate * duration, endpoint=False)
    frequencies = x * freq
    y = np.sin((2 * np.pi) * frequencies)
    return x, y


_, tone_1 = sine_wave(1, SAMPLE_RATE, DURATION)
_, tone_2 = sine_wave(4, SAMPLE_RATE, DURATION)
_, tone_3 = sine_wave(7, SAMPLE_RATE, DURATION)

mixed_tone = 3*tone_1 + tone_2 + 0.5*tone_3

def dft(f):
    n = len(f)
    f_hat = np.zeros(shape=(f.shape), dtype=np.complex)
    for k in range(n):
        for j in range(n):
            f_hat[k] += f[j] * np.exp(-2j * np.pi * j * k/n)
    return f_hat


out = dft(mixed_tone)

N = len(out)
n = np.arange(N)
T = N/SAMPLE_RATE
freq = n/T

plt.stem(freq[:50], np.abs(out[:50]))
plt.xlabel('Freq (Hz)')
plt.ylabel('DFT Amplitude')
plt.show()