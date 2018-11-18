import scipy.io
import numpy as np
import matplotlib
from scipy.io import wavfile
from matplotlib import pyplot as plt

sample_rate, data = wavfile.read('dsp.wav')
time = np.arange(len(data))/float(sample_rate)

plt.plot(time, data)
plt.title('Speech signal from Iron man')
plt.xlabel('time')
plt.ylabel('amplitude')
plt.savefig('record and display signal.jpg')