import scipy.io
import numpy as np
import matplotlib
from numpy import fft
from scipy.io import wavfile
from matplotlib import pyplot as plt

#Read the signal
sample_rate, original_signal = wavfile.read('dsp.wav')
time = np.arange(len(original_signal))/float(sample_rate)
print("The sample rate is:",sample_rate)
print("The length of the signal is:",len(original_signal))
print("The time of the signal is:",time[-1])

#Display the signal
plt.plot(time, original_signal)
plt.title('Speech signal from Iron man')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.savefig('Original_signal.jpg')

#Make TDTFT
window_length=int(0.02*sample_rate)
print("The window length is:",window_length)
print("The frequency sampling factor is:",window_length)
L=int(0.5*window_length)
print("The temporal decimation factor is:",L)
original_signal=np.append(original_signal,np.zeros(window_length))
TDTFT=np.array([0]*window_length)
for i in range(0,len(original_signal)-window_length,L):
    TDTFT_add=np.fft.fft(original_signal[i:window_length+i])
    TDTFT=np.vstack((TDTFT,TDTFT_add))
TDTFT=np.delete(TDTFT,0,axis=0)
TDTFT=np.transpose(TDTFT)
print("The shape of the TDTFT of the signal is:",TDTFT.shape)
np.savetxt("TDTFT.txt",TDTFT)

#Make inverse TDTFT
GFBS=np.array([])
for i in range(TDTFT.shape[1]):
    GFBS_add=np.fft.ifft(TDTFT[:,i])[0:L]
    GFBS=np.append(GFBS,GFBS_add)
time=np.arange(len(GFBS))/float(sample_rate)
GFBS = (GFBS.real).astype(np.int64)

#Display the signal got back from TDTFT
plt.plot(time, GFBS)
plt.title('Speech signal getting back using GFBS')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.savefig('Speech signal getting back using GFBS.jpg')
wavfile.write("GFBS.wav",sample_rate,GFBS)