import numpy as np
import matplotlib.pyplot as plt

# plot은 점과 점을 긋는 것, x,y가 확정이어야함

startt=0
endd=6
samp_t=1/15  #[sec]
sig1_freq=0.5 #[HZ]

ttime=np.arange(startt,endd,samp_t)
sig_1=np.cos(2*np.pi*sig1_freq*ttime)
sig_2=np.sin(2*np.pi*sig1_freq*ttime)

plt.plot(ttime,sig_1)
plt.plot(ttime,sig_2)
plt.xlabel("Time[sec]")
plt.ylabel("Voltage[V]")
plt.legend(["sig1","sig2"])
plt.grid(True)