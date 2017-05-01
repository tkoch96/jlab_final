import numpy as np
import pyqtgraph as pg
import pyaudio
from PyQt4 import QtCore, QtGui
from matplotlib import pyplot

FS = 44100 #Hz
tp = .04
c = 3 * 10**8
N = int(FS * tp)
CHUNKSZ = N #samples
fstart = 2402 * 10**6
fstop = 2495 * 10**6
BW = fstop-fstart
rr = c/(2*BW)
max_range = rr*N/2
ZOOM_IN = 20

fc = 2495 * 10**6



class MicrophoneRecorder():
	def __init__(self, signal):
		self.signal = signal
		self.p = pyaudio.PyAudio()
		self.stream = self.p.open(format=pyaudio.paInt16,
							channels=2,
							rate=FS,
							input=True,
							frames_per_buffer=CHUNKSZ)
		self.real_time_plot = pg.plot()
	
	def read(self):
		data = self.stream.read(CHUNKSZ)
		y = np.fromstring(data, 'int16')
		self.signal.emit(y)

	def close(self):
		self.stream.stop_stream()
		self.stream.close()
		self.p.terminate()

class SpectrogramWidget(pg.PlotWidget):
	read_collected = QtCore.pyqtSignal(np.ndarray)
	def __init__(self):
		super(SpectrogramWidget, self).__init__()

		self.img = pg.ImageItem()
		self.addItem(self.img)

		self.img_array = np.zeros((200, np.int(2*CHUNKSZ / ZOOM_IN)))

		# bipolar colormap
		pos = np.array([0., 1., 0.5, 0.25, 0.75])
		#cyan - very low, dark blue - low, black - middle, red -high, yellow - very high
		color = np.array([[0,255,255,255], [255,255,0,255], [0,0,0,255], (0, 0, 255, 255), (255, 0, 0, 255)], dtype=np.ubyte)
		cmap = pg.ColorMap(pos, color)
		lut = cmap.getLookupTable(0.0, 1.0, 256)

		# set colormap
		self.img.setLookupTable(lut)
		self.img.setLevels([30,90])

		#get max velocity
		delta_f = np.linspace(0,FS/2, 2*N)
		lambduh = c/fc
		velocity = delta_f*lambduh/2

		# setup the correct scaling for y-axis
		freq = np.arange((CHUNKSZ/2)+1)/(float(CHUNKSZ)/FS)
		yscale = 1.0/(self.img_array.shape[1]/velocity[-1])
		yscale = yscale / ZOOM_IN
		self.img.scale((1./FS)*N,yscale)
	

		self.setLabel('left', 'Velocity', units='m/s')

		self.show()

	def update(self, chunk):
		#normalized, windowed frequencies in data chunk
		# trigger = chunk[1:CHUNKSZ:2]
		# data = chunk[0:CHUNKSZ:2]
		# thresholds = trigger > 0
		# pulses = np.zeros(CHUNKSZ)
		# for i in np.arange(100,len(trigger)-N):
		 	#look for pulse being sent
		#  	if trigger[i] == 1 & int(np.mean(trigger[i-11:i-1] == 0)):
        #                         pulses = data[i:i+N]
        #                         break
		# pyplot.plot(data)
		#pyplot.show()
		# spec = np.fft.rfft(pulses,CHUNKSZ) #compute fft
		# spec = spec[0:int(len(spec)/2)] #only half
		# spec = 20*np.log10(spec)
		# spec = spec[0:int(len(spec)/ZOOM_IN)]
		data = chunk[0:-1:2]

		data = data - np.mean(data)
		
		zpad = 4 * N


		v = np.fft.fft(data,zpad)
		v = 20 * np.log10(np.abs(v))
		v = v[0:int(len(v)/2)]

		v = v[0:int(np.floor(len(v)/ZOOM_IN))]
		# mic.real_time_plot.plot(np.arange(len(trigger)),data,clear=True)
		pg.QtGui.QApplication.processEvents()
		#pyplot.plot(data)
		#pyplot.show()

		# roll down one and replace leading edge with new data
		# mmax = np.max(spec)
		self.img_array = np.roll(self.img_array, -1, 0)
		self.img_array[-1:] = v
		
		# self.img_array[-1:] = spec-mmax
		print(self.img_array[0:10])
		self.img.setImage(self.img_array, autoLevels=False)

if __name__ == '__main__':
	app = QtGui.QApplication([])
	w = SpectrogramWidget()
	w.read_collected.connect(w.update)

	mic = MicrophoneRecorder(w.read_collected)

	# time (seconds) between reads
	interval = FS/CHUNKSZ
	t = QtCore.QTimer()
	t.timeout.connect(mic.read)
	t.start(1000/interval) #QTimer takes ms

	app.exec_()
mic.close()
