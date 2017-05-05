
import alsaaudio as aa
import numpy as np
import pyqtgraph as pg
from PyQt4 import QtCore, QtGui
from matplotlib import pyplot

mode = 'range' #range or doppler
PC = False #pulse cancelling for doppler

FS = 44100 
tp = 0.04
c = 3 * 10**8
N = int(FS * tp)
FACTOR = 2
MAX_COUNTER = 4
if mode == 'Doppler':
	FACTOR = 1
CHUNKSZ = FACTOR * N
ZOOM_IN = 25 # we want to view the lower frequencies


FFT_SIZE = 4 * CHUNKSZ
WIDTH_IM = 200
HEIGHT_IM = int(FFT_SIZE / 2 / ZOOM_IN)

fmin = 2315 * 10 ** 6
fmax = 2536 * 10 ** 6
vmin = 0.5
vmax = 5
sens = (fmax-fmin)/(vmax-vmin);
vstart = 0.788
vend = 3.17
fc = 2530 * 10**6
lambduh = c / fc

fstart = vstart*sens
fstop = vend*sens
BW = fstop - fstart 

COLOR_LOW = 40
COLOR_HIGH = 90

if mode == 'range':
	if PC == True:
		COLOR_LOW = -30
		COLOR_HIGH = 35
	else:
		COLOR_LOW = 35
		COLOR_HIGH = 50
	

# card_index = 0



class MicrophoneRecorder():
	# initializing the microphone with alsaaudio
	def __init__(self, signal):
		# sample format : PCM_FORMAT_S16_LE
		# rate : 44.1 kHz
		# number of channels : 2
		self.signal = signal
		self.stream = aa.PCM(aa.PCM_CAPTURE, aa.PCM_NORMAL,cardindex = 2)
		self.stream.setchannels(2)
		self.stream.setrate(FS)
		self.stream.setformat(aa.PCM_FORMAT_S16_LE)
		if (mode == 'range'):
			self.stream.setperiodsize(2000)
		else:
			self.stream.setperiodsize(CHUNKSZ) #236 + N max
		print("I am listening to" + self.stream.cardname())
		self.real_time_plot = pg.plot()
		self.real_freq_plot = pg.plot()
		self.real_extra_plot = pg.plot()

		self.real_time_plot.setRange(yRange=(-32000,32000))
		self.real_extra_plot.setRange(yRange=(-32000,32000))
		self.real_freq_plot.setRange(yRange=(-50,100))

	# reading from the pcm input
	def read(self):
		# The reading buffer will have size Fs * tp
		data = self.stream.read()
		# print(data[0])
		if (data[0] > 0):
			# print(data[1])
			y = np.fromstring(data[1],'int16')
			self.signal.emit(y)

class SpectrogramWidget(pg.PlotWidget):
	read_collected = QtCore.pyqtSignal(np.ndarray)
	def __init__(self):
		super(SpectrogramWidget, self).__init__()
		self.img = pg.ImageItem()
		self.addItem(self.img)
		self.img_array = np.zeros((WIDTH_IM, HEIGHT_IM))
		self.counter = 0
		self.chunklist = []
	
		# setting up my color maps, 
		# pos indicates location of map scale from 0 to 1
		# color indicates color at location pos
		pos = np.array([0., 0.25, 0.5, 0.75, 1.])
		color = np.array([[0,255,255,255],[0,0,255,255],[0,0,0,255],[255,0,0,255],[255,255,0,255]], dtype=np.ubyte)
		cmap = pg.ColorMap(pos,color)
		lut = cmap.getLookupTable(0.0, 1.0, 256)
		
		# set color map to img
		self.img.setLookupTable(lut)
		# NOTE TO SELF: Susceptible to change base on what we are looking at
		self.img.setLevels([COLOR_LOW,COLOR_HIGH])


		# set up correct scaling on the y axis
		if mode == 'doppler':
			# get max velocity
			delta_f = np.linspace(0, FS/2, 2*N)
			lambduh = c/fc
			velocity = delta_f * lambduh/2
			yscale = 1.0 / (self.img_array.shape[1]/velocity[-1])
		else:
			rr = c * 2 / BW
			max_range = rr * N / 2				
			yscale = 1.0 / (self.img_array.shape[1]/max_range)
		yscale = yscale / ZOOM_IN
		self.img.scale((1./FS)*N, yscale)
		if mode == 'doppler':
			self.setLabel('left', 'Velocity', units='m/s')
		else:
			self.setLabel('left', 'Range', units='m')

		self.show()

	def updateDoppler(self, chunk):
		data = chunk[1:-1:2]
		data = np.append(data,0)
		data = data - np.mean(data)
		data = data / 1000
		#data_delay = np.append(data[1:],[0])
		#data = data - data_delay
		#data_delay = np.append(data[1:],[0])
		#data = data - data_delay		
		zpad = FFT_SIZE
		# win_black = np.blackman(int(CHUNKSZ/2))
		# data = data*win_black #window it
	
		v = np.fft.fft(data * np.blackman(len(data)),zpad)
		# v_c = v / np.amax(v) #normalize
		v_c = v
		mic.last_tick = v
		
		v_c = 20 * np.log10(np.abs(v_c))
		v_c = v_c[0:int(len(v_c)/2)]
		v_c = v_c[0:int(len(v_c)/ZOOM_IN)]
		
		mic.real_time_plot.plot(np.arange(len(data)),data,clear=True)	
		mic.real_freq_plot.plot(np.arange(len(v_c)),v_c,clear=True)
		pg = QtGui.QApplication.processEvents()
		self.img_array = np.roll(self.img_array, -1, 0)
		self.img_array[-1:] = v_c		
		self.img.setImage(self.img_array, autoLevels = False)
	
	def updateRange(self, chunk):
		# Processing data
		self.counter += 1
		self.chunklist = np.concatenate((self.chunklist,chunk))
		# print("Size of chunklist: %d" %(len(self.chunklist)))
		if (self.counter == 2):	
			data = self.chunklist[1:-1:2]
			trig = self.chunklist[0:-1:2]
			start = [el > 0 for el in trig]
			sif = []
			thresh = 0
			for ii in range(100,len(start)-N):
				if 	(start[ii] == True and np.mean(start[ii-11:ii-1]) == 0):
					sif = data[ii:ii+N-1]
					break
			if (len(sif) == 0):
				return
			if PC == True: #pulse canceller
				sif_delayed = sif[1:]
				sif = sif_delayed - sif[0:len(sif)-1]
			ave = np.mean(sif)
			sif = np.array(sif) - ave
			zpad = FFT_SIZE
			# plotting
			v = np.fft.ifft(sif, zpad)
			v = 20*np.log10(np.abs(v))
			S = v[0:(int)(len(v)/2/ZOOM_IN)]
			#Plot Signals
			mic.real_freq_plot.plot(np.arange(len(S)),S,clear=True)
			mic.real_extra_plot.plot(np.arange(len(trig)),trig,clear=True)
			mic.real_time_plot.plot(np.arange(int(len(data)/2)),data[0:int(len(data)/2)],clear=True)
			self.img_array = np.roll(self.img_array, -1, 0)
			self.img_array[-1:] = S		
			self.img.setImage(self.img_array, autoLevels = False)
			self.chunklist = []
			self.counter = 0

				

if __name__ == '__main__':
	print("Initializing GUI")
	app = QtGui.QApplication([])
	print("Initializing spectrogram")
	w = SpectrogramWidget()
	print("Connecting read collected with updates")
	if mode == 'range':
		w.read_collected.connect(w.updateRange)
	else:
		w.read_collected.connect(w.updateDoppler)
	print("Connecting to mic")
	mic = MicrophoneRecorder(w.read_collected)
	interval = tp * 2
	t = QtCore.QTimer()
	print("Starting t")
	t.timeout.connect(mic.read)
	t.start(1000 * interval)
	print("executng")
	app.exec_()

