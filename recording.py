
import alsaaudio as aa
import numpy as np
import pyqtgraph as pg
from PyQt4 import QtCore, QtGui
from matplotlib import pyplot
import sys

np.set_printoptions(threshold=np.nan)

mode = sys.argv[2] #range or doppler
PC = True #pulse cancelling for range

card_index = int(sys.argv[1])

FS = 44100 
tp = 0.02
c = 3 * 10**8
N = int(FS * tp)
FACTOR = 2
MAX_COUNTER = 4
if mode == 'doppler':
	FACTOR = 1
CHUNKSZ = FACTOR * N
ZOOM_IN = 25 # we want to view the lower frequencies


FFT_SIZE = 4 * CHUNKSZ
WIDTH_IM = 800
HEIGHT_IM = int(FFT_SIZE / 2 / ZOOM_IN)

fmin = 2315 * 10 ** 6
fmax = 2536 * 10 ** 6
vmin = 0.5
vmax = 5
sens = (fmax-fmin)/(vmax-vmin);
vstart = 0.788 #triangle wave min
vend = 3.17 #triangle wave max
fc = 2530 * 10**6
lambduh = c / fc

fstart = vstart*sens
fstop = vend*sens
BW = fstop - fstart 


#Image quality is highly dependent on good choice of these numbers

nc_decrease_max = float(sys.argv[3])
COLOR_HIGH = 0
COLOR_LOW = -50
	

# card_index = 0



class MicrophoneRecorder():
	# initializing the microphone with alsaaudio
	def __init__(self, signal):
		# sample format : PCM_FORMAT_S16_LE
		# rate : 44.1 kHz
		# number of channels : 2
		self.signal = signal
		self.stream = aa.PCM(aa.PCM_CAPTURE, aa.PCM_NORMAL,cardindex = card_index)
		self.stream.setchannels(2)
		self.stream.setrate(FS)
		self.stream.setformat(aa.PCM_FORMAT_S16_LE)
		if (mode == 'range'):
			self.stream.setperiodsize(2000)
		else:
			self.stream.setperiodsize(int(CHUNKSZ/2))
		self.sif_last_time = np.zeros(N-1)

		self.total_max = 0
		#Init plots
		self.real_time_plot = pg.plot(title="Radar Signal")
		self.real_freq_plot = pg.plot(title="Radar Spectrum")
		self.real_extra_plot = pg.plot(title="Trigger")
		
		if mode == 'range':
			self.real_time_plot.setRange(yRange=(-32000,32000))
		else:
			self.real_time_plot.setRange(yRange=(-1000,1000))
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
		if mode == 'range':
			title = 'Range vs Time'
		else:
			title = 'Velocity vs Time'
		self.img = pg.ImageItem(title=title)
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
			rr = c / 2 / BW
			max_range = rr * N / 2				
			yscale = 1.0 / (self.img_array.shape[1]/max_range)
		yscale = yscale / ZOOM_IN
		self.img.scale((1./FS)*N, yscale)
		self.setLabel('bottom', 'Time')
		if mode == 'doppler':
			self.setTitle('Doppler Radar')
			self.setLabel('left', 'Velocity', units='m/s')
		else:
			self.setTitle('Range Time Indicator')
			self.setLabel('left', 'Range', units='m')

		self.show()

	def updateDoppler(self, chunk):
			
		self.counter  += 1
		self.chunklist = np.concatenate((self.chunklist,chunk))
		if (self.counter >= 2):
			data = chunk[0:-1:2]
			trig = chunk[1:-1:2]
			zpad = FFT_SIZE
	
			v = np.fft.fft(data,zpad)
			v_c = v
		
			v_c = 20 * np.log10(np.abs(v_c))
			v_c = v_c[0:int(len(v_c)/2)]
			v_c = v_c[0:int(len(v_c)/ZOOM_IN)]
			
			mmax = np.max(v_c)

			if mmax > mic.total_max:
				mic.total_max = mmax
			else:
				mic.total_max -= nc_decrease_max
			mic.real_time_plot.plot(np.arange(len(data)),data,clear=True)	
			mic.real_freq_plot.plot(np.arange(len(v_c)),v_c,clear=True)
			mic.real_extra_plot.plot(np.arange(len(trig)),trig,clear=True)
			pg = QtGui.QApplication.processEvents()
			self.img_array = np.roll(self.img_array, -1, 0)
			self.img_array[-1:] = v_c - mic.total_max		
			self.img.setImage(self.img_array, autoLevels = False)
			self.counter=  0
			self.chunklist = []	
	def updateRange(self, chunk):
		# Processing data
		self.counter += 1
		self.chunklist = np.concatenate((self.chunklist,chunk))
		if (self.counter >= 2):	
			data = self.chunklist[0:-1:2]
			trig = self.chunklist[1:-1:2]
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
				sif = sif - mic.sif_last_time
				mic.sif_last_time = sif + mic.sif_last_time
			ave = np.mean(sif)
			sif = np.array(sif)
			zpad = FFT_SIZE
			# plotting
			v = np.fft.ifft(sif, zpad)
			v = 20*np.log10(np.abs(v))
			S = v[0:(int)(len(v)/2/ZOOM_IN)]

			mmax = np.max(S)
			if mmax > mic.total_max:
				mic.total_max = mmax
			else: #decrease the max over time
				mic.total_max -= nc_decrease_max
			#Plot Signals
			mic.real_freq_plot.plot(np.arange(len(S)),S,clear=True)
			mic.real_extra_plot.plot(np.arange(len(trig)),trig,clear=True)
			mic.real_time_plot.plot(np.arange(int(len(data)/2)),data[0:int(len(data)/2)],clear=True)
			self.img_array = np.roll(self.img_array, -1, 0)
			self.img_array[-1:] = S	- mic.total_max	
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

