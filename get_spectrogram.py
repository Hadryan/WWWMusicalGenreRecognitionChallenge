import os
import sys
sys.path.append('preprocessing/')
from utils import Standardizer
from audio import AudioProcessor
import numpy as np
import fire
import tqdm
import glob

class Extract:
	def __init__(self):
		self.ap = AudioProcessor()
		self.audio_path = '/crowdai-payload'
		self.sclr = Standardizer('preprocessing/sclr.dat.gz')
		self.get_filelist()


	def get_filelist(self):
		self.filelist = glob.glob(os.path.join(self.audio_path, '*.mp3'))


	def get_spectrogram(self, fn):
		X = self.ap.mel(self.ap.forward(self.ap.load(fn)), logamp=True)
		num_chunk = (X.shape[0]-43)/10
		spectrograms = []
		for i in range(num_chunk):
			spectrograms.append(X[i*10:i*10+43].reshape(1,43,128))
		spectrograms = np.array(spectrograms)
		spectrograms = self.sclr.transform_batch(spectrograms)
		spectrograms = spectrograms.transpose(0,3,2,1)
		return spectrograms

	def iter_get_spectrogram(self, iter_idx="0"):
		for fn in tqdm.tqdm(self.filelist):
			if fn[-5] == str(iter_idx):
				new_fn = fn[:-3] + 'npy'
				if not os.path.exists(new_fn):
					spec = self.get_spectrogram(fn)
					np.save(open(new_fn, 'wb'), spec)

if __name__ == '__main__':
	e = Extract()
	fire.Fire({'run': e.iter_get_spectrogram})
