# -*- coding: utf-8 -*-
import os
import sys
sys.path.append('models/')
sys.path.append('preprocessing/')
import numpy as np
import fire
import tqdm
import glob
import csv
from audio import AudioProcessor
from utils import Standardizer

from keras import backend as K
from machine import Network
from machine_trf import Network as NetworkT


class Transfer:
	def __init__(self):
		# get path of model checkpoints
		self.chroma_path = 'models/target_chroma.hdf5'
		self.dmfcc_path = 'models/target_dmfcc.hdf5'
		self.essentia_path = 'models/target_essentia.hdf5'
		self.genre_path = 'models/target_genre.hdf5'
		self.mfcc_path = 'models/target_mfcc.hdf5'
		self.original_path = 'models/target_original.hdf5'
		self.ultimate_path = 'models/ultimate.hdf5'
		self.audio_path = '/crowdai-payload'
		#self.audio_path = '/data/minz/sample'

		self.ap = AudioProcessor()
		self.sclr = Standardizer('preprocessing/sclr.dat.gz')
		self.output_path = '/tmp/output.csv'


	def run(self, gpu_id="0"):
		# set gpu id
		os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
		# set models for the feature extraction
		self.get_models()
		# get filelist
		self.get_filelist()
		# iteration
		self.iter_predictions()

	def get_models(self):
		# chroma
		_cn = Network(num_tags=40)
		model_chroma = _cn.model
		model_chroma.load_weights(self.chroma_path)
		self.ff_c = K.function([model_chroma.input, K.learning_phase()], 
				[model_chroma.layers[29].output])
		# dmfcc
		_dn = Network(num_tags=40)
		model_dmfcc = _dn.model
		model_dmfcc.load_weights(self.dmfcc_path)
		self.ff_d = K.function([model_dmfcc.input, K.learning_phase()], 
				[model_dmfcc.layers[29].output])
		# essentia
		_en = Network(num_tags=40)
		model_essentia = _en.model
		model_essentia.load_weights(self.essentia_path)
		self.ff_e = K.function([model_essentia.input, K.learning_phase()], 
				[model_essentia.layers[29].output])
		# genre
		_gn = Network(num_tags=40)
		model_genre = _gn.model
		model_genre.load_weights(self.genre_path)
		self.ff_g = K.function([model_genre.input, K.learning_phase()], 
				[model_genre.layers[29].output])
		# mfcc
		_mn = Network(num_tags=40)
		model_mfcc = _mn.model
		model_mfcc.load_weights(self.mfcc_path)
		self.ff_m = K.function([model_mfcc.input, K.learning_phase()], 
				[model_mfcc.layers[29].output])
		# original
		_on = Network(num_tags=16)
		model_original = _on.model
		model_original.load_weights(self.original_path)
		self.ff_o = K.function([model_original.input, K.learning_phase()], 
				[model_original.layers[29].output])
		# ultimate
		_un = NetworkT(num_tags=16)
		self.model_ultimate = _un.model
		self.model_ultimate.load_weights(self.ultimate_path)

	def get_filelist(self):
		self.filelist = glob.glob(os.path.join(self.audio_path, '*.npy'))


		
	def get_features(self, fn):
		spectrograms = np.load(fn)
		feat_c = self.ff_c([spectrograms, 0])[0]
		feat_d = self.ff_d([spectrograms, 0])[0]
		feat_e = self.ff_e([spectrograms, 0])[0]
		feat_g = self.ff_g([spectrograms, 0])[0]
		feat_m = self.ff_m([spectrograms, 0])[0]
		feat_o = self.ff_o([spectrograms, 0])[0]
		features = np.concatenate((feat_c, feat_d, feat_e, feat_g, feat_m, feat_o), axis=1)
		return features

	def get_prediction(self, fn):
		features = self.get_features(fn)
		prd = self.model_ultimate.predict(features)
		prd = np.mean(prd, axis=0)
		return prd

	def iter_predictions(self):
		CLASSES = ['Blues', 'Classical', 'Country', 'Easy Listening', 'Electronic',
				   'Experimental', 'Folk', 'Hip-Hop', 'Instrumental', 'International',
				   'Jazz', 'Old-Time / Historic', 'Pop', 'Rock', 'Soul-RnB', 'Spoken']
		HEADERS = ['file_id'] + CLASSES
		csvfile = open(self.output_path, "w")
		writer = csv.DictWriter(csvfile, fieldnames=HEADERS)
		writer.writeheader()
		TEST_FILES = sorted(self.filelist)

		for fn in tqdm.tqdm(self.filelist):
			_track_id = fn.split("/")[-1].replace(".mp3","")
			prd = self.get_prediction(fn)
			row = {}
			row['file_id'] = _track_id
			for _idx, _class in enumerate(CLASSES):
				row[_class] = prd[_idx]
			writer.writerow(row)
		csvfile.close()
		print("Output file written at ", self.output_path)

if __name__ == '__main__':
	t = Transfer()
	fire.Fire({'run': t.run})

