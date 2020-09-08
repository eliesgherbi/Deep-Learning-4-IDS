# FCN
import keras 
import numpy as np 
import pandas as pd 
import time 

from utils.utils import save_logs

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from tensorflow.compat.v1.keras.backend import set_session
import tensorflow.compat.v1 as tf1
import os

class encoder_CLASSIC:

	def __init__(self, output_directory, input_shape, nb_classes, verbose=False):
		
		config = ConfigProto()
		config.gpu_options.per_process_gpu_memory_fraction = 0.5
		config.gpu_options.allow_growth = True
		tf1.Session(config=config)
		
		self.output_directory = output_directory
		self.model = self.build_model(input_shape, nb_classes)
		if(verbose==True):
			self.model.summary()
		self.verbose = verbose
		self.model.save_weights(self.output_directory+'model_init.hdf5')

	def build_model(self, input_shape, nb_classes):
		input_layer = keras.layers.Input(input_shape)
		flat = keras.layers.Flatten()(input_layer)
		
		Dense1 = keras.layers.Dense(512, activation='relu')(flat)
		#conv1 = keras.layers.MaxPool1D(2)(conv1)
		
		Dense2 = keras.layers.Dense(128, activation='relu')(Dense1)
		#conv2 = keras.layers.MaxPool1D(2)(conv2)
		
		emb = keras.layers.Dense(60, activation='relu')(Dense2)
		
		Dense3 = keras.layers.Dense(128, activation='relu')(emb)
		#deconv3_mx = keras.layers.UpSampling1D(5)(deconv3)
		

		Dense4 = keras.layers.Dense(512, activation='relu')(Dense3)
		#deconv4 = keras.layers.UpSampling1D(2)(deconv4)
		#flat = keras.layers.Flatten()(deconv4)
		Dense5 = keras.layers.Dense(input_shape[1])(Dense4)
		output = keras.layers.RepeatVector(input_shape[0])(Dense5)
        

		#output_layer = keras.layers.Dense(nb_classes, activation='sigmoid')(deco)

		model = keras.models.Model(inputs=input_layer, outputs=output)

		model.compile(loss='mse', optimizer = keras.optimizers.Adam())
		#self.encoder = keras.models.Model(inputs=input_layer, outputs=conv2)
		#self.decoder = keras.models.Model(inputs=input_layer, outputs=deconv3)
		self.emb = keras.models.Model(inputs=input_layer, outputs=emb)
		print(model.summary())

		reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=10, 
			min_lr=0.0001)

		file_path = self.output_directory+'best_model.hdf5'

		model_checkpoint = keras.callbacks.ModelCheckpoint(filepath=file_path, monitor='loss', 
			save_best_only=True)

		self.callbacks = [reduce_lr,model_checkpoint]

		return model 

	def fit(self, x_train, y_train, x_val, y_val,y_true,test_spe=False): 
		
		
		# x_val and y_val are only used to monitor the test loss and NOT for training  
		
		batch_size = 25
		nb_epochs = 1000
		
		mini_batch_size = int(min(x_train.shape[0]/10, batch_size))

		start_time = time.time() 
		
		
		hist = self.model.fit(x_train, x_train, batch_size=mini_batch_size, epochs=nb_epochs,
			verbose=self.verbose, validation_data=(x_val,x_val), callbacks=self.callbacks, shuffle=True)
		#hist=None
		duration = time.time() - start_time
		
		
		print(self.output_directory)
		model = keras.models.load_model(self.output_directory+'best_model.hdf5')
		
		x_rec = model.predict(x_train)
		y_rec = np.array([np.linalg.norm(a-b) for a,b in zip(x_train,x_rec)])
		tresh_tr = np.percentile(y_rec,99)
		
		print(x_val.shape)
		rec = model.predict(x_val)
		y_pred = np.array([np.linalg.norm(a-b) for a,b in zip(x_val, rec)])
		tresh = np.percentile(y_pred,99)
		  
		print("generating tests embeding vector ##########################")
		#encod = self.encoder.predict(x_val)
		#decod = self.decoder.predict(x_val)
		emb = self.emb.predict(x_val)
		#y_pred_emb = np.array([np.linalg.norm(a-b) for a,b in zip(encod, decod)])
		#print("embeding vector shape is {}".format(encod.shape))
		print(y_pred.shape, y_true.shape)
		#np.save(self.output_directory+'encod_test.npy', encod)
		#np.save(self.output_directory+'decod_test.npy', decod)
		#np.save(self.output_directory+'score_embd.npy', y_pred_emb)
		np.save(self.output_directory+'embd_test.npy', emb)
		
		

		# convert the predicted from binary to integer 
		#y_pred = np.argmax(y_pred , axis=1)
		
		#hist=None
		print(y_pred.shape, y_true.shape)
		save_logs(self.output_directory, hist, y_pred, y_true, duration,tresh=tresh)
		
		print("testing !!!")
		
		if test_spe:
			hist=None
			l_atk = ['continuous',
					 'plateau',
					 'suppress',
					 'flooding',
					 'playback']

			if 'nofrq' in self.output_directory:
				print("nofreq testing*********")
				path_in = "/scratch/Project-CTI/data/SynCAN/classification_SOA/archives/mts_archive/const_nofrq_"
			else:
				print("with_freq testing*********")
				path_in = "/scratch/Project-CTI/data/SynCAN/classification_SOA/archives/mts_archive/const_"
			#x_nor = x_val
			#y_nor = y_true.reshape(-1)
			for atk in l_atk:
				start_time = time.time() 
				
				outp_atk = self.output_directory+"const_"+atk+"/"
				if not os.path.exists(outp_atk):
					os.makedirs(outp_atk)
				x_val = np.load(path_in+atk+"/x_test.npy")
				#x_val = np.concatenate([x_val,x_nor])
				y_true = np.load(path_in+atk+"/y_test.npy")
				#print(y_true.shape, y_nor.shape)
				#y_true = np.concatenate([y_true,y_nor])
				rec = model.predict(x_val)
				y_pred = np.array([np.linalg.norm(a-b) for a,b in zip(x_val, rec)])
				tresh_atk = np.percentile(y_pred,30)
				
				#tresh = y_pred[-3000]
				#encod = self.encoder.predict(x_val)
				#decod = self.decoder.predict(x_val)
				emb = self.emb.predict(x_val)
				#y_pred_emb = np.array([np.linalg.norm(a-b) for a,b in zip(encod, decod)])
				#print("embeding vector shape is {}".format(encod.shape))
				#print(y_pred.shape, y_true.shape)
				#np.save(outp_atk+'encod_test.npy', encod)
				#np.save(outp_atk+'decod_test.npy', decod)
				#np.save(outp_atk+'score_embd.npy', y_pred_emb)
				np.save(outp_atk+'embd_test.npy', emb)
				duration = time.time() - start_time
				save_logs(outp_atk, hist, y_pred, y_true, duration,tresh=tresh)
				

				"""trash test"""
				print("Trash testing")
				x_val = np.load(path_in+atk+"/x_trash.npy")
				#x_val = np.concatenate([x_val,x_nor])
				y_true = np.load(path_in+atk+"/y_trash.npy")
				#print(y_true.shape, y_nor.shape)
				#y_true = np.concatenate([y_true,y_nor])
				rec = model.predict(x_val)
				y_pred = np.array([np.linalg.norm(a-b) for a,b in zip(x_val, rec)])
				
				print("treshol atk test data is {}:".format(tresh_atk))
				#tresh = y_pred[-3000]
				#encod = self.encoder.predict(x_val)
				#decod = self.decoder.predict(x_val)
				emb = self.emb.predict(x_val)
				#y_pred_emb = np.array([np.linalg.norm(a-b) for a,b in zip(encod, decod)])
				#print("embeding vector shape is {}".format(encod.shape))
				#print(y_pred.shape, y_true.shape)
				#np.save(outp_atk+'encod_test.npy', encod)
				#np.save(outp_atk+'decod_test.npy', decod)
				#np.save(outp_atk+'score_embd.npy', y_pred_emb)
				np.save(outp_atk+'embd_trash.npy', emb)
				duration = time.time() - start_time
				outp_atk = outp_atk+"trash_"
				save_logs(outp_atk, hist, y_pred, y_true, duration,tresh=tresh_atk)
		
		print(tresh, tresh_tr)
		keras.backend.clear_session()