# LSTM autoencoder
import keras 
import numpy as np 
import pandas as pd 
import time 
from keras.models import Sequential
from utils.utils import save_logs
from keras.layers import Dense, LSTM, Dropout


from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from tensorflow.compat.v1.keras.backend import set_session
import tensorflow.compat.v1 as tf1


class encoderEmb_LSTM:

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
		encoded = keras.layers.LSTM(5,activation='relu')(input_layer)
		encoded = keras.layers.Dense(50,activation='relu')(encoded)
		embd = keras.layers.Dense(10,activation='relu')(encoded)
		decoded = keras.layers.Dense(50,activation='relu')(embd)
		recons = keras.layers.RepeatVector(input_shape[0])(decoded)
		recons = keras.layers.LSTM(5,activation='relu' ,return_sequences=True)(recons)
		recons = keras.layers.Dense(30)(recons)
		model = keras.models.Model(inputs=input_layer, outputs=recons)
		self.encoder = keras.models.Model(inputs=input_layer, outputs=encoded)
		self.decoder = keras.models.Model(inputs=input_layer, outputs=decoded)
		model.compile(loss='mean_squared_error', optimizer=keras.optimizers.Adam())

		print(model.summary())
        
		reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=50, 

			min_lr=0.0001)
		file_path = self.output_directory+'best_model.hdf5'

		model_checkpoint = keras.callbacks.ModelCheckpoint(filepath=file_path, monitor='loss', 
			save_best_only=True)

		self.callbacks = [reduce_lr,model_checkpoint]

		return model 


	def fit(self, x_train, y_train, x_val, y_val,y_true): 
		
		
		# x_val and y_val are only used to monitor the test loss and NOT for training  
		
		batch_size = 200
		nb_epochs = 40

		mini_batch_size = int(min(x_train.shape[0]/10, batch_size))

		start_time = time.time() 

		hist = self.model.fit(x_train, x_train, batch_size=mini_batch_size, epochs=nb_epochs,
			verbose=self.verbose, validation_data=(x_val,x_val), callbacks=self.callbacks)
		
		duration = time.time() - start_time

		model = keras.models.load_model(self.output_directory+'best_model.hdf5')

		rec = model.predict(x_val)
		y_pred = np.array([np.linalg.norm(a-b) for a,b in zip(x_val, rec)])
		print("generating tests embeding vector ##########################")
		encod = self.encoder.predict(x_val)
		decod = self.decoder.predict(x_val)
		y_pred_emb = np.array([np.linalg.norm(a-b) for a,b in zip(encod, decod)])
		print("embeding vector shape is {}".format(encod.shape))
		np.save(self.output_directory+'encod_test.npy', encod)
		np.save(self.output_directory+'decod_test.npy', decod)
		np.save(self.output_directory+'score_embd.npy', y_pred_emb)
		#print(y_pred.shape, y_true.shape)
		# convert the predicted from binary to integer 
		#y_pred = np.argmax(y_pred , axis=1)
		print(y_pred.shape, y_true.shape)
		save_logs(self.output_directory, hist, y_pred, y_true, duration,le_type="reconstruction")

		keras.backend.clear_session()