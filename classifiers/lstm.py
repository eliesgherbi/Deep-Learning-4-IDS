# LSTM
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

class Classifier_LSTM:

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
		
		model = Sequential()
		model.add(LSTM(input_shape=input_shape,units=10))
		#model.add(Dropout(0.5))
		#model.add(LSTM(units=30, activation='sigmoid'))
		#model.add(LSTM(units=10, activation='relu', return_sequences=False))
		#model.add(Dense(20, activation='relu'))
		model.add(Dense(1, activation='sigmoid'))

		model.compile(loss='binary_crossentropy', optimizer = 'rmsprop', 
			metrics=['accuracy'])
		
		print(model.summary())

		reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=10, 
			min_lr=0.000001)
		
		tbCallBack = keras.callbacks.TensorBoard(log_dir=self.output_directory+'Graph', histogram_freq=0, write_graph=True, write_images=True)

		file_path = self.output_directory+'best_model.hdf5'

		model_checkpoint = keras.callbacks.ModelCheckpoint(filepath=file_path, monitor='loss', 
			save_best_only=True)

		self.callbacks = [reduce_lr,model_checkpoint,tbCallBack]

		return model 

	def fit(self, x_train, y_train, x_val, y_val,y_true): 
		
		
		# x_val and y_val are only used to monitor the test loss and NOT for training  
		
		batch_size = 50
		nb_epochs = 150

		mini_batch_size = int(min(x_train.shape[0]/10, batch_size))

		start_time = time.time() 

		hist = self.model.fit(x_train, y_train, batch_size=mini_batch_size, epochs=nb_epochs,
			verbose=self.verbose, validation_data=(x_val,y_val), callbacks=self.callbacks)
		
		duration = time.time() - start_time

		model = keras.models.load_model(self.output_directory+'best_model.hdf5')
		
		
		y_pred = model.predict(x_val)
		print(y_pred.shape, y_true.shape)
		# convert the predicted from binary to integer 
		#y_pred = np.argmax(y_pred , axis=1)
		print(y_pred.shape, y_true.shape)
		save_logs(self.output_directory, hist, y_pred, y_true, duration)

		keras.backend.clear_session()