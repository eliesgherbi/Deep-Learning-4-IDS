# TSCN
import tensorflow.keras  as keras
import numpy as np 
import pandas as pd 
import time 

from utils.utils import save_logs

from tcn import TCN
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from tensorflow.compat.v1.keras.backend import set_session
import tensorflow.compat.v1 as tf1
from tensorflow.python.keras.optimizer_v2.adam import Adam

class Classifier_TCN:

	def __init__(self, output_directory, input_shape, nb_classes, verbose=True):
		
		config = ConfigProto()
		config.gpu_options.per_process_gpu_memory_fraction = 0.5
		config.gpu_options.allow_growth = True
		tf1.Session(config=config)
		
		self.output_directory = output_directory
		self.model = self.build_model(input_shape, nb_classes)
		if(verbose==True):
			self.model.summary()
		self.verbose = verbose
		self.model.save_weights(self.output_directory+'model_init',
								save_format='tf')

	def build_model(self, input_shape, nb_classes):
		input_layer = keras.layers.Input(input_shape)

		num_feat=30
		num_classes=2
		nb_filters=32
		kernel_size=5
		dilations=[2 ** i for i in range(4)]
		padding='causal'
		nb_stacks=1
		#max_len=X_train[0:1].shape[1]
		use_skip_connections=True
		use_batch_norm=True
		dropout_rate=0.05
		kernel_initializer='he_normal'
		#lr=0.00
		activation='linear'
		use_layer_norm=True

		return_sequences=true
		#name='tcn_1'
		x = TCN(nb_filters, kernel_size, nb_stacks, dilations, padding,
		    use_skip_connections, dropout_rate, return_sequences,
		    activation, kernel_initializer, use_batch_norm, use_layer_norm)(input_layer)
		"""
		return_sequences=False
		#name='tcn_1'
		x = TCN(nb_filters, kernel_size, nb_stacks, dilations, padding,
		    use_skip_connections, dropout_rate, return_sequences,
		    activation, kernel_initializer, use_batch_norm, use_layer_norm)(input_layer)
        
		"""
		output_layer = keras.layers.Dense(nb_classes, activation='sigmoid')(x)

		model = keras.models.Model(inputs=input_layer, outputs=output_layer)

		model.compile(loss='binary_crossentropy', optimizer = Adam(), 
			metrics=['accuracy'])

		print(model.summary())

		reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=50, 
			min_lr=0.0001)

		file_path = self.output_directory+'best_model'

		model_checkpoint = keras.callbacks.ModelCheckpoint(filepath=file_path, monitor='loss', 
			save_best_only=True)

		self.callbacks = [reduce_lr,model_checkpoint]

		return model 

	def fit(self, x_train, y_train, x_val, y_val,y_true): 
		
		
		# x_val and y_val are only used to monitor the test loss and NOT for training  
		batch_size = 200
		nb_epochs = 100

		mini_batch_size = int(min(x_train.shape[0]/10, batch_size))

		start_time = time.time() 

		hist = self.model.fit(x_train, y_train, batch_size=mini_batch_size, epochs=nb_epochs,
			verbose=self.verbose, validation_data=(x_val,y_val), callbacks=self.callbacks)
		
		duration = time.time() - start_time

		model = keras.models.load_model(self.output_directory+'best_model')

		y_pred = model.predict(x_val)

		# convert the predicted from binary to integer 
		#y_pred = np.argmax(y_pred , axis=1)

		save_logs(self.output_directory, hist, y_pred, y_true, duration)

		keras.backend.clear_session()

	
