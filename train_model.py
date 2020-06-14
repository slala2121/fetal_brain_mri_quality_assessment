import keras
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, CSVLogger,ReduceLROnPlateau

import dataset as ds
import argparse
import os
import shutil
import model_helpers as mh
import sys
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import time
import numpy as np

if __name__=='__main__':
	parser = argparse.ArgumentParser()

	# what stuff to save for debugging
	parser.add_argument('--save_images',metavar='int',default='0') # debugging 
	parser.add_argument('--save_model',metavar='int',default='0') # useful when tuning lr
	parser.add_argument('--enable_augmentations',metavar='int',default='0')
	# model params
	parser.add_argument('--model_type',default='single') # shallow, vgg_scratch, vgg_transfer  
	parser.add_argument('--lr',metavar='float',default='1e-4')
	parser.add_argument('--decay',metavar='float',default='0.0')
	parser.add_argument('--batch_size',default='32')
	parser.add_argument('--epochs',default='200')
	parser.add_argument('--adjust_lr_flag',default='0')
	parser.add_argument('--load_model_flag',default='0')

	# data loading/ save paths
	parser.add_argument('--data_source_dir',default='') 
	parser.add_argument('--data_partition_dir',default='')
	parser.add_argument('--main_res_path',default='results')

	# parameters
	args=parser.parse_args()

	# debugging
	save_images=int(args.save_images)
	save_model=int(args.save_model)
	enable_augmentations=int(args.enable_augmentations)

	# model parameters
	model_type=args.model_type
	lr=float(args.lr)
	decay=float(args.decay)
	batch_size=int(args.batch_size)
	epochs=int(args.epochs)
	load_model_flag=int(args.load_model_flag)
	adjust_lr_flag=int(args.adjust_lr_flag)

	# data loading/saving
	data_source_dir=args.data_source_dir
	data_partition_dir=args.data_partition_dir
	main_res_path=args.main_res_path

	# create folders to save results to
	if os.path.isdir(main_res_path):
		shutil.rmtree(main_res_path)
	os.mkdir(main_res_path)

	model_path=os.path.join(main_res_path,'models')
	os.mkdir(model_path)

	final_model_res_path=os.path.join(main_res_path,'final')
	os.mkdir(final_model_res_path)

	save_images_path=''
	if save_images:
		save_images_path=os.path.join(main_res_path,'final','sample_images')
		os.mkdir(save_images_path)
	

	transfer=True if model_type=='vgg_16_transfer' else False

	# prepare model
	num_channels=3 if transfer else 1
	input_shape=(256,256,num_channels)

	model=None
	if load_model_flag:
		print("loading model")
		model=keras.models.load_model("models/final.hdf5")
	else:
		model=mh.build_model(data_type='single',model_type=model_type,input_shape=input_shape)

	model.compile(loss='binary_crossentropy',
			  optimizer=optimizers.SGD(lr=lr, decay=decay, momentum=0.9, nesterov=False),
			  metrics=['accuracy'])

	print("compiled model")
	sys.stdout.flush()


	
	# prepare the data generators
	train_g=ds.DataGenerator(data_partition_path=os.path.join(data_partition_dir,'train.npy'),
							data_source_dir=data_source_dir,
							batch_size=batch_size,
							dim=input_shape,
							shuffle=True,
							augmentation_flag=enable_augmentations,
							save_images=save_images,save_images_path=save_images_path)


	val_g=ds.DataGenerator(data_partition_path=os.path.join(data_partition_dir,'val.npy'),
							data_source_dir=data_source_dir,
							dim=input_shape,
							save_images=save_images,save_images_path=save_images_path)


	

	# training callbacks
	csv_logger = CSVLogger(os.path.join(final_model_res_path,'training.log'))
	# tensorboard_callback=keras.callbacks.TensorBoard()
	callbacks=[csv_logger]

	if save_model:
		model_checkpoint=keras.callbacks.ModelCheckpoint(os.path.join(model_path,'best.hdf5'),
												monitor='val_loss',save_best_only=True,verbose=1)
		callbacks.append(model_checkpoint)
		
	if adjust_lr_flag:
		reduce_lr=keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, 
			verbose=1, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0)
		callbacks.append(reduce_lr)


	class_weights=np.load(os.path.join(data_partition_dir,'class_weight_bad_to_good.npy'))
	class_weights_dict={1:class_weights[0],0:class_weights[1]}
	
	t_start=time.time()
	
	
	history=model.fit_generator(
		train_g,
		epochs=epochs,
		callbacks=callbacks,
		class_weight=class_weights_dict,
		validation_data=val_g,
		use_multiprocessing=False)
	


	t_end=time.time()
	print("finished training in hours: ", (t_end-t_start)*1.0/3600)
	sys.stdout.flush()
	
	# save stats
	train_loss=history.history['loss']
	val_loss=history.history['val_loss'] 

	np.save(os.path.join(final_model_res_path,'train_loss.npy'),train_loss)

	# save final model
	if save_model:
		model.save(os.path.join(model_path,'final.hdf5'))

	plt.figure()
	plt.plot(np.arange(epochs),train_loss,'r',label='train')
	plt.plot(np.arange(epochs),val_loss,'b',label='val')
	plt.legend(loc='upper right')
	plt.savefig(os.path.join(final_model_res_path,'loss'))

	# separately plot train loss to debug whether it's actually converging
	plt.figure()
	plt.plot(np.arange(epochs),train_loss)
	plt.savefig(os.path.join(final_model_res_path,'train_loss'))
	