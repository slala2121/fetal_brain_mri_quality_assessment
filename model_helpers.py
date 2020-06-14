import numpy as np
import keras
from keras.activations import relu 
from keras.initializers import he_normal
# Import necessary components to build LeNet
from keras.models import Sequential, Model
from keras.layers import Input, Lambda
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.layers.merge import Concatenate
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras import backend as K



"""
data_type: str single/dual
model_type: shallow, vgg_scratch, vgg_transfer
input_shape: dimensionality
batch_size:  only relevant for dual models
fusion_method: only relevant for dual models
"""
def build_model(data_type,model_type,input_shape,batch_size=1,
	fusion_method='concatenation'):

	# construct the base model
	base_model=build_base_model(model_type,input_shape)

	# construct the final model
	num_fc_units= 1024 if model_type=='shallow' else 4096 # just for particular case of replicating the paper

	if data_type=='single':
		model=build_single_path_model(base_model,num_fc_units,add_dropout=True)
	elif data_type=='dual':
		model=build_dual_path_model(base_model,num_fc_units,add_dropout=True,
						batch_size=batch_size,
						fusion_method=fusion_method)

	"""
	if model_type=='shallow':
		if data_type=='single':
			return single_path_shallow_cnn(input_shape)
		if data_type=='dual':
			return dual_path_shallow_cnn(input_shape,batch_size,feature_fusion_method)
	
	# otherwise for now it's vgg
	transfer=True if model_type=='vgg_transfer' else False
	
	if data_type=='single':
		return single_path_vgg(input_shape,transfer)

	if data_type=='dual':
		return dual_path_vgg(input_shape,batch_size,transfer,feature_fusion_method)
	"""
	return model


def build_base_model(model_type,input_shape):
	base_model=None
	if model_type=='shallow':
		base_model=build_shallow_cnn(input_shape)
	elif model_type=='vgg_11':
		base_model=build_vgg_11(input_shape)
	elif model_type=='vgg_16_scratch':
		base_model=build_vgg_16_scratch(input_shape)
	elif model_type=='vgg_16_transfer':
		base_model=build_vgg_16_transfer(input_shape)

	return base_model


def build_shallow_cnn(input_shape):
	model=Sequential()
	model.add(Conv2D(filters=64,kernel_size=7,strides=4,padding='valid',
		activation='relu',kernel_initializer='he_normal',input_shape=input_shape))
	model.add(MaxPooling2D(pool_size=5,strides=1,padding='valid'))
	model.add(Conv2D(filters=32,kernel_size=9,strides=1,padding='valid',
		activation='relu',kernel_initializer='he_normal'))

	return model

def build_vgg_11(input_shape):
	model = Sequential()
	kernel_initializer='glorot_uniform' # default
	model.add(Conv2D(64, (3, 3), activation='relu', input_shape=input_shape))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Conv2D(128, (3, 3), activation='relu',kernel_initializer=kernel_initializer))
	model.add(MaxPooling2D(pool_size=(2, 2)))

	model.add(Conv2D(256, (3, 3), activation='relu',kernel_initializer=kernel_initializer))
	model.add(Conv2D(256, (3, 3), activation='relu',kernel_initializer=kernel_initializer))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Conv2D(512, (3, 3), activation='relu',kernel_initializer=kernel_initializer))
	model.add(Conv2D(512, (3, 3), activation='relu',kernel_initializer=kernel_initializer))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Conv2D(512, (3, 3), activation='relu',kernel_initializer=kernel_initializer))
	model.add(Conv2D(512, (3, 3), activation='relu',kernel_initializer=kernel_initializer))
	model.add(MaxPooling2D(pool_size=(2, 2)))

	return model

def build_vgg_16_scratch(input_shape):
	weights=None 
	model=keras.applications.vgg16.VGG16(include_top=False, 
		weights=weights, input_tensor=None, input_shape=input_shape, pooling=None)

	return model

def build_vgg_16_transfer(input_shape):
	weights='imagenet' 
	model=keras.applications.vgg16.VGG16(include_top=False, 
		weights=weights, input_tensor=None, input_shape=input_shape, pooling=None)

	return model


"""
implementation of a siamese-like/dual path CNN
model based on this paper:  Image Quality Assessment: Using Similar Scene as Reference Liang, et al.
activation, initializations are based on this paper:
-https://hackernoon.com/learning-keras-by-implementing-vgg16-from-scratch-d036733f2d5
-he et al (relu activations + initialization)

sources:
https://github.com/sorenbouma/keras-oneshot
https://software.intel.com/en-us/articles/keras-implementation-of-siamese-like-networks


use of pretrained models:
-https://github.com/ckanitkar/CS231nFinalProject/blob/master/Experiment3_SiameseNet/SiameseNetwork.py
-https://github.com/hsakas/siamese_similarity_model/blob/master/Train.py


https://github.com/topics/siamese-network
"""

def single_path_shallow_cnn(input_shape):
	print("implementing single_path_shallow_cnn")
	convnet=Sequential()
	convnet.add(Conv2D(filters=64,kernel_size=7,strides=4,padding='valid',
		activation='relu',kernel_initializer='he_normal',input_shape=input_shape))
	convnet.add(MaxPooling2D(pool_size=5,strides=1,padding='valid'))
	convnet.add(Conv2D(filters=32,kernel_size=9,strides=1,padding='valid',
		activation='relu',kernel_initializer='he_normal'))
	convnet.add(Flatten())
	convnet.add(Dense(units=1024,activation='relu',kernel_initializer='he_normal'))
	convnet.add(Dropout(0.5))
	convnet.add(Dense(units=1024,activation='relu',kernel_initializer='he_normal'))
	convnet.add(Dropout(0.5))
	convnet.add(Dense(units=1,activation='sigmoid'))
	
	return convnet


def dual_path_shallow_cnn(input_shape,batch_size=1,fusion_method='concatenation'):
	print("implementing dual_path_shallow_cnn")
	convnet=Sequential()
	convnet.add(Conv2D(filters=64,kernel_size=7,strides=4,padding='valid',
		activation='relu',kernel_initializer='he_normal'))
	convnet.add(MaxPooling2D(pool_size=5,strides=1,padding='valid'))
	convnet.add(Conv2D(filters=32,kernel_size=9,strides=1,padding='valid',
		activation='relu',kernel_initializer='he_normal'))
	convnet.add(Flatten())
	convnet.add(Dense(units=1024,activation='relu'))

	base_model=convnet
	

	ref_input=Input(input_shape)
	eval_input=Input(input_shape)

	ref_fc_activation=base_model(ref_input)
	eval_fc_activation=base_model(eval_input)


	# apply the same dropout to each input path
	fc_out_both_inputs=keras.layers.concatenate([ref_fc_activation,eval_fc_activation])
	fc_out_both_inputs=keras.layers.Reshape((2,1024))(fc_out_both_inputs)
	noise_shape=(batch_size,1,1024) # apply the same dropout mask to both inputs
	fc_out_both_inputs=Dropout(0.5,noise_shape=noise_shape)(fc_out_both_inputs)

	# fuse activations of both inputs via the 2nd FC layer
	if fusion_method=='concatenation':
		fused_tensor=keras.layers.Flatten()(fc_out_both_inputs)
	elif fusion_method=='difference':
		L1_layer = Lambda(lambda tensors:K.abs(tensors[0] - tensors[1]))

		# slice out the activations corresponding to each input 

		fc_out_inp_1=Lambda(lambda x: x[:,0,:])(fc_out_both_inputs)
		fc_out_inp_2=Lambda(lambda x: x[:,1,:])(fc_out_both_inputs)
		fused_tensor=L1_layer([fc_out_inp_1,fc_out_inp_2])


	fc_out=Dense(units=1024,activation='relu')(fused_tensor)
	fc_out=Dropout(0.5)(fc_out)

	prediction=Dense(units=1,activation='sigmoid')(fc_out)

	dual_cnn_model=Model(inputs=[ref_input,eval_input],outputs=[prediction])

	return dual_cnn_model

# https://github.com/sorenbouma/keras-oneshot/blob/master/SiameseNet.ipynb
def siamese_net(input_shape):
	print("implementing siamese net")

	import numpy.random as rng

	left_input=Input(input_shape)
	right_input=Input(input_shape)

	def W_init(shape,name=None):
		"""Initialize weights as in paper"""
		values = rng.normal(loc=0,scale=1e-2,size=shape)
		return K.variable(values,name=name)
	#//TODO: figure out how to initialize layer biases in keras.
	def b_init(shape,name=None):
		"""Initialize bias as in paper"""
		values=rng.normal(loc=0.5,scale=1e-2,size=shape)
		return K.variable(values,name=name)

	convnet = Sequential()
	convnet.add(Conv2D(64,(10,10),activation='relu',input_shape=input_shape,
					   kernel_initializer=W_init,kernel_regularizer=l2(2e-4)))
	convnet.add(MaxPooling2D())
	convnet.add(Conv2D(128,(7,7),activation='relu',
					   kernel_regularizer=l2(2e-4),kernel_initializer=W_init,bias_initializer=b_init))
	convnet.add(MaxPooling2D())
	convnet.add(Conv2D(128,(4,4),activation='relu',kernel_initializer=W_init,kernel_regularizer=l2(2e-4),bias_initializer=b_init))
	convnet.add(MaxPooling2D())
	convnet.add(Conv2D(256,(4,4),activation='relu',kernel_initializer=W_init,kernel_regularizer=l2(2e-4),bias_initializer=b_init))
	convnet.add(Flatten())
	convnet.add(Dense(4096,activation="sigmoid",kernel_regularizer=l2(1e-3),kernel_initializer=W_init,bias_initializer=b_init))

	#call the convnet Sequential model on each of the input tensors so params will be shared
	encoded_l = convnet(left_input)
	encoded_r = convnet(right_input)
	#layer to merge two encoded inputs with the l1 distance between them
	L1_layer = Lambda(lambda tensors:K.abs(tensors[0] - tensors[1]))
	#call this layer on list of two input tensors.
	L1_distance = L1_layer([encoded_l, encoded_r])
	prediction = Dense(1,activation='sigmoid',bias_initializer=b_init)(L1_distance)
	siamese_net = Model(inputs=[left_input,right_input],outputs=prediction)

	return siamese_net




"""
based on the vgg paper
"""
def single_path_vgg(input_shape,transfer=False):

	weights='imagenet' if transfer else None
	print("implemneting single_path_vgg using weights %s " %weights)
	vgg_base_model=keras.applications.vgg16.VGG16(include_top=False, 
		weights=weights, input_tensor=None, input_shape=input_shape, pooling=None)

	last=vgg_base_model.output
	x=Flatten(name="top_inp")(last)
	x=Dense(4096, activation='relu',name='top_dense_1')(x)
	x=Dropout(0.5,name='top_dropout_1')(x)
	x=Dense(4096, activation='relu',name='top_dense_2')(x)
	x=Dropout(0.5,name='top_dropout_2')(x)
	preds=Dense(1, activation='sigmoid',name="top_out")(x)
	model=keras.Model(vgg_base_model.input,preds)

	return model


"""
batch_size: only relevant for training since it is needed for the dropout mask to be the same
for both inputs


relevant literature:
-Deep Transfer Learning for Person Reidentification Geng, et al

"""
def dual_path_vgg(input_shape,batch_size,transfer=False,fusion_method='concatenation'):
	weights='imagenet' if transfer else None
	print("implemneting dual_path_vgg using weights %s " %weights)
	vgg_base_model=keras.applications.vgg16.VGG16(include_top=False, 
		weights=weights, input_tensor=None, input_shape=input_shape, pooling=None)

	last=vgg_base_model.output
	x=Flatten()(last)
	base_model_output=Dense(4096,activation='relu')(x)

	base_model=keras.Model(vgg_base_model.input,base_model_output)

	# ensure application of the same dropout
	# TODO base_model_output=Dropout(0.5)(x)

	ref_input=Input(input_shape)
	eval_input=Input(input_shape)

	ref_fc_activation=base_model(ref_input)
	eval_fc_activation=base_model(eval_input)


	# apply the same dropout to each input path
	fc_out_both_inputs=keras.layers.concatenate([ref_fc_activation,eval_fc_activation])
	fc_out_both_inputs=keras.layers.Reshape((2,4096))(fc_out_both_inputs)
	noise_shape=(batch_size,1,4096) # apply the same dropout mask to both inputs
	fc_out_both_inputs=Dropout(0.5,noise_shape=noise_shape)(fc_out_both_inputs)

	# fuse activations of both inputs via the 2nd FC layer
	if fusion_method=='concatenation':
		fused_tensor=keras.layers.Flatten()(fc_out_both_inputs)
	elif fusion_method=='difference':
		L1_layer = Lambda(lambda tensors:K.abs(tensors[0] - tensors[1]))

		# slice out the activations corresponding to each input 

		fc_out_inp_1=Lambda(lambda x: x[:,0,:])(fc_out_both_inputs)
		fc_out_inp_2=Lambda(lambda x: x[:,1,:])(fc_out_both_inputs)
		fused_tensor=L1_layer([fc_out_inp_1,fc_out_inp_2])


	fc_out=Dense(units=4096,activation='relu')(fused_tensor)
	fc_out=Dropout(0.5)(fc_out)

	prediction=Dense(units=1,activation='sigmoid')(fc_out)

	dual_cnn_model=Model(inputs=[ref_input,eval_input],outputs=[prediction])

	return dual_cnn_model




def build_single_path_model(base_model,num_fc_units,add_dropout=True):

	last=base_model.output
	x=Flatten(name="top_inp")(last)
	x=Dense(num_fc_units, activation='relu',name='top_dense_1')(x)
	if add_dropout:
		x=Dropout(0.5,name='top_dropout_1')(x)
	x=Dense(num_fc_units, activation='relu',name='top_dense_2')(x)
	if add_dropout:
		x=Dropout(0.5,name='top_dropout_2')(x)
	preds=Dense(1, activation='sigmoid',name="top_out")(x)
	model=keras.Model(base_model.input,preds)

	return model

"""
base_model: Sequential instance 

if dropout is incorporated, batch size needs to be specified

"""
def build_dual_path_model(base_model,num_fc_units,add_dropout=True,batch_size=1,
						fusion_method='concatenation'):
	last=base_model.output
	x=Flatten()(last)
	base_model_output=Dense(num_fc_units,activation='relu')(x)
	base_model_with_fc=keras.Model(base_model.input,base_model_output)

	# ensure application of the same dropout
	# TODO base_model_output=Dropout(0.5)(x)

	input_shape=base_model.input_shape[1:] # exclude batch
	ref_input=Input(input_shape)
	eval_input=Input(input_shape)

	ref_fc_activation=base_model_with_fc(ref_input)
	eval_fc_activation=base_model_with_fc(eval_input)


	# apply the same dropout to each input path
	fc_out_both_inputs=keras.layers.concatenate([ref_fc_activation,eval_fc_activation])
	fc_out_both_inputs=keras.layers.Reshape((2,num_fc_units))(fc_out_both_inputs)
	noise_shape=(batch_size,1,num_fc_units) # apply the same dropout mask to both inputs
	if add_dropout:
		fc_out_both_inputs=Dropout(0.5,noise_shape=noise_shape)(fc_out_both_inputs)

	# fuse activations of both inputs via the 2nd FC layer
	if fusion_method=='concatenation':
		fused_tensor=keras.layers.Flatten()(fc_out_both_inputs)
	elif fusion_method=='difference':
		L1_layer = Lambda(lambda tensors:K.abs(tensors[0] - tensors[1]))

		# slice out the activations corresponding to each input 

		fc_out_inp_1=Lambda(lambda x: x[:,0,:])(fc_out_both_inputs)
		fc_out_inp_2=Lambda(lambda x: x[:,1,:])(fc_out_both_inputs)
		fused_tensor=L1_layer([fc_out_inp_1,fc_out_inp_2])


	fc_out=Dense(units=num_fc_units,activation='relu')(fused_tensor)
	if add_dropout:
		fc_out=Dropout(0.5)(fc_out)

	prediction=Dense(units=1,activation='sigmoid')(fc_out)

	dual_cnn_model=Model(inputs=[ref_input,eval_input],outputs=[prediction])

	return dual_cnn_model



def alexnet_model(img_shape=(224, 224, 3), n_classes=10, l2_reg=0.,
	weights=None):

	# Initialize model
	alexnet = Sequential()

	# Layer 1
	alexnet.add(Conv2D(96, (11, 11), input_shape=img_shape,
		padding='same', kernel_regularizer=l2(l2_reg)))
	alexnet.add(BatchNormalization())
	alexnet.add(Activation('relu'))
	alexnet.add(MaxPooling2D(pool_size=(2, 2)))

	# Layer 2
	alexnet.add(Conv2D(256, (5, 5), padding='same'))
	alexnet.add(BatchNormalization())
	alexnet.add(Activation('relu'))
	alexnet.add(MaxPooling2D(pool_size=(2, 2)))

	# Layer 3
	alexnet.add(ZeroPadding2D((1, 1)))
	alexnet.add(Conv2D(512, (3, 3), padding='same'))
	alexnet.add(BatchNormalization())
	alexnet.add(Activation('relu'))
	alexnet.add(MaxPooling2D(pool_size=(2, 2)))

	# Layer 4
	alexnet.add(ZeroPadding2D((1, 1)))
	alexnet.add(Conv2D(1024, (3, 3), padding='same'))
	alexnet.add(BatchNormalization())
	alexnet.add(Activation('relu'))

	# Layer 5
	alexnet.add(ZeroPadding2D((1, 1)))
	alexnet.add(Conv2D(1024, (3, 3), padding='same'))
	alexnet.add(BatchNormalization())
	alexnet.add(Activation('relu'))
	alexnet.add(MaxPooling2D(pool_size=(2, 2)))

	# Layer 6
	alexnet.add(Flatten())
	alexnet.add(Dense(3072))
	alexnet.add(BatchNormalization())
	alexnet.add(Activation('relu'))
	alexnet.add(Dropout(0.5))

	# Layer 7
	alexnet.add(Dense(4096))
	alexnet.add(BatchNormalization())
	alexnet.add(Activation('relu'))
	alexnet.add(Dropout(0.5))

	# Layer 8
	alexnet.add(Dense(n_classes))
	alexnet.add(BatchNormalization())
	alexnet.add(Activation('softmax'))

	if weights is not None:
		alexnet.load_weights(weights)

	return alexnet





# https://keras.io/getting-started/sequential-model-guide/#examples
# modified for binary classification
# 4 conv layers
def keras_vgg_like_convnet(input_shape):
	model = Sequential()
	# this applies 32 convolution filters of size 3x3 each.
	model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
	model.add(Conv2D(32, (3, 3), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.25))

	model.add(Conv2D(64, (3, 3), activation='relu'))
	model.add(Conv2D(64, (3, 3), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.25))

	model.add(Flatten())
	model.add(Dense(256, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(1, activation='sigmoid'))

	return model

#8 conv layers
# https://arxiv.org/pdf/1409.1556.pdf
# modified for binary classification
def vgg_11(input_shape):
	model = Sequential()
	kernel_initializer='glorot_uniform' # default
	model.add(Conv2D(64, (3, 3), activation='relu', input_shape=input_shape))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Conv2D(128, (3, 3), activation='relu',kernel_initializer=kernel_initializer))
	model.add(MaxPooling2D(pool_size=(2, 2)))

	model.add(Conv2D(256, (3, 3), activation='relu',kernel_initializer=kernel_initializer))
	model.add(Conv2D(256, (3, 3), activation='relu',kernel_initializer=kernel_initializer))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Conv2D(512, (3, 3), activation='relu',kernel_initializer=kernel_initializer))
	model.add(Conv2D(512, (3, 3), activation='relu',kernel_initializer=kernel_initializer))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Conv2D(512, (3, 3), activation='relu',kernel_initializer=kernel_initializer))
	model.add(Conv2D(512, (3, 3), activation='relu',kernel_initializer=kernel_initializer))
	model.add(MaxPooling2D(pool_size=(2, 2)))

	model.add(Flatten())
	model.add(Dense(4096, activation='relu',kernel_initializer=kernel_initializer))
	model.add(Dropout(0.5))
	model.add(Dense(4096, activation='relu',kernel_initializer=kernel_initializer))
	model.add(Dropout(0.5))
	model.add(Dense(1, activation='sigmoid',kernel_initializer=kernel_initializer))

	return model


# 13 conv layers
# https://arxiv.org/pdf/1409.1556.pdf

def vgg_16(input_shape,transfer=False):
	print("implementing vgg_16")
	weights='imagenet' if transfer else None
	base_model=keras.applications.vgg16.VGG16(include_top=False, 
		weights=weights, input_tensor=None, input_shape=input_shape, pooling=None, classes=1)

	last=base_model.output
	x=Flatten(name="top_inp")(last)
	x=Dense(4096, activation='relu',name='top_dense_1')(x)
	x=Dropout(0.5,name='top_dropout_1')(x)
	x=Dense(4096, activation='relu',name='top_dense_2')(x)
	x=Dropout(0.5,name='top_dropout_2')(x)
	preds=Dense(1, activation='sigmoid',name="top_out")(x)
	model=keras.Model(base_model.input,preds)


	return model



# https://github.com/eweill/keras-deepcv/blob/master/models/classification/alexnet.py

def alexnet_model(img_shape=(224, 224, 3), n_classes=10, l2_reg=0.,
	weights=None):

	# Initialize model
	alexnet = Sequential()

	# Layer 1
	alexnet.add(Conv2D(96, (11, 11), input_shape=img_shape,
		padding='same', kernel_regularizer=l2(l2_reg)))
	alexnet.add(BatchNormalization())
	alexnet.add(Activation('relu'))
	alexnet.add(MaxPooling2D(pool_size=(2, 2)))

	# Layer 2
	alexnet.add(Conv2D(256, (5, 5), padding='same'))
	alexnet.add(BatchNormalization())
	alexnet.add(Activation('relu'))
	alexnet.add(MaxPooling2D(pool_size=(2, 2)))

	# Layer 3
	alexnet.add(ZeroPadding2D((1, 1)))
	alexnet.add(Conv2D(512, (3, 3), padding='same'))
	alexnet.add(BatchNormalization())
	alexnet.add(Activation('relu'))
	alexnet.add(MaxPooling2D(pool_size=(2, 2)))

	# Layer 4
	alexnet.add(ZeroPadding2D((1, 1)))
	alexnet.add(Conv2D(1024, (3, 3), padding='same'))
	alexnet.add(BatchNormalization())
	alexnet.add(Activation('relu'))

	# Layer 5
	alexnet.add(ZeroPadding2D((1, 1)))
	alexnet.add(Conv2D(1024, (3, 3), padding='same'))
	alexnet.add(BatchNormalization())
	alexnet.add(Activation('relu'))
	alexnet.add(MaxPooling2D(pool_size=(2, 2)))

	# Layer 6
	alexnet.add(Flatten())
	alexnet.add(Dense(3072))
	alexnet.add(BatchNormalization())
	alexnet.add(Activation('relu'))
	alexnet.add(Dropout(0.5))

	# Layer 7
	alexnet.add(Dense(4096))
	alexnet.add(BatchNormalization())
	alexnet.add(Activation('relu'))
	alexnet.add(Dropout(0.5))

	# Layer 8
	if n_classes in [1,2]:
		alexnet.add(Dense(1))
		alexnet.add(BatchNormalization())
		alexnet.add(Activation('sigmoid'))
	else:
		alexnet.add(Dense(n_classes))
		alexnet.add(BatchNormalization())
		alexnet.add(Activation('softmax'))

	if weights is not None:
		alexnet.load_weights(weights)

	return alexnet






