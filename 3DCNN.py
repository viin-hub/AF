import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Input, LeakyReLU, Add, UpSampling3D, Activation, SpatialDropout3D, Conv3D, BatchNormalization
from tensorflow.keras.optimizers import Adam
import numpy as np
import os
import random
from weighted_crossentropy import dyn_weighted_bincrossentropy
from os import listdir
from os.path import isfile, join
import pandas as pd
from data_generator import DataGenerator

def model(input_shape=(128, 128, 128, 1), n_base_filters=36, depth=2, dropout_rate=0.3,
		n_labels=2, activation_name="sigmoid"):
	
	inputs = keras.Input(input_shape)

	initializer = tf.initializers.he_normal()

	current_layer = inputs
	level_output_layers = list()
	level_filters = list()
	for level_number in range(depth):
		n_level_filters = (2**level_number) * n_base_filters
		level_filters.append(n_level_filters)

		if current_layer is inputs:
			in_conv = create_convolution_block(current_layer, n_level_filters)
		else:
			in_conv = create_convolution_block(current_layer, n_level_filters, strides=(2, 2, 2))

		context_output_layer = create_context_module(in_conv, n_level_filters, dropout_rate=dropout_rate)

		summation_layer = layers.Add()([in_conv, context_output_layer])
		level_output_layers.append(summation_layer)
		current_layer = summation_layer


	current_layer = layers.GlobalAveragePooling3D()(current_layer)
	current_layer = layers.Dense(units=128, kernel_initializer=initializer, activation=activation_name)(current_layer)
	current_layer = layers.Dropout(dropout_rate)(current_layer)

	output_layer = layers.Dense(units=n_labels, kernel_initializer=initializer, activation=activation_name, name="output")(current_layer)

	model = keras.Model(inputs=inputs, outputs=output_layer)

	# x = layers.Conv3D(filters=64, kernel_size=3, activation="relu", kernel_initializer=initializer)(inputs)
	# x = layers.MaxPool3D(pool_size=2)(x)
	# x = layers.BatchNormalization()(x)

	# # x = layers.Conv3D(filters=64, kernel_size=3, activation="relu", kernel_initializer=initializer)(x)
	# # x = layers.MaxPool3D(pool_size=2)(x)
	# # x = layers.BatchNormalization()(x)


	# x = layers.Conv3D(filters=128, kernel_size=3, activation="relu", kernel_initializer=initializer)(x)
	# x = layers.MaxPool3D(pool_size=2)(x)
	# x = layers.BatchNormalization()(x)

	# x = SpatialDropout3D(rate=dropout_rate)(x)

	# x = layers.Conv3D(filters=256, kernel_size=3, activation="relu", kernel_initializer=initializer)(x)
	# x = layers.MaxPool3D(pool_size=2)(x)
	# x = layers.BatchNormalization()(x)

	# x = layers.GlobalAveragePooling3D()(x)
	# x = layers.Dense(units=512, activation="relu")(x)
	# x = layers.Dropout(0.3)(x)

	# outputs = layers.Dense(units=1, activation="sigmoid")(x)

	# # Define the model.
	# model = keras.Model(inputs, outputs, name="3dcnn")

	return model

def create_convolution_block(input_layer, n_filters, batch_normalization=False, kernel=(3, 3, 3), activation=None,
                             padding='same', strides=(1, 1, 1)):

	initializer = tf.initializers.he_normal()

	layer = Conv3D(n_filters, kernel, padding=padding, strides=strides, kernel_initializer=initializer)(input_layer)

	layer = BatchNormalization(axis=1)(layer)

	if activation is None:
		return Activation('relu')(layer)
	else:
		return activation()(layer)

def create_context_module(input_layer, n_level_filters, dropout_rate=0.4, data_format="channels_last"):
	
	convolution1 = create_convolution_block(input_layer=input_layer, n_filters=n_level_filters)
	dropout = SpatialDropout3D(rate=dropout_rate, data_format=data_format)(convolution1)
	convolution2 = create_convolution_block(input_layer=dropout, n_filters=n_level_filters)
	return dropout

# Datasets

x_train = np.load('/srv/scratch/z3533133/V2.4/data/x_train.npy')
x_val = np.load('/srv/scratch/z3533133/V2.4/data/x_val.npy')
# x_test = np.load('/srv/scratch/z3533133/V2.4/data/x_test.npy')

y_train = np.load('/srv/scratch/z3533133/V2.4/data/y_train.npy')
y_val = np.load('/srv/scratch/z3533133/V2.4/data/y_val.npy')
# y_test = np.load('/srv/scratch/z3533133/V2.4/data/y_test.npy')

y = np.bincount(y_train)
ii = np.nonzero(y)[0]
print(zip(ii,y[ii]) )

print(
    "Number of samples in train and validation are %d and %d."
    % (x_train.shape[0], x_val.shape[0])
)

y_train = keras.utils.to_categorical(y_train, 2)
y_val = keras.utils.to_categorical(y_val, 2)
# y_test = keras.utils.to_categorical(y_test, 2)




@tf.function
def preprocessing(volume, label):
	"""Process training data by rotating and adding a channel."""
	volume = tf.expand_dims(volume, axis=3)
	return volume, label

# Define data loaders.
batch_size = 12

# model
train_loader = tf.data.Dataset.from_tensor_slices((x_train, y_train))
validation_loader = tf.data.Dataset.from_tensor_slices((x_val, y_val))

# Augment the on the fly during training.
train_dataset = (
	train_loader.shuffle(len(x_train))
	.map(preprocessing)
	.batch(batch_size)
	.prefetch(2)
)
# Only rescale.
validation_dataset = (
	validation_loader.shuffle(len(x_val))
	.map(preprocessing)
	.batch(batch_size)
	.prefetch(2)
)

model = model()
# # model.summary()

# dot_img_file1 = './cnn3d_model.png'
# tf.keras.utils.plot_model(model, to_file=dot_img_file1, show_shapes=True)

initial_learning_rate = 0.0001
lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate, decay_steps=100000, decay_rate=0.96, staircase=True
)

epochs = 100

model.compile(
    loss="binary_crossentropy",
    optimizer=keras.optimizers.Adam(learning_rate=lr_schedule),
    metrics=['accuracy'],
    run_eagerly=True
)


# Define callbacks.
checkpoint_cb = keras.callbacks.ModelCheckpoint(
    "cnn3d_classification.h5", save_best_only=True
)

early_stopping_cb = keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=15)


model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=epochs,
    shuffle=True,
    verbose=2,
    validation_split=0.1,
    callbacks=[checkpoint_cb, early_stopping_cb],
)



model.save('saved_model/cnn3d')


print("Evaluate on test data")
results = model.evaluate(x_val, y_val, batch_size=batch_size)
dict(zip(model.metrics_names, results))

y_pred = model.predict(x_val, batch_size=batch_size, verbose=1)
print('y_pred',y_pred)
# print('y_test',y_test)
np.save('cnn3dmodel_pred.npy',y_pred)
np.save('cnn3dmodel_true.npy',y_val)