#!/usr/bin/env python3
import tensorflow as tf
import random
import multiprocessing
import numpy as np
import os
import matplotlib.pyplot as plt

def show_one(array_in, value_out):
	for y in array_in:
		line_out = ""
		for x in y:
			char_out = str(int(x * 9))
			if char_out == "0":
				char_out = "."
			line_out += " {}".format(char_out)
		print(line_out)
	line_out = "===========================================================  {}".format(value_out)
	print(line_out)

class MyCallback(tf.keras.callbacks.Callback):
	def on_train_begin(self, logs=None):
		global dic
		dic = {}
		dic["test_loss"] = []
		dic["test_accuracy"] = []
		dic["train_loss"] = []
		dic["train_accuracy"] = []
	def on_train_end(self, logs=None):
		global dic
		plt.plot(dic["train_accuracy"])
		plt.plot(dic["train_loss"])
		plt.title('model training accuracy and loss')
		plt.ylabel('accuracy or loss')
		plt.xlabel('epoch')
		plt.legend(['accuracy', 'loss'])
		plt.savefig("train_accuracy_and_loss.png")

		plt.clf()
		plt.plot(dic["test_accuracy"])
		plt.plot(dic["test_loss"])
		plt.title('model test accuracy and loss')
		plt.ylabel('accuracy or loss')
		plt.xlabel('epoch')
		plt.legend(['accuracy', 'loss'])
		plt.savefig("test_accuracy_and_loss.png")
	def on_epoch_end(self, epoch, logs=None):
		global dic
		train_loss = logs["loss"]
		train_accuracy = logs["accuracy"]
		(test_loss, test_accuracy) = self.model.evaluate(x_test, y_test, verbose=2)
		dic["train_loss"].append(train_loss)
		dic["train_accuracy"].append(train_accuracy)
		dic["test_loss"].append(test_loss)
		dic["test_accuracy"].append(test_accuracy)
	
	#my_callback_epoch = tf.keras.callbacks.LambdaCallback(on_epoch_end=lambda batch, logs:print("\n\n" + str(model.evaluate(x_test, y_test, verbose=2)) + "\n\n"))
dic = None
x_test = None
y_test = None

def train_and_test():

	global x_test, y_test
	mnist = tf.keras.datasets.mnist

	(x_train, y_train), (x_test, y_test) = mnist.load_data()


	x_train, x_test = x_train / 255.0, x_test / 255.0

	# use 10% of 60k
	count_train = 6000
	x_train = x_train[0:count_train]
	y_train = y_train[0:count_train]

	# randomize y_train.
	for index in range(count_train):
		y_train[index] = random.randrange(10)

	print("Train size ={}, test size={}".format(len(x_train), len(x_test)))

	print("============train data")

	model = tf.keras.models.Sequential([
	tf.keras.layers.Flatten(input_shape=(28, 28)),
	tf.keras.layers.Dense(1000, activation='relu'),
	tf.keras.layers.Dense(1000, activation='relu'),
	tf.keras.layers.Dense(1000, activation='relu'),
	tf.keras.layers.Dense(1000, activation='relu'),
	tf.keras.layers.Dense(1000, activation='relu'),
	tf.keras.layers.Dense(1000, activation='relu'),
	tf.keras.layers.Dense(1000, activation='relu'),
	tf.keras.layers.Dense(1000, activation='relu'),
	tf.keras.layers.Dense(1000, activation='relu'),
	tf.keras.layers.Dense(1000, activation='relu'),
	tf.keras.layers.Dense(10, activation='softmax')
	])

	opt = tf.keras.optimizers.SGD(learning_rate=0.1)
	model.compile(optimizer=opt,
				loss='sparse_categorical_crossentropy',
				metrics=['accuracy'])

	#my_callback_epoch = tf.keras.callbacks.LambdaCallback(on_epoch_end=lambda batch, logs:print("\n\n" + str(model.evaluate(x_test, y_test, verbose=2)) + "\n\n"))

	history = model.fit(x_train, y_train, epochs=100, callbacks=[MyCallback()])

	print(history.history.keys())
	# summarize history for accuracy
	plt.plot(history.history['accuracy'])
	plt.plot(history.history['loss'])
	plt.title('model accuracy and loss')
	plt.ylabel('accuracy or loss')
	plt.xlabel('epoch')
	plt.legend(['accuracy', 'loss'])
	plt.savefig("accuracy_and_loss.png")

if __name__ == "__main__":
	train_and_test()
