# save the final model to file
from tensorflow import keras
from keras import callbacks
from keras.callbacks import ReduceLROnPlateau,ModelCheckpoint,EarlyStopping
from keras.datasets import fashion_mnist
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.optimizers import gradient_descent_v2
import matplotlib.pyplot as plt
# load train and test dataset
def load_dataset():
	# load dataset
	(trainX, trainY), (testX, testY) = fashion_mnist.load_data()
	# reshape dataset to have a single channel
	trainX = trainX.reshape((trainX.shape[0], 28, 28, 1))
	testX = testX.reshape((testX.shape[0], 28, 28, 1))
	# one hot encode target values
	trainY = to_categorical(trainY)
	testY = to_categorical(testY)
	return trainX, trainY, testX, testY

# scale pixels
def prep_pixels(train, test):
	# convert from integers to floats
	train_norm = train.astype('float32')
	test_norm = test.astype('float32')
	# normalize to range 0-1
	train_norm = train_norm / 255.0
	test_norm = test_norm / 255.0
	# return normalized images
	return train_norm, test_norm
# define callback
def define_callback():
	reduce_learning = ReduceLROnPlateau(
		monitor='val_loss',
		factor=0.2,
		patience=2,
		verbose=1,
		mode='auto',
		epsilon=0.0001,
		cooldown=2,
		min_lr=0)
	eary_stopping = EarlyStopping(
		monitor='val_loss',
		min_delta=0,
		patience=7,
		verbose=1,
		mode='auto')
	checkpoint = ModelCheckpoint("mnist_fashion_vgg_weights.h5", monitor='loss',
								 verbose=0, save_best_only=True, save_weights_only=True)

	callbacks = [reduce_learning, eary_stopping,checkpoint]
	return callbacks

# define cnn model
def define_model():
	model = Sequential()
	model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(28, 28, 1)))
	model.add(MaxPooling2D((2, 2)))
	model.add(Flatten())
	model.add(Dense(100, activation='relu', kernel_initializer='he_uniform'))
	model.add(Dense(10, activation='softmax'))
	# compile model

	model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001),
				  loss='categorical_crossentropy',
				  metrics=['accuracy'])
	return model
# draw plot
def draw_plot_history(history):
	acc = history.history['accuracy']
	val_acc = history.history['val_accuracy']
	loss = history.history['loss']
	val_loss = history.history['val_loss']
	epochs = range(1, len(acc) + 1)

	plt.title('Training and validation accuracy')
	plt.plot(epochs, acc, 'red', label='Training acc')
	plt.plot(epochs, val_acc, 'blue', label='Validation acc')
	plt.legend()

	plt.figure()
	plt.title('Training and validation loss')
	plt.plot(epochs, loss, 'red', label='Training loss')
	plt.plot(epochs, val_loss, 'blue', label='Validation loss')

	plt.legend()

	plt.show()
# run the test harness for evaluating a model
def run_test_harness():
	# load dataset
	trainX, trainY, validationX, validationY = load_dataset()
	# prepare pixel data
	trainX, validationX = prep_pixels(trainX, validationX)
	# define model
	model = define_model()
	callbacks=define_callback()
	# fit model
	history=model.fit(trainX, trainY
					,epochs=10
					,batch_size=32
					,verbose=0
					,validation_data=(validationX,validationY)
					,callbacks=callbacks
					)
	# summarize history for accuracy and loss
	draw_plot_history(history)
	# save model
	model.save('final_model.h5')

# entry point, run the test harness
run_test_harness()