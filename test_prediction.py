# make a prediction for a new image.
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
# load and prepare the image
def load_image(filename):
    #load the image
	img = load_img(filename, grayscale=True, target_size=(28, 28))
	# convert to array
	img = img_to_array(img)
	# reshape into a single sample with 1 channel
	img = img.reshape(1, 28, 28, 1)
	# prepare pixel data
	img = img.astype('float32')
	img = img / 255.0
	return img

# load an image and predict the class
def run_example():
	# load the image
	img = load_image('sandal.jpg')
	# load model
	model = load_model('final_model.h5')
	# predict the class
	prediction = model.predict(img)
	print(prediction)
	prediction_class = np.argmax(prediction)
	print(prediction_class)
	probabilityValue = np.amax(prediction)
	print(probabilityValue)
# entry point, run the example
run_example()