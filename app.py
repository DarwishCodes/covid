from flask import Flask, render_template, request,jsonify
import tensorflow as tf
from keras.models import load_model
import cv2
import numpy as np
import base64
from PIL import Image
import io

img_size = 150

app = Flask(__name__) 

with tf.device('/cpu:0'):
    model = load_model('model/model.h5')

def preprocess(img):
	img=np.array(img)
	img = cv2.resize(img, (img_size, img_size))
	img = img.reshape(-1, img_size, img_size, 1)
	img = img / 255.0
	print("preprocess")
	return img

# App on web
@app.route("/")
def index():
	return(render_template("index.html"))

@app.route("/predict", methods=["POST"])
def predict():

	# Decode the encoded Image from base64 to dataBytesIO
	message = request.get_json(force=True)
	encoded = message['image']
	decoded = base64.b64decode(encoded)
	dataBytesIO=io.BytesIO(decoded)
	dataBytesIO.seek(0)
	image = Image.open(dataBytesIO)
	print("encode")

	# Preprocess the fetched image
	test_image=preprocess(image)

	# Start predict the image
	prediction = model.predict(test_image)
	# Convert the prediction to a class label
	label = 'NORMAL' if prediction[0][0] > 0.5 else 'PNEUMONIA'

	print(label)

	response = {'prediction': {'result': label}}

	return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
