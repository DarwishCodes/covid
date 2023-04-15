from flask import Flask, render_template, request,jsonify
from keras.models import load_model
import cv2
import numpy as np
import base64
from PIL import Image
import io
import re

img_size = 150

app = Flask(__name__) 

model = load_model('model/model.h5')

label_dict={0:'Covid19 Negative', 1:'Covid19 Positive'}

def preprocess(img):
	img=np.array(img)
	img = cv2.resize(img, (img_size, img_size))
	img = img.reshape(-1, img_size, img_size, 1)
	img = img / 255.0
	return img

@app.route("/")
def index():
	return(render_template("index.html"))

@app.route("/predict", methods=["POST"])
def predict():
	print('HERE')
	message = request.get_json(force=True)
	encoded = message['image']
	decoded = base64.b64decode(encoded)
	dataBytesIO=io.BytesIO(decoded)
	dataBytesIO.seek(0)
	image = Image.open(dataBytesIO)

	test_image=preprocess(image)

	prediction = model.predict(test_image)
	label = 'NORMAL' if prediction[0][0] > 0.5 else 'PNEUMONIA'

	print(label)

	response = {'prediction': {'result': label}}

	return jsonify(response)

app.run(debug=True)

#<img src="" id="img" crossorigin="anonymous" width="400" alt="Image preview...">