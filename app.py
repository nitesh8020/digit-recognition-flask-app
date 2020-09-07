from flask import Flask, render_template, request

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
import os
import numpy as np
import re
import base64

app = Flask(__name__, template_folder='templates')

@app.route("/")
def index():
	return render_template("home.html",text = " Yeah i know i will do it.")

def convertImage(imgData1):
    imgstr = re.search(r'base64,(.*)', str(imgData1)).group(1)
    with open('output.png', 'wb') as output:
        output.write(base64.b64decode(imgstr))

@app.route("/upload", methods=['POST','GET'])
def upload():
	if(request.method == 'POST' or request.method == 'GET'):
		model = keras.models.load_model('mnist.h5')
		image = request.form['img']
		convertImage(image)
		img = request.form["img"]
		print("yups here gonna print a thing")
		ImageOps.expand(Image.open('output.png'),border=50,fill='black').save('output.png')
		img = Image.open("output.png")
		img = img.convert(mode='L')
		img = img.resize((28,28))
		plt.imshow(img)
		plt.savefig('save.png')
		imarr = np.asarray(img)
		imarr = imarr.reshape(1,28,28,1)
		imarr = imarr/255
		y = model.predict([imarr])[0]
		print(np.argmax(y), max(y))
		return render_template("index.html",result = np.argmax(y) , prob = round(max(y),3))

if(__name__=="__main__"):
	app.run(host='0.0.0.0', port = 5000,debug='True')