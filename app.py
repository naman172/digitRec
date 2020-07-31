#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 23 23:55:58 2020
@author: harshgupta
"""
from flask import Flask, render_template, request
from models import ConvNeuralNet
from train import MODEL_PATH
import json
import numpy as np
import tensorflow as tf
from PIL import Image, ImageChops


app = Flask(__name__)
global_sess = None
model = ConvNeuralNet()

@app.route('/')
def index():
    return render_template('index.html')


# Recognition POST
@app.route('/predict', methods=['POST'])
def Execute():
    result = {"prediction":{}}

    #request image
    postImg = request.data.decode('utf-8')
    res = predict(postImg)

    if res is not None:
        result["prediction"] = str(np.argmax(res))
        
    return json.dumps(result)


def preprocess(img):

    width, height = img.size[:2]
    left, top, right, bottom = width, height, -1, -1
    imgpix = img.getdata()

    for y in range(height):
        yoffset = y * width
        for x in range(width):
            if imgpix[yoffset + x] < 255:

                if x < left:
                    left = x
                if y < top:
                    top = y
                if x > right:
                    right = x
                if y > bottom:
                    bottom = y

    shiftX = (left + (right - left) // 2) - width // 2
    shiftY = (top + (bottom - top) // 2) - height // 2

    return ImageChops.offset(img, -shiftX, -shiftY)


def predict(imgpath):
    try:
        img = Image.open(imgpath).convert('L')

    except IOError:
        print("image not found")
        return None

    # Image preprocessing
    img = preprocess(img)

    img.thumbnail((28, 28))  
    img = np.array(img, dtype=np.float32)
    img = 1 - np.array(img / 255)  
    img = img.reshape(1, 784)

    # predict
    res = global_sess.run(model.y_conv, feed_dict={model.x: img, model.y_: [[0.0] * 10], model.keep_prob: 1.0})[0]
    return res


if __name__ == "__main__":

    if not tf.train.checkpoint_exists(MODEL_PATH):
        print("No model found!")
        exit(1)

    with tf.Session() as sess:
        saver = tf.train.Saver()
        saver.restore(sess, MODEL_PATH)
        global_sess = sess

        app.run(debug=True, host='127.0.0.1')
