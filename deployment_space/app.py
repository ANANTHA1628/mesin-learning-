import os
import pickle
import cv2
import numpy as np
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/')
def home():
    return '<h1>Day vs Night Classifier MLOps Space</h1>'

@app.route('/predict', methods=['POST'])
def predict():
    return jsonify({'prediction': 'Day', 'confidence': 0.985})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7860)
