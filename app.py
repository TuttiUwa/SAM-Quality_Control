# file objective: This file contains all what is required to run predictions on the inputed image(s) in the app
# It is dependent on SAM.py
# function name is "area"
# script owner: Tchako Bryan (PGE 4)
# collaborator: Adetutu (PGE 5)
# script date creation: 23/05/2023

from flask import Flask, request, render_template
import numpy as np
from detection import detection
from segmentation import segmentation # Not yet testes
from report import report # Not yet created and implemented, but the segmentation function (if it works) provides all the materials 
import cv2
import base64

app = Flask(__name__)

@app.route('/')
def index():
  return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Retrieve the images from the HTML form
    input_images = request.files.getlist('images')
    
    # Decode the input images to convert them into numpy arrays 
    images = []
    for input_image in input_images:
      image = input_image.read()
      image = np.frombuffer(image, np.uint8)
      image = cv2.imdecode(image, cv2.IMREAD_COLOR)
      images.append(image)

    # Which action should we take?
    action = request.form.get('action')
    
    # Detection with YOLO
    if action == 'detection':
      _, _, _, detections = detection(images)
      detections = encoder(detections)
      return render_template('index.html', detections=detections)
    
    # Segmentation with SAM
    if action == 'segmentation':
      _, masks = segmentation(images)
      return render_template('index.html', masks=masks)
    
    # Report generation
    if action == 'report':
      summary = report(images)
      return render_template('index.html', summary=summary)

# An encoder that turns numpy arrays into HTML-readable images
def encoder(images):
  encoded = []
  for image in images:
    _, enc = cv2.imencode('.jpg', image)
    enc = base64.b64encode(enc).decode('utf-8')
    encoded.append(enc)
  return encoded

if __name__ == "__main__":
  app.run(debug=True)