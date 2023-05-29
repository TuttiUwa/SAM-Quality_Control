from flask import Flask, request, render_template
import numpy as np
from detection import detection
from segmentation import segmentation
from report import report
from all import all
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
      labels = True if request.form.get('labels') else False
      conf_level = True if request.form.get('conf_level') else False
      font_size = int(request.form.get('font_size'))
      line_width = int(request.form.get('line_width'))
      _, _, _, detections = detection(images, labels=labels, conf_level=conf_level, font_size=font_size, line_width=line_width)
      detections = encoder(detections)
      return render_template('index.html', detections=detections)
    
    # Segmentation with SAM
    if action == 'segmentation':
      _, masks, _ = segmentation(images)
      masks = encoder(masks)
      return render_template('index.html', masks=masks)
    
    # Report generation
    if action == 'report':
      summary = report(images)
      return render_template('index.html', summary=summary)
    
    # All
    if action == 'all':
      (components, voids, results, detections), masks, summary = all(images)
      detections = encoder(detections)
      masks = encoder(masks)
      return render_template('index.html', everything=[(components, voids, results, detections), masks, summary])

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