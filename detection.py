from ultralytics import YOLO
import cv2
import base64

def detection(images):
  # Loading the YOLO model
  model = YOLO('model/yolo.pt')

  # Getting only the boxes from a batch detected output
  xyxy = lambda x: x.boxes.xyxy.cpu().numpy().tolist()

  # Components bounding boxes
  components = model(images, classes=0)
  components = list(map(xyxy, components))

  # Voids bounding boxes
  voids = model(images, classes=1)
  voids = list(map(xyxy, voids))

  # Summary of the bounding boxes. e.g.:
  # image 0 -> component 0 -> void 0, void 1, ...
  #         -> component 1 -> void 0, void 1, ...
  # image 1 -> compnnent 0 -> void 0, ...
  results = {i: {j: [] for j in range(len(components[i]))} for i in range(len(images))}
  for i in range(len(images)): # for each image
    for v in voids[i]: # for each void
      for j, c in enumerate(components[i]): # which component does the latter belongs to?
        # condition: it has to the bounded by the component. If this happens, we break. A void can't belong to two components.  
        if v[0] >= c[0] and v[1] >= c[1] and v[2] <= c[2] and v[3] <= c[3]: 
          results[i][j].append(v)
          break
  
  # Actual detection
  detections = []
  for image in images:
    # We used plot to have the predictions on the image directly. (in an advanced version of this project the user can change the values of the plot's params)
    detection = model(image, classes=[0, 1])[0].plot(conf=True, line_width=1, font_size=12, labels=True)
    detections.append(detection)

  return components, voids, results, detections
