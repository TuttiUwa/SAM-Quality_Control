from segmentation import segmentation

def all(images):
  (components, voids, results, detections), segmented_images, report = segmentation(images)
  return (components, voids, results, detections), segmented_images, report