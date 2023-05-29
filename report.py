from segmentation import segmentation

def report(images):
  _, _, summary = segmentation(images)
  return summary