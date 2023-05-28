from detection import detection
from segmentation import segmentation

def report(images):
    report, _ = segmentation(images)
    return report