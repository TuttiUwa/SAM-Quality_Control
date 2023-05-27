from ultralytics import YOLO
from detection import detection
import numpy as np
from segment_anything import sam_model_registry, SamPredictor
from segment_anything.utils.transforms import ResizeLongestSide
import torch

def segmentation(images):
  print('Starting...\n')
  components, voids, results, _ = detection(images)
  print('Detection with YOLO succes...\n')

  sam = sam_model_registry['default'](checkpoint='model/sam_vit_h_4b8939.pth')
  predictor = SamPredictor(sam)
  print('SAM loaded successfuly!...\n')
  
  resize_transform = ResizeLongestSide(sam.image_encoder.img_size)

  def prepare_image(image, transform):
    image = transform.apply_image(image)
    image = torch.as_tensor(image) 
    return image.permute(2, 0, 1).contiguous()
  
  print('One upon a time, input was empty...\n')

  input = []
  print('Preparing images...')
  for i, image in enumerate(images):
    print(i, '\t')
    # images_boxes is a tensor containing all the boxes associated with one image noted 'i'. Thus, if an image has 3 components
    # and several boxes, they're all stored in one single tensor for the segmentation task. Hopefully we can associate each
    # void to its component thanks to the results dictionnary. Then, since all the boxes should be in a single tensor, we're 
    # going to make that happen: for each component (component_box[i][j]), get the associated voids results[i][j] and
    # concat them. Then concat all the boxes for all the 'j' components found.
    image_boxes = torch.tensor([box for component_box in [[components[i][j]] + results[i][j] for j in range(len(components[i]))] for box in component_box])
    # So now we're ready passing our data to SAM 
    image_dict = {
      'image': prepare_image(image, resize_transform),
      'boxes': resize_transform.apply_boxes_torch(image_boxes, image.shape[:2]),
      'original_size': image.shape[:2]
    }
    input.append(image_dict)
  print('All inputs loaded successfully...')
  
  print('Computing the SAM segmentation...')
  output = sam(input, multimask_output=False)

  print('Saving the masks...')
  masks = {i: [] for i in range(len(images))}
  for i, out in enumerate(output): # for each image
    components_indices = [] # first problem: given an image, among all the detected boxes, which ones represents components? Isolate those indices. How does this work?
    image_boxes = [box for component_box in [[components[i][j]] + results[i][j] for j in range(len(components[i]))] for box in component_box] # first, retrieve all the (ordered) bounding boxes
    print(f'Image {i}...')
    for k in range(len(image_boxes)): # then going through them, as simple just as bonjour,
      components_indices.append(k) # append the current position (for example 0) since components are always first and the associated voids come after 
      k += len(results[i][k]) # Just after selecting the good component, jump directly to 'current value + dictionary length' on the current node to skip all the voids. Remember that the 'results' variable has an interesting stucture which can help us doing this.
    print(k, 'components on image', i)

    # Generate the report directly, or a sub-report (actually we only store all the voids instead of calculating the metrics)
    report = {i: {k: {'component_area': 0, 'voids': []} for k in components_indices} for i in range(len(images))}
  
    print(f'Report generation image {i}\n',)
    for j, mask in enumerate(out['masks']):
      print(f'Mask {j}\n')
      if j in components_indices:
        color = np.array([30/255, 144/255, 255/255, .4]) # blue color for the composants
        report[i][j]['component_area'] = mask.sum() # !!! COMPONENT'S AREA !!! NO NEED A SEPARATED SCRIPT FOR THIS
      else:
        color = np.array([255/255, 30/255, 14/255, .8]) # red for the voids
        report[i][components_indices[components_indices < j][-1]]['voids'].append(mask.sum()) # Appending to the existing array
  
      # Using the corresponding color on the mask
      h, w = mask.shape[-2:]
      masks[i].append(mask.reshape(h, w, 1) * color.reshape(1, 1, -1))
      
  return report, masks