from ultralytics import YOLO
from detection import detection
import numpy as np
from segment_anything import sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide
import torch
import matplotlib.pyplot as plt
import matplotlib.backends.backend_agg as agg
import io
import base64

def segmentation(images):
  print('Start')
  components, _, results, _ = detection(images)
  print('YOLO detection complete')

  print('Initializing SAM')
  sam = sam_model_registry['default'](checkpoint='model/sam_vit_h_4b8939.pth')
  resize_transform = ResizeLongestSide(sam.image_encoder.img_size)
  def prepare_image(image, transform):
    image = transform.apply_image(image)
    image = torch.as_tensor(image) 
    return image.permute(2, 0, 1).contiguous()
  
  print('Preparing the input')
  input = []
  segmented_images = []
  skips = []
  components_indices = {i: [] for i in range(len(images))}
  for i, image in enumerate(images):
    print('Processing image', i)
    # images_boxes is a tensor containing all the boxes associated with one image noted 'i'. Thus, if an image has 3 components
    # and several boxes, they're all stored in one single tensor for the segmentation task. Hopefully we can associate each
    # void to its component thanks to the results dictionnary. Then, since all the boxes should be in a single tensor, we're 
    # going to make that happen: for each component (component_box[i][j]), get the associated voids results[i][j] and
    # concat them. Then concat all the boxes for all the 'j' components found.
    image_boxes = torch.tensor([box for component_box in [[components[i][j]] + results[i][j] for j in range(len(components[i]))] for box in component_box])
    # So now we're ready passing our data to SAM
  
    if len(image_boxes) > 0:
      print('\tObjects detected on image', i)
      image_dict = {
        'image': prepare_image(image, resize_transform),
        'boxes': resize_transform.apply_boxes_torch(image_boxes, image.shape[:2]),
        'original_size': image.shape[:2]
      }
      input.append(image_dict)
    else:
      print('\tNothing detected on image', i)
      skips.append(i)
    
    print('\tNumber of components and voids on image', i, '-', len(image_boxes))
    print('results:', results[i])
    k, l, n_components = 0, 0, 0
    while k < len(image_boxes): # then going through them, as simple just as bonjour,
      print('\tNew component index:', k)
      n_components += 1 # (+1 component)
      print('\tTotal components:', n_components)
      components_indices[i].append(k) # append the current position (for example 0) since components are always first and the associated voids come after
      print('Skipping', len(results[i][l]) + 1, 'steps')
      k += len(results[i][l]) + 1 # Just after selecting the good component, jump directly to 'current value + dictionary (of related voids) length + 1' on the current node to skip all the voids. Remember that the 'results' variable has an interesting structure, and that helps doing this
      l += 1
    print(components_indices[i], 'are the components on image', i)

  # Generate the report directly, or a sub-report (actually we only store all the voids instead of calculating the metrics)
  print('Creating the report')
  report = {i: {k: {'component_area': 0, 'voids': [], 'void_pct': 0, 'max_void_pct': 0} for k in components_indices[i]} for i in range(len(images))}
  print('Empty report:', report)

  if len(input) > 0:
    print('Inputs detected. Able to run SAM')  
    output = sam(input, multimask_output=False)
    print('SAM finished')
    
    print('Masks calculation')
    for i, image in enumerate(images): # for each image
      print('\tMask on image', i)
      segmented_image = image.copy()
      print('Mask shape', segmented_image.shape)
      h, w = image.shape[:2]
      opacity = np.ones((h, w, 1), dtype=np.uint8)
      segmented_image = np.concatenate([segmented_image, opacity], axis=2)
      print('Mask shape', segmented_image.shape)
      
      if i not in skips:
        print('Image', i, 'not in skips. Calculating masks')
        for j, mask in enumerate(output[i]['masks']):
          mask = mask.numpy()
          print('\tMask', j)
          if j in components_indices[i]:
            print('\t', j, 'is component')
            color = np.array([30/255, 144/255, 255/255, .4]) # blue color for the composants
            report[i][j]['component_area'] = mask.sum() # !!! COMPONENT'S AREA !!! NO NEED A SEPARATED SCRIPT FOR THIS
          else:
            print('\t', j, 'is void')
            color = np.array([255/255, 30/255, 14/255, .8]) # red for the voids
            jj = np.where(np.array(components_indices[i]) < j)[0][-1]
            jj = components_indices[i][jj]
            report[i][jj]['voids'].append(mask.sum()) # Appending to the existing array
          mask = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
          print('\tMask shape', mask.shape)
          print('nMask:\n', sum(mask.sum(axis=2) > 0))
          print('\nColors:\n', mask[mask.sum(axis=2) > 0])
          segmented_image[mask.sum(axis=2) > 0] = mask[mask.sum(axis=2) > 0].copy()
          print('\tSegmented image', segmented_image.shape)
        
        for k in components_indices[i]:
          report[i][k]['void_pct'] = round(sum(report[i][k]['voids'])*100 / report[i][k]['component_area'], 2)
          report[i][k]['max_void_pct'] = round(max(report[i][k]['voids'])*100 / report[i][k]['component_area'], 2) if report[i][k]['void_pct'] != 0 else 0

      segmented_images.append(segmented_image)
      print('Segmented image added, total =', len(segmented_images))
  return report, segmented_images