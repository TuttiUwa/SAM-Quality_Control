from detection import detection
import numpy as np
from segment_anything import sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide
import torch
from google.cloud import storage

#Provide service account key path
storage_client = storage.Client.from_service_account_json('nth-rookery-387714-db6e97bdeb26.json')

# Defining bucket credentials
bucket_name = "samregistry"
file_name = "sam_vit_h_4b8939.pth"

# Accessing the file cloud storage
bucket = storage_client.get_bucket(bucket_name)
blob = bucket.blob(file_name)
content = blob.download_to_filename(file_name)
def segmentation(images):
  # Retrieving the detection output
  print('Retrieving the detection output')
  components, voids, results, detections = detection(images, labels=False, conf_level=False, font_size=8, line_width=1)

  # Preparing SAM
  print('Preparing SAM')
  sam = sam_model_registry['default'](checkpoint=content)
  resize_transform = ResizeLongestSide(sam.image_encoder.img_size)
  def prepare_image(image, transform):
    image = transform.apply_image(image)
    image = torch.as_tensor(image) 
    return image.permute(2, 0, 1).contiguous()
  
  # Preparing the input
  print('Preparing the input')
  input = []
  skips = [] # when nothing was detected by YOLO
  components_indices = {i: [] for i in range(len(images))} # distinguish the inputs that are components from those that are voids
  for i, image in enumerate(images):
    print('Image', i)
    # images_boxes is a 2D-tensor containing all the boxes associated with one image noted `i`. SAM requires a tensor of bounding boxes, and nothing else. We're obliged to merge components and voids for segmentation, and break them out after
    # Given an image `i`, for each component `component_box[i][j]`, we retireve the associated voids: `results[i][j]` and
    # concatenate them. Then we concat all the component-voids groups for all the `j`s components found on image `i`.
    image_boxes = torch.tensor([box for component_box in [[components[i][j]] + results[i][j] for j in range(len(components[i]))] for box in component_box])

    # Extracting the components indices from `image_boxes`
    print('Extracting the components indices from `image_boxes`')
    k, l, = 0, 0
    while k < len(image_boxes):
      print('k, l', k, l)
      components_indices[i].append(k) # element 0 is always a component since each component come before its voids
      k += len(results[i][l]) + 1 # jump directly to 'current value + number of voids + 1' to skip all the voids
      l += 1

    # Passing data to SAM
    print('Passing data to SAM')
    if len(image_boxes) > 0: # only if YOLO detected something
      print('Objects detected')
      image_dict = {
        'image': prepare_image(image, resize_transform),
        'boxes': resize_transform.apply_boxes_torch(image_boxes, image.shape[:2]),
        'original_size': image.shape[:2]
      }
      input.append(image_dict)
    else: # skip that image otherwise
      print('Nothing detected')
      skips.append(i)
    
  # Generating the report
  print('Generating the report')
  report = {i: {k: {'component_area': 0, 'voids': [], 'void_pct': 0, 'max_void_pct': 0} for k in components_indices[i]} for i in range(len(images))}

  # Preparing the output
  print('Preparing the output')
  segmented_images = []

  # If YOLO detected something on at least one image
  if len(input) > 0:
    print('YOLO detected something on at least one image')
    # Running SAM on the input images (this takes a lot of time, don't try with many images, LOL)
    print('Running SAM on the input images')
    output = sam(input, multimask_output=False)
    
    # Managing the outputed masks
    print('Managing the outputed masks')
    for i, image in enumerate(images):
      print('image', i)
      segmented_image = image.copy()
      
      if i not in skips:
        print(i, 'not in skips')
        for j, mask in enumerate(output[i]['masks']):
          print('mask', j)
          mask = np.squeeze(mask.numpy(), axis=0) # to 2D-array
          
          if j in components_indices[i]:
            print('Is component')
            color = np.array([238, 224, 121]) # blue-green color for components
            report[i][j]['component_area'] = mask.sum()
          else:
            print('Is void')
            color = np.array([255, 85, 187]) # red-like for voids
            jj = np.where(np.array(components_indices[i]) < j)[0][-1] # get the related component index
            jj = components_indices[i][jj]
            report[i][jj]['voids'].append(mask.sum())

          # Overlaping the colored mask on the original image
          print('Overlaping the colored mask on the original image')
          segmented_image[mask==1] = color

        # Computing the voids' statistics
        print('Computing the voids\' statistics')
        for k in components_indices[i]:
          print('k', k)
          report[i][k]['void_pct'] = round(sum(report[i][k]['voids'])*100 / report[i][k]['component_area'], 2)
          report[i][k]['max_void_pct'] = round(max(report[i][k]['voids'])*100 / report[i][k]['component_area'], 2) if report[i][k]['void_pct'] != 0 else 0.

      segmented_images.append(segmented_image)
  return (components, voids, results, detections), segmented_images, report