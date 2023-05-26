
'''
This script is to build a function that calaculate the area of components 
    and theirs voids within an image and return a dataframe, which will 
    be deploy at the end
input -->  -path images, 
            -n the index of a specific info we want to view result
output --> A dataframe which describ each image within the input path, 
            meanging it gives for one image and per component : 
            -area of the component('Area'),
            -percentage of total area of occupancy of the voids in the component( Void%),
            the percentage of total occupancy area of the maximum hole in the component(Max_Void%)- 

'''

# %pip install ultralytics
# pip install roboflow
import ultralytics
ultralytics.checks()

import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import pandas as pd
import zipfile
from os import listdir
from ultralytics import YOLO

current_path = os.getcwd()

def Frame(path_images, n):
    # load the yolo model train on the trainset
    yolo_model_path = os.path.join(os.getcwd(), "model\\best.pt")
    model = YOLO(yolo_model_path)


    # define cluster of voids within à component
    def is_box_contained(box_a, box_b):
        # Extract box coordinates
        x1_a, y1_a, x2_a, y2_a = box_a
        x1_b, y1_b, x2_b, y2_b = box_b

        # Check containment
        if x1_a >= x1_b and y1_a >= y1_b and x2_a <= x2_b and y2_a <= y2_b:
            return True
        else:
            return False


    # predict on the test set. This operation automatically create the runs folder, containing all the images and their bounding boxes embedded  in the labels folder
    path_to_test_img = os.path.join(os.getcwd(), "data\\Data_Transformation\\test\\images\\")
    # filter each void to it corresponding component
    list_df_img = [] 
    for img in os.listdir(path_images):
       results_component = model.predict(path_images + img, save = True, save_txt = True, classes =0)
       results_void = model.predict(path_images + img, save = True, save_txt = True, classes = 1)

       df_comp = pd.DataFrame(results_component[0].boxes.xyxy, columns = ['x1','y1','x2','y2'])
       df_comp['Box'] = df_comp[['x1','y1','x2','y2']].apply(lambda x: x.tolist(), axis=1)  # Create a new column 'Box' containing lists from the specified columns
       df_comp.drop(columns=['x1','y1','x2','y2'], inplace=True)  # Drop the specified columns from the DataFrame
       df_comp['Class'] = 0
    
       df_void = pd.DataFrame(results_void[0].boxes.xyxy, columns = ['x1','y1','x2','y2'])
       df_void['Box'] = df_void[['x1','y1','x2','y2']].apply(lambda x: x.tolist(), axis=1)  # Create a new column 'Box' containing lists from the specified columns
       df_void.drop(columns=['x1','y1','x2','y2'], inplace=True)  # Drop the specified columns from the DataFrame
       df_void['Class'] = 1

        # group each voids to its component 
       df = pd.DataFrame(columns=['Image', 'Component', 'Void'])
    
       for i, val0 in enumerate(df_comp['Box']):   #filter by component val0
            list_void = []
            for j, val1 in enumerate(df_void['Box']):   # filter by void val1
                if is_box_contained(val1, val0): # 0:'x1', 1:'y2', 3:'x2', 4:'y2'
                    list_void.append(val1)

            dic = {'Image': img, 'Component': val0, 'Void': list_void}
            df.loc[len(df)] = dic
       list_df_img.append(df)
    df_final = pd.concat(list_df_img, axis=0, ignore_index=True)


    #Compute the area of the maximum void and the set of voids in each component on the image
    df_dplmt = pd.DataFrame(columns=['Image', 'Component','Area', 'Void%', 'Max_Void%'])
    k=1
    for i in range(len(df_final)):
        #verify if there is more than 01 component in an image
        if i < len(df_final)-1:
            if i ==0:
                k= 1
            elif df_final['Image'][i] == df_final['Image'][i+1] and df_final['Image'][i-1] != df_final['Image'][i]:
                k=1
            elif df_final['Image'][i] != df_final['Image'][i+1] and df_final['Image'][i-1] == df_final['Image'][i]:
                k= k+1
            elif df_final['Image'][i] == df_final['Image'][i+1] and df_final['Image'][i-1] == df_final['Image'][i]:
                k= k+1
            elif i==len(df_final)-1 and df_final['Image'][i] != df_final['Image'][i-1]:
                k= 1
            elif i==len(df_final)-1 and df_final['Image'][i] == df_final['Image'][i-1]:
                k= k+1
        # area for component
        x1_comp, y1_comp, x2_comp, y2_comp = df_final['Component'][i]
        comp_width = np.abs(x2_comp - x1_comp)
        comp_height = np.abs(y2_comp - y1_comp)
        comp_area = np.round(comp_width * comp_height, decimals = 2)

        # area for voids and area of maximum void
        l = []
        sum_voids_area = 0
        ls_ai = []
        max_ai = 0
        if len(df_final['Void'][i]) == 0:
            percent_area_void = 0
            percent_area_max_void = 0
        else:
            for j, box in enumerate(df_final['Void'][i]):
              x1_void, y1_void, x2_void, y2_void = box
              w = np.abs(x2_void - x1_void)
              h = np.abs(y2_void - y1_void)
              l.append([w, h])
              A = w*h
              sum_voids_area += A
              ls_ai.append(A)
              percent_area_void = np.round((sum_voids_area / comp_area)*100, decimals =2)
              if A >= max_ai:
                max_ai = A
                w_max = w
                h_max = h
                max_area = w_max* h_max
                percent_area_max_void = np.round((max_ai/comp_area)*100, decimals = 2)
        # print(df_final['Image'][i])
        #print('comp_area', comp_area)
        # print(l)  # list of lists [w, h] of voids boxe in one component
        # print(ls_ai)   # list of voids area within one component
        # print(max_ai)  # keep maximum void area within one component
        # print('area',sum_voids_area)  # sum of all voids area in one component
        # print(w_max, h_max)  #w, h of the maximum void
        # print(max_area)   #  # keep the maximum void area
        # print('void%', (sum_voids_area / comp_area)*100)  # void percentage
        # print('Max_void%', (max_ai/comp_area)*100)   # MAX_void percentage
        # print('--------------------------------------')
        dic = {'Image': df_final['Image'][i], 'Component': k , 'Area':comp_area, 'Void%': percent_area_void, 'Max_Void%': percent_area_max_void}
        df_dplmt.loc[len(df_dplmt)] = dic

    df_dplmt
    print('df_dplmt\n', df_dplmt)

    # get group of the image(s) you want to get group from
    if isinstance(n, int)  and -2 < n <= len(os.listdir(path_images)) and n!= 0:
        df_dplmt = df_dplmt.groupby(by = ['Image'])
        rf = list(df_dplmt.groups.keys())[n-1]
        print('rf\n', rf)
        df_dplmt_rf = df_dplmt.get_group(rf)
        print('df_dplmt_rf\n', df_dplmt_rf)
        return df_dplmt_rf.reset_index(inplace= True)

    return df_dplmt    


result = Frame(path_images = os.path.join(os.getcwd(), "data\\Data_Transformation\\test\\images\\") ,n =5)
result
# print the mask of each image