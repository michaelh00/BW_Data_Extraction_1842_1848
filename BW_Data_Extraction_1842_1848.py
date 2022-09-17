#!/usr/bin/env python
# coding: utf-8

# # Code for data extraction: Bozner Wochenblatt 1842-1848

# In[1]:


from platform import python_version
import os
import time
import datetime
from datetime import date

import cv2
import numpy as np
import pandas as pd

import skimage
from skimage import io
from skimage import transform

import joblib
from joblib import load


# In[2]:


print("Python version: " + python_version())
print("OpenCV version: " + cv2.__version__)
print("numpy version: " + np.__version__)
print("pandas version: " + pd.__version__)
print("scikit-image version: " + skimage.__version__)
print("Joblib version: " + joblib.__version__)


# ---

# ## Table segmentation

# Extraction of the parameters for the pre-mask (exclusion of the top and bottom part of the table). What is considered is what is below param_m_u and above param_m_d.

# In[3]:


def parameter_header_fuss_grenze(img_rgb):
    
    ### Extraction of table header conturs
    img_gr = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
    gray = cv2.bitwise_not(img_gr)
    horizontal = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, -2)
    cols = horizontal.shape[1]
    horizontal_size = cols // 30
    horizontalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_size, 1))
    horizontal = cv2.erode(horizontal, horizontalStructure)
    horizontal = cv2.dilate(horizontal, horizontalStructure)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, ksize=(img_rgb.shape[1], 1))
    dilated_h = cv2.dilate(horizontal, kernel, iterations=1)
    
    contours, hierarchy = cv2.findContours(dilated_h, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    list_y_all_cont = []
    for cont in contours: 
        list_y_cont = []
        for n in range(0, len(cont)):
            a = 2*n + 1
            y = np.take(cont, a)
            list_y_cont.append(y)
        list_y_all_cont.append(list_y_cont)
    
    cont_max_y_all =[]
    cont_min_y_all =[]
    for item in list_y_all_cont:
        item_cont = np.asarray(item)
        cont_max_y = item_cont.max()
        cont_max_y_all.append(cont_max_y)
        cont_min_y = item_cont.min()
        cont_min_y_all.append(cont_min_y)
    
    param_m_u = min(cont_max_y_all, key=lambda x:abs(x-img_rgb.shape[0]/2))
    param_m_d = min(cont_min_y_all, key=lambda x:abs(x-img_rgb.shape[0]*0.9))
    
    return param_m_u, param_m_d


# ---

# Extraction of vertical mask parameters (segmentation of the table in columns).

# In[4]:


def parameter_verticale_masken(img_rgb, param_m_u, param_m_d):
        
    ### Premask: image without header
    frame = np.full((img_rgb.shape[0], img_rgb.shape[1]), fill_value=255, dtype=np.uint8)
    mask_color = (0, 0, 0)
    m_u = cv2.rectangle(frame, (0, 0), (img_rgb.shape[1], param_m_u + 2), mask_color, -1) 
    m_d = cv2.rectangle(frame, (0, param_m_d - 2), (img_rgb.shape[1], img_rgb.shape[0]), mask_color, -1) 
    m_l = cv2.rectangle(frame, (0, 0), (10, img_rgb.shape[0]), mask_color, -1) 
    m_r = cv2.rectangle(frame, (img_rgb.shape[1] - 10, 0), (img_rgb.shape[1], img_rgb.shape[0]), mask_color, -1)
    vormaske = m_u + m_d + m_l + m_r
    
    ### Parameters for the main mask Vertical Boxes ###
    ### Creation of dilated vertical fields whose contours are further used
    hilfsbild_1 = np.full((img_rgb.shape[0], img_rgb.shape[1], img_rgb.shape[2]), fill_value=0, dtype=np.uint8)
    masked_img_1 = cv2.bitwise_not(img_rgb, hilfsbild_1, mask=vormaske) 
    blur_1 = cv2.GaussianBlur(masked_img_1,(3,3), 0, 0)
    blur_gr_1 = cv2.cvtColor(blur_1, cv2.COLOR_BGR2GRAY)
    ret, out_1 = cv2.threshold(blur_gr_1, 105, 255, cv2.THRESH_BINARY)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, ksize=(5, img_rgb.shape[0]))
    dilated = cv2.dilate(out_1, kernel, iterations=1)
      
    ### Extraction of the x-coordinates of the contour points in a list of lists
    contours, hierarchy = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) > 28:
        contours_sorted = sorted(contours, key=lambda x: cv2.arcLength(x, True), reverse=True)
        contours = contours_sorted[0:28]

    cont_max_x_all = []
    cont_min_x_all = []

    list_x_all_cont = []
    for cont in contours: 
        list_x_cont = []
        for n in range(0, len(cont)):
            a = 2*n
            x = np.take(cont, a)
            list_x_cont.append(x)
        list_x_all_cont.append(list_x_cont)
    list_x_all_cont.sort()

    ### The relevant contours are filtered out of all contours
    if len(list_x_all_cont) == 28:
        spalte_1 = list_x_all_cont[1]
        spalte_2 = list_x_all_cont[3]
        spalte_3 = list_x_all_cont[5]
        spalte_4 = list_x_all_cont[7]
        spalte_5 = list_x_all_cont[9]
        spalte_6 = list_x_all_cont[11]
        spalte_7 = list_x_all_cont[13]
        spalte_8 = list_x_all_cont[16] 
        spalte_9 = list_x_all_cont[19] 
        spalte_10 = list_x_all_cont[22]
        spalte_11 = list_x_all_cont[24] 
        spalte_12 = list_x_all_cont[26]

        spaltenliste = [spalte_1, spalte_2, spalte_3, spalte_4, spalte_5, spalte_6, spalte_7, 
                        spalte_8, spalte_9, spalte_10, spalte_11, spalte_12]

        masking_params_x = [] 
        for item in spaltenliste:
            item_cont = np.asarray(item)
            cont_min_x = item_cont.min()
            cont_max_x = item_cont.max()
            masking_params_x.append((cont_min_x, cont_max_x))
    else: 
        masking_params_x = [(88,97), (135,141), (207,215), (255,262), (326,335), (376,382), (447,455), 
                            (521,527), (597,605), (672,681), (795,805), (925,935)]
        
    return masking_params_x


# ---

# Extraction of horizontal mask parameters (table segmentation in rows)

# In[5]:


def parameter_horizontale_masken(img_rgb, param_m_u, param_m_d, par_vert_mask):
    
    ### Help mask: only weather condition data columns
    frame = np.zeros((img_rgb.shape[0], img_rgb.shape[1]), dtype=np.uint8)
    frame.fill(255)
    mask_color = (0, 0, 0)
    
    m_u = cv2.rectangle(frame, (0, 0), (img_rgb.shape[1], param_m_u), mask_color, -1) 
    m_d = cv2.rectangle(frame, (0, param_m_d), (img_rgb.shape[1], img_rgb.shape[0]), mask_color, -1) 
    m_l = cv2.rectangle(frame, (0, 0), ((par_vert_mask[9][1] + 6), img_rgb.shape[0]), mask_color, -1) 
    m_r = cv2.rectangle(frame, (img_rgb.shape[1], 0), (img_rgb.shape[1], img_rgb.shape[0]), mask_color, -1)
    
    m_v11 = cv2.rectangle(frame, ((par_vert_mask[10][0] - 6), 0), ((par_vert_mask[10][1] + 6), img_rgb.shape[0]), mask_color, -1)
    m_v12 = cv2.rectangle(frame, ((par_vert_mask[11][0] - 6), 0), ((par_vert_mask[11][1] + 6), img_rgb.shape[0]), mask_color, -1)
    
    hilfsmaske = m_u + m_d + m_l + m_r + m_v11 + m_v12
    
    ### Parameters for the main mask Horizontal Boxes ###
    ### Creation of dilated horizontal fields whose contours are reused
    hilfsbild_0 = np.full((img_rgb.shape[0], img_rgb.shape[1], img_rgb.shape[2]), fill_value=0, dtype=np.uint8)
    masked_img_0 = cv2.bitwise_not(img_rgb, hilfsbild_0, mask=hilfsmaske) 
    blur_0 = cv2.GaussianBlur(masked_img_0,(3,3), 0, 0)
    blur_gr_0 = cv2.cvtColor(blur_0, cv2.COLOR_BGR2GRAY)

    ret, out_0 = cv2.threshold(blur_gr_0, 210, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, ksize=(int(img_rgb.shape[1]/4), 3))
    dilated = cv2.dilate(out_0, kernel, iterations=1)
    
    ### Extraction of the y-coordinates of the contour points in a list of lists
    contours, hierarchy = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) == 7:
        cont_max_y_all = []
        cont_min_y_all = []
        list_y_all_cont = []
        for cont in contours: 
            list_y_cont = []
            for n in range(0, len(cont)):
                a = 2*n + 1
                y = np.take(cont, a)
                list_y_cont.append(y)
            list_y_all_cont.append(list_y_cont)
            
        for item in list_y_all_cont:
            item_cont = np.asarray(item)
            cont_max_y = item_cont.max()
            cont_min_y = item_cont.min()
            cont_max_y_all.append(cont_max_y)
            cont_min_y_all.append(cont_min_y)
            
        cutting_list = [cont_max_y_all[0]]
        
        for i in range (0,6):
            first = int(cont_min_y_all[i])
            second = int(cont_max_y_all[i+1])
            cutting = int((first + second)/2)
            cutting_list.append(cutting)
        cutting_list.append(cont_min_y_all[6])
    else:
        cutting_list = [277, 259, 239, 220, 200, 180, 161, 144]
    
    return cutting_list


# ---

# Establishment of the data fields (segmentation of the table in 7 x 12 = 84 boxes (without calendar date fields))

# In[6]:


def boxes_func (img_rgb, parameter_horizontale_masken, parameter_verticale_masken, param_m_u):
    
    boxlines_h = parameter_horizontale_masken
    boxlines_v = parameter_verticale_masken
    
    frame = np.zeros((img_rgb.shape[0], img_rgb.shape[1]), dtype=np.uint8)
    frame.fill(255)
    mask_color = (0, 0, 0)

    m_u = cv2.rectangle(frame, (0, 0), (img_rgb.shape[1], param_m_u), mask_color, -1)
    m_d = cv2.rectangle(frame, (0, (img_rgb.shape[0] - 10)), (img_rgb.shape[1], img_rgb.shape[0]), mask_color, -1) 
    m_l = cv2.rectangle(frame, (0, 0), ((boxlines_v[0][1] + 6), img_rgb.shape[0]), mask_color, -1) 
    m_r = cv2.rectangle(frame, (img_rgb.shape[1] - 10, 0), (img_rgb.shape[1], img_rgb.shape[0]), mask_color, -1)

    m_v1 = cv2.rectangle(frame, ((boxlines_v[1][0] - 6), 0), ((boxlines_v[1][1] + 6), img_rgb.shape[0]), mask_color, -1)
    m_v2 = cv2.rectangle(frame, ((boxlines_v[2][0] - 6), 0), ((boxlines_v[2][1] + 6), img_rgb.shape[0]), mask_color, -1)
    m_v3 = cv2.rectangle(frame, ((boxlines_v[3][0] - 6), 0), ((boxlines_v[3][1] + 6), img_rgb.shape[0]), mask_color, -1)
    m_v4 = cv2.rectangle(frame, ((boxlines_v[4][0] - 6), 0), ((boxlines_v[4][1] + 6), img_rgb.shape[0]), mask_color, -1)
    m_v5 = cv2.rectangle(frame, ((boxlines_v[5][0] - 6), 0), ((boxlines_v[5][1] + 6), img_rgb.shape[0]), mask_color, -1)
    m_v6 = cv2.rectangle(frame, ((boxlines_v[6][0] - 6), 0), ((boxlines_v[6][1] + 6), img_rgb.shape[0]), mask_color, -1)
    m_v7 = cv2.rectangle(frame, ((boxlines_v[7][0] - 6), 0), ((boxlines_v[7][1] + 6), img_rgb.shape[0]), mask_color, -1)
    m_v8 = cv2.rectangle(frame, ((boxlines_v[8][0] - 6), 0), ((boxlines_v[8][1] + 6), img_rgb.shape[0]), mask_color, -1)

    m_v9 = cv2.rectangle(frame, ((boxlines_v[9][0] - 6), 0), ((boxlines_v[9][1] + 6), img_rgb.shape[0]), mask_color, -1)
    m_v10 = cv2.rectangle(frame, ((boxlines_v[10][0] - 6), 0), ((boxlines_v[10][1] + 6), img_rgb.shape[0]), mask_color, -1)
    m_v11 = cv2.rectangle(frame, ((boxlines_v[11][0] - 6), 0), ((boxlines_v[11][1] + 6), img_rgb.shape[0]), mask_color, -1)

    m_h1 = cv2.rectangle(frame, (0, boxlines_h[1]), (img_rgb.shape[1], boxlines_h[1]), mask_color, -1)
    m_h2 = cv2.rectangle(frame, (0, boxlines_h[2]), (img_rgb.shape[1], boxlines_h[2]), mask_color, -1)
    m_h3 = cv2.rectangle(frame, (0, boxlines_h[3]), (img_rgb.shape[1], boxlines_h[3]), mask_color, -1)
    m_h4 = cv2.rectangle(frame, (0, boxlines_h[4]), (img_rgb.shape[1], boxlines_h[4]), mask_color, -1)
    m_h5 = cv2.rectangle(frame, (0, boxlines_h[5]), (img_rgb.shape[1], boxlines_h[5]), mask_color, -1)
    m_h6 = cv2.rectangle(frame, (0, boxlines_h[6]), (img_rgb.shape[1], boxlines_h[6]), mask_color, -1)


    maske = m_u + m_d + m_l + m_r + m_v1 + m_v2 + m_v3 + m_v4 + m_v5 + m_v6 + m_v7 + m_v8 + m_v9 + m_v10 + m_v11 + m_h1 + m_h2 + m_h3 + m_h4 + m_h5 + m_h6

    contours, hierarchy = cv2.findContours(maske, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    list_of_boxes = []
    for cont in contours:
        x_c, y_c, w_c, h_c = cv2.boundingRect(cont)
        list_of_boxes.append([x_c, y_c, w_c, h_c])
    
    return list_of_boxes


# ---

# Extracted numeric data fields are brought into the standardized format of 30x80 pixels.

# In[7]:


def uni_form(segm_im):

    height, width = (30, 80)

    blank_img = np.zeros((height, width, 3), np.uint8)
    blank_img[:, 0:width] = (255, 255, 255) # (B, G, R)

    x_offset = int((width  - segm_im.shape[1])/2)
    y_offset = int((height - segm_im.shape[0])/2)

    blank_img[ y_offset:y_offset+segm_im.shape[0], x_offset:x_offset+segm_im.shape[1]] = segm_im
    
    return blank_img


# ---

# ## Data field segmentation for numeric data fields

# Segmentation of the numeric data fields depending on whether they have 1, 2 or 3 characters

# Segmentation of a data field with 3 characters

# In[8]:


def dreistellig (im2):

    gray_im = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray_im,(1,1), 0, 0)
    ret,out= cv2.threshold(blur, 105, 255, cv2.THRESH_BINARY)
    out1= cv2.bitwise_not(out)
    contours, hierarchy = cv2.findContours(out1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    list_x = []
    for cont in contours:    
        for n in range(0, len(cont)):
            a = 2*n
            x = np.take(cont, a)
            list_x.append(x)
    minimum_x = np.amin(list_x)
    maximum_x = np.amax(list_x)
    one_third_x = 1/3 * (maximum_x - minimum_x) + minimum_x

    list_x_all_cont = []
    for cont in contours: 
        list_x_cont = []
        for n in range(0, len(cont)):
            a = 2*n
            x = np.take(cont, a)
            list_x_cont.append(x)
        list_x_all_cont.append(list_x_cont)

    nearest_list = []
    for item in list_x_all_cont:
        nearest = item[np.abs(item - one_third_x).argmin()] + 1
        nearest_list.append(nearest)

    nearest_array = np.asarray(nearest_list)

    try:
        if one_third_x > nearest_array.max():
            nearest_bigger = nearest_array.max()
        elif one_third_x < nearest_array.min():
            nearest_bigger = nearest_array.min()
        else:
            nearest_bigger = nearest_array[nearest_array >= one_third_x].min()
    except:
        pass

    try: 
        if one_third_x < nearest_array.min():
            nearest_smaller = nearest_array.min()
        elif one_third_x > nearest_array.max():
            nearest_smaller = nearest_array.max()
        else:
            nearest_smaller = nearest_array[nearest_array <= one_third_x].max()  
    except:
        pass

    cutting_x = np.mean([nearest_bigger, nearest_smaller])

    y = 0
    x = 0
    h = len(im2)
    w = int(cutting_x)        
    processed_1_im2 = im2[y:y+h, x:x+w]

    y = 0
    x = int(cutting_x)
    h = len(im2)
    w = im2.shape[1] - x        
    processed_2_im2 = im2[y:y+h, x:x+w]

    
    im3 = processed_2_im2
    gray_im = cv2.cvtColor(im3, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray_im,(1,1), 0, 0)
    ret,out= cv2.threshold(blur, 105, 255, cv2.THRESH_BINARY)
    out1= cv2.bitwise_not(out)
    contours, hierarchy = cv2.findContours(out1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    list_x = []
    for cont in contours:    
        for n in range(0, len(cont)):
            a = 2*n
            x = np.take(cont, a)
            list_x.append(x)
    minimum_x = np.amin(list_x)
    maximum_x = np.amax(list_x)
    one_half_x = 1/2 * (maximum_x - minimum_x) + minimum_x

    list_x_all_cont = []
    for cont in contours: 
        list_x_cont = []
        for n in range(0, len(cont)):
            a = 2*n
            x = np.take(cont, a)
            list_x_cont.append(x)
        list_x_all_cont.append(list_x_cont)

    nearest_list = []
    for item in list_x_all_cont:
        nearest = item[np.abs(item - one_half_x).argmin()] + 1
        nearest_list.append(nearest)

    nearest_array = np.asarray(nearest_list)

    try:
        if one_half_x > nearest_array.max():
            nearest_bigger = nearest_array.max()
        elif one_half_x < nearest_array.min():
            nearest_bigger = nearest_array.min()
        else:
            nearest_bigger = nearest_array[nearest_array >= one_half_x].min()
    except:
        pass

    try: 
        if one_half_x < nearest_array.min():
            nearest_smaller = nearest_array.min()
        elif one_half_x > nearest_array.max():
            nearest_smaller = nearest_array.max()
        else:
            nearest_smaller = nearest_array[nearest_array <= one_half_x].max()  
    except:
        pass

    cutting_x = np.mean([nearest_bigger, nearest_smaller])

    y = 0
    x = 0
    h = len(im3)
    w = int(cutting_x)        
    processed_1_im3 = im3[y:y+h, x:x+w]

    y = 0
    x = int(cutting_x)
    h = len(im3)
    w = im3.shape[1] - x        
    processed_2_im3 = im3[y:y+h, x:x+w]
    
    return(processed_1_im2, processed_1_im3, processed_2_im3)


# Segmentation of a data field with 2 characters

# In[9]:


def zweistellig (im3):

    gray_im = cv2.cvtColor(im3, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray_im,(1,1), 0, 0)
    ret,out= cv2.threshold(blur, 105, 255, cv2.THRESH_BINARY)
    out1= cv2.bitwise_not(out)
    contours, hierarchy = cv2.findContours(out1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    list_x = []
    for cont in contours:    
        for n in range(0, len(cont)):
            a = 2*n
            x = np.take(cont, a)
            list_x.append(x)
    minimum_x = np.amin(list_x)
    maximum_x = np.amax(list_x)
    one_half_x = 1/2 * (maximum_x - minimum_x) + minimum_x

    list_x_all_cont = []
    for cont in contours: 
        list_x_cont = []
        for n in range(0, len(cont)):
            a = 2*n
            x = np.take(cont, a)
            list_x_cont.append(x)
        list_x_all_cont.append(list_x_cont)

    nearest_list = []
    for item in list_x_all_cont:
        nearest = item[np.abs(item - one_half_x).argmin()] + 1
        nearest_list.append(nearest)

    nearest_array = np.asarray(nearest_list)

    try:
        if one_half_x > nearest_array.max():
            nearest_bigger = nearest_array.max()
        elif one_half_x < nearest_array.min():
            nearest_bigger = nearest_array.min()
        else:
            nearest_bigger = nearest_array[nearest_array >= one_half_x].min()
    except:
        pass

    try: 
        if one_half_x < nearest_array.min():
            nearest_smaller = nearest_array.min()
        elif one_half_x > nearest_array.max():
            nearest_smaller = nearest_array.max()
        else:
            nearest_smaller = nearest_array[nearest_array <= one_half_x].max()  
    except:
        pass

    cutting_x = np.mean([nearest_bigger, nearest_smaller])

    y = 0
    x = 0
    h = len(im3)
    w = int(cutting_x)        
    processed_1_im3 = im3[y:y+h, x:x+w]

    y = 0
    x = int(cutting_x)
    h = len(im3)
    w = im3.shape[1] - x        
    processed_2_im3 = im3[y:y+h, x:x+w]
    
    return(processed_1_im3, processed_2_im3)


# Segmented characters are brought into the standardized format of 25x25 pixels.

# In[10]:


def single_crop(im_rgb):
    
    gray_im = cv2.cvtColor(im_rgb, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray_im,(1,1), 0, 0)
    ret,out= cv2.threshold(blur, 220, 255, cv2.THRESH_BINARY)
    out1= cv2.bitwise_not(out)
    contours, hierarchy = cv2.findContours(out1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    largest_cont = max(contours, key=len)
    
    largest_cont_x = [i[0][0] for i in largest_cont]
    median_largest_cont_x = np.median(largest_cont_x)
    min_largest_cont_x = np.amin(largest_cont_x)
    max_largest_cont_x = np.amax(largest_cont_x)
    
    y = 0
    x = int(min_largest_cont_x)
    h = int(im_rgb.shape[0])
    w = int(max_largest_cont_x - min_largest_cont_x)       
    processed = im_rgb[y:y+h, x:x+w]
    
    height, width = (25, 25)
    blank_image_0 = np.zeros((height, width, 3), np.uint8)
    blank_image_0[:, 0:width] = (255, 255, 255) # (B, G, R)

    x_offset = int((width  - processed.shape[1])/2)
    y_offset = int((height - processed.shape[0])/2)

    blank_image_0[ y_offset:y_offset+processed.shape[0], x_offset:x_offset+processed.shape[1]] = processed
    
    return blank_image_0


# ---

# ## Format adaptations for the machine learning pipeline

# Extracted numeric data fields are put into a format required for the ML algorithm used for data field segmentation.

# In[11]:


def process_segm_img(img):
    bild = skimage.transform.rescale(img, 1, anti_aliasing=False, multichannel=True, mode='reflect')
    bild_2 = np.reshape(bild, (1, 30*80*3))
    return bild_2


# Segmented characters are put into a format required for the ML algorithm used for optical character recognition.

# In[12]:


def process_pred_img(img):
    bild = skimage.transform.rescale(img, 1, anti_aliasing=False, multichannel=True, mode='reflect')
    bild_2 = np.reshape(bild, (1, 25*25*3))
    return bild_2


# ---

# ## Processing of descriptive data fields

# Extracted descriptive data fields are brought into the standardized format of 30x110 pixels.

# In[13]:


def witt_mitt (img_rgb):
    gray_im = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray_im,(17,17), 0, 0)
    ret,out= cv2.threshold(gray_im, 100, 255, cv2.THRESH_BINARY)
    out1= cv2.bitwise_not(out)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, ksize=(int(img_rgb.shape[1]/10), 1))
    dilated = cv2.dilate(out1, kernel, iterations=1)
    contours, hierarchy = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_cont = max(contours, key=len)
    largest_cont_x = [i[0][0] for i in largest_cont]
    median_largest_cont_x = np.median(largest_cont_x)
    min_largest_cont_x = np.amin(largest_cont_x)
    max_largest_cont_x = np.amax(largest_cont_x)

    y = 0
    x = int(min_largest_cont_x)
    h = int(img_rgb.shape[0])
    w = int(max_largest_cont_x - min_largest_cont_x)       
    processed = img_rgb[y:y+h, x:x+w]

    height, width = (30, 110)
    output_witt_mitt = np.zeros((height, width, 3), np.uint8)
    output_witt_mitt[:, 0:width] = (255, 255, 255) # (B, G, R)

    x_offset = int((width  - processed.shape[1])/2)
    y_offset = int((height - processed.shape[0])/2)

    output_witt_mitt[ y_offset:y_offset+processed.shape[0], x_offset:x_offset+processed.shape[1]] = processed
    
    return output_witt_mitt


# Required if post-normalization should be necessary during the analysis process

# In[14]:


def uni_form_witt(segm_im):

    height, width = (30, 110)

    blank_img = np.zeros((height, width, 3), np.uint8)
    blank_img[:, 0:width] = (255, 255, 255) # (B, G, R)

    x_offset = int((width  - segm_im.shape[1])/2)
    y_offset = int((height - segm_im.shape[0])/2)

    blank_img[ y_offset:y_offset+segm_im.shape[0], x_offset:x_offset+segm_im.shape[1]] = segm_im
    
    return blank_img


# Extracted descriptive data fields are put into a format required for the ML algorithm used for optical pattern recognition.

# In[15]:


def process_witt_feld(img):
    bild = skimage.transform.rescale(img, 1, anti_aliasing=False, multichannel=True, mode='reflect')
    bild_out = np.reshape(bild, (1, 30*110*3))
    return bild_out


# ---

# ## Loading of trained random forest (RF) algorithms

# The trained random forest algorithms need to be decompressed [(unzipped)](https://support.microsoft.com/en-us/windows/zip-and-unzip-files-f6dde0a7-0fec-8294-e1d3-703ed85e7ebc) from the folder **trained_RF_algorithms_zipped.zip** before use.

# For numeric data field segmentation

# In[17]:


segm_classifier = load("1_RFC_algorithm.joblib")


# For optical character recognition (OCR) of segmented characters.

# In[18]:


ocr_classifier = load("2_RFC_algorithm.joblib")


# For optical pattern recognition of descripive data fields.

# In[20]:


witt_classifier = load("3_RFC_algorithm.joblib")


# ---

# ## Extraction of data from data fields

# In[21]:


def iteration (boxes):
        
    witterungsfelder = [0,1,2,12,13,14,24,25,26,36,37,38,48,49,50,60,61,62,72,73,74]  
    
    datalist = []

    for index, box in enumerate(boxes):

        try:
            height, width = (box[3], box[2])
            blank_image = np.zeros((height, width, 3), np.uint8)
            blank_image[:, 0:width] = (255, 255, 255) # (B, G, R)

            img0 = image

            y = box[1]
            x = box[0]
            h = box[3]
            w = box[2]          
            img = img0[y:y+h, x:x+w]

            a1_1_offset = 0 # x-axis
            b1_1_offset = 0 # y-axis

            blank_image[b1_1_offset:b1_1_offset+img.shape[0], a1_1_offset:a1_1_offset+img.shape[1]] = img


            ### Extraction of the data from the descriptive data fields #########################################################
            if index in witterungsfelder:

                try:
                    outp = witt_mitt(blank_image)

                except:
                    scale_percent = 90
                    width = int(blank_image.shape[1] * scale_percent / 100)
                    height = int(blank_image.shape[0] * scale_percent / 100)
                    dim = (width, height)
                    resized_blank_image = cv2.resize(blank_image, dim, interpolation = cv2.INTER_AREA)
                    outp = uni_form_witt(resized_blank_image)

                outp_arr = process_witt_feld(outp)

                boxkind_pred = witt_classifier.predict(outp_arr)

                if boxkind_pred[0] == "0":
                    prediction = "detto"

                elif boxkind_pred[0] == "1":
                        prediction = "Donnerwetter"

                elif boxkind_pred[0] == "2":
                    prediction = "heiter"

                elif boxkind_pred[0] == "3":
                    prediction = "Regen"

                elif boxkind_pred[0] == "4":
                    prediction = "Schnee"

                elif boxkind_pred[0] == "5":
                    prediction = "trueb"
                
                elif boxkind_pred[0] == "6":
                    prediction = "Wolken"
                
                elif boxkind_pred[0] == "7":
                    prediction = "Nebel"

                else:
                    pass

                output = str(prediction)
                datalist.append(output)

            ### Extraction of the data from the numeric data fields #############################################################      
            else:
                
                try:
                    boxfeatures = uni_form(blank_image)
                except:
                    scale_percent = 90
                    width = int(blank_image.shape[1] * scale_percent / 100)
                    height = int(blank_image.shape[0] * scale_percent / 100)
                    dim = (width, height)
                    resized_blank_image = cv2.resize(blank_image, dim, interpolation = cv2.INTER_AREA)
                    boxfeatures = uni_form(resized_blank_image)

                boxfeatures_arr = process_segm_img(boxfeatures)
                boxkind_pred = segm_classifier.predict(boxfeatures_arr)

                if boxkind_pred[0] == "2":

                    teilung = dreistellig(blank_image)
                    erste_ziffer = teilung[0]
                    zweite_ziffer = teilung[1]
                    dritte_ziffer = teilung[2]

                    erste_ziffer_crop = single_crop(erste_ziffer)
                    zweite_ziffer_crop = single_crop(zweite_ziffer)
                    dritte_ziffer_crop = single_crop(dritte_ziffer)

                    erste_ziffer_crop = single_crop(erste_ziffer)
                    res_1 = process_pred_img(erste_ziffer_crop)
                    ypred_1 = ocr_classifier.predict(res_1)

                    zweite_ziffer_crop = single_crop(zweite_ziffer)
                    res_2 = process_pred_img(zweite_ziffer_crop)
                    ypred_2 = ocr_classifier.predict(res_2)

                    dritte_ziffer_crop = single_crop(dritte_ziffer)
                    res_3 = process_pred_img(dritte_ziffer_crop)
                    ypred_3 = ocr_classifier.predict(res_3)

                    if ypred_1[0] == "-2":
                        prediction_1 = ""

                    elif ypred_1[0] == "-1":
                        prediction_1 = "-"

                    else:
                        prediction_1 = ypred_1[0]

                    if ypred_2[0] == "-2":
                        prediction_2 = ""

                    elif ypred_2[0] == "-1":
                        prediction_2 = "-"

                    else:
                        prediction_2 = ypred_2[0]

                    if ypred_3[0] == "-2":
                        prediction_3 = ""

                    elif ypred_3[0] == "-1":
                        prediction_3 = "-"

                    else:
                        prediction_3 = ypred_3[0]

                    output = str(prediction_1) + str(prediction_2) + str(prediction_3)
                    datalist.append(output)


                elif boxkind_pred[0] == "1":

                    teilung = zweistellig(blank_image)
                    erste_ziffer = teilung[0]
                    zweite_ziffer = teilung[1]

                    erste_ziffer_crop = single_crop(erste_ziffer)
                    res_1 = process_pred_img(erste_ziffer_crop)
                    ypred_1 = ocr_classifier.predict(res_1)

                    zweite_ziffer_crop = single_crop(zweite_ziffer)
                    res_2 = process_pred_img(zweite_ziffer_crop)
                    ypred_2 = ocr_classifier.predict(res_2)

                    if ypred_1[0] == "-2":
                        prediction_1 = ""

                    elif ypred_1[0] == "-1":
                        prediction_1 = "-"

                    else:
                        prediction_1 = ypred_1[0]

                    if ypred_2[0] == "-2":
                        prediction_2 = ""

                    elif ypred_2[0] == "-1":
                        prediction_2 = "-"

                    else:
                        prediction_2 = ypred_2[0]

                    output = str(prediction_1) + str(prediction_2)
                    datalist.append(output)


                elif boxkind_pred == "0":

                    erste_ziffer = blank_image

                    erste_ziffer_crop = single_crop(erste_ziffer)
                    res_1 = process_pred_img(erste_ziffer_crop)
                    ypred_1 = ocr_classifier.predict(res_1)

                    if ypred_1[0] == "-2":
                        prediction_1 = ""

                    elif ypred_1[0] == "-1":
                        prediction_1 = "-"

                    else:
                        prediction_1 = ypred_1[0]

                    output = str(prediction_1)
                    datalist.append(output)

        except:
            output = " "
            datalist.append(output)
        
    return datalist


# ---

# ## Structure of the data list for the output

# In[22]:


def days_file_structure (days_file, filename, daten):

    indexlist_W_21 = [0, 12, 24, 36, 48, 60, 72]
    indexlist_W_13 = [1, 13, 25, 37, 49, 61, 73]
    indexlist_W_07 = [2, 14, 26, 38, 50, 62, 74]

    indexlist_T_21 = [3, 15, 27, 39, 51, 63, 75]
    indexlist_T_13 = [4, 16, 28, 40, 52, 64, 76]
    indexlist_T_07 = [5, 17, 29, 41, 53, 65, 77]

    indexlist_D_21_1 = [6, 18, 30, 42, 54, 66, 78]
    indexlist_D_21_2 = [7, 19, 31, 43, 55, 67, 79]

    indexlist_D_13_1 = [8, 20, 32, 44, 56, 68, 80]
    indexlist_D_13_2 = [9, 21, 33, 45, 57, 69, 81]

    indexlist_D_07_1 = [10, 22, 34, 46, 58, 70, 82]
    indexlist_D_07_2 = [11, 23, 35, 47, 59, 71, 83]

    datalist_W_21 = [daten[x] for x in indexlist_W_21]
    datalist_W_13 = [daten[x] for x in indexlist_W_13]
    datalist_W_07 = [daten[x] for x in indexlist_W_07]

    datalist_T_21 = [daten[x] for x in indexlist_T_21]
    datalist_T_13 = [daten[x] for x in indexlist_T_13]
    datalist_T_07 = [daten[x] for x in indexlist_T_07]

    datalist_D_21_1 = [daten[x] for x in indexlist_D_21_1]
    datalist_D_21_2 = [daten[x] for x in indexlist_D_21_2]

    datalist_D_13_1 = [daten[x] for x in indexlist_D_13_1]
    datalist_D_13_2 = [daten[x] for x in indexlist_D_13_2]

    datalist_D_07_1 = [daten[x] for x in indexlist_D_07_1]
    datalist_D_07_2 = [daten[x] for x in indexlist_D_07_2]

    filename_date = filename[0:10]
    date_yymmdd = datetime.datetime.strptime(filename_date, "%m-%d-%Y").strftime("%Y-%m-%d")

    for i in range (7,0, -1):
        day = date.fromisoformat(date_yymmdd)
        d = datetime.timedelta(days = i)
        dayB = day - d
        dayB_conv = date.isoformat(dayB)

        day_07 = []
        dayB_conv_1 = dayB_conv + " 07:00:00"
        day_07.append(dayB_conv_1)
        day_07.append(datalist_D_07_2[i-1])
        day_07.append(datalist_D_07_1[i-1])
        day_07.append(datalist_T_07[i-1])
        day_07.append(datalist_W_07[i-1])
        day_07.append("unknown")

        day_13 = []
        dayB_conv_2 = dayB_conv + " 13:00:00"
        day_13.append(dayB_conv_2)
        day_13.append(datalist_D_13_2[i-1])
        day_13.append(datalist_D_13_1[i-1])
        day_13.append(datalist_T_13[i-1])
        day_13.append(datalist_W_13[i-1])
        day_13.append("unknown")

        day_21 = []
        dayB_conv_3 = dayB_conv + " 21:00:00"
        day_21.append(dayB_conv_3)
        day_21.append(datalist_D_21_2[i-1])
        day_21.append(datalist_D_21_1[i-1])
        day_21.append(datalist_T_21[i-1])
        day_21.append(datalist_W_21[i-1])
        day_21.append("unknown")

        days_file.append(day_07)
        days_file.append(day_13)
        days_file.append(day_21)

    return days_file


# ---

# ## Data extraction for one example table

# In[23]:


filename = "01-20-1843.png"


# In[24]:


days_file = []
    
start = time.time()

image = cv2.imread(filename)
header = parameter_header_fuss_grenze(image)[0]
fuss = parameter_header_fuss_grenze(image)[1]
masking_params_x = parameter_verticale_masken(image, header, fuss)
masking_params_y = parameter_horizontale_masken(image, header, fuss, masking_params_x)
boxes = boxes_func(image, masking_params_y, masking_params_x, header)

daten = iteration(boxes)
    
days_file = days_file_structure(days_file, filename, daten)
        
end = time.time()
print("File: " + str(filename) + ", time required for data extraction: " + str(end-start) + " seconds.")


# In[25]:


df = pd.DataFrame(days_file, columns=["datetime", "Barometer_Zoll", "Barometer_Linien", "Thermometer", "Witterung", "manual_corr"])
df


# In[26]:


df.to_csv("result_example_raw_data.csv", index=False)


# ---

# ## Acknowledgements and Documentation

# The code in this Python Jupyter notebook is inspired by and adapted from the following sources (last access 15.09.2022): 
# 
# Image Processing Techniques:  
# https://docs.opencv.org/4.x/d2/d96/tutorial_py_table_of_contents_imgproc.html  
# https://stackoverflow.com/questions/14063070/overlay-a-smaller-image-on-a-larger-image-python-opencv  
# 
# Machine Learning Techniques:  
# https://scikit-learn.org/stable/  
# https://www.tensorflow.org/tutorials  

# This Python Jupyter notebook uses Python OpenCV (https://github.com/opencv/opencv-python), numpy (https://github.com/numpy/numpy), pandas (https://github.com/pandas-dev/pandas), scikit-image (https://github.com/scikit-image/scikit-image) and joblib (https://github.com/joblib/joblib). The code functionality was developed with the additional use of scikit-learn (https://github.com/scikit-learn/scikit-learn), tensorflow (https://github.com/tensorflow/tensorflow) and matplotlib (https://github.com/matplotlib/matplotlib).

# ---

# The data source for the data extraction is the [Bozner Wochenblatt](https://digital.tessmann.it/tessmannDigital/Zeitungsarchiv/Jahresuebersicht/Zeitung/2) (later Bozner Zeitung) for the period 1842 - 1848 (last access 15.09.2022. License: [CC BY-NC-ND 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/)

# ---
