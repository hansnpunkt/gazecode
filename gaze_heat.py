
# coding: utf-8

# In[104]:


import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import xml.etree.ElementTree as ET 
from scipy import interpolate
from scipy.ndimage.filters import gaussian_filter


# In[2]:


def create_gaze_map(input_image, input_gaze, screen_res, plot=False):
    """
    Creates a heatmap given the eyetracking data captured from a participant looking at the specified image
    :param input_image: rgb image of size [m,n,3]
    :param input_gaze: eye-tracking data of size [k,2] (first column X, second column Y)
    :param screen_res: [screen width, screen height]
    :param plot: produce plots true/false
    :return: gaze heatmap of same size [m,n]
    """

    # Unpack input data
    x_screen = input_gaze[:, 0]
    y_screen = input_gaze[:, 1]
    image_res_x = input_image.shape[1]
    image_res_y = input_image.shape[0]
    screen_res_x = screen_res[0]
    screen_res_y = screen_res[1]

    # Calculate stimulus coordinates
    sx = image_res_x / screen_res_x
    sy = image_res_y / screen_res_y
    s = max(sx, sy)
    dx = max(image_res_x * (1/sx-1/sy)/2, 0)
    dy = max(image_res_y * (1/sy-1/sx)/2, 0)
    x_stimulus = s * (x_screen-dx)
    y_stimulus = s * (y_screen-dy)

    # Create Heatmap by creating 2d histogram and gaussian smoothing
    n_bins = [np.round(image_res_x/10), np.round(image_res_y/10)]
    heatmap, x_edges, y_edges = np.histogram2d(x_stimulus,
                                               y_stimulus,
                                               bins=n_bins,
                                               range=[[0, image_res_x], [0, image_res_y]])
    heatmap = gaussian_filter(heatmap, sigma=3)

    # Bilinear interpolate to acquire full heatmap - other interpolation options are cubic/quintic
    x_bin_centers = x_edges[:-1] + (x_edges[1] - x_edges[0])/2
    y_bin_centers = y_edges[:-1] + (y_edges[1] - y_edges[0])/2
    f = interpolate.interp2d(x_bin_centers, y_bin_centers, heatmap.T, kind='linear')
    x_new = np.arange(0, image_res_x, 1)
    y_new = np.arange(0, image_res_y, 1)
    heatmap_interpolated = f(x_new, y_new)

    # Scale to [0,1]
    heatmap_interpolated = heatmap_interpolated * 1/np.max(heatmap_interpolated) 

    if plot:
        # create figure
        fig, (ax, ax2, ax3) = plt.subplots(ncols=3, figsize=(20,15))
        
        # Plot original image with eye-tracking trace
        ax.imshow(input_image)
        ax.plot(x_stimulus, y_stimulus, 'r')
        ax.axis('off')
        # Plot generated heatmap
        #plt.subplot(3, 1, 2)
        ax2.imshow(heatmap_interpolated, cmap='jet')
        ax2.axis('off')
        # Plot gaze-based attention of image
        #plt.subplot(3, 1, 3)
        ax3.imshow(input_image)
        ax3.imshow(np.dstack((np.zeros((image_res_y, image_res_x, 3)), -(heatmap_interpolated-1))))
        ax3.axis('off')
        plt.show()

    return heatmap_interpolated


# In[6]:

#
# image_dir = r'C:\Users\hansn-admin\Documents\Courses\term3\ai_vision_language_sound\project_gaze\data\VOCdevkit\VOC2012\JPEGImages'
# gaze_dir = r'C:\Users\hansn-admin\Documents\Courses\term3\ai_vision_language_sound\project_gaze\data\VOC2012_gaze\samples'
# geometry_file = r'C:/Users/hansn-admin/Documents/Courses/term3/ai_vision_language_sound/project_gaze/data/VOC2012_gaze/geometry.txt'
#
# image_name = '2010_006088.jpg'
# gaze_name = '006_2010_006088.jpg.txt'
#
# # Load image and get resolution
# img = plt.imread(os.path.join(image_dir, image_name))
#
# # Load gaze data, get X,Y eye position
# gaze_data = np.genfromtxt(os.path.join(gaze_dir, gaze_name), delimiter='\t')
# gaze_data_panda = pd.read_csv(os.path.join(gaze_dir, gaze_name), delim_whitespace=True, header=None)
# saccade_data = gaze_data_panda.loc[gaze_data_panda[5] == "S"]
# saccade_data = saccade_data.values[:, 3:5]
#
# gaze = gaze_data[:, 3:5]
#
# # Load screen geometry
# geometry = np.genfromtxt(geometry_file, delimiter='\t')
# screen = geometry[3:5]
#
# # Create gaze heat map
# gaze_map = create_gaze_map(img, gaze, screen, plot=True)
#
#
# # In[30]:
#
#
# image_dir = r'C:\Users\hansn-admin\Documents\Courses\term3\ai_vision_language_sound\project_gaze\data\VOCdevkit\VOC2012\JPEGImages'
# gaze_dir = r'C:\Users\hansn-admin\Documents\Courses\term3\ai_vision_language_sound\project_gaze\data\VOC2012_gaze\samples'
# geometry_file = r'C:/Users/hansn-admin/Documents/Courses/term3/ai_vision_language_sound/project_gaze/data/VOC2012_gaze/geometry.txt'
#
# image_name = '2010_006088.jpg'
# gaze_name = '006_2010_006088.jpg.txt'
#
# # Load image and get resolution
# img = plt.imread(os.path.join(image_dir, image_name))
#
# # Load gaze data, get X,Y eye position
# gaze_data_panda = pd.read_csv(os.path.join(gaze_dir, gaze_name), delim_whitespace=True, header=None)
# saccade_data = gaze_data_panda.loc[gaze_data_panda[5] == "S"]
# saccade = saccade_data.values[:, 3:5]
# saccade = saccade.astype(float)
#
# # Load screen geometry
# geometry = np.genfromtxt(geometry_file, delimiter='\t')
# screen = geometry[3:5]
#
# # Create gaze heat map
# gaze_map = create_gaze_map(img, saccade, screen, plot=True)
#
#
# # In[9]:
#
#
# # all gaze data as list
# gaze_list = os.listdir(r'C:\Users\hansn-admin\Documents\Courses\term3\ai_vision_language_sound\project_gaze\data\VOC2012_gaze\samples')
# image_list = os.listdir(r'C:\Users\hansn-admin\Documents\Courses\term3\ai_vision_language_sound\project_gaze\data\VOCdevkit\VOC2012\JPEGImages')
#
#
# # In[10]:
#
#
# # plot any of the many
#
# for k in range(1):
#     image_name = os.path.splitext(gaze_list[k])[0][4:]
#     gaze_name = gaze_list[k]
#
#     if os.path.splitext(gaze_list[k])[0][4:] in image_list:
#         # Load image and get resolution
#         img = plt.imread(os.path.join(image_dir, image_name))
#
#         # Load gaze data, get X,Y eye position
#         gaze_data = np.genfromtxt(os.path.join(gaze_dir, gaze_name), delimiter='\t')
#         gaze = gaze_data[:, 3:5]
#
#         # Load screen geometry
#         geometry = np.genfromtxt(geometry_file, delimiter='\t')
#         screen = geometry[3:5]
#
#         # Create gaze heat map
#         gaze_map = create_gaze_map(img, gaze, screen, plot=True)
#
#
#
#
# # In[38]:
#
#
# # only plot takingphoto
#
# # read in as pandas data frame
# action_list = pd.read_csv(r'C:\Users\hansn-admin\Documents\Courses\term3\ai_vision_language_sound\project_gaze\data\VOCdevkit\VOC2012\ImageSets\Action\usingcomputer_trainval.txt', delim_whitespace=True, header=None)
# # filter
# action_list = action_list.loc[action_list[2] == 1]
#
# for k in range(1):
#     image_name = action_list[0].iloc[k] + '.jpg'
#     gaze_name = '010_' + image_name + '.txt'
#
#     # Load image and get resolution
#     img = plt.imread(os.path.join(image_dir, image_name))
#
#     # Load gaze data, get X,Y eye position
#     gaze_data = np.genfromtxt(os.path.join(gaze_dir, gaze_name), delimiter='\t')
#     gaze = gaze_data[:, 3:5]
#
#     # Load screen geometry
#     geometry = np.genfromtxt(geometry_file, delimiter='\t')
#     screen = geometry[3:5]
#
#     # Create gaze heat map
#     gaze_map = create_gaze_map(img, gaze, screen, plot=True)
#
#
#
# # In[39]:
#
#
# # RUNNING
# # SACCADES ONLY
#
# # read in as pandas data frame
# action_list = pd.read_csv(r'C:\Users\hansn-admin\Documents\Courses\term3\ai_vision_language_sound\project_gaze\data\VOCdevkit\VOC2012\ImageSets\Action\running_trainval.txt', delim_whitespace=True, header=None)
# # filter
# action_list = action_list.loc[action_list[2] == 1]
#
# for k in range(1):
#     image_name = action_list[0].iloc[k] + '.jpg'
#     gaze_name = '010_' + image_name + '.txt'
#
#     # Load image and get resolution
#     img = plt.imread(os.path.join(image_dir, image_name))
#
#     # Load gaze data, get X,Y eye position
#     gaze_data_panda = pd.read_csv(os.path.join(gaze_dir, gaze_name), delim_whitespace=True, header=None)
#     saccade_data = gaze_data_panda.loc[gaze_data_panda[5] == "S"]
#     saccade = saccade_data.values[:, 3:5]
#     saccade = saccade.astype(float)
#
#     # Load screen geometry
#     geometry = np.genfromtxt(geometry_file, delimiter='\t')
#     screen = geometry[3:5]
#
#     # Create gaze heat map
#     gaze_map = create_gaze_map(img, saccade, screen, plot=True)
#
#
#
# # In[36]:
#
#
# # WALKING
# # SACCADES ONLY
#
# # read in as pandas data frame
# action_list = pd.read_csv(r'C:\Users\hansn-admin\Documents\Courses\term3\ai_vision_language_sound\project_gaze\data\VOCdevkit\VOC2012\ImageSets\Action\walking_trainval.txt', delim_whitespace=True, header=None)
# # filter
# action_list = action_list.loc[action_list[2] == 1]
#
# for k in range(1):
#     image_name = action_list[0].iloc[k] + '.jpg'
#     gaze_name = '010_' + image_name + '.txt'
#
#     # Load image and get resolution
#     img = plt.imread(os.path.join(image_dir, image_name))
#
#     # Load gaze data, get X,Y eye position
#     gaze_data_panda = pd.read_csv(os.path.join(gaze_dir, gaze_name), delim_whitespace=True, header=None)
#     saccade_data = gaze_data_panda.loc[gaze_data_panda[5] == "S"]
#     saccade = saccade_data.values[:, 3:5]
#     saccade = saccade.astype(float)
#
#     # Load screen geometry
#     geometry = np.genfromtxt(geometry_file, delimiter='\t')
#     screen = geometry[3:5]
#
#     # Create gaze heat map
#     gaze_map = create_gaze_map(img, saccade, screen, plot=True)
#
#
#
# # In[355]:
#
#
# # get xml files
#
# box_dir = r'C:\Users\hansn-admin\Documents\Courses\term3\ai_vision_language_sound\project_gaze\data\VOCdevkit\VOC2012\Annotations'
# box_file = '2007_000032.xml'
#
# # parse an xml file by name
# tree = ET.parse(os.path.join(box_dir, box_file))
# root = tree.getroot()
#
# # one specific item attribute
# print('Item #2 attribute:')
# print(root.tag)
#
#
# # In[357]:
#
#
# # find all names
# persons = tree.findall('.//object/name')
# # is person vector
# is_person = np.zeros((len(persons)))
# # respective bounding box array
# bnbox_array = np.zeros((4, len(persons)))
#
# if len(persons) > 0:
#     for i, names in enumerate(persons):
#         if names.text == 'person':
#             is_person[i] = 1
#
# # find all bnboxes of persons
#     xmin = tree.findall('.//object/bndbox/xmin')
#     for i, boxes in enumerate(xmin):
#         bnbox_array[0][i] = boxes.text
#     ymin = tree.findall('.//object/bndbox/ymin')
#     for i, boxes in enumerate(ymin):
#         bnbox_array[1][i] = boxes.text
#     xmax = tree.findall('.//object/bndbox/xmax')
#     for i, boxes in enumerate(xmax):
#         bnbox_array[2][i] = boxes.text
#     ymax = tree.findall('.//object/bndbox/ymax')
#     for i, boxes in enumerate(ymax):
#         bnbox_array[3][i] = boxes.text
#
# print(is_person)
# print(bnbox_array)
#
