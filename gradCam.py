from __future__ import absolute_import, division, print_function, unicode_literals
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.models import Model
import nibabel as nib
from scipy import ndimage
import random
from numpy import expand_dims

# Display
from IPython.display import Image, display
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import SimpleITK as sitk
from skimage.transform import resize
import cv2
from PIL import Image, ImageOps

###---Load The model----###
INPUT_PATCH_SIZE=(128,128,128)
width = 128
height = 128
depth = 128
inputs = keras.Input((width, height, depth, 1))

model = tf.keras.models.load_model('/home/miranda/Documents/code/official/DetectAF/saved_model/cnn3d')
# dot_img_file1 = './cnn3d_model.png'
# tf.keras.utils.plot_model(model, to_file=dot_img_file1, show_shapes=True)

# model.summary()
# x_test = np.load('/home/miranda/Documents/code/official/DetectAF/data/x_test.npy')
# y_test = np.load('/home/miranda/Documents/code/official/DetectAF/data/y_test.npy')
# y_test = keras.utils.to_categorical(y_test, 2)
# print("Evaluate on test data")
# # results = model.evaluate(x_test, y_test, batch_size=2)
# # dict(zip(model.metrics_names, results))

# y_pred = model.predict(x_test, verbose=1)
# print(y_pred)

# Create a graph that outputs target convolution and output

grad_model = tf.keras.models.Model([model.inputs], [model.layers[16].output, model.output])
# grad_model = tf.keras.models.Model([model.inputs], [model.get_layer("conv_4").output, model.output])
# grad_model.summary()

# load image
# image
img_path = '/home/miranda/Documents/data/AF/V2.4/AF/bn__InverseWarped_4019401_brain_bet.nii.nii.gz'

scan = nib.load(img_path)
# Get raw data
image = scan.get_fdata()
# image = np.array(image)

# print(image.shape) 

io_img=tf.expand_dims(image, axis=-1)
io_img=tf.expand_dims(io_img, axis=0)
# ###----index of the class
# CLASS_INDEX=2

###--Compute GRADIENT
with tf.GradientTape() as tape:
    conv_outputs, predictions = grad_model(io_img)
    CLASS_INDEX = tf.argmax(predictions[0])
    loss = predictions[:, CLASS_INDEX]

# Extract filters and gradients
output = conv_outputs[0]
grads = tape.gradient(loss, conv_outputs)[0]
print(tf.reduce_sum(grads, axis=None).numpy()) 

# Average gradients spatially
weights = tf.reduce_mean(grads, axis=(0, 1, 2))
# Build a ponderated map of filters according to gradients importance
cam = np.zeros(output.shape[0:3], dtype=np.float32)

for index, w in enumerate(weights):
    cam += w * output[:, :, :, index]

from skimage.transform import resize
from matplotlib import pyplot as plt
capi=resize(cam,(128,128,128))

capi = np.maximum(capi,0)
heatmap = (capi - capi.min()) / (capi.max() - capi.min())
f, axarr = plt.subplots(2,3,figsize=(20,10));
f.suptitle('Grad-CAM')
slice_count=65 # axial
slice_count2=57 # coronal
slice_count3=65 # sagittal

# heatmap = np.array(heatmap)
   
# plot 
axial_ct_img=np.squeeze(image[slice_count, :,:])
axial_grad_cmap_img=np.squeeze(heatmap[slice_count,:, :])
axial_ct_img = ndimage.rotate(axial_ct_img, 90, reshape=True)
axial_grad_cmap_img = ndimage.rotate(axial_grad_cmap_img, 90, reshape=True)

coronal_ct_img=np.squeeze(image[:,:,slice_count2])
coronal_grad_cmap_img=np.squeeze(heatmap[:,:,slice_count2]) 
# coronal_ct_img = ndimage.rotate(coronal_ct_img, 90, reshape=True)
# coronal_grad_cmap_img = ndimage.rotate(coronal_grad_cmap_img, 90, reshape=True)

sagittal_ct_img = np.squeeze(image[:,slice_count3,:])
sagittal_grad_cmap_img=np.squeeze(heatmap[:,slice_count3,:]) 
sagittal_ct_img = ndimage.rotate(sagittal_ct_img, 180, reshape=True)
sagittal_grad_cmap_img = ndimage.rotate(sagittal_grad_cmap_img, 180, reshape=True)


img_plot = axarr[0,0].imshow(axial_ct_img, cmap='gray');
axarr[0,0].axis('off')
axarr[0,0].set_title('CT')
    
img_plot = axarr[0,1].imshow(axial_grad_cmap_img, cmap='jet');
axarr[0,1].axis('off')
axarr[0,1].set_title('Grad-CAM')

    
# axial_overlay=cv2.addWeighted(axial_ct_img,0.3,axial_grad_cmap_img, 0.7, 0)

axial_overlay = 0.3*axial_ct_img + 0.5*axial_grad_cmap_img + 0
    
img_plot = axarr[0,2].imshow(axial_overlay,cmap='jet');
axarr[0,2].axis('off')
axarr[0,2].set_title('Overlay')


img_plot = axarr[1,0].imshow(coronal_ct_img, cmap='gray');
axarr[1,0].axis('off')
axarr[1,0].set_title('CT')
    
img_plot = axarr[1,1].imshow(coronal_grad_cmap_img, cmap='jet');
axarr[1,1].axis('off')
axarr[1,1].set_title('Grad-CAM')
    
# Coronal_overlay=cv2.addWeighted(coronal_ct_img,0.3,coronal_grad_cmap_img, 0.6, 0)

Coronal_overlay = 0.3*coronal_ct_img + 0.5*coronal_grad_cmap_img + 0
    
img_plot = axarr[1,2].imshow(Coronal_overlay,cmap='jet');
axarr[1,2].axis('off')
axarr[1,2].set_title('Overlay')

# img_plot = axarr[1,0].imshow(sagittal_ct_img, cmap='gray');
# axarr[1,0].axis('off')
# axarr[1,0].set_title('CT')
    
# img_plot = axarr[1,1].imshow(sagittal_grad_cmap_img, cmap='jet');
# axarr[1,1].axis('off')
# axarr[1,1].set_title('Grad-CAM')
    
# sagittal_overlay = 0.3*sagittal_ct_img + 0.5*sagittal_grad_cmap_img + 0
    
# img_plot = axarr[1,2].imshow(sagittal_overlay,cmap='jet');
# axarr[1,2].axis('off')
# axarr[1,2].set_title('Overlay')

# plt.show()
f.savefig('/home/miranda/Documents/code/official/DetectAF/pic/4019401_1.png')

# f2, axarr2 = plt.subplots(2,2,figsize=(10,10));
# f2.suptitle('Grad-CAM')

# img_plot2 = axarr2[0,0].imshow(axial_ct_img, cmap='gray');
# axarr2[0,0].axis('off')
# axarr2[0,0].set_title('CT')
    
# img_plot2 = axarr2[0,1].imshow(axial_overlay, cmap='jet');
# axarr2[0,1].axis('off')
# axarr2[0,1].set_title('Grad-CAM')

# img_plot2 = axarr2[1,0].imshow(coronal_ct_img, cmap='gray');
# axarr2[1,0].axis('off')
# axarr2[1,0].set_title('CT')
    
# img_plot2 = axarr2[1,1].imshow(Coronal_overlay, cmap='jet');
# axarr2[1,1].axis('off')
# axarr2[1,1].set_title('Grad-CAM')
# f2.savefig('/home/miranda/Documents/code/official/DetectAF/pic/1266198.png')


# fig1, ax1 = plt.subplots()
# ax1.imshow(coronal_ct_img,cmap='gray')
# fig1.savefig('/home/miranda/Documents/code/3DCNN/pic/CT2.png')

# fig2, ax2 = plt.subplots()
# ax2.imshow(Coronal_overlay,cmap='jet')
# fig2.savefig('/home/miranda/Documents/code/3DCNN/pic/Cam2.png')

# fig3, ax3 = plt.subplots()
# ax3.imshow(axial_ct_img,cmap='gray')
# fig3.savefig('/home/miranda/Documents/code/3DCNN/pic/CT1.png')

# fig4, ax4 = plt.subplots()
# ax4.imshow(axial_overlay,cmap='jet')
# fig4.savefig('/home/miranda/Documents/code/3DCNN/pic/Cam1.png')

def save_and_display_gradcam(img3d, heatmap3d, slice_count=60, cam_path="/home/miranda/Documents/code/official/DetectAF/pic/cam3.jpg", alpha=0.4):
    # Load the original image
    img = img3d[slice_count, :,:]
    heatmap = heatmap3d[slice_count, :,:]

    img = np.stack((img,)*3, axis=-1)

    # Rescale heatmap to a range 0-255
    heatmap = np.uint8(255 * heatmap)

    # Use jet colormap to colorize heatmap
    jet = cm.get_cmap("jet")

    # Use RGB values of the colormap
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    # Create an image with RGB colorized heatmap
    jet_heatmap = keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = keras.preprocessing.image.img_to_array(jet_heatmap)

    # Superimpose the heatmap on original image
    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = keras.preprocessing.image.array_to_img(superimposed_img)

    # axial_overlay=cv2.addWeighted(img,0.3,jet_heatmap, 0.6, 0)

    gray_image = ImageOps.grayscale(superimposed_img)
    # Save the superimposed image
    superimposed_img.save(cam_path)



# save_and_display_gradcam(image, heatmap)