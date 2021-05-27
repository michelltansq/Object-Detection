import streamlit as st
import time
import requests
from PIL import Image
import torch
import torchvision
import cv2

# Set the title of the web app as "Object Detection Tool"
st.title('Object Detection Tool')

# Creating a sidebar
# Set the header of the sidebar
st.sidebar.header('Upload your image here:')

# Add text above the upload button
st.sidebar.write('Accept png, jpg and jpeg only.')

# Add a file uploader in the sidebar
uploaded_file = st.sidebar.file_uploader('',type=['png', 'jpg', 'jpeg'],accept_multiple_files=False)

# if the user uploads a file it will show "Uploaded!" if not, the program will show a random image from the unsplash.com
if uploaded_file is None:
    url = 'https://source.unsplash.com/random/'
    image = Image.open(requests.get(url, stream=True).raw)
else:
    image = Image.open(uploaded_file)
    st.sidebar.success('Uploaded!')

# each model acts as a level of analysis which s being the smallest and x being the most in depth
model_s = torch.hub.load('ultralytics/yolov5', 'yolov5s')
model_m = torch.hub.load('ultralytics/yolov5', 'yolov5m')
model_l = torch.hub.load('ultralytics/yolov5', 'yolov5l')
model_x = torch.hub.load('ultralytics/yolov5', 'yolov5x')

# this is the title of the dropdown bar
st.sidebar.header('Select your model:')

# this is the options of the dropdown bar
option = st.sidebar.selectbox(
     'What model would you choose?',
     ['yolov5s', 'yolov5m', 'yolov5l', 'yolov5x'], index=0
)

# Subheader of the web app
st.subheader('Result:')

# if else for the rendering depending on the option chosen
if (option == 'yolov5x'):
    img = model_x(image)
elif (option == 'yolov5l'):
    img = model_l(image)
elif (option == 'yolov5m'):
    img = model_m(image)
else: 
    img = model_s(image)
    
# render image
img.render()
result = Image.fromarray(img.imgs[0])

# display image
st.image(result, use_column_width = True)
