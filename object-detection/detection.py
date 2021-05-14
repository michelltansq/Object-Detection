import streamlit as st
import time
import requests
from PIL import Image
import torch
import torchvision
import cv2

st.title('Object Detection Tool')
st.sidebar.header('Upload your image here:')
st.sidebar.write('Accept png, jpg and jpeg only.')
uploaded_file = st.sidebar.file_uploader('',type=['png', 'jpg', 'jpeg'],accept_multiple_files=False)

if uploaded_file is None:
    url = 'https://source.unsplash.com/random/'
    image = Image.open(requests.get(url, stream=True).raw)
else:
    image = Image.open(uploaded_file)
    st.sidebar.success('Uploaded!')

model_s = torch.hub.load('ultralytics/yolov5', 'yolov5s')
model_m = torch.hub.load('ultralytics/yolov5', 'yolov5m')
model_l = torch.hub.load('ultralytics/yolov5', 'yolov5l')
model_x = torch.hub.load('ultralytics/yolov5', 'yolov5x')

st.sidebar.header('Select your model:')
option = st.sidebar.selectbox(
     'What model would you choose?',
     ['yolov5s', 'yolov5m', 'yolov5l', 'yolov5x'], index=0
)

st.subheader('Result:')

if (option == 'yolov5x'):
    img = model_x(image)
elif (option == 'yolov5l'):
    img = model_l(image)
elif (option == 'yolov5m'):
    img = model_m(image)
else: 
    img = model_s(image)
img.render()
result = Image.fromarray(img.imgs[0])

st.image(result, use_column_width = True)
