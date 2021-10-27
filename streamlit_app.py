import time
import streamlit as st
import numpy as np
import pandas as pd
import cv2
from PIL import Image, ImageOps
from streamlit_MS import doing_fusion
from streamlit_PAN import doing_fusion_PAN
# from skimage import io
import matplotlib.pyplot as plt
import tifffile
from io import BytesIO
import base64
from random import randint

st.title("Image Fusion Example")

# uploaded_file1 = st.file_uploader("Input the first image file", type=["jpg","bmp","png","tif"])
# if uploaded_file1 is not None:
#     # Convert the file to an opencv image.
#     file_bytes1 = np.asarray(bytearray(uploaded_file1.read()), dtype=np.uint8)
#     opencv_image1 = cv2.imdecode(file_bytes1, 1)
#
#     # Now do something with the image! For example, let's display it:
#     st.image(opencv_image1, channels="BGR")


def get_image_download_link(img):
    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    href = f'<a href="data:file/jpg;base64,{img_str}">Download result</a>'
    return href

def get_image_download_link_cv2(img):
    is_success, buffer = cv2.imencode(".bmp", img)
    io_buf = BytesIO(buffer)
    # decode
    # decode_img = cv2.imdecode(np.frombuffer(io_buf.getbuffer(), np.uint8), -1)
    img_str = base64.b64encode(io_buf.getvalue()).decode()
    href = f'<a href="data:file/bmp;base64,{img_str}">Download result</a>'
    return href

if 'key' not in st.session_state:
    st.session_state.key = str(randint(1000, 100000000))

file_up1 = st.file_uploader("Upload First Image, size is no large than 3000", type=["jpg","bmp","png","tif"], key=st.session_state.key)
if file_up1 is not None:
    # display image that user uploaded
    head1, sep1, tail1 = str(file_up1.name).partition(".")
    if str(tail1) == 'tif':
        image1 = tifffile.imread(file_up1)
        image1 = image1/4095
        if len(image1.shape) == 3:
            image1 = image1[:,:,:3]
    else:
        image1 = Image.open(file_up1)
        image1 = np.array(image1)/255

    # if image1.shape[0] > 3000 or image1.shape[1] > 3000:
    #     st.session_state.pop('key')
    #     st.experimental_rerun()

    image1 = image1.astype('float32')
    st.write("image size:", image1.shape, "dtype:", image1.dtype, "file type：", str(tail1))
    st.image(image1, caption='Uploaded First Image.', use_column_width=True)



file_up2 = st.file_uploader("Upload Second Image, size is no large than 3000", type=["jpg","bmp","png","tif"])
if file_up2 is not None:
    # display image that user uploaded
    head2, sep2, tail2 = str(file_up2.name).partition(".")
    if str(tail2) == 'tif':
        image2 = tifffile.imread(file_up2)
        image2 = image2 / 4095
        if len(image2.shape) == 3:
            image2 = image2[:, :, :3]
    else:
        image2 = Image.open(file_up2)
        image2 = np.array(image2) / 255

    # if image2.shape[0] > 3000 or image2.shape[1] > 3000:
    #     st.session_state.pop('key')
    #     st.experimental_rerun()

    image2 = image2.astype('float32')
    st.write("image size:", image2.shape, "dtype:", image2.dtype, "file type：", str(tail2))
    st.image(image2, caption='Uploaded Second Image.', use_column_width=True)


if st.button('fusion') and file_up1 is not None and file_up2 is not None:
    if (image1.size != image2.size):
        st.write("image size is not the same")
    else:
        st.write("doing fusion...")
        if len(image1.shape) == 3:
            fused_im, time_cost = doing_fusion(image1, image2)
            st.write("processing time: {:.3} second".format(time_cost))
            st.image(fused_im, caption='Fused Image.', use_column_width=True)

        else:
            fused_im, time_cost = doing_fusion_PAN(image1, image2)
            st.write("processing time: {:.3} second".format(time_cost))
            st.image(fused_im, caption='Fused Image.', use_column_width=True)

        # result = Image.fromarray(np.array(fused_im * 255).astype('uint8'))
        # st.markdown(get_image_download_link(result), unsafe_allow_html=True)

        if len(image1.shape) == 3:
            fused_im = cv2.cvtColor(fused_im, cv2.COLOR_RGB2BGR)

        # result2 = np.array(fused_im * 255).astype('uint8')
        # st.markdown(get_image_download_link_cv2(result2), unsafe_allow_html=True)

        # cv2.imwrite("IFCNN_streamlit.bmp", (fused_im * 255).astype('uint8'))
        # with open("IFCNN_streamlit.bmp", "rb") as fp:
        #     btn = st.download_button(
        #         label="Download IMAGE",
        #         data=fp,
        #         file_name="IFCNN_streamlit.bmp",
        #         mime="image/bmp"
        #     )

        result2 = np.array(fused_im * 255).astype('uint8')
        is_success, buffer = cv2.imencode(".bmp", result2)
        io_buf = BytesIO(buffer)
        btn = st.download_button(label="Download IMAGE", data=io_buf, file_name="IFCNN_streamlit.bmp", mime="image/bmp")


