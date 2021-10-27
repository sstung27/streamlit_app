import time
import streamlit as st
import numpy as np
import pandas as pd
import cv2
from PIL import Image, ImageOps
from streamlit_MS import doing_fusion
from streamlit_PAN import doing_fusion_PAN
from streamlit_mnist import Testing
# from skimage import io
import matplotlib.pyplot as plt
import tifffile
from io import BytesIO
import base64
from random import randint
from streamlit_drawable_canvas import st_canvas

# uploaded_file1 = st.file_uploader("Input the first image file", type=["jpg","bmp","png","tif"])
# if uploaded_file1 is not None:
#     # Convert the file to an opencv image.
#     file_bytes1 = np.asarray(bytearray(uploaded_file1.read()), dtype=np.uint8)
#     opencv_image1 = cv2.imdecode(file_bytes1, 1)
#
#     # Now do something with the image! For example, let's display it:
#     st.image(opencv_image1, channels="BGR")


def MNIST():
    st.title("MNIST Example")
    # st.sidebar.header("Configuration")
    st.write("The is a three-layer fully-connected network and the weights are trained from unrectified method. "
             "The training dataset is the MNIST dataset, as shown below. The training accuracy is 0.9912 and testing accuracy is 0.9773")
#     st.image(Image.open('mnist.jpeg'), width=300)
    st.write('draw a digit then classification')

    # Specify canvas parameters in application
    # stroke_width = st.sidebar.slider("Stroke width: ", 1, 25, 3)
    # realtime_update = st.sidebar.checkbox("Update in realtime", True)
    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
        stroke_width=int(6),
        stroke_color="#FFFFFF",
        background_color="#000000",
        update_streamlit=False,
        height=int(100),
        width=int(100),
        drawing_mode="freedraw",
        display_toolbar=True,
#         key="full_app",
    )

    # Do something interesting with the image data and paths
    if canvas_result.image_data is not None:
        st.image(canvas_result.image_data[:,:,1])
#         image = np.array(canvas_result.image_data[:, :, 1] / 255).astype('float32')

        # image2 = cv2.resize(image, (24, 24))
        # image = np.zeros((28, 28)).astype('float32')
        # image[2:26,2:26] = image2
        # st.image(image)

#         scaled_img, prediction = Testing(image)
#         st.image(scaled_img)
#         pred = prediction.sort(descending=True)
#         precision1 = "Number "+str(np.array(pred.indices[0][0]))+" is "+str(np.array(pred.values[0][0]))
#         precision2 = "Number " + str(np.array(pred.indices[0][1])) + " is " + str(np.array(pred.values[0][1]))
#         precision3 = "Number " + str(np.array(pred.indices[0][2])) + " is " + str(np.array(pred.values[0][2]))
#         new_title1 = f'<p style="font-family:sans-serif; color:Green; font-size: 42px;">{precision1}</p>'
#         new_title2 = f'<p style="font-family:sans-serif; color:Green; font-size: 42px;">{precision2}</p>'
#         new_title3 = f'<p style="font-family:sans-serif; color:Green; font-size: 42px;">{precision3}</p>'
#         st.markdown(new_title1, unsafe_allow_html=True)
#         st.markdown(new_title2, unsafe_allow_html=True)
#         st.markdown(new_title3, unsafe_allow_html=True)


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


def FUSION():
    st.title("Image Fusion Example")
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


PAGES = {
    "MNIST": MNIST,
    "Image Fusion": FUSION,
}
page = st.sidebar.selectbox("Page:", options=list(PAGES.keys()))
PAGES[page]()
