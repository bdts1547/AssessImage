
from typing import List
import shutil
import sys
import os
import time

from main_tracer import find_mask
import assess
import streamlit as st
import cv2
import numpy as np
cv2.setNumThreads(0)

def main_gui():  
    st.title("Ứng dụng đánh giá chất lượng ảnh")
    # uploaded_files = st.file_uploader("Chọn file ảnh", type=["jpeg", "jpg", "png"], accept_multiple_files = True)
    
    with st.form("my-form", clear_on_submit=True):
        # file = st.file_uploader("FILE UPLOADER")
        uploaded_files = st.file_uploader("Chọn file ảnh", type=["jpeg", "jpg", "png"], accept_multiple_files = True)
        submitted = st.form_submit_button("Xác nhận")

    # print(uploaded_files)
    if uploaded_files is not None:
        if os.path.exists('data/upload/'):
            # Clear folder
            shutil.rmtree('data/upload/')
            os.makedirs('data/upload/')
        else:
            os.makedirs('data/upload/')
        
        
        for uploaded_file in uploaded_files:
            with open("data/upload/" + uploaded_file.name, "wb") as buffer:
                buffer.write(uploaded_file.getvalue())
        
        # Find SOD
        find_mask()

        # Find score symmetry
        start = time.time()
        os.system("conda activate py27 & python detect_symmetry.py")
        end = time.time()
        print("Time running Score-symmetry: {:.2f}".format(end-start))

        scores_sym = {}
        with open('score_symmetry.csv', 'r') as f:
            lines = f.read().splitlines()
            
        for line in lines:
            d = list(map(str, line.split(',')))
            scores_sym[d[0]] = list(map(float, d[1:]))


        results = []
        path_imgs = []
        path_mask = []
        for uploaded_file in uploaded_files:
            fn = uploaded_file.name
            score_sym = scores_sym[fn]
            img_path = os.path.join('data/upload/', fn)
            fn_mask = fn.split('.')[0] + '.png'
            mask_path = os.path.join('mask/upload/', fn_mask)
            rst = assess.assess_image(fn, img_path, mask_path, score_sym)
            results.append(rst)
            path_imgs.append(img_path)
            path_mask.append(mask_path)


        def header(title):
            return '<h2 style="font-size: 20px;">{}</h2>'.format(title)

        if results:

            # Show sidebar
            with st.sidebar:
                backlit = '<p style="font-size: 20px; margin-left: 12px; color: rgb(9, 171, 59);">{:.2f}</p>'.format(results[0]['Backlit'])
                contrast = '<p style="font-size: 20px; margin-left: 12px; color: rgb(9, 171, 59);">{:.2f}</p>'.format(results[0]['Contrast'])
                blur = '<p style="font-size: 20px; margin-left: 12px; color: rgb(9, 171, 59);">{:.2f}</p>'.format(results[0]['Blur'])
                layout = '<p style="font-size: 20px; margin-left: 12px; color: rgb(9, 171, 59);">{}</p>'.format(results[0]['Layout'])
                score = '<p style="font-size: 20px; margin-left: 12px; color: rgb(9, 171, 59);">{:.2f}</p>'.format(results[0]['score'])


                # st.header('Ngược sáng')
                st.markdown(header('Ngược sáng'), unsafe_allow_html=True)
                st.markdown(backlit, unsafe_allow_html=True)

                # st.header('Tương phản')
                st.markdown(header('Tương phản'), unsafe_allow_html=True)
                st.markdown(contrast, unsafe_allow_html=True)

                # st.header('Mờ')
                st.markdown(header('Độ mờ'), unsafe_allow_html=True)
                st.markdown(blur, unsafe_allow_html=True)

                # st.header('Bố cục')
                st.markdown(header('Bố cục'), unsafe_allow_html=True)
                st.markdown(layout, unsafe_allow_html=True)

                st.markdown(header('Điểm'), unsafe_allow_html=True)
                st.markdown(score, unsafe_allow_html=True)
                
            
            img = cv2.imread(path_imgs[0])
            mask = cv2.imread(path_mask[0])
            concat = np.hstack((img, mask))
            st.image(cv2.cvtColor(concat, cv2.COLOR_BGR2RGB), width=None)

        st.markdown(header('Ảnh'), unsafe_allow_html=True)
        st.write(results)
        
        if os.path.exists('data/upload/'):
            shutil.rmtree('data/upload/') # remove folder data/upload/
        print('Done')

        
if __name__ == "__main__":
    main_gui()