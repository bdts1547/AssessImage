
from typing import List
import shutil
import sys
import os

from main_tracer import find_mask
import assess
import streamlit as st
import cv2
cv2.setNumThreads(0)

def main_gui():  
    st.title("Ứng dụng đánh giá chất lượng ảnh")
    # uploaded_files = st.file_uploader("Chọn file ảnh", type=["jpeg", "jpg", "png"], accept_multiple_files = True)
    
    with st.form("my-form", clear_on_submit=True):
        # file = st.file_uploader("FILE UPLOADER")
        uploaded_files = st.file_uploader("Chọn file ảnh", type=["jpeg", "jpg", "png"], accept_multiple_files = True)
        submitted = st.form_submit_button("Xác nhận")

    print(uploaded_files)
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
        
        # # Find SOD
        find_mask()

        # Find score symmetry
        os.system("conda activate py27 & python symmary.py")
        scores_sym = {}
        with open('score_symmetry.csv', 'r') as f:
            lines = f.read().splitlines()
            
        for line in lines:
            d = list(map(str, line.split(',')))
            scores_sym[d[0]] = list(map(float, d[1:]))


        results = []
        for uploaded_file in uploaded_files:
            # if not os.path.exists('data/upload'):
            #     os.makedirs('data/upload')
            # with open("data/upload/" + uploaded_file.name, "wb") as buffer:
            #     buffer.write(uploaded_file.getvalue())
            # # Find SOD
            # find_mask()

            # # Assess Image
            # result = []
            # for img in files:
            fn = uploaded_file.name
            score_sym = scores_sym[fn]
            img_path = os.path.join('data/upload/', fn)
            fn_mask = fn.split('.')[0] + '.png'
            mask_path = os.path.join('mask/upload/', fn_mask)
            rst = assess.assess_image(fn, img_path, mask_path, score_sym)
            results.append(rst)
        st.write(results)
        
        if os.path.exists('data/upload/'):
            shutil.rmtree('data/upload/') # remove folder data/upload/
        print('Done')

        
if __name__ == "__main__":
    main_gui()