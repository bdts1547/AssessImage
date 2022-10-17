
from typing import List
import shutil
import sys
import os

from main_tracer import find_mask
import assess
import streamlit as st


def main_gui():  
    st.title("Ứng dụng đánh giá chất lượng ảnh")
    uploaded_files = st.file_uploader("Chọn file ảnh", type=["jpeg", "jpg", "png"], accept_multiple_files = True)
    if uploaded_files is not None:
        for uploaded_file in uploaded_files:
            if not os.path.exists('data/upload'):
                os.makedirs('data/upload')
            with open("data/upload/" + uploaded_file.name, "wb") as buffer:
                buffer.write(uploaded_file.getvalue())
            # Find SOD
            find_mask()

            # Assess Image
            # result = []
            # for img in files:
            #     fn = img.filename
            #     img_path = os.path.join('uploadImg/', fn)
            #     fn_mask = fn.split('.')[0] + '.png'
            #     mask_path = os.path.join('mask/upload/', fn_mask)
            #     rst = assess.assess_image(fn, img_path, mask_path)
            #     result.append(rst)
            # st.write(result)
        
if __name__ == "__main__":
    main_gui()