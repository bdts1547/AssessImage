
from fastapi import FastAPI, UploadFile, File,  Form
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware

from fastapi.staticfiles import StaticFiles

from typing import List
import shutil
import sys
import os
import time

from main_tracer import find_mask
import assess


# sys.path.append("..")
# sys.path.insert(0, "/data/upload/")
# sys.path.insert(0, "/mask/upload/")



app = FastAPI()

@app.post("/assess_image/api/upload")
async def assess_image(files: List[UploadFile]):
    if not os.path.exists('uploadImg/'):
        os.makedirs('uploadImg/')

    if os.path.exists('data/upload/'):
        # Clear folder
        shutil.rmtree('data/upload/')
        os.makedirs('data/upload/')
    else:
        os.makedirs('data/upload/')

    # Save image to UploadImg
    for image in files:
        with open("uploadImg/" + str(image.filename), "wb") as buffer:
            shutil.copyfileobj(image.file, buffer)

        # Copy file to data/upload
        file_path_1 = os.path.join('uploadImg', image.filename)
        file_path_2 = os.path.join('data/upload', image.filename)
        shutil.copy(file_path_1, file_path_2)
        
    
    
    # Find SOD
    start = time.time()
    find_mask()
    end = time.time()
    print("Time running SOD: {:.2f}".format(end-start))

    # Find score symmetry
    start = time.time()
    os.system("conda activate py27 & python detect_symmetry.py")
    end = time.time()
    print("Time running Score-symmetry: {:.2f}".format(end-start))

    scores_sym = {}
    with open('score_symmetry.csv', 'r') as f:
        lines = f.read().splitlines()
        # print("Score sym:", lines)
        
    for line in lines:
        d = list(map(str, line.split(',')))
        scores_sym[d[0]] = list(map(float, d[1:]))
    print(scores_sym)


    shutil.rmtree('data/upload/') # remove folder data/upload/



    # Assess Image
    result = []
    for img in files:
        fn = img.filename
        score_sym = scores_sym[fn]
        img_path = os.path.join('uploadImg/', fn)

        len_tail = len(fn.split('.')[-1])
        fn_mask = fn[:-len_tail-1] + '.png'
        print(fn_mask)

        mask_path = os.path.join('mask/upload/', fn_mask)
        rst = assess.assess_image(fn, img_path, mask_path, score_sym)
        result.append(rst)
    
    
    

    return {"Results": result}
        



    




@app.get("/assess_image/upload")
async def upload_image():
    content = """
<body>
<form action="/assess_image/api/upload/" enctype="multipart/form-data" method="post">
Image samples: <input name="files" type="file" multiple><br>
<input type="submit">
</form>
</body>
"""
    return HTMLResponse(content=content)


@app.get("/assess_image/layout/upload/{filename}")
async def show_image(filename: str):
    path = os.path.join('layout/upload', filename)

    return FileResponse(path)


@app.get("/")
async def index():
    content = """
        <body>
            Click <a href="http://127.0.0.1:8000/assess_image/upload">Here</a> to upload image
        </body>
    """

    return HTMLResponse(content=content)