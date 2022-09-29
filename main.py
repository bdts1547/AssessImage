
from fastapi import FastAPI, UploadFile, File,  Form
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware

from fastapi.staticfiles import StaticFiles

from typing import List
from utils import *
import shutil
import sys
import os

from main_tracer import find_mask


sys.path.append("..")
sys.path.insert(0, "/data/upload")
sys.path.insert(0, "/mask/upload")
print(sys.path)



app = FastAPI()

@app.post("/assess_image/api/upload")
async def assess_image(files: List[UploadFile]):
    if not os.path.exists('uploadImg/'):
        os.makedirs('uploadImg/')

    # Save image to UploadImg
    for image in files:
        with open("uploadImg/" + str(image.filename), "wb") as buffer:
            shutil.copyfileobj(image.file, buffer)

        with open("data/upload/" + str(image.filename), "wb") as buffer2:
            shutil.copyfileobj(image.file, buffer2)
    
    # Copy to TRACER/data
    
    print(111)
    find_mask()
    print(222)
    # for path_img in os.listdir('uploadImg/'):
    #     shutil.copy(path_img, 'TRACER/data/uploaded/')
    
    # shutil.rmtree('uploadImg/')

    return {"filename": 1}
        



    




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



