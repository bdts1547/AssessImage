
from fastapi import FastAPI, UploadFile, File,  Form
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware

from fastapi.staticfiles import StaticFiles

from typing import List
import shutil
import sys
import os

from main_tracer import find_mask
import assess


sys.path.append("..")
sys.path.insert(0, "/data/upload")
sys.path.insert(0, "/mask/upload")



app = FastAPI()

@app.post("/assess_image/api/upload")
async def assess_image(files: List[UploadFile]):
    if not os.path.exists('uploadImg/'):
        os.makedirs('uploadImg/')

    if not os.path.exists('data/upload/'):
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
    find_mask()
    shutil.rmtree('data/upload/') # remove folder data/upload/



    # Assess Image
    result = []
    for img in files:
        fn = img.filename
        img_path = os.path.join('uploadImg/', fn)
        fn_mask = fn.split('.')[0] + '.png'
        mask_path = os.path.join('mask/upload/', fn_mask)

        rst = assess.assess_image(fn, img_path, mask_path)
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


@app.get("/")
async def index():
    content = """
        Go /assess_image/upload -- to upload image
    """

    return HTMLResponse(content=content)

