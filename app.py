import os
from flask import Flask, redirect, jsonify, request, url_for, render_template, flash
from PIL import Image
from io import BytesIO
import base64
from utils import *



app = Flask(__name__)

app.config["IMAGE_UPLOADS"] = "static/upload/"


@app.route("/",methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            print('No file attached in request')
            return redirect(request.url)
        file = request.files['file']
        
        if file.filename == '':
            print('No file selected')
            return redirect(request.url)
        
        if file: # Need check file
            filename = file.filename
            file_path = os.path.join(app.root_path, 'static/upload', filename)
            file.save(file_path)
            arr = cv2.imread(file_path)
            print(arr)
            
            img = Image.open(file.stream)
            with BytesIO() as buf:
                img.save(buf, 'jpeg')
                image_bytes = buf.getvalue()
            encoded_string = base64.b64encode(image_bytes).decode()         
        return render_template('index.html', img_data=encoded_string), 200
    else:
        return render_template('index.html', img_data=""), 200


##### Route to upload image sssssssssssssssssssssssssssssssssssssssssssssssssssssadsad
@app.route('/upload-image', methods=['GET', 'POST'])
def upload_image():
    if request.method == "POST":
        if request.files:
            image = request.files["image"]
            image.save(os.path.join(app.config["IMAGE_UPLOADS"], image.filename))
            return render_template("upload_image.html", uploaded_image=image.filename)
    return render_template("upload_image.html")


@app.route('/uploads/<filename>')
def send_uploaded_file(filename=''):
    from flask import send_from_directory
    return send_from_directory(app.config["IMAGE_UPLOADS"], filename)


if __name__ == "__main__":
    app.run(debug=True)