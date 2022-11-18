# AssessImage
## Description
  The application will evaluate the image quality based on 4 criteria of backlit, contrast, blur, composition.

## Installation
- Clone project
  ```
    git clone https://github.com/bdts1547/AssessImage.git
  ```
- Download, extract **[model_symmetry](https://drive.google.com/file/d/1O5nzjuWbkFM68sFY2qYj6uFI0H14FOkZ/view?usp=share_link)** and put it in the folder AssessImage.
  ```
  AssessImage
  |-- caffe
  |-- model
  |-- MODELS (here)
  |-- ...
  ```
- Create environment python 3.7.13 with **[Anaconda](https://www.anaconda.com/)** and install requirements
  ```
    conda create -n my_env python=3.7.13 -y
    conda activate my_env
    pip install -r requirements.txt
  ```
- Create environment python 2.7.18 with name "py27" to call symmetry detection
  ```
    conda create -n py27 python=2.7.18 -y
    conda activate py27
    pip install matplotlib numpy scikit-image scipy protobuf
  ```


## Run
- Run with streamlit
  ```
    conda activate my_env
    streamlit run streamlit_app.py
  ```
