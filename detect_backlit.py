from tensorflow import keras
import numpy as np
import cv2
import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def predict_backlit(img_rgb):
    # Load model
    model = keras.models.load_model('MODELS/model_backlit')

    img_ycbcr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2YCR_CB)
    img_ycbcr = cv2.resize(img_ycbcr, (256, 256))

    img = np.array(img_ycbcr, dtype="float") / 255.0
    img = np.expand_dims(img, axis=0)

    prob = model.predict(img)

    return np.squeeze(prob)


# if __name__ == "__main__":
#     img_bgr = cv2.imread('uploadImg/backlit_2.png')
#     img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
#     print(predict_backlit(img_rgb))