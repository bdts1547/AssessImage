# from sklearn.svm import SVC
# from sklearn.calibration import CalibratedClassifierCV
from skimage.filters import laplace, sobel, roberts

import pickle
import cv2
import numpy as np


def predict_blur(image_gray):

    # Load model
    with open("MODELS/model_blur/svm_c100_linear_SDM.pkl", "rb") as f:
        clf = pickle.load(f)

    lap_feat = laplace(image_gray)
    sob_feat = sobel(image_gray)
    rob_feat = roberts(image_gray)
    feature = [ lap_feat.mean(),lap_feat.var(),np.amax(lap_feat),
                sob_feat.mean(),sob_feat.var(),np.max(sob_feat),
                rob_feat.mean(),rob_feat.var(),np.max(rob_feat)]

    proba = clf.predict_proba([feature])
    y_pred = np.argmax(proba, axis=1)
    
    # print(proba.shape)
    percent_blur = proba[0][0]
    percent_sharp = proba[0][1]

    return percent_blur, percent_sharp


if __name__ == "__main__":
    gray = cv2.imread('uploadImg/ktx.jpg', 0)
    print(predict_blur(gray))






