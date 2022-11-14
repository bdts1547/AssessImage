import numpy as np
import cv2
import os
import pickle
from skimage import io, data, color
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

with open("MODELS/model_backlit_svm/backlit_10fea_svm_c100_linear.pkl", "rb") as f:
    clf = pickle.load(f)

def get_feature(img_rgb):
   
    # img = cv2.imread(os.path.join(path_folder, img))
    # img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_ycbcr = color.rgb2ycbcr(img_rgb)

    h, w, c = img_ycbcr.shape
    s0 = h * w

    # Luminance
    luminance = img_ycbcr[:,:,0]
    lum_bot = (luminance < 32).sum()
    lum_top = (luminance > 159).sum()
    lum_m1 = (luminance > 32)
    lum_m2 = (luminance < 159)
    lum_mid = (lum_m1 == lum_m2).sum()

    lum_mean = luminance.mean()
    lum_var = luminance.var()
    lum_max = np.max(luminance)


    # Cb
    cb = img_ycbcr[:,:,1]
    cb_bot = (cb < 112).sum() # xCb < 112
    cb_top = (cb > 143).sum() # xCb > 143
    cb_m1 = (cb > 112)
    cb_m2 = (cb < 143)
    cb_mid = (cb_m1 == cb_m2).sum() # 112 < xCb < 143

    cb_mean = cb.mean()
    cb_var = cb.var()
    cb_max = np.max(cb)

    # Cr
    cr = img_ycbcr[:,:,2]
    cr_bot = (cr < 112).sum() # xCr < 112
    cr_top = (cr > 143).sum() # xCr > 143
    cr_m1 = (cr > 112)
    cr_m2 = (cr < 143)
    cr_mid = (cr_m1 == cr_m2).sum() # 112 < xCr < 143

    cr_mean = cr.mean()
    cr_var = cr.var()
    cr_max = np.max(cr)

    

    if (cb_bot + cb_top + cb_mid) != s0:
        assert False

    if (cr_bot + cr_top + cr_mid) != s0:
        assert False

    if (lum_bot + lum_top + lum_mid) != s0:
        assert False

    ### paper
    # s5 = cb_bot + cr_top
    # s6 = cb_top + cr_bot
    # s7 = cb_bot + cb_top
    # s8 = cr_bot + cr_top
    # s10 = lum_bot
    # s11 = lum_top
    
    # f1 = s5 
    # f2 = s6 
    # f3 = s7 
    # f4 = s8 
    # f5 = cb_bot*cb_top / s0
    # f6 = cr_bot*cr_top / s0
    # f7 = s10 
    # f8 = s11

    feature = [lum_bot, lum_mid, lum_top, cb_bot, cb_mid, cb_top, cr_bot, cr_mid, cr_top, lum_var]#,lum_mean lum_max, cb_mean, cb_var, cb_max, cr_mean, cr_var, cr_max]# f1, f2, f3, f4]

    return np.array(feature) / s0

def predict_backlit(img_rgb):
    feature = get_feature(img_rgb)
    feature = np.expand_dims(feature, axis=0)

    proba = clf.predict_proba(feature)
    percent_normal = proba[0][0]
    percent_backlit = proba[0][1]

    score = percent_normal * 10.0
    

    return percent_backlit, score


if __name__ == "__main__":
    img_bgr = cv2.imread('uploadImg/sharp_10.jpg')
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    print(predict_backlit(img_rgb))