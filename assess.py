
# Backlit
import matplotlib.pyplot as plt
import os
import numpy as np
import cv2
from skimage import io, data, color

# Layout
from scipy.spatial import distance

# Contrast
from imutils.paths import list_images
import argparse
import imutils

# Blur
from imutils import paths
import argparse


def backlit_detect(img, threshold=0.5):
    img_ycbcr = color.rgb2ycbcr(img)
    h, w, c = img_ycbcr.shape
    s0 = h * w
    # Calculate Cb
    cb = img_ycbcr[:,:,1]
    s1 = (cb < 112).sum().astype(np.float64) # xCb < 112
    s2 = (cb > 143).sum().astype(np.float64) # xCb > 143

    # Calculate Cr
    cr = img_ycbcr[:,:,2]
    s3 = (cr < 112).sum() # xCb < 112
    s4 = (cr > 143).sum() # xCb > 143

    s5 = s1 + s4
    s6 = s2 + s3
    s7 = s1 + s2
    s8 = s3 + s4

    # Calculate
    luminance = img_ycbcr[:,:,0]
    s9 = luminance.shape[0] * luminance.shape[1]
    s10 = (luminance < 32).sum()
    s11 = (luminance > 159).sum()


    is_candidate = False
    if (s5/s0 < 0.012 and s6/s0 < 0.012) or ((s7/s0 < 0.008 and s1*s2 == 0) \
            or (s8/s0 < 0.008 and s3*s4 == 0) or (s1*s2 == 0 and s3*s4 == 0)):
        # print("candidate")
        is_candidate = True

    # Check condition: low light intensity
    if s5/s0 >= 0.012 or s6/s0 >= 0.012:
        if s10/s9 > 0.4:
            # print("candidate")
            is_candidate = True

    # Check backlit 
    if s11/s9 < threshold and is_candidate: # Default 0.004
        # print("candidate: {}, Thres: {:.2f}".format(is_candidate, s11/s9))
        return True, s11/s9
    else:
        # print("candidate: {}, Thres: {:.2f}".format(is_candidate, s11/s9))
        return False, s11/s9



def get_bbox(thresh, img):
    # find conneted component, coordinate bbox
    num_labels, labels, stats, centers = cv2.connectedComponentsWithStats(thresh, 4, cv2.CV_32S)
    bboxes = [] # Store x1, y1 , x2, y2 to plot cv.rectange
    h, w, c = img.shape

    for stat in stats[1:]:   # stats[0] is background
        x1, y1 = stat[0], stat[1]
        x2, y2 = (stat[0] + stat[2]) , (stat[1] + stat[3]) 
        
        bboxes.append([int(x1), int(y1), int(x2), int(y2)])
    
    img_bb = img.copy()
    center_img = tuple(centers[0].astype('uint8'))
    x, y = w//2, h//2
    for bb, center in zip(bboxes, centers[1:]):
        x1, y1, x2, y2 = bb
        xc, yc = (x1 + x2) // 2, (y1 + y2) // 2
        img_bb = cv2.rectangle(img_bb, (x1, y1), (x2, y2), (255, 0, 0), 1)
        cv2.line(img_bb, (int(x), int(y)), (int(xc), int(yc)), (0,255,0), 1) # Plot center
        # cv.line(img_bb, (0, h//3), (w, h//3), (0,0,255), 1)
        # cv.line(img_bb, (0, 2*h//3), (w, 2*h//3), (0,0,255), 1)
        # cv.line(img_bb, (w//3, 0), (w//3, h), (0,0,255), 1)
        # cv.line(img_bb, (2*w//3, 0), (2*w//3, h), (0,0,255), 1)
        

    # plt.imshow(cv.cvtColor(img_bb, cv.COLOR_BGR2RGB))
    # plt.show()
    return img_bb, centers, bboxes

def is_point_in_rectangle(gpoints, bbox, img_bb):
    x1, y1, x2, y2 = bbox
    # gp1 = gpoints[0]
    # gp2 = gpoints[3]
    # img_bb = cv.rectangle(img_bb, (int(gp1[0]), int(gp1[1])), (int(gp2[0]), int(gp2[1])), (0, 0, 255), 1)

    for point in gpoints:
        # img_bb = cv.circle(img_bb, (int(point[0]), int(point[1])), radius=1, color=(0, 0, 255), thickness=-1)


        if (x1 < point[0] < x2 and y1 < point[1] < y2):
            return True

    return False

def detect_layout_center(img, centers, ratio=0.1):
    (h, w, c) = img.shape
    threshold = max(w, h) * ratio
    center_img = centers[0]
    isCenter = True
    if len(centers) <= 1:
        isCenter = False
        # return ""

    for center in centers[1:]:
        d = distance.euclidean(center_img, center)
        if (d > threshold):
            isCenter = False
        
    # print('No. Object: ', len(centers) - 1)

    # if (isCenter):
    #     return ('Center')
    # else:
    #     return ""
    return isCenter


def detect_layout_onethird(img, centers, bboxes, ratio=1/6):
    (h, w, c) = img.shape
    gpoint1_img = (w * 1/3, h * 1/3)
    gpoint2_img = (w * 2/3, h * 1/3) 
    gpoint3_img = (w * 1/3, h * 2/3) 
    gpoint4_img = (w * 2/3, h * 2/3) 
    gpoint_img = [gpoint1_img, gpoint2_img, gpoint3_img, gpoint4_img]

    margin_x = w * ratio
    margin_y = h * ratio

    isOneThird = True
    if len(centers) <= 1:
        isOneThird = False

    for center, bbox in zip(centers[1:], bboxes):
        # Check point in rectangle
        if(not(is_point_in_rectangle(gpoint_img, bbox, img))):
            isOneThird = False
    
        # Check margin < threshold ?
    
    # if (isOneThird):
    #     return ('OneThird')
    # else:
    #     return ""
    return isOneThird


def detect_layout(img_rgb, img_gray):
            # img_rgb = cv2.imread(img_path)
            # img_gray = cv2.imread(path_pred_map, 0)
            _, thresh = cv2.threshold(img_gray, 100,255,cv2.THRESH_BINARY)
            img_bb, obj_centers, bboxes = get_bbox(thresh, img_rgb)
            # title = layout(img_bb, obj_centers, bboxes)
            # print(title)
            if (detect_layout_center(img_bb, obj_centers)):
                return "Center"
            elif (detect_layout_onethird(img_bb, obj_centers, bboxes)):
                return "OneThird"
            else:
                return "No Layout"



def percent_low_contrast(image, threshold=0.8, lower_percentile=1, upper_percentile=99):
   
    from skimage.util.dtype import dtype_range, dtype_limits
    from skimage.color import rgb2gray, rgba2rgb
    
    image = np.asanyarray(image)
    if image.ndim == 3:
        if image.shape[2] == 4:
            image = rgba2rgb(image)
        if image.shape[2] == 3:
            image = rgb2gray(image)

    dlimits = dtype_limits(image, clip_negative=False)
    limits = np.percentile(image, [lower_percentile, upper_percentile])
    ratio = (limits[1] - limits[0]) / (dlimits[1] - dlimits[0])
    
    _max = threshold 
    if ratio > threshold: 
        ratio = _max
    else:
        ratio = ratio
        if ratio < 0: ratio = 0

    percent = 100 - (ratio / _max * 100)
    
    return percent



def percent_blur_(img, threshold=1000):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    fm = cv2.Laplacian(gray, cv2.CV_64F).var()

    if fm < 0: fm = 0
    if fm > threshold: fm = threshold

    percent = 100 - (fm / threshold * 100)

    return percent



def assess_image(filename, path_image, path_pred_map):
    print('go assess')
    img = cv2.imread(path_image)
    gray = cv2.imread(path_pred_map, 0)

    # Backlit
    is_backlit, thres_bl = backlit_detect(img, 0.8)
    if is_backlit:
        if thres_bl < 0.15: thres_bl = 0.15    # Min các của giá trị s11/s9
        elif thres_bl > 0.8: thres_bl = 0.8
        percent_backlit = 100 - ((thres_bl - 0.15) / (0.8 - 0.15) * 100)
    else:
        percent_backlit = 0

    # Layout
    layout = detect_layout(img, gray)

    # Contrast
    percent_lcontrast = percent_low_contrast(img)

    # Sharpness
    percent_blur = percent_blur_(img)
    
    # Write assess
    # color = (255, 0, 0)
    # cv2.putText(img, "Backlit: {:.0f}%".format(percent_backlit), (10, 30),
    #         cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    # cv2.putText(img, "Layout: {}".format(layout), (10, 55),
    #         cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    # cv2.putText(img, "Low contrast: {:.0f}%".format(percent_lcontrast), (10, 80),
    #         cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    # cv2.putText(img, "Blur: {:.0f}%".format(percent_blur), (10, 105),
    #         cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    # plt.figure(figsize=(10, 8))
    # plt.imshow(img[:,:,::-1])
    # plt.axis('off')
    # plt.show()
    
    rst = { 
            'File name': filename,
            'Backlit': percent_backlit,
            'Low contrast': percent_lcontrast,
            'Blur': percent_blur,
            'Layout': layout
           }

    return rst
