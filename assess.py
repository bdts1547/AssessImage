
# Backlit
import matplotlib.pyplot as plt
import os
import numpy as np
import cv2
import time
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
    s3 = (cr < 112).sum().astype(np.float64) # xCb < 112
    s4 = (cr > 143).sum().astype(np.float64) # xCb > 143

    s5 = s1 + s4
    s6 = s2 + s3
    s7 = s1 + s2
    s8 = s3 + s4

    # Calculate
    luminance = img_ycbcr[:,:,0]
    s9 = luminance.shape[0] * luminance.shape[1]
    s10 = (luminance < 32).sum().astype(np.float64)
    s11 = (luminance > 159).sum().astype(np.float64)


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


def save_image(img, img_thresh, title, filename):
    fig, ax = plt.subplots(1, 2, figsize=(10, 8))
    h, w, c = img.shape

    ax[0].set_title("Saliency detect {}".format(filename))
    ax[0].imshow(cv2.cvtColor(img_thresh, cv2.COLOR_BGR2RGB))

    ax[1].set_title(title)
    ax[1].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    x1 = [0, w-1]
    y1 = [h//3-1, h//3-1]
    y11 = [2*h//3-1, 2*h//3-1]
    x2 = [w//3-1, w//3-1]
    x22= [2*w//3-1, 2*w//3-1]
    y2 = [0, h-1]

    ax[1].plot(x1, y1, color='r', linestyle="-")
    ax[1].plot(x1, y11, color='r', linestyle="-")
    ax[1].plot(x2, y2, color='r', linestyle="-")
    ax[1].plot(x22, y2, color='r', linestyle="-")

    
    
    fig.savefig('layout/upload/{}'.format(filename))
    # plt.close()


def get_bbox(img_bin, img_rgb, threshold=108):
    """
    Parameters:
        img_bin: result of cv2.threshold
        img_rgb: Image rgb
        threshold: used to remove conneted components 

    """
    
    h, w, c = img_rgb.shape
    
    # find conneted component, coordinate bbox
    num_labels, labels, stats, centers = cv2.connectedComponentsWithStats(img_bin, 4, cv2.CV_32S)

    thresh = h * w / threshold # Area divide threshold
    bboxes = [] # Store x1, y1 , x2, y2 to plot cv.rectange
    new_centers = []
    for stat, center in zip(stats[1:], centers[1:]):   # stats[0] is background
        if stat[4] > thresh:    # Remove object noise (obj small)
            x1, y1 = stat[0], stat[1]
            x2, y2 = (stat[0] + stat[2]) , (stat[1] + stat[3]) 
            new_centers.append(center)
            bboxes.append([int(x1), int(y1), int(x2), int(y2)])
    
    img_bb = img_rgb.copy()
    x, y = w//2, h//2
    for bb, center in zip(bboxes, new_centers):
        x1, y1, x2, y2 = bb
        # xc, yc = (x1 + x2) // 2, (y1 + y2) // 2   # Use center bb
        xc, yc = center                             # Use center object
        img_bb = cv2.rectangle(img_bb, (x1, y1), (x2, y2), (255, 0, 0), 1)
        cv2.line(img_bb, (int(x), int(y)), (int(xc), int(yc)), (0,255,0), 1) # Plot center
        # cv.line(img_bb, (0, h//3), (w, h//3), (0,0,255), 1)
        # cv.line(img_bb, (0, 2*h//3), (w, 2*h//3), (0,0,255), 1)
        # cv.line(img_bb, (w//3, 0), (w//3, h), (0,0,255), 1)
        # cv.line(img_bb, (2*w//3, 0), (2*w//3, h), (0,0,255), 1)
 
    
    # print(f"Num obj: {len(new_centers)}")
    return img_bb, new_centers, bboxes


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


def detect_layout_center(img, centers, ratio=1/3):
    
    
    # ratio = 1/3, Acc = 0.7065
    # ratio = 1/6, Acc = 0.8184
    # ratio = 1/8, Acc = 0.7960
    # ratio = 1/8, Acc = 0.7886
    

    (h, w, c) = img.shape
    margin_width = w * ratio / 2 # Distance from image center to axis 1/3, 2/3
    
    p1 = np.asarray([w//2, 0])  # mid-top
    p2 = np.asarray([w//2, h])  # mid-bottom
    
    if len(centers) == 0:
        return False
 
    is_center = True
    for center in centers:
        # Calc distance from point to mid-line
        d = np.linalg.norm(np.cross(p2-p1, p1-center))/ np.linalg.norm(p2-p1)
        # print(d)
        if d > margin_width:
            is_center = False

    # for center in centers[1:]:
    #     d = distance.euclidean(center_img, center)
    #     if (d > threshold):
    #         isCenter = False
        
    
    return is_center


def detect_layout_onethird(img, centers, bboxes, ratio=1/3):
    # ratio = 1/3, Acc = 0.4179
    # ratio = 1/5, Acc = 0.8582
    # ratio = 1/6, Acc = 0.8930
    # ratio = 1/8, Acc = 0.8831
    
    (h, w, c) = img.shape
    
    g1 = np.asarray([w * 1/3, h * 1/3])    # left-top
    g2 = np.asarray([w * 1/3, h * 2/3])    # left-bottom
    
    g3 = np.asarray([w * 2/3, h * 1/3])    # right-top
    g4 = np.asarray([w * 2/3, h * 2/3])    # right-bottom

    margin_width = w * ratio / 2

    if len(centers) == 0:
        return False

    is_onethird = True
    for center in centers:
        # Calc distance from point to line 1-3,2-3
        d13 = np.linalg.norm(np.cross(g2-g1, g1-center)) / np.linalg.norm(g2-g1)
        d23 = np.linalg.norm(np.cross(g4-g3, g3-center)) / np.linalg.norm(g4-g3)
        d = min(d13, d23)

        # print(d)
        if d > margin_width:
            is_onethird = False

    return is_onethird


def check_is_center_or_onethird(img, centers):
    """
        Return True is center, otherwise onethird
    """

    (h, w, c) = img.shape
    
    p1 = np.asarray([w//2, 0])  # mid-top
    p2 = np.asarray([w//2, h])  # mid-bottom
    
    # gold-point
    g1 = np.asarray([w * 1/3, h * 1/3])    # left-top
    g2 = np.asarray([w * 1/3, h * 2/3])    # left-bottom
    
    g3 = np.asarray([w * 2/3, h * 1/3])    # right-top
    g4 = np.asarray([w * 2/3, h * 2/3])    # right-bottom
    
    d_sum_center = 0
    d_sum_onethird = 0
    for center in centers:
        # Calc distance from point to mid-line
        d_to_center = np.linalg.norm(np.cross(p2-p1, p1-center)) / np.linalg.norm(p2-p1)
        
        # Calc distance from point to 1-3, 2-3
        d13 = np.linalg.norm(np.cross(g2-g1, g1-center))/ np.linalg.norm(g2-g1)
        d23 = np.linalg.norm(np.cross(g4-g3, g3-center))/ np.linalg.norm(g4-g3)
        d_to_onethird = min(d13, d23)
        
        d_sum_center += d_to_center
        d_sum_onethird += d_to_onethird

    if d_sum_center <= d_sum_onethird:
        return True
    else:
        return False


def detect_layout(img_rgb, img_gray, filename, score_sym, threshold_sym=0.6):
    # print(filename)
    _, img_bin = cv2.threshold(img_gray, 100,255,cv2.THRESH_BINARY)
    img_bb, obj_centers, bboxes = get_bbox(img_bin, img_rgb)
    

    is_center = False
    is_onethird = False
    is_symmetry = max(score_sym) > threshold_sym
    
    
    # Detect center
    if detect_layout_center(img_bb, obj_centers):
        is_center = True
    
    # Detect one-third
    if detect_layout_onethird(img_bb, obj_centers, bboxes):
        is_onethird = True
    

    # Compare distince center/onethird
    if is_center and is_onethird:
        is_center = check_is_center_or_onethird(img_bb, obj_centers) # Return True is center, otherwise onethird
        is_onethird = not is_center

    if is_symmetry:
        if is_onethird: 
            save_image(img_bb, img_bin, "OneThird, Symmetry", filename)
            return "OneThird, Symmetry"
        elif is_center: 
            save_image(img_bb, img_bin, "Center, Symmetry", filename)
            return "Center, Symmetry"
        else:
            save_image(img_bb, img_bin, "Symmetry", filename)
            return "Symmetry"

    else:
        if is_onethird: 
            save_image(img_bb, img_bin, "OneThird", filename)
            return "OneThird"
        elif is_center: 
            save_image(img_bb, img_bin, "Center", filename)
            return "Center"
        else:
            save_image(img_bb, img_bin, "Not Layout", filename)
            return "No Layout"
    

def percent_low_contrast(image, threshold=0.65, lower_percentile=1, upper_percentile=99):
   
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

    # percent = 100 - (ratio / _max * 100)
    percent = (ratio / _max * 100)
    
    return percent


def percent_blur_(img, threshold=1000):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    fm = cv2.Laplacian(gray, cv2.CV_64F).var()

    if fm < 0: fm = 0
    if fm > threshold: fm = threshold

    percent = 100 - (fm / threshold * 100)

    return percent


def assess_image(filename, path_image, path_pred_map, score_sym):
    print('Asssessing')
    img = cv2.imread(path_image)
    gray = cv2.imread(path_pred_map, 0)

    
    # Backlit
    # start = time.time()
    is_backlit, thres_bl = backlit_detect(img, 0.8)
    if is_backlit:
        if thres_bl < 0.15: thres_bl = 0.15    # Min các của giá trị s11/s9
        elif thres_bl > 0.8: thres_bl = 0.8
        percent_backlit = 100 - ((thres_bl - 0.15) / (0.8 - 0.15) * 100)
    else:
        percent_backlit = 0
    # end = time.time()
    # print("Time running Backlit: {:.2f}".format(end-start))
    
    
    # Layout
    # start = time.time()
    layout = detect_layout(img, gray, filename, score_sym)
    # end = time.time()
    # print("Time running Layout: {:.2f}".format(end-start))
    
    # Contrast
    # start = time.time()
    percent_lcontrast = percent_low_contrast(img)
    # end = time.time()
    # print("Time running Contrast: {:.2f}".format(end-start))

    # Sharpness
    # start = time.time()
    percent_blur = percent_blur_(img)
    # end = time.time()
    # print("Time running Sharpness: {:.2f}".format(end-start))
  
    mask_path = 'layout/upload/' + filename
    rst = { 
            'File name': filename,
            'Backlit': percent_backlit,
            'Contrast': percent_lcontrast,
            'Blur': percent_blur,
            'Layout': layout,
            'mask_path': mask_path,
            'max_score_symmetry': max(score_sym),
           }

    return rst

