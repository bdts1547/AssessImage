
# Backlit
import matplotlib.pyplot as plt
import os
import numpy as np
import cv2
from skimage import io, data, color

# Layout
from scipy.spatial import distance
from mirror_symmetry import *

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

def save_image(img, img_thresh, title, filename, point_1, point_2):
    fig, ax = plt.subplots(1, 3, figsize=(10, 8))
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

    img_sym = img.copy()
    cv2.line(img_sym, point_1, point_2, (0,0,255), 3)
    ax[2].imshow(cv2.cvtColor(img_sym, cv2.COLOR_BGR2RGB))
    
    
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

def detect_layout_center(img, centers, ratio=1/6):
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

def detect_layout_onethird(img, centers, bboxes, ratio=1/6):
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

def is_symmetry(image, r, theta, thresh = 15, rate=1/3): 
    # 15, 1/3 , acc = 0.338

    thresh = thresh
    # Tính góc giữ trục đối xứng và trục Ox hoặc Oy
    def cal_angle_with(point_1, point_2, axis=None): # y1, y2 ~ min height, max height
        # a.b = |a||b|cos(a)
        x1, y1 = point_1
        x2, y2 = point_2

        
        if axis == 'Oy':  # vector relative to the vertical axis
            vector_sym = [x2-x1, y2-y1]
            vector_cmp = [0, y2]
        
        elif axis == 'Ox': # vector relative to the horizontal axis
            vector_sym = [x2-x1, y2-y1]
            vector_cmp = [x2, 0]
        else:
            print("Error: Must parameter 'axis' = Ox or Oy")
            return

        unit_vector_sym = vector_sym / np.linalg.norm(vector_sym)
        unit_vector_cmp = vector_cmp / np.linalg.norm(vector_cmp)
        dot_product = np.dot(unit_vector_sym, unit_vector_cmp)
        angle = np.arccos(dot_product)
        return np.degrees(angle)
    
    # x nằm trong đoạn x_min - xmax
    def is_x_within(x, x_min, x_max):
        if (x > x_min and x < x_max):
            return True
        return False
    
    def check_vertical():
        # symmetry through Oy
        point_1 = (int((r-0*np.sin(theta))/np.cos(theta)), 0) # (x, y_min)
        point_2 = (int((r-(h-1)*np.sin(theta))/np.cos(theta)), h-1) # (x, y_max)
        degree = cal_angle_with(point_1, point_2, axis='Oy')
        # print(point_1, '|', point_2)
        # print("Degree:", degree)

        x1, y1 = point_1
        x2, y2 = point_2

        x_center_min = int(w // 2 - (rate/2 * w))
        x_center_max = int(w // 2 + (rate/2 * w))


        # Vertital
        x_valid = is_x_within(x1, x_center_min, x_center_max) and is_x_within(x2, x_center_min, x_center_max)
        if (degree < thresh) and x_valid:
            return True, point_1, point_2

        return False, point_1, point_2

    def check_horizontal():
    # symmetry through Ox
        # Horizontal
        point_1 = (0, int(r / np.sin(theta))) # (x_min, y)
        point_2 = (w-1, int((r-(w-1)*np.cos(theta)) / np.sin(theta))) # (x_max, y)
        
        degree = cal_angle_with(point_1, point_2, axis='Ox')
        # print(point_1, '|', point_2)
        # print("Degree:", degree)

        x1, y1 = point_1
        x2, y2 = point_2

        y_center_min = int(h // 2 - (rate/2 * h))
        y_center_max = int(h // 2 + (rate/2 * h))


        # Vertital
        y_valid = is_x_within(y1, y_center_min, y_center_max) and is_x_within(y2, y_center_min, y_center_max)
        
        if degree < thresh and y_valid:
            return True, point_1, point_2

        return False, point_1, point_2




    h, w, c = image.shape


    _is_symmetry, p1, p2 = check_vertical()
    if _is_symmetry:
        return _is_symmetry, p1, p2
    else:
        return check_horizontal()
    


    # draw plot 

    # return _is_symmetry, p1, p2


def detect_layout(img_rgb, img_gray, filename, image_path):
    # print(filename)
    _, img_bin = cv2.threshold(img_gray, 100,255,cv2.THRESH_BINARY)
    img_bb, obj_centers, bboxes = get_bbox(img_bin, img_rgb)
    r, theta = detecting_mirrorLine(image_path)
    

    is_center = False
    is_onethird = False
    is_not_layout = False
    
    _is_symmetry, point_1, point_2 = is_symmetry(img_bb, r, theta)
    save_image(img_bb, img_bin, "test", filename, point_1, point_2)

    if (_is_symmetry):
        # save_image(img_bb, img_bin, "Symmetry", filename, point_1, point_2)
        return "Symmetry"

    if detect_layout_center(img_bb, obj_centers):
        save_image(img_bb, img_bin, "Center", filename, point_1, point_2)
        is_center = True
        # return "Center"
    
    if detect_layout_onethird(img_bb, obj_centers, bboxes):
        save_image(img_bb, img_bin, "OneThird", filename, point_1, point_2)
        is_onethird = True
        # return "OneThird"
    
    # if is_center or is_onethird or is_symmetry:
    #     is_not_layout = False
    #     save_image(img_bb, img_bin, "Not Layout", filename, point_1, point_2)
    #     return "No Layout"

    if is_center and is_onethird:
        is_center = check_is_center_or_onethird(img_bb, obj_centers) # Return True is center, otherwise onethird
        is_onethird = not is_center

    if _is_symmetry:
        if is_onethird: return "OneThird, Symmetry"
        if is_center: return "Center, Symmetry"
    else:
        if is_onethird: return "OneThird"
        if is_center: return "Center"

    save_image(img_bb, img_bin, "Not Layout", filename, point_1, point_2)
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
    print('Asssessing')
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
    layout = detect_layout(img, gray, filename, path_image)

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
    
    mask_path = 'layout/upload/' + filename
    rst = { 
            'File name': filename,
            'Backlit': percent_backlit,
            'Low contrast': percent_lcontrast,
            'Blur': percent_blur,
            'Layout': layout,
            'mask_path': mask_path,
           }

    return rst
