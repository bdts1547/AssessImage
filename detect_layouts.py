import numpy as np
import cv2


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


def detect_layout_center(img, centers, ratio=1/3):
    # ratio = 1/3, Acc = 0.7065
    # ratio = 1/6, Acc = 0.8184
    # ratio = 1/8, Acc = 0.7960
    # ratio = 1/8, Acc = 0.7886
    

    (h, w, c) = img.shape
    margin_width = w * ratio / 3 # Distance from image center to axis 1/3, 2/3
    
    p1 = np.asarray([w//2, 0])  # mid-top
    p2 = np.asarray([w//2, h])  # mid-bottom
    
    if len(centers) == 0:
        return False, 0
 
    is_center = True
    max_score = 0
    for center in centers:
        # Calc distance from point to mid-line
        d = np.linalg.norm(np.cross(p2-p1, p1-center))/ np.linalg.norm(p2-p1)
        score = 1 - d / margin_width

        if max_score < score:
            max_score = score
        if d > margin_width:
            is_center = False
            break

    # for center in centers[1:]:
    #     d = distance.euclidean(center_img, center)
    #     if (d > threshold):
    #         isCenter = False
    if is_center == False:
        max_score = 0
    
    return is_center, max_score


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

    margin_width = w * ratio / 3

    if len(centers) == 0:
        return False, 0

    is_onethird = True
    max_score = 0
    for center in centers:
        # Calc distance from point to line 1-3,2-3
        d13 = np.linalg.norm(np.cross(g2-g1, g1-center)) / np.linalg.norm(g2-g1)
        d23 = np.linalg.norm(np.cross(g4-g3, g3-center)) / np.linalg.norm(g4-g3)
        d = min(d13, d23)
        score = 1 - d / margin_width

        if max_score < score:
            max_score = score
        if d > margin_width:
            is_onethird = False
            break

    if is_onethird == False:
        max_score = 0

    return is_onethird, max_score


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
    _, img_bin = cv2.threshold(img_gray, 100,255,cv2.THRESH_BINARY)
    img_bb, obj_centers, bboxes = get_bbox(img_bin, img_rgb)
    
    # is_center = False
    # is_onethird = False
    max_score_sym = max(score_sym)
    is_symmetry = max_score_sym > threshold_sym
    
    
    # Detect center
    is_center, score_center = detect_layout_center(img_bb, obj_centers)

    
    # Detect one-third
    is_onethird, score_onethird = detect_layout_onethird(img_bb, obj_centers, bboxes)
    

    # Compare distince center/onethird
    if is_center and is_onethird:
        is_center = check_is_center_or_onethird(img_bb, obj_centers) # Return True is center, otherwise onethird
        is_onethird = not is_center

    if is_symmetry:
        if is_onethird: 
            # save_image(img_bb, img_bin, "Onethird, Symmetry", filename)
            return "Một phần ba, Đối xứng", [score_onethird, max_score_sym]
        elif is_center: 
            # save_image(img_bb, img_bin, "Center, Symmetry", filename)
            return "Trung tâm, Đối xứng", [score_center, max_score_sym]
        else:
            # save_image(img_bb, img_bin, "Symmetry", filename)
            return "Đối xứng", [max_score_sym]

    else:
        if is_onethird: 
            # save_image(img_bb, img_bin, "Onethird", filename)
            return "Một phần ba", [score_onethird]
        elif is_center: 
            # save_image(img_bb, img_bin, "Center", filename)
            return "Trung tâm", [score_center]
        else:
            # save_image(img_bb, img_bin, "Not Layout", filename)
            return "Không tìm được", [0]


if __name__ == "__main__":
    img_bgr = cv2.imread("uploadImg/sharp_10.jpg")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    mask_gray = cv2.imread("mask/upload/sharp_10.png", 0)
    filename = "sharp_10.jpg"
    layout, scores_layout = detect_layout(img_rgb, mask_gray, filename, score_sym=[0.7, 0.4, 0.3])
    score_layout = (sum(scores_layout) / len(scores_layout)) * 10
    print(layout, score_layout)

