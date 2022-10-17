from assess import *
from PIL import Image
from imutils import paths
import argparse
import cv2
import matplotlib.pyplot as plt
import os
from skimage import io, data, color
from skimage.exposure import is_low_contrast
import numpy as np

def convert_files(path_folder, prefix):

    list_images = os.listdir(path_folder)
    for i, filename in enumerate(list_images):
        old_file = os.path.join(path_folder, filename)
        new_file = os.path.join(path_folder, prefix + '_' + str(i) + '.png')

        im = Image.open(old_file)
        im.save(new_file)

    for filename in list_images:
        old_file = os.path.join(path_folder, filename)
        os.remove(old_file)

def rename_files(path_folder, prefix):
    for i, filename in enumerate(os.listdir(path_folder)):
        old_file = os.path.join(path_folder, filename)
        new_file = os.path.join(path_folder, prefix + '_' + str(i) + '.png')
        os.rename(old_file, new_file)

def plot_dataset(path_folders, xlabel, title):
    # creating the dataset
    data = {}
    list_folder = os.listdir(path_folders)
    list_folder.sort()
    for folder in list_folder:
        data[folder] = len(os.listdir(os.path.join(path_folders, folder)))

    print(data)
    courses = list(data.keys())
    values = list(data.values())
    
    fig = plt.figure()
    
    # creating the bar plot
    plt.bar(courses, values)
    
    plt.xlabel(xlabel)
    plt.ylabel("No. of images")
    plt.title(title)
    plt.show()

# Evaluate layout
def evaluate_each_layout(path_folders_images, layout=None):
    # layout: center| symmetry | onethird
    print("Evaluating...")

    list_folders = os.listdir(path_folders_images)

    x, y, pred, imgs_name = [], [], [], []

    # Process symmetry
    if layout == 'symmetry':
        for folder in list_folders:
            if folder == layout:
                list_images = os.listdir(os.path.join(path_folders_images, folder, 'img'))
                list_images.sort()
                for filename in list_images:
                    if filename.split('.')[1] == 'png':
                        image_path = os.path.join(path_folders_images, folder, 'img',filename)
                        img = cv2.imread(image_path)
                        r, theta = detecting_mirrorLine(image_path)
                        x.append([r, theta, img])
                        y.append(1)
                        imgs_name.append(filename)
            else:
                list_images = os.listdir(os.path.join(path_folders_images, folder, 'img'))
                list_images.sort()
                for filename in list_images:
                    if filename.split('.')[1] == 'png':
                        image_path = os.path.join(path_folders_images, folder, 'img',filename)
                        img = cv2.imread(image_path)
                        r, theta = detecting_mirrorLine(image_path)
                        x.append([r, theta, img])
                        y.append(0)
                        imgs_name.append(filename)

           
    else:
        for folder in list_folders:
            if folder == layout:
                list_images = os.listdir(os.path.join(path_folders_images, folder, 'img'))
                list_maps = os.listdir(os.path.join(path_folders_images, folder, 'mask'))
                list_images.sort()
                list_maps.sort()
                for filename in list_images:
                    if filename.split('.')[1] == 'png':
                        img = cv2.imread(os.path.join(path_folders_images, folder, 'img',filename))
                        img_gray = cv2.imread(os.path.join(path_folders_images, folder, 'mask', filename), 0)

                        x.append([img, img_gray])
                        y.append(1)
                        imgs_name.append(filename)
            else:
                list_images = os.listdir(os.path.join(path_folders_images, folder, 'img'))
                list_maps = os.listdir(os.path.join(path_folders_images, folder, 'mask'))
                list_images.sort()
                list_maps.sort()

                for filename in list_images:
                    if filename.split('.')[1] == 'png':
                        img = cv2.imread(os.path.join(path_folders_images, folder, 'img',filename))
                        img_gray = cv2.imread(os.path.join(path_folders_images, folder, 'mask', filename), 0)

                        x.append([img, img_gray])
                        y.append(0)
                        imgs_name.append(filename)

    # prediction
    if layout == 'center':
        for img_rgb, img_gray in x:
            _, img_bin = cv2.threshold(img_gray, 100,255,cv2.THRESH_BINARY)
            img_bb, obj_centers, bboxes = get_bbox(img_bin, img_rgb)
            tmp = detect_layout_center(img_bb, obj_centers)
            pred.append(tmp)
    
    elif layout == 'onethird':
        for img_rgb, img_gray in x:
            _, img_bin = cv2.threshold(img_gray, 100,255,cv2.THRESH_BINARY)
            img_bb, obj_centers, bboxes = get_bbox(img_bin, img_rgb)
            tmp = detect_layout_onethird(img_bb, obj_centers, bboxes)
            pred.append(tmp)

    elif layout == 'symmetry':
        for r, theta, img in x:
            tmp,_,_ = is_symmetry(img, r, theta)
            pred.append(tmp)
    
    else:
        print("Error: Parameter layout is center or onethird or symmetry")
        return

    
    


    rst = np.array(pred) == np.array(y)
    id_true = np.where(rst == True)
    id_false = np.where(rst == False)
    acc = rst.sum() / len(x)
    imgs_name = np.array(imgs_name)
    # print("Threshold: {:.4f}, Acc: {:.4f}".format(threshold, acc))
    print("Images fail:", imgs_name[id_false])
    print("Accuracy: {:.4f}".format(acc))


    return acc

def detect_blur_1(img, threshold=500):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # fm = variance_of_laplacian(gray)
    fm = cv2.Laplacian(gray, cv2.CV_64F).var()

    if fm < threshold:
        return True
    else: return False

def evaluate_blur(path_positive, path_negative, threshold, is_plot=False):
    print("Evaluating")
    x, y, pred, imgs_name = [], [], [], []
    blur_imgs = os.listdir(path_positive)
    normal_imgs = os.listdir(path_negative)
    blur_imgs.sort()
    normal_imgs.sort()
    for filename in blur_imgs:
        # print(filename)
        if filename.split('.')[1] == 'png':
            img = cv2.imread(os.path.join(path_positive, filename))
            x.append(img)
            y.append(1)
            imgs_name.append(filename)

    for filename in normal_imgs:
        if filename.split('.')[1] == 'png':
            img = cv2.imread(os.path.join(path_negative, filename))
            x.append(img)
            y.append(0)
            imgs_name.append(filename)

    if is_plot:
        accs = []
        threshold = np.arange(500, 8000, 500)
        
        for t in threshold:
            pred = []
            for img in x:
                pred.append(detect_blur_1(img, t))

            rst = np.array(pred) == np.array(y)
            id_true = np.where(rst == True)
            id_false = np.where(rst == False)
            acc = rst.sum() / len(x)
            imgs_name = np.array(imgs_name)
            print("Threshold: {}, Acc: {:.3f}".format(t, acc))
            # print(imgs_name[id_false])
            accs.append(acc)
        
        plt.plot(threshold, accs)
        plt.xlabel('Threshold')
        plt.ylabel('Acc')
        plt.title("Accuracy blur detect")
        plt.show()

    else:
        n = len(x)
        for img in x:
            pred.append(detect_blur_1(img, threshold))

        rst = np.array(pred) == np.array(y)
        id_true = np.where(rst == True)
        id_false = np.where(rst == False)
        acc = rst.sum() / n
        imgs_name = np.array(imgs_name)
        print("Threshold: {}, Acc: {:.3f}".format(threshold, acc))
        
        
        # View image
        name_true = imgs_name[id_true]
        name_false = imgs_name[id_false]
        tmp = np.array(x)
        imgs_true = tmp[id_true]
        imgs_false = tmp[id_false]
        
        # print('view true')
        # for name, img in zip(name_true, imgs_true):
        #     plt.figure()
        #     plt.imshow(img)
        #     plt.title(name)
        #     plt.show()

        print('view false')
        for name, img in zip(name_false, imgs_false):
            plt.figure()
            plt.imshow(img[:,:,:,-1])
            plt.title(name)
            plt.show()


        return acc

def evaluate_backlit(path_backlit_imgs, path_normal_imgs, threshold=0.5, is_plot=False):
    print("Evaluating...")
    x, y, pred, imgs_name = [], [], [], []
    backlit_imgs = os.listdir(path_backlit_imgs)
    normal_imgs = os.listdir(path_normal_imgs)

    backlit_imgs.sort()
    normal_imgs.sort()

    for filename in backlit_imgs:
        if filename.split('.')[1] == 'png':
            img = cv2.imread(os.path.join(path_backlit_imgs, filename))
            x.append(img)
            y.append(1)
            imgs_name.append(filename)

    for filename in normal_imgs:
        if filename.split('.')[1] == 'png':
            img = cv2.imread(os.path.join(path_normal_imgs, filename))
            x.append(img)
            y.append(0)
            imgs_name.append(filename)

    if is_plot:
        accs = []
        threshold = np.arange(0, 1, 0.1)
        for t in threshold:
            pred = []
            

            for img in x:
                pred.append(backlit_detect(img, t)[0])

            rst = np.array(pred) == np.array(y)
            # print(type(rst))
            # rst = np.array(rst)
            # return
            id_true = np.where(rst == True)
            id_false = np.where(rst == False)
            acc = rst.sum() / len(x)
            imgs_name = np.array(imgs_name)
            print("Threshold: {:.4f}, Acc: {:.4f}".format(t, acc))
            # print(imgs_name[id_false])
            accs.append(acc)
        
        
        
        plt.plot(threshold, accs)
        plt.xlabel('Threshold')
        plt.ylabel('Acc')
        plt.title("Accuracy backlight detection")
        plt.show()


    else:
        for img in x:
            pred.append(backlit_detect(img, threshold)[0])

        rst = np.array(pred) == np.array(y)
        # print(type(rst))
        # rst = np.array(rst)
        # return
        id_true = np.where(rst == True)
        id_false = np.where(rst == False)
        acc = rst.sum() / len(x)
        imgs_name = np.array(imgs_name)
        print("Threshold: {:.4f}, Acc: {:.4f}".format(threshold, acc))
        # print(imgs_name[id_false])
        return acc
    

def evaluate_contrast(path_backlit_imgs, path_normal_imgs, threshold=0.8, is_plot=False):
    print("Evaluating...")
    x, y, pred, imgs_name = [], [], [], []
    backlit_imgs = os.listdir(path_backlit_imgs)
    normal_imgs = os.listdir(path_normal_imgs)

    backlit_imgs.sort()
    normal_imgs.sort()

    for filename in backlit_imgs:
        if filename.split('.')[1] == 'png':
            img = cv2.imread(os.path.join(path_backlit_imgs, filename))
            x.append(img)
            y.append(1)
            imgs_name.append(filename)

    for filename in normal_imgs:
        if filename.split('.')[1] == 'png':
            img = cv2.imread(os.path.join(path_normal_imgs, filename))
            x.append(img)
            y.append(0)
            imgs_name.append(filename)

    if is_plot:
        accs = []
        threshold = np.arange(0, 1.1, 0.1)
        
        for t in threshold:
            pred = []
            for img in x:
                pred.append(detect_low_contrast(img, t))

            rst = np.array(pred) == np.array(y)
            id_true = np.where(rst == True)
            id_false = np.where(rst == False)
            acc = rst.sum() / len(x)
            imgs_name = np.array(imgs_name)
            print("Threshold: {}, Acc: {:.3f}".format(t, acc))
            print(imgs_name[id_true])
            accs.append(acc)
        
        plt.plot(threshold, accs)
        plt.xlabel('Threshold')
        plt.ylabel('Acc')
        plt.title("Accuracy blur detect")
        plt.show()
    else:

        n = len(x)
        for img in x:
            pred.append(detect_low_contrast(img, threshold))

        rst = np.array(pred) == np.array(y)
        id_true = np.where(rst == True)
        id_false = np.where(rst == False)
        acc = rst.sum() / n
        imgs_name = np.array(imgs_name)
        print("Threshold: {:.4f}, Acc: {:.4f}".format(threshold, acc))
        print(imgs_name[id_false])

        # View image
        name_true = imgs_name[id_true]
        name_false = imgs_name[id_false]
        tmp = np.array(x)
        imgs_true = tmp[id_true]
        imgs_false = tmp[id_false]
        
        # print('view true')
        # for name, img in zip(name_true, imgs_true):
        #     fig, ax = plt.subplots(1, 2, figsize=(16, 8))

        #     ax[0].set_title(name)
        #     ax[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        #     histr = cv2.calcHist([img],[0],None,[256],[0,256])
        #     # ax[1].hist(img.ravel(), bins=256, range=(0.0, 1.0), fc='k', ec='k')
        #     # ax[1].plot(histr)
        #     ax[1].hist(img.ravel(),256,[0,256])
            
        #     # plt.figure()
        #     # plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        #     # plt.title(name)
        #     plt.show()

        print('view false')
        for name, img in zip(name_false, imgs_false):
            fig, ax = plt.subplots(1, 2, figsize=(16, 8))

            ax[0].set_title(name)
            ax[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

            histr = cv2.calcHist([img],[0],None,[256],[0,256])
            # ax[1].hist(img.ravel(), bins=256, range=(0.0, 1.0), fc='k', ec='k')
            # ax[1].plot(histr)
            ax[1].hist(img.ravel(),256,[0,256])
            
            # plt.figure()
            # plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            # plt.title(name)
            plt.show()

        return acc



def detect_low_contrast(img, threshold=0.8):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # print(is_low_contrast(gray, threshold))
    # print(low_contrast(gray, threshold))
    if(is_low_contrast(gray, threshold)):
        # print('low contrast image')
        return True
    else:
        # print('high contrast image')  
        return False



if __name__ == "__main__":
    # plot_dataset("img_evaluate/Dataset/Dataset_Blur", " ", "Dataset_Blur")
    # evaluate_each_layout("img_evaluate/Dataset_Layout", 'symmetry')
    # evaluate_backlit("img_evaluate/Dataset/Dataset_Backlit/Backlit", 
    #     "img_evaluate/Dataset/Dataset_Backlit/Normal_light", 0.5, True)
    evaluate_contrast("img_evaluate/Dataset/Dataset_Contrast_New/low",
        "img_evaluate/Dataset/Dataset_Contrast_New/high", 0.65, False)
    # evaluate_blur( "img_evaluate/Dataset/Dataset_Blur/Blur/", 
    # "img_evaluate/Dataset/Dataset_Blur/NotBlur/", 5000, False)
    print("Done!")


    ### Backlit ####
    # rst = []
    # threshold = np.arange(0, 1.5, 0.05)
    # for t in threshold:
    #     rst.append(evaluate_backlit( "img_evaluate/Dataset/Dataset_Blur/Blur/", "img_evaluate/Dataset/Dataset_Blur/NotBlur/", t))

    # plt.plot(threshold, rst)
    # plt.xlabel('Threshold')
    # plt.ylabel('Acc')
    # plt.show()

    # for t, acc in zip(threshold, rst):
    #     print("Threshold: {}, Acc: {:.3f}".format(t, acc))


    #### Contrast ####
    # rst = []
    # threshold = np.arange(0, 1, 0.1)
    # for t in threshold:
    #     rst.append(evaluate_contrast("img_evaluate/Dataset/Dataset_Contrast/low_contrast", "img_evaluate/Dataset/Dataset_Contrast/normal", t))


    # plt.plot(threshold, rst)
    # plt.xlabel('Threshold')
    # plt.ylabel('Acc')
    # plt.show()

    # for t, acc in zip(threshold, rst):
    #     print("Threshold: {}, Acc: {:.3f}".format(t, acc))


    #### Blur ####
    # rst = []
    # rst = [0.521, 0.637, 0.678, 0.695, 0.702, 0.719, 0.743, 0.757, 0.767, 0.795, 0.795, 0.798, 0.805, 0.808, 0.812, 0.815]
    # threshold = [500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500, 6000, 6500, 7000, 7500, 8000]
    # for t in threshold:
    #     threshold.append(t)
    #     rst.append(evaluate( "img_evaluate/Dataset/Dataset_Blur/Blur/", "img_evaluate/Dataset/Dataset_Blur/NotBlur/", t))

    # plt.plot(threshold, rst)
    # plt.imshow(im_output)
    # plt.xlabel('Threshold')
    # plt.ylabel('Acc')
    # plt.show()