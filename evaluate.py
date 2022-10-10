from assess import *
from PIL import Image


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


def plot_dataset(path_folders):
    # creating the dataset
    data = {}
    list_folder = os.listdir(path_folders)
    for folder in list_folder:
        data[folder] = len(os.listdir(os.path.join(path_folders, folder, 'img')))

    print(data)
    courses = list(data.keys())
    values = list(data.values())
    
    fig = plt.figure(figsize = (10, 5))
    
    # creating the bar plot
    plt.bar(courses, values, width = 0.4)
    
    plt.xlabel("Layout")
    plt.ylabel("No. of images")
    # plt.title()
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




if __name__ == "__main__":
    # plot_dataset("img_evaluate/Eval_Layout")
    evaluate_each_layout("img_evaluate/Eval_Layout", 'symmetry')
    print("Done!")