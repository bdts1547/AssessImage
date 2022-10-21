from PIL import Image, ImageEnhance
import matplotlib.pyplot as plt
import cv2
import os





def create_imgs_low_high_contrast(path_folder):
        path_folder = 'img_evaluate/Dataset/Dataset_Contrast/normal/'
        list_name_imgs = os.listdir(path_folder)

        print(len(list_name_imgs))
    
    # for i, fn in enumerate(list_name_imgs):
        
    #     print(fn)
    #     im = Image.open('img_evaluate/Dataset/Dataset_Contrast/normal/{}'.format(fn))
    #     #image brightness enhancer
    #     enhancer = ImageEnhance.Contrast(im)

    #     # factor = 1 #gives original image
    #     # im_output = enhancer.enhance(factor)
    #     # im_output.save('original-image.png')

    #     factor = 0.5 #decrease constrast
    #     im_output = enhancer.enhance(factor)
    #     im_output.save('1.png')

        
    #     factor = 1.5 #increase contrast
    #     im_output = enhancer.enhance(factor)
    #     im_output.save('img_valuate/Dataset/Dataset_Contrast_New/high/{}'.format(fn))

def create_imgs_blur_sharp(path_folder):
    
    # path_folder = 'normal/'
    list_name_imgs = os.listdir(path_folder)
    list_name_imgs.sort()
    print(len(list_name_imgs))

    for i, fn in enumerate(list_name_imgs):
        
        print(fn)
        im = Image.open(path_folder + fn)
        #image brightness enhancer
        enhancer = ImageEnhance.Sharpness(im)

        factor = 0.05
        im_s_1 = enhancer.enhance(factor)
        im_s_1.save('img_valuate/Dataset/Dataset_Blur/blur005/blur_{}.png'.format(i))
        factor = 2
        im_s_1 = enhancer.enhance(factor)
        im_s_1.save('img_valuate/Dataset/Dataset_Blur/sharp2/sharp_{}.png'.format(i))

        break

if __name__ == "__main__":
    # create_imgs_blur_sharp('img_valuate/Dataset/Dataset_Blur/NotBlur/')
    # os.system("conda activate py27 & python symmary.py")
    # print('Done')

    scores_sym = {}
    with open('score_symmetry.csv', 'r') as f:
        lines = f.read().splitlines()
        
    for line in lines:
        d = list(map(str, line.split(',')))
        scores_sym[d[0]] = list(map(float, d[1:]))
        print(d[0], max(scores_sym[d[0]]) > 0.6)
    print(scores_sym)
    
