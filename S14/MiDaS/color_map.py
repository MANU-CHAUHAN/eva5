import cv2
'''
path = "/home/g2-test/Desktop/EVA5/Session-14/MiDas_trial_results/output/"
	 
im_gray = cv2.imread(", cv2.IMREAD_GRAYSCALE)
im_color = cv2.applyColorMap(im_gray, cv2.COLORMAP_JET)
'''
'''
import os

def load_images_from_folder(folder):
    images = []
    for bb,filename in os.listdir(folder):
        print("bb: ", bb)
        img = cv2.imread(os.path.join(folder,filename))
        im_color = cv2.applyColorMap(img, cv2.COLORMAP_JET)
        #cv2.imwrite(os.path.join(path , image{}.jpg'.format(bb)), img)
        cv2.imwrite('/home/g2-test/Desktop/EVA5/Session-14/MiDas_trial_results/color_output/image{}.jpg'.format(bb), im_color)
        #wait for 1 second
        k = cv2.waitKey(1000)
        #destroy the window
        cv2.destroyAllWindows()
        cv2.waitKey(0)
        #if img is not None:
         #   images.append(img)
    return images
folder="/home/g2-test/Desktop/EVA5/Session-14/MiDas_trial_results/output/"
#path = "/home/g2-test/Desktop/EVA5/Session-14/MiDas_trial_results/color_output/"
load_images_from_folder(folder)
'''

import glob
#select the path
path = "/home/g2-test/Desktop/EVA5/Session-14/MiDaS/output/*.*"
for bb,file in enumerate (glob.glob(path)):
    print("file: ",file)
    a= cv2.imread(file)
    print(a)
    # %%%%%%%%%%%%%%%%%%%%%
    #conversion numpy array into rgb image to show
    #c = cv2.cvtColor(a, cv2.COLOR_BGR2RGB)
    im_color = cv2.applyColorMap(a, cv2.COLORMAP_JET)
    #cv2.imshow('Color image', c)
    #writing the images in a folder output_images
    cv2.imwrite('/home/g2-test/Desktop/EVA5/Session-14/MiDaS/color_output/image{}.jpg'.format(bb), im_color) #image{}.jpg
    #cv2.imwrite('/home/g2-test/Desktop/EVA5/Session-14/MiDaS/color_output/'+'.jpg', im_color)
    #wait for 1 second
    k = cv2.waitKey(1000)
    #destroy the window
    cv2.destroyAllWindows()
