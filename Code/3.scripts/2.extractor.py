import cv2
import numpy as np
from numpy.lib.type_check import imag

path = 'D:/HISTORIFY/Dataset/22.11.2021/'
images = [x for x in range(100)]

for i in images:
    img = cv2.imread(path + '/1.training_set/building_images/extracted/image_'+str(i)+'.jpg')
    img1 = img[0:256,0:256]
    cv2.imwrite(path + '/1.training_set/building_images/extracted/original/s_image_'+str(i)+'.jpg',img1)
    img2 = img[0:256,256:512]
    #cv2.imshow("",img2)
    cv2.imwrite(path + '/1.training_set/building_images/extracted/and/a_image_'+str(i)+'.jpg',img2)
    img3 = cv2.bitwise_not(img2)
    cv2.imwrite(path + '/1.training_set/building_images/extracted/not/n_image_'+str(i)+'.jpg',img3)
    img4 = cv2.bitwise_and(img1,img2)
    cv2.imwrite(path + '/1.training_set/building_images/extracted/original_and/orig_anded_image_'+str(i)+'.jpg',img4)
    img5 = cv2.bitwise_and(img1,img3)
    cv2.imwrite(path + '/1.training_set/building_images/extracted/original_not/orig_not_image_'+str(i)+'.jpg',img5)

    #cv2.waitKey(0)