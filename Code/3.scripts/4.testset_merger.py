import cv2
import numpy as np
from numpy.lib.type_check import imag

path = 'D:/HISTORIFY/Dataset/22.11.2021/'
images = [x for x in range(100)]
styles = [x for x in range(100)]
styles_out = [x for x in range(1,6)]
for i in images:
    for s in styles:
        historified_img = cv2.imread(path + '/1.training_set/dataset/dataset_raw/image_'+str(i)+'_style_'+str(s)+'.png')
        orig_mask_mask = cv2.imread(path + '/1.training_set/building_images/extracted/and/a_image_'+str(i)+'.jpg')
        not_img = cv2.imread(path + '/1.training_set/building_images/extracted/original_not/orig_not_image_'+str(i)+'.jpg')
        second_img = cv2.imread(path + '/1.training_set/building_images/extracted/and/a_image_'+str(i)+'.jpg')
        second_not_img = cv2.bitwise_not(second_img)
        masked_historified = cv2.bitwise_and(historified_img, orig_mask_mask)
        #cv2.imshow("masked",(masked_historified))
        xored = cv2.bitwise_xor(masked_historified, second_not_img)
        anded = cv2.bitwise_or(masked_historified,not_img)
        #cv2.imshow("anded",anded)
        #cv2.waitKey(0)
        cv2.imwrite(path + '/1.training_set/dataset/dataset_processed/image_'+str(i)+'_style_'+str(s)+'.png', anded)
exit()
for i in images:
    for s in styles_out:
        historified_img = cv2.imread(path + '/2.test_set/dataset/dataset_raw/image_'+str(i)+'_style_out_'+str(s)+'.png')
        orig_mask_mask = cv2.imread(path + '/2.test_set/building_images/extracted/and/a_image_'+str(i)+'.jpg')
        not_img = cv2.imread(path + '/2.test_set/building_images/extracted/original_not/orig_not_image_'+str(i)+'.jpg')
        second_img = cv2.imread(path + '/2.test_set/building_images/extracted/and/a_image_'+str(i)+'.jpg')
        second_not_img = cv2.bitwise_not(second_img)
        masked_historified = cv2.bitwise_and(historified_img, orig_mask_mask)
        #cv2.imshow("masked",(masked_historified))
        xored = cv2.bitwise_xor(masked_historified, second_not_img)
        anded = cv2.bitwise_or(masked_historified,not_img)
        #cv2.imshow("anded",anded)
        #cv2.waitKey(0)
        cv2.imwrite(path + '/2.test_set/dataset/dataset_processed/image_'+str(i)+'_style_'+str(s)+'.png', anded)