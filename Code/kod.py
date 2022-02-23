import cv2
import os

k = 0
for i in os.listdir(os.getcwd()+'/images'):
    img = cv2.imread(os.getcwd()+'/images/'+str(i))
    img = cv2.resize(img, (256, 256), interpolation = cv2.INTER_AREA)
    cv2.imwrite(os.getcwd()+'/results/'+str(k)+'.jpg',img)
    k+=1