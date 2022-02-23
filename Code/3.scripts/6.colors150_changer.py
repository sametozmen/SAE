import scipy.io
import cv2
mat = scipy.io.loadmat('color150.mat')
for i in range(len(mat['colors'])):
    mat['colors'][i] = [0, 0, 0]
mat['colors'][1] = [255, 255, 255] #originally it is 180 120 120
scipy.io.savemat('color150_white.mat',mat)