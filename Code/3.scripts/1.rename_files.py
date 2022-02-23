import os

path = 'D:/HISTORIFY/Dataset/22.11.2021/1.training_set/content/'
files = os.listdir(path)
for i,file_name in enumerate(files):
#    os.rename(path+file_name,path+'image_'+str(i)+'.jpg')