from matplotlib import pyplot as plt
import numpy as np
import random
import re

results_str = ""
path  = "D:/HISTORIFY/Models_Trials/rapor/"

def listToString(s): 
    str1 = ""     
    for ele in s: 
        str1 += ele    
    return str1 

with open(path + "1000.txt", "r") as txt:
  results_str = txt.read()
with open(path + "2000.txt", "r") as txt:
  results_str += txt.read()
with open(path + "3000.txt", "r") as txt:
  results_str += txt.read()
with open(path + "4000.txt", "r") as txt:
  results_str += txt.read()
with open(path + "5000.txt", "r") as txt:
  results_str += txt.read()

regex_objects = ["D_R1: \d.\d{3,}",
                 " D_mix: \d.\d{3,}",
                 " D_real: \d.\d{3,}",
                 "D_rec: \d.\d{3,}",
                 "D_total: \d.\d{3,}",
                 "G_GAN_mix: \d.\d{3,}",
                 "G_GAN_rec: \d.\d{3,}",
                 "G_L1: \d.\d{3,}",
                 "G_mix: \d.\d{3,}",
                 "L1_dist: \d.\d{3,}",
                 "PatchD_mix: \d.\d{3,}",
                 "PatchD_real: \d.\d{3,}"]
regex_remove_str = ["D_R1:",
                    " D_mix:",
                    " D_real:",
                    "D_rec:",
                    "D_total:",
                    "G_GAN_mix:",
                    "G_GAN_rec:",
                    "G_L1:",
                    "G_mix:",
                    "L1_dist:",
                    "PatchD_mix:",
                    "PatchD_real:"]

for l,i in zip(regex_remove_str,regex_objects):
    reg = re.findall(i, results_str)
    if reg:
        res = re.sub(l, ' ',listToString(reg))
        res = res.split()
        plt.rcParams["figure.figsize"] = (4,4)
        plt.plot(res, c = np.random.rand(3,))
        plt.xlabel(l.replace(":",""))
        #print(reg)
        plt.savefig("D:/HISTORIFY/Models_Trials/rapor/results_"+l.replace(":","_")+".png")
        plt.show()
