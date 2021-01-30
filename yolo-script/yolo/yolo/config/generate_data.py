import pandas as pd
import numpy as np

#df = pd.read_csv("ssd_train.csv")
df = pd.read_csv("ssd_test.csv")

#train_txt = open("train_label.txt","w")
num = 68
train_list = []
val_list = []

with open("test_label.txt","w") as the_file:

    for i in range(df.shape[0]):

        im_name = df.loc[i][0]
        the_file.write("./labels/"+im_name[:-4]+".txt\n")
    the_file.close()
    '''
    img_list = []
    for j in range(1,69,4):
        if(pd.isnull(df.loc[i][j])):
            break
        img_list.append([df.loc[i][j],df.loc[i][j+1],df.loc[i][j+2],df.loc[i][j+3]])
    
    with open("./label/"+im_name[:-4]+".txt","w") as the_file:   
        for ob in img_list:
            x1 = ob[0]/1280
            y1 = ob[1]/720
            x2 = ob[2]/1280
            y2 = ob[3]/720
            
            the_file.write("0 "+str(x1)+" "+str(y1)+" "+str(x2)+" "+str(y2)+"\n")
        the_file.close()     '''
