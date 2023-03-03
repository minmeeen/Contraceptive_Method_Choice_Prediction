import numpy as np
import pandas as pd
def preprocess(df):

    # print("Unique values before data preprocessing")
    # for i in df.columns:
    #     print('Attribute:', i,'\t' ,df[i].unique())      #print only unique value. 

    #Normalize
    new_min = -1
    new_max = 1

    a0 = []
    a1 = []
    a2 = []
    a3 = []
    a4 = []
    a5 = []
    a6 = []
    a7 = []
    a8 = []

    #Binary 
    #เปลี่ยน 0 เป็น 2 ใน attribute ที่กำหนด
    df.loc[df["wife_religion"]==0,"wife_religion"]=-1    
    df.loc[df["wife_working"]==0,"wife_working"]=-1
    df.loc[df["media_exposure"]==0,"media_exposure"]=-1

    #class attribute
    #change for easy to use with np_utils.to_categorical
    df.loc[df["contraceptive_method"]==1,"contraceptive_method"]=0
    df.loc[df["contraceptive_method"]==2,"contraceptive_method"]=1
    df.loc[df["contraceptive_method"]==3,"contraceptive_method"]=2


    #numeric column
    for i in range(len(df)):    #loop เพื่อnormalize ข้อมูลทุกตัวที่ attribute = wife_age
        x = (((df.iloc[i][0] - df.min()[0])/(df.max()[0] - df.min()[0]))*(new_max-new_min))+new_min
        # x = (df.iloc[i][0]-df.mean()[0])/df.std()[0]
        a0.append(x)

    df['wife_age'] = a0
    a0 =[]

    for i in range(len(df)):    #loop เพื่อnormalize ข้อมูลทุกตัวที่ attribute = wife_education
        x = (((df.iloc[i][1] - df.min()[1])/(df.max()[1] - df.min()[1]))*(new_max-new_min))+new_min
        # x = (df.iloc[i][1]-df.mean()[1])/df.std()[1]
        a1.append(x)

    df['wife_education'] = a1

    for i in range(len(df)):    #loop เพื่อnormalize ข้อมูลทุกตัวที่ attribute = husband_education
        x = (((df.iloc[i][2] - df.min()[2])/(df.max()[2] - df.min()[2]))*(new_max-new_min))+new_min
        # x = (df.iloc[i][2]-df.mean()[2])/df.std()[2]
        a2.append(x)

    df['husband_education'] = a2

    temp = []
    

    for i in range(len(df)):    #loop เพื่อnormalize ข้อมูลทุกตัวที่ attribute = number_of_children
        x = (((df.iloc[i][3] - df.min()[3])/(df.max()[3] - df.min()[3]))*(new_max-new_min))+new_min
        # x = (df.iloc[i][3]-df.mean()[3])/df.std()[3]
        a3.append(x)

    df['number_of_children'] = a3

    a3=[]
   

    
    # for i in range(len(df)):    #loop เพื่อnormalize ข้อมูลทุกตัวที่ attribute = wife_religion
    #     x = (((df.iloc[i][4] - df.min()[4])/(df.max()[4] - df.min()[4]))*(new_max-new_min))+new_min
    #     # x = (df.iloc[i][4]-df.mean()[4])/df.std()[4]
    #     a4.append(x)

    # df['wife_religion'] = a4

    

    # for i in range(len(df)):    #loop เพื่อnormalize ข้อมูลทุกตัวที่ attribute = wife_working
    #     x = (((df.iloc[i][5] - df.min()[5])/(df.max()[5] - df.min()[5]))*(new_max-new_min))+new_min
    #     # x = (df.iloc[i][5]-df.mean()[5])/df.std()[5]
    #     a5.append(x)

    # df['wife_working'] = a5

    for i in range(len(df)):    #loop เพื่อnormalize ข้อมูลทุกตัวที่ attribute = husband_occupation
        x = (((df.iloc[i][6] - df.min()[6])/(df.max()[6] - df.min()[6]))*(new_max-new_min))+new_min
        # x = (df.iloc[i][6]-df.mean()[6])/df.std()[6]
        a6.append(x)

    df['husband_occupation'] = a6

    for i in range(len(df)):    #loop เพื่อnormalize ข้อมูลทุกตัวที่ attribute = standard_of_living_index
        x = (((df.iloc[i][7] - df.min()[7])/(df.max()[7] - df.min()[7]))*(new_max-new_min))+new_min
        # x = (df.iloc[i][7]-df.mean()[7])/df.std()[7]
        a7.append(x)

    df['standard_of_living_index'] = a7

    # for i in range(len(df)):    #loop เพื่อnormalize ข้อมูลทุกตัวที่ attribute = media_exposure
    #     x = (((df.iloc[i][8] - df.min()[8])/(df.max()[8] - df.min()[8]))*(new_max-new_min))+new_min
    #     # x = (df.iloc[i][8]-df.mean()[8])/df.std()[8]
    #     a8.append(x)

    # df['media_exposure'] = a8


    df.to_csv('data/preprocess_cmc.csv', index=False, header=False)     #write preprocessing data in csv

    return df  #return proprocessing data


def preprocessing_raisin(df):


    df.loc[df["Class"]=='Kecimen',"Class"]=0
    df.loc[df["Class"]=='Besni',"Class"]=1
    
    return df