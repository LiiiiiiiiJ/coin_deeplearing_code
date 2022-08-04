'''
@ author LiJ
@ version 1.0
@ date 2022 / 7 / 7 18: 34
@ Description:
'''
from process import select_exp
from encode import encode_dou,encode_sin
from train import train_and_pre

#open_file_name=['pit']
open_file_name=['shoot','pit','pie','py','ear']


for i in range(len(open_file_name)):
    #select_exp(open_file_name[i])
    #encode_dou(open_file_name[i])
    #encode_sin(open_file_name[i])
    train_and_pre(mode='dou',zu_name=open_file_name[i])
