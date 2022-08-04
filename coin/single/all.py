'''
panduan -> find_motif -> encode -> mutation
extend_motif
tomeme
tofa
'''

from encode import encode
from train import train_and_pre

#open_file_name=['pit']
open_file_name=['shoot','pit','pie','py','ear']
#open_file_name=['test']

for i in range(len(open_file_name)):
    #encode(open_file_name[i])
    train_and_pre(mode='cba',zu_name=open_file_name[i])
