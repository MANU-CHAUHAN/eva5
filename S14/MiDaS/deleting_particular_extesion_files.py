import os 
from os import listdir

folder_path = '/home/g2-test/Desktop/EVA5/Session-14/MiDaS/output/'

for file_name in listdir(folder_path):
    
    if file_name.endswith('.pfm'):
        
        os.remove(folder_path + file_name)
