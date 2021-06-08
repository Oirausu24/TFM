from skimage.io import imsave
import os
import numpy as np
from PIL import Image
import tifffile

root='/content/gdrive/My Drive/TFM/UVAS_JULIO2020'
destinoIR='/content/gdrive/My Drive/TFM/TODO_TIFF_IR'
destinoVISIBLE='/content/gdrive/My Drive/TFM/TODO_TIFF_VISIBLE'

for folderName, subfolders, filenames in os.walk(root):

    if os.path.basename(os.path.normpath(folderName)) == 'IR':
        new_dir='IR'
        for folderName, subfolders, filenames in os.walk(folderName):
            cont=0
            x=np.array([])
            for filename in filenames:
                im_path = os.path.join(folderName, filename)
                photo = Image.open(im_path)
                photo = photo.resize((128,128))
                data = np.array(photo)
                if cont==0:
                    x=data
                else: 
                    x=np.dstack((x, data))
                cont+=1

            if x.shape != (0,):
                cooked_name="_".join(filename.split("_", 2)[:2])
                file_name=cooked_name+'_'+new_dir+'.tiff'
                new_tiff=os.path.join(destinoIR,file_name)
                tifffile.imwrite(new_tiff, x, planarconfig='contig')

    if os.path.basename(os.path.normpath(folderName)) == 'VISIBLE':
        new_dir='VISIBLE'
        for folderName, subfolders, filenames in os.walk(folderName):
            cont=0
            x=np.array([])
            for filename in filenames:
                im_path = os.path.join(folderName, filename)
                photo = Image.open(im_path)
                photo = photo.resize((128,128))
                data = np.array(photo)
                if cont==0:
                    x=data
                else: 
                    x=np.dstack((x, data))
                cont+=1

            if x.shape != (0,):
                cooked_name="_".join(filename.split("_", 2)[:2])
                file_name=cooked_name+'_'+new_dir+'.tiff'
                new_tiff=os.path.join(destinoVISIBLE,file_name)
                tifffile.imwrite(new_tiff, x, planarconfig='contig')
