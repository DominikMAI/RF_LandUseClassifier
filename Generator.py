import os
import numpy as np
import rasterio
import glob
from skimage.transform import resize
from skimage.segmentation import mark_boundaries
from skimage.segmentation import slic
import numpy.ma as ma
from skimage.morphology import disk
from scipy.stats import *
import matplotlib.pyplot as plt
import mahotas as mt
from matplotlib import colors
from skimage.filters.rank import entropy
from skimage.filters import sobel
from skimage.measure import label
from skimage.segmentation import slic,watershed
import random
from skimage.filters.rank import gradient
from skimage.color import rgb2gray

os.chdir('./data/')
def rgb2mask(img):
    color2index = {
        (255, 255, 255) : 0,
        (0,     0, 255) : 1,
        (0,   255, 255) : 2,
        (0,   255,   0) : 3,
        (255, 255,   0) : 4,
        (255,   0,   0) : 5
    }

    height, width, ch = img.shape
    W = np.power(256, [[0],[1],[2]])

    img_id = img.dot(W).squeeze(-1) 
    values = np.unique(img_id)

    mask = np.zeros(img_id.shape)

    for i, c in enumerate(values):
        try:
            mask[img_id == c] = color2index[tuple(img[img_id==c][0])] 
        except Exception as e:
            pass
        
    return mask

def Data(start, limit, mode='training'):
    Nn = 0
    Features_vector = []
    for i in glob.glob('./All_Labels/*tif')[start:limit]:
        
        if mode=='test':
            Features_vector = []
            Features = []
            
        Nn += 1
        Label = np.zeros((512,512, 3))
        Image = np.zeros((512,512, 7))

        for b in range(1,4,1):
            label = rasterio.open(os.path.join(os.getcwd(), i)).read(b)
            label = resize(label, (512, 512),anti_aliasing=False)
            Label[:,:,b-1] = label*255

        Mask = rgb2mask(Label)
        
        Mask[Mask == 4] = 0
        Mask[Mask == 5] = 0
        #Mask[Mask == 5] = 4
        for b in range(1,5):
            f_list = os.path.split(i)[-1].split('_l')[:-1] 
            img = rasterio.open(os.path.join('./data/Ortho/', f_list[0] + "_RGBIR.tif")).read(b)
            img = resize(img, (512, 512), anti_aliasing=False)
            Image[:, :, b-1] = img

        Image[:,:,5] = gradient(Image[:,:,-1].reshape((512, 512)), disk(2)) 
        Image[:,:,4] = (Image[:,:,3] - Image[:,:,0]) / (Image[:,:,3] + Image[:,:,0])
        
        #segments_slic = slic(Image[:,:,:3], n_segments=250, compactness=10, sigma=1,start_label=1)
        sobel_gradient = sobel(rgb2gray(Image[:,:,:3]))
        segments_slic = watershed(sobel_gradient, markers=500, compactness=0.001)

        if mode == 'test':
            np.save('./data/Segments_{}.npy'.format(Nn), segments_slic)
        
        textures = entropy(Image[:,:,3], disk(2))
        Image[:,:,5] = textures
        
        print(i.split('m_')[1].split('_l')[0].split('_')[1])
        if int(i.split('m_')[1].split('_l')[0].split('_')[1]) > 9:
            name = f_list[0].split('op_')[1].split('m_')[0]+'m_0'+i.split('m_')[1].split('_l')[0]
        else:
            name = f_list[0].split('op_')[1].split('m_')[0]+'m_0'\
            +i.split('m_')[1].split('_l')[0].split('_')[0]\
            +'_'+str(0)+i.split('m_')[1].split('_l')[0].split('_')[1]
            
        dsm = rasterio.open(os.path.join('./data/1_DSM_normalisation/',
                                         'dsm_'+name + "_normalized_lastools.jpg")).read(1)
        
        dsm_resized = resize(dsm, (512, 512), anti_aliasing=False)
        Image[:,:,6] = dsm_resized #/ np.max(dsm_resized)
        
        N_ = 0
        for s in np.unique(segments_slic):
            N_ += 1
            Features = []
            Features.append(s)
            
            for b in range(1, 7, 1):

                im = Image[:,:,b].reshape((512, 512))
                v = ma.masked_array(im, mask=(segments_slic != s))
                Features.append(np.mean(im[~v.mask].reshape((-1,1))))
                Features.append(np.std(im[~v.mask].reshape((-1,1))))
                Features.append(np.min(im[~v.mask].reshape((-1,1))))
                Features.append(np.max(im[~v.mask].reshape((-1,1))))
         
            mask = ma.masked_array(Mask, mask=(segments_slic != s))
            vals,counts = np.unique(Mask[~mask.mask].reshape((-1)), return_counts=True)
            segments_slic[~mask.mask] = vals[np.argmax(counts)]
            
            Features.append(vals[np.argmax(counts)])
            
            if len(np.unique(segments_slic)) > 3:
                Features_vector.append(Features)   
            else:
                continue
        
        if mode == 'test':
            np.save('./data/Features_vector_{}.npy'.format(Nn), Features_vector)
            np.save('./data/segments_slic_{}.npy'.format(Nn), segments_slic)
            np.save('./data/Image_{}.npy'.format(Nn), Image)
        
        if len(np.unique(segments_slic)) > 3:
            cmp = colors.ListedColormap(['gray', 'firebrick', 'lightgreen', 'green','yellow' ])
            print(np.unique(segments_slic))

            bounds = [0, 1, 2, 3, 4, 5]

            norm = colors.BoundaryNorm(bounds, cmp.N)
            plt.figure(figsize=(10,10))
            plt.imshow(Image[:,:,:3])
            plt.imshow(segments_slic, cmap = cmp,norm = norm, alpha=0.5)
            plt.show()
            
    return Features_vector