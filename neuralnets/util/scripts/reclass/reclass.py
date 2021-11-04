import matplotlib.pyplot as plt
import tifffile
import sys
import torch
import numpy as np
import glob
import os
from PIL import Image

if __name__ == '__main__':
    path_in = sys.argv[1]
    path_out = sys.argv[2]
    for image in glob.glob(os.path.join(path_in, '*')):
        name = image.split('/')[-1]
        frame = np.array(Image.open(image))
        #
        # print(np.unique(frame))
        # print(frame[frame == 0].shape)
        # print(frame[frame == 1].shape)
        # print(frame[frame == 2].shape)
        # exit()

        base = np.ones_like(frame) * 0
        base[frame == 2] = 1

        out = Image.fromarray(base)
        out.save(os.path.join(path_out, name))
