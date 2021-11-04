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

    image = tifffile.imread(path_in)
    image = torch.tensor(image)

    for i, frame in enumerate(image.unbind(dim=0)):
        image = Image.fromarray(np.array(frame))
        image.save(os.path.join(path_out, f'data_{i:04}.png'))
