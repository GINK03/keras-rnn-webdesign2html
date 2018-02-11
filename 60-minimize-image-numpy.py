import glob

import pickle

from PIL import Image

import numpy as np
for name in glob.glob('./pngs/*.png'):
  ha = name.split('/').pop().replace('.png', '')
  img = Image.open(name)
  img = img.resize((150, 150))

  arr = np.array(img) - 127.0

  open(f'mini_image/{ha}', 'wb').write( pickle.dumps(arr) )

