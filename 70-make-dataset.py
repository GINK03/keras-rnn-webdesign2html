import glob

import pickle
import gzip

import json

import numpy as np
feat_index = json.load(open('feat_index.json'))
for name in sorted(glob.glob('html_vec/*')):
  ha = name.split('/').pop()

  img = pickle.load(open(f'mini_image/{ha}', 'rb'))
  text = pickle.load(open(name,'rb'))
  print(text)

  Xs1, Xs2, ys = [],[],[]
  for i in range(len(text) - 20):
    x1 = [ [0.0]*len(feat_index) for _ in range(20) ]
    for cur,index in enumerate(text[i:i+20]):
      x1[cur][index] = 1.0 
    ybase = [0.0]*len(feat_index)
    ybase[ text[i+20] ] = 1.0
    y = ybase

    Xs1.append(x1)
    Xs2.append(img)
    ys.append(y)  
  Xs1 = np.array(Xs1)
  np.save(f'dataset/{ha}-xs1.npy', Xs1)
  del Xs1
  Xs2 = np.array(Xs2)
  np.save(f'dataset/{ha}-xs2.npy', Xs2)
  del Xs2
  ys = np.array(ys)
  np.save(f'dataset/{ha}-ys.npy', ys)

