import glob

import pickle

import gzip

import json

import numpy as np

import hashlib

feat_index = json.load(open('feat_index.json'))
for name in sorted(glob.glob('html_vec/*'))[:2]:
  img = pickle.load(open(f'mini_image/{name.split("/").pop()}', 'rb'))
  text = pickle.load(open(name,'rb'))
  print(text)

  Xs1, Xs2, ys = [],[],[]
  for i in range(len(text) - 30):
    x1 = [ [0.0]*len(feat_index) for _ in range(30) ]
    for cur,index in enumerate(text[i:i+30]):
      x1[cur][index] = 1.0 
    ybase = [0.0]*len(feat_index)
    ybase[ text[i+30] ] = 1.0
    y = ybase

    Xs1.append(x1)
    Xs2.append(img)
    ys.append(y)  

    # blobのサイズを100にして、行う
    if len(Xs1) == 100: 
      Xs1 = np.array(Xs1)
      Xs2 = np.array(Xs2)
      ys = np.array(ys)
      
      blob = gzip.compress( pickle.dumps( (Xs1, Xs2, ys) ) )
      ha   = hashlib.sha256(blob).hexdigest()
      open(f'dataset/{ha}', 'wb').write( blob )
      Xs1, Xs2, ys = [],[],[]
 
  # 残った微妙なやつを回収する
  Xs1 = np.array(Xs1)
  Xs2 = np.array(Xs2)
  ys = np.array(ys)
  blob = gzip.compress( pickle.dumps( (Xs1, Xs2, ys) ) )
  ha   = hashlib.sha256(blob).hexdigest()
  open(f'dataset/{ha}', 'wb').write( blob )
  
