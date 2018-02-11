import glob

import pickle

import json

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
    y = text[i+20]

    Xs1.append(x1)
    Xs2.append(img)
    ys.append(y)
  open(f'dataset/{ha}', 'wb').write( pickle.dumps( (Xs1, Xs2, ys) ) )
