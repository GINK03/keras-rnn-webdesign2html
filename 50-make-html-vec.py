import glob

import json

import pickle

import numpy as np

feat_index = json.load(open('feat_index.json'))

for name in glob.glob('sanitize/*'):
  ha = name.split('/').pop().replace('.html', '')

  chs = [ch for ch in open(name).read()]
  chs = ['<SOS>']*20 + chs + ['<EOS>']*20 
  
  indexes = [feat_index[ch] for ch in chs]
  open(f'html_vec/{ha}', 'wb').write( pickle.dumps(indexes) )
