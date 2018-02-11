from keras.layers               import Input, Dense, GRU, LSTM, RepeatVector
from keras.models               import Model
from keras.layers.core          import Flatten
from keras.callbacks            import LambdaCallback 
from keras.optimizers           import SGD, RMSprop, Adam
from keras.layers.wrappers      import Bidirectional as Bi
from keras.layers.wrappers      import TimeDistributed as TD
from keras.layers               import merge
from keras.applications.vgg16   import VGG16 
from keras.layers.normalization import BatchNormalization as BN
from keras.layers.noise         import GaussianNoise as GN
from keras.layers.merge         import Concatenate
from keras.layers.core          import Dropout
import numpy as np
import random
import sys
import pickle
import glob
import copy
import os
import re

input_tensor1 = Input(shape=(150, 150, 3))
vgg_model     = VGG16(include_top=False, weights='imagenet', input_tensor=input_tensor1)
vgg_x         = vgg_model.layers[-1].output
vgg_x         = Flatten()(vgg_x)
vgg_x         = Dense(1800, activation='relu')(vgg_x)
vgg_x         = RepeatVector(20)(vgg_x)

input_tensor2 = Input(shape=(20, 1800))

encoded = Concatenate(axis=1)( [vgg_x, input_tensor2] )

x           = Bi(LSTM(500, recurrent_dropout=0.05, recurrent_activation='tanh', return_sequences=False))(encoded)
x           = Dropout(0.10)(x)
x           = Dense(2600, activation='relu')(x)
x           = Dropout(0.10)(x)
x           = Dense(2600, activation='relu')(x)
x           = Dropout(0.10)(x)
decoded     = Dense(1800, activation='softmax')(x)

model       = Model([input_tensor1, input_tensor2], decoded)
model.compile(optimizer=Adam(), loss='categorical_crossentropy')

"""
0 <keras.engine.topology.InputLayer object at 0x7f9ecfcea4a8>
1 <keras.layers.convolutional.Conv2D object at 0x7f9ece6220f0>
2 <keras.layers.convolutional.Conv2D object at 0x7f9e8deb02e8>
3 <keras.layers.pooling.MaxPooling2D object at 0x7f9e8de4ee10>
4 <keras.layers.convolutional.Conv2D object at 0x7f9e8de58550>
5 <keras.layers.convolutional.Conv2D object at 0x7f9e8de62e10>
6 <keras.layers.pooling.MaxPooling2D object at 0x7f9e8de6bf60>
7 <keras.layers.convolutional.Conv2D object at 0x7f9e8ddfe5c0>
8 <keras.layers.convolutional.Conv2D object at 0x7f9e8de06c50>
9 <keras.layers.convolutional.Conv2D object at 0x7f9e8de0dfd0>
10 <keras.layers.pooling.MaxPooling2D object at 0x7f9e8de20cc0>
11 <keras.layers.convolutional.Conv2D object at 0x7f9e8de29f98>
12 <keras.layers.convolutional.Conv2D object at 0x7f9e8ddbb5f8>
13 <keras.layers.convolutional.Conv2D object at 0x7f9e8ddc3eb8>
14 <keras.layers.pooling.MaxPooling2D object at 0x7f9e8ddd6d30>
15 <keras.layers.convolutional.Conv2D object at 0x7f9e8ddde630>
16 <keras.layers.convolutional.Conv2D object at 0x7f9e8dde6ef0>
17 <keras.layers.convolutional.Conv2D object at 0x7f9e8ddef588>
18 <keras.layers.pooling.MaxPooling2D object at 0x7f9e8dd81f60>
19 <keras.layers.core.Dense object at 0x7f9e8dd94a90>
20 <keras.layers.core.Flatten object at 0x7f9e8dd9c908>
21 <keras.layers.core.Dense object at 0x7f9e8dd9c6d8>
22 <keras.layers.core.RepeatVector object at 0x7f9e8dcf3978>
23 <keras.layers.wrappers.Bidirectional object at 0x7f9e8dcfd9b0>
24 <keras.layers.wrappers.TimeDistributed object at 0x7f9e8dba6ac8>
"""

for layer in model.layers[:18]:
  layer.trainable = False
  ...

buff = None
def callbacks(epoch, logs):
  global buff
  buff = copy.copy(logs)
  print("epoch" ,epoch)
  print("logs", logs)

def train():
  has = set()
  for name in glob.glob('dataset/*'):
    ha = name.split('/').pop().split('-').pop(0)
    has.add(ha)
  has = [sorted(list(has)).pop(0)]
  print(has)
  for i in range(1000):
    for name in sorted(has):
      Xs1 = np.load(f'dataset/{name}-xs1.npy')
      Xs2 = np.load(f'dataset/{name}-xs2.npy')
      ys = np.load(f'dataset/{name}-ys.npy')

      optims = [Adam(), SGD()]
      print(optims)

      print_callback = LambdaCallback(on_epoch_end=callbacks)
      batch_size = random.randint( 64, 98 )
      random_optim = random.choice( optims )
      print( random_optim )
      model.optimizer = random_optim
      model.fit( [Xs2, Xs1], ys,  shuffle=True, batch_size=batch_size, epochs=5, callbacks=[print_callback] )
      model.save("models/%9f_%09d.h5"%(buff['loss'], i))
      print("saved ..")

def predict():
  c_i = pickle.loads( open("dataset/c_i.pkl", "rb").read() )
  i_c = { i:c for c, i in c_i.items() }
  xss = []
  heads = []
  with open("dataset/wakati.distinct.txt", "r") as f:
    lines = [line for line in f]
    for fi, line in enumerate(lines):
      print("now iter ", fi)
      if fi >= 1000: 
        break
      line = line.strip()
      try:
        head, tail = line.split("___SP___")
      except ValueError as e:
        print(e)
        continue
      heads.append( head ) 
      xs = [ [0.]*DIM for _ in range(50) ]
      for i, c in enumerate(head): 
        xs[i][c_i[c]] = 1.
      xss.append( np.array( list(reversed(xs)) ) )
    
  Xs = np.array( xss[:128] )
  model = sorted( glob.glob("models/*.h5") ).pop(0)
  print("loaded model is ", model)
  autoencoder.load_weights(model)

  Ys = autoencoder.predict( Xs ).tolist()
  for head, y in zip(heads, Ys):
    terms = []
    for v in y:
      term = max( [(s, i_c[i]) for i,s in enumerate(v)] , key=lambda x:x[0])[1]
      terms.append( term )
    tail = re.sub(r"」.*?$", "」", "".join( terms ) )
    print( head, "___SP___", tail )
if __name__ == '__main__':
  if '--test' in sys.argv:
    test()

  if '--train' in sys.argv:
    train()

  if '--predict' in sys.argv:
    predict()
