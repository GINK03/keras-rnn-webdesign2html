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
vgg_x         = vgg_model.layers[-2].output
vgg_x         = Flatten()(vgg_x)
vgg_x         = Dense(1, activation='tanh')(vgg_x)
vgg_x         = RepeatVector(30)(vgg_x)

input_tensor2 = Input(shape=(30, 1800))

packed        = GRU(300, activation='relu', dropout=0.1, recurrent_dropout=0.1, recurrent_activation='tanh', return_sequences=True)(input_tensor2)

encoded       = Concatenate(axis=2)( [vgg_x, packed] )
print(encoded.shape)
x             = GRU(300, activation='relu', dropout=0.1, recurrent_dropout=0.1, recurrent_activation='tanh', return_sequences=False)(encoded)

x             = Dense(2600, activation='relu')(x)
decoded       = Dense(1800, activation='softmax')(x)

model         = Model([input_tensor1, input_tensor2], decoded)

for layer in model.layers[:18]:
  layer.trainable = False
  print(layer)
  ...
model.compile(optimizer=Adam(lr=0.0001, decay=0.03), loss='categorical_crossentropy')



buff = None
def callbacks(epoch, logs):
  global buff
  buff = copy.copy(logs)
  print("epoch" ,epoch)
  print("logs", logs)

def train():
  if '--resume' in sys.argv:
    model.load_weights(sorted(glob.glob('./models/*')).pop(0))
  has = set()
  for name in glob.glob('dataset/*'):
    ha = name.split('/').pop().split('-').pop(0)
    has.add(ha)
  #has = [sorted(list(has)).pop(0)]
  print(has)

  init_rate = 0.0002
  decay     = 0.02
  for i in range(50):
    Xs1, Xs2, ys = [None, None, None]
    for name in sorted(has):
      Xs1 = np.load(f'dataset/{name}-xs1.npy')
      Xs2 = np.load(f'dataset/{name}-xs2.npy')
      ys = np.load(f'dataset/{name}-ys.npy')

      print_callback = LambdaCallback(on_epoch_end=callbacks)
      batch_size = random.choice([50,100])
      lr = init_rate*( 1.0 - i*decay )
      print(f'now lr is {lr:0.12f}')
      model.optimizer = Adam(lr=lr, amsgrad=False)
      model.fit( [Xs2, Xs1], ys,  shuffle=True, batch_size=batch_size, epochs=1, callbacks=[print_callback] )
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
