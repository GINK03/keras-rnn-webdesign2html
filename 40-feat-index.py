import glob

feat_index = {}
for name in glob.glob('sanitize/*'):
  chars = [char for char in open(name).read()]
  for char in chars:
    if feat_index.get(char) is None:
      feat_index[char] = len(feat_index)
import json
feat_index['<SOS>'] = len(feat_index)
feat_index['<EOS>'] = len(feat_index)

json.dump(feat_index, fp=open('feat_index.json', 'w'), ensure_ascii=False, indent=2)

