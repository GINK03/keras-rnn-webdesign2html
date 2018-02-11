import glob

import bs4, lxml
from bs4 import Comment
for name in glob.glob('htmls/*'):
  ha = name.split('/').pop()
  soup = bs4.BeautifulSoup(open(name).read(), 'lxml')
  #[s.extract() for s in soup('script')]
  comments = soup.findAll(text=lambda text:isinstance(text, Comment))
  [comment.extract() for comment in comments]
  html = str(soup)

  open(f'sanitize/{ha}.html', 'w').write( html )
