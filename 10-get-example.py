import bs4

import requests


url = 'http://muuuuu.org/category/industry/hospital'

r = requests.get(url)
soup = bs4.BeautifulSoup(r.text)

hrefs = []
for a in soup.find_all('a', {'rel':'bookmark'}):
  hrefs.append(a['href'])

import json

json.dump(hrefs, fp=open('hrefs.json', 'w'), indent=2)
