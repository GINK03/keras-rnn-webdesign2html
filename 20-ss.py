import time

from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.options import Options

import json
import hashlib

options = Options()
options.binary_location = '/usr/bin/google-chrome'
options.add_argument('--headless')

hrefs = json.load(fp=open('./hrefs.json'))

for href in hrefs:
  driver = webdriver.Chrome(chrome_options=options)
  driver.set_window_size(2000, 2000) 
  driver.get(href)

  ha = hashlib.sha256(bytes(href,'utf8')).hexdigest()

  time.sleep(2)  # Chromeの場合はAjaxで遷移するので、とりあえず適当に2秒待つ。

  driver.save_screenshot(f'pngs/{ha}.png')

  html = driver.page_source
  open(f'htmls/{ha}', 'w').write( html )
  driver.quit() 
