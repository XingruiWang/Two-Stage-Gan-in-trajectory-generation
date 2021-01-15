from selenium import webdriver
import time
output_dir = 'output'
url = 'file:///home/user/path/map.html'
brower = webdriver.Chrome()
brower.get(url)
brower.maximize_window()
time.sleep(0.2)
for i in range(500):
    name = str(i)
    if len(name)<3:
        name = '0'*(3-len(name))+name
    brower.save_screenshot('output/pict%s.png'%(name))
    time.sleep(1)
brower.close()