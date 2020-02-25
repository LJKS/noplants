from os import listdir
from matplotlib import image
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
THE_GOOD = [0,1,1] #cyan
THE_BAD = [1,0,1] #magenta
THE_UGLY = None #everything else

for img_str in listdir('lbl'):
    img_str_old = 'lbl/'+img_str
    img = image.imread(img_str_old)
    good = np.all(img == THE_GOOD, -1).astype(float)
    bad = np.all(img == THE_BAD, -1).astype(float)
    ugly = np.ones(bad.shape) - good - bad
    img_new = np.stack((good, bad, ugly), -1)*255
    img_new = img_new.astype(np.uint8)
    img_string_new = 'target_container/targets/' + img_str
    img_new = Image.fromarray(img_new).convert('RGB')
    img_new.save(img_string_new)
