from scipy import signal
from PIL import Image
import numpy as np


img = Image.open("data/dataset/1_Lato-Black.72.jpg")
img2 = Image.open("digits/37.jpg")

img = np.array(255 - np.asarray(img))
img2 = np.asarray(img2)
img2.resize((28, 28))

corr = signal.correlate(img, img2)
#print(corr)
print(np.mean(corr))