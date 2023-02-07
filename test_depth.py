import matplotlib.pyplot as plt
import numpy as np

# depth = plt.imread('/home/park/tensorflow_datasets/downloads/extracted/ZIP.diode_indoor.zip/train/depth/_1000.png')

depth = np.load('/home/park/diode_converted/validation/depth/_1.npy')
plt.imshow(depth)
plt.show()