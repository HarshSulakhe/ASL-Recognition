import numpy as np
import csv
from PIL import Image
import os
import pandas as pd

path = '/home/harsh/Downloads/ASLMNIST/'

for mode in ['train','test']:
    counter = dict()
    csv_file = pd.read_csv(path + 'sign_mnist_{}'.format(mode)+'.csv')
    csv_file = csv_file.to_numpy()
    for row in csv_file:
        pixels = row[1:]
        pixels = pixels.reshape((28, 28))
        image = Image.fromarray(np.uint8(pixels),'L')
        # print(len(pixels))
        label = str((chr(ord('a')+row[0])).upper())

        if label not in counter:
            counter[label] = 0
            os.mkdir(path + mode + '/'+label)
        counter[label] += 1

        filename = path + mode +'/'+label+'/{}{}.jpg'.format(label, counter[label])
        image.save(filename)
        print("saved:",filename)
