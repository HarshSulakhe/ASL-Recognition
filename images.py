import numpy as np
import csv
from PIL import Image
import os
import pandas as pd
counter = dict()


path = '/home/harsh/Downloads/SLMNIST/'

# with open(path + 'sign_mnist_train.csv') as csv_file:
csv_file = pd.read_csv(path + 'sign_mnist_test.csv')
    # csv_reader = csv.reader(csv_file)
csv_file = csv_file.to_numpy()
for row in csv_file:
    pixels = row[1:]
    pixels = pixels.reshape((28, 28))
    image = Image.fromarray(np.uint8(pixels),'L')
    # print(len(pixels))
    label = str((chr(ord('a')+row[0])).upper())

    if label not in counter:
        counter[label] = 0
        os.mkdir(path+'valid/'+label)
    counter[label] += 1

    filename = path + 'valid/'+label+'/{}{}.jpg'.format(label, counter[label])
    image.save(filename)
    print("saved:",filename)
# print(csv_file[0])
