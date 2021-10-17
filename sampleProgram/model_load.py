# Pickleによるモデル保存
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pickle
from PIL import Image, ImageOps

test = np.array(ImageOps.invert(Image.open('sample.png').convert('L')))

test = test.astype('float32')
test = test / 255.0

test = test.reshape(-1, 28*28)

# 保存したモデルをロードする
loaded_model = pickle.load(open("model.sav", 'rb'))
result = loaded_model.predict(test)
print(result)
