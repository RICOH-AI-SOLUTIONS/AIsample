# 数字画像認識AI
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# MNISTデータを取得してnumpyの配列型に変換
mnist_x, mnist_y = fetch_openml('mnist_784', version=1, data_home="sklearn_MNIST_data", return_X_y=True)
list_mnist_x = np.array(mnist_x)
list_mnist_y = np.array(mnist_y)

# 訓練用データとテストデータに分ける
data_train, data_test, target_train, target_test = train_test_split(list_mnist_x, list_mnist_y, random_state=43)

# 学習効率を上げるため、値の範囲を0～255から0～1となるように変換する
data_train /= 255
data_test /= 255

# ニューラルネットワークによるクラス分類を行う
# MLPClassifierクラスを使って、隠れ層１（ノード数：100）で
# ニューラルネットワークを構築する
# 今回使用するマシンのスペックの考慮や少しでも変化が分かりやすくするために
# 学習回数の最大値(max_iter)を10としている
clf = MLPClassifier(hidden_layer_sizes=(100,), verbose=True, max_iter=25, random_state=43)

# 訓練用データを使って学習させる
clf.fit(data_train, target_train)

# テストデータを使って数字画像認識を行う
predict = clf.predict(data_test)

# 正解率出力
print(clf.score(data_test, target_test))
