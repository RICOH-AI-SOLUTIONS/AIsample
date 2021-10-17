# 手書き数字のデータセット取得と手書きデータの表示
from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt
import math
import numpy as np

# MNISTデータを取得してnumpyの配列型に変換
mnist_x, mnist_y = fetch_openml('mnist_784', version=1, data_home="sklearn_MNIST_data", return_X_y=True)
list_mnist_x = np.array(mnist_x)
list_mnist_y = np.array(mnist_y)

# 全部のデータを表示すると時間がかかるので
# 最初の4データのみ表示
digits = list_mnist_x[0:4]

# 手書き数字(8×8ピクセル)を表示
plt.set_cmap("binary")
fig = plt.figure()
row_and_col = math.ceil(math.sqrt(len(digits)))

# 1データずつ取得して手書き数字データを表示
for i,item in enumerate(digits):
    ax = fig.add_subplot(row_and_col,row_and_col,i + 1) # 表示位置設定
    ax.imshow(digits[i].reshape(28,28)) # 手書き数字データを表示

plt.show()

