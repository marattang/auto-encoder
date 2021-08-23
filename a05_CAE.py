# 2번 카피해서 복붙
# (CNN으로) 깊게 구성
# 2개 모델을 만드는데, 하나는 기본적 오토 인코더
# 다른 하나는 딥하게 만든 구성


import numpy as np
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
import random

(x_train, _), (x_test, _) = mnist.load_data()

# 1. 데이터
# x_train = x_train.reshape(60000, 784).astype('float')/255
# x_test = x_test.reshape(10000, 784).astype('float')/255

x_train = x_train.reshape(60000, 28, 28, 1).astype('float')/255
x_train2 = x_train.reshape(60000, 784).astype('float')/255
x_test = x_test.reshape(10000, 28, 28, 1).astype('float')/255

# 2. 모델
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Input, Conv2D, MaxPooling2D, MaxPool2D, UpSampling2D, Flatten

def autoencoder1(hidden_layer_size):         # 기본적인 오토인코더
    model = Sequential()
    model.add(Conv2D(64, input_shape=(28, 28, 1), activation='relu', kernel_size=(2,2)))
    model.add(UpSampling2D())
    model.add(Conv2D(32, kernel_size=(2,2), activation='relu'))
    model.add(UpSampling2D())
    model.add(Conv2D(12, kernel_size=(2,2), activation='relu'))
    model.add(Flatten())
    model.add(Dense(784, activation='sigmoid'))
    return model

def autoencoder2(hidden_layer_size):         # 딥~~한 오토인코더
    model = Sequential()
    model.add(Conv2D(64, input_shape=(28, 28, 1), activation='relu', kernel_size=(2,2)))
    model.add(MaxPool2D())
    model.add(Conv2D(32, kernel_size=(2,2), activation='relu'))
    model.add(MaxPool2D())
    model.add(Conv2D(12, kernel_size=(2,2), activation='relu'))
    model.add(Flatten())
    model.add(Dense(784, activation='sigmoid'))
    return model

model1 = autoencoder1(hidden_layer_size=1024)   # PCA 95% layer를 팍 줄이니까 성능이 떨어진다.
model2 = autoencoder2(hidden_layer_size=256)   # PCA 95% layer를 팍 줄이니까 성능이 떨어진다.

model1.compile(optimizer='adam', loss='mse')
model2.compile(optimizer='adam', loss='mse')


model2.summary()
print('shape : ',x_train.shape)

model1.fit(x_train, x_train2, epochs=10)
model2.fit(x_train, x_train2, epochs=10)

output1 = model1.predict(x_test)
output2 = model2.predict(x_test)

import matplotlib.pyplot as plt

fig, ((ax1, ax2, ax3, ax4, ax5), (ax6, ax7, ax8, ax9, ax10), (ax11, ax12, ax13, ax14, ax15)) = \
        plt.subplots(3, 5, figsize=(20, 7))

# 이미지 5개를 무작위로 고른다.
random_images = random.sample(range(output1.shape[0]), 5)

# 원본 이미지를 맨 위에 그린다.
for i, ax in enumerate([ax1, ax2, ax3, ax4, ax5]):
    ax.imshow(x_test[random_images[i]].reshape(28, 28,), cmap='gray')
    if i == 0:
        ax.set_ylabel("INPUT", size=20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

for i, ax in enumerate([ax6, ax7, ax8, ax9, ax10]):
    ax.imshow(output1[random_images[i]].reshape(28, 28,), cmap='gray')
    if i == 0:
        ax.set_ylabel("OUTPUT-1", size=20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])


for i, ax in enumerate([ax11, ax12, ax13, ax14, ax15]):
    ax.imshow(output2[random_images[i]].reshape(28, 28,), cmap='gray')
    if i == 0:
        ax.set_ylabel("OUTPUT-2", size=20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

plt.tight_layout()
plt.show()

