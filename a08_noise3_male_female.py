# 실습, 과제
# keras61_5 남자 여자 데이터에 노이즈를 내서
# 기미 주근깨 여드름을 제거하시오

import numpy as np
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
import random
from sklearn.model_selection import train_test_split

x = np.load('./_save/_npy/k59_5_x.npy')
x_predict = np.load('./_save/_npy/k59_5_x_predict.npy')
y = np.load('./_save/_npy/k59_5_y.npy')
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, shuffle=True, random_state=10)
# 1. 데이터
# x_train = x_train.reshape(60000, 784).astype('float')/255
# x_test = x_test.reshape(10000, 784).astype('float')/255

x_train = x_train.reshape(x_train.shape[0], 80 * 80 * 3).astype('float')/255
x_test = x_test.reshape(x_test.shape[0], 80 * 80 * 3).astype('float')/255

x_train_noised = x_train + np.random.normal(0, 0, size=x_train.shape) # 픽셀에 0에서 0.1의 난수를 더해준다.
x_test_noised = x_test + np.random.normal(0, 0, size=x_test.shape) # 픽셀에 0에서 0.1의 난수를 더해준다.
# 최대값이 1.1로 바뀌었기 때문에 다시 1로 바꿔줘야 함.

x_train_noised = np.clip(x_train_noised, a_min=0, a_max=1)
x_test_noised = np.clip(x_test_noised, a_min=0, a_max=1)

# 2. 모델
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Input, Conv2D, MaxPooling2D, MaxPool2D, UpSampling2D, Flatten

def autoencoder(hidden_layer_size):         # 기본적인 오토인코더
    model = Sequential()
    model.add(Dense(units=hidden_layer_size, input_shape=(19200,),
                    activation='relu'))
    model.add(Dense(19200, activation='sigmoid'))
    return model

model = autoencoder(hidden_layer_size=1024)   # PCA 95% layer를 팍 줄이니까 성능이 떨어진다.

model.compile(optimizer='adam', loss='mse')

model.fit(x_train_noised, x_train, epochs=10)

output = model.predict(x_test_noised)
output = output.reshape(output.shape[0], 80, 80, 3) * 255
print(output)

import matplotlib.pyplot as plt

fig,    ((ax1, ax2, ax3, ax4, ax5), (ax11, ax12, ax13, ax14, ax15),
        (ax6, ax7, ax8, ax9, ax10)) = \
        plt.subplots(3, 5, figsize=(20, 7))

# 이미지 5개를 무작위로 고른다.
random_images = random.sample(range(output.shape[0]), 5)

# 원본 이미지를 맨 위에 그린다.
for i, ax in enumerate([ax1, ax2, ax3, ax4, ax5]):
    ax.imshow(x_test[random_images[i]].reshape(80, 80,3) * 255)
    if i == 0:
        ax.set_ylabel("INPUT", size=20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

# 잡음을 넣은 이미지
for i, ax in enumerate([ax11, ax12, ax13, ax14, ax15]):
    ax.imshow(x_test_noised[random_images[i]].reshape(80, 80, 3)* 255)
    if i == 0:
        ax.set_ylabel("NOISE", size=20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

for i, ax in enumerate([ax6, ax7, ax8, ax9, ax10]):
    ax.imshow(output[random_images[i]].reshape(80, 80, 3) * 255)
    if i == 0:
        ax.set_ylabel("OUTPUT", size=20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

plt.tight_layout()
plt.show()
# noise 발생시킨 후 CNN모델로 시도하니까 성능이 현저하게 떨어졌다. ->
