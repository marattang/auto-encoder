# 오코 인코더 : 이미지 쪽에서 작업하는 것. 인코더 = 암호화
# 앞뒤가 똑같은 오~토인코더~ => 들어가는 것과 나오는게 같다. 약한 특성은 제거가 되고 강한 특성은 남는다
# GAN은 뚜렷하게 특성이 남는다. 오토 인코더는 흐릿하게 남는다. y가 필요 없다. x가 들어가서 x가 나오는 방식. 이미지 증폭은 GAN으로 한다.
# 특성이 약한걸 지우는 개념임
import numpy as np
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt

(x_train, _), (x_test, _) = mnist.load_data()

# 1. 데이터
x_train = x_train.reshape(60000, 784).astype('float')/255
x_test = x_test.reshape(10000, 784).astype('float')/255

# 2. 모델
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input

input_img = Input(shape=(784,))
encoded = Dense(1064, activation='relu')(input_img) #  해당 레이어에서 중요한 특성만 남긴다. 숫자가 높은 데이터만 남김.
decoded = Dense(784, activation='sigmoid')(encoded)
# decoded = Dense(784, activation='tanh')(encoded) # -1부터 1사이
# decoded = Dense(784, activation='relu')(encoded) # relu로 하니까 구려짐 범위가 0에서 무한대이기 때문에
# 종합적으로 봤을 때 범위가 늘어나면 흐려져서 특징을 잘 잡아내지 못함. 스케일링을 진행했기 때문에 값이 0 - 1사이의 값을 갖고 있어서
# 출력층에서 값을 제한해준 게 가장 잘 나옴
autoencoder = Model(input_img, decoded)

# autoencoder.summary()

# 3. 컴파일, 훈련
# autoencoder.compile(optimizer='adam', loss='mse')
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

autoencoder.fit(x_train, x_train, epochs=30, batch_size=128, validation_split=0.2)

# 4. 평가, 예측
decoded_img = autoencoder.predict(x_test)

n = 10
plt.figure(figsize=(20, 4))
for i in range(n):
    ax = plt.subplot(2, n, i+1)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ax = plt.subplot(2, n, i+1+n)
    plt.imshow(decoded_img[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

plt.show()