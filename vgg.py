import tensorflow as tf
import numpy as np


class VGG:
    def __init__(self, input_shape, num_classes):
        self.input_shape = input_shape
        self.num_classes = num_classes

        self.model = self.build_model()

    def build_model(self):
        model = tf.keras.Sequential()

        # 添加卷积层和最大池化层
        model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=self.input_shape))
        model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
        model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2, padding='same'))

        model.add(tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
        model.add(tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
        model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2, padding='same'))

        model.add(tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same'))
        model.add(tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same'))
        model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2, padding='same'))

        model.add(tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same'))
        model.add(tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same'))
        model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2, padding='same'))

        # self attention

        # average pool
        # model.add(tf.keras.layers.AveragePooling2D(pool_size=(2, 2)))
        # 添加全连接层
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(4096, activation='relu'))
        #model.add(tf.keras.layers.Dropout(0.5))
        model.add(tf.keras.layers.Dense(4096, activation='relu'))
        model.add(tf.keras.layers.Dense(self.num_classes, activation='softmax'))

        model.summary()
        return model

    def compile_model(self, loss, optimizer, metrics):
        self.model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

    def fit(self, x_train, y_train, batch_size, epochs, validation_data):
        self.model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=validation_data)

    def predict(self,mfccs):
        return self.model.predict(mfccs)

#vgg = VGG((300,40 , 1), 3)
#vgg.compile_model(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# vgg.fit(X_train, y_train, batch_size=32, epochs=10, validation_data=(X_test, y_test))


# 采用如下命令将3维的输入扩展维4维的输入，该命令简单明了的解释可以参考下方链接：
# x_train = np.expand_dims(x_train, axis=3)
# x_test = np.expand_dims(x_test, axis=3)
