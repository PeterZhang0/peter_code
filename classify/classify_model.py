import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import regularizers
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import cv2

#加载数据
##训练集合测试集的路径
train_dir = os.path.sep.join(['train'])
test_dir = os.path.sep.join(['test'])
##口罩路径
train_pos_path = os.path.sep.join(['train', 'pos'])
test_pos_path = os.path.sep.join(['test', 'pos'])
##非口罩路径
train_neg_path = os.path.sep.join(['train', 'neg'])
test_neg_path = os.path.sep.join(['test', 'neg'])
##查看数量
train_num_pos = len(os.listdir(train_pos_path))
test_num_pos = len(os.listdir(test_pos_path))
train_num_neg = len(os.listdir(train_neg_path))
test_num_neg = len(os.listdir(test_neg_path))
print("训练集中戴口罩的数量" + str(train_num_pos))
print("训练集中不戴口罩的数量" + str(train_num_neg))
print("测试集中戴口罩的数量" + str(test_num_pos))
print("测试集中不戴口罩的数量" + str(test_num_neg))

#设置变量
batch_size = 128
epochs = 300
image_height = 64
image_width = 64

#数据准备
##生成器准备
###测试集生成器
test_image_generator = ImageDataGenerator(rescale=1./255)

###训练数据生成器 扩充和可视化数据
image_gen_train = ImageDataGenerator(
                    rescale=1./255,
                    rotation_range=45,
                    width_shift_range=.15,
                    height_shift_range=.15,
                    horizontal_flip=True,
                    zoom_range=0.5
                    )

train_data_gen = image_gen_train.flow_from_directory(batch_size=batch_size,
                                                     directory=train_dir,
                                                     shuffle=True,
                                                     target_size=(image_height, image_width),
                                                     class_mode='binary')

augmented_images = [train_data_gen[0][0][0] for i in range(5)]

test_data_gen = test_image_generator.flow_from_directory(
    batch_size=batch_size,
    directory=test_dir,
    target_size=(image_height, image_height),
    class_mode='binary'
)
##可视化训练图像
sample_train_images, _ = next(train_data_gen)
def plotImages(images_arr):
    fig, axes = plt.subplots(1, 5, figsize=(20, 20))
    axes = axes.flatten()
    for img, ax in zip(images_arr, axes):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()

plotImages(sample_train_images)
plotImages(augmented_images)

#构建网络模型
model = Sequential([
    Conv2D(16, 3, kernel_regularizer=regularizers.l2(0.001), padding='same', activation='relu', input_shape=(image_height, image_width, 3)),
    MaxPooling2D(),
    Dropout(0.2),

    Conv2D(32, 3, kernel_regularizer=regularizers.l2(0.001), padding='same', activation='relu'),
    MaxPooling2D(),

    Conv2D(64, 3, kernel_regularizer=regularizers.l2(0.001), padding='same', activation='relu'),
    MaxPooling2D(),
    Flatten(),
    Dense(512, kernel_regularizer=regularizers.l2(0.001), activation='relu'),
    Dense(1)

])
#编译模型
model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy']
              )
#查看模型
model.summary()

#训练模型
history = model.fit_generator(
    train_data_gen,
    steps_per_epoch=(train_num_neg + train_num_pos) // batch_size,
    epochs=epochs,
    validation_data=test_data_gen,
    validation_steps=(test_num_neg + test_num_pos) // batch_size,
    # callbacks=[cp_callback]
)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

model.save('my_model.h5')

