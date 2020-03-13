import os.path
from PIL import Image
import glob
import matplotlib.image as mpimg
#改变图片大小
#改变图片大小函数
# def convertjpg(jpgfile,outdir,width,height):
#     img=Image.open(jpgfile)
#     try:
#         new_img=img.resize((width,height),Image.BILINEAR)
#         new_img.save(os.path.join(outdir,os.path.basename(jpgfile)))
#     except Exception as e:
#         print(e)

# #0类图片格式化
# for jpgfile in glob.glob(('E:/space/demo/kouzhao/train/neg/*.jpg')):
#         convertjpg(jpgfile, "E:/space/demo/kouzhao/train/new_neg", 300, 200)
# #1类图片格式化
# for jpgfile in glob.glob(('E:/space/demo/kouzhao/train/pos/*.jpg')):
#         convertjpg(jpgfile, "E:/space/demo/kouzhao/train/new_pos", 300, 200)

import os
import numpy as np
import matplotlib.pyplot as plt
import h5py


def get_files(file_dir):
    cats = []
    label_cats = []
    dogs = []
    label_dogs = []

    for file in os.listdir(file_dir + '/new_neg'):
        cats.append(file_dir + '/new_neg' + '/' + file)
        label_cats.append(0)  # 添加标签，该类标签为0，此为2分类例子，多类别识别问题自行添加
    for file in os.listdir(file_dir + '/new_pos'):
        dogs.append(file_dir + '/new_pos' + '/' + file)
        label_dogs.append(1)


    #把cat和dog合起来组成一个list（img和lab）
    image_list = np.hstack((cats, dogs))
    label_list = np.hstack((label_cats, label_dogs))
    #利用shuffle打乱顺序
    temp = np.array([image_list, label_list])
    temp = temp.transpose()
    np.random.shuffle(temp)
    #从打乱的temp中再取出list（img和lab）
    image_list = list(temp[:, 0])
    label_list = list(temp[:, 1])
    label_list = [int(i) for i in label_list]
    return image_list, label_list
# 返回两个list 分别为图片文件名及其标签  顺序已被打乱

train_dir = 'E:/space/demo/kouzhao/train'
image_list, label_list = get_files(train_dir)

# 测试为数据长度的20%
Train_image = np.zeros((len(image_list) - int(len(image_list) * 0.2), 200,300,3)).astype('u1')
Train_label = np.zeros((len(image_list) - int(len(image_list) * 0.2), 1)).astype('int')
Test_image = np.zeros((int(len(image_list) * 0.2), 200,300,3)).astype('u1')
Test_label = np.zeros((int(len(image_list) * 0.2), 1)).astype('int')


for i in range(len(image_list) - int(len(image_list) * 0.2)):
    if str(plt.imread(image_list[i]).shape) == '(200, 300)':
        continue
    else:
        Train_image[i] = np.array(mpimg.imread(image_list[i]))
        Train_label[i] = np.array(label_list[i])

for i in range(len(image_list) - int(len(image_list) * 0.2), len(image_list)):
    if str(plt.imread(image_list[i]).shape) == '(200, 300)':
        continue
    else:
        Test_image[i + int(len(image_list) * 0.2) - len(image_list)] = np.array(mpimg.imread(image_list[i]))
        Test_label[i + int(len(image_list) * 0.2) - len(image_list)] = np.array(label_list[i])

# Create a new file
f = h5py.File('data.h5', 'w')
f.create_dataset('X_train', data=Train_image)
f.create_dataset('y_train', data=Train_label)
f.create_dataset('X_test', data=Test_image)
f.create_dataset('y_test', data=Test_label)
f.close()


# train_dataset = h5py.File('data.h5', 'r')
# train_set_x_orig = np.array(train_dataset['X_train'][:]) # your train set features
# train_set_y_orig = np.array(train_dataset['y_train'][:]) # your train set labels
# test_set_x_orig = np.array(train_dataset['X_test'][:]) # your train set features
# test_set_y_orig = np.array(train_dataset['y_test'][:]) # your train set labels
# f.close()
# print("原始训练数据特征维度：" + str(train_set_x_orig.shape))
# print("原始训练数据标签维度：" + str(train_set_y_orig.shape))
# print("原始测试数据特征维度：" + str(test_set_x_orig.shape))
# print("原始测试数据标签维度：" + str(test_set_y_orig.shape))

# def load_data():
#     train_dataset = h5py.File('datasets/train_catvnoncat.h5', "r")
#     train_set_x_orig = np.array(train_dataset["train_set_x"][:])  # your train set features
#     train_set_y_orig = np.array(train_dataset["train_set_y"][:])  # your train set labels
#
#     test_dataset = h5py.File('datasets/test_catvnoncat.h5', "r")
#     test_set_x_orig = np.array(test_dataset["test_set_x"][:])  # your test set features
#     test_set_y_orig = np.array(test_dataset["test_set_y"][:])  # your test set labels
#
#     classes = np.array(test_dataset["list_classes"][:])  # the list of classes
#
#     train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
#     test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
#
#     return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes