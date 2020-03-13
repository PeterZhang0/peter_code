import os.path
from PIL import Image
import glob
import h5py
import os
import numpy as np
import matplotlib.pyplot as plt
import re
import matplotlib.image as mpimg

print("改变图片大小函数")
#改变图片大小函数
# def convertjpg(jpgfile,outdir,width,height):
#     img=Image.open(jpgfile)
#     try:
#         new_img=img.resize((width,height),Image.BILINEAR)
#         new_img.save(os.path.join(outdir,os.path.basename(jpgfile)))
#     except Exception as e:
#         print(e)
#
# #0类图片格式化
# for jpgfile in glob.glob(('E:/space/demo/kouzhao/toPredict/*.jpg')):
#         convertjpg(jpgfile, "E:/space/demo/kouzhao/new_predict", 300, 200)




#创建h5文件 并按照文件夹名排序
print("创建h5文件，并按照文件夹名字排序")
# def sort_key(s):
#     if s:
#         try:
#             c = re.findall('\d+', s)[0]
#         except:
#             c = -1
#         return int(c)
# def strsort(alist):
#     alist.sort(key=sort_key,reverse=False)
#     return alist
#
# def get_files(file_dir):
#     cats = []
#     for file in os.listdir(file_dir + '/new_predict'):
#         cats.append(file_dir + '/new_predict' + '/' + file)
#     cats = strsort(cats)
#     print(cats)
#     return cats
#
# train_dir = 'E:/space/demo/kouzhao'
# image_list = get_files(train_dir)
#
# Train_image = np.zeros((len(image_list), 200,300,3)).astype('u1')
#
# for i in range(len(image_list)):
#     if str(plt.imread(image_list[i]).shape) == '(200, 300)' or str(plt.imread(image_list[i]).shape) == '(200, 300, 4)' :
#         continue
#     else:
#         Train_image[i] = np.array(mpimg.imread(image_list[i]))
#
# f = h5py.File('predict_data.h5', 'w')
# f.create_dataset('X_train', data=Train_image)
# f.close()
# print("aaaaaa")

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

print("查看数据")
def predict_data_set():
    data_set = h5py.File('predict_data.h5', 'r')
    prediction_data_set = np.array(data_set["X_train"][:])
    return prediction_data_set
predict_data_set = predict_data_set()
plt.imshow(predict_data_set[3799])
plt.show()
#
