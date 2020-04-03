from tensorflow import keras
import csv
import codecs
from keras.preprocessing import image
import numpy as np
import glob
import re
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator


#把预测结果写入csv
def data_write_csv(file_name, datas):#file_name为写入CSV文件的路径，datas为要写入数据列表
    file_csv = codecs.open(file_name, 'w+', 'utf-8')  # 追加
    writer = csv.writer(file_csv, delimiter=' ', quotechar=' ', quoting=csv.QUOTE_MINIMAL)
    for data in datas:
        writer.writerow(data)
    print("保存文件成功，处理结束")

#加载预测数据
##用于排序
def sort_key(s):
    if s:
        try:
            c = re.findall('\d+', s)[0]
        except:
            c = -1
        return int(c)
def strsort(alist):
    alist.sort(key=sort_key,reverse=False)
    return alist

##预测数据路径
file_path = 'predictions/'
##预测数据名字
f_names = glob.glob(file_path + '*.jpg')
f_names = strsort(f_names)
images = []
# 把图片读取出来放到列表中
for i in range(len(f_names)):
    img = image.load_img(f_names[i], target_size=(64, 64))
    img= image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    images.append(img)

# 把图片数组联合在一起
images = np.concatenate([x for x in images])

# 重新创建完全相同的模型，包括其权重和优化程序
model = keras.models.load_model('my_model.h5')

# 显示网络结构
model.summary()

#进行预测
predictions = model.predict_classes(images)
#写文件
data_write_csv('prediction.csv', predictions)

