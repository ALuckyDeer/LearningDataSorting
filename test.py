#fastai new local environment test
from fastai.text.all import *
from fastai.vision.all import *
import torchvision.transforms as transforms
import math
import torch
from PIL import Image
import matplotlib.pyplot as plt
import torch.nn as nn



if __name__ == '__main__':
    path = untar_data(URLs.IMDB_SAMPLE)
    df = pd.read_csv(path/'texts.csv')
    dls = TextDataLoaders.from_df(df, path=path, text_col='text', label_col='label', valid_col='is_valid')
    learn = text_classifier_learner(dls, AWD_LSTM)
    print(learn.predict("i like fastai very much"))
    #print(math.log2(9911))
    #print(9911/(3*12))
    img = Image.open("./test.jpg")
    print("原图大小：", img.size)
    data1 = transforms.RandomResizedCrop(img.size)(img)
    print("随机裁剪后的大小:", data1.size)
    data2 = transforms.RandomResizedCrop(img.size)(img)
    data3 = transforms.RandomResizedCrop(img.size)(img)

    plt.subplot(2, 2, 1), plt.imshow(img), plt.title("原图")
    plt.subplot(2, 2, 2), plt.imshow(data1), plt.title("转换后的图1")
    plt.subplot(2, 2, 3), plt.imshow(data2), plt.title("转换后的图2")
    plt.subplot(2, 2, 4), plt.imshow(data3), plt.title("转换后的图3")
    plt.show()


