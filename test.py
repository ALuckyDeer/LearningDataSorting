#fastai new local environment test
# from fastai.text.all import *
# from fastai.vision.all import *
# import torchvision.transforms as transforms
# import math
# import torch
from PIL import Image
import matplotlib.pyplot as plt
import torch.nn as nn
import numpy as np
import pandas as pd


if __name__ == '__main__':


    # 创建一个Dataframe
    data = pd.DataFrame(np.arange(16).reshape(4, 4), index=list('abcd'), columns=list('ABCD'))
    print(data)
    print(data.iloc[1,-1])


