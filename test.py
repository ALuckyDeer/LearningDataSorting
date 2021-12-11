#fastai new local environment test
from fastai.text.all import *
from fastai.vision.all import *
from fastai.vision import *
import math
if __name__ == '__main__':
    # path = untar_data(URLs.IMDB_SAMPLE)
    # df = pd.read_csv(path/'texts.csv')
    # dls = TextDataLoaders.from_df(df, path=path, text_col='text', label_col='label', valid_col='is_valid')
    # learn = text_classifier_learner(dls, AWD_LSTM)
    # print(learn.predict("i like fastai very much"))
    print(math.log2(9911))
    print(9911/(3*12))

