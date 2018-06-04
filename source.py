import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

data_train = pd.read_csv('./data/train.csv', index_col='id')
data_test = pd.read_csv('./data/test.csv', index_col='id')

print(data_train.corr())

print(data_train.columns[data_train.isnull().values.any()].tolist())
train, validation = train_test_split(data_train, test_size=0.2)
