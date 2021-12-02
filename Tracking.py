import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
import tensorflow as tf
from functions import *

truth_file = '~/Documents/School_Documents/Classwork/Research_Boveia/Data/event100000594-truth.csv'
hits_file = '~/Documents/School_Documents/Classwork/Research_Boveia/Data/event100000594-hits.csv'


truth = pd.read_csv(truth_file)


data = pd.read_csv(hits_file)
data['particle_id'] = truth['particle_id']
data = data.loc[data['particle_id'] != 0]
group_data = data.groupby(data.particle_id)
data.drop(['hit_id','volume_id', 'layer_id','module_id'], inplace = True, axis =1)


train_data_x, train_data_y, test_data_x, test_data_y, valid_data_x, valid_data_y, max_values = train_test_split(data)
print(max_values)


