# -*- coding: utf-8 -*-

import pandas as pd
import matplotlib.pyplot as plt

path_name = r'D:\brain_tumor_segment\models\dataset\log.csv'
log = pd.read_csv(path_name)
epoch = log['epoch']
loss = log['loss']
val_loss = log['val_loss']
plt.figure()
plt.plot(epoch,loss,label = 'loss', color = 'red')
plt.plot(epoch,val_loss,label = 'val_loss', color = 'blue')
plt.show()
print(1)
