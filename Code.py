#code
import pandas as pd
import numpy as np

df=pd.read_csv("Depression.csv")
df.head()
X=df[['Insomnia','Anxiety','frustration']]
Y=df['Output']

print(df.head())
train_data=df.sample(frac=0.8)

test_data=df.drop(train_data.index)

print(train_data.shape)
print(test_data.shape)

train_data=train_data.drop(labels='Sr.no',axis=1)
test_data=test_data.drop(labels='People',axis=1)

train_tar=train_data.pop('Output')
train_data.head()
test_tar=test_data.pop("Output")
test_data.head()

from membership import membershipfunction
mfc = membershipfunction.MemFuncs(mf)
import anfis
anf = anfis.ANFIS(train_data,train_tar, mfc)
predict_train=anf.trainHybridJangOffLine(epochs=30)
anf.plotErrors()
anf.plotResults()

predict_train
train_tar
