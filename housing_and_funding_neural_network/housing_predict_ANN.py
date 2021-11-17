import pandas as pd
import numpy as np
import pickle
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

model = load_model('house.h5')

with open('house-ct.pickle', 'rb') as f:
    ct = pickle.load(f)

with open('house-scaler_X.pickle', 'rb') as f:
    scaler_X = pickle.load(f)

with open('house-scaler_y.pickle', 'rb') as f:
    scaler_y = pickle.load(f)

df = pd.read_csv('houses_shuffled.csv')
df = df[df.asuntolkm != 0]
df.drop(['id', 'kuntanro','valmvuosi'], axis = 1, inplace = True)
df = df.dropna(axis = 0, how ='any')
df = df.loc[0:199]

df = df.dropna(axis = 0, how ='any')
df.isnull().values.any()
df.isnull().sum().sum()
df = df.drop([], axis = 1)
y = df.loc[:, ['asuntolkm']]

X = df
Xnew = X
Xnew_org = Xnew
X = X.drop('asuntolkm', axis=1)
Xnew = Xnew.drop('asuntolkm', axis=1)

Xnew = ct.transform(Xnew)
Xnew = scaler_X.transform(Xnew)
ynew = model.predict(Xnew)
ynew = scaler_y.inverse_transform(ynew)
ynew = ynew.round()

f = open('text.txt','w')
for i in range (len(ynew)):
    print (f'{Xnew_org.iloc[i]}\nAsunnot: {ynew[i][0]}\n', file=f)

y.reset_index().plot(kind='scatter', x='index', y='asuntolkm')
ynew = pd.DataFrame(ynew)
plt.scatter(y=ynew,  x=ynew.index.values,c='r',marker='x')
plt.title('Asuntojen lukumäärä ennuste vs todellinen')
plt.ylabel('Asuntojen lukumäärä')
plt.xlabel('Kohde id')
plt.legend(['Todellinen', 'Ennuste'], loc='upper right')
ax = plt.gca()
plt.show()

y.reset_index().plot(kind='scatter', x='index', y='asuntolkm')
ynew = pd.DataFrame(ynew)
plt.scatter(y=ynew,  x=ynew.index.values,c='r',marker='x')
plt.title('Apartments predict vs Apartments real')
plt.ylabel('Apartments')
plt.xlabel('Building site id')
plt.legend(['Real', 'Predicted'], loc='upper right')
ax = plt.gca()
ax.set_xlim([75, 200])
ax.set_ylim([0, 5])
plt.show()
