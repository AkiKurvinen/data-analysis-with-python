import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.utils import shuffle

df = pd.read_csv('houses_shuffled.csv' )

df = df[df.asuntolkm != 0]
df.drop(['id', 'kuntanro','valmvuosi'], axis = 1, inplace = True)
df = df.loc[200:2154]

df.isnull().values.any()
df.isnull().sum().sum()

y = df.loc[:, ['asuntolkm']]
X = df.drop('asuntolkm', axis=1)

useFileds =['kuntanimi','talotyyppi','rahmuoto','rahmuoto2']
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(drop='first',sparse=False,handle_unknown = 'error'), useFileds)], remainder='passthrough')
X = ct.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.15, random_state = 42)

scaler_X = StandardScaler()
X_train = scaler_X.fit_transform(X_train)
X_test = scaler_X.transform(X_test)

scaler_y = StandardScaler()
y_train = scaler_y.fit_transform(y_train)
y_test = scaler_y.transform(y_test)

model = Sequential()
model.add(Dense(units=8, input_dim=X.shape[1],activation='relu'))
model.add(Dense(units=6, activation='relu'))
model.add(Dense(units=1, activation='linear'))

model.compile(loss='mse', optimizer='adam',metrics=['mse'])
history=model.fit(X_train, y_train, epochs=50, batch_size=32,
                  validation_data=(X_test, y_test))

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epochs')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

y_pred = model.predict(X_test)
y_pred = scaler_y.inverse_transform(y_pred)
y_test = scaler_y.inverse_transform(y_test)

r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print (f'r2: {r2}')
print (f'mae: {mae}')
print (f'rmse: {rmse}')

model.save('house.h5')

with open('house-scaler_X.pickle', 'wb') as f:
    pickle.dump(scaler_X, f)

with open('house-scaler_y.pickle', 'wb') as f:
    pickle.dump(scaler_y, f)

with open('house-ct.pickle', 'wb') as f:
    pickle.dump(ct, f)
