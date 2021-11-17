import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import pickle

df = pd.read_csv('houses_shuffled.csv')
df = df[df.asuntolkm != 0]
df.drop(['id', 'kuntanro','valmvuosi'], axis = 1, inplace = True)
df = df.loc[200:2154]

df.dropna(inplace=True)

X = df.loc[:, ['kuntanimi',
                'talotyyppi',
                'rahmuoto',
                'rahmuoto2']]
y = df.loc[:, ['asuntolkm']]

X_org = X
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(drop='first'),  ['kuntanimi',
                'talotyyppi',
                'rahmuoto',
                'rahmuoto2'])], remainder='passthrough')
X = ct.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print (f'r2: {r2}')
print (f'mae: {mae}')
print (f'rmse: {rmse}')

with open('housing-model.pickle', 'wb') as f:
    pickle.dump(model, f)

with open('housing-ct.pickle', 'wb') as f:
    pickle.dump(ct, f)

