import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt

with open('housing-model.pickle', 'rb') as f:
    model = pickle.load(f)

with open('housing-ct.pickle', 'rb') as f:
    ct = pickle.load(f)


Xnew = pd.read_csv('houses_shuffled.csv')
Xnew = Xnew[Xnew.asuntolkm != 0]
Xnew.drop(['id', 'kuntanro','valmvuosi','asuntolkm'], axis = 1, inplace = True)
Xnew = Xnew.loc[0:199]
Xnew_org = Xnew
Xnew=ct.transform(Xnew)
ynew= model.predict(Xnew)

df_result = pd.DataFrame(Xnew_org)
house_values_arr =[]

for i in range (len(ynew)):
   # print (f'{Xnew_org.iloc[i]}\nHouse value:\t\t {round(ynew[i][0])}\n')
    house_values_arr.append(round(ynew[i][0]))

df_result['Predict'] = house_values_arr

df_all_houses = pd.read_csv('houses_shuffled.csv')
df_all_houses.drop(['id', 'kuntanro', 'valmvuosi'], axis = 1, inplace = True)
df_all_houses = df_all_houses.loc[0:199]
df_result['Real'] = df_all_houses.asuntolkm

plt.plot(df_result['Real'], 'bo', label='Real')
plt.plot(df_result['Predict'] , 'rx',label='Predict')

plt.title('Asuntojen lukumäärä ennuste vs todellinen')
plt.ylabel('Asuntojen lukumäärä')
plt.xlabel('Kohde id')
plt.legend(['Todellinen', 'Ennuste'], loc='upper right')
plt.show()

plt.plot(df_result['Real'], 'bo', label='Real')
plt.plot(df_result['Predict'] , 'rx',label='Predict')

plt.title('Asuntojen lukumäärä ennuste vs todellinen')
plt.ylabel('Asuntojen lukumäärä')
plt.xlabel('Kohde id')
plt.xlim(75, 200)
plt.ylim(0, 4)
plt.legend(['Todellinen', 'Ennuste'], loc='upper right')
plt.yticks(np.arange(0,4))
plt.show()