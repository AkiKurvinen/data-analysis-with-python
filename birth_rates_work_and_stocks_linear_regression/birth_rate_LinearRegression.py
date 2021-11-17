import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
import scipy.stats as stats

df_born = pd.read_csv('born.csv')
df_born = df_born.set_index('Date').T

fields = ['Date', 'Close']
df_ndx = pd.read_csv('ndx.csv', skipinitialspace=True, usecols=fields)
df_ndx['Date'] = df_ndx['Date'].astype('datetime64[ns]')
df_ndx = df_ndx.set_index('Date').resample('Q')[ 'Close'].mean().to_frame()

df_emp = pd.read_csv('emp_per_q.csv')
df_emp.set_index('quartal', inplace=True)

df_hrs = pd.read_csv('h_per_q.csv')
df_hrs.set_index('quartal', inplace=True)

df_all = df_ndx
df_all = df_all.set_index(df_ndx.index)
df_all["emp_x1000"] = df_emp.iloc[:,0].values
df_all["h_x1M"] = df_hrs["h_x1M"].values
df_all["born"] = df_born["born"].values

all_X = df_all.iloc[:, [0]]
all_y = df_all.iloc[:, [3]]

sns.regplot(x=all_X, y=all_y,  data=df_all).set_title("NDX closing price vs Born infants")
plt.show()

pearsonr = stats.pearsonr(all_X.iloc[:,0].values, all_y.iloc[:,0].values)
print('NDX-Born')
print(f'\npearson:{pearsonr}')

chi_df = df_ndx
chi_df["born"] = df_born["born"].values
chi_df = pd.crosstab(index=chi_df['born'], columns=chi_df['Close'])
p = stats.chi2_contingency(chi_df)[1]


if p > 0.05:
    print(f'Riippuvuus ei ole tilastollisesti merkitsevä, p={round(p,4)}')
else:
    print(f'Riippuvuus on tilastollisesti merkitsevä, p={p}')

all_X = df_all.iloc[:, [1]]
all_y = df_all.iloc[:, [3]]

sns.regplot(x=all_X, y=all_y,  data=df_all).set_title("Employed (x 1000) vs Born infants")
plt.show()

pearsonr = stats.pearsonr(all_X.iloc[:,0].values, all_y.iloc[:,0].values)
print('Employed-Born')
print(f'\npearson:{pearsonr}')

chi_df = df_emp
chi_df["born"] = df_born["born"].values
chi_df = pd.crosstab(index=chi_df['born'], columns=chi_df['emp_x1000'])
p = stats.chi2_contingency(chi_df)[1]


if p > 0.05:
    print(f'Riippuvuus ei ole tilastollisesti merkitsevä, p={round(p,4)}')
else:
    print(f'Riippuvuus on tilastollisesti merkitsevä, p={p}')

all_X = df_all.iloc[:, [2]]
all_y = df_all.iloc[:, [3]]

sns.regplot(x=all_X, y=all_y,  data=df_all).set_title("Working hours (x 1M) vs vs Born infants")
plt.show()
pearsonr = stats.pearsonr(all_X.iloc[:,0].values, all_y.iloc[:,0].values)
print('Working Hrs-Born')
print(f'\npearson:{pearsonr}')

chi_df  = df_hrs
chi_df["born"] = df_born["born"].values
chi_df = pd.crosstab(index=chi_df['born'], columns=chi_df['h_x1M'])
p = stats.chi2_contingency(chi_df)[1]


if p > 0.05:
    print(f'Riippuvuus ei ole tilastollisesti merkitsevä, p={round(p,4)}')
else:
    print(f'Riippuvuus on tilastollisesti merkitsevä, p={p}')

# test without NDX
# df_all.drop(['Close'], axis = 1, inplace=True)

X = df_all.iloc[:,:-1]
y = df_all.iloc[:,[-1]]
X_original = X;

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

r2=r2_score(y_test,y_pred)
mae=mean_absolute_error(y_test,y_pred)
mse=mean_squared_error(y_test,y_pred)
rmse = np.sqrt(mse)

print('\n')
print(f'r2={r2}')
print(f'mae={mae}')
print(f'mse={mse}')
print(f'rmse={rmse}')

plt.title('Born pred vs actual')
plt.scatter(X_test.index,y_pred, c="r", label='Predicted')
plt.scatter(X_test.index,y_test, c="b", label='Actual')
plt.xlabel("Year")
plt.ylabel("infants")
plt.legend(loc="lower left")
plt.show()

plt.title('Newborns per Quartal vs Year')
plt.xlabel("Year")
plt.ylabel("newborns x10")

plt.plot(df_all.index, df_all['born']/10, c='b', label="infants/Q")

plt.scatter(X_test.index,y_pred/10, c="r", label='Predict')
plt.plot(df_all.index, df_all['Close']**0.7+1500, c='seagreen', label="NDX")
plt.plot(df_all.index, df_all['h_x1M']**1.1-600, c='c', label="Hours worked")
plt.plot(df_all.index, df_all['emp_x1000']-900, c='gray', label="Employed")
plt.legend(loc="upper left")
plt.show()
