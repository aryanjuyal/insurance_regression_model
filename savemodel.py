

import pandas as pd
import numpy as np
import pickle




dataset = pd.read_csv('insurance.csv')



dataset['sex'] = dataset['sex'].replace({"male":"1","female":"0"})
dataset['smoker'] = dataset['smoker'].replace({"yes":"1","no":"0"})



X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values



from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [5])], remainder='passthrough')
X = np.array(ct.fit_transform(X))
print(X)



from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


from sklearn.preprocessing import StandardScaler
sc=StandardScaler()




X_train[:,:5]=sc.fit_transform(X_train[:,:5])
X_test[:,:5]=sc.transform(X_test[:,:5])






from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)




y_pred = regressor.predict(X_test)



print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))



print(regressor.predict([[1,0,0,0,18,1,33,2,1]]))



from sklearn.metrics import r2_score
r2=r2_score(y_test,y_pred)
print(r2)


with open('regressor.pkl','wb') as f:
    pickle.dump(regressor,f)



