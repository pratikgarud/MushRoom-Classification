import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


df = pd.read_csv('mushroom_dataset.csv')
df.head()
df.rename(columns = {'mushroom' : 'Mushroom'}, inplace=True)

label = LabelEncoder()
df['cap-surface'] = label.fit_transform(df['cap-surface'])
df['gill-size'] = label.fit_transform(df['gill-size'])
df['veil-color'] = label.fit_transform(df['veil-color'])
df['ring-number'] = label.fit_transform(df['ring-number'])
df['ring-type'] = label.fit_transform(df['ring-type'])
df['stalk-shape'] = label.fit_transform(df['stalk-shape'])

X = df[['cap-surface','gill-size','veil-color','ring-number','ring-type','stalk-shape']]
y = df['Mushroom']

X_train, X_test , y_train, y_test = train_test_split(X,y, random_state=0, test_size=0.2)
model = RandomForestClassifier(n_estimators=142)
model.fit(X_train,y_train)
y_pred = model.predict(X_test)

from sklearn.metrics import accuracy_score, confusion_matrix
print("Accuracy", accuracy_score(y_test,y_pred)*100)

pre = model.predict([[3,1,2,2,4,0]])
pred = np.asscalar(pre)
print(pred)

import pickle
with open('Mushroom_Model.pkl', 'wb') as f:
    pickle.dump(model,f)