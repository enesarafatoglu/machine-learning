#diyabet veri setini yükleme ve özetleme
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, RepeatedKFold
from xgboost import XGBClassifier

# veri setini yükleme
dataframe = pd.read_csv('diabetes.csv')
data = dataframe.values

# veri şeklini özetleme
print(dataframe.shape)

# ilk birkaç satırı gosterme
print(dataframe.head())

# veriyi giriş ve çıkış sütunlarına bölme
X, y = data[:, :-1], data[:, -1]

# model tanimlama
model = XGBClassifier(n_estimators=100, learning_rate=0.001)

# model değerlendirme yöntemini tanımlama
cv = RepeatedKFold(n_splits=10, n_repeats=30, random_state=42)

# model degerdirme
scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)

# puanları pozitif olmaya zorlama
scores = np.absolute(scores)
print('Mean Accuracy: %.3f (%.3f)' % (scores.mean(), scores.std()) )
