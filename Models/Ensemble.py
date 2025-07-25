import pandas as pd
from sklearn.model_selection import train_test_split
from LogisticRegression import LogReg
from RandomForest import RF
from SVM import SVM


### Data inputting
data = pd.read_csv('Processed Data/finalData.csv')
futureData = data[data['year'] == 2026].drop(['republican_victory'], axis=1)
data = data[data['year'] != 2026]

# Generic data processing
y = data['republican_victory'].astype(bool)
X = data.drop(['state_po', 'year', 'district', 'republican_victory'], axis = 1)
futureX = futureData.drop(['state_po', 'year', 'district',], axis = 1)

# Data splitting
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=21)

# Model training
logreg = LogReg(X_train, y_train)
rf = RF(X_train, y_train)
svm = SVM(X_train, y_train)

# Future predictions
probs_df = pd.DataFrame()
probs_df['SVM'] = svm.predict_proba(futureX)[:,1].astype('float64')
probs_df['RF'] = rf.predict_proba(futureX)[:,1].astype('float64')
probs_df['LR'] = logreg.predict_proba(futureX)[:,1].astype('float64')

probs_df.to_csv('2026 Predictions/Probabilities.csv', index=False)
