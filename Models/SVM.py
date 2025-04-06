import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

educationData = pd.read_csv('C:/Users/finco/Documents/GitHub/DS496Capstone/Processed Data/compressedEducationData.csv')
raceData = pd.read_csv('C:/Users/finco/Documents/GitHub/DS496Capstone/Processed Data/raceData.csv')
labels = pd.read_csv('C:/Users/finco/Documents/GitHub/DS496Capstone/Processed Data/electionLabels.csv')

data = pd.merge(educationData, raceData, how='inner', on=['state_po', 'year', 'district'])
data = pd.merge(data, labels, how='inner', on=['state_po', 'year', 'district'])

y = data['republican_victory']
X = data.drop(['state_po', 'year', 'district', 'republican_victory'], axis = 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=21)

pipe_svm = Pipeline([('std', StandardScaler()), ('pca', PCA(n_components=0.95)), ('svc', SVC())])

param_range = [0.01, 0.1, 1, 10, 100]
svm_param_grid = [{'svc__C':param_range, 
                   'svc__kernel':['linear', 'rbf', 'poly', 'sigmoid'],
                    'svc__gamma':['scale','auto'] 
                   }] 

SVM_gs = GridSearchCV(estimator=pipe_svm, param_grid=svm_param_grid, scoring='f1', refit=True, cv=10, verbose=3)

SVM_gs = SVM_gs.fit(X_train, y_train)

print("Best SVM Model: ")
print("Model hyper-parameters: ", SVM_gs.best_params_)
print("Training data F1 score: ", SVM_gs.best_score_)