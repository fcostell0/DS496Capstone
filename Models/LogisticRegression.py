import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score


### Data inputting
data = pd.read_csv('C:/Users/finco/Documents/GitHub/DS496Capstone/Processed Data/finalData.csv')
futureData = data[data['year'] == 2026].drop(['republican_victory'], axis=1)
data = data[data['year'] != 2026]

y = data['republican_victory'].astype(bool)
X = data.drop(['state_po', 'year', 'district', 'republican_victory'], axis = 1)
futureX = futureData.drop(['state_po', 'year', 'district',], axis = 1)

### Data splitting
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=21)

### Pipeline/grid search
pipe_lr = Pipeline([('std', StandardScaler()), ('lr', LogisticRegression())])

param_range = [0.01, 0.1, 1, 10, 100]
lr_param_grid = {
    'lr__C':param_range,
    'lr__solver':['lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga'],
    'lr__max_iter':[100, 500, 1000]
}

lr_gs = GridSearchCV(estimator=pipe_lr, param_grid=lr_param_grid, scoring='f1', refit=True, cv=10, verbose=3)
    
lr_gs = lr_gs.fit(X_train, y_train)


### Model Diagnostics

print("Best Logistic Regression Model: ")
print("Model hyper-parameters: ", lr_gs.best_params_)
print("Validation data F1: ", lr_gs.best_score_)

print("Training Classification Report: ")
y_train_pred = lr_gs.best_estimator_.predict(X_train)
print(classification_report(y_train, y_train_pred))

print("Final Classification Report: ")
y_pred = lr_gs.best_estimator_.predict(X_test)
y_probs = lr_gs.best_estimator_.predict_proba(X_test)[:,1]
print(classification_report(y_test, y_pred))

# ROC Curve Output
plt.rcParams['font.family'] = 'Times New Roman'
fpr, tpr, thresholds = roc_curve(y_test, y_probs, pos_label=1)
roc_auc = roc_auc_score(y_test,y_probs)
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

futureData['Republican Prob'] = lr_gs.best_estimator_.predict_proba(futureX)[:,1]
futureData.to_csv('C:/Users/finco/Documents/GitHub/DS496Capstone/2026 Predictions/FutureModelPredictionData.csv', index=False)