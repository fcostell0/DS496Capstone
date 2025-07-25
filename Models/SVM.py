from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV


def SVM(X_train, y_train):
    pipe_svm = Pipeline([('std', StandardScaler()), ('svc', SVC(probability=True))])

    param_range = [0.01, 0.1, 1, 10, 100]
    svm_param_grid = [{'svc__C':param_range, 
                    'svc__kernel':['linear', 'rbf', 'poly', 'sigmoid'],
                    'svc__gamma':['scale','auto'] 
                    }] 

    SVM_gs = GridSearchCV(estimator=pipe_svm, param_grid=svm_param_grid, scoring='f1', refit=True, cv=5, verbose=3)

    SVM_gs = SVM_gs.fit(X_train, y_train)

    best_SVM = SVM_gs.best_estimator_
    
    calibrated_SVM = CalibratedClassifierCV(estimator=best_SVM.named_steps['svc'], method='sigmoid', cv=3)
    calibrated_SVM.fit(X_train, y_train)

    return calibrated_SVM
