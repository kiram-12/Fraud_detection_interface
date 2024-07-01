import pandas as pd
from sklearn.model_selection import cross_validate, StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_curve, auc, confusion_matrix
from imblearn.over_sampling import SMOTE, RandomOverSampler, ADASYN, BorderlineSMOTE
from imblearn.under_sampling import RandomUnderSampler, NearMiss, TomekLinks, CondensedNearestNeighbour, ClusterCentroids

# K_Fold Cross Validation
def K_fold_cross_validation_tec(X, y, model, k=10):
    scoring = ['accuracy', 'precision', 'recall', 'f1']
    cv = StratifiedKFold(n_splits=k)
    scores = cross_validate(model, X, y, cv=cv, scoring=scoring)
    scores_df = pd.DataFrame(scores)
    scores_df.loc['Mean'] = scores_df.mean()
    
    y_pred = cross_val_predict(model, X, y, cv=cv)
    fpr, tpr, _ = roc_curve(y, y_pred)
    roc_auc = auc(fpr, tpr)
    cm = confusion_matrix(y, y_pred)

    return scores_df, fpr, tpr, roc_auc, cm

# Decision Tree Model
def DT_modele():
    from sklearn.tree import DecisionTreeClassifier
    clf = DecisionTreeClassifier(random_state=42)
    return clf

# SVM Model
def SVM_modele():
    from sklearn.svm import SVC
    clf = SVC(random_state=42, probability=True)
    return clf

# KNN Model
def KNN_modele():
    from sklearn.neighbors import KNeighborsClassifier
    clf = KNeighborsClassifier()
    return clf

# Balancing Techniques
def apply_balancing_technique(X, y, technique):
    if technique == 'RandomUnderSampler':
        sampler = RandomUnderSampler()
    elif technique == 'NearMiss':
        sampler = NearMiss()
    elif technique == 'TomekLinks':
        sampler = TomekLinks()
    elif technique == 'CondensedNearestNeighbour':
        sampler = CondensedNearestNeighbour()
    elif technique == 'ClusterCentroids':
        sampler = ClusterCentroids()
    elif technique == 'RandomOverSampler':
        sampler = RandomOverSampler()
    elif technique == 'SMOTE':
        sampler = SMOTE()
    elif technique == 'ADASYN':
        sampler = ADASYN()
    elif technique == 'BorderlineSMOTE':
        sampler = BorderlineSMOTE()
    else:
        return X, y

    X_resampled, y_resampled = sampler.fit_resample(X, y)
    return X_resampled, y_resampled

# Test model
def Test_model(X, y, model_func, oversampling_tec):
    model = model_func()
    X_resampled, y_resampled = apply_balancing_technique(X, y, oversampling_tec)
    scores, fpr, tpr, roc_auc, cm = K_fold_cross_validation_tec(X_resampled, y_resampled, model)
    return scores, fpr, tpr, roc_auc, cm
