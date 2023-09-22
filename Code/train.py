import pandas as pd
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import pickle

def train(X_train, y_train):
 
    # Performing Cross Validation on the training dataset for all models to validate model hyperparamters and performance
    knn = KNeighborsClassifier(n_neighbors=25, weights = 'uniform', p=2, n_jobs = -1)
    svm = SVC(C=0.9, gamma=0.1, kernel='rbf', probability=True)
    rf = RandomForestClassifier(n_estimators= 130, criterion="gini", max_depth = 24, max_features = "sqrt", bootstrap=False,
                                n_jobs = -1, min_samples_leaf=3)


    knn_clr = knn.fit(X_train, y_train)
    filename_knn = "../Models/knn.pickle" # save model
    pickle.dump(knn_clr, open(filename_knn, "wb"))
    
    svm_clr = svm.fit(X_train, y_train)
    filename_svm = "../Models/svm.pickle"
    pickle.dump(svm_clr, open(filename_svm, "wb"))

    rf_clr = rf.fit(X_train, y_train)
    filename_rf = "../Models/rf.pickle"
    pickle.dump(rf_clr, open(filename_rf, "wb"))

    print("Training complete!")