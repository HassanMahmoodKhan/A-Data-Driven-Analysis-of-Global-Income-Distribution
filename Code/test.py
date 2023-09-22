import pandas as pd
import pickle
import time
from sklearn.metrics import accuracy_score
from plotting import*

def evaluate(X_train, y_train, X_test, y_test):
    # Predictions on the testing set for all models to evaluate model performance
    loaded_knn = pickle.load(open("../Models/knn.pickle", "rb"))
    loaded_svm = pickle.load(open("../Models/svm.pickle", "rb"))
    loaded_rf = pickle.load(open("../Models/rf.pickle", "rb"))

    clfs = [
                ('KNN', loaded_knn), 
                ('SVM', loaded_svm), 
                ('RF', loaded_rf)
            ]

    scores_df = pd.DataFrame(columns=['Model', 'Accuracy', 'Latency'])
    predictions_df = pd.DataFrame(columns=['Model', 'Predictions','Prediction-Probability'])
    i = 0
    for clf_name, clf in clfs:

        clf.fit(X_train, y_train)
        start_time = time.time()
        y_pred = clf.predict(X_test)
        end_time = time.time()
        inference_time = end_time - start_time
        y_prob = clf.predict_proba(X_test)
        accu = accuracy_score(y_test, y_pred)        

        data = {
                'Model' : [clf_name], 
                'Accuracy' : [accu],
                'Latency' : [inference_time]
                }
        data_df = pd.DataFrame(data)
        scores_df = pd.concat([scores_df,data_df], ignore_index=True)

        pred = {
                'Model' : [clf_name], 
                'Predictions' : [y_pred],
                'Prediction-Probability' : [y_prob]
                }
        pred_df = pd.DataFrame(pred)
        predictions_df = pd.concat([predictions_df, pred_df], ignore_index=True)
        
        ConfusionMatrix(clf_name, y_test, predictions_df['Predictions'][i])
        ROCCurve(clf_name, predictions_df['Prediction-Probability'][i], y_test)
        i+=1

    scores_df.to_csv("../Output/log.txt", sep=",", index=False)
    print("Evaluation complete!")
   
    