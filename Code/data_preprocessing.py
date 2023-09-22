import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder, MinMaxScaler
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold

def preprocessing(data):
    
    #Feature Engineering
    data.loc[data["native-country"] != "United-States", "native-country"] = 'Not United-States'
    data['capital-gain']= np.sqrt(data['capital-gain'])
    data['capital-loss']= np.sqrt(data['capital-loss'])

    #Data Pre-processing
    label_income = LabelEncoder() # Label encoding target column i.e., categorical to numerical representation
    data['income'] = label_income.fit_transform(data['income'])

    x = data.drop(['income', 'education'], axis = 1) # feature matrix; education is a redundant feature
    y = data['income'] # target series

    #Column Transformer
    ordinal_features = ['age', 'education-num', 'hours-per-week']  # Pipeline for column transformation using scaling and encoding
    numeric_features = ['fnlwgt','capital-gain', 'capital-loss']
    categorical_features = ['workclass', 'marital-status','occupation', 'relationship','race', 'sex','native-country']

    ordinal_transformer = make_pipeline(
        SimpleImputer(strategy="mean"),
        MinMaxScaler()
    )

    numeric_transformer = make_pipeline(
        SimpleImputer(strategy="median"), 
        StandardScaler()
    )

    categorical_transformer= make_pipeline(
        SimpleImputer(strategy="constant", fill_value="Unknown"),
        OneHotEncoder(drop="if_binary", handle_unknown="ignore")
    )

    col_transformer = make_column_transformer(
        (ordinal_transformer, ordinal_features),
        (numeric_transformer, numeric_features),
        (categorical_transformer, categorical_features),
        remainder='passthrough', sparse_threshold=0
    )

    col_transformer.fit(x)

    onehot_cols = (
    col_transformer
    .named_transformers_["pipeline-3"]
    .named_steps["onehotencoder"]
    .get_feature_names_out(categorical_features)
    )

    columns = ordinal_features + numeric_features + onehot_cols.tolist()

    X = col_transformer.transform(x)
    X = pd.DataFrame(X, columns = columns)
    

    print("Preprocessing completed!")
    return X,y


def gs_pipeline(X, y): 
    
    # Helper function for implementing grid search for hyperparamter tuning for all three classifiers
    parameters1 = {
    'n_neighbors': np.arange(1, 50, 2),
    'weights': ['uniform', 'distance'],
    'p': [1,2],
    'n_jobs': [-1]
    }
    
    parameters2 = {
    'C': np.arange(0.1, 1, 0.1),
    'kernel': ['rbf', 'poly'],
    'gamma': [0.01, 0.05, 0.1, 0.5, 1.0],
    'probability': [True]
    }
        
    parameters3 = {
    'n_estimators': np.arange(start=100, stop=150, step=5),
    'criterion': ['gini', 'entropy'],
    'max_features': ['sqrt', 'log2'],
    'max_depth': np.arange(start=2, stop=42, step=2),
    'min_samples_leaf':np.arange(start = 1, stop = 11, step = 1),
    'n_jobs': [-1]
    }


    models = ['K-Nearest Neighbors', 'Support Vector Machine', 'Random Forest' ]
    clfs = [ KNeighborsClassifier(), SVC(), RandomForestClassifier() ]
    parameters = [ parameters1, parameters2, parameters3 ] 
    
    grid_search = pd.DataFrame(columns=['Model', 'Best Parameters', 'Accuracy'])
    
    skf = StratifiedKFold(n_splits=5)

    for i in range(len(clfs)):
        
        results = GridSearchCV(clfs[i], param_grid = parameters[i], scoring='accuracy', n_jobs=-1, error_score='raise')
        results.fit(X,y)
    
        grid_search = grid_search.append({
                                      'Model' : models[i], 
                                      'Best Parameters' : results.best_params_,
                                      'Accuracy' : results.best_score_
                                      }, 
                                     ignore_index=True)
    print("Grid Search completed!")
    return grid_search 

def my_train_test_split(X,y):

    # Splitting the dataset into training and testing subsets
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)
    return X_train, X_test, y_train, y_test

