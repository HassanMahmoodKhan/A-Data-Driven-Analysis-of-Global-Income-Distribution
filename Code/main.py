import pandas as pd
from data_preprocessing import*
from train import train
from test import evaluate
from onnxrt import*

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

def main():

    # Read the dataset
    data = pd.read_csv("../Dataset/adult.csv", header=0)
    
    # Preprocess data
    X,y = preprocessing(data) 
    
    # Uncomment to perform hyperparamter grid search
    # grid_search = gs_pipeline(X, y) 
    # print(grid_search)
    
    # Split the dataset into training and test datasets
    X_train, X_test, y_train, y_test = my_train_test_split(X,y)

    # Uncomment to perform model training
    # train(X_train, y_train)

    # ONNX Conversion
    onnx_conversion(X, y)

    # Model Evaluation/Inference
    # evaluate(X_train, y_train, X_test, y_test)

    # ONNX Runtime Inference
    onnx_inference(X_test, y_test)


if __name__ == '__main__':
    main()



