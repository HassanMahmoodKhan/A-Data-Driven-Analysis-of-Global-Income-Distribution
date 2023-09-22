import numpy as np
import pandas as pd
import pickle
from skl2onnx import convert_sklearn
from sklearn.metrics import accuracy_score
from skl2onnx.common.data_types import FloatTensorType
import onnxruntime as ort
import time

def onnx_conversion(X, y):

    input_types = [('input', FloatTensorType(shape=[None, len(X.columns)]))]

    loaded_knn = pickle.load(open("../Models/knn.pickle", "rb"))
    loaded_svm = pickle.load(open("../Models/svm.pickle", "rb"))
    loaded_rf = pickle.load(open("../Models/rf.pickle", "rb"))

    # Convert the scikit-learn model to ONNX format and save it
    knn_onnx_model = convert_sklearn(loaded_knn, initial_types=input_types)
    with open('../Models/knn.onnx', "wb") as f:
        f.write(knn_onnx_model.SerializeToString())

    svm_onnx_model = convert_sklearn(loaded_svm, initial_types=input_types)
    with open('../Models/svm.onnx', "wb") as f:
        f.write(svm_onnx_model.SerializeToString())

    rf_onnx_model = convert_sklearn(loaded_rf, initial_types=input_types)
    with open('../Models/rf.onnx', "wb") as f:
        f.write(rf_onnx_model.SerializeToString())

    print("Models converted successfully!")


def onnx_inference(X_test, y_test):

    models = ['knn', 'svm', 'rf']
    scores_df = pd.DataFrame(columns=['Model', 'Accuracy', 'Latency'])

    for i in range(len(models)):
        sess = ort.InferenceSession('../Models/' + models[i] + '.onnx',  providers = ['CPUExecutionProvider'])
        input_name = sess.get_inputs()[0].name
        label_name = sess.get_outputs()[0].name
        input_data = np.array(X_test.astype(np.float32))
        start_time = time.time()
        predictions = sess.run([label_name], {input_name: input_data})[0]
        end_time = time.time()
        inference_time = end_time - start_time
            
        acc = accuracy_score(predictions, y_test)
        # n_correct += (predicted == np.array(y_test)).sum().item()
        # acc = 100.0 * (n_correct/len(X_test))
        scores = {
            'Model': [models[i]],
            'Accuracy': [acc],
            'Latency': [inference_time]
        }
        data_df = pd.DataFrame(scores)
        scores_df = pd.concat([scores_df,data_df], ignore_index=True)
    scores_df.to_csv("../Output/onnx_log.txt", sep=',', index=False)
    print("Inference Complete!")