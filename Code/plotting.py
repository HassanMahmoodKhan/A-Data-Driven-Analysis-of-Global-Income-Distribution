from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("darkgrid")

font = {'family' : 'DejaVu Sans',
        'size'   : 10}
plt.rc('font', **font)

def plot(data, x_axis, y_axis, plot_type, hue, h, w): # Function for plotting data
        if plot_type == "count":
            sns.catplot(data=data, x=x_axis, kind=plot_type, palette='tab10', hue=hue, height=h, aspect=w)
            plt.title(f"{plot_type} plot - {x_axis}")
            plt.xlabel(f"{x_axis}", fontsize=14)
            plt.ylabel("count", fontsize=14)
        if plot_type == "box":
            sns.catplot(data=data, x=x_axis, y=y_axis, kind=plot_type, palette='tab10', height=h, aspect=w)
            plt.title(f"{plot_type} plot - {y_axis} vs {x_axis}")
            plt.xlabel(f"{x_axis}", fontsize=14)
            plt.ylabel(f"{y_axis}", fontsize=14)

def heatmap(data): # Correlation plot showing the correlation between continuous features and the target label
    plt.figure(figsize=(8,6))
    sns.heatmap(data.corr(numeric_only=True), cmap="YlGnBu", annot=True) 

def ConfusionMatrix(model, y_test, predictions): # Function for plotting Confusion Matrix
    cm = confusion_matrix(y_test, predictions, normalize='all')
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["<=50K", ">50k"])
    sns.set_style("white")
    plt.rc('font', size=12)
    disp.plot()
    plt.savefig("../Assets/confusion_matrix_" + model)
    plt.show()
    

def ROCCurve(model, model_probs, y_test):
    ns_probs = [0 for _ in range(len(y_test))] # Function for plotting the ROC Curve and obtaining ROC Score 
    model_probs = model_probs[:, 1]
    ns_auc = roc_auc_score(y_test, ns_probs)
    model_auc = roc_auc_score(y_test,model_probs)
    print('%s ROC AUC=%.3f' % (model,model_auc))
    ns_fpr, ns_tpr, _ = roc_curve(y_test, ns_probs)
    model_fpr, model_tpr,_ = roc_curve(y_test, model_probs)
    plt.figure(figsize=[6,6])
    plt.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill', linewidth=2, color="blue")
    plt.plot(model_fpr, model_tpr, label='%s'%(model), linewidth=2, color="darkorange")
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title("Receiver Operating Characteristic (ROC) Curve - %s" %(model))
    plt.legend(loc="lower right")
    plt.savefig("../Assets/roc_curve_" + model)
    plt.show()
    