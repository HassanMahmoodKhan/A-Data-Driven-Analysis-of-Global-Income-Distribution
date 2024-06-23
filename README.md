# A Data-Driven Analysis of Global Income Distribution: Modeling and Analysis
Income disparity is a significant issue that affects the global population at varying levels. Efforts have been put together to curb it and improve the socio-economic fabric of our societies. This project aims to predict the income bracket of individuals based on a variety of features, and presents a holistic comparative analysis between multiple machine learning algorithms through hyperparameter optimization on a binary classification problem.
Using machine learning, the model attempts to predict whether (Y/N) the income of a certain individual, with certain attributes (= features), exceeds $ 50,000 per annum. Three supervised, non-paramteric algorithms have been employed for evaluation i.e., K-nearest Neighbor, Support Vector Machine, & Random Forest.

## Dataset
The Adult Data Set available at the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Adult) is worked with to obtain results. The model is trained with 80% of the dataset and validated on the remaining 20%.

The data set is decribed to have the following characteristics:
- 48842 instances
- 8 categorical attributes and 6 continous
- 3620 instances with missing values
- Target variable : income (>50K, <=50K)

The feature set is as follows:

![Feature set](https://user-images.githubusercontent.com/97694796/226029048-e4e93889-cad8-464c-86a7-2219136b4e6c.png)

## Feature Selection and Engineering
The correlation matrix for the continuous features compared with the target variable is shown below:

![Correlation Matrix](https://user-images.githubusercontent.com/97694796/226029771-515c4011-1e10-49d4-a816-98db5f5128f5.png)

I have dropped the categorical feature ‘education’ from our dataset, since it being the same as 'education-num', with the latter imposing ordinality. Features 'capital-gain' and 'capital-loss' are highly skewed and as such to minimize skewness, I have taken the square root for all instances of these features.

## Data Preprocessing
I have employed the following steps to transform the dataset into a more representative form:
1) Missing Data Imputation
2) Label Encoding
3) One-Hot Encoding
4) Feature Scaling

## Model Building & Training
There are three machine learning algorithms employed for this project:
1) K-Nearest Neighbor
2) Support Vector Machine
3) Random Forest

I have employed stratified k-fold cross validation, which is a variation of the k-fold cross validation technique that ensures each fold has approximately the same percentage of target class samples, thus addressing the dataset imbalance to an extent. In addition, it addresses the key issue of overfitting and promotes model generalization. Furthermore, the performance of a model significantly depends on the values of the model hyperparamters. I have employed the use of GridSearchCV to search all possible combinations of hyperparamter values, to determine optimal values for each of the three models.

## Model Evalution
Each model has been assesed based on these evaluation metrics:
1) Accuracy
2) Confusion Matrix
3) Reciever Characteristics Curve (ROC)

A comparison of predictive accuracy obtained with those in literature is represented in the table below:

![Comparative Analysis](https://user-images.githubusercontent.com/97694796/226034295-de5888e1-4122-4efd-8ef8-503a496e54c1.png)

Random forests classifier is the best performer out of the three classifiers and outputs the highest classification accuracy of 86.70% and an AUC score of 0.917.


