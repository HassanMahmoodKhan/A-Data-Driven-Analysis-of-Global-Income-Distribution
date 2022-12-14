# Comparative Analysis of Classifier Performance on Adult Census Income Dataset
This project aims to predict the income bracket of individuals based on a variety of features, and presents a holistic comparative analysis between multiple machine learning algorithms through hyperparameter optimization on a binary classification problem.

Using machine learning, the model attempts to predict whether (Y/N) the income of a certain individual, with certain attributes (= features), exceeds $ 50,000 per annum. Three supervised, non-paramteric algorithms have been employed for evaluation i.e., K-nearest Neighbor, Support Vector Machine, & Random Forest.

The Adult Data Set available at the UCI Machine Learning Repository (https://archive.ics.uci.edu/ml/datasets/Adult) is worked with to obtain results. The model is trained with 80% of the dataset and validated on the remaining 20%.

The data set is decribed to have the following characteristics:
- 48842 instances
- 8 categorical attributes and 6 continous
- 3620 instances with missing values

The target variable is as follows:
- income: >50K, <=50K

The feature set is as follows:
- age: continuous.
- workclass: Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked.
- fnlwgt: continuous.
- education: Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th, Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool.
education-num: continuous.
- marital-status: Married-civ-spouse, Divorced, Never-married, Separated, Widowed, Married-spouse-absent, Married-AF-spouse.
- occupation: Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty, Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv, Protective-serv, Armed-Forces.
- relationship: Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried.
- race: White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black.
- sex: Female, Male.
- capital-gain: continuous.
- capital-loss: continuous.
- hours-per-week: continuous.
- native-country: United-States, Cambodia, England, Puerto-Rico, Canada, Germany, Outlying-US(Guam-USVI-etc), India, Japan, Greece, South, China, Cuba, Iran, Honduras, Philippines, Italy, Poland, Jamaica, Vietnam, Mexico, Portugal, Ireland, France, Dominican-Republic, Laos, Ecuador, Taiwan, Haiti, Columbia, Hungary, Guatemala, Nicaragua, Scotland, Thailand, Yugoslavia, El-Salvador, Trinadad&Tobago, Peru, Hong, Holand-Netherlands.
