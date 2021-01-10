
==============================

Breast Cancer Prediction 

Breast cancer is one of the most common types of cancer in the UK. According to NHS, about 1 in 8 women are diagnosed with breast cancer during their lifetime, and early detection will significantly increase the chance of recovery.

Goal: train a machine learning model which predicts whether the cancer is benign or malignant based on a digital scan of the tumour cells.

The dataset used in this story is publicly available and was created by Dr. William H. Wolberg, physician at the University Of Wisconsin Hospital at Madison, Wisconsin, USA. To create the dataset Dr. Wolberg used fluid samples, taken from patients with solid breast masses and an easy-to-use graphical computer program called Xcyt, which is capable of perform the analysis of cytological features based on a digital scan. The program uses a curve-fitting algorithm, to compute ten features from each one of the cells in the sample, than it calculates the mean value, extreme value and standard error of each feature for the image, returning a 30 real-valuated vector.

The mean, standard error and “worst” or largest (mean of the three largest values) of these features were computed for each image, resulting in 30 features. For instance, field 3 is Mean Radius, field 13 is Radius SE, field 23 is Worst Radius.

The final dataset included 9 variables: concave points_worst, radius_se, texture_worst, smoothness_worst, symetry_worst, concave points_se, symetry_mean, franctal_dimension_worst, compactness_se and diagnosis.

Due to the nature of the problem requiring to build a model which can classify the nature of the tumour as benign or malignant, we concluded that the most suitable ML algorithm is a classification one.

Considering that each record in the data was already labelled with the diagnosis, malignant(M) vs benign (B), we decided to experiment with supervised ML algorithms. Classification models used: logistic regression, decision tree, random forest, svm and Adaboost.

The data was split into training data (80% of the entire data set) and test data (20% of the entire data set).


Logistic Regression

" Logistic regression models the probability that a response falls into a specific category. Once trained we may use the confusion matrix to evaluate the classification. The four values are true positive (predicted yes and actually was yes), true negative (predict no and actually was no), false positive (predicted yes and actually was no) and false negative (predicted no and actually was yes)." (Keith Brooks, 2018)



Confusion Matrix

"A confusion matrix is a summary of prediction results on a classification problem. The number of correct and incorrect predictions are summarized with count values and broken down by each class. This is the key to the confusion matrix. The confusion matrix shows the ways in which your classification model is confused when it makes predictions. It gives us insight not only into the errors being made by a classifier but more importantly the types of errors that are being made." (GeeksforGeeks)


![log-regression-confusion-matrix](https://raw.githubusercontent.com/puczilka/Breast-cancer-prediction/master/reports/figures/log-regression-confusion-matrix.png) 


Accuracy & Precision

"Accuracy and precision are two important factors to consider when taking data measurements. Both accuracy and precision reflect how close a measurement is to an actual value, but accuracy reflects how close a measurement is to a known or accepted value, while precision reflects how reproducible measurements are, even if they are far from the accepted value." ( Anne Marie Helmenstine, Ph.D., 2019)


Receiver Operating Characteristic (ROC) curve

The ROC curve is a graphical plot that illustrates the diagnostic ability of a binary classifier system as its discrimination threshold is varied.

We plotted the ROC curce to assess the performce of our logistic regression model as seen below.


![roc-curve](https://raw.githubusercontent.com/puczilka/Breast-cancer-prediction/master/reports/figures/roc-curve.png) 

Best recall and precision: logistic regression
Best AUC: SVM

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
