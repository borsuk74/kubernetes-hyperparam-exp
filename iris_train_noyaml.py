import pprint
import sklearn
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics



def main():
    HYPER_PARAMS_NO_YAML = [{'hyperparam_set': {'max_depth': 5, 'n_estim': 100}},
                            {'hyperparam_set': {'max_depth': 5, 'n_estim': 200}},
                            {'hyperparam_set': {'max_depth': 5, 'n_estim': 300}},
                            {'hyperparam_set': {'max_depth': 8, 'n_estim': 100}},
                            {'hyperparam_set': {'max_depth': 8, 'n_estim': 200}},
                            {'hyperparam_set': {'max_depth': 8, 'n_estim': 300}},
                            {'hyperparam_set': {'max_depth': 10, 'n_estim': 100}},
                            {'hyperparam_set': {'max_depth': 10, 'n_estim': 200}},
                            {'hyperparam_set': {'max_depth': 10, 'n_estim': 300}}]

    job_id = int(os.environ['JOB_ID'])
    DATA_DIR = '/mirror'

    df = pd.read_csv(DATA_DIR + "/iris.data", sep=",")
    df.columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'species']

    X = df[['sepal length', 'sepal width', 'petal length', 'petal width']]  # Features
    y = df['species']  # Labels

    # Split dataset into training set and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)  # 70% training and 30% test

    # Load hyperparameters

    hyperparams = HYPER_PARAMS_NO_YAML[job_id - 1]["hyperparam_set"]
    n_est = hyperparams["n_estim"]
    max_depth = hyperparams['max_depth']

    clf = RandomForestClassifier(n_estimators=n_est, max_depth=max_depth)

    # Train the model using the training sets y_pred=clf.predict(X_test)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    # Model Accuracy, how often is the classifier correct?
    acc = metrics.accuracy_score(y_test, y_pred)
    # print("Accuracy:", metrics.accuracy_score(y_test, y_pred))

    res_df = pd.DataFrame({'accuracy': acc}, index=[0])

    file_name = '/mirror/results_job_id_' + str(job_id) + '.csv'
    res_df.to_csv(file_name, index=False)


if __name__ == '__main__':
    main()