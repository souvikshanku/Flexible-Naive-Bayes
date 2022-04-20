# Importing Libraries
import os
os.makedirs('./results', exist_ok=True)

## library for parallel execution
from concurrent.futures import ThreadPoolExecutor

## library for data analysis and computation
import numpy as np
import pandas as pd

## library for model building
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split

## datasets
from sklearn.datasets import load_breast_cancer as breast_cancer
from sklearn.datasets import load_iris as iris
from sklearn.datasets import load_wine as wine

datasets = {"breast_cancer": breast_cancer, "iris": iris, "wine": wine}

## Flexible Naive Bayes Algorithm
from FlexibleNB import FlexibleNB

## Progress Bar
from tqdm import tqdm


# Defining Necessary Functions
def result_with_data(X, y, test_size=0.25):
    """
    Splits the data in training set and test set and
    applies naive bayes classifier as well as flexible
    bayes classifier with gaussian kernel and 4 bandwidth
    selection methods: 'scott', 'silverman', 'cv_ml', 'cv_ls'

    X: Input variables
    y: response variable
    test_size: proportion of test data (default 0.25)
    """

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    methods = ["scott", "silverman", "cv_ml", "cv_ls"]
    results = pd.DataFrame(
        columns=["Train Accuracy", "Test Accuracy"], index=["gaussian"] + methods
    )
    model = GaussianNB()
    model.fit(X_train, y_train)
    results.loc["gaussian", "Train Accuracy"] = model.score(X_train, y_train)
    results.loc["gaussian", "Test Accuracy"] = model.score(X_test, y_test)
    for method in methods:
        model = FlexibleNB(method, "c" * X.shape[1])
        model.fit(X_train, y_train)
        results.loc[method, "Train Accuracy"] = model.score(X_train, y_train)
        results.loc[method, "Test Accuracy"] = model.score(X_test, y_test)
    return results


def simulate_normal(sim_num, progress_bar=None):
    """
    Simulates 'sim_num' independent gaussian samples
    and output the results in a csv file

    sim_num: Number of Simulations to perform
    progress_bar: (optional) tqdm object
    """
    # C = [i for i in range(5)]
    try:
        sample_size = [100, 110, 95, 101, 99]
        cov = np.identity(6)
        X = []
        y = []
        for i in range(len(sample_size)):
            mean = [i] * 6
            sample = list(np.random.multivariate_normal(mean, cov, sample_size[i]))
            X += sample
            y += [i] * sample_size[i]
        X = np.array(X)
        y = np.array(y)
        # print("Normal Population")
        result = result_with_data(X, y, 0.25)
        result.to_csv(f"./results/normal-{sim_num:04}.csv")
        if progress_bar:
            progress_bar.update()
    except Exception as e:
        print(e)


def simulate_non_normal(sim_num, progress_bar):
    """
    Simulates 'sim_num' independent laplace samples
    and output the results in a csv file

    sim_num: Number of Simulations to perform
    progress_bar: (optional) tqdm object
    """
    # C = [i for i in range(5)]
    sample_size = [100, 110, 95, 101, 99]
    X = []
    y = []
    for i in range(len(sample_size)):
        sample = list(np.random.laplace(loc=i, scale=1, size=(sample_size[i], 6)))
        X += sample
        y += [i] * sample_size[i]
    X = np.array(X)
    y = np.array(y)
    # print("Non-Normal Population")
    result = result_with_data(X, y, 0.25)
    result.to_csv(f"./results/non-normal-{sim_num:04}.csv")
    if progress_bar:
        progress_bar.update()


# Simulating from Gaussian Distribution
with tqdm(total=200) as progress_bar:
    with ThreadPoolExecutor() as executor:
        executor.map(lambda i: simulate_normal(i, progress_bar), range(1, 201))


# Simulating from Laplace Distribution
with tqdm(total=200) as progress_bar:
    with ThreadPoolExecutor() as executor:
        executor.map(lambda i: simulate_non_normal(i, progress_bar), range(1, 201))


# Combining Results
normal_result = np.zeros((5, 2, 200))
for i in range(1, 151):
    df = pd.read_csv(f"./results/normal-{i:04}.csv", index_col=0)
    df = df.to_numpy()
    normal_result[:, :, i - 1] = df

non_normal_result = np.zeros((5, 2, 200))
for i in range(1, 151):
    df = pd.read_csv(f"./results/non-normal-{i:04}.csv", index_col=0)
    df = df.to_numpy()
    non_normal_result[:, :, i - 1] = df

print(normal_result.mean(axis=-1))
# array([[0.83458554, 0.80992126],
#        [0.99123457, 0.74892388],
#        [0.99564374, 0.7416273 ],
#        [0.97114638, 0.76587927],
#        [0.98326279, 0.72734908]])

print(non_normal_result.mean(axis=-1))
# array([[0.73479718, 0.68834646],
#        [0.97151675, 0.66671916],
#        [0.98162257, 0.6616273 ],
#        [0.93737213, 0.67191601],
#        [0.98740741, 0.61475066]])

# Real life data analysis:
# Datasets Used: Iris, Wisconsin Breast Cancer, Wine

for name, load_data in datasets.items():
    print(name)
    X, y = load_data(return_X_y=True)
    print(result_with_data(X, y, 0.25))

# breast_cancer
#                  Train Accuracy Test Accuracy
# gaussian               0.934272       0.93007
# scott                  0.995305      0.937063
# silverman              0.997653      0.951049
# normal_reference       0.946009      0.923077
# cv_ml                  0.976526      0.944056
# cv_ls                       1.0       0.41958
# iris
#                  Train Accuracy Test Accuracy
# gaussian               0.946429           1.0
# scott                       1.0           1.0
# silverman                   1.0           1.0
# normal_reference       0.982143      0.973684
# cv_ml                  0.991071      0.973684
# cv_ls                       1.0      0.789474
# wine
#                  Train Accuracy Test Accuracy
# gaussian               0.992481      0.933333
# scott                       1.0           1.0
# silverman                   1.0           1.0
# normal_reference            1.0           1.0
# cv_ml                       1.0      0.977778
# cv_ls                  0.984962      0.644444
