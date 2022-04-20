# Flexible Naive Bayes
A comparative study between Naive Bayes Classifier and Flexible Bayes Classifier

---

## Simulation
Data generated from both Gaussian and non Gaussian Distribution and compared the predictive accuracies between Naive Bayes and Flexible Bayes Classifier.

### 1. Gaussian Distribution
Data are generated from a ![equation](https://latex.codecogs.com/svg.image?N_5(\mu,&space;I_n)) distribution where $\mu = (1, 2, 3, 4, 5)'$.

* Number of Classes $=5$
* Class Sizes $ = [100, 110, 95, 101, 99]$

|Classifier|Train Accuracy|Test Accuracy|
|:-:|:-:|:-:|
|Naive Bayes |0.8237|0.8823|
|Flexible Bayes (scott)|0.9801|0.8431|
|Flexible Bayes (Silverman)|0.9845|0.8627|
|Flexible Bayes (CV-LS)|0.9713|0.7647|
|Flexible Bayes (CV-ML)|0.9581|0.8039|

