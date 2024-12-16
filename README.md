# Churn Analysis

Goal : To predict whether customer will churn or not in telecom industries
Dataset:- Churn Telecom(23 categorical variables)

majority : miniority class = 7.1 : 2.9 <br>
missing data : about 5% <br>
Evaluation Metric: Accuracy

This dataset is imbalanced dataset where we need to predict two classes "Yes" or "No".
In the dataset, there are 71% majority class which is less than 80% so I decided to go with accuracy as evaluation metric.  
    
- Best Model parameters - {'learning_rate': 0.1, 'max_depth': 4, 'min_child_weight': 5, 'n_estimators': 100, 'subsample': 0.7}       (XGBoost)
- Mean Cross validation score of Best model - 0.9281778380710121 (XGBoost)
- Test score of best model - 0.9317335945151812 (XGBoost)
- Train score of best model - 0.9323162818032666 (XGBoost)


## Acknowledgements

**Link** : https://www.kaggle.com/jpacse/datasets-for-churn-telecom*

## Dataset:-

Churn Telecom(23 categorical variables). There are total 50k+ rows and 58 columns in this dataset.
This dataset is imbalanced dataset where we need to predict two classes "Yes" or "No". In the dataset, there are 71% majority class which is less than 80% so I decided to go with accuracy as evaluation metric.


## Installation

import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

Follow these steps to set up the project:

    1. **Clone the repository**:
      Download the project code to your local machine.
     ```bash
     git clone https://github.com/Nachiketbhai-Prajapati-000513561/Churn_analysis_ML.git

    2. cd Churn_analysis_ML

    3. python -m venv venv
    source venv/bin/activate   # On Windows: venv\Scripts\activate

    4. pip install -r requirements.txt

    5. jupyter notebook



## 🚀 About Me
I'm an aspiring Data Engineer and an ML Engineer


## 🔗 Links
Nachiketbhai-Prajapati-000513561
www.linkedin.com/in/nachiket2007

## 🛠 Skills

Big Data Analytics
Data Modeling
Data Visualization
Data Manipulation
Jupyter
PySpark
Data Ingestion
Data Pipelines
Agile Methodologies
Extract, Transform, Load (ETL)
Azure Key Vault
Azure Logic Apps
Continuous Integration and Continuous Delivery (CI/CD)
Continuous Integration (CI)
Azure Data Lake
Azure Databricks
Azure Data Factory
Leadership
Microsoft Azure


## Optimizations

Algorithm Efficiency:

If there were changes in algorithm logic to reduce complexity or make operations faster, such as replacing nested loops with vectorized operations (e.g., using NumPy or Pandas).
Data Processing:

Optimized data handling by using efficient libraries like Pandas, avoiding redundant computations, or caching results.
Memory Usage:

Implementations to handle large datasets efficiently, such as processing data in chunks or removing unused variables to free up memory.
Code Modularity:

Refactoring large code blocks into reusable functions for better readability and maintenance.
Library Usage:

Replacing manual implementations with optimized library functions (e.g., using sklearn for ML models or NumPy for mathematical operations).
Performance Improvements:

Usage of multiprocessing or parallel processing libraries to speed up execution.
Leveraging JIT compilation techniques (e.g., numba or cython).
Visualization Enhancements:

Optimizing plots by limiting data points, improving readability, or reducing rendering time for complex visualizations.

## Models

**Models Used**

	Basic Algorithms 
* Naive Algorithms
* Logistic Regression
* Decision Tree
* k-Nearest Neighbors
* Support Vector Machine
* Random Forest
* Extra Trees
* Gradient Boosting
* XgBoost
* Stacking Classifiers

	Cost Sensitive Algorithms
*	Logistic Regression
*	Decision Trees
*	Support Vector Machines
*	Random Forest
*	XGBoost
*	Extra Trees
*	Bagging decision tree with under sampling

Data Sampling Algorithms
*	Decision Tree
## License

MIT License

Copyright (c) 2024 Nachiket (Nick)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.


## Deployment

To deploy this project run

```bash
  npm run deploy
```


## Running Tests

To run tests, run the following command

```bash
  npm run test
```

