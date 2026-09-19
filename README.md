# Student_Performance

This project uses machine learning to predict students' final grades based on academic and demographic features from a Portuguese school system.

A key focus of the analysis is comparing model performance **with and without first- and second-term grades (`G1` and `G2`)**. While prior grades are likely to improve prediction accuracy, excluding them allows the model to assess whether students could potentially be identified for academic intervention earlier in the school year.

## Project Workflow

The analysis includes:

* Initial data inspection and exploratory data analysis
* Selection of numeric, categorical, and ordinal features
* Custom feature transformation
* Separate preprocessing pipelines for different feature types
* Comparison of Linear Regression, Support Vector Regression, and Lasso Regression
* Cross-validation using RMSE
* Hyperparameter tuning using `GridSearchCV`
* Final evaluation on held-out test data using RMSE and R²

## Data Preparation

Selected features were divided into three groups:

**Numeric**

* `age`
* `absences_G1`
* `absences_G2`
* `absences_G3`
* `G1`
* `G2`

**Categorical**

* `higher`

**Ordinal**

* `goout`

A custom scikit-learn transformer was created to:

* Combine the three absence variables into a single `absences_sum` feature
* Optionally remove `G1` and `G2` so models could be evaluated both with and without prior-term grades

Preprocessing pipelines handle missing values, categorical and ordinal encoding, and feature scaling.

## Model Development

Three regression algorithms were compared using three-fold cross-validation:

* Linear Regression
* Support Vector Regression (SVR)
* Lasso Regression

Each model was evaluated twice: once with `G1` and `G2` included and once with those features excluded.

Support Vector Regression was subsequently fine-tuned using `GridSearchCV`, with separate searches performed for the two feature sets.

## Results

Final model performance on the test set:

| Model Inputs      | RMSE |   R² |
| ----------------- | ---: | ---: |
| With G1 and G2    | 2.18 | 0.77 |
| Without G1 and G2 | 4.39 | 0.06 |

Including prior-term grades substantially improved predictive performance. The model containing `G1` and `G2` explained approximately 77% of the variation in final grades, compared with approximately 6% when those grades were excluded.

The results demonstrate an important tradeoff: prior academic performance provides substantial predictive value, while attempting to identify students earlier in the academic year using the selected non-grade features results in considerably weaker predictions.

## Tools & Libraries

* Python
* Jupyter Notebook
* pandas
* NumPy
* Matplotlib
* Seaborn
* scikit-learn

## Key Techniques

* Exploratory data analysis
* Feature selection
* Custom scikit-learn transformers
* Preprocessing pipelines
* Column transformers
* Regression modeling
* Cross-validation
* Hyperparameter tuning
* Model evaluation using RMSE and R²

