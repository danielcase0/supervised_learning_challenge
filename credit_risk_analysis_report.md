# Module 12 Report Template

## Overview of the Analysis

The purpose of this analysis is to create a logistic regression model as a part of a credit risk evaluation.  Before lending money to a borrower, lenders want to be able to predict the riskiness of the loan.  High-risk loans are more likely to default and defaulted loans are very costly for borrowers.  As a result, there is a strong incentive for lenders to build a predictive model (a credit model) that identifies which borrowers should and should not be given a loan.

Logistic regression is a supervised learning technique that is common in credit models.  Logistic regression uses a series of independent variables to predict the binary classification (e.g., 0 or 1) of the dependent variable.  It is a supervised learning technique because the data set includes historical data about the outcome (dependent variable), which is used to train the model.  For many credit models, the dependent variable is a decision whether or not to fund the loan.  In this exercise, the dependent variable is a classification of the loan as being either healthy (presumably low- or moderate-risk) or high-risk.  The independent variables can either be continuous or binary, and the logistic regression model ultimately assigns coefficients (weights) to each of the independent variables corresponding to their influence on the classification of the dependent variable.

The initial data set includes historical data on the following loan information:

  Independent variables:
  - loan size
  - interest rate
  - borrower income
  - debt-to-income ratio
  - number of accounts
  - derogatory marks
  - total debt

  Dependent variable:
  - loan status

The purpose of this analysis is to create a logistic regression model that uses the independent variables to predict the value of the dependent variable.  If the loan status is `0` then the loan is healthy.  If the loan status is `1` then the loan is high-risk.  For the purpose of this analysis, healthy loans are considered "negatives" because of their assignment as `0` and high-risk loans are considered "positives" because of their assignment as `1`.  The possible model results are as follows:
-True negatives (TN): healthy loans that the model predicts are healthy
-False negatives (FN): high-risk loans that the model predicts are healthy
-True positives (TP): high-risk loans that the model predicts are high-risk
-False positives (FP): healthy loans that the model predicts are high-risk

The analysis is broken up into three parts.  First, a logistic regression model was created using the original data set.  Second, a logistic regression model was created using oversampled data (to create balanced classes) to see if the model performance would improve.  And third, a logisitic regression model was created using both oversampled and scaled data to better understand the relative influence of each of the independent variables on the dependent variable.  The last part wasn't required for the assignment, but I found it instructive.

To make a logistic regression model, the dependent variable (`y`) must first be separated from the independent variables (`X`).  Next, the data set must be split into training (`X_train` and `y_train`) and test data (`X_test` and `y_test`) using random sampling.  The model is built on the training data and then the model predicts what the outcomes should be for the test data (`X_test`).  The predictions (`lr_test_predictions`) are compared to the actual outcomes (`y_test`) and the performance of the model is measured.

The following performance metrics are used in the analysis:
- Accuracy: accuracy is the percent of correct predictions for both classes in the dependent variable.
- Balanced accuracy: balanced accuracy is the percent of correct predictions for both classes in the dependent variable, if the accuracy of each class were given equal weight.
- Precision: for each class, precision is the ratio of loans correctly identified in that class to the total number of loans identified as that class.
  - For example, for healthy loans, the precision of the model is the ratio of healthy loans correctly identified as being healthy (true negatives) to the total number of loans being identified as healthy (true and false negatives).
- Recall: for each class, recall is the ratio of loans correctly identified in that class to the total number of instances of that class in the data set.
  - For example, for healthy loans, the recall of the model is the ratio of healthy loans correctly identified as being healthy (true negatives) to the total number of healthy loans in the data set (true negatives and false positives).

In the second step, a second model is trained using balanced classes to see if the model performance would improve.  Credit risk modeling is a textbook example of a problem involving imbalanced classes, because there are usually many more healthy loans than high-risk ones.  As a result, there are more healthy loans than high-risk loans in the data set, and the model ends up being better at predicting healthy loans because it's ultimately trained on more healthy-loan data if the classes are left imbalanced.  To rectify the imbalance, the smaller class (e.g., high-risk loans) are randomly oversampled until the number of instances in each class is equal.  In the analysis, the purpose of conducting the `value_counts` is to illustrate the imbalanced classes in the original data and the balanced classes in the oversampled data.

Finally, in the last step, a third model is trained using balanced classes and scaled data.  This model is not intended to improve performance over the second model, because the two models are ultimately built on extremely similar data.  The purpose of the third model is to explore the coefficients (i.e., weights) of the logistic regression model to see which columns are the most influential on determining whether or not a loan is considered healthy or high-risk.  Similar coefficients exist in the first and second models as well, but without scaling the data, it's hard to estimate the relative importance of each independent variable.

## Results

A summary of the performance for each logistic regression model is provided below, and more detail can be found in the Jupyter notebook.

Model 1: Logistic Regression model trained on the original data set
- Accuracy: 99.2%
- Balanced accuracy: 95.2%
- Precision (healthy loans): 99.7%
- Precision (high-risk loans): 84.7%
- Recall (healthy loans): 99.4%
- Recall (high-risk loans): 91.0%

Model 2: Logistic Regression model trained on the oversampled data to create balanced classes
- Accuracy: 99.4%
- Balanced accuracy: 99.4%
- Precision (healthy loans): 99.98%
- Precision (high-risk loans): 84.1%
- Recall (healthy loans): 99.4%
- Recall (high-risk loans): 99.4%

Model 3: Logistic Regression model trained on the oversampled (balanced classes) and scaled data
- Accuracy: 99.3%
- Balanced accuracy: 99.3%
- Precision (healthy loans): 99.98%
- Precision (high-risk loans): 83.1%
- Recall (healthy loans): 99.3%
- Recall (high-risk loans): 99.4%

## Summary

In a credit risk situation, it is very important to reduce the number of false negatives identified by the model.  In context, false negatives are high-risk loans that the model has labeled as healthy.  High-risk loans have higher default rates, and defaulted loans are very costly to lenders.  Ideally, the credit model would correctly predict all of the loan types, but in practice, it will always mislabel some of the loans, and the lender has a much higher tolerance for false positives (healthy loans labeled as high-risk) than false negatives (high-risk loans labeled as healthy).  False positives are an example of the lender potentially leaving money on the table by not funding good customers.  False negatives are an example of the lender potentially losing their principal by lending to high-risk borrowers.  These concepts are critical for evaluating the performance of the model in the context of the exercise.

As expected, the first logistic regression model was trained on imbalanced classes (75,036 healthy loans and 2,500 high-risk loans) and the model is much better at predicting which loans are healthy than which loans are high-risk.  Only 0.6% of healthy loans are misidentified (i.e., 99.4% recall for healthy loans), but 9.0% of high-risk loans are misidentified (i.e., 91.0% recall for high-risk loans).  That disparity is reflected in the balanced accuracy score, which is 95.2%, despite the overall accuracy of the model being much higher (99.2%).

The second logistic regression model was trained on balanced classes (56,271 healthy and 56,271 high-risk loans) and the second model performs much better than the first.  Not only is the balanced accuracy much higher (99.4%), but the second model does a much better job of correctly identifying high-risk loans.  Only 0.6% of high-risk loans are mislabeled by the model, which is a significant improvement over the 9.0% mislabeled by the first one.  Additionally, the second model offers similar performance to the first model at identifying healthy loans, which for both models correctly labels the healthy loans 99.4% of the time.

Finally, the third logistic regression model was trained on balanced classes (56,271 healthy and 56,271 high-risk loans) using scaled data.  The purpose of the third model is not to improve on the performance of the second one.  Instead, the purpose is to investigate which independent variables are the most important for predicting whether a loan is healthy or high risk.  The model performance of the third model is comparable to the second, but the coefficients in the third model are much easier to interpret.  The top three most important variables for predicting whether a loan is healthy or high-risk are:
- debt-to-income ratio (4.47)
- loan size (1.58)
- derogatory marks (-1.57)

As one might expect, increases in the debt-to-income ratio of the borrower and the loan size are more likely to indicate that a loan is high-risk.  Counterintuitively, the relationship between derogatory marks and a loan being high-risk is negative, so the model treats derogatory marks as being advantageous (i.e., the more derogatory marks a loan has in the scaled data set, the less likely it is to be high risk).

The best performing models are the second and third ones.  They're essentially the same model with minor variation in how the data is preprocessed.  They perform better than the first model not only because the balanced accuracy is higher, but because they're significantly better at correctly predicting which loans are high-risk.  Obviously, a higher balanced accuracy indicates better performance, but lenders would much rather leave money on the table by mislabeling healthy loans as high-risk (false positives) than lose money by mislabeling high-risk loans as being healthy (false negatives).  Consequently, I recommend using the second or third model, which both provide better performance than the first one.