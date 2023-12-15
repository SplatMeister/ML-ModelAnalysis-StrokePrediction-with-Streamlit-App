# ML-ModelAnalysis-StrokePrediction-with-Streamlit-App
In terms of having imbalanced data, in a data set the target variable consists of different and multiple classes. When a specific class variable has a significantly lower representation than the majority class, it can lead to an imbalance in the distribution of classes.
Assessing Machine Learning Model Performance on Imbalanced Datasets

In terms of having imbalanced data, in a data set the target variable consists of different and multiple classes. When a specific class variable has a significantly lower representation than the majority class, it can lead to an imbalance in the distribution of classes. When there is a large disparity among the classes, the predictions may not provide the best performance. If the data is imbalanced, the machine learning model will predict majority class more frequently as the representation of the minority class is comparatively low. 
Simply explaining the use of metrics such as precision, recall, F1 score, and AUC-ROC are used to evaluate the performance of machine learning models. Each of these metrics provides unique insights into the performance of a specific model. These metrics are derived from a confusion matrix, where the evaluation of the model gives the counts of the true positives, true negatives, false positives, and false negatives given by the model. Based on these summary counts the following metrics are calculated.
Precision 
This measures the true positives among all other positive predictions. The formula is as follows.
Precision=  TP/((TP+FP))
In relation to imbalanced data sets, precision may be a high value if the model predicts very few positive instances, and this might not be useful. For instance, in a bank transaction data set and a model that detects fraudulent transactions. Where, only 1% of the transactions are fraudulent and the remaining is legitimate. However, it misses 20 actual fraudulent transactions. Based on this, the model makes predictions on 100 as fraud. Where, out of the 100 predicted, 90 are fraudulent and 10 are not. Therefore, the precision is 90%. Where if the model predicts a transaction as fraudulent it is correct at 90%. However, the precision may be high, the model missed 20 fraudulent transactions. Where, the model was unable to capture actual fraud cases. Therefore, it is not informative enough for imbalanced data sets. For the given fraud detection model, the objective is to capture many fraudulent transactions. Therefore, in such cases of imbalanced data sets precision only may not help evaluate in certain circumstances. 
Recall
Recall measures the true positive predictions among all actual positive instances. 
Recall=  TP/((TP+FN))
In relation to imbalanced data sets. For instance, if a model is created to detect heart disease from 10,000 patients and 200 only have heart disease. The model predicts that 150 have heart disease and out of 150 the model correctly identifies 130 patients who have heart disease. However, the model incorrectly classifies 20 without head disease and the model misses 70 actual heart disease cases. Based on that the recall is 65% and the model can identify 65% of all patients who have heart disease. But the model misses 35% of actual cases of heart disease. However, recall is important in imbalanced data sets and the model can capture rare positive cases. 
F1 Score
F1 score represents the balance between recall and precision. In other words, this helps to evaluate both other measurements. 
F1 Score=2 (recall ×precision)/(recall+precision)
In relation to a spam email classifier with an imbalanced data set and 10% of the emails are actual spam, and the remaining are legitimate emails. The model predicted that 100 emails as spam, out of that 80 are actual spam and 20 are not and 20 are misses 20. Based on the F1 score is 80% this signifies that the model can identify both spam and legitimate emails. This is a good measurement as it strikes a good balance between recall and precision. 
AUC-ROC
The area under the Receiver Operating Characteristic Curve measure, the model’s ability to distinguish between positive and negative classes at a probability threshold. This is a good measurement for imbalanced data sets. For instance, credit scoring data set the good applicants out way the bad applicants. A high AUC-ROC score results in a good score, as the model can distinguish between good and bad despite the class imbalance. 
In conclusion, imbalanced data sets and measuring the performance purely depends on the type of the data set and the objective of it. As discussed above there are a few other examples in relation to imbalanced data sets. Where identifying rare disease, precision more useful to evaluate. Then for anomaly detection in an imbalanced data set, recall is important. Thereafter, fraud detection F1 score is important and in relation to ranking job applications AUC-ROC can be useful in an imbalanced dataset. 




1.2 The Impact of Dimensionality Reduction on Model Performance

Dimensionality reduction techniques are considered simply as reducing the number of variables in each dataset and ensuring that essential information is not removed. Dimensionality reduction techniques help to simply a data set which has high dimensions. In relation to dimensionality reduction, there are several techniques, which includes principal component analysis (PCA), linear discriminant analysis and other methods. Applying these methods does affect the accuracy, training time and interpretability of various classification or regression models. 
Accuracy
Accuracy can be simply explained as the proportion of correctly classified from the total dataset as a percentage. In terms of dimensionality, reduction does improve the accuracy of the model. This is by reducing the number of attributes, where the irrelevant information is removed. This is useful in high dimensional data sets. However, there are some instances where valuable information may be removed. 
Training Time
Naturally after performing dimensionality reduction reduce the time that takes to train the mode as fewer attributes result in lesser calculations. High dimensionality data sets will take a longer period to perform. Therefore, training time is drastically reduced after performing dimensionality reduction on data sets. 


Interpretability
Interpretability can be explained as the ability to understand the reasoning for the predictions which are made by the machine learning model. Therefore, dimensionality reduction can improve the interpretability of models. This helps to better understand the relationships and impact of each feature on the model’s prediction.
Regularization techniques
These techniques are used to prevent models from overfitting and improve their performance. There are several regularization techniques that are used. 
Lasso Regularization
Least Absolute Shrinkage and Selection Operator users sparse feature weights by driving some features to zero. For instance, in a dataset that has 1000 features and after applying PCA and reducing the features to 100. Thereafter, applying Lasso regularization, some of the principal component weights to zero. Lasso regularization will help the model’s complexity and reduce the risk of overfitting. 
Ridge Regularization
In this method shrinks all feature weights towards zero. However, it does not perform feature selection. In combination with performing the dimensionality reduction can help reduce overfitting and improve models’ stability.



Elastic Net Regularization
This method combines both Lasso & Ridge regularization. This advantages in handling diverse types of datasets in providing flexibility. This is useful where its unclear which features are important and by setting some coefficients to zero. 
In conclusion, dimensionality reduction techniques play a significant role in improving the model's performance and have a positive relationship with accuracy, training time, interpretability, and regularization techniques. 
1.3 Comparing Bagging, Boosting, and Stacking: Principles, Construction, Diversity Handling, and Impact on Model Performance in Ensemble Learning

Ensemble learning is used in machine learning to improve performance by combining predictions from multiple models and improving the model's accuracy. Under ensemble learning there are several techniques, which includes Bagging, boosting, and stacking. These are the most prominent ones that are used. 
Bootstrap aggregation (bagging)
This method helps to decrease the variance and helps to avoid overfitting. This takes place when input data is divided into multiple number groups. Thereafter, each of the groups are given into the same machine learning model. The output or the prediction of each model is stored and combined. Then use voting or averaging the results for each model prediction. The construction of this ensemble is that multiple base learners are trained on different subsets of the data. Therefore, since it is trained on different data subsets, they are considered diverse. The model performance improves the model’s stability and accuracy. For instance, if a dataset has 1000 samples, bagging will divide it for number of subsets by randomly selecting data points. Each subset is used to train separate models and the outcome will be taken as an average from all models. 








Boosting 
Boosting is a technique that tries to build a classifier from the weak classifiers. For instance, this technique will start with a model and focus on the misclassified examples the first model misclassified. Thereafter, the process continues repeatedly until it tries to correct the errors and the training data is correctly predicted. In terms of the construction, base learners are trained sequentially based on the samples that were previously misclassified. Boosting helps to classify examples that are challenging to classify. The impact of the model performance, boosting can improve accuracy drastically as it keeps on taking the errors and corrects the errors.






Stacking (Blending)
Stacking involves breaking the training data set into groups and giving it to different models. Based on the problem the model will solve. For instance, if it is a regression problem the models which are used would be logistic regression, ridge regression, lasso regression and if it’s a classification problem it will be decision tree classifier, SVM classifier. Thereafter, each model will make its unique predictions. All these predictions are fed into another model, which is a second later classifier and gives the final prediction. In terms of the construction this involves two models, the first being the diverse base learners and the second is the second level meta model that takes the predictions from the base learners. The diversity of stacking is that it uses diverse types of base models, and the meta learner combines all the predictions from the baser learners. The impact on the model performance has the potential for higher accuracy and can be applied for many problems from classification to regression.













In conclusion, the common characteristics across all three ensemble techniques are used to reduce overfitting and improve the model and improve prediction accuracy in comparison to using single machine learning models. These ensemble techniques are more robust to outliers. However, bagging reduces variance, boosting reduce bias and stacking combine both aspects. These are some of the commonly used ensemble methods used to improve the performance of machine learning models. 
1.4 Comparative Analysis of Gradient Boosting Techniques: GBDT, AdaBoost, XGBoost, LightGBM, and CatBoost

Gradient boosting, adaptive boosting, XG Boost, Light GBM and Cat Boost are all considered to be gradient boosting techniques. This helps to combine multiple based models to create a more accurate predictive model. These are quite different in various aspects. 
Gradient Boosting
Gradient boosting simply is to correct the residuals from the previous model and improve accuracy. To understand how gradient boosting techniques differ in comparison to other algorithms. It is important to understand how the gradient boosting functions. For instance, if there is a data set with data that shows apartment prices based on number of square feet and the first attribute is square feet and the second being the price. As the first step of the initial prediction, the mean of the target variable is calculated. Thereafter the residuals are calculated for each observation, between the actual price and initial prediction. Thereafter, the residuals are trained on a decision tree. The model tries to capture the pattern in the residuals and correct the previous residuals. Thereafter the model is updated and improves the prediction, and the process is repeated. Finally, the prediction is summed up from all the decision trees. This model can accurately predict house prices based on the size. 
Adaptive Boosting (ADA Boost)
ADA Boost typically assigns weights to the data points, and it has a sequential learning process. For instance, for a given data set the model gives equal weights for the training set. Thereafter, AdaBoost trains weak learners and makes predictions. If the model misclassifies, the weights are increased for those to give more prominence in the next iteration. This goes on until the model performs well. Finally, the model combines the predictions from all the weak learners to make a final prediction. 
XG Boost (Extreme Gradient Boosting)
XG Boost is a superior version of gradient boosting. This model uses parallel processing and makes it one of the fastest and most efficient models. The model is known for its efficiency and performance. Hence the reason it is one of the most popular is due to the ability handle large data sets and deliver the best predictions. 
Light GBM (Light Gradient Boosting Machine)
Light GBM is a gradient boosting framework based on decision trees to increase the model's efficiency and reduce memory use. Furthermore, this method uses a histogram approach to find the best splits for the decision trees. Rather than scanning all the data it finds the best splits and creates histogram of features. This is widely used by many for large datasets with high dimensional features. What is striking is that it is efficient and fast for processing. 
Cat Boost
Cat Boost can handle categorical and numerical data types. Therefore, Cat Boost does not require any encoding to convert the data to numerical data. A key feature is that it uses an algorithm named, symmetric weighted quantile sketch that handles missing values to reduce the overfitting. Furthermore, it can scale all the columns to the same scaling level. Therefore, improve the overall performance. 
	Gradient Boosting	AdaBoost	XG Boost	Light GBM	Cat Boost
Approach	Iterative gradient descent	Sequential boosting with weighted samples	Regularized gradient	Histogram based	Ordered boosting with decision stumps.
Feature Handling	Encoding Required	Require preprocessing	Handles categorical values and missing values	Handles categorical values.	Handles categorical values and handles missing values.
Efficiency	Moderate	Low	High	Very High	High
Performance Metrics	Accuracy, Precision, Recall, F1-Score, ROC AUC for classification, and MSE, MAE, RMSE, R2 for regression	Accuracy, Precision, Recall, F1-Score, ROC AUC for classification, and MSE, MAE, RMSE, R2 for regression	Accuracy, Precision, Recall, F1-Score, ROC AUC for classification, and MSE, MAE, RMSE, R2 for regression	Accuracy, Precision, Recall, F1-Score, ROC AUC for classification, and MSE, MAE, RMSE, R2 for regression	Accuracy, Precision, Recall, F1-Score, ROC AUC for classification, and MSE, MAE, RMSE, R2 for regression
Strengths	Commonly used.	Focus on misclassified examples	High efficiency, versatility.	Low memory usage.	Efficiency in handing categorical features and missing values.
Weakness	Less efficient	Cannot handle noisy data and sequential training is slow	More memory usage.	Require many data points.	Training time is slow.
Best Use Cases	General machine learning.	For the development of simple models and classification.	Versatile on any data type.	Capability of handling large data sets.	Categorical feature heavy datasets.

Table 1 Comparison of Boosting Techniques
Based on the above given table comparison, the feature handling is different and better than one another. GBM requires one hot encoding and on the other hand AdaBoost can handle categorical features but require additional preprocessing. Light GBM efficiently handles categorical features and missing values. XG Boost supports categorical features and built-in features for missing values. Cat Boost is the best in terms of handing categorical features and does not require any preprocessing and handle missing values. 
In terms of efficiency, GBM is considered efficient. However, in comparison to XG Boost, Light GBM and Cat Boost the computational efficiency is low. XG Boost is highly efficient and is one the fastest gradient boosting libraries. Light GBM is highly efficient and reduces memory usage. Cat Boost can handle large data sets when categorical features are present in the data set.
In conclusion, Light GBM is one of the best for large datasets and Cat Boost is most suited for categorical data handling. XG Boost is highly versatile, and ADA Boost is useful and works better with less complex data. However, Gradient boosting is still the foundation and principle of boosting. 
Section 2

2.1.a Dataset selection and familiarization

The data set that is used is the Stroke Prediction Dataset. This data is predominantly used to predict if a patient is likely to get a stroke based on the attributes of the data set. The data set includes patient information ranging from demographic information, health information and personal or lifestyle information. The data set is used to provide early detection of patients who are prone to a stroke. 
Stroke prediction data set consists 12 variables and some attributes consisting unique values. Based on the overall information exploring the attributes and what each means is important. 
Attribute	Unique Values	Information	Insights
Id	N/A	Patient ID number which is unique.	These values represent all the patients the data is collected and their unique information.
Gender	"Male", "Female" or "Other"	These values represent the gender of the patient.	Apart from Male and female there is other, which could reflect that the user does not want to disclose gender. 
Age 	Between 0.8 to 82	Provides the age of the patient. 	Can categorize the age into groups to understand age brackets. 
Hypertension	0 or 1	‘0’ if the patient does not have hypertension and ‘1’ if the patient has hypertension.	Hypertension refers if a user has high blood pressure or not. This can help understand how high blood pressure can influence stroke.
Heart Disease	0 or 1	‘0’ represents that patient does not have any heart disease and ‘1’ that the patient has heart disease.	Based on these observations, having a heart disease can influence patients of having a stroke.
Ever Married	Yes or No	Understanding if the patient is married or not	This is to determine if the patient is married or single. This maybe to see how being married can influence a stroke. 
Work Type	"Children", "Govt job", "Never worked", "Private" or "Self-employed"	This is to understand the work type the patient belongs to. 	This is to understand how being unemployed may affect strokes in patients.
Residence Type	“Rural” or “Urban”	This is to understand where the users are based from. 	This helps to us understand how the living area may affect.
Average Glucose Level	55 to 271	Average glucose level in blood.	This can help understand how the level of glucose in the blood may influence a patient to having a stroke. 
BMI	10 to 97	Body Mass index. A patient’s body weight in relation to their height.	The BMI of a user will help uncover patterns have the BMI correlates with the patient’s probability of having a stroke. 
Smoking Status	"formerly smoked", "never smoked", "smokes" or "Unknown"	The status of the patient smoking habits.	This information demonstrates how a patient smoking habit are. 
Stroke	0 or 1	‘0’ if a patient didn’t have a store and ‘1’ if the patient had a stroke.	This will be the target variable to understand if the patient had a stroke or not based on the other features. 

Table 2 Data Set Information








2.1.b Data preprocessing and cleaning

After exploring the data set features and the unique values, there is a deeper understanding of the data set and what attributes are included. After using Panda’s library, the data frame is loaded as a data frame. Based on the familiarization on the previous step it was evident that the first column ‘id’ is not important and has unique values for each patient. Therefore, the specific column is dropped from the original data frame. 
Sweetviz

Using quick EDA libraries similar to pandas profiling, one of the prominent libraries is Sweetviz. This library is useful for EDA and understanding relationships of the data frame. The library has easy and relatable visuals that is summarized and given for better understanding data.  
Figure 4 Sweetviz Report
The report is equipped in providing missing values at 4% from the total data frame. At a glance a user can identify if there are any missing values for each corresponding column. In relation to the data qualities the bmi has missing values. Furthermore, each attribute and its unique values are shown on a distribution. In relation to the right-side pane there is a heat map that shows the relationships for each attribute. What is significant is that the data types are shows in different shapes including in squares and circles. 
2.1.b.1 Missing Values

In order to carry out any exploratory data analysis or machine learning models, the data needs to be checked for missing values. It was evident that there are 201 missing values in the data set. Thereafter, checking which attributes represent these missing values, ‘bmi’ column contributed all the missing values. 
 
Figure 5 Missing Values
 
Figure 6 Heat Map of Missing Values

The above missing values are only attributed to BMI and using general methods such as median or mode to replace these missing values or even dropping them may not provide the most optimal result. Therefore, further investigation on the whole data set will provide on how the missing values can be better handled. To handle the missing values, a simple linear regression model is proposed to fill these missing values. Prior to applying the linear regression model, it is important to see what attributes are correlated with age.
 
Figure 7 Heatmap of Attributes
The above heatmap the age attribute has the highest correlation of 0.33 among all the other attributes and based on that it is evident that Age has a positive correlation again BMI. Therefore, age the column is selected to develop a linear regression model to fill the missing values. 
Figure 8 Box Plot for Age Column
Based on the box plot for the age column the data is visible with in the maximum value of 82 and does not represent any outliers. This is clearly visible and in comparison, to the rest of the attributes. Since the data is distributed within the chosen criteria, a linear regression model is applied. In order to perform a linear regression scikit-learn library is used and linear regression module. 
 
Figure 9 Linear Regression Age vs BMI

After preforming a linear regression model and filling the missing data points the distribution is as follows.
 
Figure 10 Distribution of Data after Filling Missing Values

The above plot visualizes the original and the imputed ‘bmi’ values after using a learner regression. The distribution is similar to the original BMI distribution. Therefore, it is safe to say that the data points have been filled appropriately rather than dropping any data points and making the data set more effective. After running ‘isnull’ function on the data frame its mentioned that there are no missing values after using leaner regression imputation on the data set. 






2.1.b.2 Encoding Categorical Data

Thereafter the data types of each column are represented through the ‘dtypes’ function.
 
Figure 11 Data Types
Based on the above figure, some of the attributes consist ‘object’ data types. Therefore, encoding is important prior to applying any machine learning models. The following encoding methods are used for object type data types.
Attribute	Encoding Type
Gender	One Hot Encoding
Ever Married	Label Encoding
Work Type	One Hot Encoding
Residence Type	Binary Encoding
Smoking Status	One Hot Encoding

Table 3 Encoding Categorical Values
The first column is gender and based on the initial familiarization stage there were only three genders in the data set. Therefore, in order to encode the data one hot encoding is used since it has three distinct categories with no inherit order. For ‘ever married’ column label encoding is used, since the column contains unique values and has no inherent order between the categories. The work type column consists a list of types of work which has no inherit order therefore one hot encoding is used. In relation to residence type the binary encoding is used since for the nominal data where there is no inherit order among the categories this method can be used to encode the data. For the smoking status column, since there is no ordinal relationship between these categories one hot encoding is used for this specific column.
2.1.b.3 Scaling Data

Thereafter, as the next important step the remaining numeric columns requires to be normalized. The columns age, average glucose level and bmi are normalized and are in the same scale which is important for machine learning algorithms. As a result of this it will improve the model’s performance and also improve visualization. The following bar chart represent the distribution of the numeric columns, where the data normalized using z score scaling.
 
Figure 12 Before and After Scaling Data Distribution


2.1.b.4 Handling Outliers

As the final step of pre processing understanding if there are any outliers in the data set. In order to visually represent if there are any outliers box plots are used. 
 
Figure 13 Box Plot for Outliers
The above box plot represents all the outliers which are detected in the data. It is evident that there are few outliers on the bmi attribute. In order to remove these outliers z score method is used with a threshold value of 3 to remove any outliers from the data frame. After performing the z score method to identify and remove the outliers the following visual visualize how efficiently it has removed the outliers.
 
Figure 14 Box Plot for After removing Outliers

2.1.c Exploratory Data Analysis

After preprocessing and data cleaning carried out an exploratory data analysis can be carried out. Since the target variable is stroke attribute, the unique values are visualized to understand the distribution of the target variable. 
 






Based on the above-mentioned bar plot, there is drastic imbalance in the data for the stroke column and the data requires to be balanced. Furthermore, the pie chart represents the distribution of large portion of patients are males and followed by females. 
Based on the initial observation the stroke column can be replicated against the remaining attributes. 
 
Figure 17 Stroke vs Other Attributes

In relation to the above given visualization, it is visible how having a stroke and not having stroke has differed based on the unique values. At first glance the number of patients who have a higher age are prone to have a stroke. Secondly higher the glucose level and hypertension and having a heart disease has a higher possibility of having a stroke as well. For married patients having a stroke is higher based on the patient that is provided. Based on the employment type and being a self-employed patient there is a higher possibility of having a stroke. In reference to the location there is slight number of patients who live in urban areas have a stroke. Finally, patients who formally smoked have a higher count of having a stroke.
 
Figure 18 Pair Plot
Based on the previous observations it clear higher number of patients who have a stroke are who have a higher age value. In relation to the age being above 40 and glucose level higher than 150 have a large portion of stroke patients. Furthermore, a higher body mass index and higher glucose level represents patients have strokes. One interesting find is that having a low body mass index between 20 and 40 a prone to having a stroke. 


 








The above suburst takes the body mass index as the value and inner node starts with 'gender', 'ever_married', 'work_type','Residence_type', 'smoking_status'. The color represents the stroke column and blue being not had a stroke and yellow represents had a stroke. Based on the observation and the yellow highlighted spaces, a considerable amount of patients who formrly smoked have a stroke. Specifically, who are male, married, working at a private company, living in urban and formelly smoked. In relation to females who are married, working in government jobs, living in rural areas have a portion who have had a stroke and formerly smoked. The insight is that formerly smoked patients have a higher number of strokes. 
 
Figure 20  Scatter plot Age vs BMI
The above scatter plot depicts the total patient distribution by age and bmi and the color in yellow represents had a stroke and blue not had a stroke. Based on the observation, most of the patients who are age above 45 and body mass index higher than 100 has had a stroke. 
2.1.d Model Selection & Model Evaluation

After performing data preprocessing, where the missing values are highlighted and using a linear regression model to fill the missing values and performing encoding and ensuring the data is properly cleansed. 
2.1.d.1 Data Imbalance

The data set is predominantly used to predict whether an individual is at risk of having a stoke or not. Therefore, the target variable will be the ‘stroke’ column and using the feature columns age, gender etc. Where the model will learn how to classify patients into one of the two categories based on the features. Either having a stroke = 1 and not having a stroke as 0. Therefore, in conclusion this is a classification problem. Where, the objective is to predict if a patient will have a stroke or not. In relation to regression problem, the goal is to predict a person’s age based on the body mass index. 
However, based on the observation it is evident that the target variable values require to be balanced. This is predominantly due to the reason that if the majority class has a higher proportion of data points, the model will predict based on the majority class. This is an overfitting problem. Therefore, the target variable needs to be balanced. 
 
Figure 21 Data Imbalance using SMOTE
The blue bar plot clearly represents the majority class to be 0 and the minority class is 1. There is a clear data imbalance. Therefore, using SMOTE technique and using a combination of oversampling and under sampling technique as there is a large imbalance. Thereafter, using SMOTE and the green bar chart show that the 0 values are brought down from 486 to 1460 data points and the 1 value from 249 to 1460.
Thereafter 5 machine learning models are selected and using Scikit-learn library the preprocessed and encoded data set is split into training and testing data set (training 80% and testing 20%) and then fitted to the following machine learning models. 
2.1.d.2 Logistic Regression

Since the objective of the stroke prediction data set is to predict if a patient is at risk of having a stroke or not. Logistic regression can handle and interpret the importance of each feature and predict the likelihood of a having a stroke. 
Thereafter the logistic regression classifier is initialized and then fits to the training data  (X_train and y_train) data and allow the model to learn the relationship between the feature in X_train and the corresponding values in Y_train. Thereafter, the accuracy is garnered with a detailed classification report.
 
Figure 22 Logistic Regression Model and Classification Report
Based on the above report, the accuracy of the model is 82%. Which indicates that it correctly predicted 82% of the test data set. Under the performance for each class are 84%, 80% and 82%, which shows that it is well balanced performance and is able to classify both classes. Furthermore, the F1 score of 0.82 represents the effectiveness in making predictions. 
 
Figure 23 Logistic Regression Hyperparameter Tunning
Thereafter, hyperparameter tunning is performed on the logistic regressing using grid search. In order to find the best combination hyperparameters, different regularization parameters are used and solver options. Using 5-fold cross validation the best hyperparameters are selected. The best hyperparameters are ‘c’: 100 and solver: ‘liblinear’. Based on the hyperparameters, the accuracy is 83% and precision, recall and F1-scre for both classes are also balanced with a good model performance. 

2.1.d.3 Decision Tree 

Decision trees are quite useful in handling both numerical and categorical data. This model suitable for classification and regression. With feature importance score, helps identify key predictors of stroke. 
In order to make predictions as the first step a decision tree classifier is created and x train and y rain data is fitted to the model. Thereafter, a performance report is generated and the accuracy is 86% and the precision for both classes 0 and 1 is 87%. In terms of recall for both classes recall is between 86% and 87% and finally the F1-scrore is also between 86% and 87% and shows that there is a good balance. 
 
Figure 24 Decision Tree Model and Classification Report
Thereafter, hyperparameter tunning is performed on the decision tree classifier using grid search. The maximum depth, minimum sample split and minimum sample leaf hyperparameter values are included. Based on the grid search with 5-fold cross validation the best hyper parameters are minimum sample leaf is 2 and minimum samples split is 2. Finally, the report provides balanced precision, recall and F1-score for both classes and an accuracy of 86%.
 
Figure 25 Decision Tree Hyperparameter Tunning
2.1.d.4 Random Forest

Random forest results in better predictive models this model provides feature importance scores and identify which features are most influential in predicting stroke. 
After creating a random forest classifier and training the data and preparing the model. After generating the report, it gives an accuracy of 92% where it correctly predicted. Which is a high accuracy score. In relation to the precision for class 0 is 97% and for class 1 is 88%. Although the accuracy is high but based on the two classes it is not close. However, the F1-score is 91% for class 0 and 92% for class 1. 
 
Figure 26Random Forest Model and Classification Report
Thereafter, hyperparameter tunning for the random forest classifier, and defining the parameter grid for the number of trees, the maximum depth of the tree and the minimum number of samples required to split the initial node. By using 5 -folds cross validation the best hyper parameters are 100 for number of estimators and no maximum depth and minimum sample split is 2. However, after performing hyper parameter tunning the accuracy is 92%.
 
Figure 27 Random Forest Hyperparameter Tunning
2.1.d.5 Gradient Boosting

This model is useful to capture complex relationships and with a high predictive accuracy. Most importantly, the model can provide feature importance scores. For instance, hypertension and glucose level the model provides feature importance scores.
Creating a gradient boosting classifier and trained it on the training data. 
 
Figure 28 Gradient Boosting Classifier Model and Classification Report
Thereafter hyperparameter tuning for the gradient boosting classifier is carried out using Grid search. In the parameter grid the number of boosting stages, the learning rate and maximum death is given. Based on the grid search with 5-fold cross validation the best hyperparameters are for learning rate at 0.2, maximum depth at 5 and number of boosting stages at 200. Based on the hyperparameter tunning the accuracy is at 93%. 
 
Figure 29 Gradient Boosting Classifier Hyperparameter Tunning
2.1.d.6 Support Vector Machine (SVM)

This model is most suited as it can classify binary classifications. In order to check the model, a SVM classifier is created and trained on the training data. Based on the report the accuracy is 83%. In terms of precision for both classes the precision is 83%. The recall is between 82% and 83%. Finally, the F1-score is also between 82% and 83%.
 
Figure 30 Support Vector Machine Classifier Model and Classification Report
Based on the above output hyperparameter tunning is performed on SVM model using grid search. The best hyperparameters ‘c’ is 10, and kernel is radial basis function. Therefore, the balanced precision, recall and F1-score for both classes an accuracy of 88% on the test data. The hyperparameter tunning helped the model better perform. 




2.1.e Model Evaluation

After performing creating the five machine learning models and using the training data and evaluating and then performing grid search to hyper tune the models some of the model’s accuracy has increased and improved. 
2.1.e.1 Classifier performance with ROC curve evaluation. 

ROC curve or receiver operating characteristics provides the performance of each model against true positive rate, where the correctly predicted positive instances against the total number of positive instances. Fales positive rate represents the incorrectly predicted positive instances to the total number of actual negative instances in the dataset. 
 
Figure 31 ROC Curve for Classifiers

Based on the above ROC curve Random Forest and gradient boosting classifiers, are close to hugging the top left corner, TPR is close to 1 and FPR is close to 0. This demonstrates that both classifiers are good at distinguishing between positive and negative cases, with very few false positives. Furthermore, the AUC or area under the curve represents overall performance of the model. Based on the highest AUC Random Forest and gradient boosting classifiers are the two best. 
2.1.e.2 Feature Selection

Feature selection is important when there are a large number of attributes on the given data set. Feature selection help by selecting the best features and removing the irrelevant ones from the data set and bring many important efficiencies to the model. The model’s performance will improve as the relevant or important features are extracted and reducing the overall dimensionality and improving the performance. Furthermore, reduce overfitting as irrelevant features can lead to overfitting. In terms of time there a large improvement in training time for the model. 
Scikit Learn has a library for feature selection Recursive Feature Elimination. This library is used to understand what features are important in a given dataset. This works by training the data and ranking the features based on their importance scores. 
 
Figure 32 Recursive Feature Elimination for six Features
The above RFE uses random forest classifier and as an input six features are given to selected. Based on the input, the REF performs based on the feature important and provides six features that are best in comparison to the whole data set. 
However, REF can also predetermine to give the optimal number of features based on the whole data set. Using this method can help identify the important features and disregard the remaining features. 
 
Figure 33 32 Recursive Feature Elimination Ranking
 2.1.e.3 Shapley Additive Explanations

SHAP is used to explain the predictions of machine learning models. This helps to interpret the output of a model by assigning a SHAP value. The SHAP values quantify the contribution of each feature to the model’s prediction. 
Using SHAP library on the logistic regression model, under age the data points are relatively low ages between -5 to -2 highlighted in blue and has a negative impact on the model’s prediction and lower ages tend to decrease the model’s prediction. As the age moves along the x axis the age increase and the model’s prediction increase as the age increase. It is evident that the age feature helps the predication of the model and higher age groups tend to increase predictions and it is a very important feature. 

 
Figure 34 SHAP Summary Plot (Logistic Regression)
 
In relation to the random forest and decision tree classifiers the following summary plots are visualized. 












Based on the above two summary plots for both classifiers, and specifically the age feature. Where lower ages tend to increase the likelihood of predicting to class 0 and the re region or the class 1 indicates that higher ages tend to increase the likelihood of a prediction belonging to class 1. This is also visible in the random forest classifier summary plot. Based on the overall observation the age feature influence differently to the two classes. Class 0 lower ages have a positive impact and for class 1 higher ages have a positive impact. This helps understand better how the age feature influence the model’s prediction. 


2.1.f Model Interpretability, Visualization, and Deployment

2.1.f.1 Saving the Best Model

After completing the above steps and evaluating and improving the hyperparameters and the decision tree was the best model. Therefore, the selected model is saved using ‘pickle’. 
 
Figure 37 Pickle the Best Performing Model
2.1.f.2 Streamlit Development

After saving the model, the streamlit development is commenced and in order to set up the streamlit app. A new Python file is created named ‘app.py’. Prior to loading the model, streamlit and other relevant libraries are imported along with the model. 


 


Thereafter, creation of the app is executed. Firstly, the title is provided for the app and secondly a sidebar is created for users to include their input data. The inputs have options for drop down and numerical inputs as well. After filling out the inputs the user will execute the predict button.
Since the set has encoding some of the categorical variables, the number of columns has increased therefore, the processing of user submitted data is important. When the user inputs the data and clicks the predict button the data is encoded as per the model training data. This is used to process the data and applies to the predictive model and gives the output to the user if has a risk of having a stroke or not based on the provided inputs. 
 
Figure 40 Encoding User Inputs

After executing the above file and saving it. Using the command prompt the app is opened and is able to make predictions. Firstly, I have included my information to see and determine the out puts as well as some information from the dataset. 

Thereafter the model performance is visualized using the Plotly library that showcase the confusion matrix and the model performance by accuracy, precision, recall and F1-Score. These visualizations help better understand the model’s performance and get a better view on how the model is performing. 
This app can be further improved to help individuals who are prone to having a stroke to adjust some of their personal habits. For instance, based on an users input information and the app predicts the patient is prone to having a stroke, the app can provide information on what areas the patient should improve to reduce the risk of having a stroke. 

