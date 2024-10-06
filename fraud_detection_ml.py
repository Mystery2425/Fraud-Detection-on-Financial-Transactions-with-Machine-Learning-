import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import GridSearchCV

from google.colab import files
files.upload()

!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json
!kaggle datasets download -d mlg-ulb/creditcardfraud
!kaggle datasets download -d mlg-ulb/creditcardfraud

# 3. تحميل البيانات
df = pd.read_csv("creditcard.csv")
# 4. استكشاف البيانات
print("Shape of the dataset:", df.shape)
print(df.head())
print(df.info())

النتيجه
Shape of the dataset: (284807, 31)
   Time        V1        V2        V3        V4        V5        V6        V7  \
0   0.0 -1.359807 -0.072781  2.536347  1.378155 -0.338321  0.462388  0.239599   
1   0.0  1.191857  0.266151  0.166480  0.448154  0.060018 -0.082361 -0.078803   
2   1.0 -1.358354 -1.340163  1.773209  0.379780 -0.503198  1.800499  0.791461   
3   1.0 -0.966272 -0.185226  1.792993 -0.863291 -0.010309  1.247203  0.237609   
4   2.0 -1.158233  0.877737  1.548718  0.403034 -0.407193  0.095921  0.592941   

         V8        V9  ...       V21       V22       V23       V24       V25  \
0  0.098698  0.363787  ... -0.018307  0.277838 -0.110474  0.066928  0.128539   
1  0.085102 -0.255425  ... -0.225775 -0.638672  0.101288 -0.339846  0.167170   
2  0.247676 -1.514654  ...  0.247998  0.771679  0.909412 -0.689281 -0.327642   
3  0.377436 -1.387024  ... -0.108300  0.005274 -0.190321 -1.175575  0.647376   
4 -0.270533  0.817739  ... -0.009431  0.798278 -0.137458  0.141267 -0.206010   

        V26       V27       V28  Amount  Class  
0 -0.189115  0.133558 -0.021053  149.62      0  
1  0.125895 -0.008983  0.014724    2.69      0  
2 -0.139097 -0.055353 -0.059752  378.66      0  
3 -0.221929  0.062723  0.061458  123.50      0  
4  0.502292  0.219422  0.215153   69.99      0  

[5 rows x 31 columns]
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 284807 entries, 0 to 284806
Data columns (total 31 columns):
 #   Column  Non-Null Count   Dtype  
---  ------  --------------   -----  
 0   Time    284807 non-null  float64
 1   V1      284807 non-null  float64
 2   V2      284807 non-null  float64
 3   V3      284807 non-null  float64
 4   V4      284807 non-null  float64
 5   V5      284807 non-null  float64
 6   V6      284807 non-null  float64
 7   V7      284807 non-null  float64
 8   V8      284807 non-null  float64
 9   V9      284807 non-null  float64
 10  V10     284807 non-null  float64
 11  V11     284807 non-null  float64
 12  V12     284807 non-null  float64
 13  V13     284807 non-null  float64
 14  V14     284807 non-null  float64
 15  V15     284807 non-null  float64
 16  V16     284807 non-null  float64
 17  V17     284807 non-null  float64
 18  V18     284807 non-null  float64
 19  V19     284807 non-null  float64
 20  V20     284807 non-null  float64
 21  V21     284807 non-null  float64
 22  V22     284807 non-null  float64
 23  V23     284807 non-null  float64
 24  V24     284807 non-null  float64
 25  V25     284807 non-null  float64
 26  V26     284807 non-null  float64
 27  V27     284807 non-null  float64
 28  V28     284807 non-null  float64
 29  Amount  284807 non-null  float64
 30  Class   284807 non-null  int64  
dtypes: float64(30), int64(1)
memory usage: 67.4 MB

# توزيع الفئات (احتيال مقابل غير احتيال)
sns.countplot(x='Class', data=df)
plt.title('Distribution of Fraudulent and Non-Fraudulent Transactions')
plt.show()

النتيجه

# 5. معالجة البيانات
# تقسيم البيانات إلى متغيرات مستقلة (X) ومستهدفة (y)
X = df.drop('Class', axis=1)
y = df['Class']
# تقسيم البيانات إلى بيانات تدريب واختبار
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
# تحجيم البيانات باستخدام StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
# 6. بناء نماذج التعلم الآلي
# نموذج Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train)

النتيجه
  RandomForestClassifier
RandomForestClassifier(random_state=42)

# 6. بناء نماذج التعلم الآلي
# نموذج Logistic Regression
# You need to import the LogisticRegression class from sklearn.linear_model
from sklearn.linear_model import LogisticRegression
# Create and train the Logistic Regression model
lr_model = LogisticRegression(random_state=42)
lr_model.fit(X_train_scaled, y_train)
# التنبؤ باستخدام Logistic Regression
y_pred_lr = lr_model.predict(X_test_scaled)
# تقييم Logistic Regression
print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred_lr))
print("Logistic Regression Precision:", precision_score(y_test, y_pred_lr))
print("Logistic Regression Recall:", recall_score(y_test, y_pred_lr))
print("Logistic Regression F1-Score:", f1_score(y_test, y_pred_lr))

النتيجه
Logistic Regression Accuracy: 0.9992509626300574
Logistic Regression Precision: 0.8673469387755102
Logistic Regression Recall: 0.625
Logistic Regression F1-Score: 0.7264957264957265

# نموذج XGBoost
xgb_model = XGBClassifier()
xgb_model.fit(X_train_scaled, y_train)
# التنبؤ باستخدام XGBoost
y_pred_xgb = xgb_model.predict(X_test_scaled)
# تقييم XGBoost
print("XGBoost Accuracy:", accuracy_score(y_test, y_pred_xgb))
print("XGBoost Precision:", precision_score(y_test, y_pred_xgb))
print("XGBoost Recall:", recall_score(y_test, y_pred_xgb))
print("XGBoost F1-Score:", f1_score(y_test, y_pred_xgb))

النتيجه
XGBoost Accuracy: 0.9996137776061234
XGBoost Precision: 0.9327731092436975
XGBoost Recall: 0.8161764705882353
XGBoost F1-Score: 0.8705882352941177

# 7. مصفوفة الالتباس لـ Random Forest
# Import the necessary libraries
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Assuming you have X_train_scaled and y_train from previous steps
# Create and train a RandomForestClassifier
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train_scaled, y_train)
# Make predictions using the trained model
y_pred_rf = rf_model.predict(X_test_scaled)
# Now you can generate the confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred_rf)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix for Random Forest')
plt.ylabel('Actual Class')
plt.xlabel('Predicted Class')
plt.show()

النتيجه

# 8. تحسين نموذج Random Forest باستخدام GridSearchCV
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
grid_search.fit(X_train_scaled, y_train)

# عرض أفضل المعلمات
print("Best Parameters for Random Forest:", grid_search.best_params_)

# إعادة تدريب النموذج باستخدام أفضل معلمات
best_rf_model = grid_search.best_estimator_
best_rf_model.fit(X_train_scaled, y_train)
y_pred_best_rf = best_rf_model.predict(X_test_scaled)

# تقييم النموذج المحسن
print("Optimized Random Forest Accuracy:", accuracy_score(y_test, y_pred_best_rf))
print("Optimized Random Forest Precision:", precision_score(y_test, y_pred_best_rf))
print("Optimized Random Forest Recall:", recall_score(y_test, y_pred_best_rf))
print("Optimized Random Forest F1-Score:", f1_score(y_test, y_pred_best_rf))

النتيجه
Fitting 5 folds for each of 27 candidates, totalling 135 fits

# You can reduce the time required to perform Grid Search without significantly impacting the model's quality.
# 8. تحسين نموذج Random Forest باستخدام GridSearchCV
param_grid = {
    'n_estimators': [100, 200],  # تقليل العدد إلى 100 و 200
    'max_depth': [10, None],     # إبقاء خيارات عمق الشجرة أقل
    'min_samples_split': [2, 5]   # تقليل العدد هنا أيضاً
}

grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
grid_search.fit(X_train_scaled, y_train)
# عرض أفضل المعلمات
print("Best Parameters for Random Forest:", grid_search.best_params_)

النتيجه
Fitting 3 folds for each of 8 candidates, totalling 24 fits
Best Parameters for Random Forest: {'max_depth': None, 'min_samples_split': 5, 'n_estimators': 200}

# Retrain the model using the best parameters
best_rf_model = grid_search.best_estimator_
best_rf_model.fit(X_train_scaled, y_train)
# Make predictions
y_pred_best_rf = best_rf_model.predict(X_test_scaled)
# Evaluate the model
print("Optimized Random Forest Accuracy:", accuracy_score(y_test, y_pred_best_rf))
print("Optimized Random Forest Precision:", precision_score(y_test, y_pred_best_rf))
print("Optimized Random Forest Recall:", recall_score(y_test, y_pred_best_rf))
print("Optimized Random Forest F1-Score:", f1_score(y_test, y_pred_best_rf))

النتيجه
Optimized Random Forest Accuracy: 0.9996254813150287
Optimized Random Forest Precision: 0.940677966101695
Optimized Random Forest Recall: 0.8161764705882353
Optimized Random Forest F1-Score: 0.8740157480314961






