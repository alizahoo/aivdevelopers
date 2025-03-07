#!/usr/bin/env python
# coding: utf-8

# In[104]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve, f1_score, precision_score, recall_score


# In[122]:


df = pd.read_csv("survey_results_public.csv")


# In[106]:


df.shape


# In[107]:


df.dtypes.value_counts()


# In[108]:


df.columns.tolist()


# In[110]:


df.describe(include='all')


# In[111]:


missing_values = df.isnull().sum() / len(df) * 100
missing_values = missing_values[missing_values > 0].sort_values(ascending=False)


# In[112]:


plt.figure(figsize=(10, 5))
sns.barplot(x=missing_values.index[:15], y=missing_values.values[:15])
plt.xticks(rotation=90)
plt.ylabel("Percentage Missing")
plt.title("Top 15 Columns with Missing Data")
plt.show()


# In[113]:


plt.figure(figsize=(8, 4))
sns.countplot(y=df["JobSat"], order=df["JobSat"].value_counts().index, palette="viridis")
plt.title("Distribution of Job Satisfaction")
plt.xlabel("Count")
plt.ylabel("Job Satisfaction Levels")
plt.show()


# In[114]:


plt.figure(figsize=(8, 4))
sns.countplot(y=df["RemoteWork"], hue=df["JobSat"], palette="magma", order=df["RemoteWork"].value_counts().index)
plt.title("Job Satisfaction by Remote Work Status")
plt.xlabel("Count")
plt.ylabel("Remote Work Status")
plt.legend(title="JobSat", bbox_to_anchor=(1, 1))
plt.show()


# In[115]:


plt.figure(figsize=(10, 6))
sns.countplot(y=df["EdLevel"], hue=df["JobSat"], palette="Set2", order=df["EdLevel"].value_counts().index[:10])
plt.title("Job Satisfaction by Education Level")
plt.xlabel("Count")
plt.ylabel("Education Level")
plt.legend(title="JobSat", bbox_to_anchor=(1, 1))
plt.show()


# In[116]:


plt.figure(figsize=(10, 5))
sns.boxplot(x=df["JobSat"], y=df["ConvertedCompYearly"], palette="Spectral")
plt.title("Salary Distribution by Job Satisfaction")
plt.xlabel("Job Satisfaction")
plt.ylabel("Yearly Compensation (Converted)")
plt.yscale("log")
plt.show()


# In[123]:


# Data preprocessing: Select relevant columns
df = df[['ConvertedCompYearly', 'RemoteWork', 'JobSat','EdLevel']].dropna()


# In[138]:


corr_matrix = df.corr()


# In[139]:


# Plot the heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title("Correlation Heatmap of Job Satisfaction Factors")
plt.show()


# In[124]:


# Encoding categorical variables
label_encoders = {}
categorical_features = ["JobSat", "RemoteWork", "EdLevel"]


# In[125]:


for feature in categorical_features:
    le = LabelEncoder()
    df[feature] = le.fit_transform(df[feature].astype(str))
    label_encoders[feature] = le


# In[126]:


# Selecting features and target
features = ["RemoteWork", "EdLevel", "ConvertedCompYearly"]
target = "JobSat"


# In[127]:


# Handling missing salary values by filling with median
df["ConvertedCompYearly"].fillna(df["ConvertedCompYearly"].median(), inplace=True)


# In[128]:


# Splitting the dataset
X = df[features]
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[129]:


# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# In[130]:


# Train a Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)


# In[131]:


# Predictions
y_pred = model.predict(X_test)


# In[133]:


# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy:", accuracy)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# In[134]:


# Train a Logistic Regression model
log_reg_model = LogisticRegression(max_iter=1000, random_state=42)
log_reg_model.fit(X_train, y_train)


# In[135]:


# Predictions using Logistic Regression
y_pred_log = log_reg_model.predict(X_test)


# In[136]:


accuracy_log = accuracy_score(y_test, y_pred_log)
f1_log = f1_score(y_test, y_pred_log, average='weighted')
precision_log = precision_score(y_test, y_pred_log, average='weighted')
recall_log = recall_score(y_test, y_pred_log, average='weighted')


# In[103]:


print("Logistic Regression Model Accuracy:", accuracy_log)
print("F1 Score:", f1_log)
print("Precision:", precision_log)
print("Recall:", recall_log)
print("\nClassification Report:")
print(classification_report(y_test, y_pred_log))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_log))

