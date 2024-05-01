#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
get_ipython().run_line_magic('matplotlib', 'inline')

import warnings
warnings.filterwarnings('ignore')


# In[3]:


df=pd.read_csv('insurance prediction.csv')
df.head()


# In[4]:


df.shape


# In[5]:


df.info()


# In[6]:


plt.figure(figsize=(10,4))
sns.set_style(style='darkgrid', rc={"grid.color": ".8", "grid.linestyle": "--"})
sns.countplot(x=df['region'],palette='viridis',saturation=0.8,edgecolor="black")
plt.tight_layout()
plt.grid(True)
plt.show()
#Applicant from southeast have higher expenses.


# In[7]:


sns.set_style(style='darkgrid', rc={"grid.color": ".4", "grid.linestyle": "--"})
out_df = pd.DataFrame(df.groupby('sex')['sex'].count())
colors = ['#61ad66', '#3b528b']
plt.pie(out_df['sex'], labels=['Female', 'Male'], autopct='%.0f%%', colors=colors, radius=1, explode=(0, 0.1), shadow=True)
plt.show()
#the composition of Gender


# In[8]:


sns.set_style(style='darkgrid', rc={"grid.color": ".8", "grid.linestyle": "--"})
out_df=pd.DataFrame(df.groupby('smoker')['smoker'].count())
colors = ['#ff7f0e', '#1f77b4']
plt.pie(out_df['smoker'],labels=['Non-smoker','Smoker'],autopct='%.0f%%',colors=colors,radius=1,explode = (0, 0.1),shadow=True)
plt.show()
#the composition of smoker


# In[9]:


plt.figure(figsize=(10,4))
sns.set_style(style='darkgrid', rc={"grid.color": ".8", "grid.linestyle": "--"})
sns.scatterplot(y=df['age'],x=df['expenses'],palette='RdYlGn_r')
plt.tight_layout()
plt.grid(False)
plt.show()
#younger applicants have lower expenses and older applicants have higher expenses.


# In[10]:


sns.barplot(x='sex',y='expenses',data=df,palette='magma',saturation=0.5)
plt.tight_layout()
plt.grid(True)
plt.show()
#Males have high expense.


# In[11]:


plt.figure(figsize=(10,4))
sns.set_style(style='darkgrid', rc={"grid.color": ".8", "grid.linestyle": "--"})
sns.regplot(x=df['bmi'],y=df['expenses'])
plt.tight_layout()
plt.grid(True)
plt.show()
#There is a slight positive relation between BMI and expense


# In[12]:


sns.barplot(x='children',y='expenses',data=df,palette='inferno',saturation=0.9)
plt.tight_layout()
plt.grid(True)
plt.show()
#Applicant with 2 or 3 children have higher expenses


# In[13]:


sns.barplot(x='smoker',y='expenses',data=df,palette='coolwarm',saturation=0.9)
plt.tight_layout()
plt.grid(True)
plt.show()
#Smokers have high expense.


# In[14]:


def cat2col(data,col):
    dummy=pd.get_dummies(data[col],drop_first=True)
    data.drop(col, axis=1,inplace=True)
    data= pd.concat([data,dummy],axis =1)
    return data


# In[15]:


for i in df.columns:
    if df[i].dtype ==object:
        print(i)
        df =cat2col(df,i)


# In[16]:


df.head()


# In[17]:


X=df.drop('expenses',axis=1)
y=df.expenses
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=12)


# In[18]:


scores=[]
from sklearn.metrics import r2_score,mean_squared_error, mean_absolute_error


# In[19]:


from sklearn.linear_model import LinearRegression

model_lr=LinearRegression()
model_lr.fit(X_train,y_train)
pred_lr=model_lr.predict(X_test)


# In[20]:


plt.figure(figsize=(6,4))
sns.set_style(style='darkgrid', rc={"grid.color": ".8", "grid.linestyle": "--"})
sns.scatterplot(x=pred_lr,y=y_test)
plt.tight_layout()
plt.grid(True)
plt.show()


# In[21]:


from sklearn.tree import DecisionTreeRegressor

model_dt= DecisionTreeRegressor(random_state=12)
model_dt.fit(X_train,y_train)
pred_dt=model_dt.predict(X_test)
scores.append({
        'model': 'Decision Tree',
        'r2_score': r2_score(y_test, pred_dt)*100,
    'MS_score' : mean_squared_error(y_test,pred_dt)
    })

pred=pred_dt

print('R2 Score: ', r2_score(y_test, pred_dt)*100,
      '\nMean squared: ', mean_squared_error(y_test,pred_dt))


# In[22]:


plt.figure(figsize=(6,4))
sns.set_style(style='darkgrid', rc={"grid.color": ".8", "grid.linestyle": "--"})
sns.scatterplot(x=pred,y=y_test)
plt.tight_layout()
plt.grid(False)
plt.show()


# In[23]:


#extreme gradient boosting 
from xgboost import XGBRFRegressor

model_xgb = XGBRFRegressor()
model_xgb.fit(X_train, y_train)
pred_xgb = model_xgb.predict(X_test)


# In[24]:


scores.append({
        'model': 'XGB regressor',
        'r2_score': r2_score(y_test, pred_xgb)*100,
    'MS_score' : mean_squared_error(y_test,pred_xgb)
    })

pred=pred_xgb

print('R2 Score: ', r2_score(y_test, pred_xgb)*100,
      '\nMean squared: ', mean_squared_error(y_test,pred_xgb))


# In[25]:


plt.figure(figsize=(6,4))
sns.set_style(style='darkgrid', rc={"grid.color": ".8", "grid.linestyle": "--"})
sns.scatterplot(x=pred,y=y_test)
plt.tight_layout()
plt.grid(True)
plt.show()


# In[26]:


Score = pd.DataFrame(scores,columns=['model','r2_score','MS_score'])
Score.sort_values('r2_score',ascending=False,inplace=True)
Score
plt.figure(figsize=(10,4))
sns.barplot(y=Score['model'],x=Score['r2_score'],palette='Pastel1',edgecolor="black")
plt.tight_layout()
plt.grid(True)
plt.show()


# In[27]:


# Load the dataset
data = pd.read_csv("insurance prediction.csv") 

# Preprocess the data (one-hot encoding for categorical variables)
data = pd.get_dummies(data)

# Split the data into features (X) and target variable (y)
X = data.drop("expenses", axis=1)  # Replace "target_variable_column_name" with the actual column name
y = data["expenses"]

# Initialize the algorithms
linear_regression = LinearRegression()
decision_tree = DecisionTreeRegressor()
xgboost_regressor = XGBRegressor()

# Perform cross-validation and calculate mean squared error (MSE)
models = [("Linear Regression", linear_regression), ("Decision Tree", decision_tree), ("XGBoost Regressor", xgboost_regressor)]
results = []
for name, model in models:
    scores = cross_val_score(model, X, y, cv=5, scoring="neg_mean_squared_error")
    mse = -np.mean(scores)
    results.append((name, mse))

# Sort the results based on MSE in ascending order
results.sort(key=lambda x: x[1])

# Print the results
for name, mse in results:
    print(f"{name}: Mean Squared Error = {mse}")
    
# Calculate the range of the target variable
y_range = y.max() - y.min()

# Calculate the percentage of MSE
mse_percentage = (mse / y_range) * 100

# Print the MSE percentage
print(f"MSE Percentage: {mse_percentage}%")

# Identify the best model
best_model = results[0][0]
print(f"\nBest Model: {best_model}")


# In[28]:


df.shape


# In[29]:


import pandas as pd
import xgboost as xgb

def cat2col(df, column):
    return pd.concat([df, pd.get_dummies(df[column], prefix=column)], axis=1).drop(column, axis=1)

# Get input values from the user
age = int(input("Enter age: "))
sex = input("Enter sex (male/female): ")
bmi = float(input("Enter BMI: "))
children = int(input("Enter number of children: "))
smoker = input("Are you a smoker (yes/no): ")
region = input("Enter region (northeast/northwest/southeast/southwest): ")

new_data = {
    'age': [age],
    'sex': [sex],
    'bmi': [bmi],
    'children': [children],
    'smoker': [smoker],
    'region': [region]
}

new_df = pd.DataFrame(new_data)

# Apply one-hot encoding to categorical columns
categorical_columns = ['sex', 'smoker', 'region']
for column in categorical_columns:
    new_df = cat2col(new_df, column)

# Ensure that all categorical columns from training data are present in the new data
missing_columns = set(X_train.columns) - set(new_df.columns)
for col in missing_columns:
    new_df[col] = 0  # Add the missing column with value 0

# Reorder the columns to match the training data
new_df = new_df[X_train.columns]

# Make the prediction using XGBRegressor
prediction = model_xgb.predict(new_df.values)

# Display the prediction
print("Medical Expenses:", prediction)


# In[30]:


threshold = 5000  # Set your desired threshold value
prediction = model_dt.predict(new_df)
if prediction >= threshold:
    print("Insurance can be provided.")
else:
    print("Insurance may not be necessary.")


# In[ ]:


import tkinter as tk
from tkinter import messagebox
import pandas as pd
import xgboost as xgb

# Load the dataset and perform necessary preprocessing
df = pd.read_csv('insurance prediction.csv')

def cat2col(data, col):
    dummy = pd.get_dummies(data[col], drop_first=True)
    data.drop(col, axis=1, inplace=True)
    data = pd.concat([data, dummy], axis=1)
    return data

for column in df.columns:
    if df[column].dtype == object:
        df = cat2col(df, column)

X = df.drop('expenses', axis=1)
y = df.expenses

# Train the model
model_xgb = xgb.XGBRegressor()
model_xgb.fit(X, y)

# Create the UI window
window = tk.Tk()
window.title("Insurance Prediction")
window.geometry("400x300")

# Function to predict expenses and insurance
def predict_expenses():
    # Retrieve input values from the UI
    age = int(age_entry.get())
    sex = sex_var.get()
    bmi = float(bmi_entry.get())
    children = int(children_entry.get())
    smoker = smoker_var.get()
    region = region_var.get()

    # Create the new data DataFrame
    new_data = {
        'age': [age],
        'sex': [sex],
        'bmi': [bmi],
        'children': [children],
        'smoker': [smoker],
        'region': [region]
    }
    new_df = pd.DataFrame(new_data)

    # Apply one-hot encoding to categorical columns
    categorical_columns = ['sex', 'smoker', 'region']
    for column in categorical_columns:
        new_df = cat2col(new_df, column)

    # Ensure that all categorical columns from training data are present in the new data
    missing_columns = set(X.columns) - set(new_df.columns)
    for col in missing_columns:
        new_df[col] = 0  # Add the missing column with value 0

    # Reorder the columns to match the training data
    new_df = new_df[X.columns]

    # Make the prediction using XGBRegressor
    prediction = model_xgb.predict(new_df.values)[0]

    # Determine the insurance prediction based on expenses
    insurance_prediction = "Insurance can be provided." if prediction <= 5000 else "Insurance may not be necessary."

    # Display the prediction result
    prediction_text = f"Predicted Expenses: ${prediction:.2f}\n{insurance_prediction}"
    messagebox.showinfo("Prediction Result", prediction_text)

# Create the input labels and entry fields
age_label = tk.Label(window, text="Age:")
age_label.pack()
age_entry = tk.Entry(window)
age_entry.pack()

sex_label = tk.Label(window, text="Sex:")
sex_label.pack()
sex_var = tk.StringVar(window)
sex_var.set("female")
sex_option_menu = tk.OptionMenu(window, sex_var, "female", "male")
sex_option_menu.pack()

bmi_label = tk.Label(window, text="BMI:")
bmi_label.pack()
bmi_entry = tk.Entry(window)
bmi_entry.pack()

children_label = tk.Label(window, text="Children:")
children_label.pack()
children_entry = tk.Entry(window)
children_entry.pack()

smoker_label = tk.Label(window, text="Smoker:")
smoker_label.pack()
smoker_var = tk.StringVar(window)
smoker_var.set("no")
smoker_option_menu = tk.OptionMenu(window, smoker_var, "no", "yes")
smoker_option_menu.pack()

region_label = tk.Label(window, text="Region:")
region_label.pack()
region_var = tk.StringVar(window)
region_var.set("southeast")
region_option_menu = tk.OptionMenu(window, region_var, "northeast", "northwest", "southeast", "southwest")
region_option_menu.pack()

predict_button = tk.Button(window, text="Predict Expenses", command=predict_expenses)
predict_button.pack()

window.mainloop()


# In[ ]:




