# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## Aim:

To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:

1. Hardware – PCs

2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm:

1.Load the Employee.csv dataset and display the first few rows.

2.Check dataset structure and find any missing values.

3.Display the count of employees who left vs stayed.

4.Encode the "salary" column using LabelEncoder to convert it into numeric values.

5.Define features x with selected columns and target y as the "left" column.

6.Split the data into training and testing sets (80% train, 20% test).

7.Create and train a DecisionTreeClassifier model using the training data.

8.Predict the target values using the test data.

9.Evaluate the model’s accuracy using accuracy score.

10.Predict whether a new employee with specific features will leave or not.

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: Amruthavarshini Gopal
RegisterNumber: 212223230013  
*/
import pandas as pd
df=pd.read_csv("Employee.csv")
df.head()
df.isnull().sum()
df.info()
df["left"].value_counts()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
df["salary"]=le.fit_transform(df["salary"])
df.head()
x=df[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident"]]
x.head()
y=df["left"]
y
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
from sklearn.metrics import accuracy_score,confusion_matrix
accuracy=accuracy_score(y_test,y_pred)
accuracy
confusion=confusion_matrix(y_test,y_pred)
confusion
dt.predict([[0.5,0.8,9,260,6,0]])
```

## Output:

### Head of Dataset
![image](https://github.com/user-attachments/assets/232f5341-08e2-44cf-96d0-a005e45fc6ee)

### Null Count
![image](https://github.com/user-attachments/assets/26ae6f0f-d308-44eb-ae91-59879d0d317f)

### Dataset Info
![image](https://github.com/user-attachments/assets/8f589cce-bf05-4557-8fad-c36d3bbb187d)

### Value Count of Feature
![image](https://github.com/user-attachments/assets/8488f479-e839-4f92-81fa-fd324487e705)

### Encoded Data
![image](https://github.com/user-attachments/assets/b4e33d0f-49b7-4612-bad1-ff8602098a0a)

### Dataset after Feature Selection
![image](https://github.com/user-attachments/assets/78b3a26a-c3bb-4ea8-9409-c16e76c32b37)

### Y Value
![image](https://github.com/user-attachments/assets/a3b2af52-6a79-4c3f-9f1b-a8ce50541e94)

### Accuracy
![image](https://github.com/user-attachments/assets/f1c7837f-2b38-4c46-93f4-43f2e5414d41)

### Confusion Matrix
![image](https://github.com/user-attachments/assets/d6069f3a-f4b7-41bf-b198-586a3905bcf3)

### Predicted Values for new data
![image](https://github.com/user-attachments/assets/45e4fb87-89f7-4212-9d56-1138ea0575f7)


## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
