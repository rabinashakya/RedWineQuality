
# necessary imports
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score



# loading data into dataframe
df = pd.read_csv('../datasets/winequality-red.csv') 
df.info()

# Determine number of records (rows) and features (columns).
df.shape 

# Data describe
df.describe()

# Checking null value
# To detect the missing values in the cells
df.isnull().sum()

# Checking duplicated value
df.duplicated().sum()
df.drop_duplicates(inplace=True)
df.duplicated().sum()

# Checking of unique data
df['quality'].unique()

##### `Data Correlation`
df.corr()

# heatmap
sns.heatmap(df.corr(), annot=True, cmap="Blues",fmt=".2f")


##### `Preprocessing Data`
#Feature Scaling
df['quality'] = df['quality'].apply(lambda x: 1 if x >= 7 else 0)
sns.countplot(data = df, x = 'quality')
plt.xticks([0,1], ['bad wine','good wine'])
plt.title("Types of Wine")
plt.show()

#Resampling Datasets
X = df[['fixed acidity', 'volatile acidity', 'sulphates', 'alcohol', 'density']]
y = df.quality
oversample = SMOTE()
X_ros, y_ros = oversample.fit_resample(X, y)
sns.countplot(x=y_ros)
plt.xticks([0,1], ['bad wine','good wine'])
plt.title("Types of Wine")
plt.show()

# splitting the dependent and independent features.
X = df.drop('quality', axis = 1)
y = df['quality']

# splitting data into training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state=42)

# check the shape of the splited sets
X_train.shape, X_test.shape, y_train.shape, y_test.shape

##### `Evaluation of the model`
def model_evaluation(model, X_train, y_train, X_test, y_test):
    
    # Scaling Process
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Model building process
    mod = model.fit(X_train, y_train)
    mod_pred = model.predict(X_test)
    
    print("Evaluation of the Model")
    print("\n")
    print("Classification report of the Model: \n", classification_report(y_test, mod_pred))
    print("\n")
    print("Prediction of the Model: \n",mod_pred)
    print("\n")
    print("Confusion Matrix of the Model: \n ",confusion_matrix(y_test, mod_pred))
    print("\n")
    print("Accuracy score of the Model: \n", accuracy_score(y_test, mod_pred))
    print("\n")

    return mod


##### `Logistic Regression`
lr = LogisticRegression()
model_evaluation(lr,X_train,y_train, X_test, y_test)

##### `Random Forest`
rfc = RandomForestClassifier()

##### `Decision Tree`
dtc = DecisionTreeClassifier()
model_evaluation(dtc,X_train,y_train, X_test, y_test)