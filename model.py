import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import joblib

# random seed
seed = 42

# Read original dataset
iris_df= pd.read_csv("data/iris.csv")

# selecting features and target data
X = iris_df[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']]
y= iris_df[['Species']]

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3, random_state=seed,stratify=y)

clf = KNeighborsClassifier(n_neighbors=10)
clf.fit(X_train,y_train)

y_pred= clf.predict(X_test)

accuracy = accuracy_score(y_test,y_pred)

joblib.dump(clf,"output_models/kn_model.sav")