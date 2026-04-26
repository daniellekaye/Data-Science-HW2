import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
import seaborn as sns
import matplotlib.pyplot as plt

#Phase 1: Data preprocessing
data = load_breast_cancer(as_frame=True)
df = data.frame
cols = df.columns.tolist()
df = df.dropna()
sns.countplot(x='target', data=df)
plt.xlabel('Tumor Type')
plt.ylabel('Count')
plt.xticks(ticks=[0, 1], labels=['Malignant', 'Benign'])
#plt.show()

scaler = StandardScaler()
scaled_data = scaler.fit_transform(df.drop('target', axis=1))
df_scaled = pd.DataFrame(scaled_data, columns=cols[:-1])

#Phase 2: Training
randomState = 42
testSize = 0.2
X_train, X_test, y_train, y_test = train_test_split(df_scaled, df['target'], test_size=testSize, random_state=randomState)

#Model 1: Logistic Regression
maxIterations = 10000
logregmodel = LogisticRegression(max_iter=maxIterations)
logregmodel.fit(X_train, y_train)
logreg_predictions = logregmodel.predict(X_test)
logreg_cm = confusion_matrix(y_test, logreg_predictions)

#Model 2: Classification Tree
modelRandomState = 0
maxFeatures = None
minImpurityDecrease = 0.0
treemodel = DecisionTreeClassifier(max_features=maxFeatures, random_state=modelRandomState, min_impurity_decrease=minImpurityDecrease)
treemodel.fit(X_train, y_train)
treemodel_predictions = treemodel.predict(X_test)
tree_cm = confusion_matrix(y_test, treemodel_predictions)
#Model 3

#Phase 3: Evaluation and visualization
print("Logistic Regression")
logreg_accuracy = logregmodel.score(X_test, y_test)
logreg_train_accuracy = logregmodel.score(X_train, y_train)
print("Test minus Training Accuracy: " + str(logreg_accuracy - logreg_train_accuracy))

logreg_cv_scores = cross_val_score(logregmodel, X_train, y_train, scoring='accuracy', cv=5)
print("Mean CV Score: " + str(logreg_cv_scores.mean()))
print("CV Standard Deviation: " + str(logreg_cv_scores.std()))
lg_cm_disp = ConfusionMatrixDisplay(confusion_matrix=logreg_cm, display_labels=data.target_names)
lg_cm_disp.plot()
lg_cm_disp.ax_.set_title("Logistic Regression Confusion Matrix")
#plt.show()

print("\n")
print("Classification Tree")
tree_accuracy = treemodel.score(X_test, y_test)
tree_train_accuracy = treemodel.score(X_train, y_train)
print("Test minus Training Accuracy: " + str(tree_accuracy - tree_train_accuracy))

tree_cv_scores = cross_val_score(treemodel, X_train, y_train, scoring='accuracy', cv=5)
print("Mean CV Score: " + str(tree_cv_scores.mean()))
print("CV Standard Deviation: " + str(tree_cv_scores.std()))
tree_cm_disp = ConfusionMatrixDisplay(confusion_matrix=tree_cm, display_labels=data.target_names)
tree_cm_disp.plot()
tree_cm_disp.ax_.set_title("Classification Tree Confusion Matrix")
plt.show()
