import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
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

#Model 2

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
#plt.show()
