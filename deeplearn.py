'''
Deep Learning Project
Bank Authentication Data Set 
Neural Network
'''

import pandas as pd 
import numpy as np 
import seaborn as sns 
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf 

'''
Get the Data
'''

bank_data = pd.read_csv('bank_note_data.csv')
print(bank_data.head())

'''
Exploratory Data Exploratory
'''
sns.countplot(x = 'Class',data = bank_data )
plt.show()

sns.pairplot(data = bank_data, hue = 'Class', diag_kind= 'histogram')
plt.show()

'''
Data Preparation

Standard Scaling 
'''
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

scaler.fit(bank_data.drop('Class',axis=1))

scaled_features = scaler.fit_transform(bank_data.drop('Class',axis=1))

df_feat = pd.DataFrame(scaled_features,columns=bank_data.columns[:-1])
df_feat.head()


'''
Train Test Split 
'''

X = df_feat
y = bank_data['Class']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

'''
Tensorflow 
'''
image_var = tf.feature_column.numeric_column("Image.Var")
image_skew = tf.feature_column.numeric_column('Image.Skew')
image_curt = tf.feature_column.numeric_column('Image.Curt')
entropy =tf.feature_column.numeric_column('Entropy')

feat_cols = [image_var,image_skew,image_curt,entropy]

classifier = tf.estimator.DNNClassifier(hidden_units=[10, 20, 10], n_classes=2,feature_columns=feat_cols)

input_func = tf.estimator.inputs.pandas_input_fn(x=X_train,y=y_train,batch_size=20,shuffle=True)

classifier.train(input_fn=input_func,steps=500)


'''
Model Evaluation
'''

pred_fn = tf.estimator.inputs.pandas_input_fn(x=X_test,batch_size=len(X_test),shuffle=False)
note_predictions = list(classifier.predict(input_fn=pred_fn))


final_preds  = []
for pred in note_predictions:
    final_preds.append(pred['class_ids'][0])

print('DNN results')
print(confusion_matrix(y_test,final_preds))
print('\n')
print(classification_report(y_test, final_preds))


'''
Comparison with Random Forest Classifier to see if DNN is worth it 
'''

from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=200)
rfc.fit(X_train, y_train)
predictions = rfc.predict(X_test)


print('Random Forest Classifier Results')
print(confusion_matrix(y_test, predictions))
print('\n')
print(classification_report(y_test, predictions))