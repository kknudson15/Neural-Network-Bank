'''
Deep Learning Project
Bank Authentication Data Set 
Neural Network
'''


from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import tensorflow as tf
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def gather_data(file):
    bank_data = pd.read_csv(file)
    return bank_data

def data_exploration(data_frame):
    print(data_frame.head())
    sns.countplot(x = 'Class',data = data_frame )
    plt.show()

    sns.pairplot(data = data_frame, hue = 'Class', diag_kind= 'histogram')
    plt.show()


def scale_data(data_frame):
    scaler = StandardScaler()
    scaler.fit(bank_data.drop('Class',axis=1))
    scaled_features = scaler.fit_transform(bank_data.drop('Class',axis=1))
    df_feat = pd.DataFrame(scaled_features,columns=bank_data.columns[:-1])
    return df_feat

def train_test_data_split(scaled_data, data_frame):
    X = scaled_data
    y = data_frame['Class']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    return X_train, X_test,y_train, y_test

def train_tensor(X_train, y_train):
    image_var = tf.feature_column.numeric_column("Image.Var")
    image_skew = tf.feature_column.numeric_column('Image.Skew')
    image_curt = tf.feature_column.numeric_column('Image.Curt')
    entropy =tf.feature_column.numeric_column('Entropy')

    feat_cols = [image_var,image_skew,image_curt,entropy]

    classifier = tf.estimator.DNNClassifier(hidden_units=[10, 20, 10], n_classes=2,feature_columns=feat_cols)

    input_func = tf.estimator.inputs.pandas_input_fn(x=X_train,y=y_train,batch_size=20,shuffle=True)

    classifier.train(input_fn=input_func,steps=500)
    return classifier

def model_evaluation(classifier, X_test, y_test):
    pred_fn = tf.estimator.inputs.pandas_input_fn(x=X_test,batch_size=len(X_test),shuffle=False)
    note_predictions = list(classifier.predict(input_fn=pred_fn))


    final_preds  = []
    for pred in note_predictions:
        final_preds.append(pred['class_ids'][0])

    print('DNN results')
    print(confusion_matrix(y_test,final_preds))
    print('\n')
    print(classification_report(y_test, final_preds))
def Random_Forest_Classifier(X_train,X_test, y_train, y_test):
    rfc = RandomForestClassifier(n_estimators=200)
    rfc.fit(X_train, y_train)
    predictions = rfc.predict(X_test)
    print('Random Forest Classifier Results')
    print(confusion_matrix(y_test, predictions))
    print('\n')
    print(classification_report(y_test, predictions))

if __name__ == '__main__':
    file_name = 'bank_note_data.csv'
    bank_data = gather_data(file_name)
    explore = input('Do you want to visualize the data?')
    if explore == 'yes':
        data_exploration(bank_data)
    scaled_data = scale_data(bank_data)
    train_test_data = train_test_data_split(scaled_data,bank_data)
    X_train = train_test_data[0]
    X_test = train_test_data[1]
    y_train = train_test_data[2]
    y_test = train_test_data[3]
    classifier = train_tensor(X_train, y_train)
    model_evaluation(classifier, X_test, y_test)
    Random_Forest_Classifier(X_train, X_test, y_train, y_test)
