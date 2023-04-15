import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import optuna
from keras import layers
from keras.models import Sequential
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.backend import argmax
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, precision_score, f1_score, recall_score, accuracy_score
import seaborn as sns
import numpy as np

# 1. Load the Covertype Data Set
def load_dataset():
    df = pd.read_csv('covtype.data', header=None)
    df.head(5)

    df.columns = ['Elevation', 'Aspect', 'Slope', 'Horizontal_Distance_To_Hydrology', 'Vertical_Distance_To_Hydrology', 'Horizontal_Distance_To_Roadways', 'Hillshade_9am',
                  'Hillshade_Noon', 'Hillshade_3pm', 'Horizontal_Distance_To_Fire_Points', 'Wilderness_Area1', 'Wilderness_Area2', 'Wilderness_Area3', 'Wilderness_Area4',
                  'Soil_Type1', 'Soil_Type2', 'Soil_Type3', 'Soil_Type4', 'Soil_Type5', 'Soil_Type6', 'Soil_Type7', 'Soil_Type8', 'Soil_Type9', 'Soil_Type10', 'Soil_Type11',
                  'Soil_Type12', 'Soil_Type13', 'Soil_Type14', 'Soil_Type15', 'Soil_Type16', 'Soil_Type17', 'Soil_Type18', 'Soil_Type19', 'Soil_Type20', 'Soil_Type21',
                  'Soil_Type22', 'Soil_Type23', 'Soil_Type24', 'Soil_Type25', 'Soil_Type26', 'Soil_Type27', 'Soil_Type28', 'Soil_Type29', 'Soil_Type30', 'Soil_Type31',
                  'Soil_Type32', 'Soil_Type33', 'Soil_Type34', 'Soil_Type35', 'Soil_Type36', 'Soil_Type37', 'Soil_Type38', 'Soil_Type39', 'Soil_Type40', 'Cover_Type']

    x = df.loc[:50000, df.columns != 'Cover_Type']
    # y = df['Cover_Type']
    y = df.loc[:50000, df.columns == 'Cover_Type'] - 1
    y = y['Cover_Type']

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, train_size=0.7, random_state=42)

    return x_train, x_test, y_train, y_test

# 3. Use Scikit-learn library to train two simple Machine Learning models
# 3.1 SVM
def svm_classifier(x_train, y_train, x_test):
    model_svc = SVC()
    model_svc.fit(x_train, y_train)
    y_pred_svc = model_svc.predict(x_test)
    #model_svc.score(x_test, y_test)

    return y_pred_svc

# 3.2 k-NN
def knn_classifier(x_train, y_train, x_test):
    model_knn = KNeighborsClassifier()
    model_knn.fit(x_train, y_train)
    y_pred_knn = model_knn.predict(x_test)
    #model_knn.score(x_test, y_test)

    return y_pred_knn

# 4. Use TensorFlow library to train a neural network that will classify the data
# 4.1 Create a function that will find a good set of hyperparameters for the NN
def objective(trial):
    input_shape = (x_train.shape[1],)  # 54
    model = Sequential()

    # trial:
    dense = trial.suggest_categorical("dense", [64, 128, 256, 512])
    model.add(layers.Dense(dense, activation='relu', input_shape=input_shape))

    dropout = trial.suggest_categorical("dropout", [0.2, 0.5, 0.8])
    model.add(layers.Dropout(dropout))

    model.add(layers.Dense(7, activation='softmax'))

    learning_rate = trial.suggest_float("learning_rate", 0.0001, 0.001, log=True)
    model.compile(optimizer=Adam(learning_rate=learning_rate),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    epochs = trial.suggest_int("epochs", 1, 30, log=True)

    model.fit(x_train, to_categorical(y_train, num_classes=7), batch_size=8, epochs=epochs)

    y_pred = argmax(model.predict(x_test), axis=1).numpy()

    return accuracy_score(y_test, y_pred)

def study():
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=20)

    print("Number of finished trials: ", len(study.trials))

    print("Best trial:")
    trial = study.best_trial

    print("\tValue: ", trial.value)

    print("\tParams: ")
    for key, value in trial.params.items():
        print(f"\t\t{key}: {value}")


def NN(x_train, y_train, x_test, y_test, epochs):
    input_shape = (x_train.shape[1],)  # 54
    batch_size = 8
    learning_rate = 0.00055
    #epochs = 10
    epochs = epochs

    nn_model = Sequential([
        layers.Dense(512, activation='relu', input_shape=input_shape),
        layers.Dropout(0.2),
        layers.Dense(7, activation='softmax')
    ])

    nn_model.compile(optimizer=Adam(learning_rate=learning_rate),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])


    history = nn_model.fit(x_train, to_categorical(y_train, num_classes=7), batch_size=batch_size, epochs=epochs, validation_data=(x_test, to_categorical(y_test, num_classes=7)))

    y_pred = argmax(nn_model.predict(x_test), axis=1).numpy()
    #plot_curves(history)

    return y_pred, history

# 4.2 Plot training curves for the best hyperparameters
def plot_curves(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_x = range(1, len(acc) + 1)

    plt.plot(epochs_x, acc, 'bo', label='Training acc')
    plt.plot(epochs_x, val_acc, 'b', label='Validation acc')
    plt.title('NN - Training and validation accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epochs')
    plt.legend()

    plt.figure()

    plt.plot(epochs_x, loss, 'bo', label='Training loss')
    plt.plot(epochs_x, val_loss, 'b', label='Validation loss')
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.title('NN - Training and validation loss')
    plt.legend()

    plt.show()

# 5. Evaluate your neural network and other models
def get_metrics(y_test, y_pred):
    f1 = f1_score(y_test, y_pred, average='macro')
    prec = precision_score(y_test, y_pred, average='macro')
    rec = recall_score(y_test, y_pred, average='macro')
    acc = accuracy_score(y_test, y_pred)

    return f1, prec, rec, acc

def get_cm(y_test, y_pred, name):
    cm = confusion_matrix(y_test, y_pred)

    cmn = cm.astype('float')
    cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots()
    sns.heatmap(cmn, annot=True, fmt='.2f')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title(name)
    plt.show()
    #plt.savefig('static/images/CM.png')


if __name__ == '__main__':
    x_train, x_test, y_train, y_test = load_dataset()

    #study()
    '''
    Best trial:
	Value:  0.7756816212252516
	Params: 
		dense: 512
		dropout: 0.2
		learning_rate: 0.0005546715423974959
		epochs: 26
    '''

    # SVM
    y_pred_svm = svm_classifier(x_train, y_train, x_test)

    # k-NN
    y_pred_knn = knn_classifier(x_train, y_train, x_test)

    # NN
    epochs = 3
    y_pred_nn, history_nn = NN(x_train, y_train, x_test, y_test, epochs)
    plot_curves(history_nn)

    ########### Metrics ###########
    # SVM
    f1_svm, prec_svm, rec_svm, acc_svm = get_metrics(y_test, y_pred_svm)
    get_cm(y_test, y_pred_svm, 'Confusion matrix - SVM')
    print('SVM:\nF1 score: ', f1_svm, '\nPrecision: ', prec_svm, '\nRecall: ', rec_svm, '\nAccuracy: ', acc_svm)

    # k-NN
    f1_knn, prec_knn, rec_knn, acc_knn = get_metrics(y_test, y_pred_knn)
    get_cm(y_test, y_pred_knn, 'Confusion matrix - k-NN')
    print('k-NN:\nF1 score: ', f1_knn, '\nPrecision: ', prec_knn, '\nRecall: ', rec_knn, '\nAccuracy: ', acc_knn)

    # NN
    f1_nn, prec_nn, rec_nn, acc_nn = get_metrics(y_test, y_pred_nn)
    get_cm(y_test, y_pred_nn, 'Confusion matrix - NN')
    print('NN:\nF1 score: ', f1_nn, '\nPrecision: ', prec_nn, '\nRecall: ', rec_nn, '\nAccuracy: ', acc_nn)


