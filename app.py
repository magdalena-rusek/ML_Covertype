from flask import Flask, redirect, url_for, request, render_template
import main as ml

app = Flask(__name__)

@app.route('/')
def home():
    ##
    return render_template('main.html', classifiers=['Simple heuristic', 'SVM', 'k-NN', 'NN'])

@app.route('/predictions', methods=['POST'])
def predictions():
    classifier = request.form['chosen_classifier']
    x_train, x_test, y_train, y_test = ml.load_dataset()

    if classifier == 'Simple heuristic':
        y_pred = ml.heuristic(x_test)
    elif classifier == 'SVM':
        y_pred = ml.svm_classifier(x_train, y_train, x_test)
    elif classifier == 'k-NN':
        y_pred = ml.knn_classifier(x_train, y_train, x_test)
    else:
        epochs = 3
        y_pred, _ = ml.NN(x_train, y_train, x_test, y_test, epochs)

    classes = zip(y_test, y_pred)

    #f1, prec, rec, acc = ml.get_metrics(y_test, y_pred)
    return render_template('page1.html', classifier=classifier, classes=classes)

if __name__ == '__main__':
    app.run()