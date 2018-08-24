# encoding: utf-8

"""
@Author: Kaiqi Yuan
@Since: 2018-08-11 上午4:24
@Description:
"""
import sys
sys.path.extend(['/home/cdy/ykq/DDISuccess', '/home/cdy/ykq/DDISuccess/comparison'])

from comparison.comparison_dataset_process import load_comparison_dataset
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, accuracy_score, average_precision_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import numpy as np
import tensorflow as tf
from sklearn.semi_supervised import LabelPropagation

train_x, train_y, test_x, test_y = load_comparison_dataset("/data/cdy/ykq/ComparisonDataset", 28800)
print("train_X:", train_x.shape)
print("train_y:", train_y.shape)
print("test_X:", test_x.shape)
print("test_X:", test_y.shape)

def print_result(y_true, y_pred, y_logit):
    # y_logit = np.max(y_logit, axis=1)
    precision, recall, fscore, _ = precision_recall_fscore_support(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)
    print("y_true: ", y_true.shape)
    print("y_logit: ", y_logit.shape)

    y_true_matrix = []
    for label in y_true:
        if label == 0:
            y_true_matrix.append([1, 0])
        else:
            y_true_matrix.append([0, 1])
    y_true_matrix = np.array(y_true_matrix)
    print("y_true_matrix: ", y_true_matrix.shape)
    _, auroc_tensor = tf.metrics.auc(y_true_matrix, y_logit)
    _, aupr_tensor = tf.metrics.auc(y_true_matrix, y_logit, curve="PR")
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        auroc, aupr = sess.run([auroc_tensor, aupr_tensor])
    print("accuracy:{}\nprecision:{}\nrecall:{}\nfscore:{}\nauroc:{}\naupr:{}".format(
        accuracy, precision, recall, fscore, auroc, aupr))
    print("------------------------")


def knn_classifier():
    print("knn classifier: ")
    neigh = KNeighborsClassifier()
    neigh.fit(train_x, train_y)
    y_pred = neigh.predict(test_x)
    y_logit = neigh.predict_proba(test_x)
    print_result(test_y, y_pred, y_logit)
    """
    accuracy:0.81
    precision:[ 0.81632653  0.80392157]
    recall:[ 0.8   0.82]
    fscore:[ 0.80808081  0.81188119]
    auroc:0.8891666531562805
    aupr:0.8997911214828491
    """

def decision_tree_classifiler():
    print("decision tree classifier: ")
    clf = DecisionTreeClassifier()
    clf.fit(train_x, train_y)
    y_pred = clf.predict(test_x)
    y_logit = clf.predict_proba(test_x)
    print_result(test_y, y_pred, y_logit)
    """
    accuracy:0.9933333333333333
    precision:[ 0.98684211  1.        ]
    recall:[ 1.          0.98666667]
    fscore:[ 0.99337748  0.99328859]
    auroc:0.9999777674674988
    aupr:0.9999779462814331
    """



def bayes_classifier():
    print("bayes classifier: ")
    clf = BernoulliNB()
    clf.fit(train_x, train_y)
    y_pred = clf.predict(test_x)
    y_logit = clf.predict_proba(test_x)
    print_result(test_y, y_pred, y_logit)
    """
    accuracy:0.6533333333333333
    precision:[ 0.66428571  0.64375   ]
    recall:[ 0.62        0.68666667]
    fscore:[ 0.64137931  0.66451613]
    auroc:0.6806666851043701
    aupr:0.7478679418563843
    """


def logistic_classifier():
    print("logistic classifier: ")
    clf = LogisticRegression()
    clf.fit(train_x, train_y)
    y_pred = clf.predict(test_x)
    y_logit = clf.predict_proba(test_x)
    print_result(test_y, y_pred, y_logit)
    """
    ccuracy:0.8
    precision:[ 0.78846154  0.8125    ]
    recall:[ 0.82  0.78]
    fscore:[ 0.80392157  0.79591837]
    auroc:0.8869332671165466
    aupr:0.88612961769104
    """

def svm_classifier():
    print("svm classifier: ")
    clf = SVC(probability=True)
    clf.fit(train_x, train_y)
    y_pred = clf.predict(test_x)
    y_logit = clf.predict_proba(test_x)
    print_result(test_y, y_pred, y_logit)
    """
    
    """


if __name__ == '__main__':
    # knn_classifier()
    # decision_tree_classifiler()
    # bayes_classifier()    #建議用這個來測
    # logistic_classifier()
    svm_classifier()
