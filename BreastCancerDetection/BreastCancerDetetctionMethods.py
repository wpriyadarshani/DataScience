
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier


def loadData():
    # load  data
    total = pd.read_csv('data.csv')
    total_data = total.values
    np.random.shuffle(total_data)

    total_y = total_data[:,1]
    total_X = total_data[:,2:-1]


    count = 0 
    #malignant 1 else 0
    for i in range(len(total_y)):
        if total_y[i] == 'M':
            total_y[i] = 1
            count =count+1
        else:
            total_y[i] = 0

    # print(total_X)
    print("count", count)

    total_X = pd.DataFrame(total_X)
    result = total_X.copy()
    for feature_name in total_X.columns:
        max_value = total_X[feature_name].max()
        min_value = total_X[feature_name].min()
        result[feature_name] = (total_X[feature_name] - min_value) / (max_value - min_value)
    print(result)

    total_X = total_X.values

    return total_X, total_y

total_X, total_y = loadData()

#SVM 

#cross validate data
def SVM_implementation():
    final_accuracy = []
    for i in range(10):
        accuracy = []
        fpr = [0, 0,0]
        tpr = [0,0,0]
        kf = KFold(n_splits=10)
        for training_id, test_id in kf.split(total_X):
            train_x, test_x = total_X[training_id], total_X[test_id]
            train_y, test_y = total_y[training_id], total_y[test_id]

            train_y = train_y.astype('int') 
            clf = svm.SVC(gamma='scale')
            clf.fit(train_x, train_y)

            output = clf.predict(test_x)
            correct = sum(np.array(output == test_y))
            accuracy.append((correct* 1.0)/ len(output))
        final_accuracy.append(sum(accuracy)/10)

    print("overall accuracy", sum(final_accuracy)/10)


def NaiveBayes():
    final_accuracy = []
    for i in range(10):
        accuracy = []
        kf = KFold(n_splits=10)
        for training_id, test_id in kf.split(total_X):
            train_x, test_x = total_X[training_id], total_X[test_id]
            train_y, test_y = total_y[training_id], total_y[test_id]

            train_y = train_y.astype('int') 
            clf = GaussianNB()
            clf.fit(train_x, train_y)

            output = clf.predict(test_x)
            correct = sum(np.array(output == test_y))

            accuracy.append((correct* 1.0)/ len(output))
        final_accuracy.append(sum(accuracy)/10)

    print("overall accuracy", sum(final_accuracy)/10)



    
def DecisionTree():
    final_accuracy = []
    for i in range(10):
        accuracy = []
       
        kf = KFold(n_splits=10)
        for training_id, test_id in kf.split(total_X):
            train_x, test_x = total_X[training_id], total_X[test_id]
            train_y, test_y = total_y[training_id], total_y[test_id]

            train_y = train_y.astype('int') 
            clf = tree.DecisionTreeClassifier()
            clf.fit(train_x, train_y)

            output = clf.predict(test_x)
            correct = sum(np.array(output == test_y))
            accuracy.append((correct* 1.0)/ len(output))
        final_accuracy.append(sum(accuracy)/10)

    print("overall accuracy", sum(final_accuracy)/10)
   


def LDA():
    final_accuracy = []
    for i in range(10):
        accuracy = []
       
        kf = KFold(n_splits=10)
        for training_id, test_id in kf.split(total_X):
            train_x, test_x = total_X[training_id], total_X[test_id]
            train_y, test_y = total_y[training_id], total_y[test_id]

            train_y = train_y.astype('int') 
            clf = LinearDiscriminantAnalysis()
            clf.fit(train_x, train_y)

            output = clf.predict(test_x)
            correct = sum(np.array(output == test_y))


            accuracy.append((correct* 1.0)/ len(output))

        final_accuracy.append(sum(accuracy)/10)

    print("overall accuracy", sum(final_accuracy)/10)
   

 

def RF():
    final_accuracy = []
    for i in range(10):
        accuracy = []
       
        kf = KFold(n_splits=10)
        for training_id, test_id in kf.split(total_X):
            train_x, test_x = total_X[training_id], total_X[test_id]
            train_y, test_y = total_y[training_id], total_y[test_id]

            train_y = train_y.astype('int') 
            clf = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)
            clf.fit(train_x, train_y)

            output = clf.predict(test_x)
            correct = sum(np.array(output == test_y))


            accuracy.append((correct* 1.0)/ len(output))
        final_accuracy.append(sum(accuracy)/10)

    print("overall accuracy", sum(final_accuracy)/10)
   

def LR():
    #linear regreesion
    final_accuracy = []
    for i in range(10):
        kf = KFold(n_splits=10)
        K = 10
        accuracy = []
        for training_id, test_id in kf.split(total_X):
            train_x, test_x = total_X[training_id], total_X[test_id]
            train_y, test_y = total_y[training_id], total_y[test_id]

            train_y = train_y.astype('int') 
            clf = LinearRegression()
            clf.fit(train_x, train_y)

            output = clf.predict(test_x)
            predict = (2 * (clf.predict(test_x) > 0.5)) - 1
            # print("predict ", predict)

            for loc in range(len(predict)):
                if predict[loc] == -1:
                    predict[loc] = 0

            # print("predict 2 ", predict)
            correct = sum(np.array(predict == test_y))
            accuracy.append((correct* 1.0)/ len(output))

        final_accuracy.append(sum(accuracy)/10)

    print("overall accuracy", sum(final_accuracy)/10)

def KNN():
    
    kf = KFold(n_splits=10)
        # predict probabilities
    #try KNN for diffrent k nearest neighbor from 1 to 15
    neighbors_setting = range(1,16)

    for n_neighbors in neighbors_setting:

        final_accuracy = []
        for i in range(10):
            print('k nearest neighbor=',n_neighbors)
            accuracy=[]
            knn = KNeighborsClassifier(n_neighbors=n_neighbors)
            for training_id, test_id in kf.split(total_X):
                train_x, test_x = total_X[training_id], total_X[test_id]
                train_y, test_y = total_y[training_id], total_y[test_id]

                train_y=train_y.astype('int')
                knn.fit(train_x,train_y)
                output = knn.predict(test_x)
                correct = sum(np.array(output == test_y))

                accuracy.append((correct* 1.0)/ len(output))
                
                # calculate roc curve
                fpr, tpr, n_neighbors = roc_curve(output, test_y,drop_intermediate=False)
                
                auc = metrics.auc(fpr, tpr)#calculating auc for tpr and fpr
                
            # plot no skill
            plt.plot([0, 1], [0, 1], linestyle='--')
            # plot the roc curve for the model
            plt.plot(fpr, tpr, marker='.')
            txt="AUC",metrics.auc(fpr, tpr)
            plt.text(0.7,0.2,txt,ha='center')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver operating characteristic example')
            # show the plot
            plt.show()
            
            print("average accuracy KNN", (sum(accuracy)/ 10.0)*100)

            final_accuracy.append(sum(accuracy)/10)

        print("overall accuracy", sum(final_accuracy)/10)

def LogisticReg():
    
    kf = KFold(n_splits=10)
        # predict probabilities
    thresholds=[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
    for thres in thresholds:
        final_accuracy = []
        for i in range(10):

            accuracy = []
            TPR_list = [0,0,0]
            FPR_list = [0,0,0]
            for training_id, test_id in kf.split(total_X):
                train_x, test_x = total_X[training_id], total_X[test_id]
                train_y, test_y = total_y[training_id], total_y[test_id]

                train_y = train_y.astype('int') 
                log_r = LogisticRegression(solver='lbfgs')
                log_r.fit(train_x, train_y)

                output = log_r.predict(test_x)

                preds=np.where(log_r.predict_proba(test_x)[:,1]>thres,1,0)

                fpr, tpr, _ = roc_curve(preds,test_y, drop_intermediate=False)
                
                print('Accuracy from scratch: {0}'.format((preds == test_y).sum().astype(float) / len(preds)))
                
                accuracy.append((preds == test_y).sum().astype(float) / len(preds))
                
                if len(tpr) > 2:
                    TPR_list[0]+=tpr[0]
                    TPR_list[1]+=tpr[1]
                    TPR_list[2]+=tpr[2]

                if len(fpr) > 2:
                    FPR_list[0]+=fpr[0]
                    FPR_list[1]+=fpr[1]
                    FPR_list[2]+=fpr[2]
            

            #Calculating the average of TPR and FPR
            FPR_Avg = FPR_list[0]/10.0, FPR_list[1]/10.0,FPR_list[2]/10.0 
            TPR_Avg = TPR_list[0]/10.0, TPR_list[1]/10.0,TPR_list[2]/10.0

            auc = metrics.auc(FPR_Avg, TPR_Avg)#calculating auc for tpr and fpr
            print('fpr',FPR_Avg)
            print('tpr',TPR_Avg)
            print('auc',auc)

            #plotting graph for ROC
            plt.figure()
            lw = 2
            plt.plot(FPR_Avg, TPR_Avg, color='darkorange',lw=lw)
            plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            txt="AUC",metrics.auc(FPR_Avg, TPR_Avg)
            plt.text(0.7,0.2,txt,ha='center')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver operating characteristic example')
            plt.legend(loc="lower right")
            plt.show()
            
            print("average=", sum(accuracy)/ 10.0)

            final_accuracy.append(sum(accuracy)/10)

        print("overall accuracy", sum(final_accuracy)/10)


# KNN()
# SVM_implementation()
# NaiveBayes()
# DecisionTree()
# LDA()
# RF()
# LR()