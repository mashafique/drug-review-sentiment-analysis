from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from dmba import plotDecisionTree, classificationSummary, regressionSummary

def ca_score_model(train_actual_y, train_pred_y, test_actual_y, test_pred_y):
    print("Training Data Metrics:")
    print("Accuracy on the train is:",accuracy_score(train_actual_y,train_pred_y))
    classificationSummary(train_actual_y,train_pred_y)
    print('The Precision on the train is:', precision_score(train_actual_y,train_pred_y))
    print('The Recall on the train is:',recall_score(train_actual_y,train_pred_y))
    print('The F-Measure on the train is:',f1_score(train_actual_y,train_pred_y))
    
    print("\nTesting Data Metrics:")
    print("Accuracy on the test is:",accuracy_score(test_actual_y, test_pred_y))
    classificationSummary(test_actual_y, test_pred_y) 
    print('The Precision on the test is:', precision_score(test_actual_y, test_pred_y))
    print('The Recall on the test is:',recall_score(test_actual_y, test_pred_y))
    print('The F-Measure on the test is:',f1_score(test_actual_y, test_pred_y))
    
    test_accuracy = accuracy_score(test_actual_y, test_pred_y)
    test_precision = precision_score(test_actual_y, test_pred_y)
    test_recall = recall_score(test_actual_y, test_pred_y)
    test_f1 = f1_score(test_actual_y, test_pred_y)
    
    results = [test_accuracy, test_precision, test_recall, test_f1]
    
    return(results)
    
def model_comp(base,optim,case1='Base Case:',case2='Optimized:'):
    print('\n{:<28}{:<14}{:<14}{:<14}{:<14}'.format('Model Comparison','Accuracy','Precision','Recall','F-Measure'))
    print('{:<28}{:<14.4f}{:<14.4f}{:<14.4f}{:<14.4f}'.format(case1,base[0],base[1],base[2],base[3]))
    print('{:<28}{:<14.4f}{:<14.4f}{:<14.4f}{:<14.4f}'.format(case2,optim[0],optim[1],optim[2],optim[3]))
    
def get_metrics(y_test, y_predicted):
    accuracy = accuracy_score(y_test, y_predicted)
    precision = precision_score(y_test, y_predicted, pos_label=None, average='weighted')
    recall =  recall_score(y_test, y_predicted, pos_label=None, average='weighted')
    f1 = f1_score(y_test, y_predicted, pos_label=None, average='weighted')
    
    return accuracy, precision, recall, f1

def get_naive_bayes_metrics(y_test_est, y_test):
    TP = 0.00000000001
    TN = 0.00000000001
    FP = 0.00000000001
    FN = 0.00000000001

    for i in range(len(y_test)):
        if str(y_test_est[i])=="True" and y_test_est[i]==y_test[i]:
            TP = TP+1
        elif str(y_test_est[i])=="False" and y_test_est[i]==y_test[i]:
            TN = TN+1
        elif str(y_test_est[i])=="True" and y_test_est[i]!=y_test[i]:
            FP = FP+1
        else:
            FN = FN+1
            
    #Accuracy
    ACC = (TP+TN)/(TP+TN+FP+FN)
    #print("The accuracy rate is", round(ACC,4))

    #Precision
    PRE = TP/(TP+FP)
    #print("The precision is", round(PRE,4))

    #Recall
    REC = TP/(TP+FN)
    #print("The recall is", round(REC,4))

    #F-measure
    Fscore = 2*PRE*REC/(PRE+REC)
    #print("The F-measure is", round(Fscore,4))
    
    return ACC, PRE, REC, Fscore