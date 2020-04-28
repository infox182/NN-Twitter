from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix as conf_matr
from sklearn.metrics import f1_score
import numpy as np
import matplotlib.pyplot as plt

def score_test(y_true,y_pred):#на вход истинные метки и предсказанные
    
    cm = conf_matr(y_true,y_pred) #обычная
    cmn = conf_matr(y_true,y_pred,normalize ='true') #нормированная
    class_names = ['positive','negative','neutral']

    #строим графики
    for matr in [cm,cmn]:
        disp = ConfusionMatrixDisplay(confusion_matrix=matr,display_labels=class_names)
        disp = disp.plot(cmap=plt.cm.Blues)

    plt.show()
    print(cm,'\n','\n',cmn,'\n')
    accuracy = float((1 - np.mean(y_true  != y_pred)))
    f1 = f1_score(y_true,y_pred,average = 'weighted')
    print(f'accuracy = {accuracy}')
    print(f'F1 score = {f1}')
