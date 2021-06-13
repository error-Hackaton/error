import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix as CM
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt
from sklearn import svm
from skopt import BayesSearchCV
from sklearn.model_selection import cross_val_score
from skopt.space import Real, Integer

class jakosc_modelu():
    def __init__(self, model, predyktory, cel, nazwa = ''):
        self.model = model
        self.predyktory = predyktory
        self.cel = cel
        self.nazwy_predyktorow = list(predyktory.keys())
        self.klasyfikacja()
        self.nazwa = nazwa
    def klasyfikacja(self):
        self.prp = self.model.predict_proba(self.predyktory)[:,1]
        self.fpr, self.tpr, self.thr = roc_curve(self.cel, self.prp, pos_label = 1)
        self.pkt = np.abs(self.tpr-(1-self.fpr))
        self.maxpkt = np.argmin(self.pkt)
        self.przewidywane = self.prp >= self.thr[self.maxpkt]
        self.przewidywane = pd.DataFrame({'label': self.przewidywane })
        self.przewidywane.replace({'label': {True:1, False: 0}},inplace=True)
    def utworz_wskazniki(self):
        self.mx_pomylek = CM(self.cel, self.przewidywane)
        TN_U, FP_U, FN_U, TP_U = self.mx_pomylek.ravel()
        self.wskazniki = {}
        self.wskazniki['Trafność'] = round((TP_U+TN_U)/(TP_U+FP_U+TN_U+FN_U),3)
        self.wskazniki['Całkowity współczynnik błędu'] = round((FN_U+FP_U)/(TP_U+FP_U+TN_U+FN_U),3)
        self.wskazniki['Czułość'] = round(TP_U/(FN_U+TP_U),3)
        self.wskazniki['Specyficzność'] = round(TN_U/(TN_U+FP_U),3)
        self.wskazniki['Wskaźnik fałszywie negatywnych'] = round(FN_U/(FN_U+TP_U),3)
        self.wskazniki['Wskaźnik fałszywie pozytywnych'] = round(FP_U/(TN_U+FP_U),3)
        self.wskazniki['Precyzja'] = round(TP_U/(FP_U+TP_U),3)
        self.wskazniki['Proporcja prawdziewie negatywnych'] = round(TN_U/(TN_U+FN_U),3)
        self.wskazniki['Proporcja fałszywie pozytywnych'] = round(FP_U/(FP_U+TP_U),3)
        self.wskazniki['Proporcja fałszywie negatywnych'] = round(FN_U/(TN_U+FN_U),3)
    def ROC(self, rozmiar = (10,10), nazwa = ''):        
        self.auc = round(roc_auc_score(self.cel, self.prp), 6)
        plt.figure(figsize = rozmiar)
        plt.title('Krzywa ROC ' + nazwa, fontsize = 23, fontweight = 'bold')
        plt.ylabel('Czułość', fontsize = 18)
        plt.xlabel('1 - Swoistość', fontsize = 18)
        plt.xlim(0,1)
        plt.ylim(0,1)
        siatka = [x*0.1 for x in range (0,11)]
        plt.xticks(siatka)
        plt.yticks(siatka)
        plt.grid(which='major', linestyle=':', linewidth='0.5', color='black', zorder = 0)
        plt.plot([0, 1], [0, 1], color= 'black', linestyle='--', alpha = 0.6)
        plt.plot(self.fpr, self.tpr, color='blue') 
        plt.scatter(1 - self.wskazniki['Specyficzność'], self.wskazniki['Czułość'], color = 'yellow', alpha = 1)
        plt.text(0.52, 0.05, "AUC = " + str(self.auc), fontsize = 20, fontweight = 'bold', color = 'blue')
        plt.show()
class roc_auc():
    def __init__(self, model, pred_ucz, cel_ucz, pred_test, cel_test, prog = False):
        self.model = model
        self.pred_ucz = pred_ucz
        self.cel_ucz = cel_ucz
        self.pred_test = pred_test
        self.cel_test = cel_test
        if not prog:
            self.jakosc_ucz = jakosc_modelu(self.model, pred_ucz, cel_ucz)
            self.jakosc_test = jakosc_modelu(self.model, pred_test, cel_test)
        else:
            self.jakosc_ucz = jakosc_modelu(self.model, pred_ucz, cel_ucz, prog_prob = True)
            self.jakosc_test = jakosc_modelu(self.model, pred_test, cel_test, prog_prob = True)
        self.jakosc_ucz.utworz_wskazniki(); self.jakosc_test.utworz_wskazniki()
    def krzywaROC(self, rozmiar = (10,10), nazwa = ''):
        self.przewidywane_u = self.model.predict(self.pred_ucz)
        self.prawdop_u = self.model.predict_proba(self.pred_ucz)
        self.przewidywane_t = self.model.predict(self.pred_test)
        self.prawdop_t = self.model.predict_proba(self.pred_test)
        
        fpr_u, tpr_u, thr_u = roc_curve(self.cel_ucz, self.prawdop_u[:,1], pos_label = 1)
        fpr_t, tpr_t, thr_t = roc_curve(self.cel_test, self.prawdop_t[:,1], pos_label = 1)
        
        self.auc_u = round(roc_auc_score(self.cel_ucz, self.prawdop_u[:,1]), 6)
        self.auc_t = round(roc_auc_score(self.cel_test, self.prawdop_t[:,1]), 6)
        plt.figure(figsize = rozmiar)
        plt.title('Krzywa ROC ' + nazwa, fontsize = 23, fontweight = 'bold')
        plt.ylabel('Czułość', fontsize = 18)
        plt.xlabel('1 - Swoistość', fontsize = 18)
        plt.xlim(0,1)
        plt.ylim(0,1)
        siatka = [x*0.1 for x in range (0,11)]
        plt.xticks(siatka)
        plt.yticks(siatka)
        plt.grid(which='major', linestyle=':', linewidth='0.5', color='black', zorder = 0)
        plt.plot([0, 1], [0, 1], color= 'black', linestyle='--', alpha = 0.6)
        plt.plot(fpr_u, tpr_u, color='brown') 
        plt.plot(fpr_t, tpr_t, color='blue') 
        plt.text(0.52, 0.05, "Uczący AUC = " + str(self.auc_u), fontsize = 20, fontweight = 'bold', color = 'brown')
        plt.text(0.496, 0.01, "Testowy AUC = " + str(self.auc_t), fontsize = 20, fontweight = 'bold', color = 'blue')
        plt.show()
        self.prawdop = pd.DataFrame({'fpr': fpr_t, 'tpr,':tpr_t, 'thr':thr_t})

def bayes_svc_non_lin(pred, cel, liczba_iter):
    '''Szukanie najlepszych hipierparametrów modelu opartego o nieliniowy SVM'''
    opt = BayesSearchCV(svm.SVC(random_state = 2021, kernel = 'rbf'), {
        'C': Real(1e-7, 1e+3, prior='log-uniform'), 
        'gamma': Real(1e-7, 1e+3, prior='log-uniform'),
        'tol': Real(1e-7, 1e+3, prior='log-uniform'),
        'degree': Integer(1,8)
        }, n_iter = liczba_iter, cv = 5, scoring='recall',random_state = 2022)
    opt.fit(pred, cel)
    cv_scores = cross_val_score(opt, pred, cel)
    return opt