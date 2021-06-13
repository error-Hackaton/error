import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn import svm
import inne
import pickle

'''Modyfikacja danych - zmienna celu jako binarna, one hot'''
dane = pd.read_csv('node_modules\\dataset.csv', index_col = 0)
dane.replace({'bedzie_do2h': {'tak':1, 'nie':0}}, inplace = True)
for kat in ['SL', 'W', 'O', 'T']:
    nazwa = 'typ_' + kat
    dane[nazwa] = pd.Series(dane['TYP'] == kat).astype(int)
for kat in ['SL', 'W', 'O', 'T']:
    nazwa = 'pop_' + kat
    dane[nazwa] = pd.Series(dane['poprzedni_typ'] == kat).astype(int)
dane = dane.drop(['poprzedni_typ','TYP', 'index'], axis = 1)

'''Wybranie nazw predyktorów'''
nazwy_pred = list(dane.columns)
nazwy_pred.remove('czas')
nazwy_pred.remove('UWAGI')
nazwy_pred.remove('bedzie_do1h')
nazwy_pred.remove('bedzie_do2h')

'''Wsawienie wartości 0 tam gdzie przez ostatnie 4 godziny nie było aktywności'''
dane.isna().sum()/dane.shape[0]*100
dane = dane.fillna(0)

'''Podział zbioru na próbę uczącą i testową - z zachowaniem porządku czasowego'''
dane_ucz, dane_test = dane[:65075], dane[65075:]


'''Podział zbiorów uczących i testowych na predyktory i zmienną celu'''
dane_ucz = dane_ucz[dane_ucz['typ_O']==0]
dane_ucz = dane_ucz[dane_ucz['typ_T']==0]
nazwy_pred.remove('typ_O')
nazwy_pred.remove('typ_T')
pred_ucz, pred_test, cel_ucz, cel_test = dane_ucz[nazwy_pred], dane_test[nazwy_pred], dane_ucz['bedzie_do2h'], dane_test['bedzie_do2h']

'''Utworzenie pliku z ostatnimi wartociami zbioru uczącego'''
'''
x = dane_ucz[-20:]
y = pd.DataFrame(columns = ['czas', 'prognoza'])
y['czas'], y['prognoza'] = dane_ucz['czas'][-20:], 0
x = x[['czas', 'ENG', '4_2_SL', '4_2_W', '4_2_srE', '2_1_SL', '2_1_W', '1_0_SL', '1_0_W', 'typ_SL', 'typ_W', 'pop_SL', 'pop_W', 'pop_O', 'pop_T']]
x.to_csv('node_modules\\historia.csv')
y.to_csv('node_modules\\prognozy.csv')
dane_test.to_csv('node_modules\\testowe.csv')'''

'''Poszukiwanie hiperparametrów'''
#nsv = inne.bayes_svc_non_lin(pred_ucz, cel_ucz, 35)

'''Model oparty o wielowarstwowy perceptron
ml = MLPClassifier(hidden_layer_sizes=(10,4),solver = 'lbfgs', 
                   activation='tanh', alpha = 0.005)
ml.fit(pred_ucz, cel_ucz)

jakosc1 = inne.roc_auc(ml, pred_ucz, cel_ucz, pred_test, cel_test)
jakosc1.krzywaROC()'''

'''Model oparty o maszynę wektorów nośnych'''
wektory = svm.SVC()
wektory.fit(pred_ucz, cel_ucz)
wektory = CalibratedClassifierCV(wektory)
wektory.fit(pred_ucz, cel_ucz)

jakosc3 = inne.roc_auc(wektory, pred_ucz, cel_ucz, pred_test, cel_test)
jakosc3.krzywaROC(nazwa = '- SVM')
prp = jakosc3.prawdop

'''Zapis modelu do predykcji wstrząsów'''
#with open('node_modules\\model', 'wb') as plik:
#    pickle.dump(wektory, plik)


'''Model oparty o las losowy
las = RandomForestClassifier(random_state = 2021, ccp_alpha=0.02 , class_weight='balanced')
las.fit(pred_ucz, cel_ucz)

jakosc2 = jakosc_modelu.roc_auc(las, pred_ucz, cel_ucz, pred_test, cel_test)
jakosc2.krzywaROC()'''

