import pandas as pd
import pickle
import numpy as np

def przewidywanie_wstrzasu(obserwacja, t):
    '''Przewidywanie wystąpienia wstrząsu
    Funkcja w oparciu o historię zdarzeń przewiduje wystąpienie wstrząsu i dodaje  nowe obserwacje do historii
    
    PREDYKTORY:
        ENG - energia wstrząsu
        4_2_SL - ilość zdarzeń śladowych ostatnich 4 godzin wstecz
        4_2_W - ilość wstrząsów ostatnich 4 godzin wstecz
        4_2_srE - średnia energia ostatnich 4 godzin wstecz
        2_1_SL - ilość zdarzeń śladowych ostatnich 2 godzin wstecz
        2_1_W - ilość wstrząsów ostatnich 2 godzin wstecz
        1_0_SL - ilość zdarzeń śladowych ostatniej godziny
        1_0_W - ilość wstrząsów ostatniej godziny
        typ_SL - zmienna binarna - 1 jeśli nastąpiło obecnie zjawisko śladowe
        typ_W - zmienna binarna - 1 jeśli nastąpił obecnie wstrząs
        pop_SL -zmienna binarna - 1 jeśli w ostatniej obserwacji nastąpiło zjawisko śladowe
        pop_W -zmienna binarna - 1 jeśli w ostatniej obserwacji nastąpił wstrząs
        pop_O -zmienna binarna - 1 jeśli w ostatniej obserwacji nastąpiło odprężenie
        pop_T -zmienna binarna - 1 jeśli w ostatniej obserwacji nastąpiło tąpnięcie
        '''
        
    with open('node_modules\\model', 'rb') as plik:
        model = pickle.load(plik)
    ob = obserwacja.copy()
    historia = pd.read_csv('node_modules\\historia.csv', index_col = 0, parse_dates=['czas'])
    prognozy = pd.read_csv('node_modules\\prognozy.csv', index_col = 0, parse_dates=['czas'])
    i = historia.index.max()
    cechy = obserwacja[historia.columns]
    historia = historia.append(cechy)
    obserwacja = obserwacja[historia.columns].drop('czas', axis = 1)
    przewidywanie = (model.predict_proba(obserwacja)[:,1]>= 0.00291109)
    if przewidywanie:
        przewidywanie = 1
    else:
        przewidywanie = 0
    prognozy.loc[i+1] = [t, przewidywanie]
    komunikat = None
    if przewidywanie:
        wczesniej = 0
        while (t - prognozy['czas'][i]).seconds/3600 <= 2:
            if prognozy['prognoza'][i]:
                wczesniej += 1
            i -= 1
        if wczesniej <=1 :
            komunikat = 'Możliwe odprężenie lub tąpnięcie'
        else:
            komunikat = 'Wysoka szansa odprężenia lub tąpnięcia'
    prognozy.to_csv('node_modules\\prognozy.csv')
    historia.to_csv('node_modules\\historia.csv')
    return komunikat

kom =[]
dane_testowe = pd.read_csv('node_modules\\testowe.csv', index_col = 0, parse_dates=['czas'])
for i, wiersz in dane_testowe['czas'].iteritems():
    wiersz = pd.DataFrame(dane_testowe.loc[[i]])
    t = wiersz['czas'][i]
    kom.append(przewidywanie_wstrzasu(pd.DataFrame(dane_testowe.loc[[i]]),t))

kom = pd.Series(kom)
kom = kom.fillna(0)
kom.value_counts()

sprawdzenie = pd.DataFrame(columns = ['czas', 'komunikat', 'zdarzenie'])
sprawdzenie['czas'] = dane_testowe['czas']
sprawdzenie['komunikat'] = kom.values
sprawdzenie['zdarzenie'] = dane_testowe['typ_O'] + dane_testowe['typ_T']

sprawdzenie['komunikat'].value_counts()
