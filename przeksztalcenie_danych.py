import pandas as pd
import numpy as np



'''Wczytanie danych'''
dane = pd.read_excel('node_modules\\2016-2020.xlsx')

'''Zmienna czasu w jednej kolumnie, wyselekcjonowanie cech'''
dane['czas'] = dane['DATA'].astype('str') + ' ' + dane['GODZ'].astype('str')  + ':' + dane['MIN'].astype('str')  + ':' + dane['SEK'].astype('str') 
dane['czas']  = pd.to_datetime(dane['czas'] )
dane = dane.drop(['DATA', 'GODZ', 'MIN', 'SEK'], axis = 1)
pd.crosstab(dane['TYP'], dane['REJON'])
dane = dane[['czas', 'ENG', 'TYP', 'UWAGI']]

'''Usunięcie błędnej obserwacji'''
dane = dane.drop(14029, axis = 0)
dane = dane.reset_index()

'''Proste zapoznanie się ze zmienną typ'''
x = dane['TYP'].value_counts()
y = dane.groupby('TYP').aggregate('mean')
#plt.figure(figsize = (500,10))
#plt.yscale('log')
#plt.plot(dane['czas'][:2000], dane['ENG'][:2000])


'''Znalezienie indeksów odprężeń i tąpnięć'''
wst = dane['TYP'].isin(['O', 'T'])
idx = [i for i, war in enumerate(wst) if war]


'''Utworzenie nowych cech'''
dane['poprzedni_typ'] = np.nan
dane['4_2_SL'] = 0
dane['4_2_W'] = 0
dane['4_2_srE'] = 0
dane['2_1_SL'] = 0
dane['2_1_W'] = 0
dane['2_1_srE'] = 0
dane['1_0_SL'] = 0
dane['1_0_W'] = 0
dane['1_0_srE'] = 0

najblizszy = idx[0] #najwcześniejsze odprężenie lub tąpnięcie

'''Pętla obliczająca wczeniej stworzone cechy'''
for i, row in dane['czas'][10:max(idx)+1].iteritems(): #max(idx)+1 - błąd związany ze wcześniejszym podejciem
    godz_i = dane['czas'][i]
    j = i-1
    jj = i-1
    jjj = i-1
    k4 = 0
    while (godz_i - dane['czas'][j]).seconds/3600 <= 4:
        k2 = 0
        while (godz_i - dane['czas'][jj]).seconds/3600 <= 2:
            k1 = 0
            while (godz_i - dane['czas'][jjj]).seconds/3600 <= 1:
                if dane['TYP'][jjj] == 'SL':
                    dane['1_0_SL'][i] += 1
                elif dane['TYP'][jjj] == 'W':
                    dane['1_0_W'][i] += 1
                k1+=1
                dane['1_0_srE'][i] += dane['ENG'][jjj]
                jjj-=1
            try:
                dane['1_0_srE'][i] /= k1
            except:
                dane['1_0_srE'][i] = 0
            if dane['TYP'][jj] == 'SL':
                dane['2_1_SL'][i] += 1
            elif dane['TYP'][jj] == 'W':
                dane['2_1_W'][i] += 1
            k2+=1
            dane['2_1_srE'][i] += dane['ENG'][jj]
            jj-=1
        try:
            dane['2_1_srE'][i] /= k2
        except:
            dane['2_1_srE'][i] = 0
        if dane['TYP'][j] == 'SL':
            dane['4_2_SL'][i] += 1
        elif dane['TYP'][j] == 'W':
            dane['4_2_W'][i] += 1
        k4+=1
        dane['4_2_srE'][i] += dane['ENG'][j]
        j-=1
    dane['4_2_srE'][i] /= k4
    

#2_1_srE 1_0_srE - obliczają się źle - do odrzucenia
#dane = dane.drop(['2_1_srE', '1_0_srE'], axis = 1)

'''Utworzenie cech z informacją czy wystąpi odprężenie lub tąpniecie'''
dane['bedzie_do1h'] = 'nie'
dane['bedzie_do2h'] = 'nie'


'''Wstawienie informacji czy za godzinę lub dwie nastąpi tąpnięcie bądź odprężenie'''
for i, row in dane['czas'][10:max(idx)+1].iteritems():
    godz_i = dane['czas'][i]
    dane['poprzedni_typ'][i] = dane['TYP'][i-1]
    if dane['TYP'][i] in ['T', 'O']:
        wst = True
    else:
        wst = False
    j = i-1
    while (godz_i - dane['czas'][j]).seconds/3600 <= 2:
        if wst:
            dane['bedzie_do2h'][j] = 'tak'
        j -= 1
        
    j = i-1
    while (godz_i - dane['czas'][j]).seconds/3600 <= 1:
        if wst:
            dane['bedzie_do1h'][j] = 'tak'
        j -= 1
        
'''Pozostawienie obserwacji z obliczonymi nowymi cechami'''
dane = dane[10:max(idx)+1]
dane = dane.reset_index()

#dane.to_csv('node_modules\\dataset.csv')
