import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from python_speech_features import logfbank
import librosa

import sys
print('\n'.join(sys.path))
#from memory_profiler import profile
#@profile
def plot_fbank(fbank):
    fig, axes = plt.subplots(nrows=1, ncols=2, sharex=False,
                             sharey=True, figsize=(20,5))
    fig.suptitle('Filter Bank Coefficients', size=16)
    k = 0
    for x in range(1):
        for y in range(2):
            axes[x,y].set_title(list(fbank.keys())[k])
            axes[x,y].imshow(list(fbank.values())[k],
                    cmap='hot', interpolation='nearest')
            axes[x,y].get_xaxis().set_visible(False)
            axes[x,y].get_yaxis().set_visible(False)
            k += 1

#@profile()
def envelope(y, rate, threshold):
    mask = []
    y = pd.Series(y).apply(np.abs)
    y_mean = y.rolling(window = int(rate/10), min_periods=1, center = True).mean()
    for mean in y_mean:
        if mean > threshold:
            mask.append(True)
        else:
            mask.append(False)
    return mask

def Average(lst):
    return sum(lst) / len(lst)

def minimum(lst):
    return min(lst)

def maximum(lst):
    return max(lst)

def quartile2(lst):
    return np.percentile(lst, 25)

def quartile4(lst):
    return np.percentile(lst, 75)

from decimal import Decimal
import statistics

def median(lst):
    return statistics.median(map(Decimal, lst))


#@profile
def learn12(filename):
    import math
    import pandas as pd2
    signals = {}
    fft = {}
    fbank = {}
    mfccs = {}

    score = 0.0

    #for f in dfTest.index:
    Aedes = 0
    Anopheles = 0
    Culex = 0
    wav_file = filename #dfTest.File[f]
    #classification = dfTest.Label[f]
    signal, rate = librosa.load(wav_file, sr=44100)

    mask = envelope(signal, rate, 0.0005)
    signal = signal[mask]
    signals[0] = signal

    bank = logfbank(signal[:rate], rate, nfilt=26, nfft=1103).T
    fbank[0] = bank
    flist = list(fbank[0])
    print(flist)

    p1 = 0
    p2 = 0
    p3 = 0

    # getting length of list
    length = len(flist)

    import pickle
    PIK = "pickle.dat"

    with open(PIK, 'rb') as f:
        var = pickle.load(f)
        anopheles_min = var[14] #6
        anopheles_max = var[15] #7
        anopheles_median = var[5]
        anopheles_mean = var[4]
        culex_min = var[16] #10
        culex_max = var[17] #11
        culex_median = var[9]
        culex_mean = var[8]
        aedes_min = var[12] #2
        aedes_max = var[13] #3
        aedes_median = var[1]
        aedes_mean = var[0]
        aedes_2q = var[12]
        aedes_4q = var[13]
        anopheles_2q = var[14]
        anopheles_4q = var[15]
        culex_2q = var[16]
        culex_4q = var[17]


    for i in range(length):

        #RANGE SCORE
        RI_anopheles_count = (math.sqrt((Decimal(maximum(flist[i])) - Decimal(anopheles_max[i])) ** 2)) + (
            math.sqrt((Decimal(minimum(flist[i])) - Decimal(anopheles_min[i])) ** 2))
        RI_culex_count = (math.sqrt((Decimal(maximum(flist[i])) - Decimal(culex_max[i])) ** 2)) + (
            math.sqrt((Decimal(minimum(flist[i])) - Decimal(culex_min[i])) ** 2))
        RI_aedes_count = (math.sqrt((Decimal(maximum(flist[i])) - Decimal(aedes_max[i])) ** 2)) + (
            math.sqrt((Decimal(minimum(flist[i])) - Decimal(aedes_min[i])) ** 2))

        if(maximum(flist[i]) > anopheles_max[i]+ (anopheles_max[i]*0.5)):
            RI_anopheles_count = RI_anopheles_count + 1
            p1 = p1 + 1

        if(minimum(flist[i]) < anopheles_min[i]+ (anopheles_min[i]*0.5)):
            RI_anopheles_count = RI_anopheles_count + 1
            p1 = p1 + 1

        if(maximum(flist[i]) > culex_max[i]+ (culex_max[i]*0.5)):
            RI_culex_count = RI_culex_count + 1
            p2 = p2 + 1

        if(minimum(flist[i]) < culex_min[i]+ (culex_min[i]*0.5)):
            RI_culex_count = RI_culex_count + 1
            p2 = p2 + 1

        if(maximum(flist[i]) > aedes_max[i]+ (aedes_max[i]*0.5)):
            RI_aedes_count = RI_aedes_count + 1
            p3 = p3 + 1

        if(minimum(flist[i]) < aedes_min[i]+ (aedes_min[i]*0.5)):
            RI_aedes_count = RI_aedes_count + 1
            p3 = p3 + 1

        #NORMALIZE RANGE SCORE
        RI_anopheles = (RI_anopheles_count) / (RI_anopheles_count + RI_culex_count + RI_aedes_count)
        RI_culex = (RI_culex_count) / (RI_anopheles_count + RI_culex_count + RI_aedes_count)
        RI_aedes = (RI_aedes_count) / (RI_anopheles_count + RI_culex_count + RI_aedes_count)

        #MEDIAN SCORE
        median_dist_anopheles = (math.sqrt((Decimal(median(flist[i])) - Decimal(anopheles_median[i])) ** 2))
        median_dist_culex = (math.sqrt((Decimal(median(flist[i])) - Decimal(culex_median[i])) ** 2))
        median_dist_aedes = (math.sqrt((Decimal(median(flist[i])) - Decimal(aedes_median[i])) ** 2))

        #if (median(flist[i]) > (Decimal(anopheles_median[i]) + Decimal(anopheles_median[i]) * Decimal('0.5'))):
        #    p1 = p1 + 1
        #if (median(flist[i]) > (Decimal(culex_median[i]) + Decimal(culex_median[i]) * Decimal('0.5'))):
        #    p2 = p2 + 1
        #if (median(flist[i]) > (Decimal(aedes_median[i]) + Decimal(aedes_median[i]) * Decimal('0.5'))):
        #    p3 = p3 + 1

        #NORMALIZE MEDIAN SCORE
        median_anopheles = (median_dist_anopheles) / (median_dist_anopheles + median_dist_culex + median_dist_aedes)
        median_culex = (median_dist_culex) / (median_dist_anopheles + median_dist_culex + median_dist_aedes)
        median_aedes = (median_dist_aedes) / (median_dist_anopheles + median_dist_culex + median_dist_aedes)

        #MEAN SCORE
        mean_dist_anopheles = (math.sqrt((Decimal(Average(flist[i])) - Decimal(anopheles_mean[i])) ** 2))
        mean_dist_culex = (math.sqrt((Decimal(Average(flist[i])) - Decimal(culex_mean[i])) ** 2))
        mean_dist_aedes = (math.sqrt((Decimal(Average(flist[i])) - Decimal(aedes_mean[i])) ** 2))

        #NORMALIZE MEAN SCORE
        mean_anopheles = mean_dist_anopheles / (mean_dist_anopheles + mean_dist_culex + mean_dist_aedes)
        mean_culex = mean_dist_culex / (mean_dist_anopheles + mean_dist_culex + mean_dist_aedes)
        mean_aedes = mean_dist_aedes / (mean_dist_anopheles + mean_dist_culex + mean_dist_aedes)

        # df3 = pd2.DataFrame(data={'Specie': ['Anopheles', 'Culex', 'Aedes'],'Range': [RI_anopheles_count, RI_culex_count, RI_aedes_count],'Median': [median_anopheles, median_culex, median_aedes],'Mean': [mean_anopheles, mean_culex, mean_aedes], 'Total': [0,0,0]})
        df3 = pd2.DataFrame(
            data={'Specie': ['Anopheles', 'Culex', 'Aedes'],
                    'Range': [RI_anopheles, RI_culex, RI_aedes],
                    'Median': [median_anopheles, median_culex, median_aedes],
                    'Mean': [mean_anopheles, mean_culex, mean_aedes],
                    'Total': [0, 0, 0]})

        df3['Range'] = df3['Range']
        df3['Median'] = df3['Median']
        df3['Mean'] = 0  # df3['Mean']

        df3['Total'] = (df3['Range'] + df3['Median'] + df3['Mean']) / 2
            # df3['Total'] = ((((df3['Range'].rank(ascending=True))/3) * 100) *  Range_Weight) + ((((df3['Median'].rank(ascending=False))/3) * 100) * Median_Weight) + ((((df3['Mean'].rank(ascending=False))/3) * 100) * Mean_Weight)
            # print(df3)

        Prediction_Col = df3.loc[df3['Total'].idxmin()]
        print(i)
        print(df3)

        Aedes = Aedes + df3.iloc[2]['Total']
        Anopheles = Anopheles + df3.iloc[0]['Total']
        Culex = Culex + df3.iloc[1]['Total']

    if (Aedes <= Anopheles and Aedes <= Culex):
        FinalPrediction = 'Aedes'
    if (Anopheles <= Aedes and Anopheles <= Culex):
        FinalPrediction = 'Anopheles'
    if (Culex <= Aedes and Culex <= Anopheles):
        FinalPrediction = 'Culex'

    print('Aedes:' + str(Aedes))
    print('Penalty: ' + str(p3))
    print('Anopheles:' + str(Anopheles))
    print('Penalty: ' + str(p1))
    print('Culex:' + str(Culex))
    print('Penalty: ' + str(p2))
    print(FinalPrediction)

    return FinalPrediction

#instance = '2da9e12a.wav'
#instance = 'PR1.wav'
#instance = 'AES3.wav'
#instance = '01c2f88b.wav'
instance = '3a9085ca.wav'
learn12(instance)

