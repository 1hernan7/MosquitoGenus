import pickle

PIK = "pickle.dat"

with open(PIK, 'rb') as f:
    var = pickle.load(f)
    print(var[17])