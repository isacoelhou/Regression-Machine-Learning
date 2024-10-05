import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

def grid_search_KNR():

    maior = 10000000000
    for j in ("distance","uniform"):
        for i in range (1,50):

            KNN = KNeighborsRegressor(n_neighbors=i,weights=j)
            KNN.fit(x_treino, y_treino)

            opiniao = KNN.predict(x_validacao)

            mae = mean_absolute_error(y_validacao, opiniao)
            mse = mean_squared_error(y_validacao, opiniao)
            rmse = np.sqrt(mse)

            media = (mae + mse + rmse)/3

            if(media < maior):
                best_n = i
                best_w = j

    return best_n, best_w


rmse_KNN = []
mse_KNN = []
mae_KNN = []

for _ in range(20):

    dados = pd.read_excel("../Dataset/ResidencialBuilding.xlsx")

    dados = shuffle(dados)

    X = dados.iloc[:, :-2]  
    Y = dados.iloc[:, 2:]  # Outputs

    x_treino, x_temp, y_treino, y_temp = train_test_split(X, Y, test_size=0.5)
    x_validacao, x_teste, y_validacao, y_teste = train_test_split(x_temp, y_temp, test_size=0.5)

    i, j = grid_search_KNR()
    KNN = KNeighborsRegressor(n_neighbors=i,weights=j)
    KNN.fit(x_treino,y_treino)

    opiniao = KNN.predict(x_teste)

    mae_KNN.append(mean_absolute_error(y_validacao, opiniao))
    mse_KNN.append(mean_squared_error(y_validacao, opiniao))
    rmse_KNN.append(np.sqrt(mean_squared_error(y_validacao, opiniao)))