import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor


def grid_search_KNR():

    maior = 10000000000
    for j in ("distance","uniform"):
        for i in range (1,50):

            KNR = KNeighborsRegressor(n_neighbors=i,weights=j)
            KNR.fit(x_treino, y_treino)

            opiniao = KNR.predict(x_validacao)

            mae = mean_absolute_error(y_validacao, opiniao)
            mse = mean_squared_error(y_validacao, opiniao)
            rmse = np.sqrt(mse)

            media = (mae + mse + rmse)/3

            if(media < maior):
                best_n = i
                best_w = j

    return best_n, best_w

def grid_search_SVR():
    
    maior = 10000000000

    for k in ("linear", "poly", "rbf", "sigmoid"):
        for i in (0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 1): 

            SVR_ = SVR(kernel=k,C=i)
            SVR_.fit(x_treino, y_treino)

            opiniao = SVR_.predict(x_validacao)

            mae = mean_absolute_error(y_validacao, opiniao)
            mse = mean_squared_error(y_validacao, opiniao)
            rmse = np.sqrt(mse)

            media = (mae + mse + rmse)/3

            if(media < maior):
                best_k = k
                best_i = i

    return best_k, best_i
           
def grid_search_MLP():
  maior = 1000000

  for i in (5,6,10,12):
    for j in ('constant','invscaling', 'adaptive'):
        for k in (50,100,150,300,500,1000):
          for l in ('identity', 'logistic', 'tanh', 'relu'):
              MLP = MLPRegressor(hidden_layer_sizes=(i,i,i), learning_rate=j, max_iter=k, activation=l )

              SVR_.fit(x_treino, y_treino)
              opiniao = SVR_.predict(x_validacao)
              mae = mean_absolute_error(y_validacao, opiniao)
              mse = mean_squared_error(y_validacao, opiniao)
              rmse = np.sqrt(mse)
              
              media = (mae + mse + rmse)/3

              if (media > maior):
                Melhor_i = i
                Melhor_j = j
                Melhor_k = k
                Melhor_l = l

    return Melhor_i, Melhor_j, Melhor_k, Melhor_k, Melhor_l

rmse_KNR = []
mse_KNR = []
mae_KNR = []

rmse_SVR = []
mse_SVR = []
mae_SVR = []

rmse_MLP = []
mse_MLP = []
mae_MLP = []

for _ in range(20):

    dados = pd.read_excel("../Dataset/data.xlsx")

    dados = shuffle(dados)

    X = dados.iloc[:, :-1]  
    Y = dados.iloc[:, :] 

    x_treino, x_temp, y_treino, y_temp = train_test_split(X, Y, test_size=0.5)
    x_validacao, x_teste, y_validacao, y_teste = train_test_split(x_temp, y_temp, test_size=0.5)

    #############################################################################    

    i, j = grid_search_KNR()
    print(i,j)
    KNR = KNeighborsRegressor(n_neighbors=i,weights=j)
    KNR.fit(x_treino,y_treino)

    opiniao = KNR.predict(x_teste)

    mae_KNR.append(mean_absolute_error(y_validacao, opiniao))
    mse_KNR.append(mean_squared_error(y_validacao, opiniao))
    rmse_KNR.append(np.sqrt(mean_squared_error(y_validacao, opiniao)))

    #############################################################################

    i, j = grid_search_SVR()
    SVR_ = KNeighborsRegressor(kernel=i,C=j)
    SVR_.fit(x_treino,y_treino)

    opiniao = SVR_.predict(x_teste)

    mae_SVR.append(mean_absolute_error(y_validacao, opiniao))
    mse_SVR.append(mean_squared_error(y_validacao, opiniao))
    rmse_SVR.append(np.sqrt(mean_squared_error(y_validacao, opiniao)))
    
    #############################################################################

    i, j, k, l = grid_search_MLP()
    MLP = MLPRegressor(hidden_layer_sizes=(i,i,i), learning_rate=j, max_iter=k, activation=l)
    MLP.fit(x_treino,y_treino)

    opiniao = MLP.predict(x_teste)

    mae_MLP.append(mean_absolute_error(y_validacao, opiniao))
    mse_MLP.append(mean_squared_error(y_validacao, opiniao))
    rmse_MLP.append(np.sqrt(mean_squared_error(y_validacao, opiniao)))