import pandas as pd
import csv
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression

def grid_search_KNR():

    maior = 10000000000
    for j in ("distance","uniform"):
        for i in range (1,50):

            KNR = KNeighborsRegressor(n_neighbors=i,weights=j)
            KNR.fit(x_treino, y_treino)

            opiniao = KNR.predict(x_validacao)

            rmse = np.sqrt( mean_squared_error(y_validacao, opiniao))

            if(rmse < maior):
                maior = rmse
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

            rmse = np.sqrt(mean_squared_error(y_validacao, opiniao))

            if(rmse < maior):
                maior = rmse
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

              MLP.fit(x_treino, y_treino)
              opiniao = MLP.predict(x_validacao)

              rmse = np.sqrt( mean_squared_error(y_validacao, opiniao))

              if(rmse < maior):
                maior = rmse
                Melhor_i = i
                Melhor_j = j
                Melhor_k = k
                Melhor_l = l

    return Melhor_i, Melhor_j, Melhor_k, Melhor_k, Melhor_l

def grid_search_RF():
    maior = 1000000

    for n_estimators in range(100, 1000, 100):
        for criterion in ('squared_error', 'absolute_error', 'friedman_mse', 'poisson'):
            for max_deaph in range(2, 10, 1):
                for min_sample_split in (2, 3, 4, 5):
                    for min_sample_leaf in (1, 2, 3, 4):
                        RF = RandomForestRegressor(n_estimators = n_estimators, criterion = criterion, max_depth = max_deaph,
                        min_samples_split = min_sample_split, min_samples_leaf = min_sample_leaf)

                        RF.fit(x_treino, y_treino)
                        opiniao = RF.predict(x_validacao)

                        rmse = np.sqrt( mean_squared_error(y_validacao, opiniao))

                        if(rmse < maior):
                            maior = rmse
                            melhor_n_estimator = n_estimators
                            melhor_criterion = criterion
                            melhor_max_deaph = max_deaph
                            melhor_min_sample_split = min_sample_split
                            melhor_min_sample_leaf = min_sample_leaf

    return melhor_n_estimator, melhor_criterion, melhor_max_deaph, melhor_min_sample_split, melhor_min_sample_leaf

def grid_search_GB():
    maior = 1000000

    for n_estimators in range(100, 1000, 100):
        for loss in ('squared_error', 'absolute_error', 'huber', 'quantile'):
            for max_depth in range(2, 10, 1):
                for learning_rate in range(0.1, 1, 0.2):
                    for min_sample_split in (2, 3, 4, 5):
                        for min_sample_leaf in (1, 2, 3, 4):
                            GB = GradientBoostingRegressor(n_estimators=n_estimators, loss=loss, max_depth=max_depth,
                            learning_rate=learning_rate, min_samples_split=min_sample_split, min_samples_leaf=min_sample_leaf)

                            GB.fit(x_treino, y_treino)
                            opiniao = GB.predict(x_validacao)

                            rmse = np.sqrt( mean_squared_error(y_validacao, opiniao))

                            if(rmse < maior):
                                maior = rmse
                                melhor_n_estimator = n_estimators
                                melhor_loss = loss
                                melhor_max_depth = max_depth
                                melhor_learning_rate = learning_rate
                                melhor_min_sample_split = min_sample_split
                                melhor_min_sample_leaf = min_sample_leaf

    return melhor_n_estimator, melhor_loss, melhor_max_depth, melhor_learning_rate, melhor_min_sample_split, melhor_min_sample_leaf


rmse_KNR = []
mse_KNR = []
mae_KNR = []

rmse_SVR = []
mse_SVR = []
mae_SVR = []

rmse_MLP = []
mse_MLP = []
mae_MLP = []

rmse_RF = []
mse_RF = []
mae_RF = []

rmse_GB = []
mse_GB = []
mae_GB = []

rmse_RLM = []
mse_RLM = []
mae_RLM = []

for iteracao in range(20):

    print(iteracao)

    dados = pd.read_excel("../Dataset/data.xlsx")

    dados = shuffle(dados)

    X = dados.iloc[:,:-1]
    Y = dados.iloc[:,-1]

    x_treino, x_temp, y_treino, y_temp = train_test_split(X, Y, test_size=0.5)
    x_validacao, x_teste, y_validacao, y_teste = train_test_split(x_temp, y_temp, test_size=0.5)

    ############################################################################# 

    print("KNR")   

    i, j = grid_search_KNR()
    KNR = KNeighborsRegressor(n_neighbors=i,weights=j)
    KNR.fit(x_treino,y_treino)

    opiniao = KNR.predict(x_teste)

    mae_KNR.append(mean_absolute_error(y_validacao, opiniao))
    mse_KNR.append(mean_squared_error(y_validacao, opiniao))
    rmse_KNR.append(np.sqrt(mean_squared_error(y_validacao, opiniao)))



    #############################################################################
   
    print("SVR")
    i, j = grid_search_SVR()
    SVR_ = SVR(kernel=i,C=j)
    SVR_.fit(x_treino,y_treino)

    opiniao = SVR_.predict(x_teste)

    mae_SVR.append(mean_absolute_error(y_validacao, opiniao))
    mse_SVR.append(mean_squared_error(y_validacao, opiniao))
    rmse_SVR.append(np.sqrt(mean_squared_error(y_validacao, opiniao)))

    
    #############################################################################
  
    print("MLP")
    i, j, k, l = grid_search_MLP()
    MLP = MLPRegressor(hidden_layer_sizes=(i,i,i), learning_rate=j, max_iter=k, activation=l)
    MLP.fit(x_treino,y_treino)

    opiniao = MLP.predict(x_teste)

    mae_MLP.append(mean_absolute_error(y_validacao, opiniao))
    mse_MLP.append(mean_squared_error(y_validacao, opiniao))
    rmse_MLP.append(np.sqrt(mean_squared_error(y_validacao, opiniao)))



    ##############################################################################
   
    print("RF")
    i, j, k, l, m = grid_search_RF()
    RF = RandomForestRegressor(n_estimators = i, criterion = j, max_depth = k, min_samples_split = l, min_samples_leaf = m)
    RF.fit(x_treino, y_treino)

    opiniao = RF.predict(x_teste)
    mae_RF.append(mean_absolute_error(y_validacao, opiniao))
    mse_RF.append(mean_squared_error(y_validacao, opiniao))
    rmse_RF.append(np.sqrt(mean_squared_error(y_validacao, opiniao)))

  
    ##############################################################################
    
    print("GB")
    i, j, k, l, m, n = grid_search_GB()
    GB = GradientBoostingRegressor(n_estimators=i, loss=j, max_depth=k, learning_rate=l, min_samples_split=m, min_samples_leaf=n)

    GB.fit(x_treino, y_treino)

    opiniao = GB.predict(x_teste)
    mae_GB.append(mean_absolute_error(y_validacao, opiniao))
    mse_GB.append(mean_squared_error(y_validacao, opiniao))
    rmse_GB.append(np.sqrt(mean_squared_error(y_validacao, opiniao)))




    ##############################################################################
    
    print("RLM")
    RLM = LinearRegression()

    RLM.fit(x_treino, y_treino)
    opiniao = RLM.predict(x_teste)
    mae_RLM.append(mean_absolute_error(y_validacao, opiniao))
    mse_RLM.append(mean_squared_error(y_validacao, opiniao))
    rmse_RLM.append(np.sqrt(mean_squared_error(y_validacao, opiniao)))


with open("../Logs/log_rmse.csv", 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    
    # Cabeçalho
    writer.writerow(["Repetição", "KNR", "SVR", "MLP", "RF", "GB", "RLM"])
    
    # Adicionando os RMSE para cada repetição (20 repetições como na tabela)
    for i in range(20):
        writer.writerow([ i + 1,
            format(rmse_KNR[i], ".4f"),
            format(rmse_SVR[i], ".4f"),
            format(rmse_MLP[i], ".4f"),
            format(rmse_RF[i], ".4f"),
            format(rmse_GB[i], ".4f"),
            format(rmse_RLM[i], ".4f")])
        
with open("../Logs/log_mae.csv", 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    
    # Cabeçalho
    writer.writerow(["Repetição", "KNR", "SVR", "MLP", "RF", "GB", "RLM"])
    
    # Adicionando os RMSE para cada repetição (20 repetições como na tabela)
    for i in range(20):
        writer.writerow([ i + 1,
            format(mae_KNR[i], ".4f"),
            format(mae_SVR[i], ".4f"),
            format(mae_MLP[i], ".4f"),
            format(mae_RF[i], ".4f"),
            format(mae_GB[i], ".4f"),
            format(mae_RLM[i], ".4f")])

with open("../Logs/log_mse.csv", 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    
    # Cabeçalho
    writer.writerow(["Repetição", "KNR", "SVR", "MLP", "RF", "GB", "RLM"])
    
    # Adicionando os RMSE para cada repetição (20 repetições como na tabela)
    for i in range(20):
        writer.writerow([ i + 1,
            format(mse_KNR[i], ".4f"),
            format(mse_SVR[i], ".4f"),
            format(mse_MLP[i], ".4f"),
            format(mse_RF[i], ".4f"),
            format(mse_GB[i], ".4f"),
            format(mse_RLM[i], ".4f")])

