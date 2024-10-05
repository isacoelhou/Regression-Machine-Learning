import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import seaborn as sns
import matplotlib.pyplot as plt

dados = pd.read_excel("../Dataset/ResidencialBuilding.xlsx")

dados = shuffle(dados)

X = dados.iloc[:, :-2]  
Y = dados.iloc[:, -2:]  # Outputs

print(dados.info())
print(dados.describe()) 

# plt.figure(figsize=(10, 8))
# sns.heatmap(dados.corr(), annot=True, cmap='coolwarm')
# plt.title('Mapa de Calor da Correlação entre os Atributos')
# plt.show()

x_treino, x_temp, y_treino, y_temp = train_test_split(X, Y, test_size=0.5)
x_validacao, x_teste, y_validacao, y_teste = train_test_split(x_temp, y_temp, test_size=0.5)

