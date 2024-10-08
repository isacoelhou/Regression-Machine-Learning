import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Carregar a base de dados
df_origin = pd.read_excel("../Dataset/data.xlsx")

df_red = df_origin.iloc[:,:-77]
df = pd.concat([df_red, df_origin['V-9']], axis=1)


# Informações da base de dados
tamanho = df.shape[0]
dimensao = df.shape[1]
tipos_atributos = df.dtypes
valores_medios = df.mean()
valores_maximos = df.max()
valores_minimos = df.min()
correlacao = df.corr()

# Plotar a matriz de correlação
plt.figure(figsize=(20, 16))  # Ajuste o tamanho da figura conforme necessário
sns.heatmap(correlacao, annot=True, cmap='coolwarm', annot_kws={"size": 12}, fmt=".2f", cbar_kws={'shrink': .8})

# Ajustar tamanhos de fonte
plt.title('Matriz de Correlação', fontsize=24)
plt.xlabel('Variáveis', fontsize=18)
plt.ylabel('Variáveis', fontsize=18)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.show()
