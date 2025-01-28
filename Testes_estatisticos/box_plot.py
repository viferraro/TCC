import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Dados (substitua pelos seus dados reais)
data = {
    'Modelo': ['Lenet-5', 'AlexNet', 'ResNet34', 'GoogLeNet', 'MobileNet'],
    'Emissão de Carbono': [0.93286, 6.34910, 6.46300, 4.37320, 1.46991],
    'Medida-F1': [0.98682, 0.99535, 0.99126, 0.99234, 0.96869]
}
df = pd.DataFrame(data)

# Cria a coluna 'Grupo' com base nos modelos
df['Grupo'] = df['Modelo'].apply(lambda x: 'Grupo 1' if x in [
    'Lenet-5', 'MobileNet'] else 'Grupo 2')

# Cria os box plots usando Seaborn
# Ajusta o tamanho da figura para melhor visualização
plt.figure(figsize=(10, 6))

# Boxplot para Emissão de Carbono
# Cria um subplot 1 linha, 2 colunas, este é o primeiro gráfico
plt.subplot(1, 2, 1)
ax1 = sns.boxplot(x='Grupo', y='Emissão de Carbono',
            data=df, palette=["skyblue", "lightcoral"])
plt.title('Emissão de Carbono por Grupo')
plt.ylabel('Emissão de Carbono')
ax1.set_xlabel('')  # Remove o rótulo do eixo x

# Boxplot para Medida-F1
# Cria um subplot 1 linha, 2 colunas, este é o segundo gráfico
plt.subplot(1, 2, 2)
ax2 = sns.boxplot(x='Grupo', y='Medida-F1', data=df,
            palette=["skyblue", "lightcoral"])
plt.title('Medida-F1 por Grupo')
plt.ylabel('Medida-F1')
ax2.set_xlabel('')  # Remove o rótulo do eixo x

plt.tight_layout()  # Ajusta o espaçamento entre os subplots
plt.show()
