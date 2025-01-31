import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Dados (substitua pelos seus dados reais)
data = {
    'Modelo': ['Lenet-5', 'AlexNet', 'ResNet34', 'GoogLeNet', 'MobileNet'],
    'Emissão de Carbono': [0.93286, 6.34910, 6.46300, 4.37320, 1.46991],
    'Consumo de Energia': [49.60409, 95.36337, 87.37495, 75.87055, 66.01626],
    'TempoTreino': [132.05211, 546.07588, 667.21660, 626.06714, 424.10751],
    'Medida-F1': [0.98682, 0.99535, 0.99126, 0.99234, 0.96869]
}
df = pd.DataFrame(data)

modelos = df['Modelo']
xpos = np.arange(len(modelos))

# Cria a figura e os eixos
fig, ax1 = plt.subplots(figsize=(10, 6))

# Cria o gráfico de barras para a emissão de carbono
ax1.bar(xpos, df['Emissão de Carbono'], label='Emissão de Carbono', color='skyblue')
ax1.set_xlabel('Modelo')
ax1.set_ylabel('Emissão de Carbono', color='skyblue')
ax1.tick_params(axis='y', labelcolor='skyblue')
plt.xticks(xpos, modelos)

# Cria um segundo eixo y para a Medida-F1
ax2 = ax1.twinx()
ax2.plot(xpos, df['Medida-F1'], label='Medida-F1', color='red', marker='o') #marker adiciona pontos no gráfico de linhas
ax2.set_ylabel('Medida-F1', color='red')
ax2.tick_params(axis='y', labelcolor='red')

#combina as legendas dos dois eixos
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines1 + lines2, labels1 + labels2, loc=0)

# Adiciona título e grid
plt.title('Emissão de Carbono e Medida-F1 por Modelo')
plt.grid(axis='y', alpha=0.3) #grid apenas no eixo y para melhor visualização

# Mostra o gráfico
plt.show()

#Repete o código para consumo de energia e tempo de treino
# Cria a figura e os eixos
fig, ax1 = plt.subplots(figsize=(10, 6))

# Cria o gráfico de barras para o Consumo de Energia
ax1.bar(xpos, df['Consumo de Energia'], label='Consumo de Energia', color='lightgreen')
ax1.set_xlabel('Modelo')
ax1.set_ylabel('Consumo de Energia', color='lightgreen')
ax1.tick_params(axis='y', labelcolor='lightgreen')
plt.xticks(xpos, modelos)

# Cria um segundo eixo y para a Medida-F1
ax2 = ax1.twinx()
ax2.plot(xpos, df['Medida-F1'], label='Medida-F1', color='blue', marker='o') #marker adiciona pontos no gráfico de linhas
ax2.set_ylabel('Medida-F1', color='blue')
ax2.tick_params(axis='y', labelcolor='blue')

#combina as legendas dos dois eixos
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines1 + lines2, labels1 + labels2, loc=0)

# Adiciona título e grid
plt.title('Consumo de Energia e Medida-F1 por Modelo')
plt.grid(axis='y', alpha=0.3) #grid apenas no eixo y para melhor visualização

# Mostra o gráfico
plt.show()

# Cria a figura e os eixos
fig, ax1 = plt.subplots(figsize=(10, 6))

# Cria o gráfico de barras para o Tempo de Treino
ax1.bar(xpos, df['TempoTreino'], label='Tempo de Treino', color='lightcoral')
ax1.set_xlabel('Modelo')
ax1.set_ylabel('Tempo de Treino', color='lightcoral')
ax1.tick_params(axis='y', labelcolor='lightcoral')
plt.xticks(xpos, modelos)

# Cria um segundo eixo y para a Medida-F1
ax2 = ax1.twinx()
ax2.plot(xpos, df['Medida-F1'], label='Medida-F1', color='purple', marker='o') #marker adiciona pontos no gráfico de linhas
ax2.set_ylabel('Medida-F1', color='purple')
ax2.tick_params(axis='y', labelcolor='purple')

#combina as legendas dos dois eixos
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines1 + lines2, labels1 + labels2, loc=0)

# Adiciona título e grid
plt.title('Tempo de Treino e Medida-F1 por Modelo')
plt.grid(axis='y', alpha=0.3) #grid apenas no eixo y para melhor visualização

# Mostra o gráfico
plt.show()