import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Dados (substitua pelos seus dados reais)
data = {
    'Modelo': ['Lenet-5', 'AlexNet', 'ResNet34', 'GoogLeNet', 'MobileNet'],
    'Medida-F1': [0.98682, 0.99535, 0.99126, 0.99234, 0.96869]
}
df = pd.DataFrame(data)

modelos = df['Modelo']
xpos = np.arange(len(modelos))

# Cria a figura e os eixos
fig, ax1 = plt.subplots(figsize=(10, 6))

# Cria o gráfico de barras para a medida_f1
ax1.bar(xpos, df['Medida-F1'], label='Medida-F1', color='dodgerblue')
ax1.set_xlabel('Modelo')
ax1.set_ylabel('Medida-F1', color='dodgerblue')
ax1.tick_params(axis='y', labelcolor='dodgerblue')
plt.xticks(xpos, modelos)

# Definir o intervalo do eixo y de 0.9 a 1
ax1.set_ylim(0.9, 1)

# Adiciona título e grid
plt.title('Medida-F1 por Modelo')
plt.grid(axis='y', alpha=0.05)

plt.show()
