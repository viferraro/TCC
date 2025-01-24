import matplotlib.pyplot as plt
import pandas as pd

# Dados (substitua pelos seus dados reais)
data = {
    'Modelo': ['Lenet-5', 'AlexNet', 'ResNet34', 'GoogLeNet', 'MobileNet'],
    'Emissão de Carbono': [0.93286, 6.34910, 6.46300, 4.37320, 1.46991],
    'Medida-F1': [0.98682, 0.99535, 0.99126, 0.99234, 0.96869]
}
df = pd.DataFrame(data)

# Cria o gráfico de dispersão
plt.figure(figsize=(8, 6))  # Ajusta o tamanho da figura
plt.scatter(df['Emissão de Carbono'], df['Medida-F1'])

# Adiciona rótulos e título
plt.xlabel('Emissão de Carbono')
plt.ylabel('Medida-F1')
plt.title('Relação entre Emissão de Carbono e Medida-F1')

# Adiciona anotações para cada ponto (opcional)
for i, txt in enumerate(df['Modelo']):
    plt.annotate(txt, (df['Emissão de Carbono'][i], df['Medida-F1'][i]), textcoords="offset points", xytext=(0,10), ha='center')

# Mostra o gráfico
plt.grid(True) #adiciona grid ao gráfico
plt.show()