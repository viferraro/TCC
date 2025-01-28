import pandas as pd
import scipy.stats as stats

# Dados (substitua pelos seus dados reais)
data = {
    'Modelo': ['Lenet-5', 'AlexNet', 'ResNet34', 'GoogLeNet', 'MobileNet'],
    'Emissão de Carbono': [0.93286, 6.34910, 6.46300, 4.37320, 1.46991],
    'Consumo de Energia': [49.60409, 95.36337, 87.37495, 75.87055, 66.01626],
    'Medida-F1': [0.98682, 0.99535, 0.99126, 0.99234, 0.96869],
    'TempoTreino': [132.05211, 546.07588, 667.21660, 626.06714, 424.10751]
}
df = pd.DataFrame(data)

# Separa os dados em grupos
leves = df[df['Modelo'].isin(['Lenet-5', 'MobileNet'])]
complexas = df[df['Modelo'].isin(['AlexNet', 'ResNet34', 'GoogLeNet'])]

metrics = ['Emissão de Carbono', 'Consumo de Energia', 'Medida-F1', 'TempoTreino']
for metric in metrics:
    # Realiza o teste de Mann-Whitney
    u, p = stats.mannwhitneyu(leves[metric], complexas[metric])
    print(f"Teste de Mann-Whitney para {metric}: U = {u}, p = {p}")