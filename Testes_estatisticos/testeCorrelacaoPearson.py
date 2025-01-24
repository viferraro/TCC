import pandas as pd
import scipy.stats as stats

# Dados (substitua pelos seus dados reais)
data = {
    'Modelo': ['Lenet-5', 'AlexNet', 'ResNet34', 'GoogLeNet', 'MobileNet'],
    'Emissão de Carbono': [0.93286, 6.34910, 6.46300, 4.37320, 1.46991],
    'Consumo de Energia': [49.60409, 95.36337, 87.37495, 75.87055, 66.01626],
    'NumParam': [609354, 23271114, 21292042, 5977530, 24618],
    'FLOPs': [11401984, 147910912, 69921536, 36235520, 8054656],
    'TempoTreino': [132.05211, 546.07588, 667.21660, 626.06714, 424.10751]
}
df = pd.DataFrame(data)

# Calcula a correlação entre Consumo de Energia e NumParam
correlation_np = stats.pearsonr(df['Consumo de Energia'], df['NumParam'])
print(f"Correlação entre Consumo de Energia e NumParam: {correlation_np}")

# Calcula a correlação entre Consumo de Energia e FLOPs
correlation_flops = stats.pearsonr(df['Consumo de Energia'], df['FLOPs'])
print(f"Correlação entre Consumo de Energia e FLOPs: {correlation_flops}")

# Calcula a correlação entre Consumo de Energia e TempoTreino
correlation_tt = stats.pearsonr(df['Consumo de Energia'], df['TempoTreino'])
print(f"Correlação entre Consumo de Energia e TempoTreino: {correlation_tt}")

# Calcula a correlação entre Consumo de Energia e Emissão de Carbono
correlation_ec = stats.pearsonr(df['Consumo de Energia'], df['Emissão de Carbono'])
print(f"Correlação entre Consumo de Energia e Emissão de Carbono: {correlation_ec}")
