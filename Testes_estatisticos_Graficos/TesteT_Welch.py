import pandas as pd
import scipy.stats as stats

# Dados (substitua pelos seus dados reais)
data = {
    'Modelo': ['Lenet-5', 'AlexNet', 'ResNet34', 'GoogLeNet', 'MobileNet'],
    'Consumo de Energia': [49.60409, 95.36337, 87.37495, 75.87055, 66.01626],
    'Medida-F1': [0.98682, 0.99535, 0.99126, 0.99234, 0.96869]
}
df = pd.DataFrame(data)

# Separa os dados em grupos
leves = df[df['Modelo'].isin(['Lenet-5', 'MobileNet'])]
complexas = df[df['Modelo'].isin(['AlexNet', 'ResNet34', 'GoogLeNet'])]

# Lista as métricas que serão analisadas
metrics = ['Consumo de Energia', 'Medida-F1']

# Itera sobre as métricas e realiza o Teste t de Welch para cada uma
for metric in metrics:
    # Verifica se há dados suficientes em ambos os grupos para realizar o teste
    if len(leves[metric]) > 0 and len(complexas[metric]) > 0:
        # Realiza o Teste t de Welch
        t, p = stats.ttest_ind(leves[metric], complexas[metric], equal_var=False)  # equal_var=False para Welch
        print(f"Teste t de Welch para {metric}: t = {t}, p = {p}")
    else:
        print(f"Não há dados suficientes em um dos grupos para calcular o Teste t de Welch para {metric}.")