import torch.nn as nn
import graphviz
import os

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3) # Entrada com 1 canal (escala de cinza)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.dropout1 = nn.Dropout2d(0.5) # Dropout2d para camadas convolucionais
        self.fc1 = nn.Linear(64 * 12 * 12, 64) # Ajuste no tamanho da entrada da camada linear
        self.dropout2 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = nn.functional.relu(x)
        x = self.conv2(x)
        x = nn.functional.relu(x)
        x = nn.functional.max_pool2d(x, 2) # Max Pooling com stride 2 (implícito se kernel_size=2)
        x = self.dropout1(x)
        x = x.view(-1, 64 * 12 * 12)  # Flatten
        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return x

# Cria o modelo
modelo = LeNet()

# Cria o grafo do Graphviz
dot = graphviz.Digraph(comment='LeNet Architecture', graph_attr={'rankdir': 'TB'})

# Função auxiliar para adicionar blocos (sem alterações)
def adicionar_bloco(nome, label, anterior=None):
    dot.node(nome, label, shape='box')
    if anterior:
        dot.edge(anterior, nome)
    return nome

# Adiciona os blocos conceituais (agrupados com subgrafos e rank=same)
with dot.subgraph(name='cluster_input') as c:
    c.attr(rank='same')
    anterior = adicionar_bloco('input', 'Entrada (28x28x1)')

with dot.subgraph(name='cluster_conv1') as c:
    c.attr(rank='same')
    anterior = adicionar_bloco('bloco_conv1', 'Bloco Convolucional 1\n(Conv + ReLU)', anterior)

with dot.subgraph(name='cluster_conv2') as c:
    c.attr(rank='same')
    anterior = adicionar_bloco('bloco_conv2', 'Bloco Convolucional 2\n(Conv + ReLU + MaxPool)', anterior)

with dot.subgraph(name='cluster_dropout1') as c:
    c.attr(rank='same')
    anterior = adicionar_bloco('dropout1', 'Dropout (25%)', anterior)

with dot.subgraph(name='cluster_flatten') as c:
    c.attr(rank='same')
    anterior = adicionar_bloco('flatten', 'Flatten', anterior)

with dot.subgraph(name='cluster_fc1') as c:
    c.attr(rank='same')
    anterior = adicionar_bloco('bloco_fc1', 'Bloco Totalmente Conectado 1\n(Linear + ReLU)', anterior)

with dot.subgraph(name='cluster_dropout2') as c:
    c.attr(rank='same')
    anterior = adicionar_bloco('dropout2', 'Dropout (50%)', anterior)

with dot.subgraph(name='cluster_fc2') as c:
    c.attr(rank='same')
    adicionar_bloco('bloco_fc2', 'Bloco Totalmente Conectado 2\n(Softmax)', anterior)

def criar_diretorio_incrementado(diretorio_base, nome_subpasta):
    contador = 1
    diretorio_pai = os.path.join(diretorio_base, f"{nome_subpasta}_{contador}")
    while os.path.exists(diretorio_pai):
        contador += 1
        diretorio_pai = os.path.join(diretorio_base, f"{nome_subpasta}_{contador}")
    os.makedirs(diretorio_pai)
    return diretorio_pai

# Salva o diagrama
diretorio_pai = criar_diretorio_incrementado('Arch_lenet', 'lenetDiagrama')
nome_arquivo = os.path.join(diretorio_pai, "lenet_diagram_blocos_alinhados")
dot.render(nome_arquivo, view=True)

print(f"Diagrama salvo em: {nome_arquivo}.pdf")