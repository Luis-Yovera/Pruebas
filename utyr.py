# Importar dependencias
import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable

# Definir Hiperparámetros
input_size = 784
hidden_size = 128
num_classes = 10
num_epochs = 5
batch_size = 100
lr = 1e-3

# Descargando la base de datos mnist
train_data = dsets.FashionMNIST(root = './data', train = True,
                        transform = transforms.ToTensor(), download = True)

test_data = dsets.FashionMNIST(root = './data', train = False,
                       transform = transforms.ToTensor())

# Leyendo la data
train_gen = torch.utils.data.DataLoader(dataset = train_data,
                                             batch_size = batch_size,
                                             shuffle = True)

test_gen = torch.utils.data.DataLoader(dataset = test_data,
                                      batch_size = batch_size,
                                      shuffle = False)

# Definir modelo
class Net(nn.Module):
    def _init_(self, input_size, hidden_size, num_classes):
        super()._init_()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(0.2)
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.drop(out)
        out = self.fc2(out)
        return out

# Instancia del modelo
net = Net(input_size, hidden_size, num_classes)

# Verificar si CUDA está disponible, sino usar CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net.to(device)

# Compilación
loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=lr)

# Entrenamiento
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_gen):
        # Mover las imágenes y las etiquetas al dispositivo (GPU o CPU)
        images = images.view(-1, 28*28).to(device)
        labels = labels.to(device)

        # Forward, backward y optimización
        optimizer.zero_grad()
        outputs = net(images)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()

        # Imprimir cada 100 pasos
        if (i + 1) % 100 == 0:
            print(f'Epoca [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_data) // batch_size}], Loss: {loss.item():.4f}')

# Evaluación en el conjunto de prueba
correct = 0
total = 0
with torch.no_grad():  # Desactivar cálculo del gradiente para la evaluación
    for images, labels in test_gen:
        images = images.view(-1, 28*28).to(device)
        labels = labels.to(device)

        output = net(images)
        _, predicted = torch.max(output, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

# Imprimir la precisión
print(f'Accuracy: {100 * correct / total:.3f} %')