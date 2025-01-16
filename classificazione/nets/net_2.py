import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import StepLR
from pytorch_model_summary import summary

# Modified Network with Batch Normalization
class Net(nn.Module):

    def __init__(self, classes: list[str], dropout: float = 0.5) -> None:
        super(Net, self).__init__()

        # Feature extraction layers with Batch Normalization
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),  # Batch Normalization
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(192),  # Batch Normalization
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(384),  # Batch Normalization
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),  # Batch Normalization
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),  # Batch Normalization
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        # Pooling layer
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))

        # Fully connected layers with Dropout
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, len(classes)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def get_transforms(self):
        return transforms.Compose([
            transforms.Resize(256),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
            transforms.RandomRotation(30),  # Adding random rotation
            transforms.RandomGrayscale(p=0.1),  # Random grayscale
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])


if __name__ == '__main__':

    batch_size = 16
    channels = 3
    width = 224
    height = 224

    # Define input size for the network
    input_vehicles = (batch_size, channels, width, height)

    # Instantiate the model with class labels
    n = Net(['cars', 'bikes', 'motorcycles'])

    # Initialize the optimizer with Adam
    optimizer = optim.Adam(n.parameters(), lr=0.001, weight_decay=0.0005)  # Weight decay for regularization

    # Learning rate scheduler (StepLR)
    scheduler = StepLR(optimizer, step_size=7, gamma=0.1)

    # Save the trained model state
    torch.save(n.state_dict(), './out/model_state_dict.pth')

    # Save the entire model
    torch.save(n, './out/model.pth')

    # Print model summary
    print(n)
    print(summary(n, torch.ones(size=input_vehicles)))



#codice precedente test
"""import torch
import torch.nn as nn
import torchvision.transforms as transforms

from pytorch_model_summary import summary


# Rete di classificazione immagini.
class Net(nn.Module):

    def __init__(self, classes : list[str], dropout: float = 0.5) -> None:
        super(Net, self).__init__()

        # Raccolta di layer per l'estrazione delle caratteristiche.
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        # Layer di pooling.
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        
        # Raccolta di layer densi per la classificazione.
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, len(classes)),
        )

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    
    def get_transforms(self):
     return transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224),  
        transforms.RandomHorizontalFlip(),  # Riflessione orizzontale
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),  # Variazione di colore
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    
if __name__ == '__main__':

    batch_size  = 16
    channels    = 3
    width       = 224
    height      = 224

    # Definisce la dimensione dell'input che avra' la rete.
    input_vehicles = (batch_size, channels, width, height)
    
    # Crea l'oggetto che rappresenta la rete.
    # Fornisce le classi.
    n = Net(['cars', 'bikes', 'motorcycles'])
    
    # Salva i parametri addestrati della rete.
    torch.save(n.state_dict(), './out/model_state_dict.pth')
    
    # Salva l'intero modello.
    torch.save(n, './out/model.pth')
    
    # Stampa informazioni generali sul modello.
    print(n)

    # Stampa i parametri addestrabili.
    # for name, param in n.named_parameters():
    #     if param.requires_grad:
    #         print(name, param.data)

    # Stampa un recap del modello.
    print(summary(n, torch.ones(size=input_vehicles)))"""