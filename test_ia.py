import time
import torch
import torch.nn as nn
from Fonction.Versionning_Control import Version_Control
torch.set_printoptions(threshold=1000)

calcul = "GPU"
nombredimagesfinales = 10000
batch_size = 96
neuronne = 128
nombre_epochs = 1

model = nn.Sequential(                          # Modèle séquentiel : les données traversent les couches dans l'ordre
    nn.Flatten(),                               # Aplatit l'image 28x28 en un vecteur de 784 valeurs
    nn.Linear(28*28, neuronne),                 # Couche linéaire : 784 entrées -> 128 sorties (somme pondérée + biais)
    nn.ReLU(),                                  # Activation : met les valeurs négatives à 0 (apporte la non-linéarité)
    nn.Linear(neuronne, 10)                     # Couche finale : 128 -> 10 logits (un score par chiffre 0 à 9)
)

Version_Control()

from torchvision import datasets, transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if calcul == "GPU":
    device = torch.device("cuda")
if calcul == "CPU":
    device = torch.device("cpu")


transform = transforms.ToTensor()              # Convertit chaque image en tensor, valeurs normalisées entre 0 et 1
train_data = datasets.MNIST(root='.', train=True, download=True, transform=transform)
                                               # Jeu d'entraînement MNIST : 60 000 images étiquetées
loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
                                               # Découpe en batchs de 96 ; shuffle=True = ordre mélangé à chaque époque
def test_model():
    reussite = 0
    for idx in range(nombredimagesfinales):        # Phase de test : évalue le modèle sur des images
        image, true_label = train_data[idx]
        image = image.to(device)
        pred = model(image.unsqueeze(0)).argmax().item()
                                                # Prédiction : unsqueeze(0) ajoute la dimension batch ; argmax() = chiffre au plus grand logit
        if pred == true_label:                     # Compare la prédiction à la vraie réponse
            reussite += 1
    pourcentage_reussite = (reussite / nombredimagesfinales) * 100
    return pourcentage_reussite
 
model.to(device)                               # Déplace le modèle sur le GPU

optimizer = torch.optim.Adam(model.parameters())
                                               # Optimiseur Adam : ajuste les poids du modèle ; model.parameters() = tous les poids/biais à entraîner
loss_fn = nn.CrossEntropyLoss()                # Fonction de perte : mesure l'écart prédiction / vraie réponse (classification)
temp_actuelle = time.time()

images_all = train_data.data.float().unsqueeze(1) / 255.0                 # Normalisation des images : valeurs entre 0 et 1
labels_all = train_data.targets
images_all = images_all.to(device)
labels_all = labels_all.to(device)

n = images_all.shape[0]  # Nombre total d'images dans le jeu de données

for epoch in range(nombre_epochs):
    perm = torch.randperm(n, device=device)  # Mélange aléatoire des indices des images
    for start in range(0, n, batch_size):
        idx = perm[start:start + batch_size]  # Sélectionne un batch d'indices
        images, labels = images_all[idx], labels_all[idx]  # Récupère les images et labels correspondants
        optimizer.zero_grad()  # Réinitialise les gradients des poids
        output = model(images)  # Passe les images dans le modèle pour obtenir les prédictions
        loss = loss_fn(output, labels)  # Calcule la perte entre les prédictions
        loss.backward()  # Rétropropagation : calcule les gradients des poids
        optimizer.step()  # Met à jour les poids du modèle en fonction des gradients calculés

temp_aprentissage = time.time() - temp_actuelle

temp_actuelle = time.time()


temp_test = time.time() - temp_actuelle


print("\n")
print(f"Taux de réussite: {reussite}/{nombredimagesfinales} donc : ({pourcentage_reussite:.2f}%)")
print(f"Temps d'entraînement : {temp_aprentissage:.2f} secondes")
print(f"Temps de test : {temp_test:.2f} secondes")
print("\n")