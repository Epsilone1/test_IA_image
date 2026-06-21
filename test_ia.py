import time
import torch                                    # Bibliothèque principale PyTorch (tensors, calculs, réseaux de neurones)
import torch.nn as nn                           # Sous-module "neural network" : couches, fonctions de perte... renommé "nn" pour écrire plus court
from Fonction.Versionning_Control import Version_Control

calcul = "GPU"
nombredimagesfinales = 20000  
batch_size = 8
neuronne = 8192

model = nn.Sequential(                          # Crée un modèle séquentiel : les données traversent les couches dans l'ordre listé
    nn.Flatten(),                               # Aplatit l'image 28x28 (2D) en une ligne de 784 valeurs (1D)
    nn.Linear(28*28, neuronne),                      # Couche linéaire : 784 entrées -> 128 sorties (chaque sortie = somme pondérée des entrées + biais)
    nn.ReLU(),                                  # Activation : remplace les valeurs négatives par 0 (ajoute de la non-linéarité, indispensable pour apprendre des motifs complexes)
    nn.Linear(neuronne, 10)                          # Couche finale : 128 entrées -> 10 sorties (les "logits", un score par chiffre de 0 à 9)
)   


Version_Control()

from torchvision import datasets, transforms    # datasets = jeux de données prêts (MNIST...) ; transforms = outils pour transformer les images

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if calcul == "GPU":
    device = torch.device("cuda")
if calcul == "CPU":
    device = torch.device("cpu")
reussite = 0                                    # Compteur de bonnes prédictions, démarre à 0 (sera incrémenté plus bas)
                  # Nombre d'images à tester à la fin pour mesurer la performance

print("Téléchargement des données MNIST...")    # Affiche un message d'information dans le terminal
transform = transforms.ToTensor()               # Définit la transformation : convertit chaque image en tensor (format que PyTorch comprend, valeurs entre 0 et 1)
train_data = datasets.MNIST(root='.', train=True, download=True, transform=transform)
                                                # Charge les 60 000 images d'entraînement MNIST ; root='.' = dossier courant ; train=True = jeu d'entraînement ; download=True = télécharge si absent ; applique la transform
loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
                                                # Prépare la distribution des données par paquets de 96 images ; shuffle=True = mélange l'ordre à chaque époque

print("Création du modèle...")                  # Message d'information
                                            # Fin de la définition du modèle

model.to(device)                                # Déplace le modèle sur le GPU (si disponible)

optimizer = torch.optim.Adam(model.parameters())
                                                # Crée l'optimiseur Adam : c'est lui qui ajuste les poids du modèle pour réduire l'erreur ; model.parameters() = tous les poids/biais à entraîner
loss_fn = nn.CrossEntropyLoss()                 # Définit la fonction de perte : mesure l'écart entre prédiction et vraie réponse (adaptée à la classification)
temp_actuelle = time.time()                               # Récupère l'heure actuelle (en secondes depuis le 1er janvier 1970)
print("Entraînement en cours...")               # Message d'information
for i, (images, labels) in enumerate(loader):   # Boucle sur chaque paquet ; i = numéro du paquet, images = 96 images, labels = leurs 96 vraies réponses
    images, labels = images.to(device), labels.to(device)  # Déplace les images et labels sur le GPU (si disponible)
    optimizer.zero_grad()                       # Remet les gradients à zéro (sinon ils s'accumuleraient d'un paquet à l'autre)
    output = model(images)                      # Fait passer les 96 images dans le modèle -> obtient 96 prédictions (chacune = 10 logits)
    loss = loss_fn(output, labels)              # Calcule l'erreur moyenne entre les prédictions et les vraies réponses
    loss.backward()                             # Backpropagation : calcule comment chaque poids a contribué à l'erreur (les gradients)
    optimizer.step()                            # Met à jour les poids du modèle en fonction des gradients -> le modèle s'améliore d'un cran

                                                # Affiche la progression : numéro du paquet / total, et la valeur de la perte (4 décimales)
temp_aprentissage = time.time() - temp_actuelle
print("\nEntraînement terminé !")               # Message : l'entraînement est fini (\n = saute une ligne avant)

temp_actuelle = time.time()                               # Récupère l'heure actuelle (en secondes depuis le 1er janvier 1970)
for idx in range(nombredimagesfinales):         # Boucle de test : parcourt les 20000 premières images de train_data
    image, true_label = train_data[idx]         # Récupère l'image n°idx et sa vraie réponse
    image = image.to(device)                      # Déplace l'image sur le GPU (si disponible)
    pred = model(image.unsqueeze(0)).argmax().item()
                                                # Prédit le chiffre : unsqueeze(0) ajoute la dimension "paquet" ; argmax() prend l'indice du plus grand logit ; item() convertit en nombre Python
    if pred == true_label:                      # Si la prédiction est correcte...
        reussite += 1                           # ...incrémente le compteur de réussites
temp_test = time.time() - temp_actuelle
pourcentage_reussite = (reussite / nombredimagesfinales) * 100
                                                # Calcule le pourcentage de bonnes réponses

print("\n")          
print("Composant de calcul : " + calcul)                               # Affiche le résultat final : nombre de réussites / total, et le pourcentage (2 décimales)
print(f"Taux de réussite: {reussite}/{nombredimagesfinales} donc : ({pourcentage_reussite:.2f}%)")
print(f"Temps d'entraînement : {temp_aprentissage:.2f} secondes")
print(f"Temps de test : {temp_test:.2f} secondes")
print("\n")