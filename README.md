# Réseaux de Neurones DIY – Implémentation from scratch

## Présentation
Ce projet implémente, en **Python**, une bibliothèque modulaire de réseaux de neurones artificiels développée *from scratch*. Inspirée des premières versions de PyTorch, cette bibliothèque repose sur une abstraction claire des composants fondamentaux : couches linéaires, fonctions d’activation, fonctions de coût, encapsulation séquentielle et optimiseur générique.  

L’objectif est double :  
1. Mettre en œuvre les mécanismes de **l’apprentissage différentiel** sans dépendre de frameworks existants.  
2. Explorer expérimentalement la classification multi-classes et les auto-encodeurs.  

## Fonctionnalités
- **Modules implémentés** : `Linear`, `Tanh`, `Sigmoid`, `ReLU`, `Softmax`, `MSELoss`, `CrossEntropyLoss`, `Sequential`, optimiseur générique.  
- **Classification supervisée multi-classes** sur :
  - Iris (tabulaire)
  - Fashion-MNIST
  - Kuzushiji-MNIST  
- **Auto-encodeurs** : reconstruction d’images, exploration des représentations latentes, clustering par t-SNE.  
- **Prétraitement de la classification** par reconstruction.  
- Tests unitaires, sauvegarde des modèles et notebooks d’expérimentation.  

## Organisation
- [`src/modules/`](src/modules/) : implémentation des modules (couches, activations, pertes).  
- [`src/tests/`](src/tests/) : tests unitaires.  
- [`src/models/`](src/models/) : modèles sauvegardés, résultats et visualisations (non uploadés car ils sont très volumineux).  
- Notebooks principaux :
  - [`2_layers_NN.ipynb`](src/2_layers_NN.ipynb) : réseau simple (régression, classification binaire).  
  - [`multi_classe.ipynb`](src/multi_classe.ipynb) : classification multi-classes.  
  - [`autoencodeur.ipynb`](src/autoencodeur.ipynb) : entraînement et reconstruction.  
  - [`visualisation.ipynb`](src/visualisation.ipynb) : clustering et t-SNE.  
  - [`pretraitement.ipynb`](src/pretraitement.ipynb) : classification après reconstruction.  

## Résultats
- Bonne convergence et précision élevée sur **Iris**.  
- Performances satisfaisantes sur **Fashion-MNIST** et **Kuzushiji-MNIST**, avec un gain grâce à ReLU.  
- Auto-encodeurs capables de reconstruire correctement les images avec des dimensions latentes intermédiaires, mais perte d’information avec un goulot trop restreint.  
- Clustering t-SNE : séparation partielle des classes, cohérente avec leur similarité visuelle.  
- Prétraitement par auto-encodeur : performances généralement dégradées, montrant les limites d’une reconstruction purement pixel-wise.  

## Perspectives
- Implémentation de **CNNs** et d’auto-encodeurs convolutifs.  
- Ajout de pertes perceptuelles et de régularisations avancées.  
- Exploration de l’**apprentissage auto-supervisé** et optimisation automatique des hyperparamètres.  
