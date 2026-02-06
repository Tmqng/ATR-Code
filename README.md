# ATR – Automatic Target Recognition

## Résumé

> Ce projet s'intéresse au problème de classification de cibles militaires dans le cadre de l'ATR SAR : Automatic Target Recognition sur données radar de type SAR.

Le but de ce répertoire est de fournir une base de code afin de pouvoir entraîner facilement différentes architecture de réseaux de neurones sur le dataset de référence MSTAR. La pipeline est construite de façon modulable : il est aisé de rajouter un bloc de pré-traitement de données, ou de changer de modèle de DL par exemple.

La structure du projet est la suivante : 

```
ATR-Code/
│
├── datasets/                      # Datasets utilisés dans le projet
│   ├── dataset-1/
│   └── dataset-2/
│
├── experiments/                   # Expériences par modèle
│   └── nom_modele/
│       ├── config/                # Fichiers JSON de configuration (hyperparamètres)
│       ├── history/               # Historique d'entraînement (loss, accuracy, etc.)
│       └── models/                # Modèles sauvegardés pendant l'entraînement
│
├── notebooks/                     # Notebooks d'expérimentation et d'inférence
│
├── scripts/                       # Scripts exécutables depuis le terminal
│   ├── train.py                   # Entraînement (exemple)
│   ├── generate_dataset.py        # Génération des datasets (exemple)
│
│
|── src/                           # Code source principal
│   ├── data/                      # Chargement et pré-traitement des données
│   │   └── nom_dataset/           # Spécifique à chaque dataset
│   │
│   ├── models/                    # Modèles
│   │   ├── _base.py               # Classe Model générique
│   │   ├── architecture_1/        # Définition d'une architecture
│   │   └── architecture_2/
│   │
│   └── utils/                     # Fonctions utilitaires partagées
│
└── README.md
```

## Prise en main

### Installation avec uv

Ce projet est compatible avec **uv** pour la gestion des dépendances Python (rapide, reproductible, moderne).

#### Option 1 — via pip
```bash
pip install uv
```

#### Option 2 — installation standalone (recommandée)

Voir la documentation officielle :
https://docs.astral.sh/uv/getting-started/installation/

Exemple (Linux / macOS) :
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Vérifier l’installation :
```bash
uv --version
```

### Installation des dépendances

Les dépendances sont définies dans pyproject.toml et verrouillées dans uv.lock.

Pour installer exactement l’environnement du projet :
```bash
uv sync
```

👉 Cette commande :
- crée automatiquement un environnement virtuel (.venv)
- installe toutes les dépendances verrouillées

### Gestion des dépendances

#### Ajouter une dépendance :
```bash
uv add <package>
```

#### Supprimer une dépendance :
```bash
uv remove <package>
```

Les fichiers pyproject.toml et uv.lock sont automatiquement mis à jour.

### Lancer un script python :

**uv** permet de lancer un script python avec la commande:
```bash
uv run python_script.py
```
---

### Alternative : Installation avec conda et pip

Si vous préférez utiliser **conda** au lieu de **uv**, voici les étapes :

#### Étape 1 : Créer un environnement conda avec Python 3.12

```bash
conda create -n atr-code python=3.12
```

#### Étape 2 : Activer l'environnement

```bash
conda activate atr-code
```

#### Étape 3 : Installer les dépendances via pip

```bash
pip install -r requirements.txt
```

Cela créera un environnement isolé avec Python 3.12 et toutes les dépendances requises.

---

## Lancer une expérience

Les scripts d’entraînement se trouvent dans le dossier scripts/

Les datasets doivent être téléchargés dans le dossier datasets/. Il est recommandé d'organiser les données de la manière suivante :
```
nom_du_dataset/
├── train/
│   ├── class_0/
│   │   ├── sample_1.png (fichiers.png contiennent l'image)
│   │   ├── sample_1.json (fichiers.json contiennent les métadonnées 'class_name', 'class_id' etc...)
│   │   ├── sample_2.png
│   │   └── sample_2.json
│   ├── class_1/
│   │   ├── sample_1.png
│   │   └── sample_1.json
│   └── class_2/
│       ├── sample_1.png
│       └── sample_1.json
│
└── test/
    ├── class_0/
    │   ├── sample_1.png
    │   └── sample_1.json
    ├── class_1/
    │   ├── sample_1.png
    │   └── sample_1.json
    └── class_2/
        ├── sample_1.png
        └── sample_1.json
```

(ex : télécharger et décompresser le dataset suivant pour voir un example https://www.kaggle.com/datasets/minhqunnguyen/mstar-images-et-json)

Les configurations d’expériences sont définies dans experiments/<model_name>/config/.

#### Exemple : lancer un entraînement avec AConvNet
```bash
uv run python scripts/train_AConvNet.py \
  --config experiments/AConvNet/config/AConvNet-SOC.json
```

Selon la configuration :

- les résultats (logs, métriques) sont enregistrés dans experiments/AConvNet/history/
- les modèles entraînés sont sauvegardés dans experiments/AConvNet/models/

⚠️ Ces dossiers ne sont pas versionnés dans Git.



