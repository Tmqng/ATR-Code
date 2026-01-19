# ATR – Automatic Target Recognition

## Résumé
> ✏️ **À compléter**  
> Brève description du projet, du contexte (ATR, MSTAR, AConvNet, etc.) et des objectifs.

---

## Prise en main

Ce projet utilise **uv** pour la gestion des dépendances Python (rapide, reproductible, moderne).

### Installation de uv

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

(ex : télécharger et décompresser le dataset suivant https://www.kaggle.com/datasets/minhqunnguyen/mstar-images-et-json)

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



