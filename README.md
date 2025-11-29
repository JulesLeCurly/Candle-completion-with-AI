# Cryptocurrency Data Completion with AI

Système de complétion intelligente des données historiques de cryptomonnaies utilisant Deep Learning (LSTM + Attention) pour prédire les bougies manquantes.

## 📊 Vue d'ensemble

Ce projet utilise un modèle d'apprentissage profond pour combler les gaps dans les données historiques de cryptomonnaies de Binance en apprenant la relation avec Kraken comme exchange de référence.

### Caractéristiques principales

- ✅ **Architecture LSTM + Attention** pour capturer les dépendances temporelles
- ✅ **Loss function custom** avec contraintes OHLC et pénalités
- ✅ **Post-processing intelligent** pour valider et corriger les prédictions
- ✅ **Support multi-symboles** avec embeddings
- ✅ **Visualisations interactives** des prédictions
- ✅ **Cache des données** pour éviter les re-téléchargements

## 📁 Structure du projet

```
Candle-completion-with-AI/
├── config.py                 # Configuration centralisée
├── data_fetcher.py           # Récupération données via CCXT
├── feature_engineering.py    # Features dérivées et normalisation
├── dataset_builder.py        # Création datasets d'entraînement
├── model.py                  # Architecture LSTM + Attention
├── trainer.py                # Pipeline d'entraînement
├── predictor.py              # Prédiction et post-processing
├── visualizer.py             # Visualisations candlestick
├── main.py                   # Orchestration principale
├── test_predictions.py       # Tests et visualisations
├── DataBase/
│   ├── RAW_Data_1h/         # Données brutes téléchargées
│   └── Completed_Data_1h/   # Données complétées par l'IA
├── models/
│   ├── crypto_completion_model.keras
│   └── normalization_params.pkl
└── logs/
    ├── crypto_completion.log
    ├── training_history.png
    └── training_report.json
```

## 🚀 Installation

```bash
# Cloner le repository
git clone <votre-repo>
cd Candle-completion-with-AI

# Installer les dépendances
pip install tensorflow pandas numpy matplotlib ccxt scikit-learn
```

## 💻 Utilisation

### 1. Entraînement du modèle

```bash
# Entraînement complet (tous les symboles)
python main.py --mode train --epochs 100

# Entraînement sur symboles spécifiques
python main.py --mode train --symbols BTC/USDT,ETH/USDT --epochs 50

# Forcer le re-téléchargement des données
python main.py --mode train --epochs 100 --force-download
```

**Ce qui se passe:**
1. Télécharge/charge les données de Binance et Kraken
2. Calcule les features (ratios, wicks, encodages temporels)
3. Crée un dataset synthétique avec gaps artificiels
4. Entraîne le modèle LSTM + Attention
5. Évalue sur le test set
6. Génère les graphiques et rapports

**Sorties:**
- `models/crypto_completion_model.keras` - Modèle entraîné
- `logs/training_history.png` - Courbes d'entraînement
- `logs/training_report.json` - Métriques détaillées

### 2. Complétion des données

```bash
# Compléter les données avec gaps
python main.py --mode complete --input DataBase/RAW_Data_1h --output DataBase/Completed_Data_1h

# Utiliser un modèle spécifique
python main.py --mode complete --model models/my_model.keras
```

**Ce qui se passe:**
1. Charge le modèle entraîné
2. Détecte les gaps dans les données Binance
3. Prédit les bougies manquantes
4. Applique le post-processing (validation OHLC)
5. Sauvegarde les données complètes avec marqueurs

**Sorties:**
- `Completed_Data_1h/<symbol>_completed.csv` - Données complétées
- `quality_report.json` - Statistiques de complétion

### 3. Visualisation des prédictions

```bash
# Tester les prédictions avec visualisations
python test_predictions.py --symbol BTC/USDT --num-examples 5

# Tester un autre symbole
python test_predictions.py --symbol ETH/USDT --num-examples 3
```

**Ce qui se passe:**
1. Charge le modèle et les données
2. Détecte ou crée des gaps de test
3. Génère les prédictions
4. Compare visuellement avec la réalité
5. Calcule les métriques d'erreur

**Sorties:**
- Graphiques interactifs comparant réel vs prédit
- Métriques MAE/MAPE par gap
- Images sauvegardées dans `logs/`

## 📈 Comprendre les visualisations

### Training History
![Training curves](docs/training_example.png)

- **Loss**: Perte globale (plus c'est bas, mieux c'est)
- **MAE**: Erreur absolue moyenne
- **MAPE**: Erreur en pourcentage
- **Val curves**: Performance sur validation set

### Candlestick Predictions
![Predictions](docs/prediction_example.png)

- **Bleu**: Bougies réelles (contexte)
- **Orange**: Bougies prédites par l'IA
- **Zone jaune** (haut): Données réelles cachées au modèle
- **Zone verte** (bas): Prédictions de l'IA

## 📊 Format des données complétées

Les CSV complétés contiennent:

| Colonne | Description |
|---------|-------------|
| `open_time` | Timestamp de début de la bougie |
| `open`, `high`, `low`, `close` | Prix OHLC |
| `volume` | Volume échangé |
| `close_time` | Timestamp de fin |
| `number_of_trades` | Nombre de trades |
| `is_predicted` | `True` si prédit par l'IA, `False` si réel |
| `prediction_confidence` | Score de confiance (0-1) |
| `source_exchange` | `binance`, `kraken`, ou `predicted` |
| `gap_length` | Longueur du gap comblé (si prédit) |

## ⚙️ Configuration

Modifier `config.py` pour ajuster:

```python
# Symboles à traiter
SYMBOLS = ['BTC/USDT', 'ETH/USDT', ...]

# Période des données
START_DATE = '2017-01-01'
END_DATE = datetime.now()

# Architecture du modèle
LSTM_UNITS = 128
ATTENTION_UNITS = 64
LOOKBACK_WINDOW = 72  # heures de contexte

# Gaps
MAX_GAP_LENGTH = 24  # max 24h consécutives

# Entraînement
EPOCHS = 100
BATCH_SIZE = 32
LEARNING_RATE = 0.001
```

## 🎯 Cas d'usage

### 1. Recherche et backtesting
Utiliser les données complètes pour tester des stratégies de trading sur tout l'historique sans trous.

### 2. Analyse technique
Calculer des indicateurs (moyennes mobiles, RSI, etc.) sur des séries continues.

### 3. Machine Learning
Créer des datasets propres pour entraîner d'autres modèles de prédiction.

### 4. Visualisation
Créer des graphiques sans interruptions pour les analyses.

## 🔧 Résolution de problèmes

### Le modèle ne converge pas
- Réduire le `LEARNING_RATE` dans `config.py`
- Augmenter `LOOKBACK_WINDOW` pour plus de contexte
- Vérifier la qualité des données sources

### Prédictions incohérentes
- Ajuster `VIOLATION_PENALTY` dans la loss function
- Augmenter `MAX_PRICE_DEVIATION` dans le post-processing
- Entraîner plus longtemps

### Erreur de dimensions
- Vérifier que toutes les données ont les mêmes features
- S'assurer que les CSV sont bien formatés
- Relancer avec `--force-download`

### API rate limits
- Augmenter `RATE_LIMIT_DELAY` dans config
- Utiliser les données en cache (ne pas utiliser `--force-download`)

## 📚 Architecture technique

### Pipeline d'entraînement
```
Raw Data (Binance + Kraken)
    ↓
Feature Engineering (ratios, wicks, temporal)
    ↓
Synthetic Gap Creation (15% masked)
    ↓
Dataset Builder (sequences LSTM)
    ↓
Model Training (LSTM + Attention)
    ↓
Evaluation & Checkpoints
```

### Pipeline de prédiction
```
Raw Data with Gaps
    ↓
Gap Detection
    ↓
Context Extraction (72h before)
    ↓
Model Prediction
    ↓
Post-Processing (OHLC validation)
    ↓
Completed Data with Confidence
```

### Modèle
```
Input: [Primary Context (72h), Secondary Context (72h+gap), Symbol Embedding, Gap Length]
    ↓
LSTM Layers (128 units)
    ↓
Attention Mechanism (64 units)
    ↓
Dense Layers + Dropout
    ↓
Output: [OHLCV × Gap Length]
```

## 📊 Métriques de qualité

Le modèle est évalué sur:
- **MAE (Mean Absolute Error)**: Erreur moyenne absolue
- **MAPE (Mean Absolute Percentage Error)**: Erreur en %
- **OHLC Violations**: Nombre de bougies invalides
- **Confidence Score**: Score de confiance moyen

**Valeurs cibles:**
- MAE < 0.1 (après normalisation)
- MAPE < 5%
- Violations < 1%
- Confidence > 0.7

## 🚀 Prochaines améliorations

- [ ] Support de plus d'exchanges (Coinbase, Bitfinex)
- [ ] Modèle Transformer au lieu de LSTM
- [ ] Prédiction de gaps > 24h avec découpage
- [ ] Interface web pour visualisation interactive
- [ ] API REST pour prédictions en temps réel
- [ ] Support des timeframes multiples (15m, 4h, 1d)

## 📝 Licence

MIT License - Libre d'utilisation

## 🤝 Contribution

Les contributions sont bienvenues! Ouvrez une issue ou un pull request.

## ⚠️ Disclaimer

Ce projet est à des fins éducatives. Les prédictions ne constituent pas des conseils financiers. Utilisez à vos propres risques.