# 🛒 Prédiction des ventes Walmart

## 📇 Description de l’entreprise

**Walmart Inc.** est une entreprise multinationale américaine qui exploite une chaîne d’hypermarchés, de magasins de détail à bas prix et de supermarchés, dont le siège est situé à Bentonville, Arkansas.  
La société a été fondée par **Sam Walton** en **1962**.

---

## 🚧 Projet

L’équipe marketing de Walmart a demandé le développement d’un **modèle de machine learning** capable de prédire les **ventes hebdomadaires** dans ses magasins.  
L’objectif est de mieux comprendre l’influence des **indicateurs économiques** (prix du carburant, chômage, CPI, jours fériés, etc.) sur les ventes, et d’utiliser ce modèle pour **anticiper et planifier les campagnes marketing**.

---

## 🎯 Objectifs

Le projet est divisé en trois grandes étapes :

1. **Analyse exploratoire des données (EDA)** et prétraitement
2. **Modèle de base** : régression linéaire
3. **Modèles régularisés** : Ridge & Lasso pour éviter le surapprentissage

---

## 🖼️ Périmètre du projet

- Jeu de données : ventes hebdomadaires de Walmart (issu de Kaggle, version adaptée sur **JULIE**).
- Variable cible : `Weekly_Sales`.
- Variables explicatives : informations sur les magasins, indicateur de jours fériés, prix du carburant, CPI, taux de chômage, ainsi que des variables dérivées de la date (année, mois, jour, jour de la semaine).

---

## 📊 Méthodologie

### 1. Prétraitement des données & EDA

- Suppression des lignes avec valeurs manquantes dans la cible (`Weekly_Sales`).
- Gestion des valeurs aberrantes avec la règle des **3σ** sur la variable numérique `Unemployment`.
- Extraction de nouvelles variables à partir de `Date` :
  - `year`, `month`, `day_of_month`, `day_of_week`
- Analyse de la multicolinéarité avec une **matrice de corrélation** et suppression de variables redondantes.

### 2. Modèle de base : Régression linéaire

- Entraînement d’un premier modèle de **régression linéaire** comme benchmark.
- Résultats satisfaisants mais présence de **risque de surapprentissage**.

### 3. Modèles régularisés : Ridge & Lasso

- Entraînement des modèles **Ridge** et **Lasso** afin de limiter l’overfitting.
- Optimisation des hyperparamètres (`alpha`) via **GridSearchCV**.
- Évaluation avec **validation croisée KFold (7 folds)** pour mesurer la généralisation.

---

## 📈 Résultats

| Modèle    | R² Train | R² Test   | Moy. CV   | Écart-type CV |
| --------- | -------- | --------- | --------- | ------------- |
| Linéaire  | ~0.997   | ~0.944    | ~0.857    | 0.11          |
| Ridge     | ~0.996   | ~0.945    | ~0.876    | 0.09          |
| **Lasso** | ~0.994   | **0.960** | **0.883** | 0.11          |

➡️ **La régression Lasso** a montré la **meilleure généralisation** avec le R² test le plus élevé et une bonne stabilité.  
➡️ L’analyse des coefficients a permis d’identifier les **variables les plus déterminantes** dans les ventes.

---

## ✅ Conclusion .

- Le modèle final retenu est la **régression Lasso** :
  - Excellente performance prédictive
  - Capacité de sélection de variables (coefficients nuls pour les moins pertinentes)
- Ce modèle permet à Walmart de :
  - Prédire plus précisément ses ventes
  - Identifier les principaux facteurs influents
  - Orienter ses décisions marketing et logistiques

---

## 🛠️ Stack technique

- **Python**
- **pandas**, **numpy**
- **matplotlib**, **seaborn**
- **scikit-learn** (LinearRegression, Ridge, Lasso, GridSearchCV, KFold)
- **Jupyter Notebook**

---

## 📬 Livrables

- Analyse exploratoire (EDA)
- Pipeline de prétraitement
- Modèle de régression linéaire (baseline)
- Modèles régularisés (Ridge, Lasso)
- Évaluation des performances (R², MAE, RMSE, CV)
- Recommandation finale : **régression Lasso**
