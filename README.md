# 🛒 Prédiction des ventes Walmart — Rapport de synthèse

## 📇 Contexte
**Walmart Inc.** est un distributeur américain fondé par **Sam Walton** (1962), basé à Bentonville (Arkansas).  
**Objectif métier :** construire un **modèle ML** pour estimer les **ventes hebdomadaires** par magasin, mieux comprendre l’influence des indicateurs économiques et **anticiper** les campagnes marketing.

---

## 🎯 Objectifs du projet
1. **EDA + Prétraitement** des données  
2. **Modèle baseline** : régression linéaire  
3. **Réduction du surapprentissage** : régressions **Ridge** et **Lasso** (tuning d’α)

---

## 🖼️ Données & Cible
- Dataset (version **JULIE**, inspirée de Kaggle) : ventes hebdomadaires par magasin + indicateurs (`Fuel_Price`, `CPI`, `Unemployment`, `Holiday_Flag`, etc.).
- **Cible** : `Weekly_Sales`.
- **Variables explicatives** (exemples) : `Store`, `Holiday_Flag`, `Temperature`, `Fuel_Price`, `CPI`, `Unemployment`, `month`, `day_of_month` (dérivées de `Date`).

---

## 🧰 Méthodologie

### Prétraitement (pipeline scikit-learn, sans fuite)
- **Numériques** : `SimpleImputer(strategy="median")` → `StandardScaler`.  
- **Catégorielles** : `SimpleImputer(strategy="most_frequent")` → `OneHotEncoder(drop="first", handle_unknown="ignore")`.  
- **Nettoyage** : suppression des lignes avec cible manquante ; traitement des valeurs extrêmes (règle **±3σ** lorsque pertinent).  
- **Features calendrier** : `month`, `day_of_month` (autres disponibles si besoin).

### Évaluation
- **Hold-out** : `train/test = 80/20`.  
- **Validation croisée** : **KFold (10 folds)** intégrée aux GridSearch / LassoCV.  
- **Métriques** : **R²**, **MAE**, **RMSE**.  
- **Analyses** : résidus (centrage, dispersion), réels vs. prédits, coefficients.

---

## 📈 Résultats (jeu de test)

| Modèle | α* | CV R² | Train R² | **Test R²** | **Test MAE** | **Test RMSE** | # features |
|---|---:|---:|---:|---:|---:|---:|---:|
| Régression linéaire (baseline) | — | — | 0.997 | **0.954** | 119 722 | 145 447 | — |
| Ridge | 0.01 | 0.918 | 0.996 | **0.954** | 119 284 | 145 348 | — |
| Lasso | 450 | 0.941 | 0.994 | **0.950** | 119 094 | 150 965 | — |
| **Lasso (sélection) + Lasso (final)** | **≈ 100** | — | 0.982 | **0.956** | **112 010** | **142 942** | **17** |

**Lecture rapide**
- Tous les modèles expliquent ~**95 %** de la variance en test.  
- **Meilleur compromis précision / parcimonie** : **Lasso (sélection) + Lasso (final)**  
  → **Test R² = 0.956**, **MAE/RMSE** les plus bas, **17 variables** seulement.  
- **Ridge** ≈ baseline en performance mais sans parcimonie.  
- **Lasso** simple : R² test un peu inférieur et RMSE plus élevé.

---

## 🔎 Interprétation des coefficients (synthèse)
- Signal **robuste** et cohérent entre modèles : les **effets `Store_*`** dominent, complétés par quelques **effets calendrier** (ex. `month_12` positif, certains `day_of_month` négatifs).  
- **Ridge** lisse les amplitudes (réduit la variance) mais ne met pas de coefficients à 0.  
- **Lasso** annule de nombreux petits effets → **sélection de variables**.  
- Le pipeline **Lasso sélection + Lasso final** retient ~**17** variables (magasins majeurs + calendrier) → **interprétable** et **performant**.

---

## 🧪 Analyse des résidus (résumé)
- Résidus **centrés** autour de 0 (pas de biais systématique).  
- **Dispersion** accrue pour les ventes extrêmes (légère hétéroscédasticité) → cohérent avec **RMSE > MAE**.  
- Nuage **réels vs. prédits** proche de la diagonale → **bonne généralisation**.

---

## ✅ Conclusion & Recommandation
- **Modèle retenu** : **Pipeline _Lasso (sélection) + Lasso (final, α ≈ 100)_**.  
  - **Précision supérieure** (R² test **0.956**, MAE/RMSE au plus bas)  
  - **Parcimonie** (17 features) → **interprétable** et **robuste**  
  - Gestion des catégories inédites par `OneHotEncoder(handle_unknown="ignore")` (encodées à 0 par défaut)

**Usages métiers** : meilleure **prévision hebdomadaire**, **pilotage des stocks** et **planification marketing** ; compréhension des magasins/effets calendrier les plus structurants.

---

## 🔭 Pistes d’amélioration
- Ajouter des **features structurelles** (surface, trafic, mix produit) et des **interactions** (ex. `Holiday_Flag × month`).  
- Regrouper les magasins (clustering) pour mieux généraliser sur **petits volumes**.  
- Tester une cible transformée (**log(Weekly_Sales)**) pour atténuer l’hétéroscédasticité.

---

## 🛠️ Stack technique
**Python**, **pandas**, **numpy**, **matplotlib**, **seaborn**, **scikit-learn**  
(Pipelines/ColumnTransformer, `LinearRegression`, `Ridge`, `Lasso`, **`LassoCV`**, `GridSearchCV`, `KFold`).
