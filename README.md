# 📊 Customer Satisfaction & Price Prediction

*Modèles de Machine Learning pour analyser la satisfaction client et prédire les prix – données démographiques & comportementales*


## 📌 Description du Projet

Ce projet vise à **développer des modèles de machine learning** capables de :

### 🎯 **1. Classifier la satisfaction client**

* satisfied
* neutral
* dissatisfied
* puis une version binaire : satisfied vs not satisfied

### 🎯 **2. Prédire le prix des billets d’avion**

Modèles de régression appliqués sur un dataset riche contenant des informations démographiques, comportementales et opérationnelles.

Le workflow complet inclut :

✔️ Chargement des données
✔️ Nettoyage & imputation des valeurs manquantes
✔️ Encodage des variables catégorielles
✔️ Analyse exploratoire (EDA)
✔️ Tests statistiques
✔️ Corrélations & visualisations
✔️ Modèles de classification
✔️ Modèles de régression
✔️ Optimisation via GridSearchCV
✔️ Comparaison des performances


## 📂 Dataset

Le dataset contient **129 880 lignes** et **26 colonnes** liées au voyage aérien :

| Catégorie        | Variables                                                                             |
| ---------------- | ------------------------------------------------------------------------------------- |
| 🧑 Profil client | Gender, Customer Type, Age                                                            |
| ✈️ Voyage        | Type of Travel, Class, Flight Distance                                                |
| ⭐ Services       | Inflight wifi, Online booking, Gate location, Food, Seat comfort, Entertainment, etc. |
| ⏱️ Délais        | Departure Delay, Arrival Delay                                                        |
| 🎯 Cibles        | satisfaction, Price                                                                   |


## 🧹 Prétraitement des Données

### ✔️ Nettoyage & imputation

* Suppression des colonnes inutiles : `Unnamed: 0`, `id`
* Imputation des valeurs manquantes (médiane) :

  * Age
  * Ease of Online booking
  * Gate location
  * Leg room service
  * Arrival Delay in Minutes

### ✔️ Encodage des variables

* `LabelEncoder` pour les colonnes catégorielles
* Création d’une variable binaire :

  ```
  satisfied → 1  
  neutral + dissatisfied → 0
  ```

### ✔️ Normalisation

* `StandardScaler` pour améliorer la convergence des modèles


## 📊 Analyse Exploratoire (EDA)

### 🔍 Visualisations

* Histogrammes : Age, Flight Distance, Price
* Countplots : Gender, Customer Type, Type of Travel
* Boxplots : Price par Class, Age par Satisfaction

### 📈 Relations importantes observées

* Les prix varient fortement selon la classe (Eco, Eco+, Business)
* Les clients satisfaits ont en moyenne un âge légèrement supérieur
* Le vol Business a un prix moyen largement plus élevé
* Type of Travel et Online boarding influencent fortement la satisfaction


## 🧪 Tests Statistiques

| Test                       | Résultat                         | Interprétation                                        |
| -------------------------- | -------------------------------- | ----------------------------------------------------- |
| **t-test**                 | p = 0 → différence significative | Les âges diffèrent entre satisfaits/insatisfaits      |
| **Chi²**                   | p = 0 → dépendance forte         | Satisfaction dépend du type de client (loyal/déloyal) |
| **ANOVA**                  | p = 0 → différence significative | Les prix diffèrent entre classes                      |
| **Corrélation de Pearson** | r = 0.17                         | Faible corrélation positive Age ↔ Price               |


## 🔥 Modèles de Classification

Modèles testés via **GridSearchCV** :

* Logistic Regression
* Random Forest
* SVC (Support Vector Classifier)
* KNN

### 🏆 Meilleur modèle : **Random Forest**

**Résultats :**

* Accuracy : **0.96**
* AUC : **0.996**
* Très bonne capacité à généraliser
* Surapprentissage limité grâce à GridSearchCV


## 📈 Modèles de Régression (Price Prediction)

Modèles testés :

* Linear Regression
* Ridge Regression
* Lasso Regression
* Random Forest Regressor
* SVR

### 🏆 Meilleur modèle : **Random Forest Regressor**

**Performances typiques :**

* MSE minimal
* MAE réduit
* R² élevé (≈ 0.85)

La régression linéaire obtient déjà un **R² ≈ 0.854**, mais RF améliore davantage les erreurs absolues.


## 🧪 Exemple de Code d’Entraînement

```python
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(n_estimators=200, max_depth=30)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```


## 🚀 Comment Exécuter le Projet

### 1️⃣ Cloner le dépôt

```bash
git clone https://github.com/username/customer-satisfaction-ml.git
cd customer-satisfaction-ml
```

### 2️⃣ Installer les dépendances

```bash
pip install -r requirements.txt
```

### 3️⃣ Lancer les notebooks

* `Customer_Satisfaction.ipynb`
* `Price_Prediction.ipynb`

ou exécuter les scripts Python :

```bash
python train_classification.py
python train_regression.py
```


## ✨ Améliorations Futures

* Utilisation de modèles boosting : XGBoost, LightGBM
* Feature engineering :

  * Interaction features
  * Reduction via PCA
* Déploiement API FastAPI / Streamlit
* Dashboard PowerBI/Tableau pour visualisation dynamique


## 👤 Auteur

**Alex Alkhatib**
Projet Machine Learning — Satisfaction & Price Prediction


## 📄 Licence
MIT License
Copyright (c) 2025 Alex Alkhatib
