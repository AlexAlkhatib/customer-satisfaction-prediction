# ✈️ Customer Satisfaction & Price Prediction

**Comprendre la satisfaction client et prédire le prix des billets d'avion.**
Un notebook unique, entièrement exécuté, qui va de l'audit des données brutes à un modèle sauvegardé et réutilisable.

| | |
|---|---|
| 📦 Données | 129 880 passagers · 25 variables |
| 🎯 Cibles | `satisfaction` (classification binaire) · `Price` (régression) |
| 🏆 Prix | RMSE **58 €**, R² **0,997**, MAPE **2,7 %** (gradient boosting réglé) |
| 🏆 Satisfaction | Accuracy **96,5 %**, F1 **0,959**, AUC **0,995** |
| 🧪 Vérification | notebook exécuté sans erreur, 20 figures exportées, modèle rechargé et testé |

Le notebook est dans [`customer-satisfaction-prediction.ipynb`](customer-satisfaction-prediction.ipynb) — ouvert avec ses sorties, ses tableaux et ses interprétations.

---

## 📌 Objectifs

1. **Auditer et nettoyer** un jeu de données de qualité inégale (valeurs manquantes, notes ambiguës, colonnes techniques).
2. **Comprendre** les ressorts de la satisfaction par l'analyse exploratoire et des tests statistiques avec tailles d'effet.
3. **Construire des variables** (feature engineering) qui expriment des hypothèses métier, pas seulement des transformations.
4. **Prédire le prix** du billet : baseline naïve, 5 modèles, validation croisée, réglage, analyse des résidus.
5. **Prédire la satisfaction** : 4 modèles, matrice de confusion, courbes ROC, variables motrices.
6. **Distinguer ce qui est prédictible de ce qui est déployable**, et formuler des recommandations opérationnelles.

---

## 📂 Dataset

`dataset.csv` contient **129 880 lignes et 26 colonnes** : une colonne d'index exportée par erreur (`Unnamed: 0`), un identifiant technique (`id`, unique) et **25 variables métier**.

| Catégorie | Variables |
|---|---|
| 🧑 Profil client | `Gender`, `Customer Type`, `Age` (6–76 ans) |
| ✈️ Voyage | `Type of Travel`, `Class` (Eco, Eco Plus, Business), `Flight Distance` (34–5 481 miles) |
| ⭐ Services (0–5) | wifi, réservation en ligne, confort du siège, divertissement, propreté, personnel, bagages, enregistrement… (14 dimensions) |
| ⏱️ Délais | `Departure Delay in Minutes`, `Arrival Delay in Minutes` (466 et 472 valeurs distinctes, jusqu'à 1 592 min) |
| 🎯 Cibles | `satisfaction` (*dissatisfied* / *neutral* / *satisfied*), `Price` (141–4 726 €, médiane 1 472 €) |

**Qualité initiale mesurée dans le notebook (section 3) :** 0 doublon, 0 colonne constante, 913 cellules manquantes (0,70 % des lignes) réparties sur 5 colonnes — `Arrival Delay in Minutes` (393), `Leg room service` (133), `Ease of Online booking` (131), `Gate location` (130), `Age` (126).

---

## 🧭 Démarche du notebook

| # | Section | Contenu |
|---|---|---|
| 1 | Configuration | graines, style, helpers `save()` / `note()` |
| 2 | Chargement | dictionnaire des variables |
| 3 | Audit | manquants, doublons, extrêmes, distributions |
| 4 | Nettoyage | renommage `snake_case`, imputation argumentée |
| 5 | EDA (5.1–5.10) | cible, prix, âge/distance, baromètre des notes, profils, retards, corrélations, prix × distance |
| 6 | Feature engineering | indices de service, variables de retard, segments, interaction classe × distance |
| 7 | Tests statistiques | p-values **et** tailles d'effet |
| 8 | Préparation ML | split 80/20 stratifié, pipelines sans fuite |
| 9 | Régression du prix | baseline + 5 modèles, validation croisée 5 folds, résidus |
| 10 | Optimisation | recherche aléatoire, importance par permutation, modèle « déployable » |
| 11 | Classification | cible binaire, 4 modèles, ROC, importance, leviers |
| 12 | Synthèse | réponses chiffrées, 5 recommandations, 5 limites |
| 13 | Sauvegarde & inférence | `joblib`, rechargement, 6 scénarios clients |

---

## 🧹 Nettoyage et préparation

- **Renommage systématique en `snake_case`** (`Seat comfort` → `seat_comfort`) pour un code lisible.
- **Imputation par la médiane** pour l'âge et les notes : robuste aux extrêmes et respectueuse de l'échelle ordinale 0–5.
- **Règle métier pour le retard à l'arrivée** : `arrival_delay = departure_delay + décalage médian`, plus un indicateur `arrival_delay_missing` qui conserve l'information « donnée absente ».
- **Aucune fuite de données** : imputation, standardisation et one-hot vivent dans un `ColumnTransformer` intégré au `Pipeline`, ajusté sur le train uniquement (et à chaque fold de validation croisée). C'est la correction principale par rapport à la version initiale, qui encodait les modalités avant le split.
- **Split unique 80 / 20**, stratifié sur la satisfaction binaire ; le jeu de test ne sert qu'à l'évaluation finale.

---

## 🔧 Feature engineering

Trois familles de variables, chacune reliée à une hypothèse explicite :

| Famille | Variables | Hypothèse |
|---|---|---|
| **Retard** | `total_delay`, `long_delay` (≥ 60 min), `has_delay`, `arrival_extra_delay` | l'effet du retard est un **seuil**, pas une pente |
| **Qualité perçue** | `service_score`, `comfort_score`, `staff_score`, `digital_score`, `n_top_ratings`, `n_low_ratings` | les 14 notes sont corrélées : des indices les résument |
| **Segment** | `class_rank`, `class_distance`, `is_business`, `is_business_travel`, `is_loyal`, `age_group`, `distance_bucket` | la cabine est **ordinale** et interagit avec la distance |

**Apport mesuré (section 6.1, validation croisée 3 folds) :**

| Modèle | Variables brutes | Avec feature engineering | Gain |
|---|---|---|---|
| Ridge | 0,9492 | **0,9554** | **+0,0062** |
| Gradient boosting | 0,9975 | 0,9973 | −0,0002 |

Le gain est net pour les modèles linéaires (un Ridge ne sait ni moyenner 14 notes corrélées ni multiplier distance × cabine) et nul pour les arbres, déjà au plafond du jeu de données. Les variables construites sont conservées pour la **lisibilité** et la robustesse.

`add_features()` est réutilisée à l'inférence : le nouveau client traverse exactement les mêmes transformations que les données d'entraînement.

---

## 📊 Analyse exploratoire — ce que montrent les figures

| Figure | Contenu | Constat clé |
|---|---|---|
| [`01_audit_extremes.png`](figures/01_audit_extremes.png) | distributions et extrêmes | retards très asymétriques (max 1 592 min), 1 290 vols > 3 h |
| [`02_cible_satisfaction.png`](figures/02_cible_satisfaction.png) | répartition de la cible | **43,4 %** de satisfaits, 3 classes déséquilibrées |
| [`03_prix.png`](figures/03_prix.png) | histogrammes et boxplots du prix | prix multimodaux : un mode par cabine |
| [`04_age_distance.png`](figures/04_age_distance.png) | âge et distance | âge symétrique (35,5 ans en moyenne), distance très asymétrique |
| [`05_notes_services.png`](figures/05_notes_services.png) | baromètre des 14 notes | wifi **2,73/5** et réservation **2,76/5** en bas ; bagages 3,63 et service à bord 3,64 en haut |
| [`05b_notes_et_satisfaction.png`](figures/05b_notes_et_satisfaction.png) | note → taux de satisfaction | révèle l'artefact de la note 0 (voir plus bas) |
| [`06_radar_services.png`](figures/06_radar_services.png) | profils de notes par satisfaction | écarts les plus forts : embarquement (+1,43), divertissement (+1,23), siège (+1,07) |
| [`07_profil_satisfaction.png`](figures/07_profil_satisfaction.png) | satisfaction par segment | Business **69,4 %** vs Eco **18,8 %** ; voyage d'affaires **58,4 %** vs personnel **10,1 %** |
| [`08_retards.png`](figures/08_retards.png) | satisfaction par tranche de retard | **47,2 %** sans retard → **35,8 %** au-delà de 2 h, chute dès les premières minutes puis plateau |
| [`09_correlations.png`](figures/09_correlations.png) | matrice de Spearman lisible | notes de confort très corrélées ; `seat_comfort` ~ prix à 0,63 |
| [`10_prix_distance.png`](figures/10_prix_distance.png) | prix × distance par cabine | trois droites parallèles : la cabine décale le prix, la distance en fixe la pente |
| [`11_indices.png`](figures/11_indices.png) | corrélation des indices avec les cibles | `service_score` meilleur résumé univarié (r ≈ 0,50) |
| [`11b_seuil_satisfaction.png`](figures/11b_seuil_satisfaction.png) | taux de satisfaction par score et par motif | **seuil décalé** : à 3–3,5, 59,7 % (affaires) vs 9,3 % (personnel) |
| [`12_comparaison_regression.png`](figures/12_comparaison_regression.png) | R² et RMSE des 6 modèles | gradient boosting devant, baseline très loin derrière |
| [`13_residus_regression.png`](figures/13_residus_regression.png) | résidus, distribution, prédit vs réel | aucun biais, 90 % des erreurs dans ± 100 € |
| [`14_importance_prix.png`](figures/14_importance_prix.png) | importance par permutation | `class_rank` 0,56 ; `seat_comfort` 0,15 ; `flight_distance` 0,06 |
| [`15_comparaison_classification.png`](figures/15_comparaison_classification.png) | AUC et F1 des 4 modèles | forêt et boosting à égalité, logistique 7 points derrière |
| [`16_confusion_roc.png`](figures/16_confusion_roc.png) | matrice de confusion et ROC | 98,3 % des non-satisfaits et 94,2 % des satisfaits bien classés |
| [`17_importance_satisfaction.png`](figures/17_importance_satisfaction.png) | importance par permutation | `wifi` (artefact pour partie), `is_business_travel` |
| [`18_leviers.png`](figures/18_leviers.png) | leviers sans le profil commercial | wifi, embarquement en ligne, horaires, localisation de la porte |

### Deux découvertes structurantes

**1. La note 0 ne signifie pas « très mauvais » mais « non applicable ».** 7,9 % des clients ont au moins une note à 0. Pour le wifi, la note 0 correspond à 99,7 % de clients satisfaits, alors que pour le confort du siège elle correspond à 0 % : la modalité n'a pas le même sens selon la dimension. Elle est conservée et signalée, mais elle pollue les importances de variables.

**2. Le prix est presque entièrement déterminé par les variables disponibles.** Le R² de 0,997 n'est pas une performance de modèle : c'est une propriété du jeu de données, où le prix est indexé sur la cabine, la distance et les notes de service. Un tarif aérien réel dépend aussi de la date de réservation, de la concurrence et de la saisonnalité, absentes ici.

---

## 🧪 Tests statistiques

Chaque test est reporté avec sa **taille d'effet** : sur 130 000 lignes, une p-value est presque toujours significative et ne dit rien de l'ampleur du phénomène.

| Test | Question | Statistique | p-value | Taille d'effet |
|---|---|---|---|---|
| t de Welch | âge ~ satisfaction | 50,52 | < 0,001 | d de Cohen **0,274** (petit) |
| χ² (1 ddl) | fidélité ~ satisfaction | 4 493 | < 0,001 | V de Cramér **0,186** (faible) |
| χ² (2 ddl) | cabine ~ satisfaction | 32 906 | < 0,001 | V de Cramér **0,503** (fort) |
| ANOVA | prix ~ cabine | 144 545 | < 0,001 | êta² **0,690** (très fort) |
| Kruskal-Wallis | prix ~ cabine, sans normalité | 89 449 | < 0,001 | — (confirme l'ANOVA) |
| Mann-Whitney | service perçu ~ satisfaction | 3,29e9 | < 0,001 | r rang-bisérial **−0,586** |
| Pearson | prix ~ distance | — | < 0,001 | r **0,599** |
| Spearman | prix ~ distance (rangs) | — | < 0,001 | ρ **0,539** |

Lecture : la cabine est le facteur dominant du prix (69 % de la variance) et le service perçu le meilleur discriminant de la satisfaction.

---

## 🤖 Modélisation

### Régression du prix (section 9)

6 modèles comparés en validation croisée 5 folds sur le train, puis évalués une seule fois sur le test :

| Modèle | RMSE | MAE | R² | MAPE |
|---|---|---|---|---|
| **HGB réglé** | **58,31 €** | **42,28 €** | **0,997** | **2,72 %** |
| Gradient boosting (HGB) | 59,18 € | 43,11 € | 0,997 | 2,79 % |
| Forêt aléatoire | 66,77 € | 47,78 € | 0,997 | 3,08 % |
| Arbre de décision | 83,84 € | 59,85 € | 0,995 | 3,85 % |
| Ridge (α = 10) | 246,01 € | 192,44 € | 0,955 | 18,08 % |
| Régression linéaire | 246,02 € | 192,44 € | 0,955 | 18,08 % |
| Baseline (médiane) | 1 233,53 € | 998,44 € | −0,120 | 71,10 % |

- La **baseline médiane** a un R² négatif : elle est pire que la moyenne. Sans elle, un R² de 0,95 ne dit rien.
- La **linéarité plafonne à 0,955** (246 € d'erreur) : la relation change de pente selon la cabine et le niveau des notes.
- Le **réglage n'apporte que 1,5 %** (59,2 → 58,3 €) : le modèle par défaut était déjà au plafond, et ce plafond est le jeu de données. Un réglage plus fin serait du sur-ajustement de compétition.
- **Résidus** : moyenne ≈ 0 €, médiane ≈ +0,2 €, 90 % des erreurs dans **± 100 €**, pas d'écrasement des tarifs élevés.

### Audit tarifaire vs tarification (section 10.3)

Les notes de service et les retards sont connus **après** le vol : un modèle qui les utilise ne peut pas servir à tarifer.

| Modèle | RMSE | R² | MAPE | Usage |
|---|---|---|---|---|
| Toutes variables | 58,31 € | 0,997 | 2,7 % | **audit** d'un prix déjà vendu |
| Variables disponibles à la réservation | 550,22 € | 0,777 | 33,1 % | **tarification** avant le vol |

28 variables post-vol retirées, 13 conservées. Le second modèle est moins précis mais honnête sur l'information disponible : c'est celui qu'il faut déployer pour fixer un prix.

### Classification de la satisfaction (section 11)

Cible binaire : `satisfied` = 1 contre *neutral* + *dissatisfied* = 0 (43,4 % de positifs). Métriques suivies : F1, AUC, rappel et matrice de confusion — l'accuracy seule est trompeuse.

| Modèle | Accuracy | Précision | Rappel | F1 | AUC |
|---|---|---|---|---|---|
| **Gradient boosting (HGB)** | **0,965** | **0,977** | **0,942** | **0,959** | **0,995** |
| Forêt aléatoire | 0,964 | 0,976 | 0,939 | 0,957 | 0,995 |
| Régression logistique | 0,897 | 0,873 | 0,893 | 0,883 | 0,959 |
| Baseline (classe majoritaire) | 0,566 | 0,000 | 0,000 | 0,000 | 0,500 |

- La **baseline** illustre le piège de l'accuracy : 56,6 % en prédisant toujours « non satisfait », avec F1 = 0 et AUC = 0,5.
- La **régression logistique** perd ~7 points : la frontière n'est pas linéaire, elle dépend d'interactions profil × notes.
- **Forêt et boosting se rejoignent** à 96 % : au-delà d'un seuil, ce sont les variables qui décident, plus l'algorithme.
- **Sans le profil de réservation** (section 11.4), la performance reste à **95,0 % d'accuracy, F1 0,942, AUC 0,991** : le service perçu porte l'essentiel du signal. Les leviers authentiques sont `online_boarding`, `time_convenient`, `gate_location` (et `wifi`, pour partie artefact).

### Ce que vaut le modèle face à une règle simple

La règle « satisfait si score de service > 3,5 » atteint **72,5 %** d'accuracy. Le modèle apporte **+24 points** en apprenant que le seuil dépend du motif de voyage : c'est le gain réel du machine learning sur une règle de gestion, et il est substantiel.

---

## 💡 Enseignements clés

1. **Le prix est presque entièrement reconstruit** (R² ≈ 1) par la cabine, la distance et les notes — excellent outil d'audit tarifaire, pas de tarification, car les notes n'existent qu'après le vol.
2. **La satisfaction est cumulative et à seuil** : le profil fixe le niveau de référence, le score de service fait basculer, le retard agit dès les premières minutes puis sature.
3. **Prix et satisfaction partagent les mêmes déterminants** : un client qui paie cher n'est pas un client mal servi, et le retard compte surtout par ce qu'il fait aux notes.
4. **Le service explique *pourquoi*, le profil explique *qui*** : retirer le profil ne coûte que 1,7 point de F1, ce qui rend les leviers opérationnels.

---

## 🚀 Installation et exécution

### Prérequis

Python 3.12 (version d'exécution du notebook), avec :

```bash
python -m venv .venv
source .venv/bin/activate        # Windows : .venv\Scripts\activate
pip install numpy pandas matplotlib seaborn scipy scikit-learn joblib jupyter
```

Versions utilisées pour l'exécution livrée : `numpy 2.5.5`, `pandas 3.1.1`, `scikit-learn 1.9.1`, `seaborn 0.13.2`.

### Lancer le notebook

```bash
jupyter lab customer-satisfaction-prediction.ipynb
```

Puis *Run All*. Le notebook est déjà exécuté : ses 40 cellules de code, ses tableaux et ses 20 figures sont inclus dans le fichier. Un run complet prend environ **10 minutes sur 2 cœurs** (forêt aléatoire et recherche d'hyperparamètres comprises).

`dataset.csv` doit rester à la racine : le notebook y lit les données et écrit dans `figures/` et `models/`.

### Réutiliser les modèles sauvegardés

```python
import joblib
import pandas as pd

charge = joblib.load("models/modeles_avion.joblib")
# dict : regression_prix, classification_satisfaction, features, cat_features, num_features

# add_features() est définie en section 6 du notebook ; elle doit être chargée
# avant l'appel, car l'artefact contient les pipelines de pré-traitement,
# pas la construction des variables métier.
client = add_features(pd.DataFrame([{ ... }]))

prix = charge["regression_prix"].predict(client[charge["features"]])[0]
proba = charge["classification_satisfaction"].predict_proba(client[charge["features"]])[0, 1]
```

Contrôle effectué dans le notebook : sur un client du jeu de test, prix réel **868 €** → prédit **881 €**, avec rechargement depuis le fichier `.joblib`.

Exemples de scénarios (section 13) :

| Scénario | Prix estimé | P(satisfait) |
|---|---|---|
| Personnel, Eco 900 miles, notes 5/5 | 1 520 € | 0,9997 |
| … notes ramenées à 3/5 | 894 € | 0,0009 |
| … 5/5 avec 2 h de retard | 1 514 € | 0,9995 |
| Affaires, Business 1 800 miles, notes 5/5 | 4 137 € | 0,9998 |
| … Business avec des notes de 3/5 | 2 446 € | 0,0505 |
| … Business 5/5 avec 3 h de retard | 4 118 € | 0,9997 |

---

## ⚠️ Limites

- **Variables post-vol dans le modèle de prix** : les notes expliquent une grande partie du prix ici, mais n'existent qu'après le vol. Le R² ≈ 1 est un plafond artificiel pour la tarification (0,777 sans ces variables).
- **Cible générée par des règles** : la satisfaction suit des paliers nets (89,8 % des *neutral* sont des voyages personnels). Les 96 % d'accuracy sont optimistes par rapport à des données réelles.
- **Artefact de la note 0** : « non applicable » ou « très mauvaise note » selon la dimension ; conservée et signalée, mais elle pollue les importances.
- **Pas de dimension temporelle** : découpage aléatoire, donc performance surestimée pour une mise en production ; une validation temporelle serait nécessaire.
- **Winsorisation des retards** : stabilise les modèles mais écrase les cas extrêmes, précisément les plus coûteux en insatisfaction.

---

## ✨ Pistes d'amélioration

- Enrichir avec la **date de réservation**, la concurrence, la saisonnalité et le contexte de vol.
- Traiter proprement la **note 0** (modalité « non applicable » explicite) et reprendre les importances.
- **Validation temporelle** plutôt qu'aléatoire, plus proche de l'usage en production.
- Modéliser la satisfaction **par vol** avec des effets mixtes (client / route).
- Tester XGBoost / LightGBM — à titre de comparaison, le plafond étant ici fixé par les données.
- Exposer le scorer via une **API FastAPI** ou un tableau de bord Streamlit.

---

## 👤 Auteur

**Alex Alkhatib**
Projet Machine Learning — Satisfaction & Price Prediction

## 📄 Licence

MIT License — Copyright (c) 2025 Alex Alkhatib
