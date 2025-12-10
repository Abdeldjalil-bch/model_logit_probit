# 🚗 Analyse de la Sinistralité en Assurance Automobile

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/streamlit-1.28+-red.svg)
![StatsModels](https://img.shields.io/badge/statsmodels-0.14+-green.svg)

Application web interactive pour l'analyse des déterminants de la sinistralité en assurance automobile utilisant la régression logistique (modèles Logit et Probit).

## 📋 Description

Cette application permet aux professionnels de l'assurance d'identifier et quantifier les facteurs de risque influençant la probabilité de sinistre automobile. Elle fournit une analyse statistique complète avec interprétations détaillées et visualisations interactives.

## ✨ Fonctionnalités

### 📊 Analyse Complète
- **Exploration des données** : Statistiques descriptives et distributions
- **Tests préliminaires** : VIF (multicolinéarité), équilibre de l'échantillon
- **Modélisation** : Estimation des modèles Logit et Probit
- **Interprétation** : Odds Ratios et effets marginaux automatiques
- **Évaluation** : Matrices de confusion, courbes ROC, métriques de performance
- **Rapport** : Synthèse exportable avec recommandations

### 🎯 Modèles Statistiques
- **Logit** : Régression logistique avec distribution logistique
- **Probit** : Régression logistique avec distribution normale
- **Comparaison** : AIC, BIC, Pseudo R², Log-vraisemblance

### 📈 Visualisations
- Distributions de la variable cible
- Taux de sinistralité par variable
- Matrices de confusion
- Courbes ROC (AUC)
- Graphiques d'Odds Ratios

### 💡 Interprétations Automatiques
- Calcul et interprétation des Odds Ratios
- Effets marginaux sur la probabilité
- Recommandations pour la tarification
- Analyse de la capacité prédictive

## 🚀 Installation

### Prérequis
- Python 3.8 ou supérieur
- pip

### Étapes

```bash
# Cloner le repository
git clone https://github.com/VOTRE_USERNAME/model_logit_probit.git
cd model_logit_probit

# Créer un environnement virtuel
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Installer les dépendances
pip install -r requirements.txt

# Lancer l'application
streamlit run app.py
```

## 📦 Dépendances

```txt
streamlit>=1.28.0
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.12.0
scipy>=1.10.0
statsmodels>=0.14.0
scikit-learn>=1.3.0
```

## 💻 Utilisation

### 1. Lancer l'application
```bash
streamlit run app.py
```

### 2. Charger vos données
- Cliquez sur "Charger votre fichier CSV" dans la barre latérale
- Votre fichier doit contenir les colonnes requises (voir format ci-dessous)

### 3. Explorer les résultats
Naviguez entre les sections :
- 🏠 **Accueil** : Présentation et instructions
- 📈 **Données & Statistiques** : Exploration descriptive
- 🔬 **Tests Préliminaires** : VIF et tests d'équilibre
- 📊 **Modèles Logit & Probit** : Estimation et résultats
- 🎯 **Interprétation** : Odds Ratios et effets
- 📉 **Capacité Prédictive** : ROC, AUC, métriques
- 📋 **Rapport Final** : Synthèse et export

## 📊 Format des Données

Votre fichier CSV doit contenir les colonnes suivantes :

| Colonne | Type | Description | Valeurs |
|---------|------|-------------|---------|
| `sexe` | Numérique | Sexe du conducteur | 0=Femme, 1=Homme |
| `age_conducteur` | Numérique | Âge du conducteur | En années |
| `age_permis` | Numérique | Ancienneté du permis | En années |
| `age_vehicule` | Numérique | Âge du véhicule | En années |
| `genre` | Numérique | Genre du véhicule | Code véhicule |
| `puissance` | Numérique | Puissance fiscale | En CV |
| `usage` | Catégorielle | Type d'usage | 1=Fonctionnaire, 2=Affaire, 3=Commerce, 4=Taxi |
| `s` | Numérique | Nombre de sinistres | Entier |
| `s0` | Binaire | **Variable cible** | 0=Non-sinistre, 1=Sinistre |

### Exemple de données

```csv
sexe,age_conducteur,age_permis,age_vehicule,genre,puissance,usage,s,s0
1,35,15,5,1,7,1,0,0
0,28,8,2,2,5,2,1,1
1,52,30,10,1,9,3,0,0
```

## 📈 Résultats Fournis

### Modèles Statistiques
- **Coefficients** : Estimation avec erreurs standard et p-values
- **Odds Ratios** : Interprétation du risque relatif
- **Intervalles de confiance** : IC à 95%
- **Tests de significativité** : ***, **, *, ns

### Métriques de Performance
- **Pseudo R² (McFadden)** : Qualité d'ajustement
- **AIC/BIC** : Critères d'information
- **AUC** : Aire sous la courbe ROC
- **Accuracy, Sensibilité, Spécificité** : Métriques de classification

### Interprétations
- **Effets sur le risque** : Augmentation/diminution en %
- **Effets marginaux** : Changement de probabilité
- **Recommandations** : Actions pour la tarification

## 🎓 Contexte Académique

Projet développé dans le cadre de mes études en **Data Science et Intelligence Artificielle** à l'**ENSSEA**.

**Objectifs pédagogiques :**
- Maîtriser la régression logistique pour variables binaires
- Comprendre et interpréter les Odds Ratios
- Évaluer la capacité prédictive des modèles
- Appliquer les statistiques à un cas réel d'assurance

## 📖 Méthodologie

### 1. Préparation des données
- Chargement et nettoyage
- Détection de la multicolinéarité (VIF)
- Vérification de l'équilibre

### 2. Estimation des modèles
- Régression Logit (distribution logistique)
- Régression Probit (distribution normale)
- Tests de significativité globale (LR test)

### 3. Interprétation
- Calcul des Odds Ratios
- Effets marginaux
- Intervalles de confiance

### 4. Évaluation
- Matrice de confusion
- Courbe ROC et AUC
- Métriques de performance

### 5. Recommandations
- Facteurs de risque identifiés
- Implications pour la tarification
- Pistes d'amélioration

## 🔍 Exemples d'Interprétation

### Odds Ratio = 1.15 (Puissance)
> "Chaque CV supplémentaire augmente le risque de sinistre de 15%"

### Odds Ratio = 0.85 (Âge conducteur)
> "Chaque année supplémentaire réduit le risque de sinistre de 15%"

### AUC = 0.78
> "Le modèle a une bonne capacité à distinguer sinistres et non-sinistres"

## ⚠️ Limites et Améliorations

### Limites actuelles
- Variables disponibles limitées
- Données d'une seule période
- Modèle linéaire simple

### Améliorations possibles
- **Variables supplémentaires** : Données télématiques, géolocalisation
- **Modèles avancés** : Machine Learning (Random Forest, XGBoost)
- **Validation** : Cross-validation, échantillon test séparé
- **Temps** : Analyse de survie pour le délai avant sinistre

## 📥 Export et Rapports

L'application permet d'exporter :
- ✅ Résultats des modèles (CSV)
- ✅ Résumé statistique complet (TXT)
- ✅ Rapport d'analyse (Markdown)

## 🤝 Contribution

Les contributions sont bienvenues ! Pour contribuer :
1. Forkez le projet
2. Créez une branche (`git checkout -b feature/AmazingFeature`)
3. Committez vos changements (`git commit -m 'Add AmazingFeature'`)
4. Pushez vers la branche (`git push origin feature/AmazingFeature`)
5. Ouvrez une Pull Request

## 📝 License

MIT License - voir [LICENSE](LICENSE) pour plus de détails.

## 👨‍💻 Auteur

**Boucherite Ahmed Abdeldjalil**
- 🎓 Étudiant Data Science & IA - ENSSEA
- 📧 Email : a.a.boucherite@gmail.com
- 💼 LinkedIn : [linkedin.com/in/abdeldjalil-boucherite](https://www.linkedin.com/in/abdeldjalil-boucherite-745619378)

## 📚 Références

- **Régression logistique** : Hosmer, D.W. & Lemeshow, S. (2000). Applied Logistic Regression
- **Assurance automobile** : Denuit, M. & Charpentier, A. (2004). Mathématiques de l'assurance non-vie
- **Odds Ratios** : Szumilas, M. (2010). Explaining Odds Ratios. Journal of the Canadian Academy

---
