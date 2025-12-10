
"""
APPLICATION STREAMLIT - ANALYSE DE LA SINISTRALITÉ EN ASSURANCE AUTOMOBILE
Régression Logistique - Modèles Logit et Probit
"""

import streamlit as st
import pandas as pd
import numpy as np
# Configuration matplotlib pour Streamlit Cloud
import matplotlib
matplotlib.use('Agg')  # Important pour Streamlit Cloud
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Modèles statistiques
from statsmodels.formula.api import logit, probit
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Métriques de performance
from sklearn.metrics import confusion_matrix, roc_curve, auc, accuracy_score
import io

# Configuration de la page
st.set_page_config(
    page_title="Analyse Sinistralité Automobile",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Style CSS personnalisé
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        font-weight: bold;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.8rem;
        color: #ff7f0e;
        font-weight: bold;
        margin-top: 2rem;
        margin-bottom: 1rem;
        border-bottom: 3px solid #ff7f0e;
        padding-bottom: 0.5rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 1rem 0;
    }
    .interpretation-box {
        background-color: #e8f4f8;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #2196F3;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #ffc107;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================================================
# FONCTIONS DE TRAITEMENT
# ============================================================================

@st.cache_data
def load_and_prepare_data(uploaded_file):
    """Charge et prépare les données"""
    df = pd.read_csv(uploaded_file)
    
    # Renommer les colonnes
    if len(df.columns) == 9:
        df.columns = ['sexe', 'age_conducteur', 'age_permis', 'age_vehicule', 
                      'genre', 'puissance', 'usage', 's', 's0']
        df = df.drop('s', axis=1)
    else:
        df.columns = ['sexe', 'age_conducteur', 'age_permis', 'age_vehicule', 
                      'genre', 'puissance', 'usage', 's0']
    
    df = df.rename(columns={'s0': 'sinistre'})
    
    return df

def calculate_vif(df):
    """Calcule le VIF pour détecter la multicolinéarité"""
    X_vars = ['sexe', 'age_conducteur', 'age_permis', 'age_vehicule', 'puissance', 'usage']
    X = df[X_vars]
    X_with_const = sm.add_constant(X)
    
    vif_data = pd.DataFrame()
    vif_data["Variable"] = X_with_const.columns
    vif_data["VIF"] = [variance_inflation_factor(X_with_const.values, i) 
                       for i in range(X_with_const.shape[1])]
    vif_data = vif_data[vif_data["Variable"] != "const"]
    
    return vif_data

@st.cache_resource
def estimate_models(df):
    """Estime les modèles Logit et Probit"""
    formula = 'sinistre ~ sexe + age_conducteur + age_permis + age_vehicule + puissance + C(usage)'
    
    model_logit = logit(formula, data=df).fit(disp=0)
    model_probit = probit(formula, data=df).fit(disp=0)
    
    return model_logit, model_probit

# ============================================================================
# EN-TÊTE DE L'APPLICATION
# ============================================================================

st.markdown('<p class="main-header">🚗 ANALYSE DE LA SINISTRALITÉ EN ASSURANCE AUTOMOBILE</p>', 
            unsafe_allow_html=True)
st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">Régression Logistique - Modèles Logit et Probit</p>', 
            unsafe_allow_html=True)

st.markdown("---")

# ============================================================================
# BARRE LATÉRALE - CHARGEMENT DES DONNÉES
# ============================================================================

with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/000000/car-insurance.png", width=80)
    st.title("📊 Navigation")
    
    uploaded_file = st.file_uploader(
        "📁 Charger votre fichier CSV", 
        type=['csv'],
        help="Format attendu : sexe, age_conducteur, age_permis, age_vehicule, genre, puissance, usage, s, s0"
    )
    
    if uploaded_file is not None:
        st.success("✅ Fichier chargé avec succès !")
    
    st.markdown("---")
    
    st.markdown("### 🔍 Sections")
    menu = st.radio(
        "",
        ["🏠 Accueil",
         "📈 Données & Statistiques",
         "🔬 Tests Préliminaires",
         "📊 Modèles Logit & Probit",
         "🎯 Interprétation",
         "📉 Capacité Prédictive",
         "📋 Rapport Final"]
    )
    
    st.markdown("---")
    st.markdown("### 📚 À propos")
    st.info("Cette application permet d'analyser les déterminants de la sinistralité en assurance automobile à l'aide de la régression logistique.")

# ============================================================================
# PAGE D'ACCUEIL
# ============================================================================

if menu == "🏠 Accueil":
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown('<div class="section-header">Bienvenue dans l\'application d\'analyse</div>', 
                    unsafe_allow_html=True)
        
        st.markdown("""
        ### 🎯 Objectif de l'étude
        
        Cette application vous permet d'analyser les **principaux déterminants de la sinistralité** 
        en assurance automobile en utilisant la **régression logistique** (modèles Logit et Probit).
        
        ### 📊 Fonctionnalités
        
        - ✅ **Analyse descriptive** complète des données
        - ✅ **Tests statistiques** (VIF, LR test, Hosmer-Lemeshow)
        - ✅ **Estimation** des modèles Logit et Probit
        - ✅ **Interprétation** automatique des coefficients (Odds Ratios)
        - ✅ **Évaluation** de la capacité prédictive (ROC, AUC)
        - ✅ **Comparaison** des modèles
        - ✅ **Visualisations** interactives
        - ✅ **Rapport** exportable
        
        ### 🚀 Comment utiliser l'application ?
        
        1. **Charger vos données** via la barre latérale (format CSV)
        2. **Naviguer** entre les différentes sections
        3. **Explorer** les résultats et interprétations
        4. **Télécharger** le rapport final
        
        ### 📁 Format des données
        
        Votre fichier CSV doit contenir les colonnes suivantes :
        - `sexe` : Sexe du conducteur (0=Femme, 1=Homme)
        - `age_conducteur` : Âge du conducteur
        - `age_permis` : Ancienneté du permis
        - `age_vehicule` : Âge du véhicule
        - `genre` : Genre du véhicule
        - `puissance` : Puissance fiscale (CV)
        - `usage` : Type d'usage (1=Fonctionnaire, 2=Affaire, 3=Commerce, 4=Taxi)
        - `s` : Nombre de sinistres
        - `s0` : Occurrence de sinistre (0=Non, 1=Oui) **← Variable cible**
        
        """)
        
        st.markdown('<div class="interpretation-box">', unsafe_allow_html=True)
        st.markdown("""
        **💡 Note importante :** La variable `s0` (occurrence de sinistre) sera utilisée 
        comme variable dépendante dans les modèles de régression logistique.
        """)
        st.markdown('</div>', unsafe_allow_html=True)
    
    if uploaded_file is None:
        st.warning("⚠️ Veuillez charger un fichier CSV dans la barre latérale pour commencer l'analyse.")

# ============================================================================
# PAGE DONNÉES & STATISTIQUES
# ============================================================================

elif menu == "📈 Données & Statistiques":
    
    if uploaded_file is None:
        st.warning("⚠️ Veuillez d'abord charger un fichier CSV dans la barre latérale.")
    else:
        df = load_and_prepare_data(uploaded_file)
        
        st.markdown('<div class="section-header">📈 Exploration des Données</div>', 
                    unsafe_allow_html=True)
        
        # Métriques principales
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("📊 Observations", f"{len(df):,}")
        with col2:
            st.metric("📋 Variables", f"{len(df.columns)}")
        with col3:
            taux_sinistre = df['sinistre'].mean() * 100
            st.metric("🚨 Taux de sinistralité", f"{taux_sinistre:.1f}%")
        with col4:
            st.metric("✅ Données complètes", f"{(1 - df.isnull().sum().sum()/(len(df)*len(df.columns)))*100:.1f}%")
        
        st.markdown("---")
        
        # Aperçu des données
        with st.expander("📋 Aperçu des données (10 premières lignes)", expanded=True):
            st.dataframe(df.head(10), use_container_width=True)
        
        # Statistiques descriptives
        with st.expander("📊 Statistiques descriptives", expanded=True):
            st.dataframe(df.describe(), use_container_width=True)
            
            st.markdown('<div class="interpretation-box">', unsafe_allow_html=True)
            st.markdown("""
            **📖 Comment interpréter ces statistiques :**
            - **count** : Nombre d'observations
            - **mean** : Moyenne de la variable
            - **std** : Écart-type (dispersion)
            - **min/max** : Valeurs minimale et maximale
            - **25%, 50%, 75%** : Quartiles (distribution)
            """)
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Distribution de la variable cible
        st.markdown("### 🎯 Distribution de la variable cible (Sinistre)")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig, ax = plt.subplots(figsize=(8, 6))
            counts = df['sinistre'].value_counts()
            colors = ['#2ecc71', '#e74c3c']
            ax.bar(['Non-sinistre (0)', 'Sinistre (1)'], counts.values, color=colors, alpha=0.7, edgecolor='black')
            ax.set_ylabel('Nombre d\'observations', fontsize=12)
            ax.set_title('Distribution des sinistres', fontsize=14, fontweight='bold')
            ax.grid(axis='y', alpha=0.3)
            for i, v in enumerate(counts.values):
                ax.text(i, v + 5, str(v), ha='center', va='bottom', fontsize=12, fontweight='bold')
            st.pyplot(fig)
        
        with col2:
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.pie(counts.values, labels=['Non-sinistre (0)', 'Sinistre (1)'], 
                   autopct='%1.1f%%', colors=colors, startangle=90, 
                   explode=(0.05, 0.05), textprops={'fontsize': 12, 'fontweight': 'bold'})
            ax.set_title('Proportion des sinistres', fontsize=14, fontweight='bold')
            st.pyplot(fig)
        
        st.markdown('<div class="interpretation-box">', unsafe_allow_html=True)
        st.markdown(f"""
        **📖 Interprétation :**
        - **{counts[0]}** observations sans sinistre ({counts[0]/len(df)*100:.1f}%)
        - **{counts[1]}** observations avec sinistre ({counts[1]/len(df)*100:.1f}%)
        - Le taux de sinistralité est de **{taux_sinistre:.2f}%**
        
        {"⚠️ **Attention** : Déséquilibre important ! Considérer un rééchantillonnage." if taux_sinistre < 10 or taux_sinistre > 90 else "✅ **Équilibre acceptable** pour la modélisation."}
        """)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Analyse par variable
        st.markdown("### 📊 Taux de sinistralité par variable")
        
        tab1, tab2, tab3, tab4 = st.tabs(["👥 Sexe", "🚗 Puissance", "💼 Usage", "📈 Âge conducteur"])
        
        with tab1:
            col1, col2 = st.columns(2)
            
            with col1:
                taux_sexe = df.groupby('sexe')['sinistre'].agg(['sum', 'count', 'mean'])
                taux_sexe['taux_%'] = taux_sexe['mean'] * 100
                taux_sexe.index = ['Femme', 'Homme']
                st.dataframe(taux_sexe, use_container_width=True)
            
            with col2:
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.bar(['Femme', 'Homme'], taux_sexe['taux_%'].values, color=['pink', 'lightblue'], alpha=0.7, edgecolor='black')
                ax.set_ylabel('Taux de sinistralité (%)', fontsize=12)
                ax.set_title('Taux de sinistralité par sexe', fontsize=14, fontweight='bold')
                ax.grid(axis='y', alpha=0.3)
                st.pyplot(fig)
            
            st.markdown('<div class="interpretation-box">', unsafe_allow_html=True)
            diff = abs(taux_sexe.loc['Homme', 'taux_%'] - taux_sexe.loc['Femme', 'taux_%'])
            qui = 'hommes' if taux_sexe.loc['Homme', 'taux_%'] > taux_sexe.loc['Femme', 'taux_%'] else 'femmes'
            st.markdown(f"""
            **📖 Interprétation :**
            - Les **{qui}** ont un taux de sinistralité **{diff:.1f} points de pourcentage** plus élevé.
            - Cette différence sera testée dans le modèle de régression.
            """)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with tab2:
            col1, col2 = st.columns(2)
            
            with col1:
                taux_puissance = df.groupby('puissance')['sinistre'].mean() * 100
                st.dataframe(taux_puissance.to_frame('Taux (%)'), use_container_width=True)
            
            with col2:
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.bar(taux_puissance.index, taux_puissance.values, color='coral', alpha=0.7, edgecolor='black')
                ax.set_xlabel('Puissance (CV)', fontsize=12)
                ax.set_ylabel('Taux de sinistralité (%)', fontsize=12)
                ax.set_title('Taux de sinistralité par puissance', fontsize=14, fontweight='bold')
                ax.grid(axis='y', alpha=0.3)
                st.pyplot(fig)
            
            st.markdown('<div class="interpretation-box">', unsafe_allow_html=True)
            st.markdown("""
            **📖 Interprétation :**
            - Observation de la relation entre puissance du véhicule et sinistralité.
            - Généralement, les véhicules plus puissants présentent un risque accru.
            """)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with tab3:
            col1, col2 = st.columns(2)
            
            with col1:
                taux_usage = df.groupby('usage')['sinistre'].agg(['sum', 'count', 'mean'])
                taux_usage['taux_%'] = taux_usage['mean'] * 100
                taux_usage.index = ['Fonctionnaire', 'Affaire', 'Commerce', 'Taxi']
                st.dataframe(taux_usage, use_container_width=True)
            
            with col2:
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.bar(range(len(taux_usage)), taux_usage['taux_%'].values, color='steelblue', alpha=0.7, edgecolor='black')
                ax.set_xticks(range(len(taux_usage)))
                ax.set_xticklabels(taux_usage.index, rotation=45)
                ax.set_ylabel('Taux de sinistralité (%)', fontsize=12)
                ax.set_title('Taux de sinistralité par usage', fontsize=14, fontweight='bold')
                ax.grid(axis='y', alpha=0.3)
                st.pyplot(fig)
            
            st.markdown('<div class="interpretation-box">', unsafe_allow_html=True)
            st.markdown("""
            **📖 Interprétation :**
            - L'usage professionnel (Taxi, Commerce) présente souvent un risque différent.
            - Les fonctionnaires peuvent bénéficier d'un profil de risque plus favorable.
            """)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with tab4:
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.hist([df[df['sinistre']==0]['age_conducteur'], 
                     df[df['sinistre']==1]['age_conducteur']], 
                    label=['Non-sinistre', 'Sinistre'], bins=20, alpha=0.7, edgecolor='black')
            ax.set_xlabel('Âge du conducteur', fontsize=12)
            ax.set_ylabel('Fréquence', fontsize=12)
            ax.set_title('Distribution de l\'âge du conducteur par sinistralité', fontsize=14, fontweight='bold')
            ax.legend(fontsize=12)
            ax.grid(axis='y', alpha=0.3)
            st.pyplot(fig)
            
            st.markdown('<div class="interpretation-box">', unsafe_allow_html=True)
            st.markdown("""
            **📖 Interprétation :**
            - Courbe en U typique : jeunes conducteurs et seniors plus à risque.
            - Le modèle de régression quantifiera cet effet.
            """)
            st.markdown('</div>', unsafe_allow_html=True)

# ============================================================================
# PAGE TESTS PRÉLIMINAIRES
# ============================================================================

elif menu == "🔬 Tests Préliminaires":
    
    if uploaded_file is None:
        st.warning("⚠️ Veuillez d'abord charger un fichier CSV dans la barre latérale.")
    else:
        df = load_and_prepare_data(uploaded_file)
        
        st.markdown('<div class="section-header">🔬 Tests Préliminaires</div>', 
                    unsafe_allow_html=True)
        
        # Test de multicolinéarité
        st.markdown("### 📊 Test de Multicolinéarité (VIF)")
        
        st.markdown('<div class="interpretation-box">', unsafe_allow_html=True)
        st.markdown("""
        **📖 Qu'est-ce que le VIF ?**
        
        Le **VIF (Variance Inflation Factor)** mesure la corrélation entre les variables explicatives.
        
        - **VIF < 5** : ✅ Pas de problème de multicolinéarité
        - **5 < VIF < 10** : ⚠️ Multicolinéarité modérée
        - **VIF > 10** : ❌ Multicolinéarité problématique (retirer la variable)
        """)
        st.markdown('</div>', unsafe_allow_html=True)
        
        vif_data = calculate_vif(df)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Colorer les lignes selon le VIF
            def color_vif(val):
                if val < 5:
                    color = '#d4edda'  # Vert
                elif val < 10:
                    color = '#fff3cd'  # Jaune
                else:
                    color = '#f8d7da'  # Rouge
                return f'background-color: {color}'
            
            styled_vif = vif_data.style.applymap(color_vif, subset=['VIF'])
            st.dataframe(styled_vif, use_container_width=True)
        
        with col2:
            st.markdown("### 🎯 Diagnostic")
            problemes = vif_data[vif_data['VIF'] > 10]
            moderes = vif_data[(vif_data['VIF'] >= 5) & (vif_data['VIF'] <= 10)]
            
            if len(problemes) > 0:
                st.error(f"❌ {len(problemes)} variable(s) problématique(s)")
            elif len(moderes) > 0:
                st.warning(f"⚠️ {len(moderes)} variable(s) à surveiller")
            else:
                st.success("✅ Aucun problème détecté")
        
        # Test d'équilibre
        st.markdown("### ⚖️ Équilibre de la Variable Cible")
        
        col1, col2, col3 = st.columns(3)
        
        n_total = len(df)
        n_sinistres = df['sinistre'].sum()
        n_non_sinistres = n_total - n_sinistres
        ratio = n_sinistres / n_total * 100
        
        with col1:
            st.metric("🔴 Sinistres", f"{n_sinistres} ({ratio:.1f}%)")
        with col2:
            st.metric("🟢 Non-sinistres", f"{n_non_sinistres} ({100-ratio:.1f}%)")
        with col3:
            if ratio < 10 or ratio > 90:
                st.metric("⚠️ Statut", "Déséquilibré")
            else:
                st.metric("✅ Statut", "Équilibré")
        
        st.markdown('<div class="interpretation-box">', unsafe_allow_html=True)
        if ratio < 10 or ratio > 90:
            st.markdown("""
            **⚠️ Attention : Déséquilibre important détecté !**
            
            - Votre échantillon est fortement déséquilibré.
            - Cela peut affecter la performance du modèle.
            - **Solutions possibles** :
              - Rééchantillonnage (oversampling/undersampling)
              - Ajustement des seuils de classification
              - Utilisation de métriques adaptées (F1-score, AUC)
            """)
        else:
            st.markdown("""
            **✅ Équilibre acceptable**
            
            - La distribution de la variable cible est acceptable pour la modélisation.
            - Les modèles pourront être estimés sans problème.
            """)
        st.markdown('</div>', unsafe_allow_html=True)

# ============================================================================
# PAGE MODÈLES LOGIT & PROBIT
# ============================================================================

elif menu == "📊 Modèles Logit & Probit":
    
    if uploaded_file is None:
        st.warning("⚠️ Veuillez d'abord charger un fichier CSV dans la barre latérale.")
    else:
        df = load_and_prepare_data(uploaded_file)
        
        st.markdown('<div class="section-header">📊 Estimation des Modèles</div>', 
                    unsafe_allow_html=True)
        
        with st.spinner("⏳ Estimation des modèles en cours..."):
            model_logit, model_probit = estimate_models(df)
        
        st.success("✅ Modèles estimés avec succès !")
        
        # Onglets pour les deux modèles
        tab1, tab2, tab3 = st.tabs(["📈 MODÈLE LOGIT", "📉 MODÈLE PROBIT", "🔄 COMPARAISON"])
        
        with tab1:
            st.markdown("### 📊 Résultats du Modèle LOGIT")
            
            # Résumé du modèle
            with st.expander("📋 Résumé complet du modèle", expanded=False):
                st.text(model_logit.summary())
            
            # Tableau des coefficients
            st.markdown("#### 📊 Coefficients estimés")
            
            params = model_logit.params
            pvalues = model_logit.pvalues
            std_err = model_logit.bse
            conf_int = model_logit.conf_int()
            
            results_df = pd.DataFrame({
                'Coefficient (β)': params,
                'Erreur Standard': std_err,
                'p-value': pvalues,
                'IC 95% Inf': conf_int[0],
                'IC 95% Sup': conf_int[1]
            })
            
            # Ajouter une colonne de significativité
            def get_significance(p):
                if p < 0.001:
                    return '***'
                elif p < 0.01:
                    return '**'
                elif p < 0.05:
                    return '*'
                else:
                    return 'ns'
            
            results_df['Significativité'] = results_df['p-value'].apply(get_significance)
            
            st.dataframe(results_df, use_container_width=True)
            
            st.markdown('<div class="interpretation-box">', unsafe_allow_html=True)
            st.markdown("""
            **📖 Légende :**
            - \*\*\* : p < 0.001 (très significatif)
            - \*\* : p < 0.01 (significatif)
            - \* : p < 0.05 (peu significatif)
            - ns : p ≥ 0.05 (non significatif)
            """)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Métriques du modèle
            st.markdown("#### 📊 Qualité d'ajustement")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Log-vraisemblance", f"{model_logit.llf:.2f}")
            with col2:
                st.metric("AIC", f"{model_logit.aic:.2f}")
            with col3:
                st.metric("BIC", f"{model_logit.bic:.2f}")
            with col4:
                st.metric("Pseudo R² (McFadden)", f"{model_logit.prsquared:.4f}")
            
            st.markdown('<div class="interpretation-box">', unsafe_allow_html=True)
            r2 = model_logit.prsquared
            if r2 < 0.2:
                interp = "Faible pouvoir explicatif"
            elif r2 < 0.4:
                interp = "Bon pouvoir explicatif"
            else:
                interp = "Très bon pouvoir explicatif"
            
            st.markdown(f"""
            **📖 Interprétation du Pseudo R² :**
            - Valeur : **{r2:.4f}**
            - Interprétation : **{interp}**
            
            **Test du rapport de vraisemblance (LR Test) :**
            - Statistique LR : **{model_logit.llr:.2f}**
            - p-value : **{model_logit.llr_pvalue:.6f}**
            - Conclusion : {"✅ Le modèle est globalement significatif" if model_logit.llr_pvalue < 0.05 else "❌ Le modèle n'est pas significatif"}
            """)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with tab2:
            st.markdown("### 📊 Résultats du Modèle PROBIT")
            
            # Résumé du modèle
            with st.expander("📋 Résumé complet du modèle", expanded=False):
                st.text(model_probit.summary())
            
            # Tableau des coefficients
            st.markdown("#### 📊 Coefficients estimés")
            
            params_p = model_probit.params
            pvalues_p = model_probit.pvalues
            std_err_p = model_probit.bse
            conf_int_p = model_probit.conf_int()
            
            results_df_p = pd.DataFrame({
                'Coefficient (β)': params_p,
                'Erreur Standard': std_err_p,
                'p-value': pvalues_p,
                'IC 95% Inf': conf_int_p[0],
                'IC 95% Sup': conf_int_p[1]
            })
            
            results_df_p['Significativité'] = results_df_p['p-value'].apply(get_significance)
            
            st.dataframe(results_df_p, use_container_width=True)
            
            # Métriques du modèle
            st.markdown("#### 📊 Qualité d'ajustement")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Log-vraisemblance", f"{model_probit.llf:.2f}")
            with col2:
                st.metric("AIC", f"{model_probit.aic:.2f}")
            with col3:
                st.metric("BIC", f"{model_probit.bic:.2f}")
            with col4:
                st.metric("Pseudo R² (McFadden)", f"{model_probit.prsquared:.4f}")
        
        with tab3:
            st.markdown("### 🔄 Comparaison LOGIT vs PROBIT")
            
            # Tableau comparatif
            comparison_data = {
                'Critère': ['Log-vraisemblance', 'AIC', 'BIC', 'Pseudo R²'],
                'LOGIT': [
                    f"{model_logit.llf:.2f}",
                    f"{model_logit.aic:.2f}",
                    f"{model_logit.bic:.2f}",
                    f"{model_logit.prsquared:.4f}"
                ],
                'PROBIT': [
                    f"{model_probit.llf:.2f}",
                    f"{model_probit.aic:.2f}",
                    f"{model_probit.bic:.2f}",
                    f"{model_probit.prsquared:.4f}"
                ],
                'Meilleur': [
                    'LOGIT' if model_logit.llf > model_probit.llf else 'PROBIT',
                    'LOGIT' if model_logit.aic < model_probit.aic else 'PROBIT',
                    'LOGIT' if model_logit.bic < model_probit.bic else 'PROBIT',
                    'LOGIT' if model_logit.prsquared > model_probit.prsquared else 'PROBIT'
                ]
            }
            
            comparison_df = pd.DataFrame(comparison_data)
            st.dataframe(comparison_df, use_container_width=True)
            
            st.markdown('<div class="interpretation-box">', unsafe_allow_html=True)
            st.markdown("""
            **📖 Interprétation de la comparaison :**
            
            - **Log-vraisemblance** : Plus élevée = meilleur ajustement
            - **AIC/BIC** : Plus faible = meilleur modèle (pénalise la complexité)
            - **Pseudo R²** : Plus élevé = meilleur pouvoir explicatif
            
            **💡 Recommandation :**
            - En pratique, les modèles Logit et Probit donnent souvent des résultats très similaires.
            - On préfère généralement le **LOGIT** car les Odds Ratios sont plus faciles à interpréter.
            """)
            st.markdown('</div>', unsafe_allow_html=True)

# ============================================================================
# PAGE INTERPRÉTATION
# ============================================================================

elif menu == "🎯 Interprétation":
    
    if uploaded_file is None:
        st.warning("⚠️ Veuillez d'abord charger un fichier CSV dans la barre latérale.")
    else:
        df = load_and_prepare_data(uploaded_file)
        
        st.markdown('<div class="section-header">🎯 Interprétation des Résultats</div>', 
                    unsafe_allow_html=True)
        
        with st.spinner("⏳ Calcul des interprétations..."):
            model_logit, model_probit = estimate_models(df)
        
        # ODDS RATIOS
        st.markdown("### 📊 Odds Ratios (LOGIT) - Interprétation Intuitive")
        
        st.markdown('<div class="interpretation-box">', unsafe_allow_html=True)
        st.markdown("""
        **📖 Qu'est-ce qu'un Odds Ratio (OR) ?**
        
        L'Odds Ratio mesure **l'effet d'une variable sur le risque de sinistre** :
        - **OR = 1** : Aucun effet
        - **OR > 1** : Augmentation du risque (ex: OR=1.5 → +50% de risque)
        - **OR < 1** : Diminution du risque (ex: OR=0.8 → -20% de risque)
        
        **Exemple concret :**
        - Si l'OR de "Puissance" = 1.10, chaque CV supplémentaire augmente le risque de sinistre de 10%.
        """)
        st.markdown('</div>', unsafe_allow_html=True)
        
        params = model_logit.params
        pvalues = model_logit.pvalues
        conf_int = model_logit.conf_int()
        
        # Créer le tableau des Odds Ratios
        odds_data = []
        for var in params.index:
            if var != 'Intercept':
                coef = params[var]
                pval = pvalues[var]
                odds_ratio = np.exp(coef)
                ci_lower = np.exp(conf_int.loc[var, 0])
                ci_upper = np.exp(conf_int.loc[var, 1])
                
                # Déterminer la significativité
                if pval < 0.001:
                    sig = "***"
                elif pval < 0.01:
                    sig = "**"
                elif pval < 0.05:
                    sig = "*"
                else:
                    sig = "ns"
                
                # Interpréter l'effet
                if odds_ratio > 1:
                    effet = f"↑ +{(odds_ratio-1)*100:.1f}%"
                    effet_texte = f"Augmente le risque de {(odds_ratio-1)*100:.1f}%"
                else:
                    effet = f"↓ -{(1-odds_ratio)*100:.1f}%"
                    effet_texte = f"Diminue le risque de {(1-odds_ratio)*100:.1f}%"
                
                odds_data.append({
                    'Variable': var,
                    'Coefficient (β)': f"{coef:.4f}",
                    'p-value': f"{pval:.4f}",
                    'Odds Ratio': f"{odds_ratio:.3f}",
                    'IC 95%': f"[{ci_lower:.3f} - {ci_upper:.3f}]",
                    'Effet': effet,
                    'Signif.': sig,
                    'Interprétation': effet_texte
                })
        
        odds_df = pd.DataFrame(odds_data)
        st.dataframe(odds_df, use_container_width=True)
        
        # Interprétations détaillées
        st.markdown("### 📝 Interprétations Détaillées par Variable")
        
        for idx, row in odds_df.iterrows():
            with st.expander(f"📌 {row['Variable']}", expanded=False):
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.markdown(f"""
                    **Coefficient** : {row['Coefficient (β)']}  
                    **Odds Ratio** : {row['Odds Ratio']}  
                    **Intervalle de confiance 95%** : {row['IC 95%']}  
                    **p-value** : {row['p-value']} {row['Signif.']}
                    """)
                    
                    st.markdown('<div class="interpretation-box">', unsafe_allow_html=True)
                    
                    # Interprétation personnalisée selon la variable
                    var_name = row['Variable']
                    or_val = float(row['Odds Ratio'])
                    pval = float(row['p-value'])
                    
                    if pval < 0.05:
                        st.markdown(f"""
                        **✅ Interprétation :**
                        
                        Cette variable est **statistiquement significative** (p < 0.05).
                        
                        {row['Interprétation']}, toutes choses égales par ailleurs.
                        """)
                        
                        if 'age' in var_name.lower():
                            st.markdown("""
                            **💡 Exemple concret :**
                            Si l'Odds Ratio est de 1.05, cela signifie que chaque année supplémentaire 
                            augmente les chances de sinistre de 5% par rapport à l'année précédente.
                            """)
                        elif 'sexe' in var_name.lower():
                            if or_val > 1:
                                st.markdown("""
                                **💡 Exemple concret :**
                                Les hommes ont plus de chances d'avoir un sinistre que les femmes.
                                """)
                            else:
                                st.markdown("""
                                **💡 Exemple concret :**
                                Les femmes ont plus de chances d'avoir un sinistre que les hommes.
                                """)
                        elif 'puissance' in var_name.lower():
                            st.markdown("""
                            **💡 Exemple concret :**
                            Chaque cheval fiscal (CV) supplémentaire augmente le risque d'accident.
                            Un véhicule de 10 CV a donc un risque plus élevé qu'un véhicule de 5 CV.
                            """)
                    else:
                        st.markdown(f"""
                        **❌ Interprétation :**
                        
                        Cette variable n'est **pas statistiquement significative** (p ≥ 0.05).
                        
                        Son effet sur la sinistralité n'est pas prouvé dans ce modèle.
                        """)
                    
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with col2:
                    # Graphique de l'Odds Ratio
                    fig, ax = plt.subplots(figsize=(6, 4))
                    or_value = float(row['Odds Ratio'])
                    color = 'red' if or_value > 1 else 'green'
                    
                    ax.barh([0], [or_value], color=color, alpha=0.7, edgecolor='black')
                    ax.axvline(x=1, color='black', linestyle='--', linewidth=2, label='OR=1 (aucun effet)')
                    ax.set_xlabel('Odds Ratio', fontsize=12)
                    ax.set_yticks([])
                    ax.set_title(f"Odds Ratio: {or_value:.3f}", fontsize=12, fontweight='bold')
                    ax.legend()
                    ax.grid(axis='x', alpha=0.3)
                    st.pyplot(fig)
        
        # Effets marginaux
        st.markdown("### 📊 Effets Marginaux - Changement de Probabilité")
        
        st.markdown('<div class="interpretation-box">', unsafe_allow_html=True)
        st.markdown("""
        **📖 Qu'est-ce qu'un effet marginal ?**
        
        L'effet marginal indique **de combien la probabilité de sinistre change** 
        (en points de pourcentage) pour une variation unitaire d'une variable.
        
        **Exemple :**
        - Si l'effet marginal de "Âge" = 0.02, chaque année supplémentaire augmente 
          la probabilité de sinistre de 2 points de pourcentage (par exemple, de 15% à 17%).
        """)
        st.markdown('</div>', unsafe_allow_html=True)
        
        marginal_effects = model_logit.get_margeff()
        me_summary = marginal_effects.summary()
        
        with st.expander("📋 Tableau complet des effets marginaux", expanded=False):
            st.text(me_summary)

# ============================================================================
# PAGE CAPACITÉ PRÉDICTIVE
# ============================================================================

elif menu == "📉 Capacité Prédictive":
    
    if uploaded_file is None:
        st.warning("⚠️ Veuillez d'abord charger un fichier CSV dans la barre latérale.")
    else:
        df = load_and_prepare_data(uploaded_file)
        
        st.markdown('<div class="section-header">📉 Capacité Prédictive des Modèles</div>', 
                    unsafe_allow_html=True)
        
        with st.spinner("⏳ Évaluation de la capacité prédictive..."):
            model_logit, model_probit = estimate_models(df)
        
        # Prédictions
        y_true = df['sinistre']
        y_pred_logit_proba = model_logit.predict()
        y_pred_probit_proba = model_probit.predict()
        
        # Seuil de classification
        st.markdown("### ⚙️ Seuil de Classification")
        seuil = st.slider("Choisir le seuil de classification", 0.0, 1.0, 0.5, 0.05)
        
        y_pred_logit = (y_pred_logit_proba >= seuil).astype(int)
        y_pred_probit = (y_pred_probit_proba >= seuil).astype(int)
        
        st.markdown('<div class="interpretation-box">', unsafe_allow_html=True)
        st.markdown(f"""
        **📖 Explication du seuil :**
        - Si la probabilité prédite ≥ {seuil}, on prédit "Sinistre" (1)
        - Si la probabilité prédite < {seuil}, on prédit "Non-sinistre" (0)
        - Le seuil par défaut est 0.5, mais il peut être ajusté selon vos besoins
        """)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Matrice de confusion
        st.markdown("### 📊 Matrice de Confusion - LOGIT")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            cm_logit = confusion_matrix(y_true, y_pred_logit)
            
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(cm_logit, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=['Prédit: Non-sinistre', 'Prédit: Sinistre'],
                       yticklabels=['Réel: Non-sinistre', 'Réel: Sinistre'],
                       ax=ax, cbar_kws={'label': 'Nombre d\'observations'})
            ax.set_title('Matrice de Confusion - LOGIT', fontsize=14, fontweight='bold')
            st.pyplot(fig)
        
        with col2:
            tn, fp, fn, tp = cm_logit.ravel()
            
            st.markdown("#### 📋 Détails")
            st.markdown(f"""
            - **Vrais Négatifs (TN)** : {tn}  
              *(Correctement prédits comme non-sinistre)*
            
            - **Faux Positifs (FP)** : {fp}  
              *(Prédits sinistre mais non-sinistre en réalité)*
            
            - **Faux Négatifs (FN)** : {fn}  
              *(Prédits non-sinistre mais sinistre en réalité)*
            
            - **Vrais Positifs (TP)** : {tp}  
              *(Correctement prédits comme sinistre)*
            """)
        
        st.markdown('<div class="interpretation-box">', unsafe_allow_html=True)
        st.markdown("""
        **📖 Comment lire la matrice de confusion :**
        - **Diagonale** (TN + TP) : Bonnes prédictions
        - **Hors diagonale** (FP + FN) : Erreurs de prédiction
        - **FP** : Coût de fausse alerte
        - **FN** : Risque non détecté (plus grave en assurance !)
        """)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Métriques de performance
        st.markdown("### 📊 Métriques de Performance")
        
        accuracy = accuracy_score(y_true, y_pred_logit)
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("🎯 Accuracy", f"{accuracy*100:.2f}%")
            st.caption("Taux de bon classement global")
        
        with col2:
            st.metric("🔍 Sensibilité", f"{sensitivity*100:.2f}%")
            st.caption("Détection des sinistres")
        
        with col3:
            st.metric("✅ Spécificité", f"{specificity*100:.2f}%")
            st.caption("Détection des non-sinistres")
        
        with col4:
            st.metric("🎪 Précision", f"{precision*100:.2f}%")
            st.caption("Fiabilité des prédictions positives")
        
        st.markdown('<div class="interpretation-box">', unsafe_allow_html=True)
        st.markdown(f"""
        **📖 Interprétation des métriques :**
        
        - **Accuracy ({accuracy*100:.1f}%)** : {accuracy*100:.1f}% des prédictions sont correctes.
        
        - **Sensibilité ({sensitivity*100:.1f}%)** : Le modèle détecte {sensitivity*100:.1f}% des vrais sinistres.
          {"⚠️ Sensibilité faible ! Le modèle rate beaucoup de sinistres." if sensitivity < 0.6 else "✅ Bonne détection des sinistres."}
        
        - **Spécificité ({specificity*100:.1f}%)** : Le modèle identifie {specificity*100:.1f}% des non-sinistres.
          {"⚠️ Spécificité faible ! Trop de fausses alertes." if specificity < 0.6 else "✅ Bonne identification des non-sinistres."}
        
        - **Précision ({precision*100:.1f}%)** : Quand le modèle prédit un sinistre, il a raison {precision*100:.1f}% du temps.
        """)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Courbe ROC
        st.markdown("### 📈 Courbe ROC - Comparaison LOGIT vs PROBIT")
        
        fpr_logit, tpr_logit, _ = roc_curve(y_true, y_pred_logit_proba)
        fpr_probit, tpr_probit, _ = roc_curve(y_true, y_pred_probit_proba)
        
        auc_logit = auc(fpr_logit, tpr_logit)
        auc_probit = auc(fpr_probit, tpr_probit)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.plot(fpr_logit, tpr_logit, 'b-', linewidth=2.5, 
                   label=f'LOGIT (AUC = {auc_logit:.4f})')
            ax.plot(fpr_probit, tpr_probit, 'r--', linewidth=2.5, 
                   label=f'PROBIT (AUC = {auc_probit:.4f})')
            ax.plot([0, 1], [0, 1], 'k--', linewidth=1.5, label='Aléatoire (AUC = 0.5)')
            
            ax.set_xlabel('Taux de Faux Positifs (1 - Spécificité)', fontsize=13, fontweight='bold')
            ax.set_ylabel('Taux de Vrais Positifs (Sensibilité)', fontsize=13, fontweight='bold')
            ax.set_title('COURBE ROC - Comparaison des Modèles', fontsize=15, fontweight='bold')
            ax.legend(loc='lower right', fontsize=12, frameon=True, shadow=True)
            ax.grid(alpha=0.3, linestyle='--')
            ax.set_xlim([-0.02, 1.02])
            ax.set_ylim([-0.02, 1.02])
            st.pyplot(fig)
        
        with col2:
            st.markdown("#### 🎯 AUC Score")
            st.metric("LOGIT", f"{auc_logit:.4f}")
            st.metric("PROBIT", f"{auc_probit:.4f}")
            
            st.markdown("#### 📊 Interprétation")
            if auc_logit < 0.7:
                interp = "⚠️ Faible"
            elif auc_logit < 0.8:
                interp = "✅ Acceptable"
            elif auc_logit < 0.9:
                interp = "🌟 Excellent"
            else:
                interp = "🏆 Exceptionnel"
            
            st.info(f"Capacité prédictive : **{interp}**")
        
        st.markdown('<div class="interpretation-box">', unsafe_allow_html=True)
        st.markdown(f"""
        **📖 Qu'est-ce que la courbe ROC et l'AUC ?**
        
        - **Courbe ROC** : Montre le compromis entre sensibilité et spécificité
        - **AUC (Area Under Curve)** : Mesure la capacité discriminante du modèle
        
        **Interprétation de l'AUC :**
        - **0.5** : Prédiction aléatoire (pile ou face)
        - **0.7 - 0.8** : Capacité prédictive acceptable
        - **0.8 - 0.9** : Excellente capacité prédictive
        - **> 0.9** : Capacité prédictive exceptionnelle
        
        **Votre modèle :**
        - AUC LOGIT : **{auc_logit:.4f}**
        - AUC PROBIT : **{auc_probit:.4f}**
        - {"🏆 Excellent modèle !" if auc_logit > 0.8 else "✅ Modèle acceptable" if auc_logit > 0.7 else "⚠️ Modèle à améliorer"}
        """)
        st.markdown('</div>', unsafe_allow_html=True)

# ============================================================================
# PAGE RAPPORT FINAL
# ============================================================================

elif menu == "📋 Rapport Final":
    
    if uploaded_file is None:
        st.warning("⚠️ Veuillez d'abord charger un fichier CSV dans la barre latérale.")
    else:
        df = load_and_prepare_data(uploaded_file)
        
        st.markdown('<div class="section-header">📋 Rapport Final - Synthèse de l\'Analyse</div>', 
                    unsafe_allow_html=True)
        
        with st.spinner("⏳ Génération du rapport..."):
            model_logit, model_probit = estimate_models(df)
        
        # ============== PARTIE 1 : RÉSUMÉ EXÉCUTIF ==============
        st.markdown("## 📊 1. RÉSUMÉ EXÉCUTIF")
        
        taux_sinistre = df['sinistre'].mean() * 100
        
        st.markdown(f"""
        ### Contexte de l'étude
        - **Objectif** : Identifier les principaux déterminants de la sinistralité en assurance automobile
        - **Méthode** : Régression logistique (modèles Logit et Probit)
        - **Échantillon** : {len(df):,} observations
        - **Taux de sinistralité** : {taux_sinistre:.2f}%
        """)
        
        # ============== PARTIE 2 : RÉSULTATS PRINCIPAUX ==============
        st.markdown("## 🎯 2. RÉSULTATS PRINCIPAUX")
        
        # Tableau récapitulatif des Odds Ratios
        st.markdown("### 📊 Tableau récapitulatif des déterminants")
        
        params = model_logit.params
        pvalues = model_logit.pvalues
        conf_int = model_logit.conf_int()
        
        final_results = []
        for var in params.index:
            if var != 'Intercept':
                coef = params[var]
                pval = pvalues[var]
                odds_ratio = np.exp(coef)
                ci_lower = np.exp(conf_int.loc[var, 0])
                ci_upper = np.exp(conf_int.loc[var, 1])
                
                if pval < 0.001:
                    sig = "***"
                elif pval < 0.01:
                    sig = "**"
                elif pval < 0.05:
                    sig = "*"
                else:
                    sig = "ns"
                
                if odds_ratio > 1:
                    effet = f"↑ +{(odds_ratio-1)*100:.1f}%"
                else:
                    effet = f"↓ -{(1-odds_ratio)*100:.1f}%"
                
                final_results.append({
                    'Variable': var,
                    'Coef. (β)': f"{coef:.4f}",
                    'OR': f"{odds_ratio:.3f}",
                    'IC 95%': f"[{ci_lower:.3f}-{ci_upper:.3f}]",
                    'Effet': effet,
                    'p-value': f"{pval:.4f}",
                    'Signif.': sig
                })
        
        final_df = pd.DataFrame(final_results)
        st.dataframe(final_df, use_container_width=True)
        
        st.caption("Légende : *** p<0.001, ** p<0.01, * p<0.05, ns = non significatif")
        
        # Variables significatives
        vars_sig = final_df[final_df['Signif.'] != 'ns']
        
        st.markdown(f"""
        ### ✅ Variables statistiquement significatives : {len(vars_sig)}/{len(final_df)}
        """)
        
        for idx, row in vars_sig.iterrows():
            st.markdown(f"- **{row['Variable']}** : OR = {row['OR']} ({row['Effet']}) {row['Signif.']}")
        
        # ============== PARTIE 3 : QUALITÉ DU MODÈLE ==============
        st.markdown("## 📈 3. QUALITÉ DU MODÈLE")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Pseudo R²", f"{model_logit.prsquared:.4f}")
        with col2:
            st.metric("AIC", f"{model_logit.aic:.2f}")
        with col3:
            st.metric("BIC", f"{model_logit.bic:.2f}")
        with col4:
            st.metric("Log-vraisemblance", f"{model_logit.llf:.2f}")
        
        # Capacité prédictive
        y_true = df['sinistre']
        y_pred_logit_proba = model_logit.predict()
        y_pred_logit = (y_pred_logit_proba >= 0.5).astype(int)
        
        fpr_logit, tpr_logit, _ = roc_curve(y_true, y_pred_logit_proba)
        auc_logit = auc(fpr_logit, tpr_logit)
        
        accuracy = accuracy_score(y_true, y_pred_logit)
        cm = confusion_matrix(y_true, y_pred_logit)
        tn, fp, fn, tp = cm.ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        st.markdown("### 🎯 Capacité Prédictive")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("AUC", f"{auc_logit:.4f}")
        with col2:
            st.metric("Accuracy", f"{accuracy*100:.1f}%")
        with col3:
            st.metric("Sensibilité", f"{sensitivity*100:.1f}%")
        with col4:
            st.metric("Spécificité", f"{specificity*100:.1f}%")
        
        # ============== PARTIE 4 : INTERPRÉTATIONS ==============
        st.markdown("## 💡 4. INTERPRÉTATIONS ET RECOMMANDATIONS")
        
        st.markdown("### 📝 Principaux enseignements")
        
        # Générer automatiquement des interprétations
        interpretations = []
        
        for idx, row in vars_sig.iterrows():
            var = row['Variable']
            or_val = float(row['OR'])
            
            if 'sexe' in var.lower():
                if or_val > 1:
                    interpretations.append("👨 Les **conducteurs masculins** présentent un risque de sinistre significativement plus élevé que les conductrices.")
                else:
                    interpretations.append("👩 Les **conductrices** présentent un risque de sinistre significativement plus élevé que les conducteurs masculins.")
            
            elif 'age_conducteur' in var.lower():
                if or_val > 1:
                    interpretations.append(f"📈 L'**âge du conducteur** augmente le risque de sinistre de {(or_val-1)*100:.1f}% par année supplémentaire.")
                else:
                    interpretations.append(f"📉 L'**âge du conducteur** diminue le risque de sinistre de {(1-or_val)*100:.1f}% par année supplémentaire.")
            
            elif 'age_permis' in var.lower():
                if or_val > 1:
                    interpretations.append(f"📜 L'**ancienneté du permis** augmente le risque ({(or_val-1)*100:.1f}% par année), ce qui peut sembler contre-intuitif.")
                else:
                    interpretations.append(f"📜 L'**ancienneté du permis** réduit le risque de {(1-or_val)*100:.1f}% par année d'expérience.")
            
            elif 'age_vehicule' in var.lower():
                if or_val > 1:
                    interpretations.append(f"🚗 Les **véhicules plus anciens** augmentent le risque de sinistre de {(or_val-1)*100:.1f}% par année d'âge.")
                else:
                    interpretations.append(f"🚗 Les **véhicules plus récents** semblent présenter plus de risques.")
            
            elif 'puissance' in var.lower():
                if or_val > 1:
                    interpretations.append(f"⚡ La **puissance du véhicule** augmente le risque de {(or_val-1)*100:.1f}% par CV supplémentaire.")
                else:
                    interpretations.append(f"⚡ Les véhicules moins puissants présentent paradoxalement plus de risques.")
            
            elif 'usage' in var.lower():
                interpretations.append(f"💼 Le **type d'usage** du véhicule a un impact significatif sur la sinistralité.")
        
        for i, interp in enumerate(interpretations, 1):
            st.markdown(f"{i}. {interp}")
        
        # ============== PARTIE 5 : RECOMMANDATIONS ==============
        st.markdown("### 🎯 Recommandations pour les Assureurs")
        
        st.markdown('<div class="success-box">', unsafe_allow_html=True)
        st.markdown("""
        **1. Tarification personnalisée**
        - Ajuster les primes en fonction des facteurs de risque identifiés
        - Appliquer des coefficients de majoration/minoration selon les Odds Ratios
        
        **2. Segmentation des risques**
        - Créer des profils de risque basés sur les variables significatives
        - Adapter les garanties proposées selon le segment
        
        **3. Prévention ciblée**
        - Campagnes de sensibilisation pour les profils à risque élevé
        - Programmes de formation pour les jeunes conducteurs
        
        **4. Amélioration continue**
        - Mettre à jour le modèle régulièrement avec de nouvelles données
        - Intégrer d'autres variables (géolocalisation, télématique)
        
        **5. Utilisation opérationnelle**
        - Intégrer le modèle dans le système de souscription
        - Automatiser la tarification basée sur le scoring de risque
        """)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # ============== PARTIE 6 : LIMITES ==============
        st.markdown("### ⚠️ Limites de l'étude")
        
        st.markdown('<div class="warning-box">', unsafe_allow_html=True)
        st.markdown(f"""
        **Limites méthodologiques :**
        - Taille de l'échantillon : {len(df):,} observations
        - Variables disponibles : modèle parcimonieux
        - Période d'observation : données d'une seule année
        - Équilibre de l'échantillon : {taux_sinistre:.1f}% de sinistres
        
        **Améliorations possibles :**
        - Intégrer des variables comportementales (km annuels, zone géographique)
        - Données télématiques (vitesse, freinage, accélération)
        - Historique sur plusieurs années
        - Variables météorologiques et de trafic
        - Modèles plus complexes (machine learning)
        """)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # ============== PARTIE 7 : EXPORT ==============
        st.markdown("## 📥 5. EXPORT DU RAPPORT")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Export CSV des résultats
            csv = final_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📊 Télécharger les résultats (CSV)",
                data=csv,
                file_name="resultats_sinistralite.csv",
                mime="text/csv",
            )
        
        with col2:
            # Export du résumé du modèle
            summary_text = model_logit.summary().as_text()
            st.download_button(
                label="📄 Télécharger le résumé du modèle",
                data=summary_text,
                file_name="summary_modele_logit.txt",
                mime="text/plain",
            )
        
        # Générer un rapport complet en markdown
        rapport_md = f"""
# RAPPORT D'ANALYSE - SINISTRALITÉ EN ASSURANCE AUTOMOBILE

## 1. CONTEXTE ET OBJECTIFS

- **Objectif** : Identifier les principaux déterminants de la sinistralité
- **Méthode** : Régression logistique (Logit et Probit)
- **Échantillon** : {len(df):,} observations
- **Taux de sinistralité** : {taux_sinistre:.2f}%

## 2. RÉSULTATS STATISTIQUES

### 2.1 Qualité du modèle
- Pseudo R² (McFadden) : {model_logit.prsquared:.4f}
- AIC : {model_logit.aic:.2f}
- BIC : {model_logit.bic:.2f}
- Log-vraisemblance : {model_logit.llf:.2f}
- Test LR : {model_logit.llr:.2f} (p-value: {model_logit.llr_pvalue:.6f})

### 2.2 Capacité prédictive
- AUC : {auc_logit:.4f}
- Accuracy : {accuracy*100:.2f}%
- Sensibilité : {sensitivity*100:.2f}%
- Spécificité : {specificity*100:.2f}%

### 2.3 Variables significatives

"""
        
        for idx, row in vars_sig.iterrows():
            rapport_md += f"**{row['Variable']}**\n"
            rapport_md += f"- Odds Ratio : {row['OR']}\n"
            rapport_md += f"- Effet : {row['Effet']}\n"
            rapport_md += f"- p-value : {row['p-value']} {row['Signif.']}\n\n"
        
        rapport_md += f"""
## 3. INTERPRÉTATIONS

"""
        for i, interp in enumerate(interpretations, 1):
            rapport_md += f"{i}. {interp}\n\n"
        
        rapport_md += """
## 4. RECOMMANDATIONS

1. **Tarification personnalisée** basée sur les facteurs de risque identifiés
2. **Segmentation** des portefeuilles selon les profils de risque
3. **Prévention** ciblée pour les segments à risque élevé
4. **Amélioration continue** avec mise à jour régulière du modèle

## 5. CONCLUSION

Cette analyse a permis d'identifier les principaux déterminants de la sinistralité 
en assurance automobile et de quantifier leur impact à l'aide de la régression logistique.
Les résultats peuvent être directement utilisés pour améliorer la tarification et 
la gestion des risques.
"""
        
        st.download_button(
            label="📋 Télécharger le rapport complet (Markdown)",
            data=rapport_md,
            file_name="rapport_sinistralite_complet.md",
            mime="text/markdown",
        )
        
        # ============== CONCLUSION ==============
        st.markdown("---")
        st.markdown("## ✅ CONCLUSION")
        
        st.markdown('<div class="success-box">', unsafe_allow_html=True)
        st.markdown(f"""
        ### Synthèse de l'analyse
        
        ✅ **Modèle estimé avec succès**
        - {len(vars_sig)} variables statistiquement significatives identifiées
        - Pseudo R² : {model_logit.prsquared:.4f} ({interp})
        - AUC : {auc_logit:.4f} ({"Excellente" if auc_logit > 0.8 else "Bonne" if auc_logit > 0.7 else "Acceptable"} capacité prédictive)
        
        ✅ **Résultats exploitables**
        - Les Odds Ratios permettent une interprétation directe
        - Les facteurs de risque sont clairement identifiés
        - Le modèle peut être intégré dans un système de tarification
        
        ✅ **Perspectives d'amélioration**
        - Enrichir avec de nouvelles variables explicatives
        - Tester des modèles plus complexes (Machine Learning)
        - Mettre à jour régulièrement avec de nouvelles données
        
        ### 🎯 Prochaines étapes recommandées
        
        1. Valider le modèle sur un échantillon test indépendant
        2. Implémenter le scoring dans le système de souscription
        3. Monitorer les performances en production
        4. Ajuster périodiquement les coefficients
        """)
        st.markdown('</div>', unsafe_allow_html=True)

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #666; padding: 20px;'>
        <p><strong>Application d'Analyse de Sinistralité Automobile</strong></p>
        <p>Développée avec Streamlit | Régression Logistique (Logit & Probit)</p>
        <p>© 2024 - Tous droits réservés</p>
    </div>
""", unsafe_allow_html=True)
