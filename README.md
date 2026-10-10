# DiabetesRisk-AI — Système Intelligent de Prédiction du Risque de Diabète

## 1. Présentation du projet

**DiabetesRisk-AI** est un système intelligent de Machine Learning conçu pour analyser des données cliniques, identifier des profils de patients présentant des caractéristiques similaires et prédire leur groupe à partir de nouvelles observations.

Le projet repose sur deux approches complémentaires :

- **Clustering non supervisé :** utilisation de K-Means pour segmenter les patients en groupes homogènes.
- **Classification supervisée :** entraînement et comparaison de plusieurs algorithmes pour prédire le groupe d'un nouveau patient.

Le projet intègre également des outils MLOps pour assurer le suivi des expériences, l'automatisation de l'entraînement et le déploiement du modèle.

### Objectifs

- Préparer et analyser les données cliniques.
- Identifier les profils de patients grâce au clustering.
- Comparer plusieurs modèles de classification.
- Optimiser et sauvegarder le modèle final dans une pipeline complète.
- Suivre les expériences avec MLflow.
- Exposer les prédictions à travers une API REST avec FastAPI.
- Développer une interface utilisateur avec Streamlit.
- Automatiser le processus d'entraînement avec Apache Airflow.
- Conteneuriser les services avec Docker et Docker Compose.

> **Important :** le niveau de risque estimé est basé sur les profils identifiés dans le dataset et ne constitue pas un diagnostic médical.

---

## 2. Technologies utilisées

| Technologie | Utilisation |
|---|---|
| Python | Langage principal |
| Pandas & NumPy | Manipulation et analyse des données |
| Matplotlib & Seaborn | Visualisation des données |
| Scikit-learn | Prétraitement et Machine Learning |
| K-Means | Clustering non supervisé |
| Logistic Regression, KNN, SVM, Random Forest | Classification supervisée |
| GridSearchCV | Optimisation des hyperparamètres |
| Joblib | Sérialisation des pipelines et modèles |
| MLflow | Tracking des expériences et gestion des modèles |
| FastAPI | Création de l'API de prédiction |
| Streamlit | Interface web interactive |
| Apache Airflow | Orchestration du pipeline ML |
| Docker & Docker Compose | Conteneurisation des services |
| PostgreSQL | Stockage des métadonnées selon la configuration Docker |

---

## 3. Dataset

Le projet utilise le fichier `dataset-diabete.csv`, qui contient des observations cliniques de patients.

### Variables utilisées

| Variable | Description |
|---|---|
| `Pregnancies` | Nombre de grossesses |
| `Glucose` | Taux de glucose |
| `BloodPressure` | Pression artérielle |
| `SkinThickness` | Épaisseur du pli cutané |
| `Insulin` | Taux d'insuline |
| `BMI` | Indice de masse corporelle |
| `DiabetesPedigreeFunction` | Indicateur lié aux antécédents familiaux de diabète |
| `Age` | Âge du patient |

La variable `Cluster` est ajoutée après l'étape de clustering et sert de cible pour la classification supervisée.

### Source des données

Le dataset utilisé est fourni dans les ressources du brief pédagogique.

Fichier source : `data/raw/dataset-diabete.csv`

---

## 4. Architecture du projet

```text
DiabetesRisk-AI/
│
├── data/
│   ├── raw/
│   │   └── dataset-diabete.csv
│   └── processed/
│       ├── cleaned_data_diabetes.csv
│       └── clustered_data_diabetes.csv
│
├── notebooks/
│   └── diabetes_analysis.ipynb
│
├── src/
│   ├── preprocessing/
│   ├── clustering/
│   ├── classification/
│   └── ...
│
├── models/
│   ├── Logistic_Regression_model.pkl
│   ├── KNN_model.pkl
│   ├── Random_Forest_model.pkl
│   └── SVM_model.pkl
│
├── app/
│   ├── api/
│   │   └── main.py
│   └── streamlit/
│       └── app.py
│
├── dags/
│   └── retrain_diabetes_dag.py
│
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

*Cette arborescence est indicative : adaptez les noms et les chemins aux fichiers réellement présents dans votre dépôt.*

---

## 5. Démarche de Machine Learning

### Étape 1 — Analyse exploratoire des données (EDA)

L'analyse exploratoire permet de comprendre la structure du dataset avant l'entraînement des modèles.

Les opérations réalisées comprennent :

- Chargement des données avec Pandas.
- Analyse des dimensions et des types de variables.
- Détection des valeurs manquantes et des doublons.
- Analyse statistique des variables numériques.
- Visualisation des distributions et des relations entre les variables.

### Étape 2 — Prétraitement des données

Le prétraitement vise à rendre les données exploitables par les algorithmes de Machine Learning.

Les opérations comprennent :

- Identification des valeurs incohérentes et des valeurs manquantes.
- Traitement des valeurs nulles ou des zéros représentant des mesures manquantes dans certaines variables cliniques.
- Imputation des valeurs manquantes.
- Détection des valeurs aberrantes à l'aide de méthodes statistiques comme l'IQR.
- Analyse des corrélations et des relations entre les variables.
- Standardisation des variables numériques avec `StandardScaler`.

Les valeurs aberrantes sont étudiées avant toute décision de traitement, afin de préserver autant que possible les informations utiles du dataset.

### Étape 3 — Clustering avec K-Means

K-Means permet de regrouper les patients selon la similarité de leurs caractéristiques cliniques.

Les étapes comprennent :

1. Standardiser les variables numériques.
2. Étudier différentes valeurs de `k`.
3. Utiliser la méthode du coude (*Elbow Method*).
4. Calculer le coefficient de silhouette (*Silhouette Score*).
5. Sélectionner le nombre de clusters approprié.
6. Entraîner le modèle K-Means.
7. Ajouter la colonne `Cluster` au dataset.
8. Visualiser et interpréter les groupes obtenus.

### Étape 4 — Analyse et interprétation des clusters

Les clusters sont analysés à partir de leurs effectifs et des moyennes de leurs caractéristiques cliniques.

Les critères de référence du brief sont notamment :

- `Glucose > 126`
- `BMI > 30`
- `DiabetesPedigreeFunction > 0.5`

Le groupe présentant globalement les caractéristiques les plus élevées selon ces indicateurs peut être interprété comme un profil potentiellement à risque plus élevé.

Une colonne `risk_category` peut ensuite être créée à partir de la correspondance entre le numéro du cluster et son interprétation.

**Attention :** les identifiants des clusters sont arbitraires. Le cluster `0` n'est pas automatiquement à faible risque, et le cluster `1` n'est pas automatiquement à haut risque. Cette correspondance doit être établie à partir des résultats de l'analyse.

### Étape 5 — Classification supervisée

La classification apprend à prédire le groupe attribué par K-Means à partir des caractéristiques d'un patient.

Les modèles étudiés comprennent :

- Logistic Regression
- K-Nearest Neighbors (KNN)
- Support Vector Machine (SVM)
- Random Forest

La démarche comprend :

- Séparation des données en ensembles d'entraînement et de test.
- Construction de pipelines de prétraitement et de classification.
- Entraînement des différents algorithmes sur les mêmes données.
- Optimisation des hyperparamètres avec `GridSearchCV`.
- Comparaison des performances.
- Sélection et sauvegarde de la pipeline finale avec Joblib.

### Métriques d'évaluation

Les modèles sont évalués à l'aide de plusieurs métriques :

- Accuracy
- Precision
- Recall
- F1-score
- Matrice de confusion

Ces métriques permettent d'évaluer la capacité du modèle à prédire les groupes et à limiter les erreurs de classification.

### Étape 6 — Suivi des expériences avec MLflow

MLflow permet de centraliser les informations relatives aux entraînements et de faciliter leur comparaison et leur reproductibilité.

Les informations suivies comprennent :

- Les paramètres des modèles.
- Les hyperparamètres explorés.
- Les métriques d'évaluation.
- Les paramètres et métriques du clustering.
- Les matrices de confusion sous forme d'artefacts.
- Les fichiers des modèles et des pipelines.
- Les informations utiles sur les features et les versions des bibliothèques.

Le registre MLflow peut être utilisé pour gérer les versions du modèle et promouvoir une version validée vers la production, lorsque cette fonctionnalité est configurée.

### Étape 7 — API, interface web et orchestration

**FastAPI**

L'API reçoit les caractéristiques cliniques d'un patient au format JSON, charge la pipeline sauvegardée et retourne le groupe prédit ainsi que les informations de risque calculées par l'application.

**Streamlit**

L'interface web permet à l'utilisateur de saisir les informations cliniques et d'afficher le résultat retourné par l'API.

L'interface peut présenter un indicateur visuel du niveau de risque et des recommandations générales de suivi, sans remplacer une évaluation médicale.

**Apache Airflow**

Le DAG automatise les principales étapes du pipeline, selon sa configuration :

- Chargement des données.
- Prétraitement.
- Clustering.
- Entraînement et évaluation des modèles.
- Sauvegarde des artefacts.
- Suivi des expériences avec MLflow.

**Docker Compose**

Docker Compose orchestre les différents services afin de faciliter leur lancement et leur gestion dans un environnement commun.

---

## 6. Installation et exécution

### Prérequis

- Python 3.13, si compatible avec les dépendances retenues.
- Docker et Docker Compose.
- Git.
- Un environnement disposant des ressources nécessaires à l'exécution des services.

### Étape 1 — Cloner le projet

```bash
git clone <URL_DU_DEPOT_GITHUB>
cd DiabetesRisk-AI
```

Remplacez `<URL_DU_DEPOT_GITHUB>` par l'URL réelle de votre dépôt.

### Étape 2 — Lancer les services avec Docker Compose

Vérifiez que le fichier `docker-compose.yml` est présent à la racine du projet.

Construisez les images et démarrez les services :

```bash
docker compose up --build -d
```

Vérifiez leur état :

```bash
docker compose ps
```

Consultez les journaux en cas de problème :

```bash
docker compose logs -f
```

Pour consulter les journaux d'un service spécifique :

```bash
docker compose logs -f diabete-api
docker compose logs -f diabete-streamlit
docker compose logs -f diabete-mlflow
docker compose logs -f diabete-airflow
```

Ces commandes utilisent les noms de services prévus dans la configuration actuelle du projet. Si les noms changent dans `docker-compose.yml`, adaptez-les en conséquence.

### Étape 3 — Accéder aux applications

Si les ports indiqués sont exposés dans `docker-compose.yml`, les services sont accessibles aux adresses suivantes :

| Service | Adresse |
|---|---|
| Application Streamlit | http://localhost:8501 |
| Documentation FastAPI | http://localhost:8000/docs |
| Interface MLflow | http://localhost:5000 |
| Interface Airflow | http://localhost:8080 |

Ces adresses supposent que les ports correspondants sont publiés et que les services démarrent correctement.

### Étape 4 — Tester l'API

Ouvrez la documentation interactive de FastAPI :

http://localhost:8000/docs

Sélectionnez l'endpoint `POST /predict`, puis cliquez sur **Try it out**.

Exemple de données JSON :

```json
{
  "Pregnancies": 2,
  "Glucose": 140,
  "BloodPressure": 80,
  "SkinThickness": 25,
  "Insulin": 100,
  "BMI": 32.5,
  "DiabetesPedigreeFunction": 0.6,
  "Age": 35
}
```

Cliquez sur **Execute** pour envoyer la requête et examiner la réponse.

Les valeurs ci-dessus sont uniquement des données de test. Le résultat dépend du modèle entraîné et de la logique d'interprétation des clusters.

### Étape 5 — Arrêter les services

Pour arrêter les conteneurs :

```bash
docker compose down
```

Pour arrêter les conteneurs et supprimer également les volumes gérés par Compose :

```bash
docker compose down -v
```

**Attention :** la seconde commande peut supprimer les données persistantes des services, notamment celles stockées dans les volumes PostgreSQL ou MLflow selon votre configuration.

---

## 7. API de prédiction

L'API est développée avec FastAPI et expose un endpoint de prédiction.

### Endpoint principal

| Propriété | Valeur |
|---|---|
| Méthode HTTP | `POST` |
| Endpoint | `/predict` |
| Format d'entrée | JSON |
| Entrée | Caractéristiques cliniques du patient |
| Sortie | Groupe prédit et informations de risque selon l'implémentation |

La pipeline de Machine Learning doit inclure les transformations nécessaires pour que les nouvelles données soient traitées de la même manière que les données d'entraînement.

L'API doit également gérer les erreurs de validation, l'indisponibilité du modèle et les données incorrectes.

---

## 8. Résultats et performances

Les résultats définitifs doivent être renseignés à partir des fichiers de métriques et des expériences réellement enregistrées.

### Comparaison des modèles

| Modèle | Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|
| Logistic Regression | À renseigner | À renseigner | À renseigner | À renseigner |
| KNN | À renseigner | À renseigner | À renseigner | À renseigner |
| SVM | À renseigner | À renseigner | À renseigner | À renseigner |
| Random Forest | À renseigner | À renseigner | À renseigner | À renseigner |

Les métriques finales doivent être calculées sur l'ensemble de test, sans utiliser cet ensemble pour ajuster les hyperparamètres.

### Résultats du clustering

Les informations à documenter sont :

- Nombre de clusters sélectionné.
- Inertie du modèle K-Means.
- Coefficient de silhouette.
- Effectif de chaque cluster.
- Moyennes des variables cliniques par cluster.
- Interprétation des groupes et correspondance avec `risk_category`.

### Modèle retenu

Le modèle final doit être sélectionné en tenant compte des performances de classification, de la généralisation, de la robustesse et des objectifs du projet.

La pipeline complète est sauvegardée dans un fichier sérialisé afin de pouvoir être utilisée par l'API de prédiction.

---

## 9. Captures d'écran

Ajoutez des captures d'écran réelles de l'application et des outils utilisés.

### Application Streamlit

![Application Streamlit](docs/screenshots/streamlit-app.png)

### Documentation FastAPI

![Documentation FastAPI](docs/screenshots/fastapi-docs.png)

### Suivi des expériences MLflow

![Interface MLflow](docs/screenshots/mlflow-tracking.png)

### Orchestration avec Airflow

![Interface Airflow](docs/screenshots/airflow-dag.png)

Les images ci-dessus correspondent à des chemins à créer dans le dépôt. Remplacez-les par vos propres captures d'écran pour qu'elles s'affichent correctement sur GitHub.

---

## 10. Limites du projet

- Les prédictions dépendent de la qualité et de la représentativité du dataset utilisé.
- Les groupes produits par K-Means ne correspondent pas nécessairement à des catégories médicales validées.
- Les performances doivent être vérifiées sur des données inédites.
- Une classification parfaite sur un ensemble de test ne garantit pas une bonne généralisation ; les éventuelles fuites de données doivent être vérifiées.
- Les recommandations affichées ne constituent pas un avis médical.
- La mise en production nécessite une gestion appropriée des erreurs, des versions des modèles, de la sécurité et des données.

---

## 11. Perspectives d'amélioration

- Enrichir le dataset avec des données plus diversifiées.
- Renforcer la validation et la reproductibilité des expériences.
- Ajouter des tests unitaires et des tests d'intégration.
- Mettre en place une validation automatique des performances avant de promouvoir un modèle.
- Améliorer la surveillance des prédictions et la détection de dérive des données.
- Renforcer la sécurité de l'API et la gestion des accès.
- Automatiser les tests et le déploiement avec une pipeline CI/CD.

---

## 12. Auteur et contexte pédagogique

**Projet :** Système Intelligent de Prédiction du Risque de Diabète  
**Nom du dépôt :** DiabetesRisk-AI  
**Formateur / référent :** Hamid OUFAKIR  
**Date de création du brief :** 23/09/2026  
**Période de réalisation prévue :** du 28/09/2026 au 09/10/2026  
**Formation / référentiel :** Certification RNCP Développeur·se en Intelligence Artificielle

**Dépôt GitHub :** À compléter  
**Lien Jira :** À compléter

---

## Conclusion

DiabetesRisk-AI met en œuvre un pipeline complet de Machine Learning, depuis la préparation des données jusqu'à la prédiction via une API et une interface web. L'intégration de MLflow, d'Airflow et de Docker permet d'aborder les principes fondamentaux du MLOps, notamment le suivi des expériences, la reproductibilité, l'orchestration et le déploiement des modèles.