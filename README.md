# Milestone 2 Project: Fraud Detection & Health Outcome Analysis

## Introduction & Overview

This project combines supervised learning to classify CFPB consumer complaints as fraud versus legitimate cases, and unsupervised learning to discover patterns in county-level mortality data linked to socioeconomic factors. For Part A, I aim to build models with 85%+ accuracy to help investigators prioritize fraud cases. For Part B, I'll use clustering to identify county groups with similar overdose mortality trajectories and demographic characteristics.

---

## Part A: Supervised Learning - Financial Fraud Detection

**Dataset**: CFPB Consumer Complaints Database (https://www.consumerfinance.gov/data-research/consumer-complaints/) with 165,881 complaints from 2020-2025, 40% fraud-labeled (66,077 fraud cases). Features include complaint narrative text, product type, company name, state, submission date, and response time.

**Learning Approaches & Features**: I'll start with Logistic Regression + TF-IDF (5000 terms) as an interpretable baseline, then try ensemble methods (Random Forest with 100 trees, XGBoost), and use MiniLM (all-MiniLM-L6-v2 from Hugging Face) which is optimized for CPU inference. Text features will include TF-IDF vectors, Word2Vec embeddings (300-dim), and MiniLM embeddings (384-dim). Additional features: response time, complaint length, historical state fraud rate, and one-hot encoded product type. I'll use SMOTE for oversampling, class weights, and threshold tuning to handle imbalance. This combination balances interpretability (TF-IDF), semantic understanding (MiniLM - 3x faster than DistilBERT, only 22M parameters), and interaction detection (tree methods).

**Evaluation & Visualization**: Primary metrics are F1-score and F2-score (weights recall 2x precision) since missing fraud is costlier than false alarms. Secondary metrics include precision, recall, accuracy, ROC-AUC, and PR-AUC. Visualizations: feature importance charts (TF-IDF terms), ROC and PR curves comparing models, confusion matrices, and temporal performance analysis (2020-2025).

---

## Part B: Unsupervised Learning - Health Outcome Patterns

**Dataset**: CDC WONDER mortality data (https://wonder.cdc.gov/) with county-level death rates (2018-2023) for drug overdose, suicide, and alcohol-related deaths, and American Community Survey 5-year estimates (https://data.census.gov/) with county-level demographics including income, poverty, education, unemployment, age, population density, and insurance coverage. Datasets will be merged using FIPS county codes and year.

**Research Questions**: (1) Do counties cluster into distinct mortality  patterns? (2) What socioeconomic factors differentiate clusters? (3) Do clusters show spatial coherence (e.g., regionalization)? (4) Which demographic variables most strongly predict cluster membership?

**Data Preparation**: Calculate crude mortality rates per 100k, handle CDC suppression (<20 deaths) via KNN imputation or county exclusion, compute year-over-year rate changes and 3-year trends, create composite indices (economic stress, health access, rurality), merge ACS with CDC via FIPS codes aligning temporal periods, and apply PCA to reduce variables to components explaining 80%+ variance.

**Unsupervised Approaches**: (1) K-Means for mortality trajectories using optimal k from elbow method and silhouette scores, (2) K-Means and hierarchical clustering (Ward linkage) on combined mortality + PCA-reduced demographics, (3) DBSCAN for outlier detection.

**External Resources**: Census Bureau FIPS codes and county shapefiles for geographic mapping, libraries including scikit-learn (clustering, PCA), geopandas (spatial), and folium/plotly (interactive maps). 

**Evaluation Methods**: Internal validation using silhouette score, Davies-Bouldin index, and elbow method to determine optimum K-Value. Ablation testing on variables of the clusters. 

**Visualizations**: Elbow and silhouette plots for cluster quality, interactive geographic maps colored by cluster assignment with county-level tooltips, feature distribution heatmaps comparing cluster characteristics, hierarchical clustering dendrograms, box plots comparing key variables across clusters, and PCA biplots with feature vectors.

---
