# Milestone 2 Project: Fraud Detection & Health Outcome Analysis

## Introduction & Overview

This project combines supervised learning to classify CFPB consumer complaints as fraud versus legitimate cases, and unsupervised learning to discover patterns in county-level mortality data linked to socioeconomic factors. For Part A, I aim to build models  to help investigators prioritize fraud cases. For Part B, I'll use clustering to identify county groups with similar overdose mortality trajectories and demographic characteristics.

---

## Part A: Supervised Learning - Financial Fraud Detection

**Problem**
  Binary classification of consumer complaints as fraud vs. legitimate using the CFPB Consumer Complaints Database.
  
**Dataset**
  Source: CFPB (consumerfinance.gov), 165,605 complaints, 18 columns
  Date range: January 2020 – February 2026
  Class distribution: 6,244 fraud (3.77%), 159,361 non-fraud
  Top fraud products: Debt Collection (2,274), Money Transfer (1,469), Checking/Savings (924)
  
**Features (1,000+ total)**
  Text: TF-IDF (1,000 terms) on complaint narratives
  Categorical (one-hot): Product, company, state, submission channel, company response, timely response
  Numeric: Complaint narrative length, word count, day of week

**Additional Analyses**
  SMOTE: XGB+SMOTE achieves F1=0.544, Recall=75.4% (best overall)
  Ablation: Hybrid (text+structured) F1=0.451 > Structured-only (0.393) > Text-only (0.313)
  HP Sensitivity: max_depth is dominant (~50% F1 gain from 10→30); n_estimators negligible
  5-Fold CV: RF=0.324±0.107, LR=0.255±0.039, XGB=0.059±0.027 (default threshold)
  Failure Analysis: 635/1,127 fraud missed (56.3%), top product=Debt Collection, top state=TX
  Feature Importance: Top-20 features visualized (mix of TF-IDF terms and structured fields)
  
**Visualizations**
  EDA panel, ROC + confusion matrices (3 models), model comparison bar chart, ablation chart, feature importance, HP sensitivity heatmap, failure analysis breakdowns

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
