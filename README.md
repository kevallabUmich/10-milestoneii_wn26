# Milestone 2 Project: Fraud Detection & Health Outcome Analysis

## Introduction & Overview

This project combines supervised learning to classify CFPB consumer complaints as fraud versus legitimate cases, and unsupervised learning to discover patterns in county-level mortality data linked to socioeconomic factors. For Part A, I aim to build models with 85%+ accuracy to help investigators prioritize fraud cases. For Part B, I'll use clustering to identify county groups with similar overdose/suicide mortality trajectories and demographic characteristics.

The main challenges include: (1) class imbalance in the CFPB data despite 40% fraud prevalence, requiring SMOTE and careful metric selection (prioritizing recall over accuracy), (2) messy unstructured complaint text needing NLP preprocessing, (3) model interpretability requirements for regulatory compliance, (4) CDC WONDER data suppression creating ~15-20% missing values in rural counties, (5) high-dimensional ACS census data requiring dimensionality reduction, and (6) spatial autocorrelation between neighboring counties violating clustering independence assumptions.

## Related Work

The most similar project is "Automated Complaint Classification using Machine Learning" by Adarsh Singh on Kaggle (https://www.kaggle.com/code/adarshsng/consumer-complaint-classification-nlp), which uses the same CFPB dataset to classify complaints into 18 product categories (credit card, mortgage, etc.) with 70% accuracy using Random Forest and TF-IDF.

My project differs by: (1) predicting fraud vs non-fraud (binary classification with regulatory implications) rather than product categories, (2) using advanced methods like XGBoost, MiniLM (compact sentence transformer for CPU), and SMOTE for imbalance handling, (3) including a separate unsupervised component with CDC mortality and ACS demographic data for public health insights, (4) optimizing for recall/F2-score rather than accuracy since missing fraud is costlier than false alarms, and (5) implementing LIME/SHAP for model explainability.

---

## Part A: Supervised Learning - Financial Fraud Detection

**Dataset**: CFPB Consumer Complaints Database (https://www.consumerfinance.gov/data-research/consumer-complaints/) with 165,881 complaints from 2020-2025, 40% fraud-labeled (66,077 fraud cases). Features include complaint narrative text, product type, company name, state, submission date, and response time.

**Learning Approaches & Features**: I'll start with Logistic Regression + TF-IDF (5000 terms) as an interpretable baseline, then try ensemble methods (Random Forest with 100 trees, XGBoost), and use MiniLM (all-MiniLM-L6-v2 from Hugging Face) which is optimized for CPU inference. Text features will include TF-IDF vectors, Word2Vec embeddings (300-dim), and MiniLM embeddings (384-dim). Additional features: response time, complaint length, historical state fraud rate, and one-hot encoded product type. I'll use SMOTE for oversampling, class weights, and threshold tuning to handle imbalance. This combination balances interpretability (TF-IDF), semantic understanding (MiniLM - 3x faster than DistilBERT, only 22M parameters), and interaction detection (tree methods).

**External Datasets/Tools**: FTC fraud reports for state-level and product-specific fraud statistics to create additional features. Pre-trained MiniLM from Hugging Face (sentence-transformers library, specifically optimized for CPU and semantic embeddings). LIME/SHAP libraries for model explainability. Work required: downloading/scraping FTC data, merging with CFPB data by state/product, generating MiniLM embeddings for complaint text, and implementing SHAP wrappers for Random Forest and XGBoost.

**Evaluation & Visualization**: Primary metrics are F1-score and F2-score (weights recall 2x precision) since missing fraud is costlier than false alarms. Secondary metrics include precision, recall, accuracy, ROC-AUC, and PR-AUC. Visualizations: feature importance charts (TF-IDF terms, MiniLM embedding clusters), ROC and PR curves comparing models, confusion matrices, SHAP waterfall plots for individual predictions, word clouds comparing false negatives vs positives, and temporal performance analysis (2020-2025).

---

## Part B: Unsupervised Learning - Health Outcome Patterns

**Dataset**: CDC WONDER mortality data (https://wonder.cdc.gov/) with county-level death rates (2015-2023) for drug overdose, suicide, and alcohol-related deaths, and American Community Survey 5-year estimates (https://data.census.gov/) with county-level demographics including income, poverty, education, unemployment, age, population density, and insurance coverage. Datasets will be merged using FIPS county codes.

**Research Questions**: (1) Do counties cluster into distinct mortality trajectory patterns (e.g., stable, epidemic peak, accelerating)? (2) What socioeconomic factors differentiate high-risk from low-risk clusters? (3) Do clusters show spatial coherence (e.g., Appalachian opioid belt)? (4) Which demographic variables most strongly predict cluster membership?

**Data Preparation**: Calculate age-adjusted mortality rates per 100k, handle CDC suppression (<10 deaths) via KNN imputation or county exclusion, compute year-over-year rate changes and 3-year trends, create composite indices (economic stress, health access, rurality), merge ACS with CDC via FIPS codes aligning temporal periods, and apply PCA to reduce 100+ ACS variables to 10-15 components explaining 80%+ variance.

**Unsupervised Approaches**: (1) K-Means with Dynamic Time Warping distance for mortality trajectories using optimal k from elbow method and silhouette scores, (2) K-Means and hierarchical clustering (Ward linkage) on combined mortality + PCA-reduced demographics, (3) DBSCAN for outlier detection, and (4) t-SNE/UMAP for 2D visualization. Features include mortality rates, rate changes, trend slopes, and PCA components from ACS demographics.

**External Resources**: Census Bureau FIPS codes and county shapefiles for geographic mapping, libraries including scikit-learn (clustering, PCA), tslearn (DTW), geopandas (spatial), and folium/plotly (interactive maps). Post-hoc explainability via training Random Forest to predict cluster labels and extracting feature importance/SHAP values.

**Evaluation Methods**: Internal validation using silhouette score (>0.3 target), Davies-Bouldin index, Calinski-Harabasz index, elbow method, and bootstrap stability (1000 iterations). External validation via geographic coherence with known epidemic regions and correlation with CDC opioid prescription rates. Spatial analysis using Moran's I for autocorrelation testing.

**Visualizations**: Elbow and silhouette plots for cluster quality, interactive choropleth maps colored by cluster assignment with county-level tooltips, t-SNE/UMAP scatterplots with points colored by cluster and sized by population, trajectory line plots showing average mortality by cluster (2015-2023), feature distribution heatmaps comparing cluster characteristics, hierarchical clustering dendrograms, box plots comparing key variables across clusters, PCA biplots with feature vectors, and feature importance bar charts from post-hoc Random Forest.

---
