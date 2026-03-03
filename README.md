# Filippo Teodorani Master's Thesis

This repository contains the full analytical pipeline developed for the Master's thesis:

**“Epidemiological and Economic Impact Assessment of RSV Vaccination in Adults Aged 65+ in Lombardy”**

The project reconstructs latent RSV incidence using heterogeneous surveillance systems and evaluates counterfactual vaccination strategies through stochastic epidemic simulation and economic modeling.

## Project Overview

The analytical framework is structured into three sequential components:

1. **RSV Incidence Reconstruction (Step 1)**  
   Reconstruction of weekly RSV incidence among adults aged 65+ in Lombardy using:
   - Italian ARI surveillance data  
   - U.S. RSV laboratory circulation signals  
   - Machine learning (Poisson Gradient Boosting)

2. **Vaccination Scenario Generation (Step 2)**  
   Simulation of alternative RSV vaccination adoption pathways using:
   - Historical influenza vaccination behavior  
   - Logistic diffusion modeling  
   - Weekly within-season allocation template  
   - Two-week immunological protection delay  

3. **Counterfactual Epidemiological & Economic Evaluation (Step 3)**  
   Simulation of future epidemic seasons and estimation of:
   - Hospitalizations avoided  
   - Deaths avoided  
   - Healthcare costs saved  
   - Net budget impact  
   - Return on Investment (ROI)  

## Repository Structure

### data/
- raw/  
  Original raw datasets  

- interim/  
  Cleaned and harmonized intermediate datasets  

- processed/  
  Final analytical datasets used in modeling  


### src/
- data_preparation/  
  Python scripts transforming raw → interim  

- dataset_creation/  
  Python scripts transforming interim → processed  


### notebook/
Jupyter notebooks for analytical steps:

- Step 1 – RSV reconstruction  
- Step 2 – Vaccination scenario generation  
- Step 3 – Epidemiological & economic simulation  


### outputs/
- step1/  
- step2/  
- step3/  

Generated figures, tables, and model outputs.

## Data Pipeline

The full data flow follows a structured pipeline:

Raw data
    |
    v
Data preparation scripts (raw → interim)
    |
    v
Dataset creation scripts (interim → processed)
    |
    v
Analytical notebooks (modeling)
    |
    v
Outputs (tables, figures, final results)

## Analytical Structure

All processed datasets are harmonized to a unified panel structure:

**Influenza Season × Epidemiological Week × Geographic Unit × Age Class**

## Modeling Components

### 1. RSV Reconstruction
- ARI-based scaling
- RSV laboratory proxy integration
- Feature engineering (lags, seasonality, meteorology)
- Histogram-based Gradient Boosting (Poisson loss)
- Age redistribution and epidemiological calibration

### 2. Vaccination Modeling
- Logistic adoption function
- Scenario-based plateau assumptions (low / mid / high)
- Weekly campaign allocation template
- Effective coverage with 2-week delay

### 3. Impact & Economic Evaluation
- Stochastic seasonal incidence simulation
- Fixed hospitalization risk mapping
- Case fatality application
- Cost-of-illness framework
- Net cost and ROI computation

## Outputs

The repository generates:

- Weekly RSV incidence projections
- Seasonal hospitalization and mortality estimates
- Five-year cumulative impact tables
- Economic performance indicators
- Publication-ready figures

All results are stored in the `outputs/` directory.

## Reproducibility

To reproduce the analysis:

1. Execute scripts in `src/data_preparation/`
2. Execute scripts in `src/dataset_creation/`
3. Run notebooks in order:
   - Step 1
   - Step 2
   - Step 3

## Author

Master’s Thesis — Filippo Teodorani
Università di Bologna - Alma Mater Studiorum 
Supervisor: Dott. Claudio Sartori



