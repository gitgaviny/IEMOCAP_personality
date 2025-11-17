# IEMOCAP Personality Labels (Big Five Personality, OCEAN)

This repository provides **personality annotations** for the IEMOCAP dataset, including all preprocessing scripts and consistency/correlation analysis.  
All annotations were collected from **six independent raters** based on the **Ten-Item Personality Inventory (TIPI)**.

---

## Repository Structure

```
dataset/
    tipi_labels                 # Raw TIPI annotations from 6 raters
    cluster_ocean               # Convert TIPI → OCEAN (speaker-level)
fleisskappa/
    filtering                   # Remove farthest-from-mean rater
    binarize_5raters            # Convert continuous OCEAN → binary labels
    cal                         # Compute Fleiss’ kappa
icc/
    cal                         # Compute ICC (intra-class correlation)
correlation_analysis/
    ...                         # Big Five & valence/arousal correlation
```

---

## Data Description

### **1. TIPI Labels (dataset/tipi_labels)**
Contains the **raw TIPI annotations** from six raters:
- `dialogue_id`, `gender`, `tipi` (1–10)
- `rater1`–`rater6`

---

## Processing Pipeline

### **Step 1 — Convert TIPI → OCEAN**

```
dataset/cluster_ocean.py
```

Converts 10 TIPI items into the five OCEAN personality traits.

---

### **Step 2 — Filter Raters**


```
fleisskappa/filtering.py
```

Removes the rater **farthest from the mean** (per TIPI item) and produces 5-rater filtered labels.

---

### **Step 3 — Binarize Personality Traits**


```
fleisskappa/binarize_5raters.py
```

Binarizes OCEAN traits using each trait’s **median**  

---

### **Step 4 — Fleiss’ Kappa**


```
fleisskappa/cal.py
```

Computes **Fleiss’ kappa** for inter-rater agreement.

---

### **Step 5 — ICC (Intra-Class Correlation)**


```
icc/cal/
```

Computes **ICC** (e.g., ICC(2,k)) for rater reliability.

---

## 📊 Correlation Analysis

Scripts in:

```
correlation_analysis/
```

Compute PCC between **Big Five traits** and **arousal/valence**, with optional gender-specific analysis.



These resources allow rigorous study of personality and affective behavior using the IEMOCAP dataset.
