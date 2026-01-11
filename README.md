# REMAP: Parkinson’s Disease Motor Symptom Analysis
> **Objective movement quantification using Computer Vision, 3D Kinematics, and Generative Transformers.**

---

## Project Overview
This project is a core component of the **REMAP (Research on Mobility in Aging and Parkinson’s)** open dataset study . It addresses the limitations of traditional clinical assessments, such as the **MDS-UPDRS scale**, which often suffer from inter-rater variability and a lack of fine-grained temporal resolution.

---

## Research Background
* **Generative Augmentation**: To overcome the scarcity of clinical PD datasets, we employ **Transformers** to generate synthetic skeletal sequences, enhancing the robustness of downstream diagnostic models.
* **Technical Approach**: 
    * **Reconstruction**: VideoPose3D for 3D skeletal lifting.
    * **Generation**: Transformer-based time-series modeling for skeletal joints.

---

## Repository Structure

### 1. Generative Models (Transformer Pipeline)
* **`generate_data_0810_third.ipynb`**: The initial Transformer generative pipeline for temporal skeletal sequences.
* **`Transvae.ipynb`** *(Upgrade)*: The upgraded generative engine.
    * **Integration of VAE (Variational Autoencoder)**: By incorporating VAE, the model learns the underlying probability distribution of human motion. This allows for the generation of synthetic skeletal data that maintains anatomical plausibility while introducing realistic biological variability.
    * **Optimized Attention**: Enhanced temporal dependencies for smoother skeletal transitions.
    * **Clinical Alignment**: Precise mapping between generated sequences and clinical labels (Turn ID, Participant ID).
    * **Performance**: Achieved significant reduction in **Mean Absolute Error (MAE)**.

### 2. Data Exploration & Analysis
* **`DataExploration/`**: Signal processing, Butterworth filtering, and interactive 2D/3D trajectory visualization.
* **`SitToStand/`**: Kinematic analysis of postural transitions, focusing on velocity and stability.

---

##  Data Availability
The dataset used in this research is part of the **REMAP Study**. 
* **Data Source**: The original datasets are publicly available at [[REMAP](https://data.bris.ac.uk/data/dataset/9e748876b7bf30218ef7e4ec4d7f026a)].
* **Usage**: This repository contains code for processing these datasets and generating synthetic samples. Due to GitHub storage limits and data integrity protocols, we recommend downloading the raw data directly from the official source provided above.

---

## Tech Stack
| Category | Tools/Libraries |
| :--- | :--- |
| **Generative AI** | **Transformers**, Time-series Data Augmentation |
| **Pose Estimation** | VideoPose3D, Detectron2 |
| **Data Science** | NumPy, Pandas, SciPy, Scikit-learn |

---

## Important Disclaimers
> [!IMPORTANT]
> **Data Privacy**: This repository only includes de-identified skeletal coordinates and synthetic data. No raw patient videos or HIPAA-protected information are hosted here.

---
© 2025 Yankai Zhao
