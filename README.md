# gene-knockout-pred
Predicts transcriptional consequences of gene knockout in spatial transcriptomics micro environment. Identifies target cells, simulates knockout, and analyzes downstream effects in neighboring cells.
# Spatial Transcriptomics Knockout Predictor (SOX2)

Predicts transcriptional consequences of **SOX2 knockout** in spatially resolved tissues using **Celcomen** and **Simcomen** models.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 📌 About

This tool performs **spatially-aware in silico knockout** of the transcription factor **SOX2** in 10x Visium spatial transcriptomics data. It identifies cells with high SOX2 expression, simulates SOX2 knockout using the **Celcomen** and **Simcomen** frameworks, and predicts downstream transcriptional changes in neighbouring cells.

**Key capabilities:**
- Automatic selection of target cells (top 5% SOX2 expression with sufficient neighbours).
- Patch extraction around each target cell (spatial neighbourhood).
- Celcomen training to model gene–gene interactions.
- Simcomen inference to simulate knockout and predict expression changes.
- Differential expression analysis (up/downregulated genes) in neighbouring cells.
- GSEA (Enrichr) for functional interpretation of regulated genes.
- Spatial visualisation of predicted knockout effects.

---

## 🧬 Methodological Framework

The analysis follows a three‑step process:

1. **Celcomen training** – learns a latent representation of gene–gene interactions from the spatial expression data of a local patch.
2. **SOX2 knockout** – sets SOX2 expression to zero in the target cell.
3. **Simcomen inference** – predicts the perturbed expression profile of the target cell and its neighbours, quantifying the propagated effect of the knockout.

The models are built on the **Celcomen** and **Simcomen** frameworks (see [celcomen](https://github.com/your-org/celcomen) for details).

---

## 🚀 Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/kzhezhel/sox2-knockout-predictor.git
cd sox2-knockout-predictor
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
