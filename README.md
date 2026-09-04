# RCA_Explainer

**RAG-based Root Cause Analysis and Explainability for Anomalies in 5G/O-RAN Systems**

RCA_Explainer is a research-oriented framework for performing **Root Cause Analysis (RCA) of anomalies in 5G/O-RAN systems** using **Retrieval-Augmented Generation (RAG)**, domain-specific 3GPP specifications, machine learning models, and knowledge graphs.

The framework combines monitoring data with knowledge extracted from **3GPP and O-RAN technical specifications** to identify potential root causes and generate explanations that are grounded in telecom-domain knowledge.

---

## Overview

Modern 5G and O-RAN networks generate large volumes of heterogeneous telemetry data. Detecting an anomaly is only the first step; understanding **why the anomaly occurred** is critical for effective network management and troubleshooting.

This project aims to address this problem by combining:

* Anomaly information from network monitoring data
* 3GPP/O-RAN technical specifications
* Retrieval-Augmented Generation (RAG)
* Large Language Models (LLMs)
* Knowledge Graphs (KG)
* Machine Learning-based explanations
* Structured explanations of detected anomalies

The overall objective is to move from:

**Anomaly Detection → Root Cause Identification → Explainable RCA**

---

## Key Features

### 1. RAG-based Root Cause Analysis

The framework retrieves relevant information from telecom specifications and uses it as contextual knowledge for root cause analysis.

This helps the LLM generate explanations grounded in domain-specific information instead of relying only on its internal knowledge.

---

### 2. 3GPP / O-RAN Specification Integration

The repository contains utilities for downloading, processing, and extracting information from technical specifications.

Relevant components include:

* 3GPP specification downloading
* O-RAN specification processing
* Text extraction
* Image extraction
* Table summarization
* Image summarization

The processed specifications serve as the knowledge source for the RCA pipeline.

---

### 3. Retrieval-Augmented Generation Pipeline

The RAG pipeline connects anomalous network behavior with relevant technical knowledge.

A simplified workflow is:

```text
Network Monitoring Data
          │
          ▼
    Anomaly Detection
          │
          ▼
   Anomaly / Metrics
          │
          ▼
     RAG Retrieval
          │
          ▼
  3GPP / O-RAN Knowledge
          │
          ▼
       LLM / RCA
          │
          ▼
 Root Cause Explanation
```

---

### 4. Knowledge Graph-based RCA

The repository also includes components for constructing and using a **Knowledge Graph (KG)** from telecom-domain information.

The KG represents relationships between entities, metrics, network components, and possible causes.

Relevant components include:

* `generate_triplets.py`
* `entity_alignment.py`
* `json_to_db.py`
* `kg_results/`
* `count_entities.py`

This enables structured relationships to complement the retrieval-based reasoning process.

---

### 5. Machine Learning Explainability

The project also includes experiments for explaining anomaly predictions using machine learning models.

The repository contains Random Forest explanation components under:

```text
RF_shape_explaination/
```

and corresponding experimental/ablation code.

---

### 6. Ablation Studies

Several experiments are included to evaluate different components of the proposed approach.

Examples include:

```text
ablation_result/
ablation_test.py
subjective_ablation_test.py
subjective_pipeline.py
```

These experiments are intended to study the contribution of different components of the RCA framework.

---

## Repository Structure

```text
RCA_Explainer/
│
├── 3GPPfiles/
│   └── 3GPP specification files
│
├── RCA_Explaination/
│   └── RCA explanation components
│
├── RCA_Infrence/
│   └── RCA inference components
│
├── RF_shape_explaination/
│   └── Random Forest explanation
│
├── ablation_result/
│   └── Ablation study results
│
├── kg_results/
│   └── Knowledge Graph results
│
├── sample_output/
│   └── Sample RCA outputs
│
├── MCQ_Generator.py
├── RCA_RAG_Pipeline.py
├── TS_ORAN_downloader.py
├── oran_spec_downloader.py
│
├── ablation_test.py
├── subjective_ablation_test.py
├── subjective_pipeline.py
│
├── deepseek.py
├── internvl.py
│
├── extracter.py
├── extract_images.py
├── image_summariser.py
├── image_summarisation_parallel.py
├── table_summariser.py
│
├── generate_triplets.py
├── entity_alignment.py
├── json_to_db.py
├── vectordb_loader.py
│
├── metrics_description.json
├── requirements.txt
└── README.md
```

---

## Main Components

### RCA RAG Pipeline

`RCA_RAG_Pipeline.py`

Implements the RAG-based RCA workflow by combining anomaly information with retrieved domain-specific knowledge.

---

### Specification Processing

The repository provides scripts for working with 3GPP and O-RAN specifications:

```text
TS_ORAN_downloader.py
oran_spec_downloader.py
extracter.py
extract_images.py
```

These components prepare technical specifications for downstream retrieval and reasoning.

---

### Knowledge Graph Construction

The KG pipeline includes:

```text
generate_triplets.py
        │
        ▼
entity_alignment.py
        │
        ▼
json_to_db.py
        │
        ▼
Knowledge Graph
```

The resulting information is used to represent relationships between telecom entities and concepts.

---

### Multimodal Information Extraction

Telecom specifications contain more than plain text. This repository therefore includes components for processing:

* Text
* Tables
* Images

Relevant scripts include:

```text
image_summariser.py
image_summarisation_parallel.py
table_summariser.py
extract_images.py
```

---

## Technologies

The project uses a combination of:

* **Python**
* **Large Language Models (LLMs)**
* **Retrieval-Augmented Generation (RAG)**
* **Knowledge Graphs**
* **Vector Databases**
* **Machine Learning**
* **Random Forest**
* **3GPP / O-RAN specifications**
* **DeepSeek**
* **InternVL**
* **Natural Language Processing**

---

## Installation

Clone the repository:

```bash
git clone https://github.com/AbdullaOvais/RCA_Explainer.git
cd RCA_Explainer
```

Create a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

---

## Usage

The repository contains multiple pipelines and experimental components.

The main RAG-based RCA pipeline can be explored through:

```bash
python RCA_RAG_Pipeline.py
```

Other scripts can be executed independently depending on the experiment or processing stage.

---

## RCA Workflow

The overall research workflow can be summarized as:

```text
        3GPP / O-RAN Specifications
                    │
                    ▼
          Specification Processing
                    │
          ┌─────────┴─────────┐
          ▼                   ▼
     Vector Store        Knowledge Graph
          │                   │
          └─────────┬─────────┘
                    ▼
             Anomaly Information
                    │
                    ▼
              RAG Retrieval
                    │
                    ▼
                   LLM
                    │
                    ▼
             Root Cause Analysis
                    │
                    ▼
          Explainable RCA Output
```

---

## Sample Outputs

Example RCA outputs are available in:

```text
sample_output/
```

These examples demonstrate the type of explanations produced by the framework.

---

## Research Applications

This framework is designed for research in:

* 5G networks
* O-RAN
* Network intelligence
* Anomaly detection
* Root Cause Analysis
* Explainable AI
* Network troubleshooting
* RAG-based network management
* Knowledge Graph-based reasoning
* Telecom-specific LLM applications

---

## Disclaimer

This repository is intended primarily for **research and experimental purposes**.

The generated root-cause explanations should be treated as research outputs and should be independently validated before being used for operational network decisions.

---

## Author

**Abdulla Ovais**

IIT Hyderabad

---

## Citation

If you use this repository or its components in your research, please cite the corresponding work/repository.


