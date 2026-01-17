![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=fff) ![HuggingFace](https://img.shields.io/badge/HuggingFace-F18C00?logo=huggingface&logoColor=fff) ![SentenceTransformers](https://img.shields.io/badge/SentenceTransformers-FF6F61?logo=transformers&logoColor=fff) ![FAISS](https://img.shields.io/badge/FAISS-3C3C3C?logo=faiss&logoColor=fff) ![Flask](https://img.shields.io/badge/Flask-000000?logo=flask&logoColor=fff) ![DVC](https://img.shields.io/badge/DVC-007ACC?logo=dvc&logoColor=fff) ![DagsHub](https://img.shields.io/badge/DagsHub-FF7F50?logo=dagshub&logoColor=fff) ![MLflow](https://img.shields.io/badge/MLflow-00BFFF?logo=mlflow&logoColor=fff)



**Reasona** is a modular AI/ML pipeline framework for **streaming, preprocessing, embedding, indexing, retrieval, reranking, and inference** on large-scale datasets. It supports **streaming-first workflows** to process data end-to-end efficiently, with vector-based retrieval and transformer-based inference, and a Flask web app for interactive usage.

---

## Features

* **Streaming Data Ingestion**: Stream large datasets directly from [Hugging Face Datasets] or Wikimedia without downloading entire files.
* **Data Cleaning & Transformation**: Remove duplicates, handle missing values, and convert to instruction-based JSON for embedding.
* **Chunking & Embedding**: Split long documents into configurable chunks with optional overlap; embed using [SentenceTransformers].
* **Vector Indexing**: Store embeddings in [FAISS] for fast similarity search.
* **Retrieval & Reranking**: Retrieve top-k results via FAISS and optionally rerank results using transformer-based models.
* **Inference**: Generate answers using pretrained models with configurable parameters.
* **Flask Web App**: Interactive interface for querying Reasona pipelines, managing chats, and visualizing results.
* **Scalable & Configurable**: Centralized YAML configuration for all pipelines.
* **Logging & Monitoring**: JSON logs and runtime metrics for all stages, including checkpoints and progress tracking.

---

## Project Structure

```
Reasona/
│
├── src/
│   └── Reasona/
│       ├── config/
│       │   ├── config_manager.py
│       │   └── params.yaml
│       ├── data/
│       │   ├── loader.py         
│       │   ├── cleaner.py         
│       │   ├── formatter.py      
│       │   ├── chunker.py        
│       │   └── embedder.py        
│       ├── pipeline/
│       │   ├── preprocess_pipeline.py   # Streaming + preprocessing
│       │   ├── indexing_pipeline.py     # Chunking + embedding + FAISS
│       │   ├── reranking_pipeline.py    # Transformer reranking
│       │   ├── inference_pipeline.py    # Inference & retrieval
│       │   └── training_pipeline.py     # Qwen/Qwen2.5-1.5B-Instruct
│       ├── vectorstore/
│       │   └── faiss_store.py          
│       ├── services/
│       │   └── reasona_service.py      
│       └── utils/
│           ├── logger.py               
│           └── helpers.py              
├── config/
│   ├── config.yaml                     
│   └── params.yaml                     
├── artifacts/                           
├── logs/                                
├── main.py                              
├── app.py                               
└── README.md
```

---

## Installation

1. Clone the repository:

```bash
git clone https://github.com/louayamor/Reasona.git
cd Reasona
```

2. Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Configuration

Control pipelines via `config/config.yaml` and `config/params.yaml`. Key sections:

* **preprocess**: dataset, split, max samples, batch size, shuffle, prefetch buffer.
* **indexing**: embedding model, chunk size/overlap, batch size, queue size, vector store directory, checkpoint frequency.
* **reranking**: reranker model, tokenizer, top-k reranking.
* **retrieval**: top-k results, embedding model, vector store path.
* **inference**: model path, tokenizer path, engine type, max tokens, temperature.
* **flask**: host, port, debug mode.

---

## Usage

### Preprocessing + Indexing (Streaming Mode)

```bash
python main.py
```

Pipeline flow:

1. **Preprocessing Pipeline (Producer)**

   * Streams data from Hugging Face or Wikimedia.
   * Cleans, formats, and converts to instruction-based JSON.
   * Stops at `max_samples`.

2. **Indexing Pipeline (Consumer)**

   * Chunks data.
   * Embeds chunks using `SentenceTransformers`.
   * Stores vectors in FAISS with checkpoints.

> Pipelines communicate via queues to handle large datasets efficiently.

### Reranking & Inference

```bash
python src/Reasona/pipeline/reranking_pipeline.py
python src/Reasona/pipeline/inference_pipeline.py
```

* Retrieve top-k results from FAISS.
* Optionally rerank using transformer models.
* Generate answers or code snippets.

### Run Flask App

```bash
python app.py
```

* Access interactive web interface at `http://localhost:5000`.
* Query datasets, retrieve top-k results, and perform inference.

---

## Supported Datasets

* Hugging Face Datasets (streamable)
* [`wikimedia/wikipedia`](https://huggingface.co/datasets/wikimedia/wikipedia)

---

## Logging & Monitoring

* JSON-based logs:

  * `logs/pipeline/preprocess_pipeline.json`
  * `logs/pipeline/indexing_pipeline.json`
  * `logs/pipeline/reranking_pipeline.json`
  * `logs/pipeline/inference_pipeline.json`

* Logs include progress, checkpoints, runtime, and embedding statistics.

---

## Author

**Louay Amor** – [GitHub](https://github.com/louayamor) | [LinkedIn](https://linkedin.com/in/louayamor)
