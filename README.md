# Reasona

**Reasona** is a modular AI/ML pipeline framework designed for **streaming, preprocessing, embedding, indexing and inference** on large-scale datasets. It is optimized for **streaming-first workflows**, allowing end-to-end processing without intermediate storage, while supporting vector-based retrieval and transformer-based inference.

---

## Features

* **Streaming Data Ingestion**: Stream large datasets directly from [Hugging Face Datasets] without downloading entire files.
* **Data Cleaning & Transformation**: Remove duplicates, handle missing values, and format data into instruction-based JSON suitable for embedding.
* **Chunking & Embedding**: Break long documents into configurable chunks (with optional overlap) and embed using [SentenceTransformers].
* **Vector Indexing**: Store embeddings in [FAISS] for fast similarity search and retrieval.
* **Retrieval & Inference**: Perform top-k search over embeddings and support inference using pretrained transformer models.
* **Scalable & Configurable**: Centralized YAML configuration for preprocessing, indexing, retrieval, training, and inference.
* **Logging & Monitoring**: JSON-based logs with progress, checkpoints, and runtime metrics for each pipeline stage.

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
│       │   ├── loader.py          # StreamingDatasetLoader
│       │   ├── cleaner.py         # Data cleaning utilities
│       │   ├── formatter.py       # DataFormatter
│       │   ├── chunker.py         # TextChunker for embeddings
│       │   └── embedder.py        # SentenceTransformers embedding
│       ├── pipeline/
│       │   ├── preprocess_pipeline.py  # Producer (streaming + preprocessing)
│       │   ├── indexing_pipeline.py    # Consumer (chunking + embedding + FAISS)
│       │   ├── training_pipeline.py    # Training LoRA models
│       │   └── inference_pipeline.py   # Inference & retrieval
│       ├── vectorstore/
│       │   └── faiss_store.py          # FAISS integration
│       └── utils/
│           ├── logger.py               # Logging utilities
│           └── helpers.py              # Misc helpers
├── config/
│   ├── config.yaml                     # Main pipeline configuration
│   └── params.yaml                     # Parameters like chunk size, batch size
├── artifacts/                           # Saved models, embeddings, vector stores
├── logs/                                # JSON log files
├── main.py                              # Entry point for preprocessing + indexing
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

All pipelines are controlled via `config/config.yaml` and `config/params.yaml`.
Key sections:

* **preprocess**: dataset name, split, max samples, batch size, shuffle and prefetch buffers.
* **indexing**: embedding model, chunk size/overlap, batch size, queue size, vector store directory, checkpoint frequency.
* **retrieval**: top-k results, embedding model, vector store directory.
* **inference**: model path, tokenizer path, engine type, max tokens, temperature.


## Usage

### Run Preprocessing + Indexing (Producer → Consumer)

```bash
python main.py
```

Pipeline flow:

1. **Preprocessing Pipeline (Producer)**

   * Streams data from Hugging Face.
   * Cleans and formats into instruction-based JSON.
   * Stops automatically when `max_samples` is reached.

2. **Indexing Pipeline (Consumer)**

   * Chunks data with configurable overlap.
   * Embeds chunks using `SentenceTransformers`.
   * Stores vectors in FAISS and saves checkpoints.

> Both pipelines run in **streaming mode** and communicate through queues to handle large datasets efficiently.

### Training & Inference

Configure `config.yaml` for your model, then run:

```bash
python src/Reasona/pipeline/training_pipeline.py
python src/Reasona/pipeline/inference_pipeline.py
```

---

## Supported Datasets

* Hugging Face Datasets (streamable)
* [`PleIAs/SYNTH`](https://huggingface.co/datasets/PleIAs/SYNTH)
* `wikimedia/wikipedia` (configurable language versions)

---

## Logging & Monitoring

* JSON-based logs for each pipeline:

  * `logs/pipeline/preprocess_pipeline.json`
  * `logs/pipeline/indexing_pipeline.json`
  * `logs/pipeline/main_pipeline.log`
* Logs include progress, checkpoint saving, runtime, and embedding statistics.

---

## Tools & Libraries Used

* **Python 3.10+**
* [Hugging Face Datasets](https://huggingface.co/docs/datasets) – streaming and processing large datasets
* [SentenceTransformers](https://www.sbert.net/) – embeddings
* [FAISS](https://github.com/facebookresearch/faiss) – vector storage and search
* [PyYAML](https://pyyaml.org/) – configuration management

---

## Author

**Louay Amor** – [GitHub](https://github.com/louayamor) | [LinkedIn](https://linkedin.com/in/louayamor)
