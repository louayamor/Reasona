# Reasona

**Reasona** is a modular AI/ML pipeline framework designed for **streaming, preprocessing, embedding, indexing, training, and inference** on synthetic datasets. It provides end-to-end capabilities for data ingestion, cleaning, transformation, model training, vector-based retrieval, and inference with structured configuration.

---

## Features

* **Streaming Data Ingestion**: Stream large datasets directly from [Hugging Face Datasets].
* **Data Cleaning & Transformation**: Remove duplicates, handle missing values, and format data into instruction-based JSON suitable for training or embedding.
* **Indexing & Embedding**: Chunk and embed streaming data using [SentenceTransformers] and store embeddings in [FAISS].
* **Retrieval & Inference**: Perform vector-based retrieval with top-k search; support transformer-based inference pipelines.
* **Training & Fine-Tuning**: LoRA-based fine-tuning support for custom models.
* **Configuration Management**: Centralized YAML configuration for preprocessing, indexing, retrieval, training, and inference.
* **Modular Architecture**: Clear separation between **producer (preprocessing)** and **consumer (indexing/inference)** pipelines for scalable streaming workflows.

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
│       │   ├── loader.py          # StreamingDatasetProcessor
│       │   ├── cleaner.py         # Data cleaning utilities
│       │   ├── formatter.py       # DataFormatter
│       │   ├── chunker.py         # TextChunker for embeddings
│       │   └── __init__.py
│       ├── pipeline/
│       │   ├── preprocess_pipeline.py  # Producer
│       │   ├── indexing_pipeline.py    # Consumer
│       │   ├── training_pipeline.py    # Training
│       │   └── inference_pipeline.py   # Inference / Retrieval
│       ├── vectorstore/
│       │   └── faiss_store.py          # FAISS integration
│       └── utils/
│           ├── logger.py               # Logging
│           └── helpers.py              # Utility helpers
├── config/
│   ├── config.yaml
│   └── params.yaml
├── artifacts/                # Saved models, embeddings, vector stores
├── main.py                   # Entry point for streaming + indexing pipelines
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

* **preprocess**: dataset name, split, max samples, batch size.
* **embedding**: embedding model, chunk size/overlap, vector store directory.
* **retrieval**: top-k results, embedding model, vector store directory.
* **inference**: model path, tokenizer path, engine type, max tokens, temperature.

Example `preprocess` section:

```yaml
preprocess:
  dataset_name: "PleIAs/SYNTH"
  split: "train"
  max_samples: 10000
  batch_size: 500
```

---

## Usage

### Run Preprocessing + Indexing (Producer → Consumer)

```bash
python main.py
```

Pipeline flow:

1. **Preprocessing Pipeline (Producer)**

   * Streams data from Hugging Face.
   * Cleans and formats into instruction-based JSON.

2. **Indexing Pipeline (Consumer)**

   * Chunks streamed data using `TextChunker`.
   * Embeds chunks using `SentenceTransformers`.
   * Stores vectors in FAISS for later retrieval.

> The pipelines are **streaming-first**, avoiding intermediate file storage.

### Training & Inference

Configure `config.yaml` and `params.yaml` for your model, then run:

```bash
python src/Reasona/pipeline/training_pipeline.py
python src/Reasona/pipeline/inference_pipeline.py
```

---

## Supported Datasets

* Hugging Face Datasets (streamable).
* [`PleIAs/SYNTH`](https://huggingface.co/datasets/PleIAs/SYNTH).

---

## Logging

All pipelines produce **JSON-based logs**:

* `logs/pipeline/preprocess_pipeline.json`
* `logs/pipeline/indexing_pipeline.json`
* `logs/pipeline/main_pipeline.log`

---

## Tools & Libraries Used

* **Python 3.10+**
* [Hugging Face Datasets](https://huggingface.co/docs/datasets) – streaming large datasets
* [SentenceTransformers](https://www.sbert.net/) – embeddings
* [FAISS](https://github.com/facebookresearch/faiss) – vector storage and search
* [PyYAML](https://pyyaml.org/) – configuration
* [LoRA / Transformers](https://huggingface.co/docs/transformers/) – model training & fine-tuning

---

## Author

**Louay Amor** – [GitHub](https://github.com/louayamor) | [LinkedIn](https://linkedin.com/in/louayamor)
