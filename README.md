# Reasona

**Reasona** is a modular AI/ML pipeline framework designed for processing, cleaning, transforming, and training on synthetic datasets. It provides end-to-end capabilities for data ingestion, preprocessing, model training, and inference with structured configuration.

---

## 🚀 Features

- **Data Ingestion**: Stream and combine large datasets efficiently, optimized for limited RAM (16GB tested).  
- **Data Cleaning**: Remove duplicates, handle missing values, and preprocess data for downstream tasks.  
- **Data Transformation**: Format datasets into instruction-based formats suitable for model training.  
- **Training & Fine-Tuning**: Support for LoRA-based fine-tuning pipelines.  
- **Inference**: Flexible inference engine integration with customizable parameters.  
- **Configuration Management**: Centralized YAML-based configuration for preprocessing, training, and inference.  

---

## 💾 Project Structure

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
│       │   └── __init__.py
│       ├── pipeline/
│       │   ├── preprocess_pipeline.py
│       │   ├── training_pipeline.py
│       │   └── inference_pipeline.py
│       └── utils/
│           ├── logger.py
│           └── helpers.py
├── config/
│   ├── config.yaml
│   └── params.yaml
├── artifacts/                # Generated datasets and outputs
├── main.py                   # Entry point to run pipelines
└── README.md
```

---

## ⚙️ Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/Reasona.git
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

## 💪 Usage

### Run Preprocessing Pipeline

```bash
python main.py
```

This will execute the preprocessing pipeline including:

1. **Data Ingestion** – combining dataset files (streaming from Hugging Face if configured).  
2. **Data Cleaning** – removing duplicates and handling missing values.  
3. **Data Transformation** – formatting into instruction-based JSON for training.  

### Training and Inference

After preprocessing, configure `config.yaml` and `params.yaml` for model training or inference pipelines. Then run:

```bash
python src/Reasona/pipeline/training_pipeline.py
python src/Reasona/pipeline/inference_pipeline.py
```

---

## 🌐 Data Source

- The pipeline supports streaming large datasets directly from [Hugging Face Datasets](https://huggingface.co/datasets).  
- Example dataset used: [`PleIAs/SYNTH`](https://huggingface.co/datasets/PleIAs/SYNTH)  

---

## 👤 Author

**Louay Amor** – [GitHub](https://github.com/louayamor) | [LinkedIn](https://linkedin.com/in/louayamor)

