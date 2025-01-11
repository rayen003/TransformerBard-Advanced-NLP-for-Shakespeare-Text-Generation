# Shakespeare Text Generation with Transformers

This project implements a transformer-based model for generating Shakespeare-style text. The repository documents various experiments and improvements in the model architecture, embeddings, and training approaches.

## Project Structure
```
.
├── README.md
├── data/
│   └── shakespeare.txt
├── src/
│   ├── preprocessing_pipeline.py
│   ├── training_pipeline.py
│   ├── inference_pipeline.py
│   └── inference.py
├── experiments/
│   └── README.md
└── requirements.txt
```

## Experiments
Each experiment is documented in the `experiments/` directory with its own configuration, results, and analysis.

## Setup
1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Run training: `python src/training_pipeline.py`

## Current Results
Initial model (2024-01-12):
- Training data: 5% of Shakespeare text
- Vocabulary size: 11,914 words
- Test accuracy: 2.80%
- Test loss: 7.5568
