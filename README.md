# Reproducing a Question-Answering System with NLP

This project implements a QA system using a Dynamic Memory Network (DMN), inspired by the paper ["Dynamic Memory Networks for NLP"](https://arxiv.org/pdf/1603.01417.pdf). It processes the bAbI dataset and leverages GloVe embeddings for context and question understanding.

## Setup Instructions

1. Clone the repository and install dependencies:
   ```bash
   git clone https://github.com/your-username/qa-system-nlp.git
   cd qa-system-nlp
   pip install -r requirements.txt
   ```
## Download and Place the Datasets

- **GloVe Embeddings** in `data/`
- **bAbI Dataset** in `data/`

## Verify the Structure

```plaintext
project_root/
├── data/                # GloVe and bAbI datasets
├── models/              # Neural modules
├── utils/               # Utility scripts
├── main.py              # Main script
├── requirements.txt     # Dependencies
```


## Running the Project

To train and evaluate the model:
```bash
python main.py
```

## Project Structure

- **`main.py`**: Trains and evaluates the DMN model.
- **`utils/`**: Utility scripts for preprocessing and batching.
- **`models/`**: Implements the DMN components:
  - `input_module.py`
  - `question_module.py`
  - `episodic_memory.py`
  - `answer_module.py`

## Key Components

1. **Input Module**: Encodes the input context with GRU.
2. **Question Module**: Encodes the question with GRU.
3. **Episodic Memory**: Updates memory iteratively using attention.
4. **Answer Module**: Generates the answer from the final memory.

## Acknowledgments

- *Dynamic Memory Networks for NLP* by Ankit Kumar et al.
- **Stanford NLP Group** for GloVe.
- **Facebook Research** for the bAbI dataset.