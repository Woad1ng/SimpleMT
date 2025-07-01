# SimpleMT

This is the official repo for " Simplify  & Translate: A Unified Framework for Accessible Machine Translation"

## Installation

It is recommended to use Python 3.11 and have CUDA installed for GPU acceleration.

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

Some evaluation metrics (such as COMET and BARTScore) may require additional installation:

```bash
pip install comet-ml bart-score
```

## Data Preparation

Place your input data (e.g., `asset.test.zh.orig`) in the `data/asset/` directory, or modify the script parameters as needed.

## Running Translation/Simplification

Use the following shell script to batch run different models and methods:

```bash
bash run_simplemt.sh
```



## Evaluation

Use the following script to automatically evaluate the generated results:

```bash
bash evaluate.sh
```




## Main Scripts

- `SimpleMT.py`: Main translation/simplification script, supporting custom logits processors (e.g., AoA, word frequency).
- `src/evaluate.py`: Automated evaluation script, supporting SARI, BLEU, FKGL, METEOR, COMET, BARTScore, BERTScore, and more.

## Contributing

Feel free to open an issue or pull request if you have questions or suggestions!

---