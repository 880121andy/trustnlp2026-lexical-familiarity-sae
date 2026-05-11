# Scripts

Scripts for reproducing the experiments including:
- `run.py`: main SAE analysis script
- `lf_simplified.py`, `lf_full.py`: compute lexical familiarity score script

(Optional)Scripts for Gemma-12B-PT & Qwen-9B-Base model-generalization experiments can be accessed in respective folders.

## How To Run

1. `run.py`:

```bash
pip install -e requirements.txt

python run.py
```

2. `lf_simplified.py` / `lf_full.py`:

```bash
# Make sure you have transformers installed
pip install transformers

# Run the full analysis
python lf_full.py --data-dir /path/to/data --output ./lf_results

# or, the simplified analysis
python lf_simplified.py --data-dir /path/to/data --output ./lf_results
```
