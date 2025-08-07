# 🕒 TimeAgent: Forecasting Made Easy

**TimeAgent** is a modular and powerful framework for time series forecasting — built for Kaggle competitions, research, and real-world applications.

Whether you're a beginner with a CSV or a researcher doing AutoML, TimeAgent helps you quickly load data, train models, evaluate, and export submissions.

---

## 🚀 Quick Start (No Hassle)

If you're **not a Python expert**, here's the easiest way to run everything.

### ✅ What You Need First

- [Anaconda](https://www.anaconda.com/products/distribution) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html)
- Your dataset (train/test/sample files)
- The `TimeAgent.zip` file (shared with you)
- This script: [`run_agent.sh`](run_agent.sh)

---

### 📦 Step-by-step

```bash
bash run_agent.sh TimeAgent.zip /path/to/your/data --description "Forecasting wind for August"
```

> 📂 `/path/to/your/data` must contain your `train.csv`, and optionally  `test.csv` and `sample_submission.csv`.

That’s it! This will:
- 🔓 Unzip the project
- ⚙️ Create a clean Conda environment from `environment.yml`
- 🧠 Run the agent on your data
- 📄 Save predictions in `submissions/submission.csv`

---

## 🧠 What TimeAgent Can Do

- 🔍 Automatically detect time, group, and target columns
- 🧪 Smart train/test splitting for time series
- 📈 Model training (transformer-based, DLinear, etc.)
- 🧬 Post-training reinforcement learning (optional)
- 📤 Kaggle-style submission output
- 📊 Built-in metrics and error reporting

---

## 💻 For Developers / Researchers

If you're technical and want more control:

### 1. Clone or unzip the repo manually

```bash
unzip TimeAgent.zip
cd TimeAgent
```

### 2. Set up the environment

If using Conda:

```bash
conda env create -f environment.yml
conda activate timeagent
```

> Or, if you prefer `pip`:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Run the agent

```bash
python main_agent.py --inputs path/to/data --description "My forecasting project"
```

---

## ⚙️ CLI Options

| Option                | Description                                      |
|-----------------------|--------------------------------------------------|
| `--inputs`            | Path to data folder or CSV files                 |
| `--grouping-cols`     | (Optional) Columns to group by (e.g., location)  |
| `--target-cols`       | (Optional) Target columns (e.g., temperature)    |
| `--output-dir`        | Where to save `submission.csv` (default: `submissions/`) |
| `--description`       | (Optional) Short description of your project     |

**Example:**

```bash
python automated_framework.py --inputs data/ --grouping-cols location --target-cols temperature --description "Temp forecasting"
```

---

## 🎛️ Streamlit App (Optional UI)

If you want an interactive interface:

```bash
streamlit run stream_app.py
```

This will launch a local web app where you can:
- Upload train/test/sample files
- View column summaries
- Choose grouping and target columns
- Launch training directly from UI
- Export predictions

---

## 📁 Project Structure

```
TimeAgent/
├── automated_framework.py              # CLI entry point
├── environment.yml            # Conda environment
├── requirements.txt           # (Optional) pip install fallback
├── run_agent.sh               # One-line execution script
├── src/
│   ├── csv_merging/           # Merging logic
│   ├── time_series_training/  # Model training
│   ├── kaggle_utilities/      # Submission prep
│   ├── post_training/         # RL optimization
│   ├── utilities/             # Setup, helpers
│   └── app/                   # Streamlit UI
```

---

## 📤 Output

After running, your predictions will be saved in:

```
submissions/
└── submission.csv
```

---

## ❓ FAQ

### What if I only have `train.csv`?

No problem! The agent will automatically split it into train/test (80/20).

---

### What kinds of models does it use?

We use deep learning models like:
- [DLinear](https://arxiv.org/abs/2205.13504)
- [Transformer variants]

---

### Can I use this in Kaggle?

Yes. It supports Kaggle-ready formats and generates `submission.csv`.

---

## 📬 Questions?

Open an issue or reach out to the maintainer.

---

**Happy Forecasting! ⏳**
