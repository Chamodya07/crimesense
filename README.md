# Crime Sense App

This repository powers the Streamlit-based Crime Sense profiling demo.

## Environment setup

The tabular model was trained in a Colab environment running **Python 3.12** with the
following packages:

- scikit-learn 1.6.1
- joblib 1.5.3
- numpy 2.0.2
- pandas 2.2.2

If you run the app with mismatched versions you may see errors like:

```
AttributeError: Can't get attribute '_RemainderColsList' on sklearn.compose._column_transformer
```

This is caused by a scikit-learn/joblib version mismatch during unpickling of the
model. To avoid it, create a fresh Python 3.12 virtual environment and install
from `requirements.txt`, which now pins the exact training versions.

```bash
python -m venv .venv  # make sure this uses Python 3.12
.\.venv\Scripts\activate
pip install -r requirements.txt
```

Once the dependencies match, the tabular model will load and predictions will vary
with your inputs.

## RAG retrieval index

The prototype can optionally provide similar past-case evidence using a
RAG-style vector search. Before using the Streamlit app you must build the
index from a historical dataset. Two approaches are supported:

**Local file**

```bash
python -m services.rag_index path/to/your/cases.csv  # writes artifacts/rag/*
```

**Via Kaggle** – if you have a `kaggle.json` file in the project root (or
one of the standard locations) the app can download a dataset and build the
index automatically. On the profiling page select one of the supported
slugs and click **Download & Build RAG Index**. The available slugs are:

- `chicago/chicago-crime`
- `brunacmendes/nypd-complaint-data-historic-20062019`
- `venkatsairo4899/los-angeles-crime-data-2020-2023`

The Kaggle CLI must be installed (`pip install kaggle`), and the JSON file
will be copied to `~/.kaggle/kaggle.json` when the button is pressed.

If the artifacts are missing the app will display a friendly message and
continue running without crashing. The index is not required for basic
profiling functionality.
