# Setup Instructions for Windows 🛠️

Follow these steps to create a matching Python environment and run the Streamlit app with the exact versions used during model training.

1. **Create a Python 3.12 virtual environment** (run from project root):

   ```powershell
   py -3.12 -m venv .venv
   ```

2. **Activate the virtual environment**:

   ```powershell
   .\.venv\Scripts\activate
   ```

3. **Upgrade `pip` and install dependencies**:

   ```powershell
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Verify that the pinned versions are active**:

   ```powershell
   python -c "import sklearn,joblib,numpy,pandas; print('sklearn', sklearn.__version__); print('joblib', joblib.__version__); print('numpy', numpy.__version__); print('pandas', pandas.__version__)"
   ```

5. **Run Streamlit using the virtual environment interpreter**:
   ```powershell
   python -m streamlit run app.py
   ```

---

### Troubleshooting

- If the runtime output still shows Python 3.10 or scikit-learn 1.7.x, you're not using the virtual environment. Activate it before running commands.
- **Model load errors** (e.g. "Failed to load tabular model..." or joblib/unpickle problems) mean the scikit-learn/numpy/pandas versions don't match training. Re‑create the venv and reinstall with the pinned versions in `requirements.txt`.
- In VS Code, select the correct interpreter via **Python: Select Interpreter** and choose `.venv\Scripts\python.exe`.
- Prefer `python -m streamlit` rather than the bare `streamlit` command to guarantee the correct environment is used.
