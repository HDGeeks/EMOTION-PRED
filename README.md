

# 🐍 Virtual Environment Setup Guide

This project requires a Python virtual environment.  
Python **3.9 or higher** is recommended.

Follow the steps below depending on your operating system.

---

## 1. Create a Virtual Environment

### macOS / Linux
```bash
python3 -m venv venv

Windows

python -m venv venv


⸻

2. Activate the Virtual Environment

macOS / Linux

source venv/bin/activate

Windows (PowerShell)

.\venv\Scripts\activate

When activated, your terminal prompt will show:

(venv) $

This confirms that the environment is active.

⸻

3. Install Project Dependencies

Make sure the virtual environment is still activated, then run:

pip install -r requirements.txt


⸻

4. Set Up Jupyter Kernel (Optional but Recommended)

If you are using Jupyter Notebook or JupyterLab, register the virtual environment as a kernel:

python -m ipykernel install --user --name=venv --display-name="Python (venv)"

Then inside Jupyter:
	•	Open Kernel → Change Kernel
	•	Select Python (venv)

⸻

🎉 You’re Ready!

Run the notebooks, experiment, break things, fix things —
enjoy the project!

