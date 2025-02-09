import subprocess
libraries = [
    "numpy", "pandas", "tqdm", "rdkit", 
    "scikit-learn", "CGRtools", "matplotlib", 
    "seaborn", "imblearn", "dill", "xgboost", "torch"
]

result = subprocess.run(["pip", "freeze"], stdout=subprocess.PIPE, text=True)


with open("../requirements.txt", "w") as f:
    for line in result.stdout.splitlines():
        if any(line.startswith(lib) for lib in libraries):
            f.write(line + "\n")
