import subprocess
# Run tabular.py
print("Running tabular baseline (CatBoost)")
subprocess.run(["python3", "tabular.py"])
# Run multimodal.py
print("Running multimodal model (CNN + GRU + MLP)")
subprocess.run(["python3", "multimodal.py"])
