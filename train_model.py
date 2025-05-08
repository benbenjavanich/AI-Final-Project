# train_model.py  (at project root)

import pandas as pd
from pathlib import Path
from src.models import train_and_save

BASE = Path(__file__).parent
# load the patient CSV
df = pd.read_csv(
    BASE/"data"/"patients.csv",
    converters={"med_times": lambda s: s.split(";")}
)
# train & write urgency_model.pt
train_and_save(df, out_path=str(BASE/"urgency_model.pt"))
