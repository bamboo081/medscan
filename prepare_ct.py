"""Create exploratory image-level CT splits; patient independence is unknown."""
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
root = Path('data/raw/ct/dataset').resolve()
df = pd.read_csv(root / 'kidneyData.csv')
df = pd.DataFrame({'path': df['path'].map(lambda p: str(root / p)), 'label': df['diag'].str.lower()})
if len(df) != 12446 or df.path.duplicated().any() or set(df.label) != {'normal', 'cyst', 'stone', 'tumor'}:
    raise ValueError('Unexpected dataset metadata')
train, rest = train_test_split(df.index, test_size=0.2, stratify=df.label, random_state=42)
val, test = train_test_split(rest, test_size=0.5, stratify=df.loc[rest, 'label'], random_state=42)
df['split'] = ''; df.loc[train, 'split'] = 'train'; df.loc[val, 'split'] = 'val'; df.loc[test, 'split'] = 'test'
out = Path('data/processed/ct'); out.mkdir(parents=True, exist_ok=True)
df.to_csv(out / 'splits.csv', index=False)
print(pd.crosstab(df.split, df.label))
