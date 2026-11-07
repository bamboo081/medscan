"""Create NIH splits using official test list and patient-level train/val split."""
from pathlib import Path
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

root = Path('data/archive (2)').resolve()
out = Path('data/processed/nih')
out.mkdir(parents=True, exist_ok=True)
df = pd.read_csv(root / 'Data_Entry_2017.csv')
files = {p.name: str(p.resolve()) for p in root.glob('images_*/images/*.png')}
test = set((root / 'test_list.txt').read_text().splitlines())
trainval = set((root / 'train_val_list.txt').read_text().splitlines())
if set(df['Image Index']) != test | trainval or test & trainval:
    raise ValueError('Metadata and official partition lists differ')
df['path'] = df['Image Index'].map(files)
if df.path.isna().any():
    raise ValueError('Some images are missing')
df = df.rename(columns={'Patient ID': 'patient_id', 'Finding Labels': 'labels'})
df['split'] = 'test'
idx = df.index[df['Image Index'].isin(trainval)]
tr, va = next(GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=42).split(df.loc[idx], groups=df.loc[idx, 'patient_id']))
df.loc[idx[tr], 'split'] = 'train'
df.loc[idx[va], 'split'] = 'val'
df[['path', 'patient_id', 'labels', 'split']].to_csv(out / 'splits.csv', index=False)
print(df.groupby('split').agg(images=('path', 'size'), patients=('patient_id', 'nunique')))
print('Saved', out / 'splits.csv')
