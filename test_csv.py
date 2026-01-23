import pandas as pd
from pathlib import Path

df = pd.read_csv('data/parametrar/Beställningsfrekvens.csv', sep=';', encoding='utf-8')

print("Columns:", list(df.columns))
print("\nFirst 3 rows:")
for idx in range(min(3, len(df))):
    print(f"\nRow {idx}:")
    row = df.iloc[idx]
    print(f"  Col 0 ({df.columns[0]}): {row.iloc[0]}")
    print(f"  Col 1 ({df.columns[1]}): {row.iloc[1]}")
    print(f"  Col 2 ({df.columns[2]}): {row.iloc[2]}")
    print(f"  Col 3 ({df.columns[3]}): {row.iloc[3]}")
