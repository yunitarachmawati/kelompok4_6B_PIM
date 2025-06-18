import pandas as pd

# Baca file CSV raw (yang bermasalah)
df = pd.read_csv("users.csv", header=0, names=["username", "password", "email"], skip_blank_lines=True)

# Pastikan kolom email tidak kosong, isi dengan string kosong kalau kosong
df['email'] = df['email'].fillna('')

# Ubah dulu jadi string, baru strip
df['username'] = df['username'].astype(str).str.strip()
df['password'] = df['password'].astype(str).str.strip()
df['email'] = df['email'].astype(str).str.strip()

# Simpan ulang file CSV yang sudah bersih
df.to_csv("users_clean.csv", index=False)
