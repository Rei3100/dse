import os

# テストでは Demucs (重い・モデルDL) を無効化し、dedup を full-MFCC 縮退で高速に回す。
# 実機 (DSRE.exe) は DSRE_DEMUCS 未設定 = 既定 ON で vocal stem 編成判定を使う。
os.environ.setdefault("DSRE_DEMUCS", "0")
