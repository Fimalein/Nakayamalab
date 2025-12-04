import pyvisa
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from logging import getLogger, StreamHandler, DEBUG
import h5py
from datetime import datetime

# ===== ログ設定 =====
logger = getLogger('logging1')
handler = StreamHandler()
handler.setLevel(DEBUG)
logger.setLevel(DEBUG)
logger.addHandler(handler)
logger.debug('start')

# ===== VISA初期化 =====
rm = pyvisa.ResourceManager()
rm_list = rm.list_resources()
print(rm_list)

inst1 = rm.open_resource(rm_list[1])  
inst2 = rm.open_resource(rm_list[2])
inst1.write_termination = '\n'
inst1.read_termination  = '\n'
inst1.timeout = 10000

logger.debug(inst1.query('*IDN?'))

# ===== バイナリ取得（生カウント値） =====
inst1.write("HEADer OFF")
inst1.write("DATa:ENCdg SRIBinary")      
inst1.write("DATa:STARt 1")
inst1.write("DATa:STOP 500")
inst1.write("ACQuire:STOPAfter SEQuence")
inst1.write("ACQuire:STATE RUN")
inst1.query("*OPC?")                     

inst1.write("DATa:SOURce CH1")
wave1 = np.array(inst1.query_binary_values("CURVe?", datatype="h", is_big_endian=False),
                 dtype=np.int16)
inst1.write("DATa:SOURce CH2")
wave2 = np.array(inst1.query_binary_values("CURVe?", datatype="h", is_big_endian=False),
                 dtype=np.int16)

n_bin = min(len(wave1), len(wave2))
ch1_b = wave1[:n_bin]
ch2_b = wave2[:n_bin]

# バイナリ版CSV
df_bin = pd.DataFrame({"CH1_raw": ch1_b, "CH2_raw": ch2_b})
df_bin.to_csv("wave_binary.csv", index=False)

# ===== ASCII取得（スケール済み想定） =====
inst1.write("HEADer OFF")
inst1.write("DATa:ENCdg ASCii")
inst1.write("DATa:STARt 1")
inst1.write("DATa:STOP 500")
inst1.write("ACQuire:STOPAfter SEQuence")
inst1.write("ACQuire:STATE RUN")
inst1.query("*OPC?")

inst1.write("DATa:SOURce CH1")
ch1 = np.array(inst1.query_ascii_values("CURVe?"), dtype=float)
inst1.write("DATa:SOURce CH2")
ch2 = np.array(inst1.query_ascii_values("CURVe?"), dtype=float)

n_asc = min(len(ch1), len(ch2))
ch1_a = ch1[:n_asc]
ch2_a = ch2[:n_asc]

# ASCII版CSV
df_asc = pd.DataFrame({"CH1_V": ch1_a, "CH2_V": ch2_a})
df_asc.to_csv("wave_ascii.csv", index=False)

# ===== HDF5保存 =====
ts = datetime.now().strftime("%Y%m%d_%H%M%S")
h5_path = f"{ts}.h5"

with h5py.File(h5_path, "w") as f:
    run = f.create_group(f"runs/{ts}")

    # インデックス（サンプル番号）を共通軸として保存
    idx_ascii = np.arange(n_asc, dtype=np.int64)
    idx_binary = np.arange(n_bin, dtype=np.int64)

    # スケール済み(とみなす)波形
    g_v = run.create_group("ascii_like")
    g_v.create_dataset(
        "index", data=idx_ascii,
        compression="gzip", compression_opts=5,
        shuffle=True, fletcher32=True, chunks=True
    )
    g_v.create_dataset(
        "CH1_V", data=ch1_a, dtype="f8",
        compression="gzip", compression_opts=5,
        shuffle=True, fletcher32=True, chunks=True
    )
    g_v.create_dataset(
        "CH2_V", data=ch2_a, dtype="f8",
        compression="gzip", compression_opts=5,
        shuffle=True, fletcher32=True, chunks=True
    )

    # RAW波形
    g_raw = run.create_group("binary")
    g_raw.create_dataset(
        "index", data=idx_binary,
        compression="gzip", compression_opts=5,
        shuffle=True, fletcher32=True, chunks=True
    )
    g_raw.create_dataset(
        "CH1_raw", data=ch1_b, dtype="i2",
        compression="gzip", compression_opts=5,
        shuffle=True, fletcher32=True, chunks=True
    )
    g_raw.create_dataset(
        "CH2_raw", data=ch2_b, dtype="i2",
        compression="gzip", compression_opts=5,
        shuffle=True, fletcher32=True, chunks=True
    )

    # メタ情報
    g_meta = run.create_group("meta")
    g_meta.attrs["オシロ"] = inst1.query("*IDN?").strip()
    g_meta.attrs["発振器"] = inst2.query("*IDN?").strip()
    g_meta.attrs["タイムスタンプ"] = ts
    g_meta.attrs["wfmpre?"] = inst1.query("wfmpre?").strip()
    

logger.debug(f"saved HDF5: {h5_path}")

# ===== HDF5読み出し =====
with h5py.File(h5_path, "r") as f:
    run_key = list(f["runs"].keys())[0]
    g_v = f[f"runs/{run_key}/ascii_like"]
    idx = g_v["index"][:]
    ch1_h5 = g_v["CH1_V"][:]
    ch2_h5 = g_v["CH2_V"][:]

# プロット（
plt.figure()
plt.plot(idx, ch1_h5, label="CH1 (ASCII-like)")
plt.plot(idx, ch2_h5, label="CH2 (ASCII-like)")
plt.xlabel("Sample index")
plt.ylabel("Amplitude (unit from scope)")
plt.legend()
plt.tight_layout()
plt.show()

inst1.close()
logger.debug("finish")
