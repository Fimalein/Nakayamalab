import argparse
import pyvisa
import numpy as np
import matplotlib.pyplot as plt
import h5py
from datetime import datetime
import pandas as pd
import os
from io import BytesIO

# 引数パース
parser = argparse.ArgumentParser()
parser.add_argument("--type", choices=["YT", "XY"], default="YT")
parser.add_argument("--title", type=str, default="YT graph")
parser.add_argument("--xlabel", type=str, default="T")
parser.add_argument("--ylabel", type=str, default="Y")
args = parser.parse_args()

# 保存ディレクトリの指定
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
outdir = os.path.join(BASE_DIR, "data")
os.makedirs(outdir, exist_ok=True)

# VISA接続
rm = pyvisa.ResourceManager()
inst_list = rm.list_resources()
print("接続可能な機器:", inst_list)

inst = rm.open_resource(inst_list[1])
print("接続中:", inst.query("*idn?"))

inst.write("header off")
inst.write("data:encdg sribinary")
inst.write("data:width 2")
inst.write("data:start 1")
inst.write("data:stop 500")

# データ取得
inst.write("data:source ch1")
wave_data_1 = inst.query_binary_values("curv?", datatype="h", is_big_endian=False, container=np.array)

inst.write("data:source ch2")
wave_data_2 = inst.query_binary_values("curv?", datatype="h", is_big_endian=False, container=np.array)

# CSVデータ
if args.type == "YT":
    df = pd.DataFrame({"T": wave_data_1.flatten(), "Y": wave_data_2.flatten()})
else:
    df = pd.DataFrame({"X": wave_data_1.flatten(), "Y": wave_data_2.flatten()})
csv_string = df.to_csv(index=False)

# プロット生成
fig, ax = plt.subplots()
if args.type == "YT":
    ax.plot(df["T"], df["Y"])
else:
    ax.plot(df["X"], df["Y"])
ax.set_xlabel(args.xlabel)
ax.set_ylabel(args.ylabel)
ax.set_title(args.title)
ax.grid(True)

# PNG画像をバイト列に
buf = BytesIO()
fig.savefig(buf, format='png')
buf.seek(0)
png_bytes = np.frombuffer(buf.getvalue(), dtype='uint8')
buf.close()
plt.close(fig)

# HDF5保存
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
filename = f"scope_data_{timestamp}.h5"
h5_path = os.path.join(outdir, filename)

with h5py.File(h5_path, 'w') as f:
    f.create_dataset("T", data=wave_data_1.flatten(), dtype='int16')
    f.create_dataset("Y", data=wave_data_2.flatten(), dtype='int16')
    f.create_dataset("csv_data", data=np.string_(csv_string))
    f.create_dataset("plot_image", data=png_bytes)
    f.attrs["source"] = inst_list[0]
    f.attrs["created_at"] = timestamp
    f.attrs["graph_type"] = args.type
    f.attrs["title"] = args.title
