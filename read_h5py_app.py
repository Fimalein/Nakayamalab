import h5py
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import sys
from io import BytesIO


def load_scope_data(filepath: str):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    with h5py.File(filepath, 'r') as f:
        ch1 = f['T'][()]
        ch2 = f['Y'][()]
        source = f.attrs.get('source', 'unknown')
        created_at = f.attrs.get('created_at', 'unknown')

        base_name = os.path.splitext(os.path.basename(filepath))[0]
        extract_dir = os.path.join(os.path.dirname(filepath), base_name)
        os.makedirs(extract_dir, exist_ok = True)

# csv
        csv_bytes = f["csv_data"][()]
        csv_str = csv_bytes.decode("utf-8")
        df = pd.read_csv(BytesIO(csv_str.encode()))
        csv_path = os.path.join(extract_dir, f"{base_name}.csv")
        with open(csv_path, "w", encoding="utf-8") as csv_out:
            csv_out.write(csv_str)
        print(f"CSVを保存しました: {csv_path}")

# png
        img_data = f["plot_image"][()]
        image_path = os.path.join(extract_dir, f"{base_name}.png")
        with open(image_path, "wb") as out:
            out.write(img_data.tobytes())
        
        print(f"画像を保存しました： {image_path}")

    return ch1, ch2, source, created_at


def plot_scope_data(ch1, ch2, title="Scope Data"):
    plt.figure(figsize=(10, 5))
    plt.plot(ch1, ch2, label="YTグラフ")
    plt.xlabel("T")
    plt.ylabel("Y")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # --- 修正前 ---
    # filedate = input(" please input h5file date:")
    # filepath = "./data/scope_data_" + filedate + ".h5"

    # --- 修正後 ---
    if len(sys.argv) < 2:
        print("Usage: python script.py <HDF5ファイルのパス>")
        sys.exit(1)

    filepath = sys.argv[1]

    T, Y, source, created_at = load_scope_data(filepath)

    print("source:", source)
    print("created_at:", created_at)

    plot_scope_data(T, Y, title=f"{source} ({created_at})")
