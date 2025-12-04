#!/usr/bin/env python3

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import subprocess
import os
import io

import h5py
import numpy as np
from PIL import Image, ImageTk

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# --- OneDrive 設定（rclone） ---
# 事前に `rclone config` で "onedrive" というリモートを作成しておくこと
ONEDRIVE_REMOTE = "onedrive"      # rclone のリモート名
ONEDRIVE_DIR = "waveforms"        # OneDrive 内のアップロード先フォルダ（空文字ならルート）

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# HDF5 ビュー用の状態
hdf5_current_file = None          # 最後に開いた HDF5 ファイルパス
hdf5_tree_path_map = {}           # Treeview の item_id -> HDF5 パス の対応表


# ----------------------------------------------------------------------
# OneDrive アップロード
# ----------------------------------------------------------------------
def confirm_upload(file_list):
    """OneDrive (rclone) にアップロードするか確認して、必要ならアップロード"""
    if not file_list:
        return

    answer = messagebox.askyesno("送信確認", "OneDrive にアップロードしますか？")
    if not answer:
        return

    dest_base = f"{ONEDRIVE_REMOTE}:{ONEDRIVE_DIR}" if ONEDRIVE_DIR else f"{ONEDRIVE_REMOTE}:"

    errors = []
    for file_path in file_list:
        if not os.path.exists(file_path):
            errors.append(f"存在しないファイル: {file_path}")
            continue
        try:
            subprocess.run(
                ["rclone", "copy", file_path, dest_base],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
        except subprocess.CalledProcessError as e:
            errors.append(f"{file_path} のアップロード失敗: {e.stderr.strip()}")

    if errors:
        messagebox.showerror("アップロードエラー", "\n".join(errors))
    else:
        messagebox.showinfo("アップロード完了", "すべてのファイルを OneDrive にアップロードしました。")


# ----------------------------------------------------------------------
# 周波数入力有効/無効
# ----------------------------------------------------------------------
def toggle_freq_inputs():
    """周波数チェックの ON/OFF で入力欄の有効・無効を切り替える"""
    state = "normal" if freq_enable_var.get() else "disabled"
    entry_start.configure(state=state)
    entry_end.configure(state=state)
    entry_step.configure(state=state)


# ----------------------------------------------------------------------
# GPIB/VISA 機器スキャン用ポップアップ
# ----------------------------------------------------------------------
def open_gpib_popup():
    try:
        import pyvisa
    except ImportError:
        messagebox.showerror("VISAエラー", "pyvisa がインストールされていません。\n\npip install pyvisa")
        return

    popup = tk.Toplevel(root)
    popup.title("GPIB/VISA 機器選択")

    tk.Label(popup, text="検出した機器一覧（*IDN?）").grid(row=0, column=0, columnspan=2, sticky="w", padx=5, pady=5)

    list_inst1 = tk.Listbox(popup, width=80, height=6, exportselection=False)
    list_inst2 = tk.Listbox(popup, width=80, height=6, exportselection=False)

    tk.Label(popup, text="オシロスコープとして選択（inst1）").grid(row=1, column=0, columnspan=2, sticky="w", padx=5)
    list_inst1.grid(row=2, column=0, columnspan=2, padx=5, pady=2)

    tk.Label(popup, text="発振器として選択（inst2）").grid(row=3, column=0, columnspan=2, sticky="w", padx=5)
    list_inst2.grid(row=4, column=0, columnspan=2, padx=5, pady=2)

    instruments = []  # [(addr, idn_str), ...]

    try:
        rm = pyvisa.ResourceManager()
        resources = rm.list_resources()
        # ASRL (シリアル) は除外しておく
        resources = [r for r in resources if not r.startswith("ASRL")]

        if not resources:
            messagebox.showinfo("VISA", "接続されている機器が見つかりませんでした。")

        for addr in resources:
            desc = ""
            try:
                with rm.open_resource(addr) as inst:
                    inst.timeout = 1000  # ms
                    try:
                        idn = inst.query("*IDN?")
                        desc = idn.strip()
                    except Exception:
                        desc = "応答なし (*IDN?)"
            except Exception as e:
                desc = f"接続エラー: {e.__class__.__name__}"
            instruments.append((addr, desc))

        for addr, desc in instruments:
            text = f"{addr}  |  {desc}"
            list_inst1.insert(tk.END, text)
            list_inst2.insert(tk.END, text)

    except Exception as e:
        messagebox.showerror("VISAエラー", f"機器スキャンに失敗しました:\n{e}")
        popup.destroy()
        return

    def on_ok():
        # inst1 / inst2 はどちらかだけ選ばれていてもOK
        sel1 = list_inst1.curselection()
        sel2 = list_inst2.curselection()

        if sel1:
            inst1_addr_var.set(instruments[sel1[0]][0])
        else:
            inst1_addr_var.set("")

        if sel2:
            inst2_addr_var.set(instruments[sel2[0]][0])
        else:
            inst2_addr_var.set("")

        popup.destroy()

    def on_cancel():
        popup.destroy()

    btn_frame = tk.Frame(popup)
    btn_frame.grid(row=5, column=0, columnspan=2, pady=5)
    tk.Button(btn_frame, text="決定", command=on_ok).pack(side="left", padx=5)
    tk.Button(btn_frame, text="キャンセル", command=on_cancel).pack(side="left", padx=5)

    popup.grab_set()  # モーダルっぽくする


# ----------------------------------------------------------------------
# HDF5 ツリー構築・選択・プロット・保存
# ----------------------------------------------------------------------
def build_hdf5_tree(tree, filepath):
    """HDF5ファイルを開いて、Treeview に構造を全部突っ込む"""
    global hdf5_tree_path_map
    hdf5_tree_path_map.clear()
    tree.delete(*tree.get_children())

    with h5py.File(filepath, "r") as f:
        # ルートノード
        root_id = tree.insert("", "end", text="/", open=True)
        hdf5_tree_path_map[root_id] = "/"

        def add_node(parent_id, parent_path, obj):
            # Group のとき、中身をアルファベット順で並べる
            if isinstance(obj, h5py.Group):
                for name, child in sorted(obj.items(), key=lambda kv: kv[0]):
                    if parent_path == "/":
                        child_path = "/" + name
                    else:
                        child_path = parent_path.rstrip("/") + "/" + name
                    label = name
                    if isinstance(child, h5py.Dataset):
                        label += " [D]"
                    elif isinstance(child, h5py.Group):
                        label += " [G]"
                    item_id = tree.insert(parent_id, "end", text=label, open=False)
                    hdf5_tree_path_map[item_id] = child_path
                    if isinstance(child, h5py.Group):
                        add_node(item_id, child_path, child)

        add_node(root_id, "/", f)


def on_hdf5_tree_select(event):
    """ツリーで選択されたノードに応じて右側に情報を表示"""
    global hdf5_current_file, hdf5_tree_path_map

    if hdf5_current_file is None:
        return

    tree = event.widget
    sel = tree.selection()
    if not sel:
        return
    item_id = sel[0]
    path = hdf5_tree_path_map.get(item_id)
    if path is None:
        return

    hdf5_text.delete("1.0", tk.END)
    image_label.configure(image=None)
    image_label.image = None
    csv_text.delete("1.0", tk.END)

    try:
        with h5py.File(hdf5_current_file, "r") as f:
            obj = f[path]

            # Group の場合
            if isinstance(obj, h5py.Group):
                hdf5_text.insert(tk.END, f"[Group] {path}\n\n")
                if obj.attrs:
                    hdf5_text.insert(tk.END, "@attrs:\n")
                    for k, v in obj.attrs.items():
                        hdf5_text.insert(tk.END, f"  {k} = {repr(v)}\n")
                else:
                    hdf5_text.insert(tk.END, "(属性なし)\n")
                return

            # Dataset の場合
            if isinstance(obj, h5py.Dataset):
                hdf5_text.insert(
                    tk.END,
                    f"[Dataset] {path}\nshape={obj.shape}, dtype={obj.dtype}\n\n"
                )
                # attrs
                if obj.attrs:
                    hdf5_text.insert(tk.END, "@attrs:\n")
                    for k, v in obj.attrs.items():
                        hdf5_text.insert(tk.END, f"  {k} = {repr(v)}\n")
                    hdf5_text.insert(tk.END, "\n")

                # データプレビュー & 画像 / テキスト推定
                data = obj[()]

                # バイト列（画像 or テキストの可能性）
                if isinstance(data, (bytes, bytearray)) or (
                    isinstance(data, np.ndarray) and data.dtype == np.uint8
                ):
                    # 画像として試す
                    raw_bytes = bytes(data) if isinstance(data, np.ndarray) else data
                    try:
                        img = Image.open(io.BytesIO(raw_bytes))
                        img = img.resize((300, 200))
                        photo = ImageTk.PhotoImage(img)
                        image_label.configure(image=photo)
                        image_label.image = photo
                        hdf5_text.insert(tk.END, "推定: 画像データ\n")
                    except Exception:
                        # テキストとして試す
                        try:
                            txt = raw_bytes.decode("utf-8", errors="replace")
                            preview = txt[:500]
                            if len(txt) > 500:
                                preview += "...(省略)"
                            csv_text.insert(tk.END, preview)
                            hdf5_text.insert(tk.END, "推定: テキスト/CSV データ\n")
                        except Exception:
                            hdf5_text.insert(tk.END, "バイナリデータ（画像・テキスト判別不可）\n")
                    return

                # 数値配列など
                arr = np.array(data)
                flat = arr.ravel()
                n_all = flat.size
                n_show = min(n_all, 50)
                preview = flat[:n_show]
                hdf5_text.insert(
                    tk.END,
                    f"data preview ({n_show}/{n_all} elements):\n{preview}\n"
                )

    except Exception as e:
        messagebox.showerror("HDF5エラー", str(e))


def plot_dataset_in_window(path):
    """選択された HDF5 の Dataset を別ウィンドウでプロット"""
    global hdf5_current_file

    if hdf5_current_file is None:
        messagebox.showerror("プロットエラー", "HDF5ファイルが開かれていません。")
        return

    try:
        with h5py.File(hdf5_current_file, "r") as f:
            obj = f[path]
            if not isinstance(obj, h5py.Dataset):
                messagebox.showerror("プロットエラー", "選択されたノードは Dataset ではありません。")
                return

            data = np.array(obj[()])  # numpy配列化
    except Exception as e:
        messagebox.showerror("プロットエラー", f"データ読み込みに失敗しました:\n{e}")
        return

    # データの形に応じて描画方針を決める
    if data.ndim == 0:
        messagebox.showinfo("プロット", f"スカラー値です: {data}")
        return
    elif data.ndim == 1:
        x = np.arange(data.size)
        y = data
        plot_type = "1D"
    elif data.ndim == 2:
        plot_type = "2D"
    else:
        messagebox.showerror("プロットエラー", f"{data.ndim}次元配列は簡易プロット対象外です。")
        return

    # Tk の別ウィンドウを開いて Matplotlib を埋め込む
    win = tk.Toplevel(root)
    win.title(f"Plot: {path}")

    fig = Figure(figsize=(5, 4))
    ax = fig.add_subplot(111)

    if plot_type == "1D":
        ax.plot(x, y)
        ax.set_xlabel("Index")
        ax.set_ylabel("Value")
    elif plot_type == "2D":
        im = ax.imshow(data, aspect="auto")
        ax.set_xlabel("Index (axis 1)")
        ax.set_ylabel("Index (axis 0)")
        fig.colorbar(im, ax=ax)

    ax.set_title(path)

    canvas = FigureCanvasTkAgg(fig, master=win)
    canvas.draw()
    canvas.get_tk_widget().pack(fill="both", expand=True)


def plot_selected_dataset():
    """Treeviewで選択されているノードを Dataset とみなしてプロット"""
    global hdf5_tree_path_map

    sel = hdf5_tree.selection()
    if not sel:
        messagebox.showerror("プロットエラー", "ノードが選択されていません。")
        return

    item_id = sel[0]
    path = hdf5_tree_path_map.get(item_id)
    if path is None:
        messagebox.showerror("プロットエラー", "内部パス情報が見つかりません。")
        return

    plot_dataset_in_window(path)


def plot_first_dataset_from_file(filepath: str):
    """指定された HDF5 ファイルの中で、最初に見つかった Dataset をプロット"""
    global hdf5_current_file

    if not os.path.exists(filepath):
        messagebox.showerror("グラフ表示エラー", f"HDF5ファイルが見つかりません:\n{filepath}")
        return

    hdf5_current_file = filepath
    first_path = None

    try:
        with h5py.File(filepath, "r") as f:
            def visit(name, obj):
                nonlocal first_path
                if first_path is None and isinstance(obj, h5py.Dataset):
                    first_path = "/" + name if not name.startswith("/") else name

            f.visititems(visit)

        if first_path is None:
            messagebox.showinfo("グラフ表示", "Dataset が1つも見つかりませんでした。")
            return

        plot_dataset_in_window(first_path)

    except Exception as e:
        messagebox.showerror("グラフ表示エラー", f"HDF5 読み込みに失敗しました:\n{e}")


def save_image_from_dataset():
    """選択中の Dataset を画像として保存（PNG想定）"""
    global hdf5_current_file, hdf5_tree_path_map

    if hdf5_current_file is None:
        messagebox.showerror("保存エラー", "HDF5ファイルが開かれていません。")
        return

    sel = hdf5_tree.selection()
    if not sel:
        messagebox.showerror("保存エラー", "ツリーから Dataset を選択してください。")
        return

    item_id = sel[0]
    path = hdf5_tree_path_map.get(item_id)
    if path is None:
        messagebox.showerror("保存エラー", "内部パス情報が見つかりません。")
        return

    try:
        with h5py.File(hdf5_current_file, "r") as f:
            obj = f[path]
            data = obj[()]
    except Exception as e:
        messagebox.showerror("保存エラー", f"HDF5 読み込みに失敗しました:\n{e}")
        return

    # バイト列 or uint8配列として扱って画像判定
    raw_bytes = None
    if isinstance(data, (bytes, bytearray)):
        raw_bytes = data
    elif isinstance(data, np.ndarray) and data.dtype == np.uint8:
        raw_bytes = bytes(data)

    if raw_bytes is None:
        messagebox.showerror("保存エラー", "この Dataset は画像形式のバイナリではなさそうです。")
        return

    try:
        img = Image.open(io.BytesIO(raw_bytes))
    except Exception as e:
        messagebox.showerror("保存エラー", f"画像として解釈できませんでした:\n{e}")
        return

    # デフォルトファイル名
    base = os.path.splitext(os.path.basename(hdf5_current_file))[0]
    dname = path.replace("/", "_").lstrip("_") or "dataset"
    default_name = f"{base}_{dname}.png"

    save_path = filedialog.asksaveasfilename(
        title="画像を保存",
        defaultextension=".png",
        initialfile=default_name,
        filetypes=[("PNG images", "*.png"), ("All files", "*.*")]
    )
    if not save_path:
        return

    try:
        img.save(save_path, format="PNG")
        messagebox.showinfo("保存完了", f"画像を保存しました:\n{save_path}")
    except Exception as e:
        messagebox.showerror("保存エラー", f"保存に失敗しました:\n{e}")


def save_csv_from_dataset():
    """選択中の Dataset を CSV 形式テキストとして保存"""
    global hdf5_current_file, hdf5_tree_path_map

    if hdf5_current_file is None:
        messagebox.showerror("保存エラー", "HDF5ファイルが開かれていません。")
        return

    sel = hdf5_tree.selection()
    if not sel:
        messagebox.showerror("保存エラー", "ツリーから Dataset を選択してください。")
        return

    item_id = sel[0]
    path = hdf5_tree_path_map.get(item_id)
    if path is None:
        messagebox.showerror("保存エラー", "内部パス情報が見つかりません。")
        return

    try:
        with h5py.File(hdf5_current_file, "r") as f:
            obj = f[path]
            data = obj[()]
    except Exception as e:
        messagebox.showerror("保存エラー", f"HDF5 読み込みに失敗しました:\n{e}")
        return

    # デフォルトファイル名
    base = os.path.splitext(os.path.basename(hdf5_current_file))[0]
    dname = path.replace("/", "_").lstrip("_") or "dataset"
    default_name = f"{base}_{dname}.csv"

    save_path = filedialog.asksaveasfilename(
        title="CSVを保存",
        defaultextension=".csv",
        initialfile=default_name,
        filetypes=[("CSV files", "*.csv"), ("Text files", "*.txt"), ("All files", "*.*")]
    )
    if not save_path:
        return

    try:
        # バイト列ならテキストとして保存
        if isinstance(data, (bytes, bytearray)):
            txt = data.decode("utf-8", errors="replace")
            with open(save_path, "w", encoding="utf-8") as f:
                f.write(txt)
        else:
            arr = np.array(data)
            # 1次元なら列ベクトルに
            if arr.ndim == 1:
                arr = arr.reshape(-1, 1)
            # 数値以外が混じっていると savetxt がコケるので文字列化
            if not np.issubdtype(arr.dtype, np.number):
                arr_str = arr.astype(str)
                np.savetxt(save_path, arr_str, delimiter=",", fmt="%s")
            else:
                np.savetxt(save_path, arr, delimiter=",")
        messagebox.showinfo("保存完了", f"CSVを保存しました:\n{save_path}")
    except Exception as e:
        messagebox.showerror("保存エラー", f"保存に失敗しました:\n{e}")


def open_hdf5():
    """HDF5ファイルを選択してツリーに表示"""
    global hdf5_current_file
    filepath = filedialog.askopenfilename(filetypes=[("HDF5 files", "*.h5 *.hdf5")])
    if not filepath:
        return

    hdf5_current_file = filepath

    try:
        build_hdf5_tree(hdf5_tree, filepath)
    except Exception as e:
        messagebox.showerror("HDF5読み込みエラー", str(e))
        return

    # 先頭ノードを選択して情報表示
    root_items = hdf5_tree.get_children()
    if root_items:
        hdf5_tree.selection_set(root_items[0])
        hdf5_tree.event_generate("<<TreeviewSelect>>")


# ----------------------------------------------------------------------
# 測定実行＋HDF5/CSV/PNG 出力＋グラフ表示
# ----------------------------------------------------------------------
def run_script():
    """周波数設定＋計測スクリプト実行＋HDF5/CSV/PNG 出力＋グラフ表示"""
    # 子プロセスに渡す環境変数を準備
    env = os.environ.copy()

    inst1 = inst1_addr_var.get().strip()
    inst2 = inst2_addr_var.get().strip()
    if inst1:
        env["INST1_ADDRESS"] = inst1
    if inst2:
        env["INST2_ADDRESS"] = inst2

    # --- 周波数設定（必要なときだけ） ---
    freq_args = []
    if freq_enable_var.get():
        try:
            start = float(entry_start.get())
            end   = float(entry_end.get())
            step  = float(entry_step.get())
            freq_args = [str(start), str(end), str(step)]
            subprocess.run(
                ["pkexec", "python3", "set_frequency.py"] + freq_args,
                check=True,
                env=env
            )
        except ValueError:
            messagebox.showerror("入力エラー", "周波数を正しく入力してください")
            return
        except Exception as e:
            messagebox.showerror("周波数設定エラー", str(e))
            return

    file_paths = []
    errors = []

    # --- ① HDF5 は常に出力する ---
    h5_path = None
    try:
        graph_args = [
            "--type",  graph_type_var.get(),
            "--title", plot_title_var.get(),
            "--xlabel", plot_xlabel_var.get(),
            "--ylabel", plot_ylabel_var.get(),
        ]
        script_path = os.path.join(BASE_DIR, "hdf5_plot.py")
        completed = subprocess.run(
            ["pkexec", "python3", script_path] + freq_args + graph_args,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env
        )
        h5_path = completed.stdout.strip()

        if not h5_path:
            # stdout に何も返していない場合は、固定パスを仮定
            h5_path = os.path.join(BASE_DIR, "output", "output.h5")

        file_paths.append(h5_path)

    except subprocess.CalledProcessError as e:
        errors.append(f"H5出力失敗: {e.stderr}")
    except Exception as e:
        errors.append(f"H5出力失敗: {e}")

    # H5 が作れていないなら終了
    if not h5_path or not os.path.exists(h5_path):
        report = "出力確認:\n"
        for path in file_paths:
            report += f"{'✅' if os.path.exists(path) else '❌'} {path}\n"
        if errors:
            report += "\n⚠ エラー:\n" + "\n".join(errors)
        messagebox.showerror("エラー", report)
        return

    # --- ② CSV の出力（チェック ON のときだけ） ---
    if csv_var.get():
        try:
            csv_script = os.path.join(BASE_DIR, "csv.py")
            completed = subprocess.run(
                ["python3", csv_script, "--input", h5_path],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            csv_path = completed.stdout.strip()
            if not csv_path:
                csv_path = os.path.join(BASE_DIR, "output", "output.csv")
            file_paths.append(csv_path)
        except subprocess.CalledProcessError as e:
            errors.append(f"CSV出力失敗: {e.stderr}")
        except Exception as e:
            errors.append(f"CSV出力失敗: {e}")

    # --- ③ PNG の出力（チェック ON のときだけ） ---
    if png_var.get():
        try:
            plot_script = os.path.join(BASE_DIR, "plot.py")
            completed = subprocess.run(
                ["python3", plot_script, "--input", h5_path, "--save"],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            img_path = completed.stdout.strip()
            if not img_path:
                img_path = os.path.join(BASE_DIR, "output", "output.png")
            file_paths.append(img_path)
        except subprocess.CalledProcessError as e:
            errors.append(f"PNG出力失敗: {e.stderr}")
        except Exception as e:
            errors.append(f"PNG出力失敗: {e}")

    # --- ④ 測定後にグラフ表示（ラジオボタンの設定による） ---
    if plot_mode_var.get() == "show":
        try:
            plot_first_dataset_from_file(h5_path)
        except Exception as e:
            errors.append(f"グラフ表示失敗: {e}")

    # --- ⑤ 出力確認ダイアログ ---
    report = "出力確認:\n"
    for path in file_paths:
        report += f"{'✅' if os.path.exists(path) else '❌'} {path}\n"
    if errors:
        report += "\n⚠ エラー:\n" + "\n".join(errors)

    messagebox.showinfo("結果", report)

    # OneDrive へアップロードするか確認
    confirm_upload(file_paths)


def on_exit():
    """ウィンドウを閉じるときに OneDrive アップロード確認を行う"""
    files = []

    # デフォルトの出力パスに合わせる（必要に応じて変更）
    if csv_var.get():
        files.append(os.path.join(BASE_DIR, "output", "output.csv"))
    if png_var.get():
        files.append(os.path.join(BASE_DIR, "output", "output.png"))

    # HDF5 は常に出力される前提
    files.append(os.path.join(BASE_DIR, "output", "output.h5"))

    confirm_upload(files)
    root.destroy()


# ----------------------------------------------------------------------
# GUI 本体
# ----------------------------------------------------------------------
root = tk.Tk()
root.title("波形保存 & HDF5ビューア")

# inst1 / inst2 の VISA アドレス（inst1=オシロ, inst2=発振器）
inst1_addr_var = tk.StringVar(value="")
inst2_addr_var = tk.StringVar(value="")

notebook = ttk.Notebook(root)
frame_main = ttk.Frame(notebook, padding=10)
frame_hdf5 = ttk.Frame(notebook, padding=10)
notebook.add(frame_main, text="出力・設定")
notebook.add(frame_hdf5, text="HDF5読み込み")
notebook.pack(expand=1, fill="both")

# --- タブ1 レイアウト調整 ---
row = 0
freq_enable_var = tk.BooleanVar()
tk.Checkbutton(
    frame_main,
    text="周波数を設定する",
    variable=freq_enable_var,
    command=toggle_freq_inputs
).grid(row=row, column=0, columnspan=2, sticky="w")
row += 1

tk.Label(frame_main, text="開始周波数（Hz）").grid(row=row, column=0, sticky="e")
entry_start = tk.Entry(frame_main, state="disabled")
entry_start.grid(row=row, column=1, sticky="w")
row += 1

tk.Label(frame_main, text="終了周波数（Hz）").grid(row=row, column=0, sticky="e")
entry_end = tk.Entry(frame_main, state="disabled")
entry_end.grid(row=row, column=1, sticky="w")
row += 1

tk.Label(frame_main, text="間隔（Hz）").grid(row=row, column=0, sticky="e")
entry_step = tk.Entry(frame_main, state="disabled")
entry_step.grid(row=row, column=1, sticky="w")
row += 1

# 出力ファイル形式（HDF5 は常に出力 / CSV・PNG はオプション）
csv_var = tk.BooleanVar(value=False)
png_var = tk.BooleanVar(value=False)

tk.Label(frame_main, text="追加で保存する形式:").grid(row=row, column=0, sticky="w")
row += 1
tk.Checkbutton(frame_main, text="CSV", variable=csv_var).grid(row=row, column=0, sticky="w")
tk.Checkbutton(frame_main, text="PNG", variable=png_var).grid(row=row, column=1, sticky="w")
row += 1

tk.Label(frame_main, text="※ HDF5 は常に出力されます").grid(row=row, column=0, columnspan=2, sticky="w")
row += 1

# グラフ種類選択（XY or YT）
graph_type_var = tk.StringVar(value="YT")
tk.Label(frame_main, text="グラフ種類:").grid(row=row, column=0, sticky="e")
tk.OptionMenu(frame_main, graph_type_var, "YT", "XY").grid(row=row, column=1, sticky="w")
row += 1

# グラフタイトル・軸ラベル
plot_title_var = tk.StringVar(value="")
plot_xlabel_var = tk.StringVar(value="Time")
plot_ylabel_var = tk.StringVar(value="Voltage")

tk.Label(frame_main, text="タイトル:").grid(row=row, column=0, sticky="e")
tk.Entry(frame_main, textvariable=plot_title_var, width=25).grid(row=row, column=1, sticky="w")
row += 1

tk.Label(frame_main, text="X軸ラベル:").grid(row=row, column=0, sticky="e")
tk.Entry(frame_main, textvariable=plot_xlabel_var, width=25).grid(row=row, column=1, sticky="w")
row += 1

tk.Label(frame_main, text="Y軸ラベル:").grid(row=row, column=0, sticky="e")
tk.Entry(frame_main, textvariable=plot_ylabel_var, width=25).grid(row=row, column=1, sticky="w")
row += 1

# グラフ表示モード（ラジオボタン）
plot_mode_var = tk.StringVar(value="none")  # "none" or "show"
tk.Label(frame_main, text="グラフ表示:").grid(row=row, column=0, sticky="e")
tk.Radiobutton(
    frame_main, text="表示しない",
    variable=plot_mode_var, value="none"
).grid(row=row, column=1, sticky="w")
row += 1

tk.Radiobutton(
    frame_main, text="測定後に表示する",
    variable=plot_mode_var, value="show"
).grid(row=row, column=1, sticky="w")
row += 1

# inst1 / inst2 表示
tk.Label(frame_main, text="オシロ (inst1):").grid(row=row, column=0, sticky="e")
tk.Label(frame_main, textvariable=inst1_addr_var, width=25, anchor="w").grid(row=row, column=1, sticky="w")
row += 1

tk.Label(frame_main, text="発振器 (inst2):").grid(row=row, column=0, sticky="e")
tk.Label(frame_main, textvariable=inst2_addr_var, width=25, anchor="w").grid(row=row, column=1, sticky="w")
row += 1

tk.Button(frame_main, text="GPIB 機器スキャン", command=open_gpib_popup)\
    .grid(row=row, column=0, columnspan=2, pady=5)
row += 1

# 実行ボタン
run_btn = tk.Button(frame_main, text="スクリプトを実行", command=run_script)
run_btn.grid(row=row, column=0, columnspan=2, pady=10)
row += 1

# --- タブ2: HDF5 ビューア ---
top_frame = ttk.Frame(frame_hdf5)
top_frame.pack(fill="x", pady=5)
tk.Button(top_frame, text="HDF5ファイルを開く", command=open_hdf5).pack(side="left")

hdf5_main_pane = ttk.PanedWindow(frame_hdf5, orient="horizontal")
hdf5_main_pane.pack(fill="both", expand=True)

# 左：ツリー
hdf5_tree_frame = ttk.Frame(hdf5_main_pane)
hdf5_tree = ttk.Treeview(hdf5_tree_frame)
hdf5_tree.pack(fill="both", expand=True)
hdf5_main_pane.add(hdf5_tree_frame, weight=1)

# 右：情報＋プレビュー＋保存ボタン
right_frame = ttk.Frame(hdf5_main_pane)
hdf5_main_pane.add(right_frame, weight=3)

tk.Label(right_frame, text="ノード情報").pack(anchor="w")
hdf5_text = tk.Text(right_frame, height=10)
hdf5_text.pack(fill="both", expand=False, pady=5)

image_label = tk.Label(right_frame)
image_label.pack()

csv_text = tk.Text(right_frame, height=10)
csv_text.pack(fill="both", expand=True, pady=5)

btn_frame = tk.Frame(right_frame)
btn_frame.pack(fill="x", pady=5)
tk.Button(btn_frame, text="画像を保存する", command=save_image_from_dataset).pack(side="left", padx=2)
tk.Button(btn_frame, text="CSVを保存する", command=save_csv_from_dataset).pack(side="left", padx=2)

# コンテキストメニュー（右クリック用）
hdf5_tree_menu = tk.Menu(hdf5_tree, tearoff=0)
hdf5_tree_menu.add_command(label="このDatasetをプロット", command=plot_selected_dataset)


def on_hdf5_tree_right_click(event):
    # 右クリックした行を選択状態にしてメニュー表示
    iid = hdf5_tree.identify_row(event.y)
    if iid:
        hdf5_tree.selection_set(iid)
        hdf5_tree_menu.tk_popup(event.x_root, event.y_root)
        


hdf5_tree.bind("<Button-3>", on_hdf5_tree_right_click)       # 右クリック
hdf5_tree.bind("<<TreeviewSelect>>", on_hdf5_tree_select)    # 選択時イベント

root.protocol("WM_DELETE_WINDOW", on_exit)
root.mainloop()
