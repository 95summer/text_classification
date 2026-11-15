import csv
import jieba
import pandas as pd
import importlib.util
import pathlib

_cfg_path = pathlib.Path(__file__).resolve().parent.parent / "RF" / "config.py"
_spec = importlib.util.spec_from_file_location("rf_config", _cfg_path)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
Config = _mod.Config

config = Config()


def data_preprocessing(task_type, datapath):
    # 读取文件
    # quoting=csv.QUOTE_NONE：对话文本里含英文引号（"），pandas 默认会把引号当包裹符，
    # 导致一行被读成跨行、label 列变成 NaN，RandomForest 直接报 "Input y contains NaN"。
    df = pd.read_csv(datapath, sep="\t", encoding="UTF-8",
                     names=["text_classification", "label"],
                     quoting=csv.QUOTE_NONE)
    # 丢掉解析异常的行
    bad = df["label"].isna().sum()
    if bad:
        print(f"  [{task_type}] 丢弃 {bad} 行解析异常数据")
        df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)

    # 分词 只取前30个
    df["words"] = df["text_classification"].apply(lambda line: " ".join(jieba.lcut(str(line))[:30]))
    # 存储文件
    if task_type in datapath:
        df.to_csv(config.process_datapath.get(task_type), sep=",", header=True, index=False, mode="w", encoding="UTF-8")
        print(f"{task_type}: {len(df)} 条 -> {config.process_datapath.get(task_type)}")


if __name__ == "__main__":
    for task_type, datapath in config.original_datapath.items():
        data_preprocessing(task_type, datapath)
