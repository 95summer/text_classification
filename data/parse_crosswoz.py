# -*- coding: utf-8 -*-
"""把 CrossWOZ 原始对话解析成意图分类数据。

输入（CrossWOZ 官方划分，原样使用，不自己切分）：
    <crosswoz>/train.json  val.json  test.json

输出：
    data/processed/train.txt / dev.txt / test.txt   → "用户原话<TAB>领域编号"

标签来源是每条消息自带的 dialog_act（对话行为标注）第 2 位，不是猜的。
领域只取 5 类：酒店 / 餐馆 / 景点 / 地铁 / 出租。
"""

import argparse
import json
import os
import re
from collections import Counter

DOMAINS = ["酒店", "餐馆", "景点", "地铁", "出租"]
LABEL = {d: i for i, d in enumerate(DOMAINS)}

SPLITS = [("train", "train.json"), ("dev", "val.json"), ("test", "test.json")]

# 一句话里出现制表符或换行会让下游按行读取的文件错位（表现为 label 变 NaN 或 split 报错）
_WS = re.compile(r"[\t\r\n]+")


def clean(text):
    """把制表符、换行压成空格，并去掉首尾空白。"""
    return _WS.sub(" ", (text or "")).strip()


def extract(path):
    """返回 (样本列表, 丢弃数)。"""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    samples, dropped = [], 0
    for dial in data.values():
        for m in dial.get("messages", []):
            if m.get("role") != "usr":
                continue
            text = clean(m.get("content"))
            if not text:
                continue
            # 从对话行为里取领域；General 行为没有领域，跳过
            doms = {a[1] for a in m.get("dialog_act", [])
                    if len(a) >= 4 and a[1] in LABEL}
            if len(doms) != 1:
                dropped += 1
                continue
            samples.append((text, LABEL[doms.pop()]))
    return samples, dropped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--crosswoz", default="raw/crosswoz", help="CrossWOZ json 所在目录")
    ap.add_argument("--out", default="processed", help="输出目录")
    ap.add_argument("--dry-run", action="store_true", help="只统计，不写文件")
    args = ap.parse_args()

    for split, fname in SPLITS:
        path = os.path.join(args.crosswoz, fname)
        if not os.path.exists(path):
            print("缺少文件：%s" % path)
            return
        samples, dropped = extract(path)
        dist = Counter(y for _, y in samples)
        lens = [len(t) for t, _ in samples]
        print("[%s] 样本 %d 条，丢弃 %d 条（无领域或跨多个领域）" % (split, len(samples), dropped))
        print("      领域分布 %s" % {DOMAINS[k]: v for k, v in sorted(dist.items())})
        print("      平均句长 %.1f 字，最长 %d 字" % (sum(lens) / len(lens), max(lens)))
        if not args.dry_run:
            os.makedirs(args.out, exist_ok=True)
            out_path = os.path.join(args.out, "%s.txt" % split)
            with open(out_path, "w", encoding="utf-8") as f:
                for text, y in samples:
                    f.write("%s\t%d\n" % (text, y))
            print("      写入 %s" % out_path)


if __name__ == "__main__":
    main()
