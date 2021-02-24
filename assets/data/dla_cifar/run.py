#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2021/2/24 12:42
# @Author  : yangqiang
import numpy as np

for i in range(1, 12):
    file = f"DLAAgent-dist-{i}.log"
    best_acc = float(open(f"logs/{file}").read().splitlines()[-1].split("best acc: ")[-1])

    for line in open(f"logs/{file}"):
        if "acc of 10 classes" in line:
            acc = np.array(line.split("acc of 10 classes: [")[-1].rstrip(']\n').split()).astype('float')
            if round(acc.mean(), 4) == best_acc:
                print(file, best_acc, acc)
