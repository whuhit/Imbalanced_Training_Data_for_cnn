#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/2/24 12:42
# @Author  : yangqiang
import numpy as np


def print_res_acc():
    for i in range(12, 16):
        file = f"DLAAgent_diff_huge_{i}.log"
        best_acc = float(
            open(f"logs/{file}").read().splitlines()[-1].split("best acc: ")[-1])

        for line in open(f"logs/{file}"):
            if "acc of 10 classes" in line:
                acc = np.array(line.split(
                    "acc of 10 classes: [")[-1].rstrip(']\n').split()).astype('float')
                if round(acc.mean(), 4) == best_acc:
                    print(file, best_acc, acc)


def write_dist():
    from easyPlog import Plog
    nums_array = np.array([[500, 500, 3641, 3642, 3642, 3642, 3642, 3641, 5000, 5000],
                           [5000, 5000, 3641, 3642, 3642, 3642, 3642, 3641, 500, 500],
                           [50, 50, 3791, 3792, 3792, 3792, 3792, 3791, 5000, 5000],
                           [5000, 5000, 3791, 3792, 3792, 3792, 3792, 3791, 50, 50]])

    for i in range(12, 16):
        log = Plog(f"diff_huge/dist_{i}.txt", cover=True)
        count = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0}
        for line in open("/assets/data/dla_cifar/dist_default/train.txt"):
            label = int(line.split()[1])
            count[label] += 1
            if count[label] > nums_array[i - 12][label]:
                continue
            else:
                log.log(line.strip())

print_res_acc()