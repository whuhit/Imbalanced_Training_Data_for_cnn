#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2021/2/23 10:38
# @Author  : yangqiang
# @File    : dla_config.py
import argparse
import json


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--opt', type=str, default="configs/cifar_dla/dla.json",
                        help='parse options from json file')

    return parser.parse_args()


config = parse_args()


if config.opt:
    with open(config.opt, 'r') as f:
        a = json.load(f)

    for k, v in a.items():
        setattr(config, k, v)

