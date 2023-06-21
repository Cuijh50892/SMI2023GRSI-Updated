#!/usr/bin/env python
# # -*- coding: utf-8 -*-
import os
import math
import json
import argparse
import numpy as np
import glob2
from tqdm import tqdm
import csv
import openpyxl
import xml.dom.minidom as minidom

file_num = 800
mat = np.zeros((1,16))

for idx in range(file_num):
    file_name = 'vector' + str(idx) + '.csv'
    a = np.loadtxt(file_name, delimiter=",")
    a = np.atleast_2d(a)
    mat = np.concatenate((mat, a), axis=0)

np.savetxt('codesX.csv', mat, fmt='%0.8f', delimiter = ',')

















