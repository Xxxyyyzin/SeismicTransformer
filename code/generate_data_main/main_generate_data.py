# -*- coding: utf-8 -*-
"""
@Time ： 2021/8/25 11:36
@Auth ： yyyzzzin
@File ：main_SeismicTrans.py
@IDE ：PyCharm
"""
import os
import sys
import time
import numpy as np

from generate_velocity import main
from generate_simulation import generate_simulation
from convert_to_binary import convert_to_binary_func

sys.path.insert(0, '../shared_modules/')
import io_utils


N_VELOCITY_EXAMPLES=100000  #
EACH_EPOCH=200
EPOCH=N_VELOCITY_EXAMPLES//EACH_EPOCH



ROOT_DIR = ""
VELOCITY_DIR = ROOT_DIR + "velocity/fault/"
GATHER_DIR = ROOT_DIR + "gather/fault_2ms_r/"
DATA_PATH = ROOT_DIR + "data/fault_2ms_r.bin"


def generate_main(EPOCH):
    for i in range(EPOCH):
        print("%s / %s"%(i+1,EPOCH))
        main()
        print("%s / %s :Velocity models have been generated!" % (i + 1, EPOCH))
        generate_simulation()
        print("%s / %s :Simulations are finished!" % (i + 1, EPOCH))
        if i == 0:
            convert_to_binary_func(True)
        else:
            convert_to_binary_func(False)
        print("%s / %s :Datas have been converted to the binary!" % (i + 1, EPOCH))
        io_utils.clear_dir(VELOCITY_DIR)
        io_utils.clear_dir(GATHER_DIR)
        io_utils.remove_dir(VELOCITY_DIR)
        io_utils.remove_dir(GATHER_DIR)
        print("%s / %s :Related directories have been cleaned !" % (i + 1, EPOCH))
        print("%s th is finished,data size is %s" % (i + 1,os.path.getsize(DATA_PATH)))
        print("========================================")

generate_main(EPOCH)
