#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
"""
@Time ： 2021/9/15 12:06
@Auth ：Xxxyyyzin
@File ：main_SeismicTrans.py
@IDE ：PyCharm
"""

import socket
import torch.nn.functional as F
from SeimiscTransformer import SeismicTrans_base_patch16_128
import losses
from datasets import SeismicBinaryDBDataset
import sys

sys.path.insert(0, '../shared_modules/')
from constantsBase import ConstantsBase


class Constants(ConstantsBase):

    def __init__(self, **kwargs):
        "Define default parameters"

        ######################################
        ##### GLOBAL CONSTANTS FOR MODEL
        ######################################

        self.RUN = "fault_SeismicTrans_Adam_l2_mean_loss_gain"
        #"marmousi_SeismicTrans_Adam_l1_mean_loss_gain"

        self.DATA ="fault_2ms_r.bin"
        #"marmousi_2ms.bin"

        # GPU parameters
        self.DEVICE = 0  # cuda device

        # Model parameters
        # self.MODEL = xyj_model.XYJ_model
        self.MODEL =SeismicTrans_base_patch16_128
        self.MODEL_NAME="SeismicTransformer"

        self.MODEL_LOAD_PATH = None
        # self.MODEL_LOAD_PATH = "server/models/layers_new_lr1e4_b100_constant8_vvdeep_r_l1/model_03000000.torch"

        self.DROPOUT_RATE = 0.0  # probability to drop
        self.ACTIVATION = F.relu

        # Optimisation parameters
        self.OPTIMIZER = "Adam"  # SGD Adam RMSprop Adadelta Adagrad Adamax
        self.LOSS_NAME="l1_mean_loss_gain"  #l1_mean_loss l2_mean_loss l1_half_loss  l2_mean_loss_gain
        self.LOSS_FUNC= losses.l1_mean_loss_gain
        self.BATCH_SIZE = 2
        self.LRATE = 1e-4
        self.WEIGHT_DECAY = 0  # L2 weight decay parameter

        # seed
        self.SEED = 123

        # training length
        self.N_STEPS =100

        # CPU parameters
        self.N_CPU_WORKERS = 1  # number of multiprocessing workers for DataLoader

        self.DATASET = SeismicBinaryDBDataset

        # input dataset properties
        self.N_EXAMPLES =30
        self.VELOCITY_SHAPE = (1, 128, 128)  # 1, NX, NZ
        self.GATHER_SHAPE = (1, 32, 512)  # 1, NREC, NSTEPS
        self.SOURCE_SHAPE = (2, 1, 1)  # 2, 1, 1

        # pre-processing
        self.T_GAIN = 2.5  # gain on gather
        self.VELOCITY_MU = 2700.0  # m/s , for normalising the velocity models in pre-processing
        self.VELOCITY_SIGMA = 560.0  # m/s , for normalising the velocity models in pre-processing
        self.GATHER_MU = 0.
        self.GATHER_SIGMA = 1.0

        ## 3. SUMMARY OUTPUT FREQUENCIES
        self.SUMMARY_FREQ = 100  # how often to save the summaries, in # steps
        self.TEST_FREQ = 100  # how often to test the model on test data, in # steps
        self.MODEL_SAVE_FREQ =20  # how often to save the model, in # steps


        # overwrite with input arguments
        for key in kwargs.keys(): self[key] = kwargs[key]

        self.SUMMARY_OUT_DIR = "results/summaries/%s/%s/" % (self.OPTIMIZER, self.RUN)
        self.MODEL_OUT_DIR = "results/models/%s/%s/" % (self.OPTIMIZER, self.RUN)

        self.DATA_PATH = "../generate_data/data/" + self.DATA
