# -*- coding: utf-8 -*-
"""
@Time ： 2021/9/15 12:06
@Auth ： Xxxyyzin
@File ：main_SeismicTrans.py
@IDE ：PyCharm
"""
import sys
import os
import time
import matplotlib


if 'linux' in sys.platform.lower(): matplotlib.use('Agg')  # use a non-interactive backend (ie plotting without windows)
import matplotlib.pyplot as plt

import numpy as np
from tensorboardX import SummaryWriter
import torch
import torch.optim
from torch.utils.data import RandomSampler, DataLoader
from torch_utils import get_weights, get_weights_update_percent

from mysampler import BatchSampler
from constants_SeismicTrans import Constants
import losses



## This needs to be specified - problem dependent
def plot_result(inputs_array, outputs_array, labels_array, sample_batch, ib=0, isource=0,
                aspect=0.2):
    "Plot a network prediction, compare to ground truth and input"
    f = plt.figure(figsize=(12, 5))

    # define gain profile for display
    t_gain = np.arange(outputs_array.shape[-1], dtype=np.float32) ** 2.5
    t_gain = t_gain / np.median(t_gain)
    t_gain = t_gain.reshape((1, 1, 1, outputs_array.shape[-1]))  # along NSTEPS

    plt.subplot2grid((1, 4), (0, 0), colspan=2)
    plt.imshow(inputs_array[ib, 0, :, :].T, vmin=-1, vmax=1)
    plt.colorbar()

    plt.subplot2grid((1, 4), (0, 2), colspan=1)
    plt.imshow((t_gain * outputs_array)[ib, isource, :, :].T,
               aspect=aspect, cmap="Greys", vmin=-1, vmax=1)
    plt.colorbar()
    plt.title("%f, %f" % (np.min(outputs_array), np.max(outputs_array)))

    plt.subplot2grid((1, 4), (0, 3), colspan=1)
    plt.imshow((t_gain * labels_array)[ib, isource, :, :].T,
               aspect=aspect, cmap="Greys", vmin=-1, vmax=1)
    plt.colorbar()
    plt.title(
        "%s" % (sample_batch["inputs"][1].detach().cpu().numpy().copy()[ib, :, 0, 0]))  # label with source position

    return f


class Trainer:

    def __init__(self, c):
        "Initialise torch, output directories, training dataset and model"

        ## INITIALISE

        # set seed
        if c.SEED == None:
            c.SEED = torch.initial_seed()
        else:
            torch.manual_seed(c.SEED)  # likely independent of numpy
        np.random.seed(c.SEED)

        # clear directories
        c.get_outdirs()  # 将这个函数的内部函数进行修改，把清空文件夹的函数注释掉
        # （不能清空文件夹不然，每次一运行就会将之前的文件全部删除）
        c.save_constants_file()  # saves torch seed too
        print(c)

        # set device/ threads
        # 指认设备和进程数
        device = torch.device("cuda:%i" % (c.DEVICE) if torch.cuda.is_available() else "cpu")
        print("Device: %s" % (device))
        torch.backends.cudnn.benchmark = False
        # let cudnn find the best algorithm to use for your hardware (not good for dynamic nets)
        torch.set_num_threads(1)  # for main inference

        print("Main thread ID: %i" % os.getpid())
        print("Number of CPU threads: ", torch.get_num_threads())
        print("Torch seed: ", torch.initial_seed())

        # initialise summary writer
        writer = SummaryWriter(c.SUMMARY_OUT_DIR)

        ### DEFINE TRAIN/TEST DATASETS

        # split dataset 80:20
        irange = np.arange(0, c.N_EXAMPLES)
        np.random.shuffle(irange)
        # randomly shuffle the indicies (in place) before splitting. To get diversity in train/test split.
        traindataset = c.DATASET(c,irange=irange[0:(8 * c.N_EXAMPLES // 10)],verbose=True)
        testdataset = c.DATASET(c,irange=irange[(8 * c.N_EXAMPLES // 10):c.N_EXAMPLES],verbose=True)
        assert len(set(traindataset.irange).intersection(testdataset.irange)) == 0  # make sure examples aren't shared!

        #### DEFINE MODEL
        model = c.MODEL()

        # load previous weights
        if c.MODEL_LOAD_PATH != None:
            cp = torch.load(c.MODEL_LOAD_PATH,map_location=torch.device('cpu'))  # remap tensors from gpu to cpu if needed
            model.load_state_dict(cp['model_state_dict'])
            ioffset = cp["i"]
            print("Loaded model weights from: %s" % (c.MODEL_LOAD_PATH))
        else:
            ioffset = 0


        # writer.add_graph(model, torch.zeros((1,)+c.VELOCITY_SHAPE))# write graph before placing on GPU
        print()
        print("Model: %s" % (model.name))
        total_params = sum(p.numel() for p in model.parameters())
        total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print("Total number of parameters: %i" % (total_params))
        print("Total number of trainable parameters: %i" % (total_trainable_params))


        model.to(device)

        self.c, self.device, self.writer = c, device, writer
        self.traindataset, self.testdataset = traindataset, testdataset
        self.model, self.ioffset = model, ioffset

    def train(self):
        "train model"
        trainingloss=[]
        testingloss=[]

        c, device, writer = self.c, self.device, self.writer
        traindataset, testdataset = self.traindataset, self.testdataset
        model, ioffset = self.model, self.ioffset

        ### TRAIN

        print()
        print("Training..")

        N_BATCHES = len(traindataset) // c.BATCH_SIZE
        N_EPOCHS = int(np.ceil(c.N_STEPS / N_BATCHES))

        # below uses my own batch sampler so that dataloader iterators run over n_epochs
        # also uses dataset.initialise_file_reader method to open a file handle in each worker process,
        # instead of a shared one on the main thread
        # DataLoader essentially iterates through iter(batch_sampler) or iter(sampler) depending on inputs
        # calling worker_init in each worker process
        trainloader = DataLoader(traindataset,
                                 batch_sampler=BatchSampler(RandomSampler(traindataset, replacement=True),
                                                            # randomly sample with replacement
                                                            batch_size=c.BATCH_SIZE,
                                                            drop_last=True,
                                                            n_epochs=1),
                                 worker_init_fn=traindataset.initialise_worker_fn,
                                 num_workers=c.N_CPU_WORKERS,  # num_workers = spawns multiprocessing subprocess workers
                                 timeout=300)  # timeout after 5 mins of no data loading

        testloader = DataLoader(testdataset,
                                batch_sampler=BatchSampler(RandomSampler(testdataset, replacement=True),
                                                           # randomly sample with replacement
                                                           batch_size=c.BATCH_SIZE,
                                                           drop_last=True,
                                                           n_epochs=N_EPOCHS),
                                worker_init_fn=testdataset.initialise_worker_fn,
                                num_workers=1,  # num_workers = spawns multiprocessing subprocess workers
                                timeout=300)  # timeout after 5 mins of no data loading

        testloader_iterator = iter(testloader)
        trainloader_iterator = iter(trainloader)


        ######################
        #####采用不同的优化器###
        ######################

        # optimizer=torch.optim.Adadelta(model.parameters(), lr=1.0, rho=0.9, eps=1e-06, weight_decay=0)
        # optimizer=torch.optim.RMSprop(model.parameters(), lr=0.01, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0, centered=False)
        optimizer = torch.optim.Adam(model.parameters(), lr=c.LRATE, weight_decay=c.WEIGHT_DECAY)
        # optimizer = torch.optim.SGD(model.parameters(), lr=c.LRATE, momentum=0.9)
        # optimizer=torch.optim.Adagrad(model.parameters(), lr=0.01, lr_decay=0, weight_decay=0, initial_accumulator_value=0, eps=1e-10)
        # optimizer = torch.optim.Adamax(model.parameters(), lr=0.002, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

        start0 = start1 = time.time();
        w1 = get_weights(model)
        for ie in range(N_EPOCHS):  # loop over the dataset multiple times
            wait_start, wait_time, gpu_time, gpu_utilisation = time.time(), 0., 0., 0.
            for ib in range(N_BATCHES):
                i = ioffset + ie * N_BATCHES + ib

                try:  # get next sample_batch
                    sample_batch = next(trainloader_iterator)
                except StopIteration:  # restart iterator
                    del trainloader_iterator
                    trainloader_iterator = iter(trainloader)  # re-initiates batch/sampler iterators, with new random starts
                    sample_batch = next(trainloader_iterator)

                wait_time += time.time() - wait_start

                ## TRAIN
                gpu_start = time.time()
                model.train()  # switch to train mode (for dropout/ batch norm layers)

                # get the data
                inputs = sample_batch["inputs"]  # expects list of inputs
                labels = sample_batch["labels"]  # expects list of labels
                inputs = [inp.to(device) for inp in inputs]
                labels = [lab.to(device) for lab in labels]

                # zero the parameter gradients  AT EACH STEP
                optimizer.zero_grad()  # zeros all parameter gradient buffers

                # forward + backward + optimize
                outputs = model(*inputs)  # expect tuple of output


                # labels1 = torch.tensor([item.cpu().detach().numpy() for item in labels])


                # print(type(labels[0])) #<class 'torch.Tensor'>
                labels=labels[0]
                # print(labels.shape)
                # print(outputs.shape)

                loss = c.LOSS_FUNC(labels,outputs,c)  # use the <Loss> standard to update the gradient
                loss1=loss.item()
                with open(c.MODEL_OUT_DIR + "training111_%s_%s_%s.txt" % (
                        c.MODEL_NAME, c.LOSS_NAME, c.OPTIMIZER), 'a') as f1:
                    f1.write(str(loss1)+"\n")

                trainingloss.append(loss1)

                loss.backward()  # updates all gradients in model
                optimizer.step()  # updates all parameters using their gradients

                gpu_time += time.time() - gpu_start

                ## TRAIN STATISTICS

                if (i + 1) % 100 == 0:
                    gpu_utilisation = 100 * gpu_time / (wait_time + gpu_time)
                    print("Wait time average: %.4f s GPU time average: %.4f s GPU util: %.2f %% device: %i" % (
                        wait_time / 100, gpu_time / 100, gpu_utilisation, c.DEVICE))
                    gpu_time, wait_time = 0., 0.

                if (i + 1) % c.SUMMARY_FREQ == 0:
                    print("---------------training begin---------------")  # 后期加的！

                    rate = c.SUMMARY_FREQ / (time.time() - start1)

                    with torch.no_grad():  # faster inference without tracking
                        model.eval()

                        # get example outputs and losses
                        inputs = sample_batch["inputs"]  # expects list of inputs
                        labels = sample_batch["labels"]  # expects list of labels
                        inputs = [inp.to(device) for inp in inputs]
                        labels = [lab.to(device) for lab in labels]
                        outputs = model(*inputs)

                        labels=labels[0]
                        # loss=c.LOSS_FUNC(labels[0], outputs[0]).item()
                        loss = c.LOSS_FUNC(labels, outputs,c).item()
                        writer.add_scalar("loss/"+c.LOSS_NAME+"/train", loss, i + 1)

                        inputs_array = inputs[0].detach().cpu().numpy().copy()
                        # detach returns a new tensor, detached from the current graph
                        outputs_array = outputs[0].detach().cpu().numpy().copy()
                        labels_array = labels[0].detach().cpu().numpy().copy()
                        if (i + 1) % (10 * c.SUMMARY_FREQ) == 0:
                            f = plot_result(inputs_array, outputs_array, labels_array, sample_batch)
                            writer.add_figure("compare/train", f, i + 1, close=True)

                        # check weight updates from previous summary
                        w2 = get_weights(model)
                        mu, _, av = get_weights_update_percent(w1, w2)
                        s = "Weight updates (%.1f %% average): " % (100 * av)
                        for m in mu: s += "%.1f " % (100 * m)
                        print(s)
                        del w1;
                        w1 = w2

                        # add run statistics
                        writer.add_scalar("stats/epoch", ie, i + 1)
                        writer.add_scalar("stats/rate/batch", rate, i + 1)
                        writer.add_scalar("stats/rate/gpu_utilisation", gpu_utilisation, i + 1)

                        # output to screen
                        print(
                            '[epoch: %i/%i, batch: %i/%i i: %i] loss: %.4f elapsed: %.2f hr %s %s' % (
                                ie + 1,
                                N_EPOCHS,
                                ib + 1,
                                N_BATCHES,
                                i + 1,
                                loss,
                                (time.time() - start0) / (60 * 60),
                                time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
                                c.RUN))
                    print("---------------training end---------------")  # 后期加的！
                    print()  # 后期加的！

                    start1 = time.time()

                ## TEST STATISTICS

                if (i + 1) % c.TEST_FREQ == 0:
                    print("---------------testing begin---------------")  # 后期加的！

                    with torch.no_grad():  # faster inference without tracking

                        # try:  # get next sample_batch
                        #     sample_batch = next(testloader_iterator)
                        # except StopIteration:  # restart iterator
                        #     del testloader_iterator
                        #     testloader_iterator = iter(testloader)  # re-initiates batch/sampler iterators, with new random starts
                        #     sample_batch = next(testloader_iterator)
                        #     # print(sample_batch["i"])# check

                        model.eval()

                        # get example outputs and losses
                        inputs = sample_batch["inputs"]  # expects list of inputs
                        labels = sample_batch["labels"]  # expects list of labels
                        inputs = [inp.to(device) for inp in inputs]
                        labels = [lab.to(device) for lab in labels]
                        outputs = model(*inputs)
                        labels=labels[0]

                        # loss=c.LOSS_FUNC(labels[0], outputs[0]).item()
                        loss = c.LOSS_FUNC(labels, outputs,c).item()
                        with open(
                                c.MODEL_OUT_DIR + "testing111_%s_%s_%s.txt" % (c.MODEL_NAME, c.LOSS_NAME, c.OPTIMIZER),
                                'a') as f2:
                            f2.write(str(loss)+"\n")

                        testingloss.append(loss)
                        writer.add_scalar("loss/"+c.LOSS_NAME+"/test", loss, i + 1)

                        inputs_array = inputs[0].detach().cpu().numpy().copy()
                        # detach returns a new tensor, detached from the current graph
                        outputs_array = outputs[0].detach().cpu().numpy().copy()
                        labels_array = labels[0].detach().cpu().numpy().copy()
                        if (i + 1) % (10 * c.TEST_FREQ) == 0:
                            f = plot_result(inputs_array, outputs_array, labels_array, sample_batch)
                            writer.add_figure("compare/test", f, i + 1, close=True)
                        # print函数为后期加的！
                        print(
                            '[epoch: %i/%i, batch: %i/%i i: %i] loss: %.4f  elapsed: %.2f hr %s %s' % (
                                ie + 1,
                                N_EPOCHS,
                                ib + 1,
                                N_BATCHES,
                                i + 1,
                                loss,
                                (time.time() - start0) / (60 * 60),
                                time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
                                c.RUN
                            ))
                    print("---------------testing end-----------------")  # 后期加的！
                    print()  # 后期加的！

                ## SAVE

                if (i + 1) % c.MODEL_SAVE_FREQ == 0:
                    print("------------Saving model begin-------------")  # 后期加的！

                    model.eval()

                    model.to(torch.device('cpu'))  # put model on cpu before saving
                    # to avoid out-of-memory error

                    # save a checkpoint
                    # torch.save({
                    # 'i': i + 1,
                    # 'model_state_dict': model.state_dict(),
                    # }, c.MODEL_OUT_DIR+"model_%.8i.torch"%(i + 1))

                    # 加上关于优化器的
                    torch.save({
                        'i': i + 1,
                        'model_state_dict': model.state_dict(),
                    }, c.MODEL_OUT_DIR + "%s_%s_%s_%.8i.torch" % (c.MODEL_NAME,c.OPTIMIZER, c.LOSS_NAME,i + 1))

                    model.to(device)
                    print("-------------Saving model end--------------")  # 后期加的！
                    print()  # 后期加的！

                wait_start = time.time()

        del trainloader_iterator, testloader_iterator

        #save
        # 训练、测试损失保存
        with open(c.MODEL_OUT_DIR + "training_%s_%s_%s.txt" % (
                c.MODEL_NAME, c.LOSS_NAME, c.OPTIMIZER), 'w') as f:
            for _ in trainingloss:
                f.write("%f \n" % _)

        with open(
                c.MODEL_OUT_DIR + "testing_%s_%s_%s.txt" % (c.MODEL_NAME, c.LOSS_NAME, c.OPTIMIZER),
                'w') as f:
            for _ in testingloss:
                f.write("%f \n" % _)


        print('Finished Training (total runtime: %.1f hrs)' % (
                (time.time() - start0) / (60 * 60)))

        with open(
            c.MODEL_OUT_DIR+"TraningTime_%s_%s_%s.txt"% (c.MODEL_NAME, c.LOSS_NAME, c.OPTIMIZER),
        "w") as f :
            f.write('Finished Training (total runtime: %.1f hrs)' % (
                (time.time() - start0) / (60 * 60)))


    def close(self):
        self.writer.close()


if __name__ == "__main__":

    # cs = [Constants(), ]

    DEVICE = 0

    cs = [Constants(RUN="fault_SeismicTrans_Adam_l1_mean_loss_gain",
                    LOSS_NAME="l1_mean_loss_gain",
                    LOSS_FUNC=losses.l1_mean_loss_gain,
                    T_GAIN=2.5,
                    LRATE=1e-4,
                    DEVICE=DEVICE,
                    ), ]


    for c in cs:
        run = Trainer(c)
        run.train()
        run.close()

