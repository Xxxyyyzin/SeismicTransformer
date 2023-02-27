import os
import sys
import numpy as np



ROOT_DIR = ""

VELOCITY_DIR = ROOT_DIR + "velocity/fault/"
GATHER_DIR = ROOT_DIR + "gather/fault_2ms_r/"
DATA_PATH = ROOT_DIR + "data/fault_2ms_r.bin"
DATA_DIR=ROOT_DIR+"data/"

sys.path.insert(0, '../shared_modules/')
import io_utils
io_utils.get_dir(DATA_DIR)

N_VELS = 200
N_SOURCES = 3




def convert_to_binary_func(first_epoch):
    if first_epoch == True:
        source_is = np.load(GATHER_DIR + "source_is.npy")
        with open(DATA_PATH, 'wb') as f_data:
            # parse to flat binary
            for ivel in range(N_VELS):
                # load velocity data from .npy
                velocity = np.load(VELOCITY_DIR + "velocity_%.8i.npy" % (ivel))  # shape: (NX, NY)

                for isource in range(N_SOURCES):
                    gather = np.load(
                        GATHER_DIR + "gather_%.8i_%.8i.npy" % (ivel, isource))  # shape: (NREC, NSTEPS)

                    source_i = source_is[ivel, isource].astype(np.float32)

                    # write individual examples to file
                    # (sacrifices size for readability (duplicates velocity model))
                    velocity = velocity.astype("<f4")  # ensure little endian float32
                    gather = gather.astype("<f4")
                    source_i = source_i.astype("<f4")
                    f_data.write(velocity.tobytes(order="C"))  # ensure C order
                    f_data.write(gather.tobytes(order="C"))
                    f_data.write(source_i.tobytes(order="C"))

    else:
        source_is = np.load(GATHER_DIR + "source_is.npy")
        with open(DATA_PATH, 'ab') as f_data:
            # parse to flat binary
            for ivel in range(N_VELS):
                # load velocity data from .npy
                velocity = np.load(VELOCITY_DIR + "velocity_%.8i.npy" % (ivel))  # shape: (NX, NY)

                for isource in range(N_SOURCES):
                    gather = np.load(
                        GATHER_DIR + "gather_%.8i_%.8i.npy" % (ivel, isource))  # shape: (NREC, NSTEPS)

                    source_i = source_is[ivel, isource].astype(np.float32)

                    # write individual examples to file
                    # (sacrifices size for readability (duplicates velocity model))
                    velocity = velocity.astype("<f4")  # ensure little endian float32
                    gather = gather.astype("<f4")
                    source_i = source_i.astype("<f4")
                    f_data.write(velocity.tobytes(order="C"))  # ensure C order
                    f_data.write(gather.tobytes(order="C"))
                    f_data.write(source_i.tobytes(order="C"))
    print(os.path.getsize(DATA_PATH))




