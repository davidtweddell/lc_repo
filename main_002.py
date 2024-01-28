#===============================================
# import libraries
#===============================================
import math
import os
import timeit

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from rich.pretty import pprint

# output options for pandas and numpy
pd.options.display.float_format  = "{:0.2f}".format
np.set_printoptions(precision=3)

DEBUG = True

#===============================================
# set up logging
#===============================================
from project_modules.utils import get_logger

print("="*50)
logger = get_logger("log.log")

#===============================================
# logical control variables
#===============================================
DEBUG   = False
VERBOSE = False

#===============================================
# check GPU availability
#===============================================
from project_modules.utils import check_gpu
device = check_gpu()

#===============================================================================
# read data from yaml file
#===============================================================================
from project_modules.utils import read_parameters
config = read_parameters("config.yaml", verbose = False, DEBUG = True)

#===============================================================
# Code to do the work should go here
#===============================================================
def work():
    # announce start
    print("="*50)
    logger.info("Starting.")

    print("="*50)
    logger.info("Next step.")

    print("="*50)
    logger.info("Done.")


    # targzip something
    from project_modules.io import save_tgz

    save_tgz(source_dir  = "project_modules", 
             target_dir  = ".", 
             content_dir = "utils", 
             verbose     = True,
             DEBUG       = False)



#===============================================================
# Timing wrapper around the main code
#===============================================================
if __name__ == "__main__":

    from project_modules.utils import Timer

    with Timer() as t:
        work()

    print("="*50)
    logger.info(f"Program executed in {t.interval:.3f} [s]")
    print("="*50)