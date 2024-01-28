#===============================================
# import libraries
#===============================================

import torch
import logging


#========================================
def check_gpu():
#========================================
    # check for GPU availability.

    print("="*50)
    logging.info("Checking for GPU availability.")

    # cuda first
    if torch.cuda.is_available():
        device = 'cuda'
        use_mps_device = False

    # mps next 
    elif torch.backends.mps.is_available():
        device = 'mps'
        use_mps_device = True

    # cpu as the last possibility
    else:
        device = 'cpu'
        use_mps_device = False

    # print the results
    logging.info(f'... using {device} for pytorch.')

    return device