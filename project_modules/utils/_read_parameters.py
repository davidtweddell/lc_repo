#===============================================
# import libraries
#===============================================

from pathlib import Path
from rich.pretty import pprint
from typing import Dict, List, Union, Any
import yaml

import logging

#===============================================
# logical control variables
#===============================================
DEBUG   = False
VERBOSE = False

#================================================================
def read_parameters(
                    p: Union[Path, str], 
                    verbose: bool = VERBOSE, 
                    DEBUG: bool = False) -> Dict[str, Union[str, int, float, List[str]]]:
#================================================================

    print('='*50)
    logging.info(f'Reading parameters.')
    # print('-'*50)

    logging.info(f"... reading {p}")

    with open(Path(p), 'r') as stream:
        parms = yaml.safe_load(stream)
    if verbose:
        pprint(parms, expand_all = True)
    if DEBUG:
        logging.debug(f'... {parms}')

    return parms


# #================================================================
# def make_paths(parms: Dict, 
#                verbose: bool = VERBOSE, 
#                DEBUG: bool = False) -> Dict:

# #================================================================
    
#     raw_data_p = Path(parms['raw_data_p'])
#     raw_data_f = Path(parms['raw_data_f'])
#     raw_data_fp = raw_data_p / raw_data_f
    
#     proc_data_p = Path(parms['proc_data_p'])
