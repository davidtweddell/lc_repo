import os, tarfile
import pathlib
from pathlib import Path
from typing import Union, List, Dict, Tuple, Optional

import logging

DEBUG = False

#========================================
def save_tgz(
                source_dir:  Union[Path, str], 
                content_dir: Union[Path, str],
                target_dir:  Union[Path, str, None] = None,
                verbose:     bool = False,
                DEBUG:       bool = False,
                ):
#========================================
# create a gzipped tarball of the finetuned model

    print("="*50)
    logging.info(f"Preparing tgz file.")

    # resolve the paths
    source_dir = Path(source_dir).resolve()
    target_dir = Path(target_dir).resolve() if target_dir is not None else None

    # check to make sure the source directory exists
    if not source_dir.is_dir():
        logging.error(f"Source directory {source_dir} does not exist.")
        raise FileNotFoundError(f"Source directory {source_dir} does not exist.")
    
    # check to make sure the content directory exists
    if not (source_dir / content_dir).is_dir():
        logging.error(f"Content directory {source_dir / content_dir} does not exist.")
        raise FileNotFoundError(f"Content directory {source_dir / content_dir} does not exist.")
    
    # check to make sure the target directory exists
    if target_dir is not None:
        if not target_dir.is_dir():
            logging.error(f"Target directory {target_dir} does not exist.")
            raise FileNotFoundError(f"Target directory {target_dir} does not exist.")


    # the name of the tarfile
    tar_fn = f"{content_dir}.tar.gz"

    # if the target directory is not specified, put the output in the source_dir:
    if target_dir is None:
        target_dir = source_dir


    if DEBUG:
        logging.debug(f"source_dir:  {source_dir}")
        logging.debug(f"target_dir:  {target_dir}")
        logging.debug(f"content_dir: {content_dir}")
        logging.debug(f"tar_fn:      {tar_fn}")

    # # check if the parent directory exists
    # if os.path.exists(source_dir):
    logging.info(f"... creating {tar_fn}")

    # what is our current working directory?
    # logging.info(os.getcwd())
    original_cwd = os.getcwd()

    # when we tar, we want to use the basename of the path
    # so that the tarball does not have the full path
    # in it

    # move the cwd to the source_dir
    os.chdir(source_dir)
    if DEBUG:
        logging.debug(f"os.getcwd(): {os.getcwd()}")

    # make the tarfile
    with tarfile.open(tar_fn, "w:gz") as tar:
        tar.add(content_dir)
        # if DEBUG:
        #     logging.debug("tarfile contents:")
        #     logging.debug(tar.list(verbose=True))

    # return to the original cwd
    os.chdir(original_cwd)
    if DEBUG:
        logging.info(f"os.getcwd(): {os.getcwd()}")

    logging.info(f"... done.")
