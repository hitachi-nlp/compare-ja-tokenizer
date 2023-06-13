import logging
import pprint
import shutil
import time
from logging import getLogger
from pathlib import Path
from typing import List, Tuple

from transformers import TrainerCallback

logger = getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO
)


class LoggingCallback(TrainerCallback):
    """A `TrainerCallback` for saving weights periodically."""
    def __init__(self, save_interval: float):
        """
        Args:
            save_interval (float): An interval to save weights in seconds.  
        """
        self.save_interval = save_interval
        self.start_time = time.time()
        self.save_counter = 1

    
    def on_log(self, args, state, control, logs=None, **kwargs):
        current_duration = time.time() - self.start_time
        if (current_duration // (self.save_interval * self.save_counter)) >= 1:
            logger.info(f'Save weights at {state.global_step} steps trained for '
                        f'{self.save_interval} * {self.save_counter} seconds!')
            self.save_counter += 1
            control.should_save = True
