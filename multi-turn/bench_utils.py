"""
Copied from https://github.com/vllm-project/vllm/blob/9e77ffca3f41e0e73879098f1686a4c82b8619d9/benchmarks/multi_turn/bench_utils.py
"""

import logging
from enum import Enum


class Color(Enum):
    RED = "\033[91m"
    GREEN = "\033[92m"
    BLUE = "\033[94m"
    PURPLE = "\033[95m"
    CYAN = "\033[96m"
    YELLOW = "\033[93m"
    RESET = "\033[0m"

    def __str__(self):
        return self.value


TEXT_SEPARATOR = "-" * 100

# Configure the logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] - %(message)s",
    datefmt="%d-%m-%Y %H:%M:%S",
)
logger = logging.getLogger(__name__)