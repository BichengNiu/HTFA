# -*- coding: utf-8 -*-
"""
可重现性管理模块

确保实验结果的可重现性
"""

import random
import numpy as np
import os
from typing import Optional
from dashboard.DFM.train_ref.utils.logger import get_logger


logger = get_logger(__name__)


def set_seed(seed: int) -> None:
    """设置全局随机种子

    Args:
        seed: 随机种子
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    logger.info(f"设置全局随机种子: {seed}")


def ensure_reproducibility(seed: Optional[int] = None) -> int:
    """确保可重现性

    Args:
        seed: 随机种子，如果为None则使用默认值42

    Returns:
        int: 使用的随机种子
    """
    if seed is None:
        seed = 42

    set_seed(seed)

    return seed
