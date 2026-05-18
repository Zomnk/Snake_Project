# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Python module for snake robot RL project.
"""

import os

SNAKE_ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# Register Gym environments.
from .tasks import *

# Register UI extensions.
from .ui_extension_example import *
