import os
import sys

from absl import app, flags, logging

from src.models import alexnet
from src.utils import common

# project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# sys.path.append(os.path.join(project_root, "src"))

common.set_random_seed(42)

model_str = alexnet.AlexNet().model_str
flags.DEFINE_string(
    "experiments_path", os.path.join(common.project_root, "experiments"), help=""
)
flags.DEFINE_string("config_name", f"{model_str}/config/{model_str}-SOC.json", help="")
FLAGS = flags.FLAGS

# TODO: Load dataset
# TODO: Train model
# TODO:

