import os
from yacs.config import CfgNode as CN

# global config object
_C = CN()

# ----------------configuration options------------------------ #

# number of GPUS to use in the experiment
_C.NUM_GPUS = 1
# number of workers for doing things
_C.NUM_WORKERS = 12
# random seed
_C.RNG_SEED = 42
# configuration directory
_C.CFG_DIR = 'cfgs'
# base configuration yaml file
_C.CFG_BASE = 'adapter.yaml'
# train batch size
_C.TRAIN_BATCH = 32
# val batch size
_C.VAL_BATCH = 32
# task name
_C.TASK_NAME = 'cola'
# whether to use full-finetuning or adapter
_C.TRAINING_STRATEGY = 'adapter'
# adapter bottleneck size
_C.ADAPTER_BOTTLENECK = 64
#max sequece length
_C.MAX_SEQ_LENGTH = 128
# model name
_C.MODEL_NAME = 'bert-large-uncased'
# lr
_C.LEARNING_RATE = 3e-5
# weight decay
_C.WEIGHT_DECAY = 0.0
# warmup steps
_C.WARMUP_STEPS = 0
# number of epochs
_C.EPOCHS = 50
# trainer accelerator
_C.ACCELERATOR = 'auto'

# ----------------default config-------------------------------- #

# import the defaults as a global singleton:
cfg = _C  # `from config import cfg`

_CFG_DEFAULT = _C.clone()
_CFG_DEFAULT.freeze()

def dump_cfg(config_name='cfg.yaml'):
    """Dumps the config to the output directory."""
    cfg_file = os.path.join(_C.CFG_DIR, config_name)
    with open(cfg_file, "w") as f:
        _C.dump(stream=f)

if __name__ == '__main__':
    dump_cfg(_C.CFG_BASE)