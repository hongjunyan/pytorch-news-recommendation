import re
import yaml
from typing import Union
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
import mlflow


def flat_config(config):
    """Flat config loaded from a yaml file to a flat dict.
    Args:
        config (dict): Configuration loaded from a yaml file.
    Returns:
        dict: Configuration dictionary.
    """
    f_config = {}
    category = config.keys()
    for cate in category:
        for key, val in config[cate].items():
            f_config[key] = val
    return f_config


class HyperParams(object):
    def __init__(self, yaml_file: Union[Path, str]):
        yaml_file = Path(yaml_file)
        # https://stackoverflow.com/questions/30458977/yaml-loads-5e-6-as-string-and-not-a-number
        loader = yaml.SafeLoader
        loader.add_implicit_resolver(
            u'tag:yaml.org,2002:float',
            re.compile(u'''^(?:[-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
                    |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
                    |\\.[0-9_]+(?:[eE][-+][0-9]+)?
                    |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
                    |[-+]?\\.(?:inf|Inf|INF)
                    |\\.(?:nan|NaN|NAN))$''', re.X),
            list(u'-+0123456789.'))

        if yaml_file.exists():
            with open(yaml_file, "r") as f:
                cfg = yaml.load(f, Loader=loader)
                cfg = flat_config(cfg)

            self._cfg = cfg
            for hparam in cfg:
                setattr(self, hparam, cfg[hparam])

    def update(self, **kargs):
        self._cfg.update(kargs)
        for hparam in kargs:
            setattr(self, hparam, kargs[hparam])

    def __repr__(self):
        return f"Hyper-Parameter: {self._cfg}"


class ModelLogger(object):
    def __init__(self, tensorboard_dir: str):
        self.writer = SummaryWriter(tensorboard_dir)

    def log_hparam(self, key, value):
        mlflow.log_param(key, value)

    def log_pr_curve(self, preds, labels):
        pass

    def log_scalar(self, name: str, value: float, step: int):
        """Log a scalar value to both MLflow and TensorBoard"""
        self.writer.add_scalar(name, value, step)
        mlflow.log_metric(name, value, step=step)







