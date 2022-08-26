import copy
from pathlib import Path
import torch
import torch.nn as nn


class BasicTorchModule(nn.Module):
    def __init__(self, hparams):
        super(BasicTorchModule, self).__init__()
        self.params = copy.deepcopy(hparams)
        self.save_dir = Path(self.params.save_dir)

    def save(self, file_name: str = None) -> str:
        checkpoint = {"model": self.state_dict(), "params": self.params}
        if not self.save_dir.exists():
            self.save_dir.mkdir(parents=True, exist_ok=True)
        if file_name is None:
            best_path = self.save_dir.joinpath(f"{self.params.model_type}.pt")
        else:
            best_path = self.save_dir.joinpath(file_name)
        torch.save(checkpoint, str(best_path))
        return str(best_path)

    def load(self, file_name: str = None) -> nn.Module:
        if file_name is None:
            best_path = self.save_dir.joinpath(f"{self.params.model_type}.pt")
        else:
            best_path = self.save_dir.joinpath(file_name)
        if self.params.use_gpu:
            data = torch.load(best_path)["model"]  # load gpu model
        else:
            data = torch.load(best_path, map_location=lambda storage, loc: storage)['model']  # load cpu model
        self.load_state_dict(data)

        if self.params.use_gpu:
            return self.cuda()
        else:
            return self