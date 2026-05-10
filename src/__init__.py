import yaml
from pathlib import Path
import logging
from types import SimpleNamespace

def setup_logger():
    log_dir = Path(__file__).resolve().parent.parent / "logs"
    log_dir.mkdir(exist_ok=True)
    logger = logging.getLogger("L2O")
    logger.setLevel(logging.INFO)
    
    if not logger.handlers:
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        )
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        file_handler = logging.FileHandler(log_dir / "log_baseline.log", encoding='utf-8')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger
logger = setup_logger()

_device = None
def get_device():
    global _device
    if _device is None:
        import torch
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using {_device}")
    return _device

def dict_to_sns(d):
    if isinstance(d, dict):
        return SimpleNamespace(**{k: dict_to_sns(v) for k, v in d.items()})
    return d

def load_all_config():
    doc_path = Path(__file__).parent.parent
    config_path = doc_path / "config/config.yaml"

    if not config_path:
        raise FileNotFoundError("Connot found config file")
    
    with open(config_path,"r",encoding="utf-8") as f:
        config_dict = yaml.safe_load(f)

    return dict_to_sns(config_dict)
cfg = load_all_config()
sample_dir = Path(__file__).resolve().parent.parent / "samples"