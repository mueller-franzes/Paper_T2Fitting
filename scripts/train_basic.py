import warnings
warnings.filterwarnings("ignore", message="Unsupported `ReduceOp` for distributed computing.")

from pathlib import Path
from fitting.models import FCN
from fitting.data import ExpDecayDataModule
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
import torch
from datetime import datetime


if __name__ == "__main__":
    # ----------- Settings/Defaults Initialization ----------------
    current_time = datetime.now().strftime("%Y_%m_%d_%H%M%S")
    root_dir = Path.cwd() / 'runs' / str(current_time)
    path_run_dir = root_dir
    root_dir.mkdir(parents=True, exist_ok=True)
    gpus = [0] if torch.cuda.is_available() else None

    # ------------ Load Data ----------------
    dm = ExpDecayDataModule(path_checkpoint_dir=root_dir, batch_size=1024, ram=False)
    dm.setup('fit')

    # ------------ Initialize Model ------------
    model = FCN(
        n_signalpoints=len(dm.ds_train[0]['timepoints']),
        n_params=len(dm.params_dict()),
        # model_func=dm.ds_train.model_func,
    )
    model.model_func = dm.ds_train.model_func  # WORKAROUND: see init function

    # -------------- Training Initialization ---------------
    to_monitor = "Val/loss"
    min_max = "min"

    checkpointing = ModelCheckpoint(
        filepath=str(path_run_dir),
        monitor=to_monitor,
        save_last=True,
        save_top_k=3,
        mode=min_max,
    )
    trainer = Trainer(
        gpus=gpus,
        default_root_dir=str(path_run_dir),
        callbacks=[checkpointing],
        checkpoint_callback=True,
        check_val_every_n_epoch=1,
        log_every_n_steps=50,
        auto_lr_find=False,
        limit_train_batches=1.0,
        limit_val_batches=1.0,  # 0 = disable validation
        min_epochs=1,
        max_epochs=500,
        num_sanity_val_steps=2
    )

    # ---------------- Execute Training ----------------
    trainer.fit(model, datamodule=dm)

    # ------------- Save config -------------
    model.save_best_checkpoint(dm.path_checkpoint_dir, checkpointing.best_model_path)
    dm.save()
