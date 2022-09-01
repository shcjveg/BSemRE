import pytorch_lightning as pl
import sastvd.linevd as lvd
# from ray.tune.integration.pytorch_lightning import (
#     TuneReportCallback,
#     TuneReportCheckpointCallback,
# )
import os
import sastvd as svd
from pytorch_lightning.loggers import CSVLogger

os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["SLURM_JOB_NAME"] = "bash"

# "modeltype": tune.choice(["gat1layer", "gat2layer", "mlponly"]),
# "gnntype": tune.choice(["gat", "gcn"]),
# "gtype": tune.choice(["pdg", "pdg+raw", "cfgcdg", "cfgcdg+raw"]),

config = {
    "hfeat": 512,
    "embtype": "doc2vec",
    "stmtweight": 1,
    "hdropout": 0.3,
    "gatdropout": 0.2,
    "modeltype": "gat2layer",
    "gnntype": "gat",
    "loss": "ce",
    "scea": 0.5,
    "gtype": "pdg+raw",
    "batch_size": 256, # 1024
    "multitask": "linemethod",
    "splits": "default",
    "lr": 1e-4,
}
samplesz = -1
run_id = svd.get_run_id()
sp = svd.get_dir(svd.processed_dir() / f"poison_{config['embtype']}" / run_id)
logger_dir = svd.get_dir(svd.processed_dir() / f"logger"/ f"poison_{config['embtype']}_{config['gtype']}")
logger_csv = CSVLogger(logger_dir, name="poison")

def train_linevd(
    config, savepath, samplesz=-1, max_epochs=130, num_gpus=1, checkpoint_dir=None
):
    """Wrap Pytorch Lightning to pass to RayTune."""
    model = lvd.LitGNN(
        hfeat=config["hfeat"],
        embtype=config["embtype"],
        methodlevel=False,
        nsampling=True,
        model=config["modeltype"],
        loss=config["loss"],
        hdropout=config["hdropout"],
        gatdropout=config["gatdropout"],
        num_heads=4,
        multitask=config["multitask"],
        stmtweight=config["stmtweight"],
        gnntype=config["gnntype"],
        scea=config["scea"],
        lr=config["lr"],
        batch_size=config["batch_size"],
    )

    # Load data
    data = lvd.BigVulDatasetLineVDDataModule(
        batch_size=config["batch_size"],
        sample=samplesz,
        methodlevel=False,
        nsampling=True,
        nsampling_hops=2,
        gtype=config["gtype"],
        splits=config["splits"],
    )

    # # Train model
    checkpoint_callback = pl.callbacks.ModelCheckpoint(monitor="val_loss",dirpath=sp)
    # metrics = ["train_loss", "val_loss", "val_auroc"]
    # raytune_callback = TuneReportCallback(metrics, on="validation_end")
    # rtckpt_callback = TuneReportCheckpointCallback(metrics, on="validation_end")

    trainer = pl.Trainer(
        # gpus=[0],
        logger=logger_csv,
        strategy="ddp",
        accelerator="gpu",
        devices=[1],
        auto_lr_find=False,
        default_root_dir=savepath,
        num_sanity_val_steps=0,
        # callbacks=[checkpoint_callback, raytune_callback, rtckpt_callback],
        callbacks=[checkpoint_callback],
        max_epochs=max_epochs,
    )
    trainer.fit(model, data)

if __name__ == '__main__':
    train_linevd(config=config,savepath=sp,samplesz=samplesz,max_epochs=10)