# -*-coding:utf-8 -*-
import pickle
from copy import deepcopy
from pathlib import Path
from typing import IO, Literal
import sys
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import yaml
from sklearn.metrics import roc_curve

from . import models
from .data.load_data import MyData
from loguru import logger
from functools import partial

from ray import tune, air
from ray.air import Checkpoint, session
from ray.tune.schedulers import ASHAScheduler


class Trained_BinaryModel:
    """
    train mode: normal training
    turn mode: train with ray-tune. hidden_dim,batch_size is ignored
    load mode: load model from saved files.
    """

    def __init__(
        self,
        info_path: str | Path = None,
        info: pd.DataFrame = None,
        all_feature_path: str | Path = None,
        save_path: str | Path = None,
        feature: str = None,
        domain_cols: list = None,
        class_col: str = None,
        control_label: str = None,
        split_col: str = None,
        trait_include: list = None,
        model: Literal["MDAB", "DANN"] = "MDAB",
        mode: Literal["train", "load", "tune"] = "load",
        load: bool | Path | str = True,
        hidden_dim: int = 128,
        first_conv_out: int = 16,
        batch_size: int = 256,
        num_epochs: int = 1000,
        domain_lambda: float = 0.1,
        log_file: IO | Path | str = None,
        num_samples=10,
    ) -> None:
        self.info_path = Path(info_path) if info_path is not None else None
        self.all_feature_path = Path(all_feature_path)
        self.save_path = Path(save_path)
        self.feature = feature
        self.domain_cols = domain_cols
        self.class_col = class_col
        self.control_label = control_label
        self.split_col = split_col
        self.model_name = model
        self.load = load
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.domain_lambda = domain_lambda
        self.first_conv_out = first_conv_out

        # logging
        if log_file is not None:
            self.log_file = log_file
            logger.remove()
            logger.add(self.log_file, level="INFO")
            logger.add(sys.stderr, level="ERROR")

        self.info = info
        self.trait_include = trait_include
        self.model = None
        self.dataset = None
        self.best_state_dict = None
        self.threshold_95 = None
        self.threshold_98 = None
        self.test_predictions = None
        self.test_true_labels = None

        if mode == "train":
            self._train_func()
            self._save()
        elif mode == "retrain":
            if self.load is True:
                self._load_paras(self.save_path)
                self._train_func()
            elif isinstance(self.load, (Path, str)):
                load_path = Path(self.load)
                self._load_paras(load_path)
                self._train_func()
                self._save()
        elif mode == "tune":
            self._train_with_tune(num_samples=num_samples)
            self._process_tune_result()
            self._save()
        elif mode == "retune":
            load_path = Path(self.load)
            self._load_tune_result(load_path)
            self._process_tune_result()
            self._save()
        elif mode == "load":
            if self.load is True:
                self._load_paras(self.save_path)
                self._load_trained_model()
            elif isinstance(self.load, (Path, str)):
                load_path = Path(self.load)
                self._load_paras(load_path)
                self._load_trained_model()
                self._save()
            else:
                self.paras = None
                self._load_trained_model()
                self._save()

        else:
            logger.error("No")

    def _data_process(self, batch_size, val_shuffle=True):
        # load self.info
        if self.info is None:
            self.info = pd.read_csv(self.info_path, low_memory=False)

        self.info["domain"] = self.info[self.domain_cols].apply(
            lambda x: "+".join(x.dropna().astype(str)), axis=1
        )

        # load feature
        if self.all_feature_path.suffix == ".pkl":
            with open(self.all_feature_path, "rb") as f:
                self.df_features = pickle.load(f)
        elif self.all_feature_path.suffix == ".csv":
            self.df_features = pd.read_csv(self.all_feature_path)
        else:
            logger.error("No feature file")
            return 0

        self.df_feature = self.df_features.filter(
            regex=f"{self.feature}|SampleID", axis=1
        )

        df = (
            pd.merge(
                self.df_feature,
                self.info[["SampleID"] + self.trait_include],
                on="SampleID",
                how="inner",
            )
            .drop_duplicates(subset=["SampleID"])
            .set_index("SampleID")
        )

        # Prepare the data
        dataset = MyData(
            df,
            class_col=self.class_col,
            domain_col="domain",
            split_col=self.split_col,
            info_include=self.trait_include,
            batch_size=batch_size,
            control_label=self.control_label,
            val_shuffle=val_shuffle,
        )
        return dataset

    def _init_model(
        self,
        model_name,
        input_dim,
        hidden_dim,
        output_dim,
        num_domains,
        first_conv_out=16,
    ):
        if model_name == "MDAB":
            Mymodel = models.MDAB.MDAB
        elif model_name == "DANN":
            Mymodel = models.model.DANN

        print("input_dim: ", input_dim)
        print("hidden_dim: ", hidden_dim)
        print("output_dim: ", output_dim)
        print("num_domains: ", num_domains)

        model = Mymodel(input_dim, hidden_dim, output_dim, num_domains, first_conv_out)
        return model

    def _train_with_tune(self, num_samples=1):
        config = {
            "first_conv_out": tune.grid_search([8, 16, 32]),
            "hidden_dim": tune.grid_search([64, 128, 256, 512]),
            "domain_lambda": tune.grid_search([0.2, 0.5, 0.8]),
            "batch_size": tune.grid_search([128, 256, 512]),
        }
        config = {
                "first_conv_out": tune.choice([8,16,32]),
                "hidden_dim": tune.choice([64, 128, 256, 512]),
                "domain_lambda": tune.choice( [ 0.2,0.5,0.8]),
                "batch_size": tune.choice([128,256,512])
            }

        scheduler = ASHAScheduler(
            # metric="loss",
            # mode="min",
            max_t=self.num_epochs,
            grace_period=1,
            reduction_factor=2,
        )
        tuner = tune.Tuner(
            partial(self._tune_train_func),
            param_space=config,
            tune_config=tune.TuneConfig(
                metric="loss",
                mode="min",
                num_samples=num_samples,
                scheduler=scheduler,
            ),
            run_config=air.RunConfig(
                name=f"{self.feature}_tune",
                stop={"training_iteration": 100},
                checkpoint_config=air.CheckpointConfig(
                    checkpoint_score_attribute="min-loss",
                    num_to_keep=10,
                ),
                storage_path=self.save_path,
            ),
        )
        self.tune_result = tuner.fit()

    def _load_tune_result(self, load_path):
        experiment_path = load_path / f"{self.feature}_tune"
        if not experiment_path.is_dir():
            raise NotADirectoryError("No tune resuls dir found")

        logger.info(f"Loading results from {experiment_path}...")
        restored_tuner = tune.Tuner.restore(
            experiment_path.__str__(), trainable=partial(self._tune_train_func)
        )
        self.tune_result = restored_tuner.get_results()

    def _process_tune_result(self):
        best_result = self.tune_result.get_best_result("loss", "min", "all")
        
        logger.info(f"Best trial of {self.feature}")
        logger.info(f"Best trial config: {best_result.config}")
        logger.info(
            f"Best trial final validation:{best_result.metrics}"
        )
        
        # save para
        self.hidden_dim = best_result.config["hidden_dim"]
        self.first_conv_out = best_result.config["first_conv_out"]
        self.batch_size = best_result.config["batch_size"]
        self.domain_lambda = best_result.config["domain_lambda"]

        # dataset
        self.dataset = self._data_process(self.batch_size)
        # save best model
        self.model = self._init_model(
            model_name=self.model_name,
            input_dim=self.dataset.feature_size,
            output_dim=self.dataset.num_classes,
            hidden_dim=self.hidden_dim,
            num_domains=self.dataset.num_domains,
            first_conv_out=self.first_conv_out,
        )
        
        best_checkpoint_data = best_result.checkpoint.to_dict()
        # save best model state dict
        self.best_state_dict = best_checkpoint_data["model_state_dict"]

    def _tune_train_func(self, config):
        # data and model
        dataset = self._data_process(batch_size=config["batch_size"])
        model = self._init_model(
            model_name=self.model_name,
            input_dim=dataset.feature_size,
            output_dim=dataset.num_classes,
            hidden_dim=config["hidden_dim"],
            num_domains=dataset.num_domains,
            first_conv_out=config["first_conv_out"],
        )

        print(f"|--Training {self.feature}-----------------|")
        print(f"domain lambda: {self.domain_lambda}")

        # model training
        optimizer = model._optimizer()

        # ray tune checkpoint
        checkpoint = session.get_checkpoint()

        if checkpoint:
            checkpoint_state = checkpoint.to_dict()
            start_epoch = checkpoint_state["epoch"]
            self.model.load_state_dict(checkpoint_state["model_state_dict"])
            optimizer.load_state_dict(checkpoint_state["optimizer_state_dict"])
        else:
            start_epoch = 0

        # Training loop
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        for epoch in range(start_epoch, self.num_epochs):
            model.train_step(
                dataset.train_loader,
                optimizer=optimizer,
                epoch=epoch,
                num_epochs=self.num_epochs,
                domain_lambda=config["domain_lambda"],
            )

            # Evaluation on the validation set
            val_rlt = model.evaluate(dataset.val_loader, cal_auc=False)
            val_loss, val_accuracy, val_auc = val_rlt["results"]

            checkpoint_data = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            }
            checkpoint = Checkpoint.from_dict(checkpoint_data)

            session.report(
                {"loss": val_loss, "accuracy": val_accuracy, "auc": val_auc},
                checkpoint=checkpoint,
            )

        print("Finished Training")

    def _train_func(self):
        # model and dataset
        self.dataset = self._data_process(batch_size=self.batch_size)
        self.model = self._init_model(
            model_name=self.model_name,
            input_dim=self.dataset.feature_size,
            output_dim=self.dataset.num_classes,
            hidden_dim=self.hidden_dim,
            num_domains=self.dataset.num_domains,
        )

        logger.info(f"|--Training {self.feature}-----------------|")
        logger.info(f"domain lambda: {self.domain_lambda}")

        # model training
        optimizer = self.model._optimizer()
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.1, patience=50, min_lr=1e-5
        )

        # Variables to keep track of best validation loss and number of epochs without improvement
        self.num_epochs_no_improvement = 0
        max_epochs_without_improvement = 101

        # Training loop
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)

        best_state_dict = None
        best_val_loss = float("inf")
        for epoch in range(self.num_epochs):
            train_loss_allbatches = self.model.train_step(
                self.dataset.train_loader,
                optimizer=optimizer,
                epoch=epoch,
                num_epochs=self.num_epochs,
                domain_lambda=self.domain_lambda,
                best_model=best_state_dict,
            )

            train_loss = train_loss_allbatches / len(self.dataset.train_loader.dataset)

            # Evaluation on the validation set
            val_rlt = self.model.evaluate(self.dataset.val_loader)
            val_loss, val_accuracy, val_auc = val_rlt["results"]

            scheduler.step(val_loss)

            # Print training and validation statistics
            logger.info(
                f"Epoch {epoch+1}/{self.num_epochs}:Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Accuracy: {val_accuracy:.2%} | Val AUC: {val_auc:.4f} |"
                f"leanring rate: {scheduler.optimizer.param_groups[0]['lr']}"
            )

            # Check if this is the best model so far based on validation loss
            if val_loss <= best_val_loss:
                best_val_loss = val_loss
                best_accuracy = val_accuracy
                best_state_dict = deepcopy(self.model.state_dict())
                best_val_auc = val_auc
                self.num_epochs_no_improvement = 0
            else:
                self.num_epochs_no_improvement += 1

            if self.num_epochs_no_improvement >= max_epochs_without_improvement:
                logger.info("Early stopping triggered!")
                break

        logger.info(
            f"Best Val Loss: {best_val_loss:.4f} | Val Accuracy: {best_accuracy:.2%} | Val AUC: {best_val_auc:.4f}"
        )

        self.best_state_dict = best_state_dict

    def eval(self):
        self.model.load_state_dict(self.best_state_dict)

        val_eval = self.model.evaluate(self.dataset.val_loader)
        fpr, _, thresholds = roc_curve(val_eval["true_labels"], val_eval["predictions"])
        self.threshold_95 = thresholds[np.argmin(1 - fpr >= 0.95)]
        self.threshold_98 = thresholds[np.argmin(1 - fpr >= 0.98)]

        test_eval = self.model.evaluate(self.dataset.test_loader)
        test_loss, test_accuracy, test_auc = test_eval["results"]

        # Print testing statistics
        logger.info(
            f"Test Loss: {test_loss:.4f} | Test Accuracy: {test_accuracy:.2%} | AUC: {test_auc:.4f}"
        )

        self.test_predictions = test_eval["predictions"]
        self.test_true_labels = test_eval["true_labels"]

        test_binary_predictions_95 = (
            self.test_predictions >= self.threshold_95
        ).astype(int)
        test_binary_predictions_98 = (
            self.test_predictions >= self.threshold_98
        ).astype(int)
        test_sensitivity_95 = np.sum(
            (test_binary_predictions_95 == 1) & (np.array(self.test_true_labels) == 1)
        ) / np.sum(self.test_true_labels)
        test_sensitivity_98 = np.sum(
            (test_binary_predictions_98 == 1) & (np.array(self.test_true_labels) == 1)
        ) / np.sum(self.test_true_labels)

        logger.info(
            f"Test Sens at spec 95: {test_sensitivity_95:.2%}\n"
            f"Test Sens at spec 98: {test_sensitivity_98:.2%}"
        )

    def _save(self):
        if not self.save_path.exists():
            self.save_path.mkdir(parents=True)
            logger.info(f"Folder '{self.save_path}' created.")
        model_filename = Path(self.save_path, f"{self.feature}.pth")
        torch.save(self.best_state_dict, model_filename)

        # parameter
        paras = {
            "domain_cols": self.domain_cols,
            "domain_lambda": self.domain_lambda,
            "batch_size": self.batch_size,
            "model_paras": {
                "model_name": self.model_name,
                "input_dim": self.dataset.feature_size,
                "hidden_dim": self.hidden_dim,
                "output_dim": self.dataset.num_classes,
                "num_domains": self.dataset.num_domains,
                "first_conv_out": self.first_conv_out,
            },
        }

        with open(self.save_path / f"{self.feature}_paras.yml", "w") as outfile:
            yaml.dump(paras, outfile)

    def _load_paras(self, load_path):
        paras_file = load_path / f"{self.feature}_paras.yml"
        if paras_file.is_file():
            with open(paras_file, "r") as f:
                self.paras = yaml.safe_load(f)
        else:
            logger.error("No config file found")
            raise FileNotFoundError("No config file found")

        self.domain_cols = self.paras["domain_cols"]

    def _load_trained_model(self):
        if self.paras is None:
            self.paras = {
                "domain_cols": self.domain_cols,
                "domain_lambda": self.domain_lambda,
                "batch_size": self.batch_size,
                "model_paras": {
                    "model_name": self.model_name,
                    "input_dim": self.dataset.feature_size,
                    "hidden_dim": self.hidden_dim,
                    "output_dim": self.dataset.num_classes,
                    "num_domains": self.dataset.num_domains,
                    "first_conv_out": self.first_conv_out,
                },
            }

        self.model = self._init_model(**self.paras["model_paras"])
        self.dataset = self._data_process(batch_size=self.paras["batch_size"])
        model_filename = Path(self.save_path, f"{self.feature}.pth")
        self.best_state_dict = torch.load(model_filename)
        self.model.load_state_dict(self.best_state_dict)
