# Copyright 2021 The Trieste Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from typing import Any, Dict, Callable

import torch as t
import torch.nn as nn
from torch.utils.data import TensorDataset

import bayesfunc as bf

import tensorflow as tf

from ...data import Dataset
from ...types import TensorType
from ..interfaces import TrainableProbabilisticModel
from .interface import BayesFuncPredictor


class BayesFuncModel(BayesFuncPredictor, TrainableProbabilisticModel):
    def __init__(
        self,
        model: nn.Module,
        fit_args: Dict[Any] | None = None,
        dtype=t.float64,
        device: str = "cpu",
        num_train_samples: int = 10,
    ):
        super().__init__()

        self._model = model

        if fit_args is None:
            self.fit_args = dict({
                "verbose": True,
                "lr": 0.01,
                "epochs": 400,
                "batch_size": 100,
                "scheduler": True
            })
        else:
            self.fit_args = fit_args

        self.num_data = None

        self.dtype = dtype
        self.device = device

        self.num_train_samples = num_train_samples

    @property
    def model(self) -> nn.Module:
        return self._model

    def update(self, dataset: Dataset) -> None:
        self.num_data = dataset.query_points.shape[0]

    def optimize(self, dataset: Dataset) -> None:
        bf.set_full_cov(self.model, False)

        query_points = dataset.query_points
        if isinstance(query_points, tf.Tensor):
            query_points = query_points.numpy()
        observations = dataset.observations
        if isinstance(observations, tf.Tensor):
            observations = observations.numpy()

        X = t.from_numpy(query_points).to(device=self.device, dtype=self.dtype)
        y = t.from_numpy(observations).to(device=self.device, dtype=self.dtype)

        dataset = TensorDataset(X, y)

        trainloader = t.utils.data.DataLoader(
            dataset,
            batch_size=self.fit_args["batch_size"],
            shuffle=True,
            num_workers=0
        )

        opt = t.optim.Adam(self.model.parameters(), lr=self.fit_args.get("lr"))

        if self.fit_args.get("scheduler"):
            scheduler = t.optim.lr_scheduler.MultiStepLR(
                opt,
                milestones=[self.fit_args.get("epochs")//2]
            )
        else:
            scheduler = None

        for epoch in range(self.fit_args["epochs"]):
            total_elbo = 0.
            for data, target in trainloader:
                opt.zero_grad()
                data = data.expand(self.num_train_samples, *data.shape)
                output, logpq, _ = bf.propagate(self.model, data)

                assert target.shape == output.loc.shape[1:]

                ll = output.log_prob(target.unsqueeze(0)).mean(0).sum()
                elbo = ll + logpq.mean()*target.shape[0] / self.num_data

                (-elbo).backward()
                opt.step()

                total_elbo += elbo.detach().cpu().item()

            if scheduler is not None:
                scheduler.step()

            if self.fit_args.get("verbose"):
                print(f'Epoch: {epoch}/{self.fit_args["epochs"]}, ELBO: {total_elbo}')

    def sample(self, query_points: TensorType, num_samples: int) -> TensorType:
        with t.no_grad():
            bf.set_full_cov(self.model, True)

            if isinstance(query_points, tf.Tensor):
                query_points = query_points.numpy()

            X = t.from_numpy(query_points).to(device=self.device, dtype=self.dtype)

            ys, _, _ = bf.propagate(self.model, X.expand(num_samples, *X.shape))

            samples_np = ys.loc.detach().cpu().numpy()

            return samples_np
