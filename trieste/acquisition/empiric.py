# Copyright 2020 The Trieste Contributors
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
from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Generic, TypeVar

from trieste.data import Dataset
from trieste.models import ProbabilisticModel

T = TypeVar("T")
T_co = TypeVar("T_co", covariant=True)


class Empiric(ABC, Generic[T_co]):
    """An :class:`Empiric` builds a value from historic data and models of that data."""

    @abstractmethod
    def acquire(
        self, datasets: Mapping[str, Dataset], models: Mapping[str, ProbabilisticModel]
    ) -> T_co:
        """
        :param datasets: The data from the observer.
        :param models: The models over each dataset in ``datasets``.
        :return: A value of type `T_co`.
        """


def unit(t: T) -> Empiric[T]:
    class _Anon(Empiric[T]):
        def acquire(
            self, datasets: Mapping[str, Dataset], models: Mapping[str, ProbabilisticModel]
        ) -> T:
            return t

    return _Anon()


class SingleModelEmpiric(ABC, Generic[T_co]):
    def using(self, tag: str) -> Empiric[T_co]:
        """
        :param tag: The tag for the model and dataset to use.
        :return: An :class:`Empiric` that selects the model and dataset specified by ``tag``, as
            defined in :meth:`acquire`.
        """
        single_model_empiric = self

        class _Anon(Empiric[T_co]):
            def acquire(
                self, datasets: Mapping[str, Dataset], models: Mapping[str, ProbabilisticModel]
            ) -> T_co:
                return single_model_empiric.acquire(datasets[tag], models[tag])

            def __repr__(self) -> str:
                return f"{single_model_empiric!r} using tag {tag!r}"

        return _Anon()

    @abstractmethod
    def acquire(self, dataset: Dataset, model: ProbabilisticModel) -> T_co:
        """
        :param dataset: The observer values.
        :param model: The model over the specified ``dataset``.
        :return: A value of type `T_co`.
        """
