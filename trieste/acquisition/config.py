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
""" This module contains functionality for building an acquisition step from config. """
from __future__ import annotations
from typing import Any, List, Mapping, NoReturn, Tuple, Union, cast

from trieste.acquisition.function import (
    ExpectedImprovement,
    NegativeLowerConfidenceBound,
    NegativePredictiveMean,
    ProbabilityOfFeasibility,
)
from trieste.acquisition.rule import (
    AcquisitionRule,
    EfficientGlobalOptimization,
    ThompsonSampling,
    TrustRegion,
)

_Args = Union[List[object], Tuple[object, ...]]
_Kwargs = Mapping[str, object]

AcquisitionConfig = Union[
    str, Tuple[str, _Args, _Kwargs], Tuple[str, str], Tuple[str, _Kwargs, str, _Args, _Kwargs]
]
""" Type alias for valid Schema for constructing an acquisition step from config. """


# todo what if we only support config for the single-model case atm? For those constructing more
#  complex problems, can we expect them to use more complex tooling?
def create_acquisition_rule(config: AcquisitionConfig) -> AcquisitionRule[Any, NoReturn]:
    """
    Construct and return an acquisition rule, and any encapsulated acquisition functions, from
    ``config``. ``config`` can take any of the following forms:

        - `(str, list|tuple, dict)`: build a rule, described by the `str`, which does not use an
          :class:`AcquisitionFunctionBuilder`. Constructs the rule with the proceeding positional
          and keyword arguments.
        - `str`: shorthand for `(str, [], {})`, i.e. uses default arguments.
        - `(str, dict, str, list|tuple, dict)`: build a rule described by the first `str`, with
          the proceeding keyword arguments. Positional arguments are not allowed for the rule, and
          the keyword arguments should omit the :class:`AcquisitionFunctionBuilder`. The rule will
          accept one :class:`AcquisitionFunctionBuilder`, which is described by the second `str` and
          is constructed with the proceeding positional and keyword arguments.
        - `(str, str)`: shorthand for `(str, {}, str, [], {})`, i.e. uses default arguments.

    The valid names for each rule are (case insensitive):

        - :class:`EfficientGlobalOptimization`: `"efficient_global_optimization"`, `"ego"`
        - :class:`TrustRegion`: `"trust_region"`, `"tr"`
        - :class:`ThompsonSampling`: `"thompson_sampling"`, `"ts"`

    and for each builder:

        - :class:`ExpectedImprovement`: `"expected_improvement"`, `"ei"`
        - :class:`NegativeLowerConfidenceBound`: `"negative_lower_confidence_bound"`, `"-lcb"`
        - :class:`NegativePredictiveMean`: `"negative_predictive_mean"`, `"-predictive_mean"`
        - :class:`ProbabilityOfFeasibility`: `"probability_of_feasibility"`, `"pof"`

    For example,

        >>> create_acquisition_rule(("EGO", "EI"))
        EfficientGlobalOptimization(ExpectedImprovement())
        >>> create_acquisition_rule(("thompson_sampling", [1000], {"num_query_points": 100}))
        ThompsonSampling(1000, 100)

    It's possible to create the same rule with different syntaxes by switching positional and
    keyword arguments, or utilising default arguments.

        >>> create_acquisition_rule(("trust_region", {"kappa": 1e-4}, "-lcb", [], {"beta": 1.96}))
        TrustRegion(NegativeLowerConfidenceBound(1.96), 0.7, 0.0001)
        >>> create_acquisition_rule(("trust_region", {"kappa": 1e-4}, "-lcb", [1.96], {}))
        TrustRegion(NegativeLowerConfidenceBound(1.96), 0.7, 0.0001)
        >>> create_acquisition_rule(("trust_region", {}, "-lcb", [1.96], {}))
        TrustRegion(NegativeLowerConfidenceBound(1.96), 0.7, 0.0001)
        >>> create_acquisition_rule(("trust_region", "-lcb"))
        TrustRegion(NegativeLowerConfidenceBound(1.96), 0.7, 0.0001)

    :param config: The configuration for the specified acquisition rule (and
        :class:`AcquisitionFunctionBuilder` if required).
    :return: An acquisition rule built from ``config``.
    :raise TypeError: If the arguments specified in ``config`` would raise `TypeError` when applied
        to the corresponding rule or function builder.
    :raise ValueError: If the acquisition specification cannot be parsed.
    """
    _EGO = ("efficient_global_optimization", "ego")
    _TRUST_REGION = ("trust_region", "tr")
    _THOMPSON_SAMPLING = ("thompson_sampling", "ts")

    _BUILDERS = [
        (("expected_improvement", "ei"), ExpectedImprovement),
        (("negative_lower_confidence_bound", "-lcb"), NegativeLowerConfidenceBound),
        (("negative_predictive_mean", "-predictive_mean"), NegativePredictiveMean),
        (("probability_of_feasibility", "pof"), ProbabilityOfFeasibility),
    ]

    if _matches_schema(config, str):
        config = (cast(str, config), (), {})

    if _matches_schema(config, (str, (list, tuple), dict)):
        rule_name, args, kwargs = cast(Tuple[str, _Args, _Kwargs], config)

        if rule_name.lower() in _THOMPSON_SAMPLING:
            return ThompsonSampling(*args, **kwargs)

    if _matches_schema(config, (str, str)):
        rule_name, builder_name = cast(Tuple[str, str], config)
        config = (rule_name, {}, builder_name, (), {})

    if _matches_schema(config, (str, dict, str, (list, tuple), dict)):
        rule_name, rule_kwargs, builder_name, builder_args, builder_kwargs = cast(
            Tuple[str, _Kwargs, str, _Args, _Kwargs], config
        )

        for allowed_builder_names, builder_type in _BUILDERS:
            if builder_name.lower() in allowed_builder_names:
                builder = builder_type(*builder_args, **builder_kwargs)

                if rule_name.lower() in _EGO:
                    return EfficientGlobalOptimization(builder=builder, **rule_kwargs)  # type: ignore
                if rule_name.lower() in _TRUST_REGION:
                    return TrustRegion(
                        builder=builder,
                        **rule_kwargs  # type: ignore
                    )

    raise ValueError(f"Unrecognised acquisition specification {config}")


def _matches_schema(
    config: AcquisitionConfig, spec: Union[type, Tuple[Union[type, Tuple[type, ...]], ...]]
) -> bool:
    if isinstance(spec, type):
        return isinstance(config, spec)

    if len(config) != len(spec):
        return False

    return all(isinstance(c, s) for c, s in zip(config, spec))
