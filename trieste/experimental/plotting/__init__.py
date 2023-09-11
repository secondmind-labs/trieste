# Copyright 2022 The Trieste Contributors
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
"""
Experimental plotting package. Not intended for production code, as it is not yet fully
tested and may change quickly.
"""

try:
    from .inequality_constraints import (
        plot_2obj_cst_query_points,
        plot_init_query_points,
        plot_objective_and_constraints,
    )
    from .plotting import (
        convert_figure_to_frame,
        convert_frames_to_gif,
        plot_acq_function_2d,
        plot_bo_points,
        plot_function_2d,
        plot_gp_2d,
        plot_mobo_history,
        plot_mobo_points_in_obj_space,
        plot_regret,
        plot_trust_region_history_2d,
    )
    from .plotting_plotly import (
        add_bo_points_plotly,
        plot_function_plotly,
        plot_model_predictions_plotly,
    )
except Exception as e:
    print(
        "trieste.experimental.plotting requires matplotlib and plotly to be installed."
        "\nOne way to do this is to install 'trieste[plotting]'."
    )
    raise e
