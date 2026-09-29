# Copyright 2026 Enactic, Inc.
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

import pytest

from dora_openarm_dataset_recorder.main import (
    data_path,
    legacy_frequencies,
    parse_frequencies,
    resolve_frequencies,
)

# What a real dataflow wires into the recorder.
INPUTS = {
    "command": "ui/command",
    "ker_metadata": "leader/metadata",
    "arm_right_action": "leader/follower_position_right",
    "arm_right_observation": "follower-right/state",
    "arm_left_action": "leader/follower_position_left",
    "arm_left_observation": "follower-left/state",
    "elevation_action": "lifter/elevation_action",
    "elevation_observation": "lifter/elevation_observation",
    "camera_wrist_right": {"source": "camera-wrist-right/image", "queue_size": 1},
    "camera_head_left": "camera-head-stereo-splitter/image_0",
}


@pytest.mark.parametrize(
    ("input_name", "expected"),
    [
        ("arm_right_action", "action/arms/right"),
        ("arm_left_observation", "obs/arms/left"),
        ("elevation_action", "action/lifter/elevation"),
        ("elevation_observation", "obs/lifter/elevation"),
        ("camera_head_left", "cameras/head_left"),
        ("command", None),
        ("ker_metadata", None),
    ],
)
def test_data_path(input_name, expected):
    assert data_path(input_name) == expected


def test_defaults_cover_every_recorded_stream():
    frequencies = resolve_frequencies(INPUTS)
    # Only the recorded streams, keyed by where their data is written.
    assert set(frequencies) == {
        "action/arms/right",
        "obs/arms/right",
        "action/arms/left",
        "obs/arms/left",
        "action/lifter/elevation",
        "obs/lifter/elevation",
        "cameras/wrist_right",
        "cameras/head_left",
    }
    assert frequencies["action/arms/right"] == {
        "nominal_hz": 250.0,
        "source": "default arms",
    }
    assert frequencies["obs/lifter/elevation"]["nominal_hz"] == 250.0
    assert frequencies["cameras/head_left"] == {
        "nominal_hz": 30.0,
        "source": "default cameras",
    }


def test_declared_group_overrides_default():
    frequencies = resolve_frequencies(INPUTS, {"arms": 500.0})
    assert frequencies["action/arms/left"] == {
        "nominal_hz": 500.0,
        "source": "declared arms",
    }
    assert frequencies["cameras/head_left"]["nominal_hz"] == 30.0


def test_declared_path_overrides_group():
    frequencies = resolve_frequencies(
        INPUTS, {"cameras": 30.0, "cameras/head_left": 15.0}
    )
    assert frequencies["cameras/head_left"] == {
        "nominal_hz": 15.0,
        "source": "declared",
    }
    assert frequencies["cameras/wrist_right"]["source"] == "declared cameras"


def test_parse_frequencies():
    assert parse_frequencies("{arms: 500, cameras/ceiling: 15}") == {
        "arms": 500.0,
        "cameras/ceiling": 15.0,
    }
    assert parse_frequencies(None) == {}
    assert parse_frequencies("") == {}


@pytest.mark.parametrize(
    "raw",
    [
        "250",  # not a map
        "{legs: 250}",  # unknown group
        "{arms: 0}",  # not positive
        "{arms: fast}",  # not a number
    ],
)
def test_parse_frequencies_rejects(raw):
    with pytest.raises(ValueError):
        parse_frequencies(raw)


def test_legacy_frequencies():
    legacy = legacy_frequencies(resolve_frequencies(INPUTS))
    assert legacy == {
        "action": {"arms": {"left": 250.0, "right": 250.0}, "lifter": 250.0},
        "obs": {"arms": {"left": 250.0, "right": 250.0}, "lifter": 250.0},
        "cameras": {"head_left": 30.0, "wrist_right": 30.0},
    }
