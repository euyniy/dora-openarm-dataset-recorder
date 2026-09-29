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
    FrequencyDetector,
    detect_frequencies,
    parse_frequencies,
)

# The teleoperation shape every real dataflow has: one leader tick for the arms
# and the lifter, one camera tick for the cameras, both through a quitter so
# they stop on quit. The nodes that only take data inputs -- the splitter, IK,
# the lifter -- are given the tick that drives them, which is what makes their
# rate readable here.
CONFIGS = [
    {"id": "ui", "inputs": {"tick": "dora/timer/secs/1"}},
    {
        "id": "quittable-tick-leader",
        "inputs": {"command": "ui/command", "tick": "dora/timer/millis/4"},
    },
    {
        "id": "quittable-tick-camera",
        "inputs": {"command": "ui/command", "tick": "dora/timer/millis/33"},
    },
    {"id": "webxr", "inputs": {"tick": "quittable-tick-leader/tick"}},
    {
        # IK takes no tick: it solves when a target arrives, and its README
        # defines --tick-hz as the nominal rate of those targets.
        "id": "ik",
        "args": "--mode bimanual --tick-hz 250 --limit-velocity",
        "inputs": {
            "target_right": "webxr/pose_right",
            "target_left": "webxr/pose_left",
        },
    },
    {
        "id": "follower-right",
        "inputs": {
            "request_state": "ik/position_right",
            "move_position": "ik/position_right",
            "command": "ui/arm_command",
        },
    },
    {
        "id": "lifter",
        "inputs": {
            "tick": "quittable-tick-leader/tick",
            "joystick_y": "webxr/joystick_y_left",
        },
    },
    {
        "id": "camera-wrist-right",
        "inputs": {"tick": "quittable-tick-camera/tick"},
    },
    {
        "id": "camera-head-stereo",
        "inputs": {"tick": "quittable-tick-camera/tick"},
    },
    {
        "id": "camera-head-stereo-splitter",
        "inputs": {
            "tick": "quittable-tick-camera/tick",
            "image": "camera-head-stereo/image",
        },
    },
]

INPUTS = {
    "command": "ui/command",
    "ker_metadata": "leader/metadata",
    "arm_right_action": "ik/position_right",
    "arm_right_observation": "follower-right/state",
    "elevation_action": "lifter/elevation_action",
    "elevation_observation": "lifter/elevation_observation",
    "camera_wrist_right": {"source": "camera-wrist-right/image", "queue_size": 1},
    "camera_head_left": "camera-head-stereo-splitter/image_0",
}


def test_detect_through_quitter():
    detector = FrequencyDetector(CONFIGS)
    assert detector.detect("camera-wrist-right/image") == pytest.approx(1_000.0 / 33)


def test_detect_declared_tick_hz():
    detector = FrequencyDetector(CONFIGS)
    assert detector.detect("ik/position_right") == pytest.approx(250.0)


def test_detect_through_follower_and_ik():
    detector = FrequencyDetector(CONFIGS)
    # follower-right -> ik (request_state) -> ik's --tick-hz.
    assert detector.detect("follower-right/state") == pytest.approx(250.0)


def test_detect_through_splitter():
    detector = FrequencyDetector(CONFIGS)
    frequency = detector.detect("camera-head-stereo-splitter/image_0")
    assert frequency == pytest.approx(1_000.0 / 33)


def test_detect_lifter():
    detector = FrequencyDetector(CONFIGS)
    assert detector.detect("lifter/elevation_action") == pytest.approx(250.0)


def test_detect_dict_input():
    detector = FrequencyDetector(CONFIGS)
    frequency = detector.detect({"source": "camera-wrist-right/image", "queue_size": 1})
    assert frequency == pytest.approx(1_000.0 / 33)


def test_detect_ignores_other_args():
    detector = FrequencyDetector([{"id": "ik", "args": "--mode bimanual --dt 0.1"}])
    assert detector.detect("ik/position_right") is None


def test_detect_node_without_tick():
    # A node that states no rate: the dataflow has to wire a tick to it.
    detector = FrequencyDetector(
        [{"id": "splitter", "inputs": {"image": "camera/image"}}]
    )
    assert detector.detect("splitter/image_0") is None


def test_detect_node_without_inputs():
    detector = FrequencyDetector([{"id": "webxr"}])
    assert detector.detect("webxr/pose_right") is None


def test_detect_unknown_node():
    detector = FrequencyDetector(CONFIGS)
    assert detector.detect("nowhere/image") is None


def test_detect_frequencies():
    frequencies, unknown = detect_frequencies(INPUTS, CONFIGS)
    assert unknown == []
    assert frequencies == {
        "action": {"arms": {"right": 250.0}, "lifter": 250.0},
        "obs": {"arms": {"right": 250.0}, "lifter": 250.0},
        # 1000/33 is 30.303030303030305 without the rounding.
        "cameras": {"wrist_right": 30.303, "head_left": 30.303},
    }


def test_detect_frequencies_reports_unknown():
    configs = [config for config in CONFIGS if config["id"] != "lifter"]
    configs.append({"id": "lifter", "inputs": {"joystick_y": "webxr/joystick_y_left"}})
    frequencies, unknown = detect_frequencies(INPUTS, configs)
    assert sorted(unknown) == ["elevation_action", "elevation_observation"]
    assert "lifter" not in frequencies["action"]
    assert "lifter" not in frequencies["obs"]


def test_override_wins():
    # The MuJoCo node answers state requests at the leader tick but renders its
    # cameras at 30 Hz, so its camera rate can only be stated by the dataflow.
    configs = [
        {"id": "ui", "inputs": {"tick": "dora/timer/secs/1"}},
        {
            "id": "quittable-tick-leader",
            "inputs": {"command": "ui/command", "tick": "dora/timer/millis/4"},
        },
        {
            "id": "mujoco-collect",
            "inputs": {"request_state": "quittable-tick-leader/tick"},
        },
    ]
    inputs = {
        "arm_right_observation": "mujoco-collect/state_right",
        "camera_ceiling": "mujoco-collect/camera_ceiling",
    }
    frequencies, unknown = detect_frequencies(inputs, configs, {"camera_ceiling": 30.0})
    assert unknown == []
    assert frequencies["obs"]["arms"]["right"] == pytest.approx(250.0)
    assert frequencies["cameras"]["ceiling"] == 30.0


def test_parse_frequencies():
    assert parse_frequencies("{camera_ceiling: 30}") == {"camera_ceiling": 30.0}
    assert parse_frequencies(None) == {}
    assert parse_frequencies("") == {}


@pytest.mark.parametrize("raw", ["30", "{camera_ceiling: 0}", "{camera_ceiling: fast}"])
def test_parse_frequencies_rejects(raw):
    with pytest.raises(ValueError):
        parse_frequencies(raw)
