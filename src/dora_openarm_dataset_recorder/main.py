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

"""Node to record data from OpenArm and cameras as OpenArm dataset."""

import argparse
from dataclasses import dataclass, field
import datetime
import copy
import dora
import json
import os
import pathlib
import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np
import math
from numpy.typing import ArrayLike
import shutil
import sys
import yaml


@dataclass
class Episode:
    """Episode related data."""

    number: int = 0
    success: bool = False
    task_index: int = 0

    right_action_timestamps: ArrayLike = field(default_factory=list)
    right_actions: ArrayLike = field(default_factory=list)
    right_observation_timestamps: ArrayLike = field(default_factory=list)
    right_observations: ArrayLike = field(default_factory=list)
    left_action_timestamps: ArrayLike = field(default_factory=list)
    left_actions: ArrayLike = field(default_factory=list)
    left_observation_timestamps: ArrayLike = field(default_factory=list)
    left_observations: ArrayLike = field(default_factory=list)
    elevation_action_timestamps: ArrayLike = field(default_factory=list)
    elevation_actions: ArrayLike = field(default_factory=list)
    elevation_observation_timestamps: ArrayLike = field(default_factory=list)
    elevation_observations: ArrayLike = field(default_factory=list)


def extract_values(value: pa.Array, key: str) -> np.ndarray:
    """Read `key` from a length-1 StructArray, or a flat array as-is."""
    if pa.types.is_struct(value.type):
        value = value.field(key)[0].values
    return np.array(value, dtype=np.float32)


class EpisodeWriter:
    """Writer an episode."""

    def __init__(self, directory, episode):
        """Initialize variables."""
        self._directory = directory
        self._episode = episode
        self._base_directory = self._directory / "episodes" / str(self._episode.number)

    def write_camera_image(self, name, image, timestamp, format):
        """Write an image from a camera."""
        output_path = self._base_directory / "cameras" / name / f"{timestamp}.{format}"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("wb") as output:
            # Using pa.PythonFile here is for zero-copy. We can't
            # write pa.Buffer data to Python's IO directory. If we use
            # Python's IO, we need to copy data in pa.Buffer as
            # Python's bytes. We want to avoid it.
            with pa.PythonFile(output) as pa_output:
                pa_output.write(image.buffers()[1])

    def finish(self):
        """Write all pending data."""
        if self._episode.right_actions:
            self._write_kinematic_state(
                self._base_directory / "action" / "arms" / "right",
                self._episode.right_action_timestamps,
                self._episode.right_actions,
            )
        if self._episode.right_observations:
            self._write_kinematic_state(
                self._base_directory / "obs" / "arms" / "right",
                self._episode.right_observation_timestamps,
                self._episode.right_observations,
            )
        if self._episode.left_actions:
            self._write_kinematic_state(
                self._base_directory / "action" / "arms" / "left",
                self._episode.left_action_timestamps,
                self._episode.left_actions,
            )
        if self._episode.left_observations:
            self._write_kinematic_state(
                self._base_directory / "obs" / "arms" / "left",
                self._episode.left_observation_timestamps,
                self._episode.left_observations,
            )
        if self._episode.elevation_actions:
            self._write_positions(
                self._base_directory / "action" / "lifter" / "elevation.parquet",
                self._episode.elevation_action_timestamps,
                self._episode.elevation_actions,
            )
        if self._episode.elevation_observations:
            self._write_positions(
                self._base_directory / "obs" / "lifter" / "elevation.parquet",
                self._episode.elevation_observation_timestamps,
                self._episode.elevation_observations,
            )

    def cancel(self):
        """Cancel this episode."""
        shutil.rmtree(self._base_directory, ignore_errors=True)

    def _write_positions(self, output_path, timestamps, positions):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        list_type = pa.list_(pa.float32())
        table = pa.table(
            {
                "timestamp": pa.array(timestamps, type=pa.timestamp("ns")),
                "value": pa.array(positions, type=list_type),
            }
        )
        pq.write_table(table, output_path)

    def _write_kinematic_state(self, base_path, timestamps, states):
        output_path = base_path / "state.parquet"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        list_type = pa.list_(pa.float32())
        first = states[0]

        if isinstance(first, pa.StructArray):
            field_names = first.type.names
            available_field_names = [
                "qpos",
                "qvel",
                "qtorque",
                "pose",
            ]  # currently only support these fields
            state_fields = {"timestamp": pa.array(timestamps, type=pa.timestamp("ns"))}
            for field_name in field_names:
                if field_name not in available_field_names:
                    continue
                state_fields[field_name] = pa.array(
                    [extract_values(s, field_name) for s in states],
                    type=list_type,
                )
            table = pa.table(state_fields)
        else:
            # if the observation is not a struct, it should be qpos.
            table = pa.table(
                {
                    "timestamp": pa.array(timestamps, type=pa.timestamp("ns")),
                    "qpos": pa.array(states, type=list_type),
                }
            )
        pq.write_table(table, output_path)


class DatasetWriter:
    """Write a dataset."""

    _VERSION = "0.4.0"

    def __init__(self, directory, name, metadata):
        """Initialize variables."""
        self._directory = directory
        self._name = name
        self._metadata = metadata
        self._base_directory = self._directory / name
        self._episode_results = []
        if self._base_directory.exists():
            existing = self._read_existing_metadata()
            if existing is not None:
                mismatched = {
                    k: v
                    for k, v in self._metadata.items()
                    if k in existing and existing[k] != v
                }
                if mismatched:
                    raise ValueError(
                        f"Existing dataset metadata does not match: {self._base_directory / 'metadata.yaml'}\n"
                        + "\n".join(
                            f"  {k}: existing={existing.get(k)!r}, current={v!r}"
                            for k, v in mismatched.items()
                        )
                    )
        else:
            self._base_directory.mkdir(parents=True)

    def create_episode_writer(self, episode):
        """Create a writer for the given episode."""
        episode_id = str(episode.number)
        if episode_id in {result["id"] for result in self._episode_results}:
            raise ValueError(f"Episode {episode_id} already exists in dataset")
        episode_directory = self._base_directory / "episodes" / episode_id
        if episode_directory.exists():
            raise ValueError(f"Episode directory already exists: {episode_directory}")
        return EpisodeWriter(self._base_directory, episode)

    def finish_episode(self, episode):
        """Add a finished episode to the writer."""
        self._episode_results.append(
            dict(
                id=str(episode.number),
                success=episode.success,
                task_index=episode.task_index,
            )
        )
        self._write_metadata_file()

    def set_leader_ker_metadata(self, ker_metadata):
        """Record KER leader device metadata under equipment.leader.ker."""
        equipment = self._metadata.setdefault("equipment", {})
        leader = equipment.setdefault("leader", {})
        ker = leader.setdefault("ker", {})
        ker["id"] = "OpenArmKER"
        ker["firmware_version"] = ker_metadata.get("fw")
        ker["hardware_version"] = ker_metadata.get("hw")
        self._write_metadata_file()

    def _write_metadata_file(self):
        metadata = copy.deepcopy(self._metadata)
        metadata["version"] = self._VERSION
        metadata["episodes"] = self._episode_results
        output_path = self._base_directory / "metadata.yaml"
        with open(output_path, "w", encoding="utf-8") as f:
            yaml.dump(metadata, f, default_flow_style=False, allow_unicode=True)

    def _read_existing_metadata(self):
        metadata_path = self._base_directory / "metadata.yaml"
        if not metadata_path.exists():
            return None
        with open(metadata_path, encoding="utf-8") as f:
            existing_metadata = yaml.safe_load(f) or {}
        self._episode_results = existing_metadata.get("episodes", [])
        return existing_metadata


class FrequencyDetector:
    """Detect the nominal frequency of an input from the dataflow.

    A node states its rate in one of two ways: it takes a timer, directly or
    through a quitter, on an input named ``tick``, ``request_state`` or
    ``request_position``; or, when it is event-driven and takes no tick at all,
    it declares ``--tick-hz`` in its arguments (the IK node documents that
    argument as its nominal rate). A node that states neither is walked no
    further: wire a tick to it in the dataflow rather than guessing here.

    The frequency is the rate the dataflow asks for, not the rate the data
    arrived at; what a camera actually delivered is in its image timestamps.
    """

    _RATE_INPUTS = ("tick", "request_state", "request_position")

    def __init__(self, configs):
        """Initialize with node configurations."""
        self._configs = {config["id"]: config for config in configs}

    def detect(self, input):
        """Detect frequency of the given input, or ``None`` if it has no rate."""
        node_id = self._source(input).split("/", 1)[0]
        config = self._configs.get(node_id)
        if config is None:
            return None
        inputs = config.get("inputs") or {}
        for name in self._RATE_INPUTS:
            if name in inputs:
                next_input = self._source(inputs[name])
                if next_input.startswith("dora/timer/"):
                    return self._timer_frequency(next_input)
                return self.detect(next_input)
        return self._declared_frequency(config)

    def _declared_frequency(self, config):
        args = (config.get("args") or "").split()
        if "--tick-hz" not in args:
            return None
        value = args[args.index("--tick-hz") + 1]
        return float(value)

    def _source(self, input):
        # An input is either "node/output" or {source: ..., queue_size: ...}.
        if isinstance(input, dict):
            return input["source"]
        return input

    def _timer_frequency(self, timer):
        unit, value = timer.split("/")[2:4]
        if unit == "secs":
            return 1.0 / int(value)
        elif unit == "millis":
            return 1_000.0 / int(value)
        return None


def parse_frequencies(raw):
    """Parse the ``FREQUENCIES`` setting: a YAML map of input name to Hz.

    For streams whose rate the dataflow cannot state, because the node that
    sends them throttles internally (the MuJoCo node renders its cameras at
    30 Hz while answering state requests at the leader tick).
    """
    if not raw:
        return {}
    frequencies = yaml.safe_load(raw)
    if not isinstance(frequencies, dict):
        raise ValueError(f"FREQUENCIES must be a map of input name to Hz: {raw!r}")
    for name, value in frequencies.items():
        if not isinstance(value, (int, float)) or value <= 0:
            raise ValueError(f"FREQUENCIES[{name!r}] must be positive: {value!r}")
    return {name: float(value) for name, value in frequencies.items()}


def detect_frequencies(inputs, configs, overrides=None):
    """Return the frequencies of the recorder's inputs, in the dataset shape.

    ``action.arms.<side>`` / ``obs.arms.<side>`` / ``action.lifter`` /
    ``obs.lifter`` / ``cameras.<name>``, in Hz. An input whose rate is neither
    detected nor overridden is left out, and reported in ``unknown``.
    """
    detector = FrequencyDetector(configs)
    frequencies = {"action": {"arms": {}}, "obs": {"arms": {}}, "cameras": {}}
    overrides = overrides or {}
    unknown = []
    for name in inputs:
        if name in overrides:
            frequency = overrides[name]
        else:
            frequency = detector.detect(inputs[name])
        if name.startswith("arm_"):
            # arm_right_action -> right, action
            side, type = name.split("_")[1:3]
            slot, key = frequencies[TYPES[type]]["arms"], side
        elif name.startswith("elevation_"):
            # elevation_action -> action, lifter
            type = name.removeprefix("elevation_")
            slot, key = frequencies[TYPES[type]], "lifter"
        elif name.startswith("camera_"):
            # camera_wrist_right -> wrist_right
            slot, key = frequencies["cameras"], name.removeprefix("camera_")
        else:
            continue
        if frequency is None:
            unknown.append(name)
            continue
        slot[key] = round(frequency, 3)
    return frequencies, unknown


TYPES = {"action": "action", "observation": "obs"}


def _collect_dynamic_metadata(metadata, args, node):
    metadata["operation_type"] = args.operation_type
    if args.operation_type == "teleop":
        if "equipment" not in metadata:
            metadata["equipment"] = {}
        if "embodiments" not in metadata["equipment"]:
            metadata["equipment"]["embodiments"] = {}
        # equipment.leader.ker is filled at runtime from the KER node's
        # metadata reply (see main()'s "ker_metadata" handling).
    elif args.operation_type == "rollout":
        if "model" not in metadata:
            metadata["model"] = {}
        if args.docker_image:
            metadata["model"]["docker_image"] = args.docker_image

    frequencies, unknown = detect_frequencies(
        node.node_config()["inputs"],
        node.dataflow_descriptor()["nodes"],
        parse_frequencies(args.frequencies),
    )
    for name in unknown:
        print(
            f"Warning: no frequency for {name}: no tick above it. Wire a tick "
            "to the node that sends it, or set FREQUENCIES.",
            file=sys.stderr,
        )
    metadata["frequencies"] = frequencies


def main():
    """Collect data and record them."""
    parser = argparse.ArgumentParser(description="Record data as OpenArm dataset")
    parser.add_argument(
        "--directory",
        default=os.getenv("DIRECTORY", os.getcwd()),
        help="The output directory",
        type=pathlib.Path,
    )
    parser.add_argument(
        "--docker-image",
        default=os.getenv("DOCKER_IMAGE", os.getenv("IMAGE")),
        help="The Docker image used for this rollout",
        type=str,
    )
    parser.add_argument(
        "--frequencies",
        default=os.getenv("FREQUENCIES"),
        help="YAML map of data path to frequency in Hz, for streams the "
        'dataflow cannot be walked for (e.g. "cameras/ceiling: 30")',
        type=str,
    )
    parser.add_argument(
        "--metadata-file",
        default=os.getenv("METADATA_FILE"),
        help="The metadata file",
        type=pathlib.Path,
    )
    parser.add_argument(
        "--name",
        default=os.getenv("NAME", "dataset"),
        help="The dataset name",
        type=str,
    )
    parser.add_argument(
        "--operation-type",
        choices=["teleop", "rollout"],
        default=os.getenv("OPERATION_TYPE", "teleop"),
        help="The operation type",
        type=str,
    )
    args = parser.parse_args()

    node = dora.Node()
    if args.metadata_file is None:
        metadata = {}
    else:
        with open(args.metadata_file, encoding="utf-8") as f:
            metadata = yaml.safe_load(f)
    _collect_dynamic_metadata(metadata, args, node)
    dataset_writer = DatasetWriter(args.directory, args.name, metadata)
    episode = None
    episode_writer = None
    arm_observation_timestamp_key = None

    for event in node:
        if event["type"] != "INPUT":
            continue

        event_id = event["id"]
        if event_id == "command":
            command = event["value"][0].as_py()
            if command == "start":
                episode = Episode()
                episode.number = event["metadata"].get("episode_number", 0)
                episode.task_index = event["metadata"].get("task_index", 0)
                episode_writer = dataset_writer.create_episode_writer(episode)
            elif command in ("success", "fail"):
                if command == "success":
                    episode.success = True
                episode_writer.finish()
                dataset_writer.finish_episode(episode)
                episode = None
                episode_writer = None
            elif command == "cancel":
                episode_writer.cancel()
                episode = None
                episode_writer = None
            elif command == "quit":
                if episode is not None:
                    episode_writer.finish()
                    dataset_writer.finish_episode(episode)
                    episode = None
                    episode_writer = None
                break
            continue

        if event_id == "ker_metadata":
            # KER leader device metadata (JSON) from the KER node.
            ker_metadata = json.loads(event["value"][0].as_py())
            dataset_writer.set_leader_ker_metadata(ker_metadata)
            continue

        timestamp_key = "timestamp"
        if event_id in ("arm_right_observation", "arm_left_observation"):
            if arm_observation_timestamp_key is None:
                arm_observation_timestamp_key = (
                    "observation_timestamp"
                    if "observation_timestamp" in event["metadata"]
                    else "timestamp"
                )
                print(
                    f"Arm observation timestamp field: {arm_observation_timestamp_key}",
                    flush=True,
                )
            timestamp_key = arm_observation_timestamp_key
            if timestamp_key not in event["metadata"]:
                raise ValueError(
                    f"{event_id} is missing locked timestamp field {timestamp_key!r}"
                )

        # Main process
        if episode is None:
            continue
        timestamp = event["metadata"][timestamp_key]
        if isinstance(timestamp, datetime.datetime):
            # Added by dora-rs automatically.
            # Convert to POSIX timestamp in nanosecond.
            timestamp = math.ceil(timestamp.timestamp() * 1_000_000_000)
        if event_id.startswith("arm_"):
            value = event["value"]
            if isinstance(value, pa.StructArray) and "new_position" in value.type.names:
                value = value.field("new_position")

            # arm_right_action ->
            # right_action
            key_prefix = event_id.removeprefix("arm_")
            # right_action ->
            # right_actions
            values_key = f"{key_prefix}s"
            getattr(episode, values_key).append(value)
            # right_action ->
            # right_action_timestamps
            timestamps_key = f"{key_prefix}_timestamps"
            getattr(episode, timestamps_key).append(timestamp)
        elif event_id.startswith("elevation_"):
            # elevation_observation -> elevation_observations, elevation_observation_timestamps
            # elevation_action -> elevation_actions, elevation_action_timestamps
            getattr(episode, f"{event_id}s").append(event["value"])
            getattr(episode, f"{event_id}_timestamps").append(timestamp)
        elif event_id.startswith("camera_"):
            name = event_id.removeprefix("camera_")
            image = event["value"]
            format = event["metadata"]["encoding"]
            episode_writer.write_camera_image(name, image, timestamp, format)


if __name__ == "__main__":
    main()
