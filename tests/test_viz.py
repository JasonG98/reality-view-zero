"""viz-episode 的图片导出与位姿投影叠加测试。"""

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from rvz.config import SimConfig
from rvz.data.viz import viz_episode
from rvz.sim.record import generate_episode


def test_viz_episode_exports_context_and_reprojection(tmp_path: Path) -> None:
    cfg = SimConfig(robot_res=48, drone_res=48, steps=1, n_drones=1)
    episode = generate_episode(9, cfg)
    episode_path = tmp_path / "ep_000009.npz"
    np.savez_compressed(episode_path, **episode)
    outputs = viz_episode(episode_path, tmp_path / "viz")
    assert set(outputs) == {"context", "drone", "reprojection", "overview"}
    for path in outputs.values():
        assert path.is_file()
        with Image.open(path) as image:
            image.verify()
    with Image.open(outputs["drone"]) as drone, Image.open(outputs["reprojection"]) as overlay:
        assert drone.size == (48, 48)
        assert overlay.size == (256, 256)
        enlarged = drone.resize(overlay.size, Image.Resampling.NEAREST)
        assert np.any(np.asarray(enlarged) != np.asarray(overlay))


def test_viz_episode_rejects_out_of_range_indices(tmp_path: Path) -> None:
    cfg = SimConfig(robot_res=48, drone_res=48, steps=1, n_drones=1)
    episode_path = tmp_path / "episode.npz"
    np.savez_compressed(episode_path, **generate_episode(10, cfg))
    with pytest.raises(ValueError, match="step"):
        viz_episode(episode_path, tmp_path / "bad", step=1)
    with pytest.raises(ValueError, match="drone_index"):
        viz_episode(episode_path, tmp_path / "bad", drone_index=1)
    with np.load(episode_path, allow_pickle=False) as loaded:
        episode = {name: loaded[name] for name in loaded.files}
    meta = json.loads(str(episode["meta_json"].item()))
    meta.pop("max_reprojection_error_px")
    episode["meta_json"] = json.dumps(meta)
    np.savez_compressed(episode_path, **episode)
    outputs = viz_episode(episode_path, tmp_path / "without_optional_stats")
    assert outputs["reprojection"].is_file()
