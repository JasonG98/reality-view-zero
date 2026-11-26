"""仓库包围盒布局、AABB 占用栅格与平滑 A* 路径。"""

import heapq
from dataclasses import dataclass
from itertools import pairwise

import numpy as np

from rvz.config import SimConfig


@dataclass(frozen=True)
class Box:
    """轴对齐基本体；center/size 均为全尺寸米制。"""

    name: str
    kind: str
    center: np.ndarray
    size: np.ndarray
    rgba: np.ndarray


@dataclass(frozen=True)
class Light:
    position: np.ndarray
    direction: np.ndarray
    diffuse: np.ndarray


@dataclass
class Layout:
    seed: int
    width: float
    depth: float
    boxes: tuple[Box, ...]
    lights: tuple[Light, ...]
    grid_resolution: float
    robot_radius: float
    occupied: np.ndarray
    min_path_length: float


def required_path_length(cfg: SimConfig) -> float:
    """小车走完 cfg.steps 步所需的最短路径长度。"""
    return (cfg.steps - 1) * cfg.robot_speed * cfg.dt


def occupied_grid(
    width: float, depth: float, resolution: float, robot_radius: float, boxes: tuple[Box, ...]
) -> np.ndarray:
    """按规划安全半径膨胀场景包围盒，得到二维占用栅格。"""
    nx, ny = int(width / resolution), int(depth / resolution)
    xs = (np.arange(nx) + 0.5) * resolution - width / 2.0
    ys = (np.arange(ny) + 0.5) * resolution - depth / 2.0
    xx, yy = np.meshgrid(xs, ys)
    occupied = np.zeros((ny, nx), dtype=bool)
    for box in boxes:
        half = box.size / 2.0
        inside = (np.abs(xx - box.center[0]) <= half[0] + robot_radius) & (
            np.abs(yy - box.center[1]) <= half[1] + robot_radius
        )
        occupied |= inside
    return occupied


def validate_layout(layout: Layout) -> None:
    """检查几何体有效性、空闲栅格一致性与可导航性。"""
    names = [box.name for box in layout.boxes]
    if len(names) != len(set(names)):
        raise ValueError("布局几何体名称不唯一")
    if layout.width <= 0.0 or layout.depth <= 0.0:
        raise ValueError("仓库尺寸必须为正")
    for box in layout.boxes:
        if np.any(box.size <= 0.0) or not np.all(np.isfinite(box.center)) or not np.all(np.isfinite(box.size)):
            raise ValueError(f"{box.name} 的尺寸或中心无效")
    expected = occupied_grid(layout.width, layout.depth, layout.grid_resolution, layout.robot_radius, layout.boxes)
    if not np.array_equal(layout.occupied, expected):
        raise ValueError("空闲栅格与几何体 AABB 不一致")
    if int((~layout.occupied).sum()) < 2:
        raise ValueError("布局没有足够的空闲栅格")


_SQRT2 = float(np.sqrt(2.0))
_MIN_START_REAR_CLEARANCE_M = 0.68
_REAR_CLEARANCE_MARGIN_M = 0.06
_REAR_OBSTACLE_MIN_TOP_M = 0.20
_ASTAR_DIRS = (
    (0, 1, 1.0),
    (1, 0, 1.0),
    (0, -1, 1.0),
    (-1, 0, 1.0),
    (1, 1, _SQRT2),
    (1, -1, _SQRT2),
    (-1, 1, _SQRT2),
    (-1, -1, _SQRT2),
)


def _astar(layout: Layout, start: tuple[int, int], goal: tuple[int, int]) -> list[tuple[int, int]] | None:
    """8 邻接 A*（禁止斜穿障碍物角点）。"""
    occupied = layout.occupied
    if occupied[start] or occupied[goal]:
        return None
    rows, cols = occupied.shape
    frontier: list[tuple[float, float, tuple[int, int]]] = [(0.0, 0.0, start)]
    costs = {start: 0.0}
    parents: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
    while frontier:
        _, cost, current = heapq.heappop(frontier)
        if current == goal:
            path = []
            cursor: tuple[int, int] | None = current
            while cursor is not None:
                path.append(cursor)
                cursor = parents[cursor]
            return path[::-1]
        if cost > costs[current] + 1e-9:
            continue
        row, col = current
        for dy, dx, step_cost in _ASTAR_DIRS:
            next_row, next_col = row + dy, col + dx
            if not (0 <= next_row < rows and 0 <= next_col < cols) or occupied[next_row, next_col]:
                continue
            if dy != 0 and dx != 0 and (occupied[row + dy, col] or occupied[row, col + dx]):
                continue
            new_cost = cost + step_cost
            neighbor = (next_row, next_col)
            if new_cost >= costs.get(neighbor, 1e18):
                continue
            costs[neighbor] = new_cost
            parents[neighbor] = current
            dr, dc = abs(goal[0] - next_row), abs(goal[1] - next_col)
            heuristic = max(dr, dc) + (_SQRT2 - 1.0) * min(dr, dc)
            heapq.heappush(frontier, (new_cost + heuristic, new_cost, neighbor))
    return None


def _cell_world(layout: Layout, cell: tuple[int, int]) -> np.ndarray:
    return np.array(
        [
            (cell[1] + 0.5) * layout.grid_resolution - layout.width / 2.0,
            (cell[0] + 0.5) * layout.grid_resolution - layout.depth / 2.0,
        ]
    )


def _shortcut_segment(layout: Layout, points: np.ndarray) -> np.ndarray:
    """用视线剪枝消除栅格锯齿，只保留必要的绕障折点。"""
    simplified = [points[0]]
    index = 0
    count = len(points)
    while index < count - 1:
        far = count - 1
        while far > index + 1:
            if path_is_free(layout, np.vstack([points[index], points[far]]).astype(np.float32)):
                break
            far -= 1
        simplified.append(points[far])
        index = far
    return np.asarray(simplified, dtype=np.float64)


def _arc_is_free(layout: Layout, arc: np.ndarray) -> bool:
    arc_f32 = arc.astype(np.float32)
    if not path_is_free(layout, arc_f32):
        return False
    # 同步检查跨步弦，避免后续等距采样在贴障弯道处擦角穿格。
    for stride in (2, 4):
        sub = arc_f32[::stride]
        if len(sub) >= 2 and not path_is_free(layout, sub):
            return False
    return path_is_free(layout, np.vstack([arc_f32[0], arc_f32[len(arc_f32) // 2], arc_f32[-1]]))


def _smooth_path(layout: Layout, vertices: np.ndarray) -> np.ndarray:
    """用无碰撞二次 Bézier 圆角平滑折线转角，消除 90°/180° 航向跳变。"""
    deduped = [vertices[0]]
    for point in vertices[1:]:
        if np.linalg.norm(point - deduped[-1]) >= 1e-5:
            deduped.append(point)
    if len(deduped) <= 2:
        return np.asarray(deduped, dtype=np.float32)

    pts = np.asarray(deduped, dtype=np.float64)
    out = [pts[0]]
    for idx in range(1, len(pts) - 1):
        prev_pt, cur_pt, next_pt = pts[idx - 1], pts[idx], pts[idx + 1]
        vec_in, vec_out = cur_pt - prev_pt, next_pt - cur_pt
        len_in, len_out = float(np.linalg.norm(vec_in)), float(np.linalg.norm(vec_out))
        if len_in < 1e-5 or len_out < 1e-5:
            continue
        dir_in, dir_out = vec_in / len_in, vec_out / len_out
        if float(np.clip(np.dot(dir_in, dir_out), -1.0, 1.0)) > 0.9985:
            out.append(cur_pt)
            continue
        max_offset = min(0.45 * len_in, 0.45 * len_out, 0.75)
        placed = False
        for scale in (1.0, 0.7, 0.45, 0.25):
            offset = max_offset * scale
            if offset < 0.12:
                continue
            start_pt = cur_pt - offset * dir_in
            end_pt = cur_pt + offset * dir_out
            ts = np.linspace(0.0, 1.0, max(9, int(np.ceil(2.0 * offset / 0.025)) + 1))[:, None]
            arc = (1.0 - ts) ** 2 * start_pt + 2.0 * (1.0 - ts) * ts * cur_pt + ts**2 * end_pt
            if _arc_is_free(layout, arc):
                out.extend(arc)
                placed = True
                break
        if not placed:
            out.append(cur_pt)
    out.append(pts[-1])

    compact = [out[0]]
    for point in out[1:]:
        if np.linalg.norm(point - compact[-1]) > 1e-4:
            compact.append(point)
    return np.asarray(compact, dtype=np.float32)


def _path_length(path: np.ndarray) -> float:
    return float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum()) if len(path) >= 2 else 0.0


def _max_turn_rad(path: np.ndarray) -> float:
    if len(path) < 3:
        return 0.0
    segs = np.diff(path.astype(np.float64), axis=0)
    norms = np.linalg.norm(segs, axis=1, keepdims=True)
    dirs = segs / np.maximum(norms, 1e-12)
    dots = np.clip(np.sum(dirs[:-1] * dirs[1:], axis=1), -1.0, 1.0)
    return float(np.arccos(dots).max())


def _rear_clearance_ok(layout: Layout, origin: np.ndarray, heading: np.ndarray) -> bool:
    """确保起点正后方留有足够空间，避免低机位跟拍相机在首步被墙或货架逼出视野。"""
    samples = origin[None, :] - np.linspace(0.0, _MIN_START_REAR_CLEARANCE_M, 12)[:, None] * heading[None, :]
    for box in layout.boxes:
        if float(box.center[2] + box.size[2] / 2.0) <= _REAR_OBSTACLE_MIN_TOP_M:
            continue
        half_xy = box.size[:2] / 2.0 + _REAR_CLEARANCE_MARGIN_M
        if bool(np.all(np.abs(samples - box.center[:2]) <= half_xy, axis=1).any()):
            return False
    return True


def plan_path(layout: Layout, rng: np.random.Generator) -> np.ndarray:
    """用 8 邻接 A* + 视线剪枝 + 圆角平滑生成无碰撞、无突兀折返的米制路径 (K,2)。"""
    free_cells = np.argwhere(~layout.occupied)
    if len(free_cells) < 2:
        raise ValueError("布局没有足够的空闲栅格")
    min_seg = min(0.65, layout.min_path_length)
    for _ in range(12):
        start = tuple(int(v) for v in free_cells[rng.integers(len(free_cells))])
        vertices = [_cell_world(layout, start)]
        prev_dir: np.ndarray | None = None
        for hop in range(24):
            smoothed = _smooth_path(layout, np.asarray(vertices))
            if _path_length(smoothed) >= layout.min_path_length + 0.05 and path_is_free(layout, smoothed):
                return smoothed
            for attempt in range(200):
                goal = tuple(int(v) for v in free_cells[rng.integers(len(free_cells))])
                dist = float(np.linalg.norm(_cell_world(layout, goal) - vertices[-1]))
                if layout.min_path_length >= 1.2 and attempt < 90:
                    if hop == 0 and not (0.7 <= dist <= 1.35):
                        continue
                    if 0 < hop < 4 and not (0.8 <= dist <= 2.2):
                        continue
                cells = _astar(layout, start, goal)
                if cells is None or len(cells) < 2:
                    continue
                segment = _shortcut_segment(layout, np.asarray([_cell_world(layout, cell) for cell in cells]))
                segment_length = _path_length(segment)
                if segment_length < min_seg:
                    continue
                if layout.min_path_length >= 1.2 and hop == 0 and attempt < 90 and segment_length > 1.5:
                    continue
                d_start = segment[1] - segment[0]
                d_start /= np.linalg.norm(d_start)
                if prev_dir is None:
                    if not _rear_clearance_ok(layout, segment[0], d_start):
                        continue
                else:
                    dot = float(np.dot(prev_dir, d_start))
                    # 防止段间 180° 瞬时折返；前段尽量包含转弯以覆盖进弯/出弯过渡。
                    if attempt < 150 and dot < -0.25:
                        continue
                    if attempt < 80 and dot > 0.92:
                        continue
                    if attempt >= 150 and dot < -0.65:
                        continue
                candidate = _smooth_path(layout, np.vstack([vertices, segment[1:]]))
                if not path_is_free(layout, candidate):
                    continue
                if attempt < 160 and _max_turn_rad(candidate) > 0.35:
                    continue
                vertices.extend(segment[1:])
                d_end = segment[-1] - segment[-2]
                prev_dir = d_end / np.linalg.norm(d_end)
                start = goal
                break
            else:
                break
        else:
            smoothed = _smooth_path(layout, np.asarray(vertices))
            if _path_length(smoothed) >= layout.min_path_length and path_is_free(layout, smoothed):
                return smoothed
    raise ValueError("无法规划满足长度与平滑要求的无碰撞路径")


def path_is_free(layout: Layout, path: np.ndarray) -> bool:
    """逐段密采样检查路径中心是否位于膨胀占用栅格的空闲区域。"""
    points = np.asarray(path)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) == 0:
        return False
    dense = [points[:1]]
    for start, end in pairwise(points):
        count = max(2, int(np.ceil(np.linalg.norm(end - start) / (layout.grid_resolution / 4.0))) + 1)
        dense.append(np.linspace(start, end, count)[1:])
    samples = np.concatenate(dense)
    cols = np.floor((samples[:, 0] + layout.width / 2.0) / layout.grid_resolution).astype(int)
    rows = np.floor((samples[:, 1] + layout.depth / 2.0) / layout.grid_resolution).astype(int)
    if (
        np.any(rows < 0)
        or np.any(rows >= layout.occupied.shape[0])
        or np.any(cols < 0)
        or np.any(cols >= layout.occupied.shape[1])
    ):
        return False
    return not bool(layout.occupied[rows, cols].any())
