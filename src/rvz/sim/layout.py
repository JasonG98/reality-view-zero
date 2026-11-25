"""确定性仓库布局、AABB 占用栅格与 A* 路径。"""

import heapq
from dataclasses import dataclass
from itertools import pairwise

import numpy as np

from rvz.config import SimConfig

_PALETTE = np.array(
    [
        [0.80, 0.35, 0.10],
        [0.15, 0.45, 0.80],
        [0.20, 0.65, 0.30],
        [0.85, 0.70, 0.15],
        [0.75, 0.20, 0.15],
        [0.55, 0.30, 0.70],
    ],
    dtype=np.float32,
)


@dataclass(frozen=True)
class Box:
    """轴对齐基本体；center/size 均为全尺寸米制，z 从底面支撑关系校验。"""

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
    floor_colors: np.ndarray
    wall_colors: np.ndarray
    sky_colors: np.ndarray
    grid_resolution: float
    robot_radius: float
    occupied: np.ndarray
    min_path_length: float


def _color(rng: np.random.Generator) -> np.ndarray:
    color = _PALETTE[rng.integers(len(_PALETTE))] + rng.uniform(-0.04, 0.04, 3)
    return np.concatenate([np.clip(color, 0.05, 0.95), [1.0]]).astype(np.float32)


def _overlaps(first: Box, second: Box) -> bool:
    first_min, first_max = first.center - first.size / 2.0, first.center + first.size / 2.0
    second_min, second_max = second.center - second.size / 2.0, second.center + second.size / 2.0
    return bool(np.all(np.minimum(first_max, second_max) - np.maximum(first_min, second_min) > 1e-5))


def _occupied_grid(
    width: float, depth: float, resolution: float, robot_radius: float, boxes: tuple[Box, ...]
) -> np.ndarray:
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


def _free_box(
    rng: np.random.Generator,
    width: float,
    depth: float,
    boxes: list[Box],
    name: str,
    kind: str,
    size: np.ndarray,
) -> Box | None:
    for _ in range(100):
        margin = 0.55
        center = np.array(
            [
                rng.uniform(-width / 2.0 + margin + size[0] / 2.0, width / 2.0 - margin - size[0] / 2.0),
                rng.uniform(-depth / 2.0 + margin + size[1] / 2.0, depth / 2.0 - margin - size[1] / 2.0),
                size[2] / 2.0,
            ],
            dtype=np.float32,
        )
        candidate = Box(name, kind, center, size.astype(np.float32), _color(rng))
        if not any(_overlaps(candidate, box) for box in boxes):
            return candidate
    return None


def sample_layout(seed: int, cfg: SimConfig) -> Layout:
    """由 seed 确定性生成仓库、随机纹理颜色和空闲栅格。"""
    rng = np.random.default_rng(seed)
    width = float(rng.uniform(cfg.warehouse_min_size, cfg.warehouse_max_size))
    depth = float(rng.uniform(cfg.warehouse_min_size, cfg.warehouse_max_size))
    n_aisles = int(rng.integers(cfg.min_aisles, cfg.max_aisles + 1))
    wall_colors = rng.uniform(0.25, 0.75, (2, 3)).astype(np.float32)
    floor_colors = rng.uniform(0.20, 0.65, (2, 3)).astype(np.float32)
    sky_colors = rng.uniform(0.30, 0.85, (2, 3)).astype(np.float32)
    boxes: list[Box] = []
    wall_specs = [
        ("wall_north", (0.0, depth / 2.0 - 0.1, 1.0), (width, 0.2, 2.0)),
        ("wall_south", (0.0, -depth / 2.0 + 0.1, 1.0), (width, 0.2, 2.0)),
        ("wall_east", (width / 2.0 - 0.1, 0.0, 1.0), (0.2, depth - 0.4, 2.0)),
        ("wall_west", (-width / 2.0 + 0.1, 0.0, 1.0), (0.2, depth - 0.4, 2.0)),
    ]
    for name, center, size in wall_specs:
        boxes.append(
            Box(
                name,
                "wall",
                np.array(center, dtype=np.float32),
                np.array(size, dtype=np.float32),
                np.ones(4, dtype=np.float32),
            )
        )

    shelf_count = n_aisles - 1
    shelf_ys = np.linspace(-depth / 2.0 + 1.5, depth / 2.0 - 1.5, shelf_count)
    for shelf_index, shelf_y in enumerate(shelf_ys):
        shelf_size = np.array([width * rng.uniform(0.45, 0.65), rng.uniform(0.55, 0.75), rng.uniform(1.2, 1.6)])
        shelf_center = np.array([width * rng.uniform(-0.08, 0.08), shelf_y, shelf_size[2] / 2.0])
        shelf = Box(
            f"shelf_{shelf_index}", "shelf", shelf_center.astype(np.float32), shelf_size.astype(np.float32), _color(rng)
        )
        boxes.append(shelf)
        for stack_index in range(int(rng.integers(1, 4))):
            stack_size = np.array([rng.uniform(0.25, 0.45), rng.uniform(0.25, 0.40), rng.uniform(0.25, 0.50)])
            for _ in range(30):
                stack_center = np.array(
                    [
                        rng.uniform(
                            shelf_center[0] - shelf_size[0] / 2.0 + stack_size[0] / 2.0,
                            shelf_center[0] + shelf_size[0] / 2.0 - stack_size[0] / 2.0,
                        ),
                        rng.uniform(
                            shelf_center[1] - shelf_size[1] / 2.0 + stack_size[1] / 2.0,
                            shelf_center[1] + shelf_size[1] / 2.0 - stack_size[1] / 2.0,
                        ),
                        shelf_size[2] + stack_size[2] / 2.0,
                    ],
                    dtype=np.float32,
                )
                candidate = Box(
                    f"shelf_box_{shelf_index}_{stack_index}",
                    "carton",
                    stack_center,
                    stack_size.astype(np.float32),
                    _color(rng),
                )
                if not any(_overlaps(candidate, box) for box in boxes):
                    boxes.append(candidate)
                    break

    object_counts = (("carton", 5 + n_aisles), ("pallet", 2), ("obstacle", 3))
    for kind, count in object_counts:
        for index in range(count):
            if kind == "pallet":
                size = np.array([rng.uniform(0.7, 1.0), rng.uniform(0.6, 0.9), 0.12])
            elif kind == "carton":
                size = rng.uniform(0.25, 0.60, 3)
            else:
                size = rng.uniform(0.20, 0.50, 3)
            candidate = _free_box(rng, width, depth, boxes, f"{kind}_{index}", kind, size)
            if candidate is not None:
                boxes.append(candidate)

    lights = []
    for _ in range(int(rng.integers(1, 4))):
        position = np.array(
            [rng.uniform(-width / 3.0, width / 3.0), rng.uniform(-depth / 3.0, depth / 3.0), rng.uniform(4.5, 6.5)],
            dtype=np.float32,
        )
        direction = -position / np.linalg.norm(position)
        diffuse = rng.uniform(0.65, 1.0, 3).astype(np.float32)
        lights.append(Light(position, direction.astype(np.float32), diffuse))
    occupied = _occupied_grid(width, depth, cfg.grid_resolution, cfg.robot_radius, tuple(boxes))
    layout = Layout(
        seed=seed,
        width=width,
        depth=depth,
        boxes=tuple(boxes),
        lights=tuple(lights),
        floor_colors=floor_colors,
        wall_colors=wall_colors,
        sky_colors=sky_colors,
        grid_resolution=cfg.grid_resolution,
        robot_radius=cfg.robot_radius,
        occupied=occupied,
        min_path_length=(cfg.steps - 1) * cfg.robot_speed * cfg.dt,
    )
    validate_layout(layout)
    return layout


def validate_layout(layout: Layout) -> None:
    """检查几何边界、正体积重叠、支撑关系和占用栅格一致性。"""
    names = [box.name for box in layout.boxes]
    if len(names) != len(set(names)):
        raise ValueError("布局几何体名称不唯一")
    for index, box in enumerate(layout.boxes):
        if np.any(box.size <= 0.0) or not np.all(np.isfinite(box.center)):
            raise ValueError(f"{box.name} 的尺寸或中心无效")
        low, high = box.center - box.size / 2.0, box.center + box.size / 2.0
        if low[0] < -layout.width / 2.0 - 1e-4 or high[0] > layout.width / 2.0 + 1e-4:
            raise ValueError(f"{box.name} 超出仓库宽度")
        if low[1] < -layout.depth / 2.0 - 1e-4 or high[1] > layout.depth / 2.0 + 1e-4:
            raise ValueError(f"{box.name} 超出仓库深度")
        if abs(low[2]) > 1e-4:
            supported = False
            for other in layout.boxes:
                other_top = other.center[2] + other.size[2] / 2.0
                horizontal_support = np.all(
                    np.minimum(high[:2], other.center[:2] + other.size[:2] / 2.0)
                    - np.maximum(low[:2], other.center[:2] - other.size[:2] / 2.0)
                    > 0.0
                )
                if abs(other_top - low[2]) < 1e-4 and horizontal_support:
                    supported = True
                    break
            if not supported:
                raise ValueError(f"{box.name} 悬空")
        for other in layout.boxes[index + 1 :]:
            if _overlaps(box, other):
                raise ValueError(f"几何体重叠：{box.name} 与 {other.name}")
    expected = _occupied_grid(layout.width, layout.depth, layout.grid_resolution, layout.robot_radius, layout.boxes)
    if not np.array_equal(layout.occupied, expected):
        raise ValueError("空闲栅格与几何体 AABB 不一致")


def _astar(layout: Layout, start: tuple[int, int], goal: tuple[int, int]) -> list[tuple[int, int]] | None:
    occupied = layout.occupied
    if occupied[start] or occupied[goal]:
        return None
    frontier = [(0, 0, start)]
    costs = {start: 0}
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
        if cost != costs[current]:
            continue
        for dy, dx in ((0, 1), (1, 0), (0, -1), (-1, 0)):
            neighbor = (current[0] + dy, current[1] + dx)
            if not (0 <= neighbor[0] < occupied.shape[0] and 0 <= neighbor[1] < occupied.shape[1]):
                continue
            if occupied[neighbor]:
                continue
            new_cost = cost + 1
            if new_cost >= costs.get(neighbor, 1 << 30):
                continue
            costs[neighbor] = new_cost
            parents[neighbor] = current
            priority = new_cost + abs(goal[0] - neighbor[0]) + abs(goal[1] - neighbor[1])
            heapq.heappush(frontier, (priority, new_cost, neighbor))
    return None


def _cell_world(layout: Layout, cell: tuple[int, int]) -> np.ndarray:
    return np.array(
        [
            (cell[1] + 0.5) * layout.grid_resolution - layout.width / 2.0,
            (cell[0] + 0.5) * layout.grid_resolution - layout.depth / 2.0,
        ]
    )


def plan_path(layout: Layout, rng: np.random.Generator) -> np.ndarray:
    """随机选点并用 4 邻接 A* 连接，返回总长足够的米制路径 (K,2)。"""
    free_cells = np.argwhere(~layout.occupied)
    if len(free_cells) < 2:
        raise ValueError("布局没有足够的空闲栅格")
    start = tuple(int(v) for v in free_cells[rng.integers(len(free_cells))])
    world_path = [_cell_world(layout, start)]
    total_length = 0.0
    for _ in range(12):
        if total_length >= layout.min_path_length:
            return np.asarray(world_path, dtype=np.float32)
        for _ in range(200):
            goal = tuple(int(v) for v in free_cells[rng.integers(len(free_cells))])
            cells = _astar(layout, start, goal)
            if cells is None or len(cells) < 2:
                continue
            segment = np.asarray([_cell_world(layout, cell) for cell in cells])
            segment_length = float(np.linalg.norm(np.diff(segment, axis=0), axis=1).sum())
            if segment_length < min(2.0, layout.min_path_length):
                continue
            world_path.extend(segment[1:])
            total_length += segment_length
            start = goal
            break
        else:
            raise ValueError("无法规划满足长度要求的无碰撞路径")
    if total_length < layout.min_path_length:
        raise ValueError("路径段数达到上限，仍无法满足轨迹长度")
    return np.asarray(world_path, dtype=np.float32)


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
