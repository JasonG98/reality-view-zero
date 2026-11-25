"""在桌面窗口同步显示仿真的五路相机。"""

from contextlib import closing
from time import perf_counter

from PIL import Image

from rvz.config import SimConfig
from rvz.geometry import CAM_ORDER
from rvz.sim.record import live_camera_frames

# 精简版 Python 可能不含 Tk，其他 CLI 子命令仍需能导入。
try:
    import tkinter

    from PIL import ImageTk
except ImportError as error:
    if error.name not in {"tkinter", "_tkinter"}:
        raise
    tkinter = None


_CAMERA_LABELS = ("前视", "左视", "右视", "后视")  # 与 CAM_ORDER 对齐
_PANEL_GAP = 12
_TILE_GAP = 4
_LABEL_INSET = 8
_LAYOUT_RETRY_MS = 10
_SCREEN_MARGIN = 80
_DEFAULT_WINDOW_WIDTH = 1060
_DEFAULT_WINDOW_HEIGHT = 536


def run_viewer(seed: int, cfg: SimConfig, drone_index: int = 0) -> None:
    """按仿真步长刷新桌面窗口，关闭窗口时释放渲染器。"""
    if tkinter is None:
        raise RuntimeError("当前 Python 缺少 Tk 支持，无法打开实时预览窗口")
    try:
        window = tkinter.Tk()
    except tkinter.TclError as error:
        raise RuntimeError("无法打开实时预览窗口，请在图形桌面会话中运行") from error

    try:
        window.title(f"rvz 实时预览 · seed {seed}")
        window_width = min(
            max(_DEFAULT_WINDOW_WIDTH, 2 * cfg.drone_res + 3 * _PANEL_GAP),
            window.winfo_screenwidth() - _SCREEN_MARGIN,
        )
        window_height = min(
            max(_DEFAULT_WINDOW_HEIGHT, cfg.drone_res + 2 * _PANEL_GAP),
            window.winfo_screenheight() - _SCREEN_MARGIN,
        )
        window.geometry(f"{window_width}x{window_height}")
        window.minsize(480, 260)
        window.protocol("WM_DELETE_WINDOW", window.quit)
        window.bind("<Escape>", lambda _event: window.quit())
        canvas = tkinter.Canvas(window, background="#11161b", highlightthickness=0)
        canvas.pack(fill="both", expand=True)

        camera_labels = (*_CAMERA_LABELS, "无人机视角")
        image_items = [canvas.create_image(0, 0, anchor="nw") for _ in camera_labels]
        label_items = [
            canvas.create_window(
                0,
                0,
                anchor="nw",
                window=tkinter.Label(canvas, text=label, background="#11161b", foreground="#f5f7fa", padx=8, pady=3),
            )
            for label in camera_labels
        ]

        frame_error: Exception | KeyboardInterrupt | None = None
        photos: list[ImageTk.PhotoImage] = []
        with closing(live_camera_frames(seed, cfg, drone_index)) as frames:

            def draw_frame() -> None:
                nonlocal frame_error, next_draw
                started = perf_counter()
                try:
                    width, height = canvas.winfo_width(), canvas.winfo_height()
                    tile_size = min(
                        (width - 3 * _PANEL_GAP - 2 * _TILE_GAP) // 4,
                        (height - 2 * _PANEL_GAP - _TILE_GAP) // 2,
                    )
                    if tile_size < 1:
                        # Tk 首次映射时画布可能只有 1×1，等布局完成后再推进仿真。
                        next_draw = window.after(_LAYOUT_RETRY_MS, draw_frame)
                        return
                    robot_rgb, drone_rgb = next(frames)
                    panel_size = 2 * tile_size + _TILE_GAP
                    left = (width - 2 * panel_size - _PANEL_GAP) // 2
                    top = (height - panel_size) // 2
                    stride = tile_size + _TILE_GAP
                    positions = (
                        (left, top),
                        (left + stride, top),
                        (left, top + stride),
                        (left + stride, top + stride),
                        (left + panel_size + _PANEL_GAP, top),
                    )
                    sizes = (tile_size,) * len(CAM_ORDER) + (panel_size,)
                    photos[:] = [
                        ImageTk.PhotoImage(
                            Image.fromarray(rgb).resize((size, size), Image.Resampling.LANCZOS), master=window
                        )
                        for rgb, size in zip((*robot_rgb, drone_rgb), sizes, strict=True)
                    ]
                    for image_item, label_item, photo, (x, y) in zip(
                        image_items, label_items, photos, positions, strict=True
                    ):
                        canvas.coords(image_item, x, y)
                        canvas.itemconfigure(image_item, image=photo)
                        canvas.coords(label_item, x + _LABEL_INSET, y + _LABEL_INSET)
                    delay_ms = max(1, round((cfg.dt - (perf_counter() - started)) * 1000))
                    next_draw = window.after(delay_ms, draw_frame)
                except (Exception, KeyboardInterrupt) as error:
                    # Tk 的回调异常默认只打印，交还给 CLI 才能以失败退出。
                    frame_error = error
                    window.quit()

            next_draw = window.after_idle(draw_frame)
            try:
                window.mainloop()
            finally:
                window.after_cancel(next_draw)
        if frame_error is not None:
            raise frame_error
    finally:
        window.destroy()
