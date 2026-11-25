# AWS RoboMaker 小型仓库资产

`small_warehouse.zip` 来自 AWS RoboMaker Small Warehouse World 的 `worlds/no_roof_small_warehouse.world`，版本固定为 `3c23a698bf0b4e366ddf8b084af507c519bd3483`。归档内含原版 MIT-0 授权文件。[原始场景](https://github.com/aws-robotics/aws-robomaker-small-warehouse-world/blob/3c23a698bf0b4e366ddf8b084af507c519bd3483/worlds/no_roof_small_warehouse.world) · [授权](https://github.com/aws-robotics/aws-robomaker-small-warehouse-world/blob/3c23a698bf0b4e366ddf8b084af507c519bd3483/LICENSE)

归档保留原场景 25 个模型实例及其相对位置。COLLADA 按材质转成 OBJ，保留 UV 与 PNG 贴图。MJCF 将地面平移至小车使用的 `z=0`，用 MuJoCo 聚光灯适配原场景灯光。`navigation.json` 记录从原几何得到的仓库尺寸与障碍包围盒。
