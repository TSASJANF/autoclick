# 自动点击 - 屏幕识别工具

基于 YOLO 与 OpenCV 模板匹配的 Windows 屏幕识别自动点击工具，
内置「截图采集 → 标注 → 训练 → 检测 → 自动点击」完整工作流。

## 快速开始

```bash
pip install -e .          # 或 pip install -r requirements.txt
python main.py
```

界面基于 **qt-material**（Material Design，MIT 协议）主题，语义按钮
（主操作/成功/危险）由 `ui/theme.py` 叠加补充样式实现。

## 典型工作流

1. **标注页**：新建样本集 → 截图（F12）采集屏幕 → 选择/新建目标 → 拖框标注 → 保存（S）
2. **训练页**：选择目标 → 自动推荐超参 → 开始训练（停止为协作式，在 epoch 边界生效）
3. **检测页**：勾选目标 → 设置置信度/帧率/优先级 → 开始检测（全局热键 F9 切换）
4. **OpenCV 模板页**：无需训练的轻量方案：截图 → 框选模板 → 测试匹配

## 项目结构

```
main.py                  入口（logging + QSS + QApplication）
core/                    领域与平台层（零 Qt 依赖，可独立测试）
  models.py              领域数据类（Target/SampleSet/Detection + NMS）
  paths.py               路径治理：APP_ROOT 解析、失效路径重定位修复
  json_store.py          原子 JSON 读写（损坏自动备份）
  image_io.py            Unicode 安全的图像读写（修复中文路径）
  detection_engine.py    统一检测引擎 + 点击决策（纯逻辑）
  detector.py            YOLO 检测器封装
  opencv_detector.py     模板匹配检测器（top-K + NMS）
  annotations.py         YOLO 格式标注读写
  dataset.py             训练数据集组装（负样本策略）
  train_runner.py        训练运行器（协作停止）
  capture.py / windows.py / clicker.py / hotkey.py   Win32 平台能力
  target_manager.py / sample_set_manager.py          持久化仓库
  config.py / state_manager.py                       配置与会话状态
ui/                      表现层（PySide6）
  app_context.py         组合根：共享服务的唯一组装点
  widgets/               共享组件（画布、图像转换）
  workers/               QThread 适配层（检测/训练/模型加载）
  dialogs.py             共享对话框
  *_tab.py               各功能页签
tests/                   单元测试（core 纯逻辑全覆盖）
scripts/                 一次性维护脚本
```

## 架构规则

1. **core 不得导入 PySide6** —— 业务逻辑与 UI 框架彻底解耦，
   Qt 线程适配统一放在 ui/workers；
2. **所有持久化路径经过 core.paths** —— 存储相对 APP_ROOT，运行时
   绝对路径，项目目录整体迁移不失效（旧绝对路径自动重定位）；
3. **图像读写一律用 core.image_io** —— cv2.imread/imwrite 在 Windows
   上不支持非 ASCII 路径；
4. **JSON 读写一律用 core.json_store** —— 原子写 + 损坏备份；
5. **状态变更走 ctx.hub 信号广播** —— 页签修改目标后 emit
   `hub.targets_changed`，无需在 MainWindow 补连散装信号。

## 行为要点

- **训练**：数据集组装（负样本 + train/val 80/20 划分，固定种子可复现）
  在后台线程执行；停止为协作式（epoch 边界生效），不强杀线程；
- **检测**：单/多目标统一点击语义（优先级 + 只点一个/点全部对单目标
  同样生效）；「ai+opencv」双模式结果跨源去重；点击带屏幕边缘安全边距；
  连续 10 帧异常自动停止并提示（锁屏/休眠恢复后不再假死）；
- **安全**：单实例锁（双开会相互覆盖配置）；导入 .pt 模型前有 pickle
  RCE 风险提示；截图目录按保留上限自动清理；
- **多显示器**：点击坐标基于虚拟桌面换算，副屏目标不会点错位；
- **截图隐藏窗口**：只隐藏本进程自己的窗口，不会隐藏游戏/其他应用。

## 测试

```bash
pip install -e .[dev]
pytest                           # 78 个单元测试
python tests/smoke_offscreen.py  # 离屏冒烟测试（完整构建主窗口）
python tests/layout_audit.py     # 布局审计：自动检测文字截断
python tests/visual_check.py     # 各页签截图输出到 logs/ui_shots/
```

## 数据目录

- `data/sets/<set_id>/` — 样本集（images/ + targets/ + session.json）
- `data/screenshots/` — 截图输出
- `logs/app.log` — 运行日志
- `targets.json` / `config.json` / `app_state.json` — 持久化状态
