# RoboRender on one AMD Radeon Pro W7900

上游是 [robo-render/RoboRender-Video-Model](https://github.com/robo-render/RoboRender-Video-Model)，`main` @ `16850eba`。论文是 [arXiv:2610.09254](https://arxiv.org/abs/2610.09254)，项目页 [robo-render.github.io](https://robo-render.github.io/)。上游说明原样保留在 [`UPSTREAM_README.md`](UPSTREAM_README.md)。

这次在一张 AMD Radeon Pro W7900D 上跑完了上游 `infer.py` 的主路径：三视角、81 帧、50 步、CFG 5.0、seed 42。权重、容器层和编译缓存不在这个目录里。

上游仓库没有 `LICENSE` 文件。`core/` 里每个文件写明改编自 [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio) 和 [Wan2.1](https://github.com/Wan-Video/Wan2.1)，这两个上游是 Apache-2.0。RoboRender 自己加的推理驱动和裁剪后的 pipeline 没有单独的许可。在作者允许再分发之前，不要把这棵树推到新的公开仓库。

## 改了什么

GPU 专用依赖只有可选的 `flash_attn` 和 `sageattention`，`requirements.txt` 里本来就没写它们。缺这两个包时，原代码走 PyTorch 的 SDPA。ROCm 上这条路要改两处，否则要么选到会爆显存的数学核，要么 safetensors 用 CUDA 分配器直接失败。

| 文件 | 改了什么 |
| --- | --- |
| `core/dit.py` | 注意力只走 `SDPBackend.FLASH_ATTENTION`，不行再走 `EFFICIENT_ATTENTION`。数学核会把约 2.5 万 token 的分数矩阵展开，48 GB 也放不下 |
| `core/loader.py` | safetensors 若不能直接载到 `cuda` 设备，先载到 CPU 再搬。这次实际加载没有触发这条退路 |
| `requirements-rocm.txt` | 去掉 CUDA 版 `torch==2.9.1`。镜像里的 PyTorch 是 `2.11.0+rocm7.14.0` |
| `scripts/` | 单卡容器、依赖安装、合成条件样本 |

第一个提交 `da1d8b8` 的树与上游 `16850eba` 一致。`git diff da1d8b8` 能看到代码改动，对应 [`results/port.diff`](results/port.diff)。

## 环境

- 镜像：`rocm/pytorch:rocm7.14_ubuntu24.04_py3.12_pytorch_release_2.11.0`（HIP 7.14.60850）
- GPU：一张 AMD Radeon Pro W7900D（gfx1100，48 GB）
- 容器只挂 `/dev/kfd` 和一张卡的渲染节点。进程里 `torch.cuda.device_count()` 为 1，见 `results/isolation.log`
- 容器内设置 `HIP_VISIBLE_DEVICES=0` 和 `ROCR_VISIBLE_DEVICES=0`
- 从镜像自带的 `/opt/venv` 再套一层 venv 时，`--system-site-packages` 看不到父 venv 里的 PyTorch。`scripts/radeon_install_deps.sh` 会把 `/opt/venv/lib/python3.12/site-packages` 写进 `.pth`

## 复现

宿主机工作目录挂到容器 `/workspace`。根分区如果装不下约 20 GB 的权重，把目录放在大磁盘上。

| 容器内路径 | 内容 |
| --- | --- |
| `/workspace/src` | 本目录 |
| `/workspace/models/Wan2.1-Fun-V1.1-1.3B-Control` | [alibaba-pai/Wan2.1-Fun-V1.1-1.3B-Control](https://huggingface.co/alibaba-pai/Wan2.1-Fun-V1.1-1.3B-Control) 里推理要用的文件 |
| `/workspace/ckpt/roborender_lora.safetensors` | [RoboRender/roborender-video](https://huggingface.co/RoboRender/roborender-video)，699,995,040 字节 |
| `/workspace/venv` | torch 以外的依赖 |
| `/workspace/results` | 日志和视频 |

```bash
WORK=/path/to/workdir
git clone <this-repo> "$WORK/src"
# 在 src 里用相对链接，容器内外都能解析
ln -sfn ../models "$WORK/src/models"
ln -sfn ../ckpt "$WORK/src/ckpt"
ln -sfn ../data "$WORK/src/data"
ln -sfn ../results "$WORK/src/outputs"
bash "$WORK/src/scripts/radeon_start_container.sh" "$WORK" /dev/dri/renderD131
docker exec radeon-roborender-video-model bash /workspace/src/scripts/radeon_install_deps.sh
```

权重按上游 README 用 `hf download`。这台机器访问不了 huggingface.co，基座模型改从 ModelScope 的 `PAI/Wan2.1-Fun-V1.1-1.3B-Control` 拉取，LoRA 走 `hf-mirror.com`。需要的文件是 `diffusion_pytorch_model.safetensors`、`Wan2.1_VAE.pth`、`models_t5_umt5-xxl-enc-bf16.pth`、`models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth` 和 `google/umt5-xxl/`。

上游 README 里的 `data/test_long_sample_00502` 没有随仓库发布。`scripts/make_synthetic_sample.py` 按同一目录结构写了一个 81 帧、三视角的合成深度和机械臂掩码，所以 `infer.py` 的命令行可以原样跑。这不是论文里的 DROID / AgiBot 测试片段。

```bash
docker exec radeon-roborender-video-model /workspace/venv/bin/python \
  /workspace/src/scripts/make_synthetic_sample.py /workspace/data/synthetic_clip81
docker exec -w /workspace/src radeon-roborender-video-model \
  /workspace/venv/bin/python infer.py \
  --sample_dir data/synthetic_clip81 \
  --views hand_left hand_right head_color \
  --output_dir /workspace/results/demo
```

默认就是 `--height 240 --width 416 --clip_length 81 --num_inference_steps 50 --cfg_scale 5.0 --seed 42`。

## 验证

一次 50 步、三视角、81 帧的片段，从 VAE 编码到解码，用时 399.7 秒。去噪过程中 `rocm-smi` 读到 GPU 3 占用 18.4 GiB（19,749,101,568 字节），48 GB 的卡只用一张就够。LoRA 融合了 300 个张量。

输出在 `results/`：

| 文件 | 尺寸 | sha256 |
| --- | --- | --- |
| `generated_hand_left.mp4` | 81 帧，416×240，15 fps | `4ec2771f…b80914` |
| `generated_hand_right.mp4` | 同上 | `401c90da…f18717` |
| `generated_head_color.mp4` | 同上 | `3837b3ea…d712b9` |
| `comparison.mp4` | 81 帧，1248×720（每行是深度、掩码、生成结果，三行叠在一起） | `fdbbb468…e4d6f1` |

完整哈希在 `results/metrics.txt`，日志在 `results/verify.log`。第 40 帧的对比图是 `results/frame40_comparison.png`。合成条件是一根横条加一个椭圆；模型画出了木纹桌面、黑色夹爪，以及一个浅色勺子。提示词写的是红杯子。几何把形状钉死了，语言没有把椭圆改成杯子。

论文在单张 H200 上给同规格片段的时间是 68.8 秒。这次是 399.7 秒，大约慢 5.8 倍。论文里的 6 步 DMD2 学生模型没有出现在这个仓库里，issue #1 还在问它会不会开源，所以验证停在发布出来的 50 步教师路径。
