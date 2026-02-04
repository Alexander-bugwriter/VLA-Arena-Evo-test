# vla_arena_client.py

import json
import logging
import math
import random
import asyncio
import websockets
import numpy as np
import draccus
import imageio
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from tqdm import tqdm

# VLA-Arena 依赖
from vla_arena.vla_arena import benchmark, get_vla_arena_path
from vla_arena.vla_arena.envs import OffScreenRenderEnv

# 日志配置
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', datefmt='%H:%M:%S')
log = logging.getLogger(__name__)

VLA_ARENA_ENV_RESOLUTION = 256
VLA_ARENA_DUMMY_ACTION = [0.0] * 6 + [-1.0]


@dataclass
class Args:
    # === Server Settings ===
    server_url: str = "ws://localhost:9000"
    # [关键参数] execution_horizon
    # 意义：从模型生成的 Chunk 中实际采纳前多少步。
    # 例如：Evo1 生成 50 步，这里设为 14，则只执行前 14 步。
    execution_horizon: int = 14
    # === Task Settings ===
    task_suite_name: str | list[str] = 'safety_dynamic_obstacles'
    task_level: int = 0
    num_trials_per_task: int = 10
    # [新增参数] max_episode_steps
    # 意义：强制指定最大步数。如果不指定 (None)，则使用 Benchmark 默认值 (300 或 600)。
    max_episode_steps: int | None = None
    # === Output Settings ===
    video_out_path: str = f'smolvla_server_test/{datetime.now().strftime("%Y%m%d")}'
    seed: int = 7
    save_video_mode: str = 'first_success_failure'  # 'all', 'first_success_failure', 'none'
    # === Env Config (Don't change usually) ===
    add_noise: bool = False
    randomize_color: bool = False
    adjust_light: bool = False
    camera_offset: bool = False
    policy_path: str = "remote"  # 占位符


def construct_payload(obs, task_description):
    """打包发送给 Evo1 Server 的标准数据包"""
    # 1. 获取图像数据
    agentview = np.ascontiguousarray(obs['agentview_image'][::-1, ::-1])
    wrist = np.ascontiguousarray(obs['robot0_eye_in_hand_image'][::-1, ::-1])

    # 2. [关键修复] 创建第3张 Dummy 图片 (Evo1 Server 强制要求 len(images) == 3)
    # 使用与 agentview 相同的尺寸 (H, W, 3)，全黑
    dummy_img = np.zeros_like(agentview)

    return {
        "image": [
            agentview.tolist(),  # image[0]
            wrist.tolist(),  # image[1]
            dummy_img.tolist()  # image[2] (Dummy)
        ],
        "state": np.concatenate((
            obs['robot0_eef_pos'],
            _quat2axisangle(obs['robot0_eef_quat']),
            obs['robot0_gripper_qpos'],
        )).astype(np.float32).tolist(),
        "prompt": task_description,
        "image_mask": [1, 1, 0],  # 标记前两张有效，第三张无效
        "action_mask": [1] * 7 + [0] * 17
    }


def _quat2axisangle(quat):
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0): return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


async def eval_vla_arena(args: Args) -> None:
    # 设置随机种子
    np.random.seed(args.seed);
    random.seed(args.seed);
    rng = np.random.default_rng(args.seed)

    async with websockets.connect(args.server_url, max_size=200_000_000) as websocket:
        log.info(f"✅ Connected to Inference Server: {args.server_url}")
        log.info(f"⚙️  Execution Horizon: {args.execution_horizon} (Only affects chunk models)")

        benchmark_dict = benchmark.get_benchmark_dict()
        suite_names = [args.task_suite_name] if isinstance(args.task_suite_name, str) else args.task_suite_name

        for suite_name in suite_names:
            task_suite = benchmark_dict[suite_name]()

            # [关键逻辑 1] 确定任务的最大允许步数 (Failure Threshold)
            # 优先级：Args 参数 > 任务默认值
            if args.max_episode_steps is not None:
                max_allowed_steps = args.max_episode_steps
                log.info(f"⚙️  Overriding Max Steps with User Argument: {max_allowed_steps}")
            else:
                max_allowed_steps = 600 if suite_name == 'long_horizon' else 300
                log.info(f"⚙️  Using Default Max Steps for {suite_name}: {max_allowed_steps}")

            video_dir = Path(args.video_out_path) / suite_name
            video_dir.mkdir(parents=True, exist_ok=True)

            total_success = 0

            for task_id in tqdm(range(5), desc="Tasks"):
                task = task_suite.get_task_by_level_id(args.task_level, task_id)
                env, _ = _get_env(task, args)

                for ep_idx in tqdm(range(args.num_trials_per_task), desc=f"Task {task_id}", leave=False):
                    # Reset Env
                    obs = env.reset();
                    obs = env.set_init_state(
                        task_suite.get_task_init_states(args.task_level, task_id)[(ep_idx + rng.integers(0, 10)) % 10])
                    for _ in range(10): obs, _, _, _ = env.step(VLA_ARENA_DUMMY_ACTION)  # Warmup

                    # 初始化状态
                    total_steps = 0  # 累计执行步数
                    frames = []
                    done = False

                    # === 主循环：直到成功 或 超过最大步数 ===
                    while total_steps < max_allowed_steps:
                        try:
                            # 1. 发送请求
                            frames.append(np.ascontiguousarray(obs['agentview_image'][::-1, ::-1]))
                            await websocket.send(json.dumps(construct_payload(obs, task.language)))

                            # 2. 接收响应
                            actions = json.loads(await websocket.recv())
                            server_gen_len = len(actions)

                            # [关键逻辑 2] 决定本次要执行多少步
                            if server_gen_len == 1:
                                steps_to_execute = 1
                            else:
                                steps_to_execute = min(server_gen_len, args.execution_horizon)

                            # 3. 执行动作 (Inner Execution Loop)
                            for i in range(steps_to_execute):
                                # [关键逻辑 3] 每一小步都要检查是否超时
                                if total_steps >= max_allowed_steps:
                                    log.info(f"⏳ Failed: Timeout at step {total_steps}")
                                    break  # 跳出 Inner Loop

                                # === [关键修复] 动作维度适配 ===
                                raw_action = actions[i]

                                # 1. 截取前 7 维 (针对 Evo1 的 24 维输出)
                                # 如果是 SmolVLA 输出 7 维，切片操作 [:7] 也是安全的
                                action = raw_action[:7]
                                obs, _, done, info = env.step(action)
                                total_steps += 1  # 累计步数 +1

                                if done:
                                    break  # 成功，跳出 Inner Loop

                            if done or total_steps >= max_allowed_steps:
                                break  # 跳出 Outer While Loop

                        except Exception as e:
                            log.error(f"Error: {e}");
                            break

                    # === 结算 ===
                    if done:
                        total_success += 1

                    # 视频保存逻辑
                    if args.save_video_mode != 'none':
                        should_save = False
                        if args.save_video_mode == 'all':
                            should_save = True
                        elif args.save_video_mode == 'first_success_failure':
                            should_save = True

                        if should_save:
                            suffix = "success" if done else "failure"
                            vid_path = video_dir / f"task{task_id}_ep{ep_idx}_{suffix}.mp4"
                            imageio.mimsave(vid_path, frames, fps=30)

            log.info(
                f"Suite {suite_name} Finished. Success Rate: {total_success}/{5 * args.num_trials_per_task} ({total_success / (5 * args.num_trials_per_task) * 100:.1f}%)")


def _get_env(task, args):
    bddl = Path(get_vla_arena_path('bddl_files')) / task.problem_folder / f'level_{task.level}' / task.bddl_file
    return OffScreenRenderEnv(
        bddl_file_name=str(bddl), camera_heights=VLA_ARENA_ENV_RESOLUTION, camera_widths=VLA_ARENA_ENV_RESOLUTION,
        camera_offset=args.camera_offset, color_randomize=args.randomize_color, add_noise=args.add_noise,
        light_adjustment=args.adjust_light
    ), task.language


if __name__ == '__main__':
    args = draccus.parse(Args)
    try:
        asyncio.run(eval_vla_arena(args))
    except KeyboardInterrupt:
        pass