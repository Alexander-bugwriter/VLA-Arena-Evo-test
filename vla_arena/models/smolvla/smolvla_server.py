import asyncio
import websockets
import json
import torch
import numpy as np
import argparse
import logging
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

# ========= Logging =========
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
log = logging.getLogger(__name__)


def load_model(policy_path, device='cuda'):
    log.info(f"Loading SmolVLA policy from: {policy_path}")
    policy = SmolVLAPolicy.from_pretrained(policy_path)
    policy.to(device)
    policy.eval()
    log.info("✅ Model loaded successfully.")
    return policy


def infer_from_json_dict(data: dict, policy, device='cuda'):
    """
    完全对齐 Libero Client 的协议
    Input JSON:
      - image: [agentview(uint8 list), wrist(uint8 list), dummy(uint8 list)]
      - state: [pos(3), axis_angle(3), gripper(2)] -> Total 8 dims
      - prompt: string
      - image_mask: [1, 1, 0] (ignored)
      - action_mask: [...] (ignored)

    Output:
      - actions: List[List[float]] -> shape [Horizon, 7]
    """
    try:
        # === 1. 解析数据 ===
        images_list = data.get("image")
        prompt = data.get("prompt")
        state_list = data.get("state")

        if images_list is None or len(images_list) < 2:
            raise ValueError("Input 'image' list error: Need at least agentview and wrist.")

        # === 2. 图像处理 (List -> Tensor) ===
        # Libero Client 发送的是 [H, W, 3] 的 uint8 列表
        # image[0] -> AgentView
        # image[1] -> Wrist

        # 转换为 numpy
        agentview_np = np.array(images_list[0], dtype=np.uint8)
        wrist_np = np.array(images_list[1], dtype=np.uint8)

        # 预处理: [H, W, C] -> [1, C, H, W], 归一化 /255.0
        # 注意: Libero Client 发送的是 448x448。SmolVLA 会根据配置自动处理或接受该尺寸。
        agentview_tensor = torch.from_numpy(agentview_np / 255.0).permute(2, 0, 1).float().to(device).unsqueeze(0)
        wrist_tensor = torch.from_numpy(wrist_np / 255.0).permute(2, 0, 1).float().to(device).unsqueeze(0)

        # === 3. 状态处理 ===
        # Client 发送 8 维状态，转为 Tensor [1, 8]
        state_np = np.array(state_list, dtype=np.float32)
        state_tensor = torch.from_numpy(state_np).float().to(device).unsqueeze(0)

        # === 4. 构建 Observation ===
        observation = {
            'observation.images.image': agentview_tensor,
            'observation.images.wrist_image': wrist_tensor,
            'observation.state': state_tensor,
            'task': prompt,
        }

        # === 5. 推理 (Action Chunking) ===
        with torch.inference_mode():
            # select_action 返回 [Batch, Horizon, Action_Dim]
            # 例如 [1, 14, 7]
            action_tensor = policy.select_action(observation)

        # === 6. 格式化输出 ===
        # 取 Batch 0，转为 List of Lists
        # 结果形如: [[a1..a7], [a1..a7], ... (14 times)]
        actions = action_tensor[0].cpu().numpy().tolist()
        if len(actions) > 0 and isinstance(actions[0], (float, int, np.number)):
            actions = [actions]

        return actions

    except Exception as e:
        log.error(f"❌ Inference error: {e}")
        # 发生错误时，返回全0动作防止Client崩溃，或者抛出异常
        # 假设 Horizon=14, Dim=7
        return [[0.0] * 7] * 14


async def handle_request(websocket, policy, device):
    log.info(f"Client connected: {websocket.remote_address}")
    try:
        async for message in websocket:
            # 1. 接收数据
            # log.info("Received observation chunk...")
            try:
                json_data = json.loads(message)
            except json.JSONDecodeError:
                log.error("❌ Received invalid JSON")
                continue

            # 2. 模型推理
            actions_list = infer_from_json_dict(json_data, policy, device)

            # 3. 发送动作序列
            # Client 会执行: actions = np.array(json.loads(result))
            response = json.dumps(actions_list)
            await websocket.send(response)
            # log.info(f"Sent action chunk (length {len(actions_list)})")

    except websockets.exceptions.ConnectionClosed:
        log.info("Client disconnected.")
    except Exception as e:
        log.error(f"Unexpected error: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--policy_path', type=str, required=True, help='Path to pretrained SmolVLA checkpoint')
    parser.add_argument('--port', type=int, default=9000, help='Port to serve on (matches client default)')
    parser.add_argument('--device', type=str, default='cuda', help='Device')

    args = parser.parse_args()

    # 加载模型
    policy = load_model(args.policy_path, args.device)


    # 启动 WebSocket 服务
    async def main():
        log.info(f"🚀 SmolVLA Server running at ws://0.0.0.0:{args.port}")
        # max_size 设置为 200MB 以支持接收高清图像数据
        async with websockets.serve(
                lambda ws: handle_request(ws, policy, args.device),
                "0.0.0.0", args.port, max_size=200_000_000, ping_interval=None
        ):
            await asyncio.Future()  # Keep running


    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        log.info("Server stopped.")