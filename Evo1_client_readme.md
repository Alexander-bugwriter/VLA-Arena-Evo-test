原始的smolvla启动测试的指令是这样的
```bash
vla-arena eval --model smolvla --config vla_arena/configs/evaluation/smolvla.yaml
```
不用cli封装的话，直接用这个脚本也可以运行
```bash
python vla_arena/models/smolvla/evaluator.py --model smolvla --config vla_arena/configs/evaluation/smolvla.yaml
```

关键是把evaluator.py脚本拆分成client版本和server版本 server发送的关键字要和evo1_server一致 client要和原来的脚本环境对齐

步骤1 先创建smolvla_server.py 把归一化后的关键字和libero_client对齐
```bash
python vla_arena/models/smolvla/smolvla_server.py --policy_path /home/lyh/PycharmProjects/VLA-Arena/smolvla-vla-arena/pretrained_model
```

步骤2 创建vla_arena_client
```bash
python vla_arena/vla_arena_client.py
```

具体的文件名什么的在vla_arena_client.py文件里改就行了。

这个vla_arena_client和原生的evo1_server是完全对齐的。只需要修改一下evo1_server的端口确保端口对齐就行了。
接下来考虑训练并且检查一下server就行了。