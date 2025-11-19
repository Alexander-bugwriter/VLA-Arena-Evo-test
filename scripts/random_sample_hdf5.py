import argparse
import os
from pathlib import Path
import h5py
import numpy as np
import json
import random


def copy_hdf5_group(source_group, target_group):
    """
    递归复制HDF5组的所有数据和属性
    
    Args:
        source_group: 源HDF5组
        target_group: 目标HDF5组
    """
    # 复制所有属性
    for key, value in source_group.attrs.items():
        target_group.attrs[key] = value
    
    # 复制所有数据集和子组
    for key in source_group.keys():
        source_item = source_group[key]
        if isinstance(source_item, h5py.Dataset):
            # 复制数据集
            target_group.create_dataset(key, data=source_item[:])
        elif isinstance(source_item, h5py.Group):
            # 递归复制子组
            target_subgroup = target_group.create_group(key)
            copy_hdf5_group(source_item, target_subgroup)


def sample_hdf5_file(input_file, output_file, sample_ratio, random_seed=None):
    """
    从HDF5文件中随机抽样一定比例的demo，创建新的HDF5文件
    
    Args:
        input_file: 输入HDF5文件路径
        output_file: 输出HDF5文件路径
        sample_ratio: 抽样比例 (0.0 - 1.0)
        random_seed: 随机种子，用于可重复性
    """
    if random_seed is not None:
        random.seed(random_seed)
        np.random.seed(random_seed)
    
    print(f"处理文件: {input_file}")
    
    # 打开输入文件
    try:
        with h5py.File(input_file, "r") as f_in:
            # 检查文件结构
            if "data" not in f_in.keys():
                print(f"错误: 文件 {input_file} 中没有找到 'data' 组")
                return False
            
            data_group = f_in["data"]
            
            # 获取所有demo的名称
            demo_names = [key for key in data_group.keys() if key.startswith("demo_")]
            demo_names.sort()  # 确保顺序一致
            
            if not demo_names:
                print(f"错误: 文件 {input_file} 中没有找到demo数据")
                return False
            
            total_demos = len(demo_names)
            num_samples = max(1, int(total_demos * sample_ratio))
            
            print(f"  总demo数: {total_demos}")
            print(f"  抽样比例: {sample_ratio:.1%}")
            print(f"  抽样数量: {num_samples}")
            
            # 随机选择demo
            selected_demos = random.sample(demo_names, num_samples)
            selected_demos.sort()  # 保持排序，便于阅读
            
            print(f"  选中的demo: {selected_demos[:5]}{'...' if len(selected_demos) > 5 else ''}")
            
            # 创建输出目录
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # 创建输出文件并复制数据
            with h5py.File(output_file, "w") as f_out:
                # 创建data组
                data_group_out = f_out.create_group("data")
                
                # 复制data组的所有属性
                for key, value in data_group.attrs.items():
                    data_group_out.attrs[key] = value
                
                # 复制选中的demo
                total_samples = 0
                for i, demo_name in enumerate(selected_demos):
                    # 创建新的demo组（重新编号）
                    new_demo_name = f"demo_{i}"
                    demo_group_out = data_group_out.create_group(new_demo_name)
                    
                    # 复制demo组的所有数据
                    demo_group_in = data_group[demo_name]
                    copy_hdf5_group(demo_group_in, demo_group_out)
                    
                    # 累计样本数
                    if "num_samples" in demo_group_in.attrs:
                        total_samples += demo_group_in.attrs["num_samples"]
                    elif "obs" in demo_group_in:
                        # 如果没有num_samples属性，尝试从obs中推断
                        obs_group = demo_group_in["obs"]
                        # 查找任意一个数据集来推断长度
                        for key in obs_group.keys():
                            if isinstance(obs_group[key], h5py.Dataset):
                                total_samples += len(obs_group[key])
                                break
                
                # 更新统计信息
                if "num_demos" in data_group_out.attrs:
                    data_group_out.attrs["num_demos"] = num_samples
                if "total" in data_group_out.attrs:
                    data_group_out.attrs["total"] = total_samples
                
                print(f"  输出文件: {output_file}")
                print(f"  保留demo数: {num_samples}")
                print(f"  总样本数: {total_samples}")
            
            return True
            
    except Exception as e:
        print(f"处理文件 {input_file} 时出错: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(
        description="从HDF5文件中随机抽样一定比例的数据，创建新的HDF5文件"
    )
    parser.add_argument(
        "--input-file",
        type=str,
        help="输入HDF5文件路径"
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default=None,
        help="输出HDF5文件路径（默认：在输入文件名后添加_sampled后缀）"
    )
    parser.add_argument(
        "--ratio",
        type=float,
        required=True,
        help="抽样比例 (0.0 - 1.0)，例如 0.5 表示抽样50%%"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="随机种子，用于可重复性"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default=None,
        help="输入目录，批量处理目录下的所有HDF5文件"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="输出目录，与--input-dir一起使用"
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*.hdf5",
        help="文件名模式（默认: *.hdf5）"
    )
    parser.add_argument(
        "--not-recursive",
        action="store_true",
        help="不递归搜索子目录"
    )
    
    args = parser.parse_args()
    
    # 验证抽样比例
    if args.ratio < 0.0 or args.ratio > 1.0:
        print("错误: 抽样比例必须在0.0到1.0之间")
        return
    
    # 批量处理模式
    if args.input_dir:
        if not args.output_dir:
            print("错误: 使用--input-dir时必须指定--output-dir")
            return
        
        input_dir = Path(args.input_dir)
        output_dir = Path(args.output_dir)
        
        # 查找所有HDF5文件
        if args.not_recursive:
            demo_files = list(input_dir.glob(args.pattern))
        else:
            demo_files = list(input_dir.rglob(args.pattern))
        
        if not demo_files:
            print(f"在 {args.input_dir} 中没有找到匹配 {args.pattern} 的文件")
            return
        
        print(f"找到 {len(demo_files)} 个文件待处理\n")
        
        success_count = 0
        for demo_file in demo_files:
            # 生成输出文件路径
            relative_path = demo_file.relative_to(input_dir)
            output_file = output_dir / relative_path
            
            # 如果输出文件名与输入相同，添加后缀
            if output_file == demo_file:
                output_file = output_file.parent / f"{output_file.stem}_sampled{output_file.suffix}"
            
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            if sample_hdf5_file(str(demo_file), str(output_file), args.ratio, args.seed):
                success_count += 1
            print()
        
        print(f"处理完成: {success_count}/{len(demo_files)} 个文件成功")
    
    # 单文件处理模式
    else:
        if not args.input_file:
            print("错误: 必须指定--input-file或--input-dir")
            return
        
        # 确定输出文件路径
        if args.output_file:
            output_file = args.output_file
        else:
            input_path = Path(args.input_file)
            output_file = str(input_path.parent / f"{input_path.stem}_sampled{input_path.suffix}")
        
        success = sample_hdf5_file(args.input_file, output_file, args.ratio, args.seed)
        if success:
            print("\n处理完成!")
        else:
            print("\n处理失败!")


if __name__ == "__main__":
    main()

