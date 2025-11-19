import argparse
import h5py


def print_dataset_info(name, obj):
    """回调函数，用于打印HDF5对象的信息。"""
    indent_level = name.count('/')
    indent = '  ' * indent_level

    if isinstance(obj, h5py.Dataset):
        # 打印数据集信息
        shape = obj.shape
        dtype = obj.dtype
        print(f"{indent}- 数据集: {name} | 形状: {shape} | 类型: {dtype}")

        # 尝试展示前几个数据
        try:
            data_preview = obj[...]
            if data_preview.size > 0:
                # 限制显示数量，避免输出过多数据
                preview_flat = data_preview.flatten()
                preview_size = min(5, preview_flat.size)
                preview_str = ', '.join(str(x) for x in preview_flat[:preview_size])
                print(f"{indent}    示例数据: {preview_str}{' ...' if preview_flat.size > preview_size else ''}")
        except Exception:
            print(f"{indent}    (无法读取数据示例)" )

        # 打印属性
        if obj.attrs:
            print(f"{indent}    属性:")
            for key, value in obj.attrs.items():
                print(f"{indent}      - {key}: {value}")

    elif isinstance(obj, h5py.Group):
        # 打印组信息
        print(f"{indent}+ 组: {name if name else '/'}")
        if obj.attrs:
            print(f"{indent}    属性:")
            for key, value in obj.attrs.items():
                print(f"{indent}      - {key}: {value}")


def inspect_hdf5(file_path, dataset_path=None):
    """检查HDF5文件的结构及内容示例。"""
    print(f"正在检查文件: {file_path}")

    with h5py.File(file_path, 'r') as h5_file:
        if dataset_path:
            if dataset_path in h5_file:
                obj = h5_file[dataset_path]
                print_dataset_info(dataset_path, obj)
            else:
                print(f"路径 {dataset_path} 不存在。可用的键包括:")
                for key in h5_file.keys():
                    print(f"- {key}")
        else:
            h5_file.visititems(print_dataset_info)


def main():
    parser = argparse.ArgumentParser(description="打印HDF5文件中的键和值示例")
    parser.add_argument("file", type=str, help="HDF5 文件路径")
    parser.add_argument(
        "--path",
        type=str,
        default=None,
        help="指定要查看的数据集路径，默认打印整个文件结构"
    )

    args = parser.parse_args()
    inspect_hdf5(args.file, args.path)


if __name__ == "__main__":
    main()
