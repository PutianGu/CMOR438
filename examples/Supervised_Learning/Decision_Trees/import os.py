import os

def list_files(startpath):
    print(f"Project Structure for: {startpath}\n")
    
    # 遍历目录
    for root, dirs, files in os.walk(startpath):
        # 忽略 .git 目录和 __pycache__ 目录，让输出更干净
        if '.git' in dirs:
            dirs.remove('.git')
        if '__pycache__' in dirs:
            dirs.remove('__pycache__')
            
        # 计算缩进级别
        level = root.replace(startpath, '').count(os.sep)
        indent = '│   ' * (level - 1) + '├── ' if level > 0 else ''
        
        # 打印文件夹名称
        print(f"{indent}{os.path.basename(root)}/")
        
        # 打印该文件夹下的所有文件
        sub_indent = '│   ' * level + '├── '
        for f in files:
            print(f"{sub_indent}{f}")

# 你的目标路径 (使用了 r 前缀来处理 Windows 反斜杠)
target_path = r"C:\Users\sutt6\OneDrive\Desktop\CMOR438\CMOR438"

if os.path.exists(target_path):
    list_files(target_path)
else:
    print("错误：找不到指定的路径，请检查拼写。")