import subprocess
import os
import sys
import datetime
import argparse

def run_fold(fold, config_file, additional_args=None):
    """运行指定的fold配置"""
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(log_dir, f"fold_{fold}.log")
    print(f"开始运行 fold {fold} ({datetime.datetime.now()})")
    print(f"日志保存在: {log_file}")
    
    # 构建命令
    cmd = ["python", "双卡main_causal.py", "--config", config_file, "--fold", str(fold)]
    if additional_args:
        cmd.extend(additional_args)
    
    # 运行命令并捕获输出
    with open(log_file, "w") as f:
        process = subprocess.Popen(
            cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        
        # 实时显示输出并写入日志
        for line in process.stdout:
            sys.stdout.write(line)
            f.write(line)
            f.flush()
        
        # 等待进程完成
        return_code = process.wait()
        
    return return_code == 0

def main():
    parser = argparse.ArgumentParser(description="运行多个fold配置")
    parser.add_argument("--config", default="./train_causal.yaml", help="配置文件路径")
    parser.add_argument("--start", type=int, default=0, help="起始fold编号")
    parser.add_argument("--end", type=int, default=4, help="结束fold编号")
    args = parser.parse_args()
    
    print(f"开始运行从fold {args.start}到{args.end}的配置... ({datetime.datetime.now()})")
    
    for fold in range(args.start, args.end + 1):
        print("=" * 50)
        success = run_fold(fold, args.config)
        
        if success:
            print(f"Fold {fold} 成功完成!")
        else:
            print(f"Fold {fold} 运行失败")
            answer = input("是否继续运行其他fold? (y/n): ")
            if answer.lower() != "y":
                print("中止运行")
                break
        print("=" * 50)
    
    print(f"完成! ({datetime.datetime.now()})")

if __name__ == "__main__":
    main()