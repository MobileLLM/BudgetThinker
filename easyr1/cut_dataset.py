import pandas as pd

def cut_data():
    file_path = "datasets/train_first_half.parquet"
    
    data = pd.read_parquet(file_path)

    print(data['problem'][0])
    
    half_size = len(data) // 2
    data_first_half = data.iloc[:half_size]
    data_second_half = data.iloc[half_size:]
    
    print(f"First half length: {len(data_first_half)}")
    print(f"Second half length: {len(data_second_half)}")
    
    data_first_half.to_parquet("datasets/train_1_in_4.parquet", index=False)
    data_second_half.to_parquet("datasets/train_2_in_4.parquet", index=False)


def formatted_data():
    file_path = "datasets/train_first_half.parquet"
    
    data = pd.read_parquet(file_path)
    
    data['problem'] = data['problem'].apply(lambda x: "Return your final response within \\boxed{}. " + x)
    
    print(data['problem'][0])
    
    target_path = file_path.replace(".parquet", "_formatted.parquet")
    data.to_parquet(target_path, index=False)


def visualize_data():
        # 定义文件路径
    file_path = "datasets/train-00000-of-00001_formatted.parquet"
    
    # 读取数据
    data = pd.read_parquet(file_path)

    print(data.head())
    
            
if __name__ == "__main__":
    formatted_data()
    visualize_data()
    cut_data()