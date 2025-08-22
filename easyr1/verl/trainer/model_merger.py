# model_merger.py
import os
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Tuple, Optional

import torch
from torch.distributed._tensor import DTensor, Placement, Shard
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForTokenClassification,
    AutoModelForVision2Seq,
    PreTrainedModel,
)


def merge_by_placement(tensors: List[torch.Tensor], placement: Placement) -> torch.Tensor:
    """Merge tensors according to their placement."""
    if placement.is_replicate():
        return tensors[0]
    elif placement.is_partial():
        raise NotImplementedError("Partial placement is not supported yet")
    elif placement.is_shard():
        return torch.cat(tensors, dim=placement.dim).contiguous()
    else:
        raise ValueError(f"Unsupported placement: {placement}")


def get_model_class(config: AutoConfig) -> PreTrainedModel:
    """Determine the appropriate model class based on config."""
    if "ForTokenClassification" in config.architectures[0]:
        return AutoModelForTokenClassification
    elif "ForCausalLM" in config.architectures[0]:
        return AutoModelForCausalLM
    elif "ForConditionalGeneration" in config.architectures[0]:
        return AutoModelForVision2Seq
    else:
        raise NotImplementedError(f"Unknown architecture {config.architectures}")


def load_sharded_state_dicts(local_dir: str) -> Tuple[List[dict], int, Tuple[int, ...], Tuple[str, ...]]:
    """Load all sharded state dicts and return mesh information."""
    # Find world size and rank 0 file
    world_size = 0
    for filename in os.listdir(local_dir):
        match = re.match(r"model_world_size_(\d+)_rank_0\.pt", filename)
        if match:
            world_size = int(match.group(1))
            break
    if not world_size:
        raise ValueError("No model file with the proper format found")

    # Load rank 0 to get mesh info
    rank0_state = torch.load(
        os.path.join(local_dir, f"model_world_size_{world_size}_rank_0.pt"), 
        map_location="cpu"
    )
    pivot_key = sorted(rank0_state.keys())[0]
    weight = rank0_state[pivot_key]
    
    if not isinstance(weight, DTensor):
        raise TypeError("Expected DTensor in state dict")
    
    device_mesh = weight.device_mesh
    mesh = device_mesh.mesh
    mesh_dim_names = device_mesh.mesh_dim_names

    print(f"Got device mesh {mesh}, mesh_dim_names {mesh_dim_names}")

    if mesh_dim_names not in (("fsdp",),):
        raise ValueError(f"Unsupported mesh_dim_names {mesh_dim_names}")

    # Prepare list for all state dicts
    state_dicts = [rank0_state] + [None] * (world_size - 1)

    # Load remaining shards in parallel
    def load_shard(rank):
        if rank == 0:
            return rank0_state
        model_path = os.path.join(local_dir, f"model_world_size_{world_size}_rank_{rank}.pt")
        return torch.load(model_path, map_location="cpu", weights_only=False)

    with ThreadPoolExecutor(max_workers=min(32, os.cpu_count())) as executor:
        for rank, state_dict in enumerate(executor.map(load_shard, range(world_size))):
            state_dicts[rank] = state_dict

    return state_dicts, world_size, mesh.shape, mesh_dim_names


def merge_state_dicts(
    state_dicts: List[dict], 
    world_size: int, 
    mesh_shape: Tuple[int, ...], 
    mesh_dim_names: Tuple[str, ...]
) -> dict:
    """Merge sharded state dicts into a single state dict."""
    merged_state = {}
    param_placements: Dict[str, List[Placement]] = {}
    keys = set(state_dicts[0].keys())

    for key in keys:
        shards = []
        for state_dict in state_dicts:
            tensor = state_dict[key]
            if isinstance(tensor, DTensor):
                shards.append(tensor._local_tensor.bfloat16())
                placements = tuple(tensor.placements)
                # Handle replicated placement at dp dimension
                if mesh_dim_names[0] == "dp":
                    placements = placements[1:]
                if key not in param_placements:
                    param_placements[key] = placements
                else:
                    assert param_placements[key] == placements
            else:
                # Non-DTensor values (like buffers) are the same across ranks
                merged_state[key] = tensor.bfloat16()
                break
        
        if key in merged_state:
            continue
            
        # Merge shards according to their placements
        placements = param_placements[key]
        if len(mesh_shape) == 1:
            # 1-D sharding (FSDP only)
            assert len(placements) == 1
            merged_state[key] = merge_by_placement(shards, placements[0])
        else:
            # 2-D sharding (FSDP + TP)
            raise NotImplementedError("FSDP + TP is not supported yet")

    return merged_state


def save_merged_model(
    local_dir: str, 
    merged_state: dict, 
    hf_upload_path: Optional[str] = None
) -> None:
    """Save merged model and optionally upload to Hugging Face Hub."""
    hf_path = os.path.join(local_dir, "huggingface")
    config = AutoConfig.from_pretrained(hf_path)
    model_class = get_model_class(config)

    # Create model on meta device first to save memory
    with torch.device("meta"):
        model = model_class.from_config(config, torch_dtype=torch.bfloat16)
    
    # Load state dict onto CPU
    model.to_empty(device="cpu")
    model.load_state_dict(merged_state)
    
    print(f"Saving model to {hf_path}")
    model.save_pretrained(hf_path)

    if hf_upload_path:
        from huggingface_hub import HfApi
        api = HfApi()
        api.create_repo(repo_id=hf_upload_path, private=False, exist_ok=True)
        api.upload_folder(folder_path=hf_path, repo_id=hf_upload_path, repo_type="model")


def merge_and_save_model(local_dir: str, hf_upload_path: Optional[str] = None) -> None:
    """Main function to merge sharded models and save the result."""
    # Load all sharded state dicts
    state_dicts, world_size, mesh_shape, mesh_dim_names = load_sharded_state_dicts(local_dir)
    
    # Merge state dicts
    merged_state = merge_state_dicts(state_dicts, world_size, mesh_shape, mesh_dim_names)
    
    # Save merged model
    save_merged_model(local_dir, merged_state, hf_upload_path)
    
    # reorganize_folders(local_dir)



import os
import shutil
from pathlib import Path

def reorganize_folders(root_dir: str) -> None:
    """
    重组文件夹结构：
    1. 将actor/huggingface重命名为models并移动到父目录
    2. 删除除新models文件夹外的所有内容
    
    参数:
        root_dir: 最外层目录路径 (示例中的'step_20_reward_0.676')
    """
    root_path = Path(root_dir)
    actor_path = root_path / "actor"
    huggingface_path = actor_path / "huggingface"
    
    # 验证目录结构是否符合预期
    if not actor_path.exists():
        raise FileNotFoundError(f"未找到actor目录: {actor_path}")
    if not huggingface_path.exists():
        raise FileNotFoundError(f"未找到huggingface目录: {huggingface_path}")
    
    # 新models目录路径 (与actor同级)
    models_path = root_path / "models"
    
    print(f"正在将 {huggingface_path} 移动到 {models_path}")
    
    # 移动并重命名huggingface文件夹
    shutil.move(str(huggingface_path), str(models_path))
    
    print("正在清理原始文件...")
    
    # 删除原始actor目录及其内容
    shutil.rmtree(str(actor_path))
    
    # 删除其他可能存在的文件 (根据图片描述)
    for item in root_path.glob("*"):
        if item.name != "models":
            if item.is_file():
                item.unlink()
            elif item.is_dir():
                shutil.rmtree(str(item))
    
    print("文件夹重组完成！")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", required=False, type=str, help="The path for your saved model")
    parser.add_argument("--hf_upload_path", default=None, type=str, 
                       help="The path of the huggingface repo to upload")
    args = parser.parse_args()
    
    merge_and_save_model("/mnt/lyc/wuxinrui/R1_training/training/TCM4_addthinkprunedata/step_17_reward_0.668/actor")
    reorganize_folders("/mnt/lyc/wuxinrui/R1_training/training/TCM4_addthinkprunedata/step_17_reward_0.668")