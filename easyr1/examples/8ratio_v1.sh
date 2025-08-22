
set -x
export stage=2
export VLLM_ATTENTION_BACKEND=XFORMERS
export CUDA_VISIBLE_DEVICES=0,1,2,3
export steady=8ratio_v1
export TENSORBOARD_DIR=tensorlog_${steady}

MODEL_PATH=/path/to/your/model
export remaining=8ratio

python3 -m verl.trainer.main \
    config=examples/8ratio_v1.yaml \
    worker.actor.model.model_path=${MODEL_PATH} \
    trainer.n_gpus_per_node=4