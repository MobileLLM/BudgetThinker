export special_token_loss=T


deepspeed --num_gpus 8 src/train.py \
--deepspeed examples/deepspeed/ds_z0_config.json \
--stage sft \
--model_name_or_path /path/to/your/model \
--do_train \
--dataset 8ratio_SFT_below10000 \
--template deepseek3 \
--finetuning_type full \
--output_dir  /path/to/your/output_1 \
--overwrite_cache \
--per_device_train_batch_size 2 \
--gradient_accumulation_steps 8 \
--lr_scheduler_type cosine \
--logging_steps 10 \
--save_steps 2000 \
--learning_rate 2e-5 \
--num_train_epochs 2.0 \
--plot_loss \
--bf16


deepspeed --num_gpus 8 src/train.py \
--deepspeed examples/deepspeed/ds_z0_config.json \
--stage sft \
--model_name_or_path /path/to/your/output_1 \
--do_train \
--dataset 8ratio_SFT_below10000 \
--template deepseek3 \
--finetuning_type full \
--output_dir  /path/to/your/output_2 \
--overwrite_cache \
--per_device_train_batch_size 2 \
--gradient_accumulation_steps 8 \
--lr_scheduler_type cosine \
--logging_steps 10 \
--save_steps 2000 \
--learning_rate 2e-5 \
--num_train_epochs 4.0 \
--plot_loss \
--bf16