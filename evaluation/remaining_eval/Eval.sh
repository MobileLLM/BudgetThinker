
# Qwen2.5-Math-Instruct Series
PROMPT_TYPE="deepseek3"

DATA_NAME="math500"
# DATA_NAME="amc23"
# DATA_NAME="aime24"

export CUDA_VISIBLE_DEVICES="0,1,2,3"

export remaining=8ratio
# export remaining=default
# export remaining=remaining250

MODEL_NAME_OR_PATH='/path1/path2/Model_Name/models'
# notice!
# the path must form like: /path1/path2/Model_Name/models
# the "Model_Name" must be the penultimate directory

PARENT_DIR=$(dirname "$MODEL_NAME_OR_PATH")
MODEL_NAME=$(basename "$PARENT_DIR")
echo MODEL_NAME: $MODEL_NAME

export tip=8ratio
# export tip=prompt_v2
# export tip=prompt_v1

export stage=2
# export stage=1

export mode=TIP-$tip-STAGE-$stage
export model=MODEL-$MODEL_NAME
export modelname=MODEL-$MODEL_NAME-TIP-$tip-STAGE-$stage-DATA-$DATA_NAME

bash ./sh/remaining.sh $PROMPT_TYPE $MODEL_NAME_OR_PATH $DATA_NAME
# bash ./sh/remaining_sample_n.sh $PROMPT_TYPE $MODEL_NAME_OR_PATH $DATA_NAME