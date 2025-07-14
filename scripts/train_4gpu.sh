for ARGUMENT in "$@"
do
    KEY=$(echo $ARGUMENT | cut -f1 -d=)

    KEY_LENGTH=${#KEY}
    VALUE="${ARGUMENT:$KEY_LENGTH+1}"


    count=$(($count+1))
    if [ $((${#ARGUMENT} > $KEY_LENGTH)) == 1 ]
    then
        export "$KEY"="$VALUE"
        echo "$KEY = $VALUE | $KEY_LENGTH | ${#ARGUMENT} | $((${#ARGUMENT} > $KEY_LENGTH))"
    fi
done

export WANDB__SERVICE_WAIT=300
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
NUM_GPU="$(nvidia-smi --list-gpus | wc -l)"
echo "NUM_GPU=$NUM_GPU"

eval "$(python /ariesdv0/discovery.py 1)"
TF_CPP_MIN_LOG_LEVEL=2 OMP_NUM_THREADS=32 python -m torch.distributed.launch --nnodes 1 --nproc-per-node 2 --use-env \
    --master-addr ${MASTER_ADDR} --node-rank ${NODE_RANK} \
    scripts/run.py \
  --config-name=fractal \
  action_lr=0.00005 \
  vlm_lr=0.00005 \
  flow_sampling=beta \
  use_torch_compile=True \
  use_bf16=True \
  use_amp=True
