cd eval/efficiency

METHOD="pbs" # baselines: flashattn, minference, flexprefill, xattention, meanpooling


# Evaluate on 8k, 16k, 32k, 64k, 128k
for len in 8 16 32 64 128; do
     CUDA_VISIBLE_DEVICES=0 python eval_efficiency.py \
          --method $METHOD \
          --len $((len * 1024))
done

# Evaluate on 256k, tp=4
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --standalone eval_efficiency.py \
     --method $METHOD \
     --len $((256 * 1024))


# Evaluate on 512k, tp=8
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 --standalone eval_efficiency.py \
     --method $METHOD \
     --len $((512 * 1024))
