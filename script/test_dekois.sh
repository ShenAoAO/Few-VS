python /root/autodl-tmp/Drug-fewshot/unimol/test.py \
    "/root/autodl-tmp/Drug-fewshot/data/" \
    --user-dir /root/autodl-tmp/Drug-fewshot/unimol \
    --valid-subset test \
    --results-path /root/autodl-tmp/Drug-fewshot/test \
    --num-workers 8 \
    --ddp-backend=c10d \
    --batch-size 2 \
    --task drugclip \
    --loss in_batch_softmax \
    --arch fewshot \
    --fp16 \
    --fp16-init-scale 4 \
    --fp16-scale-window 256 \
    --seed 1 \
    --path /root/autodl-tmp/Drug-fewshot/checkpoint_best.pt \
    --finetune-pocket-model pocket_pre_220816.pt \
    --finetune-mol-model mol_pre_no_h_220816.pt \
    --log-interval 100 \
    --ft 16 \
    --log-format simple \
    --max-pocket-atoms 511 \
    --test-task DEKOIS \
    --lr 0.01 \
    --sample-time 2 \
    --mol-token 5 \
    --pocket-token 3 \
    --epoch-train 10

python /root/autodl-tmp/Drug-fewshot/unimol/test.py \
    "/root/autodl-tmp/Drug-fewshot/data/" \
    --user-dir /root/autodl-tmp/Drug-fewshot/unimol \
    --valid-subset test \
    --results-path /root/autodl-tmp/Drug-fewshot/test \
    --num-workers 8 \
    --ddp-backend=c10d \
    --batch-size 2 \
    --task drugclip \
    --loss in_batch_softmax \
    --arch fewshot \
    --fp16 \
    --fp16-init-scale 4 \
    --fp16-scale-window 256 \
    --seed 1 \
    --path /root/autodl-tmp/Drug-fewshot/checkpoint_best.pt \
    --finetune-pocket-model pocket_pre_220816.pt \
    --finetune-mol-model mol_pre_no_h_220816.pt \
    --log-interval 100 \
    --ft 8 \
    --log-format simple \
    --max-pocket-atoms 511 \
    --test-task DEKOIS \
    --lr 0.01 \
    --sample-time 2 \
    --mol-token 5 \
    --pocket-token 3 \
    --epoch-train 10

python /root/autodl-tmp/Drug-fewshot/unimol/test.py \
    "/root/autodl-tmp/Drug-fewshot/data/" \
    --user-dir /root/autodl-tmp/Drug-fewshot/unimol \
    --valid-subset test \
    --results-path /root/autodl-tmp/Drug-fewshot/test \
    --num-workers 8 \
    --ddp-backend=c10d \
    --batch-size 2 \
    --task drugclip \
    --loss in_batch_softmax \
    --arch fewshot \
    --fp16 \
    --fp16-init-scale 4 \
    --fp16-scale-window 256 \
    --seed 1 \
    --path /root/autodl-tmp/Drug-fewshot/checkpoint_best.pt \
    --finetune-pocket-model pocket_pre_220816.pt \
    --finetune-mol-model mol_pre_no_h_220816.pt \
    --log-interval 100 \
    --ft 4 \
    --log-format simple \
    --max-pocket-atoms 511 \
    --test-task DEKOIS \
    --lr 0.01 \
    --sample-time 2 \
    --mol-token 5 \
    --pocket-token 3 \
    --epoch-train 10

python /root/autodl-tmp/Drug-fewshot/unimol/test.py \
    "/root/autodl-tmp/Drug-fewshot/data/" \
    --user-dir /root/autodl-tmp/Drug-fewshot/unimol \
    --valid-subset test \
    --results-path /root/autodl-tmp/Drug-fewshot/test \
    --num-workers 8 \
    --ddp-backend=c10d \
    --batch-size 2 \
    --task drugclip \
    --loss in_batch_softmax \
    --arch fewshot \
    --fp16 \
    --fp16-init-scale 4 \
    --fp16-scale-window 256 \
    --seed 1 \
    --path /root/autodl-tmp/Drug-fewshot/checkpoint_best.pt \
    --finetune-pocket-model pocket_pre_220816.pt \
    --finetune-mol-model mol_pre_no_h_220816.pt \
    --log-interval 100 \
    --ft 2 \
    --log-format simple \
    --max-pocket-atoms 511 \
    --test-task DEKOIS \
    --lr 0.01 \
    --sample-time 2 \
    --mol-token 5 \
    --pocket-token 3 \
    --epoch-train 10

python /root/autodl-tmp/Drug-fewshot/unimol/test.py \
    "/root/autodl-tmp/Drug-fewshot/data/" \
    --user-dir /root/autodl-tmp/Drug-fewshot/unimol \
    --valid-subset test \
    --results-path /root/autodl-tmp/Drug-fewshot/test \
    --num-workers 8 \
    --ddp-backend=c10d \
    --batch-size 2 \
    --task drugclip \
    --loss in_batch_softmax \
    --arch fewshot \
    --fp16 \
    --fp16-init-scale 4 \
    --fp16-scale-window 256 \
    --seed 1 \
    --path /root/autodl-tmp/Drug-fewshot/checkpoint_best.pt \
    --finetune-pocket-model pocket_pre_220816.pt \
    --finetune-mol-model mol_pre_no_h_220816.pt \
    --log-interval 100 \
    --ft 16 \
    --log-format simple \
    --max-pocket-atoms 511 \
    --test-task DEKOIS \
    --lr 0.01 \
    --sample-time 1 \
    --mol-token 5 \
    --pocket-token 3 \
    --epoch-train 10

python /root/autodl-tmp/Drug-fewshot/unimol/test.py \
    "/root/autodl-tmp/Drug-fewshot/data/" \
    --user-dir /root/autodl-tmp/Drug-fewshot/unimol \
    --valid-subset test \
    --results-path /root/autodl-tmp/Drug-fewshot/test \
    --num-workers 8 \
    --ddp-backend=c10d \
    --batch-size 2 \
    --task drugclip \
    --loss in_batch_softmax \
    --arch fewshot \
    --fp16 \
    --fp16-init-scale 4 \
    --fp16-scale-window 256 \
    --seed 1 \
    --path /root/autodl-tmp/Drug-fewshot/checkpoint_best.pt \
    --finetune-pocket-model pocket_pre_220816.pt \
    --finetune-mol-model mol_pre_no_h_220816.pt \
    --log-interval 100 \
    --ft 8 \
    --log-format simple \
    --max-pocket-atoms 511 \
    --test-task DEKOIS \
    --lr 0.01 \
    --sample-time 1 \
    --mol-token 5 \
    --pocket-token 3 \
    --epoch-train 10

python /root/autodl-tmp/Drug-fewshot/unimol/test.py \
    "/root/autodl-tmp/Drug-fewshot/data/" \
    --user-dir /root/autodl-tmp/Drug-fewshot/unimol \
    --valid-subset test \
    --results-path /root/autodl-tmp/Drug-fewshot/test \
    --num-workers 8 \
    --ddp-backend=c10d \
    --batch-size 2 \
    --task drugclip \
    --loss in_batch_softmax \
    --arch fewshot \
    --fp16 \
    --fp16-init-scale 4 \
    --fp16-scale-window 256 \
    --seed 1 \
    --path /root/autodl-tmp/Drug-fewshot/checkpoint_best.pt \
    --finetune-pocket-model pocket_pre_220816.pt \
    --finetune-mol-model mol_pre_no_h_220816.pt \
    --log-interval 100 \
    --ft 4 \
    --log-format simple \
    --max-pocket-atoms 511 \
    --test-task DEKOIS \
    --lr 0.01 \
    --sample-time 1 \
    --mol-token 5 \
    --pocket-token 3 \
    --epoch-train 10

python /root/autodl-tmp/Drug-fewshot/unimol/test.py \
    "/root/autodl-tmp/Drug-fewshot/data/" \
    --user-dir /root/autodl-tmp/Drug-fewshot/unimol \
    --valid-subset test \
    --results-path /root/autodl-tmp/Drug-fewshot/test \
    --num-workers 8 \
    --ddp-backend=c10d \
    --batch-size 2 \
    --task drugclip \
    --loss in_batch_softmax \
    --arch fewshot \
    --fp16 \
    --fp16-init-scale 4 \
    --fp16-scale-window 256 \
    --seed 1 \
    --path /root/autodl-tmp/Drug-fewshot/checkpoint_best.pt \
    --finetune-pocket-model pocket_pre_220816.pt \
    --finetune-mol-model mol_pre_no_h_220816.pt \
    --log-interval 100 \
    --ft 2 \
    --log-format simple \
    --max-pocket-atoms 511 \
    --test-task DEKOIS \
    --lr 0.01 \
    --sample-time 1 \
    --mol-token 5 \
    --pocket-token 3 \
    --epoch-train 10

python /root/autodl-tmp/Drug-fewshot/unimol/test.py \
    "/root/autodl-tmp/Drug-fewshot/data/" \
    --user-dir /root/autodl-tmp/Drug-fewshot/unimol \
    --valid-subset test \
    --results-path /root/autodl-tmp/Drug-fewshot/test \
    --num-workers 8 \
    --ddp-backend=c10d \
    --batch-size 2 \
    --task drugclip \
    --loss in_batch_softmax \
    --arch fewshot \
    --fp16 \
    --fp16-init-scale 4 \
    --fp16-scale-window 256 \
    --seed 1 \
    --path /root/autodl-tmp/Drug-fewshot/checkpoint_best.pt \
    --finetune-pocket-model pocket_pre_220816.pt \
    --finetune-mol-model mol_pre_no_h_220816.pt \
    --log-interval 100 \
    --ft 16 \
    --log-format simple \
    --max-pocket-atoms 511 \
    --test-task DEKOIS \
    --lr 0.01 \
    --sample-time 3 \
    --mol-token 5 \
    --pocket-token 3 \
    --epoch-train 10

python /root/autodl-tmp/Drug-fewshot/unimol/test.py \
    "/root/autodl-tmp/Drug-fewshot/data/" \
    --user-dir /root/autodl-tmp/Drug-fewshot/unimol \
    --valid-subset test \
    --results-path /root/autodl-tmp/Drug-fewshot/test \
    --num-workers 8 \
    --ddp-backend=c10d \
    --batch-size 2 \
    --task drugclip \
    --loss in_batch_softmax \
    --arch fewshot \
    --fp16 \
    --fp16-init-scale 4 \
    --fp16-scale-window 256 \
    --seed 1 \
    --path /root/autodl-tmp/Drug-fewshot/checkpoint_best.pt \
    --finetune-pocket-model pocket_pre_220816.pt \
    --finetune-mol-model mol_pre_no_h_220816.pt \
    --log-interval 100 \
    --ft 8 \
    --log-format simple \
    --max-pocket-atoms 511 \
    --test-task DEKOIS \
    --lr 0.01 \
    --sample-time 3 \
    --mol-token 5 \
    --pocket-token 3 \
    --epoch-train 10

python /root/autodl-tmp/Drug-fewshot/unimol/test.py \
    "/root/autodl-tmp/Drug-fewshot/data/" \
    --user-dir /root/autodl-tmp/Drug-fewshot/unimol \
    --valid-subset test \
    --results-path /root/autodl-tmp/Drug-fewshot/test \
    --num-workers 8 \
    --ddp-backend=c10d \
    --batch-size 2 \
    --task drugclip \
    --loss in_batch_softmax \
    --arch fewshot \
    --fp16 \
    --fp16-init-scale 4 \
    --fp16-scale-window 256 \
    --seed 1 \
    --path /root/autodl-tmp/Drug-fewshot/checkpoint_best.pt \
    --finetune-pocket-model pocket_pre_220816.pt \
    --finetune-mol-model mol_pre_no_h_220816.pt \
    --log-interval 100 \
    --ft 4 \
    --log-format simple \
    --max-pocket-atoms 511 \
    --test-task DEKOIS \
    --lr 0.01 \
    --sample-time 3 \
    --mol-token 5 \
    --pocket-token 3 \
    --epoch-train 10

python /root/autodl-tmp/Drug-fewshot/unimol/test.py \
    "/root/autodl-tmp/Drug-fewshot/data/" \
    --user-dir /root/autodl-tmp/Drug-fewshot/unimol \
    --valid-subset test \
    --results-path /root/autodl-tmp/Drug-fewshot/test \
    --num-workers 8 \
    --ddp-backend=c10d \
    --batch-size 2 \
    --task drugclip \
    --loss in_batch_softmax \
    --arch fewshot \
    --fp16 \
    --fp16-init-scale 4 \
    --fp16-scale-window 256 \
    --seed 1 \
    --path /root/autodl-tmp/Drug-fewshot/checkpoint_best.pt \
    --finetune-pocket-model pocket_pre_220816.pt \
    --finetune-mol-model mol_pre_no_h_220816.pt \
    --log-interval 100 \
    --ft 2 \
    --log-format simple \
    --max-pocket-atoms 511 \
    --test-task DEKOIS \
    --lr 0.01 \
    --sample-time 3 \
    --mol-token 5 \
    --pocket-token 3 \
    --epoch-train 10