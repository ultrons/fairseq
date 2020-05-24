# Training
SAVE="save/dynamic_conv_iwslt"
mkdir -p $SAVE 

/home/sivaibhav/xla/scripts/debug_run.py \
 --tidy --outfile /tmp/dynamic_conv_debug_run_`date +%d%m%y_%H_%M_%S`.tar.gz -- python -u \
$(which fairseq-train) data-bin/iwslt14.tokenized.de-en \
    --tpu \
    --clip-norm 0 --optimizer adam --lr 0.0005 \
    --source-lang de --target-lang en --max-tokens 4000 --no-progress-bar \
    --log-interval 100 --min-lr '1e-09' --weight-decay 0.0001 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --lr-scheduler inverse_sqrt \
    --ddp-backend=no_c10d \
    --max-update 50000 --warmup-updates 4000 --warmup-init-lr '1e-07' \
    --adam-betas '(0.9, 0.98)' --keep-last-epochs 10 \
    -a lightconv_iwslt_de_en --save-dir $SAVE \
    --dropout 0.3 --attention-dropout 0.1 --weight-dropout 0.1 \
    --encoder-glu 0 --decoder-glu 0


