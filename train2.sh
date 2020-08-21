TAG=$1
#/usr/share/torch-xla-nightly/pytorch/xla/scripts/debug_run.py \
# --tidy --outfile /tmp/debug_run_${TAG}_`date +%d%m%y_%H_%M_%S`.tar.gz -- python -u \
python \
 train.py \
 /home/sivaibhav/wav2vec/manifest \
 --tpu \
	 --distributed-world-size 1 \
--max-update 500 \
--log-interval 1 \
--encoder-layers $1 \
--num-workers 6 \
--task audio_pretraining \
--criterion wav2vec \
--arch wav2vec2 \
--quantize-targets \
--extractor-mode default \
--conv-feature-layers '[(512, 10, 5)] + [(512, 3, 2)] * 4 + [(512,2,2)] * 2' \
--final-dim 256 \
--latent-vars 320 \
--latent-groups 2 \
--latent-temp '(2,0.5,0.999995)' \
--infonce \
--optimizer adam \
--adam-betas '(0.9,0.98)' \
--adam-eps 1e-06 \
--lr-scheduler polynomial_decay \
--lr 0.0005 \
--warmup-updates 32000 \
--mask-length 10 \
--mask-prob 0.65 \
--mask-selection static \
--mask-other 0 \
--encoder-layerdrop 0.05 \
--dropout-input 0.1 \
--dropout-features 0.1 \
--feature-grad-mult 0.1 \
--loss-weights '[0.1, 10]' \
--conv-pos 128 \
--conv-pos-groups 16 \
--num-negatives 100 \
--cross-sample-negatives 0 \
--max-sample-size 250000 \
--min-sample-size 32000 \
--no-save \
--dropout 0.1 \
--attention-dropout 0.1 \
--weight-decay 0.01 \
--max-tokens 1400000 \
--skip-invalid-size-inputs-valid-test \
--log-format simple \
--disable-validation \
--ddp-backend no_c10d
