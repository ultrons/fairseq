#/home/sivaibhav/xla/scripts/debug_run.py \
# --tidy --outfile /tmp/debug_run.tar.gz -- python -u \
python \
train.py \
 /home/sivaibhav/fairseq/wav2vec/manifest \
	 --tpu \
--max-sentences 16 \
--save-dir /home/sivaibhav/wav2vec/model \
--num-workers 6 \
--bf16 \
--max-update 400 \
--save-interval 1 \
--no-epoch-checkpoints \
--arch wav2vec \
--task audio_pretraining \
--lr 1e-06 \
--min-lr 1e-09 \
--optimizer adam \
--max-lr 0.005 \
--lr-scheduler cosine \
--conv-feature-layers '[(512, 10, 5), (512, 8, 4), (512, 4, 2), (512, 4, 2), (512, 4, 2), (512, 1, 1), (512, 1, 1)]' \
--conv-aggregator-layers '[(512, 2, 1), (512, 3, 1), (512, 4, 1), (512, 5, 1), (512, 6, 1), (512, 7, 1), (512, 8, 1), (512, 9, 1), (512, 10, 1), (512, 11, 1), (512, 12, 1), (512, 13, 1)]' \
--skip-connections-agg \
--residual-scale 0.5 \
--log-compression \
--warmup-updates 500 \
--warmup-init-lr 1e-07 \
--criterion binary_cross_entropy \
--num-negatives 10 \
--max-sample-size 150000 \
--max-tokens 1500000 \
--skip-invalid-size-inputs-valid-test \
--log-interval 1
