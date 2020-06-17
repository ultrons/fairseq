#/usr/share/torch-xla-nightly/pytorch/xla/scripts/debug_run.py \
# --tidy --outfile /tmp/debug_run_`date +%d%m%y_%H_%M_%S`.tar.gz -- python -u \
python \
 train.py \
 /home/sivaibhav/wav2vec/manifest \
	 --tpu \
	 --bf16 \
	 --distributed-world-size 8 \
--max-sentences 1 \
--num-workers 6 \
--max-update 40000 \
--save-interval 1 \
--no-save \
--disable-validation \
--no-epoch-checkpoints \
--arch wav2vec \
--task audio_pretraining \
--lr 1e-06 \
--min-lr 1e-09 \
--optimizer adam \
--max-lr 1e-05 \
--lr-scheduler cosine \
--conv-feature-layers '[(512, 10, 5), (512, 8, 4), (512, 4, 2), (512, 4, 2), (512, 4, 2), (512, 1, 1), (512, 1, 1), (512, 1, 1)]' \
--conv-aggregator-layers '[(512, 2, 1), (512, 3, 1), (512, 4, 1), (512, 5, 1), (512, 6, 1), (512, 7, 1), (512, 8, 1), (512, 9, 1), (512, 10, 1), (512, 11, 1), (512, 12, 1), (512, 13, 1)]' \
	--activation gelu --offset auto --skip-connections-agg --residual-scale 0.5 \
	--log-keys '["prob_perplexity","code_perplexity","temp"]' --vq-type gumbel --vq-groups 2 --vq-depth 2 \
	--combine-groups --vq-vars 320 --vq-temp '(2,0.5,0.999995)' --prediction-steps 12 --warmup-updates 1000 \
--log-compression \
--warmup-init-lr 1e-07 \
--criterion binary_cross_entropy \
--num-negatives 10 \
--max-sample-size 150000 \
	--max-tokens 300000 --cross-sample-negatives 0 --update-freq 1 --seed 2 --skip-invalid-size-inputs-valid-test \
--skip-invalid-size-inputs-valid-test \
--log-interval 20 \
--log-format simple
