# TODO: cleanup in case of no .arrow files but cache-* files available.
python3 ~/wtpsplit/xla_spawn.py --num_cores ${TPU_NUM_DEVICES} wtpsplit/train/train_xlmr.py $1