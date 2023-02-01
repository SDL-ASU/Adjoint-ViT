NUM_PROC=$1
shift
python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_PROC main_adjoined.py "$@"