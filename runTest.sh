modal shell --gpu a100 \
 --image "nvidia/cuda:12.2.0-devel-ubuntu22.04" \
 --add-python 3.11 \
 -c "git clone https://github.com/ajuneja23/paged_attn_dlm && cd paged_attn_dlm && sh runFA.sh"