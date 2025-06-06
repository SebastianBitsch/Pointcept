### 
### Test that finetuning on the scannet provided dataset works, this helps just debug that we can run it 
### 

#!/bin/sh
### General options
### –- specify queue --
#BSUB -q gpua100
### -- set the job Name --
#BSUB -J ego-ft

### -- ask for number of cores --
#BSUB -n 8
#BSUB -R "rusage[mem=3GB]"

### -- Select the resources --
#BSUB -gpu "num=1:mode=exclusive_process"

### -- set walltime limit: hh:mm --
#BSUB -W 24:00
#BSUB -o logs/ego_%J.out
#BSUB -e logs/ego_%J.err

nvidia-smi
# Load the cuda module
module load cuda/12.4

# print some info
/appl/cuda/11.6.0/samples/bin/x86_64/linux/release/deviceQuery

export CUDA_HOME="/appl/cuda/12.4.0"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"
# See: https://www.hpc.dtu.dk/?page_id=2129, volta=8.0
export TORCH_CUDA_ARCH_LIST="8.0"

cd ~/Pointcept
source ~/sonata/.venv/bin/activate

# Arguments
# -m : NUM_MACHINE
# -g : NUM_GPU
# -c : CONFIG
# -d : DATASET
# -n : EXP_NAME
# -w : WEIGHT
# -p : PYTHON
# -r : RESUME

# See: https://github.com/Pointcept/Pointcept/blob/59891348502bca9ae58dcdffa605d62a519f4cbf/pointcept/models/sonata/README.md?plain=1#L52
bash scripts/finetune.sh \
    -m 1 \
    -g 1 \
    -c _semseg-sonata-v1m1-0c-scannet-ft-ego \
    -d sonata \
    -n ego-debug \
    -w /work3/s204157/data/ego3d/exp/scannet/debug/model/sonata.pth

