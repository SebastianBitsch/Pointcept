export CUDA_HOME="/appl/cuda/12.4.0"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"
export TORCH_CUDA_ARCH_LIST="8.0"


Lowered batchsize from 24 to 4, unchanged learning rate - might fuck us up idk

# download the submodules
git submodule add git@github.com:jensModvig/02501-Ego3D.git
git submodule update --init --recursive
