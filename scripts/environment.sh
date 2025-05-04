mamba create -n diffilo python=3.8
mamba activate diffilo
mamba install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
pip install torch_geometric
mamba install -c conda-forge pyscipopt
mamba install -c conda-forge ecole
pip install hydra-core --upgrade
pip install tensorboardX tensorboard
python -m pip install gurobipy==10.0
