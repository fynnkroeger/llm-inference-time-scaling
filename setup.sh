mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm ~/miniconda3/miniconda.sh
source ~/miniconda3/bin/activate

conda create -n inference-time-scaling -c defaults python=3.12 pip git
conda activate inference-time-scaling
pip install -r requirements.txt
git clone https://github.com/openai/human-eval
pip install -e human-eval
