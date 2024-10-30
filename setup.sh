conda create -n inference-time-scaling -c defaults python=3.12 pip git
conda activate inference-time-scaling
pip install -r requirements.txt
git clone https://github.com/openai/human-eval
pip install -e human-eval
