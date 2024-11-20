# llm-inference-time-scaling

## Working Conda
### First time installation
Install conda or miniconda, then run the following command. This might take some minutes due to installing pytorch. 
```bash
conda env create -f environment.yml
```

### Env Activation
```bash
conda activate inference-time-scaling 
```

### Keeping up to date
Do this everytime when somebody else has updated the environment.yml file.
```bash
conda env update
```

### Adding packages
If the package is available on conda.
```bash
conda install <package_name>
```

If not use pip:
```bash
pip install <package_name>
```

Then always export your enviroment to make it reproducible!
```bash
./conda-export-complete
```

Conda has an issue that pip installs are not exported when you run `conda env export --from-history`. This provided script fixes it and automatically updates the `environment.yaml`.