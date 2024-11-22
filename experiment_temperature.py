from inference import run_experiment
import yaml


try:
    config_path = "./config.yaml"
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    for temperature in [0.2, 0.4, 0.6, 0.8, 1, 1.2]:
        # Set temperature for each run
        config["sampling"]["temperature"] = temperature
        out_path = run_experiment(config)
        print(f"Results successfully stored in {out_path}")
except FileNotFoundError:
    print(f"Config file not found: {config_path}")
except yaml.YAMLError as e:
    print(f"Error in YAML file: {e}")
