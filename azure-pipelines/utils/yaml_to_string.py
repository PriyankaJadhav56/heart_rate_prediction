import json
import yaml
import os
import sys

json_path = sys.argv[1]

def replace_solution_config(data, solution_config_data):
    if isinstance(data, dict):
        for key, value in data.items():
            if key == 'solution_config':
                data[key] = f"{solution_config_data}"
            elif isinstance(value, (dict, list)):
                replace_solution_config(value, solution_config_data)
    elif isinstance(data, list):
        for item in data:
            replace_solution_config(item, solution_config_data)
    return data

with open('data_config/SolutionConfig.yaml', 'r') as yaml_file:
    yaml_data = yaml.safe_load(yaml_file)

solution_config = json.dumps(yaml_data)
solution_config_data = json.loads(solution_config)

with open(json_path, 'r') as file:
    data = json.load(file)

modified_data = replace_solution_config(data, solution_config_data)

with open(json_path, 'w') as file:
    json.dump(modified_data, file, indent=4)



