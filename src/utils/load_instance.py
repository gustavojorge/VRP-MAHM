import json

def load_instance(instance_path: str) -> dict:
    with open(instance_path, "r") as f:
        return json.load(f)
