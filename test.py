import json

def parse_methods_with_samples(file_path):
    """
    Parses a JSON file where each top-level key except 'samples' is a method,
    each containing 'root_causes' and 'edges'.
    'samples' key contains common samples shared by all methods.
    
    Args:
        file_path (str): Path to JSON file.
    
    Returns:
        dict: {
            'methods': {
                method_name: {
                    'root_causes': [...],
                    'edges': [...]
                },
                ...
            },
            'samples': [...]
        }
    """
    result = {
        "methods": {},
        "samples": []
    }

    try:
        with open(file_path, 'r') as f:
            data = json.load(f)

        for key, value in data.items():
            print(key)
            if key == "samples":
                if isinstance(value, list):
                    result["samples"] = value
            else:
                # Assume other keys are methods
                if isinstance(value, dict):
                    root_causes = value.get("root_causes", [])
                    edges = value.get("edges", [])
                    result["methods"][key] = {
                        "root_causes": root_causes,
                        "edges": edges
                    }

    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
    except json.JSONDecodeError:
        print(f"Error: File '{file_path}' is not valid JSON.")
    except Exception as e:
        print(f"Unexpected error: {e}")

    return result


filename = "RCA_inferences/sample_0_step_5_explanation.json"
parsed = parse_methods_with_samples(filename)

print("Methods found:", list(parsed["methods"].keys()))
for method, details in parsed["methods"].items():
    print(f"\nMethod: {method}")
    print("Root Causes:", details["root_causes"])
    print("Edges:", details["edges"])

print(f"\nTotal samples: {len(parsed['samples'])}")
print(f"First sample example: {parsed['samples'][0] if parsed['samples'] else 'No samples'}")
