from typing import Dict, Any, List

def format_value_for_scip(v):
    if isinstance(v, bool):
        return "TRUE" if v else "FALSE"
    return str(v)

def dict_to_scip_set_commands(params: Dict[str, Any]) -> List[str]:
    return [f"set {k} {format_value_for_scip(v)}" for k, v in params.items()]

def write_param_file(path: str, params: Dict[str, Any]):
    with open(path, "w") as f:
        for line in dict_to_scip_set_commands(params):
            f.write(line + "\n")
