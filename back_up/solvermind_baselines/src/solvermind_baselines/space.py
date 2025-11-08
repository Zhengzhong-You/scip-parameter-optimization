import yaml
from typing import Dict, Any, List, Tuple
import ConfigSpace as CS

def load_whitelist(path: str) -> List[Dict[str, Any]]:
    with open(path, "r") as f:
        data = yaml.safe_load(f) or []
    return data

def build_configspace(wl: List[Dict[str, Any]]) -> Tuple[CS.ConfigurationSpace, List[Dict[str, Any]]]:
    cs = CS.ConfigurationSpace()
    enriched = []
    for item in wl:
        name = item["name"]; typ = item["type"]
        if typ == "float":
            hp = CS.hyperparameters.FloatHyperparameter(
                name, lower=float(item["lower"]), upper=float(item["upper"]),
                log=bool(item.get("log", False)))
        elif typ == "int":
            hp = CS.hyperparameters.IntegerHyperparameter(
                name, lower=int(item["lower"]), upper=int(item["upper"]))
        elif typ == "bool":
            choices = item.get("choices", [False, True])
            hp = CS.hyperparameters.CategoricalHyperparameter(name, choices=[bool(x) for x in choices])
        elif typ == "cat":
            choices = item["choices"]
            hp = CS.hyperparameters.CategoricalHyperparameter(name, choices=choices)
        else:
            raise ValueError(f"Unknown type in whitelist: {typ}")
        cs.add_hyperparameter(hp)
        enriched.append(item)
    return cs, enriched

def rbfopt_variables(wl: List[Dict[str, Any]]):
    var_lower = []; var_upper = []; var_type = ""; cat_maps = []
    for item in wl:
        typ = item["type"]
        if typ == "float":
            var_lower.append(float(item["lower"])); var_upper.append(float(item["upper"])); var_type += "R"; cat_maps.append(None)
        elif typ == "int":
            var_lower.append(int(item["lower"])); var_upper.append(int(item["upper"])); var_type += "I"; cat_maps.append(None)
        elif typ == "bool":
            var_lower.append(0); var_upper.append(1); var_type += "I"; cat_maps.append(("__bool__", [False, True]))
        elif typ == "cat":
            choices = list(item["choices"])
            var_lower.append(0); var_upper.append(len(choices)-1); var_type += "I"; cat_maps.append((item["name"], choices))
        else:
            raise ValueError(f"Unknown type {typ}")
    dim = len(wl)
    return dim, var_lower, var_upper, var_type, cat_maps
