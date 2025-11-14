import ConfigSpace as CS
from typing import Dict, Any, List, Tuple


def build_configspace(wl: List[Dict[str, Any]]) -> Tuple[CS.ConfigurationSpace, List[Dict[str, Any]]]:
    cs = CS.ConfigurationSpace()
    enriched = []
    for item in wl:
        name = item["name"]; typ = item["type"]
        if typ == "float":
            hp = CS.hyperparameters.UniformFloatHyperparameter(
                name, lower=float(item["lower"]), upper=float(item["upper"]), log=bool(item.get("log", False))
            )
        elif typ == "int":
            hp = CS.hyperparameters.UniformIntegerHyperparameter(name, lower=int(item["lower"]), upper=int(item["upper"]))
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

