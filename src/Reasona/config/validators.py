def require(cfg: dict, key: str, section: str):
    if key not in cfg or cfg[key] is None:
        raise ValueError(
            f"Missing required config key '{key}' in section '{section}'"
        )
    return cfg[key]