"""Load Ragas Personas from a YAML file.

The YAML schema:

    personas:
      - name: <str>
        role_description: <str>
      - name: <str>
        role_description: <str>
"""

from __future__ import annotations

from pathlib import Path

import yaml
from ragas.testset.persona import Persona


def load_personas(path: Path) -> list[Persona]:
    """Load Personas from a YAML file. Fails loud on missing/malformed input."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Persona file not found: {path}")

    data = yaml.safe_load(path.read_text()) or {}
    if "personas" not in data:
        raise ValueError(f"{path} is missing the top-level 'personas' key")

    raw = data["personas"]
    if not isinstance(raw, list) or len(raw) == 0:
        raise ValueError(f"{path} must contain at least one persona under 'personas'")

    personas: list[Persona] = []
    for i, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ValueError(f"{path}: personas[{i}] is not a mapping")
        name = item.get("name")
        role = item.get("role_description")
        if not name or not role:
            raise ValueError(f"{path}: personas[{i}] missing name or role_description")
        personas.append(Persona(name=str(name), role_description=str(role).strip()))

    return personas
