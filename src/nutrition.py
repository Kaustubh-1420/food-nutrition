"""Nutrition lookup and portion scaling."""
from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from .config import NUTRITION_DB_PATH, display_name


@dataclass
class Nutrition:
    dish: str
    display: str
    serving_g: float
    calories: float
    protein_g: float
    carbs_g: float
    fat_g: float
    fiber_g: float
    category: str

    def as_table_rows(self) -> list[list]:
        return [
            ["Calories", f"{self.calories:.0f} kcal"],
            ["Protein",  f"{self.protein_g:.1f} g"],
            ["Carbs",    f"{self.carbs_g:.1f} g"],
            ["Fat",      f"{self.fat_g:.1f} g"],
            ["Fiber",    f"{self.fiber_g:.1f} g"],
            ["Serving",  f"{self.serving_g:.0f} g"],
        ]


@lru_cache(maxsize=1)
def _db() -> dict:
    with Path(NUTRITION_DB_PATH).open() as f:
        return json.load(f)


def lookup(dish_key: str, portion: float = 1.0) -> Nutrition | None:
    """Return nutrition for one serving × portion multiplier."""
    db = _db()
    entry = db.get(dish_key)
    if entry is None:
        return None
    serving_g = entry["serving_g"] * portion
    factor = serving_g / 100.0
    return Nutrition(
        dish=dish_key,
        display=display_name(dish_key),
        serving_g=serving_g,
        calories=entry["calories"] * factor,
        protein_g=entry["protein_g"] * factor,
        carbs_g=entry["carbs_g"] * factor,
        fat_g=entry["fat_g"] * factor,
        fiber_g=entry["fiber_g"] * factor,
        category=entry.get("category", "unknown"),
    )
