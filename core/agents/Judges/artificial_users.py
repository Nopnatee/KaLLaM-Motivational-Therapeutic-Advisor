from dataclasses import dataclass
from typing import Literal, List
import random

@dataclass(frozen=True)
class Persona:
    depression_severity: Literal["mild","moderate","severe"]
    openness: Literal["low","medium","high"]
    dominance: Literal["submissive","neutral","dominant"]
    age_group: Literal["teen","young_adult","adult"]
    willingness_to_disclose: Literal["low","medium","high"]

class PersonaFactory:
    """
    Generates artificial users from clinical-like dimensions.
    """
    CHOICES = {
        "depression_severity": ["mild", "moderate", "severe"],
        "openness": ["low", "medium", "high"],
        "dominance": ["submissive", "neutral", "dominant"],
        "age_group": ["teen", "young_adult", "adult"],
        "willingness_to_disclose": ["low", "medium", "high"],
    }

    def sample(self, n: int = 10, seed: int | None = None) -> List[Persona]:
        if seed is not None:
            random.seed(seed)
        personas = []
        for _ in range(n):
            kwargs = {k: random.choice(v) for k, v in self.CHOICES.items()}
            personas.append(Persona(**kwargs))
        return personas
