"""Mode-specific trainers for ProEssenLLM."""

from .species_holdout_trainer import SpeciesHoldoutTrainer
from .within_species_trainer import WithinSpeciesTrainer


def create_trainer(args):
    if args.mode == "within_species":
        return WithinSpeciesTrainer(args)
    if args.mode == "species_holdout":
        return SpeciesHoldoutTrainer(args)
    raise ValueError(f"Unsupported mode: {args.mode}")


__all__ = ["SpeciesHoldoutTrainer", "WithinSpeciesTrainer", "create_trainer"]
