"""Unknown-species trainer with target species reserved for final evaluation."""

from .base_trainer import BaseTrainer


class SpeciesHoldoutTrainer(BaseTrainer):
    expected_mode = "species_holdout"
