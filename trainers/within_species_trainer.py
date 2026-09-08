"""Known-species trainer using per-species sample-level splits."""

from .base_trainer import BaseTrainer


class WithinSpeciesTrainer(BaseTrainer):
    expected_mode = "within_species"
