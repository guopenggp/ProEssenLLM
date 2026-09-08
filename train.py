#!/usr/bin/env python
"""Unified ProEssenLLM training entry point."""

from configs import parse_configuration
from trainers import create_trainer


def main() -> None:
    arguments = parse_configuration()
    trainer = create_trainer(arguments)
    trainer.run()


if __name__ == "__main__":
    main()
