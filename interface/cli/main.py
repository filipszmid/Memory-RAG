"""
src/cli.py
====================
CLI interface for the Personal Memory Module.
"""

import sys
from pathlib import Path
from typing import Optional

import click
from loguru import logger

from src.config.settings import settings
from src.pipeline.extraction import FactExtractionPipeline


@click.command()
@click.option(
    "--input",
    "-i",
    "input_dir",
    default=str(Path(__file__).parent.parent.parent / "example_conversations"),
    show_default=True,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Folder containing conversation JSON files.",
)
@click.option(
    "--output",
    "-o",
    "output_dir",
    default=str(Path(__file__).parent.parent.parent / "outputs"),
    show_default=True,
    type=click.Path(file_okay=False, path_type=Path),
    help="Folder to write output JSON files.",
)
@click.option(
    "--provider",
    "-p",
    type=click.Choice(["gemini", "openai"], case_sensitive=False),
    default="gemini",
    show_default=True,
    help="LLM provider.",
)
@click.option(
    "--model",
    "-m",
    default=None,
    help="Model override.",
)
@click.option(
    "--delay",
    "-d",
    default=settings.default_delay,
    show_default=True,
    type=int,
    help="Seconds to wait between API calls.",
)
def cli(
    input_dir: Path, output_dir: Path, provider: str, model: Optional[str], delay: int
):
    """
    Extract atomic user facts from LLM conversation logs.
    """
    chosen_model = model or (
        settings.gemini_default_model
        if provider == "gemini"
        else settings.openai_default_model
    )

    try:
        pipeline = FactExtractionPipeline(
            input_dir=input_dir,
            output_dir=output_dir,
            provider_name=provider,
            model=chosen_model,
            delay=delay,
        )
        pipeline.run()
    except Exception as e:
        logger.error(f"Pipeline failed to initialize or execute: {e}")
        sys.exit(1)


if __name__ == "__main__":
    cli()
