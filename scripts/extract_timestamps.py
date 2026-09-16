from pathlib import Path

import click
import numpy as np

from gmri2fem.utils import find_timestamps


@click.command()
@click.option("--timetable", "-t", type=click.Path(exists=True), required=True)
@click.option("--subject", "-sub", type=str, required=True)
@click.option("--sequence", "-seq", type=str, required=True)
@click.option("--output", "-o", type=click.Path(), required=True)
def extract_sequence_timestamps(
    timetable: Path,
    output: Path,
    subject: str,
    sequence: str,
):
    acq_times = find_timestamps(timetable, sequence, subject)
    times = np.array(acq_times)
    np.savetxt(output, np.maximum(0.0, times))


if __name__ == "__main__":
    extract_sequence_timestamps()
