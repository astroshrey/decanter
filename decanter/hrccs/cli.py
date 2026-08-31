"""Command-line interface for ``decanter-hrccs``."""

from __future__ import annotations

import argparse

from decanter.hrccs.config import load_config
from decanter.hrccs.pipeline import run


def main() -> None:
    parser = argparse.ArgumentParser(
        description="SVD-based HRCCS analysis of wavelength-calibrated Decanter spectra"
    )
    parser.add_argument("config", help="TOML configuration file")
    args = parser.parse_args()
    config = load_config(args.config)
    results = run(config)
    for result in results:
        print(f"{result.species}: SVD={result.selected.count}, "
              f"observed local max={result.selected.local_peak_snr:+.2f} sigma "
              f"at ({result.selected.local_peak_kp_kms:.1f}, "
              f"{result.selected.local_peak_vsys_kms:.1f}) km/s, "
              f"injected local max={result.injected.local_peak_snr:+.2f} sigma, "
              f"global-null FAP={result.null_false_alarm_fraction:.3f}")


if __name__ == "__main__":
    main()
