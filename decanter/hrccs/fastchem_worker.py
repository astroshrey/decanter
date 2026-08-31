"""Private FastChem subprocess entry point used for OpenMP isolation."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from decanter.hrccs.models import _equilibrium_profiles_local


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("usage: python -m decanter.hrccs.fastchem_worker REQUEST RESPONSE")
    request, response = map(Path, sys.argv[1:])
    with np.load(request, allow_pickle=False) as data:
        pressure = np.asarray(data["pressure_bar"], dtype=float)
        temperature = np.asarray(data["temperature_k"], dtype=float)
        payload = json.loads(str(data["payload_json"]))
    species = tuple(str(value) for value in payload["species"])
    vmr, mmw, metadata = _equilibrium_profiles_local(
        SimpleNamespace(**payload["config"]),
        float(payload["metallicity_dex"]),
        pressure,
        temperature,
        species,
    )
    np.savez_compressed(
        response,
        species=np.asarray(species, dtype="U32"),
        vmr=np.stack([vmr[name] for name in species]),
        mean_molecular_weight=mmw,
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )


if __name__ == "__main__":
    main()
