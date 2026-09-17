from __future__ import annotations

import argparse

from config.settings import settings
from config.startup import (
    StartupConfigurationError,
    validate_dashboard_settings,
    validate_runtime_settings,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate Sapient service settings.")
    parser.add_argument("service", choices=("runtime", "dashboard"))
    args = parser.parse_args()

    validator = (
        validate_runtime_settings
        if args.service == "runtime"
        else validate_dashboard_settings
    )
    try:
        validator(settings)
    except StartupConfigurationError as exc:
        print(f"Configuration invalid: {exc}")
        return 1

    print(f"{args.service.capitalize()} configuration is valid.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
