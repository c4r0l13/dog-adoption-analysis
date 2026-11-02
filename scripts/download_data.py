"""Download the City of Long Beach Animal Care Services intake/outcome records (2017-2024).

Source: City of Long Beach open data portal, as curated by TidyTuesday (2025-03-04):
https://github.com/rfordatascience/tidytuesday/tree/main/data/2025/2025-03-04
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

URL = "https://raw.githubusercontent.com/rfordatascience/tidytuesday/main/data/2025/2025-03-04/longbeach.csv"


def main() -> None:
    out = Path("data/raw")
    out.mkdir(parents=True, exist_ok=True)
    target = out / "longbeach.csv"
    if not target.exists():
        urllib.request.urlretrieve(URL, target)
    print(f"Data ready in {target.resolve()}")


if __name__ == "__main__":
    main()
