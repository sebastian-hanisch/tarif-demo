"""Sucht für jedes Preset Zufalls-Seeds außerhalb der Messreihe (ab 500), auf denen die Abnahmekriterien aus tar_stories.py gelten.

Aufruf:  python -B tools/preset_search.py [erster Seed] [Anzahl]    (Standard: 500, 40)
Ausgabe: je Preset die Seeds, die das Kriterium erfüllen (mit Reserve: Kennzahl und Abstand zur Schwelle stehen in tests/test_stories.py)."""
import os
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tar_constants as C  # noqa: E402
import tar_evaluation as ev  # noqa: E402
import tar_stories as ST  # noqa: E402


def main(first=500, count=40):
    good = {name: [] for name in C.PRESETS}
    for seed in range(first, first + count):
        for name, preset in C.PRESETS.items():
            s = ST.settings_of(preset, seed=seed)
            a = ev.analyse(s)
            a_ref = ev.analyse(ST.reference_settings(s)) if name in ST.NEEDS_REFERENCE else None
            if ST.check(name, a, a_ref):
                lv = a.price_level["R"]
                ref = a_ref.price_level["R"] if a_ref else float("nan")
                good[name].append((seed, round(lv, 3), round(ref, 3), round(a.s_null / a.model.C0, 2)))
    for name, rows in good.items():
        print(f"{name}: {len(rows)}/{count} Seeds erfüllen das Kriterium; erste: {rows[:6]}  (Seed, Preisniveau Max-R, Referenzniveau, S_null/C0)")


if __name__ == "__main__":
    args = [int(x) for x in sys.argv[1:3]]
    main(*args)
