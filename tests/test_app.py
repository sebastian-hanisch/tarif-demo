"""End-to-end Smoke-Test via Streamlits offiziellem AppTest-Framework: laedt app.py mit den
Standardeinstellungen, klickt jeden Preset-Button und faehrt jeden Slider an seine Grenzen und prueft, dass
kein Python-Fehler auftritt - insbesondere `streamlit.errors.StreamlitDuplicateElementId` (mehrere
st.plotly_chart-Aufrufe ohne eindeutiges key= koennen zufaellig identischen Inhalt rendern und kollidieren)
und `StreamlitAPIException` bei Slidern mit berechneten Grenzen (min == max)."""

import os

from streamlit.testing.v1 import AppTest

APP_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")
TIMEOUT = int(os.environ.get("APPTEST_TIMEOUT", "300"))  # unter Last (parallele Läufe) reichen 180 s nicht immer; Umgebungsvariable zum Anpassen
AUTOPLAY_MARKER = "\u25b6"  # Auto-Play-Buttons (Animation mit sleep) im Smoke-Test auslassen


def _fresh_app() -> AppTest:
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    return at


def test_app_loads_without_exception():
    _fresh_app()


def test_every_button_click_does_not_raise():
    labels = [b.label for b in _fresh_app().button]
    assert labels, "keine Buttons gefunden"
    for index, label in enumerate(labels):
        if AUTOPLAY_MARKER in label:
            continue
        at = _fresh_app()
        at.button[index].click().run(timeout=TIMEOUT)
        assert not at.exception, (label, [str(e) for e in at.exception])


def test_every_slider_at_its_min_and_max_does_not_raise():
    labels = [s.label for s in _fresh_app().slider]
    if not labels:
        return  # diese App hat keine Schieberegler; die Auswahl-Regler prüft test_every_selectbox_option_does_not_raise (unten)
    for index, label in enumerate(labels):
        for edge in ("min", "max"):
            at = _fresh_app()
            slider = at.slider[index]
            value = slider.min if edge == "min" else slider.max
            if isinstance(slider.value, int):
                value = int(value)
            slider.set_value(value).run(timeout=TIMEOUT)
            assert not at.exception, (label, edge, [str(e) for e in at.exception])


def test_every_selectbox_option_does_not_raise():
    """Die App hat Auswahl-Regler statt Schieberegler: jede Stufe jedes Reglers einmal setzen (Seitenleiste und Abschnitte)."""
    boxes = [(i, sb.label, list(sb.options)) for i, sb in enumerate(_fresh_app().selectbox)]
    assert boxes, "keine Auswahl-Regler gefunden"
    for index, label, options in boxes:
        for option in options:
            at = _fresh_app()
            at.selectbox[index].select(option).run(timeout=TIMEOUT)
            assert not at.exception, (label, option, [str(e) for e in at.exception])


def test_seed_input_at_its_limits_does_not_raise():
    for value in (0, 9999):
        at = _fresh_app()
        at.number_input[0].set_value(value).run(timeout=TIMEOUT)
        assert not at.exception, (value, [str(e) for e in at.exception])
