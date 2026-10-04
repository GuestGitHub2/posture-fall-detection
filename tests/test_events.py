import json
from pathlib import Path

from src.events import EventBus


def test_callbacks_are_isolated_and_jsonl_is_local(tmp_path: Path) -> None:
    bus = EventBus(tmp_path / "events.jsonl")
    events = []

    def failed(event):
        raise RuntimeError("integration unavailable")

    bus.subscribe(failed)
    bus.subscribe(events.append)
    event = bus.emit("fall_detected", 3, 0.94, "lying", 1.25)
    assert events == [event]
    assert json.loads(bus.path.read_text())["track_id"] == 3
