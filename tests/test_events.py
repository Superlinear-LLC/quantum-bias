import asyncio
import json

from quantum_lite.app import create_app
from quantum_lite.config import Settings
from quantum_lite import events
from quantum_lite.events import EventBus, sse_stream


def parse(chunk):
    lines = chunk.strip().split("\n")
    return lines[0].removeprefix("event: "), json.loads(lines[1].removeprefix("data: "))


def test_queue_is_bounded_and_drops_are_reported(monkeypatch):
    monkeypatch.setattr(events, "QUEUE_SIZE", 3)

    async def main():
        bus = EventBus()
        sub = bus.subscribe("s1")
        for i in range(5):
            bus.publish("bit", {"i": i}, "s1")
        assert sub.queue.qsize() == 3 and sub.dropped == 2
        gen = sse_stream(sub, bus)
        ev, data = parse(await gen.__anext__())
        assert ev == "bit" and data["i"] == 0 and data["dropped"] == 2
        ev, data = parse(await gen.__anext__())
        assert data["i"] == 1 and "dropped" not in data
        await gen.aclose()
        assert not bus._subs

    asyncio.run(main())


def test_heartbeat_and_filtering():
    async def main():
        bus = EventBus()
        sess, glob = bus.subscribe("s1"), bus.subscribe(None)
        bus.publish("bit", {"x": 1}, "s2")  # other session
        bus.publish("job", {"x": 2}, "s2")
        bus.publish("batch", {"x": 3}, "s1")
        assert sess.queue.qsize() == 1  # only its own session
        assert glob.queue.qsize() == 1  # global: job events only (batch/bit are not global)
        gen = sse_stream(bus.subscribe("quiet"), bus, heartbeat=0.05)
        ev, data = parse(await gen.__anext__())
        assert ev == "heartbeat"
        await gen.aclose()

    asyncio.run(main())


def test_live_job_states_are_published(tmp_path):
    settings = Settings(data_dir=tmp_path / "d", sim_latency_seconds=0.05)

    async def main():
        app = create_app(settings)
        mgr = app.state.mgr
        s = await mgr.create({"provider": "sim", "mode": "live", "num_qubits": 3, "aggregation": "xor",
                              "shots_per_job": 5, "interval_ms": 20, "auto_refill": False,
                              "refill_below": 0, "refill_shots": None,
                              "stream_seconds": 60, "stream_max_jobs": None})
        sub = mgr.bus.subscribe(s.id)
        await mgr.press(s)
        got = []
        while not sub.queue.empty():
            got.append(sub.queue.get_nowait())
        states = [d["state"] for e, d in got if e == "job"]
        assert states[0] == "submitting" and "queued" in states and "running" in states and states[-1] == "done"
        kinds = [e for e, _ in got]
        assert "bit" in kinds and "stats" in kinds
        job = next(d for e, d in got if e == "job" and d["state"] == "done")
        assert job["simulated_latency"] is True and job["queue_position"] is None

    asyncio.run(main())
