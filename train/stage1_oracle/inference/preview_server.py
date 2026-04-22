from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Iterator, Protocol

from flask import Flask, Response, send_file

from ..events.tokens import Stage1Vocab
from .assembler import StreamingAssembler
from .canvas_preview import CANVAS_PREVIEW_HTML
from .checkpoint_runner import Stage1CheckpointRunner
from .feature_source import FullSongFeatureSource
from .scheduler import BufferPolicy


_LOGGER = logging.getLogger(__name__)
_PUBLIC_STREAM_ERROR = {
    "message": "Stage 1 preview generation failed. Check server logs for details.",
    "code": "preview_stream_failed",
}


@dataclass(frozen=True)
class PreviewEvent:
    name: str
    data: dict[str, object]

    def to_sse(self) -> str:
        return f"event: {self.name}\ndata: {json.dumps(self.data, sort_keys=True, separators=(',', ':'))}\n\n"


class _FeatureSource(Protocol):
    audio_duration_ms: float

    def iter_windows(self): ...

    def features_for_window(self, window): ...


class _Runner(Protocol):
    checkpoint_name: str

    def decode_window(self, *, features, window, difficulty: float, open_hold_mask: int): ...


EventFactory = Callable[[], Iterable[PreviewEvent | tuple[str, dict[str, object]]]]


class PreviewServer:
    def __init__(
        self,
        *,
        audio_path: str | Path,
        event_factory: EventFactory,
    ) -> None:
        self.audio_path = Path(audio_path).resolve()
        self.event_factory = event_factory

    def create_app(self) -> Flask:
        app = Flask(__name__)

        @app.get("/")
        def index() -> Response:
            return Response(CANVAS_PREVIEW_HTML, mimetype="text/html")

        @app.get("/audio")
        def audio() -> Response:
            return send_file(self.audio_path, conditional=True)

        @app.get("/events")
        def events() -> Response:
            def generate() -> Iterator[str]:
                for event in self.event_factory():
                    preview_event = event if isinstance(event, PreviewEvent) else PreviewEvent(event[0], event[1])
                    yield preview_event.to_sse()

            return Response(generate(), mimetype="text/event-stream")

        return app


def build_preview_event_stream(
    *,
    source: _FeatureSource,
    runner: _Runner,
    assembler: StreamingAssembler,
    difficulty: float,
    target_buffer_ms: int = 2000,
    rebuffer_floor_ms: int = 750,
) -> Iterator[PreviewEvent]:
    buffer_policy = BufferPolicy(target_buffer_ms=target_buffer_ms, rebuffer_floor_ms=rebuffer_floor_ms)
    ready_sent = False
    try:
        yield PreviewEvent(
            "metadata",
            {
                "audio_duration_ms": int(round(source.audio_duration_ms)),
                "difficulty": float(difficulty),
                "checkpoint_name": runner.checkpoint_name,
                "target_buffer_ms": target_buffer_ms,
                "rebuffer_floor_ms": rebuffer_floor_ms,
            },
        )
        for window in source.iter_windows():
            features = source.features_for_window(window)
            decoded = runner.decode_window(
                features=features,
                window=window,
                difficulty=difficulty,
                open_hold_mask=assembler.open_hold_mask,
            )
            for batch in assembler.process_window(decoded):
                yield PreviewEvent("batch", batch.to_json())
            status = buffer_policy.update(
                playhead_ms=0,
                committed_through_ms=assembler.committed_through_ms,
            )
            if status.ready and not ready_sent:
                ready_sent = True
                yield PreviewEvent(
                    "ready",
                    {
                        "committed_through_ms": assembler.committed_through_ms,
                        "generated_through_ms": assembler.generated_through_ms,
                    },
                )
            yield PreviewEvent(
                "status",
                {
                    "generated_through_ms": assembler.generated_through_ms,
                    "committed_through_ms": assembler.committed_through_ms,
                    "buffer_state": status.state,
                    "future_ms": status.future_ms,
                },
            )
        for batch in assembler.finish():
            yield PreviewEvent("batch", batch.to_json())
        done_status = buffer_policy.update(
            playhead_ms=0,
            committed_through_ms=assembler.committed_through_ms,
            done=True,
        )
        if done_status.ready and not ready_sent:
            yield PreviewEvent(
                "ready",
                {
                    "committed_through_ms": assembler.committed_through_ms,
                    "generated_through_ms": assembler.generated_through_ms,
                },
            )
        yield PreviewEvent(
            "done",
            {
                "generated_through_ms": assembler.generated_through_ms,
                "committed_through_ms": assembler.committed_through_ms,
            },
        )
    except Exception:
        _LOGGER.exception("Stage 1 preview event stream failed")
        yield PreviewEvent("error", dict(_PUBLIC_STREAM_ERROR))


def create_preview_server(
    *,
    checkpoint_path: str | Path,
    audio_path: str | Path,
    beatmap_path: str | Path,
    difficulty: float,
    device_name: str = "auto",
    target_buffer_ms: int = 2000,
) -> PreviewServer:
    Stage1Vocab().difficulty_bucket_id(difficulty)
    runner = Stage1CheckpointRunner.load(checkpoint_path, device_name=device_name)
    source = FullSongFeatureSource(
        audio_path=audio_path,
        beatmap_path=beatmap_path,
        bpm_log_mean=runner.bpm_log_mean,
        bpm_log_std=runner.bpm_log_std,
    )
    return PreviewServer(
        audio_path=audio_path,
        event_factory=lambda: build_preview_event_stream(
            source=source,
            runner=runner,
            assembler=StreamingAssembler(vocab=runner.vocab),
            difficulty=difficulty,
            target_buffer_ms=target_buffer_ms,
        ),
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the Stage 1 oracle streaming mania preview server.")
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--audio-path", required=True)
    parser.add_argument("--beatmap-path", required=True)
    parser.add_argument("--difficulty", required=True, type=float)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--target-buffer-ms", default=2000, type=int)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=5000, type=int)
    args = parser.parse_args(argv)

    server = create_preview_server(
        checkpoint_path=args.checkpoint_path,
        audio_path=args.audio_path,
        beatmap_path=args.beatmap_path,
        difficulty=args.difficulty,
        device_name=args.device,
        target_buffer_ms=args.target_buffer_ms,
    )
    app = server.create_app()
    print(f"Stage 1 preview listening at http://{args.host}:{args.port}", flush=True)
    app.run(host=args.host, port=args.port, threaded=True)


if __name__ == "__main__":
    main()
