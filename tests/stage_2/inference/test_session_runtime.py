from __future__ import annotations

import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from train.stage_2.inference.session_runtime import SessionRuntime, SessionRuntimeConfig
from train.stage_2.timing.grid_fitting.types import TimingFitDiagnostics, TimingFitResult
from train.stage_2.timing.schema import FittedTimingGrid, FrameTimingPrediction, TimingSegment


class SessionRuntimeTests(unittest.TestCase):
    def test_prepare_audio_caches_mel_dense_timing_and_padding_mask(self) -> None:
        audio_path = Path("song.wav")
        mel = np.arange(3 * 160, dtype=np.float32).reshape(3, 160)
        prediction = _prediction(frame_count=512, source_path=audio_path)
        provider = _FakeTimingProvider(prediction)
        fitter = _FakeGridFitter()
        control_model = _FakeControlModel(control_dim=6)
        runtime = SessionRuntime(
            session_id="s1",
            model_runtime=_fake_model_runtime(provider, control_model=control_model),
            config=SessionRuntimeConfig(minimum_frame_count=4),
            mel_loader=_fake_mel_loader(mel),
            grid_fitter=fitter,
        )

        cache = runtime.prepare_audio(audio_path, audio_length_ms=1_234)

        self.assertIs(runtime.audio_cache, cache)
        self.assertEqual(cache.session_id, "s1")
        self.assertEqual(cache.audio_path, audio_path)
        self.assertEqual(cache.audio_length_ms, 1_234)
        self.assertEqual(cache.audio_length_source, "provided")
        self.assertEqual(cache.source_frame_count, 3)
        self.assertEqual(cache.padded_frame_count, 400)
        self.assertEqual(tuple(cache.full_mel.shape), (1, 400, 160))
        self.assertEqual(tuple(cache.full_dense_timing_v2.shape), (1, 400, 4))
        self.assertEqual(tuple(cache.padding_mask.shape), (1, 400))
        self.assertEqual(cache.padding_mask[:, :5].tolist(), [[False, False, False, True, True]])
        self.assertEqual(cache.frame_count_tensor.tolist(), [400])
        self.assertEqual(cache.source_frame_count_tensor.tolist(), [3])
        self.assertTrue(torch.equal(cache.full_mel[0, :3], torch.as_tensor(mel)))
        self.assertTrue(torch.equal(cache.full_mel[0, 3:], torch.zeros(397, 160)))
        self.assertTrue(torch.isfinite(cache.full_dense_timing_v2).all())
        self.assertIs(cache.beatthis_prediction, prediction)
        self.assertIs(cache.timing_fit_result, fitter.result)
        self.assertIs(cache.timing_grid, fitter.result.grid)
        self.assertEqual(provider.paths, [audio_path])
        self.assertEqual(fitter.predictions, [prediction])
        self.assertIsNotNone(runtime.control_cache)
        assert runtime.control_cache is not None
        self.assertEqual(runtime.control_cache.start_ms, 0)
        self.assertEqual(runtime.control_cache.target_start_frame, 0)
        self.assertEqual(runtime.control_cache.control_slice_start_frames.tolist(), [[0, 100, 200, 300]])
        self.assertEqual(tuple(runtime.control_cache.control_memory_8s.shape), (1, 400, 6))
        self.assertNotIn("density_teacher_8s", runtime.control_cache.as_model_batch())
        self.assertEqual(control_model.calls[0]["target_start_frame"], [0, 100, 200, 300])
        self.assertFalse(control_model.calls[0]["grad_enabled"])
        self.assertTrue(control_model.calls[0]["inference_mode"])

        batch = cache.as_model_batch()
        self.assertIs(batch["full_mel"], cache.full_mel)
        self.assertIs(batch["full_dense_timing_v2"], cache.full_dense_timing_v2)
        self.assertIs(batch["padding_mask"], cache.padding_mask)
        self.assertIs(batch["frame_count"], cache.frame_count_tensor)
        self.assertIs(batch["source_frame_count"], cache.source_frame_count_tensor)

    def test_prepare_audio_estimates_audio_length_from_mel_when_missing(self) -> None:
        runtime = SessionRuntime(
            session_id="s1",
            model_runtime=_fake_model_runtime(_FakeTimingProvider(_prediction())),
            config=SessionRuntimeConfig(minimum_frame_count=4),
            mel_loader=_fake_mel_loader(np.zeros((5, 160), dtype=np.float32)),
            grid_fitter=_FakeGridFitter(),
        )

        cache = runtime.prepare_audio("song.wav")

        self.assertEqual(cache.audio_length_ms, 100)
        self.assertEqual(cache.audio_length_source, "mel_frame_estimate")

    def test_reset_audio_cache_drops_cached_tensors(self) -> None:
        runtime = SessionRuntime(
            session_id="s1",
            model_runtime=_fake_model_runtime(_FakeTimingProvider(_prediction())),
            config=SessionRuntimeConfig(minimum_frame_count=4),
            mel_loader=_fake_mel_loader(np.zeros((5, 160), dtype=np.float32)),
            grid_fitter=_FakeGridFitter(),
        )
        runtime.prepare_audio("song.wav")

        runtime.reset_audio_cache()

        self.assertIsNone(runtime.audio_cache)
        self.assertIsNone(runtime.control_cache)

    def test_prepare_audio_accepts_start_ms_for_initial_control_window(self) -> None:
        control_model = _FakeControlModel(control_dim=3)
        runtime = SessionRuntime(
            session_id="s1",
            model_runtime=_fake_model_runtime(_FakeTimingProvider(_prediction()), control_model=control_model),
            config=SessionRuntimeConfig(minimum_frame_count=4),
            mel_loader=_fake_mel_loader(np.zeros((450, 160), dtype=np.float32)),
            grid_fitter=_FakeGridFitter(),
        )

        cache = runtime.prepare_audio("song.wav", start_ms=2_000)

        self.assertEqual(cache.padded_frame_count, 500)
        self.assertEqual(cache.padding_mask[:, 448:500].tolist()[0][:5], [False, False, True, True, True])
        self.assertIsNotNone(runtime.control_cache)
        assert runtime.control_cache is not None
        self.assertEqual(runtime.control_cache.start_ms, 2_000)
        self.assertEqual(runtime.control_cache.target_start_frame, 100)
        self.assertEqual(runtime.control_cache.control_slice_start_frames.tolist(), [[100, 200, 300, 400]])
        self.assertEqual(control_model.calls[0]["target_start_frame"], [100, 200, 300, 400])

    def test_prepare_control_reuses_audio_cache_without_rerunning_timing(self) -> None:
        prediction = _prediction()
        provider = _FakeTimingProvider(prediction)
        fitter = _FakeGridFitter()
        control_model = _FakeControlModel(control_dim=3)
        runtime = SessionRuntime(
            session_id="s1",
            model_runtime=_fake_model_runtime(provider, control_model=control_model),
            config=SessionRuntimeConfig(minimum_frame_count=4),
            mel_loader=_fake_mel_loader(np.zeros((500, 160), dtype=np.float32)),
            grid_fitter=fitter,
        )
        runtime.prepare_audio("song.wav")
        first_audio_cache = runtime.audio_cache

        control_cache = runtime.prepare_control(start_ms=1_000)

        self.assertIs(runtime.audio_cache, first_audio_cache)
        self.assertIs(runtime.control_cache, control_cache)
        self.assertEqual(provider.paths, [Path("song.wav")])
        self.assertEqual(fitter.predictions, [prediction])
        self.assertEqual(len(control_model.calls), 2)
        self.assertEqual(control_cache.control_slice_start_frames.tolist(), [[50, 150, 250, 350]])

    def test_prepare_control_batch_computes_multiple_windows(self) -> None:
        control_model = _FakeControlModel(control_dim=3)
        runtime = SessionRuntime(
            session_id="s1",
            model_runtime=_fake_model_runtime(_FakeTimingProvider(_prediction()), control_model=control_model),
            config=SessionRuntimeConfig(minimum_frame_count=4, max_control_batch_size=12),
            mel_loader=_fake_mel_loader(np.zeros((900, 160), dtype=np.float32)),
            grid_fitter=_FakeGridFitter(),
        )
        runtime.prepare_audio("song.wav")

        batch_cache = runtime.prepare_control_batch(start_ms_values=(0, 8_000, 16_000))

        self.assertIs(runtime.control_batch_cache, batch_cache)
        self.assertIsNone(runtime.control_cache)
        self.assertEqual(batch_cache.start_ms_values, (0, 8_000, 16_000))
        self.assertEqual(batch_cache.target_start_frames, (0, 400, 800))
        self.assertEqual(batch_cache.control_slice_start_frames.tolist(), [[0, 100, 200, 300], [400, 500, 600, 700], [800, 900, 1000, 1100]])
        self.assertEqual(tuple(batch_cache.control_memory_8s.shape), (3, 400, 3))
        self.assertEqual(float(batch_cache.control_memory_8s[1, 0, 0].item()), 400.0)
        self.assertEqual(float(batch_cache.control_memory_8s[1, 100, 0].item()), 500.0)
        self.assertEqual(float(batch_cache.control_memory_8s[2, 300, 0].item()), 1100.0)
        self.assertEqual(control_model.calls[-1]["target_start_frame"], [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100])

    def test_prepare_control_batch_rejects_more_than_max(self) -> None:
        runtime = SessionRuntime(
            session_id="s1",
            model_runtime=_fake_model_runtime(_FakeTimingProvider(_prediction())),
            config=SessionRuntimeConfig(minimum_frame_count=4, max_control_batch_size=2),
            mel_loader=_fake_mel_loader(np.zeros((900, 160), dtype=np.float32)),
            grid_fitter=_FakeGridFitter(),
        )
        runtime.prepare_audio("song.wav")

        with self.assertRaisesRegex(ValueError, "<= 2"):
            runtime.prepare_control_batch(start_ms_values=(0, 8_000, 16_000))

    def test_prepare_full_control_batches_whole_song(self) -> None:
        control_model = _FakeControlModel(control_dim=2)
        runtime = SessionRuntime(
            session_id="s1",
            model_runtime=_fake_model_runtime(_FakeTimingProvider(_prediction()), control_model=control_model),
            config=SessionRuntimeConfig(minimum_frame_count=4, max_control_batch_size=2),
            mel_loader=_fake_mel_loader(np.zeros((850, 160), dtype=np.float32)),
            grid_fitter=_FakeGridFitter(),
        )
        runtime.prepare_audio("song.wav")

        full_cache = runtime.prepare_full_control()

        self.assertIs(runtime.full_control_cache, full_cache)
        self.assertEqual(full_cache.max_batch_size, 2)
        self.assertEqual(full_cache.start_ms_values, (0, 8_000, 16_000))
        self.assertEqual(full_cache.target_start_frames, (0, 400, 800))
        self.assertEqual(tuple(full_cache.control_memory_8s.shape), (3, 400, 2))
        self.assertEqual(float(full_cache.control_memory_8s[2, 0, 0].item()), 800.0)
        self.assertEqual(float(full_cache.control_memory_8s[2, 300, 0].item()), 1100.0)
        self.assertEqual(len(control_model.calls), 3)
        self.assertEqual(control_model.calls[-2]["target_start_frame"], [0, 100, 200, 300, 400, 500, 600, 700])
        self.assertEqual(control_model.calls[-1]["target_start_frame"], [800, 900, 1000, 1100])

    def test_prepare_control_requires_audio_cache(self) -> None:
        runtime = SessionRuntime(
            session_id="s1",
            model_runtime=_fake_model_runtime(_FakeTimingProvider(_prediction())),
            config=SessionRuntimeConfig(minimum_frame_count=4),
            mel_loader=_fake_mel_loader(np.zeros((5, 160), dtype=np.float32)),
            grid_fitter=_FakeGridFitter(),
        )

        with self.assertRaisesRegex(RuntimeError, "prepare_audio"):
            runtime.prepare_control()

    def test_prepare_control_rejects_unaligned_start_ms(self) -> None:
        runtime = SessionRuntime(
            session_id="s1",
            model_runtime=_fake_model_runtime(_FakeTimingProvider(_prediction())),
            config=SessionRuntimeConfig(minimum_frame_count=4),
            mel_loader=_fake_mel_loader(np.zeros((5, 160), dtype=np.float32)),
            grid_fitter=_FakeGridFitter(),
        )
        runtime.prepare_audio("song.wav")

        with self.assertRaisesRegex(ValueError, "divisible"):
            runtime.prepare_control(start_ms=21)

    def test_prepare_audio_rejects_invalid_mel_shape(self) -> None:
        runtime = SessionRuntime(
            session_id="s1",
            model_runtime=_fake_model_runtime(_FakeTimingProvider(_prediction())),
            config=SessionRuntimeConfig(minimum_frame_count=4),
            mel_loader=_fake_mel_loader(np.zeros((5, 159), dtype=np.float32)),
            grid_fitter=_FakeGridFitter(),
        )

        with self.assertRaisesRegex(ValueError, "packed_mel"):
            runtime.prepare_audio("song.wav")


def _prediction(*, frame_count: int = 512, source_path: Path | str = "song.wav") -> FrameTimingPrediction:
    frame_indexes = np.arange(frame_count, dtype=np.float32)
    beat_prob = (np.sin(frame_indexes / 8.0) * 0.25 + 0.5).astype(np.float32)
    downbeat_prob = (np.cos(frame_indexes / 32.0) * 0.25 + 0.5).astype(np.float32)
    return FrameTimingPrediction(
        provider="fake",
        checkpoint_path="fake-checkpoint",
        source_path=Path(source_path).as_posix(),
        beat_prob=beat_prob,
        downbeat_prob=downbeat_prob,
        frame_rate_hz=50.0,
    )


def _fake_mel_loader(mel: np.ndarray):
    def load(audio_path: str | Path) -> np.ndarray:
        del audio_path
        return mel

    return load


def _fake_model_runtime(provider: _FakeTimingProvider, *, control_model: torch.nn.Module | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        device=torch.device("cpu"),
        beatthis_provider=provider,
        control_model=_FakeControlModel() if control_model is None else control_model,
    )


class _FakeTimingProvider:
    def __init__(self, prediction: FrameTimingPrediction) -> None:
        self.prediction = prediction
        self.paths: list[Path] = []

    def predict_file(self, audio_path: str | Path) -> FrameTimingPrediction:
        self.paths.append(Path(audio_path))
        return self.prediction


class _FakeControlModel(torch.nn.Module):
    def __init__(self, control_dim: int = 4) -> None:
        super().__init__()
        self.control_dim = int(control_dim)
        self.calls: list[dict[str, object]] = []

    def forward(
        self,
        *,
        context_mel: torch.Tensor,
        context_dense_timing_v2: torch.Tensor,
        normalized_difficulty: torch.Tensor,
        context_padding_mask: torch.Tensor,
        target_start_frame: torch.Tensor | None = None,
        **kwargs: object,
    ) -> SimpleNamespace:
        del context_dense_timing_v2, normalized_difficulty, context_padding_mask, kwargs
        batch_size = int(context_mel.shape[0])
        frames = int(context_mel.shape[1])
        if target_start_frame is None:
            start_values = torch.zeros(batch_size, dtype=torch.float32, device=context_mel.device)
            recorded_starts: list[int] | None = None
        else:
            start_values = target_start_frame.to(device=context_mel.device, dtype=torch.float32).reshape(batch_size)
            recorded_starts = [int(value) for value in target_start_frame.detach().cpu().reshape(-1).tolist()]
        self.calls.append(
            {
                "target_start_frame": recorded_starts,
                "grad_enabled": torch.is_grad_enabled(),
                "inference_mode": torch.is_inference_mode_enabled(),
            },
        )
        control_memory = torch.zeros(
            batch_size,
            frames,
            self.control_dim,
            dtype=context_mel.dtype,
            device=context_mel.device,
        )
        control_memory[:, :, 0] = start_values.reshape(batch_size, 1)
        value_pred = start_values.reshape(batch_size, 1, 1).expand(batch_size, 100, 1).to(dtype=context_mel.dtype)
        return SimpleNamespace(control_memory=control_memory, value_pred=value_pred)


class _FakeGridFitter:
    def __init__(self) -> None:
        self.predictions: list[FrameTimingPrediction] = []
        self.result = TimingFitResult(
            grid=FittedTimingGrid((TimingSegment(offset_ms=0.0, beat_length_ms=500.0, meter=4),)),
            score=0.99,
            diagnostics=TimingFitDiagnostics(
                selected_period_frames=25.0,
                selected_offset_frames=0.0,
                selected_bpm=120.0,
                candidate_count=1,
                half_tempo_score=0.0,
                double_tempo_score=0.0,
                raw_selected_bpm=120.0,
                raw_score=0.99,
                tempo_multiplier=1.0,
                segment_alias_switch_count=0,
                tempo_multiplier_distribution={"1.0": 1},
            ),
        )

    def fit(self, prediction: FrameTimingPrediction) -> TimingFitResult:
        self.predictions.append(prediction)
        return self.result


if __name__ == "__main__":
    unittest.main()
