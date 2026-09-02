"""
Free-running multi-year rollout for WeatherGenerator.

One prognostic stream, named by ``rollout.feedback_stream``, is fed back on itself:
its predicted state becomes its own next input, with no nudging and no
re-initialisation. Every other stream is re-read from the dataset at each step,
so prescribed boundary conditions stay external to the simulation. Running the
atmosphere against prescribed SST and sea ice is the AMIP case of this.

Each outer iteration is a full forward pass that re-reads the other streams,
re-encodes the carried state and predicts; the latent advances inside
``model.forward`` are not refreshed in between.

Tokenization is not reimplemented. Each step asks ``MultiStreamDataSampler``
for a real batch and overwrites only the fed-back stream's source values, so
coordinates, geoinfos and binning stay identical to training.
"""

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import zarr

logger = logging.getLogger(__name__)

# Fields exceeding this many standard deviations from their training mean are
# treated as a diverged rollout. A blown-up run that continues to completion
# wastes far more compute than one that aborts early.
BLOWUP_ZSCORE = 10.0


@dataclass
class RolloutState:
    """Physical atmospheric state carried between outer-loop iterations."""

    timestamp: np.datetime64
    # (n_points, n_source_channels), physical units, SOURCE channel order
    data: np.ndarray
    step: int = 0


class ChannelMap:
    """
    Maps predicted TARGET channels onto the SOURCE channels the encoder expects.

    Raises at construction if any source channel is not predicted -- that would
    make the feedback loop impossible to close, and is far better caught here
    than as a silently wrong multi-year rollout.
    """

    def __init__(self, source_channels: list[str], target_channels: list[str]):
        self.source_channels = list(source_channels)
        self.target_channels = list(target_channels)

        missing = set(self.source_channels) - set(self.target_channels)
        if missing:
            raise ValueError(
                f"Feedback loop cannot close: {len(missing)} source channel(s) are "
                f"never predicted by the model: {sorted(missing)}. "
                "The model cannot supply its own next input. Check source_exclude / "
                "target_exclude in the stream config."
            )

        t_index = {name: i for i, name in enumerate(self.target_channels)}
        self.gather_idx = np.array(
            [t_index[name] for name in self.source_channels], dtype=np.int64
        )
        self.dropped = sorted(set(self.target_channels) - set(self.source_channels))

        logger.info(
            "ChannelMap: %d target -> %d source channels (dropped, predicted but "
            "not fed back: %s)",
            len(self.target_channels),
            len(self.source_channels),
            self.dropped or "none",
        )

    def target_to_source(self, pred: np.ndarray) -> np.ndarray:
        """(n_points, n_target_channels) -> (n_points, n_source_channels)."""
        assert pred.shape[-1] == len(self.target_channels), (
            f"expected {len(self.target_channels)} target channels, got {pred.shape[-1]}"
        )
        return pred[..., self.gather_idx]


class _StateInjectingReader:
    """
    Reader shim: serves a caller-supplied physical state in place of the values
    on disk, while keeping the real coordinates, geoinfos and datetimes.

    Everything not overridden is delegated to the wrapped reader, so normalization
    statistics, channel lists and grid metadata all remain the genuine article.

    ``override`` must be in PHYSICAL units and SOURCE channel order --
    ``collect_datasources`` applies ``normalize_source_channels`` after
    ``get_source`` returns, so handing it normalized data would double-normalize.
    """

    def __init__(self, wrapped):
        # bypass __setattr__ delegation for our own attributes
        object.__setattr__(self, "_wrapped", wrapped)
        object.__setattr__(self, "override", None)
        object.__setattr__(self, "n_injected", 0)

    def get_source(self, idx):
        rdata = self._wrapped.get_source(idx)
        override = self.override
        if override is None:
            return rdata
        if override.shape != rdata.data.shape:
            raise ValueError(
                f"Injected state shape {override.shape} != reader source shape "
                f"{rdata.data.shape}. The carried state and the on-disk grid have "
                "diverged; refusing to roll out."
            )
        rdata.data = np.asarray(override, dtype=rdata.data.dtype)
        object.__setattr__(self, "n_injected", self.n_injected + 1)
        return rdata

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "_wrapped"), name)

    def __setattr__(self, name, value):
        if name in ("override", "n_injected"):
            object.__setattr__(self, name, value)
        else:
            setattr(self._wrapped, name, value)


class AMIPRollout:
    """
    Multi-year free-running rollout with prescribed boundary conditions.

    Parameters
    ----------
    model, model_params
        As returned by ``init_model_and_shard``. The model must have been trained
        with a forecasting objective -- ``fe_num_blocks > 0`` and
        ``masking_strategy: "forecast"``. A model trained as an autoencoder will
        run here and produce meaningless output.
    sampler
        A ``MultiStreamDataSampler``. Supplies batch geometry and fresh forcing.
    feedback_stream
        Name of the prognostic stream that is fed back. The forcing stream is
        NOT fed back -- it is re-read from disk every step, which is the point.
    """

    def __init__(
        self,
        model,
        model_params,
        sampler,
        cf,
        feedback_stream: str = "ERA5",
        device: str = "cuda",
    ):
        self.model = model
        self.model_params = model_params
        self.sampler = sampler
        self.cf = cf
        self.feedback_stream = feedback_stream
        self.device = device

        self.reader = self._get_reader(feedback_stream)
        self.channel_map = ChannelMap(
            source_channels=list(self.reader.source_channels),
            target_channels=list(self.reader.target_channels),
        )

        fc = cf.training_config.get("forecast", {})
        trained_num_steps = int(fc.get("num_steps", 1))
        if trained_num_steps < 1:
            raise ValueError(
                f"forecast.num_steps={trained_num_steps}; a model with no forecast steps "
                "cannot roll out. This is the autoencoder configuration."
            )

        # Base step in hours as the model was trained; hardcoding it would misalign
        # every _index_for() lookup for models trained at a different time_step.
        time_step = fc.get("time_step", None)
        base_step_hours = (
            int(np.timedelta64(time_step, "h") / np.timedelta64(1, "h"))
            if time_step is not None else 6
        )
        if base_step_hours < 1:
            raise ValueError(f"forecast.time_step resolved to {base_step_hours}h; expected >=1h.")

        # Outer-loop cadence H hours must be alias-free (H % 24 == 0 or 24 % H == 0),
        # or the sampled clock hour walks backward each iteration and aliases the diurnal
        # cycle into a spurious beat. Use the full trained chunk when it already is,
        # otherwise take the largest alias-free num_steps below it.
        def _alias_free(hours: int) -> bool:
            return hours % 24 == 0 or 24 % hours == 0

        rc = cf.get("rollout", {})
        requested = rc.get("num_steps", None)
        if requested is not None:
            self.num_steps = int(requested)
            if not (1 <= self.num_steps <= trained_num_steps):
                raise ValueError(
                    f"rollout.num_steps={self.num_steps} must be between 1 and the "
                    f"trained forecast.num_steps={trained_num_steps}; the model "
                    "cannot forecast more chunked steps per outer iteration than "
                    "it was trained for."
                )
        elif _alias_free(base_step_hours * trained_num_steps):
            self.num_steps = trained_num_steps
        else:
            self.num_steps = 1  # base_step_hours * 1 always satisfies 24 % H == 0 or H == 24
            for candidate in range(trained_num_steps, 0, -1):
                if _alias_free(base_step_hours * candidate):
                    self.num_steps = candidate
                    break
            logger.info(
                "AMIPRollout: trained forecast.num_steps=%d at a %dh base step "
                "(%dh/iter) does not divide 24h evenly; using num_steps=%d "
                "(%dh/iter) instead to avoid diurnal-cycle aliasing. Override "
                "with rollout.num_steps=%d to restore the old cadence.",
                trained_num_steps, base_step_hours, base_step_hours * trained_num_steps,
                self.num_steps, base_step_hours * self.num_steps, trained_num_steps,
            )

        # hours advanced per outer iteration == the forcing refresh interval
        self.hours_per_iter = base_step_hours * self.num_steps

        from weathergen.utils.utils import get_dtype

        self.mixed_precision_dtype = get_dtype(cf.get("mixed_precision_dtype", "bf16"))

        # we call _get_batch() directly and never reset(), where the rng is normally seeded
        if getattr(self.sampler, "rng", None) is None:
            self.sampler.rng = np.random.default_rng(self.sampler.data_loader_rng_seed)

        self.injector = self._install_injector()
        self.state: RolloutState | None = None
        logger.info(
            "AMIPRollout: %d forecast steps/pass -> forcing refreshed every %dh",
            self.num_steps,
            self.hours_per_iter,
        )

    def _index_for(self, t: np.datetime64) -> int:
        """
        Map a timestamp to the sampler's window index.

        ``TimeWindowHandler`` provides only the forward map (``window(idx)``), but
        the mapping is linear in ``t_start``/``t_window_step``, so the inverse is
        exact. We compute it and then **round-trip through ``window()`` to verify**
        -- an off-by-one here would silently roll out from the wrong date and
        pair the carried state with the wrong step's boundary data.
        """
        tw = self.sampler.time_window_handler
        idx = int((np.datetime64(t) - tw.t_start) // tw.t_window_step)

        rng = tw.get_index_range()
        if not (rng.start <= idx <= rng.end):
            raise IndexError(
                f"Timestamp {t} maps to window index {idx}, outside the dataset range "
                f"[{rng.start}, {rng.end}] (covering {tw.t_start} to {tw.t_end})."
            )

        got = tw.window(idx)
        got_start = getattr(got, "start", got)
        if np.datetime64(got_start) != np.datetime64(t):
            raise ValueError(
                f"Window index round-trip failed: {t} -> idx {idx} -> {got_start}. "
                f"The requested time is not aligned to the {tw.t_window_step} window "
                "grid; rolling out would silently use the wrong date."
            )
        return idx

    def _get_reader(self, stream_name: str):
        sd = self.sampler.streams_datasets[stream_name]
        return sd.readers[0] if hasattr(sd, "readers") else sd

    # ---------------------------------------------------------------- stepping

    def _install_injector(self) -> "_StateInjectingReader":
        """
        Wrap the atmospheric stream's reader so it serves our carried state
        instead of reading truth from zarr.

        Injection MUST happen here, upstream of tokenization. By the time data
        reaches ``StreamData.source_tokens_cells`` it has already been binned into
        healpix cells and padded; patching it there would mean reimplementing the
        tokenizer and risking divergence from what training did. Substituting at
        the reader means the framework tokenizes and normalizes our state through
        exactly the same code path it used during training.

        The other streams' readers are deliberately left untouched, so the prescribed
        boundary data is genuinely re-read every outer iteration. That is the whole
        point of the outer loop.
        """
        sd = self.sampler.streams_datasets[self.feedback_stream]
        if not hasattr(sd, "readers") or not sd.readers:
            raise RuntimeError(
                f"Stream '{self.feedback_stream}' exposes no readers to wrap; cannot inject "
                "state. Without injection the rollout would silently re-read truth from "
                "disk and produce a fake 'perfect' rollout."
            )
        if isinstance(sd.readers[0], _StateInjectingReader):
            return sd.readers[0]
        inj = _StateInjectingReader(sd.readers[0])
        sd.readers[0] = inj
        logger.info("State injector installed on stream '%s'", self.feedback_stream)
        return inj

    def _check_finite(self, arr: np.ndarray, t: np.datetime64) -> None:
        if not np.all(np.isfinite(arr)):
            n_bad = int((~np.isfinite(arr)).sum())
            raise FloatingPointError(
                f"Rollout diverged at {t}: {n_bad} non-finite values in predicted state."
            )
        mu = np.asarray(self.reader.mean)[self.reader.source_idx]
        sd = np.asarray(self.reader.stdev)[self.reader.source_idx]
        sd = np.where(sd > 0, sd, 1.0)
        z = np.abs((np.nanmean(arr, axis=0) - mu) / sd)
        if np.nanmax(z) > BLOWUP_ZSCORE:
            worst = int(np.nanargmax(z))
            raise FloatingPointError(
                f"Rollout blew up at {t}: channel "
                f"'{self.channel_map.source_channels[worst]}' is {z[worst]:.1f} sigma "
                f"from its training mean (threshold {BLOWUP_ZSCORE})."
            )

    @torch.no_grad()
    def step(self) -> RolloutState:
        """Advance one outer iteration: refresh forcing, re-encode, predict."""
        if self.state is None:
            raise RuntimeError("Rollout not initialized; call initialize() first.")

        t = self.state.timestamp
        idx = self._index_for(t)

        # only this stream is overridden; the others read fresh from the dataset
        self.injector.override = self.state.data
        before = self.injector.n_injected
        try:
            batch = self.sampler._get_batch(idx, self.num_steps)
        finally:
            self.injector.override = None

        if self.injector.n_injected == before:
            raise RuntimeError(
                "State injection did not fire: the sampler never called get_source on "
                "the wrapped reader. The rollout would have silently re-read truth "
                "from disk and produced a fake 'perfect' result."
            )

        # Mirror Trainer.train/validate exactly: move the whole ModelBatch, then
        # hand the model *source samples* (not the ModelBatch), under the same
        # autocast as training so inference numerics match.
        batch.to_device(self.device)
        source_samples = batch.get_source_samples()

        with torch.autocast(
            device_type="cuda",
            dtype=self.mixed_precision_dtype,
            enabled=bool(self.cf.get("with_mixed_precision", False)),
        ):
            # Model.forward takes (model_params, batch); how many steps it advances comes
            # from the batch itself via get_output_idxs(), not from an argument.
            out = self.model(self.model_params, source_samples)

        last = max(source_samples.get_output_idxs())
        pred = out.get_physical_prediction(last, self.feedback_stream)
        if isinstance(pred, (list, tuple)):
            pred = pred[0]

        # Predictions come back in tokenized (cell-packed) order and must be un-permuted
        # with idxs_inv, as validation_io.py does. Omitting this does not crash and the
        # fields stay plausible, but the spatial pattern is scrambled (18 h 2t: RMSE
        # 20.4 K vs 2.5 K for persistence).
        sdata = batch.target_samples.get_samples()[0].streams_data[self.feedback_stream]
        idxs_inv = sdata.idxs_inv[last]
        if idxs_inv is None or len(idxs_inv) == 0:
            raise RuntimeError(
                f"No idxs_inv for stream '{self.feedback_stream}' at step {last}; cannot "
                "restore predictions to physical point order. Rolling out on "
                "cell-packed order would silently produce a spatially scrambled state."
            )
        pred = pred[:, idxs_inv]

        pred = pred.detach().float().cpu().numpy()
        if pred.ndim == 3:  # (ensemble, points, channels) -> deterministic member
            pred = pred[0]

        physical = self.reader.denormalize_target_channels(pred)
        next_data = self.channel_map.target_to_source(physical)

        t_next = t + np.timedelta64(self.hours_per_iter, "h")

        # A shrinking point count means the reader handed back spoofed data, which happens
        # silently once the targets run past the dataset's end date (observed 40320 -> 2).
        if next_data.shape != self.state.data.shape:
            raise ValueError(
                f"State shape changed {self.state.data.shape} -> {next_data.shape} at "
                f"{t_next}. Almost certainly the forecast window ran past the end of "
                f"the dataset and the reader returned spoofed data. The last usable "
                f"start time is (test_config.end_date - {self.hours_per_iter}h)."
            )
        self._check_finite(next_data, t_next)

        self.state = RolloutState(
            timestamp=t_next, data=next_data, step=self.state.step + 1
        )
        return self.state

    # ------------------------------------------------------------ orchestration

    def initialize(self, start: np.datetime64 | str) -> RolloutState:
        """Seed the rollout from the dataset's own state at ``start``."""
        if isinstance(start, str):
            start = np.datetime64(start)
        idx = self._index_for(start)
        rdata = self.reader.get_source(idx)
        data = self.reader.denormalize_source_channels(
            self.reader.normalize_source_channels(rdata.data)
        )
        self.state = RolloutState(timestamp=start, data=np.asarray(data), step=0)
        logger.info("Initialized at %s, state shape %s", start, self.state.data.shape)
        return self.state

    def run(
        self,
        start: np.datetime64 | str,
        end: np.datetime64 | str,
        output_path: str | Path,
        checkpoint_every: int = 200,
        resume: bool = False,
    ) -> None:
        """
        Execute the rollout, streaming to zarr.

        Output is written incrementally -- a 44-year 6-hourly rollout of ~100
        channels over 65k points does not fit in memory.
        """
        start = np.datetime64(start) if isinstance(start, str) else start
        end = np.datetime64(end) if isinstance(end, str) else end
        output_path = Path(output_path)

        # Each iteration starting at t needs targets out to t + hours_per_iter, so
        # the last usable start is t_end - hours_per_iter. Going beyond makes the
        # reader return spoofed data instead of raising, so clamp up front rather
        # than discovering it mid-rollout.
        tw = self.sampler.time_window_handler
        last_usable = np.datetime64(tw.t_end) - np.timedelta64(self.hours_per_iter, "h")
        if end > last_usable:
            logger.warning(
                "Requested end %s exceeds the last usable start %s "
                "(dataset ends %s, %dh needed per iteration); clamping.",
                end, last_usable, tw.t_end, self.hours_per_iter,
            )
            end = last_usable
        if end <= start:
            raise ValueError(
                f"No usable rollout window: start={start}, clamped end={end}. The "
                f"dataset covers {tw.t_start} to {tw.t_end} and each iteration needs "
                f"{self.hours_per_iter}h of lead. Widen test_config.end_date."
            )

        # The rollout is Markovian in the carried physical state, so the last written
        # prediction is a valid restart: seed from it and append, continuing the same
        # integration rather than re-seeding from truth.
        resumed = False
        if resume and Path(output_path).exists():
            store = zarr.open(str(output_path), mode="a")
            if "predictions" in store and min(
                store["predictions"].shape[0], store["dates"].shape[0]
            ) > 0:
                preds = store["predictions"]
                dates = store["dates"]
                # The step loop appends predictions before dates, so a kill between the two
                # leaves predictions one row longer and would shift every later timestamp.
                # Truncate both: append() writes at the end, so an orphan row would persist.
                n_done = int(min(preds.shape[0], dates.shape[0]))
                if preds.shape[0] != dates.shape[0]:
                    logger.warning(
                        "torn write in %s: predictions=%d dates=%d; truncating to %d",
                        output_path, preds.shape[0], dates.shape[0], n_done,
                    )
                    preds.resize((n_done,) + tuple(preds.shape[1:]))
                    dates.resize((n_done,))
                last_data = np.asarray(preds[-1])
                if not np.isfinite(last_data).all() or not last_data.any():
                    raise ValueError(
                        f"refusing to resume {output_path}: the restart state at step "
                        f"{n_done} is unusable (finite="
                        f"{bool(np.isfinite(last_data).all())}, "
                        f"nonzero={bool(last_data.any())}). The last write was probably "
                        "interrupted mid-chunk; truncate one more step and retry."
                    )
                last_time = np.datetime64(dates[-1].astype("datetime64[s]"))
                self.state = RolloutState(
                    timestamp=last_time, data=last_data, step=n_done
                )
                store.attrs["end"] = str(end)
                store.attrs["completed"] = False
                logger.info(
                    "RESUMING rollout from step %d at %s (appending to %s)",
                    n_done, last_time, output_path,
                )
                resumed = True
            else:
                logger.warning("resume requested but no steps in %s; starting fresh", output_path)
        if not resumed:
            self.initialize(start)
            n_pts, n_ch = self.state.data.shape
            store = zarr.open(str(output_path), mode="w")
            preds = store.create_dataset(
                "predictions",
                shape=(0, n_pts, n_ch),
                chunks=(1, n_pts, n_ch),
                dtype="f4",
            )
            dates = store.create_dataset(
                "dates", shape=(0,), chunks=(1024,), dtype="M8[s]"
            )
            store.attrs["source_channels"] = self.channel_map.source_channels
            store.attrs["hours_per_iteration"] = self.hours_per_iter
            store.attrs["num_forecast_steps"] = self.num_steps
            store.attrs["start"] = str(start)
            store.attrs["end"] = str(end)
        n_expected = int((end - start) / np.timedelta64(self.hours_per_iter, "h"))

        logger.info(
            "Rollout %s -> %s: %d iterations of %dh", start, end, n_expected, self.hours_per_iter
        )

        try:
            while self.state.timestamp < end:
                s = self.step()
                preds.append(s.data[None].astype("f4"))
                dates.append(np.array([s.timestamp], dtype="M8[s]"))

                if s.step % checkpoint_every == 0:
                    store.attrs["last_completed_step"] = s.step
                    store.attrs["last_completed_time"] = str(s.timestamp)
                    logger.info("step %d/%d  t=%s", s.step, n_expected, s.timestamp)
        except FloatingPointError as e:
            # Record where it died rather than losing the diagnosis with the process.
            store.attrs["diverged"] = True
            store.attrs["diverged_at_step"] = self.state.step
            store.attrs["diverged_at_time"] = str(self.state.timestamp)
            store.attrs["diverged_reason"] = str(e)
            logger.error("Rollout diverged and was stopped: %s", e)
            raise

        store.attrs["completed"] = True
        logger.info("Rollout complete: %d steps -> %s", self.state.step, output_path)
