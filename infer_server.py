#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import json
import math
import sys
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
import tempfile

import torch
import torchaudio

from irodori_tts.inference_runtime import (
    RuntimeKey,
    SamplingRequest,
    download_hf_checkpoint,
    get_cached_runtime,
    resolve_cfg_scales,
)


def _parse_optional_float(value: str) -> float | None:
    raw = str(value).strip().lower()
    if raw in {"none", "null", "off", "disable", "disabled"}:
        return None
    try:
        out = float(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "Expected float or one of [none, null, off, disable, disabled]."
        ) from exc
    if not math.isfinite(out):
        raise argparse.ArgumentTypeError(f"Expected finite float for value={value!r}.")
    return out


def _load_audio(path: str | Path) -> tuple[torch.Tensor, int]:
    try:
        return torchaudio.load(str(path))
    except RuntimeError:
        import soundfile as sf

        data, sr = sf.read(str(path), dtype="float32")
        wav = torch.from_numpy(data)
        if wav.ndim == 1:
            wav = wav.unsqueeze(0)
        else:
            wav = wav.T
        return wav, sr


def _encode_wav_bytes(audio: torch.Tensor, sample_rate: int) -> bytes:
    audio_cpu = audio.detach().to(device="cpu", dtype=torch.float32)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        try:
            torchaudio.save(str(tmp_path), audio_cpu, sample_rate)
        except RuntimeError:
            import soundfile as sf

            audio_np = (
                audio_cpu.squeeze(0).numpy() if audio_cpu.shape[0] == 1 else audio_cpu.T.numpy()
            )
            sf.write(str(tmp_path), audio_np, sample_rate, format="WAV")
        return tmp_path.read_bytes()
    finally:
        tmp_path.unlink(missing_ok=True)


def _resolve_checkpoint_path(args: argparse.Namespace) -> str:
    if args.checkpoint is not None:
        checkpoint_path = Path(str(args.checkpoint)).expanduser()
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        print(f"[checkpoint] using local file: {checkpoint_path}", flush=True)
        return str(checkpoint_path)

    repo_id = str(args.hf_checkpoint).strip()
    if repo_id == "":
        raise ValueError("hf_checkpoint must be non-empty.")

    checkpoint_path = download_hf_checkpoint(repo_id)
    print(
        f"[checkpoint] downloaded model.safetensors from hf://{repo_id} -> {checkpoint_path}",
        flush=True,
    )
    return str(checkpoint_path)


def _resolve_existing_file(path: str | None) -> str | None:
    if path is None:
        return None
    resolved = Path(str(path)).expanduser()
    if not resolved.is_file():
        raise FileNotFoundError(f"File not found: {resolved}")
    return str(resolved.resolve())


def _maybe_create_default_ref_latent(
    runtime,
    *,
    default_ref_wav: str | None,
    default_ref_latent: str | None,
    default_ref_auto_create: bool,
    ref_normalize_db: float | None,
    ref_ensure_max: bool,
) -> str | None:
    if default_ref_latent is not None:
        return _resolve_existing_file(default_ref_latent)
    if not default_ref_auto_create or default_ref_wav is None:
        return None

    wav_path = Path(default_ref_wav)
    out_path = wav_path.with_suffix(wav_path.suffix + ".latent.pt")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    wav, sr = _load_audio(wav_path)
    latent = runtime.codec.encode_waveform(
        wav.unsqueeze(0),
        sample_rate=int(sr),
        normalize_db=ref_normalize_db,
        ensure_max=ref_ensure_max,
    )[0].cpu()
    torch.save(latent, out_path)
    print(f"[server] created default ref latent: {out_path}", flush=True)
    return str(out_path.resolve())


class InferServerState:
    def __init__(self, args: argparse.Namespace) -> None:
        checkpoint_path = _resolve_checkpoint_path(args)
        self.runtime_key = RuntimeKey(
            checkpoint=checkpoint_path,
            model_device=str(args.model_device),
            codec_repo=str(args.codec_repo),
            model_precision=str(args.model_precision),
            codec_device=str(args.codec_device),
            codec_precision=str(args.codec_precision),
            codec_deterministic_encode=bool(args.codec_deterministic_encode),
            codec_deterministic_decode=bool(args.codec_deterministic_decode),
            compile_model=bool(args.compile_model),
            compile_dynamic=bool(args.compile_dynamic),
        )
        self.runtime, reloaded = get_cached_runtime(self.runtime_key)
        print(f"[server] runtime: {'reloaded' if reloaded else 'reused'}", flush=True)

        self.default_ref_wav = _resolve_existing_file(args.default_ref_wav)
        self.default_ref_latent = _maybe_create_default_ref_latent(
            self.runtime,
            default_ref_wav=self.default_ref_wav,
            default_ref_latent=args.default_ref_latent,
            default_ref_auto_create=bool(args.default_ref_auto_create),
            ref_normalize_db=args.ref_normalize_db,
            ref_ensure_max=bool(args.ref_ensure_max),
        )

        self.request_defaults = {
            "caption": None,
            "ref_normalize_db": args.ref_normalize_db,
            "ref_ensure_max": bool(args.ref_ensure_max),
            "num_candidates": int(args.num_candidates),
            "decode_mode": str(args.decode_mode),
            "seconds": None if args.seconds is None else float(args.seconds),
            "duration_scale": float(args.duration_scale),
            "min_seconds": float(args.min_seconds),
            "max_seconds": float(args.max_seconds),
            "max_ref_seconds": None if args.max_ref_seconds is None else float(args.max_ref_seconds),
            "max_text_len": None if args.max_text_len is None else int(args.max_text_len),
            "max_caption_len": None
            if args.max_caption_len is None
            else int(args.max_caption_len),
            "num_steps": int(args.num_steps),
            "cfg_scale_text": float(args.cfg_scale_text),
            "cfg_scale_caption": float(args.cfg_scale_caption),
            "cfg_scale_speaker": float(args.cfg_scale_speaker),
            "cfg_guidance_mode": str(args.cfg_guidance_mode),
            "cfg_scale": None if args.cfg_scale is None else float(args.cfg_scale),
            "cfg_min_t": float(args.cfg_min_t),
            "cfg_max_t": float(args.cfg_max_t),
            "truncation_factor": None
            if args.truncation_factor is None
            else float(args.truncation_factor),
            "rescale_k": None if args.rescale_k is None else float(args.rescale_k),
            "rescale_sigma": None if args.rescale_sigma is None else float(args.rescale_sigma),
            "context_kv_cache": bool(args.context_kv_cache),
            "speaker_kv_scale": None
            if args.speaker_kv_scale is None
            else float(args.speaker_kv_scale),
            "speaker_kv_min_t": None
            if args.speaker_kv_scale is None
            else float(args.speaker_kv_min_t),
            "speaker_kv_max_layers": None
            if args.speaker_kv_max_layers is None
            else int(args.speaker_kv_max_layers),
            "speaker_uncond_mode": str(args.speaker_uncond_mode),
            "seed": None if args.seed is None else int(args.seed),
            "t_schedule_mode": str(args.t_schedule_mode),
            "sway_coeff": float(args.sway_coeff),
            "trim_tail": bool(args.trim_tail),
            "tail_window_size": int(args.tail_window_size),
            "tail_std_threshold": float(args.tail_std_threshold),
            "tail_mean_threshold": float(args.tail_mean_threshold),
            "lora_adapter": None if args.lora_adapter is None else str(args.lora_adapter),
        }

    def build_sampling_request(self, payload: dict[str, Any]) -> SamplingRequest:
        if not isinstance(payload, dict):
            raise ValueError("Request body must be a JSON object.")
        if "text" not in payload:
            raise ValueError("Missing required field: text")

        req_data = dict(self.request_defaults)
        req_data["text"] = str(payload["text"])
        req_data["caption"] = payload.get("caption", req_data["caption"])

        for key in (
            "ref_normalize_db",
            "ref_ensure_max",
            "num_candidates",
            "decode_mode",
            "seconds",
            "duration_scale",
            "min_seconds",
            "max_seconds",
            "max_ref_seconds",
            "max_text_len",
            "max_caption_len",
            "num_steps",
            "cfg_scale_text",
            "cfg_scale_caption",
            "cfg_scale_speaker",
            "cfg_guidance_mode",
            "cfg_scale",
            "cfg_min_t",
            "cfg_max_t",
            "truncation_factor",
            "rescale_k",
            "rescale_sigma",
            "context_kv_cache",
            "speaker_kv_scale",
            "speaker_kv_min_t",
            "speaker_kv_max_layers",
            "speaker_uncond_mode",
            "seed",
            "t_schedule_mode",
            "sway_coeff",
            "trim_tail",
            "tail_window_size",
            "tail_std_threshold",
            "tail_mean_threshold",
            "lora_adapter",
        ):
            if key in payload:
                req_data[key] = payload[key]

        no_ref = bool(payload.get("no_ref", False))
        ref_wav = payload.get("ref_wav")
        ref_wavs = payload.get("ref_wavs")
        ref_latent = payload.get("ref_latent")
        ref_latents = payload.get("ref_latents")
        ref_embed = payload.get("ref_embed")

        if ref_wav is None and ref_wavs is None and ref_latent is None and ref_latents is None and ref_embed is None and not no_ref:
            if self.default_ref_latent is not None:
                ref_latent = self.default_ref_latent
            elif self.default_ref_wav is not None:
                ref_wav = self.default_ref_wav

        use_speaker_for_request = bool(
            self.runtime.model_cfg.use_speaker_condition_resolved and not no_ref
        )
        cfg_scale_text, cfg_scale_caption, cfg_scale_speaker, _ = resolve_cfg_scales(
            cfg_guidance_mode=str(req_data["cfg_guidance_mode"]),
            cfg_scale_text=float(req_data["cfg_scale_text"]),
            cfg_scale_caption=float(req_data["cfg_scale_caption"]),
            cfg_scale_speaker=float(req_data["cfg_scale_speaker"]),
            cfg_scale=(
                None if req_data["cfg_scale"] is None else float(req_data["cfg_scale"])
            ),
            use_caption_condition=bool(
                self.runtime.model_cfg.use_caption_condition
                and req_data["caption"] is not None
                and str(req_data["caption"]).strip() != ""
            ),
            use_speaker_condition=use_speaker_for_request,
        )

        if self.runtime.model_cfg.use_speaker_condition_resolved and not (
            no_ref
            or ref_wav is not None
            or ref_wavs is not None
            or ref_latent is not None
            or ref_latents is not None
            or ref_embed is not None
        ):
            raise ValueError(
                "speaker-conditioned checkpoint requires ref_wav/ref_wavs/ref_latent/ref_latents/ref_embed, "
                "or no_ref=true."
            )

        return SamplingRequest(
            text=str(req_data["text"]),
            caption=None if req_data["caption"] is None else str(req_data["caption"]),
            ref_wav=None if ref_wav is None else str(ref_wav),
            ref_wavs=None if ref_wavs is None else [str(item) for item in ref_wavs],
            ref_latent=None if ref_latent is None else str(ref_latent),
            ref_latents=None if ref_latents is None else [str(item) for item in ref_latents],
            ref_embed=None if ref_embed is None else str(ref_embed),
            no_ref=no_ref,
            ref_normalize_db=(
                None if req_data["ref_normalize_db"] is None else float(req_data["ref_normalize_db"])
            ),
            ref_ensure_max=bool(req_data["ref_ensure_max"]),
            num_candidates=int(req_data["num_candidates"]),
            decode_mode=str(req_data["decode_mode"]),
            seconds=None if req_data["seconds"] is None else float(req_data["seconds"]),
            duration_scale=float(req_data["duration_scale"]),
            min_seconds=float(req_data["min_seconds"]),
            max_seconds=float(req_data["max_seconds"]),
            max_ref_seconds=(
                None if req_data["max_ref_seconds"] is None else float(req_data["max_ref_seconds"])
            ),
            max_text_len=None
            if req_data["max_text_len"] is None
            else int(req_data["max_text_len"]),
            max_caption_len=None
            if req_data["max_caption_len"] is None
            else int(req_data["max_caption_len"]),
            num_steps=int(req_data["num_steps"]),
            cfg_scale_text=cfg_scale_text,
            cfg_scale_caption=cfg_scale_caption,
            cfg_scale_speaker=cfg_scale_speaker,
            cfg_guidance_mode=str(req_data["cfg_guidance_mode"]),
            cfg_scale=None,
            cfg_min_t=float(req_data["cfg_min_t"]),
            cfg_max_t=float(req_data["cfg_max_t"]),
            truncation_factor=None
            if req_data["truncation_factor"] is None
            else float(req_data["truncation_factor"]),
            rescale_k=None if req_data["rescale_k"] is None else float(req_data["rescale_k"]),
            rescale_sigma=None
            if req_data["rescale_sigma"] is None
            else float(req_data["rescale_sigma"]),
            context_kv_cache=bool(req_data["context_kv_cache"]),
            speaker_kv_scale=None
            if req_data["speaker_kv_scale"] is None
            else float(req_data["speaker_kv_scale"]),
            speaker_kv_min_t=None
            if req_data["speaker_kv_min_t"] is None
            else float(req_data["speaker_kv_min_t"]),
            speaker_kv_max_layers=None
            if req_data["speaker_kv_max_layers"] is None
            else int(req_data["speaker_kv_max_layers"]),
            speaker_uncond_mode=str(req_data["speaker_uncond_mode"]),
            seed=None if req_data["seed"] is None else int(req_data["seed"]),
            t_schedule_mode=str(req_data["t_schedule_mode"]),
            sway_coeff=float(req_data["sway_coeff"]),
            trim_tail=bool(req_data["trim_tail"]),
            tail_window_size=int(req_data["tail_window_size"]),
            tail_std_threshold=float(req_data["tail_std_threshold"]),
            tail_mean_threshold=float(req_data["tail_mean_threshold"]),
            lora_adapter=None if req_data["lora_adapter"] is None else str(req_data["lora_adapter"]),
        )

    def describe(self) -> dict[str, Any]:
        return {
            "checkpoint": self.runtime_key.checkpoint,
            "model_device": self.runtime_key.model_device,
            "model_precision": self.runtime_key.model_precision,
            "codec_device": self.runtime_key.codec_device,
            "codec_precision": self.runtime_key.codec_precision,
            "default_ref_wav": self.default_ref_wav,
            "default_ref_latent": self.default_ref_latent,
            "use_speaker_condition": bool(self.runtime.model_cfg.use_speaker_condition_resolved),
            "use_caption_condition": bool(self.runtime.model_cfg.use_caption_condition),
            "default_text_max_len": int(self.runtime.default_text_max_len),
            "default_caption_max_len": int(self.runtime.default_caption_max_len),
            "default_max_ref_seconds": float(self.runtime.default_max_ref_seconds),
        }


class InferRequestHandler(BaseHTTPRequestHandler):
    server_version = "IrodoriInferServer/0.1"

    @property
    def state(self) -> InferServerState:
        return self.server.state  # type: ignore[attr-defined]

    def log_message(self, format: str, *args: Any) -> None:
        sys.stdout.write("[http] " + (format % args) + "\n")

    def _send_json(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path == "/" or self.path == "/health":
            self._send_json(
                HTTPStatus.OK,
                {
                    "ok": True,
                    "service": "irodori-tts",
                    "endpoints": ["GET /health", "GET /info", "POST /infer", "POST /infer/json"],
                },
            )
            return
        if self.path == "/info":
            self._send_json(HTTPStatus.OK, {"ok": True, "runtime": self.state.describe()})
            return
        self._send_json(HTTPStatus.NOT_FOUND, {"ok": False, "error": "Not found"})

    def do_POST(self) -> None:
        if self.path not in {"/infer", "/infer/json"}:
            self._send_json(HTTPStatus.NOT_FOUND, {"ok": False, "error": "Not found"})
            return

        try:
            content_length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self._send_json(HTTPStatus.BAD_REQUEST, {"ok": False, "error": "Invalid Content-Length"})
            return

        try:
            raw_body = self.rfile.read(content_length)
            payload = json.loads(raw_body.decode("utf-8"))
            req = self.state.build_sampling_request(payload)
            result = self.state.runtime.synthesize(req)
            wav_bytes = _encode_wav_bytes(result.audio, result.sample_rate)
        except Exception as exc:
            self._send_json(
                HTTPStatus.BAD_REQUEST,
                {"ok": False, "error": type(exc).__name__, "message": str(exc)},
            )
            return

        if self.path == "/infer/json":
            self._send_json(
                HTTPStatus.OK,
                {
                    "ok": True,
                    "sample_rate": result.sample_rate,
                    "used_seed": result.used_seed,
                    "messages": result.messages,
                    "stage_timings": result.stage_timings,
                    "total_to_decode": result.total_to_decode,
                    "audio_base64": base64.b64encode(wav_bytes).decode("ascii"),
                },
            )
            return

        filename = f"irodori_{result.used_seed}.wav"
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "audio/wav")
        self.send_header("Content-Length", str(len(wav_bytes)))
        self.send_header("Content-Disposition", f'inline; filename="{filename}"')
        self.send_header("X-Irodori-Seed", str(result.used_seed))
        self.end_headers()
        self.wfile.write(wav_bytes)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="HTTP inference server for Irodori-TTS.")
    checkpoint_group = parser.add_mutually_exclusive_group(required=True)
    checkpoint_group.add_argument(
        "--checkpoint",
        default=None,
        help="Local model checkpoint path (.pt or .safetensors).",
    )
    checkpoint_group.add_argument(
        "--hf-checkpoint",
        default=None,
        help=(
            "Hugging Face model repo id or repo/subfolder containing model.safetensors "
            "(e.g. your-org/your-model or your-org/your-model/int8-weight-only)."
        ),
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--lora-adapter", default=None)
    parser.add_argument("--model-device", default="cpu")
    parser.add_argument("--model-precision", choices=["fp32", "bf16"], default="fp32")
    parser.add_argument("--codec-device", default="cpu")
    parser.add_argument("--codec-precision", choices=["fp32", "bf16"], default="fp32")
    parser.add_argument(
        "--codec-deterministic-encode",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--codec-deterministic-decode",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--codec-repo", default="Aratako/Semantic-DACVAE-Japanese-32dim")
    parser.add_argument("--default-ref-wav", default=None)
    parser.add_argument("--default-ref-latent", default=None)
    parser.add_argument(
        "--default-ref-auto-create",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="If default ref latent is omitted, encode --default-ref-wav once at startup.",
    )
    parser.add_argument("--max-ref-seconds", type=float, default=None)
    parser.add_argument("--ref-normalize-db", type=_parse_optional_float, default=-16.0)
    parser.add_argument(
        "--ref-ensure-max",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--max-text-len", type=int, default=None)
    parser.add_argument("--max-caption-len", type=int, default=None)
    parser.add_argument("--num-steps", type=int, default=40)
    parser.add_argument(
        "--t-schedule-mode",
        choices=["linear", "sway"],
        default="linear",
    )
    parser.add_argument("--sway-coeff", type=float, default=-1.0)
    parser.add_argument("--seconds", type=float, default=None)
    parser.add_argument("--duration-scale", type=float, default=1.0)
    parser.add_argument("--min-seconds", type=float, default=0.5)
    parser.add_argument("--max-seconds", type=float, default=30.0)
    parser.add_argument("--num-candidates", type=int, default=1)
    parser.add_argument("--decode-mode", choices=["sequential", "batch"], default="sequential")
    parser.add_argument(
        "--compile-model",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--compile-dynamic",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--cfg-scale-text", type=float, default=3.0)
    parser.add_argument("--cfg-scale-caption", type=float, default=3.0)
    parser.add_argument("--cfg-scale-speaker", type=float, default=5.0)
    parser.add_argument(
        "--cfg-guidance-mode",
        choices=["independent", "joint", "alternating"],
        default="independent",
    )
    parser.add_argument("--cfg-scale", type=float, default=None)
    parser.add_argument("--cfg-min-t", type=float, default=0.5)
    parser.add_argument("--cfg-max-t", type=float, default=1.0)
    parser.add_argument("--truncation-factor", type=float, default=None)
    parser.add_argument("--rescale-k", type=float, default=None)
    parser.add_argument("--rescale-sigma", type=float, default=None)
    parser.add_argument(
        "--context-kv-cache",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--speaker-kv-scale", type=float, default=None)
    parser.add_argument("--speaker-kv-min-t", type=float, default=0.9)
    parser.add_argument("--speaker-kv-max-layers", type=int, default=None)
    parser.add_argument(
        "--speaker-uncond-mode",
        choices=["mask", "noise"],
        default="mask",
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--trim-tail",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--tail-window-size", type=int, default=20)
    parser.add_argument("--tail-std-threshold", type=float, default=0.05)
    parser.add_argument("--tail-mean-threshold", type=float, default=0.1)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    state = InferServerState(args)
    httpd = ThreadingHTTPServer((str(args.host), int(args.port)), InferRequestHandler)
    httpd.state = state  # type: ignore[attr-defined]
    print(f"[server] listening on http://{args.host}:{args.port}", flush=True)
    httpd.serve_forever()


if __name__ == "__main__":
    main()
