#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import queue
import re
import threading
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any


_SENTENCE_SPLIT_RE = re.compile(
    r"[^\u3002\uff1f\uff01!?]+[\u3002\uff1f\uff01!?]?|[\u3002\uff1f\uff01!?]"
)


def split_text_for_streaming(text: str) -> list[str]:
    chunks: list[str] = []
    for match in _SENTENCE_SPLIT_RE.finditer(text):
        chunk = match.group(0).strip()
        if chunk:
            chunks.append(chunk)
    return chunks


def request_tts_audio(
    server_url: str,
    payload: dict[str, Any],
    *,
    timeout: float,
) -> bytes:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        server_url,
        data=body,
        headers={"Content-Type": "application/json; charset=utf-8"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return response.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Server connection failed: {exc}") from exc


def play_wav_bytes(wav_bytes: bytes) -> None:
    try:
        import winsound
    except ImportError as exc:
        raise RuntimeError("This client currently supports Windows winsound playback only.") from exc

    winsound.PlaySound(wav_bytes, winsound.SND_MEMORY)


@dataclass
class ChunkResult:
    index: int
    text: str
    wav_bytes: bytes | None = None
    error: Exception | None = None


def producer_loop(
    chunks: list[str],
    server_url: str,
    base_payload: dict[str, Any],
    timeout: float,
    out_queue: queue.Queue[ChunkResult],
) -> None:
    for index, chunk in enumerate(chunks, start=1):
        payload = dict(base_payload)
        payload["text"] = chunk
        if payload.get("seed") is not None:
            payload["seed"] = int(payload["seed"]) + (index - 1)
        try:
            wav_bytes = request_tts_audio(server_url, payload, timeout=timeout)
            out_queue.put(ChunkResult(index=index, text=chunk, wav_bytes=wav_bytes))
        except Exception as exc:
            out_queue.put(ChunkResult(index=index, text=chunk, error=exc))
            return
    out_queue.put(ChunkResult(index=-1, text="__end__"))


def build_base_payload(args: argparse.Namespace) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "caption": args.caption,
        "no_ref": bool(args.no_ref),
        "num_steps": args.num_steps,
        "cfg_scale_text": args.cfg_scale_text,
        "cfg_scale_caption": args.cfg_scale_caption,
        "cfg_scale_speaker": args.cfg_scale_speaker,
        "cfg_guidance_mode": args.cfg_guidance_mode,
        "seconds": args.seconds,
        "duration_scale": args.duration_scale,
        "seed": args.seed,
    }

    if args.ref_wav is not None:
        payload["ref_wav"] = args.ref_wav
    if args.ref_latent is not None:
        payload["ref_latent"] = args.ref_latent
    if args.ref_embed is not None:
        payload["ref_embed"] = args.ref_embed
    if args.max_ref_seconds is not None:
        payload["max_ref_seconds"] = args.max_ref_seconds
    if args.decode_mode is not None:
        payload["decode_mode"] = args.decode_mode
    if args.speaker_kv_scale is not None:
        payload["speaker_kv_scale"] = args.speaker_kv_scale
    if args.speaker_kv_min_t is not None:
        payload["speaker_kv_min_t"] = args.speaker_kv_min_t
    if args.speaker_kv_max_layers is not None:
        payload["speaker_kv_max_layers"] = args.speaker_kv_max_layers
    if args.lora_adapter is not None:
        payload["lora_adapter"] = args.lora_adapter

    return payload


def load_text(args: argparse.Namespace) -> str:
    if args.text is not None:
        return args.text
    with open(args.text_file, "r", encoding="utf-8") as f:
        return f.read()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Irodori-TTS infer_server.py client. Split text by 。？！ and play sequentially."
    )
    text_group = parser.add_mutually_exclusive_group(required=True)
    text_group.add_argument("--text", default=None, help="Text to speak.")
    text_group.add_argument("--text-file", default=None, help="UTF-8 text file to speak.")
    parser.add_argument("--server-url", default="http://127.0.0.1:8000/infer")
    parser.add_argument("--caption", default=None)
    parser.add_argument("--ref-wav", default=None)
    parser.add_argument("--ref-latent", default=None)
    parser.add_argument("--ref-embed", default=None)
    parser.add_argument("--no-ref", action="store_true")
    parser.add_argument("--num-steps", type=int, default=40)
    parser.add_argument("--cfg-scale-text", type=float, default=3.0)
    parser.add_argument("--cfg-scale-caption", type=float, default=3.0)
    parser.add_argument("--cfg-scale-speaker", type=float, default=5.0)
    parser.add_argument(
        "--cfg-guidance-mode",
        choices=["independent", "joint", "alternating"],
        default="independent",
    )
    parser.add_argument("--seconds", type=float, default=None)
    parser.add_argument("--duration-scale", type=float, default=1.0)
    parser.add_argument("--max-ref-seconds", type=float, default=None)
    parser.add_argument("--decode-mode", choices=["sequential", "batch"], default=None)
    parser.add_argument("--speaker-kv-scale", type=float, default=None)
    parser.add_argument("--speaker-kv-min-t", type=float, default=None)
    parser.add_argument("--speaker-kv-max-layers", type=int, default=None)
    parser.add_argument("--lora-adapter", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--show-chunks", action="store_true")
    args = parser.parse_args()

    text = load_text(args).strip()
    if not text:
        raise SystemExit("Text is empty.")

    chunks = split_text_for_streaming(text)
    if not chunks:
        raise SystemExit("No readable chunks were produced from the text.")

    if args.show_chunks:
        for i, chunk in enumerate(chunks, start=1):
            print(f"[chunk {i}/{len(chunks)}] {chunk}")

    base_payload = build_base_payload(args)
    result_queue: queue.Queue[ChunkResult] = queue.Queue(maxsize=2)
    worker = threading.Thread(
        target=producer_loop,
        args=(chunks, args.server_url, base_payload, args.timeout, result_queue),
        daemon=True,
    )
    worker.start()

    total = len(chunks)
    while True:
        result = result_queue.get()
        if result.index == -1:
            break
        if result.error is not None:
            raise SystemExit(f"[error chunk {result.index}/{total}] {result.error}")

        print(f"[play {result.index}/{total}] {result.text}")
        play_wav_bytes(result.wav_bytes or b"")


if __name__ == "__main__":
    main()
