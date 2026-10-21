"""Transcribe complete audio with the pinned TalTech streaming Zipformer.

The archive pipeline invokes this adapter with an audio path, --output-dir and
--formats json. Audio is streamed through ffmpeg and sherpa-onnx so recordings
do not have to be decoded into large temporary files.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path

import numpy as np
import sherpa_onnx
from huggingface_hub import snapshot_download

REPOSITORY = "TalTechNLP/streaming-zipformer-large.et-en"
REVISION = "2812262bd4fd5d4327dc09ca0e6c19e8a32a03d6"
SAMPLE_RATE = 16000
FRAME_SAMPLES = SAMPLE_RATE // 2


def model_path() -> Path:
    return Path(
        snapshot_download(
            repo_id=REPOSITORY,
            revision=REVISION,
            allow_patterns=(
                "encoder.int8.onnx",
                "decoder.int8.onnx",
                "joiner.int8.onnx",
                "tokens.txt",
            ),
        )
    )


def channel_count(path: Path) -> int:
    output = subprocess.check_output(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "stream=channels",
            "-of",
            "json",
            str(path),
        ],
        text=True,
    )
    return int(json.loads(output)["streams"][0]["channels"])


def _start_ffmpeg(path: Path, *, stereo: bool, channel: str = "mix") -> subprocess.Popen[bytes]:
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(path)]
    if channel != "mix":
        index = 0 if channel == "left" else 1
        command += ["-af", f"pan=mono|c0=c{index}"]
    command += ["-ar", str(SAMPLE_RATE), "-ac", "2" if stereo else "1", "-f", "f32le", "pipe:1"]
    return subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def _finish_ffmpeg(process: subprocess.Popen[bytes]) -> None:
    if process.stdout is not None:
        process.stdout.close()
    stderr = process.stderr.read().decode(errors="replace") if process.stderr else ""
    if process.wait() != 0:
        raise RuntimeError(f"ffmpeg failed: {stderr[-1000:]}")


def select_channel(path: Path) -> str:
    """Keep the mix unless phase cancellation makes it 12 dB weaker."""
    if channel_count(path) < 2:
        return "mix"
    process = _start_ffmpeg(path, stereo=True)
    sums = np.zeros(3, dtype=np.float64)
    assert process.stdout is not None
    while chunk := process.stdout.read(262144):
        samples = np.frombuffer(chunk, dtype="<f4")
        if len(samples) % 2:
            raise ValueError("ffmpeg returned an incomplete stereo frame")
        channels = samples.reshape(-1, 2).astype(np.float64)
        sums[0] += np.square(channels[:, 0]).sum()
        sums[1] += np.square(channels[:, 1]).sum()
        sums[2] += np.square(channels.mean(axis=1)).sum()
    _finish_ffmpeg(process)
    stronger = 0 if sums[0] >= sums[1] else 1
    if sums[stronger] == 0 or math.sqrt(sums[2] / sums[stronger]) >= 10 ** (-12 / 20):
        return "mix"
    return "left" if stronger == 0 else "right"


def _recognizer(model: Path) -> sherpa_onnx.OnlineRecognizer:
    return sherpa_onnx.OnlineRecognizer.from_transducer(
        tokens=str(model / "tokens.txt"),
        encoder=str(model / "encoder.int8.onnx"),
        decoder=str(model / "decoder.int8.onnx"),
        joiner=str(model / "joiner.int8.onnx"),
        num_threads=4,
        provider="cpu",
        sample_rate=SAMPLE_RATE,
        feature_dim=80,
        decoding_method="modified_beam_search",
        max_active_paths=4,
        enable_endpoint_detection=True,
        rule3_min_utterance_length=20,
    )


def _cues(tokens: list[str], times: list[float], offset: float) -> list[dict]:
    cues: list[dict] = []
    group: list[str] = []
    start = 0.0
    for index, (token, time) in enumerate(zip(tokens, times, strict=True)):
        if not group:
            start = offset + time
        group.append(token)
        content = "".join(group).strip()
        next_time = offset + (times[index + 1] if index + 1 < len(times) else time + 0.3)
        at_end = index + 1 == len(tokens)
        next_word = not at_end and tokens[index + 1].startswith(" ")
        if at_end or (next_word and (len(content) >= 80 or next_time - start >= 8)):
            if content:
                cues.append(
                    {
                        "start": round(max(0.0, start), 2),
                        "end": round(max(start + 0.04, next_time), 2),
                        "text": content,
                    }
                )
            group = []
    return cues


def transcribe(path: Path, model: Path) -> dict:
    channel = select_channel(path)
    process = _start_ffmpeg(path, stereo=False, channel=channel)
    assert process.stdout is not None
    recognizer = _recognizer(model)
    stream = recognizer.create_stream()
    cues: list[dict] = []
    samples_fed = 0
    segment_start = 0
    while chunk := process.stdout.read(FRAME_SAMPLES * 4):
        samples = np.frombuffer(chunk, dtype="<f4")
        stream.accept_waveform(SAMPLE_RATE, samples)
        samples_fed += len(samples)
        while recognizer.is_ready(stream):
            recognizer.decode_stream(stream)
        if recognizer.is_endpoint(stream):
            result = recognizer.get_result_all(stream)
            cues.extend(_cues(result.tokens, result.timestamps, segment_start / SAMPLE_RATE))
            recognizer.reset(stream)
            segment_start = samples_fed
    _finish_ffmpeg(process)
    stream.accept_waveform(SAMPLE_RATE, np.zeros(SAMPLE_RATE, dtype=np.float32))
    stream.input_finished()
    while recognizer.is_ready(stream):
        recognizer.decode_stream(stream)
    result = recognizer.get_result_all(stream)
    cues.extend(_cues(result.tokens, result.timestamps, segment_start / SAMPLE_RATE))
    if not cues:
        raise ValueError("recognizer returned no speech")
    # Endpoint resets can place the next segment a few milliseconds before the
    # previous one. Preserve the recognized text while keeping cue starts ordered.
    previous_start = 0.0
    for cue in cues:
        if cue["start"] < previous_start:
            cue["start"] = previous_start
            cue["end"] = round(max(cue["end"], previous_start + 0.04), 2)
        previous_start = cue["start"]
    return {
        "model": {
            "name": "large",
            "repository": REPOSITORY,
            "revision": REVISION,
            "precision": "int8",
        },
        "runtime": {
            "sherpa_onnx": sherpa_onnx.__version__,
            "decoding_method": "modified_beam_search",
            "threads": 4,
            "audio_channel": channel,
        },
        "transcript": " ".join(cue["text"] for cue in cues),
        "cues": cues,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", nargs="?", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--formats", default="json")
    parser.add_argument("--download-only", action="store_true")
    parser.add_argument("--model", default="large")
    parser.add_argument("--precision", default="int8")
    args = parser.parse_args()
    if args.model != "large" or args.precision != "int8" or args.formats != "json":
        parser.error("only the locked large/int8 JSON configuration is supported")
    model = model_path()
    if args.download_only:
        print(model)
        return
    if args.audio is None or args.output_dir is None:
        parser.error("audio and --output-dir are required")
    payload = transcribe(args.audio.resolve(), model)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / f"{args.audio.stem}.json"
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
