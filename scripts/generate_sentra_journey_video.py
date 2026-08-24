#!/usr/bin/env python3
"""Generate ~60s SENTRA journey MP4 with ambient-tech BGM (no narration)."""
from __future__ import annotations

import math
import shutil
import struct
import subprocess
import tempfile
import wave
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "frontend" / "public" / "sentra_journey.mp4"
W, H = 1280, 720
FPS = 6
DURATION = 60
SR = 44100

SCENES = [
    {
        "start": 0, "end": 7, "id": "intro", "badge": "SENTRA",
        "title": "BioSCADA AI — SENTRA",
        "subtitle": "Streaming Event-driven Network of Threshold Response Agents",
        "bullets": [
            "Biopharma SCADA threshold response platform",
            "Live telemetry → prediction → agent remedy → GxP sign-off",
            "Five phases form one closed-loop control journey",
        ],
    },
    {
        "start": 7, "end": 17, "id": "p0", "badge": "PHASE 0",
        "title": "Edge Telemetry Stream",
        "subtitle": "Kafka / Redpanda ingest for five bioprocess parameters",
        "bullets": [
            "Topics stream Temperature, pH, Pressure, Conductivity, Humidity",
            "Each tick carries asset, value, zone, and event time",
            "Breach candidates are flagged before they reach the agents",
        ],
    },
    {
        "start": 17, "end": 27, "id": "p1", "badge": "PHASE 1",
        "title": "Flink Stream Engine",
        "subtitle": "PyFlink continuous SQL on tumbling event-time windows",
        "bullets": [
            "10-second tumbling windows with watermarks",
            "Aggregates v_avg, v_std, and v_delta features",
            "Emits structured breach events with top drivers",
        ],
    },
    {
        "start": 27, "end": 37, "id": "p2", "badge": "PHASE 2",
        "title": "PdM Predictive Models",
        "subtitle": "Supervised models estimate real-time P(breach)",
        "bullets": [
            "GBM, Random Forest, SVM, and MLP per parameter",
            "Outputs predicted breach probability and ranked drivers",
            "RED flags become the hand-off trigger to agents",
        ],
    },
    {
        "start": 37, "end": 47, "id": "p3", "badge": "PHASE 3",
        "title": "Agentic RAG Remedy",
        "subtitle": "Specialized agents retrieve SOP, CAPA, and OEM knowledge",
        "bullets": [
            "Orchestrator routes RED events to the owning parameter agent",
            "Hybrid vector search over SOP, CAPA, and OEM manuals",
            "LLM synthesizes citable, actionable remediation steps",
        ],
    },
    {
        "start": 47, "end": 55, "id": "p4", "badge": "PHASE 4",
        "title": "GxP Governance & Audit",
        "subtitle": "Policy decision, e-signature, immutable WORM trail",
        "bullets": [
            "OPA evaluates ALLOW versus REQUIRE e-signature",
            "21 CFR Part 11 signer identity and meaning of signature",
            "Hash-chained WORM audit captures every commit or reject",
        ],
    },
    {
        "start": 55, "end": 60, "id": "close", "badge": "JOURNEY COMPLETE",
        "title": "Closed-Loop Threshold Response",
        "subtitle": "From sensor tick to governed plant action",
        "bullets": [
            "Telemetry → Flink → PdM → Agents → Governance",
            "Start Studio to walk the interactive thirteen-step workflow",
            "SENTRA keeps operators ahead of excursions — not behind them",
        ],
    },
]


def font(size: int, bold: bool = False):
    candidates = [
        "C:/Windows/Fonts/segoeuib.ttf" if bold else "C:/Windows/Fonts/segoeui.ttf",
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            try:
                return ImageFont.truetype(path, size)
            except OSError:
                pass
    return ImageFont.load_default()


def scene_at(t: float):
    for s in SCENES:
        if s["start"] <= t < s["end"] or (t >= DURATION - 0.05 and s["end"] == DURATION):
            local = (t - s["start"]) / max(s["end"] - s["start"], 1e-6)
            return s, local
    return SCENES[-1], 1.0


def draw_frame(t: float) -> Image.Image:
    img = Image.new("RGB", (W, H), (6, 18, 22))
    d = ImageDraw.Draw(img)

    for i in range(0, H, 4):
        shade = int(6 + 10 * (1 - abs(i - H / 2) / (H / 2)))
        d.rectangle([0, i, W, i + 4], fill=(shade, shade + 10, shade + 14))

    pulse = 0.45 + 0.15 * math.sin(t * 1.4)
    for r in range(200, 40, -20):
        alpha = int(16 * pulse * (r / 200))
        color = (8 + alpha // 3, 36 + alpha, 44 + alpha)
        d.ellipse([W // 2 - r, 250 - r // 2, W // 2 + r, 250 + r // 2], outline=color)

    s, local = scene_at(t)
    fade = max(0.35, min(1.0, min(1.0, local * 5.0, (1.0 - local) * 6.0 + 0.35)))

    d.rectangle([0, 0, W, 56], fill=(8, 28, 34))
    d.line([0, 56, W, 56], fill=(43, 179, 192), width=2)
    d.text((28, 14), "SENTRA", font=font(24, True), fill=(43, 179, 192))
    d.text((140, 20), "BioSCADA AI  ·  Threshold Response Journey", font=font(16), fill=(170, 200, 200))

    prog = t / DURATION
    d.rectangle([40, H - 34, W - 40, H - 26], fill=(20, 48, 54))
    d.rectangle([40, H - 34, 40 + int((W - 80) * prog), H - 26], fill=(43, 179, 192))
    d.text((40, H - 56), f"{int(t):02d}s / {DURATION}s  ·  ambient journey score", font=font(14), fill=(120, 150, 150))

    phases = ["0 Telemetry", "1 Flink", "2 PdM", "3 Agents", "4 Govern"]
    active_idx = {"intro": -1, "p0": 0, "p1": 1, "p2": 2, "p3": 3, "p4": 4, "close": 4}.get(s["id"], 0)
    chip_w = 150
    x0 = (W - (chip_w * 5 + 12 * 4)) // 2
    for i, label in enumerate(phases):
        x = x0 + i * (chip_w + 12)
        on = i <= active_idx
        fill = (18, 70, 78) if on else (12, 32, 38)
        outline = (43, 179, 192) if i == active_idx else ((40, 90, 100) if on else (30, 50, 58))
        d.rounded_rectangle([x, 74, x + chip_w, 108], radius=8, fill=fill, outline=outline, width=2)
        d.text((x + 14, 82), label, font=font(14, True), fill=(220, 240, 240) if on else (110, 130, 130))

    d.rounded_rectangle([70, 130, W - 70, H - 78], radius=18, fill=(10, 32, 38), outline=(43, 179, 192), width=2)

    badge_col = (255, 120, 80) if s["badge"].startswith("PHASE") else (43, 179, 192)
    d.rounded_rectangle([100, 156, 340, 196], radius=8, fill=(14, 42, 50), outline=badge_col, width=2)
    d.text((118, 164), s["badge"], font=font(18, True), fill=badge_col)

    d.text((100, 220), s["title"], font=font(38, True), fill=(255, 255, 255))
    d.text((100, 275), s["subtitle"], font=font(20), fill=(150, 200, 205))

    y = 330
    for bullet in s["bullets"]:
        d.ellipse([110, y + 8, 122, y + 20], fill=(43, 179, 192))
        d.text((136, y), bullet, font=font(20), fill=(210, 230, 230))
        y += 42

    d.rounded_rectangle([W - 360, 160, W - 100, 470], radius=12, fill=(8, 26, 32), outline=(30, 70, 78), width=1)
    d.text((W - 340, 178), "IN THIS PHASE", font=font(13, True), fill=(43, 179, 192))
    rail = [
        "Sensors → Kafka topics",
        "Windowed feature math",
        "P(breach) + why drivers",
        "Agent RAG remediation",
        "E-sign + WORM audit",
    ]
    ry = 215
    for i, row in enumerate(rail):
        lit = i <= max(0, active_idx)
        d.text((W - 340, ry), ("● " if lit else "○ ") + row, font=font(15),
               fill=(200, 230, 230) if lit else (90, 110, 115))
        ry += 36

    if fade < 0.95:
        overlay = Image.new("RGBA", (W, H), (6, 18, 22, int(255 * (1 - fade) * 0.4)))
        img = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")
    return img


def find_ffmpeg() -> str:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        return ffmpeg
    candidate = Path(
        r"C:\Users\arnab.das\AppData\Local\Microsoft\WinGet\Packages"
        r"\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.1.2-full_build\bin\ffmpeg.exe"
    )
    if candidate.exists():
        return str(candidate)
    raise SystemExit("ffmpeg not found")


def _clamp16(x: float) -> int:
    return max(-32767, min(32767, int(x)))


def synthesize_ambient_bgm(wav_path: Path, seconds: float = DURATION) -> None:
    """Generate an ambient-tech bed: soft pads + pulsing low motif + airy high shimmer."""
    n = int(seconds * SR)
    # Precompute samples
    samples = []
    for i in range(n):
        t = i / SR
        # Soft evolving pad (detuned fifths / octaves)
        pad = (
            0.18 * math.sin(2 * math.pi * 110 * t + 0.15 * math.sin(0.07 * t))
            + 0.12 * math.sin(2 * math.pi * 164.8 * t + 0.2 * math.sin(0.11 * t))
            + 0.10 * math.sin(2 * math.pi * 220 * t)
            + 0.06 * math.sin(2 * math.pi * 329.6 * t + 0.4 * math.sin(0.2 * t))
        )
        # Slow pulse / heartbeat tech tick (every ~1.5s)
        pulse_env = max(0.0, math.sin(2 * math.pi * (t / 1.5))) ** 8
        pulse = pulse_env * (
            0.12 * math.sin(2 * math.pi * 55 * t)
            + 0.05 * math.sin(2 * math.pi * 880 * t)
        )
        # Airy shimmer
        shimmer = 0.035 * math.sin(2 * math.pi * 1760 * t + math.sin(2 * math.pi * 0.4 * t))
        # Scene lift — slight brightening mid journey
        lift = 1.0 + 0.15 * math.sin(2 * math.pi * t / seconds)
        # Fade in/out
        fade = 1.0
        if t < 1.5:
            fade = t / 1.5
        elif t > seconds - 2.0:
            fade = max(0.0, (seconds - t) / 2.0)
        val = (pad + pulse + shimmer) * lift * fade * 0.85
        samples.append(_clamp16(val * 32767))

    with wave.open(str(wav_path), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SR)
        wf.writeframes(b"".join(struct.pack("<h", s) for s in samples))


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg = find_ffmpeg()
    total_frames = DURATION * FPS

    with tempfile.TemporaryDirectory(prefix="sentra_vid_") as tmp:
        tmp_path = Path(tmp)
        print(f"Rendering {total_frames} frames @ {FPS} fps …")
        for i in range(total_frames):
            t = i / FPS
            draw_frame(t).save(tmp_path / f"frame_{i:05d}.png")
            if i % 30 == 0:
                print(f"  frame {i}/{total_frames}")

        bgm = tmp_path / "ambient_bgm.wav"
        print("Synthesizing ambient-tech background music …")
        synthesize_ambient_bgm(bgm, DURATION)

        silent_mp4 = tmp_path / "video_silent.mp4"
        print("Encoding silent video …")
        subprocess.run([
            ffmpeg, "-y",
            "-framerate", str(FPS),
            "-i", str(tmp_path / "frame_%05d.png"),
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "22",
            "-movflags", "+faststart",
            str(silent_mp4),
        ], check=True, capture_output=True)

        print("Muxing ambient BGM …")
        subprocess.run([
            ffmpeg, "-y",
            "-i", str(silent_mp4),
            "-i", str(bgm),
            "-c:v", "copy",
            "-c:a", "aac", "-b:a", "160k",
            "-shortest",
            "-movflags", "+faststart",
            str(OUT),
        ], check=True, capture_output=True)

    print(f"Wrote {OUT} ({OUT.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
