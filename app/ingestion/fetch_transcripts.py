from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

from deep_translator import GoogleTranslator
from youtube_transcript_api import YouTubeTranscriptApi

from app.config import ensure_runtime_dirs, get_settings
from app.utils import MinIntervalLimiter, configure_logging


LOGGER = logging.getLogger(__name__)

DEFAULT_VIDEOS = {
    "3b1b_nn": "aircAruvnKk",
    "3b1b_transformers": "wjZofJX0v4M",
    "campusx_dl": "fHF22Wxuyw4",
    "codewithharry_ml": "C6YtPJxNULA",
}


def _parse_video_args(video_args: list[str] | None) -> dict[str, str]:
    if not video_args:
        return DEFAULT_VIDEOS

    videos: dict[str, str] = {}
    for item in video_args:
        if "=" not in item:
            raise ValueError(f"Invalid --video value `{item}`. Expected format: name=video_id")
        name, video_id = item.split("=", 1)
        name = name.strip()
        video_id = video_id.strip()
        if not name or not video_id:
            raise ValueError(f"Invalid --video value `{item}`. Name and video_id are required.")
        videos[name] = video_id
    return videos


def _fetch_transcript(video_id: str, languages: list[str]) -> list[dict[str, Any]]:
    # Compatibility across youtube-transcript-api versions:
    # - Older versions: YouTubeTranscriptApi.get_transcript(...)
    # - Newer versions: YouTubeTranscriptApi().fetch(...).to_raw_data()
    if hasattr(YouTubeTranscriptApi, "get_transcript"):
        return YouTubeTranscriptApi.get_transcript(video_id, languages=languages)

    api = YouTubeTranscriptApi()
    fetched = api.fetch(video_id, languages=languages)
    if hasattr(fetched, "to_raw_data"):
        return fetched.to_raw_data()
    return [
        {
            "text": getattr(item, "text", ""),
            "start": getattr(item, "start", 0.0),
            "duration": getattr(item, "duration", 0.0),
        }
        for item in fetched
    ]


def _one_line_error(error: Exception) -> str:
    message = str(error).strip()
    if not message:
        return error.__class__.__name__
    return message.splitlines()[0].strip()


def _translate_segments(
    segments: list[dict[str, Any]],
    source_lang: str,
    target_lang: str,
    min_interval_seconds: float,
) -> tuple[list[dict[str, Any]], int]:
    translator = GoogleTranslator(source=source_lang, target=target_lang)
    limiter = MinIntervalLimiter(min_interval_seconds=min_interval_seconds)
    translated: list[dict[str, Any]] = []
    failed_count = 0
    for idx, segment in enumerate(segments):
        text = segment.get("text", "")
        if text.strip():
            limiter.wait()
            try:
                translated_text = translator.translate(text)
                if translated_text:
                    text = translated_text
                else:
                    failed_count += 1
            except Exception as translation_error:
                failed_count += 1
                LOGGER.warning(
                    "Translation failed for segment %s; keeping original text. Reason: %s",
                    idx,
                    _one_line_error(translation_error),
                )
        translated.append({**segment, "text": text})
    return translated, failed_count


def fetch_and_normalize(
    video_id: str,
    name: str,
    output_dir: Path,
    translation_sleep_seconds: float,
) -> Path:
    language_used = "en"
    try:
        segments = _fetch_transcript(video_id, languages=["en"])
    except Exception as first_error:
        LOGGER.warning(
            "English transcript unavailable for %s (%s). Falling back to Hindi. Reason: %s",
            name,
            video_id,
            _one_line_error(first_error),
        )
        language_used = "hi"
        segments = _fetch_transcript(video_id, languages=["hi"])
        segments, failed_count = _translate_segments(
            segments=segments,
            source_lang="hi",
            target_lang="en",
            min_interval_seconds=translation_sleep_seconds,
        )
        if failed_count:
            LOGGER.warning(
                "%s: %d Hindi segment(s) were not translated and were kept as original text.",
                name,
                failed_count,
            )

    full_text = " ".join(segment.get("text", "") for segment in segments).strip()
    payload = {
        "video_id": video_id,
        "name": name,
        "original_language": language_used,
        "segment_count": len(segments),
        "text": full_text,
        "segments": segments,
    }

    output_path = output_dir / f"{name}.json"
    with output_path.open("w", encoding="utf-8") as file_handle:
        json.dump(payload, file_handle, ensure_ascii=False, indent=2)
    return output_path


def run_ingestion(videos: dict[str, str], overwrite: bool = False) -> None:
    settings = get_settings()
    ensure_runtime_dirs(settings)

    for name, video_id in videos.items():
        output_path = settings.raw_data_dir / f"{name}.json"
        if output_path.exists() and not overwrite:
            LOGGER.info("Skipping %s (already exists at %s). Use --overwrite to refresh.", name, output_path)
            continue

        try:
            written_path = fetch_and_normalize(
                video_id=video_id,
                name=name,
                output_dir=settings.raw_data_dir,
                translation_sleep_seconds=settings.ingestion_translation_sleep_seconds,
            )
            with written_path.open("r", encoding="utf-8") as file_handle:
                text_len = len(json.load(file_handle).get("text", ""))
            LOGGER.info("Ingested %s (%s chars) -> %s", name, f"{text_len:,}", written_path)
        except Exception:
            LOGGER.exception("Failed to ingest %s (%s).", name, video_id)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fetch YouTube transcripts and normalize into English JSON files."
    )
    parser.add_argument(
        "--video",
        action="append",
        help="Repeatable argument in the format name=video_id. If omitted, built-in defaults are used.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing transcript JSON files in raw data directory.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING, ERROR). Default: INFO.",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    configure_logging(args.log_level)
    videos = _parse_video_args(args.video)
    run_ingestion(videos=videos, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
