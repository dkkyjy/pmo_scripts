"""Streaming fingerprint-based fixed-source filtering helpers.

This module provides:
1. Core duplicate filtering algorithm (remove_fixed_sources_streaming)
2. Structured YAML payload filtering helper
"""

from __future__ import annotations

import re
from collections import deque
from dataclasses import dataclass
from itertools import combinations
from typing import Any, Dict, List, Optional, Tuple

from logger_config import logger

PairKey = Tuple[str, str]
Match = Tuple[str, int]
EventPayload = Dict[str, Any]
EventMap = Dict[str, EventPayload]

FINGERPRINT_THRESHOLD_NS = 20
FINGERPRINT_BUFFER_SIZE = 50
FINGERPRINT_MIN_PAIR = 5
FINGERPRINT_HISTORY_SIZE = 200


@dataclass
class EventFingerprint:
    """Event fingerprint container used by streaming duplicate filtering."""

    line_content: str
    timestamp: float
    pairs: Dict[PairKey, int]
    matches: List[Match]
    event_key: Optional[str] = None
    payload: Optional[EventPayload] = None
    is_duplicate: bool = False


def parse_line_for_fingerprint(
    line: str,
) -> Tuple[Optional[float], List[Match], Dict[PairKey, int]]:
    """Parse one raw text line into timestamp, matches and pair fingerprint."""
    match = re.match(r"([\d.]+):\s*\[(.*)\]", line)
    if not match:
        return None, [], {}

    timestamp = float(match.group(1))
    detectors: List[Match] = []
    matches_str = match.group(2).rstrip("]")

    items = matches_str.split("), (")
    for item in items:
        item = item.strip("() ")
        if not item:
            continue
        parts = item.split(",")
        if len(parts) < 2:
            continue
        du_id = parts[0].strip().strip("'\"")
        try:
            time_ns = int(parts[1].strip())
            detectors.append((du_id, time_ns))
        except ValueError:
            continue

    pairs = _build_pairs(detectors)
    return timestamp, detectors, pairs


def _build_pairs(matches: List[Match]) -> Dict[PairKey, int]:
    """Build pairwise time-difference fingerprint for one event."""
    pairs: Dict[PairKey, int] = {}
    for (id1, t1), (id2, t2) in combinations(matches, 2):
        pair_key = _pair_key(id1, id2)
        pairs[pair_key] = t1 - t2
    return pairs


def _pair_key(id1: str, id2: str) -> PairKey:
    """Return a stable pair key for two detector IDs."""
    if id1 <= id2:
        return (id1, id2)
    return (id2, id1)


def is_fingerprint_similar(
    pairs1: Dict[PairKey, int],
    pairs2: Dict[PairKey, int],
    threshold: int,
    min_pair: int,
) -> bool:
    """Judge whether two event fingerprints are similar enough."""
    common_pairs = set(pairs1.keys()) & set(pairs2.keys())
    if len(common_pairs) < min_pair:
        return False

    similar_count = sum(
        1 for pair in common_pairs if abs(pairs1[pair] - pairs2[pair]) < threshold
    )
    return similar_count >= min_pair


def remove_fixed_sources_streaming(
    matching_times: List[Tuple[float, List[Match]]],
    threshold: int = FINGERPRINT_THRESHOLD_NS,
    buffer_size: int = FINGERPRINT_BUFFER_SIZE,
    min_pair: int = FINGERPRINT_MIN_PAIR,
    history_size: int = FINGERPRINT_HISTORY_SIZE,
) -> List[Tuple[float, List[Match]]]:
    """Filter duplicate fixed-source events using streaming fingerprints."""
    logger.info("[Step 2] Removing fixed sources (Streaming Fingerprint Method)...")
    logger.info(
        "[Step 2] Params: threshold={}ns, buffer_size={}, min_pair={}, history_size={}",
        threshold,
        buffer_size,
        min_pair,
        history_size,
    )

    events: List[EventFingerprint] = []
    for gps_time, matches in matching_times:
        pairs = _build_pairs(matches)
        line_repr = f"{gps_time}: {matches}"
        events.append(EventFingerprint(line_repr, gps_time, pairs, matches))

    output: List[Tuple[float, List[Match]]] = []
    buffer_list: List[EventFingerprint] = []
    history_fingerprints = deque(maxlen=history_size)
    removed_count = 0

    for event in events:
        if any(
            is_fingerprint_similar(event.pairs, hist, threshold, min_pair)
            for hist in history_fingerprints
        ):
            history_fingerprints.append(event.pairs)
            removed_count += 1
            continue

        is_dup = False
        for buffered in buffer_list:
            if buffered.is_duplicate:
                continue
            if is_fingerprint_similar(event.pairs, buffered.pairs, threshold, min_pair):
                buffered.is_duplicate = True
                history_fingerprints.append(buffered.pairs)
                history_fingerprints.append(event.pairs)
                is_dup = True
                removed_count += 1
                break

        if is_dup:
            continue

        buffer_list.append(event)

        while len(buffer_list) > buffer_size:
            oldest = buffer_list.pop(0)
            if not oldest.is_duplicate:
                output.append((oldest.timestamp, oldest.matches))
            else:
                removed_count += 1

    for event in buffer_list:
        if not event.is_duplicate:
            output.append((event.timestamp, event.matches))
        else:
            removed_count += 1

    logger.info(
        "[Step 2] Complete: Removed {} background events. {} signal events retained.",
        removed_count,
        len(output),
    )
    return output


def _extract_time_ns(value: Any) -> Optional[int]:
    """Extract one nanosecond sample from YAML time payload entry."""
    if isinstance(value, list) and len(value) > 0:
        sample = value[0]
    else:
        sample = value

    if isinstance(sample, (int, float)):
        return int(sample)
    return None


def _extract_event_fingerprint(
    event_key: str,
    payload: EventPayload,
) -> Optional[EventFingerprint]:
    """Convert one matching YAML event payload to EventFingerprint."""
    if not isinstance(payload, dict):
        return None

    gps_time = payload.get("gps_time", None)
    if not isinstance(gps_time, (int, float)):
        return None

    time_map = payload.get("time", None)
    if not isinstance(time_map, dict):
        return None

    matches: List[Match] = []
    for du_id, ns_value in time_map.items():
        time_ns = _extract_time_ns(ns_value)
        if time_ns is None:
            continue
        matches.append((str(du_id), time_ns))

    if len(matches) < 2:
        return None

    return EventFingerprint(
        line_content=f"{gps_time}: {matches}",
        timestamp=float(gps_time),
        pairs=_build_pairs(matches),
        matches=matches,
        event_key=event_key,
        payload=payload,
    )


def filter_fixed_sources_from_payload(
    matching_payload: EventMap,
    threshold: int = FINGERPRINT_THRESHOLD_NS,
    buffer_size: int = FINGERPRINT_BUFFER_SIZE,
    min_pair: int = FINGERPRINT_MIN_PAIR,
    history_size: int = FINGERPRINT_HISTORY_SIZE,
) -> EventMap:
    """Filter duplicate fixed-source events from structured matching payload."""
    events: List[EventFingerprint] = []
    passthrough_keys = set()

    for event_key, payload in matching_payload.items():
        event = _extract_event_fingerprint(event_key, payload)
        if event is None:
            passthrough_keys.add(event_key)
            logger.warning(
                "Event {} missing structured time/gps data for fingerprint filtering; keep as-is.",
                event_key,
            )
            continue
        events.append(event)

    retained_keys = set()
    buffer_list: List[EventFingerprint] = []
    history_fingerprints = deque(maxlen=history_size)

    for event in events:
        if any(
            is_fingerprint_similar(event.pairs, hist, threshold, min_pair)
            for hist in history_fingerprints
        ):
            history_fingerprints.append(event.pairs)
            continue

        is_dup = False
        for buffered in buffer_list:
            if buffered.is_duplicate:
                continue
            if is_fingerprint_similar(event.pairs, buffered.pairs, threshold, min_pair):
                buffered.is_duplicate = True
                history_fingerprints.append(buffered.pairs)
                history_fingerprints.append(event.pairs)
                is_dup = True
                break

        if is_dup:
            continue

        buffer_list.append(event)
        while len(buffer_list) > buffer_size:
            oldest = buffer_list.pop(0)
            if not oldest.is_duplicate and oldest.event_key is not None:
                retained_keys.add(oldest.event_key)

    for event in buffer_list:
        if not event.is_duplicate and event.event_key is not None:
            retained_keys.add(event.event_key)

    filtered_payload: EventMap = {}
    for event_key, payload in matching_payload.items():
        if event_key in retained_keys or event_key in passthrough_keys:
            filtered_payload[event_key] = payload
    return filtered_payload
