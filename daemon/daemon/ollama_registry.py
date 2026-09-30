"""
Ollama registry client — fetch a GGUF blob by library path.

Resolves an Ollama library reference (e.g. ``qwen3:4b``) against
``registry.ollama.ai``'s Docker-v2-compatible surface:

    1. GET  /v2/library/<name>/manifests/<tag>   → manifest JSON
    2. Find the layer with media type ``application/vnd.ollama.image.model``
    3. GET  /v2/library/<name>/blobs/<digest>     → raw GGUF bytes

The blob is streamed to a temp file (``<dest>.part``) while sha256 is
computed incrementally — nothing is buffered in memory, since these files
run to 17 GB. A pre-existing ``.part`` file is resumed when the registry
responds to range requests; otherwise it is restarted cleanly.

This module is **runtime-agnostic**: it knows nothing about llama.cpp,
executors or the catalogue. It returns ``(path, computed_sha256)`` and
the caller decides whether the hash matches expectations. Kept separate
so a future runtime can reuse it without importing llama.cpp internals.

The registry-pull protocol against ``registry.ollama.ai`` is not a
documented public contract; digest verification at the call site is the
safety net if it drifts.
"""

from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path
from typing import Awaitable, Callable, Optional, Tuple

import httpx

logger = logging.getLogger(__name__)

REGISTRY_BASE = "https://registry.ollama.ai"
MODEL_MEDIA_TYPE = "application/vnd.ollama.image.model"

# Stream in 8 MB chunks — large enough for throughput on a fast link,
# small enough that a progress callback fires often enough to look alive.
_CHUNK_SIZE = 8 * 1024 * 1024


def _parse_ref(source_ref: str) -> Tuple[str, str]:
    """Split an Ollama library path into (name, tag).

    ``qwen3:4b``  → ``("qwen3", "4b")``
    ``mistral``   → ``("mistral", "latest")``
    """
    if ":" in source_ref:
        name, tag = source_ref.split(":", 1)
    else:
        name, tag = source_ref, "latest"
    return name.strip(), tag.strip()


async def _fetch_manifest(
    client: httpx.AsyncClient, name: str, tag: str,
) -> dict:
    """GET the manifest and return it as parsed JSON.

    Raises on non-200 or if the response is not valid JSON.
    """
    url = f"/v2/library/{name}/manifests/{tag}"
    response = await client.get(url)
    response.raise_for_status()
    return response.json()


def _find_model_layer(manifest: dict) -> dict:
    """Extract the GGUF model layer from a manifest.

    The manifest lists several layers (config, template, params, model).
    We want the one whose ``mediaType`` is the model blob.

    Raises ValueError when no matching layer exists.
    """
    for layer in manifest.get("layers", []):
        if layer.get("mediaType") == MODEL_MEDIA_TYPE:
            return layer
    raise ValueError(
        f"No layer with mediaType '{MODEL_MEDIA_TYPE}' in manifest "
        f"(layers: {[l.get('mediaType') for l in manifest.get('layers', [])]})"
    )


async def pull(
    source_ref: str,
    dest_dir: Path,
    *,
    progress_callback: Optional[Callable[[dict], Awaitable[None]]] = None,
) -> Tuple[Path, str]:
    """Download a GGUF blob from the Ollama registry.

    Args:
        source_ref:        Ollama library path, e.g. ``qwen3:4b``.
        dest_dir:          Directory to write the temp file into.
        progress_callback: Optional async callable receiving
                           ``{"status": ..., "completed": N, "total": N}``.

    Returns:
        ``(temp_path, sha256_hex)`` — the caller verifies the digest and
        renames the file into its final location.

    Raises:
        httpx.HTTPStatusError: On non-2xx responses from the registry.
        ValueError:            When the manifest has no model layer.
    """
    name, tag = _parse_ref(source_ref)
    dest_dir.mkdir(parents=True, exist_ok=True)
    part_path = dest_dir / f"{name.replace('/', '_')}_{tag}.part"

    async with httpx.AsyncClient(
        base_url=REGISTRY_BASE,
        timeout=httpx.Timeout(30.0, read=600.0),
        follow_redirects=True,
    ) as client:
        # ── Resolve manifest ─────────────────────────────────────
        logger.info(f"Fetching manifest for '{source_ref}' from Ollama registry")
        manifest = await _fetch_manifest(client, name, tag)
        layer = _find_model_layer(manifest)

        blob_digest = layer["digest"]       # e.g. "sha256:abcdef..."
        blob_size = layer.get("size", 0)    # total bytes
        blob_url = f"/v2/library/{name}/blobs/{blob_digest}"

        # ── Resume or restart ────────────────────────────────────
        headers: dict = {}
        existing_bytes = 0
        hasher = hashlib.sha256()

        if part_path.exists():
            existing_bytes = part_path.stat().st_size
            if existing_bytes > 0 and existing_bytes < blob_size:
                # Hash what we already have so the final digest is correct.
                logger.info(
                    f"Resuming download at {existing_bytes / 1e6:.0f} MB "
                    f"of {blob_size / 1e6:.0f} MB"
                )
                with open(part_path, "rb") as fh:
                    while True:
                        chunk = fh.read(_CHUNK_SIZE)
                        if not chunk:
                            break
                        hasher.update(chunk)
                headers["Range"] = f"bytes={existing_bytes}-"
            elif existing_bytes >= blob_size:
                # Already have the full file from a previous attempt.
                logger.info("Part file is already complete — hashing")
                with open(part_path, "rb") as fh:
                    while True:
                        chunk = fh.read(_CHUNK_SIZE)
                        if not chunk:
                            break
                        hasher.update(chunk)
                return part_path, hasher.hexdigest()

        # ── Stream the blob ──────────────────────────────────────
        logger.info(
            f"Downloading blob {blob_digest[:20]}... "
            f"({blob_size / 1e9:.1f} GB)"
        )
        async with client.stream("GET", blob_url, headers=headers) as response:
            # If the server doesn't support range requests and we asked
            # for a range, restart from scratch.
            if headers.get("Range") and response.status_code == 200:
                logger.info(
                    "Registry does not support range requests — "
                    "restarting download"
                )
                existing_bytes = 0
                hasher = hashlib.sha256()
                mode = "wb"
            elif response.status_code == 206:
                mode = "ab"
            else:
                response.raise_for_status()
                mode = "wb"
                existing_bytes = 0
                hasher = hashlib.sha256()

            downloaded = existing_bytes
            with open(part_path, mode) as fh:
                async for chunk in response.aiter_bytes(
                    chunk_size=_CHUNK_SIZE,
                ):
                    fh.write(chunk)
                    hasher.update(chunk)
                    downloaded += len(chunk)

                    if progress_callback:
                        await progress_callback({
                            "status": "downloading",
                            "completed": downloaded,
                            "total": blob_size,
                        })

    computed = hasher.hexdigest()
    logger.info(
        f"Download complete — {downloaded / 1e9:.1f} GB, "
        f"sha256:{computed[:12]}..."
    )
    return part_path, computed
