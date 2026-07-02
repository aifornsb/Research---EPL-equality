"""
Browser automation for the TxDOT Dallas ITS camera page.

Uses Playwright (Chromium) to load the public camera page, locate a
camera tile/panel by its exact display name, and save a screenshot of
that camera's actual video/image element — not the surrounding card or
its text label.

Notes on the target site:
The TxDOT camera page renders a list/grid of camera tiles, each showing
a still image or video feed with a name label. Because the DOM structure
of public traffic-camera sites changes over time, this module uses a
resilient text-based search strategy (find the element containing the
camera's exact name, then locate an <img>/<video>/<canvas> element inside
that same tile) rather than hardcoded CSS selectors. If the site
structure changes, update the selector hints in config/cameras.yaml or
the fallback logic below.

Correctness guarantees enforced here:
1. Exact-name matching only (no ambiguous substring matches), so we never
   grab the wrong camera when two names share a common prefix (e.g.
   "IH30 @ Carrier Pkwy" vs. a hypothetical "IH30 @ Carrier Pkwy East").
2. The screenshot target must be a real media element (img/video/canvas)
   with pixel dimensions above a configurable minimum — a screenshot of
   just the text label or an empty/broken-image placeholder is rejected
   and triggers a retry, rather than being silently accepted.
3. A single timestamp is generated once per capture attempt and reused
   for both the CSV row and the saved image filename, so the two never
   drift apart even if the browser work takes a few seconds.
"""
from __future__ import annotations

import base64
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from playwright.sync_api import sync_playwright, Page, Locator, TimeoutError as PWTimeoutError
from tenacity import retry, stop_after_attempt, wait_fixed, retry_if_exception_type

from src.config import AppConfig, IMAGES_DIR
from src.utils import image_path_for, now_local, safe_folder_name

logger = logging.getLogger("collector")


class CameraNotFoundError(Exception):
    """Raised when a camera with the given exact name cannot be located."""


class CameraMediaInvalidError(Exception):
    """Raised when a camera tile is found but no valid, sufficiently large
    media element (img/video/canvas) can be captured from it."""


@dataclass
class SnapshotResult:
    camera_name: str
    camera_slug: str
    captured_at: datetime
    image_path: Path
    success: bool
    error: str | None = None


def _find_camera_element(page: Page, camera_name: str) -> Locator:
    """Locate the DOM element representing a camera tile by EXACT name.

    Strategy (in order), all requiring an exact (not substring) match so
    that similarly-named cameras are never confused with each other:
    1. Playwright's built-in exact text matcher.
    2. Playwright's quoted-text selector (`text="..."`), which also
       performs an exact, whitespace-normalized match.
    3. A manual scan for elements whose fully whitespace-normalized text
       equals the target exactly (covers sites that mark up the label in
       an element type Playwright's text engine doesn't traverse well).
    """
    normalized_target = " ".join(camera_name.split()).lower()

    # Strategy 1: exact text
    locator = page.get_by_text(camera_name, exact=True)
    if locator.count() > 0:
        return locator.first

    # Strategy 2: quoted text= selector also performs an exact match
    quoted = camera_name.replace('"', '\\"')
    candidates = page.locator(f'text="{quoted}"')
    if candidates.count() > 0:
        return candidates.first

    # Strategy 3: loose scan, but still require full-string equality
    all_text_nodes = page.locator("body *:visible")
    count = min(all_text_nodes.count(), 800)  # safety cap
    for i in range(count):
        el = all_text_nodes.nth(i)
        try:
            txt = el.inner_text(timeout=200)
        except Exception:
            continue
        if txt and " ".join(txt.split()).lower() == normalized_target:
            return el

    raise CameraNotFoundError(f"Camera not found on page: {camera_name!r}")


def _media_natural_size(media: Locator) -> tuple[int, int]:
    """Return (width, height) in pixels for an img/video/canvas element,
    using the element's intrinsic/natural size where available so we
    detect broken images (naturalWidth == 0) or tiny icons."""
    try:
        size = media.evaluate(
            """(el) => {
                if (el.tagName === 'IMG') {
                    return [el.naturalWidth || 0, el.naturalHeight || 0];
                }
                if (el.tagName === 'VIDEO') {
                    return [el.videoWidth || 0, el.videoHeight || 0];
                }
                const rect = el.getBoundingClientRect();
                return [Math.round(rect.width), Math.round(rect.height)];
            }"""
        )
        return int(size[0]), int(size[1])
    except Exception:
        return (0, 0)


def _find_media_in_tile(container: Locator) -> Locator | None:
    """Find the actual camera feed element (img/video/canvas) within a
    camera tile container. Returns None if no media element exists."""
    media = container.locator("img, video, canvas")
    if media.count() == 0:
        return None
    return media.first


def _locate_camera_tile(page: Page, camera_name: str) -> Locator:
    """Find the camera name label, then walk up to its enclosing tile
    container (the smallest ancestor that also contains a media element)."""
    label_element = _find_camera_element(page, camera_name)

    # Try a sequence of ancestor levels, preferring the closest ancestor
    # that actually contains an image/video, rather than assuming the
    # immediate parent is the right container.
    for level in range(1, 6):
        container = label_element.locator(
            f"xpath=ancestor-or-self::*[self::div or self::li or self::article "
            f"or self::section][{level}]"
        )
        if container.count() == 0:
            continue
        if _find_media_in_tile(container) is not None:
            return container

    # Fall back to the immediate ancestor even without confirmed media;
    # the caller will raise CameraMediaInvalidError if no media is found.
    fallback = label_element.locator(
        "xpath=ancestor-or-self::*[self::div or self::li or self::article][1]"
    )
    return fallback if fallback.count() > 0 else label_element


def _try_download_native_image(page: Page, media: Locator, dest_path: Path) -> bool:
    """Best-quality path: save the media at its ORIGINAL, native resolution.

    - <img>: read its resolved URL (el.currentSrc/el.src) and download the
      raw file bytes through the browser's own request context (shares
      cookies/headers), writing them to disk untouched — no scaling, no
      re-encoding, exactly the pixels the camera server published.
      data: URIs are decoded directly.
    - <video>: draw the current frame onto an offscreen canvas at the
      video's native videoWidth x videoHeight and export as JPEG q=0.95.
      (May fail on cross-origin streams due to canvas tainting — that's
      fine, we fall back to a screenshot.)

    Returns True if a native-resolution file was written, False otherwise.
    Never raises — any failure just means "use the screenshot fallback".
    """
    try:
        tag = media.evaluate("el => el.tagName")

        if tag == "IMG":
            src = media.evaluate("el => el.currentSrc || el.src || ''")
            if not src:
                return False
            if src.startswith("data:"):
                _, b64 = src.split(",", 1)
                dest_path.write_bytes(base64.b64decode(b64))
                logger.info("Saved native image from data: URI -> %s", dest_path)
                return True
            response = page.request.get(src, timeout=15000)
            if not response.ok:
                logger.warning(
                    "Native image download returned HTTP %s for %s", response.status, src
                )
                return False
            body = response.body()
            if len(body) < 1000:  # sanity check: reject tiny error blobs
                logger.warning("Native image download too small (%d bytes)", len(body))
                return False
            dest_path.write_bytes(body)
            logger.info(
                "Saved native-resolution image (%d bytes) from %s -> %s",
                len(body),
                src,
                dest_path,
            )
            return True

        if tag == "VIDEO":
            data_url = media.evaluate(
                """(el) => {
                    if (!el.videoWidth || !el.videoHeight) return null;
                    const c = document.createElement('canvas');
                    c.width = el.videoWidth;
                    c.height = el.videoHeight;
                    c.getContext('2d').drawImage(el, 0, 0, c.width, c.height);
                    try { return c.toDataURL('image/jpeg', 0.95); }
                    catch (e) { return null; }  // tainted canvas (cross-origin)
                }"""
            )
            if data_url:
                _, b64 = data_url.split(",", 1)
                dest_path.write_bytes(base64.b64decode(b64))
                logger.info("Saved native video frame -> %s", dest_path)
                return True
            return False

    except Exception as exc:  # noqa: BLE001 - fall back to screenshot on any error
        logger.warning("Native-resolution capture failed (%s); using screenshot fallback.", exc)
    return False


def _capture_camera_media(
    page: Page,
    camera_name: str,
    dest_path: Path,
    min_width: int,
    min_height: int,
    jpeg_quality: int,
) -> None:
    """Find the camera's real media element and save it at the highest
    quality possible (never the label text or the whole card).

    Quality strategy, best first:
    1. Download the original image file / grab a native-resolution video
       frame (see _try_download_native_image) — full source quality.
    2. Fall back to an element screenshot at the page's device scale
       factor (2x by default) saved as JPEG at `jpeg_quality`.

    Raises CameraNotFoundError if the camera can't be located, or
    CameraMediaInvalidError if a media element can't be found or is too
    small / not actually loaded (e.g. a broken image icon).
    """
    tile = _locate_camera_tile(page, camera_name)
    media = _find_media_in_tile(tile)

    if media is None:
        raise CameraMediaInvalidError(
            f"No img/video/canvas element found in tile for camera: {camera_name!r}"
        )

    media.scroll_into_view_if_needed(timeout=5000)

    # Give lazy-loaded / streaming media a brief moment to actually paint
    # a frame before we measure and capture it.
    page.wait_for_timeout(500)

    width, height = _media_natural_size(media)
    if width < min_width or height < min_height:
        raise CameraMediaInvalidError(
            f"Media element for {camera_name!r} too small or not loaded "
            f"(got {width}x{height}, need >= {min_width}x{min_height})"
        )

    # Tier 1: native-resolution original (no scaling, no re-encode)
    if _try_download_native_image(page, media, dest_path):
        return

    # Tier 2: high-quality element screenshot (rendered at 2x DPI via the
    # context's device_scale_factor, encoded at high JPEG quality)
    media.screenshot(path=str(dest_path), type="jpeg", quality=jpeg_quality)
    logger.info("Saved screenshot fallback (quality=%d) -> %s", jpeg_quality, dest_path)


@retry(
    reraise=True,
    stop=stop_after_attempt(3),
    wait=wait_fixed(5),
    retry=retry_if_exception_type(
        (CameraNotFoundError, CameraMediaInvalidError, PWTimeoutError)
    ),
)
def _load_page_and_capture(
    config: AppConfig, camera_name: str, dest_path: Path
) -> None:
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        try:
            page = browser.new_page(
                viewport={
                    "width": config.screenshot.viewport_width,
                    "height": config.screenshot.viewport_height,
                },
                # Render at higher DPI so the screenshot fallback (tier 2)
                # captures roughly 2x the pixels. Has no effect on tier 1
                # (native file download), which is already full quality.
                device_scale_factor=config.screenshot.device_scale_factor,
            )
            page.goto(config.site_url, wait_until="networkidle", timeout=30000)
            page.wait_for_timeout(config.screenshot.wait_after_load_ms)
            _capture_camera_media(
                page,
                camera_name,
                dest_path,
                config.screenshot.min_media_width,
                config.screenshot.min_media_height,
                config.screenshot.jpeg_quality,
            )
        finally:
            browser.close()


def capture_snapshot(config: AppConfig, camera_name: str, camera_slug: str) -> SnapshotResult:
    """Capture a single timestamped snapshot for the given camera.

    A single `captured_at` timestamp is generated once here and used for
    both the image filename and the value returned in `SnapshotResult`,
    so downstream CSV rows and image paths always agree — even across
    internal retries, which all target the same destination path.

    Returns a SnapshotResult indicating success/failure. Never raises —
    callers should check `.success` and `.error` so one bad snapshot
    doesn't crash the collection loop.
    """
    captured_at = now_local(config.timezone)
    # Images are grouped by the camera's human-readable display name
    # (sanitized for the filesystem), e.g. data/images/IH30 @ Carrier Pkwy/...
    # The slug is still used for vehicle_id values in the traffic CSV.
    dest_path = image_path_for(IMAGES_DIR, safe_folder_name(camera_name), captured_at)

    try:
        _load_page_and_capture(config, camera_name, dest_path)
        logger.info(
            "Snapshot captured | camera=%s | path=%s | timestamp=%s",
            camera_name,
            dest_path,
            captured_at.isoformat(),
        )
        return SnapshotResult(
            camera_name=camera_name,
            camera_slug=camera_slug,
            captured_at=captured_at,
            image_path=dest_path,
            success=True,
        )
    except Exception as exc:  # noqa: BLE001 - we intentionally capture everything
        logger.error(
            "Snapshot FAILED | camera=%s | error=%s", camera_name, exc, exc_info=True
        )
        return SnapshotResult(
            camera_name=camera_name,
            camera_slug=camera_slug,
            captured_at=captured_at,
            image_path=dest_path,
            success=False,
            error=str(exc),
        )


def capture_both_cameras(config: AppConfig) -> tuple[SnapshotResult, SnapshotResult]:
    """Capture the toll camera and traffic camera snapshots for one cycle."""
    toll_result = capture_snapshot(
        config, config.toll_camera.name, config.toll_camera.slug
    )
    traffic_result = capture_snapshot(
        config, config.traffic_camera.name, config.traffic_camera.slug
    )
    return toll_result, traffic_result
