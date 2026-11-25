# LUMEN

Browser-based optical placard positioning research application. Use standard iPhone rear RGB cameras, multiple photographs or video, optional flashlight illumination, and commercially available removable optical fiducials. No CAD import, native iOS app, LiDAR, NFC, or custom hardware required.

## Build with Fable 5.2
Read every file in `prompt_series/` before coding. Execute in numeric order, starting with 00_MASTER.md. Maintain a working app at each milestone. Improve the proposed algorithms and UX when evidence supports it; document decisions and test results. Do not stop at design documents or pretend unimplemented features work.

Target: single-origin FastAPI web application deployed to Posit Connect, CPU-first Python/OpenCV/NumPy/SciPy and locally bundled JavaScript/HTML/CSS. Verify target platform dependencies and browser behavior. Use synthetic fixtures and generic configurations only; no employer data, photos, credentials or internal requirements in this repository.

## Workflow
Identify operation, place 2–4 distinct known-size AprilTags near the target, capture overlapping images or video of the side boundary/curvature and floor interface, establish scale and local geometry, confirm datums, calculate specified target corners, show registered placement guidance, then measure observed placement from a verification image. Photometric analysis is a research extension to reliable planar metrology. Require explicit coordinate definitions and fail closed on ambiguous or unvalidated geometry. All outputs are prototype guidance, not product acceptance.

## Reference links
- https://docs.posit.co/connect/user/fastapi/
- https://developer.mozilla.org/en-US/docs/Web/API/MediaDevices/getUserMedia
- https://developer.mozilla.org/en-US/docs/Web/API/MediaStreamTrack/getCapabilities
- https://docs.opencv.org/4.x/d9/dab/tutorial_homography.html
- https://github.com/AprilRobotics/apriltag
- https://texxary-shop.com/products/april-tags-sticker-sheet
- https://arxiv.org/abs/2303.14190 (research background)

Use only approved environments for real manufacturing data and validate actual physical accuracy, surface compatibility and tooling controls before production use.
