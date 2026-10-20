import json
from pathlib import Path

import cv2
import numpy as np

from .cameras import Cameras
from .stereo import calibrate_views, markers


def targets(directory, square_mm=25., marker_mm=200.):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    # SVG dimensions preserve physical scale when printed at 100%, no fit-to-page.
    def svg(name, width, height, cells):
        content = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}mm" '
                   f'height="{height}mm" viewBox="0 0 {width} {height}">'
                   '<rect width="100%" height="100%" fill="white"/>' + "".join(cells) + '</svg>')
        (directory / name).write_text(content)
    margin = square_mm
    svg("checkerboard.svg", 12*square_mm, 9*square_mm, [
        f'<rect x="{margin+x*square_mm}" y="{margin+y*square_mm}" width="{square_mm}" height="{square_mm}"/>'
        for y in range(7) for x in range(10) if (x+y) % 2 == 0])
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    for marker_id in (0, 1):
        image = cv2.aruco.generateImageMarker(dictionary, marker_id, 6)
        cell = marker_mm / 6
        svg(f"marker-{marker_id}.svg", marker_mm + 2*cell, marker_mm + 2*cell, [
            f'<rect x="{(x+1)*cell}" y="{(y+1)*cell}" width="{cell}" height="{cell}"/>'
            for y in range(6) for x in range(6) if image[y, x] == 0])


def run(args):
    objects, left, right, signatures = [], [], [], []
    grid = np.zeros((9*6, 3), dtype=np.float32)
    grid[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2) * args.square_mm / 1000
    window = "MagicPointer calibration - Q cancels"
    try:
        with Cameras(args.left, args.right, args.fps) as cameras:
            print("Cameras:", cameras.info)
            print("SPACE: capture varied checkerboard views; C: solve after 15; Q: cancel")
            while True:
                frames = cameras.read()
                if frames[0].image.shape != frames[1].image.shape:
                    raise ValueError("Both cameras must use the same resolution")
                corners = []
                panels = []
                paired = abs(frames[0].grabbed_at - frames[1].grabbed_at) <= .04
                for frame in frames:
                    gray = cv2.cvtColor(frame.image, cv2.COLOR_BGR2GRAY)
                    found, points = cv2.findChessboardCorners(gray, (9, 6))
                    if found:
                        points = cv2.cornerSubPix(gray, points, (11, 11), (-1, -1),
                            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, .001))
                    corners.append(points if found else None)
                    panel = frame.image.copy()
                    if found:
                        cv2.drawChessboardCorners(panel, (9, 6), points, True)
                    panels.append(cv2.resize(panel, (640, 360)))
                preview = np.hstack(panels)
                text = f"{len(objects)}/15+ views | SPACE capture | C solve | Q cancel"
                cv2.putText(preview, text, (15, 28), cv2.FONT_HERSHEY_SIMPLEX, .65, (0, 220, 255), 2)
                cv2.imshow(window, preview)
                key = cv2.waitKey(1) & 255
                if key in (ord('q'), 27) or cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                    print("Calibration canceled; existing file preserved")
                    return 0
                if key == ord(' ') and paired and all(c is not None for c in corners):
                    # Point displacement rejects repeated identical views while allowing tilt.
                    signature = np.concatenate([c.ravel() for c in corners])
                    if any(np.sqrt(np.mean((signature-old)**2)) < 15 for old in signatures):
                        print("Move/tilt the checkerboard to capture a different view")
                    else:
                        signatures.append(signature)
                        objects.append(grid.copy())
                        left.append(corners[0])
                        right.append(corners[1])
                        print(f"Captured view {len(objects)}")
                if key == ord('c'):
                    if len(objects) < 15:
                        print("Capture at least 15 varied views first")
                        continue
                    break
            stereo = calibrate_views(objects, left, right, tuple(frames[0].image.shape[1::-1]))
            print("RMS pixel errors:", stereo.data["rms"])
            print("Remove checkerboard; show wall markers 0 and 1 to both cameras. SPACE saves.")
            while True:
                frames = cameras.read()
                found = [markers(f.image) for f in frames]
                preview = np.hstack([cv2.resize(f.image, (640, 360)) for f in frames])
                cv2.putText(preview, "Show markers 0 + 1 in BOTH views | SPACE save | Q cancel",
                            (10, 25), cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 220, 255), 2)
                cv2.imshow(window, preview)
                key = cv2.waitKey(1) & 255
                if key in (ord('q'), 27) or cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                    return 0
                if key != ord(' ') or any(not {0, 1}.issubset(m) for m in found):
                    continue
                if not stereo.check_frames(frames):
                    print("Camera pair arrived too far apart; try again")
                    continue
                a, b = (np.concatenate([m[i] for i in (0, 1)]) for m in found)
                points = stereo.reconstruct(a, b)
                if points is None:
                    print("Marker reconstruction failed; check marker visibility and calibration")
                    continue
                spacing = float(np.linalg.norm(points[:4].mean(axis=0)-points[4:].mean(axis=0)))
                sizes = [np.linalg.norm(points[i]-points[(i//4)*4+(i+1)%4]) for i in range(8)]
                size = float(np.mean(sizes))
                if abs(spacing - args.marker_spacing) > args.marker_spacing * .05 or abs(size - args.marker_size) > args.marker_size * .08:
                    print(f"Measured spacing {spacing:.3f} m, marker size {size:.3f} m disagree with supplied measurements")
                    continue
                stereo.data.update({"wall_markers": [{str(i): m[i].tolist() for i in (0, 1)} for m in found],
                                    "marker_spacing_m": spacing, "marker_size_m": size,
                                    "camera_indices": [args.left, args.right], "capture": cameras.info})
                path = Path(args.calibration)
                path.parent.mkdir(parents=True, exist_ok=True)
                temporary = path.with_suffix(".tmp")
                temporary.write_text(json.dumps(stereo.data, indent=2, allow_nan=False))
                temporary.replace(path)
                print(f"Saved {path}. Keep cameras and wall markers fixed.")
                return 0
    finally:
        cv2.destroyAllWindows()
