import json
from pathlib import Path

import cv2
import numpy as np

from .cameras import Cameras
from .stereo import calibrate_views, markers


def targets(directory, square_mm=25., marker_mm=80.):
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
    cards = []
    card_width, card_height = marker_mm * 1.625, marker_mm * 1.675
    for marker_id in (0, 1):
        image = cv2.aruco.generateImageMarker(dictionary, marker_id, 6)
        cell = marker_mm / 6
        # A full white cell around the unchanged code isolates it from the art.
        quiet = marker_mm * 4/3
        white_x = (card_width - quiet)/2
        pattern_x = white_x + cell
        pattern_y = white_x + cell
        card = [
            f'<title>Wall reference {marker_id}: {marker_mm:g} mm black square</title>',
            f'<rect width="{card_width}" height="{card_height}" fill="#ede8dd"/>',
            f'<path d="M 0 {marker_mm*.39} Q {marker_mm*.43} {marker_mm*.39} {marker_mm*.43} 0 L 0 0 Z" fill="#bcc4b5"/>',
            f'<path d="M {card_width} {card_height-marker_mm*.42} Q {card_width-marker_mm*.43} {card_height-marker_mm*.42} {card_width-marker_mm*.43} {card_height} L {card_width} {card_height} Z" fill="#bcc4b5"/>',
            f'<rect x="{marker_mm*.04}" y="{marker_mm*.04}" width="{card_width-marker_mm*.08}" height="{card_height-marker_mm*.08}" fill="none" stroke="#a39885" stroke-width="{marker_mm*.003}"/>',
            f'<rect x="{white_x}" y="{white_x}" width="{quiet}" height="{quiet}" fill="white"/>',
            f'<g id="aruco-pattern-{marker_id}" fill="#000000" shape-rendering="crispEdges">',
            *[f'<rect x="{pattern_x+x*cell}" y="{pattern_y+y*cell}" width="{cell}" height="{cell}"/>'
              for y in range(6) for x in range(6) if image[y, x] == 0],
            '</g>',
            f'<text x="{card_width/2}" y="{card_height-marker_mm*.06}" text-anchor="middle" font-family="Georgia,serif" font-size="{marker_mm*.045}" fill="#655e50">{"I" if marker_id == 0 else "II"}</text>',
        ]
        svg(f"marker-{marker_id}.svg", card_width, card_height, card)
        cards.append(card)
    # Default 80-mm markers make two 130 x 134-mm cards on a portrait A4 sheet.
    # Explicit larger sizes retain their physical dimensions on a larger sheet.
    page_width = max(210., card_width + 20)
    page_height = max(297., 2*card_height + 29)
    gap = 9.
    x = (page_width-card_width)/2
    y = (page_height-2*card_height-gap)/2
    svg("markers-pair.svg", page_width, page_height, [
        f'<g transform="translate({x} {y+i*(card_height+gap)})">{"".join(card)}</g>'
        for i, card in enumerate(cards)])


def run(args):
    objects, left, right, signatures = [], [], [], []
    grid = np.zeros((9*6, 3), dtype=np.float32)
    grid[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2) * args.square_mm / 1000
    window = "MagicPointer calibration - Q cancels"
    try:
        with Cameras(args.left, args.right, args.fps, backend=args.camera_backend,
                     width=args.capture_width, height=args.capture_height) as cameras:
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
