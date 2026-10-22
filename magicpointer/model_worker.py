"""WiLoR subprocess: raw BGR bytes in RAM, JSON landmarks out; no image files."""
import argparse
import contextlib
import json
import os
from pathlib import Path
import sys
import time

# Executed by absolute script path in a separate venv, not an installed package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def reply(value):
    print('MAGICPOINTER_JSON ' + json.dumps(value, allow_nan=False), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda'), required=True)
    parser.add_argument('--limit', type=int, required=True)
    args = parser.parse_args()
    # Dependencies are installed explicitly by the guide, never by an upstream
    # auto-installer running an unqualified pip in a different environment.
    os.environ['YOLO_AUTOINSTALL'] = 'false'
    model = None
    try:
        with contextlib.redirect_stdout(sys.stderr):
            import numpy as np
            from magicpointer.comparison_models import WiLoRModel
            model = WiLoRModel(args.directory, args.device, args.limit)
        reply({'ready': True, 'info': model.info})
        while True:
            header = sys.stdin.buffer.readline()
            if not header:
                break
            request = json.loads(header)
            shape = request['shape']
            size = int(np.prod(shape))
            if len(shape) != 3 or shape[2] != 3 or size <= 0 or size > 128*1024*1024:
                raise ValueError('Invalid worker frame shape')
            data = sys.stdin.buffer.read(size)
            if len(data) != size:
                raise ValueError('Incomplete worker frame')
            image = np.frombuffer(data, dtype=np.uint8).reshape(shape).copy()
            started = time.perf_counter()
            with contextlib.redirect_stdout(sys.stderr):
                hands = model.detect(image, request['timestamp'], request['view'])
            reply({'hands': hands, 'inference_ms': (time.perf_counter()-started)*1000})
    except Exception as error:
        reply({'error': f'{type(error).__name__}: {error}'})
        return 1
    finally:
        if model:
            model.close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
