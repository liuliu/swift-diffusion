"""Audit raw FP32 SQLite weights using actual FP16 round-to-nearest conversion."""

import argparse
import json
import math
from pathlib import Path
import sqlite3
import struct

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path,
                        default=Path('/slow/Data/moss_image_sft_dit_f32.ckpt'))
    parser.add_argument('--output', type=Path,
                        default=Path(__file__).resolve().parent / 'artifacts/weights_fp16_audit.json')
    args = parser.parse_args()
    limits = np.finfo(np.float16)
    counters = ['elements', 'source_zeros', 'source_nonfinite', 'outside_fp16_finite_range',
                'overflow_to_inf', 'underflow_to_zero', 'below_fp16_min_subnormal',
                'below_fp16_min_normal', 'fp16_subnormal']
    totals = dict.fromkeys(counters, 0)
    rows = []
    with sqlite3.connect(args.checkpoint.resolve().as_uri() + '?mode=ro', uri=True) as db:
        metadata = db.execute('SELECT rowid,name,datatype,dim,length(data) FROM tensors ORDER BY rowid').fetchall()
        for rowid, name, datatype, dimensions, byte_count in metadata:
            assert datatype == 16384, (name, datatype)  # CCV_32F, uncompressed raw data.
            shape = [x for x in struct.unpack('<' + 'i' * (len(dimensions)//4), dimensions) if x]
            assert byte_count == math.prod(shape) * 4, (name, shape, byte_count)
            row = dict(name=name, shape=shape, **dict.fromkeys(counters, 0))
            peak, minimum = 0., math.inf
            with db.blobopen('tensors', 'data', rowid, readonly=True) as blob:
                while data := blob.read(16 * 1024 * 1024):
                    source = np.frombuffer(data, dtype='<f4')
                    finite = np.isfinite(source)
                    absolute = np.abs(source)
                    with np.errstate(over='ignore', under='ignore', invalid='ignore'):
                        half = source.astype(np.float16)
                    nonzero = source != 0
                    magnitude = np.abs(half)
                    row['elements'] += source.size
                    row['source_zeros'] += int(np.count_nonzero(~nonzero))
                    row['source_nonfinite'] += int(np.count_nonzero(~finite))
                    row['outside_fp16_finite_range'] += int(np.count_nonzero(finite & (absolute > limits.max)))
                    row['overflow_to_inf'] += int(np.count_nonzero(finite & np.isinf(half)))
                    row['underflow_to_zero'] += int(np.count_nonzero(nonzero & (half == 0)))
                    row['below_fp16_min_subnormal'] += int(np.count_nonzero(nonzero & (absolute < limits.smallest_subnormal)))
                    row['below_fp16_min_normal'] += int(np.count_nonzero(nonzero & (absolute < limits.smallest_normal)))
                    row['fp16_subnormal'] += int(np.count_nonzero((magnitude > 0) & (magnitude < limits.smallest_normal)))
                    peak = max(peak, float(np.max(absolute, where=finite, initial=0)))
                    minimum = min(minimum, float(np.min(absolute, where=nonzero & finite, initial=np.inf)))
            row['max_abs'] = peak
            row['min_nonzero_abs'] = minimum if math.isfinite(minimum) else None
            rows.append(row)
            for key in counters:
                totals[key] += row[key]
            if len(rows) % 64 == 0:
                print(f'Audited {len(rows)}/{len(metadata)} tensors', flush=True)
    report = dict(checkpoint=str(args.checkpoint.resolve()), tensors=len(rows), totals=totals,
                  fp16_max=float(limits.max), fp16_min_normal=float(limits.smallest_normal),
                  fp16_min_subnormal=float(limits.smallest_subnormal),
                  rounding='IEEE FP16 round to nearest, ties to even',
                  underflow_definition='Nonzero FP32 weight rounds to positive or negative zero',
                  overflow_definition='Finite FP32 weight rounds to positive or negative infinity',
                  tensors_with_underflow=sum(r['underflow_to_zero'] > 0 for r in rows),
                  tensors_with_overflow=sum(r['overflow_to_inf'] > 0 for r in rows),
                  max_abs=max(r['max_abs'] for r in rows),
                  min_nonzero_abs=min(r['min_nonzero_abs'] for r in rows if r['min_nonzero_abs'] is not None),
                  results=rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'results'}, indent=2), flush=True)
    print('Largest underflow counts:', flush=True)
    for row in sorted(rows, key=lambda r: r['underflow_to_zero'], reverse=True)[:12]:
        print(row['name'], row['underflow_to_zero'], '/', row['elements'], flush=True)
    print('Saved:', args.output, flush=True)


if __name__ == '__main__':
    main()
