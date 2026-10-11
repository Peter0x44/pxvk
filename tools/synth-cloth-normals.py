"""Synthetic pxvk-replay captures for PhysX 2.8.4's cloth normal kernel.

usage: synth-cloth-normals.py <PhysXCore.dll> <fatbin offset hex> <out dir>

The fatbin offset is module 14's `fatbin_offset` in the manifest that
tools/extract-physx-ptx.py writes for a 2.8.4 PhysXCore.dll. Output belongs
in tmp/ (it contains the proprietary fatbin). Replay the result with
pxvk-replay (--time N for timing): no native capture of these kernels
exists until a 2.8.4 game with GPU cloth is traced.

Writes captures of PxgCudaDeformableUberPreCalNormals<B,1> on an n x n cloth
grid (n*n <= B vertices, one triangle per thread), with expected normals
computed in float32 (sum of unnormalized face normals per vertex).
"""
import math
import os
import struct
import sys

import numpy as np

dll, offset, out = sys.argv[1], int(sys.argv[2], 16), sys.argv[3]
os.makedirs(out, exist_ok=True)

data = open(dll, 'rb').read()
magic, version, header_size, fat_size = struct.unpack_from('<IHHQ', data, offset)
assert magic == 0xBA55ED50, hex(magic)
open(os.path.join(out, 'fatbin_0001.bin'), 'wb').write(data[offset:offset + header_size + fat_size])

KERNEL = ('_Z34PxgCudaDeformableUberPreCalNormalsILj{b}ELj1EEvPK32PxgCudaDeformableTriangleIndicesPK12'
          'PxcVectorPadP9PxcVectorjj')
BASE = 0x28800000
INDICES, POSITIONS, NORMALS = BASE, BASE + 0x40000, BASE + 0x80000


def sparse(regions):
    return b''.join(struct.pack('<II', a, len(b)) + b for a, b in regions)


ordinal = 0
for block, n in ((1024, 32), (256, 16), (64, 8)):
    ordinal += 1
    rng = np.random.default_rng(block)
    # A gently rumpled cloth: grid in x/z, small random height.
    pos = np.zeros((n * n, 4), dtype=np.float32)
    for y in range(n):
        for x in range(n):
            pos[y * n + x, :3] = (x * 0.1, rng.uniform(-0.02, 0.02), y * 0.1)
    tris = []
    for y in range(n - 1):
        for x in range(n - 1):
            a, b, c, d = y * n + x, y * n + x + 1, (y + 1) * n + x, (y + 1) * n + x + 1
            tris += [(a, b, c), (b, d, c)]
    tris = np.array(tris, dtype=np.uint32)
    normals = np.zeros((n * n, 3), dtype=np.float32)
    for i0, i1, i2 in tris:
        p0, p1, p2 = pos[i0, :3], pos[i1, :3], pos[i2, :3]
        e1, e2 = (p1 - p0).astype(np.float32), (p2 - p0).astype(np.float32)
        nrm = np.array([e1[1] * e2[2] - e1[2] * e2[1],
                        e1[2] * e2[0] - e1[0] * e2[2],
                        e1[0] * e2[1] - e1[1] * e2[0]], dtype=np.float32)
        for v in (i0, i1, i2):
            normals[v] += nrm
    grid = math.ceil(len(tris) / block)
    name = KERNEL.format(b=block)
    base = os.path.join(out, f'capture_{ordinal:06d}')
    params = struct.pack('<IIIII', INDICES, POSITIONS, NORMALS, len(tris), n * n)
    with open(base + '.txt', 'w') as f:
        f.write(f'kernel {name}\nframe 0\nlaunch {ordinal}\nfatbin 1\n')
        f.write(f'grid {grid} 1 1\nblock {block} 1 1\nshared 0\nparams {params.hex()}\n')
        f.write(f'allocation {BASE:08x} {0x100000}\n')
    before = [(INDICES, tris.tobytes()), (POSITIONS, pos.tobytes()), (NORMALS, bytes(normals.nbytes))]
    after = [(INDICES, tris.tobytes()), (POSITIONS, pos.tobytes()), (NORMALS, normals.tobytes())]
    open(base + '.before', 'wb').write(sparse(before))
    open(base + '.after', 'wb').write(sparse(after))
    print(f'{ordinal}: block {block}, {n}x{n} vertices, {len(tris)} triangles, grid {grid}')
