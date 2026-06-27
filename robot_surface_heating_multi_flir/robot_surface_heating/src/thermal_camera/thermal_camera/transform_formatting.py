#!/usr/bin/env python3

import re
import numpy as np

RAW_TEXT = r"""
[INFO] [1778609088.515173477] [pnp_extrinsic]: EEF pose: [0.916, -0.651, 0.181]
[INFO] [1778609088.515547355] [pnp_extrinsic]: FK: [[-0.04665562  0.9988191  -0.01355225  0.91629551]
 [ 0.99259383  0.04787969  0.1116469  -0.65115599]
 [ 0.11216393 -0.00824293 -0.99365553  0.18104249]
 [ 0.          0.          0.          1.        ]]
[INFO] [1778609121.837222940] [pnp_extrinsic]: EEF pose: [0.348, -0.523, 0.171]
[INFO] [1778609121.837553855] [pnp_extrinsic]: FK: [[ 0.18761593  0.91976111  0.34473145  0.34793364]
 [ 0.98204452 -0.16860038 -0.08463144 -0.52302896]
 [-0.01971885  0.35441983 -0.93487847  0.17109212]
 [ 0.          0.          0.          1.        ]]
[INFO] [1778609186.477948256] [pnp_extrinsic]: EEF pose: [0.364, -0.055, 0.139]
[INFO] [1778609186.478267190] [pnp_extrinsic]: FK: [[ 0.00946937  0.99727816  0.07312044  0.36409055]
 [ 0.88986645 -0.04175919  0.45430593 -0.05495848]
 [ 0.45612283  0.06076544 -0.88783981  0.13907436]
 [ 0.          0.          0.          1.        ]]
[INFO] [1778609266.882243050] [pnp_extrinsic]: EEF pose: [0.990, -0.181, 0.184]
[INFO] [1778609266.882595012] [pnp_extrinsic]: FK: [[ 0.19985015  0.92749848 -0.31592165  0.99014369]
 [ 0.90172617 -0.04795438  0.42963973 -0.18146892]
 [ 0.38334037 -0.37073838 -0.84593334  0.18422947]
 [ 0.          0.          0.          1.        ]]
"""


def extract_fk_matrices(raw_text: str):
    lines = raw_text.splitlines()

    matrices = []
    collecting = False
    current = []

    for line in lines:
        if "FK:" in line:
            collecting = True
            current = []

        if collecting:
            # remove log prefix
            clean = re.sub(r".*?\[\s*", "[", line)

            # extract all numbers from the line
            nums = re.findall(r"[-+]?\d*\.\d+|[-+]?\d+", clean)
            current.extend([float(n) for n in nums])

            # end of matrix
            if "]]" in line:
                if len(current) == 16:
                    matrices.append(np.array(current).reshape(4, 4))
                else:
                    print(f"Warning: expected 16 values, got {len(current)}")
                collecting = False

    return matrices


def format_output(matrices):
    print("raw_world_pts = np.array([")
    for i, m in enumerate(matrices):
        print("    [")
        for row in m:
            row_str = ", ".join(f"{v: .8f}" for v in row)
            print(f"        [{row_str}],")
        if i < len(matrices) - 1:
            print("    ],\n")
        else:
            print("    ],")
    print("], dtype=np.float64)")


def main():
    mats = extract_fk_matrices(RAW_TEXT)

    if not mats:
        print("No FK matrices found.")
        return

    format_output(mats)


if __name__ == "__main__":
    main()