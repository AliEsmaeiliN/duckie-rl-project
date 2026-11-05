"""
Live window for evaluate.py --show: what the camera sees next to what the policy sees.

    +-----------------------------+-----------+
    | camera (crop line at 1/3)   | stacked   |
    |                             | policy    |
    |                             | input 2x2 |
    +-----------------------------+-----------+
    | v / omega bars, lane offset, heading, step         |
    +-----------------------------------------------------+

Keys: q / Esc = stop the run (results are still written), p / space = pause.
"""
from typing import Optional, Tuple

import cv2
import numpy as np

CAM_W, CAM_H = 480, 360
TILE = 168  # each policy frame upscaled to this size
BAR_H = 70
FONT = cv2.FONT_HERSHEY_SIMPLEX


class LiveView:
    def __init__(self, title: str = "matrix_eval"):
        self.title = title
        self.ok = True
        try:
            cv2.namedWindow(self.title, cv2.WINDOW_NORMAL)
        except cv2.error as e:
            print(f"[show] cannot open a window ({e}); continuing without --show")
            self.ok = False

    def _policy_grid(self, obs: np.ndarray, grayscale: bool) -> np.ndarray:
        c = 1 if grayscale else 3
        frames = [obs[k * c:(k + 1) * c] for k in range(obs.shape[0] // c)]
        tiles = []
        for k, f in enumerate(frames):
            img = f[0] if grayscale else f.transpose(1, 2, 0)[:, :, ::-1]  # RGB -> BGR for display
            img = cv2.resize(img, (TILE, TILE), interpolation=cv2.INTER_NEAREST)
            if img.ndim == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            label = "t" if k == len(frames) - 1 else f"t-{len(frames) - 1 - k}"
            cv2.putText(img, label, (4, 16), FONT, 0.45, (0, 255, 255), 1)
            tiles.append(img)
        while len(tiles) < 4:
            tiles.append(np.zeros((TILE, TILE, 3), np.uint8))
        grid = np.vstack([np.hstack(tiles[:2]), np.hstack(tiles[2:4])])
        return cv2.resize(grid, (grid.shape[1] * CAM_H // grid.shape[0], CAM_H))

    @staticmethod
    def _bar(img, x, y, w, value, lo, hi, label, color):
        cv2.rectangle(img, (x, y), (x + w, y + 14), (70, 70, 70), -1)
        zero = x + int(w * (0 - lo) / (hi - lo))
        pos = x + int(w * (np.clip(value, lo, hi) - lo) / (hi - lo))
        cv2.rectangle(img, (min(zero, pos), y), (max(zero, pos), y + 14), color, -1)
        cv2.line(img, (zero, y - 2), (zero, y + 16), (200, 200, 200), 1)
        cv2.putText(img, f"{label} {value:+.2f}", (x + w + 8, y + 12), FONT, 0.45, (230, 230, 230), 1)

    def update(self, view_bgr: np.ndarray, obs: np.ndarray, grayscale: bool, action: np.ndarray,
               pwm: Tuple[float, float], lane: Optional[Tuple[float, float]], on_road: Optional[bool],
               header: str, on_pause=None) -> Optional[str]:
        """Draw one frame. Returns 'quit' when the user asked to stop."""
        if not self.ok:
            return None
        cam = cv2.resize(view_bgr, (CAM_W, CAM_H))
        cv2.line(cam, (0, CAM_H // 3), (CAM_W, CAM_H // 3), (0, 200, 255), 1)  # CropResizeWrapper cut
        grid = self._policy_grid(obs, grayscale)
        top = np.hstack([cam, grid])

        bar = np.full((BAR_H, top.shape[1], 3), 30, np.uint8)
        self._bar(bar, 10, 10, 140, float(action[0]), 0.0, 1.0, "v", (80, 200, 80))
        self._bar(bar, 10, 40, 140, float(action[1]), -1.0, 1.0, "omega", (80, 160, 240))
        self._bar(bar, 262, 10, 100, pwm[0], -1.0, 1.0, "pwm L", (180, 180, 80))
        self._bar(bar, 262, 40, 100, pwm[1], -1.0, 1.0, "pwm R", (180, 180, 80))
        if lane is not None:
            txt, col = f"lane d {lane[0]:+.3f} m   hdg {np.degrees(lane[1]):+5.1f} deg", (230, 230, 230)
        elif on_road is False:
            txt, col = "OFF ROAD", (60, 60, 255)
        else:
            txt, col = "no pose", (160, 160, 160)
        cv2.putText(bar, txt, (520, 24), FONT, 0.55, col, 1)
        cv2.putText(bar, header, (520, 54), FONT, 0.45, (200, 200, 200), 1)

        cv2.imshow(self.title, np.vstack([top, bar]))
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("p"), ord(" ")):
            if on_pause is not None:
                on_pause()  # stop the wheels while paused
            while True:
                key = cv2.waitKey(50) & 0xFF
                if key in (ord("p"), ord(" "), ord("q"), 27):
                    break
        if key in (ord("q"), 27):
            return "quit"
        return None

    def close(self):
        if self.ok:
            cv2.destroyAllWindows()
