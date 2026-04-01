import numpy as np
import cv2
import torch
from torchvision import transforms
from screen_grab.grab import ScreenGrab
from player_api.player import Player
from player_api.label_cnn import LabelCNN
from health_api.health import HealthAPI

PLAYER_ONE_ID = 0
PLAYER_TWO_ID = 1

CONFIDENCE_THRESHOLD = 0.85

PATCH_SIZE = 64
STRIDE = 32


class PlayerDetector:
    def __init__(self, monitor: int,
                 p1_model_path:  str = "player_api/models/p1_cnn.pth",
                 cpu_model_path: str = "player_api/models/cpu_cnn.pth"):

        self.player1 = Player(player_id=PLAYER_ONE_ID)
        self.player2 = Player(player_id=PLAYER_TWO_ID)

        self.health_api = HealthAPI(monitor=monitor)
        self.screen     = ScreenGrab(monitor=monitor)

        self.p1_model  = self._load_model(p1_model_path)
        self.cpu_model = self._load_model(cpu_model_path)

        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])

    @staticmethod
    def _load_model(path: str) -> LabelCNN:
        model = LabelCNN()
        model.load_state_dict(torch.load(path, map_location="cpu"))
        model.eval()
        return model

    def _scan_frame(self, frame_bgr: np.ndarray, model: LabelCNN) -> tuple | None:
        h, w = frame_bgr.shape[:2]
        y_start = int(h * 0.15)
        y_end   = int(h * 0.70)
        cropped = frame_bgr[y_start:y_end, :]
        ch, cw  = cropped.shape[:2]
        half    = PATCH_SIZE // 2
        best_score = 0.0
        best_pos   = None
        patches, positions = [], []

        for y in range(half, ch - half, STRIDE):
            for x in range(half, cw - half, STRIDE):
                patch     = cropped[y - half:y + half, x - half:x + half]
                patch_rgb = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
                patch_pil = transforms.functional.to_pil_image(patch_rgb)
                patches.append(self.transform(patch_pil))
                # Adjust y back to full frame coordinates
                positions.append((x, y + y_start))

        if not patches:
            return None

        batch = torch.stack(patches)
        with torch.no_grad():
            scores = model(batch).squeeze(1).numpy()

        for score, pos in zip(scores, positions):
            if score > best_score:
                best_score = score
                best_pos   = pos

        return best_pos if best_score >= CONFIDENCE_THRESHOLD else None

    def update(self, color_frame: np.ndarray | None = None):
        if color_frame is None:
            color_frame = self.screen.grab(greyscale=False)

        frame_bgr = cv2.cvtColor(color_frame, cv2.COLOR_BGRA2BGR)

        p1_pos  = self._scan_frame(frame_bgr, self.p1_model)
        cpu_pos = self._scan_frame(frame_bgr, self.cpu_model)

        if p1_pos is not None:
            self.player1.update_position(p1_pos)
        if cpu_pos is not None:
            self.player2.update_position(cpu_pos)

        # Update health
        self.health_api.process_health()
        health_vec = self.health_api.get_vector()
        if health_vec.size == 2:
            self.player1.update_health(float(health_vec[0]))
            self.player2.update_health(float(health_vec[1]))

    def get_players(self) -> tuple[Player, Player]:
        return self.player1, self.player2

    def debug_frame(self, color_frame: np.ndarray | None = None) -> np.ndarray:
        if color_frame is None:
            color_frame = self.screen.grab(greyscale=False)

        frame_bgr = cv2.cvtColor(color_frame, cv2.COLOR_BGRA2BGR)

        for model, label, color in [
            (self.p1_model,  "P1",  (0, 255, 0)),
            (self.cpu_model, "CPU", (0, 0, 255)),
        ]:
            pos = self._scan_frame(frame_bgr, model)
            print(f"{label} detected at: {pos}")
            if pos:
                cx, cy = pos
                cv2.circle(frame_bgr, (cx, cy), 6, color, -1)
                cv2.putText(frame_bgr, label, (cx + 8, cy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        return frame_bgr