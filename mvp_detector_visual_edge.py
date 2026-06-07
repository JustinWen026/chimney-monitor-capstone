"""
mvp_detector_visual_edge_v3.py
NCKU CSIE — AI Petrochemical Chimney Safety Monitor
新增：
  - 多煙囪獨立狀態機（Per-Chimney FSM）：空間 IoU 關聯煙霧與煙囪 bounding box
  - 滑動視窗事件密度統計（Temporal Sliding Window Density）：
      過去 WINDOW_SEC 秒內偵測命中率須超過 DENSITY_THRESHOLD 才觸發告警
      參考：Park & Ko (2020) Temporal Fire-Tube, Sensors 20, 2202
"""

import os
import time
import cv2
import collections
from datetime import datetime
from ultralytics import YOLO

from discord_notifier import send_discord_alert_async

# ==============================================================================
# 1. 系統參數配置
# ==============================================================================
PROJECT_ROOT        = r"D:\Programming\NCKU_Project_V2"
VIDEO_PATH          = os.path.join(PROJECT_ROOT, "clean_video_perfect.mp4")
MODEL_PATH          = os.path.join(PROJECT_ROOT, r"runs\detect\ncku_smoke_8to2_v2\weights\best.pt")
SCREENSHOT_DIR      = os.path.join(PROJECT_ROOT, "alerts")

ALERT_THRESHOLD_SEC = 3.0    # 狀態機持續時間閾值（秒）
ALERT_COOLDOWN_SEC  = 60.0   # 每根煙囪的告警冷卻時間（秒）
MIN_CONFIDENCE      = 0.45   # 低於此信心度的偵測不納入狀態機（雜訊過濾）

# 滑動視窗密度統計參數（Temporal Sliding Window，參考 Temporal Fire-Tube 概念）
WINDOW_SEC          = 5.0    # 滑動視窗長度（秒）
DENSITY_THRESHOLD   = 0.60   # 視窗內偵測命中率須達此值才允許觸發

# 多煙囪空間關聯參數
CHIMNEY_SMOKE_IOU_THRESH = 0.05   # 煙霧 box 與煙囪 box 的最小垂直重疊比（較寬鬆，因煙會往上飄）

ATTRIBUTION_TEXT = {
    "black_smoky": "Root Cause: Combustion Ctrl Failure",
    "white_smoky": "Root Cause: Process Leakage / Steam Anomaly",
    "unknown":     "Root Cause: Pending Confirmation",
}

os.makedirs(SCREENSHOT_DIR, exist_ok=True)

# ==============================================================================
# 2. 工具函式
# ==============================================================================
DIVIDER = "=" * 110

def print_banner():
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(DIVIDER)
    print(f"  NCKU CSIE — AI Petrochemical Chimney Safety Monitor v3  |  啟動: {now}")
    print(f"  模型   : {MODEL_PATH}")
    print(f"  影像   : {VIDEO_PATH}")
    print(f"  閾值   : 持續>{ALERT_THRESHOLD_SEC}s  密度>{DENSITY_THRESHOLD:.0%}/{WINDOW_SEC}s  信心>{MIN_CONFIDENCE}")
    print(f"  冷卻   : {ALERT_COOLDOWN_SEC}s/煙囪  截圖: {SCREENSHOT_DIR}")
    print(DIVIDER + "\n")


def compute_iou(boxA, boxB):
    """計算兩個 bounding box 的 IoU（xywh 格式 → xyxy）"""
    ax1, ay1, ax2, ay2 = boxA
    bx1, by1, bx2, by2 = boxB
    ix1 = max(ax1, bx1); iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2); iy2 = min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    areaA = (ax2 - ax1) * (ay2 - ay1)
    areaB = (bx2 - bx1) * (by2 - by1)
    return inter / (areaA + areaB - inter + 1e-6)


def vertical_overlap_ratio(smoke_box, chimney_box):
    """
    計算煙霧 box 與煙囪 box 的水平軸重疊比（x 軸）。
    因為煙霧通常在煙囪正上方飄散，垂直方向可能無交集，
    所以改用水平重疊來判斷「此煙屬於哪根煙囪」。
    """
    sx1, _, sx2, _ = smoke_box
    cx1, _, cx2, _ = chimney_box
    overlap = max(0, min(sx2, cx2) - max(sx1, cx1))
    smoke_width = max(sx2 - sx1, 1)
    return overlap / smoke_width


def save_screenshot(frame, chimney_id: str, smoke_type: str) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_id = chimney_id.replace(" ", "_").replace("/", "-")
    filename = f"alert_{safe_id}_{smoke_type}_{ts}.jpg"
    path = os.path.join(SCREENSHOT_DIR, filename)
    cv2.imwrite(path, frame)
    return path


def print_timing(t_pre, t_inf, t_post, t_state, t_draw, fps):
    print(
        f"\r  [PERF] "
        f"讀取:{t_pre:5.1f}ms  推論:{t_inf:6.1f}ms  後處理:{t_post:5.1f}ms  "
        f"狀態機:{t_state:5.2f}ms  畫框:{t_draw:5.2f}ms  "
        f"| Total:{t_pre+t_inf+t_post+t_state+t_draw:6.1f}ms  FPS:{fps:5.1f}   ",
        end="", flush=True
    )


# ==============================================================================
# 3. 煙囪狀態機物件（Per-Chimney FSM）
# ==============================================================================
class ChimneyFSM:
    """
    每根煙囪獨立維護一個有限狀態機（FSM）。
    狀態：NORMAL → ACCUMULATING → TRIGGERED → NORMAL
    整合滑動視窗密度統計，過濾偶發誤報。
    """

    def __init__(self, chimney_id: str):
        self.chimney_id      = chimney_id
        self.state           = "NORMAL"
        self.smoke_start_t   = None
        self.active_type     = None
        self.duration        = 0.0
        self.last_alert_t    = 0.0
        self.last_conf       = 0.0

        # 滑動視窗：deque 存 (timestamp, hit: bool)
        self._window: collections.deque = collections.deque()

    # ── 滑動視窗密度計算 ─────────────────────────────────────────────────────
    def _push_window(self, t: float, hit: bool):
        self._window.append((t, hit))
        # 移除過期資料
        while self._window and (t - self._window[0][0]) > WINDOW_SEC:
            self._window.popleft()

    def _density(self) -> float:
        if not self._window:
            return 0.0
        hits = sum(1 for _, h in self._window if h)
        return hits / len(self._window)

    # ── 狀態機更新（每幀呼叫）──────────────────────────────────────────────
    def update(self, t: float, has_smoke: bool, smoke_type: str | None, confidence: float):
        """
        t          : 當前時間（time.time()）
        has_smoke  : 本幀是否偵測到關聯煙霧
        smoke_type : 煙霧類型標籤（或 None）
        confidence : YOLO 信心度
        返回: (triggered_new: bool, alert_info: dict | None)
              triggered_new=True 表示本幀剛觸發告警，alert_info 包含通報所需資料
        """
        self._push_window(t, has_smoke)
        density = self._density()

        triggered_new = False
        alert_info    = None

        if self.state == "NORMAL":
            self.duration = 0.0
            if has_smoke:
                self.state         = "ACCUMULATING"
                self.smoke_start_t = t
                self.active_type   = smoke_type
                self.last_conf     = confidence
                print(f"\n[{datetime.now().strftime('%H:%M:%S')}] 🚨 [{self.chimney_id}] NORMAL ➔ ACCUMULATING  |  煙霧: {smoke_type}  信心: {confidence:.2f}")

        elif self.state == "ACCUMULATING":
            if has_smoke:
                self.duration  = t - self.smoke_start_t
                self.last_conf = max(self.last_conf, confidence)
                print(
                    f"\r[{datetime.now().strftime('%H:%M:%S')}] [{self.chimney_id}] ⏳ 蓄能: {self.duration:.2f}s/{ALERT_THRESHOLD_SEC}s  "
                    f"密度: {density:.1%}  信心: {confidence:.2f}   ",
                    end="", flush=True
                )

                # 雙重門檻：時間 AND 密度
                if self.duration >= ALERT_THRESHOLD_SEC and density >= DENSITY_THRESHOLD:
                    self.state = "TRIGGERED"
                    readable_t = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    print(f"\n[{readable_t}] 🔥 [{self.chimney_id}] ➔ TRIGGERED  "
                          f"時間:{self.duration:.2f}s  密度:{density:.1%}  信心:{self.last_conf:.2f}")

                    if (t - self.last_alert_t) > ALERT_COOLDOWN_SEC:
                        self.last_alert_t = t
                        triggered_new = True
                        alert_info = {
                            "timestamp":   readable_t,
                            "duration":    round(self.duration, 1),
                            "smoke_type":  self.active_type or "unknown",
                            "chimney_id":  self.chimney_id,
                            "density":     density,
                            "confidence":  round(self.last_conf, 2),
                        }
                    else:
                        remaining = ALERT_COOLDOWN_SEC - (t - self.last_alert_t)
                        print(f"[{datetime.now().strftime('%H:%M:%S')}] 🛡️  [{self.chimney_id}] 冷卻中，剩餘 {remaining:.0f}s，跳過 Discord。")
            else:
                print(f"\n[{datetime.now().strftime('%H:%M:%S')}] 🟢 [{self.chimney_id}] 煙霧中斷 → NORMAL  蓄能: {self.duration:.2f}s（未達閾值）")
                self._reset()

        elif self.state == "TRIGGERED":
            if has_smoke:
                self.duration = t - self.smoke_start_t
                print(
                    f"\r[{datetime.now().strftime('%H:%M:%S')}] [{self.chimney_id}] 🔴 警報持續！累計: {self.duration:.2f}s  "
                    f"密度: {density:.1%}   ",
                    end="", flush=True
                )
            else:
                print(f"\n[{datetime.now().strftime('%H:%M:%S')}] 🟢 [{self.chimney_id}] 事件解除 → NORMAL  最終持續: {self.duration:.2f}s")
                self._reset()

        return triggered_new, alert_info

    def _reset(self):
        self.state         = "NORMAL"
        self.smoke_start_t = None
        self.active_type   = None
        self.duration      = 0.0
        self.last_conf     = 0.0


# ==============================================================================
# 4. OpenCV UI 繪製
# ==============================================================================
def draw_ui(frame, chimney_fsms: dict, frame_counter: int, fps: float):
    """
    左上角顯示各煙囪狀態（緊湊版），TRIGGERED 時加紅框閃爍與歸因文字。
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    thin = 1
    y_base = 20

    # 全域 FPS + 時間戳（右上角）
    ts_text = datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
    cv2.putText(frame, ts_text, (frame.shape[1] - 280, 18),
                font, 0.38, (160, 160, 160), thin, cv2.LINE_AA)
    cv2.putText(frame, f"FPS: {fps:.1f}", (frame.shape[1] - 90, 36),
                font, 0.38, (200, 200, 200), thin, cv2.LINE_AA)

    any_triggered = False

    for idx, (cid, fsm) in enumerate(chimney_fsms.items()):
        y = y_base + idx * 62

        if fsm.state == "NORMAL":
            color = (0, 210, 0)
            prog  = 0.0
        elif fsm.state == "ACCUMULATING":
            color = (0, 165, 255)
            prog  = min(fsm.duration / ALERT_THRESHOLD_SEC, 1.0)
        else:
            color = (0, 0, 210)
            prog  = 1.0
            any_triggered = True

        # 狀態文字
        label = f"{cid}: {fsm.state}"
        cv2.putText(frame, label, (10, y + 12), font, 0.44, color, thin, cv2.LINE_AA)

        # 進度條
        bar_x, bar_y, bar_w, bar_h = 10, y + 18, 170, 6
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (180, 180, 180), 1)
        fw = int(bar_w * prog)
        if fw > 0:
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fw, bar_y + bar_h), color, -1)

        # 詳細資訊
        density_val = fsm._density()
        detail = f"T:{fsm.duration:.1f}s  D:{density_val:.0%}  Conf:{fsm.last_conf:.2f}"
        cv2.putText(frame, detail, (10, y + 36), font, 0.35, (190, 190, 190), thin, cv2.LINE_AA)

        # 歸因（TRIGGERED 才顯示）
        if fsm.state == "TRIGGERED":
            attr = ATTRIBUTION_TEXT.get(fsm.active_type or "unknown", ATTRIBUTION_TEXT["unknown"])
            cv2.putText(frame, attr, (10, y + 52), font, 0.33, (0, 100, 255), thin, cv2.LINE_AA)

    # 全畫面紅框閃爍（任一煙囪 TRIGGERED）
    if any_triggered and (frame_counter // 15) % 2 == 0:
        h, w = frame.shape[:2]
        cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 0, 210), 6)

    return frame


# ==============================================================================
# 5. 空間關聯：將煙霧 box 歸屬到最近的煙囪
# ==============================================================================
def associate_smoke_to_chimney(smoke_boxes, chimney_boxes):
    """
    smoke_boxes  : list of (xyxy, smoke_type, confidence)
    chimney_boxes: list of (xyxy, chimney_id)
    回傳: dict { chimney_id: list of (smoke_type, confidence) }
    """
    result = {cid: [] for _, cid in chimney_boxes}
    result["Unassigned"] = []

    for s_box, s_type, s_conf in smoke_boxes:
        best_cid   = None
        best_score = CHIMNEY_SMOKE_IOU_THRESH  # 最低門檻

        for c_box, c_id in chimney_boxes:
            score = vertical_overlap_ratio(s_box, c_box)
            if score > best_score:
                best_score = score
                best_cid   = c_id

        target = best_cid if best_cid else "Unassigned"
        if target not in result:
            result[target] = []
        result[target].append((s_type, s_conf))

    return result


# ==============================================================================
# 6. 主迴圈
# ==============================================================================
def main():
    print_banner()
    print(f"[{datetime.now().strftime('%H:%M:%S')}] [INIT] 初始化 YOLOv11 推論引擎...")
    model = YOLO(MODEL_PATH)

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print("❌ 無法開啟影片！")
        return

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    wait_ms   = int(1000 / video_fps) if video_fps > 0 else 30
    print(f"[{datetime.now().strftime('%H:%M:%S')}] [INIT] 影片 FPS={video_fps:.1f}，每幀等待={wait_ms}ms")
    print(f"[{datetime.now().strftime('%H:%M:%S')}] [INIT] 等待首幀偵測煙囪佈局後自動建立各煙囪 FSM...\n")

    # 煙囪 FSM 字典（首幀偵測後動態建立）
    chimney_fsms: dict[str, ChimneyFSM] = {}

    frame_counter = 0
    display_fps   = 0.0

    print(DIVIDER)
    print("  🚀 推論迴圈啟動 — 按下 [Q] 結束")
    print(DIVIDER + "\n")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_counter += 1
        current_time = time.time()

        # ── 推論 ──────────────────────────────────────────────────────────────
        results         = model(frame, device=0, verbose=False)
        annotated_frame = results[0].plot()

        t_pre  = results[0].speed.get('preprocess',  0.0)
        t_inf  = results[0].speed.get('inference',   0.0)
        t_post = results[0].speed.get('postprocess', 0.0)

        if frame_counter % 10 == 0 or frame_counter == 1:
            total_pipe  = t_pre + t_inf + t_post
            display_fps = 1000.0 / total_pipe if total_pipe > 0 else 0.0

        # ── 解析偵測結果 ──────────────────────────────────────────────────────
        smoke_boxes   = []   # (xyxy_tuple, smoke_type, confidence)
        chimney_boxes = []   # (xyxy_tuple, chimney_id_str)

        for i, box in enumerate(results[0].boxes):
            cls_name = model.names[int(box.cls[0])]
            conf     = float(box.conf[0])
            xyxy     = tuple(map(int, box.xyxy[0].tolist()))  # (x1,y1,x2,y2)

            if cls_name in ['white_smoky', 'black_smoky']:
                if conf >= MIN_CONFIDENCE:
                    smoke_boxes.append((xyxy, cls_name, conf))

            elif cls_name == 'chimney':
                # 以煙囪中心 x 座標編號，方便識別
                cx = (xyxy[0] + xyxy[2]) // 2
                cid = f"Chimney-{chr(65 + len(chimney_boxes))}"  # A, B, C...
                chimney_boxes.append((xyxy, cid))

        # 建立尚未見過的煙囪 FSM
        for _, cid in chimney_boxes:
            if cid not in chimney_fsms:
                chimney_fsms[cid] = ChimneyFSM(cid)
                print(f"[{datetime.now().strftime('%H:%M:%S')}] [FSM] 新增煙囪狀態機: {cid}")

        # ── 狀態機更新 ────────────────────────────────────────────────────────
        t_sm_start = time.perf_counter()

        # 空間關聯：把每個煙霧 box 歸屬到對應煙囪
        associations = associate_smoke_to_chimney(smoke_boxes, chimney_boxes)

        for cid, fsm in chimney_fsms.items():
            hits = associations.get(cid, [])
            if hits:
                # 取最高信心度的那筆
                best_type, best_conf = max(hits, key=lambda x: x[1])
                triggered, alert = fsm.update(current_time, True, best_type, best_conf)
            else:
                triggered, alert = fsm.update(current_time, False, None, 0.0)

            if triggered and alert:
                # 截圖
                shot_path = save_screenshot(annotated_frame, alert["chimney_id"], alert["smoke_type"])
                print(f"[{datetime.now().strftime('%H:%M:%S')}] 📸 事件截圖: {shot_path}")

                # Discord 告警
                send_discord_alert_async(
                    timestamp     = alert["timestamp"],
                    duration      = alert["duration"],
                    smoke_type    = alert["smoke_type"],
                    chimney_id    = alert["chimney_id"],
                    density       = alert["density"],
                    confidence    = alert["confidence"],
                    screenshot_path = shot_path,
                )

        t_sm_end   = time.perf_counter()
        t_state_ms = (t_sm_end - t_sm_start) * 1000.0

        # ── UI 繪製 ───────────────────────────────────────────────────────────
        t_draw_start    = time.perf_counter()
        annotated_frame = draw_ui(annotated_frame, chimney_fsms, frame_counter, display_fps)
        t_draw_ms       = (time.perf_counter() - t_draw_start) * 1000.0

        # ── 詳細耗時（每 10 幀）──────────────────────────────────────────────
        if frame_counter % 10 == 0:
            print_timing(t_pre, t_inf, t_post, t_state_ms, t_draw_ms, display_fps)

        # ── 顯示 ──────────────────────────────────────────────────────────────
        cv2.imshow("NCKU Petrochemical Chimney — AI Safety Monitor v3", annotated_frame)
        if cv2.waitKey(wait_ms) & 0xFF == ord('q'):
            print(f"\n\n[{datetime.now().strftime('%H:%M:%S')}] [EXIT] 使用者手動結束。")
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"[{datetime.now().strftime('%H:%M:%S')}] [EXIT] 資源釋放完畢。\n")

    # 結束摘要
    print(DIVIDER)
    print("  執行摘要")
    print(DIVIDER)
    for cid, fsm in chimney_fsms.items():
        print(f"  {cid}: 最終狀態={fsm.state}  最終持續={fsm.duration:.2f}s")
    print(DIVIDER + "\n")


if __name__ == "__main__":
    main()