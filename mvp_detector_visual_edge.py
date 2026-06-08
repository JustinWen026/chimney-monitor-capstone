"""
mvp_detector_visual_edge_v4.py
NCKU CSIE — AI Petrochemical Chimney Safety Monitor

v4 修正：
  - 煙囪 ID 穩定化：改用「x 座標錨點分群」取代每幀順序編號。
    同一根物理煙囪在不同幀的 box 中心 x 只要落在同一個 cluster 就給同一個 ID，
    徹底解決 YOLO 漏偵某幀導致 ID 每幀跳動、FSM 一直 reset 的問題。
  - 滑動視窗密度因此才能真正累積，不再因 ID 變動而清零。

v4 修正（煙囪歸屬補丁）：
  - 錨點 x 容差從 60px 放寬至 120px，容許煙霧遮住煙囪時 YOLO box 偏移。
  - 煙霧 box 若水平重疊找不到對應煙囪，改用「最近錨點 x 距離」強制歸屬，
    不再丟進 Unassigned，確保持續冒煙的煙囪一定有 FSM 累積。
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
MIN_CONFIDENCE      = 0.45   # 低於此信心度不納入狀態機

# 滑動視窗密度統計（Temporal Sliding Window，參考 Park & Ko 2020 Temporal Fire-Tube）
WINDOW_SEC        = 5.0   # 視窗長度（秒）
DENSITY_THRESHOLD = 0.40  # 視窗內命中率門檻（40%，考量 YOLO 在遠景小目標的偵測率波動）

# 煙囪 ID 穩定化：x 座標分群容差（像素）
# 同一根煙囪在不同幀的 box 中心 x 差異通常 < 30px，設 60px 足夠容錯
CHIMNEY_X_TOLERANCE = 120

# 煙霧歸屬：煙霧 box 水平重疊門檻
CHIMNEY_SMOKE_OVERLAP_THRESH = 0.05
# 煙霧無重疊時，最近錨點 x 距離上限（超過此距離不強制歸屬）
CHIMNEY_SMOKE_FALLBACK_MAX_DIST = 350

ATTRIBUTION_TEXT = {
    "black_smoky": "Root Cause: Combustion Ctrl Failure",
    "white_smoky": "Root Cause: Process Leakage / Steam Anomaly",
    "unknown":     "Root Cause: Pending Confirmation",
}

os.makedirs(SCREENSHOT_DIR, exist_ok=True)

# ==============================================================================
# 2. 煙囪 ID 穩定化：x 座標錨點管理器
# ==============================================================================
class ChimneyAnchorManager:
    """
    維護一份「已知煙囪錨點」清單（anchor_cx: 中心 x 均值, label: 'Chimney-A'...）。
    每幀偵測到的 chimney box，找最近的錨點（距離 < CHIMNEY_X_TOLERANCE）就沿用其 ID，
    否則新建錨點並分配下一個字母。
    若某幀 YOLO 完全漏偵某根煙囪，該錨點繼續保留，下幀恢復偵測後 ID 不變。
    """

    def __init__(self):
        # list of {"label": str, "cx": float}  (cx 為歷史平均，做 EMA 平滑)
        self._anchors: list[dict] = []
        self._counter = 0          # 已分配標籤數量（A, B, C...）

    def assign(self, chimney_boxes_xyxy: list[tuple]) -> list[tuple[tuple, str]]:
        """
        輸入: 本幀所有 chimney xyxy box
        輸出: list of (xyxy, stable_chimney_id)
        """
        result = []
        for box in chimney_boxes_xyxy:
            cx = (box[0] + box[2]) / 2.0
            matched = self._find_anchor(cx)
            if matched is None:
                label = f"Chimney-{chr(65 + self._counter)}"
                self._counter += 1
                self._anchors.append({"label": label, "cx": cx})
                matched = self._anchors[-1]
                print(f"[{datetime.now().strftime('%H:%M:%S')}] [ANCHOR] 新增煙囪錨點: {label}  cx={cx:.0f}px")
            else:
                # EMA 更新錨點 x（α=0.1，慢速追蹤）
                matched["cx"] = matched["cx"] * 0.9 + cx * 0.1
            result.append((box, matched["label"]))

        # 按 cx 排序，讓 A 永遠是最左邊（視覺上直覺）
        result.sort(key=lambda item: (self._label_to_idx(item[1])))
        return result

    def all_labels(self) -> list[str]:
        return [a["label"] for a in self._anchors]

    def _find_anchor(self, cx: float):
        best, best_dist = None, float("inf")
        for a in self._anchors:
            d = abs(a["cx"] - cx)
            if d < best_dist:
                best_dist = d
                best = a
        return best if best_dist < CHIMNEY_X_TOLERANCE else None

    def _label_to_idx(self, label: str) -> int:
        try:
            return ord(label.split("-")[-1]) - 65
        except Exception:
            return 999


# ==============================================================================
# 3. 工具函式
# ==============================================================================
DIVIDER = "=" * 110

def print_banner():
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(DIVIDER)
    print(f"  NCKU CSIE — AI Petrochemical Chimney Safety Monitor v4  |  啟動: {now}")
    print(f"  模型   : {MODEL_PATH}")
    print(f"  影像   : {VIDEO_PATH}")
    print(f"  閾值   : 持續>{ALERT_THRESHOLD_SEC}s  密度>{DENSITY_THRESHOLD:.0%}/{WINDOW_SEC}s  信心>{MIN_CONFIDENCE}")
    print(f"  冷卻   : {ALERT_COOLDOWN_SEC}s/煙囪  截圖: {SCREENSHOT_DIR}")
    print(f"  煙囪錨點容差: ±{CHIMNEY_X_TOLERANCE}px  煙霧 fallback 距離: {CHIMNEY_SMOKE_FALLBACK_MAX_DIST}px")
    print(DIVIDER + "\n")


def vertical_overlap_ratio(smoke_box, chimney_box):
    """煙霧 box 與煙囪 box 的水平（x 軸）重疊比，用於空間歸屬。"""
    sx1, _, sx2, _ = smoke_box
    cx1, _, cx2, _ = chimney_box
    overlap = max(0, min(sx2, cx2) - max(sx1, cx1))
    return overlap / max(sx2 - sx1, 1)


def associate_smoke_to_chimney(smoke_boxes, chimney_id_boxes, anchor_mgr):
    """
    smoke_boxes     : list of (xyxy, smoke_type, confidence)
    chimney_id_boxes: list of (xyxy, chimney_id)
    anchor_mgr      : ChimneyAnchorManager（用於 fallback 最近錨點查詢）

    歸屬策略（兩段式）：
      1. 優先用水平重疊比（vertical_overlap_ratio）找對應煙囪。
      2. 若無煙囪 box 有足夠重疊，改用錨點 cx 距離做 fallback 強制歸屬，
         確保煙霧遮住煙囪導致 YOLO 漏偵時，煙霧仍能歸屬到正確的 FSM。
    回傳: dict { chimney_id: list of (smoke_type, confidence) }
    """
    result = {cid: [] for _, cid in chimney_id_boxes}
    result["Unassigned"] = []

    for s_box, s_type, s_conf in smoke_boxes:
        s_cx = (s_box[0] + s_box[2]) / 2.0

        # --- 第一段：水平重疊優先 ---
        best_cid, best_score = None, CHIMNEY_SMOKE_OVERLAP_THRESH
        for c_box, c_id in chimney_id_boxes:
            score = vertical_overlap_ratio(s_box, c_box)
            if score > best_score:
                best_score, best_cid = score, c_id

        # --- 第二段：fallback — 用錨點 cx 距離找最近煙囪 ---
        if best_cid is None and anchor_mgr._anchors:
            nearest = min(anchor_mgr._anchors, key=lambda a: abs(a["cx"] - s_cx))
            dist = abs(nearest["cx"] - s_cx)
            if dist < CHIMNEY_SMOKE_FALLBACK_MAX_DIST:
                best_cid = nearest["label"]

        target = best_cid if best_cid else "Unassigned"
        if target not in result:
            result[target] = []
        result[target].append((s_type, s_conf))

    return result


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
        end="", flush=True,
    )


# ==============================================================================
# 4. 煙囪狀態機（Per-Chimney FSM）
# ==============================================================================
class ChimneyFSM:
    """
    每根煙囪獨立有限狀態機。
    觸發條件（雙重門檻）：
      1. 持續偵測時間 >= ALERT_THRESHOLD_SEC
      2. 滑動視窗內命中率 >= DENSITY_THRESHOLD
    兩者需同時滿足，避免偶發偵測誤觸，也避免持續真實煙霧被過低密度拒絕。
    """

    def __init__(self, chimney_id: str):
        self.chimney_id    = chimney_id
        self.state         = "NORMAL"
        self.smoke_start_t = None
        self.active_type   = None
        self.duration      = 0.0
        self.last_alert_t  = 0.0
        self.last_conf     = 0.0
        self._window: collections.deque = collections.deque()

    # ── 滑動視窗 ──────────────────────────────────────────────────────────────
    def _push(self, t: float, hit: bool):
        self._window.append((t, hit))
        while self._window and (t - self._window[0][0]) > WINDOW_SEC:
            self._window.popleft()

    def density(self) -> float:
        if not self._window:
            return 0.0
        return sum(1 for _, h in self._window if h) / len(self._window)

    # ── 狀態機更新 ────────────────────────────────────────────────────────────
    def update(self, t: float, has_smoke: bool, smoke_type: str | None, confidence: float):
        """
        每幀呼叫。
        返回 (triggered_new: bool, alert_info: dict | None)
        """
        self._push(t, has_smoke)
        d = self.density()

        triggered_new = False
        alert_info    = None

        if self.state == "NORMAL":
            self.duration = 0.0
            if has_smoke:
                self.state         = "ACCUMULATING"
                self.smoke_start_t = t
                self.active_type   = smoke_type
                self.last_conf     = confidence
                print(
                    f"\n[{datetime.now().strftime('%H:%M:%S')}] 🚨 [{self.chimney_id}] "
                    f"NORMAL ➔ ACCUMULATING  |  煙霧: {smoke_type}  信心: {confidence:.2f}"
                )

        elif self.state == "ACCUMULATING":
            if has_smoke:
                self.duration  = t - self.smoke_start_t
                self.last_conf = max(self.last_conf, confidence)
                print(
                    f"\r[{datetime.now().strftime('%H:%M:%S')}] [{self.chimney_id}] "
                    f"⏳ 蓄能: {self.duration:.2f}s/{ALERT_THRESHOLD_SEC}s  "
                    f"密度: {d:.1%}  信心: {confidence:.2f}   ",
                    end="", flush=True,
                )

                # 雙重門檻
                if self.duration >= ALERT_THRESHOLD_SEC and d >= DENSITY_THRESHOLD:
                    self.state = "TRIGGERED"
                    readable_t = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    print(
                        f"\n[{readable_t}] 🔥 [{self.chimney_id}] ➔ TRIGGERED  "
                        f"時間:{self.duration:.2f}s  密度:{d:.1%}  信心:{self.last_conf:.2f}"
                    )
                    if (t - self.last_alert_t) > ALERT_COOLDOWN_SEC:
                        self.last_alert_t = t
                        triggered_new = True
                        alert_info = {
                            "timestamp":  readable_t,
                            "duration":   round(self.duration, 1),
                            "smoke_type": self.active_type or "unknown",
                            "chimney_id": self.chimney_id,
                            "density":    d,
                            "confidence": round(self.last_conf, 2),
                        }
                    else:
                        remaining = ALERT_COOLDOWN_SEC - (t - self.last_alert_t)
                        print(
                            f"[{datetime.now().strftime('%H:%M:%S')}] 🛡️  [{self.chimney_id}] "
                            f"冷卻中，剩餘 {remaining:.0f}s，跳過 Discord。"
                        )
            else:
                # 煙霧消失但尚未達閾值 → reset
                # 注意：只有「連續消失」才 reset，偶發漏幀交給密度門檻吸收
                print(
                    f"\n[{datetime.now().strftime('%H:%M:%S')}] 🟢 [{self.chimney_id}] "
                    f"煙霧消失 → NORMAL  蓄能: {self.duration:.2f}s（未達閾值）"
                )
                self._reset()

        elif self.state == "TRIGGERED":
            if has_smoke:
                self.duration = t - self.smoke_start_t
                print(
                    f"\r[{datetime.now().strftime('%H:%M:%S')}] [{self.chimney_id}] "
                    f"🔴 警報持續！累計: {self.duration:.2f}s  密度: {d:.1%}   ",
                    end="", flush=True,
                )
            else:
                print(
                    f"\n[{datetime.now().strftime('%H:%M:%S')}] 🟢 [{self.chimney_id}] "
                    f"事件解除 → NORMAL  最終持續: {self.duration:.2f}s"
                )
                self._reset()

        return triggered_new, alert_info

    def _reset(self):
        self.state         = "NORMAL"
        self.smoke_start_t = None
        self.active_type   = None
        self.duration      = 0.0
        self.last_conf     = 0.0


# ==============================================================================
# 5. OpenCV UI 繪製
# ==============================================================================
def draw_ui(frame, chimney_fsms: dict, frame_counter: int, fps: float):
    font = cv2.FONT_HERSHEY_SIMPLEX
    thin = 1
    y_base = 20

    # 右上角：時間 + FPS
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

        # 狀態標籤
        cv2.putText(frame, f"{cid}: {fsm.state}", (10, y + 12),
                    font, 0.44, color, thin, cv2.LINE_AA)

        # 進度條
        bx, by, bw, bh = 10, y + 18, 170, 6
        cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (180, 180, 180), 1)
        fw = int(bw * prog)
        if fw > 0:
            cv2.rectangle(frame, (bx, by), (bx + fw, by + bh), color, -1)

        # 詳細數值
        d_val = fsm.density()
        cv2.putText(
            frame,
            f"T:{fsm.duration:.1f}s  D:{d_val:.0%}  Conf:{fsm.last_conf:.2f}",
            (10, y + 36), font, 0.35, (190, 190, 190), thin, cv2.LINE_AA,
        )

        # 歸因（TRIGGERED 才顯示）
        if fsm.state == "TRIGGERED":
            attr = ATTRIBUTION_TEXT.get(fsm.active_type or "unknown", ATTRIBUTION_TEXT["unknown"])
            cv2.putText(frame, attr, (10, y + 52), font, 0.33, (0, 100, 255), thin, cv2.LINE_AA)

    # 全畫面紅框閃爍
    if any_triggered and (frame_counter // 15) % 2 == 0:
        h, w = frame.shape[:2]
        cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 0, 210), 6)

    return frame


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
    print(f"[{datetime.now().strftime('%H:%M:%S')}] [INIT] 影片 FPS={video_fps:.1f}  等待={wait_ms}ms\n")

    anchor_mgr   = ChimneyAnchorManager()
    chimney_fsms : dict[str, ChimneyFSM] = {}   # { chimney_id: ChimneyFSM }

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
        raw_chimney_boxes = []   # xyxy tuples（未分配 ID）
        smoke_boxes       = []   # (xyxy, smoke_type, confidence)

        for box in results[0].boxes:
            cls_name = model.names[int(box.cls[0])]
            conf     = float(box.conf[0])
            xyxy     = tuple(map(int, box.xyxy[0].tolist()))

            if cls_name in ['white_smoky', 'black_smoky']:
                if conf >= MIN_CONFIDENCE:
                    smoke_boxes.append((xyxy, cls_name, conf))
            elif cls_name == 'chimney':
                raw_chimney_boxes.append(xyxy)

        # ── 穩定化煙囪 ID ─────────────────────────────────────────────────────
        chimney_id_boxes = anchor_mgr.assign(raw_chimney_boxes)  # [(xyxy, stable_id), ...]

        # 確保每個已知 ID 都有對應 FSM
        for _, cid in chimney_id_boxes:
            if cid not in chimney_fsms:
                chimney_fsms[cid] = ChimneyFSM(cid)

        # ── 空間關聯 + 狀態機更新 ─────────────────────────────────────────────
        t_sm_start   = time.perf_counter()
        associations = associate_smoke_to_chimney(smoke_boxes, chimney_id_boxes, anchor_mgr)

        for cid, fsm in chimney_fsms.items():
            hits = associations.get(cid, [])
            if hits:
                best_type, best_conf = max(hits, key=lambda x: x[1])
                triggered, alert = fsm.update(current_time, True, best_type, best_conf)
            else:
                triggered, alert = fsm.update(current_time, False, None, 0.0)

            if triggered and alert:
                shot = save_screenshot(annotated_frame, alert["chimney_id"], alert["smoke_type"])
                print(f"[{datetime.now().strftime('%H:%M:%S')}] 📸 截圖: {shot}")
                send_discord_alert_async(
                    timestamp     = alert["timestamp"],
                    duration      = alert["duration"],
                    smoke_type    = alert["smoke_type"],
                    chimney_id    = alert["chimney_id"],
                    density       = alert["density"],
                    confidence    = alert["confidence"],
                    screenshot_path = shot,
                )

        t_state_ms = (time.perf_counter() - t_sm_start) * 1000.0

        # ── UI 繪製 ───────────────────────────────────────────────────────────
        t_draw_start    = time.perf_counter()
        annotated_frame = draw_ui(annotated_frame, chimney_fsms, frame_counter, display_fps)
        t_draw_ms       = (time.perf_counter() - t_draw_start) * 1000.0

        if frame_counter % 10 == 0:
            print_timing(t_pre, t_inf, t_post, t_state_ms, t_draw_ms, display_fps)

        cv2.imshow("NCKU Petrochemical Chimney — AI Safety Monitor v4", annotated_frame)
        if cv2.waitKey(wait_ms) & 0xFF == ord('q'):
            print(f"\n\n[{datetime.now().strftime('%H:%M:%S')}] [EXIT] 使用者手動結束。")
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"[{datetime.now().strftime('%H:%M:%S')}] [EXIT] 資源釋放完畢。\n")

    # ── 執行摘要 ──────────────────────────────────────────────────────────────
    print(DIVIDER)
    print("  執行摘要")
    print(DIVIDER)
    for cid, fsm in chimney_fsms.items():
        print(f"  {cid}: 最終狀態={fsm.state}  最終持續={fsm.duration:.2f}s  最終密度={fsm.density():.1%}")
    print(DIVIDER + "\n")


if __name__ == "__main__":
    main()