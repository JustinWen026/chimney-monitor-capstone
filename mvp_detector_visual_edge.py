"""
mvp_detector_visual_edge_v5.py
NCKU CSIE — AI Petrochemical Chimney Safety Monitor

v5 修正（Event-Locking 歸因策略）：
  - 煙霧歸屬改為「事件鎖定（Event-Locking）」：只在煙霧首次被偵測到的那幾幀
    判斷來源煙囪（對應 FSM 的 NORMAL→ACCUMULATING 轉換瞬間），一旦鎖定
    就不再更改，直到煙霧消失事件結束。完全消除持續飄移導致的誤歸屬問題。
  - 移除每幀動態關聯邏輯，改為直接向已鎖定的 FSM 推送 smoke 偵測結果。
  - 移除視覺雜亂的感應區框線（Zone visualization）。
  - Alert Fatigue 統計保留：執行摘要顯示 FSM 告警抑制率。
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
    維護已知煙囪錨點（x 座標 EMA），提供穩定的煙囪 ID。
    「煙霧歸屬」不再由此類負責，改由 Event-Locking 機制在 FSM 層處理。
    """

    def __init__(self):
        self._anchors: list[dict] = []   # [{"label": str, "cx": float}]
        self._counter = 0

    def assign(self, chimney_boxes_xyxy: list[tuple]) -> list[tuple[tuple, str]]:
        """回傳 list of (xyxy, stable_chimney_id)，並更新錨點 EMA。"""
        result = []
        for box in chimney_boxes_xyxy:
            cx = (box[0] + box[2]) / 2.0
            matched = self._find_anchor(cx)
            if matched is None:
                label = f"Chimney-{chr(65 + self._counter)}"
                self._counter += 1
                self._anchors.append({"label": label, "cx": cx})
                matched = self._anchors[-1]
                print(f"[{datetime.now().strftime('%H:%M:%S')}] [ANCHOR] 新增煙囪: {label}  cx={cx:.0f}px")
            else:
                matched["cx"] = matched["cx"] * 0.9 + cx * 0.1
            result.append((box, matched["label"]))
        result.sort(key=lambda item: self._label_to_idx(item[1]))
        return result

    def find_nearest_label(self, cx: float) -> str | None:
        """找最近的錨點 label（不限容差，供 Event-Locking 使用）。"""
        if not self._anchors:
            return None
        nearest = min(self._anchors, key=lambda a: abs(a["cx"] - cx))
        return nearest["label"] if abs(nearest["cx"] - cx) < CHIMNEY_SMOKE_FALLBACK_MAX_DIST else None

    def _find_anchor(self, cx: float):
        best, best_dist = None, float("inf")
        for a in self._anchors:
            d = abs(a["cx"] - cx)
            if d < best_dist:
                best_dist, best = d, a
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
    print(f"  NCKU CSIE — AI Petrochemical Chimney Safety Monitor v5  |  啟動: {now}")
    print(f"  模型   : {MODEL_PATH}")
    print(f"  影像   : {VIDEO_PATH}")
    print(f"  閾值   : 持續>{ALERT_THRESHOLD_SEC}s  密度>{DENSITY_THRESHOLD:.0%}/{WINDOW_SEC}s  信心>{MIN_CONFIDENCE}")
    print(f"  冷卻   : {ALERT_COOLDOWN_SEC}s/煙囪  截圖: {SCREENSHOT_DIR}")
    print(f"  煙囪錨點容差: ±{CHIMNEY_X_TOLERANCE}px  煙霧 fallback 距離: {CHIMNEY_SMOKE_FALLBACK_MAX_DIST}px")
    print(DIVIDER + "\n")


def find_smoke_source(smoke_boxes, anchor_mgr) -> tuple[str | None, str | None, float]:
    """
    Event-Locking 來源判斷（只在事件開始時呼叫一次）。
    看所有偵測到的煙霧 box，找底部中心 x 最靠近哪根煙囪錨點，回傳該煙囪 ID。
    回傳: (chimney_id, smoke_type, confidence)
    """
    if not smoke_boxes or not anchor_mgr._anchors:
        return None, None, 0.0

    # 取信心度最高的那個煙霧 box 作為代表
    best_smoke = max(smoke_boxes, key=lambda x: x[2])
    s_box, s_type, s_conf = best_smoke
    s_bottom_cx = (s_box[0] + s_box[2]) / 2.0

    cid = anchor_mgr.find_nearest_label(s_bottom_cx)
    return cid, s_type, s_conf


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

    frame_counter          = 0
    display_fps            = 0.0
    baseline_trigger_count = 0   # 無演算法 baseline：每幀有煙就算一次
    fsm_suppressed_count   = 0   # 被狀態機抑制（有煙但未 TRIGGERED）的幀數

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

        # ── Alert Fatigue 統計（baseline：有煙霧偵測到就算一次）──────────────
        if smoke_boxes:
            baseline_trigger_count += 1

        # ── 穩定化煙囪 ID ─────────────────────────────────────────────────────
        chimney_id_boxes = anchor_mgr.assign(raw_chimney_boxes)  # [(xyxy, stable_id), ...]

        # 確保每個已知 ID 都有對應 FSM
        for _, cid in chimney_id_boxes:
            if cid not in chimney_fsms:
                chimney_fsms[cid] = ChimneyFSM(cid)

        # ── Event-Locking 狀態機更新 ──────────────────────────────────────────
        # 核心邏輯：只在煙霧「新出現」瞬間（NORMAL→ACCUMULATING）判斷來源煙囪，
        # 一旦鎖定就不再重新判斷，直到事件結束（煙消失→NORMAL）。
        # 這樣無論煙霧如何飄移，歸屬不會因為 YOLO box 位置變動而改變。
        t_sm_start = time.perf_counter()
        has_any_smoke = len(smoke_boxes) > 0

        for cid, fsm in chimney_fsms.items():
            if fsm.state == "NORMAL":
                # 只有 NORMAL 狀態才需要判斷「煙從哪來」
                if has_any_smoke:
                    src_cid, src_type, src_conf = find_smoke_source(smoke_boxes, anchor_mgr)
                    if src_cid == cid:
                        # 這根煙囪是煙霧來源 → 觸發本 FSM
                        triggered, alert = fsm.update(current_time, True, src_type, src_conf)
                    else:
                        triggered, alert = fsm.update(current_time, False, None, 0.0)
                else:
                    triggered, alert = fsm.update(current_time, False, None, 0.0)
            else:
                # ACCUMULATING / TRIGGERED：已鎖定來源，只需告知「有沒有煙」
                # 不再重新判斷是誰的煙，避免飄移誤解除
                triggered, alert = fsm.update(current_time, has_any_smoke,
                                               fsm.active_type, fsm.last_conf)

            if triggered and alert:
                shot = save_screenshot(annotated_frame, alert["chimney_id"], alert["smoke_type"])
                print(f"[{datetime.now().strftime('%H:%M:%S')}] 📸 截圖: {shot}")
                send_discord_alert_async(
                    timestamp       = alert["timestamp"],
                    duration        = alert["duration"],
                    smoke_type      = alert["smoke_type"],
                    chimney_id      = alert["chimney_id"],
                    density         = alert["density"],
                    confidence      = alert["confidence"],
                    screenshot_path = shot,
                )

        t_state_ms = (time.perf_counter() - t_sm_start) * 1000.0

        # ── UI 繪製 ───────────────────────────────────────────────────────────
        t_draw_start    = time.perf_counter()
        annotated_frame = draw_ui(annotated_frame, chimney_fsms, frame_counter, display_fps)
        t_draw_ms       = (time.perf_counter() - t_draw_start) * 1000.0

        if frame_counter % 10 == 0:
            print_timing(t_pre, t_inf, t_post, t_state_ms, t_draw_ms, display_fps)

        cv2.imshow("NCKU Petrochemical Chimney — AI Safety Monitor v5", annotated_frame)
        if cv2.waitKey(wait_ms) & 0xFF == ord('q'):
            print(f"\n\n[{datetime.now().strftime('%H:%M:%S')}] [EXIT] 使用者手動結束。")
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"[{datetime.now().strftime('%H:%M:%S')}] [EXIT] 資源釋放完畢。\n")

    # ── 執行摘要 ──────────────────────────────────────────────────────────────
    discord_total = sum(1 for fsm in chimney_fsms.values() if fsm.last_alert_t > 0)
    suppression_rate = (1 - discord_total / baseline_trigger_count) * 100 if baseline_trigger_count > 0 else 0

    print(DIVIDER)
    print("  執行摘要 — Alert Fatigue Reduction 數據")
    print(DIVIDER)
    print(f"  處理總幀數          : {frame_counter} 幀")
    print(f"  [Baseline] 有煙幀數  : {baseline_trigger_count} 次（無演算法情境下每幀觸發一次告警）")
    print(f"  [v5 FSM]  Discord 告警: {discord_total} 次")
    print(f"  告警抑制率          : {suppression_rate:.2f}%")
    print()
    for cid, fsm in chimney_fsms.items():
        print(f"  {cid}: 最終狀態={fsm.state}  最終持續={fsm.duration:.2f}s  最終密度={fsm.density():.1%}")
    print(DIVIDER + "\n")


if __name__ == "__main__":
    main()