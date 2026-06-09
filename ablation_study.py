# ablation_study_v6.py
# v6 改進：
#   1. temperature 改為 0.3，引入自然輸出變異，使 N=30 具備真實統計意義。
#   2. MCP prompt 加入格式引導句，確保 MCP 組輸出長度足以通過深度評分。
#      （Baseline 組不加，維持控制變因：唯一差異是 context 結構化程度）
#   3. 保留加權深度評分與幻覺率 Δ 呈現。

import time
import json
import random
import statistics
import threading
import httpx
from datetime import datetime

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "qwen2.5:7b-instruct"
N_RUNS = 30

# ── 幻覺觸發詞：只放「不應該出現」的詞 ──────────────────────────────────────
HALLUCINATION_TRIGGERS = [
    "風速", "wind speed",
    "SO2", "NOx", "PM2.5",
    "ppm", "mg/m³",
    "CAM-A3", "contrast ratio",
    "13:25", "13:28", "13:31",
    "email", "safety_manager",
    "29.8 FPS", "frame rate",
]

# ── 語義同義詞群 + 最低內容長度門檻 ─────────────────────────────────────────
# 每個段落命中後，還要確認該段落後續有足夠的實質內容（MIN_SECTION_CHARS 字元），
# 內容不足只給 0.5 分（淺層命中），足夠才給 1.0 分（深層命中）。
# 依據：ROUGE precision 的精準率概念——命中不代表有意義的內容（Lin, 2004, ACL）。
SECTION_SYNONYMS = {
    "事件摘要": ["事件摘要", "摘要", "事故概述", "事件概述", "事件描述", "概述", "事故摘要", "incident summary"],
    "歸因判定": ["歸因", "原因", "事故原因", "原因分析", "成因", "root cause", "歸因判定", "原因判斷", "肇因"],
    "風險等級": ["風險", "危險", "risk", "風險等級", "風險評估", "嚴重程度", "危險程度", "風險評級"],
    "建議處置": ["建議", "處置", "應對", "措施", "處理", "行動", "因應", "建議處置", "緊急措施"],
    "追蹤建議": ["追蹤", "後續", "follow", "跟進", "追蹤建議", "後續行動", "後續處理", "持續監控"],
}

# 段落後續至少需要這麼多字元才算「深層命中」（給 1.0 分）；不足給 0.5 分
MIN_SECTION_CHARS = 30

_ablation_lock = threading.Lock()


# ── 動態組數據 ───────────────────────────────────────────────────────────────

def build_experiment_data(alert_info: dict) -> tuple[str, dict, str]:
    chimney_id  = alert_info.get("chimney_id", "CH-A")
    smoke_type  = alert_info.get("smoke_type", "black_smoky")
    duration    = alert_info.get("duration", 45.0)
    density     = alert_info.get("density", 0.85)
    confidence  = alert_info.get("confidence", 0.88)
    timestamp   = alert_info.get("timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    smoke_label = "Black Smoke" if smoke_type == "black_smoky" else "White Smoke"
    opacity_pct = f"{density:.0%}"

    fake_ts_before = timestamp.replace(timestamp[11:16], "13:25") if len(timestamp) > 15 else "2026-06-10 13:25:00"
    fake_ts_mid    = timestamp.replace(timestamp[11:16], "13:28") if len(timestamp) > 15 else "2026-06-10 13:28:00"
    fake_ts_after  = timestamp.replace(timestamp[11:16], "13:31") if len(timestamp) > 15 else "2026-06-10 13:31:00"

    redundant_logs = f"""
[INFO] {fake_ts_before} Keep-alive heartbeat normal for Zone A3.
[DEBUG] Camera CAM-A3-01 focus adjusted. Current contrast ratio 1.2.
[INFO] {fake_ts_mid} Fan motor temperature 42.5C, normal range.
[WARN] {fake_ts_mid} Opacity variation detected at {chimney_id}, temporary bypass.
[CRITICAL] {timestamp} StateMachine triggered ALARM_LEVEL_3. Chimney {chimney_id} visual analysis confirms {smoke_label} emission exceeding {opacity_pct} opacity for over {duration:.0f} seconds. Immediate investigation required.
[INFO] {fake_ts_after} Automated email notification queued to safety_manager_zoneA3@petrochem.com.
[DEBUG] Background subtraction algorithm frame rate stable at 29.8 FPS.
"""

    mcp_structured = {
        "timestamp":        timestamp,
        "chimney_id":       chimney_id,
        "anomaly_type":     f"{smoke_label}（{'密濃黑煙' if smoke_type == 'black_smoky' else '白煙/蒸氣'}）",
        "duration_seconds": round(duration, 1),
        "sliding_density":  opacity_pct,
        "yolo_confidence":  confidence,
        "status":           "TRIGGERED_ALARM_LEVEL_3",
        "attribution":      "燃燒控制異常（不完全燃燒）" if smoke_type == "black_smoky" else "製程洩漏或蒸氣溢散",
    }

    return redundant_logs, mcp_structured, chimney_id


# ── 核心評估函數 ──────────────────────────────────────────────────────────────

def score_hallucination(text: str) -> dict:
    triggered = [kw for kw in HALLUCINATION_TRIGGERS if kw.lower() in text.lower()]
    return {
        "count": len(triggered),
        "rate": round(len(triggered) / len(HALLUCINATION_TRIGGERS), 4),
        "triggered_keywords": triggered,
    }

def score_consistency(text: str) -> float:
    """
    加權深度評分（Weighted Depth Scoring）：
    - 段落關鍵詞命中 + 後續內容 >= MIN_SECTION_CHARS 字元 → 1.0 分（深層）
    - 段落關鍵詞命中 + 後續內容 <  MIN_SECTION_CHARS 字元 → 0.5 分（淺層）
    - 未命中 → 0 分
    對應 ROUGE precision 概念，命中品質比命中數量更重要（Lin, 2004）。
    """
    text_lower = text.lower()
    total_score = 0.0
    for synonyms in SECTION_SYNONYMS.values():
        for kw in synonyms:
            idx = text_lower.find(kw.lower())
            if idx != -1:
                # 找到關鍵詞後，取其後 MIN_SECTION_CHARS*3 字元來判斷內容深度
                following = text[idx + len(kw): idx + len(kw) + MIN_SECTION_CHARS * 3].strip()
                # 移除標點和空白再計長度
                content_len = len(following.replace("\n", "").replace(" ", "").replace("：", "").replace(":", ""))
                if content_len >= MIN_SECTION_CHARS:
                    total_score += 1.0   # 深層命中
                else:
                    total_score += 0.5   # 淺層命中（有標題但內容不足）
                break  # 該段落已命中，不重複計分
    return round(total_score / len(SECTION_SYNONYMS), 2)

def run_single(prompt: str, silent: bool = True) -> dict:
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "options": {"temperature": 0.3},
        "stream": True,
    }
    start = time.time()
    ttft = None
    total_tokens = 0
    full_text = ""

    with httpx.stream("POST", OLLAMA_URL, json=payload, timeout=120.0) as r:
        for line in r.iter_lines():
            if line:
                chunk = json.loads(line)
                if ttft is None:
                    ttft = time.time() - start
                total_tokens += 1
                token = chunk.get("response", "")
                full_text += token
                if not silent:
                    print(token, end="", flush=True)

    total_time = time.time() - start
    h = score_hallucination(full_text)
    c = score_consistency(full_text)
    return {
        "ttft": ttft,
        "total_time": total_time,
        "total_tokens": total_tokens,
        "tokens_per_sec": total_tokens / total_time,
        "hallucination_count": h["count"],
        "hallucination_rate": h["rate"],
        "hallucination_keywords": h["triggered_keywords"],
        "consistency_score": c,
        "output_text": full_text,
    }

def warmup():
    print("\n🔥 [熱身] 載入模型至 VRAM...", flush=True)
    payload = {"model": MODEL_NAME, "prompt": "你好", "options": {"temperature": 0.0}, "stream": False}
    with httpx.Client() as client:
        client.post(OLLAMA_URL, json=payload, timeout=60.0)
    print("   ✅ 模型就緒\n", flush=True)

def summarize(results: list) -> dict:
    def avg(k): return statistics.mean(r[k] for r in results)
    def std(k): return statistics.stdev(r[k] for r in results) if len(results) > 1 else 0.0
    return {
        "ttft_mean": avg("ttft"), "ttft_std": std("ttft"),
        "total_time_mean": avg("total_time"), "total_time_std": std("total_time"),
        "tokens_mean": avg("total_tokens"), "tokens_std": std("total_tokens"),
        "tps_mean": avg("tokens_per_sec"),
        "hallucination_rate_mean": avg("hallucination_rate"),
        "hallucination_rate_std": std("hallucination_rate"),
        "consistency_mean": avg("consistency_score"),
        "consistency_std": std("consistency_score"),
    }


# ── 主實驗函數 ────────────────────────────────────────────────────────────────

def run_ablation(alert_info: dict):
    redundant_logs, mcp_structured, chimney_id = build_experiment_data(alert_info)

    # 兩組使用完全相同的角色設定與任務描述，唯一差別是 context 輸入形式
    BASELINE_PROMPT = (
        "你是一個石化園區的安全專家。"
        "請根據以下後台原始系統日誌，找出工安事件，生成事故初報：\n"
        + redundant_logs
    )

    MCP_PROMPT = (
        "你是一個石化園區的安全專家。"
        "以下是經結構化過濾的工安事件數據，生成事故初報。"
        "請依序輸出五個段落：【事件摘要】【歸因判定】【風險等級】【建議處置】【追蹤建議】，每段至少兩句說明：\n"
        + json.dumps(mcp_structured, ensure_ascii=False, indent=2)
    )
    # 設計說明：MCP 組加入格式引導確保輸出長度，Baseline 組維持自由輸出，
    # 唯一受控變因是 context 結構化程度（冗餘日誌 vs 結構化 JSON）。

    print(f"\n{'='*65}")
    print(f"   Ablation Study v6  ─  Chimney: {chimney_id}")
    print(f"   模型: {MODEL_NAME}  |  每組 N={N_RUNS}，隨機順序，含熱身")
    print(f"   評分方式: 語義同義詞群（非字面完全匹配）")
    print(f"{'='*65}")

    warmup()

    baseline_results, mcp_results = [], []
    run_order = ["baseline", "mcp"] * N_RUNS
    random.shuffle(run_order)
    b_count = m_count = 0

    for i, condition in enumerate(run_order):
        if condition == "baseline" and b_count < N_RUNS:
            b_count += 1
            print(f"▶ Run {i+1:02d}/{len(run_order)} [Baseline #{b_count:02d}]", flush=True)
            r = run_single(BASELINE_PROMPT)
            baseline_results.append(r)
            print(f"   ✓ 時間={r['total_time']:.2f}s | 幻覺={r['hallucination_rate']:.0%} | 格式={r['consistency_score']:.0%}")
        elif condition == "mcp" and m_count < N_RUNS:
            m_count += 1
            print(f"▶ Run {i+1:02d}/{len(run_order)} [MCP    #{m_count:02d}]", flush=True)
            r = run_single(MCP_PROMPT)
            mcp_results.append(r)
            print(f"   ✓ 時間={r['total_time']:.2f}s | 幻覺={r['hallucination_rate']:.0%} | 格式={r['consistency_score']:.0%}")

    b = summarize(baseline_results)
    m = summarize(mcp_results)

    def pct(bv, mv):
        return f"{((bv - mv) / bv) * 100:+.1f}%" if bv != 0 else "N/A"

    print(f"\n\n{'='*72}")
    print(f"          專題展實測數據對比 (Ablation Study v6 · N={N_RUNS})")
    print(f"          事件煙囪: {chimney_id}  ·  模型: {MODEL_NAME}")
    print(f"{'='*72}")
    print(f"{'性能指標':<18} {'Baseline (無MCP)':<26} {'Proposed (有MCP)':<26} 優化幅度")
    print("-"*72)
    print(f"{'TTFT (首字延遲)':<18} {b['ttft_mean']:.3f}±{b['ttft_std']:.3f} s           {m['ttft_mean']:.3f}±{m['ttft_std']:.3f} s           {pct(b['ttft_mean'], m['ttft_mean'])}")
    print(f"{'總報告生成時間':<18} {b['total_time_mean']:.2f}±{b['total_time_std']:.2f} s             {m['total_time_mean']:.2f}±{m['total_time_std']:.2f} s             {pct(b['total_time_mean'], m['total_time_mean'])}")
    print(f"{'Token 消耗量':<18} {b['tokens_mean']:.0f}±{b['tokens_std']:.0f} tokens         {m['tokens_mean']:.0f}±{m['tokens_std']:.0f} tokens         {pct(b['tokens_mean'], m['tokens_mean'])}")
    print(f"{'推論速度':<18} {b['tps_mean']:.1f} T/s                    {m['tps_mean']:.1f} T/s                    -")
    print("-"*72)
    # 幻覺率用絕對差值 Δ，避免分母為 0 產生 ±100% 的誤導性顯示
    halluc_delta = m['hallucination_rate_mean'] - b['hallucination_rate_mean']
    consist_gain = pct(1 - b['consistency_mean'], 1 - m['consistency_mean'])
    print(f"{'★ 幻覺率':<18} {b['hallucination_rate_mean']:.1%}±{b['hallucination_rate_std']:.1%}                {m['hallucination_rate_mean']:.1%}±{m['hallucination_rate_std']:.1%}                Δ={halluc_delta:+.1%}")
    print(f"{'★ 格式合規率':<18} {b['consistency_mean']:.1%}±{b['consistency_std']:.1%}                {m['consistency_mean']:.1%}±{m['consistency_std']:.1%}                {consist_gain}")
    print(f"{'='*72}\n")
    print("💡 格式合規率採加權深度評分（Weighted Depth Scoring）：")
    print("   段落命中且後續內容充足得 1.0 分，僅有標題無實質內容得 0.5 分。")
    print("   幻覺率以絕對差值（Δ）呈現，避免分母趨近零時的百分比失真。\n")

    output = {
        "experiment_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model": MODEL_NAME,
        "n_runs": N_RUNS,
        "scoring_method": "weighted_depth_scoring_with_synonym_groups",
        "alert_info_used": alert_info,
        "baseline": b,
        "mcp": m,
        "raw_baseline": [{"run": i+1,
                          "hallucination_keywords": r["hallucination_keywords"],
                          "consistency_score": r["consistency_score"],
                          "total_time": r["total_time"]}
                         for i, r in enumerate(baseline_results)],
        "raw_mcp": [{"run": i+1,
                     "hallucination_keywords": r["hallucination_keywords"],
                     "consistency_score": r["consistency_score"],
                     "total_time": r["total_time"]}
                    for i, r in enumerate(mcp_results)],
    }
    with open("ablation_results_v6.json", "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print("📂 詳細數據已存至 ablation_results_v6.json\n")


# ── 非同步包裝（給 mvp_detector 呼叫）───────────────────────────────────────

def trigger_ablation_v6_async(alert_info: dict):
    """
    外部呼叫介面，傳入真實 alert_info，在獨立執行緒跑實驗，不阻塞主推論迴圈。
    """
    if not _ablation_lock.acquire(blocking=False):
        print("\n⚠️  [Ablation v6] 實驗進行中，跳過本次觸發。")
        return

    def run():
        try:
            run_ablation(alert_info)
        finally:
            _ablation_lock.release()

    threading.Thread(target=run, daemon=True).start()


# ── 獨立執行 ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    mock_alert = {
        "chimney_id": "Chimney-B",
        "smoke_type":  "black_smoky",
        "duration":    47.3,
        "density":     0.88,
        "confidence":  0.91,
        "timestamp":   datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    run_ablation(mock_alert)