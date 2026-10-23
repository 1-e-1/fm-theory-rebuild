"""Read-only training inspection plus atomic factual progress persistence."""
from collections import Counter
import csv
from datetime import datetime
import json
from pathlib import Path
from zoneinfo import ZoneInfo
from experiment.scope import load_manifest, active_attempt_path

ROOT = Path(__file__).resolve().parents[1]


def inspect():
    manifest = load_manifest()
    counts = dict(Counter(r["status"] for r in manifest["runs"]))
    running = []
    for run in manifest["runs"]:
        if run["status"] != "RUNNING":
            continue
        attempt = active_attempt_path(run)
        row = {}
        if attempt and (attempt / "training.csv").exists():
            with (attempt / "training.csv").open(encoding="utf-8", newline="") as f:
                for item in csv.DictReader(f):
                    if item.get("step", "").isdigit():
                        row = item
        running.append({"run_id": run["run_id"], "step": int(row.get("step", 0)),
                        "seconds": round(float(row.get("elapsed_seconds", 0)), 1),
                        "phase": "sampling/evaluation/checks" if attempt and (attempt / "checkpoint_final.pt").exists() else "training",
                        "attempt": str(attempt)})
    value = {"updated_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
             "completed": counts.get("COMPLETED", 0), "denominator": 50,
             "statuses": counts, "running": running}
    temporary = ROOT / "PROGRESS.md.tmp"
    lines = ["# Phase 1 实际执行进度", "", f"更新时间：{value['updated_at']}", "",
             f"正式完成：**{value['completed']}/50**。清单分母固定；测试、pilot和采样检查不计正式配置。", "",
             "当前状态：" + json.dumps(counts, ensure_ascii=False), "",
             "SEQ_IID共25项按seed成组推进；ROBOT_8DOF共25项因缺用户实际资产路径/正确几何/FK及条件专家数据BLOCKED。",
             "具体缺失见ROBOT_RESOURCE_AUDIT.json。DEFERRED_BY_USER独立列出。", "",
             "模型原版已保存；模型源码未修改。23项模型correctness通过，独立pilot完成，共同12000更新正式协议已锁定。",
             "不以达到预算宣称收敛。实际每项状态以manifest及齐全artifact门禁为准。", ""]
    for item in running:
        lines.append(f"- {item['run_id']}: {item['step']}/12000, {item['phase']}, elapsed={item['seconds']}s, {item['attempt']}")
    lines += ["", "断点每100更新保存optimizer/RNG；恢复前保留未提交日志，不重置正式预算。", "",
              "```powershell", "python -m experiment.monitor", "python -m experiment.scope --verify",
              "python -m experiment.runner resume", "python -m experiment.verify",
              "python -m experiment.reporting --package", "```", "",
              "完成本阶段后停止，不运行其余数据、控制组或消融。"]
    temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    temporary.replace(ROOT / "PROGRESS.md")
    (ROOT / "LIVE_PROGRESS.json").write_text(json.dumps(value, indent=2), encoding="utf-8")
    print(json.dumps(value, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    inspect()
