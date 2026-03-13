"""
monitoring/dashboard.py
Monitoring dashboard — terminal metrics view and HTML report generator.

Two modes:
  python monitoring/dashboard.py             → terminal live view
  python monitoring/dashboard.py --html      → generate HTML report → logs/dashboard.html
  python monitoring/dashboard.py --watch 30  → refresh every 30 seconds

Monitors:
  - Query volume and safety flag rate
  - Average faithfulness and latency trends
  - Feedback satisfaction rate
  - Corpus health (stale documents, chunk counts)
  - Human review queue size
"""

import json
import time
import argparse
from datetime import datetime
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).parent.parent))
from pipeline.audit_log import audit_logger, QUERY_LOG, FEEDBACK_LOG, FLAGGED_LOG


# ── Terminal dashboard ─────────────────────────────────────────────────────────

def render_terminal(stats: dict, health: dict):
    """Print a clean terminal metrics view."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    total = stats.get("total_queries", 0)

    def bar(val: float, width: int = 20) -> str:
        filled = int(val * width)
        return "█" * filled + "░" * (width - filled)

    print("\033[2J\033[H", end="")  # Clear screen
    print("═" * 60)
    print(f"  MEDQUERY MONITORING DASHBOARD   {now}")
    print("═" * 60)

    if total == 0:
        print("\n  No queries logged yet. Run the system and try again.\n")
        print("═" * 60)
        return

    faith = stats.get("avg_faithfulness", 0)
    flag_rate = stats.get("flag_rate", 0)
    latency = stats.get("avg_latency_ms", 0)

    print(f"\n  QUERY METRICS  (last {total} queries)")
    print(f"  {'Total queries':<26} {total}")
    print(f"  {'Safe queries':<26} {stats['safe_queries']} ({stats['safe_queries']/total:.0%})")
    print(f"  {'Flagged queries':<26} {stats['flagged_queries']} ({flag_rate:.1%})")
    print()
    print(f"  {'Avg faithfulness':<26} {faith:.2f}  {bar(faith)}")
    print(f"  {'Avg latency':<26} {latency}ms")

    breakdown = stats.get("safety_breakdown", {})
    if breakdown:
        print(f"\n  SAFETY BREAKDOWN")
        for cat, count in sorted(breakdown.items(), key=lambda x: -x[1]):
            pct = count / total
            print(f"  {'  '+cat:<26} {count:>4}  {bar(pct, 15)} {pct:.0%}")

    feedback = stats.get("feedback", {})
    fb_total = feedback.get("total", 0)
    if fb_total > 0:
        sat = feedback.get("satisfaction_rate", 0)
        print(f"\n  USER FEEDBACK")
        print(f"  {'Total feedback':<26} {fb_total}")
        print(f"  {'Satisfaction rate':<26} {sat:.0%}  {bar(sat)}")
        print(f"  {'  Positive':<26} {feedback.get('positive', 0)}")
        print(f"  {'  Negative':<26} {feedback.get('negative', 0)}")

    print(f"\n  CORPUS HEALTH")
    print(f"  {'Total documents':<26} {health.get('total_documents', '—')}")
    print(f"  {'Fresh documents':<26} {health.get('fresh_documents', '—')}")
    print(f"  {'Stale documents':<26} {health.get('stale_documents', '—')}")
    print(f"  {'Total chunks':<26} {health.get('total_chunks', '—')}")

    review_count = _count_lines(FLAGGED_LOG)
    print(f"\n  REVIEW QUEUE")
    print(f"  {'Items pending review':<26} {review_count}")

    print("\n" + "═" * 60)


# ── HTML report generator ──────────────────────────────────────────────────────

def generate_html_report(stats: dict, health: dict) -> str:
    """Generate a standalone HTML monitoring report."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    total = stats.get("total_queries", 0)

    def pct_bar(val: float, color: str = "#1d6b4e") -> str:
        w = round(val * 100)
        return (
            f'<div style="background:#e8f4ef;border-radius:4px;height:8px;width:100%;margin-top:4px">'
            f'<div style="background:{color};border-radius:4px;height:8px;width:{w}%"></div></div>'
        )

    def metric_card(label: str, value, sub: str = "", bar: str = "") -> str:
        return f"""
        <div style="background:#fdfcf8;border:1px solid #e0dbd0;border-radius:10px;
                    padding:1rem 1.2rem;min-width:160px;flex:1">
          <div style="font-size:.75rem;color:#8a8580;text-transform:uppercase;
                      letter-spacing:.06em;margin-bottom:.3rem">{label}</div>
          <div style="font-size:1.5rem;font-weight:500;color:#1a1814">{value}</div>
          {f'<div style="font-size:.75rem;color:#8a8580;margin-top:.2rem">{sub}</div>' if sub else ''}
          {bar}
        </div>"""

    faith = stats.get("avg_faithfulness", 0)
    flag_rate = stats.get("flag_rate", 0)
    latency = stats.get("avg_latency_ms", 0)
    feedback = stats.get("feedback", {})
    sat_rate = feedback.get("satisfaction_rate") or 0

    faith_color = "#1d6b4e" if faith >= 0.80 else "#c47a1a" if faith >= 0.60 else "#8b1a1a"
    flag_color  = "#1d6b4e" if flag_rate <= 0.1 else "#c47a1a" if flag_rate <= 0.2 else "#8b1a1a"

    # Safety breakdown rows
    breakdown = stats.get("safety_breakdown", {})
    breakdown_rows = "".join(
        f"<tr><td style='padding:.4rem .8rem'>{cat}</td>"
        f"<td style='padding:.4rem .8rem;text-align:right'>{count}</td>"
        f"<td style='padding:.4rem .8rem;text-align:right'>{count/(total or 1):.0%}</td></tr>"
        for cat, count in sorted(breakdown.items(), key=lambda x: -x[1])
    )

    # Recent flagged items
    flagged_items = audit_logger.get_review_queue(limit=10)
    flagged_rows = "".join(
        f"<tr>"
        f"<td style='padding:.35rem .8rem;font-size:.8rem;color:#8a8580'>{item.get('timestamp','')[:16]}</td>"
        f"<td style='padding:.35rem .8rem;font-size:.8rem'>{item.get('query','')[:60]}...</td>"
        f"<td style='padding:.35rem .8rem;font-size:.8rem'>{item.get('review_reason','')}</td>"
        f"<td style='padding:.35rem .8rem;font-size:.8rem'>{item.get('faithfulness',0):.2f}</td>"
        f"</tr>"
        for item in flagged_items
    ) or "<tr><td colspan='4' style='padding:.5rem;color:#8a8580;text-align:center'>No flagged items</td></tr>"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>MedQuery Monitoring — {now}</title>
<link href="https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500&display=swap" rel="stylesheet">
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: 'DM Sans', sans-serif; background: #f5f2eb;
          color: #1a1814; padding: 2rem; }}
  h1 {{ font-size: 1.4rem; font-weight: 500; margin-bottom: .3rem; }}
  h2 {{ font-size: 1rem; font-weight: 500; margin: 2rem 0 .8rem; color: #4a4740; }}
  .ts {{ font-size: .8rem; color: #8a8580; }}
  .cards {{ display: flex; gap: .8rem; flex-wrap: wrap; margin-bottom: 1.5rem; }}
  table {{ width: 100%; border-collapse: collapse; background: #fdfcf8;
           border: 1px solid #e0dbd0; border-radius: 10px; overflow: hidden; }}
  thead {{ background: #f5f2eb; }}
  th {{ padding: .5rem .8rem; font-size: .78rem; font-weight: 500;
        text-align: left; color: #4a4740; border-bottom: 1px solid #e0dbd0; }}
  td {{ border-bottom: 1px solid #f0ede6; }}
  tr:last-child td {{ border-bottom: none; }}
</style>
</head>
<body>
<div style="max-width:900px;margin:0 auto">
  <h1>MedQuery Monitoring Dashboard</h1>
  <p class="ts">Generated: {now} &nbsp;·&nbsp; Last {total} queries</p>

  <h2>Query metrics</h2>
  <div class="cards">
    {metric_card("Total queries", total)}
    {metric_card("Faithfulness", f"{faith:.0%}",
                 "avg NLI score", pct_bar(faith, faith_color))}
    {metric_card("Flag rate", f"{flag_rate:.1%}",
                 "queries flagged", pct_bar(1-flag_rate, flag_color))}
    {metric_card("Avg latency", f"{latency}ms")}
    {metric_card("Satisfaction", f"{sat_rate:.0%}" if feedback.get('total') else "—",
                 f"{feedback.get('total',0)} votes", pct_bar(sat_rate) if feedback.get('total') else "")}
  </div>

  <h2>Safety breakdown</h2>
  <table>
    <thead><tr><th>Category</th><th style="text-align:right">Count</th><th style="text-align:right">%</th></tr></thead>
    <tbody>{breakdown_rows}</tbody>
  </table>

  <h2>Corpus health</h2>
  <div class="cards">
    {metric_card("Documents", health.get('total_documents','—'))}
    {metric_card("Fresh", health.get('fresh_documents','—'), f"< {health.get('stale_threshold_days',30)}d old")}
    {metric_card("Stale", health.get('stale_documents','—'), "need refresh")}
    {metric_card("Chunks", health.get('total_chunks','—'))}
  </div>

  <h2>Review queue (last 10 flagged)</h2>
  <table>
    <thead><tr><th>Time</th><th>Query</th><th>Reason</th><th>Faithfulness</th></tr></thead>
    <tbody>{flagged_rows}</tbody>
  </table>
</div>
</body>
</html>"""
    return html


# ── Helpers ───────────────────────────────────────────────────────────────────

def _count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    with open(path) as f:
        return sum(1 for line in f if line.strip())


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MedQuery monitoring dashboard")
    parser.add_argument("--html",  action="store_true", help="Generate HTML report")
    parser.add_argument("--watch", type=int, metavar="SECS",
                        help="Refresh terminal view every N seconds")
    args = parser.parse_args()

    from monitoring.corpus_refresh import CorpusHealthChecker
    checker = CorpusHealthChecker()

    if args.html:
        stats  = audit_logger.get_stats()
        health = checker.check()
        html   = generate_html_report(stats, health)
        out = Path(__file__).parent.parent / "logs" / "dashboard.html"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(html)
        print(f"Dashboard saved → {out}")
    elif args.watch:
        print(f"Watching (refresh every {args.watch}s) — Ctrl+C to stop")
        while True:
            stats  = audit_logger.get_stats()
            health = checker.check()
            render_terminal(stats, health)
            time.sleep(args.watch)
    else:
        stats  = audit_logger.get_stats()
        health = checker.check()
        render_terminal(stats, health)
