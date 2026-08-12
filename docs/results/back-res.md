# Stock Alpha v2 최신 ML 실행 결과

- 실행일: `2026-08-12`
- Snapshot: `research_provisional_20160104_20260227_cost_master_v2_r1`
- Artifact: `lambdarank_v2_20260812_replay_guard_1trial`
- 실행 모드: `research`
- 실행 명령: `uv run python -m src.stocks.cli.train --artifact-id lambdarank_v2_20260812_replay_guard_1trial --snapshot-id research_provisional_20160104_20260227_cost_master_v2_r1 --mode research --optuna-trials 1 --max-rss-mib 8000`
- 최종 결과: **NO_TRADE / 미승격 (replay 용량 초과로 fail-closed)**

## 실행·자원

| 항목 | 수치 |
|---|---:|
| Optuna terminal trials | 1 |
| Screened / pruned trials | 1 / 0 |
| Shortlisted trials | 1 |
| Economically eligible trials | 0 |
| Screen time | 3.43 s |
| Full refit time | 161.16 s |
| Economic replay time | 1.53 s |
| 총 wall time (`/usr/bin/time`) | 188.68 s |
| Peak RSS (workflow telemetry) | 6,256.6 MiB |
| Peak RSS (OS measurement) | 약 6,260 MiB |
| Replay cache bytes | 674,288,600 bytes |
| Replay decision preparation 예상치 | 14,301,256,442 bytes |
| RSS limit | 8,000 MiB |

## 백테스팅·승격 판정

- Best screen Rank-IC: `0.06937652`
- Replay는 `INNER_SELECTION_BASE_ONLY` 단계에서 decision preparation 용량 가드에 걸려 실행 전에 중단되었다.
- `capacity_failure_reason`: `replay_capacity_exceeded`
- 따라서 ledger metrics, filled-order 수익률, stress metrics는 생성되지 않았다.
- `promoted=false`, `no_trade=true`, `promotion_reasons=["no-champion-trial"]`

이번 실행은 OOM으로 종료되지 않았으며, 예상 materialization 크기(약 14.3GB)가 RSS 상한을 넘는 것을 감지해 fail-closed로 보호했다. 유효한 경제적 백테스트 성과가 생성되지 않았으므로 paper/live 승격은 금지된다.

## 산출물

- [Metrics](../../data/artifacts/stocks/lambdarank_v2_20260812_replay_guard_1trial/metrics.json)
- [Manifest](../../data/artifacts/stocks/lambdarank_v2_20260812_replay_guard_1trial/manifest.json)
