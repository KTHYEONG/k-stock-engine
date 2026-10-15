# VPS Daily Automation Audit Runbook (K-Stock Engine)

> **Agent-Executable Operational Runbook:**  
> An AI auditor or site reliability engineer runs this runbook to verify that every automated routine of K-Stock Engine ran on schedule, **and that the data produced adheres strictly to financial invariants and Point-in-Time contracts**.

- **Target Host:** `or-vps` (SSH alias, tailnet) / local workstation
- **Systemd Units:** `kse-morning.service`, `kse-auction-open.service`, `kse-collect-eve.service`, `kse-daily-audit.service`, `kse-backup.service`, `kse-weekly-report.service`, `kse-alert@.service`
- **Output Language:** Report findings in Korean (한국어); keys, IDs, badges stay English.

---

## 1. Operating Rules (Non-Negotiable)

1. **Read-Only Inspection**: VPS 상에서 파일 수정, 임의 프로세스 재시작, `git pull`, `rclone` 쓰기를 절대 수행하지 않음. 진단 결과는 로컬 터미널 및 `scratch/`에만 기록.
2. **No Secrets Leakage**: `.env`, `.quant.env`, API 키, KIS 토큰, 계좌번호 등 민감 정보를 출력에 절대 표시하지 않음.
3. **UNKNOWN is Not PASS**: 로그 누락, 파일 미발견, 호스트 접속 불가 상태는 PASS가 아니며 반드시 원인과 함께 `UNKNOWN`으로 판정.
4. **Day-Kind Determination First**: 검사 전 당일이 `BUSINESS`(영업일), `WEEKEND`(주말), `HOLIDAY`(공휴일)인지 캘린더 캐시(`data/state/calendar/`)로 선제 확정.
5. **Fail-Closed Verification**: 시스템 상태 불일치, 쿼터 고갈, 원장 불일치가 감지되면 즉시 SEV1로 보고하고 다음 파이프라인 진행을 차단.

---

## 2. Severity Classification & Escalation Matrix

| Severity | 정의 및 기준 | 대표 발생 사례 | 대응 프로토콜 |
| :---: | :--- | :--- | :--- |
| **SEV1** | 데이터 영구 유실, 인과성 파괴, 원장 불일치 | 영업일 08:32 결정 봉인 실패, 장전 호가 누락, 디스크 잔여 < 5GB, NTP Drift > 2.0s | 사용자 즉시 경보(텔레그램), 신규 주문 즉시 동결 |
| **SEV2** | 보조 데이터 결측, 쿼터 한도 임박 | LS 수급 결측(KIS 백업 투입됨), DART 16,000건 쿼터 소진, 일일 감사 WARN 발생 | 당일 리포트 헤드라인 기재 및 야간 점검 조치 |
| **SEV3** | 경미한 일시적 지연 | 1회 재시도로 복구된 네트워크 일시 오류, 백업 대기 지연 | 리포트 본문 기록 |
| **INFO** | 정상 완료 지표 | 타이머 정시 실행, 레코드 건수, 피크 RSS 메모리 | 본문 상태 테이블 표시 |

---

## 3. Automation Cadence & Daily Audit Windows

| Audit Mode | 실행 시각 (KST) | 검사 목적 및 주안점 | 점검 대상 서비스 / 산출물 |
| :---: | :--- | :--- | :--- |
| **M0 (아침 봉인)** | **평일 08:33 ~ 08:38** | 당일 영업일 판별 및 결정 체인 봉인 확인 | `kse-morning.service`<br/>`data/state/decisions/YYYY-MM-DD.json` |
| **M1 (장전 경매)** | **평일 08:42 ~ 08:50** | 시초가 10호가/예상체결가 수집 확인 | `kse-auction-open.service`<br/>`data/bronze/kis/auction_open/dt=YYYY-MM-DD/` |
| **M2 (장개시 집행)** | **평일 09:05 ~ 09:30** | 시가 단일가 체결 및 정수 원장 기록 확인 | `data/state/shadow/official/fills.parquet`<br/>`data/state/shadow/official/ledger.parquet` |
| **M3 (저녁 수집)** | **평일 19:30 ~ 20:30** | 시세, 수급, DART, KIND 수집 및 Silver 빌드 | `kse-collect-eve.service`<br/>`ReceiptCatalog` SQLite3 WAL 동기화 |
| **M4 (야간 감사)** | **매일 21:45 ~ 22:30** | 12대 금융 무결성 및 시계열 연속성 전수 감사 | `kse-daily-audit.service`<br/>`data/ops/events/daily_audit_YYYY-MM-DD.json` |
| **M5 (원격 백업)** | **매일 02:40 ~ 03:30** | Google Drive 원격 대사 백업 완료 검증 | `kse-backup.service`<br/>`rclone` 원격 파일 크기 일치 확인 |

---

## 4. Phase-by-Phase Verification Procedures

### 4.1 M0: 아침 결정 봉인 검증 (`kse-morning`)
```bash
# 1. systemd 서비스 실행 성공 여부 확인
systemctl status kse-morning.service --no-pager

# 2. 당일 결정 파일 존재 및 무결성 확인
ls -la data/state/decisions/$(date +%Y-%m-%d).json
cat data/state/decisions/$(date +%Y-%m-%d).json | jq '{session: .session, candidate_count: (.targets | length)}'
```
* **PASS 조건**: `exit_code == 0`, 당일 결정 파일 존재, 종목 수가 설정된 상한(`n=20`)과 일치.

### 4.2 M1: 장전 시초가 경매 수집 검증 (`kse-auction-open`)
```bash
# 1. 경매 수집 서비스 상태 확인
systemctl status kse-auction-open.service --no-pager

# 2. 당일 수집된 10호가/예상체결 JSONL 확인
ls -la data/bronze/kis/auction_open/dt=$(date +%Y-%m-%d)/
wc -l data/bronze/kis/auction_open/dt=$(date +%Y-%m-%d)/*.jsonl
```
* **PASS 조건**: 예상 종목 수(기본 80종목)의 95% 이상에서 `antc_cnpr > 0` 확인.

### 4.3 M3: 저녁 수집 및 정규화 검증 (`kse-collect-eve`)
```bash
# 1. 저녁 수집 서비스 로그 확인
journalctl -u kse-collect-eve.service -n 50 --no-pager

# 2. SQLite 영수증 카탈로그 무결성 검사
sqlite3 data/state/receipts.db "PRAGMA integrity_check;"
sqlite3 data/state/receipts.db "SELECT count(*) FROM receipts WHERE collected_at >= datetime('now', '-1 day');"
```
* **PASS 조건**: SQLite `integrity_check` 결과 `ok`, 당일 영수증 신규 등록 건수 정상 증가.

### 4.4 M4: 일일 감사 결과 검증 (`kse-daily-audit`)
```bash
# 1. 감사 이벤트 결과 JSON 확인
cat data/ops/events/daily_audit_$(date +%Y-%m-%d).json | jq '.'
```
* **PASS 조건**: `outcome == "PASS"`, 결측치 0건, 인과성 역전 0건.

### 4.5 M5: 오프사이트 백업 검증 (`kse-backup`)
```bash
# 1. 백업 서비스 종료 상태 확인
systemctl status kse-backup.service --no-pager

# 2. 백업 로그에서 rclone 대사 결과 확인
journalctl -u kse-backup.service -n 30 | grep -E "Transferred|Checks|Elapsed"
```
* **PASS 조건**: 에러 0건, 로컬 아카이브와 원격 드라이브 간 해시/바이트 일치.

---

## 5. Structured Audit Output Template

감사 완료 후 아래 템플릿에 따라 한국어로 보고서를 작성합니다:

```markdown
### 🛡️ K-Stock Engine 일일 무인 운영 감사 보고서 (YYYY-MM-DD)

| Audit Phase | 상태 | 주요 지표 및 세부 내용 | 판정 사유 |
| :--- | :---: | :--- | :--- |
| **M0 Morning Seal** | `PASS` | 08:32:04 완료, 20종목 목표 봉인 (`decisions/YYYY-MM-DD.json`) | 정상 봉인 완료 |
| **M1 Auction Capture** | `PASS` | 08:39:31 완료, 80개 종목 10호가 수집 (`antc_cnpr` 100%) | 결측치 없음 |
| **M2 Intraday Fills** | `PASS` | 09:00:15 체결, 정수 원장 잔고 오차 0원, 미수금 0건 | 정수 원장 무결 |
| **M3 Evening Collect** | `PASS` | 18:30:12 기동, DART 1,240건 / KRX 2,750건 / LS 수급 적재 | SQLite WAL 정상 |
| **M4 Daily Audit** | `PASS` | 21:40:02 실행, 12개 Invariant Rule 전수 Green | 불변식 위반 0건 |
| **M5 Offsite Backup** | `PASS` | 02:30:15 백업 완료, Google Drive 원격 크기 100% 대사 | 백업 완료 |
```
