# GitHub Actions Secrets 설정 가이드

이 워크플로우를 실행하려면 GitHub 저장소에 다음 Secrets를 설정해야 합니다.

## 설정 방법

1. GitHub 저장소로 이동
2. **Settings** → **Secrets and variables** → **Actions** 클릭
3. **New repository secret** 버튼 클릭
4. 아래 목록의 각 Secret를 추가

## 필요한 Secrets 목록

### Airtable
- `AIRTABLE_API_KEY`: Airtable API 키
- `AIRTABLE_BASE_ID`: Airtable Base ID (기본값: `appW0RZZJFdNc1D8H`)

### Supabase
- `SUPABASE_HOST`: Supabase 호스트 주소
- `SUPABASE_DB`: 데이터베이스 이름 (기본값: `postgres`)
- `SUPABASE_USER`: 데이터베이스 사용자명
- `SUPABASE_PASSWORD`: 데이터베이스 비밀번호
- `SUPABASE_PORT`: 포트 번호 (기본값: `6543`)

### Slack
- `SLACK_BOT_TOKEN`: Slack Bot Token
- `SLACK_CHANNEL`: 알림을 보낼 채널 (예: `#airtable-progresql-sync-log`)

### Naver Search Ads
- `NAVER_API_KEY`: 네이버 검색광고 API 키
- `NAVER_API_SECRET`: 네이버 검색광고 API Secret
- `NAVER_CUSTOMER_ID`: 네이버 검색광고 고객 ID

## 실행 스케줄

기본적으로 매일 오전 9시(UTC)에 실행됩니다. (한국시간 오후 6시)

스케줄을 변경하려면 `.github/workflows/sync-airtable.yml` 파일의 `cron` 값을 수정하세요.

## 수동 실행

GitHub Actions 탭에서 **Run workflow** 버튼을 클릭하여 수동으로 실행할 수 있습니다.

