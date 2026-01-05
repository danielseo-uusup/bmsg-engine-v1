import os
import requests
import psycopg2
import json
import time
import hmac
import hashlib
import base64
from datetime import datetime, date, timedelta
from psycopg2.extras import execute_batch
from concurrent.futures import ThreadPoolExecutor, as_completed

# =====================
# 환경 설정 (GitHub Actions용 - 환경변수 사용)
# =====================
class Config:
    # Airtable
    AIRTABLE_API_KEY = os.getenv("AIRTABLE_API_KEY")
    AIRTABLE_BASE_ID = os.getenv("AIRTABLE_BASE_ID", "appW0RZZJFdNc1D8H")
    
    # Supabase
    SUPABASE_HOST = os.getenv("SUPABASE_HOST")
    SUPABASE_DB = os.getenv("SUPABASE_DB", "postgres")
    SUPABASE_USER = os.getenv("SUPABASE_USER")
    SUPABASE_PASSWORD = os.getenv("SUPABASE_PASSWORD")
    SUPABASE_PORT = os.getenv("SUPABASE_PORT", "6543")
    
    # Slack
    SLACK_BOT_TOKEN = os.getenv("SLACK_BOT_TOKEN")
    SLACK_CHANNEL = os.getenv("SLACK_CHANNEL", "#airtable-progresql-sync-log")
    SLACK_USER_ID = os.getenv("SLACK_USER_ID")  # DM용 (선택사항)
    
    # Naver SA
    NAVER_API_KEY = os.getenv("NAVER_API_KEY")
    NAVER_API_SECRET = os.getenv("NAVER_API_SECRET")
    NAVER_CUSTOMER_ID = os.getenv("NAVER_CUSTOMER_ID")
    
    @classmethod
    def validate(cls):
        """필수 환경변수 검증"""
        required_vars = [
            "AIRTABLE_API_KEY", "SUPABASE_HOST", "SUPABASE_USER", 
            "SUPABASE_PASSWORD", "SLACK_BOT_TOKEN",
            "NAVER_API_KEY", "NAVER_API_SECRET", "NAVER_CUSTOMER_ID"
        ]
        missing = [var for var in required_vars if not getattr(cls, var)]
        if missing:
            raise ValueError(f"필수 환경변수가 설정되지 않았습니다: {', '.join(missing)}")

# =====================
# 유틸리티 함수
# =====================
class Utils:
    @staticmethod
    def convert_to_string(value):
        if isinstance(value, (dict, list)):
            return json.dumps(value, ensure_ascii=False, default=str)
        return value
    
    @staticmethod
    def send_slack_message(message, channel=None):
        try:
            url = "https://slack.com/api/chat.postMessage"
            headers = {
                "Authorization": f"Bearer {Config.SLACK_BOT_TOKEN}",
                "Content-Type": "application/json"
            }
            data = {
                "channel": channel or Config.SLACK_CHANNEL,
                "text": message
            }
            
            response = requests.post(url, headers=headers, json=data)
            if response.status_code == 200:
                result = response.json()
                if result.get("ok"):
                    print("✓ Slack 알림 전송 성공")
                else:
                    print(f"✗ Slack 알림 실패: {result.get('error')}")
            else:
                print(f"✗ Slack API 요청 실패: {response.status_code}")
        except Exception as e:
            print(f"✗ Slack 알림 전송 중 오류: {e}")

# =====================
# 동기화 작업 클래스
# =====================
class SyncJob:
    def __init__(self, name, table_id, db_table, sync_type="incremental", days=30):
        self.name = name
        self.table_id = table_id
        self.db_table = db_table
        self.sync_type = sync_type
        self.days = days
        self.insert_count = 0
        self.update_count = 0
        self.start_time = None
        self.end_time = None
    
    def get_date_filter(self):
        if self.sync_type == "full":
            return None
        
        cutoff_date = (datetime.now() - timedelta(days=self.days)).strftime("%Y-%m-%d")
        
        # 테이블별 날짜 필드명 매핑
        date_fields = {
            "airtable_records": ["생성일", "Last Modified"],
            "airtable_inventory": ["재고변동일시", "Last Modified"],
            "airtable_revenue": ["날짜", "Last Modified"],
            "airtable_pickup_dashboard": ["Last Modified"],
        }
        
        fields = date_fields.get(self.db_table, ["Last Modified"])
        
        if len(fields) == 1:
            return f"IS_AFTER({{{fields[0]}}}, '{cutoff_date}')"
        else:
            conditions = [f"IS_AFTER({{{f}}}, '{cutoff_date}')" for f in fields]
            return f"OR({', '.join(conditions)})"
    
    def fetch_records(self):
        headers = {"Authorization": f"Bearer {Config.AIRTABLE_API_KEY}"}
        url = f"https://api.airtable.com/v0/{Config.AIRTABLE_BASE_ID}/{self.table_id}"
        
        offset = None
        records = []
        
        while True:
            params = {}
            
            filter_formula = self.get_date_filter()
            if filter_formula:
                params["filterByFormula"] = filter_formula
            
            if offset:
                params["offset"] = offset
            
            r = requests.get(url, headers=headers, params=params, timeout=30)
            data = r.json()
            
            if "records" not in data:
                print(f"✗ 레코드 조회 실패: {data}")
                break
            
            batch = data.get("records", [])
            records.extend(batch)
            print(f"  조회: {len(batch)}건, 누적: {len(records)}건")
            
            offset = data.get("offset")
            if not offset:
                break
            
            time.sleep(0.2)  # Rate limit 방어
        
        return records
    
    def execute(self, conn, cur):
        self.start_time = time.time()
        
        print(f"\n{'='*60}")
        print(f"작업 시작: {self.name}")
        print(f"테이블: {self.db_table}")
        print(f"동기화 방식: {self.sync_type}")
        if self.sync_type == "incremental":
            print(f"동기화 범위: 최근 {self.days}일")
        print(f"{'='*60}")
        
        # 1. Airtable에서 레코드 조회
        records = self.fetch_records()
        print(f"✓ 총 {len(records)}건 조회 완료")
        
        if not records:
            print("✓ 동기화할 레코드 없음")
            self.end_time = time.time()
            return
        
        # 2. 기존 레코드 확인
        record_ids = [rec["id"] for rec in records]
        cur.execute(
            f"SELECT record_id FROM {self.db_table} WHERE record_id = ANY(%s)",
            (record_ids,)
        )
        existing_ids = set(row[0] for row in cur.fetchall())
        print(f"✓ 기존 레코드 {len(existing_ids)}건 확인")
        
        # 3. 데이터 준비 및 저장 (테이블별 로직)
        self._save_records(records, existing_ids, cur, conn)
        
        # 4. 삭제된 레코드 정리 (전체 레코드 비교)
        self._cleanup_deleted_records(conn, cur)
        
        self.end_time = time.time()
        elapsed = self.end_time - self.start_time
        
        print(f"\n✓ 작업 완료: {self.name}")
        print(f"  신규: {self.insert_count}건")
        print(f"  업데이트: {self.update_count}건")
        print(f"  총: {self.insert_count + self.update_count}건")
        print(f"  실행시간: {elapsed:.1f}초")
    
    def _cleanup_deleted_records(self, conn, cur):
        """Airtable에서 삭제된 레코드를 Supabase에서도 삭제"""
        try:
            # Airtable에서 전체 record_id 조회
            headers = {"Authorization": f"Bearer {Config.AIRTABLE_API_KEY}"}
            url = f"https://api.airtable.com/v0/{Config.AIRTABLE_BASE_ID}/{self.table_id}"
            
            airtable_ids = set()
            offset = None
            
            while True:
                params = {"pageSize": 100}
                if offset:
                    params["offset"] = offset
                
                r = requests.get(url, headers=headers, params=params, timeout=30)
                data = r.json()
                
                if "records" not in data:
                    break
                
                for rec in data.get("records", []):
                    airtable_ids.add(rec["id"])
                
                offset = data.get("offset")
                if not offset:
                    break
                
                time.sleep(0.1)  # Rate limit 방어
            
            # Supabase에서 전체 record_id 조회
            cur.execute(f"SELECT record_id FROM {self.db_table}")
            supabase_ids = set(row[0] for row in cur.fetchall())
            
            # Supabase에만 있는 레코드 (Airtable에서 삭제된 것)
            to_delete = supabase_ids - airtable_ids
            
            if to_delete:
                delete_count = len(to_delete)
                cur.execute(
                    f"DELETE FROM {self.db_table} WHERE record_id = ANY(%s)",
                    (list(to_delete),)
                )
                conn.commit()
                print(f"✓ 삭제된 레코드 정리: {delete_count}건 삭제")
            else:
                print(f"✓ 삭제된 레코드 없음")
                
        except Exception as e:
            print(f"⚠️ 삭제된 레코드 정리 중 오류 (무시): {str(e)}")
    
    def _save_records(self, records, existing_ids, cur, conn):
        # 이 메서드는 하위 클래스에서 구현
        raise NotImplementedError("하위 클래스에서 구현 필요")
    
    def get_summary(self):
        elapsed = (self.end_time - self.start_time) if self.end_time else 0
        return {
            "name": self.name,
            "insert": self.insert_count,
            "update": self.update_count,
            "total": self.insert_count + self.update_count,
            "elapsed": round(elapsed, 1)
        }

# =====================
# 개별 동기화 작업 구현
# =====================
class BmsgRecordsSync(SyncJob):
    def __init__(self):
        super().__init__(
            name="수거기록(bmsg_records)",
            table_id="tblT0tygjCwD4jpni",
            db_table="airtable_records",
            sync_type="incremental",
            days=100
        )
    
    def _save_records(self, records, existing_ids, cur, conn):
        batch_data = []
        
        for rec in records:
            fields = rec.get("fields", {})
            
            if rec["id"] in existing_ids:
                self.update_count += 1
            else:
                self.insert_count += 1
            
            batch_data.append({
                "record_id": rec["id"],
                "created_time": rec.get("createdTime"),
                "last_modified_time": Utils.convert_to_string(fields.get("Last Modified")),
                "gu_dong": Utils.convert_to_string(fields.get("구+동")),
                "maeip_total": Utils.convert_to_string(fields.get("매입액합계")),
                "visit_request_date": Utils.convert_to_string(fields.get("방문 희망 날짜")),
                "created_date": Utils.convert_to_string(fields.get("생성일")),
                "assigned_rider": Utils.convert_to_string(fields.get("수거기사배정(text)")),
                "pickup_number": Utils.convert_to_string(fields.get("수거번호")),
                "pickup_number_text": Utils.convert_to_string(fields.get("수거번호 text")),
                "pickup_datetime": Utils.convert_to_string(fields.get("수거일시")),
                "pickup_date_std": Utils.convert_to_string(fields.get("수거일자표준화(복사)")),
                "application_unique": Utils.convert_to_string(fields.get("신청_unique")),
                "real_weight_bag": Utils.convert_to_string(fields.get("실제무게_가방")),
                "real_weight_shoes": Utils.convert_to_string(fields.get("실제무게_신발")),
                "real_weight_clothes": Utils.convert_to_string(fields.get("실제무게_의류")),
                "real_weight_padding": Utils.convert_to_string(fields.get("실제무게_패딩")),
                "real_weight_total": Utils.convert_to_string(fields.get("실제무게합계")),
                "expected_weight": Utils.convert_to_string(fields.get("예상무게(kg)")),
                "reservation_date": Utils.convert_to_string(fields.get("예약신청일")),
                "awareness_path": Utils.convert_to_string(fields.get("인지경로(dropdown)")),
                "address": Utils.convert_to_string(fields.get("주소")),
                "status": Utils.convert_to_string(fields.get("처리여부")),
                "cancel_reason": Utils.convert_to_string(fields.get("취소사유")),
                "raw_json": json.dumps(rec, ensure_ascii=False, default=str)
            })
        
        if batch_data:
            execute_batch(cur, """
                INSERT INTO airtable_records(
                    record_id, created_time, last_modified_time, gu_dong, maeip_total,
                    visit_request_date, created_date, assigned_rider, pickup_number,
                    pickup_number_text, pickup_datetime, pickup_date_std, application_unique,
                    real_weight_bag, real_weight_shoes, real_weight_clothes, real_weight_padding,
                    real_weight_total, expected_weight, reservation_date, awareness_path,
                    address, status, cancel_reason, raw_json
                )
                VALUES (
                    %(record_id)s, %(created_time)s, %(last_modified_time)s, %(gu_dong)s,
                    %(maeip_total)s, %(visit_request_date)s, %(created_date)s, %(assigned_rider)s,
                    %(pickup_number)s, %(pickup_number_text)s, %(pickup_datetime)s,
                    %(pickup_date_std)s, %(application_unique)s, %(real_weight_bag)s,
                    %(real_weight_shoes)s, %(real_weight_clothes)s, %(real_weight_padding)s,
                    %(real_weight_total)s, %(expected_weight)s, %(reservation_date)s,
                    %(awareness_path)s, %(address)s, %(status)s, %(cancel_reason)s, %(raw_json)s
                )
                ON CONFLICT (record_id) DO UPDATE SET
                    last_modified_time = EXCLUDED.last_modified_time,
                    raw_json = EXCLUDED.raw_json,
                    updated_at = now(),
                    gu_dong = EXCLUDED.gu_dong,
                    maeip_total = EXCLUDED.maeip_total,
                    visit_request_date = EXCLUDED.visit_request_date,
                    created_date = EXCLUDED.created_date,
                    assigned_rider = EXCLUDED.assigned_rider,
                    pickup_number = EXCLUDED.pickup_number,
                    pickup_number_text = EXCLUDED.pickup_number_text,
                    pickup_datetime = EXCLUDED.pickup_datetime,
                    pickup_date_std = EXCLUDED.pickup_date_std,
                    application_unique = EXCLUDED.application_unique,
                    real_weight_bag = EXCLUDED.real_weight_bag,
                    real_weight_shoes = EXCLUDED.real_weight_shoes,
                    real_weight_clothes = EXCLUDED.real_weight_clothes,
                    real_weight_padding = EXCLUDED.real_weight_padding,
                    real_weight_total = EXCLUDED.real_weight_total,
                    expected_weight = EXCLUDED.expected_weight,
                    reservation_date = EXCLUDED.reservation_date,
                    awareness_path = EXCLUDED.awareness_path,
                    address = EXCLUDED.address,
                    status = EXCLUDED.status,
                    cancel_reason = EXCLUDED.cancel_reason
            """, batch_data, page_size=100)
            
            conn.commit()
            print(f"✓ 배치 저장 완료: {len(batch_data)}건")

class InventorySync(SyncJob):
    def __init__(self):
        super().__init__(
            name="재고(inventory)",
            table_id="tblVcKkL4Qj2I9QBy",
            db_table="airtable_inventory",
            sync_type="incremental",
            days=30
        )
    
    def _save_records(self, records, existing_ids, cur, conn):
        batch_data = []
        
        for rec in records:
            fields = rec.get("fields", {})
            
            if rec["id"] in existing_ids:
                self.update_count += 1
            else:
                self.insert_count += 1
            
            batch_data.append({
                "record_id": rec["id"],
                "created_time": rec.get("createdTime"),
                "last_modified_time": fields.get("Last Modified"),
                "batch_number": Utils.convert_to_string(fields.get("배치번호")),
                "batch_ts": Utils.convert_to_string(fields.get("Batch_TS")),
                "batch_type": Utils.convert_to_string(fields.get("배치유형")),
                "batch_number_text": Utils.convert_to_string(fields.get("배치번호_text")),
                "batch_number_change_datetime": Utils.convert_to_string(fields.get("배치번호변동일시")),
                "batchTS": Utils.convert_to_string(fields.get("배치TS")),
                "stock_change_datetime": Utils.convert_to_string(fields.get("재고변동일시")),
                "stock_change_date_parsed": Utils.convert_to_string(fields.get("재고변동날짜파싱")),
                "date_reference": Utils.convert_to_string(fields.get("날짜참조")),
                "sales_number": Utils.convert_to_string(fields.get("판매번호")),
                "customer_name": Utils.convert_to_string(fields.get("판매고객명")),
                "site": Utils.convert_to_string(fields.get("Site")),
                "stock_change": Utils.convert_to_string(fields.get("재고변동")),
                "stock_change_channel": Utils.convert_to_string(fields.get("재고변동-채널")),
                "rollup_manager": Utils.convert_to_string(fields.get("차란 담당자 Rollup (from 배치번호)")),
                "apartment": Utils.convert_to_string(fields.get("아파트 (from 배치번호)")),
                "sales_weight": Utils.convert_to_string(fields.get("판매무게")),
                "weight": Utils.convert_to_string(fields.get("무게")),
                "change_weight": Utils.convert_to_string(fields.get("변동무게")),
                "site_text": Utils.convert_to_string(fields.get("Site_Text")),
                "batch_weight": Utils.convert_to_string(fields.get("무게 (from 배치번호)")),
                "note": Utils.convert_to_string(fields.get("note")),
                "outbound_weight_check": Utils.convert_to_string(fields.get("출고무게 확인")),
                "inbound_weight_check": Utils.convert_to_string(fields.get("입고무게 확인")),
                "temp_value": Utils.convert_to_string(fields.get("temp")),
                "last_modified_by": Utils.convert_to_string(fields.get("Last Modified By")),
                "input_manager": Utils.convert_to_string(fields.get("입력담당")),
                "input_manager_text": Utils.convert_to_string(fields.get("입력담당_text")),
                "channel_2": Utils.convert_to_string(fields.get("채널-2 (from 판매번호)")),
                "batch_total_weight": Utils.convert_to_string(fields.get("배치 무게 total (from 배치번호)")),
                "last_modified_2": Utils.convert_to_string(fields.get("Last Modified 2")),
                "last_modified_by_2": Utils.convert_to_string(fields.get("Last Modified By 2")),
                "input_manager_2": Utils.convert_to_string(fields.get("입력담당 2")),
                "input_manager_text_2": Utils.convert_to_string(fields.get("입력담당_text 2")),
                "raw_json": json.dumps(rec, ensure_ascii=False, default=str)
            })
        
        if batch_data:
            execute_batch(cur, """
                INSERT INTO airtable_inventory(
                    record_id, created_time, last_modified_time, batch_number, batch_ts,
                    batch_type, batch_number_text, batch_number_change_datetime, batchTS,
                    stock_change_datetime, stock_change_date_parsed, date_reference,
                    sales_number, customer_name, site, stock_change, stock_change_channel,
                    rollup_manager, apartment, sales_weight, weight, change_weight,
                    site_text, batch_weight, note, outbound_weight_check, inbound_weight_check,
                    temp_value, last_modified_by, input_manager, input_manager_text,
                    channel_2, batch_total_weight, last_modified_2, last_modified_by_2,
                    input_manager_2, input_manager_text_2, raw_json
                )
                VALUES (
                    %(record_id)s, %(created_time)s, %(last_modified_time)s, %(batch_number)s,
                    %(batch_ts)s, %(batch_type)s, %(batch_number_text)s,
                    %(batch_number_change_datetime)s, %(batchTS)s, %(stock_change_datetime)s,
                    %(stock_change_date_parsed)s, %(date_reference)s, %(sales_number)s,
                    %(customer_name)s, %(site)s, %(stock_change)s, %(stock_change_channel)s,
                    %(rollup_manager)s, %(apartment)s, %(sales_weight)s, %(weight)s,
                    %(change_weight)s, %(site_text)s, %(batch_weight)s, %(note)s,
                    %(outbound_weight_check)s, %(inbound_weight_check)s, %(temp_value)s,
                    %(last_modified_by)s, %(input_manager)s, %(input_manager_text)s,
                    %(channel_2)s, %(batch_total_weight)s, %(last_modified_2)s,
                    %(last_modified_by_2)s, %(input_manager_2)s, %(input_manager_text_2)s,
                    %(raw_json)s
                )
                ON CONFLICT (record_id) DO UPDATE SET
                    last_modified_time = EXCLUDED.last_modified_time,
                    raw_json = EXCLUDED.raw_json,
                    updated_at = now(),
                    batch_number = EXCLUDED.batch_number,
                    batch_ts = EXCLUDED.batch_ts,
                    batch_type = EXCLUDED.batch_type,
                    batch_number_text = EXCLUDED.batch_number_text,
                    batch_number_change_datetime = EXCLUDED.batch_number_change_datetime,
                    batchTS = EXCLUDED.batchTS,
                    stock_change_datetime = EXCLUDED.stock_change_datetime,
                    stock_change_date_parsed = EXCLUDED.stock_change_date_parsed,
                    date_reference = EXCLUDED.date_reference,
                    sales_number = EXCLUDED.sales_number,
                    customer_name = EXCLUDED.customer_name,
                    site = EXCLUDED.site,
                    stock_change = EXCLUDED.stock_change,
                    stock_change_channel = EXCLUDED.stock_change_channel,
                    rollup_manager = EXCLUDED.rollup_manager,
                    apartment = EXCLUDED.apartment,
                    sales_weight = EXCLUDED.sales_weight,
                    weight = EXCLUDED.weight,
                    change_weight = EXCLUDED.change_weight,
                    site_text = EXCLUDED.site_text,
                    batch_weight = EXCLUDED.batch_weight,
                    note = EXCLUDED.note,
                    outbound_weight_check = EXCLUDED.outbound_weight_check,
                    inbound_weight_check = EXCLUDED.inbound_weight_check,
                    temp_value = EXCLUDED.temp_value,
                    last_modified_by = EXCLUDED.last_modified_by,
                    input_manager = EXCLUDED.input_manager,
                    input_manager_text = EXCLUDED.input_manager_text,
                    channel_2 = EXCLUDED.channel_2,
                    batch_total_weight = EXCLUDED.batch_total_weight,
                    last_modified_2 = EXCLUDED.last_modified_2,
                    last_modified_by_2 = EXCLUDED.last_modified_by_2,
                    input_manager_2 = EXCLUDED.input_manager_2,
                    input_manager_text_2 = EXCLUDED.input_manager_text_2
            """, batch_data, page_size=100)
            
            conn.commit()
            print(f"✓ 배치 저장 완료: {len(batch_data)}건")

class RevenueSync(SyncJob):
    def __init__(self):
        super().__init__(
            name="판매(revenue)",
            table_id="tblFBD10lHsDna9Ox",
            db_table="airtable_revenue",
            sync_type="incremental",
            days=30
        )

    def _save_records(self, records, existing_ids, cur, conn):
        batch_data = []

        for rec in records:
            fields = rec.get("fields", {})

            if rec["id"] in existing_ids:
                self.update_count += 1
            else:
                self.insert_count += 1

            batch_data.append({
                "record_id": rec["id"],
                "created_time": rec.get("createdTime"),
                "last_modified_time": Utils.convert_to_string(fields.get("Last Modified")),
                "date_value": Utils.convert_to_string(fields.get("날짜")),
                "customer_name": Utils.convert_to_string(fields.get("고객명")),
                "channel_1": Utils.convert_to_string(fields.get("채널-1")),
                "channel_2": Utils.convert_to_string(fields.get("채널-2")),
                "product": Utils.convert_to_string(fields.get("상품")),
                "payment_type": Utils.convert_to_string(fields.get("지불형태")),
                "revenue_migration": Utils.convert_to_string(fields.get("매출액(migration)")),
                "stock_change_weight": Utils.convert_to_string(fields.get("재고변동 무게")),
                "weight": Utils.convert_to_string(fields.get("무게")),
                "unit_price": Utils.convert_to_string(fields.get("단가")),
                "base_price": Utils.convert_to_string(fields.get("기준가격")),
                "discount_amount": Utils.convert_to_string(fields.get("할인금액")),
                "revenue": Utils.convert_to_string(fields.get("매출액")),
                "discount_rate": Utils.convert_to_string(fields.get("할인율")),
                "tax_invoice": Utils.convert_to_string(fields.get("세금계산서발행")),
                "revenue_vat_included": Utils.convert_to_string(fields.get("매출액(vat포함)")),
                "vat": Utils.convert_to_string(fields.get("vat")),
                "real_unit_price": Utils.convert_to_string(fields.get("실제단가")),
                "deposit_date": Utils.convert_to_string(fields.get("입금일자")),
                "deposit_status": Utils.convert_to_string(fields.get("입금여부")),
                "cash_receipt": Utils.convert_to_string(fields.get("현금영수증 발행")),
                "depositor_name": Utils.convert_to_string(fields.get("입금자명")),
                "batch": Utils.convert_to_string(fields.get("배치")),
                "outbound_location": Utils.convert_to_string(fields.get("출고위치")),
                "stock_change": Utils.convert_to_string(fields.get("재고변동")),
                "transaction_amount": Utils.convert_to_string(fields.get("거래금액 (from 고객명)")),
                "avg_weight_per_transaction": Utils.convert_to_string(fields.get("평균거래당무게 (from 고객명)")),
                "avg_price_per_weight": Utils.convert_to_string(fields.get("평균무게당단가 (from 고객명)")),
                "total_transaction_weight": Utils.convert_to_string(fields.get("총 거래무게")),
                "project": Utils.convert_to_string(fields.get("프로젝트")),
                "revenue_per_weight": Utils.convert_to_string(fields.get("매출액/무게")),
                "note": Utils.convert_to_string(fields.get("비고")),
                "stock_instagram": Utils.convert_to_string(fields.get("Stock_Instagram")),
                "stock_change_type": Utils.convert_to_string(fields.get("재고변동유형")),
                "unique_sale": Utils.convert_to_string(fields.get("unique_판매")),
                "sale_order": Utils.convert_to_string(fields.get("판매_order")),
                "last_modified_by": Utils.convert_to_string(fields.get("Last Modified By")),
                "customer_name_text": Utils.convert_to_string(fields.get("고객명_text")),
                "author": Utils.convert_to_string(fields.get("작성자")),
                "customer_value": Utils.convert_to_string(fields.get("고객")),
                "raw_json": json.dumps(rec, ensure_ascii=False, default=str)
            })

        if batch_data:
            execute_batch(cur, """
                INSERT INTO airtable_revenue(
                    record_id, created_time, last_modified_time, date_value,
                    customer_name, channel_1, channel_2, product, payment_type,
                    revenue_migration, stock_change_weight, weight, unit_price,
                    base_price, discount_amount, revenue, discount_rate, tax_invoice,
                    revenue_vat_included, vat, real_unit_price, deposit_date,
                    deposit_status, cash_receipt, depositor_name, batch,
                    outbound_location, stock_change, transaction_amount,
                    avg_weight_per_transaction, avg_price_per_weight,
                    total_transaction_weight, project, revenue_per_weight, note,
                    stock_instagram, stock_change_type, unique_sale, sale_order,
                    last_modified_by, customer_name_text, author, customer_value,
                    raw_json
                )
                VALUES (
                    %(record_id)s, %(created_time)s, %(last_modified_time)s,
                    %(date_value)s, %(customer_name)s, %(channel_1)s, %(channel_2)s,
                    %(product)s, %(payment_type)s, %(revenue_migration)s,
                    %(stock_change_weight)s, %(weight)s, %(unit_price)s,
                    %(base_price)s, %(discount_amount)s, %(revenue)s,
                    %(discount_rate)s, %(tax_invoice)s, %(revenue_vat_included)s,
                    %(vat)s, %(real_unit_price)s, %(deposit_date)s,
                    %(deposit_status)s, %(cash_receipt)s, %(depositor_name)s,
                    %(batch)s, %(outbound_location)s, %(stock_change)s,
                    %(transaction_amount)s, %(avg_weight_per_transaction)s,
                    %(avg_price_per_weight)s, %(total_transaction_weight)s,
                    %(project)s, %(revenue_per_weight)s, %(note)s,
                    %(stock_instagram)s, %(stock_change_type)s, %(unique_sale)s,
                    %(sale_order)s, %(last_modified_by)s, %(customer_name_text)s,
                    %(author)s, %(customer_value)s, %(raw_json)s
                )
                ON CONFLICT (record_id) DO UPDATE SET
                    last_modified_time = EXCLUDED.last_modified_time,
                    raw_json = EXCLUDED.raw_json,
                    updated_at = now(),
                    date_value = EXCLUDED.date_value,
                    customer_name = EXCLUDED.customer_name,
                    channel_1 = EXCLUDED.channel_1,
                    channel_2 = EXCLUDED.channel_2,
                    product = EXCLUDED.product,
                    payment_type = EXCLUDED.payment_type,
                    revenue_migration = EXCLUDED.revenue_migration,
                    stock_change_weight = EXCLUDED.stock_change_weight,
                    weight = EXCLUDED.weight,
                    unit_price = EXCLUDED.unit_price,
                    base_price = EXCLUDED.base_price,
                    discount_amount = EXCLUDED.discount_amount,
                    revenue = EXCLUDED.revenue,
                    discount_rate = EXCLUDED.discount_rate,
                    tax_invoice = EXCLUDED.tax_invoice,
                    revenue_vat_included = EXCLUDED.revenue_vat_included,
                    vat = EXCLUDED.vat,
                    real_unit_price = EXCLUDED.real_unit_price,
                    deposit_date = EXCLUDED.deposit_date,
                    deposit_status = EXCLUDED.deposit_status,
                    cash_receipt = EXCLUDED.cash_receipt,
                    depositor_name = EXCLUDED.depositor_name,
                    batch = EXCLUDED.batch,
                    outbound_location = EXCLUDED.outbound_location,
                    stock_change = EXCLUDED.stock_change,
                    transaction_amount = EXCLUDED.transaction_amount,
                    avg_weight_per_transaction = EXCLUDED.avg_weight_per_transaction,
                    avg_price_per_weight = EXCLUDED.avg_price_per_weight,
                    total_transaction_weight = EXCLUDED.total_transaction_weight,
                    project = EXCLUDED.project,
                    revenue_per_weight = EXCLUDED.revenue_per_weight,
                    note = EXCLUDED.note,
                    stock_instagram = EXCLUDED.stock_instagram,
                    stock_change_type = EXCLUDED.stock_change_type,
                    unique_sale = EXCLUDED.unique_sale,
                    sale_order = EXCLUDED.sale_order,
                    last_modified_by = EXCLUDED.last_modified_by,
                    customer_name_text = EXCLUDED.customer_name_text,
                    author = EXCLUDED.author,
                    customer_value = EXCLUDED.customer_value
            """, batch_data, page_size=100)

            conn.commit()
            print(f"✓ 배치 저장 완료: {len(batch_data)}건")

class PickupDashboardSync(SyncJob):
    def __init__(self):
        super().__init__(
            name="방문수거 대시보드(pickup_dashboard)",
            table_id="tblDTb1NKe8T8y8AB",
            db_table="airtable_pickup_dashboard",
            sync_type="incremental",
            days=365
        )

    def _save_records(self, records, existing_ids, cur, conn):
        batch_data = []

        for rec in records:
            fields = rec.get("fields", {})

            if rec["id"] in existing_ids:
                self.update_count += 1
            else:
                self.insert_count += 1

            batch_data.append({
                "record_id": rec["id"],
                "created_time": rec.get("createdTime"),
                "last_modified_time": Utils.convert_to_string(fields.get("Last Modified")),
                "total_weight_kg": Utils.convert_to_string(fields.get("총 무게_Kg")),
                "goal_self_weight_2024": Utils.convert_to_string(fields.get("자체무게 목표달성(~24.12)")),
                "visit_apartment_kg": Utils.convert_to_string(fields.get("방문수거+아파트_kg")),
                "visit_kg": Utils.convert_to_string(fields.get("방문수거_Kg")),
                "apartment_kg": Utils.convert_to_string(fields.get("아파트수거_kg")),
                "charan_kg": Utils.convert_to_string(fields.get("차란수거_kg")),
                "visit_ratio": Utils.convert_to_string(fields.get("방문수거(%)")),
                "apartment_ratio": Utils.convert_to_string(fields.get("아파트수거(%)")),
                "self_ratio": Utils.convert_to_string(fields.get("자체수거(%)")),
                "charan_ratio": Utils.convert_to_string(fields.get("차란수거(%)")),
                "apartment_event": Utils.convert_to_string(fields.get("아파트수거_event")),
                "charan_event": Utils.convert_to_string(fields.get("차란수거_event")),
                "visit_event": Utils.convert_to_string(fields.get("방문수거_event 2")),
                "month_value": Utils.convert_to_string(fields.get("월")),
                "week_value": Utils.convert_to_string(fields.get("주")),
                "date_text": Utils.convert_to_string(fields.get("날짜_text")),
                "raw_json": json.dumps(rec, ensure_ascii=False, default=str)
            })

        if batch_data:
            execute_batch(cur, """
                INSERT INTO airtable_pickup_dashboard(
                    record_id, created_time, last_modified_time, total_weight_kg,
                    goal_self_weight_2024, visit_apartment_kg, visit_kg, apartment_kg,
                    charan_kg, visit_ratio, apartment_ratio, self_ratio, charan_ratio,
                    apartment_event, charan_event, visit_event, month_value,
                    week_value, date_text, raw_json
                )
                VALUES (
                    %(record_id)s, %(created_time)s, %(last_modified_time)s,
                    %(total_weight_kg)s, %(goal_self_weight_2024)s,
                    %(visit_apartment_kg)s, %(visit_kg)s, %(apartment_kg)s,
                    %(charan_kg)s, %(visit_ratio)s, %(apartment_ratio)s,
                    %(self_ratio)s, %(charan_ratio)s, %(apartment_event)s,
                    %(charan_event)s, %(visit_event)s, %(month_value)s,
                    %(week_value)s, %(date_text)s, %(raw_json)s
                )
                ON CONFLICT (record_id) DO UPDATE SET
                    last_modified_time = EXCLUDED.last_modified_time,
                    raw_json = EXCLUDED.raw_json,
                    updated_at = now(),
                    total_weight_kg = EXCLUDED.total_weight_kg,
                    goal_self_weight_2024 = EXCLUDED.goal_self_weight_2024,
                    visit_apartment_kg = EXCLUDED.visit_apartment_kg,
                    visit_kg = EXCLUDED.visit_kg,
                    apartment_kg = EXCLUDED.apartment_kg,
                    charan_kg = EXCLUDED.charan_kg,
                    visit_ratio = EXCLUDED.visit_ratio,
                    apartment_ratio = EXCLUDED.apartment_ratio,
                    self_ratio = EXCLUDED.self_ratio,
                    charan_ratio = EXCLUDED.charan_ratio,
                    apartment_event = EXCLUDED.apartment_event,
                    charan_event = EXCLUDED.charan_event,
                    visit_event = EXCLUDED.visit_event,
                    month_value = EXCLUDED.month_value,
                    week_value = EXCLUDED.week_value,
                    date_text = EXCLUDED.date_text
            """, batch_data, page_size=100)

            conn.commit()
            print(f"✓ 배치 저장 완료: {len(batch_data)}건")

class NaverAdSync:
    def __init__(self, days=7, max_workers=10):
        self.name = "네이버 광고 통계"
        self.days = days
        self.max_workers = max_workers
        self.insert_count = 0
        self.update_count = 0
        self.start_time = None
        self.end_time = None

    def get_summary(self):
        elapsed = (self.end_time - self.start_time) if self.end_time else 0
        return {
            "name": self.name,
            "insert": self.insert_count,
            "update": self.update_count,
            "total": self.insert_count + self.update_count,
            "elapsed": round(elapsed, 1)
        }

    def _make_headers(self, method, uri):
        if not (Config.NAVER_API_KEY and Config.NAVER_API_SECRET and Config.NAVER_CUSTOMER_ID):
            raise RuntimeError("네이버 API 자격증명이 설정되지 않았습니다.")

        timestamp = str(int(time.time() * 1000))
        message = f"{timestamp}.{method}.{uri}"
        signature = hmac.new(
            Config.NAVER_API_SECRET.encode("utf-8"),
            message.encode("utf-8"),
            hashlib.sha256
        ).digest()
        signature_b64 = base64.b64encode(signature).decode()

        return {
            "X-Timestamp": timestamp,
            "X-API-KEY": Config.NAVER_API_KEY,
            "X-Customer": Config.NAVER_CUSTOMER_ID,
            "X-Signature": signature_b64,
            "Content-Type": "application/json"
        }

    def _create_tables(self, cur, conn):
        cur.execute("""
            CREATE TABLE IF NOT EXISTS adgroup_daily_stats (
                id SERIAL PRIMARY KEY,
                date DATE,
                ad_group_id VARCHAR(50),
                impression BIGINT DEFAULT 0,
                click BIGINT DEFAULT 0,
                cost BIGINT DEFAULT 0,
                created_at TIMESTAMP DEFAULT NOW(),
                updated_at TIMESTAMP DEFAULT NOW(),
                UNIQUE(date, ad_group_id)
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS ad_keyword_by_date (
                id SERIAL PRIMARY KEY,
                date DATE,
                ad_group_id VARCHAR(50),
                ad_group_name TEXT,
                keyword_id VARCHAR(50),
                keyword_text TEXT,
                impression BIGINT DEFAULT 0,
                click BIGINT DEFAULT 0,
                cost BIGINT DEFAULT 0,
                ctr DECIMAL(10,4),
                cpc DECIMAL(10,2),
                avg_rank DECIMAL(10,2),
                created_at TIMESTAMP DEFAULT NOW(),
                updated_at TIMESTAMP DEFAULT NOW()
            )
        """)
        cur.execute("""
            CREATE UNIQUE INDEX IF NOT EXISTS idx_ad_keyword_by_date_unique
            ON ad_keyword_by_date (date, keyword_id)
        """)
        cur.execute("""
            ALTER TABLE ad_keyword_by_date
            ADD COLUMN IF NOT EXISTS ad_group_name TEXT
        """)
        conn.commit()

    def _fetch_active_campaigns(self):
        url = "https://api.searchad.naver.com/ncc/campaigns"
        response = requests.get(url, headers=self._make_headers("GET", "/ncc/campaigns"), timeout=10)
        response.raise_for_status()
        campaigns = response.json()
        return [c for c in campaigns if c.get("status") == "ELIGIBLE"]

    def _get_adgroup_daily_stats(self, adgroup_id, since, until):
        endpoint = "/stats"
        url = f"https://api.searchad.naver.com{endpoint}"
        params = {
            "id": adgroup_id,
            "fields": json.dumps(["impCnt", "clkCnt", "salesAmt"]),
            "timeRange": json.dumps({"since": since, "until": until}),
            "timeIncrement": "1"
        }
        response = requests.get(url, params=params, headers=self._make_headers("GET", endpoint), timeout=10)
        if response.status_code == 200:
            return response.json()
        return None

    def _get_keyword_daily_stats(self, keyword_id, since, until):
        endpoint = "/stats"
        url = f"https://api.searchad.naver.com{endpoint}"
        params = {
            "id": keyword_id,
            "fields": json.dumps(["impCnt", "clkCnt", "salesAmt", "ctr", "cpc", "avgRnk"]),
            "timeRange": json.dumps({"since": since, "until": until}),
            "timeIncrement": "1"
        }
        response = requests.get(url, params=params, headers=self._make_headers("GET", endpoint), timeout=10)
        if response.status_code == 200:
            return response.json()
        return None

    def _batch_save_adgroup_stats(self, stats_list, cur, conn):
        if not stats_list:
            return

        records_to_insert = []
        for adgroup_id, stats_response in stats_list:
            if not stats_response or "data" not in stats_response:
                continue
            for stat in stats_response["data"]:
                records_to_insert.append((
                    stat.get("dateEnd"),
                    adgroup_id,
                    stat.get("impCnt", 0),
                    stat.get("clkCnt", 0),
                    stat.get("salesAmt", 0)
                ))

        if records_to_insert:
            execute_batch(cur, """
                INSERT INTO adgroup_daily_stats (date, ad_group_id, impression, click, cost, created_at, updated_at)
                VALUES (%s, %s, %s, %s, %s, NOW(), NOW())
                ON CONFLICT (date, ad_group_id)
                DO UPDATE SET
                    impression = EXCLUDED.impression,
                    click = EXCLUDED.click,
                    cost = EXCLUDED.cost,
                    updated_at = NOW()
                WHERE
                    adgroup_daily_stats.impression != EXCLUDED.impression OR
                    adgroup_daily_stats.click != EXCLUDED.click OR
                    adgroup_daily_stats.cost != EXCLUDED.cost
            """, records_to_insert, page_size=100)
            conn.commit()
            self.update_count += len(records_to_insert)

    def _batch_save_keyword_stats(self, stats_list, cur, conn):
        if not stats_list:
            return

        records_to_insert = []
        for keyword_info, adgroup_id, adgroup_name, stats_response in stats_list:
            if not stats_response or "data" not in stats_response:
                continue
            for stat in stats_response["data"]:
                records_to_insert.append((
                    stat.get("dateEnd"),
                    adgroup_id,
                    adgroup_name,
                    keyword_info["nccKeywordId"],
                    keyword_info["keyword"],
                    stat.get("impCnt", 0),
                    stat.get("clkCnt", 0),
                    stat.get("salesAmt", 0),
                    stat.get("ctr"),
                    stat.get("cpc"),
                    stat.get("avgRnk")
                ))

        if records_to_insert:
            execute_batch(cur, """
                INSERT INTO ad_keyword_by_date (
                    date, ad_group_id, ad_group_name, keyword_id, keyword_text,
                    impression, click, cost, ctr, cpc, avg_rank, created_at, updated_at
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW(), NOW())
                ON CONFLICT (date, keyword_id) DO UPDATE SET
                    impression = EXCLUDED.impression,
                    click = EXCLUDED.click,
                    cost = EXCLUDED.cost,
                    ctr = EXCLUDED.ctr,
                    cpc = EXCLUDED.cpc,
                    avg_rank = EXCLUDED.avg_rank,
                    updated_at = NOW()
                WHERE
                    ad_keyword_by_date.impression != EXCLUDED.impression OR
                    ad_keyword_by_date.click != EXCLUDED.click OR
                    ad_keyword_by_date.cost != EXCLUDED.cost OR
                    ad_keyword_by_date.ctr != EXCLUDED.ctr OR
                    ad_keyword_by_date.cpc != EXCLUDED.cpc OR
                    ad_keyword_by_date.avg_rank != EXCLUDED.avg_rank
            """, records_to_insert, page_size=100)
            conn.commit()
            self.update_count += len(records_to_insert)

    def execute(self, conn, cur):
        self.start_time = time.time()

        END_DATE = date.today().strftime("%Y-%m-%d")
        START_DATE = (date.today() - timedelta(days=self.days)).strftime("%Y-%m-%d")

        print(f"\n{'='*60}")
        print(f"작업 시작: {self.name}")
        print(f"기간: {START_DATE} ~ {END_DATE} (최근 {self.days}일)")
        print(f"{'='*60}")

        self._create_tables(cur, conn)

        print("\n1) 활성 캠페인 조회")
        active_campaigns = self._fetch_active_campaigns()
        print(f"✓ 활성 캠페인 {len(active_campaigns)}개")

        adgroups_resp = requests.get(
            "https://api.searchad.naver.com/ncc/adgroups",
            headers=self._make_headers("GET", "/ncc/adgroups"),
            timeout=10
        )
        adgroups_resp.raise_for_status()
        all_adgroups = adgroups_resp.json()

        active_campaign_ids = [c["nccCampaignId"] for c in active_campaigns]
        active_adgroups = [
            ag for ag in all_adgroups
            if ag.get("nccCampaignId") in active_campaign_ids and ag.get("status") == "ELIGIBLE"
        ]
        print(f"✓ 활성 광고그룹 {len(active_adgroups)}개")

        # 광고그룹 통계 병렬 수집
        adgroup_stats_list = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(self._get_adgroup_daily_stats, ag["nccAdgroupId"], START_DATE, END_DATE): ag
                for ag in active_adgroups
            }
            for future in as_completed(futures):
                try:
                    result = future.result()
                    if result:
                        adgroup_stats_list.append((futures[future]["nccAdgroupId"], result))
                except Exception:
                    continue

        self._batch_save_adgroup_stats(adgroup_stats_list, cur, conn)

        # 키워드 통계 병렬 수집
        keyword_tasks = []
        for adgroup in active_adgroups:
            keywords_url = "https://api.searchad.naver.com/ncc/keywords"
            keywords_params = {"nccAdgroupId": adgroup["nccAdgroupId"]}
            keywords_resp = requests.get(
                keywords_url,
                params=keywords_params,
                headers=self._make_headers("GET", "/ncc/keywords"),
                timeout=10
            )
            if keywords_resp.status_code == 200:
                keywords = keywords_resp.json()
                active_keywords = [kw for kw in keywords if kw.get("status") == "ELIGIBLE"]
                for kw_info in active_keywords:
                    keyword_tasks.append((kw_info, adgroup["nccAdgroupId"], adgroup["name"]))

        print(f"✓ 키워드 {len(keyword_tasks)}개 수집 대상")

        keyword_stats_list = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(
                    self._get_keyword_daily_stats,
                    kw_info["nccKeywordId"],
                    START_DATE,
                    END_DATE
                ): (kw_info, ag_id, ag_name)
                for kw_info, ag_id, ag_name in keyword_tasks
            }
            for idx, future in enumerate(as_completed(futures), 1):
                try:
                    res = future.result()
                    kw_info, ag_id, ag_name = futures[future]
                    if res:
                        keyword_stats_list.append((kw_info, ag_id, ag_name, res))
                except Exception:
                    continue
                if idx % 50 == 0:
                    print(f"   진행: {idx}/{len(keyword_tasks)}")

        self._batch_save_keyword_stats(keyword_stats_list, cur, conn)

        self.end_time = time.time()
        elapsed = self.end_time - self.start_time
        Utils.send_slack_message(
            f"** bmsg mkt synced (fast)\n"
            f"📊 광고그룹 {len(active_adgroups)}개\n"
            f"🎯 키워드 {len(keyword_tasks)}개\n"
            f"⏰ 기간: 최근 {self.days}일\n"
            f"⚡ 실행시간: {elapsed:.1f}초"
        )

class CountComparisonJob:
    def __init__(self):
        self.name = "전체 레코드 개수 비교"
        self.insert_count = 0
        self.update_count = 0
        self.start_time = None
        self.end_time = None

    def get_summary(self):
        elapsed = (self.end_time - self.start_time) if self.end_time else 0
        return {
            "name": self.name,
            "insert": self.insert_count,
            "update": self.update_count,
            "total": self.insert_count + self.update_count,
            "elapsed": round(elapsed, 1)
        }

    def _get_airtable_count(self, table_id):
        headers = {"Authorization": f"Bearer {Config.AIRTABLE_API_KEY}"}
        url = f"https://api.airtable.com/v0/{Config.AIRTABLE_BASE_ID}/{table_id}"
        total = 0
        offset = None

        while True:
            params = {"pageSize": 100, "fields[]": []}
            if offset:
                params["offset"] = offset
            response = requests.get(url, headers=headers, params=params, timeout=15)
            data = response.json()
            if "records" not in data:
                break
            recs = data["records"]
            total += len(recs)
            offset = data.get("offset")
            if not offset:
                break
            time.sleep(0.25)
        return total

    def _get_supabase_count(self, cur, table_name):
        cur.execute(f"SELECT COUNT(*) FROM {table_name}")
        return cur.fetchone()[0]

    def execute(self, conn, cur):
        self.start_time = time.time()
        airtable_tables = {
            "revenue": "tblFBD10lHsDna9Ox",
            "bmsg_records": "tblT0tygjCwD4jpni",
            "inventory": "tblVcKkL4Qj2I9QBy",
            "pickup_dashboard": "tblDTb1NKe8T8y8AB",
        }
        supabase_tables = {
            "revenue": "airtable_revenue",
            "bmsg_records": "airtable_records",
            "inventory": "airtable_inventory",
            "pickup_dashboard": "airtable_pickup_dashboard",
        }

        airtable_results = {}
        supabase_results = {}

        # Airtable 병렬 조회
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = {executor.submit(self._get_airtable_count, tbl_id): key for key, tbl_id in airtable_tables.items()}
            for future in as_completed(futures):
                key = futures[future]
                airtable_results[key] = future.result()

        # Supabase 카운트
        for key, table_name in supabase_tables.items():
            supabase_results[key] = self._get_supabase_count(cur, table_name)

        log_lines = []
        def log_add(x):
            log_lines.append(x)
            print(x)

        total_airtable = 0
        total_supabase = 0
        differences = []

        log_add("=" * 60)
        log_add("TABLE COUNT COMPARISON")
        log_add("=" * 60)

        for key in sorted(airtable_tables.keys()):
            a_count = airtable_results.get(key, 0)
            s_count = supabase_results.get(key, 0)
            diff = a_count - s_count
            total_airtable += a_count
            total_supabase += s_count
            status = "✅ MATCH" if diff == 0 else f"❌ DIFF: {diff}"
            log_add(f"{key.upper()}")
            log_add(f"  Airtable: {a_count:,}")
            log_add(f"  Supabase: {s_count:,}")
            log_add(f"  Status:   {status}\n")
            if diff != 0:
                differences.append((key, diff))

        elapsed = round(time.time() - self.start_time, 1)
        log_add("=" * 60)
        log_add("SUMMARY")
        log_add("=" * 60)
        log_add(f"Total Airtable records: {total_airtable:,}")
        log_add(f"Total Supabase records: {total_supabase:,}")
        log_add(f"Overall difference:     {total_airtable - total_supabase:,}")

        if differences:
            log_add(f"\n❌ {len(differences)} mismatched tables:")
            for key, diff in differences:
                log_add(f"  - {key}: {diff:+,}")
        else:
            log_add("\n✅ All tables match perfectly!")

        log_add(f"\n⏱ Execution time: {elapsed} seconds")
        Utils.send_slack_message("\n".join(log_lines))
        self.insert_count = total_airtable
        self.update_count = total_supabase
        self.end_time = time.time()

# =====================
# 메인 실행 오케스트레이터
# =====================
class SyncOrchestrator:
    def __init__(self):
        self.jobs = []
        self.overall_start = None
        self.overall_end = None
        
    def add_job(self, job):
        self.jobs.append(job)
    
    def execute_all(self):
        self.overall_start = time.time()
        
        print("\n" + "="*80)
        print("Airtable → Supabase 통합 동기화 시작")
        print(f"시작 시각: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"총 작업 수: {len(self.jobs)}개")
        print("="*80)
        
        # DB 연결
        conn = psycopg2.connect(
            host=Config.SUPABASE_HOST,
            port=Config.SUPABASE_PORT,
            database=Config.SUPABASE_DB,
            user=Config.SUPABASE_USER,
            password=Config.SUPABASE_PASSWORD
        )
        cur = conn.cursor()
        
        # 테이블 생성
        self._create_tables(cur, conn)
        
        # 각 작업 실행
        for idx, job in enumerate(self.jobs, 1):
            try:
                print(f"\n[{idx}/{len(self.jobs)}] ", end="")
                job.execute(conn, cur)
            except Exception as e:
                print(f"\n✗ 작업 실패: {job.name}")
                print(f"  오류: {str(e)}")
                Utils.send_slack_message(f"❌ {job.name} 동기화 실패: {str(e)}")
        
        cur.close()
        conn.close()
        
        self.overall_end = time.time()
        
        # 최종 리포트
        self._send_final_report()
    
    def _create_tables(self, cur, conn):
        print("\n데이터베이스 테이블 확인 중...")
        
        # bmsg_records 테이블
        cur.execute("""
        CREATE TABLE IF NOT EXISTS airtable_records (
            record_id VARCHAR PRIMARY KEY,
            created_time TIMESTAMP,
            last_modified_time TEXT,
            gu_dong TEXT,
            maeip_total TEXT,
            visit_request_date TEXT,
            created_date TEXT,
            assigned_rider TEXT,
            pickup_number TEXT,
            pickup_number_text TEXT,
            pickup_datetime TEXT,
            pickup_date_std TEXT,
            application_unique TEXT,
            real_weight_bag TEXT,
            real_weight_shoes TEXT,
            real_weight_clothes TEXT,
            real_weight_padding TEXT,
            real_weight_total TEXT,
            expected_weight TEXT,
            reservation_date TEXT,
            awareness_path TEXT,
            address TEXT,
            status TEXT,
            cancel_reason TEXT,
            raw_json JSONB,
            created_at TIMESTAMP DEFAULT NOW(),
            updated_at TIMESTAMP DEFAULT NOW()
        )
        """)
        
        # inventory 테이블
        cur.execute("""
        CREATE TABLE IF NOT EXISTS airtable_inventory (
            record_id VARCHAR PRIMARY KEY,
            created_time TIMESTAMP,
            last_modified_time TEXT,
            batch_number TEXT,
            batch_ts TEXT,
            batch_type TEXT,
            batch_number_text TEXT,
            batch_number_change_datetime TEXT,
            batchTS TEXT,
            stock_change_datetime TEXT,
            stock_change_date_parsed TEXT,
            date_reference TEXT,
            sales_number TEXT,
            customer_name TEXT,
            site TEXT,
            stock_change TEXT,
            stock_change_channel TEXT,
            rollup_manager TEXT,
            apartment TEXT,
            sales_weight TEXT,
            weight TEXT,
            change_weight TEXT,
            site_text TEXT,
            batch_weight TEXT,
            note TEXT,
            outbound_weight_check TEXT,
            inbound_weight_check TEXT,
            temp_value TEXT,
            last_modified_by TEXT,
            input_manager TEXT,
            input_manager_text TEXT,
            channel_2 TEXT,
            batch_total_weight TEXT,
            last_modified_2 TEXT,
            last_modified_by_2 TEXT,
            input_manager_2 TEXT,
            input_manager_text_2 TEXT,
            raw_json JSONB,
            created_at TIMESTAMP DEFAULT NOW(),
            updated_at TIMESTAMP DEFAULT NOW()
        )
        """)

        # revenue 테이블
        cur.execute("""
        CREATE TABLE IF NOT EXISTS airtable_revenue (
            record_id VARCHAR PRIMARY KEY,
            created_time TIMESTAMP,
            last_modified_time TEXT,
            date_value TEXT,
            customer_name TEXT,
            channel_1 TEXT,
            channel_2 TEXT,
            product TEXT,
            payment_type TEXT,
            revenue_migration TEXT,
            stock_change_weight TEXT,
            weight TEXT,
            unit_price TEXT,
            base_price TEXT,
            discount_amount TEXT,
            revenue TEXT,
            discount_rate TEXT,
            tax_invoice TEXT,
            revenue_vat_included TEXT,
            vat TEXT,
            real_unit_price TEXT,
            deposit_date TEXT,
            deposit_status TEXT,
            cash_receipt TEXT,
            depositor_name TEXT,
            batch TEXT,
            outbound_location TEXT,
            stock_change TEXT,
            transaction_amount TEXT,
            avg_weight_per_transaction TEXT,
            avg_price_per_weight TEXT,
            total_transaction_weight TEXT,
            project TEXT,
            revenue_per_weight TEXT,
            note TEXT,
            stock_instagram TEXT,
            stock_change_type TEXT,
            unique_sale TEXT,
            sale_order TEXT,
            last_modified_by TEXT,
            customer_name_text TEXT,
            author TEXT,
            customer_value TEXT,
            raw_json JSONB,
            created_at TIMESTAMP DEFAULT NOW(),
            updated_at TIMESTAMP DEFAULT NOW()
        )
        """)

        # pickup dashboard 테이블
        cur.execute("""
        CREATE TABLE IF NOT EXISTS airtable_pickup_dashboard (
            record_id VARCHAR PRIMARY KEY,
            created_time TIMESTAMP,
            last_modified_time TEXT,
            total_weight_kg TEXT,
            goal_self_weight_2024 TEXT,
            visit_apartment_kg TEXT,
            visit_kg TEXT,
            apartment_kg TEXT,
            charan_kg TEXT,
            visit_ratio TEXT,
            apartment_ratio TEXT,
            self_ratio TEXT,
            charan_ratio TEXT,
            apartment_event TEXT,
            charan_event TEXT,
            visit_event TEXT,
            month_value TEXT,
            week_value TEXT,
            date_text TEXT,
            raw_json JSONB,
            created_at TIMESTAMP DEFAULT NOW(),
            updated_at TIMESTAMP DEFAULT NOW()
        )
        """)
        
        conn.commit()
        print("✓ 테이블 생성/확인 완료")
    
    def _send_final_report(self):
        elapsed = self.overall_end - self.overall_start
        
        total_insert = sum(job.insert_count for job in self.jobs)
        total_update = sum(job.update_count for job in self.jobs)
        total_records = total_insert + total_update
        
        print("\n" + "="*80)
        print("동기화 완료 요약")
        print("="*80)
        
        for job in self.jobs:
            summary = job.get_summary()
            print(f"\n{summary['name']}")
            print(f"  신규: {summary['insert']:,}건")
            print(f"  업데이트: {summary['update']:,}건")
            print(f"  소계: {summary['total']:,}건")
            print(f"  실행시간: {summary['elapsed']}초")
        
        print(f"\n{'='*80}")
        print(f"전체 통계")
        print(f"  총 신규: {total_insert:,}건")
        print(f"  총 업데이트: {total_update:,}건")
        print(f"  총 레코드: {total_records:,}건")
        print(f"  총 실행시간: {elapsed:.1f}초")
        print(f"{'='*80}")
        
        # Slack 알림 - 수정된 부분
        slack_msg = "🎉 *Airtable-Supabase 통합 동기화 완료*\n\n"
        
        for job in self.jobs:
            summary = job.get_summary()
            slack_msg += f"\n*{summary['name']}*\n"
            slack_msg += f"➕ 신규 {summary['insert']:,}건\n"
            slack_msg += f"♻️ 업데이트 {summary['update']:,}건\n"
            slack_msg += f"📊 소계 {summary['total']:,}건\n"
            slack_msg += f"⚡ {summary['elapsed']}초\n"
        
        slack_msg += "\n─────────────────────\n"
        slack_msg += "📈 *전체 통계*\n"
        slack_msg += f"총 신규: {total_insert:,}건\n"
        slack_msg += f"총 업데이트: {total_update:,}건\n"
        slack_msg += f"총 레코드: {total_records:,}건\n"
        slack_msg += f"⏱ 총 실행시간: {elapsed:.1f}초"
        
        Utils.send_slack_message(slack_msg)

# =====================
# 실행
# =====================
if __name__ == "__main__":
    # 환경변수 검증
    Config.validate()
    
    orchestrator = SyncOrchestrator()
    
    # 동기화 작업 등록
    orchestrator.add_job(BmsgRecordsSync())
    orchestrator.add_job(InventorySync())
    orchestrator.add_job(RevenueSync())
    orchestrator.add_job(PickupDashboardSync())
    orchestrator.add_job(NaverAdSync())
    orchestrator.add_job(CountComparisonJob())
    
    # 실행
    orchestrator.execute_all()
