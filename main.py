from fastapi import FastAPI
from pydantic import BaseModel
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
import pandas as pd
import numpy as np

app = FastAPI()


# ============================================================
# 운영 철학 (설계 상위 규칙)
# ============================================================
# 1. 수요는 가능한 한 전부 흡수한다 (미수거 0)
# 2. 이동 효율은 최대화한다 (지역 밀도)
# 3. 기사 간 콜 수는 10~25건 심리적 안정 구간 유지
# ============================================================


# --- 상수 정의 ---
VEHICLE_CAPACITY_KG = 1200  # 하드 제약: 기사당 최대 적재량
MIN_CALLS_SOFT = 10         # 소프트 제약: 콜 수 하한 (디모티베이션 방지)
MAX_CALLS_SOFT = 25         # 소프트 제약: 콜 수 상한 (과부하 방지)
DEFAULT_WEIGHT_KG = 15      # 무게 정보 없을 때 기본값


# --- 데이터 모델 ---
class Location(BaseModel):
    id: str
    lat: float
    lon: float
    weight: int = DEFAULT_WEIGHT_KG  # kg 단위 예상 무게


class RequestBody(BaseModel):
    locations: list[Location]
    num_vehicles: int = 4
    vehicle_capacity: int = VEHICLE_CAPACITY_KG


# --- 유틸리티 함수 ---
def haversine(lat1, lon1, lat2, lon2):
    """두 좌표 간 거리 계산 (km)"""
    R = 6371
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2) * np.sin(dlambda/2)**2
    return 2 * R * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


def create_distance_matrix(df):
    """거리 행렬 생성 (미터 단위 정수)"""
    n = len(df)
    coords = df[['lat', 'lon']].values
    dist_matrix = np.zeros((n, n), dtype=int)
    
    for i in range(n):
        for j in range(n):
            if i != j:
                dist_km = haversine(coords[i][0], coords[i][1],
                                   coords[j][0], coords[j][1])
                dist_matrix[i][j] = int(dist_km * 1000)  # km → m
    
    return dist_matrix


@app.get("/")
def read_root():
    return {
        "status": "active", 
        "message": "VRP Engine V8 (CVRP + Soft Fairness)",
        "philosophy": "미수거 0 → 이동효율 → 콜수 안정(10-25)"
    }


@app.post("/optimize")
def optimize_routes(body: RequestBody):
    """
    OR-Tools CVRP 단일 모델 최적화
    
    우선순위:
    1순위: 미수거 최소화 (모든 콜 배정)
    2순위: 이동 효율 최대화 (총 이동거리 최소화)
    3순위: 콜 수 안정성 (10~25건 범위 유지)
    
    하드 제약:
    - 기사당 적재량 ≤ 1,200kg
    
    소프트 제약:
    - 콜 수 < 10 → 패널티
    - 콜 수 > 25 → 패널티
    """
    
    # 1. 데이터 준비
    data = [loc.dict() for loc in body.locations]
    df = pd.DataFrame(data)
    df = df.reset_index(drop=True)
    
    num_locations = len(df)
    num_vehicles = body.num_vehicles
    vehicle_capacity = body.vehicle_capacity
    depot_idx = 0
    
    if num_locations < 2:
        return {"status": "error", "message": "Not enough locations"}
    
    # 무게 기본값 처리
    if 'weight' not in df.columns:
        df['weight'] = DEFAULT_WEIGHT_KG
    df['weight'] = df['weight'].fillna(DEFAULT_WEIGHT_KG).astype(int)
    
    # depot 무게는 0
    df.loc[depot_idx, 'weight'] = 0
    
    # 2. 거리 행렬 생성
    dist_matrix = create_distance_matrix(df)
    
    # 3. OR-Tools 매니저 설정
    manager = pywrapcp.RoutingIndexManager(num_locations, num_vehicles, depot_idx)
    routing = pywrapcp.RoutingModel(manager)
    
    # ============================================================
    # [비용 함수] 이동 거리 (2순위: 이동 효율)
    # ============================================================
    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return dist_matrix[from_node][to_node]
    
    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
    
    # ============================================================
    # [하드 제약 1] 미배정 절대 방지 (1순위: 미수거 0)
    # ============================================================
    # 모든 노드에 대해 방문하지 않으면 극단적 패널티
    # 이 패널티가 충분히 크면 OR-Tools는 절대 노드를 빼지 않음
    UNASSIGNED_PENALTY = 10000000000  # 100억
    
    for node_idx in range(1, num_locations):  # depot 제외
        index = manager.NodeToIndex(node_idx)
        routing.AddDisjunction([index], UNASSIGNED_PENALTY)
    
    # ============================================================
    # [하드 제약 2] 적재량 제한 (1,200kg)
    # ============================================================
    def demand_callback(from_index):
        from_node = manager.IndexToNode(from_index)
        return int(df.iloc[from_node]['weight'])
    
    demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
    
    routing.AddDimensionWithVehicleCapacity(
        demand_callback_index,
        0,  # slack 없음
        [vehicle_capacity] * num_vehicles,  # 각 차량 최대 적재량
        True,  # 시작점에서 0부터
        'Capacity'
    )
    
    # ============================================================
    # [소프트 제약] 콜 수 안정성 (10~25건)
    # ============================================================
    def count_callback(from_index):
        """각 방문을 1건으로 카운트 (depot 제외)"""
        from_node = manager.IndexToNode(from_index)
        if from_node == depot_idx:
            return 0
        return 1
    
    count_callback_index = routing.RegisterUnaryTransitCallback(count_callback)
    
    routing.AddDimension(
        count_callback_index,
        0,  # slack
        100,  # 최대값 (충분히 큰 값)
        True,
        'Count'
    )
    
    count_dimension = routing.GetDimensionOrDie('Count')
    
    # 콜 수 소프트 제약 설정
    CALL_PENALTY = 100000  # 범위 이탈 시 패널티
    
    for vehicle_id in range(num_vehicles):
        end_index = routing.End(vehicle_id)
        
        # 하한: 10건 미만이면 패널티
        count_dimension.SetCumulVarSoftLowerBound(end_index, MIN_CALLS_SOFT, CALL_PENALTY)
        
        # 상한: 25건 초과하면 패널티
        count_dimension.SetCumulVarSoftUpperBound(end_index, MAX_CALLS_SOFT, CALL_PENALTY)
    
    # ============================================================
    # [추가 최적화] 이동거리 균등화 (지역 클러스터링 유도)
    # ============================================================
    routing.AddDimension(
        transit_callback_index,
        0,  # slack
        10000000,  # 최대 이동거리 (충분히 큰 값)
        True,
        'Distance'
    )
    
    distance_dimension = routing.GetDimensionOrDie('Distance')
    
    # GlobalSpanCost: 기사 간 이동거리 편차 최소화
    # → 자연스럽게 지역별 클러스터링 효과
    distance_dimension.SetGlobalSpanCostCoefficient(100)
    
    # ============================================================
    # [검색 전략 설정]
    # ============================================================
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    
    # 초기 해 전략: 병렬 삽입 (여러 기사가 동시에 가까운 점 선점)
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PARALLEL_CHEAPEST_INSERTION
    )
    
    # 메타휴리스틱: Guided Local Search (지역 최적해 탈출)
    search_parameters.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    )
    
    # 충분한 계산 시간 부여
    search_parameters.time_limit.seconds = 30
    
    # 로그 출력 (디버깅용, 운영 시 제거 가능)
    search_parameters.log_search = False
    
    # ============================================================
    # [풀이 실행]
    # ============================================================
    solution = routing.SolveWithParameters(search_parameters)
    
    # ============================================================
    # [결과 처리]
    # ============================================================
    if not solution:
        return {
            "status": "fail",
            "message": "Solution not found. 적재량 제약으로 모든 콜 배정 불가능할 수 있음.",
            "suggestion": "기사 수를 늘리거나 적재량 제한을 확인하세요."
        }
    
    results = []
    stats = []
    total_distance = 0
    
    for vehicle_id in range(num_vehicles):
        route = []
        route_distance = 0
        route_weight = 0
        route_order = 1
        
        index = routing.Start(vehicle_id)
        prev_node = depot_idx
        
        while not routing.IsEnd(index):
            node_idx = manager.IndexToNode(index)
            
            if node_idx != depot_idx:
                route.append({
                    "id": df.iloc[node_idx]['id'],
                    "order": route_order,
                    "weight_kg": int(df.iloc[node_idx]['weight']),
                    "cumulative_weight_kg": route_weight + int(df.iloc[node_idx]['weight'])
                })
                route_weight += int(df.iloc[node_idx]['weight'])
                route_distance += dist_matrix[prev_node][node_idx]
                prev_node = node_idx
                route_order += 1
            
            index = solution.Value(routing.NextVar(index))
        
        # depot 복귀 거리
        route_distance += dist_matrix[prev_node][depot_idx]
        total_distance += route_distance
        
        # 결과에 추가
        for stop in route:
            results.append({
                "id": stop["id"],
                "driver": f"기사 {vehicle_id + 1}",
                "visit_order": stop["order"],
                "weight_kg": stop["weight_kg"],
                "cumulative_weight_kg": stop["cumulative_weight_kg"]
            })
        
        # 통계
        call_count = len(route)
        status = "정상"
        if call_count < MIN_CALLS_SOFT:
            status = f"⚠️ 하한 미달 ({call_count} < {MIN_CALLS_SOFT})"
        elif call_count > MAX_CALLS_SOFT:
            status = f"⚠️ 상한 초과 ({call_count} > {MAX_CALLS_SOFT})"
        
        stats.append({
            "driver": f"기사 {vehicle_id + 1}",
            "call_count": call_count,
            "total_weight_kg": route_weight,
            "distance_km": round(route_distance / 1000, 2),
            "status": status
        })
    
    # 미배정 체크
    assigned_ids = set([r['id'] for r in results])
    all_ids = set(df[df.index != depot_idx]['id'].tolist())
    unassigned_ids = all_ids - assigned_ids
    
    return {
        "status": "success",
        "updates": results,
        "statistics": stats,
        "summary": {
            "total_locations": num_locations - 1,
            "total_assigned": len(results),
            "unassigned": len(unassigned_ids),
            "unassigned_ids": list(unassigned_ids) if unassigned_ids else [],
            "total_distance_km": round(total_distance / 1000, 2),
            "avg_distance_km": round(total_distance / 1000 / num_vehicles, 2)
        },
        "constraints": {
            "vehicle_capacity_kg": vehicle_capacity,
            "min_calls_soft": MIN_CALLS_SOFT,
            "max_calls_soft": MAX_CALLS_SOFT
        }
    }


@app.post("/optimize/debug")
def optimize_routes_debug(body: RequestBody):
    """
    디버그용 엔드포인트
    - 입력 데이터 검증
    - 총 무게 계산
    - 배정 가능 여부 사전 체크
    """
    data = [loc.dict() for loc in body.locations]
    df = pd.DataFrame(data)
    
    if 'weight' not in df.columns:
        df['weight'] = DEFAULT_WEIGHT_KG
    df['weight'] = df['weight'].fillna(DEFAULT_WEIGHT_KG).astype(int)
    
    total_weight = df['weight'].sum() - df.iloc[0]['weight']  # depot 제외
    total_calls = len(df) - 1
    max_capacity = body.num_vehicles * body.vehicle_capacity
    
    avg_calls = total_calls / body.num_vehicles
    avg_weight = total_weight / body.num_vehicles
    
    feasibility = "✅ 배정 가능"
    warnings = []
    
    if total_weight > max_capacity:
        feasibility = "❌ 배정 불가능"
        warnings.append(f"총 무게({total_weight}kg)가 총 적재량({max_capacity}kg)을 초과")
    
    if avg_calls < MIN_CALLS_SOFT:
        warnings.append(f"평균 콜 수({avg_calls:.1f})가 하한({MIN_CALLS_SOFT}) 미만")
    
    if avg_calls > MAX_CALLS_SOFT:
        warnings.append(f"평균 콜 수({avg_calls:.1f})가 상한({MAX_CALLS_SOFT}) 초과")
    
    return {
        "input_summary": {
            "total_locations": len(df),
            "total_calls": total_calls,
            "total_weight_kg": int(total_weight),
            "num_vehicles": body.num_vehicles,
            "vehicle_capacity_kg": body.vehicle_capacity
        },
        "per_vehicle_avg": {
            "avg_calls": round(avg_calls, 1),
            "avg_weight_kg": round(avg_weight, 1)
        },
        "capacity_check": {
            "total_capacity_kg": max_capacity,
            "utilization_pct": round(total_weight / max_capacity * 100, 1)
        },
        "feasibility": feasibility,
        "warnings": warnings if warnings else ["없음"]
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
