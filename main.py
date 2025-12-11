from fastapi import FastAPI
from pydantic import BaseModel
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
from scipy.spatial.distance import pdist, squareform
import pandas as pd
import numpy as np

app = FastAPI()

# --- 데이터 모델 ---
class Location(BaseModel):
    id: str
    lat: float
    lon: float
    weight: int = 0

class RequestBody(BaseModel):
    locations: list[Location]
    num_vehicles: int = 4

@app.get("/")
def read_root():
    return {"status": "active", "message": "VRP Engine V4 (Parallel Clustering)"}

@app.post("/optimize")
def optimize_routes(body: RequestBody):
    # 1. 데이터 준비
    df = pd.DataFrame([loc.dict() for loc in body.locations])
    
    if len(df) < 2:
        return {"status": "error", "message": "Not enough locations"}

    num_vehicles = body.num_vehicles
    
    # 2. 거리 계산 (Haversine)
    def haversine_vectorized(lat1, lon1, lat2, lon2):
        R = 6371
        phi1, phi2 = np.radians(lat1), np.radians(lat2)
        dphi = np.radians(lat2 - lat1)
        dlambda = np.radians(lon2 - lon1)
        a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2) * np.sin(dlambda/2)**2
        return 2 * R * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    coords = df[['lat', 'lon']].values
    # 거리에 가중치(1000)를 곱해 미터 단위로 변환
    dist_matrix = squareform(pdist(coords, lambda u, v: haversine_vectorized(u[0], u[1], v[0], v[1]))) * 1000

    # 3. OR-Tools 설정
    manager = pywrapcp.RoutingIndexManager(len(df), num_vehicles, 0)
    routing = pywrapcp.RoutingModel(manager)

    # [비용 1] 거리 비용
    def distance_callback(from_index, to_index):
        return int(dist_matrix[manager.IndexToNode(from_index)][manager.IndexToNode(to_index)])
    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # [제약 1] 미배정 절대 방지 (Penalty: 1000억 점)
    # 지구 끝까지 가는 비용보다 미배정 벌점이 더 커야 함 -> 무조건 방문
    for node_index in range(1, len(df)):
        routing.AddDisjunction([manager.NodeToIndex(node_index)], 100000000000)

    # [제약 2] 4명 강제 투입 & 건수 균등화
    def count_callback(from_index):
        return 1
    count_callback_index = routing.RegisterUnaryTransitCallback(count_callback)
    
    routing.AddDimension(count_callback_index, 0, 200, True, 'Count')
    count_dimension = routing.GetDimensionOrDie('Count')
    
    # (A) 최소 업무량 강제: 전체 건수의 10% 이상은 무조건 해야 함
    # 예: 80건이면 최소 8건은 해야 함. 안 하면 벌점 1억 점.
    min_count_per_vehicle = int((len(df) - 1) / num_vehicles * 0.5) # 평균의 50%
    if min_count_per_vehicle < 1: min_count_per_vehicle = 1
    
    for i in range(num_vehicles):
        # SetCumulVarSoftLowerBound(차량번호, 최소값, 미달시벌점)
        count_dimension.SetCumulVarSoftLowerBound(i, min_count_per_vehicle, 100000000)

    # (B) 최대 격차 줄이기: 1등과 꼴등의 건수 차이를 줄임
    count_dimension.SetGlobalSpanCostCoefficient(5000)

    # [제약 3] 이동 거리 균등화 (클러스터링 유도)
    # 한 명이 너무 멀리(넓게) 다니지 못하게 막음 -> 구역이 뭉치게 됨
    routing.AddDimension(
        transit_callback_index,
        0,  # slack
        500000, # 최대 이동 거리 (넉넉하게)
        True,  # start cumul to zero
        'Distance'
    )
    dist_dimension = routing.GetDimensionOrDie('Distance')
    # 기사들 간의 총 이동 거리 차이를 줄임 -> 비슷한 반경의 구역을 맡게 됨
    dist_dimension.SetGlobalSpanCostCoefficient(5000)

    # 4. 풀이 전략 설정 (여기가 핵심!)
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    
    # ★ PARALLEL_CHEAPEST_INSERTION: 
    # 차고지에서 4명이 동시에 뻗어나가며 가장 싼 곳을 먹는 방식. 
    # 이 전략이 "꽃잎 모양" 구역 나누기에 가장 효과적임.
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PARALLEL_CHEAPEST_INSERTION)
    
    # ★ GUIDED_LOCAL_SEARCH:
    # 꼬인 선을 풀고 최적해를 찾아가는 후처리 (필수)
    search_parameters.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH)
    
    # 계산 시간 60초 (충분히 풀도록)
    search_parameters.time_limit.seconds = 60

    solution = routing.SolveWithParameters(search_parameters)

    # 5. 결과 반환
    if not solution:
        return {"status": "fail", "message": "Solution not found"}

    results = []
    for vehicle_id in range(num_vehicles):
        index = routing.Start(vehicle_id)
        while not routing.IsEnd(index):
            node_index = manager.IndexToNode(index)
            if node_index != 0:
                results.append({
                    "id": df.iloc[node_index]['id'],
                    "driver": f"기사 {vehicle_id + 1}"
                })
            index = solution.Value(routing.NextVar(index))

    return {"status": "success", "updates": results}
