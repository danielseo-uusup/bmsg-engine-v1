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
    return {"status": "active", "message": "VRP Engine V3 (Clustering Optimized)"}

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
    # 거리에 가중치를 주어 멀리 있는 점을 더 비싸게 인식시킴
    dist_matrix = squareform(pdist(coords, lambda u, v: haversine_vectorized(u[0], u[1], v[0], v[1]))) * 1000

    # 3. OR-Tools 설정
    manager = pywrapcp.RoutingIndexManager(len(df), num_vehicles, 0)
    routing = pywrapcp.RoutingModel(manager)

    # [비용] 거리 비용
    def distance_callback(from_index, to_index):
        return int(dist_matrix[manager.IndexToNode(from_index)][manager.IndexToNode(to_index)])
    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # [핵심 1] 미배정 절대 방지 (Penalty: 100억 점)
    # 어떤 이유로든 배정 안 하면 계산 자체가 망가지도록 설정
    for node_index in range(1, len(df)):
        routing.AddDisjunction([manager.NodeToIndex(node_index)], 10000000000)

    # [핵심 2] 4명 강제 투입 (최소 업무량 설정)
    def count_callback(from_index):
        return 1
    count_callback_index = routing.RegisterUnaryTransitCallback(count_callback)
    
    routing.AddDimension(count_callback_index, 0, 200, True, 'Count')
    count_dimension = routing.GetDimensionOrDie('Count')
    
    # 총 건수를 차량 수로 나눈 뒤, 그 70% 정도는 무조건 하도록 강제
    # 예: 80건 / 4명 = 20건 -> 최소 14건은 해야 함.
    # (차고지 노드 1개가 포함되므로 실제 업무량은 min_count - 1)
    total_orders = len(df) - 1 # 차고지 제외
    min_count = int((total_orders / num_vehicles) * 0.7) 
    
    # 모든 차량에 최소 업무량 강제 (Soft Lower Bound)
    # 이걸 어기면 벌점 100만점 -> 2명만 일하는 게 불가능해짐
    for i in range(num_vehicles):
        count_dimension.SetCumulVarSoftLowerBound(i, min_count + 1, 1000000)

    # [핵심 3] 클러스터링 유도 (최대 거리 제한)
    # 한 기사가 너무 멀리(지구 끝에서 끝) 이동하지 못하게 막음
    routing.AddDimension(
        transit_callback_index,
        0,  # slack
        300000, # 차량당 최대 이동 거리 (300km)
        True,  # start cumul to zero
        'Distance'
    )
    dist_dimension = routing.GetDimensionOrDie('Distance')
    # 기사들 간의 이동 거리 격차를 줄임 -> 비슷한 크기의 구역을 맡게 됨
    dist_dimension.SetGlobalSpanCostCoefficient(100)

    # 4. 풀이 전략 설정 (여기가 "침팬지" 탈출 열쇠)
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    
    # ★ PARALLEL_CHEAPEST_INSERTION: 
    # 차고지에서 꽃잎처럼 동시에 퍼져나가며 경로를 만듭니다. (군집화에 유리)
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PARALLEL_CHEAPEST_INSERTION)
    
    # ★ GUIDED_LOCAL_SEARCH:
    # 꼬인 선을 푸는 후처리 알고리즘
    search_parameters.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH)
    
    # 계산 시간 60초 (충분히 풀도록)
    search_parameters.time_limit.seconds = 60

    solution = routing.SolveWithParameters(search_parameters)

    # 5. 결과 반환
    if not solution:
        # 실패 시 로그를 위해 
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
