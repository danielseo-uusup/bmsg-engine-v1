from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
from scipy.spatial.distance import pdist, squareform
import pandas as pd
import numpy as np

app = FastAPI()

# --- 데이터 모델 정의 ---
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
    return {"status": "active", "message": "VRP Optimization Engine is Running!"}

@app.post("/optimize")
def optimize_routes(body: RequestBody):
    # 1. 데이터 준비
    df = pd.DataFrame([loc.dict() for loc in body.locations])
    
    if len(df) < 2:
        return {"status": "error", "message": "Not enough locations"}

    # 2. 거리 계산 (Haversine)
    def haversine_vectorized(lat1, lon1, lat2, lon2):
        R = 6371
        phi1, phi2 = np.radians(lat1), np.radians(lat2)
        dphi = np.radians(lat2 - lat1)
        dlambda = np.radians(lon2 - lon1)
        a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2) * np.sin(dlambda/2)**2
        return 2 * R * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    coords = df[['lat', 'lon']].values
    dist_matrix = squareform(pdist(coords, lambda u, v: haversine_vectorized(u[0], u[1], v[0], v[1]))) * 1000

    # 3. OR-Tools 설정
    manager = pywrapcp.RoutingIndexManager(len(df), body.num_vehicles, 0)
    routing = pywrapcp.RoutingModel(manager)

    # [중요] 거리 비용 설정
    def distance_callback(from_index, to_index):
        return int(dist_matrix[manager.IndexToNode(from_index)][manager.IndexToNode(to_index)])
    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # [핵심 1] 미배정 절대 방지 (Penalty: 10억 점)
    # 거리가 아무리 멀어도 배정 안 하는 것보다 배정하는 게 이득이게 만듦
    for node_index in range(1, len(df)):
        routing.AddDisjunction([manager.NodeToIndex(node_index)], 1000000000)

    # [핵심 2] 클러스터링 강화 (Count Dimension)
    def count_callback(from_index):
        return 1
    count_callback_index = routing.RegisterUnaryTransitCallback(count_callback)
    
    # 인당 최대 100건까지 가능 (유연하게)
    routing.AddDimension(count_callback_index, 0, 100, True, 'Count')
    count_dimension = routing.GetDimensionOrDie('Count')
    
    # ★★★ 여기를 수정했습니다 ★★★
    # 기존: 100,000 (무조건 똑같이 나눠라 -> 동선 꼬임)
    # 변경: 2,500 (적당히 비슷하게 나누되, 동선 효율을 더 챙겨라 -> 클러스터링)
    count_dimension.SetGlobalSpanCostCoefficient(2500)

    # 무게 균등 배분 (보조)
    weights = df['weight'].tolist()
    def weight_callback(from_index):
        return weights[manager.IndexToNode(from_index)]
    weight_callback_index = routing.RegisterUnaryTransitCallback(weight_callback)
    routing.AddDimension(weight_callback_index, 0, 50000, True, 'Weight')
    
    # 4. 풀이 설정
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    
    # [핵심 3] 계산 시간 30초로 대폭 연장 (복잡한 꼬임을 풀 시간 확보)
    search_parameters.time_limit.seconds = 30
    
    # 지역 최적해 탈출 알고리즘 (Guided Local Search)
    search_parameters.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH

    solution = routing.SolveWithParameters(search_parameters)

    # 5. 결과 반환
    if not solution:
        return {"status": "fail", "message": "No solution found"}

    results = []
    for vehicle_id in range(body.num_vehicles):
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
