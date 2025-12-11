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

    # [핵심 1] 미배정 방지 (모든 노드에 강력한 방문 의무 부여)
    # 방문 안 할 경우 패널티 1,000,000,000점 -> 무조건 방문하게 됨
    for node_index in range(1, len(df)):
        routing.AddDisjunction([manager.NodeToIndex(node_index)], 1000000000)

    # [핵심 2] 건수 균등 배분 (Count Dimension)
    def count_callback(from_index):
        return 1
    count_callback_index = routing.RegisterUnaryTransitCallback(count_callback)
    
    # 넉넉하게 인당 최대 100건까지 가능하게 설정 (제약으로 실패하지 않도록)
    routing.AddDimension(count_callback_index, 0, 200, True, 'Count')
    count_dimension = routing.GetDimensionOrDie('Count')
    
    # "가장 많이 한 사람"과 "가장 적게 한 사람"의 차이를 줄이는 데 100,000배 가중치
    # 이 값이 높을수록 거리가 좀 멀어져도 개수를 맞추려고 노력함
    count_dimension.SetGlobalSpanCostCoefficient(100000)

    # 무게 균등 배분 (보조)
    weights = df['weight'].tolist()
    def weight_callback(from_index):
        return weights[manager.IndexToNode(from_index)]
    weight_callback_index = routing.RegisterUnaryTransitCallback(weight_callback)
    routing.AddDimension(weight_callback_index, 0, 50000, True, 'Weight')
    
    # 4. 풀이 설정
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    
    # [핵심 3] 계산 시간 30초로 연장 (데이터가 많아졌으므로 필수)
    search_parameters.time_limit.seconds = 30
    
    # 지역 최적해에 빠지지 않도록 탐색 알고리즘 강화
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
