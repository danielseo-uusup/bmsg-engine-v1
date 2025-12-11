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

# --- 헬스 체크용 (잘 돌아가는지 확인) ---
@app.get("/")
def read_root():
    return {"status": "active", "message": "VRP Optimization Engine is Running!"}

# --- 핵심 최적화 API ---
@app.post("/optimize")
def optimize_routes(body: RequestBody):
    # 1. 데이터 준비
    df = pd.DataFrame([loc.dict() for loc in body.locations])
    
    # 데이터가 너무 적으면 에러 처리
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

    # 비용(거리) 설정
    def distance_callback(from_index, to_index):
        return int(dist_matrix[manager.IndexToNode(from_index)][manager.IndexToNode(to_index)])
    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # 균등 배분 (건수)
    def count_callback(from_index):
        return 1
    count_callback_index = routing.RegisterUnaryTransitCallback(count_callback)
    routing.AddDimension(count_callback_index, 0, 100, True, 'Count')
    routing.GetDimensionOrDie('Count').SetGlobalSpanCostCoefficient(500000)

    # 균등 배분 (무게) - 2순위
    weights = df['weight'].tolist()
    def weight_callback(from_index):
        return weights[manager.IndexToNode(from_index)]
    weight_callback_index = routing.RegisterUnaryTransitCallback(weight_callback)
    routing.AddDimension(weight_callback_index, 0, 10000, True, 'Weight')
    routing.GetDimensionOrDie('Weight').SetGlobalSpanCostCoefficient(100)

    # 풀이
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    search_parameters.time_limit.seconds = 5 # 5초 제한

    solution = routing.SolveWithParameters(search_parameters)

    # 4. 결과 반환
    if not solution:
        return {"status": "fail", "message": "No solution found"}

    results = []
    for vehicle_id in range(body.num_vehicles):
        index = routing.Start(vehicle_id)
        while not routing.IsEnd(index):
            node_index = manager.IndexToNode(index)
            if node_index != 0: # 0번(차고지) 제외
                results.append({
                    "id": df.iloc[node_index]['id'],
                    "driver": f"기사 {vehicle_id + 1}"
                })
            index = solution.Value(routing.NextVar(index))

    return {"status": "success", "updates": results}