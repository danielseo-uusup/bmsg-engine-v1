from fastapi import FastAPI
from pydantic import BaseModel
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
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
    """거리 행렬 생성 (미터 단위)"""
    n = len(df)
    dist_matrix = np.zeros((n, n))
    coords = df[['lat', 'lon']].values
    
    for i in range(n):
        for j in range(n):
            if i != j:
                dist_matrix[i][j] = haversine(
                    coords[i][0], coords[i][1],
                    coords[j][0], coords[j][1]
                ) * 1000  # km → m
    
    return dist_matrix


def create_sector_clusters(df, num_vehicles, depot_idx=0):
    """
    Depot 기준 방위각으로 섹터 분할 (꽃잎 모양)
    - 각 기사가 명확한 지리적 구역을 담당하게 됨
    """
    depot_lat = df.iloc[depot_idx]['lat']
    depot_lon = df.iloc[depot_idx]['lon']
    
    # 각 포인트의 depot 기준 방위각 계산
    angles = []
    indices = []
    
    for idx in range(len(df)):
        if idx == depot_idx:
            continue
        
        dy = df.iloc[idx]['lat'] - depot_lat
        dx = df.iloc[idx]['lon'] - depot_lon
        angle = np.arctan2(dy, dx)
        angles.append(angle)
        indices.append(idx)
    
    # 방위각 기준 정렬
    sorted_pairs = sorted(zip(angles, indices), key=lambda x: x[0])
    sorted_indices = [pair[1] for pair in sorted_pairs]
    
    # 균등 분할
    splits = np.array_split(sorted_indices, num_vehicles)
    
    # 클러스터 라벨 할당
    cluster_labels = {depot_idx: -1}  # depot은 -1
    for cluster_id, split in enumerate(splits):
        for idx in split:
            cluster_labels[idx] = cluster_id
    
    df = df.copy()
    df['cluster'] = df.index.map(lambda x: cluster_labels.get(x, -1))
    
    return df


def solve_tsp_within_cluster(cluster_df, depot_coords):
    """
    클러스터 내 TSP 최적화 (OR-Tools 사용)
    - depot에서 출발하여 클러스터 내 모든 지점 방문 후 depot 복귀
    """
    if len(cluster_df) == 0:
        return []
    
    if len(cluster_df) == 1:
        return [cluster_df.iloc[0]['id']]
    
    # depot을 임시로 추가 (인덱스 0)
    temp_df = pd.DataFrame([{
        'id': '__depot__',
        'lat': depot_coords[0],
        'lon': depot_coords[1]
    }])
    temp_df = pd.concat([temp_df, cluster_df[['id', 'lat', 'lon']].reset_index(drop=True)], ignore_index=True)
    
    # 거리 행렬 생성
    dist_matrix = create_distance_matrix(temp_df)
    
    # OR-Tools 설정
    manager = pywrapcp.RoutingIndexManager(len(temp_df), 1, 0)
    routing = pywrapcp.RoutingModel(manager)
    
    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return int(dist_matrix[from_node][to_node])
    
    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
    
    # 검색 파라미터
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
    search_parameters.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH)
    search_parameters.time_limit.seconds = 5
    
    solution = routing.SolveWithParameters(search_parameters)
    
    if not solution:
        # 풀이 실패 시 Nearest Neighbor 휴리스틱
        return nearest_neighbor_order(cluster_df, depot_coords)
    
    # 결과 추출 (depot 제외)
    route = []
    index = routing.Start(0)
    while not routing.IsEnd(index):
        node_index = manager.IndexToNode(index)
        if node_index != 0:  # depot 제외
            route.append(temp_df.iloc[node_index]['id'])
        index = solution.Value(routing.NextVar(index))
    
    return route


def nearest_neighbor_order(cluster_df, depot_coords):
    """Nearest Neighbor 휴리스틱 (백업용)"""
    if len(cluster_df) == 0:
        return []
    
    coords = cluster_df[['lat', 'lon']].values
    ids = cluster_df['id'].tolist()
    n = len(cluster_df)
    
    visited = [False] * n
    order = []
    
    # depot에서 가장 가까운 점부터 시작
    current_lat, current_lon = depot_coords
    
    for _ in range(n):
        min_dist = float('inf')
        nearest_idx = -1
        
        for j in range(n):
            if not visited[j]:
                dist = haversine(current_lat, current_lon, coords[j][0], coords[j][1])
                if dist < min_dist:
                    min_dist = dist
                    nearest_idx = j
        
        if nearest_idx != -1:
            visited[nearest_idx] = True
            order.append(ids[nearest_idx])
            current_lat, current_lon = coords[nearest_idx]
    
    return order


@app.get("/")
def read_root():
    return {"status": "active", "message": "VRP Engine V5 (Sector Clustering + TSP)"}


@app.post("/optimize")
def optimize_routes(body: RequestBody):
    """
    2단계 최적화:
    1) 방위각 기반 섹터 클러스터링 → 구역 분리
    2) 각 클러스터 내 TSP 최적화 → 동선 최적화
    """
    # 1. 데이터 준비
    df = pd.DataFrame([loc.dict() for loc in body.locations])
    
    if len(df) < 2:
        return {"status": "error", "message": "Not enough locations"}
    
    num_vehicles = body.num_vehicles
    depot_coords = (df.iloc[0]['lat'], df.iloc[0]['lon'])
    
    # 2. 섹터 클러스터링 (방위각 기반)
    df = create_sector_clusters(df, num_vehicles, depot_idx=0)
    
    # 3. 각 클러스터별 TSP 최적화
    results = []
    stats = []
    
    for cluster_id in range(num_vehicles):
        cluster_df = df[df['cluster'] == cluster_id].reset_index(drop=True)
        
        if len(cluster_df) == 0:
            stats.append({
                "driver": f"기사 {cluster_id + 1}",
                "count": 0,
                "distance_km": 0
            })
            continue
        
        # TSP 최적화
        optimized_order = solve_tsp_within_cluster(cluster_df, depot_coords)
        
        # 결과 저장
        for node_id in optimized_order:
            results.append({
                "id": node_id,
                "driver": f"기사 {cluster_id + 1}"
            })
        
        # 통계 계산
        total_distance = 0
        prev_lat, prev_lon = depot_coords
        
        for node_id in optimized_order:
            node_data = cluster_df[cluster_df['id'] == node_id].iloc[0]
            total_distance += haversine(prev_lat, prev_lon, node_data['lat'], node_data['lon'])
            prev_lat, prev_lon = node_data['lat'], node_data['lon']
        
        # depot 복귀 거리 추가
        total_distance += haversine(prev_lat, prev_lon, depot_coords[0], depot_coords[1])
        
        stats.append({
            "driver": f"기사 {cluster_id + 1}",
            "count": len(optimized_order),
            "distance_km": round(total_distance, 2)
        })
    
    return {
        "status": "success",
        "updates": results,
        "statistics": stats,
        "total_assigned": len(results),
        "total_locations": len(df) - 1  # depot 제외
    }


@app.post("/optimize_v2")
def optimize_routes_v2(body: RequestBody):
    """
    대안: OR-Tools 단일 VRP (제약조건 강화 버전)
    - 모든 차량 강제 사용
    - 구역 분리 강화
    """
    # 1. 데이터 준비
    df = pd.DataFrame([loc.dict() for loc in body.locations])
    
    if len(df) < 2:
        return {"status": "error", "message": "Not enough locations"}
    
    num_vehicles = body.num_vehicles
    num_locations = len(df)
    
    # 2. 거리 행렬 생성
    dist_matrix = create_distance_matrix(df)
    
    # 3. OR-Tools 설정
    manager = pywrapcp.RoutingIndexManager(num_locations, num_vehicles, 0)
    routing = pywrapcp.RoutingModel(manager)
    
    # [비용] 거리 비용
    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return int(dist_matrix[from_node][to_node])
    
    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
    
    # [제약 1] 미배정 절대 방지 - 인덱스 정확히 처리
    penalty = 100000000000  # 1000억
    for node_index in range(1, num_locations):
        index = manager.NodeToIndex(node_index)
        if index >= 0:  # 유효한 인덱스만
            routing.AddDisjunction([index], penalty)
    
    # [제약 2] 건수 균등화 + 모든 차량 사용 강제
    def count_callback(from_index):
        return 1
    
    count_callback_index = routing.RegisterUnaryTransitCallback(count_callback)
    routing.AddDimension(count_callback_index, 0, 200, True, 'Count')
    count_dimension = routing.GetDimensionOrDie('Count')
    
    # 평균 건수 계산 (depot 제외)
    avg_count = (num_locations - 1) / num_vehicles
    min_count = max(1, int(avg_count * 0.5))  # 최소: 평균의 50%
    max_count = int(avg_count * 1.5) + 1      # 최대: 평균의 150%
    
    for i in range(num_vehicles):
        end_index = routing.End(i)
        # Hard constraint: 최소 건수 강제
        routing.solver().Add(count_dimension.CumulVar(end_index) >= min_count)
        # Soft constraint: 최대 건수 초과 시 벌점
        count_dimension.SetCumulVarSoftUpperBound(end_index, max_count, 10000000)
    
    # 건수 격차 최소화
    count_dimension.SetGlobalSpanCostCoefficient(100000)
    
    # [제약 3] 거리 균등화 (구역 분리 유도)
    routing.AddDimension(
        transit_callback_index,
        0,
        1000000,  # 최대 이동 거리 (1000km)
        True,
        'Distance'
    )
    dist_dimension = routing.GetDimensionOrDie('Distance')
    dist_dimension.SetGlobalSpanCostCoefficient(50000)  # 강화
    
    # 4. 검색 전략
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PARALLEL_CHEAPEST_INSERTION)
    search_parameters.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH)
    search_parameters.time_limit.seconds = 60
    
    solution = routing.SolveWithParameters(search_parameters)
    
    # 5. 결과 반환
    if not solution:
        return {"status": "fail", "message": "Solution not found"}
    
    results = []
    stats = []
    
    for vehicle_id in range(num_vehicles):
        route = []
        total_distance = 0
        index = routing.Start(vehicle_id)
        prev_node = 0
        
        while not routing.IsEnd(index):
            node_index = manager.IndexToNode(index)
            if node_index != 0:
                route.append(df.iloc[node_index]['id'])
                total_distance += dist_matrix[prev_node][node_index]
                prev_node = node_index
            index = solution.Value(routing.NextVar(index))
        
        # depot 복귀 거리
        total_distance += dist_matrix[prev_node][0]
        
        for node_id in route:
            results.append({
                "id": node_id,
                "driver": f"기사 {vehicle_id + 1}"
            })
        
        stats.append({
            "driver": f"기사 {vehicle_id + 1}",
            "count": len(route),
            "distance_km": round(total_distance / 1000, 2)
        })
    
    return {
        "status": "success",
        "updates": results,
        "statistics": stats,
        "total_assigned": len(results),
        "total_locations": num_locations - 1
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
