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


def kmeans_clustering(coords, num_clusters, max_iter=100):
    """
    순수 NumPy K-Means 구현 (sklearn 의존성 제거)
    - 지리적 근접성 기반 클러스터링
    - 가까운 점들끼리 자연스럽게 묶임
    """
    n = len(coords)
    
    if n <= num_clusters:
        return list(range(n))
    
    # 초기 중심점: 데이터에서 균등하게 선택
    indices = np.linspace(0, n - 1, num_clusters, dtype=int)
    centroids = coords[indices].copy()
    
    labels = np.zeros(n, dtype=int)
    
    for _ in range(max_iter):
        # 각 점을 가장 가까운 중심점에 할당
        for i in range(n):
            min_dist = float('inf')
            for j in range(num_clusters):
                dist = haversine(coords[i][0], coords[i][1], 
                               centroids[j][0], centroids[j][1])
                if dist < min_dist:
                    min_dist = dist
                    labels[i] = j
        
        # 중심점 업데이트
        new_centroids = np.zeros_like(centroids)
        for j in range(num_clusters):
            cluster_points = coords[labels == j]
            if len(cluster_points) > 0:
                new_centroids[j] = cluster_points.mean(axis=0)
            else:
                new_centroids[j] = centroids[j]
        
        # 수렴 체크
        if np.allclose(centroids, new_centroids, atol=1e-6):
            break
        
        centroids = new_centroids
    
    return labels.tolist()


def balance_clusters(df, labels, num_clusters):
    """
    클러스터 크기 균형 조정
    - 너무 큰 클러스터에서 가장자리 점을 인접 클러스터로 이동
    """
    df = df.copy()
    df['cluster'] = labels
    
    target_size = len(df) // num_clusters
    max_size = target_size + 2
    min_size = max(1, target_size - 2)
    
    # 클러스터 중심점 계산
    centroids = {}
    for c in range(num_clusters):
        cluster_points = df[df['cluster'] == c][['lat', 'lon']].values
        if len(cluster_points) > 0:
            centroids[c] = cluster_points.mean(axis=0)
    
    # 반복적으로 균형 조정
    for _ in range(10):
        cluster_sizes = df['cluster'].value_counts().to_dict()
        
        balanced = True
        for c in range(num_clusters):
            size = cluster_sizes.get(c, 0)
            if size > max_size:
                balanced = False
                # 가장 먼 점을 찾아서 다른 클러스터로 이동
                cluster_df = df[df['cluster'] == c]
                
                # 중심에서 가장 먼 점 찾기
                max_dist = -1
                farthest_idx = None
                for idx, row in cluster_df.iterrows():
                    dist = haversine(row['lat'], row['lon'], 
                                   centroids[c][0], centroids[c][1])
                    if dist > max_dist:
                        max_dist = dist
                        farthest_idx = idx
                
                if farthest_idx is not None:
                    # 가장 가까운 다른 클러스터 찾기
                    point = df.loc[farthest_idx, ['lat', 'lon']].values
                    min_dist = float('inf')
                    best_cluster = c
                    
                    for other_c in range(num_clusters):
                        if other_c != c and cluster_sizes.get(other_c, 0) < max_size:
                            dist = haversine(point[0], point[1],
                                           centroids[other_c][0], centroids[other_c][1])
                            if dist < min_dist:
                                min_dist = dist
                                best_cluster = other_c
                    
                    if best_cluster != c:
                        df.loc[farthest_idx, 'cluster'] = best_cluster
        
        if balanced:
            break
    
    return df['cluster'].tolist()


def create_geo_clusters(df, num_vehicles, depot_idx=0):
    """
    K-Means 기반 지리적 클러스터링
    - 가까운 점들끼리 자연스럽게 묶임
    - 클러스터 크기 균형 조정 포함
    """
    # depot 제외한 좌표
    non_depot_mask = df.index != depot_idx
    non_depot_df = df[non_depot_mask].copy()
    
    if len(non_depot_df) == 0:
        df = df.copy()
        df['cluster'] = -1
        return df
    
    coords = non_depot_df[['lat', 'lon']].values
    
    # K-Means 클러스터링
    labels = kmeans_clustering(coords, num_vehicles)
    
    # 균형 조정
    non_depot_df['cluster'] = labels
    balanced_labels = balance_clusters(non_depot_df, labels, num_vehicles)
    non_depot_df['cluster'] = balanced_labels
    
    # 전체 DataFrame에 클러스터 할당
    df = df.copy()
    df['cluster'] = -1  # 기본값 (depot)
    
    for idx, cluster in zip(non_depot_df.index, non_depot_df['cluster']):
        df.loc[idx, 'cluster'] = cluster
    
    return df


def solve_tsp_within_cluster(cluster_df, depot_coords):
    """
    클러스터 내 TSP 최적화 (OR-Tools 사용)
    """
    if len(cluster_df) == 0:
        return []
    
    if len(cluster_df) == 1:
        return [cluster_df.iloc[0]['id']]
    
    # depot을 임시로 추가 (인덱스 0)
    temp_data = [{
        'id': '__depot__',
        'lat': depot_coords[0],
        'lon': depot_coords[1]
    }]
    
    for _, row in cluster_df.iterrows():
        temp_data.append({
            'id': row['id'],
            'lat': row['lat'],
            'lon': row['lon']
        })
    
    temp_df = pd.DataFrame(temp_data)
    
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
        # 풀이 실패 시 Nearest Neighbor
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
    return {"status": "active", "message": "VRP Engine V6 (K-Means Geo Clustering)"}


@app.post("/optimize")
def optimize_routes(body: RequestBody):
    """
    2단계 최적화:
    1) K-Means 지리적 클러스터링 → 가까운 점끼리 묶임
    2) 각 클러스터 내 TSP 최적화 → 동선 최적화
    
    ★ 미배정 0건 보장
    """
    # 1. 데이터 준비
    df = pd.DataFrame([loc.dict() for loc in body.locations])
    df = df.reset_index(drop=True)  # 인덱스 정리
    
    if len(df) < 2:
        return {"status": "error", "message": "Not enough locations"}
    
    num_vehicles = body.num_vehicles
    depot_coords = (df.iloc[0]['lat'], df.iloc[0]['lon'])
    
    # 2. K-Means 지리적 클러스터링
    df = create_geo_clusters(df, num_vehicles, depot_idx=0)
    
    # ★ 미배정 체크 및 강제 할당
    unassigned = df[(df['cluster'] == -1) & (df.index != 0)]
    if len(unassigned) > 0:
        # 미배정 점들을 가장 가까운 클러스터에 강제 할당
        for idx, row in unassigned.iterrows():
            min_dist = float('inf')
            best_cluster = 0
            
            for c in range(num_vehicles):
                cluster_points = df[df['cluster'] == c][['lat', 'lon']].values
                if len(cluster_points) > 0:
                    centroid = cluster_points.mean(axis=0)
                    dist = haversine(row['lat'], row['lon'], centroid[0], centroid[1])
                    if dist < min_dist:
                        min_dist = dist
                        best_cluster = c
            
            df.loc[idx, 'cluster'] = best_cluster
    
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
            node_row = cluster_df[cluster_df['id'] == node_id]
            if len(node_row) > 0:
                node_data = node_row.iloc[0]
                total_distance += haversine(prev_lat, prev_lon, node_data['lat'], node_data['lon'])
                prev_lat, prev_lon = node_data['lat'], node_data['lon']
        
        # depot 복귀 거리 추가
        total_distance += haversine(prev_lat, prev_lon, depot_coords[0], depot_coords[1])
        
        stats.append({
            "driver": f"기사 {cluster_id + 1}",
            "count": len(optimized_order),
            "distance_km": round(total_distance, 2)
        })
    
    # 4. 최종 검증
    total_assigned = len(results)
    total_locations = len(df) - 1  # depot 제외
    
    return {
        "status": "success",
        "updates": results,
        "statistics": stats,
        "total_assigned": total_assigned,
        "total_locations": total_locations,
        "unassigned": total_locations - total_assigned  # 항상 0이어야 함
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
