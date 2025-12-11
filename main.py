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


def kmeans_plusplus_init(coords, num_clusters):
    """K-Means++ 초기화: 더 나은 초기 중심점 선택"""
    n = len(coords)
    centroids = []
    
    # 첫 번째 중심점: 랜덤 선택
    first_idx = np.random.randint(0, n)
    centroids.append(coords[first_idx])
    
    for _ in range(1, num_clusters):
        # 각 점에서 가장 가까운 중심점까지의 거리 계산
        min_distances = []
        for i in range(n):
            min_dist = float('inf')
            for c in centroids:
                dist = haversine(coords[i][0], coords[i][1], c[0], c[1])
                min_dist = min(min_dist, dist)
            min_distances.append(min_dist)
        
        # 거리의 제곱에 비례하는 확률로 다음 중심점 선택
        min_distances = np.array(min_distances)
        probs = min_distances ** 2
        probs = probs / probs.sum()
        
        next_idx = np.random.choice(n, p=probs)
        centroids.append(coords[next_idx])
    
    return np.array(centroids)


def kmeans_clustering(coords, num_clusters, max_iter=100):
    """K-Means++ 클러스터링"""
    n = len(coords)
    
    if n <= num_clusters:
        return list(range(n))
    
    # K-Means++ 초기화
    np.random.seed(42)  # 재현성
    centroids = kmeans_plusplus_init(coords, num_clusters)
    
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
        
        if np.allclose(centroids, new_centroids, atol=1e-6):
            break
        
        centroids = new_centroids
    
    return labels.tolist(), centroids


def reassign_outliers(df, num_clusters):
    """
    Outlier 재배정: 클러스터 중심에서 너무 먼 점은 더 가까운 클러스터로 이동
    - 파란색처럼 두 지역에 걸친 클러스터 문제 해결
    """
    df = df.copy()
    
    # 클러스터 중심점 계산
    centroids = {}
    for c in range(num_clusters):
        cluster_points = df[df['cluster'] == c][['lat', 'lon']].values
        if len(cluster_points) > 0:
            centroids[c] = cluster_points.mean(axis=0)
    
    # 각 클러스터의 평균 거리(반경) 계산
    cluster_radius = {}
    for c in range(num_clusters):
        cluster_df = df[df['cluster'] == c]
        if len(cluster_df) > 0:
            distances = []
            for _, row in cluster_df.iterrows():
                dist = haversine(row['lat'], row['lon'], 
                               centroids[c][0], centroids[c][1])
                distances.append(dist)
            cluster_radius[c] = np.mean(distances) if distances else 0
    
    # Outlier 감지 및 재배정
    changed = True
    iterations = 0
    max_iterations = 10
    
    while changed and iterations < max_iterations:
        changed = False
        iterations += 1
        
        for idx, row in df.iterrows():
            if row['cluster'] == -1:
                continue
            
            current_cluster = int(row['cluster'])
            current_dist = haversine(row['lat'], row['lon'],
                                    centroids[current_cluster][0], 
                                    centroids[current_cluster][1])
            
            # 현재 클러스터 평균 반경의 2배 이상 떨어져 있으면 outlier
            threshold = cluster_radius.get(current_cluster, 10) * 2
            
            if current_dist > threshold:
                # 더 가까운 클러스터 찾기
                best_cluster = current_cluster
                min_dist = current_dist
                
                for c in range(num_clusters):
                    if c != current_cluster and c in centroids:
                        dist = haversine(row['lat'], row['lon'],
                                       centroids[c][0], centroids[c][1])
                        # 다른 클러스터가 현재보다 50% 이상 가까우면 이동
                        if dist < min_dist * 0.7:
                            min_dist = dist
                            best_cluster = c
                
                if best_cluster != current_cluster:
                    df.loc[idx, 'cluster'] = best_cluster
                    changed = True
        
        # 중심점 및 반경 재계산
        if changed:
            for c in range(num_clusters):
                cluster_points = df[df['cluster'] == c][['lat', 'lon']].values
                if len(cluster_points) > 0:
                    centroids[c] = cluster_points.mean(axis=0)
                    distances = []
                    cluster_df = df[df['cluster'] == c]
                    for _, row in cluster_df.iterrows():
                        dist = haversine(row['lat'], row['lon'], 
                                       centroids[c][0], centroids[c][1])
                        distances.append(dist)
                    cluster_radius[c] = np.mean(distances) if distances else 0
    
    return df


def balance_clusters(df, num_clusters, target_variance=3):
    """클러스터 크기 균형 조정"""
    df = df.copy()
    
    non_depot = df[df['cluster'] != -1]
    target_size = len(non_depot) // num_clusters
    max_size = target_size + target_variance
    min_size = max(1, target_size - target_variance)
    
    # 클러스터 중심점
    centroids = {}
    for c in range(num_clusters):
        cluster_points = df[df['cluster'] == c][['lat', 'lon']].values
        if len(cluster_points) > 0:
            centroids[c] = cluster_points.mean(axis=0)
    
    for _ in range(20):
        cluster_sizes = df[df['cluster'] != -1]['cluster'].value_counts().to_dict()
        
        balanced = True
        for c in range(num_clusters):
            size = cluster_sizes.get(c, 0)
            
            if size > max_size:
                balanced = False
                cluster_df = df[df['cluster'] == c]
                
                # 중심에서 가장 먼 점 찾기
                max_dist = -1
                farthest_idx = None
                for idx, row in cluster_df.iterrows():
                    if c not in centroids:
                        continue
                    dist = haversine(row['lat'], row['lon'], 
                                   centroids[c][0], centroids[c][1])
                    if dist > max_dist:
                        max_dist = dist
                        farthest_idx = idx
                
                if farthest_idx is not None:
                    point = df.loc[farthest_idx, ['lat', 'lon']].values
                    min_dist = float('inf')
                    best_cluster = c
                    
                    for other_c in range(num_clusters):
                        other_size = cluster_sizes.get(other_c, 0)
                        if other_c != c and other_size < max_size and other_c in centroids:
                            dist = haversine(point[0], point[1],
                                           centroids[other_c][0], centroids[other_c][1])
                            if dist < min_dist:
                                min_dist = dist
                                best_cluster = other_c
                    
                    if best_cluster != c:
                        df.loc[farthest_idx, 'cluster'] = best_cluster
                        # 중심점 재계산
                        for cc in [c, best_cluster]:
                            cluster_points = df[df['cluster'] == cc][['lat', 'lon']].values
                            if len(cluster_points) > 0:
                                centroids[cc] = cluster_points.mean(axis=0)
        
        if balanced:
            break
    
    return df


def create_geo_clusters(df, num_vehicles, depot_idx=0):
    """K-Means++ 지리적 클러스터링 + Outlier 재배정"""
    df = df.copy()
    df['cluster'] = -1  # 초기화
    
    # depot 제외
    non_depot_indices = [i for i in range(len(df)) if i != depot_idx]
    
    if len(non_depot_indices) == 0:
        return df
    
    coords = df.loc[non_depot_indices, ['lat', 'lon']].values
    
    # K-Means++ 클러스터링
    labels, _ = kmeans_clustering(coords, num_vehicles)
    
    # 라벨 할당
    for i, idx in enumerate(non_depot_indices):
        df.loc[idx, 'cluster'] = labels[i]
    
    # Outlier 재배정 (핵심!)
    df = reassign_outliers(df, num_vehicles)
    
    # 균형 조정
    df = balance_clusters(df, num_vehicles)
    
    return df


def solve_tsp_within_cluster(cluster_df, depot_coords):
    """클러스터 내 TSP 최적화"""
    if len(cluster_df) == 0:
        return []
    
    if len(cluster_df) == 1:
        return [cluster_df.iloc[0]['id']]
    
    # depot 추가
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
    dist_matrix = create_distance_matrix(temp_df)
    
    # OR-Tools
    manager = pywrapcp.RoutingIndexManager(len(temp_df), 1, 0)
    routing = pywrapcp.RoutingModel(manager)
    
    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return int(dist_matrix[from_node][to_node])
    
    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
    
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
    search_parameters.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH)
    search_parameters.time_limit.seconds = 5
    
    solution = routing.SolveWithParameters(search_parameters)
    
    if not solution:
        return nearest_neighbor_order(cluster_df, depot_coords)
    
    route = []
    index = routing.Start(0)
    while not routing.IsEnd(index):
        node_index = manager.IndexToNode(index)
        if node_index != 0:
            route.append(temp_df.iloc[node_index]['id'])
        index = solution.Value(routing.NextVar(index))
    
    return route


def nearest_neighbor_order(cluster_df, depot_coords):
    """Nearest Neighbor 휴리스틱"""
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
    return {"status": "active", "message": "VRP Engine V7 (K-Means++ with Outlier Reassignment)"}


@app.post("/optimize")
def optimize_routes(body: RequestBody):
    """
    3단계 최적화:
    1) K-Means++ 클러스터링
    2) Outlier 재배정 (두 지역에 걸친 클러스터 해결)
    3) 각 클러스터 내 TSP 최적화
    
    ★ 미배정 0건 절대 보장
    """
    # 1. 데이터 준비
    data = [loc.dict() for loc in body.locations]
    df = pd.DataFrame(data)
    df = df.reset_index(drop=True)
    
    total_locations = len(df) - 1  # depot 제외
    
    if total_locations < 1:
        return {"status": "error", "message": "Not enough locations"}
    
    num_vehicles = body.num_vehicles
    depot_coords = (df.iloc[0]['lat'], df.iloc[0]['lon'])
    
    # 2. 클러스터링 (K-Means++ + Outlier 재배정)
    df = create_geo_clusters(df, num_vehicles, depot_idx=0)
    
    # ★★★ 미배정 강제 할당 (절대 미배정 없음) ★★★
    unassigned_mask = (df['cluster'] == -1) & (df.index != 0)
    unassigned_indices = df[unassigned_mask].index.tolist()
    
    if len(unassigned_indices) > 0:
        # 각 클러스터 중심 계산
        centroids = {}
        for c in range(num_vehicles):
            cluster_points = df[df['cluster'] == c][['lat', 'lon']].values
            if len(cluster_points) > 0:
                centroids[c] = cluster_points.mean(axis=0)
            else:
                # 빈 클러스터면 depot 위치 사용
                centroids[c] = np.array(depot_coords)
        
        for idx in unassigned_indices:
            row = df.loc[idx]
            min_dist = float('inf')
            best_cluster = 0
            
            for c in range(num_vehicles):
                dist = haversine(row['lat'], row['lon'], 
                               centroids[c][0], centroids[c][1])
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
        
        optimized_order = solve_tsp_within_cluster(cluster_df, depot_coords)
        
        for node_id in optimized_order:
            results.append({
                "id": node_id,
                "driver": f"기사 {cluster_id + 1}"
            })
        
        # 통계
        total_distance = 0
        prev_lat, prev_lon = depot_coords
        
        for node_id in optimized_order:
            node_row = cluster_df[cluster_df['id'] == node_id]
            if len(node_row) > 0:
                node_data = node_row.iloc[0]
                total_distance += haversine(prev_lat, prev_lon, 
                                          node_data['lat'], node_data['lon'])
                prev_lat, prev_lon = node_data['lat'], node_data['lon']
        
        total_distance += haversine(prev_lat, prev_lon, depot_coords[0], depot_coords[1])
        
        stats.append({
            "driver": f"기사 {cluster_id + 1}",
            "count": len(optimized_order),
            "distance_km": round(total_distance, 2)
        })
    
    # 4. 최종 검증
    total_assigned = len(results)
    unassigned_count = total_locations - total_assigned
    
    # 만약 아직도 미배정이 있으면 에러 로그
    if unassigned_count > 0:
        # 디버깅: 어떤 점이 미배정인지 확인
        assigned_ids = set([r['id'] for r in results])
        all_ids = set(df[df.index != 0]['id'].tolist())
        missing_ids = all_ids - assigned_ids
        
        return {
            "status": "warning",
            "message": f"미배정 {unassigned_count}건 발생",
            "missing_ids": list(missing_ids),
            "updates": results,
            "statistics": stats,
            "total_assigned": total_assigned,
            "total_locations": total_locations,
            "unassigned": unassigned_count
        }
    
    return {
        "status": "success",
        "updates": results,
        "statistics": stats,
        "total_assigned": total_assigned,
        "total_locations": total_locations,
        "unassigned": 0
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
