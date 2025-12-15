from fastapi import FastAPI
from pydantic import BaseModel
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
import pandas as pd
import numpy as np
import traceback
from typing import Optional, List
from sklearn.cluster import KMeans
from scipy.spatial import ConvexHull
from shapely.geometry import Point, Polygon

app = FastAPI()


# --- 상수 정의 ---
VEHICLE_CAPACITY_KG = 1200
DEFAULT_MAX_CAPA = 25
MIN_CALLS_SOFT = 10
DEFAULT_WEIGHT_KG = 15
CORE_NODE_RATIO = 0.9  # ★ V10.4: 70% → 90%로 상향


# --- 데이터 모델 ---
class Location(BaseModel):
    id: str
    lat: float
    lon: float
    weight: int = DEFAULT_WEIGHT_KG


class Driver(BaseModel):
    id: str
    name: str
    max_capa: Optional[int] = DEFAULT_MAX_CAPA
    base_lat: Optional[float] = None
    base_lng: Optional[float] = None
    vehicle_capacity_kg: Optional[int] = VEHICLE_CAPACITY_KG


class RequestBody(BaseModel):
    locations: List[Location]
    drivers: Optional[List[Driver]] = None
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
                dist_matrix[i][j] = int(dist_km * 1000)
    
    return dist_matrix


def calculate_cluster_centroid(df, assignments, vehicle_id):
    """특정 클러스터의 중심점 계산"""
    assigned_indices = [idx for idx, vid in assignments.items() if vid == vehicle_id]
    if not assigned_indices:
        return None
    
    lats = [float(df.iloc[idx]['lat']) for idx in assigned_indices]
    lons = [float(df.iloc[idx]['lon']) for idx in assigned_indices]
    
    return (np.mean(lats), np.mean(lons))


def get_cluster_stats(df, assignments, vehicle_id):
    """특정 클러스터 통계"""
    assigned_indices = [idx for idx, vid in assignments.items() if vid == vehicle_id]
    
    total_weight = sum(int(df.iloc[idx]['weight']) for idx in assigned_indices)
    call_count = len(assigned_indices)
    
    return {
        'indices': list(assigned_indices),
        'call_count': call_count,
        'total_weight': total_weight
    }


def create_geographic_clusters(df, num_clusters, depot_idx=0):
    """
    K-Means 기반 지리적 클러스터링
    """
    coords = []
    node_indices = []
    
    for i in range(len(df)):
        if i == depot_idx:
            continue
        coords.append([df.iloc[i]['lat'], df.iloc[i]['lon']])
        node_indices.append(i)
    
    if len(coords) < num_clusters:
        return {idx: i % num_clusters for i, idx in enumerate(node_indices)}, None
    
    coords_array = np.array(coords)
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(coords_array)
    
    node_to_cluster = {}
    for i, node_idx in enumerate(node_indices):
        node_to_cluster[node_idx] = int(labels[i])
    
    return node_to_cluster, kmeans.cluster_centers_


def create_convex_hulls(df, node_to_cluster, num_clusters, depot_idx=0):
    """
    ★ V10.4 신규: 각 클러스터의 Convex Hull(볼록 껍질) 생성 ★
    
    목적:
    - 각 클러스터의 지리적 경계를 명확하게 정의
    - 경계 내부/외부 판정에 사용
    
    Returns:
        cluster_hulls: {cluster_id: Polygon} 매핑
    """
    cluster_hulls = {}
    
    for cluster_id in range(num_clusters):
        # 해당 클러스터의 노드 좌표 수집
        cluster_coords = []
        for node_idx, cid in node_to_cluster.items():
            if cid == cluster_id:
                lat = float(df.iloc[node_idx]['lat'])
                lon = float(df.iloc[node_idx]['lon'])
                cluster_coords.append([lon, lat])  # Shapely는 (x, y) = (lon, lat)
        
        if len(cluster_coords) < 3:
            # 3개 미만이면 Convex Hull 생성 불가 → 버퍼로 대체
            if len(cluster_coords) == 1:
                point = Point(cluster_coords[0])
                cluster_hulls[cluster_id] = point.buffer(0.01)  # 약 1km 버퍼
            elif len(cluster_coords) == 2:
                from shapely.geometry import LineString
                line = LineString(cluster_coords)
                cluster_hulls[cluster_id] = line.buffer(0.005)
            continue
        
        try:
            coords_array = np.array(cluster_coords)
            hull = ConvexHull(coords_array)
            hull_points = coords_array[hull.vertices]
            polygon = Polygon(hull_points)
            
            # 약간의 버퍼 추가 (경계에 걸친 노드 포함)
            cluster_hulls[cluster_id] = polygon.buffer(0.002)  # 약 200m 버퍼
            
            print(f"  클러스터 {cluster_id}: Convex Hull 생성 (노드 {len(cluster_coords)}개, 꼭짓점 {len(hull.vertices)}개)")
        except Exception as e:
            print(f"  클러스터 {cluster_id}: Convex Hull 생성 실패 - {e}")
            # 실패 시 모든 점을 포함하는 큰 원으로 대체
            center = np.mean(coords_array, axis=0)
            cluster_hulls[cluster_id] = Point(center).buffer(0.05)
    
    return cluster_hulls


def assign_node_to_best_cluster(df, node_idx, cluster_hulls, node_to_cluster):
    """
    ★ V10.4 신규: Convex Hull 기반으로 노드의 최적 클러스터 결정 ★
    
    로직:
    1. 노드가 어떤 Hull 내부에 있는지 확인
    2. 여러 Hull에 속하면 (겹치는 영역) → 중심에 가장 가까운 클러스터
    3. 어느 Hull에도 안 속하면 → 가장 가까운 Hull의 클러스터
    """
    lat = float(df.iloc[node_idx]['lat'])
    lon = float(df.iloc[node_idx]['lon'])
    point = Point(lon, lat)
    
    # 1. 어떤 Hull에 속하는지 확인
    containing_clusters = []
    for cluster_id, hull in cluster_hulls.items():
        if hull.contains(point):
            containing_clusters.append(cluster_id)
    
    if len(containing_clusters) == 1:
        return containing_clusters[0]
    
    if len(containing_clusters) > 1:
        # 여러 Hull에 속함 (겹치는 영역) → 원래 K-Means 할당 유지
        return node_to_cluster.get(node_idx, containing_clusters[0])
    
    # 2. 어느 Hull에도 안 속함 → 가장 가까운 Hull
    min_dist = float('inf')
    best_cluster = 0
    
    for cluster_id, hull in cluster_hulls.items():
        dist = point.distance(hull)
        if dist < min_dist:
            min_dist = dist
            best_cluster = cluster_id
    
    return best_cluster


def enforce_convex_hull_boundaries(df, node_to_cluster, cluster_hulls, num_clusters, depot_idx=0):
    """
    ★ V10.4 신규: Convex Hull 경계 기반으로 클러스터 재할당 ★
    
    목적:
    - K-Means 결과를 Convex Hull 경계로 보정
    - 경계 밖에 있는 노드를 올바른 클러스터로 이동
    """
    print("\n=== Convex Hull 경계 보정 ===")
    
    reassigned_count = 0
    new_assignments = dict(node_to_cluster)
    
    for node_idx in node_to_cluster.keys():
        if node_idx == depot_idx:
            continue
        
        original_cluster = node_to_cluster[node_idx]
        best_cluster = assign_node_to_best_cluster(df, node_idx, cluster_hulls, node_to_cluster)
        
        if best_cluster != original_cluster:
            new_assignments[node_idx] = best_cluster
            reassigned_count += 1
            print(f"  노드 {node_idx}: 클러스터 {original_cluster} → {best_cluster}")
    
    print(f"  Convex Hull 보정으로 {reassigned_count}개 노드 재할당")
    
    return new_assignments


def match_clusters_to_drivers_v2(df, assignments, drivers, num_clusters, depot_idx=0):
    """
    V10.2: max_capa를 고려한 클러스터-기사 매칭
    """
    cluster_stats = {}
    for cluster_id in range(num_clusters):
        stats = get_cluster_stats(df, assignments, cluster_id)
        cluster_stats[cluster_id] = stats
        print(f"  클러스터 {cluster_id}: {stats['call_count']}건, {stats['total_weight']}kg")
    
    cluster_centroids = {}
    for cluster_id in range(num_clusters):
        centroid = calculate_cluster_centroid(df, assignments, cluster_id)
        if centroid:
            cluster_centroids[cluster_id] = centroid
    
    driver_info = {}
    for i, driver in enumerate(drivers):
        max_capa = driver.max_capa if driver.max_capa else DEFAULT_MAX_CAPA
        if driver.base_lat is not None and driver.base_lng is not None:
            base = (driver.base_lat, driver.base_lng)
        else:
            base = (float(df.iloc[depot_idx]['lat']), float(df.iloc[depot_idx]['lon']))
        
        driver_info[i] = {
            'name': driver.name,
            'max_capa': max_capa,
            'base': base
        }
        print(f"  기사 {i} ({driver.name}): max_capa={max_capa}")
    
    cluster_to_driver = {}
    used_drivers = set()
    
    sorted_clusters = sorted(
        cluster_stats.keys(),
        key=lambda c: cluster_stats[c]['call_count'],
        reverse=True
    )
    
    print(f"\n  클러스터 처리 순서 (콜 수 내림차순): {sorted_clusters}")
    
    for cluster_id in sorted_clusters:
        if cluster_id not in cluster_centroids:
            continue
        
        cluster_calls = cluster_stats[cluster_id]['call_count']
        centroid = cluster_centroids[cluster_id]
        
        candidates = []
        for driver_id, info in driver_info.items():
            if driver_id in used_drivers:
                continue
            
            if info['max_capa'] >= cluster_calls:
                dist = haversine(centroid[0], centroid[1], info['base'][0], info['base'][1])
                candidates.append((driver_id, dist, info['max_capa']))
        
        if candidates:
            candidates.sort(key=lambda x: x[1])
            best_driver = candidates[0][0]
            print(f"  클러스터 {cluster_id} ({cluster_calls}건) → 기사 {best_driver} ({driver_info[best_driver]['name']}) [거리 기반]")
        else:
            remaining = [(d, info['max_capa']) for d, info in driver_info.items() if d not in used_drivers]
            if remaining:
                remaining.sort(key=lambda x: -x[1])
                best_driver = remaining[0][0]
                print(f"  클러스터 {cluster_id} ({cluster_calls}건) → 기사 {best_driver} ({driver_info[best_driver]['name']}) [차선책]")
            else:
                best_driver = 0
        
        cluster_to_driver[cluster_id] = best_driver
        used_drivers.add(best_driver)
    
    remaining_drivers = set(range(len(drivers))) - used_drivers
    for cluster_id in range(num_clusters):
        if cluster_id not in cluster_to_driver:
            if remaining_drivers:
                cluster_to_driver[cluster_id] = remaining_drivers.pop()
            else:
                cluster_to_driver[cluster_id] = 0
    
    print(f"\n  최종 매칭 결과: {cluster_to_driver}")
    return cluster_to_driver


def optimize_visit_order(df, assignments, cluster_id, depot_idx=0):
    """클러스터별 방문 순서 최적화 (Nearest Neighbor)"""
    assigned_indices = [idx for idx, vid in assignments.items() if vid == cluster_id and idx != depot_idx]
    
    if len(assigned_indices) <= 1:
        return assigned_indices
    
    try:
        depot_lat = float(df.iloc[depot_idx]['lat'])
        depot_lon = float(df.iloc[depot_idx]['lon'])
        
        visited = []
        remaining = set(assigned_indices)
        current_lat, current_lon = depot_lat, depot_lon
        
        while remaining:
            nearest = min(remaining, key=lambda idx: haversine(
                current_lat, current_lon,
                float(df.iloc[idx]['lat']), float(df.iloc[idx]['lon'])
            ))
            visited.append(nearest)
            remaining.remove(nearest)
            current_lat = float(df.iloc[nearest]['lat'])
            current_lon = float(df.iloc[nearest]['lon'])
        
        return visited
        
    except Exception as e:
        print(f"optimize_visit_order error: {e}")
        return assigned_indices


def check_cluster_overlap(df, assignments, num_clusters):
    """클러스터 간 지리적 교차 정도 측정"""
    centroids = {}
    for cluster_id in range(num_clusters):
        centroid = calculate_cluster_centroid(df, assignments, cluster_id)
        if centroid:
            centroids[cluster_id] = centroid
    
    if len(centroids) < 2:
        return 0.0
    
    overlap_count = 0
    total_count = 0
    
    for node_idx, assigned_cluster in assignments.items():
        if assigned_cluster not in centroids:
            continue
        
        node_lat = float(df.iloc[node_idx]['lat'])
        node_lon = float(df.iloc[node_idx]['lon'])
        
        own_centroid = centroids[assigned_cluster]
        own_dist = haversine(node_lat, node_lon, own_centroid[0], own_centroid[1])
        
        min_other_dist = float('inf')
        for other_cluster, other_centroid in centroids.items():
            if other_cluster == assigned_cluster:
                continue
            other_dist = haversine(node_lat, node_lon, other_centroid[0], other_centroid[1])
            min_other_dist = min(min_other_dist, other_dist)
        
        if min_other_dist < own_dist:
            overlap_count += 1
        
        total_count += 1
    
    if total_count == 0:
        return 0.0
    
    return round(overlap_count / total_count * 100, 1)


@app.get("/")
def read_root():
    return {
        "status": "active",
        "message": "VRP Engine V10.4 (Convex Hull Boundary Enforcement)",
        "features": [
            "drivers 필드 선택적",
            "기사별 max_capa 하드캡",
            "K-Means 사전 클러스터링",
            "★ V10.4: 핵심 노드 비율 90%로 상향",
            "★ V10.4: Convex Hull 경계 기반 클러스터 강제",
            "Same Vehicle Constraint로 클러스터 교차 방지",
            "max_capa 고려한 클러스터-기사 매칭"
        ],
        "changelog": {
            "v10.4": "Convex Hull 경계 강제 + 핵심 노드 90%",
            "v10.3": "K-Means + Same Vehicle Constraint",
            "v10.2": "클러스터-기사 매칭 시 max_capa 제약"
        }
    }


@app.post("/optimize")
def optimize_routes(body: RequestBody):
    """
    OR-Tools CVRP + ★ V10.4: Convex Hull 경계 강제 ★
    
    핵심 변경:
    1. K-Means로 초기 클러스터링
    2. 각 클러스터의 Convex Hull 생성
    3. Convex Hull 경계 기반으로 클러스터 재할당
    4. Same Vehicle Constraint (핵심 노드 90%)
    5. OR-Tools 최적화
    """
    
    try:
        # 1. 데이터 준비
        data = [loc.dict() for loc in body.locations]
        df = pd.DataFrame(data)
        df = df.reset_index(drop=True)
        
        num_locations = len(df)
        depot_idx = 0
        
        if num_locations < 2:
            return {"status": "error", "message": "Not enough locations"}
        
        if body.drivers and len(body.drivers) > 0:
            drivers = body.drivers
            num_vehicles = len(drivers)
            use_driver_features = True
        else:
            num_vehicles = body.num_vehicles
            drivers = [
                Driver(
                    id=f"driver_{i+1}",
                    name=f"기사 {i+1}",
                    max_capa=DEFAULT_MAX_CAPA,
                    base_lat=None,
                    base_lng=None,
                    vehicle_capacity_kg=body.vehicle_capacity
                )
                for i in range(num_vehicles)
            ]
            use_driver_features = False
        
        driver_max_capas = []
        driver_kg_capas = []
        for driver in drivers:
            max_capa = driver.max_capa if driver.max_capa else DEFAULT_MAX_CAPA
            kg_capa = driver.vehicle_capacity_kg if driver.vehicle_capacity_kg else body.vehicle_capacity
            driver_max_capas.append(max_capa)
            driver_kg_capas.append(kg_capa)
        
        print(f"\n=== VRP V10.4 최적화 시작 ===")
        print(f"총 위치: {num_locations}개, 기사: {num_vehicles}명")
        print(f"기사별 max_capa: {driver_max_capas}")
        print(f"총 수용 가능: {sum(driver_max_capas)}건")
        
        if 'weight' not in df.columns:
            df['weight'] = DEFAULT_WEIGHT_KG
        df['weight'] = pd.to_numeric(df['weight'], errors='coerce').fillna(DEFAULT_WEIGHT_KG).astype(int)
        df.loc[depot_idx, 'weight'] = 0
        
        # 2. K-Means 사전 클러스터링
        print(f"\n=== 1단계: K-Means 클러스터링 ===")
        node_to_cluster, cluster_centers = create_geographic_clusters(df, num_vehicles, depot_idx)
        
        cluster_node_counts = {}
        for node_idx, cluster_id in node_to_cluster.items():
            cluster_node_counts[cluster_id] = cluster_node_counts.get(cluster_id, 0) + 1
        print(f"K-Means 결과: {cluster_node_counts}")
        
        # 3. ★ V10.4: Convex Hull 생성 ★
        print(f"\n=== 2단계: Convex Hull 생성 ===")
        cluster_hulls = create_convex_hulls(df, node_to_cluster, num_vehicles, depot_idx)
        
        # 4. ★ V10.4: Convex Hull 경계 기반 재할당 ★
        node_to_cluster = enforce_convex_hull_boundaries(df, node_to_cluster, cluster_hulls, num_vehicles, depot_idx)
        
        # 재할당 후 클러스터별 노드 수
        cluster_node_counts_after = {}
        for node_idx, cluster_id in node_to_cluster.items():
            cluster_node_counts_after[cluster_id] = cluster_node_counts_after.get(cluster_id, 0) + 1
        print(f"Convex Hull 보정 후: {cluster_node_counts_after}")
        
        # 5. 거리 행렬
        dist_matrix = create_distance_matrix(df)
        
        # 6. OR-Tools 설정
        manager = pywrapcp.RoutingIndexManager(num_locations, num_vehicles, depot_idx)
        routing = pywrapcp.RoutingModel(manager)
        
        def distance_callback(from_index, to_index):
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            return dist_matrix[from_node][to_node]
        
        transit_callback_index = routing.RegisterTransitCallback(distance_callback)
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
        
        UNASSIGNED_PENALTY = 10000000000
        for node_idx in range(1, num_locations):
            index = manager.NodeToIndex(node_idx)
            routing.AddDisjunction([index], UNASSIGNED_PENALTY)
        
        def demand_callback(from_index):
            from_node = manager.IndexToNode(from_index)
            return int(df.iloc[from_node]['weight'])
        
        demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
        routing.AddDimensionWithVehicleCapacity(
            demand_callback_index, 0, driver_kg_capas, True, 'Capacity')
        
        def count_callback(from_index):
            from_node = manager.IndexToNode(from_index)
            return 0 if from_node == depot_idx else 1
        
        count_callback_index = routing.RegisterUnaryTransitCallback(count_callback)
        routing.AddDimensionWithVehicleCapacity(
            count_callback_index, 0, driver_max_capas, True, 'CallCount')
        
        # 7. ★ V10.4: Same Vehicle Constraint (90% 핵심 노드) ★
        print(f"\n=== 3단계: Same Vehicle Constraint (핵심 노드 {int(CORE_NODE_RATIO*100)}%) ===")
        
        cluster_to_nodes = {}
        for node_idx, cluster_id in node_to_cluster.items():
            if cluster_id not in cluster_to_nodes:
                cluster_to_nodes[cluster_id] = []
            cluster_to_nodes[cluster_id].append(node_idx)
        
        constraints_added = 0
        
        for cluster_id, nodes in cluster_to_nodes.items():
            if len(nodes) < 2:
                continue
            
            # 클러스터 중심점 계산
            center_lat = np.mean([float(df.iloc[n]['lat']) for n in nodes])
            center_lon = np.mean([float(df.iloc[n]['lon']) for n in nodes])
            
            # 중심에 가까운 순으로 정렬
            nodes_with_dist = []
            for node_idx in nodes:
                node_lat = float(df.iloc[node_idx]['lat'])
                node_lon = float(df.iloc[node_idx]['lon'])
                dist = haversine(node_lat, node_lon, center_lat, center_lon)
                nodes_with_dist.append((node_idx, dist))
            
            nodes_with_dist.sort(key=lambda x: x[1])
            
            # ★ V10.4: 90% 핵심 노드 ★
            core_count = max(2, int(len(nodes) * CORE_NODE_RATIO))
            core_nodes = [n[0] for n in nodes_with_dist[:core_count]]
            
            print(f"  클러스터 {cluster_id}: 전체 {len(nodes)}개 중 핵심 {len(core_nodes)}개")
            
            # 체인 방식으로 Same Vehicle Constraint 적용
            for i in range(len(core_nodes) - 1):
                node_a = core_nodes[i]
                node_b = core_nodes[i + 1]
                
                index_a = manager.NodeToIndex(node_a)
                index_b = manager.NodeToIndex(node_b)
                
                routing.AddPickupAndDelivery(index_a, index_b)
                routing.solver().Add(
                    routing.VehicleVar(index_a) == routing.VehicleVar(index_b)
                )
                constraints_added += 1
        
        print(f"  총 Same Vehicle Constraints: {constraints_added}개")
        
        # 콜 수 하한 (소프트)
        count_dimension = routing.GetDimensionOrDie('CallCount')
        CALL_PENALTY = 50000
        
        for vehicle_id in range(num_vehicles):
            end_index = routing.End(vehicle_id)
            count_dimension.SetCumulVarSoftLowerBound(end_index, MIN_CALLS_SOFT, CALL_PENALTY)
        
        # 거리 균등화
        routing.AddDimension(transit_callback_index, 0, 10000000, True, 'Distance')
        distance_dimension = routing.GetDimensionOrDie('Distance')
        distance_dimension.SetGlobalSpanCostCoefficient(200)
        
        # 8. 검색
        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.PARALLEL_CHEAPEST_INSERTION)
        search_parameters.local_search_metaheuristic = (
            routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH)
        search_parameters.time_limit.seconds = 45  # 제약 많아서 시간 증가
        
        print(f"\n=== 4단계: OR-Tools 최적화 ===")
        solution = routing.SolveWithParameters(search_parameters)
        
        if not solution:
            print("⚠️ Same Vehicle Constraint로 해 찾기 실패, Fallback 시도...")
            
            # Fallback: 제약 완화 (70%로 낮추고 재시도)
            manager2 = pywrapcp.RoutingIndexManager(num_locations, num_vehicles, depot_idx)
            routing2 = pywrapcp.RoutingModel(manager2)
            
            transit_callback_index2 = routing2.RegisterTransitCallback(distance_callback)
            routing2.SetArcCostEvaluatorOfAllVehicles(transit_callback_index2)
            
            for node_idx in range(1, num_locations):
                index = manager2.NodeToIndex(node_idx)
                routing2.AddDisjunction([index], UNASSIGNED_PENALTY)
            
            demand_callback_index2 = routing2.RegisterUnaryTransitCallback(demand_callback)
            routing2.AddDimensionWithVehicleCapacity(
                demand_callback_index2, 0, driver_kg_capas, True, 'Capacity')
            
            count_callback_index2 = routing2.RegisterUnaryTransitCallback(count_callback)
            routing2.AddDimensionWithVehicleCapacity(
                count_callback_index2, 0, driver_max_capas, True, 'CallCount')
            
            # 70% 핵심 노드로 재시도
            constraints_added_fallback = 0
            for cluster_id, nodes in cluster_to_nodes.items():
                if len(nodes) < 2:
                    continue
                
                center_lat = np.mean([float(df.iloc[n]['lat']) for n in nodes])
                center_lon = np.mean([float(df.iloc[n]['lon']) for n in nodes])
                
                nodes_with_dist = []
                for node_idx in nodes:
                    node_lat = float(df.iloc[node_idx]['lat'])
                    node_lon = float(df.iloc[node_idx]['lon'])
                    dist = haversine(node_lat, node_lon, center_lat, center_lon)
                    nodes_with_dist.append((node_idx, dist))
                
                nodes_with_dist.sort(key=lambda x: x[1])
                
                core_count = max(2, int(len(nodes) * 0.7))  # 70%로 완화
                core_nodes = [n[0] for n in nodes_with_dist[:core_count]]
                
                for i in range(len(core_nodes) - 1):
                    node_a = core_nodes[i]
                    node_b = core_nodes[i + 1]
                    
                    index_a = manager2.NodeToIndex(node_a)
                    index_b = manager2.NodeToIndex(node_b)
                    
                    routing2.AddPickupAndDelivery(index_a, index_b)
                    routing2.solver().Add(
                        routing2.VehicleVar(index_a) == routing2.VehicleVar(index_b)
                    )
                    constraints_added_fallback += 1
            
            solution = routing2.SolveWithParameters(search_parameters)
            manager = manager2
            routing = routing2
            
            if not solution:
                return {
                    "status": "fail",
                    "message": "Solution not found. 제약 조건 충돌.",
                    "debug": {
                        "total_calls": num_locations - 1,
                        "total_max_capa": sum(driver_max_capas),
                        "constraints_tried": constraints_added
                    }
                }
        
        # 9. 결과 추출
        cluster_assignments = {}
        
        for cluster_id in range(num_vehicles):
            index = routing.Start(cluster_id)
            while not routing.IsEnd(index):
                node_idx = manager.IndexToNode(index)
                if node_idx != depot_idx:
                    cluster_assignments[node_idx] = cluster_id
                index = solution.Value(routing.NextVar(index))
        
        # 10. 클러스터-기사 매칭
        print("\n=== 5단계: 클러스터-기사 매칭 ===")
        if use_driver_features:
            cluster_to_driver = match_clusters_to_drivers_v2(
                df, cluster_assignments, drivers, num_vehicles, depot_idx)
        else:
            cluster_to_driver = {i: i for i in range(num_vehicles)}
        
        # 11. 결과 생성
        results = []
        stats = []
        total_distance = 0
        
        for cluster_id in range(num_vehicles):
            driver_id = cluster_to_driver.get(cluster_id, cluster_id)
            driver = drivers[driver_id] if driver_id < len(drivers) else drivers[0]
            
            visit_order = optimize_visit_order(df, cluster_assignments, cluster_id, depot_idx)
            
            route_distance = 0
            route_weight = 0
            prev_lat = float(df.iloc[depot_idx]['lat'])
            prev_lon = float(df.iloc[depot_idx]['lon'])
            
            for order, node_idx in enumerate(visit_order, 1):
                try:
                    node = df.iloc[node_idx]
                    node_weight = int(node['weight'])
                    route_weight += node_weight
                    route_distance += haversine(prev_lat, prev_lon, 
                                               float(node['lat']), float(node['lon']))
                    prev_lat, prev_lon = float(node['lat']), float(node['lon'])
                    
                    results.append({
                        "id": str(node['id']),
                        "driver_id": driver.id,
                        "driver_name": driver.name,
                        "visit_order": order,
                        "weight_kg": node_weight,
                        "cumulative_weight_kg": route_weight
                    })
                except Exception as e:
                    print(f"Error processing node {node_idx}: {e}")
                    continue
            
            if visit_order:
                route_distance += haversine(prev_lat, prev_lon,
                                           float(df.iloc[depot_idx]['lat']),
                                           float(df.iloc[depot_idx]['lon']))
            
            total_distance += route_distance
            
            call_count = len(visit_order)
            max_capa = driver.max_capa if driver.max_capa else DEFAULT_MAX_CAPA
            
            status = "정상"
            if call_count < MIN_CALLS_SOFT:
                status = f"⚠️ 하한 미달 ({call_count} < {MIN_CALLS_SOFT})"
            elif call_count > max_capa:
                status = f"🚨 상한 초과 ({call_count} > {max_capa})"
            
            cluster_centroid = calculate_cluster_centroid(df, cluster_assignments, cluster_id)
            base_distance = 0
            if cluster_centroid and driver.base_lat and driver.base_lng:
                base_distance = haversine(cluster_centroid[0], cluster_centroid[1],
                                         driver.base_lat, driver.base_lng)
            
            stats.append({
                "driver_id": driver.id,
                "driver_name": driver.name,
                "call_count": call_count,
                "max_capa": max_capa,
                "total_weight_kg": route_weight,
                "vehicle_capacity_kg": driver.vehicle_capacity_kg or body.vehicle_capacity,
                "distance_km": round(route_distance, 2),
                "base_to_cluster_km": round(base_distance, 2),
                "status": status
            })
        
        # 미배정 체크
        assigned_ids = set([r['id'] for r in results])
        all_ids = set(str(df.iloc[i]['id']) for i in range(1, len(df)))
        unassigned_ids = all_ids - assigned_ids
        
        # 클러스터 교차 점수
        overlap_score = check_cluster_overlap(df, cluster_assignments, num_vehicles)
        
        print(f"\n=== 최적화 완료 ===")
        print(f"배정: {len(results)}건, 미배정: {len(unassigned_ids)}건")
        print(f"클러스터 교차 점수: {overlap_score}%")
        
        return {
            "status": "success",
            "updates": results,
            "statistics": stats,
            "summary": {
                "total_locations": num_locations - 1,
                "total_assigned": len(results),
                "unassigned": len(unassigned_ids),
                "unassigned_ids": list(unassigned_ids) if unassigned_ids else [],
                "total_distance_km": round(total_distance, 2),
                "avg_distance_km": round(total_distance / num_vehicles, 2) if num_vehicles > 0 else 0
            },
            "optimization_info": {
                "algorithm": "V10.4: K-Means + Convex Hull + Same Vehicle (90%)",
                "same_vehicle_constraints": constraints_added,
                "cluster_overlap_score": overlap_score,
                "core_node_ratio": CORE_NODE_RATIO,
                "use_driver_features": use_driver_features
            },
            "matching_info": {
                "cluster_to_driver": {f"클러스터{k}": drivers[v].name for k, v in cluster_to_driver.items()},
                "algorithm": "V10.2: max_capa 고려 매칭"
            } if use_driver_features else None
        }
        
    except Exception as e:
        traceback.print_exc()
        return {
            "status": "error",
            "message": str(e),
            "traceback": traceback.format_exc()
        }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
