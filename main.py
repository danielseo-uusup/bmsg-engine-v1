from fastapi import FastAPI
from pydantic import BaseModel
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
import pandas as pd
import numpy as np
import traceback
from typing import Optional, List

app = FastAPI()


# --- 상수 정의 ---
VEHICLE_CAPACITY_KG = 1200
DEFAULT_MAX_CAPA = 25
MIN_CALLS_SOFT = 10
DEFAULT_WEIGHT_KG = 15


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


def match_clusters_to_drivers_v2(df, assignments, drivers, num_clusters, depot_idx=0):
    """
    ★ V10.2 신규: max_capa를 고려한 클러스터-기사 매칭 ★
    
    기존 문제점:
    - 거점 거리만 고려해서 매칭
    - max_capa가 20인 기사가 콜 10개 클러스터에, max_capa가 10인 기사가 콜 20개 클러스터에 매칭되는 문제
    
    해결 방안:
    1. 각 클러스터의 콜 수 계산
    2. 기사의 max_capa가 클러스터 콜 수보다 작으면 매칭 후보에서 제외
    3. 적합한 후보 중에서 거점 거리가 가장 가까운 기사 선택
    4. 적합한 후보가 없으면 max_capa가 가장 큰 기사 선택 (차선책)
    """
    
    # 1. 클러스터별 통계 계산
    cluster_stats = {}
    for cluster_id in range(num_clusters):
        stats = get_cluster_stats(df, assignments, cluster_id)
        cluster_stats[cluster_id] = stats
        print(f"  클러스터 {cluster_id}: {stats['call_count']}건, {stats['total_weight']}kg")
    
    # 2. 클러스터 중심점 계산
    cluster_centroids = {}
    for cluster_id in range(num_clusters):
        centroid = calculate_cluster_centroid(df, assignments, cluster_id)
        if centroid:
            cluster_centroids[cluster_id] = centroid
    
    # 3. 기사 정보 정리
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
    
    # 4. ★ 핵심 로직: max_capa를 고려한 매칭 ★
    cluster_to_driver = {}
    used_drivers = set()
    
    # 클러스터를 콜 수 내림차순으로 정렬 (콜 수 많은 클러스터부터 처리)
    # → max_capa가 큰 기사를 먼저 배정받을 수 있도록
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
        
        # 적합한 기사 후보 찾기 (max_capa >= 클러스터 콜 수)
        candidates = []
        for driver_id, info in driver_info.items():
            if driver_id in used_drivers:
                continue
            
            # ★ 핵심: max_capa가 클러스터 콜 수 이상인 기사만 후보로
            if info['max_capa'] >= cluster_calls:
                dist = haversine(centroid[0], centroid[1], info['base'][0], info['base'][1])
                candidates.append((driver_id, dist, info['max_capa']))
        
        if candidates:
            # 적합한 후보 중 거리가 가장 가까운 기사 선택
            candidates.sort(key=lambda x: x[1])  # 거리순 정렬
            best_driver = candidates[0][0]
            print(f"  클러스터 {cluster_id} ({cluster_calls}건) → 기사 {best_driver} ({driver_info[best_driver]['name']}, max_capa={driver_info[best_driver]['max_capa']}) [거리 기반]")
        else:
            # 적합한 후보가 없으면 남은 기사 중 max_capa가 가장 큰 기사 선택
            remaining = [(d, info['max_capa']) for d, info in driver_info.items() if d not in used_drivers]
            if remaining:
                remaining.sort(key=lambda x: -x[1])  # max_capa 내림차순
                best_driver = remaining[0][0]
                print(f"  클러스터 {cluster_id} ({cluster_calls}건) → 기사 {best_driver} ({driver_info[best_driver]['name']}, max_capa={driver_info[best_driver]['max_capa']}) [차선책: 적합 후보 없음]")
            else:
                best_driver = 0
                print(f"  클러스터 {cluster_id} ({cluster_calls}건) → 기사 0 [fallback]")
        
        cluster_to_driver[cluster_id] = best_driver
        used_drivers.add(best_driver)
    
    # 매칭 안 된 클러스터 처리 (빈 클러스터 등)
    remaining_drivers = set(range(len(drivers))) - used_drivers
    for cluster_id in range(num_clusters):
        if cluster_id not in cluster_to_driver:
            if remaining_drivers:
                cluster_to_driver[cluster_id] = remaining_drivers.pop()
            else:
                cluster_to_driver[cluster_id] = 0
    
    print(f"\n  최종 매칭 결과: {cluster_to_driver}")
    return cluster_to_driver


def smart_swap_optimization(df, assignments, num_clusters, driver_capacities, depot_idx=0):
    """상식적 교환 최적화"""
    assignments = dict(assignments)
    swaps_made = 0
    max_iterations = 5
    
    try:
        for iteration in range(max_iterations):
            made_swap_this_round = False
            
            centroids = {}
            for vid in range(num_clusters):
                centroid = calculate_cluster_centroid(df, assignments, vid)
                if centroid is not None:
                    centroids[vid] = centroid
            
            if not centroids:
                break
            
            nodes_to_check = list(assignments.keys())
            
            for node_idx in nodes_to_check:
                if node_idx == depot_idx or node_idx not in assignments:
                    continue
                
                current_cluster = assignments[node_idx]
                
                if current_cluster not in centroids:
                    continue
                
                try:
                    node_lat = float(df.iloc[node_idx]['lat'])
                    node_lon = float(df.iloc[node_idx]['lon'])
                except:
                    continue
                
                current_centroid = centroids[current_cluster]
                current_dist = haversine(node_lat, node_lon, 
                                        current_centroid[0], current_centroid[1])
                
                if current_dist == 0:
                    continue
                
                best_alternative = None
                best_alternative_dist = current_dist
                
                for other_cluster in range(num_clusters):
                    if other_cluster == current_cluster or other_cluster not in centroids:
                        continue
                    
                    other_centroid = centroids[other_cluster]
                    other_dist = haversine(node_lat, node_lon,
                                          other_centroid[0], other_centroid[1])
                    
                    if other_dist < current_dist * 0.7 and other_dist < best_alternative_dist:
                        best_alternative = other_cluster
                        best_alternative_dist = other_dist
                
                if best_alternative is not None:
                    current_stats = get_cluster_stats(df, assignments, current_cluster)
                    target_stats = get_cluster_stats(df, assignments, best_alternative)
                    
                    can_swap = True
                    
                    target_max_capa = driver_capacities.get(best_alternative, DEFAULT_MAX_CAPA)
                    if target_stats['call_count'] + 1 > target_max_capa:
                        can_swap = False
                    
                    if current_stats['call_count'] - 1 < MIN_CALLS_SOFT:
                        if current_stats['call_count'] >= MIN_CALLS_SOFT:
                            can_swap = False
                    
                    if can_swap:
                        assignments[node_idx] = best_alternative
                        swaps_made += 1
                        made_swap_this_round = True
            
            if not made_swap_this_round:
                break
                
    except Exception as e:
        print(f"smart_swap error: {e}")
    
    return assignments, swaps_made


def mutual_swap_optimization(df, assignments, num_clusters, depot_idx=0):
    """상호 교환 최적화"""
    assignments = dict(assignments)
    swaps_made = 0
    
    try:
        centroids = {}
        for vid in range(num_clusters):
            centroid = calculate_cluster_centroid(df, assignments, vid)
            if centroid is not None:
                centroids[vid] = centroid
        
        if len(centroids) < 2:
            return assignments, swaps_made
        
        for vid_a in range(num_clusters):
            for vid_b in range(vid_a + 1, num_clusters):
                if vid_a not in centroids or vid_b not in centroids:
                    continue
                
                indices_a = [idx for idx, vid in assignments.items() if vid == vid_a and idx != depot_idx]
                indices_b = [idx for idx, vid in assignments.items() if vid == vid_b and idx != depot_idx]
                
                if not indices_a or not indices_b:
                    continue
                
                a_to_b_candidates = []
                for idx in indices_a:
                    try:
                        node_lat = float(df.iloc[idx]['lat'])
                        node_lon = float(df.iloc[idx]['lon'])
                        dist_to_a = haversine(node_lat, node_lon, centroids[vid_a][0], centroids[vid_a][1])
                        dist_to_b = haversine(node_lat, node_lon, centroids[vid_b][0], centroids[vid_b][1])
                        if dist_to_b < dist_to_a:
                            a_to_b_candidates.append((idx, dist_to_a - dist_to_b))
                    except:
                        continue
                
                b_to_a_candidates = []
                for idx in indices_b:
                    try:
                        node_lat = float(df.iloc[idx]['lat'])
                        node_lon = float(df.iloc[idx]['lon'])
                        dist_to_a = haversine(node_lat, node_lon, centroids[vid_a][0], centroids[vid_a][1])
                        dist_to_b = haversine(node_lat, node_lon, centroids[vid_b][0], centroids[vid_b][1])
                        if dist_to_a < dist_to_b:
                            b_to_a_candidates.append((idx, dist_to_b - dist_to_a))
                    except:
                        continue
                
                if not a_to_b_candidates or not b_to_a_candidates:
                    continue
                
                a_to_b_candidates.sort(key=lambda x: -x[1])
                b_to_a_candidates.sort(key=lambda x: -x[1])
                
                for a_cand in a_to_b_candidates[:3]:
                    for b_cand in b_to_a_candidates[:3]:
                        idx_a = a_cand[0]
                        idx_b = b_cand[0]
                        
                        if assignments.get(idx_a) != vid_a or assignments.get(idx_b) != vid_b:
                            continue
                        
                        assignments[idx_a] = vid_b
                        assignments[idx_b] = vid_a
                        swaps_made += 1
                        break
                    else:
                        continue
                    break
                    
    except Exception as e:
        print(f"mutual_swap error: {e}")
    
    return assignments, swaps_made


def optimize_visit_order(df, assignments, cluster_id, depot_idx=0):
    """클러스터별 방문 순서 최적화"""
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


@app.get("/")
def read_root():
    return {
        "status": "active",
        "message": "VRP Engine V10.2 (Capacity-Aware Matching)",
        "features": [
            "drivers 필드 선택적 (없으면 num_vehicles 사용)",
            "기사별 max_capa 하드캡",
            "★ V10.2: max_capa 고려한 클러스터-기사 매칭",
            "콜 수 많은 클러스터 → max_capa 큰 기사 우선 배정",
            "Smart Swap / Mutual Swap 후처리"
        ],
        "changelog": {
            "v10.2": "클러스터-기사 매칭 시 max_capa 제약 추가. 콜 수 초과 매칭 방지."
        }
    }


@app.post("/optimize")
def optimize_routes(body: RequestBody):
    """
    OR-Tools CVRP + 기사별 제약 + ★ max_capa 고려 매칭 ★
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
        
        # drivers 유무에 따라 분기
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
        
        # 기사별 설정 추출
        driver_max_capas = []
        driver_kg_capas = []
        for driver in drivers:
            max_capa = driver.max_capa if driver.max_capa else DEFAULT_MAX_CAPA
            kg_capa = driver.vehicle_capacity_kg if driver.vehicle_capacity_kg else body.vehicle_capacity
            driver_max_capas.append(max_capa)
            driver_kg_capas.append(kg_capa)
        
        print(f"\n=== VRP V10.2 최적화 시작 ===")
        print(f"총 위치: {num_locations}개, 기사: {num_vehicles}명")
        print(f"기사별 max_capa: {driver_max_capas}")
        print(f"총 수용 가능: {sum(driver_max_capas)}건")
        
        # weight 처리
        if 'weight' not in df.columns:
            df['weight'] = DEFAULT_WEIGHT_KG
        df['weight'] = pd.to_numeric(df['weight'], errors='coerce').fillna(DEFAULT_WEIGHT_KG).astype(int)
        df.loc[depot_idx, 'weight'] = 0
        
        # 2. 거리 행렬
        dist_matrix = create_distance_matrix(df)
        
        # 3. OR-Tools CVRP
        manager = pywrapcp.RoutingIndexManager(num_locations, num_vehicles, depot_idx)
        routing = pywrapcp.RoutingModel(manager)
        
        def distance_callback(from_index, to_index):
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            return dist_matrix[from_node][to_node]
        
        transit_callback_index = routing.RegisterTransitCallback(distance_callback)
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
        
        # 미배정 방지
        UNASSIGNED_PENALTY = 10000000000
        for node_idx in range(1, num_locations):
            index = manager.NodeToIndex(node_idx)
            routing.AddDisjunction([index], UNASSIGNED_PENALTY)
        
        # 적재량 제한 (기사별)
        def demand_callback(from_index):
            from_node = manager.IndexToNode(from_index)
            return int(df.iloc[from_node]['weight'])
        
        demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
        routing.AddDimensionWithVehicleCapacity(
            demand_callback_index, 0,
            driver_kg_capas,
            True, 'Capacity'
        )
        
        # 콜 수 제한 (기사별 max_capa 하드캡)
        def count_callback(from_index):
            from_node = manager.IndexToNode(from_index)
            return 0 if from_node == depot_idx else 1
        
        count_callback_index = routing.RegisterUnaryTransitCallback(count_callback)
        routing.AddDimensionWithVehicleCapacity(
            count_callback_index, 0,
            driver_max_capas,
            True, 'CallCount'
        )
        
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
        
        # 검색 전략
        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.PARALLEL_CHEAPEST_INSERTION)
        search_parameters.local_search_metaheuristic = (
            routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH)
        search_parameters.time_limit.seconds = 30
        
        solution = routing.SolveWithParameters(search_parameters)
        
        if not solution:
            return {
                "status": "fail",
                "message": "Solution not found. 적재량/콜수 제약 확인 필요.",
                "debug": {
                    "total_calls": num_locations - 1,
                    "total_max_capa": sum(driver_max_capas),
                    "driver_max_capas": driver_max_capas,
                    "total_weight": int(df['weight'].sum()),
                    "total_kg_capacity": sum(driver_kg_capas)
                }
            }
        
        # 4. 클러스터 추출
        cluster_assignments = {}
        
        for cluster_id in range(num_vehicles):
            index = routing.Start(cluster_id)
            while not routing.IsEnd(index):
                node_idx = manager.IndexToNode(index)
                if node_idx != depot_idx:
                    cluster_assignments[node_idx] = cluster_id
                index = solution.Value(routing.NextVar(index))
        
        # 5. Smart Swap 후처리
        cluster_capa_map = {i: driver_max_capas[i] for i in range(num_vehicles)}
        cluster_assignments, smart_swaps = smart_swap_optimization(
            df, cluster_assignments, num_vehicles, cluster_capa_map, depot_idx)
        
        # 6. Mutual Swap 후처리
        cluster_assignments, mutual_swaps = mutual_swap_optimization(
            df, cluster_assignments, num_vehicles, depot_idx)
        
        total_swaps = smart_swaps + mutual_swaps
        
        # 7. ★ V10.2: max_capa 고려한 클러스터-기사 매칭 ★
        print("\n=== 클러스터-기사 매칭 (V10.2) ===")
        if use_driver_features:
            cluster_to_driver = match_clusters_to_drivers_v2(
                df, cluster_assignments, drivers, num_vehicles, depot_idx)
        else:
            cluster_to_driver = {i: i for i in range(num_vehicles)}
        
        # 8. 결과 생성
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
            
            # depot 복귀
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
        
        print(f"\n=== 최적화 완료 ===")
        print(f"배정: {len(results)}건, 미배정: {len(unassigned_ids)}건")
        
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
                "smart_swaps": smart_swaps,
                "mutual_swaps": mutual_swaps,
                "total_swaps": total_swaps,
                "use_driver_features": use_driver_features,
                "message": f"후처리로 {total_swaps}건 재배정"
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
