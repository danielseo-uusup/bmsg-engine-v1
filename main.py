from fastapi import FastAPI
from pydantic import BaseModel
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
import pandas as pd
import numpy as np
import traceback
from typing import Optional

app = FastAPI()


# ============================================================
# 운영 철학 (설계 상위 규칙)
# ============================================================
# 1. 수요는 가능한 한 전부 흡수한다 (미수거 0)
# 2. 이동 효율은 최대화한다 (지역 밀도)
# 3. 기사 간 콜 수는 개인별 max_capa 내에서 관리
# 4. 기사 거점과 가장 가까운 클러스터에 배정
# ============================================================


# --- 상수 정의 ---
VEHICLE_CAPACITY_KG = 1200
DEFAULT_MAX_CAPA = 25  # max_capa 없을 때 기본값
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
    locations: list[Location]
    drivers: list[Driver]
    vehicle_capacity: int = VEHICLE_CAPACITY_KG  # 기본값 (개별 설정 없을 때)


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


def match_clusters_to_drivers(df, assignments, drivers, num_clusters, depot_idx=0):
    """
    클러스터-기사 최적 매칭
    
    1. 각 클러스터의 중심점 계산
    2. 각 기사의 거점(base_lat, base_lng)과 클러스터 중심 간 거리 계산
    3. Hungarian Algorithm으로 최적 매칭 (또는 Greedy)
    """
    # 클러스터 중심점 계산
    cluster_centroids = {}
    for cluster_id in range(num_clusters):
        centroid = calculate_cluster_centroid(df, assignments, cluster_id)
        if centroid:
            cluster_centroids[cluster_id] = centroid
    
    # 기사 거점 정보
    driver_bases = {}
    for i, driver in enumerate(drivers):
        if driver.base_lat is not None and driver.base_lng is not None:
            driver_bases[i] = (driver.base_lat, driver.base_lng)
        else:
            # 거점 정보 없으면 depot 사용
            driver_bases[i] = (float(df.iloc[depot_idx]['lat']), float(df.iloc[depot_idx]['lon']))
    
    # 거리 행렬: cluster_id × driver_id
    n = max(len(cluster_centroids), len(drivers))
    cost_matrix = np.full((n, n), 1e9)  # 큰 값으로 초기화
    
    for cluster_id, centroid in cluster_centroids.items():
        for driver_id, base in driver_bases.items():
            if cluster_id < n and driver_id < n:
                dist = haversine(centroid[0], centroid[1], base[0], base[1])
                cost_matrix[cluster_id][driver_id] = dist
    
    # Greedy 매칭 (간단한 구현)
    # 더 정교하게 하려면 scipy.optimize.linear_sum_assignment 사용 가능
    cluster_to_driver = {}
    used_drivers = set()
    
    # 거리가 가장 짧은 쌍부터 매칭
    pairs = []
    for cluster_id in cluster_centroids.keys():
        for driver_id in driver_bases.keys():
            pairs.append((cost_matrix[cluster_id][driver_id], cluster_id, driver_id))
    
    pairs.sort(key=lambda x: x[0])
    
    for dist, cluster_id, driver_id in pairs:
        if cluster_id not in cluster_to_driver and driver_id not in used_drivers:
            cluster_to_driver[cluster_id] = driver_id
            used_drivers.add(driver_id)
    
    # 매칭 안 된 클러스터는 남은 기사에게 배정
    remaining_drivers = set(range(len(drivers))) - used_drivers
    for cluster_id in range(num_clusters):
        if cluster_id not in cluster_to_driver:
            if remaining_drivers:
                cluster_to_driver[cluster_id] = remaining_drivers.pop()
            else:
                # 기사가 부족하면 첫 번째 기사에게
                cluster_to_driver[cluster_id] = 0
    
    return cluster_to_driver


def smart_swap_optimization(df, assignments, num_clusters, driver_capacities, depot_idx=0):
    """
    상식적 교환 최적화 (후처리)
    - driver_capacities: {cluster_id: max_capa} 매핑
    """
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
                if node_idx == depot_idx:
                    continue
                
                if node_idx not in assignments:
                    continue
                
                current_cluster = assignments[node_idx]
                
                if current_cluster not in centroids:
                    continue
                
                try:
                    node_lat = float(df.iloc[node_idx]['lat'])
                    node_lon = float(df.iloc[node_idx]['lon'])
                    node_weight = int(df.iloc[node_idx]['weight'])
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
                    if other_cluster == current_cluster:
                        continue
                    if other_cluster not in centroids:
                        continue
                    
                    other_centroid = centroids[other_cluster]
                    other_dist = haversine(node_lat, node_lon,
                                          other_centroid[0], other_centroid[1])
                    
                    if other_dist < current_dist * 0.7:
                        if other_dist < best_alternative_dist:
                            best_alternative = other_cluster
                            best_alternative_dist = other_dist
                
                if best_alternative is not None:
                    current_stats = get_cluster_stats(df, assignments, current_cluster)
                    target_stats = get_cluster_stats(df, assignments, best_alternative)
                    
                    can_swap = True
                    
                    # 대상 클러스터의 max_capa 체크
                    target_max_capa = driver_capacities.get(best_alternative, DEFAULT_MAX_CAPA)
                    if target_stats['call_count'] + 1 > target_max_capa:
                        can_swap = False
                    
                    # 현재 클러스터 콜 수 하한 체크
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
        traceback.print_exc()
    
    return assignments, swaps_made


def mutual_swap_optimization(df, assignments, num_clusters, driver_capacities, depot_idx=0):
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
                
                # A → B 후보
                a_to_b_candidates = []
                for idx in indices_a:
                    try:
                        node_lat = float(df.iloc[idx]['lat'])
                        node_lon = float(df.iloc[idx]['lon'])
                        
                        dist_to_a = haversine(node_lat, node_lon, centroids[vid_a][0], centroids[vid_a][1])
                        dist_to_b = haversine(node_lat, node_lon, centroids[vid_b][0], centroids[vid_b][1])
                        
                        if dist_to_b < dist_to_a:
                            improvement = dist_to_a - dist_to_b
                            a_to_b_candidates.append((idx, improvement, int(df.iloc[idx]['weight'])))
                    except:
                        continue
                
                # B → A 후보
                b_to_a_candidates = []
                for idx in indices_b:
                    try:
                        node_lat = float(df.iloc[idx]['lat'])
                        node_lon = float(df.iloc[idx]['lon'])
                        
                        dist_to_a = haversine(node_lat, node_lon, centroids[vid_a][0], centroids[vid_a][1])
                        dist_to_b = haversine(node_lat, node_lon, centroids[vid_b][0], centroids[vid_b][1])
                        
                        if dist_to_a < dist_to_b:
                            improvement = dist_to_b - dist_to_a
                            b_to_a_candidates.append((idx, improvement, int(df.iloc[idx]['weight'])))
                    except:
                        continue
                
                if not a_to_b_candidates or not b_to_a_candidates:
                    continue
                
                a_to_b_candidates.sort(key=lambda x: -x[1])
                b_to_a_candidates.sort(key=lambda x: -x[1])
                
                for a_cand in a_to_b_candidates[:3]:
                    for b_cand in b_to_a_candidates[:3]:
                        idx_a, _, weight_a = a_cand
                        idx_b, _, weight_b = b_cand
                        
                        if assignments.get(idx_a) != vid_a or assignments.get(idx_b) != vid_b:
                            continue
                        
                        # 맞교환은 콜 수 변화 없으므로 용량만 체크
                        assignments[idx_a] = vid_b
                        assignments[idx_b] = vid_a
                        swaps_made += 1
                        break
                    else:
                        continue
                    break
                    
    except Exception as e:
        print(f"mutual_swap error: {e}")
        traceback.print_exc()
    
    return assignments, swaps_made


def optimize_visit_order(df, assignments, cluster_id, depot_idx=0):
    """클러스터별 방문 순서 최적화 (Nearest Neighbor)"""
    assigned_indices = [idx for idx, vid in assignments.items() if vid == cluster_id and idx != depot_idx]
    
    if not assigned_indices:
        return []
    
    if len(assigned_indices) == 1:
        return assigned_indices
    
    try:
        depot_lat = float(df.iloc[depot_idx]['lat'])
        depot_lon = float(df.iloc[depot_idx]['lon'])
        
        visited = []
        remaining = set(assigned_indices)
        current_lat, current_lon = depot_lat, depot_lon
        
        while remaining:
            nearest = None
            nearest_dist = float('inf')
            
            for idx in remaining:
                try:
                    dist = haversine(current_lat, current_lon,
                                   float(df.iloc[idx]['lat']), 
                                   float(df.iloc[idx]['lon']))
                    if dist < nearest_dist:
                        nearest_dist = dist
                        nearest = idx
                except:
                    continue
            
            if nearest is not None:
                visited.append(nearest)
                remaining.remove(nearest)
                current_lat = float(df.iloc[nearest]['lat'])
                current_lon = float(df.iloc[nearest]['lon'])
            else:
                visited.extend(list(remaining))
                break
        
        return visited
        
    except Exception as e:
        print(f"optimize_visit_order error: {e}")
        return assigned_indices


@app.get("/")
def read_root():
    return {
        "status": "active",
        "message": "VRP Engine V10 (Driver-specific Capa + Base Location Matching)",
        "features": [
            "기사별 max_capa 하드캡",
            "기사 거점 기반 클러스터 매칭",
            "Smart Swap 후처리",
            "Mutual Swap 후처리"
        ]
    }


@app.post("/optimize")
def optimize_routes(body: RequestBody):
    """
    OR-Tools CVRP + 기사별 제약 + 거점 매칭
    
    단계:
    1. OR-Tools CVRP로 클러스터 생성 (기사별 max_capa 반영)
    2. Smart Swap / Mutual Swap 후처리
    3. 클러스터-기사 매칭 (거점 거리 기반)
    4. 방문 순서 최적화
    """
    
    try:
        # 1. 데이터 준비
        data = [loc.dict() for loc in body.locations]
        df = pd.DataFrame(data)
        df = df.reset_index(drop=True)
        
        num_locations = len(df)
        drivers = body.drivers
        num_vehicles = len(drivers)
        depot_idx = 0
        
        if num_locations < 2:
            return {"status": "error", "message": "Not enough locations"}
        
        if num_vehicles < 1:
            return {"status": "error", "message": "No drivers provided"}
        
        # 기사별 설정 추출
        driver_max_capas = []
        driver_kg_capas = []
        for driver in drivers:
            max_capa = driver.max_capa if driver.max_capa else DEFAULT_MAX_CAPA
            kg_capa = driver.vehicle_capacity_kg if driver.vehicle_capacity_kg else body.vehicle_capacity
            driver_max_capas.append(max_capa)
            driver_kg_capas.append(kg_capa)
        
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
            driver_kg_capas,  # 기사별 적재량
            True, 'Capacity'
        )
        
        # ★ 콜 수 제한 (기사별 max_capa) ★
        def count_callback(from_index):
            from_node = manager.IndexToNode(from_index)
            return 0 if from_node == depot_idx else 1
        
        count_callback_index = routing.RegisterUnaryTransitCallback(count_callback)
        routing.AddDimensionWithVehicleCapacity(
            count_callback_index, 0,
            driver_max_capas,  # ★ 기사별 max_capa 하드캡 ★
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
        
        # 4. OR-Tools 결과에서 클러스터 추출
        cluster_assignments = {}  # node_idx → cluster_id
        
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
            df, cluster_assignments, num_vehicles, cluster_capa_map, depot_idx)
        
        total_swaps = smart_swaps + mutual_swaps
        
        # 7. ★ 클러스터-기사 매칭 (거점 기반) ★
        cluster_to_driver = match_clusters_to_drivers(
            df, cluster_assignments, drivers, num_vehicles, depot_idx)
        
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
            
            # 거점과 클러스터 중심 거리 계산
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
                "message": f"후처리로 {total_swaps}건 재배정하여 클러스터 최적화"
            },
            "matching_info": {
                "cluster_to_driver": {f"클러스터{k}": drivers[v].name for k, v in cluster_to_driver.items()}
            }
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
