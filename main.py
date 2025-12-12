from fastapi import FastAPI
from pydantic import BaseModel
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
import pandas as pd
import numpy as np
import traceback

app = FastAPI()


# --- 상수 정의 ---
VEHICLE_CAPACITY_KG = 1200
MIN_CALLS_SOFT = 10
MAX_CALLS_SOFT = 25
DEFAULT_WEIGHT_KG = 15


# --- 데이터 모델 ---
class Location(BaseModel):
    id: str
    lat: float
    lon: float
    weight: int = DEFAULT_WEIGHT_KG


class RequestBody(BaseModel):
    locations: list[Location]
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
    """특정 기사에게 배정된 점들의 중심점 계산"""
    assigned_indices = [idx for idx, vid in assignments.items() if vid == vehicle_id]
    if not assigned_indices:
        return None
    
    lats = [float(df.iloc[idx]['lat']) for idx in assigned_indices]
    lons = [float(df.iloc[idx]['lon']) for idx in assigned_indices]
    
    return (np.mean(lats), np.mean(lons))


def get_cluster_stats(df, assignments, vehicle_id, vehicle_capacity):
    """특정 기사의 클러스터 통계"""
    assigned_indices = [idx for idx, vid in assignments.items() if vid == vehicle_id]
    
    total_weight = sum(int(df.iloc[idx]['weight']) for idx in assigned_indices)
    call_count = len(assigned_indices)
    
    return {
        'indices': list(assigned_indices),  # 복사본 사용
        'call_count': call_count,
        'total_weight': total_weight,
        'remaining_capacity': vehicle_capacity - total_weight
    }


def smart_swap_optimization(df, assignments, num_vehicles, vehicle_capacity, depot_idx=0):
    """
    상식적 교환 최적화 (후처리)
    - 안전한 버전: 모든 예외 상황 처리
    """
    assignments = dict(assignments)  # 복사
    swaps_made = 0
    max_iterations = 5
    
    try:
        for iteration in range(max_iterations):
            made_swap_this_round = False
            
            # 각 기사별 중심점 계산
            centroids = {}
            for vid in range(num_vehicles):
                centroid = calculate_cluster_centroid(df, assignments, vid)
                if centroid is not None:
                    centroids[vid] = centroid
            
            if not centroids:
                break
            
            # 각 점에 대해 교환 검토
            nodes_to_check = list(assignments.keys())
            
            for node_idx in nodes_to_check:
                if node_idx == depot_idx:
                    continue
                
                if node_idx not in assignments:
                    continue
                
                current_vehicle = assignments[node_idx]
                
                if current_vehicle not in centroids:
                    continue
                
                try:
                    node_lat = float(df.iloc[node_idx]['lat'])
                    node_lon = float(df.iloc[node_idx]['lon'])
                    node_weight = int(df.iloc[node_idx]['weight'])
                except (IndexError, KeyError, ValueError):
                    continue
                
                # 현재 클러스터 중심까지 거리
                current_centroid = centroids[current_vehicle]
                current_dist = haversine(node_lat, node_lon, 
                                        current_centroid[0], current_centroid[1])
                
                if current_dist == 0:
                    continue
                
                # 다른 클러스터 중심까지 거리 비교
                best_alternative = None
                best_alternative_dist = current_dist
                
                for other_vehicle in range(num_vehicles):
                    if other_vehicle == current_vehicle:
                        continue
                    if other_vehicle not in centroids:
                        continue
                    
                    other_centroid = centroids[other_vehicle]
                    other_dist = haversine(node_lat, node_lon,
                                          other_centroid[0], other_centroid[1])
                    
                    # 30% 이상 가까워야 교환 고려
                    if other_dist < current_dist * 0.7:
                        if other_dist < best_alternative_dist:
                            best_alternative = other_vehicle
                            best_alternative_dist = other_dist
                
                # 교환 후보가 있으면 제약 조건 확인
                if best_alternative is not None:
                    current_stats = get_cluster_stats(df, assignments, current_vehicle, vehicle_capacity)
                    target_stats = get_cluster_stats(df, assignments, best_alternative, vehicle_capacity)
                    
                    can_swap = True
                    
                    # 1. 대상 기사 용량 초과 체크
                    if target_stats['total_weight'] + node_weight > vehicle_capacity:
                        can_swap = False
                    
                    # 2. 현재 기사 콜 수 하한 체크
                    if current_stats['call_count'] - 1 < MIN_CALLS_SOFT:
                        if current_stats['call_count'] >= MIN_CALLS_SOFT:
                            can_swap = False
                    
                    # 3. 대상 기사 콜 수 상한 체크
                    if target_stats['call_count'] + 1 > MAX_CALLS_SOFT:
                        improvement = (current_dist - best_alternative_dist) / current_dist
                        if improvement < 0.5:
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


def mutual_swap_optimization(df, assignments, num_vehicles, vehicle_capacity, depot_idx=0):
    """
    상호 교환 최적화 - 안전한 버전
    """
    assignments = dict(assignments)  # 복사
    swaps_made = 0
    
    try:
        # 각 기사별 중심점
        centroids = {}
        for vid in range(num_vehicles):
            centroid = calculate_cluster_centroid(df, assignments, vid)
            if centroid is not None:
                centroids[vid] = centroid
        
        if len(centroids) < 2:
            return assignments, swaps_made
        
        # 모든 기사 쌍에 대해 교환 검토
        for vid_a in range(num_vehicles):
            for vid_b in range(vid_a + 1, num_vehicles):
                if vid_a not in centroids or vid_b not in centroids:
                    continue
                
                # 현재 배정 상태 확인
                indices_a = [idx for idx, vid in assignments.items() if vid == vid_a and idx != depot_idx]
                indices_b = [idx for idx, vid in assignments.items() if vid == vid_b and idx != depot_idx]
                
                if not indices_a or not indices_b:
                    continue
                
                weight_a = sum(int(df.iloc[idx]['weight']) for idx in indices_a)
                weight_b = sum(int(df.iloc[idx]['weight']) for idx in indices_b)
                
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
                
                # 개선도 높은 순으로 정렬
                a_to_b_candidates.sort(key=lambda x: -x[1])
                b_to_a_candidates.sort(key=lambda x: -x[1])
                
                # 맞교환 실행 (한 쌍만)
                for a_cand in a_to_b_candidates[:3]:  # 상위 3개만 검토
                    for b_cand in b_to_a_candidates[:3]:
                        idx_a, _, weight_a_point = a_cand
                        idx_b, _, weight_b_point = b_cand
                        
                        # 이미 교환된 점인지 확인
                        if assignments.get(idx_a) != vid_a or assignments.get(idx_b) != vid_b:
                            continue
                        
                        # 교환 후 용량 체크
                        new_weight_a = weight_a - weight_a_point + weight_b_point
                        new_weight_b = weight_b - weight_b_point + weight_a_point
                        
                        if new_weight_a <= vehicle_capacity and new_weight_b <= vehicle_capacity:
                            assignments[idx_a] = vid_b
                            assignments[idx_b] = vid_a
                            swaps_made += 1
                            
                            # 무게 업데이트
                            weight_a = new_weight_a
                            weight_b = new_weight_b
                            break
                    else:
                        continue
                    break
                    
    except Exception as e:
        print(f"mutual_swap error: {e}")
        traceback.print_exc()
    
    return assignments, swaps_made


def optimize_visit_order(df, assignments, vehicle_id, depot_idx=0):
    """기사별 방문 순서 최적화 (Nearest Neighbor)"""
    assigned_indices = [idx for idx, vid in assignments.items() if vid == vehicle_id and idx != depot_idx]
    
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
                # 못 찾으면 나머지 다 추가
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
        "message": "VRP Engine V9.1 (CVRP + Smart Swap - Bug Fixed)",
        "philosophy": "미수거 0 → 이동효율 → 콜수 안정(10-25)"
    }


@app.post("/optimize")
def optimize_routes(body: RequestBody):
    """
    OR-Tools CVRP + Smart Swap 후처리 (버그 수정 버전)
    """
    
    try:
        # 1. 데이터 준비
        data = [loc.dict() for loc in body.locations]
        df = pd.DataFrame(data)
        df = df.reset_index(drop=True)
        
        num_locations = len(df)
        num_vehicles = body.num_vehicles
        vehicle_capacity = body.vehicle_capacity
        depot_idx = 0
        
        if num_locations < 2:
            return {"status": "error", "message": "Not enough locations"}
        
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
        
        # 적재량 제한
        def demand_callback(from_index):
            from_node = manager.IndexToNode(from_index)
            return int(df.iloc[from_node]['weight'])
        
        demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
        routing.AddDimensionWithVehicleCapacity(
            demand_callback_index, 0,
            [vehicle_capacity] * num_vehicles,
            True, 'Capacity'
        )
        
        # 콜 수 제약
        def count_callback(from_index):
            from_node = manager.IndexToNode(from_index)
            return 0 if from_node == depot_idx else 1
        
        count_callback_index = routing.RegisterUnaryTransitCallback(count_callback)
        routing.AddDimension(count_callback_index, 0, 100, True, 'Count')
        
        count_dimension = routing.GetDimensionOrDie('Count')
        CALL_PENALTY = 50000
        
        for vehicle_id in range(num_vehicles):
            end_index = routing.End(vehicle_id)
            count_dimension.SetCumulVarSoftLowerBound(end_index, MIN_CALLS_SOFT, CALL_PENALTY)
            count_dimension.SetCumulVarSoftUpperBound(end_index, MAX_CALLS_SOFT, CALL_PENALTY)
        
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
                "message": "Solution not found. 적재량 제약 확인 필요.",
                "debug": {
                    "total_weight": int(df['weight'].sum()),
                    "total_capacity": vehicle_capacity * num_vehicles,
                    "num_locations": num_locations
                }
            }
        
        # 4. OR-Tools 결과에서 배정 추출
        assignments = {}
        
        for vehicle_id in range(num_vehicles):
            index = routing.Start(vehicle_id)
            while not routing.IsEnd(index):
                node_idx = manager.IndexToNode(index)
                if node_idx != depot_idx:
                    assignments[node_idx] = vehicle_id
                index = solution.Value(routing.NextVar(index))
        
        # 5. Smart Swap 후처리
        assignments, smart_swaps = smart_swap_optimization(
            df, assignments, num_vehicles, vehicle_capacity, depot_idx)
        
        # 6. Mutual Swap 후처리
        assignments, mutual_swaps = mutual_swap_optimization(
            df, assignments, num_vehicles, vehicle_capacity, depot_idx)
        
        total_swaps = smart_swaps + mutual_swaps
        
        # 7. 결과 생성
        results = []
        stats = []
        total_distance = 0
        
        for vehicle_id in range(num_vehicles):
            visit_order = optimize_visit_order(df, assignments, vehicle_id, depot_idx)
            
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
                        "driver": f"기사 {vehicle_id + 1}",
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
            status = "정상"
            if call_count < MIN_CALLS_SOFT:
                status = f"⚠️ 하한 미달 ({call_count} < {MIN_CALLS_SOFT})"
            elif call_count > MAX_CALLS_SOFT:
                status = f"⚠️ 상한 초과 ({call_count} > {MAX_CALLS_SOFT})"
            
            stats.append({
                "driver": f"기사 {vehicle_id + 1}",
                "call_count": call_count,
                "total_weight_kg": route_weight,
                "distance_km": round(route_distance, 2),
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
            "constraints": {
                "vehicle_capacity_kg": vehicle_capacity,
                "min_calls_soft": MIN_CALLS_SOFT,
                "max_calls_soft": MAX_CALLS_SOFT
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
