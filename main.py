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
    dist_matrix = np.zeros((n, n), dtype=int)
    
    for i in range(n):
        for j in range(n):
            if i != j:
                dist_km = haversine(
                    float(df.iloc[i]['lat']), float(df.iloc[i]['lon']),
                    float(df.iloc[j]['lat']), float(df.iloc[j]['lon'])
                )
                dist_matrix[i][j] = int(dist_km * 1000)  # 미터 단위
    
    return dist_matrix


@app.get("/")
def read_root():
    return {
        "status": "active",
        "message": "VRP Engine V12 (Pure OR-Tools CVRP)",
        "features": [
            "★ 순수 OR-Tools CVRP",
            "★ 클러스터링 없음",
            "★ 목표: 총 이동 거리 최소화",
            "★ 제약: 기사별 max_capa 준수",
            "결과가 어떻게 나오는지 테스트용"
        ],
        "algorithm": "OR-Tools CVRP (no pre-clustering)"
    }


@app.post("/optimize")
def optimize_routes(body: RequestBody):
    """
    ★ V12: 순수 OR-Tools CVRP ★
    
    - 클러스터링 없음
    - OR-Tools가 알아서 최적의 경로 결정
    - max_capa만 제약으로 설정
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
        
        # 기사 설정
        if body.drivers and len(body.drivers) > 0:
            drivers = body.drivers
        else:
            drivers = [
                Driver(id=f"driver_{i+1}", name=f"기사 {i+1}", max_capa=DEFAULT_MAX_CAPA)
                for i in range(body.num_vehicles)
            ]
        
        num_vehicles = len(drivers)
        
        # 기사별 max_capa
        driver_max_capas = []
        driver_info = []
        for i, driver in enumerate(drivers):
            max_capa = driver.max_capa if driver.max_capa else DEFAULT_MAX_CAPA
            base_lat = driver.base_lat if driver.base_lat else float(df.iloc[depot_idx]['lat'])
            base_lng = driver.base_lng if driver.base_lng else float(df.iloc[depot_idx]['lon'])
            
            driver_max_capas.append(max_capa)
            driver_info.append({
                'driver_idx': i,
                'driver': driver,
                'max_capa': max_capa,
                'base_lat': base_lat,
                'base_lng': base_lng
            })
        
        total_max_capa = sum(driver_max_capas)
        total_calls = num_locations - 1
        
        print(f"\n{'='*50}")
        print(f"VRP V12 - Pure OR-Tools CVRP")
        print(f"{'='*50}")
        print(f"총 콜: {total_calls}건, 수용량: {total_max_capa}건")
        print(f"기사별 max_capa: {driver_max_capas}")
        
        if 'weight' not in df.columns:
            df['weight'] = DEFAULT_WEIGHT_KG
        df['weight'] = pd.to_numeric(df['weight'], errors='coerce').fillna(DEFAULT_WEIGHT_KG).astype(int)
        df.loc[depot_idx, 'weight'] = 0
        
        # 2. 거리 행렬 생성
        print("\n=== 거리 행렬 생성 ===")
        dist_matrix = create_distance_matrix(df)
        
        # 3. OR-Tools 설정
        print("\n=== OR-Tools CVRP 설정 ===")
        manager = pywrapcp.RoutingIndexManager(num_locations, num_vehicles, depot_idx)
        routing = pywrapcp.RoutingModel(manager)
        
        # 거리 콜백
        def distance_callback(from_index, to_index):
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            return dist_matrix[from_node][to_node]
        
        transit_callback_index = routing.RegisterTransitCallback(distance_callback)
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
        
        # ★ 핵심: 콜 수 제한 (max_capa) ★
        def count_callback(from_index):
            from_node = manager.IndexToNode(from_index)
            return 0 if from_node == depot_idx else 1
        
        count_callback_index = routing.RegisterUnaryTransitCallback(count_callback)
        routing.AddDimensionWithVehicleCapacity(
            count_callback_index,
            0,  # slack
            driver_max_capas,  # 기사별 max_capa
            True,  # start_cumul_to_zero
            'CallCount'
        )
        
        # 미배정 허용 (총 콜 > 총 수용량인 경우)
        UNASSIGNED_PENALTY = 100000000
        for node_idx in range(1, num_locations):
            index = manager.NodeToIndex(node_idx)
            routing.AddDisjunction([index], UNASSIGNED_PENALTY)
        
        # 4. 검색 파라미터
        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
        search_parameters.local_search_metaheuristic = (
            routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH)
        search_parameters.time_limit.seconds = 30
        
        # 5. 최적화 실행
        print("\n=== OR-Tools 최적화 실행 ===")
        solution = routing.SolveWithParameters(search_parameters)
        
        if not solution:
            return {
                "status": "fail",
                "message": "OR-Tools가 해를 찾지 못했습니다."
            }
        
        # 6. 결과 추출
        print("\n=== 결과 추출 ===")
        
        results = []
        stats = []
        total_distance = 0
        
        for vehicle_id in range(num_vehicles):
            d_info = driver_info[vehicle_id]
            driver = d_info['driver']
            max_capa = d_info['max_capa']
            base_lat = d_info['base_lat']
            base_lng = d_info['base_lng']
            
            # 경로 추출
            route_nodes = []
            index = routing.Start(vehicle_id)
            
            while not routing.IsEnd(index):
                node_idx = manager.IndexToNode(index)
                if node_idx != depot_idx:
                    route_nodes.append(node_idx)
                index = solution.Value(routing.NextVar(index))
            
            # 거리 계산
            route_distance = 0
            if route_nodes:
                # 거점 → 첫 노드
                route_distance += haversine(
                    base_lat, base_lng,
                    float(df.iloc[route_nodes[0]]['lat']),
                    float(df.iloc[route_nodes[0]]['lon'])
                )
                # 노드 간 이동
                for i in range(len(route_nodes) - 1):
                    route_distance += haversine(
                        float(df.iloc[route_nodes[i]]['lat']),
                        float(df.iloc[route_nodes[i]]['lon']),
                        float(df.iloc[route_nodes[i+1]]['lat']),
                        float(df.iloc[route_nodes[i+1]]['lon'])
                    )
            
            total_distance += route_distance
            
            # 결과 생성
            route_weight = 0
            for order, node_idx in enumerate(route_nodes, 1):
                node = df.iloc[node_idx]
                node_weight = int(node['weight'])
                route_weight += node_weight
                
                results.append({
                    "id": str(node['id']),
                    "driver_id": driver.id,
                    "driver_name": driver.name,
                    "visit_order": order,
                    "weight_kg": node_weight,
                    "cumulative_weight_kg": route_weight
                })
            
            call_count = len(route_nodes)
            status = "정상" if call_count <= max_capa else f"🚨 초과"
            
            stats.append({
                "driver_id": driver.id,
                "driver_name": driver.name,
                "call_count": call_count,
                "max_capa": max_capa,
                "total_weight_kg": route_weight,
                "distance_km": round(route_distance, 2),
                "status": status
            })
            
            print(f"  {driver.name}: {call_count}건, {route_distance:.1f}km")
        
        # 미배정 노드 확인
        assigned_ids = set(r['id'] for r in results)
        all_ids = set(str(df.iloc[i]['id']) for i in range(1, len(df)))
        unassigned_ids = list(all_ids - assigned_ids)
        
        violations = [s for s in stats if s['call_count'] > s['max_capa']]
        
        print(f"\n{'='*50}")
        print(f"완료: 배정 {len(results)}건, 미배정 {len(unassigned_ids)}건")
        print(f"총 거리: {total_distance:.1f}km")
        print(f"{'='*50}")
        
        return {
            "status": "success",
            "updates": results,
            "statistics": stats,
            "summary": {
                "total_locations": total_calls,
                "total_assigned": len(results),
                "unassigned": len(unassigned_ids),
                "unassigned_ids": unassigned_ids,
                "total_distance_km": round(total_distance, 2)
            },
            "optimization_info": {
                "algorithm": "V12: Pure OR-Tools CVRP",
                "max_capa_violations": len(violations),
                "cluster_overlap": "N/A (OR-Tools가 자유롭게 배정)",
                "note": "클러스터링 없이 순수 거리 최적화"
            }
        }
        
    except Exception as e:
        traceback.print_exc()
        return {"status": "error", "message": str(e)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
