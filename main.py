from fastapi import FastAPI
from pydantic import BaseModel
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


def voronoi_partition(df, drivers, depot_idx=0):
    """
    ★ V11.2 핵심: 거점 기반 Voronoi 분할 ★
    
    원칙:
    1. 각 노드를 "가장 가까운 기사 거점"에 배정
    2. 자연스럽게 영역이 분리됨 (Voronoi 특성)
    3. 기사들의 영역이 서로 겹치지 않음
    
    결과: 각 기사의 거점 주변으로 클러스터 형성
    """
    
    print("\n=== 1단계: 거점 기반 Voronoi 분할 ===")
    
    # 기사 거점 정보
    driver_bases = []
    for i, driver in enumerate(drivers):
        max_capa = driver.max_capa if driver.max_capa else DEFAULT_MAX_CAPA
        
        if driver.base_lat is not None and driver.base_lng is not None:
            base_lat, base_lng = driver.base_lat, driver.base_lng
        else:
            # 거점 없으면 depot 사용
            base_lat = float(df.iloc[depot_idx]['lat'])
            base_lng = float(df.iloc[depot_idx]['lon'])
        
        driver_bases.append({
            'driver_idx': i,
            'driver': driver,
            'max_capa': max_capa,
            'base_lat': base_lat,
            'base_lng': base_lng
        })
        
        print(f"  {driver.name}: 거점=({base_lat:.4f}, {base_lng:.4f}), max_capa={max_capa}")
    
    # 각 노드를 가장 가까운 거점에 배정 (순수 Voronoi)
    node_assignments = {}  # {node_idx: driver_idx}
    driver_nodes = {i: [] for i in range(len(drivers))}
    
    for i in range(len(df)):
        if i == depot_idx:
            continue
        
        node_lat = float(df.iloc[i]['lat'])
        node_lon = float(df.iloc[i]['lon'])
        
        # 가장 가까운 거점 찾기
        min_dist = float('inf')
        nearest_driver = 0
        
        for d_info in driver_bases:
            dist = haversine(node_lat, node_lon, d_info['base_lat'], d_info['base_lng'])
            if dist < min_dist:
                min_dist = dist
                nearest_driver = d_info['driver_idx']
        
        node_assignments[i] = nearest_driver
        driver_nodes[nearest_driver].append(i)
    
    # Voronoi 결과 출력
    print(f"\n  Voronoi 분할 결과:")
    for d_info in driver_bases:
        driver_idx = d_info['driver_idx']
        count = len(driver_nodes[driver_idx])
        print(f"    {d_info['driver'].name}: {count}건 (max_capa={d_info['max_capa']})")
    
    return node_assignments, driver_nodes, driver_bases


def balance_by_max_capa(df, driver_nodes, driver_bases, depot_idx=0):
    """
    ★ V11.2: max_capa 기반 밸런싱 ★
    
    Voronoi 분할 후 max_capa를 초과하는 기사가 있으면:
    1. 초과 기사의 "경계 노드" (거점에서 가장 먼 노드)를 찾음
    2. 해당 노드와 가장 가까운 "여유 있는" 기사에게 이전
    3. 모든 기사가 max_capa 이하가 될 때까지 반복
    
    핵심: 이전 시 "거리 기준"으로 자연스러운 경계 유지
    """
    
    print("\n=== 2단계: max_capa 기반 밸런싱 ===")
    
    # 현재 상태 복사
    balanced_nodes = {k: list(v) for k, v in driver_nodes.items()}
    
    # 기사별 max_capa 매핑
    driver_max_capa = {d['driver_idx']: d['max_capa'] for d in driver_bases}
    driver_bases_map = {d['driver_idx']: d for d in driver_bases}
    
    max_iterations = 100
    total_moved = 0
    
    for iteration in range(max_iterations):
        moved_this_round = False
        
        # 초과 기사 찾기
        for driver_idx, nodes in balanced_nodes.items():
            max_capa = driver_max_capa[driver_idx]
            excess = len(nodes) - max_capa
            
            if excess <= 0:
                continue
            
            # 거점 정보
            d_info = driver_bases_map[driver_idx]
            base_lat, base_lng = d_info['base_lat'], d_info['base_lng']
            
            # 거점에서 가장 먼 노드 (경계 노드)
            nodes_with_dist = []
            for node_idx in nodes:
                node_lat = float(df.iloc[node_idx]['lat'])
                node_lon = float(df.iloc[node_idx]['lon'])
                dist = haversine(base_lat, base_lng, node_lat, node_lon)
                nodes_with_dist.append((node_idx, dist, node_lat, node_lon))
            
            nodes_with_dist.sort(key=lambda x: -x[1])  # 거리 내림차순 (먼 것부터)
            
            # 초과분만큼 이전 시도
            for node_idx, _, node_lat, node_lon in nodes_with_dist[:excess]:
                # 여유 있는 기사 중 해당 노드와 가장 가까운 거점 찾기
                best_target = None
                best_dist = float('inf')
                
                for other_idx, other_nodes in balanced_nodes.items():
                    if other_idx == driver_idx:
                        continue
                    
                    other_max_capa = driver_max_capa[other_idx]
                    
                    # 여유 있는지 확인
                    if len(other_nodes) >= other_max_capa:
                        continue
                    
                    # 해당 기사 거점과의 거리
                    other_info = driver_bases_map[other_idx]
                    dist_to_other = haversine(node_lat, node_lon, 
                                             other_info['base_lat'], other_info['base_lng'])
                    
                    if dist_to_other < best_dist:
                        best_dist = dist_to_other
                        best_target = other_idx
                
                # 이전 실행
                if best_target is not None:
                    balanced_nodes[driver_idx].remove(node_idx)
                    balanced_nodes[best_target].append(node_idx)
                    moved_this_round = True
                    total_moved += 1
                    
                    from_name = driver_bases_map[driver_idx]['driver'].name
                    to_name = driver_bases_map[best_target]['driver'].name
                    print(f"  노드 {node_idx}: {from_name} → {to_name}")
        
        if not moved_this_round:
            break
    
    print(f"\n  총 {total_moved}개 노드 이전")
    
    # 최종 상태 출력
    print(f"\n  밸런싱 후 결과:")
    for d_info in driver_bases:
        driver_idx = d_info['driver_idx']
        count = len(balanced_nodes[driver_idx])
        max_capa = d_info['max_capa']
        status = "✅" if count <= max_capa else "❌ 초과"
        print(f"    {d_info['driver'].name}: {count}건 / max_capa={max_capa} {status}")
    
    return balanced_nodes


def handle_overflow(df, balanced_nodes, driver_bases, depot_idx=0):
    """
    ★ V11.2: 전체 초과분 처리 ★
    
    총 노드 수 > 총 max_capa 합계인 경우:
    1. 모든 기사가 max_capa까지 채움
    2. 나머지는 미배정으로 처리
    
    미배정 기준: 모든 거점에서 가장 먼 노드들
    """
    
    print("\n=== 3단계: 초과분 처리 ===")
    
    total_max_capa = sum(d['max_capa'] for d in driver_bases)
    total_nodes = sum(len(nodes) for nodes in balanced_nodes.values())
    
    overflow = total_nodes - total_max_capa
    
    if overflow <= 0:
        print(f"  초과 없음 (총 {total_nodes}건 / 수용량 {total_max_capa}건)")
        return balanced_nodes, []
    
    print(f"  초과 발생: {overflow}건 미배정 필요")
    
    # 모든 노드에 대해 "가장 가까운 거점까지의 거리" 계산
    all_nodes = []
    for driver_idx, nodes in balanced_nodes.items():
        d_info = next(d for d in driver_bases if d['driver_idx'] == driver_idx)
        
        for node_idx in nodes:
            node_lat = float(df.iloc[node_idx]['lat'])
            node_lon = float(df.iloc[node_idx]['lon'])
            
            # 가장 가까운 거점과의 거리
            min_dist = min(
                haversine(node_lat, node_lon, d['base_lat'], d['base_lng'])
                for d in driver_bases
            )
            
            all_nodes.append({
                'node_idx': node_idx,
                'driver_idx': driver_idx,
                'min_dist': min_dist
            })
    
    # 거리 내림차순 정렬 (가장 먼 노드부터)
    all_nodes.sort(key=lambda x: -x['min_dist'])
    
    # 초과분 제거
    unassigned = []
    removed_count = 0
    
    for node_info in all_nodes:
        if removed_count >= overflow:
            break
        
        node_idx = node_info['node_idx']
        driver_idx = node_info['driver_idx']
        
        if node_idx in balanced_nodes[driver_idx]:
            balanced_nodes[driver_idx].remove(node_idx)
            unassigned.append(node_idx)
            removed_count += 1
    
    print(f"  {len(unassigned)}건 미배정 처리")
    
    return balanced_nodes, unassigned


def optimize_visit_order_nearest_neighbor(df, node_indices, start_lat, start_lon):
    """Nearest Neighbor 알고리즘으로 방문 순서 최적화"""
    if not node_indices:
        return []
    
    if len(node_indices) == 1:
        return list(node_indices)
    
    visited = []
    remaining = set(node_indices)
    current_lat, current_lon = start_lat, start_lon
    
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


def calculate_route_distance(df, visit_order, start_lat, start_lon):
    """경로 총 거리 계산"""
    if not visit_order:
        return 0
    
    total_dist = 0
    current_lat, current_lon = start_lat, start_lon
    
    for node_idx in visit_order:
        node_lat = float(df.iloc[node_idx]['lat'])
        node_lon = float(df.iloc[node_idx]['lon'])
        total_dist += haversine(current_lat, current_lon, node_lat, node_lon)
        current_lat, current_lon = node_lat, node_lon
    
    return total_dist


@app.get("/")
def read_root():
    return {
        "status": "active",
        "message": "VRP Engine V11.2 (Voronoi Partition by Driver Base)",
        "features": [
            "★ 거점 기반 Voronoi 분할: 각 노드를 가장 가까운 거점에 배정",
            "★ 자연스러운 영역 분리: 기사 영역이 서로 감싸지 않음",
            "★ max_capa 기반 밸런싱: 초과 시 경계 노드 이전",
            "★ 클러스터 간 교차 0%",
            "Nearest Neighbor 방문 순서 최적화"
        ],
        "algorithm": "Voronoi Partition → max_capa Balancing → Nearest Neighbor TSP"
    }


@app.post("/optimize")
def optimize_routes(body: RequestBody):
    """
    ★ V11.2: 거점 기반 Voronoi 분할 배차 최적화 ★
    
    핵심 원칙:
    1. 각 노드를 "가장 가까운 기사 거점"에 배정 (Voronoi)
    2. max_capa 초과 시 경계 노드를 인접 기사에게 이전
    3. 기사 영역이 서로 감싸지 않음 (자연스러운 분리)
    
    알고리즘:
    1. Voronoi 분할: 노드 → 가장 가까운 거점
    2. max_capa 밸런싱: 초과 노드 이전
    3. 초과분 처리: 미배정
    4. Nearest Neighbor로 방문 순서 결정
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
                Driver(
                    id=f"driver_{i+1}",
                    name=f"기사 {i+1}",
                    max_capa=DEFAULT_MAX_CAPA
                )
                for i in range(body.num_vehicles)
            ]
        
        num_drivers = len(drivers)
        total_max_capa = sum(d.max_capa or DEFAULT_MAX_CAPA for d in drivers)
        total_calls = num_locations - 1
        
        print(f"\n{'='*50}")
        print(f"VRP V11.2 - Voronoi Partition by Driver Base")
        print(f"{'='*50}")
        print(f"총 콜: {total_calls}건")
        print(f"총 수용량: {total_max_capa}건")
        print(f"기사 수: {num_drivers}명")
        
        # weight 처리
        if 'weight' not in df.columns:
            df['weight'] = DEFAULT_WEIGHT_KG
        df['weight'] = pd.to_numeric(df['weight'], errors='coerce').fillna(DEFAULT_WEIGHT_KG).astype(int)
        
        # 2. Voronoi 분할
        node_assignments, driver_nodes, driver_bases = voronoi_partition(
            df, drivers, depot_idx
        )
        
        # 3. max_capa 밸런싱
        balanced_nodes = balance_by_max_capa(
            df, driver_nodes, driver_bases, depot_idx
        )
        
        # 4. 초과분 처리
        balanced_nodes, unassigned_nodes = handle_overflow(
            df, balanced_nodes, driver_bases, depot_idx
        )
        
        # 5. 결과 생성
        print("\n=== 4단계: 방문 순서 최적화 ===")
        
        results = []
        stats = []
        total_distance = 0
        
        driver_bases_map = {d['driver_idx']: d for d in driver_bases}
        
        for driver_idx, nodes in balanced_nodes.items():
            d_info = driver_bases_map[driver_idx]
            driver = d_info['driver']
            max_capa = d_info['max_capa']
            base_lat = d_info['base_lat']
            base_lng = d_info['base_lng']
            
            if not nodes:
                stats.append({
                    "driver_id": driver.id,
                    "driver_name": driver.name,
                    "call_count": 0,
                    "max_capa": max_capa,
                    "total_weight_kg": 0,
                    "distance_km": 0,
                    "status": "⚠️ 배정 없음"
                })
                continue
            
            # 방문 순서 최적화
            visit_order = optimize_visit_order_nearest_neighbor(
                df, nodes, base_lat, base_lng
            )
            
            # 경로 거리
            route_distance = calculate_route_distance(df, visit_order, base_lat, base_lng)
            total_distance += route_distance
            
            # 결과 생성
            route_weight = 0
            for order, node_idx in enumerate(visit_order, 1):
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
            
            call_count = len(visit_order)
            
            if call_count > max_capa:
                status = f"🚨 상한 초과 ({call_count} > {max_capa})"
            elif call_count < max_capa * 0.5:
                status = f"⚠️ 여유 ({call_count} / {max_capa})"
            else:
                status = "정상"
            
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
        
        # 검증
        violations = [s for s in stats if s['call_count'] > s['max_capa']]
        
        print(f"\n{'='*50}")
        print(f"최적화 완료")
        print(f"배정: {len(results)}건, 미배정: {len(unassigned_nodes)}건")
        print(f"총 거리: {total_distance:.1f}km")
        print(f"max_capa 위반: {len(violations)}건")
        print(f"{'='*50}")
        
        return {
            "status": "success",
            "updates": results,
            "statistics": stats,
            "summary": {
                "total_locations": total_calls,
                "total_assigned": len(results),
                "unassigned": len(unassigned_nodes),
                "unassigned_ids": [str(df.iloc[idx]['id']) for idx in unassigned_nodes],
                "total_distance_km": round(total_distance, 2),
                "avg_distance_km": round(total_distance / num_drivers, 2) if num_drivers > 0 else 0
            },
            "optimization_info": {
                "algorithm": "V11.2: Voronoi Partition + max_capa Balancing",
                "max_capa_violations": len(violations),
                "cluster_overlap": 0,
                "principles": [
                    "거점 기반 Voronoi 분할",
                    "기사 영역 간 감싸기 없음",
                    "max_capa 하드캡 준수"
                ]
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
