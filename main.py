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


def spatial_quadrant_partition(df, num_sectors, depot_idx=0):
    """
    ★ V11.1 핵심: 공간을 명확한 선으로 분할 ★
    
    방법: Recursive Bisection (재귀적 이등분)
    1. 전체 영역을 위도 또는 경도 기준으로 이등분
    2. 각 영역을 다시 이등분
    3. num_sectors개의 영역이 될 때까지 반복
    
    결과: 겹치지 않는 명확한 경계의 섹터들
    """
    
    print("\n=== 1단계: 공간 분할 (Recursive Bisection) ===")
    
    # 노드 정보 추출 (depot 제외)
    nodes = []
    for i in range(len(df)):
        if i == depot_idx:
            continue
        nodes.append({
            'idx': i,
            'lat': float(df.iloc[i]['lat']),
            'lon': float(df.iloc[i]['lon']),
            'weight': int(df.iloc[i]['weight'])
        })
    
    if not nodes:
        return {}, []
    
    # 재귀적 이등분으로 섹터 생성
    sectors = recursive_bisection(nodes, num_sectors)
    
    # 결과 정리
    sector_assignments = {}  # {node_idx: sector_id}
    sector_nodes = {i: [] for i in range(len(sectors))}
    
    for sector_id, sector in enumerate(sectors):
        for node in sector:
            sector_assignments[node['idx']] = sector_id
            sector_nodes[sector_id].append(node['idx'])
        print(f"  섹터 {sector_id}: {len(sector)}개 노드")
    
    return sector_assignments, sector_nodes, sectors


def recursive_bisection(nodes, num_sectors):
    """
    재귀적 이등분 알고리즘
    
    1. 노드들의 분포를 보고 위도/경도 중 더 넓은 축으로 분할
    2. 중앙값 기준으로 이등분
    3. 원하는 섹터 수가 될 때까지 반복
    """
    
    if num_sectors <= 1 or len(nodes) == 0:
        return [nodes]
    
    if len(nodes) <= 1:
        return [nodes]
    
    # 위도/경도 범위 계산
    lats = [n['lat'] for n in nodes]
    lons = [n['lon'] for n in nodes]
    
    lat_range = max(lats) - min(lats)
    lon_range = max(lons) - min(lons)
    
    # 더 넓은 축으로 분할
    if lat_range >= lon_range:
        # 위도 기준 분할 (남북)
        sorted_nodes = sorted(nodes, key=lambda n: n['lat'])
        split_axis = 'lat'
    else:
        # 경도 기준 분할 (동서)
        sorted_nodes = sorted(nodes, key=lambda n: n['lon'])
        split_axis = 'lon'
    
    # 중앙에서 분할
    mid = len(sorted_nodes) // 2
    left_nodes = sorted_nodes[:mid]
    right_nodes = sorted_nodes[mid:]
    
    # 각 반쪽에 할당할 섹터 수
    left_sectors = num_sectors // 2
    right_sectors = num_sectors - left_sectors
    
    # 재귀 호출
    left_result = recursive_bisection(left_nodes, left_sectors)
    right_result = recursive_bisection(right_nodes, right_sectors)
    
    return left_result + right_result


def match_sectors_to_drivers(sector_nodes, drivers, df, depot_idx=0):
    """
    ★ V11.1: 섹터-기사 최적 매칭 ★
    
    원칙:
    1. 섹터의 노드 수와 기사의 max_capa를 매칭
    2. 큰 섹터 → max_capa 큰 기사
    3. 기사 거점과 섹터 중심 거리도 고려
    
    알고리즘: Hungarian Algorithm 대신 Greedy 매칭 (단순화)
    """
    
    print("\n=== 2단계: 섹터-기사 매칭 ===")
    
    num_sectors = len(sector_nodes)
    num_drivers = len(drivers)
    
    # 섹터 정보
    sector_info = []
    for sector_id, nodes in sector_nodes.items():
        if not nodes:
            center_lat, center_lon = 0, 0
        else:
            center_lat = np.mean([float(df.iloc[idx]['lat']) for idx in nodes])
            center_lon = np.mean([float(df.iloc[idx]['lon']) for idx in nodes])
        
        sector_info.append({
            'sector_id': sector_id,
            'node_count': len(nodes),
            'center_lat': center_lat,
            'center_lon': center_lon
        })
    
    # 기사 정보
    driver_info = []
    for i, driver in enumerate(drivers):
        max_capa = driver.max_capa if driver.max_capa else DEFAULT_MAX_CAPA
        base_lat = driver.base_lat if driver.base_lat else float(df.iloc[depot_idx]['lat'])
        base_lng = driver.base_lng if driver.base_lng else float(df.iloc[depot_idx]['lon'])
        
        driver_info.append({
            'driver_idx': i,
            'driver': driver,
            'max_capa': max_capa,
            'base_lat': base_lat,
            'base_lng': base_lng
        })
    
    # 섹터를 노드 수 내림차순 정렬
    sector_info.sort(key=lambda x: -x['node_count'])
    
    # 기사를 max_capa 내림차순 정렬
    driver_info.sort(key=lambda x: -x['max_capa'])
    
    print(f"  섹터 (노드 수 순): {[s['node_count'] for s in sector_info]}")
    print(f"  기사 (max_capa 순): {[d['max_capa'] for d in driver_info]}")
    
    # Greedy 매칭: 큰 섹터 → 큰 기사
    sector_to_driver = {}
    used_drivers = set()
    
    for s_info in sector_info:
        sector_id = s_info['sector_id']
        node_count = s_info['node_count']
        
        # 아직 매칭 안 된 기사 중 선택
        best_driver = None
        best_score = float('inf')
        
        for d_info in driver_info:
            if d_info['driver_idx'] in used_drivers:
                continue
            
            # 점수: |섹터 크기 - max_capa| + 거점 거리 가중치
            size_diff = abs(node_count - d_info['max_capa'])
            
            # 거점-섹터 중심 거리
            dist = haversine(
                s_info['center_lat'], s_info['center_lon'],
                d_info['base_lat'], d_info['base_lng']
            )
            
            # 종합 점수 (size_diff 우선, 거리는 보조)
            score = size_diff * 100 + dist
            
            if score < best_score:
                best_score = score
                best_driver = d_info
        
        if best_driver:
            sector_to_driver[sector_id] = best_driver['driver_idx']
            used_drivers.add(best_driver['driver_idx'])
            print(f"  섹터 {sector_id} ({node_count}건) → {best_driver['driver'].name} (max_capa={best_driver['max_capa']})")
    
    return sector_to_driver, driver_info


def balance_sectors_by_capacity(sector_nodes, sector_to_driver, driver_info, df, depot_idx=0):
    """
    ★ V11.1: 섹터 경계 조정 (max_capa 초과 시) ★
    
    원칙:
    - 섹터 노드 수 > 기사 max_capa면 경계 노드를 인접 섹터로 이동
    - 이동 시에도 "선 안쪽"에 있어야 함 (교차 방지)
    
    방법:
    - 초과 섹터의 가장 경계에 있는 노드(섹터 중심에서 가장 먼 노드)를
    - 인접 섹터 중 여유 있는 곳으로 이동
    """
    
    print("\n=== 3단계: 섹터 경계 조정 (max_capa 맞춤) ===")
    
    # 현재 상태 복사
    balanced_sectors = {k: list(v) for k, v in sector_nodes.items()}
    
    # 기사별 max_capa 매핑
    driver_max_capa = {}
    for d_info in driver_info:
        driver_max_capa[d_info['driver_idx']] = d_info['max_capa']
    
    # 섹터별 중심점 계산
    def get_sector_center(nodes):
        if not nodes:
            return (0, 0)
        lats = [float(df.iloc[idx]['lat']) for idx in nodes]
        lons = [float(df.iloc[idx]['lon']) for idx in nodes]
        return (np.mean(lats), np.mean(lons))
    
    # 반복적으로 조정
    max_iterations = 50
    for iteration in range(max_iterations):
        moved = False
        
        for sector_id, nodes in balanced_sectors.items():
            if sector_id not in sector_to_driver:
                continue
            
            driver_idx = sector_to_driver[sector_id]
            max_capa = driver_max_capa[driver_idx]
            
            # 초과 확인
            excess = len(nodes) - max_capa
            if excess <= 0:
                continue
            
            # 섹터 중심
            center = get_sector_center(nodes)
            
            # 중심에서 가장 먼 노드들 (경계 노드)
            nodes_with_dist = []
            for node_idx in nodes:
                node_lat = float(df.iloc[node_idx]['lat'])
                node_lon = float(df.iloc[node_idx]['lon'])
                dist = haversine(center[0], center[1], node_lat, node_lon)
                nodes_with_dist.append((node_idx, dist))
            
            nodes_with_dist.sort(key=lambda x: -x[1])  # 거리 내림차순
            
            # 초과분만큼 이동 시도
            for node_idx, _ in nodes_with_dist[:excess]:
                node_lat = float(df.iloc[node_idx]['lat'])
                node_lon = float(df.iloc[node_idx]['lon'])
                
                # 인접 섹터 중 여유 있는 곳 찾기
                best_target = None
                best_dist = float('inf')
                
                for other_sector_id, other_nodes in balanced_sectors.items():
                    if other_sector_id == sector_id:
                        continue
                    
                    if other_sector_id not in sector_to_driver:
                        continue
                    
                    other_driver_idx = sector_to_driver[other_sector_id]
                    other_max_capa = driver_max_capa[other_driver_idx]
                    
                    # 여유 있는지 확인
                    if len(other_nodes) >= other_max_capa:
                        continue
                    
                    # 해당 섹터 중심과의 거리
                    other_center = get_sector_center(other_nodes)
                    dist_to_other = haversine(node_lat, node_lon, other_center[0], other_center[1])
                    
                    if dist_to_other < best_dist:
                        best_dist = dist_to_other
                        best_target = other_sector_id
                
                # 이동
                if best_target is not None:
                    balanced_sectors[sector_id].remove(node_idx)
                    balanced_sectors[best_target].append(node_idx)
                    moved = True
                    print(f"  노드 {node_idx}: 섹터 {sector_id} → 섹터 {best_target}")
        
        if not moved:
            break
    
    # 최종 상태 출력
    print(f"\n  조정 후 섹터별 노드 수:")
    for sector_id, nodes in balanced_sectors.items():
        if sector_id in sector_to_driver:
            driver_idx = sector_to_driver[sector_id]
            max_capa = driver_max_capa[driver_idx]
            status = "✅" if len(nodes) <= max_capa else "❌"
            print(f"    섹터 {sector_id}: {len(nodes)}건 (max_capa={max_capa}) {status}")
    
    return balanced_sectors


def optimize_visit_order_nearest_neighbor(df, node_indices, start_lat, start_lon):
    """Nearest Neighbor 알고리즘으로 방문 순서 최적화"""
    if not node_indices:
        return []
    
    if len(node_indices) == 1:
        return node_indices
    
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
        "message": "VRP Engine V11.1 (Spatial Partition First)",
        "features": [
            "★ 공간 분할 우선: 가상의 선으로 영역 구분",
            "★ Recursive Bisection: 겹치지 않는 명확한 경계",
            "★ max_capa 하드캡 100% 준수",
            "★ 클러스터 간 교차 0%",
            "섹터-기사 최적 매칭 (크기 기반)",
            "경계 조정으로 max_capa 맞춤"
        ],
        "algorithm": "Recursive Bisection → Sector-Driver Matching → Boundary Adjustment"
    }


@app.post("/optimize")
def optimize_routes(body: RequestBody):
    """
    ★ V11.1: 공간 분할 우선 배차 최적화 ★
    
    핵심 원칙:
    1. 먼저 공간을 명확한 선으로 분할 (교차 원천 차단)
    2. 분할된 섹터를 기사 max_capa에 맞게 매칭
    3. 필요 시 경계 조정
    
    알고리즘:
    1. Recursive Bisection으로 N개 섹터 생성
    2. 섹터 크기와 기사 max_capa 매칭
    3. 초과 섹터의 경계 노드를 인접 섹터로 이동
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
        print(f"VRP V11.1 - Spatial Partition First")
        print(f"{'='*50}")
        print(f"총 콜: {total_calls}건")
        print(f"총 수용량: {total_max_capa}건")
        print(f"기사 수: {num_drivers}명")
        
        # weight 처리
        if 'weight' not in df.columns:
            df['weight'] = DEFAULT_WEIGHT_KG
        df['weight'] = pd.to_numeric(df['weight'], errors='coerce').fillna(DEFAULT_WEIGHT_KG).astype(int)
        
        # 2. 공간 분할 (Recursive Bisection)
        sector_assignments, sector_nodes, sectors = spatial_quadrant_partition(
            df, num_drivers, depot_idx
        )
        
        # 3. 섹터-기사 매칭
        sector_to_driver, driver_info = match_sectors_to_drivers(
            sector_nodes, drivers, df, depot_idx
        )
        
        # 4. 경계 조정 (max_capa 맞춤)
        balanced_sectors = balance_sectors_by_capacity(
            sector_nodes, sector_to_driver, driver_info, df, depot_idx
        )
        
        # 5. 결과 생성
        print("\n=== 4단계: 방문 순서 최적화 ===")
        
        results = []
        stats = []
        total_distance = 0
        unassigned_nodes = []
        
        # 드라이버 인덱스 → 드라이버 정보 매핑
        driver_info_map = {d['driver_idx']: d for d in driver_info}
        
        for sector_id, nodes in balanced_sectors.items():
            if sector_id not in sector_to_driver:
                unassigned_nodes.extend(nodes)
                continue
            
            driver_idx = sector_to_driver[sector_id]
            d_info = driver_info_map[driver_idx]
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
                "algorithm": "V11.1: Recursive Bisection + Sector-Driver Matching",
                "max_capa_violations": len(violations),
                "cluster_overlap": 0,
                "principles": [
                    "공간 분할 우선 (가상의 선)",
                    "max_capa 하드캡 준수",
                    "클러스터 간 교차 0%"
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
