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


def capacity_aware_bisection(nodes_data, capacities, depth=0):
    """
    ★ V11.4 핵심: Capacity-Aware Recursive Bisection ★
    
    원칙:
    1. 노드들을 경도/위도 기준으로 "선을 그어" 분할
    2. 분할 비율은 capacities에 맞게 조정
    3. 한 영역이 다른 영역을 감싸는 것 불가능 (물리적으로)
    
    Args:
        nodes_data: [{'idx': int, 'lat': float, 'lon': float}, ...]
        capacities: [20, 20, 20, 13] - 각 클러스터가 가져야 할 크기
    
    Returns:
        [cluster0_nodes, cluster1_nodes, ...]
    """
    
    n_clusters = len(capacities)
    
    # 종료 조건: 클러스터 1개
    if n_clusters == 1:
        return [nodes_data]
    
    if len(nodes_data) == 0:
        return [[] for _ in range(n_clusters)]
    
    # 위도/경도 범위 계산
    lats = [n['lat'] for n in nodes_data]
    lons = [n['lon'] for n in nodes_data]
    
    lat_range = max(lats) - min(lats)
    lon_range = max(lons) - min(lons)
    
    # 더 넓은 축으로 분할 (직관적인 경계 생성)
    if lat_range >= lon_range:
        # 위도 기준 분할 (남북)
        sorted_nodes = sorted(nodes_data, key=lambda n: n['lat'])
        axis = 'lat'
    else:
        # 경도 기준 분할 (동서)
        sorted_nodes = sorted(nodes_data, key=lambda n: n['lon'])
        axis = 'lon'
    
    # 클러스터를 두 그룹으로 나누기
    left_clusters = n_clusters // 2
    right_clusters = n_clusters - left_clusters
    
    left_capacities = capacities[:left_clusters]
    right_capacities = capacities[left_clusters:]
    
    # 분할 비율 계산 (capacity 비율에 따라)
    total_left_cap = sum(left_capacities)
    total_right_cap = sum(right_capacities)
    total_cap = total_left_cap + total_right_cap
    
    # 분할 위치: capacity 비율에 맞게
    split_ratio = total_left_cap / total_cap
    split_idx = int(len(sorted_nodes) * split_ratio)
    
    # 최소 1개는 각 그룹에
    split_idx = max(1, min(split_idx, len(sorted_nodes) - 1))
    
    left_nodes = sorted_nodes[:split_idx]
    right_nodes = sorted_nodes[split_idx:]
    
    indent = "  " * depth
    print(f"{indent}분할 ({axis}): 왼쪽 {len(left_nodes)}개 (cap={total_left_cap}), 오른쪽 {len(right_nodes)}개 (cap={total_right_cap})")
    
    # 재귀 호출
    left_result = capacity_aware_bisection(left_nodes, left_capacities, depth + 1)
    right_result = capacity_aware_bisection(right_nodes, right_capacities, depth + 1)
    
    return left_result + right_result


def assign_clusters_to_drivers(clusters, drivers, df, depot_idx=0):
    """
    ★ V11.4: 클러스터-기사 매칭 ★
    
    원칙:
    - 클러스터 크기와 기사 max_capa가 이미 맞춰져 있음
    - 기사 거점과 클러스터 중심 거리로 최적 매칭
    """
    
    print("\n=== 2단계: 클러스터-기사 매칭 ===")
    
    n_clusters = len(clusters)
    n_drivers = len(drivers)
    
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
    
    # 클러스터 중심점 계산
    cluster_centers = []
    for i, cluster in enumerate(clusters):
        if cluster:
            center_lat = np.mean([n['lat'] for n in cluster])
            center_lon = np.mean([n['lon'] for n in cluster])
        else:
            center_lat, center_lon = 0, 0
        cluster_centers.append((center_lat, center_lon))
        print(f"  클러스터 {i}: {len(cluster)}건, 중심=({center_lat:.4f}, {center_lon:.4f})")
    
    # Greedy 매칭: 각 클러스터에 가장 적합한 기사 배정
    # 기준: 크기 매칭 + 거리
    cluster_to_driver = {}
    used_drivers = set()
    
    # 클러스터를 크기 내림차순으로 처리
    cluster_order = sorted(range(n_clusters), key=lambda i: -len(clusters[i]))
    
    for cluster_idx in cluster_order:
        cluster = clusters[cluster_idx]
        center = cluster_centers[cluster_idx]
        cluster_size = len(cluster)
        
        best_driver = None
        best_score = float('inf')
        
        for d_info in driver_info:
            if d_info['driver_idx'] in used_drivers:
                continue
            
            # 점수: |클러스터 크기 - max_capa| * 1000 + 거리
            size_diff = abs(cluster_size - d_info['max_capa'])
            dist = haversine(center[0], center[1], d_info['base_lat'], d_info['base_lng'])
            score = size_diff * 1000 + dist
            
            if score < best_score:
                best_score = score
                best_driver = d_info
        
        if best_driver:
            cluster_to_driver[cluster_idx] = best_driver['driver_idx']
            used_drivers.add(best_driver['driver_idx'])
            print(f"  클러스터 {cluster_idx} ({cluster_size}건) → {best_driver['driver'].name} (max_capa={best_driver['max_capa']})")
    
    return cluster_to_driver, driver_info


def trim_excess_nodes(clusters, cluster_to_driver, driver_info, df):
    """
    ★ V11.4: 초과 노드 제거 (미배정 처리) ★
    
    클러스터 내 노드 수 > 기사 max_capa인 경우:
    - 클러스터 중심에서 가장 먼 노드를 미배정 처리
    - 다른 클러스터로 이전하지 않음! (경계 유지)
    """
    
    print("\n=== 3단계: 초과 노드 처리 ===")
    
    driver_max_capa = {d['driver_idx']: d['max_capa'] for d in driver_info}
    
    trimmed_clusters = []
    all_unassigned = []
    
    for cluster_idx, cluster in enumerate(clusters):
        if cluster_idx not in cluster_to_driver:
            trimmed_clusters.append(cluster)
            continue
        
        driver_idx = cluster_to_driver[cluster_idx]
        max_capa = driver_max_capa[driver_idx]
        
        if len(cluster) <= max_capa:
            trimmed_clusters.append(cluster)
            continue
        
        # 초과분 제거
        excess = len(cluster) - max_capa
        
        # 클러스터 중심
        center_lat = np.mean([n['lat'] for n in cluster])
        center_lon = np.mean([n['lon'] for n in cluster])
        
        # 중심에서 거리 계산
        nodes_with_dist = []
        for node in cluster:
            dist = haversine(center_lat, center_lon, node['lat'], node['lon'])
            nodes_with_dist.append((node, dist))
        
        # 거리순 정렬 (가까운 것부터)
        nodes_with_dist.sort(key=lambda x: x[1])
        
        # max_capa만큼만 유지, 나머지는 미배정
        kept = [n[0] for n in nodes_with_dist[:max_capa]]
        removed = [n[0] for n in nodes_with_dist[max_capa:]]
        
        trimmed_clusters.append(kept)
        all_unassigned.extend(removed)
        
        d_info = next(d for d in driver_info if d['driver_idx'] == driver_idx)
        print(f"  클러스터 {cluster_idx} ({d_info['driver'].name}): {len(cluster)} → {len(kept)}건 (미배정 {len(removed)}건)")
    
    print(f"  총 미배정: {len(all_unassigned)}건")
    
    return trimmed_clusters, all_unassigned


def optimize_visit_order(df, nodes, start_lat, start_lon):
    """Nearest Neighbor TSP"""
    if not nodes:
        return []
    if len(nodes) == 1:
        return [nodes[0]['idx']]
    
    visited = []
    remaining = list(nodes)
    current_lat, current_lon = start_lat, start_lon
    
    while remaining:
        nearest_idx = min(range(len(remaining)), key=lambda i: haversine(
            current_lat, current_lon,
            remaining[i]['lat'], remaining[i]['lon']
        ))
        nearest = remaining.pop(nearest_idx)
        visited.append(nearest['idx'])
        current_lat, current_lon = nearest['lat'], nearest['lon']
    
    return visited


def calculate_route_distance(df, visit_order, start_lat, start_lon):
    """경로 거리 계산"""
    if not visit_order:
        return 0
    
    total = 0
    current_lat, current_lon = start_lat, start_lon
    
    for node_idx in visit_order:
        node_lat = float(df.iloc[node_idx]['lat'])
        node_lon = float(df.iloc[node_idx]['lon'])
        total += haversine(current_lat, current_lon, node_lat, node_lon)
        current_lat, current_lon = node_lat, node_lon
    
    return total


@app.get("/")
def read_root():
    return {
        "status": "active",
        "message": "VRP Engine V11.4 (Capacity-Aware Recursive Bisection)",
        "features": [
            "★ 경도/위도 기준 선 긋기 (물리적 분할)",
            "★ capacity 비율에 따른 분할 크기 조정",
            "★ 한 영역이 다른 영역을 감쌀 수 없음",
            "★ 경계 조정 없음 (다른 클러스터로 이전 X)",
            "★ 초과분은 미배정 처리",
            "Nearest Neighbor TSP"
        ],
        "algorithm": "Capacity-Aware Recursive Bisection → Driver Matching → Trim Excess"
    }


@app.post("/optimize")
def optimize_routes(body: RequestBody):
    """
    ★ V11.4: Capacity-Aware Recursive Bisection ★
    
    핵심 원칙:
    1. 경도/위도 기준으로 "선을 그어" 분할 (감싸기 불가능)
    2. 분할 크기는 max_capa 비율에 맞게
    3. 초과분은 미배정 (다른 클러스터로 이전 X)
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
        
        num_drivers = len(drivers)
        
        # max_capa 리스트 (내림차순 정렬)
        capacities = sorted([d.max_capa or DEFAULT_MAX_CAPA for d in drivers], reverse=True)
        total_max_capa = sum(capacities)
        total_calls = num_locations - 1
        
        print(f"\n{'='*50}")
        print(f"VRP V11.4 - Capacity-Aware Recursive Bisection")
        print(f"{'='*50}")
        print(f"총 콜: {total_calls}건, 수용량: {total_max_capa}건")
        print(f"Capacities (정렬): {capacities}")
        
        if 'weight' not in df.columns:
            df['weight'] = DEFAULT_WEIGHT_KG
        df['weight'] = pd.to_numeric(df['weight'], errors='coerce').fillna(DEFAULT_WEIGHT_KG).astype(int)
        
        # 노드 데이터 준비 (depot 제외)
        nodes_data = []
        for i in range(len(df)):
            if i == depot_idx:
                continue
            nodes_data.append({
                'idx': i,
                'lat': float(df.iloc[i]['lat']),
                'lon': float(df.iloc[i]['lon']),
                'weight': int(df.iloc[i]['weight'])
            })
        
        # 2. Capacity-Aware Recursive Bisection
        print("\n=== 1단계: Capacity-Aware Bisection ===")
        clusters = capacity_aware_bisection(nodes_data, capacities)
        
        print(f"\n  분할 결과: {[len(c) for c in clusters]}")
        
        # 3. 클러스터-기사 매칭
        cluster_to_driver, driver_info = assign_clusters_to_drivers(clusters, drivers, df, depot_idx)
        
        # 4. 초과 노드 제거
        trimmed_clusters, unassigned_nodes = trim_excess_nodes(clusters, cluster_to_driver, driver_info, df)
        
        # 5. 결과 생성
        print("\n=== 4단계: 방문 순서 최적화 ===")
        
        results = []
        stats = []
        total_distance = 0
        
        driver_info_map = {d['driver_idx']: d for d in driver_info}
        
        for cluster_idx, cluster in enumerate(trimmed_clusters):
            if cluster_idx not in cluster_to_driver:
                continue
            
            driver_idx = cluster_to_driver[cluster_idx]
            d_info = driver_info_map[driver_idx]
            driver = d_info['driver']
            max_capa = d_info['max_capa']
            base_lat = d_info['base_lat']
            base_lng = d_info['base_lng']
            
            if not cluster:
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
            
            visit_order = optimize_visit_order(df, cluster, base_lat, base_lng)
            route_distance = calculate_route_distance(df, visit_order, base_lat, base_lng)
            total_distance += route_distance
            
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
        
        violations = [s for s in stats if s['call_count'] > s['max_capa']]
        unassigned_ids = [str(df.iloc[n['idx']]['id']) for n in unassigned_nodes]
        
        print(f"\n{'='*50}")
        print(f"완료: 배정 {len(results)}건, 미배정 {len(unassigned_nodes)}건")
        print(f"{'='*50}")
        
        return {
            "status": "success",
            "updates": results,
            "statistics": stats,
            "summary": {
                "total_locations": total_calls,
                "total_assigned": len(results),
                "unassigned": len(unassigned_nodes),
                "unassigned_ids": unassigned_ids,
                "total_distance_km": round(total_distance, 2)
            },
            "optimization_info": {
                "algorithm": "V11.4: Capacity-Aware Recursive Bisection",
                "max_capa_violations": len(violations),
                "cluster_overlap": 0,
                "note": "경도/위도 기준 선 긋기로 영역 분할, 감싸기 불가능"
            }
        }
        
    except Exception as e:
        traceback.print_exc()
        return {"status": "error", "message": str(e)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
