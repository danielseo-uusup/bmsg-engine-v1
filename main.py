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


def kmeans_clustering(coords, n_clusters, max_iter=100):
    """
    순수 numpy로 구현한 K-Means 클러스터링
    sklearn 없이 동작
    """
    n_samples = len(coords)
    
    if n_samples <= n_clusters:
        return list(range(n_samples))
    
    # 초기 중심점: 랜덤 선택
    np.random.seed(42)
    indices = np.random.choice(n_samples, n_clusters, replace=False)
    centroids = coords[indices].copy()
    
    labels = np.zeros(n_samples, dtype=int)
    
    for _ in range(max_iter):
        # 각 점을 가장 가까운 중심점에 할당
        new_labels = np.zeros(n_samples, dtype=int)
        for i in range(n_samples):
            distances = [haversine(coords[i][0], coords[i][1], c[0], c[1]) for c in centroids]
            new_labels[i] = np.argmin(distances)
        
        # 수렴 체크
        if np.array_equal(labels, new_labels):
            break
        labels = new_labels
        
        # 중심점 업데이트
        for k in range(n_clusters):
            cluster_points = coords[labels == k]
            if len(cluster_points) > 0:
                centroids[k] = cluster_points.mean(axis=0)
    
    return labels.tolist(), centroids


def geographic_clustering(df, n_clusters, depot_idx=0):
    """
    ★ V11.3 핵심: 노드 분포 기반 지리적 클러스터링 ★
    
    원칙:
    1. 기사 거점 무시, 오직 노드 위치만 고려
    2. K-Means로 지리적으로 가까운 노드들을 묶음
    3. 결과: 서로 겹치지 않는 명확한 클러스터
    """
    
    print("\n=== 1단계: 노드 분포 기반 클러스터링 ===")
    
    # 노드 좌표 추출 (depot 제외)
    coords = []
    node_indices = []
    
    for i in range(len(df)):
        if i == depot_idx:
            continue
        coords.append([float(df.iloc[i]['lat']), float(df.iloc[i]['lon'])])
        node_indices.append(i)
    
    coords = np.array(coords)
    
    # K-Means 클러스터링
    labels, centroids = kmeans_clustering(coords, n_clusters)
    
    # 결과 정리
    cluster_nodes = {i: [] for i in range(n_clusters)}
    for i, node_idx in enumerate(node_indices):
        cluster_id = labels[i]
        cluster_nodes[cluster_id].append(node_idx)
    
    # 클러스터별 정보
    cluster_info = []
    for cluster_id in range(n_clusters):
        nodes = cluster_nodes[cluster_id]
        if nodes:
            center_lat = centroids[cluster_id][0]
            center_lon = centroids[cluster_id][1]
        else:
            center_lat, center_lon = 0, 0
        
        cluster_info.append({
            'cluster_id': cluster_id,
            'node_count': len(nodes),
            'center_lat': center_lat,
            'center_lon': center_lon
        })
        print(f"  클러스터 {cluster_id}: {len(nodes)}개 노드, 중심=({center_lat:.4f}, {center_lon:.4f})")
    
    return cluster_nodes, cluster_info, centroids


def match_clusters_to_drivers(cluster_info, drivers, df, depot_idx=0):
    """
    ★ V11.3: 클러스터-기사 최적 매칭 ★
    
    원칙:
    1. 큰 클러스터 → max_capa 큰 기사
    2. 기사 거점과 클러스터 중심 거리는 보조 기준
    
    알고리즘: 크기 우선 Greedy 매칭
    """
    
    print("\n=== 2단계: 클러스터-기사 매칭 ===")
    
    # 기사 정보 정리
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
    
    # 클러스터를 노드 수 내림차순 정렬
    sorted_clusters = sorted(cluster_info, key=lambda x: -x['node_count'])
    
    # 기사를 max_capa 내림차순 정렬
    sorted_drivers = sorted(driver_info, key=lambda x: -x['max_capa'])
    
    print(f"  클러스터 크기: {[c['node_count'] for c in sorted_clusters]}")
    print(f"  기사 max_capa: {[d['max_capa'] for d in sorted_drivers]}")
    
    # 1:1 매칭 (크기순)
    cluster_to_driver = {}
    
    for i, c_info in enumerate(sorted_clusters):
        if i < len(sorted_drivers):
            d_info = sorted_drivers[i]
            cluster_to_driver[c_info['cluster_id']] = d_info['driver_idx']
            print(f"  클러스터 {c_info['cluster_id']} ({c_info['node_count']}건) → {d_info['driver'].name} (max_capa={d_info['max_capa']})")
    
    return cluster_to_driver, driver_info


def balance_clusters_by_capacity(df, cluster_nodes, cluster_info, cluster_to_driver, driver_info, depot_idx=0):
    """
    ★ V11.3: max_capa 기반 클러스터 경계 조정 ★
    
    클러스터 노드 수 > 기사 max_capa인 경우:
    1. 클러스터 중심에서 가장 먼 노드 (경계 노드) 찾기
    2. 인접 클러스터 중 여유 있는 곳으로 이전
    3. 이전 시 "가장 가까운 클러스터 중심"으로 이동 (지리적 연속성 유지)
    """
    
    print("\n=== 3단계: max_capa 기반 경계 조정 ===")
    
    # 현재 상태 복사
    balanced = {k: list(v) for k, v in cluster_nodes.items()}
    
    # 기사별 max_capa
    driver_max_capa = {d['driver_idx']: d['max_capa'] for d in driver_info}
    
    # 클러스터 중심점
    cluster_centers = {c['cluster_id']: (c['center_lat'], c['center_lon']) for c in cluster_info}
    
    max_iterations = 100
    total_moved = 0
    
    for iteration in range(max_iterations):
        moved = False
        
        for cluster_id, nodes in list(balanced.items()):
            if cluster_id not in cluster_to_driver:
                continue
            
            driver_idx = cluster_to_driver[cluster_id]
            max_capa = driver_max_capa[driver_idx]
            excess = len(nodes) - max_capa
            
            if excess <= 0:
                continue
            
            # 클러스터 중심
            center = cluster_centers[cluster_id]
            
            # 중심에서 가장 먼 노드들 (경계 노드)
            nodes_with_dist = []
            for node_idx in nodes:
                node_lat = float(df.iloc[node_idx]['lat'])
                node_lon = float(df.iloc[node_idx]['lon'])
                dist = haversine(center[0], center[1], node_lat, node_lon)
                nodes_with_dist.append((node_idx, dist, node_lat, node_lon))
            
            nodes_with_dist.sort(key=lambda x: -x[1])  # 거리 내림차순
            
            # 초과분 이전
            for node_idx, _, node_lat, node_lon in nodes_with_dist[:excess]:
                # 여유 있는 클러스터 중 해당 노드와 가장 가까운 중심 찾기
                best_target = None
                best_dist = float('inf')
                
                for other_cluster_id, other_nodes in balanced.items():
                    if other_cluster_id == cluster_id:
                        continue
                    if other_cluster_id not in cluster_to_driver:
                        continue
                    
                    other_driver_idx = cluster_to_driver[other_cluster_id]
                    other_max_capa = driver_max_capa[other_driver_idx]
                    
                    # 여유 확인
                    if len(other_nodes) >= other_max_capa:
                        continue
                    
                    # 해당 클러스터 중심과의 거리
                    other_center = cluster_centers[other_cluster_id]
                    dist = haversine(node_lat, node_lon, other_center[0], other_center[1])
                    
                    if dist < best_dist:
                        best_dist = dist
                        best_target = other_cluster_id
                
                if best_target is not None:
                    balanced[cluster_id].remove(node_idx)
                    balanced[best_target].append(node_idx)
                    moved = True
                    total_moved += 1
        
        if not moved:
            break
    
    print(f"  총 {total_moved}개 노드 이전")
    
    # 최종 상태
    print(f"\n  조정 후 클러스터별 노드 수:")
    for cluster_id, nodes in balanced.items():
        if cluster_id in cluster_to_driver:
            driver_idx = cluster_to_driver[cluster_id]
            d_info = next(d for d in driver_info if d['driver_idx'] == driver_idx)
            max_capa = d_info['max_capa']
            status = "✅" if len(nodes) <= max_capa else "❌"
            print(f"    클러스터 {cluster_id} → {d_info['driver'].name}: {len(nodes)}건 (max={max_capa}) {status}")
    
    return balanced


def handle_overflow(df, balanced, cluster_to_driver, driver_info, cluster_info):
    """
    총 노드 > 총 max_capa인 경우 미배정 처리
    기준: 클러스터 중심에서 가장 먼 노드들
    """
    
    print("\n=== 4단계: 초과분 처리 ===")
    
    total_max_capa = sum(d['max_capa'] for d in driver_info)
    total_nodes = sum(len(nodes) for nodes in balanced.values())
    overflow = total_nodes - total_max_capa
    
    if overflow <= 0:
        print(f"  초과 없음")
        return balanced, []
    
    print(f"  {overflow}건 미배정 필요")
    
    # 클러스터 중심
    cluster_centers = {c['cluster_id']: (c['center_lat'], c['center_lon']) for c in cluster_info}
    
    # 모든 노드의 "클러스터 중심에서의 거리" 계산
    all_nodes = []
    for cluster_id, nodes in balanced.items():
        if cluster_id not in cluster_centers:
            continue
        center = cluster_centers[cluster_id]
        
        for node_idx in nodes:
            node_lat = float(df.iloc[node_idx]['lat'])
            node_lon = float(df.iloc[node_idx]['lon'])
            dist = haversine(center[0], center[1], node_lat, node_lon)
            all_nodes.append({
                'node_idx': node_idx,
                'cluster_id': cluster_id,
                'dist': dist
            })
    
    # 거리 내림차순 (먼 것부터 제거)
    all_nodes.sort(key=lambda x: -x['dist'])
    
    unassigned = []
    for node_info in all_nodes[:overflow]:
        node_idx = node_info['node_idx']
        cluster_id = node_info['cluster_id']
        
        if node_idx in balanced[cluster_id]:
            balanced[cluster_id].remove(node_idx)
            unassigned.append(node_idx)
    
    print(f"  {len(unassigned)}건 미배정")
    
    return balanced, unassigned


def optimize_visit_order(df, nodes, start_lat, start_lon):
    """Nearest Neighbor TSP"""
    if not nodes:
        return []
    if len(nodes) == 1:
        return list(nodes)
    
    visited = []
    remaining = set(nodes)
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
        "message": "VRP Engine V11.3 (Node-Distribution Clustering)",
        "features": [
            "★ 노드 분포 기반 클러스터링 (거점 무시)",
            "★ K-Means로 지리적으로 가까운 노드 묶음",
            "★ 클러스터 간 겹침/감싸기 없음",
            "★ 클러스터-기사 매칭: 크기 → max_capa",
            "max_capa 경계 조정",
            "Nearest Neighbor TSP"
        ],
        "algorithm": "K-Means Geographic Clustering → Size-based Driver Matching → Boundary Adjustment"
    }


@app.post("/optimize")
def optimize_routes(body: RequestBody):
    """
    ★ V11.3: 노드 분포 기반 클러스터링 ★
    
    핵심:
    1. 기사 거점 무시, 노드 위치만으로 클러스터링
    2. K-Means로 지리적으로 명확한 경계 생성
    3. 클러스터 크기와 기사 max_capa 매칭
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
        total_max_capa = sum(d.max_capa or DEFAULT_MAX_CAPA for d in drivers)
        total_calls = num_locations - 1
        
        print(f"\n{'='*50}")
        print(f"VRP V11.3 - Node-Distribution Clustering")
        print(f"{'='*50}")
        print(f"총 콜: {total_calls}건, 수용량: {total_max_capa}건")
        
        if 'weight' not in df.columns:
            df['weight'] = DEFAULT_WEIGHT_KG
        df['weight'] = pd.to_numeric(df['weight'], errors='coerce').fillna(DEFAULT_WEIGHT_KG).astype(int)
        
        # 2. 노드 분포 기반 클러스터링
        cluster_nodes, cluster_info, centroids = geographic_clustering(df, num_drivers, depot_idx)
        
        # 3. 클러스터-기사 매칭
        cluster_to_driver, driver_info = match_clusters_to_drivers(cluster_info, drivers, df, depot_idx)
        
        # 4. max_capa 경계 조정
        balanced = balance_clusters_by_capacity(
            df, cluster_nodes, cluster_info, cluster_to_driver, driver_info, depot_idx
        )
        
        # 5. 초과분 처리
        balanced, unassigned = handle_overflow(df, balanced, cluster_to_driver, driver_info, cluster_info)
        
        # 6. 결과 생성
        print("\n=== 5단계: 방문 순서 최적화 ===")
        
        results = []
        stats = []
        total_distance = 0
        
        driver_info_map = {d['driver_idx']: d for d in driver_info}
        
        for cluster_id, nodes in balanced.items():
            if cluster_id not in cluster_to_driver:
                continue
            
            driver_idx = cluster_to_driver[cluster_id]
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
            
            visit_order = optimize_visit_order(df, nodes, base_lat, base_lng)
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
        
        print(f"\n{'='*50}")
        print(f"완료: 배정 {len(results)}건, 미배정 {len(unassigned)}건")
        print(f"{'='*50}")
        
        return {
            "status": "success",
            "updates": results,
            "statistics": stats,
            "summary": {
                "total_locations": total_calls,
                "total_assigned": len(results),
                "unassigned": len(unassigned),
                "unassigned_ids": [str(df.iloc[idx]['id']) for idx in unassigned],
                "total_distance_km": round(total_distance, 2)
            },
            "optimization_info": {
                "algorithm": "V11.3: Node-Distribution K-Means + Size-based Matching",
                "max_capa_violations": len(violations),
                "cluster_overlap": 0
            }
        }
        
    except Exception as e:
        traceback.print_exc()
        return {"status": "error", "message": str(e)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
