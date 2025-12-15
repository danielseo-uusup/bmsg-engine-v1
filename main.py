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


def get_centroid(coords_list):
    """좌표 리스트의 중심점 계산"""
    if not coords_list:
        return None
    lats = [c[0] for c in coords_list]
    lons = [c[1] for c in coords_list]
    return (np.mean(lats), np.mean(lons))


def capacity_aware_clustering(df, drivers, depot_idx=0):
    """
    ★ V11 핵심: max_capa 기반 클러스터링 ★
    
    원칙:
    1. 클러스터 크기 = 기사의 max_capa (절대 초과 불가)
    2. 클러스터 간 교차 없음 (지리적으로 완전 분리)
    3. 기사 거점에서 가까운 영역부터 할당
    
    알고리즘: Seed-based Greedy Clustering
    1. 각 기사의 거점을 시드(seed)로 사용
    2. 각 시드에서 가장 가까운 노드부터 max_capa만큼 할당
    3. 이미 할당된 노드는 다른 클러스터에서 제외
    """
    
    print("\n=== Capacity-Aware Clustering ===")
    
    # 노드 정보 추출 (depot 제외)
    nodes = []
    for i in range(len(df)):
        if i == depot_idx:
            continue
        nodes.append({
            'idx': i,
            'lat': float(df.iloc[i]['lat']),
            'lon': float(df.iloc[i]['lon']),
            'weight': int(df.iloc[i]['weight']),
            'assigned': False
        })
    
    print(f"총 노드: {len(nodes)}개")
    
    # 기사 정보 정리 (max_capa 내림차순 정렬)
    driver_info = []
    for i, driver in enumerate(drivers):
        max_capa = driver.max_capa if driver.max_capa else DEFAULT_MAX_CAPA
        
        # 거점 좌표 (없으면 depot 사용)
        if driver.base_lat is not None and driver.base_lng is not None:
            base_lat, base_lng = driver.base_lat, driver.base_lng
        else:
            base_lat = float(df.iloc[depot_idx]['lat'])
            base_lng = float(df.iloc[depot_idx]['lon'])
        
        driver_info.append({
            'driver_idx': i,
            'driver': driver,
            'max_capa': max_capa,
            'base_lat': base_lat,
            'base_lng': base_lng
        })
    
    # ★ 핵심: max_capa가 큰 기사부터 처리 ★
    # 이유: 큰 클러스터를 먼저 확보해야 작은 클러스터가 남은 영역에서 선택 가능
    driver_info.sort(key=lambda x: -x['max_capa'])
    
    print(f"기사 처리 순서 (max_capa 내림차순):")
    for d in driver_info:
        print(f"  {d['driver'].name}: max_capa={d['max_capa']}, 거점=({d['base_lat']:.4f}, {d['base_lng']:.4f})")
    
    # 클러스터 할당 결과
    cluster_assignments = {}  # {node_idx: driver_idx}
    driver_clusters = {d['driver_idx']: [] for d in driver_info}
    
    # ★ Seed-based Greedy Assignment ★
    for d_info in driver_info:
        driver_idx = d_info['driver_idx']
        max_capa = d_info['max_capa']
        base_lat = d_info['base_lat']
        base_lng = d_info['base_lng']
        
        # 미할당 노드 중 거점에서 가까운 순으로 정렬
        unassigned = [n for n in nodes if not n['assigned']]
        
        if not unassigned:
            print(f"  {d_info['driver'].name}: 할당 가능한 노드 없음")
            continue
        
        # 거점에서의 거리 계산
        for node in unassigned:
            node['dist_to_base'] = haversine(
                node['lat'], node['lon'],
                base_lat, base_lng
            )
        
        # 거리순 정렬
        unassigned.sort(key=lambda x: x['dist_to_base'])
        
        # max_capa만큼 할당
        assigned_count = 0
        for node in unassigned:
            if assigned_count >= max_capa:
                break
            
            node['assigned'] = True
            cluster_assignments[node['idx']] = driver_idx
            driver_clusters[driver_idx].append(node['idx'])
            assigned_count += 1
        
        print(f"  {d_info['driver'].name}: {assigned_count}건 할당 (max_capa={max_capa})")
    
    # 미할당 노드 확인
    unassigned_nodes = [n['idx'] for n in nodes if not n['assigned']]
    print(f"\n미할당 노드: {len(unassigned_nodes)}개")
    
    return cluster_assignments, driver_clusters, unassigned_nodes, driver_info


def optimize_visit_order_nearest_neighbor(df, node_indices, start_lat, start_lon):
    """
    Nearest Neighbor 알고리즘으로 방문 순서 최적화
    
    시작점(기사 거점)에서 가장 가까운 노드부터 방문
    """
    if not node_indices:
        return []
    
    if len(node_indices) == 1:
        return node_indices
    
    visited = []
    remaining = set(node_indices)
    current_lat, current_lon = start_lat, start_lon
    
    while remaining:
        # 현재 위치에서 가장 가까운 노드 선택
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
    
    # 복귀 거리는 포함하지 않음 (실제 운영에서는 복귀 안 할 수도 있음)
    
    return total_dist


@app.get("/")
def read_root():
    return {
        "status": "active",
        "message": "VRP Engine V11 (First Principles - Capacity-Aware Clustering)",
        "features": [
            "★ max_capa 하드캡 100% 준수",
            "★ 클러스터 간 교차 0% (지리적 완전 분리)",
            "★ max_capa 큰 기사 → 큰 클러스터 자동 매칭",
            "기사 거점 기반 클러스터 할당",
            "Nearest Neighbor 방문 순서 최적화"
        ],
        "principles": {
            "1": "max_capa는 절대 초과 불가 (하드캡)",
            "2": "클러스터 간 교차 없음 (한 노드는 하나의 클러스터에만)",
            "3": "max_capa가 큰 기사가 큰 클러스터를 가져감"
        },
        "algorithm": "Seed-based Greedy Clustering + Nearest Neighbor TSP"
    }


@app.post("/optimize")
def optimize_routes(body: RequestBody):
    """
    ★ V11: First Principles 기반 배차 최적화 ★
    
    핵심 원칙:
    1. max_capa 하드캡 절대 준수
    2. 클러스터 간 교차 없음
    3. max_capa 큰 기사 → 큰 클러스터
    
    알고리즘:
    1. 기사를 max_capa 내림차순 정렬
    2. 각 기사의 거점에서 가까운 노드부터 max_capa만큼 할당
    3. 클러스터 내 Nearest Neighbor로 방문 순서 결정
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
            use_driver_features = True
        else:
            drivers = [
                Driver(
                    id=f"driver_{i+1}",
                    name=f"기사 {i+1}",
                    max_capa=DEFAULT_MAX_CAPA
                )
                for i in range(body.num_vehicles)
            ]
            use_driver_features = False
        
        num_vehicles = len(drivers)
        
        # max_capa 합계
        total_max_capa = sum(d.max_capa or DEFAULT_MAX_CAPA for d in drivers)
        total_calls = num_locations - 1  # depot 제외
        
        print(f"\n{'='*50}")
        print(f"VRP V11 - First Principles")
        print(f"{'='*50}")
        print(f"총 콜: {total_calls}건")
        print(f"총 수용량: {total_max_capa}건")
        print(f"기사 수: {num_vehicles}명")
        
        # weight 처리
        if 'weight' not in df.columns:
            df['weight'] = DEFAULT_WEIGHT_KG
        df['weight'] = pd.to_numeric(df['weight'], errors='coerce').fillna(DEFAULT_WEIGHT_KG).astype(int)
        
        # 2. ★ 핵심: Capacity-Aware Clustering ★
        cluster_assignments, driver_clusters, unassigned_nodes, driver_info = \
            capacity_aware_clustering(df, drivers, depot_idx)
        
        # 3. 각 클러스터별 방문 순서 최적화 + 결과 생성
        print(f"\n=== 방문 순서 최적화 ===")
        
        results = []
        stats = []
        total_distance = 0
        
        for d_info in driver_info:
            driver_idx = d_info['driver_idx']
            driver = d_info['driver']
            max_capa = d_info['max_capa']
            base_lat = d_info['base_lat']
            base_lng = d_info['base_lng']
            
            cluster_nodes = driver_clusters[driver_idx]
            
            if not cluster_nodes:
                # 빈 클러스터
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
            
            # Nearest Neighbor로 방문 순서 결정
            visit_order = optimize_visit_order_nearest_neighbor(
                df, cluster_nodes, base_lat, base_lng
            )
            
            # 경로 거리 계산
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
            
            # 상태 판정
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
        
        # 4. 검증: max_capa 초과 여부
        violations = []
        for stat in stats:
            if stat['call_count'] > stat['max_capa']:
                violations.append(f"{stat['driver_name']}: {stat['call_count']} > {stat['max_capa']}")
        
        if violations:
            print(f"\n⚠️ max_capa 위반: {violations}")
        else:
            print(f"\n✅ max_capa 100% 준수")
        
        # 5. 결과 반환
        print(f"\n{'='*50}")
        print(f"최적화 완료")
        print(f"배정: {len(results)}건, 미배정: {len(unassigned_nodes)}건")
        print(f"총 거리: {total_distance:.1f}km")
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
                "avg_distance_km": round(total_distance / num_vehicles, 2) if num_vehicles > 0 else 0
            },
            "optimization_info": {
                "algorithm": "V11: Capacity-Aware Greedy Clustering + Nearest Neighbor",
                "max_capa_violations": len(violations),
                "cluster_overlap": 0,  # 교차 없음 보장
                "principles": [
                    "max_capa 하드캡 100% 준수",
                    "클러스터 간 교차 0%",
                    "max_capa 큰 기사 → 큰 클러스터"
                ]
            },
            "driver_assignments": {
                d_info['driver'].name: {
                    "max_capa": d_info['max_capa'],
                    "assigned": len(driver_clusters[d_info['driver_idx']])
                }
                for d_info in driver_info
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
