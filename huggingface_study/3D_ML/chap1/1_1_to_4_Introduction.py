# Mesh - 3D 객체를 정의하는 모든 정점, 모서리 면의 집합 -> 이를 컴퓨터가 이해하는 데 많은 어려움이 존재함
# 3D pipeline는 현재, multi-view diffusion -> 머신러닝을 통해서 non-mesh representation로 변환한 뒤 한 차례를 더 걸쳐서 3D 객체인 Mesh를 생성

# Non-mesh representation -> ML-friendly 3D representation
 # triplanes - 3개의 평면이 교차하는 지점에서 3D 객체를 표현 - InstantMesh
 # voxel grids - 3D 격자 구조로 객체를 표현
 # nerts - 신경망을 통해 3D 객체를 표현 - NeRF
 # Splats - 3D 공간에 점들을 흩뿌려 객체를 표현, 실시간으로 렌더링을 통해서 확인이 가능 - LGM

# 3D Vision Tasks에서 크게 두가지 형태로 접근이 이루어짐.
    # 머신러닝 파이프라인은 메시가 아닌 3D 표현을 사용하여 3D 객체를 생성 -> 연구가 활발하게 이루어졌으며 현재는 변화가 거의 없음
    # 생성된 3D 객체를 메시로 변환하여 최종 출력 -> 해당 분야의 연구 성과가 미비하며 다양한 접근으로 연구가 활발히 이루어지고 있음


# Gaussian Splatting
 # 3D 공간에 점들을 흩뿌려 객체를 표현하는 방법
 # 각 점들은 가우시안 분포를 가지며, 이 분포의 밀도에 따라 색상과 투명도가 결정됨
 # 이러한 점들을 조합하여 3D 객체를 형성하며, 실시간 렌더링이 가능하여 빠른 시각화가 가능
 # LGM(Learning Generative Models of 3D Objects with Gaussian Splatting) 논문에서 제안됨
    # 실제 3D Vision 분야에서 생태계 전체가 현재 Mesh로 표현을 하고 있기에 splat으로 대체될 가능성은 낮음
    # 다만, splat을 통해서 non-mesh representation에서 mesh로 변환하는 과정에서 활용하여 이후에 mesh로 변환하는 연구가 활발히 이루어질 가능성은 존재함


# Non-mesh representation to Mesh를 하는 과정에서 wnfh Marching Cubes 알고리즘을 활용하여 3D 객체를 메시로 변환
    # 첫 번째 스텝에서 View를 여러 관점에서 볼 수 있도록 다중 시점 확산을 수행하게 됩니다. (Stable Diffusion, Diffusion 등)