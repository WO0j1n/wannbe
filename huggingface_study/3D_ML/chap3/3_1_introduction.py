# 3D Pipeline 중 2번쨰 step인 ML - friendly3D 과정에 대해서 배워보도록 하겠습니다.
# 먼저, 다양한 기법들이 존재하게 됩니다.
 # triplanes - 3개의 평면이 교차하는 지점에서 3D 객체를 표현 - InstantMesh
 # voxel grids - 3D 격자 구조로 객체를 표현
 # nerts - 신경망을 통해 3D 객체를 표현 - NeRF
 # Splats - 3D 공간에 점들을 흩뿌려 객체를 표현, 실시간으로 렌더링을 통해서 확인이 가능 - LGM

# 이중에서도  Gaussian Splatting에 대해서 알아보도록 하겠습니다.
 # 3D 공간에 점들을 흩뿌려 객체를 표현하는 방법
 # 각 점들은 가우시안 분포를 가지며, 이 분포의 밀도에 따라 색상과 투명도가 결정됨
 # 이러한 점들을 조합하여 3D 객체를 형성하며, 실시간 렌더링이 가능하여 빠른 시각화가 가능
 # LGM(Learning Generative Models of 3D Objects with Gaussian Splatting) 논문에서 제안됨
    # 실제 3D Vision 분야에서 생태계 전체가 현재 Mesh로 표현을 하고 있기에 splat으로 대체될 가능성은 낮음
    # 다만, splat을 통해서 non-mesh representation에서 mesh로 변환하는 과정에서 활용하여 이후에 mesh로 변환하는 연구가 활발히 이루어질 가능성은 존재함

# 가우스 스플래팅의 겨웅, 미분이 가능한 기법이라는 점이 장점입니다.
# 이미지를 가져와서 2D에서 3d이미지로 변환하여 화면에 그리는 Triangle rasterization 방식을 사용하며 대부분의 Mesh로 표현할 때 해당 기법을 그대로 활용함 -> 그러나 해당 기법의 경우 미분이 불가능한 단점이 존재함.

# 그렇기 떄문에 Gaussian Splatting을 활용하고자 함. 해당 기법의 경우,
 # Location - 물테가 있는 위치(X, Y, Z)
 # Covariance - (3x3 matrix)
 # Color
 # alpah - 투명도
 # 위 4가지를 기준으로 미분이 가능한 장점이 존재함.

# 먼저, Split을 rasterization을 하기 위해서 이 2D로 Projection을 수행하게 되며 각 픽셀에 대해서 모든 점의 기여를 정렬하여 미분에 수행함.

