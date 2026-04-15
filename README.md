# Intro. 들어가며.
본 프로젝트는 AI 모델 학습에 적합한 데이터를 만들기 위해 Hugging Face 데이터셋을 활용하여 이미지 전처리(Preprocessing), 데이터 증강(Augmentation), 그리고 이상치 필터링(Outlier Filtering) 파이프라인을 구축한 결과물입니다.

# Body. 전처리과정.

1. 이상치 탐지 및 필터링 (Outlier Detection)
학습에 방해가 될 수 있는 "질 낮은 데이터"를 자동으로 걸러냅니다.

평균 밝기 필터링: np.mean()을 활용해 너무 어두운(Dark) 이미지를 제거합니다.

객체 크기 필터링: cv2.threshold와 findContours를 사용하여 이미지 내 주요 물체의 면적을 계산합니다. 전체 면적의 5% 미만인 이미지는 학습 가치가 낮다고 판단해 제외합니다.

2. 표준 전처리 (Standard Preprocessing)
이미지의 규격을 맞추고 노이즈를 제어합니다.

Resize: 모든 이미지를 동일한 크기인 224x224로 고정합니다.

Gaussian Blur: 픽셀 단위의 자잘한 노이즈를 제거하여 모델이 특징을 더 잘 잡게 합니다.

Grayscale: 색상 정보가 중요하지 않은 학습 환경을 가정하여 흑백으로 변환합니다.

Normalization: 픽셀 값을 0~255 사이로 정규화하여 데이터 분포를 일정하게 만듭니다.

3. 데이터 증강 (Data Augmentation)
데이터 부족 문제를 해결하고 모델의 일반화 성능을 높입니다.

Flip: 좌우 반전을 통해 방향에 대한 저항성을 키웁니다.

Rotation: 중심축 기준 15도 회전을 적용합니다.

Color Variation: 밝기와 대비를 조절하여 다양한 조명 환경에 대비합니다.

# Information about files.

image_preprocessing.py: 전처리 전체 로직이 담긴 파이썬 스크립트

preprocessed_samples/: 전처리가 완료된 이미지 5장 (merge, stack)

이미지는 [ 좌우반전 | 회전 | 색상변화] 순으로 가로로 병합되어 저장됩니다.
