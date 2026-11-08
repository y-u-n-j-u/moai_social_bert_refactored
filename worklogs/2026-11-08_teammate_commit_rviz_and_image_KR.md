# 2026-11-08 팀원 커밋 받기, 이미지 drive-1028, RViz 정리 (건식만, 주행 없음)

코드는 `capstone_navi` 브랜치 `jackal-real-drive-fixes-20260912`의 `205059e`와 `94605c9`까지 푸시했다. 오프라인 테스트 248개 통과.
**오늘은 로봇을 움직이지 않았다**(건식, `cmd_vel_topic:=/spu_bert/cmd_vel_dryrun`, 로봇 `cmd_vel` 발행자 0). 새 코드는 실제 주행으로 확인하지 않았다.

## 1. 팀원 커밋 `d159252` ("Expose Jackal model path decisions and following evidence", `junwoo@naver.com`, 11-06)

내 11-02 커밋(`70519eb`) 위에 쌓여 있었고 충돌은 없었다. `capstone_navi`의 `git fetch`가 로컬의 깨진 참조 `refs/codex/turn-diffs/captures/1788572071669/.../base`
(0바이트 빈 파일) 때문에 "did not send all necessary objects"로 실패해서, 사용자의 승인을 받고 **그 참조 파일 하나만 삭제**한 뒤 빨리감기로 당겼다(나머지 `refs/codex` 9개는 그대로).

| 추가 | 내용 |
|---|---|
| 후보/선택 증거 | 브릿지가 `/spu_bert/selection_evidence`(후보별 좌표, 탈락 이유, 비용, 선택 이유)와 `/spu_bert/selection_markers`를 발행 |
| 선택 규칙 | 점수 = 모델 순서 + 이전 경로와의 거리 + 시작 방향 + 소프트 마진. `selection_heading_limit_rad 0.6`(약 34°)보다 급한 후보는 `eligible=false`(34° 안에 후보가 없으면 제한을 풀어 다시 고름) |
| `selection_view.py` | 후보 점수/이유를 표로 보여 주는 Tk 창 |
| `model_evidence_only:=true` | 모델 후보만 발행(대체/유지/복구/조향 보조 끔). 이 모드에서만 사람 거부/정지를 1.20/0.90으로 복원하고 사람 점유 지우기/스캔 마스크를 끔 |
| 데드맨 | `deadman.py`, 조이스틱 버튼을 누르는 동안만 RUN. 기본은 꺼짐(`deadman_button -1`) |
| 감시 스크립트 | RUN 서비스가 `success=True`여야 진행, 조이스틱 해제/비상정지면 감시 종료 |
| 분석 | `model_following_evidence.py`, `MODEL_FOLLOWING_EVIDENCE_KR.md` |

우리 사람 설정(거부/정지/감속 0, 제외 반경 0.8, 하드스톱 마스크)은 설정 파일이 안 바뀌어서 그대로다. 대신 RViz 기본값이 바뀌었다: 사람 궤적(최대 5초, 기본 꺼짐), 로봇 궤적(꺼짐), Nav2 경로(꺼짐).

## 2. 오늘 바꾼 것

| 항목 | 내용 |
|---|---|
| 이미지 | `drive-1027`(팀원 커밋 반영) → **`drive-1028`**(+오늘 변경). 런처 기본값 `drive-1028` |
| **카메라 "No Image"** | 원인: RViz가 `/camera/camera/color/image_raw`와 `/ped_yolo/instance_mask`를 **구독하지 않은 채** 떠 있었다(카메라 15Hz 정상, 레이아웃도 변경 없음). RViz만 다시 띄우자 구독하고 이미지가 나옴. 셰이더 오류(`indexed_8bit_image`)는 지난 며칠 로그에도 있어 원인 아님. 내부 원인은 모름(소프트웨어 렌더링 RViz의 기존 불안정). 수정: `rviz_retry.bash`가 카메라 발행 중인데 RViz가 구독 안 하면 자동 재시작 |
| 색 변경 (사용자 요청) | 선택됐지만 교체된 후보(노랑)와 실제 발행된 대체 경로(굵은 주황)를 **초록**으로, 방향 제외 후보(주황)를 **빨강**으로. 브릿지 코드와 `MODEL_FOLLOWING_EVIDENCE_KR.md` 수정 |
| RViz 끔 | 추종기 흰 글씨(`tracker_command_source`), 연한 회색 경로 |

현재 화면의 의미(코드 근거): 얇은 파랑=통과 후보, 얇은 빨강=탈락 또는 방향 제외, 굵은 초록 0.075=선택된 후보, 굵은 초록 0.085=발행된 비모델 경로(유지/대체/복구), 큰 노란 구=유도점(GP), 작은 초록/빨간 구=MGP 목표 샘플(안전/위험), 주황 화살표=최종 목표, 주황 작은 구=추종 목표점, 자홍 구(30cm)=사람.

## 3. 알아 둘 점

- **색을 합쳐서 구별이 안 된다**: 굵은 초록이 "모델이 고른 새 경로"인지 "유지/대체 경로"인지 색으로 알 수 없다(굵기 0.075 대 0.085는 화면에서 거의 안 보임). 빨강도 탈락과 방향 제외가 섞인다. 구별은 터미널 로그(`path_valid rank=` 대 `retained`/`global_fallback`)나 `selection_evidence`로 한다. **이 색 변경은 팀원이 만든 구분을 바꾼 것이니 팀원에게 알릴 것.**
- 사용자는 선택 후보를 보려고 **얇은 후보 선은 남기기로** 했다. 작은 구/큰 노란 구/연한 선 정리는 흰 글씨와 회색 선만 적용했고 구 두 종류는 아직 켜져 있다.
- `2D Pose Estimate` 대신 목표(`2D Goal Pose`)를 찍어야 한다. 브릿지를 다시 올리면 목표가 지워진다(건식 시험에서는 `/goal_pose`로 로봇 앞 8m를 직접 넣었다).
- 네트워크: 부팅 후 `192.168.131.50`이 없으면 컨테이너 시작 점검에서 멈춘다. `sudo ... configure_jackal_network.bash up`은 **일반 터미널**에서 해야 한다(이 세션의 `!` 입력은 비밀번호를 못 받음).

## 4. TODO (11-02 일지 6절 + 오늘)

1. **새 코드(후보/선택 증거, 색 변경, `rviz_retry` 검사)를 실제 주행으로 확인.** 건식 확인만 했다. `drive-1028`로 시작.
2. 정지 줄이기: 몸체 마진 0.10m, "모름" 셀 처리, 시작 방향 34° 제한이 정지의 원인인지 기록(`bag_104424`의 롤링 맵과 `selection_evidence`)으로 먼저 재계산. 11-02의 5.0m 지점 고정 물체 확인.
3. `odom_missing_or_stale`(RViz와 무관): 브릿지가 오도메트리 나이를 재는 방식 확인.
4. 11-02 4차 주행의 위치 추정 튐(map→odom 1.20m) 원인, 추적 안 된 사람의 정지, 모델이 사람을 우회하는지 평가(11-02 일지 6절 3~6번).
5. 팀원에게 색 변경(구분 없어짐) 알리고, 모델/비모델을 구별하는 표시(선 끝 글씨 등)가 필요한지 정하기.
6. 정리: 기록 폴더 정리(`pedtest_20261023_*`, `20261102_*`, `20261108_*`), 키보드 Ctrl/Alt 교환(`ctrl:swap_lalt_lctl`, 왼쪽 Ctrl+C/V가 안 됨) 결정, `jackal_humble_hmmdyn`(남의 것, 묻고 나서만).

## 5. 내일 시작 순서
`DRIVE_PROCEDURE_KR.md` 13절. 요약: (부팅 후) sudo 네트워크(일반 터미널) → `xhost +SI:localuser:root` → `./open_jackal_terminals.bash`(기본 `drive-1028`) → 2번 칸(센서+인식+Nav2, 약 60초) → 4번 RViz(위치 찍기, 방향까지 끌기) → 3번 브릿지 → 목표 → "run"이라고 할 때만 주행.
