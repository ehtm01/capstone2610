"""팀 노트북(최최종코드.ipynb) → 센서 입력 + 공정 파생 + As-Is/To-Be 성능 그래프 노트북 생성.
원본은 수정하지 않는다. usage: python build_notebook.py <src.ipynb> <out.ipynb>"""
import json, re, sys, copy

src, out = sys.argv[1:3]
nb = json.load(open(src, encoding="utf-8"))
code = "".join(nb["cells"][0]["source"])


def sub(old, new):
    global code
    assert code.count(old) == 1, (old[:60], code.count(old))
    code = code.replace(old, new)


# 0) 로컬(Windows) 실행 시 한글 폰트
sub('plt.rcParams.update({', '''# [수정] 로컬 Windows 실행 시 한글 폰트
if not font_paths and sys.platform.startswith("win"):
    plt.rcParams["font.family"] = "Malgun Gothic"

plt.rcParams.update({''')

# 1) 입력: 계획 7개 + 센서 12개
sub('''    "planned_lead_time_min"
]''', '''    "planned_lead_time_min",

    # [수정] 공정 센서·이벤트 12개 (Blind 입력에도 존재)
    "changeover_min",
    "station_a_cycle_sec",
    "station_b_cycle_sec",
    "station_c_cycle_sec",
    "station_d_cycle_sec",
    "wait_before_b_min",
    "wait_before_c_min",
    "wait_before_d_min",
    "wip_start_qty",
    "wip_end_qty",
    "equipment_stop_min",
    "material_wait_min"
]''')

# 2) 파생: 계획 파생 4개 + 공정 파생 5개
sub('''    "operator_min_per_qty"
]''', '''    "operator_min_per_qty",

    # [수정] 공정 파생 5개
    "max_cycle_sec",
    "bottleneck_proc_min",
    "total_wait_min",
    "event_total_min",
    "wip_delta"
]''')

sub('''    d[DERIVED] = d[DERIVED].replace(''', '''    # [수정] 공정 파생변수
    cycles = d[[
        "station_a_cycle_sec",
        "station_b_cycle_sec",
        "station_c_cycle_sec",
        "station_d_cycle_sec"
    ]]

    d["max_cycle_sec"] = cycles.max(axis=1)                  # 가장 느린 공정 사이클
    d["bottleneck_proc_min"] = d["max_cycle_sec"] * q / 60   # 병목 공정 처리시간(분)

    d["total_wait_min"] = d[[
        "wait_before_b_min",
        "wait_before_c_min",
        "wait_before_d_min"
    ]].sum(axis=1, min_count=1)                              # 공정 앞 대기 합계

    d["event_total_min"] = (
        d["changeover_min"]
        + d["equipment_stop_min"]
        + d["material_wait_min"]
    )                                                        # 교체·정지·자재 이벤트 합계

    d["wip_delta"] = d["wip_end_qty"] - d["wip_start_qty"]   # WIP 증감

    d[DERIVED] = d[DERIVED].replace(''')

# 2-1) 로컬(Jupyter/VS Code) 실행 지원: 데이터 폴더 자동 탐색, Colab일 때만 업로드
sub('''folder = Path(
    os.environ.get("TEAM3_DATA_DIR", "/content")
)

folder.mkdir(
    parents=True,
    exist_ok=True
)

sources = find_csvs(folder)

if len(sources) != 3:
''', '''# [수정] Colab이 아니면 로컬 폴더에서 CSV 3개를 찾는다
IN_COLAB = "google.colab" in sys.modules

candidates = [
    os.environ.get("TEAM3_DATA_DIR"),
    "/content",
    ".",
    "data",
    "review_exp/nb_runs/data",
    Path.home() / "Documents" / "카카오톡 받은 파일",
    Path.home() / "Downloads",
]

folder, sources = None, {}

for c in candidates:
    if c and Path(c).is_dir():
        found = find_csvs(Path(c))
        # 원본·Blind 입력·Blind 정답이 서로 다른 폴더에 있어도 합친다
        for k, v in found.items():
            sources.setdefault(k, v)
        if folder is None and found:
            folder = Path(c)
    if len(sources) == 3:
        break

if folder is None:
    folder = Path("/content") if IN_COLAB else Path(".")

print("CSV 탐색 결과:", {k: str(v) for k, v in sources.items()})

if len(sources) != 3 and IN_COLAB:
''')
sub('''    raise ValueError(
        f"CSV 3종이 필요합니다. "
        f"발견한 유형: {list(sources)}"
    )''', '''    raise ValueError(
        f"CSV 3종이 필요합니다. "
        f"발견한 유형: {list(sources)}. "
        f"로컬에서는 원본·Blind 입력·Blind 정답 CSV를 노트북과 같은 폴더(또는 data 폴더)에 두거나 "
        f"환경변수 TEAM3_DATA_DIR로 폴더를 지정하세요."
    )''')
# 2-2) Windows 로컬: sklearn 모델 HTML 표시가 cp949로 파일을 읽다 실패 → 텍스트 표시로 전환
sub('from sklearn.inspection import permutation_importance\n',
    'from sklearn.inspection import permutation_importance\n'
    'from sklearn import set_config\n'
    'set_config(display="text")  # [수정] Windows 로컬에서 모델 HTML 표시 인코딩 오류 방지\n')

# 3) STEP·그래프 헤더 기준으로 셀 분할 (최상위 주석 블록에서만)
lines = code.split("\n")
cuts = [0]
for i in range(1, len(lines) - 1):
    if re.fullmatch(r"# =+", lines[i]) and re.match(r"# (STEP|그래프|히트맵)", lines[i + 1]):
        cuts.append(i)
cuts.append(len(lines))
chunks = ["\n".join(lines[a:b]).strip("\n") for a, b in zip(cuts, cuts[1:])]
chunks = [c for c in chunks if c.strip()]
for c in chunks:
    compile(c, "<cell>", "exec")  # 각 셀이 문법상 완결인지 확인


def md(t):
    return {"cell_type": "markdown", "metadata": {}, "source": t}


def cc(t):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": t}


intro = md("""# TEAM 05 · 센서 입력 확장 모델 성능 평가

원본 `최최종코드.ipynb`의 STEP 0~5 코드를 그대로 쓰고 **입력 변수만 바꾼 버전**입니다. 알고리즘, 하이퍼파라미터, 시간순 분할, Top-k 선택, 최종 모델 선택 규칙은 원본과 같습니다. 바꾼 곳은 코드에 `# [수정]`으로 표시했습니다.

| 구분 | 원본 (As-Is) | 이 노트북 (To-Be) |
|---|---|---|
| 입력 변수 | 계획 7개 | 계획 7개 + **공정 센서·이벤트 12개** = 19개 |
| 파생 변수 | 계획 파생 4개 | 계획 파생 4개 + **공정 파생 5개** |
| 제외 변수 | — | `actual_qty`(Blind 입력에 없음), `actual_lead_time_min`·`schedule_delay_min`·`on_time_completion_flag`·`bottleneck_station`(정답·사후 정보) |

- **공정 센서 12개**: 공정 A~D 사이클, B·C·D 앞 대기, 시작·종료 WIP, 교체·설비 정지·자재 대기 시간
- **공정 파생 5개**: 최대 사이클, 병목 처리시간(최대 사이클 × 계획 수량), 대기 합계, 이벤트 합계(교체 + 정지 + 자재), WIP 증감
- **예측 시점**: 계획 시점이 아니라 **공정 진행 중(센서값 수집 후)** 재예측이다. 계획 시점 모델(As-Is)과의 비교는 마지막 **STEP 6**에 있다.

**실행 방법 (Colab)**: 실행하면 CSV 3개(원본 생산 데이터, Blind 입력, Blind 정답)를 업로드하라고 나온다. 로컬에서는 환경변수 `TEAM3_DATA_DIR`에 세 파일이 있는 폴더를 지정한다.""")

step6_md = md("""## STEP 6. As-Is(계획 7개) vs To-Be(센서 포함) 성능 평가

- **As-Is**: 원본 노트북의 최종 모델과 같은 구성. 계획 변수 Top5(`planned_lead_time_min`, `available_operators`, `line_id`, `product_family`, `shift`) + RandomForest
- **To-Be**: 이 노트북이 내부 검증으로 고른 최종 모델(`best_key`)
- 두 모델 모두 같은 방식으로 평가한다. 내부 검증은 학습 구간으로 학습해 검증 401건으로 평가하고, Blind는 원본 1,600건 전체로 다시 학습해 Blind 1,600건으로 평가한다.
- ⚠️ Blind 정답은 평가에만 쓰고 모델 선택이나 임계값 조정에는 쓰지 않는다.""")

step6 = cc('''# ============================================================
# STEP 6-1. As-Is / To-Be 모델 학습·평가
# ============================================================

from sklearn.metrics import confusion_matrix, precision_recall_curve

ASIS_COLS = [
    "planned_lead_time_min",
    "available_operators",
    "line_id",
    "product_family",
    "shift"
]
ASIS_ALG = "RandomForest"

# 원본 노트북 fit_eval과 같은 열 순서(EXT 순서)로 맞춘다
ASIS_COLS = [c for c in EXT if c in ASIS_COLS]

tobe_cols, tobe_alg, tobe_pca = configurations[best_key]


def fit_pair(cols, alg, pca=False):
    """내부 검증용(학습 구간)과 Blind용(원본 전체) 모델을 각각 학습"""
    m_valid = pipeline(cols, alg, pca).fit(train_ex[cols], y_train)
    m_blind = pipeline(cols, alg, pca).fit(full_ex[cols], original.y)
    return m_valid, m_blind


asis_valid, asis_blind = fit_pair(ASIS_COLS, ASIS_ALG)
tobe_valid, tobe_blind = fit_pair(tobe_cols, tobe_alg, tobe_pca)

rows = []
for name, mv, mb, cols in [
    ("As-Is 계획 Top5", asis_valid, asis_blind, ASIS_COLS),
    (f"To-Be {best_key}", tobe_valid, tobe_blind, tobe_cols)
]:
    rows.append({"모델": name, "평가": "내부 검증 401", **score(mv, valid_ex[cols], y_valid)})
    rows.append({"모델": name, "평가": "Blind 1600", **score(mb, blind_ex[cols], y_blind)})

asis_tobe = pd.DataFrame(rows)
display(asis_tobe.round(4))

asis_tobe.to_csv(
    OUT / "10_asis_tobe.csv",
    index=False,
    encoding="utf-8-sig"
)

pred_asis = asis_blind.predict(blind_ex[ASIS_COLS])
pred_tobe = tobe_blind.predict(blind_ex[tobe_cols])
p3_asis = class3_probability(asis_blind, blind_ex[ASIS_COLS])
p3_tobe = class3_probability(tobe_blind, blind_ex[tobe_cols])

for name, pred in [("As-Is", pred_asis), ("To-Be", pred_tobe)]:
    print(
        f"{name}: Failure 경보 {(pred == 3).sum()}건 / "
        f"실제 Failure {(y_blind == 3).sum()}건 / "
        f"놓침 {((y_blind == 3) & (pred != 3)).sum()}건"
    )''')

g1 = cc('''# ============================================================
# 그래프 30. As-Is vs To-Be 지표 비교 (내부 검증 / Blind)
# ============================================================

metric_names = {
    "macro_f1": "Macro F1",
    "failure_recall": "Failure Recall",
    "failure_precision": "Failure Precision",
    "failure_ap": "Failure AP",
    "qwk": "QWK"
}

fig, axes = plt.subplots(1, 2, figsize=(16, 5.5), sharey=True)
width = 0.38
x = np.arange(len(metric_names))

for ax, part in zip(axes, ["내부 검증 401", "Blind 1600"]):
    sub_df = asis_tobe[asis_tobe["평가"] == part].set_index("모델")
    for k, (model_name, color) in enumerate(zip(sub_df.index, ["#9aa5b1", "#2bb3a3"])):
        vals = sub_df.loc[model_name, list(metric_names)].astype(float).values
        bars = ax.bar(x + (k - 0.5) * width, vals, width, label=model_name, color=color)
        ax.bar_label(bars, fmt="%.2f", padding=2, fontsize=9)
    ax.set_xticks(x, list(metric_names.values()))
    ax.set_ylim(0, 1.08)
    ax.set_title(part)
    ax.grid(axis="y", alpha=0.3)

axes[0].legend(loc="upper left")
plt.suptitle("As-Is(계획 7개) vs To-Be(센서 포함) 성능 비교", fontsize=14)
show_fig("30_asis_tobe_metrics")''')

g2 = cc('''# ============================================================
# 그래프 31. Blind 혼동행렬 비교 (행 = 실제, 열 = 예측)
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

for ax, (title, pred) in zip(axes, [
    ("As-Is 계획 Top5", pred_asis),
    (f"To-Be {best_key}", pred_tobe)
]):
    cm = confusion_matrix(y_blind, pred, labels=range(4))
    ax.imshow(cm, cmap="Blues")
    for i in range(4):
        for j in range(4):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() * 0.6 else "black", fontsize=11)
    ax.set_xticks(range(4), NAMES, rotation=20)
    ax.set_yticks(range(4), NAMES)
    ax.set_xlabel("예측")
    ax.set_ylabel("실제")
    ax.set_title(f"{title}\\nMacro F1 {f1_score(y_blind, pred, average='macro'):.3f}")

show_fig("31_blind_confusion_compare")''')

g3 = cc('''# ============================================================
# 그래프 32. Blind Failure Precision-Recall 곡선 비교
# ============================================================

fig, ax = plt.subplots(figsize=(7.5, 6))

for name, p3, color in [
    ("As-Is 계획 Top5", p3_asis, "#9aa5b1"),
    (f"To-Be {best_key}", p3_tobe, "#2bb3a3")
]:
    prec, rec, _ = precision_recall_curve(y_blind == 3, p3)
    ap = average_precision_score(y_blind == 3, p3)
    ax.plot(rec, prec, color=color, lw=2.2, label=f"{name} (AP {ap:.3f})")

base_rate = (y_blind == 3).mean()
ax.axhline(base_rate, ls="--", color="gray", label=f"무작위 기준 {base_rate:.3f}")
ax.set_xlabel("Failure Recall")
ax.set_ylabel("Failure Precision")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1.02)
ax.legend(loc="upper right")
ax.set_title("Blind Failure 탐지 PR 곡선")
ax.grid(alpha=0.3)

show_fig("32_blind_pr_compare")''')

g4 = cc('''# ============================================================
# 그래프 33. 단계별(1→2→3) 내부 검증 Macro F1 추이
# ============================================================

fig, ax = plt.subplots(figsize=(9, 5.5))

for k, alg in enumerate(stage_f1.columns):
    vals = stage_f1[alg].values
    line, = ax.plot(range(len(vals)), vals, marker="o", lw=2.2, label=alg)
    for i, v in enumerate(vals):
        # 알고리즘마다 라벨을 좌우로 엇갈려 겹침 방지
        ax.annotate(f"{v:.3f}", (i, v), textcoords="offset points",
                    xytext=(-42 + 28 * k, 8), ha="center", fontsize=8,
                    color=line.get_color())

ax.set_xticks(range(len(stage_f1.index)), stage_f1.index)
ax.set_ylabel("내부 검증 Macro F1")
ax.set_title("단계별 성능 변화 (센서 포함 입력)")
ax.grid(alpha=0.3)
ax.legend()

show_fig("33_stage_f1_trend")''')

g5 = cc('''# ============================================================
# 그래프 34. To-Be 최종 모델 Permutation Importance (내부 검증)
# ============================================================

pi = permutation_importance(
    tobe_valid,
    valid_ex[tobe_cols],
    y_valid,
    scoring="f1_macro",
    n_repeats=10,
    random_state=SEED
)

tobe_imp = pd.Series(pi.importances_mean, index=tobe_cols).sort_values()

plt.figure(figsize=(9, 0.5 * len(tobe_imp) + 1.5))
plt.barh(tobe_imp.index, tobe_imp.values,
         color=["#2bb3a3" if c in DERIVED else "#5b7a99" for c in tobe_imp.index])
plt.xlabel("Permutation Importance (Macro F1 감소)")
plt.title(f"To-Be 최종 모델 변수 중요도 · {best_key}\\n(청록 = 파생변수)")
show_fig("34_tobe_importance")''')

cells = [intro] + [cc(c) for c in chunks] + [step6_md, step6, g1, g2, g3, g4, g5]
new = copy.deepcopy(nb)
new["cells"] = cells
new["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
json.dump(new, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("cells:", len(cells), "code chunks:", len(chunks))
