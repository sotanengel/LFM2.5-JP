# dpo-001 preference pair stats report (Issue #115)

- Total pairs: 970
- Total candidate prompts: 5461

## カテゴリ別ペア数

| category | pairs |
|---|---|
| char_count | 312 |
| bullet_count | 2 |
| format_json | 15 |
| paragraph_count | 1 |
| forbidden_word | 1 |
| polite_form | 393 |
| no_constraint | 246 |

## ペア不成立理由内訳

| reason | count |
|---|---|
| no_chosen_candidate | 3767 |
| no_rejected_candidate | 724 |

## chosen / rejected 長さ分布

- chosen: mean=192.2 median=205.5 min=23 max=671
- rejected: mean=259.8 median=263.0 min=41 max=781

## judge スコア分布

- chosen score mean: 4.18
- rejected score mean: 4.11

## 応答長ガード(ハードゲート)

- chosen: mean=192.2 median=205.5 min=23 max=671
- base_mean=176.0 tolerance=0.2 -> band=[140.8, 211.2] -> 判定: PASS

## 評価非重複アサーション(ハードゲート)

- 評価 char_count 値集合: [50, 60, 70, 80, 90, 100, 120, 150]
- ペア化プロンプト char_count/compound max 値集合: [40, 45, 55, 65, 75, 85, 95, 105, 110, 115, 130, 135, 140, 145, 160, 165]
- 値の重複: (なし)
- topic の評価プロンプトへの出現: (なし)

