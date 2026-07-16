# dpo-001 preference prompt pool stats report (Issue #115)

- Seed: 42
- Total pool prompts: 5461
- Source A (distill CSV, eval-collision drops): 3988 kept, 12 dropped
- Source B (coverage-gap, programmatic): 853
- Source C (joryu bank, open-ended): 620 (drops: {'no_category': 100, 'eval_topic_keyword': 56, 'not_sampled': 3325})
- Output: data/processed/dpo/pref_prompts.jsonl

## カテゴリ別件数

| category | count |
|---|---|
| char_count | 1584 |
| bullet_count | 606 |
| format_json | 466 |
| keyword_include | 93 |
| paragraph_count | 93 |
| forbidden_word | 93 |
| start_word | 93 |
| compound | 80 |
| polite_form | 1133 |
| no_constraint | 1220 |

## 評価非重複アサーション(ハードゲート)

- 評価 char_count 値集合: [50, 60, 70, 80, 90, 100, 120, 150]
- プール char_count/compound min/max 値集合(一部): [30, 40, 45, 55, 65, 75, 85, 95, 105, 110, 115, 125, 130, 135, 140, 145, 155, 160, 165, 170, 175, 180, 185, 190, 195, 200, 205, 210, 215, 220, 230, 240, 250, 260, 270, 280, 290, 300, 310, 320, 330, 340]
- 値の重複: (なし)
- topic の評価プロンプトへの出現: (なし)

