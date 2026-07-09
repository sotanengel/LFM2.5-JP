# Phase 1 パイロットレポート (Issue #20 / #23 ゲート実走)

実施日: 2026-07-09 / 環境: WSL2 Ubuntu + RTX 3060 Ti(torch 2.13.0+cu130、flash SDP)

## 1. データ準備(end-to-end、`configs/data/corpus_pilot.yaml`)

`python -m lfm25_ja.data.prepare --config configs/data/corpus_pilot.yaml --sample-limit 20000`

| コーパス | 入力 | 出力 | 主な除去 |
|---|---|---|---|
| aozora(青空文庫) | 16,951 | 15,976 | 長さ 5.6% / 重複 0.2% |
| wikitext_en | 20,000 | 8,748 | 長さ 56.2%(短行)/ 重複 0.1% |

混合(seed 42・documents 単位): **18,783 docs、ja 85.00% / en 15.00%**(目標比率と完全一致)。
download → clean(NFKC/言語判定/MinHash/長さ)→ mix → レポート出力まで 1 コマンドで完走。

## 2. 350M パイロット CPT(`configs/cpt/cpt_350m_pilot.yaml`)

- モデル: LiquidAI/LFM2-350M(bf16、層 FT: index 15 のみ可変 = 5.18%)
- データ: 上記 mixture の packed seq 1024 × sample_fraction 0.01(~86 万トークン)、1 epoch
- batch 1 × grad_accum 4、lr 1e-4、paged_adamw_8bit、grad ckpt

**結果**:

| 項目 | 値 |
|---|---|
| loss | **5.72 → 4.88**(単調改善、発散・スパイクなし) |
| perplexity | ~305 → ~132 |
| VRAM ピーク | 693 MiB allocated / 1.6 GiB reserved |
| 所要 | 学習 108 s(2.0 optimizer steps/s、8.0 samples/s) |

## 3. 生成サンプル(temperature 0.8、抜粋)

- 「吾輩は猫である。名前は」→ 「…小助の妖風をやらぬ二匹の猫の一組に、私達は彼らの家へ来たのだ。『ああ、彼らは可愛い。…』」
- 「明治時代の東京では、」→ 「…幕府の法制度を維持して続けようとする努力が、この近代化の舞台作りとして重要な役割を果たしました。…」
- "The history of science shows that" → 英語でも整合的な継続(忘却の兆候なし)

**文字化け・トークン崩れ・反復ループなし**。日本語は青空文庫調の文体を獲得しており CPT が効いている。

## Phase 1 ゲート判定: **通過**

- [x] テスト全通過(CPU 85 件、CI 緑)
- [x] パイロットで loss / perplexity が単調改善
- [x] 生成テキストに文字化け・テンプレート崩れなし

## 備考

- パイロットコーパスは軽量代替(青空文庫 + wikitext)。本番 CPT(Phase 2)では corpus.yaml の
  wikipedia_ja / cc100_ja を使うが、`prepare.py` が非ストリーミングのためフルダウンロード
  (数十 GB)が発生する。Phase 2 開始時に streaming 対応または subset 指定を検討すること
- 生成品質は 350M + ~86 万トークンとしては期待どおり(意味の通る文だが内容の飛躍はある)
