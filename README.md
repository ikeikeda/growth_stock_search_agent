# growth_stock_search_agent

日本株の「市場でまだ注目されていない割安成長株」を発掘する CrewAI エージェント。

## 機能

- **CrewAI** 4エージェント（Researcher / Analyst / Ranker / Evaluator）
- **Ollama** ローカル LLM（デフォルト: `gemma4:12b`）
- **Tavily** Web 検索・一次情報抽出
- **Google Spreadsheet** 新規銘柄のみ行追加
- **DSPy** プロンプト最適化（文字数削減 + 品質維持）
- **フィードバックループ** 合格銘柄 0 件のとき失敗原因を記録し、次回リサーチへ教訓を自動注入

## 前提条件

- Python 3.11+
- [uv](https://docs.astral.sh/uv/)
- [Ollama](https://ollama.ai/)（ローカル LLM サーバー）
- Tavily API キー
- Google Cloud サービスアカウント + Spreadsheet

## セットアップ

```bash
# 依存関係インストール
uv sync

# 環境変数
cp .env.example .env
# .env を編集（TAVILY_API_KEY, GOOGLE_SHEETS_ID 等）

# Ollama モデル取得（モデル名は ollama list で確認）
ollama pull gemma4:12b

# サービスアカウント JSON を配置
# credentials/service_account.json

# Spreadsheet をサービスアカウントのメールアドレスと共有（編集権限）
```

## 使い方

```bash
# 前提条件チェック
uv run check

# Spreadsheet 書き込みだけを試す（リサーチなし・サンプルデータ）
uv run test-sheets
uv run test-sheets --preview   # 書き込まず内容確認のみ

# リサーチ実行（Spreadsheet 追記）
uv run research

# ドライラン（Sheets 書き込みなし）
uv run research --dry-run

# 品質閾値未満でも強制書き込み
uv run research --force-write

# 最適化前プロンプトで実行
uv run research --use-base

# 失敗時の教訓更新・注入をスキップ
uv run research --no-feedback

# 蓄積した失敗件数と現行の教訓を表示
uv run inspect-feedback

# ユニットテスト
uv run pytest

# DSPy プロンプト最適化（週1回程度推奨）
uv run optimize-prompt
```

## 定期実行（Windows タスクスケジューラ）

1. 「タスクの作成」→ トリガー: 日次または週次
2. 操作: プログラム `scripts\run_scheduled.bat`
3. 開始: プロジェクトルート

## プロジェクト構成

```
src/growth_stock_search_agent/
├── main.py              # CLI エントリポイント
├── config.py            # 設定・ヘルスチェック
├── models.py            # Pydantic スキーマ
├── crew/                # CrewAI エージェント
├── feedback/            # 未検出時の次回向け教訓
├── output/              # Google Sheets 書き込み
├── prompts/             # プロンプト管理
└── dspy_opt/            # DSPy 最適化
```

## 環境変数

| 変数 | 説明 |
|------|------|
| `OLLAMA_BASE_URL` | Ollama API URL（デフォルト: `http://localhost:11434`） |
| `OLLAMA_MODEL` | モデル名（デフォルト: `gemma4:12b`） |
| `OLLAMA_MAX_TOKENS` | 生成予算（デフォルト: `16384`）。thinking モデルは足りないと空応答になる |
| `OLLAMA_DISABLE_THINKING` | thinking 無効化（デフォルト: `false`）。gemma4 では `false` 推奨 |
| `TAVILY_API_KEY` | Tavily API キー |
| `GOOGLE_SHEETS_ID` | スプレッドシート ID |
| `GOOGLE_SERVICE_ACCOUNT_JSON` | サービスアカウント JSON パス |
| `GOOGLE_SHEETS_WORKSHEET` | ワークシート名 |
| `EVAL_QUALITY_THRESHOLD` | 書き込み品質閾値（デフォルト: 0.6） |
| `FEEDBACK_ENABLED` | 未検出時の教訓更新と次回注入（デフォルト: `true`） |
| `FEEDBACK_UNKNOWN_CODE_THRESHOLD` | 未確認コードを禁止する連続失敗回数（デフォルト: 2） |
| `FEEDBACK_LESSONS_MAX_CHARS` | プロンプトへ注入する教訓の上限文字数（デフォルト: 1500） |

## 出力

- **Spreadsheet**: Pass 判定かつ未登録の銘柄のみ新規行追加
- **logs/evaluation_*.json**: 評価結果・不合格銘柄の監査ログ
- **logs/feedback/failures.jsonl**: 合格 0 件実行の分類ログ
- **logs/feedback/lessons.json**: 次回リサーチへ注入する教訓
