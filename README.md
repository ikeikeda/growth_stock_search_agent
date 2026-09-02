# growth_stock_search_agent

日本株の「市場でまだ注目されていない割安成長株」を発掘する CrewAI エージェント。

## 機能

- **CrewAI** 4エージェント（Researcher / Analyst / Ranker / Evaluator）
- **Ollama** ローカル LLM（デフォルト: `gemma4:12b`）
- **Tavily** Web 検索・一次情報抽出
- **Google Spreadsheet** 新規銘柄のみ行追加
- **DSPy** プロンプト最適化（文字数削減 + 品質維持）

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
├── output/              # Google Sheets 書き込み
├── prompts/             # プロンプト管理
└── dspy_opt/            # DSPy 最適化
```

## 環境変数

| 変数 | 説明 |
|------|------|
| `OLLAMA_BASE_URL` | Ollama API URL（デフォルト: `http://localhost:11434`） |
| `OLLAMA_MODEL` | モデル名（デフォルト: `gemma4:12b`） |
| `TAVILY_API_KEY` | Tavily API キー |
| `GOOGLE_SHEETS_ID` | スプレッドシート ID |
| `GOOGLE_SERVICE_ACCOUNT_JSON` | サービスアカウント JSON パス |
| `GOOGLE_SHEETS_WORKSHEET` | ワークシート名 |
| `EVAL_QUALITY_THRESHOLD` | 書き込み品質閾値（デフォルト: 0.6） |

## 出力

- **Spreadsheet**: Pass 判定かつ未登録の銘柄のみ新規行追加
- **logs/evaluation_*.json**: 評価結果・不合格銘柄の監査ログ
