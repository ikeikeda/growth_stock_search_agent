from __future__ import annotations

from pathlib import Path

import gspread
from google.oauth2.service_account import Credentials

from growth_stock_search_agent.config import PROJECT_ROOT, get_settings
from growth_stock_search_agent.models import ResearchReport, StockEvaluation

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]

HEADERS = [
    "実行日時",
    "順位",
    "銘柄名",
    "銘柄コード",
    "現在株価",
    "時価総額",
    "予想PER",
    "売上高成長率",
    "営業利益成長率",
    "割安理由",
    "未注目理由",
    "成長材料",
    "リスク",
    "Top3フラグ",
    "Top3比較",
    "評価スコア",
    "目的適合",
    "評価コメント",
]


def _get_worksheet():
    settings = get_settings()
    credentials_path = PROJECT_ROOT / settings.google_service_account_json
    credentials = Credentials.from_service_account_file(
        str(credentials_path),
        scopes=SCOPES,
    )
    client = gspread.authorize(credentials)
    spreadsheet = client.open_by_key(settings.google_sheets_id)
    try:
        return spreadsheet.worksheet(settings.google_sheets_worksheet)
    except gspread.WorksheetNotFound:
        worksheet = spreadsheet.add_worksheet(
            title=settings.google_sheets_worksheet,
            rows=1000,
            cols=len(HEADERS),
        )
        worksheet.append_row(HEADERS)
        return worksheet


def _ensure_headers(worksheet) -> None:
    first_row = worksheet.row_values(1)
    if not first_row:
        worksheet.append_row(HEADERS)


def get_existing_codes() -> set[str]:
    worksheet = _get_worksheet()
    _ensure_headers(worksheet)
    codes = worksheet.col_values(4)
    if len(codes) <= 1:
        return set()
    return {code.strip() for code in codes[1:] if code.strip()}


def _evaluation_map(report: ResearchReport) -> dict[str, StockEvaluation]:
    return {item.code: item for item in report.evaluation.stock_evaluations}


def append_new_candidates(report: ResearchReport) -> list[str]:
    """Append pass-rated candidates not already in the sheet. Returns appended codes."""
    worksheet = _get_worksheet()
    _ensure_headers(worksheet)
    existing_codes = get_existing_codes()
    eval_by_code = _evaluation_map(report)

    appended: list[str] = []
    rows: list[list[str]] = []

    for candidate in report.candidates:
        code = candidate.code.strip()
        if not code or code in existing_codes:
            continue

        evaluation = eval_by_code.get(code)
        if evaluation is None or not evaluation.passes_criteria:
            continue

        top3_text = report.top3_comparison if candidate.is_top3 else ""
        issues_text = "; ".join(evaluation.issues) if evaluation.issues else ""

        rows.append(
            [
                report.run_date,
                str(candidate.rank),
                candidate.name,
                code,
                candidate.current_price,
                candidate.market_cap,
                candidate.forecast_per,
                candidate.revenue_growth,
                candidate.operating_profit_growth,
                candidate.undervalued_reason,
                candidate.unnoticed_reason,
                candidate.growth_drivers,
                candidate.risks,
                "Yes" if candidate.is_top3 else "No",
                top3_text,
                f"{evaluation.overall_score:.2f}",
                "Pass",
                issues_text,
            ]
        )
        appended.append(code)

    if rows:
        worksheet.append_rows(rows, value_input_option="USER_ENTERED")

    return appended
