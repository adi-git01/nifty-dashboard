"""
Excel Export Utility for Dashboard
"""

import pandas as pd
from io import BytesIO
import streamlit as st

def create_excel_download(df: pd.DataFrame, filename: str = "export.xlsx", sheet_name: str = "Data") -> bytes:
    """
    Create an Excel file from a DataFrame and return as bytes for download.
    
    Args:
        df: DataFrame to export
        filename: Name of the file (for reference)
        sheet_name: Name of the Excel sheet
    
    Returns:
        bytes: Excel file as bytes
    """
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name=sheet_name, index=False)
    return output.getvalue()


def create_multi_sheet_excel(sheets: dict) -> bytes:
    """
    Create an Excel file with multiple sheets.
    
    Args:
        sheets: dict of {sheet_name: DataFrame}
    
    Returns:
        bytes: Excel file as bytes
    """
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        for sheet_name, df in sheets.items():
            df.to_excel(writer, sheet_name=sheet_name[:31], index=False)  # Excel limit 31 chars
    return output.getvalue()


def add_excel_download_button(df: pd.DataFrame, filename: str, label: str = "📥 Export to Excel", key: str = None):
    """
    Add an Excel download button to Streamlit.
    
    Args:
        df: DataFrame to export
        filename: Name of the downloaded file
        label: Button label
        key: Unique key for the button
    """
    excel_data = create_excel_download(df, filename)
    st.download_button(
        label=label,
        data=excel_data,
        file_name=filename,
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        key=key
    )
