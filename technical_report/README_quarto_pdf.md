# Technical Report — Quarto PDF Generation

This document provides instructions for generating the Customer Churn Analysis Report as a PDF using Quarto.

---

## 1. Cover Page Setup

The cover page is implemented as a separate content file:

technical_report/content/cover.qmd

### Requirements:
- The cover page should contain:
  - Report title
  - Subtitle
  - Student information table
  - Declaration of originality
- The cover page should end with:

\newpage

This ensures the rest of the report begins on a new page.

### Inclusion in Report

The cover page must be included at the top of index.qmd using the include directive.

---

## 2. Rendering the PDF

### Step 1: Ensure Quarto is Installed

Verify installation:

quarto --version

---

### Step 2: Install TinyTeX (One-Time Setup)

quarto install tinytex

This provides the LaTeX engine required for PDF generation.

---

### Step 3: Render the Report

From the technical_report directory:

quarto render report.qmd --to pdf

---

## 3. Output Location

The generated PDF will be saved to:

technical_report/reports/

---

## 4. Notes

- Use \newpage only where a full page break is required (e.g., after the cover page).
- Subsections will not automatically create new pages.
- Ensure all include paths are correct relative to index.qmd.

---

## 5. Troubleshooting

HTML instead of PDF:
- Ensure TinyTeX is installed
- Ensure the --to pdf flag is used

LaTeX errors:
- Re-run: quarto install tinytex

File not found errors:
- Confirm include paths are correct

---

End of instructions.
