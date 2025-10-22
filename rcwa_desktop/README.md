# RCWA Desktop UI (Stage 3)

Standalone PyQt6 application for configuring and launching RCWA simulations.

## جدید چه خبر؟

- موتور بهینه‌سازی داخلی برای پیشنهاد ابعاد سوراخ‌ها، ضخامت لایه‌ها و انتخاب مواد
  بر اساس فرکانس هدف.
- رابط کاربری با حافظه داخلی: آخرین تنظیمات به صورت خودکار ذخیره و پس از باز کردن
  برنامه بازیابی می‌شود.
- قابلیت ذخیره و بارگذاری پیکربندی‌های دلخواه از فایل‌های JSON.
- سیستم لاگ‌گیری یکپارچه که برای هر ران پوشه‌ای اختصاصی زیر `logs/` ایجاد و خروجی
  رابط کاربری و آداپتور را ذخیره می‌کند.

## Features

- General tab: frequency sweep, cell dimensions, output prefix with validation.
- Materials tab: select CSV datasets and thickness (mm) for L1, mask (solid/hole) and L3.
- Geometry tab: editable table of circular mask holes with coordinates and diameters in millimeters.
- Execution tab: run the existing `adapter_step1.py`, view stdout/stderr logs, and visualize Reflection Loss (dB) vs frequency immediately inside the UI.
- Built-in optimiser tab section to tune the mask geometry before running.
- Load / save configuration JSON compatible with the RCWA adapter, including automatic persistence of the last run.
- Material CSV paths are normalised to absolute locations inside the repository, so you can simply pick files such as
  `m1.csv` without entering full paths.

## Requirements

```bash
pip install PyQt6 matplotlib
```

## Running

From the repository root run the module entry point so Python can resolve the package-relative imports correctly:

```bash
python -m rcwa_desktop.main
```

On Windows PowerShell the equivalent command is:

```powershell
py -m rcwa_desktop.main
```

Both commands assume you execute them inside the project directory (the folder that contains `rcwa_desktop/`). If you prefer launching from the package directory itself, change into `rcwa_desktop/` first and run `python -m main`.

This is the first fully functional milestone. Future enhancements can add batch sweeps, job history, and advanced geometry primitives.
