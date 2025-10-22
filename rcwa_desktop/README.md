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
- Geometry tab: editable table of mask holes (circle or square) with coordinates in millimeters.
- Execution tab: run the existing `adapter_step1.py`, view stdout/stderr logs, and visualize Reflection Loss (dB) vs frequency immediately inside the UI.
- Built-in optimiser tab section to tune the mask geometry before running.
- Load / save configuration JSON compatible with the RCWA adapter, including automatic persistence of the last run.

## Requirements

```bash
pip install PyQt6 matplotlib
```

## Running

```bash
python main.py
```

This is the first fully functional milestone. Future enhancements can add batch sweeps, job history, and advanced geometry primitives.
