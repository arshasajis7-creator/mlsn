# RCWA Command-Line Guide

این راهنما توضیح می‌دهد چگونه بدون رابط گرافیکی و صرفاً در خط فرمان، مسائل RCWA را تعریف، اجرا و نتایج را تحلیل کنید. فارغ از دیتاست یا هندسه‌ی خاص، ساختار کلی مراحل به شرح زیر است.

---

## 1. ساختار کلی پیکربندی

تمام اسکریپت‌ها و داده‌ها در پوشه‌ی `rcwa_adaptor/` قرار دارند. شبیه‌سازی با فایل‌های JSON انجام می‌شود که ساختاری مشابه زیر دارند:

```json
{
  "schema_version": "step3-0.7",
  "freq_GHz": {"start": 1.0, "stop": 18.0, "points": 21},
  "cell": {"Lx_m": 0.03, "Ly_m": 0.03},
  "layers": {
    "L1": {"csv": "m1.csv", "thickness_m": 0.004},
    "L2_mask": {
      "csv_solid": "m2.csv",
      "csv_hole": "mhole.csv",
      "thickness_m": 0.0019,
      "grid": {"Nx": 48, "Ny": 48},
      "holes": [
        {"type": "circle", "x_m": 0.0, "y_m": 0.0, "diameter_m": 0.006},
        {"type": "circle", "x_m": 0.006, "y_m": 0.0, "diameter_m": 0.0035}
      ]
    },
    "L3": {"csv": "m3.csv", "thickness_m": 0.010}
  },
  "polarization": "TE",
  "n_harmonics": [11, 11],
  "backing": {"type": "metal", "eps_imag_clamp": 1e8},
  "angles": {"theta_deg": 0.0, "phi_deg": 0.0},
  "solver": {"check_convergence": false, "atol": 1e-3, "rtol": 1e-2, "max_iters": 30},
  "tolerances": {"energy": 2.0, "nonnegativity": 1e-3, "strict": false},
  "output_prefix": "example_run"
}
```

### فیلدهای مهم
- `freq_GHz`: بازه و تعداد نقاط فرکانس (GHz).
- `cell`: ابعاد سلول دوره‌ای (m).
- `layers`:
  - `L1`, `L3`: لایه‌های همگن با ضخامت (m) و فایل ماده.
  - `L2_mask`: لایه‌ی دوره‌ای با ماده‌ی صلب، ماده‌ی سوراخ، ضخامت، grid و سوراخ‌ها.
  - `holes`: اشیای دایره‌ای یا مربعی با مختصات و اندازه‌ی بر حسب متر.
- `polarization`: یکی از `TE`, `TM`, یا `AVG`.
- `n_harmonics`: برای لایه‌های دوره‌ای (اگر یک عدد باشد، 1D؛ اگر دو عدد باشد 2D).
- `backing`: معمولاً `metal` (PEC) با پارامتر خیالی بزرگ.
- `angles`: زاویه‌ی ورود (theta/phi).
- `solver`: تنظیمات همگرایی.
- `tolerances`: کنترل انرژی (اگر strict=false باشد فقط هشدار می‌دهد).
- `output_prefix`: نام فایل خروجی CSV.

---

## 2. اجرای شبیه‌سازی

با فرض اینکه در ریشه‌ی پروژه (`C:\mlsn`) هستید:

```bash
python rcwa_adaptor\adapter_step1.py rcwa_adaptor\configs\my_case.json
```

این دستور:
1. فایل پیکربندی `my_case.json` را می‌خواند،
2. آرایه‌ی فرکانس را sweep می‌کند،
3. نتایج را در فایل CSV با نام `output_prefix_step1_results.csv` ذخیره می‌کند،
4. فایل JSONL در `step1_logs/` برای جزئیات اجرا تولید می‌کند.

اگر می‌خواهید TE و TM را جداگانه اجرا و نتایج را میانگین بگیرید:

```bash
python rcwa_adaptor\step3_batch.py rcwa_adaptor\configs\my_case.json --suffix avg
```

این اسکریپت دو بار شبیه‌سازی (TE و TM) را اجرا کرده و نتیجه‌ی میانگین را در فایل `output_prefix_avg_step1_results.csv` می‌نویسد.

### پارامترهای مفید `step3_batch.py`
- `--polarizations TE TM` برای تعیین قطبش‌های اجرا.
- `--suffix avg` برای افزودن پسوند به نام فایل خروجی.
- `--log-dir custom_logs` جهت تغییر مسیر ذخیره‌ی فایل‌های JSONL.

---

## 3. شرایط متداول

### 3.1 فیلم‌های دولایه (بدون ماسک)

در صورتی که لایه‌ی دوره‌ای ندارید، می‌توانید `L2_mask` را حذف کنید یا ضخامت آن را صفر بگذارید:

```json
"layers": {
  "L1": {"csv": "m1.csv", "thickness_m": 0.004},
  "L3": {"csv": "m3.csv", "thickness_m": 0.010}
}
```

این حالت مثل روش TMM است و `n_harmonics` می‌تواند `1` باشد.

### 3.2 ماسک با سوراخ‌های دایره‌ای

یک نمونه‌ی متداول:

```json
"holes": [
  {"type": "circle", "x_m": 0.0, "y_m": 0.0, "diameter_m": 0.006},
  {"type": "circle", "x_m": 0.006, "y_m": 0.0, "diameter_m": 0.003}
]
```

### 3.3 ماسک با سوراخ‌های مربعی

```json
"holes": [
  {"type": "square", "x_m": 0.0, "y_m": 0.0, "width_m": 0.004, "height_m": 0.004}
]
```

### 3.4 ماسک با شکل مختلط (پلاس)

```json
"holes": [
  {"type": "circle", "x_m": 0.0, "y_m": 0.0, "diameter_m": 0.006},
  {"type": "circle", "x_m": 0.006, "y_m": 0.0, "diameter_m": 0.0035},
  {"type": "circle", "x_m": -0.006, "y_m": 0.0, "diameter_m": 0.0035},
  {"type": "circle", "x_m": 0.0, "y_m": 0.006, "diameter_m": 0.0035},
  {"type": "circle", "x_m": 0.0, "y_m": -0.006, "diameter_m": 0.0035}
]
```

---

## 4. تحلیل نتایج

نتایج در فایل CSV شامل ستون‌های `freq_GHz`, `wavelength_m`, `R`, `T`, `A_raw`, `A`, `RL_dB` هستند.

برای نمایش خلاصه، اسکریپت آماده‌ای داریم:

```bash
python rcwa_adaptor\step3_analyze.py rcwa_adaptor\results\example_run_step1_results.csv
```

خروجی شامل:
- بازه‌ی فرکانس،
- حداقل/حداکثر بازتاب و RL،
- 최소/حداکثر جذب خام (`A_raw`)،
- فرکانسی که بیشترین مشکل (بازتاب بالا یا جذب منفی) دارد.

---

## 5. سناریوهای نمونه

### 5.1 تنظیم یک لایه‌ی دولایه‌ی بهینه (m1/m3)

```json
{
  "freq_GHz": {"start": 1.0, "stop": 18.0, "points": 21},
  "cell": {"Lx_m": 0.03, "Ly_m": 0.03},
  "layers": {
    "L1": {"csv": "m1.csv", "thickness_m": 0.0045},
    "L3": {"csv": "m3.csv", "thickness_m": 0.010}
  },
  "polarization": "AVG",
  "n_harmonics": 1,
  "backing": {"type": "metal", "eps_imag_clamp": 1e8},
  "angles": {"theta_deg": 0.0, "phi_deg": 0.0},
  "solver": {"check_convergence": false, "atol": 1e-3, "rtol": 1e-2, "max_iters": 20},
  "tolerances": {"energy": 1e-3, "nonnegativity": 1e-4, "strict": true},
  "output_prefix": "two_layer_final"
}
```

اجرای آن:

```bash
python rcwa_adaptor\adapter_step1.py rcwa_adaptor\configs\two_layer_final.json
python rcwa_adaptor\step3_analyze.py rcwa_adaptor\two_layer_final_step1_results.csv
```

### 5.2 ماسک سه‌سوراخ

```json
{
  "layers": {
    "L1": {"csv": "m1.csv", "thickness_m": 0.006},
    "L2_mask": {
      "csv_solid": "m2.csv",
      "csv_hole": "mhole.csv",
      "thickness_m": 0.0019,
      "grid": {"Nx": 48, "Ny": 48},
      "holes": [
        {"type": "circle", "x_m": -0.0075, "y_m": 0.0, "diameter_m": 0.005},
        {"type": "circle", "x_m": 0.0075, "y_m": 0.0, "diameter_m": 0.007},
        {"type": "circle", "x_m": 0.0, "y_m": 0.008, "diameter_m": 0.006}
      ]
    },
    "L3": {"csv": "m3.csv", "thickness_m": 0.0009}
  }
}
```

---

## 6. نکات و مشکلات رایج

1. **خطای انرژی یا جذب منفی**: `A_raw` منفی یا `RL_dB` مثبت به معنای بازتاب بسیار زیاد است (نه جذب). باید هندسه یا مواد را تغییر دهید.
2. **عدم اجرا یا زمان طولانی**: تنظیمات مثل `grid Nx/Ny` بزرگ یا `[n_harmonics]` بالا باعث کندی شدید می‌شود. از مقادیر 32–56، 11–15 شروع کنید و در صورت نیاز افزایش دهید.
3. **شناسه‌ی فایل خروجی**: در صورت اجرای چندباره با `output_prefix` یکسان، فایل‌های خروجی overwrite می‌شوند.
4. **فراخوانی batch با زاویه**: برای زاویه‌های مایل باید `angles.theta_deg` را در فایل config تغییر دهید؛ سپس batch را با همان فایل اجرا کنید.

---

## 7. بهترین روش‌ها

- از نام‌گذاری منظم `output_prefix` استفاده کنید (مثلاً `projA_L1_4.5mm`).
- بعد از هر اجرا `step3_analyze.py` را برای کنترل کیفیت (بازتاب/جذب) اجرا کنید.
- جمله‌ی `strict=true` را وقتی مدل معتبر شد فعال کنید تا انرژی دقیق بررسی شود.
- اگر `A_raw` همچنان منفی است یا RL بالا، نشان‌دهنده‌ی این است که ساختار از نظر فیزیکی جاذب نیست؛ باید طراحی لایه‌ها/ماسک تغییر کند.

---

## 8. منابع

- `adapter_step1.py`: اجرای پایهٔ RCWA.
- `step3_batch.py`: اجرای قطبش‌های مختلف و میانگین‌گیری.
- `step3_analyze.py`: خلاصه و تحلیل نتایج.
- `step3_thickness_sweep.py` و `step3_two_layer_grid.py`: نمونه اسکریپت‌های sweep ضخامت‌ها و لایه‌ها.

با این راهنما می‌توانید انواع سناریوهای جذب یا ساختارهای دوره‌ای را در خط فرمان تعریف و اجرا کنید و نمودار Reflection Loss را برای تحلیل سریع به دست آورید.
