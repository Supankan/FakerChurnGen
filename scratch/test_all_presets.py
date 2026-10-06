import requests

presets = ['telecom', 'saas', 'banking', 'ecommerce', 'gaming', 'streaming', 'ridehailing', 'fitness', 'edtech']

all_passed = True
for p in presets:
    # 1. Test preview
    r_prev = requests.post('http://127.0.0.1:5000/api/preview', json={'preset_name': p, 'preview_rows': 25})
    if r_prev.status_code != 200:
        print(f"FAILED PREVIEW for {p}: {r_prev.status_code} {r_prev.text}")
        all_passed = False
        continue

    prev_data = r_prev.json()
    num_rows = prev_data['stats']['num_rows']
    num_cols = prev_data['stats']['num_cols']
    churn_pct = round(prev_data['stats']['actual_churn_rate'] * 100, 1)
    rule_stats = prev_data['stats']['rule_stats']
    print(f"[{p.upper()}] Preview: OK ({num_rows} rows, {num_cols} cols, churn {churn_pct}%, {len(rule_stats)} rules tracked)")

    # 2. Test generate (CSV download)
    r_gen = requests.post('http://127.0.0.1:5000/api/generate', json={'preset_name': p, 'num_rows': 100})
    if r_gen.status_code != 200:
        print(f"FAILED GENERATE for {p}: {r_gen.status_code} {r_gen.text}")
        all_passed = False
    else:
        print(f"[{p.upper()}] Generate CSV: OK ({len(r_gen.content)} bytes)")

if all_passed:
    print("\nALL 9 PRESETS TESTED CLEANLY FOR BOTH PREVIEW AND GENERATE/DOWNLOAD!")
else:
    print("\nSOME PRESET TESTS FAILED!")
