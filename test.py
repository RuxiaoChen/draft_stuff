from datetime import datetime, timedelta

def iso8601_to_mdy_hms(iso_str):
    dt = datetime.fromisoformat(iso_str)
    return dt.strftime('%m/%d/%Y %H:%M:%S')

print(iso8601_to_mdy_hms('2022-08-01T06:52:08-04:00'))