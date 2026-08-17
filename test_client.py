import urllib.request
import json

# Define the absolute target endpoint
url = "http://127.0.0"
 
# Prepare the data packet exactly how our data contract requires it
payload = {
    "TransactionAmt": 4500.00,
    "ProductCD": "W",
    "card1": 13926.0,
    "card4": "visa",
    "card6": "credit",
    "P_emaildomain": "gmail.com",
    "DeviceType": "mobile",
    "DeviceInfo": "iPhone"
}

# Encode the dictionary into standard JSON bytes
json_data = json.dumps(payload).encode('utf-8')

# Construct the transaction call with strict headers
req = urllib.request.Request(
    url, 
    data=json_data, 
    headers={'Content-Type': 'application/json', 'accept': 'application/json'},
    method='POST'
)

print("⏳ Sending credit card transaction payload to FastAPI core...")
try:
    with urllib.request.urlopen(req) as response:
        response_body = response.read().decode('utf-8')
        print("\n--- 🏦 LIVE TRANSACTION SYSTEM RESPONSE ---")
        print(json.dumps(json.loads(response_body), indent=2))
except Exception as e:
    print(f"\n❌ Network Connection Error: {str(e)}")
    print("👉 Action: Make sure your first terminal is running 'uvicorn app:app --reload'")


