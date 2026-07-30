# Kotak Neo API Setup Guide — Read + Trade Only (No Withdrawals)

This guide walks you through configuring Kotak Neo API access
with the correct permissions for safe automated trading.

## Step 1: Log In to Kotak Neo

1. Go to https://neo.kotaksecurities.com/
2. Log in with your **Trading User ID** and **Password**

## Step 2: Navigate to API Dashboard

1. Click **Invest** in the top menu
2. Select **TradeAPI**
3. Click **API Dashboard**

## Step 3: Register TOTP (Time-Based One-Time Password)

TOTP is required for API authentication — it replaces the OTP sent to your phone.

1. In the API Dashboard, find the TOTP registration section
2. Open your authenticator app (Google Authenticator, Microsoft Authenticator, etc.)
3. Scan the QR code shown on screen
4. Enter the 6-digit code from your authenticator app
5. Click **Continue** — you'll see a success confirmation

## Step 4: Get Your API Credentials

After TOTP registration:

1. **API Key (Unique Client Code)**: A 5-character code (e.g., "AB123")
   - Copy this — it's your `BROKER_API_KEY` in OpenAlgo
2. **API Secret (Token)**: A UUID format string (e.g., "ec6a746c-e44b-455e-abf2-c13352b2fc45")
   - Copy this — it's your `BROKER_API_SECRET` in OpenAlgo

## Step 5: Configure API Permissions — CRITICAL

**Use ONLY Read + Trade permissions. Never enable withdrawal.**

When creating/editing your API application:
- ✅ **Read** — view positions, holdings, order book
- ✅ **Trade** — place, modify, cancel orders
- ❌ **Withdrawal** — NEVER enable this
- ❌ **Fund Transfer** — NEVER enable this

Even if you trust your code, an AI agent with withdrawal access
is a security risk. The code in this project also blocks withdrawal
endpoints at the application level as a second safety net.

## Step 6: Register Static IP (if required)

Kotak Neo may require a static IP for API access:

1. In the API Dashboard, find the **Static IP** registration section
2. Enter your server's public IP address
3. If running locally, you may need your ISP's external IP
4. For cloud VPS (AWS, DigitalOcean), use the server's public IP

To find your public IP:
```bash
curl -s ifconfig.me
```

## Step 7: Configure OpenAlgo

In OpenAlgo's `.env` file:

```env
BROKER_API_KEY=your_kotak_unique_client_code
BROKER_API_SECRET=your_kotak_token_uuid
REDIRECT_URL=http://127.0.0.1:5000/kotak/callback
```

Then start OpenAlgo:
```bash
cd openalgo
uv run app.py
```

Visit http://127.0.0.1:5000 and log in with your Kotak Neo credentials.

## Step 8: Configure This Application

In `config/settings.env`:

```env
OPENALGO_HOST=http://127.0.0.1:5000
OPENALGO_API_KEY=your_openalgo_api_key
API_KEY_PERMISSIONS=READ_TRADE_ONLY
```

The `OPENALGO_API_KEY` is generated inside OpenAlgo's dashboard
(not the same as your Kotak API key).

## Step 9: Verify the Connection

```bash
python scripts/verify_setup.py
```

This will:
1. Check that OpenAlgo is reachable
2. Verify your API key works
3. Confirm withdrawal endpoints are blocked
4. Test a quote fetch for NIFTY 50
5. Verify email and WhatsApp notification settings

## Brokerage

Orders placed via Kotak Neo Trade API have **₹0 brokerage** on
all trade-free plans. You only pay:
- Exchange transaction charges
- SEBI turnover fees
- GST (18%)
- STT/CTT (as applicable)
- Stamp duty

## Rate Limits

Kotak Neo allows up to **10 orders per second** via API.
This application defaults to **8/sec** to leave headroom.

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Login fails in OpenAlgo | Ensure TOTP is registered and authenticator is synced |
| "Invalid API key" | Check that Unique Client Code (5 chars) is correct |
| "IP not whitelisted" | Register your server's public IP in API Dashboard |
| Rate limit errors | Reduce `RATE_LIMIT_PER_SEC` in settings.env |
| Orders not executing | Check if market hours (9:15 AM - 3:30 PM IST) |
