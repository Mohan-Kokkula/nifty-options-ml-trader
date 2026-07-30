# Notification Setup Guide — Email + WhatsApp

## Email Setup (Gmail)

### Step 1: Enable 2-Factor Authentication
1. Go to https://myaccount.google.com/security
2. Enable **2-Step Verification** if not already on

### Step 2: Create an App Password
1. Go to https://myaccount.google.com/apppasswords
2. Select **Mail** and **Other (Custom name)**
3. Name it "Nifty Trader"
4. Click **Generate**
5. Copy the 16-character password (e.g., `abcd efgh ijkl mnop`)

### Step 3: Configure settings.env
```env
EMAIL_ENABLED=true
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=your_email@gmail.com
SMTP_PASSWORD=abcdefghijklmnop    # App Password (no spaces)
EMAIL_FROM=your_email@gmail.com
EMAIL_TO=your_email@gmail.com     # Can be a different email
```

### Other Email Providers

| Provider | SMTP Host | Port |
|----------|-----------|------|
| Gmail | smtp.gmail.com | 587 |
| Outlook | smtp-mail.outlook.com | 587 |
| Yahoo | smtp.mail.yahoo.com | 587 |
| Zoho | smtp.zoho.in | 587 |

---

## WhatsApp Setup (Twilio)

### Step 1: Create a Twilio Account
1. Go to https://www.twilio.com/try-twilio
2. Sign up (free tier includes WhatsApp sandbox)
3. Verify your phone number

### Step 2: Set Up WhatsApp Sandbox
1. In Twilio Console, go to **Messaging** → **Try it out** → **Send a WhatsApp message**
2. You'll see a sandbox number (usually +1 415 523 8886)
3. Send the join message from YOUR WhatsApp:
   - Open WhatsApp on your phone
   - Send a message to the sandbox number: `join <your-sandbox-keyword>`
   - Example: `join hungry-elephant`
4. You'll receive a confirmation reply

### Step 3: Get Your Twilio Credentials
1. Go to https://console.twilio.com
2. On the dashboard, find:
   - **Account SID** (starts with `AC...`)
   - **Auth Token** (click to reveal)

### Step 4: Configure settings.env
```env
WHATSAPP_ENABLED=true
TWILIO_ACCOUNT_SID=ACxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
TWILIO_AUTH_TOKEN=your_auth_token_here
TWILIO_WHATSAPP_FROM=whatsapp:+14155238886    # Twilio sandbox number
WHATSAPP_TO=whatsapp:+91XXXXXXXXXX            # Your Indian mobile number
```

### Step 5: Test It
```bash
export $(cat config/settings.env | grep -v '^#' | xargs)
python scripts/test_notifications.py
```

### Production WhatsApp (Optional)
For production (no sandbox limitations):
1. Apply for WhatsApp Business API via Twilio
2. Register your business number
3. Get message templates approved
4. Update `TWILIO_WHATSAPP_FROM` with your business number

### Twilio Free Tier Limits
- Sandbox: Messages only to numbers that joined the sandbox
- Trial account: $15.50 credit
- WhatsApp messages: ~$0.005 per message
- Sufficient for hundreds of trade alerts

---

## Verify Both Notifications

```bash
# Load environment
export $(cat config/settings.env | grep -v '^#' | xargs)

# Run verification
python scripts/verify_setup.py

# Send test messages
python scripts/test_notifications.py
```

You should receive:
- An HTML email with a formatted trade alert table
- A WhatsApp message with the trade details

## What Triggers Notifications

| Event | Email | WhatsApp |
|-------|-------|----------|
| Order placed (BUY/SELL) | ✅ | ✅ |
| Order failed | ✅ | ✅ |
| Risk limit blocked | ✅ | ✅ |
| Position closed | ✅ | ✅ |
| All orders cancelled | ✅ | ✅ |
| Withdrawal attempt blocked | ✅ | ✅ |
| API/connection error | ✅ | ✅ |

Notifications are sent in background threads so they never
block or slow down trade execution.
