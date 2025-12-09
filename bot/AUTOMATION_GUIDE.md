# Trading Bot Automation Guide
## Complete Guide for 24/7 Automated Trading

> **Bottom Line**: For a trading bot that needs to run **continuously during market hours** (9:15 AM - 3:30 PM), **Oracle Cloud Free Tier** is the **BEST option**. GitHub Actions is great for scheduled tasks but not continuous processes.

---

## 📊 Comparison: All Automation Options

| Feature | GitHub Actions | Oracle Cloud Free | AWS Free Tier | Local PC | Railway/Render |
|---------|---------------|-------------------|---------------|----------|----------------|
| **Cost** | FREE (2000 min/mo) | FREE (forever) | FREE (12 months) | Electricity | $5-20/month |
| **Always On** | ❌ No | ✅ Yes | ✅ Yes | ✅ Yes | ✅ Yes |
| **Max Runtime** | 6 hours | Unlimited | Unlimited | Unlimited | Unlimited |
| **Best For** | Scheduled tasks | Full bot | Full bot | Testing | Production |
| **Setup Difficulty** | Easy | Medium | Medium | Easy | Easy |
| **Reliability** | High | High | Very High | Low (power cuts) | High |

---

## 🏆 RECOMMENDED: Oracle Cloud Free Tier

### Why Oracle Cloud?

1. **Always Free** - Not a trial, genuinely free forever
2. **2 AMD VMs** with 1GB RAM each (enough for bot)
3. **24/7 uptime** - Perfect for trading hours
4. **Static IP** - Consistent connectivity
5. **No credit card charges** - Won't accidentally bill you

---

## 🚀 Option 1: Oracle Cloud Setup (BEST)

### Step 1: Create Oracle Cloud Account

1. Go to: https://www.oracle.com/cloud/free/
2. Click "Start for Free"
3. Fill in details:
   - Email (use personal, not temporary)
   - Country: India
   - **Home Region: India West (Mumbai)** ← Important for low latency!
4. Add payment method (won't be charged for Always Free)
5. Verify email

### Step 2: Create a Free VM

1. Login to Oracle Cloud Console
2. Go to **Compute → Instances → Create Instance**

3. Configure:
   ```
   Name: trading-bot
   Compartment: (default)
   
   Image: Ubuntu 22.04 (or 24.04)
   Shape: VM.Standard.E2.1.Micro (Always Free)
   
   Networking:
   - Create new VCN
   - Public IP: Assign (for SSH access)
   
   SSH Keys:
   - Generate new key pair
   - Download BOTH keys (public + private)
   ```

4. Click **Create** and wait ~2 minutes

### Step 3: Connect to Your VM

**Windows (using PuTTY):**
```
1. Download PuTTY: https://www.putty.org/
2. Convert .key to .ppk using PuTTYgen
3. Open PuTTY:
   - Host: <public-ip-from-oracle>
   - Port: 22
   - Connection → SSH → Auth → Private key: your.ppk
4. Click Open
5. Login as: ubuntu
```

**Windows (using OpenSSH in CMD/PowerShell):**
```powershell
ssh -i path\to\private-key.key ubuntu@<public-ip>
```

### Step 4: Install Python & Dependencies

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Python 3.11
sudo apt install -y python3.11 python3.11-venv python3-pip

# Install git
sudo apt install -y git

# Clone your repo
git clone https://github.com/Sahilbhatane/openalgo.git
cd openalgo

# Create virtual environment
python3.11 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 5: Configure Environment

```bash
# Create .env file
nano .env
```

Add your configuration:
```env
# Trading Mode
TRADING_MODE=PAPER

# AngelOne Credentials
ANGEL_API_KEY=your_api_key
ANGEL_CLIENT_ID=your_client_id
ANGEL_PASSWORD=your_password
ANGEL_TOTP_SECRET=your_totp_secret

# OpenAlgo
OPENALGO_API_KEY=your_openalgo_key
OPENALGO_HOST=http://127.0.0.1:5000

# Telegram (optional)
TELEGRAM_BOT_TOKEN=your_bot_token
TELEGRAM_CHAT_ID=your_chat_id
```

Press `Ctrl+X`, then `Y`, then `Enter` to save.

### Step 6: Create Systemd Service (Auto-Start)

```bash
sudo nano /etc/systemd/system/trading-bot.service
```

Add this content:
```ini
[Unit]
Description=OpenAlgo Trading Bot
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/openalgo
Environment=PATH=/home/ubuntu/openalgo/.venv/bin
ExecStart=/home/ubuntu/openalgo/.venv/bin/python -m bot.main
Restart=always
RestartSec=10

# Logging
StandardOutput=append:/home/ubuntu/openalgo/logs/bot.log
StandardError=append:/home/ubuntu/openalgo/logs/bot_error.log

[Install]
WantedBy=multi-user.target
```

### Step 7: Enable and Start Service

```bash
# Create logs directory
mkdir -p ~/openalgo/logs

# Reload systemd
sudo systemctl daemon-reload

# Enable auto-start on boot
sudo systemctl enable trading-bot

# Start the service
sudo systemctl start trading-bot

# Check status
sudo systemctl status trading-bot
```

### Step 8: Useful Commands

```bash
# View live logs
tail -f ~/openalgo/logs/bot.log

# Stop bot
sudo systemctl stop trading-bot

# Restart bot
sudo systemctl restart trading-bot

# View service status
sudo systemctl status trading-bot

# View last 100 lines of logs
tail -100 ~/openalgo/logs/bot.log
```

### Step 9: Daily Token Refresh (Cron Job)

The bot needs a fresh TOTP token daily. Since automated TOTP is a security risk, you have two options:

**Option A: Manual Daily Login (Recommended)**
- Login to broker app/website at 7:30 AM daily
- Token will be refreshed

**Option B: Semi-Automated with Telegram Alert**

Create a reminder script:
```bash
nano ~/login_reminder.sh
```

```bash
#!/bin/bash
# Send Telegram reminder to login

TELEGRAM_TOKEN="your_bot_token"
CHAT_ID="your_chat_id"
MESSAGE="🔔 TRADING BOT REMINDER%0A%0APlease login to AngelOne to refresh session token.%0A%0ATime: $(TZ='Asia/Kolkata' date '+%H:%M IST')"

curl -s "https://api.telegram.org/bot${TELEGRAM_TOKEN}/sendMessage?chat_id=${CHAT_ID}&text=${MESSAGE}"
```

```bash
chmod +x ~/login_reminder.sh

# Add to cron (runs at 7:15 AM IST = 1:45 AM UTC)
crontab -e
# Add this line:
45 1 * * 1-5 /home/ubuntu/login_reminder.sh
```

---

## 📋 Option 2: GitHub Actions (For Scheduled Tasks Only)

GitHub Actions is **already configured** in your repo for morning learning:
- File: `.github/workflows/morning-learning.yml`
- Runs at: 8:00 AM IST on weekdays
- Purpose: Generate daily trading plan before market opens

### Limitations for Full Bot:
- ❌ Maximum 6-hour runtime (market is open 6+ hours)
- ❌ May be delayed during high GitHub load
- ❌ Not designed for continuous processes

### What It's Good For:
- ✅ Morning analysis (8:00 AM - before market)
- ✅ End-of-day reports (after 3:30 PM)
- ✅ Nightly data processing
- ✅ Weekly performance reports

### Enable GitHub Actions:

1. Go to: https://github.com/Sahilbhatane/openalgo/settings/secrets/actions
2. Add these secrets:
   ```
   ANGEL_API_KEY
   ANGEL_CLIENT_ID
   OPENALGO_API_KEY
   TELEGRAM_BOT_TOKEN (optional)
   TELEGRAM_CHAT_ID (optional)
   ```
3. Go to **Actions** tab → Enable workflows

### Usage:
- **Automatic**: Runs at 8:00 AM IST weekdays
- **Manual**: Actions → Daily Morning Learning → Run workflow

---

## 💻 Option 3: Local PC with Task Scheduler

### Windows Task Scheduler Setup:

**Step 1: Create a batch file**
```batch
@echo off
cd /d C:\Users\sahil\OneDrive\Desktop\CODE\Projects\Task\openalgo
call .venv\Scripts\activate
python -m bot.main
```
Save as: `run_bot.bat`

**Step 2: Create Scheduled Task**
1. Open Task Scheduler (`taskschd.msc`)
2. Create Task (not Basic Task)
3. Configure:
   - **Name**: Trading Bot
   - **Run whether user is logged on or not**: ✅
   - **Run with highest privileges**: ✅

4. **Triggers**:
   - New → Daily
   - Start: 7:30 AM
   - Recur every: 1 day
   - Stop if runs longer than: 10 hours

5. **Actions**:
   - Start a program
   - Program: `C:\Users\sahil\OneDrive\Desktop\CODE\Projects\Task\openalgo\run_bot.bat`

6. **Conditions**:
   - Wake computer: ✅
   - Start only if AC power: ✅

7. **Settings**:
   - If task fails, restart every: 5 minutes
   - Attempt to restart up to: 3 times

### Limitations:
- ❌ Power cuts stop the bot
- ❌ Computer must stay on
- ❌ Windows updates may restart PC
- ❌ Internet outages

---

## ☁️ Option 4: Other Cloud Providers

### AWS Free Tier (12 months free)
- 750 hours/month EC2 t2.micro
- Setup similar to Oracle Cloud
- After 12 months: ~$5-10/month

### Google Cloud (Always Free)
- f1-micro instance (0.6GB RAM - tight!)
- US regions only for free tier
- Higher latency to India

### Railway.app
- $5/month (no free tier for always-on)
- Very easy deployment
- Good for production

### Render.com
- Free tier sleeps after 15 min inactivity
- Paid: $7/month for always-on
- Easy GitHub integration

---

## 🔐 Security Best Practices

### 1. Never Store Secrets in Code
```bash
# Use environment variables
export ANGEL_API_KEY="your_key"

# Or .env file (add to .gitignore!)
echo ".env" >> .gitignore
```

### 2. Use SSH Keys, Not Passwords
```bash
# Generate SSH key
ssh-keygen -t ed25519 -C "trading-bot"

# Add public key to Oracle Cloud
```

### 3. Firewall Rules
```bash
# Allow only SSH (Oracle Cloud default)
# Block all other incoming traffic

# On Ubuntu (if needed):
sudo ufw allow 22/tcp
sudo ufw enable
```

### 4. Keep System Updated
```bash
# Weekly update cron job
sudo crontab -e
# Add:
0 3 * * 0 apt update && apt upgrade -y
```

---

## 📊 Monitoring Your Bot

### Option 1: Telegram Alerts (Recommended)

Already configured in your bot. Add these environment variables:
```env
TELEGRAM_BOT_TOKEN=your_bot_token
TELEGRAM_CHAT_ID=your_chat_id
```

You'll receive:
- 🟢 Bot started
- 📊 Trade entries/exits
- 🔴 Errors and warnings
- 📈 Daily P&L summary

### Option 2: UptimeRobot (Free)

1. Go to https://uptimerobot.com
2. Create free account
3. Add monitor:
   - Type: Ping or Port
   - URL: Your server IP
4. Get alerts when server goes down

### Option 3: Simple Health Check Script

```bash
# Create health check
nano ~/health_check.sh
```

```bash
#!/bin/bash
if ! pgrep -f "bot.main" > /dev/null; then
    # Bot not running - restart and alert
    sudo systemctl restart trading-bot
    
    # Send Telegram alert
    curl -s "https://api.telegram.org/bot${TELEGRAM_TOKEN}/sendMessage?chat_id=${CHAT_ID}&text=⚠️ Bot was down, restarted automatically"
fi
```

```bash
chmod +x ~/health_check.sh

# Run every 5 minutes
crontab -e
# Add:
*/5 * * * * /home/ubuntu/health_check.sh
```

---

## 🎯 Quick Start Summary

### For Paper Trading (Testing):
1. Use **Local PC** with Task Scheduler
2. Run for 3-4 weeks
3. Review daily reports

### For Live Trading:
1. Sign up for **Oracle Cloud Free Tier**
2. Create Always Free VM
3. Install bot as systemd service
4. Configure Telegram alerts
5. Login manually each morning (7:30 AM)

### Cost Breakdown:

| Solution | Monthly Cost | Annual Cost |
|----------|--------------|-------------|
| Oracle Cloud Free | ₹0 | ₹0 |
| GitHub Actions | ₹0 | ₹0 |
| AWS (after free tier) | ₹400-800 | ₹4,800-9,600 |
| Local PC electricity | ₹200-500 | ₹2,400-6,000 |
| Railway/Render | ₹400-1,600 | ₹4,800-19,200 |

---

## ❓ FAQ

**Q: Can I automate the TOTP login?**
A: Technically yes, but NOT recommended. Storing TOTP secret is a security risk. Manual login takes 30 seconds and protects your account.

**Q: What if Oracle Cloud is down?**
A: Oracle Cloud has 99.95% SLA. In rare cases, have broker mobile app ready for manual trading.

**Q: Can I run multiple bots?**
A: Yes! Oracle Free Tier gives 2 VMs. Run paper trading on one, live on another.

**Q: How do I update the bot?**
A: 
```bash
cd ~/openalgo
git pull origin auto-bot
sudo systemctl restart trading-bot
```

**Q: What's the latency from Oracle Mumbai to NSE?**
A: ~2-5ms. Fast enough for any non-HFT strategy.

---

## 📞 Support

If you face issues:
1. Check logs: `tail -100 ~/openalgo/logs/bot.log`
2. Check service: `sudo systemctl status trading-bot`
3. Check internet: `ping google.com`
4. Restart: `sudo systemctl restart trading-bot`

---

*Last Updated: December 2025*
*Author: OpenAlgo Trading Bot*
