# Model Orders: real-money copy of the model portfolio

Portfolio Manager → Return Tracker → **🎫 Model Orders**. Code: `utils/model_book.py` (ticket, state),
`utils/model_orders_view.py` (tab), `utils/dhan_orders.py` (the only module that can place an order).

## How it works

1. The paper model (OptComp V2.2) runs every evening as before.
2. The ticket compares the model's holdings with your model book: **sell** what the model dropped, **buy**
   what it holds and you do not. One slot = your model equity ÷ 15, whole shares, limit 1% through the last close.
3. You confirm. A fill updates the model book, My Book and the Journal (tag `model`), with the gap to the model's price.
4. The daily Telegram brief carries the ticket when there is one.

## Stage 2 — you confirm each ticket

| Mode | What you need | Status |
|---|---|---|
| Place the orders yourself in the Dhan app, record fills in the tab | nothing | works now |
| "Send to Dhan" from the tab after typing CONFIRM | the three items below | built, **not yet tried against Dhan** |

1. Secrets: under `[dhan]` add `orders_enabled = true` (the `client_id` is already there).
2. Today's Dhan login: sync Dhan on the Import tab (the token lasts a day).
3. A fixed IP address registered with Dhan (web.dhan.co → Profile → DhanHQ Trading APIs → Add IP; changeable once
   in 7 days). Streamlit Cloud has no fixed address. Either set `order_proxy = "http://user:pass@host:port"` to a
   fixed-IP exit, or run the dashboard on a small server with a fixed address.

### Setting up the fixed-IP exit (about 15 minutes, free)

`deploy/order-proxy/` is a tiny tunnel that only reaches Dhan's API and only with a password. Your Dhan token
passes through it encrypted; the server never sees it.

1. Create a small always-on Linux server with a **static public IP**. Free choices: Oracle Cloud "Always Free"
   VM (Mumbai region), or Google Cloud e2-micro. Any ₹300–500/month VPS also works. Pick Ubuntu.
2. In the provider's firewall, allow inbound **TCP 8443**.
3. On the server run:
   `curl -fsSL https://raw.githubusercontent.com/adi-git01/nifty-dashboard/main/deploy/order-proxy/setup.sh | sudo bash`
   It prints the server's IP and the two secrets lines.
4. Register that IP with Dhan (web.dhan.co → Profile → DhanHQ Trading APIs → Add IP).
5. Paste the two lines under `[dhan]` in the Streamlit app's secrets and save.
6. In the dashboard: Model Orders → Settings → **Check the address Dhan will see**. It should show the server's IP.

First live use: untick everything except one cheap order and send that alone.

## Stage 3 — fully automatic (not built)

| Needed | Why |
|---|---|
| A host with a fixed IP that runs after the evening engine (small VPS, or Dhan Cloud, which Dhan says is exempt from the IP rule) | GitHub Actions and Streamlit Cloud have changing addresses |
| A Dhan token without a daily manual login | tokens last 24 hours; needs Dhan's token-renewal or TOTP login flow, with the secret kept on that host only |
| A few weeks of stage 2 fills | the Order history shows the real gap to the model's price; automate only if it is small |
| Limits agreed in advance | per-order and per-day caps, a pause switch (already in Settings), and an alert on every order and every refusal |
| A check of the current SEBI retail-algo rules with Dhan | whether a self-run strategy at this order rate needs registering |

Guard rails already enforced in `utils/dhan_orders.py`: delivery limit orders on NSE only; one order per order id;
caps per order and per day; a sale never exceeds the model book; off unless `orders_enabled` is set.
