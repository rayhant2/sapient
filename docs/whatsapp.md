# WhatsApp delivery

`notifications/whatsapp.py` is the runtime output sink for Twilio WhatsApp. It:

1. Loads the output owner through the database interface.
2. Formats a message for the relevant alert type.
3. Sends the message through Twilio.
4. Records the accepted message in the `alerts` table.

Unflagged hypothesis scans are intentionally silent. Cross-portfolio messages are
stored without a ticker, which the existing Supabase schema supports.

## Configuration

Set these values only when real WhatsApp delivery is required:

```dotenv
TWILIO_ACCOUNT_SID=
TWILIO_AUTH_TOKEN=
TWILIO_WHATSAPP_FROM=whatsapp:+14155238886
```

Each user's `whatsapp_number` must include Twilio's `whatsapp:+` prefix. For the
Twilio Sandbox, the recipient must first join the sandbox. Production delivery
requires an approved WhatsApp sender.

The implementation never includes LLM API keys or Twilio credentials in message
bodies or alert records. Delivery exceptions are replaced with sanitized errors.
Unit tests use a fake Twilio client and do not send live messages.
