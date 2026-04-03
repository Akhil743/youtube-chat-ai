import os
import logging
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from app.services.quota import quota_service

log = logging.getLogger(__name__)
router = APIRouter(prefix="/api/billing", tags=["billing"])

STRIPE_SECRET_KEY = os.getenv("STRIPE_SECRET_KEY")
STRIPE_WEBHOOK_SECRET = os.getenv("STRIPE_WEBHOOK_SECRET")
STRIPE_PRICE_ID = os.getenv("STRIPE_PRICE_ID")

stripe = None
if STRIPE_SECRET_KEY:
    try:
        import stripe as _stripe
        _stripe.api_key = STRIPE_SECRET_KEY
        stripe = _stripe
    except ImportError:
        log.warning("stripe package not installed — billing endpoints disabled")


class CheckoutRequest(BaseModel):
    user_id: str


class SubscriptionStatus(BaseModel):
    is_pro: bool
    user_id: str


@router.post("/create-checkout-session")
async def create_checkout_session(body: CheckoutRequest):
    if not stripe or not STRIPE_PRICE_ID:
        raise HTTPException(status_code=503, detail="Billing not configured")

    try:
        session = stripe.checkout.Session.create(
            mode="subscription",
            line_items=[{"price": STRIPE_PRICE_ID, "quantity": 1}],
            success_url="https://akhilrajs.com/youtube-chat/?upgrade=success",
            cancel_url="https://akhilrajs.com/youtube-chat/?upgrade=cancelled",
            client_reference_id=body.user_id,
            metadata={"user_id": body.user_id},
        )
        return {"checkout_url": session.url}
    except Exception as e:
        log.error(f"Stripe checkout error: {e}")
        raise HTTPException(status_code=500, detail="Failed to create checkout session")


@router.post("/webhook")
async def stripe_webhook(request: Request):
    if not stripe or not STRIPE_WEBHOOK_SECRET:
        raise HTTPException(status_code=503, detail="Billing not configured")

    payload = await request.body()
    sig = request.headers.get("stripe-signature")

    try:
        event = stripe.Webhook.construct_event(payload, sig, STRIPE_WEBHOOK_SECRET)
    except Exception as e:
        log.warning(f"Webhook verification failed: {e}")
        raise HTTPException(status_code=400, detail="Invalid signature")

    if event["type"] == "checkout.session.completed":
        session = event["data"]["object"]
        user_id = session.get("client_reference_id") or session.get("metadata", {}).get("user_id")
        customer_id = session.get("customer")
        if user_id:
            quota_service.set_pro(user_id, customer_id)
            log.info(f"User {user_id} upgraded to Pro (customer: {customer_id})")

    elif event["type"] == "customer.subscription.deleted":
        sub = event["data"]["object"]
        customer_id = sub.get("customer")
        for record in quota_service._users.values():
            if record.stripe_customer_id == customer_id:
                quota_service.revoke_pro(record.user_id)
                log.info(f"User {record.user_id} Pro revoked (subscription cancelled)")
                break

    return {"status": "ok"}


@router.get("/status/{user_id}", response_model=SubscriptionStatus)
async def subscription_status(user_id: str):
    info = quota_service.get_quota_info(user_id)
    return SubscriptionStatus(
        is_pro=info["is_pro"] if info else False,
        user_id=user_id,
    )
