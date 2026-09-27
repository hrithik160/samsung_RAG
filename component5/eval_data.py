"""Labelled evaluation set for Phase 5 (Telemetry + Eval).

Labels are written from the *user's intent*, not from what any component
happens to output:
  should_retrieve : does answering this turn genuinely need fresh evidence?
  gold_docs       : doc_ids that contain that evidence (only when should_retrieve)
  kind            : fact | refine | cosmetic | ack | partial
      fact     - a fresh information need
      refine   - adds a NEW constraint/topic to an ongoing conversation
      cosmetic - presentation-only ask (bullets, shorter, translate...) that
                 must be served from the previous answer without retrieval
      ack      - "thanks" / "okay" -- nothing to answer
      partial  - utterance cut off mid-thought; controller should WAIT
"""
from __future__ import annotations

CORPUS = {
    "S24_Display": "The Galaxy S24 features a Dynamic AMOLED 2X display with a 120Hz refresh rate.",
    "S24_Camera": "The Galaxy S24 camera system includes a 50MP main sensor, a 12MP ultrawide, and a 10MP telephoto with 3x optical zoom.",
    "S24_Battery": "The Galaxy S24 battery capacity is 4000mAh. It supports 25W fast charging and 15W wireless charging.",
    "S24_Water": "The Galaxy S24 is IP68 rated, meaning it is water and dust resistant.",
    "S24_Stylus": "Unlike the S24 Ultra, the base Galaxy S24 does not support the S Pen stylus.",
    "Ultra_Camera": "The Galaxy S24 Ultra camera has a 200MP main sensor and a 5x optical telephoto lens with up to 100x Space Zoom.",
    "Ultra_SPen": "The Galaxy S24 Ultra includes a built-in S Pen stylus that docks inside the phone body with air gestures and note-taking.",
    "Ultra_Battery": "The Galaxy S24 Ultra battery capacity is 5000mAh with 45W wired fast charging.",
    "Buds_Battery": "Galaxy Buds Pro last up to 5 hours per charge with noise cancellation on, and 18 hours with the charging case.",
    "Buds_ANC": "Galaxy Buds Pro offer active noise cancellation with three ambient sound levels and voice detect mode.",
    "Buds_Pairing": "To pair Galaxy Buds Pro open the case near your phone and accept the Bluetooth pairing prompt.",
    "Watch_Water": "The Galaxy Watch is water resistant to 5ATM and IP68, safe for swimming in a pool.",
    "Watch_Battery": "The Galaxy Watch battery lasts about 40 hours with always-on display and supports wireless charging.",
    "Watch_Sleep": "Galaxy Watch sleep tracking measures sleep stages, snoring, and blood oxygen overnight and gives a sleep score.",
    "Warranty": "The phone warranty period is one year. Samsung smartphone coverage includes manufacturing defects but not accidental damage.",
    "Returns": "Customers may return unopened or unused Samsung products within 14 days of delivery for a full refund.",
    "Venue_Pune": "Venue A in Pune accommodates up to 30 people for workshops. Venue B accommodates up to 50 attendees.",
    "Venue_Cancellation": "Venue A allows cancellation without charge up to 48 hours before the workshop. Later cancellation is charged one day.",
    "Venue_Catering": "Venue A offers on-site catering. External caterers are allowed with prior approval.",
    "Travel_Base": "Standard employee travel reimbursement follows the approved booking rate and requires itemized receipts.",
    "Travel_International": "International travel requires foreign-currency receipts and finance pre-approval.",
    "Travel_LateBooking": "Bookings made after travel require a written exception and senior director approval.",
}

def T(text, kind, should_retrieve, gold=(), is_final=True):
    return {"text": text, "kind": kind, "should_retrieve": should_retrieve,
            "gold_docs": list(gold), "is_final": is_final}

SESSIONS = [
    {"session_id": "battery_pen", "turns": [
        T("I need to know", "partial", False),
        T("What is the Galaxy S24 battery capacity and does it support the S Pen?", "fact", True, ["S24_Battery", "S24_Stylus"]),
        T("Okay, thanks", "ack", False),
        T("put that in bullet points", "cosmetic", False),
        T("what about the display refresh rate?", "refine", True, ["S24_Display"]),
    ]},
    {"session_id": "ultra", "turns": [
        T("How good is the S24 Ultra camera zoom?", "fact", True, ["Ultra_Camera"]),
        T("make it shorter", "cosmetic", False),
        T("and what about its S Pen?", "refine", True, ["Ultra_SPen"]),
        T("thanks", "ack", False),
    ]},
    {"session_id": "buds", "turns": [
        T("How long does the Galaxy Buds battery last?", "fact", True, ["Buds_Battery"]),
        T("does it have noise cancellation?", "refine", True, ["Buds_ANC"]),
        T("summarize that", "cosmetic", False),
    ]},
    {"session_id": "warranty", "turns": [
        T("What is the warranty period for the phone?", "fact", True, ["Warranty"]),
        T("can I return it if I don't like it?", "refine", True, ["Returns"]),
        T("rephrase that more simply", "cosmetic", False),
    ]},
    {"session_id": "watch", "turns": [
        T("Is the Galaxy Watch water resistant and how long is the battery life?", "fact", True, ["Watch_Water", "Watch_Battery"]),
        T("translate that", "cosmetic", False),
        T("what about sleep tracking?", "refine", True, ["Watch_Sleep"]),
    ]},
    {"session_id": "venue_stream", "turns": [
        {**T("I need a workshop venue in Pune", "fact", True, ["Venue_Pune"], is_final=False), "compound": False},
        {**T("Actually, the workshop is for 30 people and I need cancellation terms and catering options", "refine", True,
             ["Venue_Pune", "Venue_Cancellation", "Venue_Catering"]), "compound": True},
        T("put that in bullet points", "cosmetic", False),
    ]},
    {"session_id": "travel_late", "turns": [
        T("Summarize the travel reimbursement rule for an employee trip", "fact", True, ["Travel_Base"]),
        T("The trip was international and the booking was made after travel", "refine", True,
          ["Travel_International", "Travel_LateBooking"]),
        T("please repeat the answer in two bullets", "cosmetic", False),
    ]},
]
