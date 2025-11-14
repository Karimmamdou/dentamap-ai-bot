import os
import csv
import random

from telegram import (
    Update,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
)
from telegram.constants import ParseMode
from telegram.ext import (
    ApplicationBuilder,
    ContextTypes,
    MessageHandler,
    CallbackQueryHandler,
    filters,
)

from openai import OpenAI

# ------------------- ENV VARIABLES -------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

client = OpenAI(api_key=OPENAI_API_KEY)

# ------------------- LOAD PRODUCTS CSV -------------------
PRODUCTS = []  # list of dicts


def load_products():
    """Load products from products.csv into memory."""
    global PRODUCTS
    PRODUCTS = []
    csv_path = "products.csv"
    if not os.path.exists(csv_path):
        print(f"❌ products.csv not found in working directory.")
        return

    with open(csv_path, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Ensure we have clean strings
            clean_row = {k: (v or "").strip() for k, v in row.items()}
            PRODUCTS.append(clean_row)

    print(f"✅ Loaded {len(PRODUCTS)} products from products.csv")


load_products()

# ------------------- GPT-4o MINI (HYBRID) -------------------


async def ask_gpt(query_text: str) -> str:
    """
    Use GPT-4o Mini to refine / rewrite unclear dental questions
    into a clean product search query (Arabic if input is Arabic).
    """
    system_prompt = (
        "You are DentaMap AI. Your job is to rewrite unclear or vague dentist "
        "questions into short, clear dental product search queries. "
        "Correct spelling, keep only important product-related words, and reply "
        "in Arabic if the user writes Arabic. Keep the output short."
    )

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query_text},
        ],
        max_tokens=150,
        temperature=0.2,
    )
    return resp.choices[0].message.content


# -------- WHEN TO USE GPT (HYBRID DECISION) --------


def needs_ai_help(text: str) -> bool:
    """
    Decide if we should call GPT-4o Mini to clarify the query.
    """
    t = (text or "").strip().lower()

    # Very short / unclear
    if len(t) < 4:
        return True

    # words that imply comparison / vague request
    unclear_words = [
        "قارن",
        "مقارنة",
        "افضل",
        "أفضل",
        "احسن",
        "variant",
        "quality",
        "compare",
        "ايه رأيك",
        "اختار لي",
    ]
    if any(w in t for w in unclear_words):
        return True

    # presence of dental keywords = probably clear enough
    keywords = [
        "composite",
        "bond",
        "sealer",
        "etch",
        "edta",
        "gutta",
        "paper",
        "niti",
        "file",
        "brackets",
        "ortho",
        "flow",
        "resin",
        "adhesive",
    ]
    if not any(k in t for k in keywords):
        # no strong product keyword → ask GPT to clarify
        return True

    return False


# ------------------- SEARCH IN PRODUCTS -------------------


def text_score_for_query(text: str, query: str) -> int:
    """
    Very simple scoring: count occurrences of each token of query in text.
    """
    t = text.lower()
    q = query.lower().strip()
    if not q:
        return 0

    score = 0
    tokens = [w for w in q.split() if w]
    for tok in tokens:
        if tok in t:
            score += 1
    return score


def rank_products(query: str, topn: int = 5):
    """
    Rank products by simple textual similarity to the query.
    Uses product_name + brand + Keywords.
    """
    scored = []
    for row in PRODUCTS:
        combined_text = (
            f"{row.get('product_name','')} "
            f"{row.get('brand','')} "
            f"{row.get('Keywords','')}"
        )
        score = text_score_for_query(combined_text, query)
        if score > 0:
            scored.append((row, score))

    scored.sort(key=lambda x: x[1], reverse=True)
    return [r for r, _ in scored[:topn]]


def alt_candidates(main_row: dict, topk: int = 3):
    """
    Suggest alternatives:
    - Prefer same Material Type
    - Else same brand
    - Else random
    """
    mt = main_row.get("Material Type", "").strip()
    brand = main_row.get("brand", "").strip()

    # 1) same Material Type
    same_type = [
        r
        for r in PRODUCTS
        if r is not main_row and r.get("Material Type", "").strip() == mt and mt
    ]
    if len(same_type) >= topk:
        return random.sample(same_type, topk)

    # 2) same brand
    same_brand = [
        r for r in PRODUCTS if r is not main_row and r.get("brand", "").strip() == brand
    ]
    pool = same_type + same_brand
    pool = list({id(r): r for r in pool}.values())  # unique

    # 3) fallback: any
    if len(pool) < topk:
        others = [r for r in PRODUCTS if r is not main_row and r not in pool]
        pool += others

    if not pool:
        return []

    if len(pool) <= topk:
        return pool
    return random.sample(pool, topk)


# ------------------- MESSAGE FORMATTING -------------------


def format_card(row: dict) -> str:
    name = row.get("product_name", "—")
    brand = row.get("brand", "—")
    price = row.get("price_sar", "—")
    dist = row.get("distributor_name", "—")

    # نختار رقم الموزع من distributor_phone أو Contact Number
    phone = row.get("distributor_phone", "").strip()
    if not phone:
        phone = row.get("Contact Number", "").strip()
    if not phone:
        phone = "—"

    mat_type = row.get("Material Type", "").strip()

    lines = [
        f"🦷 **{name}**",
        f"🏷 **البراند:** {brand}",
    ]

    if mat_type:
        lines.append(f"🧪 **الفئة / النوع:** {mat_type}")

    lines.extend(
        [
            f"💰 **السعر التقريبي:** {price} ريال",
            f"🏢 **الموزع المعتمد:** {dist}",
            f"📞 **رقم التواصل:** {phone}",
            "",
            "⚠️ الأسعار قابلة للتغيّر — يُرجى التواصل مع الموزع للحصول على أفضل سعر متاح في السوق.",
        ]
    )

    return "\n".join(lines)


def main_keyboard(product_name: str):
    """
    Inline keyboard with 'Alternatives' button.
    """
    return InlineKeyboardMarkup(
        [
            [InlineKeyboardButton("🔄 عرض بدائل مشابهة", callback_data=f"alts::{product_name}")]
        ]
    )


# ------------------- TELEGRAM HANDLERS -------------------


async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = (update.message.text or "").strip()
    if not text:
        return

    # 1) HYBRID: decide if we need GPT-4o mini
    if needs_ai_help(text):
        try:
            refined = await ask_gpt(text)
            query = refined.strip() or text
        except Exception as e:
            print(f"GPT error: {e}")
            query = text
    else:
        query = text

    # 2) search products
    results = rank_products(query, topn=5)

    if not results:
        await update.message.reply_text(
            "⚠️ لم يتم العثور على المنتج المطلوب.\n"
            "جرّب كتابة اسم المنتج أو جزء منه فقط.",
            parse_mode=ParseMode.MARKDOWN,
        )
        return

    first = results[0]
    caption = format_card(first)
    img = (first.get("image_url") or "").strip()
    kb = main_keyboard(first.get("product_name", ""))

    # send main product with photo if available
    if img.lower().endswith((".jpg", ".jpeg", ".png")):
        try:
            await update.message.reply_photo(
                photo=img,
                caption=caption,
                parse_mode=ParseMode.MARKDOWN,
                reply_markup=kb,
            )
        except Exception as e:
            print(f"Photo send error: {e}")
            await update.message.reply_text(
                caption, parse_mode=ParseMode.MARKDOWN, reply_markup=kb
            )
    else:
        await update.message.reply_text(
            caption, parse_mode=ParseMode.MARKDOWN, reply_markup=kb
        )

    # send quick alternatives list as text
    alts = alt_candidates(first, topk=2)
    if alts:
        lines = []
        for r in alts:
            lines.append(
                f"• {r.get('product_name','')} — {r.get('brand','')} — {r.get('price_sar','')} ريال"
            )
        await update.message.reply_text(
            "✨ **بدائل مقترحة:**\n" + "\n".join(lines),
            parse_mode=ParseMode.MARKDOWN,
        )


async def handle_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()

    data = query.data or ""
    if data.startswith("alts::"):
        name = data.split("::", 1)[1].strip().lower()
        target = None
        for r in PRODUCTS:
            if r.get("product_name", "").strip().lower() == name:
                target = r
                break

        if not target:
            await query.edit_message_reply_markup(reply_markup=None)
            await query.message.reply_text(
                "⚠️ لم يمكن العثور على هذا المنتج لعرض بدائل.",
                parse_mode=ParseMode.MARKDOWN,
            )
            return

        alts = alt_candidates(target, topk=3)
        if not alts:
            await query.message.reply_text(
                "لا توجد بدائل متاحة حاليًا لهذا المنتج.",
                parse_mode=ParseMode.MARKDOWN,
            )
            return

        lines = []
        for r in alts:
            lines.append(
                f"• {r.get('product_name','')} — {r.get('brand','')} — {r.get('price_sar','')} ريال"
            )
        await query.message.reply_text(
            "🔄 بدائل مشابهة:\n" + "\n".join(lines),
            parse_mode=ParseMode.MARKDOWN,
        )


# ------------------- MAIN ENTRY -------------------


# ------------------- MAIN ENTRY -------------------


def main():
    if not TELEGRAM_TOKEN:
        print("❌ TELEGRAM_BOT_TOKEN is missing in environment.")
        return

    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()

    # Handlers
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
    app.add_handler(CallbackQueryHandler(handle_callback))

    print("🚀 DentaMap AI bot is running on Render (CSV + HYBRID GPT-4o mini)...")
    app.run_polling()


if __name__ == "__main__":
    main()
