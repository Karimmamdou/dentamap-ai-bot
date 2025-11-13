import os
import pandas as pd
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.constants import ParseMode
from telegram.ext import (
    ApplicationBuilder,
    ContextTypes,
    MessageHandler,
    filters,
)
from openai import OpenAI

# -------------------------------------------------------------------
# ENVIRONMENT VARIABLES
# -------------------------------------------------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

client = OpenAI(api_key=OPENAI_API_KEY)

# -------------------------------------------------------------------
# LOAD PRODUCT DATABASE (Excel)
# -------------------------------------------------------------------
df = pd.read_excel("products.xlsx")

# Normalize text fields for better matching
df["product_name"] = df["product_name"].astype(str)
df["brand"] = df["brand"].astype(str)
df["keywords"] = df["keywords"].astype(str)
df["distributor"] = df["distributor"].astype(str)


# -------------------------------------------------------------------
# GPT-4o MINI (only used for unclear queries)
# -------------------------------------------------------------------
async def ask_gpt(query_text):
    """Use GPT-4o Mini to refine ambiguous or unclear questions."""
    system_prompt = (
        "You are DentaMap AI. Your job is to rewrite unclear or vague user questions "
        "into clean, structured dental product search queries. "
        "Correct spelling, extract product-related meaning, and keep content short. "
        "Arabic answers only unless user writes English."
    )

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": query_text}
        ],
        max_tokens=200,
        temperature=0.2
    )

    return response.choices[0].message.content


# -------------------------------------------------------------------
# Detect if GPT-4o Mini is needed
# -------------------------------------------------------------------
def needs_ai_help(text: str) -> bool:
    t = text.strip().lower()

    # too short
    if len(t) < 4:
        return True

    unclear_words = [
        "قارن", "مقارنة", "افضل", "أفضل", "احسن",
        "variant", "quality", "compare"
    ]

    if any(w in t for w in unclear_words):
        return True

    keywords = [
        "composite", "bond", "sealer", "etch", "edta", "gutta",
        "paper", "niti", "file", "brackets", "ortho", "flow",
        "resin", "adhesive"
    ]

    if not any(k in t for k in keywords):
        return True

    return False


# -------------------------------------------------------------------
# Product Search System (Excel Based)
# -------------------------------------------------------------------
def rank_products(query, topn=5):
    q = query.lower()
    scores = []

    for idx, row in df.iterrows():
        text = f"{row['product_name']} {row['brand']} {row['keywords']}".lower()
        score = text.count(q)
        if score > 0:
            scores.append((idx, score))

    scores.sort(key=lambda x: x[1], reverse=True)

    if not scores:
        return pd.DataFrame()

    top_idx = [s[0] for s in scores[:topn]]
    return df.loc[top_idx]


def alt_candidates(main_row, topk=3):
    brand = main_row["brand"]
    mask = (df["brand"] != brand)
    tmp = df[mask].sample(min(topk, len(df[mask])))
    return tmp


# -------------------------------------------------------------------
# Format product card
# -------------------------------------------------------------------
def format_card(row):
    name = row["product_name"]
    brand = row["brand"]
    price = row.get("price_sar", "—")
    dist = row.get("distributor", "—")
    phone = row.get("phone", "—")
    origin = row.get("origin", "—")

    txt = (
        f"🦷 **{name}**\n"
        f"🏷 **البراند:** {brand}\n"
        f"💰 **السعر:** {price} ريال\n"
        f"🌍 **بلد المنشأ:** {origin}\n"
        f"🏢 **الموزع المعتمد:** {dist}\n"
        f"📞 **رقم التواصل:** {phone}\n\n"
        "⚠️ الأسعار قابلة للتغير — يُرجى التواصل مع الموزع للحصول على أفضل سعر.\n"
    )
    return txt


# -------------------------------------------------------------------
# Keyboard
# -------------------------------------------------------------------
def main_keyboard(name):
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("🔄 بدائل", callback_data=f"alt:{name}")],
        [InlineKeyboardButton("🔥 عروض", callback_data="offers")],
    ])


# -------------------------------------------------------------------
# EVENTS — Incoming Messages
# -------------------------------------------------------------------
async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.message.text.strip()

    # decide if we need GPT help
    if needs_ai_help(q):
        refined = await ask_gpt(q)
        q = refined

    # search excel
    results = rank_products(q, topn=5)

    if results.empty:
        await update.message.reply_text(
            "⚠️ لم يتم العثور على المنتج. يرجى كتابة الاسم أو جزء منه."
        )
        return

    first = results.iloc[0]
    caption = format_card(first)
    img = str(first.get("image_url", "")).strip()

    kb = main_keyboard(first["product_name"])

    if img.lower().endswith((".jpg", ".jpeg", ".png")):
        try:
            await update.message.reply_photo(
                photo=img,
                caption=caption,
                parse_mode=ParseMode.MARKDOWN,
                reply_markup=kb
            )
        except:
            await update.message.reply_text(caption, parse_mode=ParseMode.MARKDOWN, reply_markup=kb)
    else:
        await update.message.reply_text(caption, parse_mode=ParseMode.MARKDOWN, reply_markup=kb)

    # send alternatives
    alts = alt_candidates(first, topk=2)
    if not alts.empty:
        lines = []
        for _, r in alts.iterrows():
            lines.append(f"• {r['product_name']} — {r['brand']} — {r['price_sar']} ريال")

        await update.message.reply_text(
            "✨ **بدائل مقترحة:**\n" + "\n".join(lines),
            parse_mode=ParseMode.MARKDOWN
        )


# -------------------------------------------------------------------
# BOT LAUNCHER
# -------------------------------------------------------------------
async def main():
    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
    print("🚀 Bot is running on Render 24/7...")
    await app.run_polling()


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
