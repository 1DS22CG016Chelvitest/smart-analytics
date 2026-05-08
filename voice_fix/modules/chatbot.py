import json
import re
from fuzzywuzzy import fuzz
import pandas as pd

# ── Load FAQ ────────────────────────────────────────────────────
with open('data/chatbot.json', 'r', encoding='utf-8') as f:
    raw = json.load(f)

faq_pairs = []
if isinstance(raw, dict) and 'questions' in raw:
    for item in raw['questions']:
        faq_pairs.append({
            'question': str(item.get('question', '')),
            'answer':   str(item.get('answer', ''))
        })
print(f"FAQ loaded: {len(faq_pairs)} pairs")

# ── Load Products ───────────────────────────────────────────────
df = pd.read_csv('data/products.csv')

CAT_COL   = 'Category'
PRICE_COL = 'Price (Rs.)'
FINAL_COL = 'Final_Price(Rs.)'
DISC_COL  = 'Discount (%)'
PAY_COL   = 'Payment_Method'

catalog = df.groupby(CAT_COL).agg(
    avg_price    = (PRICE_COL, 'mean'),
    min_price    = (PRICE_COL, 'min'),
    max_price    = (PRICE_COL, 'max'),
    avg_discount = (DISC_COL,  'mean'),
    count        = (PRICE_COL, 'count')
).reset_index().round(2)

categories = df[CAT_COL].dropna().unique().tolist()
print(f"Categories: {categories}")

# ── Store contact details ───────────────────────────────────────
STORE_NAME    = "Smart Analytics Store"
PHONE         = "+91-98765-43210"
WHATSAPP      = "+91-91234-56789"
EMAIL         = "support@smartanalytics.in"
SALES_EMAIL   = "sales@smartanalytics.in"
WORKING_HOURS = "Monday to Saturday, 9:00 AM to 6:00 PM IST"

# ── FAQ search ──────────────────────────────────────────────────
def search_faq(user_input):
    best_score, best_answer = 0, None
    for pair in faq_pairs:
        score = fuzz.partial_ratio(
            user_input.lower(),
            pair['question'].lower()
        )
        if score > best_score:
            best_score  = score
            best_answer = pair['answer']
    return best_answer if best_score > 50 else None

# ── Category summary ────────────────────────────────────────────
def category_summary(cat):
    sub = df[df[CAT_COL] == cat]
    return (
        f"Here is what we have for **{cat}**:\n\n"
        f"- Total items: {len(sub)}\n"
        f"- Price range: "
        f"₹{sub[PRICE_COL].min()} – "
        f"₹{sub[PRICE_COL].max()}\n"
        f"- Average price: "
        f"₹{round(sub[PRICE_COL].mean(), 2)}\n"
        f"- Average discount: "
        f"{round(sub[DISC_COL].mean(), 1)}%\n\n"
        f"Want to know cheapest or best deals "
        f"in {cat}?"
    )

# ── Recommendations ─────────────────────────────────────────────
def get_recommendations():
    top = catalog.sort_values(
        'avg_discount', ascending=False).head(5)
    reply = "Our best deals right now:\n\n"
    for _, row in top.iterrows():
        reply += (
            f"- **{row[CAT_COL]}** — "
            f"avg ₹{row['avg_price']} "
            f"with {row['avg_discount']}% discount\n"
        )
    return reply

# ── Contact info response ───────────────────────────────────────
def get_contact_info():
    return (
        f"Here are our contact details:\n\n"
        f"- Phone: {PHONE}\n"
        f"- WhatsApp: {WHATSAPP}\n"
        f"- Email: {EMAIL}\n"
        f"- Sales enquiry: {SALES_EMAIL}\n"
        f"- Working hours: {WORKING_HOURS}\n\n"
        f"For damaged or missing items, "
        f"email us within 48 hours of delivery "
        f"with your order details and a photo."
    )

# ── Exact keyword match helper ──────────────────────────────────
def has_word(text, words):
    for w in words:
        pattern = r'\b' + re.escape(w) + r'\b'
        if re.search(pattern, text):
            return True
    return False

# ── Main chatbot function ───────────────────────────────────────
def chatbot_response(user_input):
    if not user_input.strip():
        return "Please type something!"

    u = user_input.lower().strip()

    # ── 1. Greeting ─────────────────────────────────────────────
    if has_word(u, ['hi', 'hello', 'hey',
                    'good morning', 'good evening']):
        return (
            f"Hello! Welcome to {STORE_NAME}! 👋\n\n"
            f"I can help you with:\n"
            f"- Product categories and prices\n"
            f"- Discounts and best deals\n"
            f"- Store policies and FAQs\n"
            f"- Price comparisons\n\n"
            f"We sell: {', '.join(categories)}\n\n"
            f"What are you looking for?"
        )

    # ── 2. Goodbye ──────────────────────────────────────────────
    if has_word(u, ['bye', 'goodbye',
                    'thank you', 'thanks']):
        return (
            f"Thank you for visiting "
            f"{STORE_NAME}! "
            f"Have a great day! 😊\n\n"
            f"For any help, reach us at "
            f"{PHONE} or {EMAIL}."
        )

    # ── 3. Direct contact queries ────────────────────────────────
    contact_triggers = [
        'contact', 'phone number', 'call us',
        'whatsapp', 'email', 'email id',
        'email address', 'reach you',
        'reach us', 'get in touch',
        'customer care', 'helpline',
        'contact number', 'support number',
        'contact details', 'contact info'
    ]
    if any(kw in u for kw in contact_triggers):
        return get_contact_info()

    # ── 4. Working hours ─────────────────────────────────────────
    if any(w in u for w in [
            'working hours', 'open', 'timing',
            'when are you open', 'office hours',
            'available time', 'business hours']):
        return (
            f"We are available "
            f"{WORKING_HOURS}.\n\n"
            f"Outside working hours you can:\n"
            f"- Email us at {EMAIL}\n"
            f"- WhatsApp us at {WHATSAPP}\n"
            f"We respond within 24 hours."
        )

    # ── 5. Damaged / missing product ─────────────────────────────
    if any(w in u for w in [
            'damaged', 'broken', 'defective',
            'wrong product', 'missing item',
            'not received', 'missing',
            'wrong order', 'incomplete']):
        return (
            f"We are sorry to hear that! 😟\n\n"
            f"Please follow these steps:\n"
            f"1. Take a clear photo of the item\n"
            f"2. Email us at {EMAIL} within "
            f"48 hours of delivery\n"
            f"3. Or call us at {PHONE}\n\n"
            f"We will arrange a free replacement "
            f"or full refund immediately."
        )

    # ── 6. Bulk order query ──────────────────────────────────────
    if any(w in u for w in [
            'bulk order', 'bulk purchase',
            'wholesale', 'large quantity',
            'business order', 'corporate order']):
        return (
            f"Yes, we accept bulk orders! 📦\n\n"
            f"For bulk orders and special pricing:\n"
            f"- Email: {SALES_EMAIL}\n"
            f"- Phone: {PHONE}\n"
            f"- WhatsApp: {WHATSAPP}\n\n"
            f"Our sales team is available "
            f"{WORKING_HOURS}."
        )

    # ── 7. FAQ — policy / support questions ──────────────────────
    faq_triggers = [
        'policy', 'policies', 'return', 'refund',
        'delivery', 'shipping', 'track', 'tracking',
        'cancel', 'cancellation', 'exchange',
        'warranty', 'support',
        'create an account', 'register',
        'forgot password', 'reset password',
        'login', 'sign up', 'sign in',
        'how do i', 'how can i', 'can i',
        'what is your', 'do you accept',
        'tell me your', 'ur policies',
        'your policies', 'help me',
        'problem with', 'issue with',
        'payment issue', 'coupon', 'promo',
        'cod', 'cash on delivery',
        'international', 'delete account',
        'privacy', 'data safe', 'personal info',
        'update address', 'change address'
    ]
    is_faq = any(kw in u for kw in faq_triggers)
    if is_faq:
        answer = search_faq(user_input)
        if answer:
            return answer

    # ── 8. All categories ────────────────────────────────────────
    if any(w in u for w in [
            'categories', 'what do you sell',
            'what do you have', 'all products',
            'show all', 'what categories',
            'what items', 'what products',
            'list products', 'list categories']):
        reply = (
            "We sell products in these "
            "categories:\n\n"
        )
        for _, row in catalog.iterrows():
            reply += (
                f"- **{row[CAT_COL]}** — "
                f"{int(row['count'])} items, "
                f"₹{row['min_price']} to "
                f"₹{row['max_price']}, "
                f"avg {row['avg_discount']}% off\n"
            )
        return reply

    # ── 9. Show specific category ────────────────────────────────
    for cat in categories:
        if cat.lower() in u and any(
                w in u for w in [
                    'show', 'tell', 'give',
                    'about', 'products', 'items']):
            return category_summary(cat)

    # ── 10. Cheapest products ────────────────────────────────────
    if any(w in u for w in [
            'cheapest', 'lowest price',
            'most affordable', 'cheap',
            'budget', 'low price',
            'minimum price', 'least expensive']):
        for cat in categories:
            if cat.lower() in u:
                sub  = df[df[CAT_COL] == cat]
                rows = sub.nsmallest(5, PRICE_COL)
                reply = (
                    f"Cheapest {cat} "
                    f"products:\n\n"
                )
                for i, (_, r) in enumerate(
                        rows.iterrows()):
                    reply += (
                        f"{i+1}. Category: {cat}"
                        f" | Price: ₹{r[PRICE_COL]}"
                        f" | Discount: "
                        f"{r[DISC_COL]}% off"
                        f" | You pay: "
                        f"₹{r[FINAL_COL]}\n"
                    )
                return reply
        rows  = df.nsmallest(5, PRICE_COL)
        reply = "Most affordable items overall:\n\n"
        for i, (_, r) in enumerate(rows.iterrows()):
            reply += (
                f"{i+1}. Category: {r[CAT_COL]}"
                f" | Price: ₹{r[PRICE_COL]}"
                f" | Discount: {r[DISC_COL]}% off"
                f" | You pay: ₹{r[FINAL_COL]}\n"
            )
        return reply

    # ── 11. Most expensive / premium ─────────────────────────────
    if any(w in u for w in [
            'expensive', 'premium', 'luxury',
            'highest price', 'most costly',
            'top price', 'costliest']):
        for cat in categories:
            if cat.lower() in u:
                sub  = df[df[CAT_COL] == cat]
                rows = sub.nlargest(5, PRICE_COL)
                reply = f"Premium {cat} products:\n\n"
                for i, (_, r) in enumerate(
                        rows.iterrows()):
                    reply += (
                        f"{i+1}. Category: {cat}"
                        f" | Price: ₹{r[PRICE_COL]}"
                        f" | Discount: "
                        f"{r[DISC_COL]}% off\n"
                    )
                return reply
        rows  = df.nlargest(5, PRICE_COL)
        reply = "Our most premium products:\n\n"
        for i, (_, r) in enumerate(rows.iterrows()):
            reply += (
                f"{i+1}. Category: {r[CAT_COL]}"
                f" | Price: ₹{r[PRICE_COL]}\n"
            )
        return reply

    # ── 12. Best deals / discount ────────────────────────────────
    if any(w in u for w in [
            'best deals', 'top deals',
            'best offer', 'recommend',
            'suggest', 'popular deals',
            'deal', 'offer', 'discount',
            'sale', 'saving',
            'maximum discount',
            'biggest discount']):
        return get_recommendations()

    # ── 13. Price range between X and Y ─────────────────────────
    if 'between' in u or (
            'range' in u and 'price' in u):
        nums = re.findall(r'\d+', u)
        if len(nums) >= 2:
            low   = int(nums[0])
            high  = int(nums[1])
            results = df[
                (df[PRICE_COL] >= low) &
                (df[PRICE_COL] <= high)
            ]
            if len(results) == 0:
                return (
                    f"No products found between "
                    f"₹{low} and ₹{high}.\n\n"
                    f"Our price range is "
                    f"₹{df[PRICE_COL].min()} to "
                    f"₹{df[PRICE_COL].max()}.\n"
                    f"Try within that range."
                )
            reply = (
                f"Products between "
                f"₹{low} and ₹{high} "
                f"({len(results)} items):\n\n"
            )
            summary = results.groupby(
                CAT_COL)[PRICE_COL].count()
            for cat, cnt in summary.items():
                avg = results[
                    results[CAT_COL] == cat
                ][PRICE_COL].mean().round(2)
                reply += (
                    f"- {cat}: {cnt} items "
                    f"(avg ₹{avg})\n"
                )
            return reply

    # ── 14. Products under X price ───────────────────────────────
    if 'under' in u:
        nums = re.findall(r'\d+', u)
        if nums:
            limit   = int(nums[0])
            results = df[df[PRICE_COL] <= limit]
            if len(results) == 0:
                return (
                    f"No products found "
                    f"under ₹{limit}.\n\n"
                    f"Our minimum price is "
                    f"₹{df[PRICE_COL].min()}."
                )
            reply = (
                f"Products under ₹{limit} "
                f"({len(results)} items):\n\n"
            )
            summary = results.groupby(
                CAT_COL)[PRICE_COL].count()
            for cat, cnt in summary.items():
                minp = results[
                    results[CAT_COL] == cat
                ][PRICE_COL].min()
                reply += (
                    f"- {cat}: {cnt} items "
                    f"(from ₹{minp})\n"
                )
            return reply

    # ── 15. Payment methods ──────────────────────────────────────
    if any(w in u for w in [
            'payment method', 'payment option',
            'payment mode', 'accepted payment',
            'pay using', 'pay by',
            'upi', 'net banking',
            'debit card', 'credit card']):
        methods = df[PAY_COL].value_counts()
        reply   = "Payment methods we accept:\n\n"
        for method in methods.index:
            reply += f"- {method}\n"
        reply += (
            f"\nFor payment issues contact us:\n"
            f"Phone: {PHONE}\n"
            f"Email: {EMAIL}"
        )
        return reply

    # ── 16. Store summary / statistics ───────────────────────────
    if any(w in u for w in [
            'store summary', 'store overview',
            'store info', 'store statistics',
            'about store', 'about the store',
            'total products', 'how many products',
            'store details']):
        top_cat = df[CAT_COL].value_counts().index[0]
        return (
            f"Store Overview — {STORE_NAME}:\n\n"
            f"- Total products: {len(df)}\n"
            f"- Categories: {len(categories)}\n"
            f"- Average price: "
            f"₹{round(df[PRICE_COL].mean(), 2)}\n"
            f"- Average discount: "
            f"{round(df[DISC_COL].mean(), 1)}%\n"
            f"- Most popular category: {top_cat}\n"
            f"- Price range: "
            f"₹{df[PRICE_COL].min()} to "
            f"₹{df[PRICE_COL].max()}\n\n"
            f"Contact us: {PHONE} | {EMAIL}"
        )

    # ── 17. Compare two categories ───────────────────────────────
    if 'compare' in u or ' vs ' in u:
        matched = [
            cat for cat in categories
            if cat.lower() in u
        ]
        if len(matched) >= 2:
            reply = (
                f"Comparing "
                f"**{matched[0]}** vs "
                f"**{matched[1]}**:\n\n"
            )
            for cat in matched[:2]:
                sub = df[df[CAT_COL] == cat]
                reply += (
                    f"**{cat}:**\n"
                    f"- Items: {len(sub)}\n"
                    f"- Avg price: "
                    f"₹{round(sub[PRICE_COL].mean(), 2)}\n"
                    f"- Avg discount: "
                    f"{round(sub[DISC_COL].mean(), 1)}%\n"
                    f"- Price range: "
                    f"₹{sub[PRICE_COL].min()} – "
                    f"₹{sub[PRICE_COL].max()}\n\n"
                )
            return reply
        return (
            "Please mention two categories "
            "to compare.\n"
            f"Available: {', '.join(categories)}"
        )

    # ── 18. Most popular / trending ──────────────────────────────
    if any(w in u for w in [
            'popular', 'trending',
            'most bought', 'top selling',
            'most sold', 'most purchased']):
        top   = df[CAT_COL].value_counts()
        reply = "Most popular categories:\n\n"
        for i, (cat, cnt) in enumerate(
                top.head(5).items()):
            reply += (
                f"{i+1}. {cat} "
                f"— {cnt} purchases\n"
            )
        return reply

    # ── 19. Average price ────────────────────────────────────────
    if any(w in u for w in [
            'average price', 'avg price',
            'mean price', 'typical price']):
        for cat in categories:
            if cat.lower() in u:
                sub = df[df[CAT_COL] == cat]
                avg = round(sub[PRICE_COL].mean(), 2)
                return (
                    f"Average price of "
                    f"**{cat}**: ₹{avg}\n\n"
                    f"Range: "
                    f"₹{sub[PRICE_COL].min()} to "
                    f"₹{sub[PRICE_COL].max()}"
                )
        avg = round(df[PRICE_COL].mean(), 2)
        return (
            f"Average price across all "
            f"products: ₹{avg}"
        )

    # ── 20. Default fallback ─────────────────────────────────────
    return (
        "I didn't quite understand that.\n\n"
        "Here are things you can ask me:\n\n"
        "- 'show me electronics'\n"
        "- 'cheapest products'\n"
        "- 'best deals'\n"
        "- 'products under 300'\n"
        "- 'products between 100 and 300'\n"
        "- 'compare sports vs clothing'\n"
        "- 'store summary'\n"
        "- 'payment methods'\n"
        "- 'how do I track my order'\n"
        "- 'contact details'\n"
        "- 'tell me your policies'\n\n"
        f"Or call us directly: {PHONE}"
    )
