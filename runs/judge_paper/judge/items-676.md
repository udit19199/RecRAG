Prompt version 2026-11-10.haiku.2

You check whether retrieved text states the facts of gold evidence sentences.

Use only the retrieved items. Do not use outside knowledge and do not reason
your way to a fact the text does not state; plain rewording, synonyms,
abbreviations and different formatting are fine.

For every fact give one status:
- stated: some item says it, with every specific matching (names, numbers,
  dates, places, who did what, qualifiers).
- contradicted: some item gives a different value for the same specific, such
  as another year or another person.
- missing: neither of the above.
If one item states the fact and another contradicts it, the status is contradicted.

Evidence for stated or contradicted facts: copy one continuous passage from
one item, character for character, with no ellipses or edits, and give that
item's number. Never join separate words into a quote that does not appear as
written; if the words are apart, quote the longer continuous passage that
contains them. A fact spread across items may take several quotes. The quotes
together must contain the words that state every specific of the fact (who,
what, numbers, dates, reasons); a quote that only sets the topic, or stops
before the specific, is not evidence. Missing facts take no evidence. Every quote is checked against the item text by a
program, and a quote that does not appear there counts as missing.

Write the verdict file named in your task as a JSON array with one object per
item, in any order:
{"id": "<item id>", "facts": [{"fact": "S.F", "status": "stated", "evidence": [{"item": 3, "quote": "..."}]}, ...]}
where "S.F" is the fact's label as given (sentence number, fact number).
Give exactly one entry for every fact label of the item. Use only ids from
this file and do not skip any item.

This file has 4 items.

=== ITEM id=b9a0c63e72592c49 ===
Gold sentences and facts:
Sentence 0: Cardholders can get 20% back as a statement credit when they purchase eligible pre-purchased meals, in-flight food purchases, alcoholic beverages, and audio headsets on Delta flights.
  Fact 0.0: Cardholders can get 20% back as a statement credit.
  Fact 0.1: Cardholders can get 20% back when they purchase eligible pre-purchased meals on Delta flights.
  Fact 0.2: Cardholders can get 20% back when they purchase eligible in-flight food purchases on Delta flights.
  Fact 0.3: Cardholders can get 20% back when they purchase eligible alcoholic beverages on Delta flights.
  Fact 0.4: Cardholders can get 20% back when they purchase eligible audio headsets on Delta flights.
Sentence 1: And for Black Friday, it's on sale for over $400 off, knocking the price down to a super impressive $297.
  Fact 1.0: The product is on sale for Black Friday.
  Fact 1.1: The product is on sale for over $400 off.
  Fact 1.2: The product's sale price is $297.
  Fact 1.3: The $297 price is described as super impressive.

Retrieved items:
<item 1>
200+ of the best deals from Amazon's Cyber Monday sale (Mashable, 2023-11-27T08:45:59+00:00)
Table of Contents Table of Contents Echo, Fire TV, and Kindle deals Apple deals TV deals Laptop deals Headphone and earbud deals Tablet deals Gaming deals
Speaker deals Vacuum deals Kitchen deals Smart home deals Fitness deals Beauty tech deals Drone deals Camera deals Lego deals Gift card deals

UPDATE: Nov. 27, 2023, 5:00 a.m. EST This post has been updated with all of the latest Cyber Monday deals available at Amazon.

Amazon is dragging out the year's biggest shopping holiday(s) into 11 days of deals.

The retail giant began its Black Friday sale in the early morning of Friday, Nov. 17 (a week ahead of schedule) and was on top of making the switch to Cyber
Monday language in the wee hours of Saturday, Nov. 25. Official Cyber Monday mode, which is currently on through Monday, Nov. 27, includes both a ton of deals
carried over from Black Friday plus some new ones.
</item 1>

<item 2>
The best Black Friday deals you can get for under $25 (The Verge, 2023-11-23T17:58:09+00:00)
Come on in and peruse these affordable wares.

Phone accessories and smart trackers

Apple AirTag $ 24 $ 29 17 % off $ 24 $ 24 $ 29 17 % off Apple’s AirTag can help you find your lost items with its ultra wideband technology. You’ll get the best
compatibility with an iPhone, though Apple released an Android app that can detect an AirTag’s location and notify you if one seems to be following you. Read
our review. $24 at Amazon$80 at Amazon (four-pack)

Anker 511 Charger (30W) $ 16 $ 23 30 % off $ 16 $ 16 $ 23 30 % off Anker’s 30W 511 Charger juices phones quickly and is small enough to carry around in a purse.
</item 2>

<item 3>
The best Black Friday deals you can get for under $25 (The Verge, 2023-11-23T17:58:09+00:00)
It doesn’t have Matter or energy monitoring, but that’s alright at this price. $18 at Amazon (two-pack, with on-page coupon)

Amazon Smart Plug $ 15 $ 25 40 % off $ 15 $ 15 $ 25 40 % off Amazon’s apt-titled Smart Plug doesn’t require a hub and lets you add Alexa functionality to any
outlet, allowing you to control a range of devices with just your voice. $15 at Amazon$15 at Best Buy

Amazon Echo Glow $ 17 $ 30 43 % off $ 17 $ 17 $ 30 43 % off The Amazon Echo Glow is a multicolored smart lamp for children, one that can cycle through colors
and function as an Alexa-enabled sleep timer if needed. $17 at Amazon

TP-Link’s RE315 Wi-Fi extender is on sale for just $24.99 ($25 off), matching its best price to date.
</item 3>

<item 4>
Some of our favorite devices are on sale for Black Friday. (The Verge, 2023-11-24T15:42:21+00:00)
I’ve had my Slice Micro for a few months, so I’m not worried about it dulling yet, but it is unclear if the blade is replaceable.

Headphone hanger

Kaitlin Hatton, audience manager

Anchor Pro under-desk headphone hanger $ 12 $ 15 20 % off $ 12 $ 12 $ 15 20 % off An under-desk dual headphone hanger that uses 3M adhesive for mounting and an
included Velcro strap to anchor a headphone cable. $12 at Amazon

I gave this a try for one of our TikTok videos, and it has not disappointed me yet. I use it to hold my Razer Kraken headset and some extra cords. It’s small
enough to remain out of the way but large enough to hold more than the headset itself. It has a pretty strong hold and doesn’t give, even as I raise my standing
desk up and down several times a day.
</item 4>

<item 5>
200+ of the best deals from Amazon's Cyber Monday sale (Mashable, 2023-11-27T08:45:59+00:00)
Beauty tech deals

Drone deals

Camera deals

GoPro

Lego deals

Gift card deals
</item 5>

<item 6>
Fly with Delta frequently? Small business owners can score a solid welcome bonus and other perks with the Delta SkyMiles Gold Business Card (Fortune,
2023-12-12T21:01:43+00:00)
Our take: If you’re a small business owner who flies with Delta regularly and wants to score a solid welcome bonus, consider the Delta SkyMiles® Gold Business
American Express Card. This card has a welcome bonus of up to 75,000 miles and perks like free checked bags and priority boarding with a $0 intro annual fee for
the first year (then $99).

Pros Free checked bag free on Delta flights

Generous welcome bonus of 75,000 miles

No foreign transaction fees Cons No lounge access

$99 annual fee ($0 intro fee for the first year)

To view rates and fees of the Delta SkyMiles® Gold Business American Express Card, see this page.
</item 6>

<item 7>
The best Apple Cyber Monday deals of 2023: Save on AirPods, iPads, Apple Watches and more (Engadget, 2023-11-27T10:00:30+00:00)
Just use the code APPLET5USA at checkout to see the offer. The deal applies to the digital version of the card; as of this writing, you can get the credit with
a physical Apple gift card as well by using the code APPLET5USAP, though you may not want to clip the coupon on those cards' product page, as Amazon's listing
says that will only give you a $10 credit. As a reminder, you can use an Apple gift card for App Store credit, subscription services like Apple Music or iCloud,
and/or purchases made at a physical Apple store.

Apple AirTags

A four-pack of Apple AirTags is $80 right now at Amazon, thanks to a 19 percent discount. They go for $29 each at full price, so the deal will save you $9 a
pop. AirTags are our top picks for Bluetooth trackers for iPhone users as they tap into Apple’s disturbingly vast FindMy network, using other Apple mobiles to
find your lost stuff.
</item 7>

<item 8>
Fly with Delta frequently? Small business owners can score a solid welcome bonus and other perks with the Delta SkyMiles Gold Business Card (Fortune,
2023-12-12T21:01:43+00:00)
Is Delta SkyMiles Gold Card hard to get?

Cardholders should aim for a ‘good’ FICO credit score of 690 or better to qualify for the Delta SkyMiles Gold Card, though card issuers look at other factors
like income, too.

Please note that card details are accurate as of the publish date, but are subject to change at any time at the discretion of the issuer. Please contact the
card issuer to verify rates, fees, and benefits before applying.

Terms apply to American Express benefits and offers. Enrollment may be required for select American Express benefits and offers. Visit americanexpress.com to
learn more.

Eligibility and Benefit level varies by Card. Terms, Conditions, and Limitations Apply. Please visit americanexpress.com/benefitsguide for more details.
Underwritten by Amex Assurance Company.
</item 8>

<item 9>
Adobe: Thanksgiving US online sales nudge up to $5.6B; Salesforce: $31.7B spent globally (TechCrunch, 2023-11-24T15:43:51+00:00)
Salesforce’s calculations of average order value were equally modest. Globally, average orders were up just 2% to $103 per “basket” while in the U.S. they were
up a paltry 1% to $119.

Mobile devices had a standout year: Adobe said that some $3.3 billion was spent over mobile devices on Thanksgiving, up 14% and an all-time record for the day.

Salesforce added that online traffic in general to e-commerce sites, which includes browsing, was up too, but again only in single digits of 4% globally and 6%
in the U.S.

The two are actively tracking sales for today, Black Friday, and for the whole of the “Cyber Weekend” — which not only kicks off holiday shopping but has a
particularly strong showing in online sales due to people traveling to be with family.
</item 9>

<item 10>
Some of our favorite devices are on sale for Black Friday. (The Verge, 2023-11-24T15:42:21+00:00)
Black Friday and Cyber Monday may be in the past, but some sales are still hanging around (although maybe not for long). For more leftover deals we recommend
across all categories, be sure to check out the rest of the Cyber Monday still available here.

Every month or so, we like to ask our staff about their favorite stuff — whether it’s pet toys, travel aids, kitchen gadgets, or straightforward tech. And the
results are usually very different, very interesting, and a lot of fun.

In celebration of the annual post-Thanksgiving sales, we looked through some of our recent “favorites” articles and found deals on a lot of the tech, kitchen
tools, travel aids, and pet toys we like. We thought we’d list a few in case you’ve read about them in the past and thought, “Well, that sort of sounds good,
but it’s a bit pricey.”
</item 10>

<item 11>
Adobe: Thanksgiving US online sales nudge up to $5.6B; Salesforce: $31.7B spent globally (TechCrunch, 2023-11-24T15:43:51+00:00)
“Consumers are embracing mobile wallets to break down friction between discovering on social and purchasing on mobile.”

— Retailers are really pushing out discounts to get people more willing to spend money this year. Both Adobe and Salesforce said in the U.S. discounts were up
by about 28%. Categories that saw dramatic markdowns included toys, electronics and computers, per Adobe’s figures; respectively these saw sales up by 182% and
113% over last month.

— Buy Now Pay Later (BNPL) remains a popular option for paying upfront. BNPL drove $390 million in online spend, Adobe said, up 7.5% on last year.

We’ll update with more data later, and with Black Friday figures as they start to emerge.
</item 11>

<item 12>
The 53 best Black Friday deals we could find at Amazon, Walmart, Target and more (Engadget, 2023-11-19T15:13:23+00:00)
If you're a frequent Target shopper, note that the retailer is selling this same bundle with a $75 store gift card for $450.

Xbox Series S

A starter bundle that includes a 512GB capacity Xbox Series S and a three-month subscription to Game Pass Ultimate is currently $240 at Microsoft. The console
by itself has a list price of $300 and three months of access to Xbox's service will set you back $51 after this year's price hike (though the first month is
often just $1 for new members).

Anker accessories sale

A slew of Anker chargers, cables, power adapters and other accessories are on sale for Black Friday at Amazon, with discounts ranging up to 37 percent. The
deals extend to our favorite premium power bank, Anker's Prime 20,000 mAH battery with charging base, which is $130 for the set and its lowest price ever.
</item 12>

<item 13>
American Express Cash Magnet® Card review: a simple, no-nonsense cash back card (Fortune, 2023-11-03T20:07:05+00:00)
Is the American Express Cash Magnet® Card right for you?

The American Express Cash Magnet® Card is best for those who are looking for a simple and easy-to-use card they can use for all of their purchases. With no
annual fee and 1.5% cash back on every purchase, it’s a no-frills, no-hassle addition to your wallet.

But if you love to travel, you’ll likely need another credit card. Other options offer more comprehensive travel benefits, such as reimbursing you for TSA
PreCheck fees or travel interruption/cancellation insurance, and you can find other cards that don’t charge foreign transaction fees.

Credit cards similar to American Express Cash Magnet® Card

If the American Express Cash Magnet® Card isn’t the best choice for you, consider these alternatives:

Amex Blue Cash Everyday vs. ​​American Express Cash Magnet® Card

The Blue Cash Everyday® Card from American Express is similar to the American Express Cash Magnet® Card in its fees and APR offers, but it has a different
rewards structure.
</item 13>

<item 14>
American Express Cash Magnet® Card review: a simple, no-nonsense cash back card (Fortune, 2023-11-03T20:07:05+00:00)
The only transactions you won’t earn rewards on are the following:

Balance transfers

Cash advances

Purchases of traveler’s checks

Purchases of prepaid cards or reloading card balances

Person-to-person transfers, such as Venmo transactions

One important thing to note: American Express doesn’t allow cardholders to earn rewards during a billing period if the cardmember hasn’t paid the minimum
payment due by the due date.

Does American Express Cash Magnet® Card offer a new customer bonus?

If you are a new customer, you can qualify for a $200 statement credit if you spend $2,000 in the first 6 months of card ownership. Once you’ve reached the
spending threshold, American Express will credit the $200 to your account as a statement credit. However, it can take eight to 12 weeks after reaching the
spending requirement to receive the credit.

The bonus on the American Express Cash Magnet® card is relatively low—it’s possible to find cards without annual fees that offer higher bonus amounts—but the
bonus offer gives customers more time to reach the spending threshold.
</item 14>

<item 15>
The 53 best Black Friday deals we could find at Amazon, Walmart, Target and more (Engadget, 2023-11-19T15:13:23+00:00)
The the magnetic clip mount and long-lasting, fast-charging battery are also reasons to buy.

Masterclass Black Friday deal

Masterclass’ Black Friday deal gives you two memberships for the price of one, starting at $120 per year. That provides two individual memberships, while you
can pay a total of $180 for access for two devices simultaneously, content downloads and a Family membership that covers six devices. Masterclass could be a
good gift option for people who love to learn or collect hobbies as it provides access to hundreds of hours of video of masters of their craft explaining how
they do their work. Some classes have exercises you can complete as well, making it feel more like a real class. This deal will be available until the end of
the day on Black Friday proper.

Headspace Black Friday deal

Headspace Headspace $35 $70 Save $35 Headspace has knocked 50 percent off its memberships for Black Friday, so you can pay only $35 to get one year of access.
</item 15>

<item 16>
The 53 best Black Friday deals we could find at Amazon, Walmart, Target and more (Engadget, 2023-11-19T15:13:23+00:00)
$35 at Headspace

Headspace’s Black Friday deal is back for 2023, knocking half off subscriptions for new and returning subscribers. You’ll spend $35 if you pay for one year
upfront, or $6.50 per month if you choose the more frequent payment method. Headspace is a solid meditation and mindfulness app that packs a lot of value,
including dozens of guided meditation sessions to choose from at various durations, “sleepcasts” and soothing audio to help you get better shut eye and curated
yoga and fitness videos that can help you stick to your physical health goals.

Surfshark VPN

Surfshark’s Black Friday deal knocks up to 86 percent off two-year plans, and you’ll get a few extra months of service for free depending on the plan you
select. We consider Surfshark to be one of the best VPN services available right now, particularly if you want a solid VPN at an affordable price.
</item 16>

<item 17>
The 53 best Black Friday deals we could find at Amazon, Walmart, Target and more (Engadget, 2023-11-19T15:13:23+00:00)
We believe the mid-tier plan, Surfshark One, will be the best for most people and a two-year plan is down to $2.69 per month, with an additional four months of
services thrown in for free.

Samsung M80C Smart Monitor

Samsung’s Smart Monitor M80C is $300 off and down to $400 right now. The selling point for this display, and really any Samsung Smart Monitor, is that it pulls
triple duty as an external display, a smart TV and an IoT hub. In addition to connecting it to your computer, you can use it standalone as a TV thanks to the
built-in TV operating system and included remote, and the embedded SmartThings hub lets you directly connect things like smart lights and other gear to the
display.

Motorola razr+

The Moto razr+ is 30 percent off and down to a record low of $700.
</item 17>

<item 18>
Fly with Delta frequently? Small business owners can score a solid welcome bonus and other perks with the Delta SkyMiles Gold Business Card (Fortune,
2023-12-12T21:01:43+00:00)
Main Cabin 1 Priority boarding. Cardholders (and up to nine guests on the reservation) can get priority boarding if they use their SkyMiles number when booking
their Delta flights.

Cardholders (and up to nine guests on the reservation) can get priority boarding if they use their SkyMiles number when booking their Delta flights. Flight
credit. If cardholders make eligible purchases totaling $10,000 or more in a year, they will receive a $100 Delta eCredit that can be used to purchase Delta
flights.

If cardholders make eligible purchases totaling $10,000 or more in a year, they will receive a $100 Delta eCredit that can be used to purchase Delta flights.
20% back on in-flight purchases. Cardholders can get 20% back as a statement credit when they purchase eligible pre-purchased meals, in-flight food purchases,
alcoholic beverages, and audio headsets on Delta flights.
</item 18>

<item 19>
Adobe: Thanksgiving US online sales nudge up to $5.6B; Salesforce: $31.7B spent globally (TechCrunch, 2023-11-24T15:43:51+00:00)
Thanksgiving Thursday, when stores in the U.S. are closed and many are spending time off work, has become the de facto start of the holiday shopping season both
for those looking to get bargains online and for online retailers to kick off holiday sales deals to meet that demand. But if there is a message from this
year’s Thanksgiving sales, it is that consumers are holding steady, but don’t hold your breath for a growth boom.

Adobe Analytics said that people in the U.S. on Thursday spent $5.6 billion online, which it calculates at an increase of just 5.5% on last year.

Salesforce, which also calculates sales based on data collected by its Commerce Cloud division, noted that globally, online sales reached $31.7 billion with its
U.S. tally at $7.5 billion — both up only 1%. Salesforce’s calculations of average order value were equally modest.
</item 19>

<item 20>
The 53 best Black Friday deals we could find at Amazon, Walmart, Target and more (Engadget, 2023-11-19T15:13:23+00:00)
This Breville machine is much more than an air fryer, though — it has 13 cooking functions and a capacity that can handle a full turkey, so it’s more like a
mini convection oven that sits on your countertop. It’s best for people who love to cook and have big families, or those who maybe don’t have a traditional oven
and want a versatile (and relatively small) appliance for their kitchen.

Fitbit Charge 6

Fitbit’s newest fitness tracker, the Charge 6, is $60 off right now and down to a record low of $100. It’s the follow-up to one of our favorite fitness
trackers, the Fitbit Charge 5, and it brings back the physical side button that previous generations had and integrates even more with Google services. It has
an AMOLED touchscreen that displays all of your activity stats and smart notifications, a built-in GPS for mapping outdoor workouts, a week-long battery life
and the ability to display turn-by-turn directions on its screen from Google Maps.
</item 20>

<item 21>
Fly with Delta frequently? Small business owners can score a solid welcome bonus and other perks with the Delta SkyMiles Gold Business Card (Fortune,
2023-12-12T21:01:43+00:00)
Is the Delta SkyMiles Gold Business Card right for you?

The Delta SkyMiles® Gold Business American Express Card is best for business owners who plan to fly with Delta frequently. Many of the card’s perks can be used
specifically on Delta flights. It’s a good card to hold on to long-term for the perks and discounts even if your spending is best put on another business card.

Frequently asked questions

Is the Delta SkyMiles Gold Business American Express card worth it?

The $99 annual fee on the Delta SkyMiles Gold Business American Express card may be worth it for business owners who frequently travel with Delta. The card
offers perks like a free checked bag that could easily cover the annual fee.

How much is the Delta Gold Business Card annual fee?

The Delta Gold Business card has a $99 annual fee, but there’s a $0 intro fee for the first year.

Is Delta SkyMiles Gold Card hard to get?
</item 21>

<item 22>
How to tackle Black Friday shopping like a boss (Polygon, 2023-11-12T15:00:00+00:00)
However, it likely won’t be the last opportunity for you to save. As someone who’s been covering deals for The Verge and Polygon for years, I’m shocked by how
often I see these low prices, often spoken of as if they’re a flash in the pan, extending through much of December. That said, it’s more likely that price cuts
on products you want can go by unnoticed unless you’re tracking deals yourself.

Track deals yourself

We love to tell you about great deals. But there’s a limit to the kind of products that we write about, and the ones that we see. If you’re not comfortable
leaving it up to the chance that we’re going to tell you about price cuts on the products you really want, you have the power to track deals with relative ease.

If you only shop on Amazon, CamelCamelCamel remains one of the best tools for tracking price changes and history for products.
</item 22>

<item 23>
Adobe: Thanksgiving US online sales nudge up to $5.6B; Salesforce: $31.7B spent globally (TechCrunch, 2023-11-24T15:43:51+00:00)
As a bellwether for the rest of the holiday period — traditionally the biggest period for retail sales in the year — the flat sales speak to another tough year
for online retailers.

Adobe Analytics is predicting $37.2 billion in online spend for the full five days, up just 5.4% year on year and accounting for 16.8% of all holiday spend. And
Black Friday will see $9.6 billion in sales, up roughly the same, 5.7%, versus figures last year. (The published figures in 2022 were $9.13 billion.) Salesforce
has not provided forecasts.

For some context on today’s Thanksgiving figures, last year’s published figures from Adobe were $5.29 billion, which actually represents an increase of just
under 4%. (It’s likely Adobe Analytics readjusted its final figures for last year, which is why we see a higher percentage of growth.)
</item 23>

<item 24>
Adobe: Thanksgiving US online sales nudge up to $5.6B; Salesforce: $31.7B spent globally (TechCrunch, 2023-11-24T15:43:51+00:00)
Today’s 5.5% rate is definitely an improvement on last year’s 2.9%. But it’s nothing compared to the years preceding COVID-19, such as 2017, where we were
seeing growth of 18% or more.

Inflation is making an impact but not as much as the worry around consumer spending, said Adobe, which said that spending is being led actually by more
discounts to encourage buying, rather than less buying of more expensive products. That leads one to wonder what kind of impact that’s having on retailers’
margins.

“Cyber Week is off to a strong start with Thanksgiving driving a record $5.6 billion in online spend as consumers took advantage of strong discounts and
continued their shopping plans, virtually,” said Vivek Pandya, lead analyst, Adobe Digital Insights, in a statement. “Mobile shopping hit an all-time high, as
shoppers took to their smartphones to get the best deals during holiday gatherings, further solidifying mobile’s growing importance in e-commerce.”
</item 24>

<item 25>
The 53 best Black Friday deals we could find at Amazon, Walmart, Target and more (Engadget, 2023-11-19T15:13:23+00:00)
In the consumer electronics space, we expect most companies to have some sort of Black Friday promotion, so that means you will likely be able to get big-ticket
items from brands like Apple, Samsung, Google, Sony, LG, Sonos and others for much cheaper prices.

Your Black Friday Shopping Guide: See all of Yahoo’s Black Friday coverage, here. Follow Engadget for Black Friday tech deals. Learn about Black Friday trends
on In The Know. Hear from Autoblog’s experts on the best Black Friday deals for your car, garage, and home, and find Black Friday sales to shop on AOL,
handpicked just for you.
</item 25>

<item 26>
American Express Cash Magnet® Card review: a simple, no-nonsense cash back card (Fortune, 2023-11-03T20:07:05+00:00)
When you use the Amex Cash Magnet Card to pay for your entire car rental cost and decline the rental company’s collision damage waiver, you automatically
receive secondary coverage against damage or theft of the rental vehicle. Global assist hotline: If you are traveling 100 miles or more from home, you can use
the Global Assist Hotline to get emergency assistance, including medical referrals, emergency cash wires and missing luggage services. However, you are
responsible for paying for the charges of any third-party service provided.

If you are traveling 100 miles or more from home, you can use the Global Assist Hotline to get emergency assistance, including medical referrals, emergency cash
wires and missing luggage services. However, you are responsible for paying for the charges of any third-party service provided. Plan It: On purchases of $100
or more, you can take advantage of Amex’s Plan It feature. It allows you to split purchases into equal monthly payments at a fixed fee—similar to how buy now,
pay later apps work.
</item 26>

<item 27>
The best Black Friday deals you can get for under $25 (The Verge, 2023-11-23T17:58:09+00:00)
Tile Slim (2022 version) $ 20 $ 35 43 % off $ 20 $ 20 $ 35 43 % off The Tile Slim is as thin as two credit cards, allowing you to stash the simple Bluetooth
tracker in your wallet, passport holder, or luggage. It also features a 250-foot range and, more importantly, compatibility with both Android and iOS devices.
$20 at Amazon$25 at Best Buy

Smart home tech

Meross Smart Wi-Fi Plug Mini (MSS110) $ 18 $ 23 22 % off $ 18 $ 18 $ 23 22 % off With solid performance and support for all of the major smart home platforms,
this affordable Wi-Fi smart plug gives you the basics you need to control non-smart devices for less. It doesn’t have Matter or energy monitoring, but that’s
alright at this price.
</item 27>

<item 28>
Adobe: Thanksgiving US online sales nudge up to $5.6B; Salesforce: $31.7B spent globally (TechCrunch, 2023-11-24T15:43:51+00:00)
— More on mobile: Salesforce said that for the day, mobile accounted for 79% of all online traffic globally and 82% in the U.S., and that mobile wallets were
really making a mark for reducing some of the buying friction on the smaller devices. Apple Pay saw transactions up by 47% compared to 44% up for all mobile
wallets overall. Social on mobile — think Instagram links, TikTok and Snapchat — has become an influencer in itself. These collectively accounted for 13% of all
traffic referrals to sites. Notably, they are not where the purchases are being made for the most part, though.

“Mobile traffic and sales are soaring as people are on the go once again this holiday weekend,” said Salesforce VP and GM, Retail, Rob Garf, in a statement.
“Consumers are embracing mobile wallets to break down friction between discovering on social and purchasing on mobile.”
</item 28>

<item 29>
Fly with Delta frequently? Small business owners can score a solid welcome bonus and other perks with the Delta SkyMiles Gold Business Card (Fortune,
2023-12-12T21:01:43+00:00)
Cardholders can get 20% back as a statement credit when they purchase eligible pre-purchased meals, in-flight food purchases, alcoholic beverages, and audio
headsets on Delta flights. Baggage insurance. Cardholders who purchase plane, train, ship or bus ride tickets with their card are eligible for baggage insurance
coverage—which kicks in if your baggage is stolen, lost, or damaged.

Cardholders who purchase plane, train, ship or bus ride tickets with their card are eligible for baggage insurance coverage—which kicks in if your baggage is
stolen, lost, or damaged. Car Rental Loss and Damage Insurance. A cardholder can get damage and theft insurance on their car rental. Cardholders must use their
card to book the rental and decline the collision damage waiver at the agency to receive it.
</item 29>

<item 30>
Fly with Delta frequently? Small business owners can score a solid welcome bonus and other perks with the Delta SkyMiles Gold Business Card (Fortune,
2023-12-12T21:01:43+00:00)
For $99 a year, cardholders receive benefits like priority boarding, the first checked bag benefit (and for guests), discounts on in-flight purchases, and
baggage insurance. The card, however, doesn’t provide access to Delta Sky Club or Delta’s airport lounges.

The rewards structure on this card is decent, but it’s nothing to write home about—the card offers 2x miles on just a handful of categories like advertising,
shipping, and dining.

A big benefit of the Delta SkyMiles Gold Business Card is the welcome bonus: 75,000 Bonus Miles after you spend $6,000 in purchases on your new Card in your
first 6 months of card opening. And cardholders who use their miles to purchase flights get even more benefits—customers receive a 15% discount when they use
miles to buy Delta flights on the Fly Delta app or the Delta website.
</item 30>

<item 31>
Fly with Delta frequently? Small business owners can score a solid welcome bonus and other perks with the Delta SkyMiles Gold Business Card (Fortune,
2023-12-12T21:01:43+00:00)
Credit cards similar to Delta SkyMiles Gold Business Card

Cardholders who aren’t loyal to Delta and want more flexible options for redeeming their rewards should consider the Chase Ink Business Preferred® Credit Card,
which has a similar annual fee but a more generous welcome bonus and rewards structure.

Ink Business Preferred credit card vs. Delta SkyMiles Gold Business Card

The Chase Ink Business Preferred® Credit Card has an annual fee of $95 and a large welcome bonus of 100,000 points after spending $8,000 in the first three
months of card opening.

With the Ink Business Preferred Card, customers receive 3x points per $1 (up to $150,000 combined) spent on shipping purchases, internet, cable, phone, and
advertising with social media sites and search engines. Ultimate Rewards points can be transferred 1:1 to Virgin Atlantic or Flying Blue to book flights on
Delta.

Is the Delta SkyMiles Gold Business Card right for you?
</item 31>

<item 32>
Adobe: Thanksgiving US online sales nudge up to $5.6B; Salesforce: $31.7B spent globally (TechCrunch, 2023-11-24T15:43:51+00:00)
Adobe Analytics’ figures are based, it says, on 1 trillion visits to U.S. retail sites, 100 million SKUs and 18 product categories. Salesforce says it taps data
from 1.5 billion consumers in its research. (Both companies work with a number of giant and smaller retailers, so they have the infrastructure to provide this
kind of intel to these customers on an ongoing basis.)

Breaking out some of the trends in how people are shopping:

— Overall, desktop sales are still exceeding other kinds of screens when it comes to conversions and number of items purchased, but mobile is the standout
device in the evening hours, when it accounted for a whopping 59% of all online sales (likely because people were using that time to socialize and buy on the
sly).
</item 32>

<item 33>
Fly with Delta frequently? Small business owners can score a solid welcome bonus and other perks with the Delta SkyMiles Gold Business Card (Fortune,
2023-12-12T21:01:43+00:00)
Delta SkyMiles Gold Business Card: Rewards

The Delta SkyMiles® Gold Business American Express Card provides higher rewards rates on select purchases that might be helpful to small business owners:

2x miles on purchases made directly with Delta

2x miles on U.S. purchases for advertising in select media

2x miles on U.S. shipping purchases

2x miles on dining at restaurants

1x mile on every eligible dollar spent on other purchases

New cardholders can receive a welcome bonus of 75,000 Bonus Miles after you spend $6,000 in purchases on your new Card in your first 6 months of card opening.

How to redeem Delta SkyMiles Gold Business Card miles

Purchase flights

Cardholders can use their miles (and their cash) to buy flights. When cardholders book flights through the Fly Delta app or Delta SkyMiles account with their
cards, they can get a 15% discount called Takeoff15.
</item 33>

<item 34>
Shop 200+ Walmart doorbuster deals for Black Friday (Mashable, 2023-11-25T01:13:44+00:00)
Read Mashable's review of the Solo Stove Bonfire 2.0.

More outdoor deals

Apple deals

Why we like it

The Apple Watch Series 9 is "a robust and elite wearable device – the best Apple has ever made," according to Mashable Senior Editor Stan Schroeder. While it
looks almost identical to its predecessor, it's equipped with a new s9 chip, which makes it about 30 percent snappier than before. It also boasts double the
storage and has double the peak brightness and a fun new Double Tap feature. It's not an essential upgrade if you already have a Series 7 or 8, but it's
certainly a step in the right direction for those with older watches. At $359 for the 45mm GPS version, it's currently sitting at an all-time low cost at
Walmart.

More Apple deals

Pet deals

Why we like it

A solid dog tracking device for pet parents on a budget, Tractive gives you the ability to track your dog's live location within the companion app.
</item 34>

<item 35>
American Express Cash Magnet® Card review: a simple, no-nonsense cash back card (Fortune, 2023-11-03T20:07:05+00:00)
Statement credits will be applied to your account within three days, but statement credits cannot be used toward the minimum amount due. Unless the statement
credit reduces your balance to $0, you’re still required to make the minimum payment.

Additional benefits

Compared to the benefits of other credit cards, the American Express Cash Magnet® Card is more limited. However, it does offer the following perks:

American Express Experiences: With American Express Experiences, you get access to exclusive ticket pre-sales and cardmember-only events. Events include major
Broadway shows, concerts, and sport competitions.

With American Express Experiences, you get access to exclusive ticket pre-sales and cardmember-only events. Events include major Broadway shows, concerts, and
sport competitions. Car rental loss and damage insurance: When you use the Amex Cash Magnet Card to pay for your entire car rental cost and decline the rental
company’s collision damage waiver, you automatically receive secondary coverage against damage or theft of the rental vehicle.
</item 35>

<item 36>
200+ of the best deals from Amazon's Cyber Monday sale (Mashable, 2023-11-27T08:45:59+00:00)
More headphone and earbud deals

Headphones

Earbuds

Tablet deals

Gaming deals

Gaming headsets

Mashable Deals Want more hand-picked deals in your inbox? Sign up for Mashable's daily Deals newsletter. Loading... Sign Me Up By signing up you agree to our
Terms of Use and Privacy Policy Thanks for signing up!

Gaming mice

Gaming keyboards

VR headsets

Meta Quest 2 — $249 $299.99 (save $50.99) + free $50 Amazon credit with code META50 🔥

Speaker deals

Vacuum deals

Cordless vacuums

Robot vacuums and vacuum/mop hybrids

Kitchen deals

Pizza ovens

Sparkling water makers

Toaster ovens

Smart home deals

Note: Echo devices are listed above under "Amazon device and service deals."

Blink

Chromecast

Google Nest

Fitness deals

Smartwatches and fitness trackers

Note: Apple Watches are listed above under "Apple deals."
</item 36>

<item 37>
Fly with Delta frequently? Small business owners can score a solid welcome bonus and other perks with the Delta SkyMiles Gold Business Card (Fortune,
2023-12-12T21:01:43+00:00)
Delta SkyMiles Gold Business Card Highlights

Card Type: Airline

Welcome bonus: 75,000 Bonus Miles after you spend $6,000 in purchases on your new Card in your first 6 months

75,000 Bonus Miles after you spend $6,000 in purchases on your new Card in your first 6 months Annual fee: $0 intro annual fee for the first year (then $99)

$0 intro annual fee for the first year (then $99) Rewards: 2x miles on purchases made directly with Delta, 2x miles on U.S. purchases for advertising in select
media, 2x miles on U.S. shipping purchases, 2x miles on dining at restaurants, 1x mile on every eligible dollar spent on other purchases

With the Delta SkyMiles® Gold Business American Express Card, cardholders who frequently fly with Delta can reduce travel costs and upgrade their travel
experience.
</item 37>

<item 38>
American Express Cash Magnet® Card review: a simple, no-nonsense cash back card (Fortune, 2023-11-03T20:07:05+00:00)
With this card, you’ll earn:

3% cash back on up to $6,000 spent on groceries per year

3% cash back on up to $6,000 spent on U.S. online retail purchases per year

3% cash back on up to $6,000 spent at gas stations per year

1% cash back on all other purchases

If you tend to spend more money in select categories, such as groceries or online retail purchases, the Blue Cash Everyday® Card from American Express may be a
better fit.

Discover it® Cash Back Credit Card vs. ​​American Express Cash Magnet® Card

Like the American Express Cash Magnet®, the Discover it® Cash Back Card doesn’t have an annual fee. But it has a very different rewards system than the Amex
card; rather than earning a flat cash back rate, you can earn 5% cash back on up to $1,500 in combined quarterly purchases in various categories throughout the
year (then 1%).
</item 38>

<item 39>
Shop 200+ Walmart doorbuster deals for Black Friday (Mashable, 2023-11-25T01:13:44+00:00)
It also allows you to set up a virtual fence and notifies you when your pet steps out of (or enters) the safe area. It's not perfect — the battery is just so-so
and the live tracking can be a bit glitchy — but for the price, it's hard to beat. Save $20 for Black Friday at Walmart and snag it for only $29.99.

More pet deals

Fitness deals

Why we like it

The Walmart-exclusive Echelon Connect Sport is a solid Peloton alternative for a retail price that won't make you cringe. And for Black Friday, it's on sale for
over $400 off, knocking the price down to a super impressive $297. But just because it's wallet-friendly doesn't mean its specs don't impress. The Connect Sport
features 32 levels of silent magnetic resistance, aero handlebars and adjustable pedals, and Bluetooth technology that lets you connect any smart device to the
Echelon Fit app.
</item 39>

<item 40>
200+ of the best deals from Amazon's Cyber Monday sale (Mashable, 2023-11-27T08:45:59+00:00)
The smart screen responds to Alexa commands that are particularly handy when your hands are full, like asking for measurement conversions mid-cooking, checking
the weather mid-rushing out the door, or turning off your smart lights with a sleeping kid in hand. Plus, it's got a 5.5-inch screen and better sound than its
predecessor, making it perfect for watching videos or video calling friends and family.

More Amazon device and service deals

Amazon services

Audible Premium Plus — $5.95/month $14.95/month for four months (save $9/month; new customers only; get an additional $20 Audible credit)

Echo Buds

Echo smart displays

Echo smart speakers

Fire tablets

Fire TVs

Note: All Fire TVs come with a free 6-month subscription to MGM+ (a $35.94 value).
</item 40>

<item 41>
Fly with Delta frequently? Small business owners can score a solid welcome bonus and other perks with the Delta SkyMiles Gold Business Card (Fortune,
2023-12-12T21:01:43+00:00)
Delta has more than 20 airline partners, so cardholders can book through airlines like Air France and Virgin Atlantic—though the 15% discount only applies to
Delta and Delta Connection® flights.

This discount is a reason to hold onto the card long-term as you could save thousands of miles over the course of a year if you are a heavy Delta user. The best
bargains on award flights are found through Delta SkyMiles Award Deals. Recent deals from Seattle (SEA) have included Chicago (ORD) for 8,000 miles or Auckland,
New Zealand (AKL) for 64,000 one way in economy plus taxes and fees. Flash sales come and go so it’s a good idea to keep an eye on the site.

Other options include booking travel through Delta Vacations, using miles for upgrades or even buying champagne at the Delta Sky Club. That said, your best
value will most likely be on award tickets.
</item 41>

<item 42>
54 Best Apple Black Friday Deals (2023): iPad, Apple Watch, AirPods (Wired, 2023-11-25T00:00:29+00:00)
This helps support our journalism. Learn more.

Apple's Cyber Monday Promotion

Apple notoriously doesn't run sales on its own storefront. However, it does usually throw in a gift card for Apple Store credit if you purchase select items.
This year, now through November 27, you can get up to $200 in Apple Store credit to use later if you purchase certain items. For example, if you buy an iPhone
14 or iPhone SE, you can get up to $75 in store credit. Buy a MacBook Air and you'll get up to $200. You can find the full list here.

Separately, there's also a sale on Apple gift cards on Amazon. Each one has a unique corresponding coupon code that you can see on this page. For example, you
can get a $100 Apple Gift Card and a free $15 Amazon credit with code APPLET5USAP. There are tons of cards to choose from. You can browse them all here.
</item 42>

<item 43>
The 53 best Black Friday deals we could find at Amazon, Walmart, Target and more (Engadget, 2023-11-19T15:13:23+00:00)
Plugs like these are a simple way to add some smart capabilities to any home, letting you turn on lights with just your voice, set automated schedules and
routines that are triggered by other activities. These would make a great stocking stuffer for anyone you know who's curious about smart home connectivity but
hasn't yet taken the plunge.

Xbox gift card

You can pick up a $50 Xbox gift card for $45 right now at Amazon and Target, a digital item that would make a good gift for a gamer who’s otherwise hard to shop
for. This card can be used on a number of things, including game downloads, in-game purchases, membership costs like Xbox Game Pass and more. If you or a loved
one is loyal to a different console, there are other digital gift cards on sale, too. A $70 PlayStation Plus gift card is going for $63 right now, while a
12-month Nintendo Switch Family plan is slightly discounted from $35 to $32.
</item 43>

<item 44>
The 53 best Black Friday deals we could find at Amazon, Walmart, Target and more (Engadget, 2023-11-19T15:13:23+00:00)
Samsung microSD card sale

If you need a new microSD card for your Nintendo Switch, Steam Deck or GoPro, a trio of Samsung microSD cards we recommend are also down their lowest prices to
date. The 128GB version of the Samsung Pro Plus is down to $11 at Amazon, B&H, Best Buy and others, while the 256GB and 512GB models are down to $18 and $32,
respectively. Beyond that, the 256GB Samsung Evo Select is on sale for a new low of $15 and the 256GB Samsung Pro Ultimate is available for $25.

The Pro Plus is the top overall pick in our microSD card buying guide, as it consistently ranked among the fastest cards we tested despite its relatively
affordable price tag. The Pro Ultimate offers faster read speeds, so it could be worth the step up if you often move files between your card and a computer.
</item 44>

<item 45>
Shop 200+ Walmart doorbuster deals for Black Friday (Mashable, 2023-11-25T01:13:44+00:00)
Speaking of which, the app gives you access to 40+ daily live and 15,000+ on-demand connected fitness classes. The $297 Black Friday cost even includes a 30-day
free trial of Echelon Premier membership to get you started.

More fitness deals

Toys and gift deals

Why we like it

In days gone by, Walmart's Black Friday sale was full of 40% or higher discounts, but the era of the doorbuster deal is ending. And that's why it's so
refreshing to see a full 50% discount on a popular item like this. The LEGO Star Wars line includes tons of great gift ideas for fans of all ages, from Jedis to
Padawans. This building kit of The Child, aka Baby Yoda, is just begging to live on your office desk.

More deals on toys and gifts
</item 45>

<item 46>
Fly with Delta frequently? Small business owners can score a solid welcome bonus and other perks with the Delta SkyMiles Gold Business Card (Fortune,
2023-12-12T21:01:43+00:00)
That said, your best value will most likely be on award tickets.

Rates and fees

Annual fee: $0 intro annual fee for the first year (then $99)

$0 intro annual fee for the first year (then $99) Foreign transaction fee: None

None Purchase APR: 20.99%–29.99% variable

Additional benefits

Free first checked bag. Cardholders (and up to nine of their guests) can get the first checked bag for free. The standard bag fee is $30, so if you use this
perk more than three times, you’ll save more in bag fees than the annual fee costs you.

Cardholders (and up to nine of their guests) can get the first checked bag for free. The standard bag fee is $30, so if you use this perk more than three times,
you’ll save more in bag fees than the annual fee costs you. Main Cabin 1 Priority boarding.
</item 46>

<item 47>
The best Black Friday deals you can get for under $25 (The Verge, 2023-11-23T17:58:09+00:00)
It streams in 4K, supports HDR color, and is compatible with Alexa.

Games and gaming accessories

Mass Effect Legendary Edition $ 12 $ 60 80 % off $ 12 $ 12 $ 60 80 % off A compilation of the first three Mass Effect games repackaged and optimized for 4K with
reduced load times, improved performance, and various gameplay tweaks. $12 at Amazon (Steam code)

Samsung Evo Select microSD card (512GB) $ 25 $ 35 29 % off $ 25 $ 25 $ 35 29 % off This Samsung Evo Select microSD card offers 512GB of extra storage for a
Nintendo Switch, Steam Deck, or Asus ROG Ally at a good price point. $25 at Amazon

Staff favorites and miscellaneous tech

ChomChom pet hair remover $ 20 $ 32 38 % off $ 20 $ 20 $ 32 38 % off The ChomChom is a reusable cat and dog hair remover that works great for furniture.
</item 47>

<item 48>
The best Apple Black Friday deals on iPads, AirPods, Apple Watches and more (Engadget, 2023-11-24T09:30:32+00:00)
Engadget has been testing and reviewing consumer tech since 2004. Our stories may include affiliate links; if you buy something through a link, we may earn a
commission. Read more about how we evaluate products.

There are still tons of Apple Black Friday deals you can shop for online; these are our favorites.

Most of Apple’s current-generation devices went on sale for Black Friday — and most of those deals look like they’ll carry over through the weekend and into
Cyber Monday. That means if you were otherwise occupied on Friday, you can still save on iPads, Apple Watches, MacBooks, AirPods and more. Some of the prices
are new lows and many others match the record prices we’ve seen previously. We’ve tested and reviewed nearly every device below, so check out the links to our
full reviews and guides if you need help deciding which deals to snag. These are the best Apple Black Friday deals you can still get today from retailers like
Amazon, Target, Best Buy and B&H Photo.
</item 48>

<item 49>
200+ of the best deals from Amazon's Cyber Monday sale (Mashable, 2023-11-27T08:45:59+00:00)
27, includes both a ton of deals carried over from Black Friday plus some new ones.

We're curating a running list of Amazon's best Cyber Weekend deals, spotlighting some of our favorites and noting when good deals inevitably sell out. Read on
for the full rundown, and check back often: We're going to be updating this story incessantly as the sale continues, as well as our even bigger (if you can
imagine) list of Cyber Monday deals across more retailers.

Note: All newly added deals are marked with a ✨, while deals with a 🔥 have dropped to an all-time low price. Amazon's invite-only deals for Prime members are
marked with a 📨. Deals with a strikeout were either sold out or expired at the time of writing.

Echo, Fire TV, and Kindle deals

Why we like it

An Echo Show is a subtle yet game-changing addition to any room — and for less than $40 with this rollover Black Friday to Cyber Monday deal, there's little
reason to not make your life easier.
</item 49>

<item 50>
The 53 best Black Friday deals we could find at Amazon, Walmart, Target and more (Engadget, 2023-11-19T15:13:23+00:00)
It’s hard to tell if this is a record-low price, but considering the standard V15 Detect is going for between $650 and $750 across the internet, we consider
this to be a good deal. In addition to its strong cleaning power, the V15 Detect has a laser-powered optic cleaner head that illuminates the floor before you as
you’re cleaning so you can see dust and grime more clearly. It also has a piezo sensor, which sizes and counts dust particles as you clean and shows you that
information on its LCD display.

Amazon Echo Black Friday sale

Almost the entirety of Amazon's Echo speaker lineup is on sale for Black Friday, with most bundled alongside a TP-Link Kasa smart bulb for no extra cost.
</item 50>
=== END ITEM id=b9a0c63e72592c49 ===

=== ITEM id=1f335281acad8879 ===
Gold sentences and facts:
Sentence 0: For all the other benefits they get from Android, Google could solve this problem today if they wanted to.
  Fact 0.0: The parties in question receive other benefits from Android
  Fact 0.1: Google could solve this problem today if it wanted to
Sentence 1: It will attempt to argue that Google restrains competition within two separate markets, including the distribution of apps to Android users and the market for payment processing solutions for content inside Android apps.
  Fact 1.0: It will attempt to argue that Google restrains competition within two separate markets
  Fact 1.1: One of the two markets is the distribution of apps to Android users
  Fact 1.2: The other market is the market for payment processing solutions for content inside Android apps
Sentence 2: The case, filed by Arkansas-based publisher Helena World Chronicle, argues that Google “siphons off” news publishers’ content, their readers and ad revenue through anticompetitive means.
  Fact 2.0: The case was filed by Arkansas-based publisher Helena World Chronicle
  Fact 2.1: The case argues that Google siphons off news publishers' content
  Fact 2.2: The case argues that Google siphons off news publishers' readers
  Fact 2.3: The case argues that Google siphons off news publishers' ad revenue
  Fact 2.4: The case argues that Google does this through anticompetitive means

Retrieved items:
<item 1>
Epic v. Google, explained (The Verge, 2023-11-05T11:00:00+00:00)
Epic also claims that Google is illegally tying its Google Play payments platform to the Google Play app store, keeping other potential app payment mechanisms
from competing as well.

So, does Google have a monopoly?

A monopoly on what? That’s one of the biggest questions the court will decide. Epic claims that Google has illegal monopolies in “Android app distribution” and
“Android in-app payment processing.” It’ll argue that if you buy or develop for Android phones, you’re stuck paying the Google tax.

But Google will say the real competition is from Apple, because people can choose to buy an iPhone instead. It’s pretty tough to argue Google has a monopoly on
app stores in general.

This whole line of argument is called “market definition.” If the court decides the relevant market is phones and app stores, not specifically Android ones,
then Google’s on the road to victory. If it accepts that Android apps are their own market, Epic will be in better shape. Or, it could pick a different market
definition entirely, like the judge in the Apple case ended up doing.
</item 1>

<item 2>
News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line (TechCrunch, 2023-12-15T17:56:02+00:00)
SGE, it argues, offers web searchers a way to seek information in a conversational mode, but ultimately keeps users in Google’s “walled garden” as it
“plagiarizes” their content. Publishers also can’t block SGE because it uses the same web crawler as Google’s general search service, GoogleBot.

Plus, it says Google’s Bard AI was trained on a dataset that included “news, magazine and digital publications,” citing both a 2023 report from the News Media
Alliance and a Washington Post article about AI training data for reference. (The Post, which worked with researchers at the Allen Institute for AI, had found
that News and Media sites were the third largest category of AI training data.)

The case points to other concerns, too, like changing AdSense rates and evidence of improper spoliation of evidence on Google’s part, by its destruction of chat
messages — an issue raised in the recent Epic Games lawsuit against Google over app store antitrust issues, which Epic won.
</item 2>

<item 3>
Tim Sweeney on Epic’s victory royale over Google (The Verge, 2023-12-12T19:17:14+00:00)
What would you say the differences are between the Apple and Google cases?

I would say Apple was ice and Google was fire.

The thing with Apple is all of their antitrust trickery is internal to the company. They use their store, their payments, they force developers to all have the
same terms, they force OEMs and carriers to all have the same terms.

Whereas Google, to achieve things with Android, they were going around and paying off game developers, dozens of game developers, to not compete. And they’re
paying off dozens of carriers and OEMs to not compete — and when all of these different companies do deals together, lots of people put things in writing, and
it’s right there for everybody to read and to see plainly.

I think the Apple case would be no less interesting if we could see all of their internal thoughts and deliberations, but Apple was not putting it in writing,
whereas Google was. You know, I think Apple is... it’s a little bit unfortunate that in a lot of ways Apple’s restrictions on competition are absolute.
</item 3>

<item 4>
The Epic v. Google trial may come down to simple v. complicated (The Verge, 2023-11-07T15:00:40+00:00)
Google spent its 45 minutes attempting to explain that its dominance over the Android app market isn’t anything nefarious but simply the natural outcome of
Google fiercely competing with the iPhone and its iOS App Store, where Google would like the court to believe that competition truly lies.

If Google can convince the jury of that, it could be a winning argument in the case — because obviously, Google doesn’t have a monopoly on app stores or phones
in general. “You cannot separate the quality of a phone from the quality of the apps in its app store, and that means Google and Apple compete against each
other,” began Google lead attorney Glenn Pomerantz.

But Google wound up spending much of its opening statement attempting to explain away its seemingly bad behavior as normal business practices and didn’t always
succeed out of the gate.
</item 4>

<item 5>
Epic v. Google, explained (The Verge, 2023-11-05T11:00:00+00:00)
Google will say that every Samsung Galaxy phone has “a competing app store right there on the homescreen,” even though Google apparently tried to quash that
store, too, fearing it could cost the company billions.

We caught Google trying to play the China card in a pre-briefing with press, reminding us that Epic is 40 percent owned by Chinese company Tencent — even though
Google itself allegedly considered teaming up with Tencent to eliminate the Epic threat. While the parties can’t argue it in court, Google can attempt to subtly
play the China card: “Google may ask a witness once about Tencent’s domicile,” reads one of Judge Donato’s orders.

In the same press briefing, Google suggested to journalists that Epic should have joined the company’s “User Choice Billing” program if it wanted an alternative
payment system, instead of suing. But that program didn’t exist until 2022, long after the lawsuit, and it only knocks 4 percentage points off Google’s fee.

When should we expect a jury verdict?
</item 5>

<item 6>
Tim Sweeney on Epic’s victory royale over Google (The Verge, 2023-12-12T19:17:14+00:00)
Thou shalt not have a competing store on iOS and thou shalt not use a competing payment method. And I think Apple should be receiving at least as harsh
antitrust scrutiny as Google.

It’s interesting to me that because Google distributes the Android operating system as open source, they had to put all these deals out in the open. More out in
the open, I should say — certainly they still wanted to keep them secret.

But I’m going down my story about all the best emails from the Epic v. Apple trial — and we do have a lot of documents from both Apple and Google that show they
were similarly self-serving in terms of deals.

I’d say this is the thing that’s disappointed me the most with Apple and Google: even at the peak of the antitrust trial against Microsoft, Microsoft was
awesome to developers. Microsoft has always been awesome to developers, always being respectful, giving developers a great deal and treating them as partners,
you know? And so even as Microsoft was crushing corporate competitors, the developer experience was excellent.
</item 6>

<item 7>
The Epic v. Google trial may come down to simple v. complicated (The Verge, 2023-11-07T15:00:40+00:00)
I did like Pomerantz’s commonsense argument that Google can’t possibly have a monopoly on Android app stores when “every single Samsung phone comes with two app
stores right on the homescreen,” which continued:

When they show these charts that show all these downloads from Play and not from the Galaxy Store, that’s what the Samsung phone users are choosing. They’re
touching Play. Nothing’s keeping them from touching the Galaxy Store; it’s just what works for them.

I called Google “comparatively open” earlier, and that openness will likely be heavily debated in the weeks to come. Epic promised to “show that Google has
closed off each and every other option” to the Play Store during this trial. But Google points to the simple fact that it allows alternate app channels at all —
something Android rival iOS doesn’t.

Pomerantz boasted that over a billion people have gone through the process Epic portrays as needlessly onerous to get apps outside the Play Store.
</item 7>

<item 8>
The Epic v. Google trial may come down to simple v. complicated (The Verge, 2023-11-07T15:00:40+00:00)
The future of Google’s app store is at stake in a lawsuit by Fortnite publisher Epic Games. Epic sued Google in 2020 after a fight over in-app purchase fees,
claiming the Android operating system’s Google Play Store constituted an unlawful monopoly — while Google says its demands would damage Android’s ability to
offer a secure user experience and compete with Apple’s iOS. Follow along with updates here.

Bornstein showed jurors charts of Google’s fat app profit margins (70 percent on $12 billion in revenue a year, says Epic) and pointed out several ugly-seeming
ways Google has allegedly attempted to keep anyone from taking that money away — like paying game developers not to build their own app stores or standalone app
launchers like Epic did with Fortnite.

“Google pays actual and potential competitors not to compete. Literally gives them money and other things of value,” said Bornstein. “It’s like Google saying,
‘Here’s $360 million’ — that’s an actual number you’ll hear about — why don’t you sit this one out and let me win?”
</item 8>

<item 9>
More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo (TechCrunch, 2023-11-13T14:37:44+00:00)
We’ve already established that Google and Apple have a long and mutually beneficial relationship, even while competing, so it’s not surprising to see three
Apple search-related queries bringing in the big bucks — not least since September 22, 2017 was the official release date of the iPhone 8.

Meanwhile, queries like “car insurance,” “cheap flights” and “credit report” are perennial favorites and they speak to how much Google dominates vertical search
— that is, search in very specific market categories. As for LifeLock… the big Equifax data breach of 2017 was a hot topic in September 2017 and LifeLock was
making a big push to win business with people who wanted to buy identity theft protection.

Revenue-shares to pre-install Google apps on Androids

Jamie Rosenberg, a Google employee who focuses on Android and Google Play, testified in Google’s defense on November 8. He said that the competition between
Google and Apple is “as intense as it gets,” reports Bloomberg.
</item 9>

<item 10>
What to know about Fortnite maker Epic Games’ antitrust battle with Google, starting today (TechCrunch, 2023-11-06T16:47:11+00:00)
This program, still in pilot testing, is open to all developers who sell apps in the 35 markets where it’s now available, including the U.S., and reduces the
standard commission by 4% for companies who choose to use their own payment processing solution. Spotify and Bumble were the initial testers for the new system,
first introduced in November 2022, and Match, as part of its settlement agreement, will also now take advantage of this option.

Google will also argue that its commissions aren’t just tied to payment processing, but rather help it to develop other features and controls to keep Android
users safe and guide them to discovering new applications, as well as help fund the tools and services developers need to build and grow their apps. Google also
helps developers reach a broader audience beyond the mobile phone, with support for Android across platforms, ranging from tablets to TVs to autos and more. And
it will note that 99% of its developers qualify for a commission of 15% or less, based on their revenues.
</item 10>

<item 11>
What to know about Fortnite maker Epic Games’ antitrust battle with Google, starting today (TechCrunch, 2023-11-06T16:47:11+00:00)
The competitive landscape with other app stores, OEMs

The tech giant additionally plans to push back at Epic’s claims that it doesn’t have competition in the app store business. It will point out that not only does
the Play Store compete with Apple’s App Store, which the Ninth Circuit ruling in the Apple case agreed upon, too, but Android also supports competitive app
stores like Samsung’s Galaxy Store and Amazon’s Appstore.

Epic, however, will attempt to demonstrate that Google makes it hard for alternative app stores to succeed and reach consumers, noting that 90% of apps are
still downloaded through Google Play. And it will point out that Google bundles the Play Store with other Google apps that Android OEMs (original equipment
manufacturers) have to agree to in order to use Android.

Epic will also bring up “Anti-Fragmentation Agreements” (AFAs) that prevent the OEMs from modifying Android to allow for frictionless downloading of apps
outside of Google Play.
</item 11>

<item 12>
More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo (TechCrunch, 2023-11-13T14:37:44+00:00)
He said that the competition between Google and Apple is “as intense as it gets,” reports Bloomberg.

Rosenberg explained how Google gets manufacturers to sign a mobile app distribution agreement (MADA) that requires Android smartphone makers (like Samsung or
Oppo) to pre-load a bundle of 11 Google apps on the device, including Search, Chrome and Play. They don’t have to be the default choices, he said.

Google also has revenue share agreements (RSAs) with smartphone makers and wireless carriers (like Verizon) that require them to set Google search and Chrome
web browser as defaults. Rosenberg defended the move and said it was because Google apps [like Search] are “best in class.” The RSAs also motivate other
companies to make or sell more Android devices, he said.
</item 12>

<item 13>
The Epic v. Google trial may come down to simple v. complicated (The Verge, 2023-11-07T15:00:40+00:00)
The upshot for consumers, Epic’s earlier legal filings have suggested, is that we pay higher prices for apps than we would if there were more competition and /
or lower app store and payment processing fees. But while this will probably come up later in the trial, Epic chose to focus more on simply painting Google as
the bad guy on day one.

It’s not clear how much of that evidence will hold up on closer examination. That $360 million, for instance, refers to an alleged payment that kept Activision
from opening an app store that could compete with Google Play. But Activision told The Verge in 2022 that it “never entered into an agreement that Activision
would not open its own app store” — and Google is now, it says, armed with the evidence to prove it. On Monday, Epic’s attorney admitted Google “was too clever”
to draw up contracts that specifically forced developers not to compete with the Play Store. The overall narrative is compelling, though — and I’m not sure
Google’s opening statement countered it.
</item 13>

<item 14>
What to know about Fortnite maker Epic Games’ antitrust battle with Google, starting today (TechCrunch, 2023-11-06T16:47:11+00:00)
Epic Games will specifically reference one agreement it had made with the OEM OnePlus, to make Epic games available on its devices through the Epic Games app.
But OnePlus had to cancel the deal over concerns that the arrangement would have it bypassing the Google Play Store. The case will also examine a deal between
Google and Samsung that Epic says was designed to prevent the Galaxy Store from being a competitive threat and require Google Play Billing on apps distributed
by Samsung.

Epic will argue Google’s developer agreements are anti-competitive

A second set of agreements Epic will highlight are those between Google and Android app developers themselves. One such program, the “Project Hug” initiative, a
part of the Google Games Velocity Program, will be cited as an example of where Google paid competitors not to distribute their apps on Android outside of the
Play Store by targeting those that were most at risk of attrition from Play, then approaching them with a monetary offer.
</item 14>

<item 15>
Is Google Search better than the rest? And is that fair? (The Age, 2023-10-22T18:00:00+00:00)
The question remains, why do users prefer Google? And if the answer, as Walker says, is that Google is simply better, the question becomes whether that’s down
to its scale, its longevity, its ubiquity and all the user data it sucks up, which no other company could possibly match.

For the US, which has to prove that Google somehow broke the law to achieve the status quo, this legal case could be an uphill battle. But for the rest of us,
it offers a rare opportunity for some insight into what the tech giant does with its enormous market share. Is it still focused on being the best? Or, as has
been alleged, does it use its ubiquity to squeeze us for money even at the expense of product quality? How ‘organic’ are Google’s results? This month, Wired
posted an opinion article by lawyer and privacy advocate Megan Gray, which alleged that Google had accidentally revealed during the trial that it manipulates
people’s search queries for maximum ad revenue.
</item 15>

<item 16>
What to know about Fortnite maker Epic Games’ antitrust battle with Google, starting today (TechCrunch, 2023-11-06T16:47:11+00:00)
To make its case, Epic plans to bring claims under Sections 1 and 2 of U.S. antitrust law, the Sherman Act, as well as California’s anticompetitive laws, the
Cartwright Act, and the Unfair Competition Law. It will attempt to argue that Google restrains competition within two separate markets, including the
distribution of apps to Android users and the market for payment processing solutions for content inside Android apps.

Google argues its commissions aren’t just tied to billing, but offers discount for third-party billing options

Today, Google requires apps to use its own first-party billing system, and charges app developers a 15% to 30% commission on the sales it processes.

However, Google will counter Epic’s arguments by pointing out that it has rolled out a new option for app developers called User Choice Billing, which Epic has
declined to use.
</item 16>

<item 17>
Epic v. Google, explained (The Verge, 2023-11-05T11:00:00+00:00)
That sounds fascinating, but if I don’t care about Fortnite or rich companies suing each other, what’s in it for me?

The future of Google’s app store could depend on this trial — both Epic and Google agree on that. Epic wants to break up Google’s alleged monopoly on Android
app stores and payment methods, so developers aren’t stuck paying the “Google Tax” or passing that fee along to you.

But if Epic wins — according to Google, anyhow — it could make Android phones less safe by dismantling basic protections against sideloaded apps, and damage
Android’s ability to compete with the iPhone because it (arguably) can’t run a competitive app store by giving it away for free.

Or maybe you just want juicy dirt on both companies? More on that later!

What exactly is Epic claiming?

Epic argues that Google makes it so difficult for developers and users to get around its Android app store, and that app store’s standard fees, that it’s
created an illegal monopoly that unjustly enriches Google while artificially driving up app prices because other stores can’t compete.
</item 17>

<item 18>
Epic v. Google, explained (The Verge, 2023-11-05T11:00:00+00:00)
But if you do want to see a vision of a future that Epic would accept, look no further than this letter it sent to state attorneys general — laying out a
settlement that would basically prohibit Google from doing the anticompetitive things that Epic claims it did.

There’s also this tweet from Sweeney:

Google wouldn’t tell The Verge if it’s offered Epic any kind of settlement yet.

Why if I just want the dirty laundry on both these companies? You promised dirt...

Oh, it could get good.

The allegations made before trial suggest Google was playing loads of dirty tricks with names like “Project Hug,” “Project Agave,” and “Project Banyan”, with
Epic claiming it’s seen evidence Google was paying off game developers and phone makers not to abandon its app store — to the tune of hundreds of millions of
dollars. Google allegedly even toyed with the idea of buying Epic to keep Fortnite from inspiring other game developers to abandon ship.

Plus, there’s evidence Google deleted potentially incriminating messages to keep them out of a court’s hands.
</item 18>

<item 19>
What to know about Fortnite maker Epic Games’ antitrust battle with Google, starting today (TechCrunch, 2023-11-06T16:47:11+00:00)
As an example, Epic will point to Activision Blizzard, the owner of King.com and maker of Candy Crush and Call of Duty, as one developer that wanted to go the
route of opening its own app store. Google and Activision Blizzard then entered into a three-year agreement in January 2020, worth approximately $360 million,
to keep the developer on Google Play, Epic will argue.

Google, however, will characterize this program as a way to incentivize developers to launch their apps on Android at the same time as they launch on iOS and
plans to dispute Epic’s claims that it prevented developers from opening their own app stores.

Android allows sideloading

Google will make note that, unlike Apple, it also allows apps to be sideloaded on Android devices — something Epic takes advantage of today. It will note this
process is streamlined and helps keep users safe.
</item 19>

<item 20>
Tim Sweeney on Epic’s victory royale over Google (The Verge, 2023-12-12T19:17:14+00:00)
It’s always been in Google’s power to solve this problem. They make several billion dollars a year in unfairly earned profits from imposing this tax, which is
nothing compared to the money they make from search. For all the other benefits they get from Android, Google could solve this problem today if they wanted to.

Do you today believe that Activision Blizzard was really intending to build its own app store?

I don’t know about Supercell, but we know from the documents in the case that Riot was planning to distribute League of Legends directly through their website,
on mobile, exactly as they do on PC. And that’s what they planned to do until Google paid them off to not do that. Google’s payoff at the minimum dissuaded Riot
from distributing off Google Play.

I think they’re the strongest example of the three, but I’m curious about Activision.

Activision was building a store! We know they had a large team of dozens of engineers building a mobile app store to launch on Android.
</item 20>
=== END ITEM id=1f335281acad8879 ===

=== ITEM id=7f4aa2a908914ef7 ===
Gold sentences and facts:
Sentence 0: “She’s got the Eras Tour booked all through next year, and she’d rather die than cancel a show.
  Fact 0.0: The person quoted has the Eras Tour booked all through next year.
  Fact 0.1: The person quoted would rather die than cancel a show.
Sentence 1: "It was almost like this very strange, subtle clue to the media that they don't know everything that happened in that relationship, and I can have something really major and traumatic happen to me and they don't know about it," she told NPR.
  Fact 1.0: The speaker said it was almost like a very strange, subtle clue to the media that they don't know everything that happened in that relationship.
  Fact 1.1: The speaker said she can have something really major and traumatic happen to her and they don't know about it.
  Fact 1.2: The speaker told NPR this.
Sentence 2: Now, Swift has shared that she has nothing to hide in her new relationship with Kelce.
  Fact 2.0: Taylor Swift has shared that she has nothing to hide in her new relationship with Kelce.

Retrieved items:
<item 1>
Taylor Swift reveals secret start to her relationship with Travis Kelce (The Independent - Life and Style, 2023-12-06T14:23:01+00:00)
“This all started when Travis very adorably put me on blast on his podcast, which I thought was metal as hell,” she explained. “We started hanging out right
after that. So we actually had a significant amount of time that no one knew, which I’m grateful for, because we got to get to know each other.”

Swift also confirmed that when she famously first attended Kelce’s game at Arrowhead Stadium in September, she was already dating the NFL star.

“By the time I went to that first game, we were a couple. I think some people think that they saw our first date at that game? We would never be psychotic
enough to hard launch a first date,” she said.

The “Blank Space” singer also acknowledged that although her love life continues to make headlines with every Kansas City Chiefs game she attends, she’s focused
on being there for her partner. She specified that she isn’t trying to hide her relationship from the public eye.
</item 1>

<item 2>
Taylor Swift's 1989: The stories behind her biggest album (BBC News - Entertainment & Arts, 2023-10-13T23:00:08+00:00)
"I would never have said anything like that on a previous album," she told Ryan Seacrest.

"My previous albums have always been, 'I was right, you were wrong'. What happens when you grow up is you realise that the rules in a relationship are very
blurred. It gets very complicated, very quickly."

4) Out Of The Woods

One of 1989's strengths is how the music reflects the emotional tenor of the words. Here, that means frantic anxiety.

As Taylor and her partner run from forces that seek to destroy them, they're pursued by pounding drums and rushing synths. And there's a desperation to her
voice, as though she's repeating "everything's fine, everything's fine, everything's fine," against evidence to the contrary.

By the middle 8, it seems they've made a clean getaway. Then car brakes are slammed, and it's "twenty stitches in a hospital room".

In interviews, Taylor delighted in revealing this was based on a true story.
</item 2>

<item 3>
Taylor Swift's 1989: The stories behind her biggest album (BBC News - Entertainment & Arts, 2023-10-13T23:00:08+00:00)
It's certainly not her finest work - a vendetta disguised as a pop song, with a ham-fisted chorus that lacks her usual melodic instincts.

Swift hinted in a Rolling Stone interview that the track was about a high-profile female musician, saying the person did something "so horrible" that made them
"straight-up enemies."

Although she tried not to identify the target, it became common knowledge that Katy Perry was in the crosshairs, after allegedly "stealing" Taylor's backing
dancers just before a tour.

Perry confirmed the situation on James Corden's Carpool Karaoke, adding that she "tried to talk to [Swift] about it and she wouldn't speak to me."

The pop divas eventually reconciled in the 2019 video for You Need To Calm Down... but Taylor's still performing Bad Blood on her Eras tour.
</item 3>

<item 4>
Taylor Swift is Time’s Person of the Year (The Independent - Life and Style, 2023-12-06T13:55:17+00:00)
“I think some people think that they saw our first date at that game? We would never be psychotic enough to hard launch a first date,” Swift joked.

Back in April, it was revealed that Swift and British actor Joe Alwyn had broken up after six years of dating. The former couple were notoriously private, as
the Conversations with Friends star previously expressed he didn’t want to discuss their relationship because it “feeds into a weird part of the culture”.

Now, Swift has shared that she has nothing to hide in her new relationship with Kelce. “When you say a relationship is public, that means I’m going to see him
do what he loves, we’re showing up for each other, other people are there and we don’t care,” she said. “The opposite of that is you have to go to an extreme
amount of effort to make sure no one knows that you’re seeing someone. And we’re just proud of each other.”
</item 4>

<item 5>
Taylor Swift's 1989: The stories behind her biggest album (BBC News - Entertainment & Arts, 2023-10-13T23:00:08+00:00)
No-one, it turns out.

The song is thematically similar to Blank Space, commenting on the media's portrayal of Taylor's relationships: "But if I'm all dressed up / They might as well
be lookin' at us / And if they call me a slut / You know it might be worth it for once."

The best of the new tracks is Is It Over Now. An angrier take on 1989's philosophical break-up songs, it calls back to the Harry Styles snowmobile incident (see
also, Out Of The Woods) then, deliciously, references the "Sad Taylor Boat" meme - a photo of Taylor fleeing a vacation alone after the couple split up.

She goes on to call an ex a "lying traitor", accuses them of parading their new relationships in public, and noting that every new girlfriend looks like her.

"If she's got blue eyes, I will surmise that you'll probably date her."
</item 5>

<item 6>
Taylor Swift is Time’s Person of the Year (The Independent - Life and Style, 2023-12-06T13:55:17+00:00)
The decision came after one of the singer’s fans, 23-year-old Ana Clara Benevides, died from a cardiac arrest during Swift’s first performance in the country.
Swift explained to Time that “part of my identity as a human” is putting on a show for her fans, no matter the circumstances. “I know I’m going on that stage
whether I’m sick, injured, heartbroken, uncomfortable, or stressed,” she said. “That’s part of my identity as a human being now. If someone buys a ticket to my
show, I’m going to play it unless we have some sort of force majeure.”

Perhaps it was her decision to re-record her music catalogue that catapulted Swift into such success. When music mogul Scooter Braun aquired Swift’s longtime
label, Big Machine Records, in 2019, he also gained the rights to the master recordings of Swift’s first six studio albums. In November 2020, Braun sold the
masters to an investment fund in a deal that was believed to be over $300 million.
</item 6>

<item 7>
Taylor Swift-Travis Kelce timeline: Everything to know about rumored romance between pop star, Chiefs TE (CBSSports.com, 2023-10-12T23:21:52+00:00)
- Enchanted

Mahomes said he had not met Swift when asked in the postgame press conference, but he did meet her at the afterparty.

"Yea I met her. She's really cool. Good people," Mahomes said, via KSHB 41 in Kansas City. "But like Trav said, imma let them have their privacy and just keep
it moving."

Sept. 27: Kelce breaks his silence

"And they said speak now." -Speak Now

Kelce addressed Swift's attendance on his podcast, saying, "To see the slow-motion chest bumps, to see the high-fives with Mom, to see how Chiefs Kingdom was
all excited that she was there -- that s--- was absolutely hysterical. It's definitely a game I'll remember, that's for damn sure."

He is not going to speak too much more about the relationship, noting that he wants "to respect both of our lives."

"Yeah, my personal life -- that's not so personal.
</item 7>

<item 8>
Taylor Swift reveals secret start to her relationship with Travis Kelce (The Independent - Life and Style, 2023-12-06T14:23:01+00:00)
Elsewhere in the conversation with Time, the “Anti-Hero” singer said that she tries not to let the paparazzi get to her, even though so many of her outings with
pals go on to make headlines. “Over the years, I’ve learned I don’t have the time or bandwidth to get pressed about things that don’t matter. Yes, if I go out
to dinner, there’s going to be a whole chaotic situation outside the restaurant. But I still want to go to dinner with my friends,” she said.

She emphasised how her perspective about the media has changed over the years, adding: “Life is short. Have adventures. Me locking myself away in my house for a
lot of years - I’ll never get that time back. I’m more trusting now than I was six years ago.”

With his love life in the public eye, Kelce has also opened up about his relationship with Swift.
</item 8>

<item 9>
Taylor Swift-Travis Kelce timeline: Everything to know about rumored romance between pop star, Chiefs TE (CBSSports.com, 2023-10-12T23:21:52+00:00)
"Yeah, my personal life -- that's not so personal. I did this to myself Jason, I know this. What's real is that you know it is my personal life. ... She's not
in the media as much as I am doing this show every single week."

Sept. 27: Jerry Jones comments on the Swift effect

Swift is getting the owners talking, too.

"It is hard to move that needle dramatically on a great feeling, a great attraction in the NFL," Jones said Wednesday on 105.3FM The Fan in Dallas. "We get a
lot of eye balls. She blew it off the charts. It is amazing. It makes you smile. But it just shows the depth of her following, especially with her core
constituency. We all, I think, felt the same way when she's sitting up there with [Travis Kelce's] mom and she's going nuts when he's playing and those hearts
are popping out everywhere.
</item 9>

<item 10>
Taylor Swift is Time’s Person of the Year (The Independent - Life and Style, 2023-12-06T13:55:17+00:00)
Stay ahead of the trend in fashion and beyond with our free weekly Lifestyle Edit newsletter Stay ahead of the trend in fashion and beyond with our free weekly
Lifestyle Edit newsletter Please enter a valid email address Please enter a valid email address SIGN UP I would like to be emailed about offers, events and
updates from The Independent. Read our privacy notice Thanks for signing up to the

Lifestyle Edit email {{ #verifyErrors }} {{ message }} {{ /verifyErrors }} {{ ^verifyErrors }} Something went wrong. Please try again later {{ /verifyErrors }}

Taylor Swift has officially been named Time’s 2023 Person of the Year.

On Wednesday 6 December, the magazine announced Swift had earned the honour after an “editors’ assessment of the individual who most shaped the headlines over
the previous 12 months, for better or for worse”.

The recognition of Swift comes on the heels of a whirlwind year for the popstar, whose Eras Tour dominated ticket sales in 2023.
</item 10>

<item 11>
Taylor Swift is Time’s Person of the Year (The Independent - Life and Style, 2023-12-06T13:55:17+00:00)
And we’re just proud of each other.”

Speaking to Time, Swift detailed exactly how much work went into preparing for her Eras Tour - where each concert spans over 180 minutes, with more than 40
songs from at least nine albums and 16 costume changes. “Every day I would run on the treadmill, singing the entire set list out loud,” she told the outlet.
“Fast for fast songs, and a jog or a fast walk for slow songs.”

She also stopped drinking and underwent three months of dance training, something that she admitted is “not my strong suit”. After she plays a run of shows,
Swift said she takes a day to rest and recover before taking the stage once again. “I do not leave my bed except to get food and take it back to my bed and eat
it there,” she revealed. “It’s a dream scenario.”

Most recently, a heat wave in Brazil forced Swift to postpone the second night of her Eras Tour shows in Rio de Janeiro.
</item 11>

<item 12>
Taylor Swift is Time’s Person of the Year (The Independent - Life and Style, 2023-12-06T13:55:17+00:00)
Not to mention her rumoured relationship with Kansas City Chiefs tight end Travis Kelce caused a 400 per cent spike in NFL jersey sales.

Since her first NFL appearance in September, Swift’s occasional attendance at Kelce’s games has received pushback from NFL fans who believe her presence is
distracting from the game itself. “I’m just there to support Travis,” Swift admitted. “I have no awareness of if I’m being shown too much and pissing off a few
dads, Brads, and Chads.”

She playfully added: “Football is awesome, it turns out. I’ve been missing out my whole life.”

The “Bad Blood” singer revealed that their romance began when Kelce gave her shoutout on his podcast, admitting he attempted to give Swift his phone number
after one of her Eras Tour shows. The pair immediately began “hanging out” and by the time she attended her first Chiefs game in September, they were already a
couple. “I think some people think that they saw our first date at that game?
</item 12>

<item 13>
Taylor Swift's 1989: The stories behind her biggest album (BBC News - Entertainment & Arts, 2023-10-13T23:00:08+00:00)
Taylor Swift's biggest album 1989 returns with new tracks from the vault Published 27 October

Image source, Getty Images Image caption, Taylor Swift's 1989 tour sold 2.2m tickets and earned more than $250m at the box office

On Friday, Taylor Swift released a new version of 1989 - the biggest-selling album of her career, and the one that definitively turned her into a pop star.

Featuring hits like Shake It Off, Blank Space and Style, it was originally written during the 2013-14 Red Tour, with demos stored on her phone in a folder named
"Sailor Twips".

Awarded a Grammy for album of the year, it has spent 325 weeks in the UK charts.

But now she has re-recorded it as the latest part of an ongoing campaign to regain control of her work, after an investment company bought her master tapes in
2019.

This is the biggest and riskiest part of the project.
</item 13>

<item 14>
Taylor Swift's 1989: The stories behind her biggest album (BBC News - Entertainment & Arts, 2023-10-13T23:00:08+00:00)
"A farewell to twang," the New York Times called it.

Taylor knew change was necessary. "I don't have the option of making music that sounds just like what I've done before," she told Popcrush. "People will see
right through it. They'll see that I was lazy".

Her label begged her to add three country-leaning songs to the album, but she stood her ground.

"At a certain point, if you chase two rabbits, you lose them both."

Welcome To New York itself is the sound of creative and personal freedom. Taylor moved to the city in 2014 and was amazed to discover she could go grocery
shopping largely incognito.

"My friends and people who know me have commented that I'm physically different since being here," she told NYC radio station WKTU. "That song is about the
endless hopefulness and endless possibilities that New York seems to present," she added in an interview with Access Hollywood.

All proceeds from the track are donated to New York City's Public Schools.
</item 14>

<item 15>
Taylor Swift's 1989: The stories behind her biggest album (BBC News - Entertainment & Arts, 2023-10-13T23:00:08+00:00)
"If she's got blue eyes, I will surmise that you'll probably date her."

The fact that all the new songs were co-written and produced by Jack Antonoff suggests there may have been rights issues over the tracks she wrote with Max
Martin - who has not been involved with the new re-recordings.

That means the vault tracks are all bathed in the dreamy, soft-focus atmospherics of Out Of The Woods, rather than the upfront pop of Shake It Off.

Speaking in 2015, Taylor admitted she'd been brutal about pruning songs from 1989's original tracklist.

"There were so many songs I wrote for this album that were really good, but if I felt in any way that they would have belonged on my last album, Red, they got
cut," she told New York radio's The Todd Show in the Morning.

"There are really, really good songs on the cutting room floor because they weren't innovative enough or new enough.
</item 15>

<item 16>
Taylor Swift is Time’s Person of the Year (The Independent - Life and Style, 2023-12-06T13:55:17+00:00)
Since then, Swift has released four “Taylor’s Version” albums: Fearless (Taylor’s Version) in April 2021, Red (Taylor’s Version) in November 2021, Speak Now
(Taylor’s Version) in July 2023, and 1989 (Taylor’s Version) in October 2023.

“It’s not lost on me that the two great catalysts for this happening were two horrendous things that happened to me,” Swift said, when asked about the success
of her tour. “The first was getting cancelled within an inch of my life and sanity. The second was having my life’s work taken away from me by someone who hates
me.”

While Swift is arguably more famous now than ever before, the Grammy winner revealed that the constant attention and swarms of paparazzi won’t stop her from
living her life. There’s also constant rumours surrounding her personal life, specifically rumours about her relationship with Alwyn that her publicist, Tree
Paine, was forced to publicly deny.
</item 16>

<item 17>
Taylor Swift is Time’s Person of the Year (The Independent - Life and Style, 2023-12-06T13:55:17+00:00)
“Over the years, I’ve learned I don’t have the time or bandwidth to get pressed about things that don’t matter,” Swift told Time. “Yes, if I go out to dinner,
there’s going to be a whole chaotic situation outside the restaurant. But I still want to go to dinner with my friends.

“Life is short. Have adventures. Me locking myself away in my house for a lot of years - I’ll never get that time back,” she added. “I’m more trusting now than
I was six years ago.”
</item 17>

<item 18>
Taylor Swift's 1989: The stories behind her biggest album (BBC News - Entertainment & Arts, 2023-10-13T23:00:08+00:00)
Like, through thick and thin, sickness and health love," she told MTV.

Drawing directly on conversations she'd had with Lena, the song blends scenes of everyday life (burnt toast, borrowed shirts) with moments of incandescent
tenderness (kisses on sidewalks and late night heart-to-hearts).

In an interview with Elle, Taylor talked wistfully about the couple's ease around each other.

"I've never had that," she said. "[But] now I have more of a grasp on the fact that when you're in a state of infatuation and you think everything that person
does is perfect, it then - if you're lucky - morphs into a real relationship when you see that that person is not in fact perfect, but you still want to see
them every day."
</item 18>

<item 19>
Taylor Swift is Time’s Person of the Year (The Independent - Life and Style, 2023-12-06T13:55:17+00:00)
“This is the proudest and happiest I’ve ever felt, and the most creatively fulfilled and free I’ve ever been,” Swift told Time. “Ultimately, we can convolute it
all we want, or try to overcomplicate it, but there’s only one question... Are you not entertained?”

Swift was chosen among a strong list of contenders, including Chinese President Xi Jinping, King Charles III, the live-action Barbie movie, and Vladimir Putin.
“While her popularity has grown across the decades, this is the year that Swift, 33, achieved a kind of nuclear fusion: shooting art and commerce together to
release an energy of historic force,” the magazine said about its selection.

This year alone, Swift has broken records with her career-spanning Eras Tour, which is estimated to have grossed over $1.4bn. Her concert film, the Eras Tour
movie, has surpassed $250m at the global box office since its opening weekend in October.
</item 19>

<item 20>
Taylor Swift's 1989: The stories behind her biggest album (BBC News - Entertainment & Arts, 2023-10-13T23:00:08+00:00)
In interviews, Taylor delighted in revealing this was based on a true story.

"It was almost like this very strange, subtle clue to the media that they don't know everything that happened in that relationship, and I can have something
really major and traumatic happen to me and they don't know about it," she told NPR.

In her Rolling Stone cover story, Taylor said the crash actually occurred on a snowmobile "with an ex who lost control and wrecked it so badly that she saw her
life flash before her eyes".

That prompted speculation the track was also about Styles: The couple had gone skiing with Justin Bieber and Selena Gomez in late 2012.

The lyrics also reference their identical aeroplane necklaces and a December break-up, which ties in with the "Sad Taylor on a boat" meme.

But the singer steadfastly refuses to name names: "The fact that I've never confirmed who those songs are about makes me feel like there is still one card I'm
holding," she told GQ.
</item 20>

<item 21>
Taylor Swift reveals secret start to her relationship with Travis Kelce (The Independent - Life and Style, 2023-12-06T14:23:01+00:00)
The “All Too Well” singer’s interview with Time marks her first time publicly speaking out about Kelce, who she’s been romantically linked to since this summer,
when the football tight end first confessed to attempting to give her his number.

While Swift has attended a few of Kelce’s games since their relationship started, he’s also gone on to support her career. In November, he made his way to
Buenos Aires, Argentina, to attend one of her Eras Tour shows. During the concert, Swift gave a shout-out to the Chiefs player when she changed her “Karma”
lyrics to reflect their relationship. “Karma is the guy on the Chiefs, coming straight home to me,” she sang, letting out a laugh as fans in the audience
screamed in response.

After the show, the “Style” singer was seen running into Kelce’s arms and kissing him, before he whisked her away backstage.
</item 21>

<item 22>
Taylor Swift-Travis Kelce timeline: Everything to know about rumored romance between pop star, Chiefs TE (CBSSports.com, 2023-10-12T23:21:52+00:00)
Some found issue with these changes, but the NFL said this is a "pop cultural moment" and they are acting as such.

"We frequently change our bios and profile imagery based on what's happening in and around our games, as well as culturally," the NFL wrote in a statement. "The
Taylor Swift and Travis Kelce news has been a pop cultural moment we've leaned into in real time, as it's an intersection of sport and entertainment, and we've
seen an incredible amount of positivity around the sport."

Oct. 6: Travis and Jason think the NFL is doing too much

The NFL is really digging into the Swift-Kelce coverage, and not everyone is a fan, including the people involved.

"I think it's fun when they show who's at the game," Travis said. "I think it brings a little bit more to the atmosphere, brings a little bit more to what
you're watching. But at the same time, they're overdoing it a little bit."
</item 22>

<item 23>
When pop culture and sport collide: A timeline of Taylor Swift’s NFL takeover (The Age, 2023-09-26T05:26:05+00:00)
Suddenly, football fans and Swifties were living through this surprising collision. Swift lyrics were infiltrating NFL game commentary, and her fans were
furiously becoming football experts. All that needed to happen was for Swift herself to infiltrate a game, determining once and for all whether there was
anything to these rumours or not. This Is Me Trying

On September 21, Kelce announced on The Pat McAfee Show he had invited Swift to watch one of his games. “I told her, ‘I’ve seen you rock the stage in Arrowhead,
you might have to come see me rock a stage at Arrowhead and we can see which one’s a little more lit,’” he said. “We’ll see what happens in the near future.”
Were they finally about to hard-launch their friendship/potential romance to the world, laying all speculation to rest? Love Story This finally brings us to the
fated sighting of Swift at Arrowhead Stadium on Monday, cheering on No. 87 with gusto.
</item 23>

<item 24>
Taylor Swift reveals secret start to her relationship with Travis Kelce (The Independent - Life and Style, 2023-12-06T14:23:01+00:00)
Stay ahead of the trend in fashion and beyond with our free weekly Lifestyle Edit newsletter Stay ahead of the trend in fashion and beyond with our free weekly
Lifestyle Edit newsletter Please enter a valid email address Please enter a valid email address SIGN UP I would like to be emailed about offers, events and
updates from The Independent. Read our privacy notice Thanks for signing up to the

Lifestyle Edit email {{ #verifyErrors }} {{ message }} {{ /verifyErrors }} {{ ^verifyErrors }} Something went wrong. Please try again later {{ /verifyErrors }}

Taylor Swift has revealed when her relationship with Travis Kelce really started.

The 33-year-old singer spoke candidly about the romance during a recent interview with Time after she was named the magazine’s 2023 Person of the Year. Speaking
to the publication, she said she connected with Kelce in July, after he confessed on his podcast that he tried and failed to give her a friendship bracelet with
his phone number on it during one of her Eras Tour shows.
</item 24>

<item 25>
Taylor Swift reveals secret start to her relationship with Travis Kelce (The Independent - Life and Style, 2023-12-06T14:23:01+00:00)
In an interview withWSJ Magazine, published on 20 November, he described how much he admires her skills as a musician. “I’ve never been a man of words. Being
around her, seeing how smart Taylor is, has been f***ing mind-blowing. I’m learning every day,” he explained.

However, he also acknowledged the heavy public attention around his private life as a result of his relationship, noting he’s “never dated anyone with that kind
of aura around them”. But, according to Kelce, he’s not letting the media affect his relationship.

“But at the same time, I’m not running away from any of it,” he said. “The scrutiny she gets, how much she has a magnifying glass on her, every single day,
paparazzi outside her house, outside every restaurant she goes to, after every flight she gets off, and she’s just living, enjoying life. When she acts like
that I better not be the one acting all strange.”
</item 25>

<item 26>
Travis Kelce says he tried to delete his ‘nonsense’ tweets before they went viral (The Independent - Life and Style, 2023-11-25T14:50:53+00:00)
In an interview withWSJ Magazine published on 20 November, Kelce described how much he admires Swift’s skills as a musician. “I’ve never been a man of words.
Being around her, seeing how smart Taylor is, has been f***ing mind-blowing. I’m learning every day,” he explained.

Although he acknowledged the public attention his personal life has received because he’s “never dated anyone with that kind of aura around them,” Kelce
specified that he’s not letting the media affect his relationship.

“But at the same time, I’m not running away from any of it,” he said. “The scrutiny she gets, how much she has a magnifying glass on her, every single day,
paparazzi outside her house, outside every restaurant she goes to, after every flight she gets off, and she’s just living, enjoying life.

“When she acts like that I better not be the one acting all strange.”
</item 26>

<item 27>
Taylor Swift's 1989: The stories behind her biggest album (BBC News - Entertainment & Arts, 2023-10-13T23:00:08+00:00)
All proceeds from the track are donated to New York City's Public Schools.

Image source, Getty Images Image caption, The original version of 1989 won three Grammy Awards, including Album of the Year

2) Blank Space

Pretty much the perfect Taylor Swift song, Blank Space takes every horrible rumour that was being spread about her love life, and amplifies it with satirical
relish.

"In the last couple of years the media have had a wonderful fixation on painting me as the psycho serial dater girl," she told the Grammy Museum. "Every article
was like, 'Here's Taylor Swift standing near some guy. WATCH OUT GUY!'

"My first reaction was to be like, 'That's a bummer. This isn't fun for me'.
</item 27>

<item 28>
Taylor Swift reveals secret start to her relationship with Travis Kelce (The Independent - Life and Style, 2023-12-06T14:23:01+00:00)
She specified that she isn’t trying to hide her relationship from the public eye.

“When you say a relationship is public, that means I’m going to see him do what he loves, we’re showing up for each other, other people are there and we don’t
care,” she said. “The opposite of that is you have to go to an extreme amount of effort to make sure no one knows that you’re seeing someone. And we’re just
proud of each other.”

During the interview with Time, the Grammy winner addressed how her appearances at Kelce’s games are heavily publicised. So far, she’s attended five of Kelce’s
games, including one on 3 December, and she’s often seated in the Kelce family suite during the events.

“There’s a camera, like, a half-mile away, and you don’t know where it is, and you have no idea when the camera is putting you in the broadcast, so I don’t know
if I’m being shown 17 times or once,” she said.
</item 28>

<item 29>
Taylor Swift's 1989: The stories behind her biggest album (BBC News - Entertainment & Arts, 2023-10-13T23:00:08+00:00)
Those search engine clicks gotta be lucrative, right?

Image source, Getty Images Image caption, Many of 1989's songs were supposedly inspired by Taylor's relationship with Harry Styles

3) Style

An epic aural painting of windswept coastlines and propulsive wanderlust, Style is (supposedly) named for Harry Styles, with whom Taylor had a brief
relationship in 2012.

Except she says it was inspired by fashion, and how certain items like a black dress or red lipstick will "never go out of style". That prompted her to write
about the people who'll always be part of your life, even after the flame of romantic love wanes (all of which sounds like a classic piece of deflection to a
probing journalist).

Whichever interpretation you prefer, the song marks a shift in Swift's psychology, as she briefly admits to having been unfaithful.

"I would never have said anything like that on a previous album," she told Ryan Seacrest.
</item 29>

<item 30>
Taylor Swift's 1989: The stories behind her biggest album (BBC News - Entertainment & Arts, 2023-10-13T23:00:08+00:00)
It's so weird to look back on that - you've got heaping piles of discarded songs that I really loved when I wrote them and I still love now.

"I just need each album to have its own sound and if it doesn't fit that sound, it doesn't matter if it's a great song to me."
</item 30>
=== END ITEM id=7f4aa2a908914ef7 ===

=== ITEM id=44dedc4d346c2faf ===
Gold sentences and facts:
Sentence 0: He revealed that he made a friendship bracelet, a fan tradition at her concerts, with his number on it to give to her, seemingly expressing his interest in Swift.
  Fact 0.0: He revealed that he made a friendship bracelet.
  Fact 0.1: The friendship bracelet is a fan tradition at Swift's concerts.
  Fact 0.2: The friendship bracelet had his number on it.
  Fact 0.3: He made the friendship bracelet to give to Swift.
  Fact 0.4: Making the friendship bracelet seemingly expressed his interest in Swift.
Sentence 1: On his podcast, he later invited the “Anti-Hero” singer to one of his games at Arrowhead Stadium - an invitation she famously accepted on 24 September.
  Fact 1.0: He later invited the 'Anti-Hero' singer on his podcast to one of his games at Arrowhead Stadium.
  Fact 1.1: The 'Anti-Hero' singer famously accepted the invitation on 24 September.
Sentence 2: Now, Swift has shared that she has nothing to hide in her new relationship with Kelce.
  Fact 2.0: Swift has now shared that she has nothing to hide.
  Fact 2.1: Swift has shared that she has nothing to hide in her relationship with Kelce.
  Fact 2.2: Swift's relationship with Kelce is new.

Retrieved items:
<item 1>
Taylor Swift reveals secret start to her relationship with Travis Kelce (The Independent - Life and Style, 2023-12-06T14:23:01+00:00)
Stay ahead of the trend in fashion and beyond with our free weekly Lifestyle Edit newsletter Stay ahead of the trend in fashion and beyond with our free weekly
Lifestyle Edit newsletter Please enter a valid email address Please enter a valid email address SIGN UP I would like to be emailed about offers, events and
updates from The Independent. Read our privacy notice Thanks for signing up to the

Lifestyle Edit email {{ #verifyErrors }} {{ message }} {{ /verifyErrors }} {{ ^verifyErrors }} Something went wrong. Please try again later {{ /verifyErrors }}

Taylor Swift has revealed when her relationship with Travis Kelce really started.

The 33-year-old singer spoke candidly about the romance during a recent interview with Time after she was named the magazine’s 2023 Person of the Year. Speaking
to the publication, she said she connected with Kelce in July, after he confessed on his podcast that he tried and failed to give her a friendship bracelet with
his phone number on it during one of her Eras Tour shows.
</item 1>

<item 2>
Taylor Swift-Travis Kelce timeline: Everything to know about rumored romance between pop star, Chiefs TE (CBSSports.com, 2023-10-12T23:21:52+00:00)
- Enchanted

Mahomes said he had not met Swift when asked in the postgame press conference, but he did meet her at the afterparty.

"Yea I met her. She's really cool. Good people," Mahomes said, via KSHB 41 in Kansas City. "But like Trav said, imma let them have their privacy and just keep
it moving."

Sept. 27: Kelce breaks his silence

"And they said speak now." -Speak Now

Kelce addressed Swift's attendance on his podcast, saying, "To see the slow-motion chest bumps, to see the high-fives with Mom, to see how Chiefs Kingdom was
all excited that she was there -- that s--- was absolutely hysterical. It's definitely a game I'll remember, that's for damn sure."

He is not going to speak too much more about the relationship, noting that he wants "to respect both of our lives."

"Yeah, my personal life -- that's not so personal.
</item 2>

<item 3>
Taylor Swift-Travis Kelce timeline: Everything to know about rumored romance between pop star, Chiefs TE (CBSSports.com, 2023-10-12T23:21:52+00:00)
Here's a look at how it all started and where it has gone since:

June 29: Kelce makes friendship bracelets for fellow TEs

Is this where it all begin? Kelce made a friendship bracelet to give to the newcomers at a camp for NFL tight ends. He said he was "taking a page out of Taylor
Swift's book."

July 26: Kelce makes friendship bracelet for Swift

"So make the friendship bracelets, take the moment and taste it." - You're On Your Own Kid

Kelce announced that he went to Swift's highly anticipated concert, but said he did not have the chance to meet the 12-time Grammy winner.
</item 3>

<item 4>
Taylor Swift-Travis Kelce timeline: Everything to know about rumored romance between pop star, Chiefs TE (CBSSports.com, 2023-10-12T23:21:52+00:00)
Singer Taylor Swift has a blank space, and she's writing Travis Kelce's name. There were rumors the two were dating for weeks, and those talks only intensified
with swift attending five Chiefs games so far this season, including Sunday's game agains the Chargers.

The Era's Tour singer was seen chatting with the tight end's mom, Donna Kelce, and cheering emphatically when No. 87 scored a touchdown in the first game she
attended last month. Last week, she was in attendance to watch the Chiefs as they defeated the Denver Broncos in Week 6 on "Thursday Night Football."

She is once again at Arrowhead Stadium, this time to watch the Chiefs take on another AFC West opponent, the Los Angeles Chargers.

For those wondering how this apparent relationship came to be, wonder no longer. We have created an extensive timeline of the high-profile couple that is
currently the talk of the league, and even added some Swift lyrics to go along with some of the highlights.
</item 4>

<item 5>
Taylor Swift reveals secret start to her relationship with Travis Kelce (The Independent - Life and Style, 2023-12-06T14:23:01+00:00)
In an interview withWSJ Magazine, published on 20 November, he described how much he admires her skills as a musician. “I’ve never been a man of words. Being
around her, seeing how smart Taylor is, has been f***ing mind-blowing. I’m learning every day,” he explained.

However, he also acknowledged the heavy public attention around his private life as a result of his relationship, noting he’s “never dated anyone with that kind
of aura around them”. But, according to Kelce, he’s not letting the media affect his relationship.

“But at the same time, I’m not running away from any of it,” he said. “The scrutiny she gets, how much she has a magnifying glass on her, every single day,
paparazzi outside her house, outside every restaurant she goes to, after every flight she gets off, and she’s just living, enjoying life. When she acts like
that I better not be the one acting all strange.”
</item 5>

<item 6>
Taylor Swift reveals secret start to her relationship with Travis Kelce (The Independent - Life and Style, 2023-12-06T14:23:01+00:00)
“This all started when Travis very adorably put me on blast on his podcast, which I thought was metal as hell,” she explained. “We started hanging out right
after that. So we actually had a significant amount of time that no one knew, which I’m grateful for, because we got to get to know each other.”

Swift also confirmed that when she famously first attended Kelce’s game at Arrowhead Stadium in September, she was already dating the NFL star.

“By the time I went to that first game, we were a couple. I think some people think that they saw our first date at that game? We would never be psychotic
enough to hard launch a first date,” she said.

The “Blank Space” singer also acknowledged that although her love life continues to make headlines with every Kansas City Chiefs game she attends, she’s focused
on being there for her partner. She specified that she isn’t trying to hide her relationship from the public eye.
</item 6>

<item 7>
Taylor Swift-Travis Kelce timeline: Everything to know about rumored romance between pop star, Chiefs TE (CBSSports.com, 2023-10-12T23:21:52+00:00)
24: Swift mingles with Donna Kelce



"Talkin' over a football team." - Mean

Swift already seems to be making friends with the family. Swift was sitting next to Donna in the box and the two were seen talking and laughing as the game went
along.

Sept. 24: Postgame interviews turn into Swiftie talk

The postgame press conference seemed more about Swift and Kelce than it did about the actual game. The media wanted any insight it could get into the
relationship and how Swift ended up at Arrowhead.

"He told me at the last minute," Mahomes said when asked if he knew Swift was coming. "There's some things with Trav where he kind of just says it and you don't
know if it's true or not. He says it so calmly. ... Friday he was like, 'Yeah, I think she's coming to the game this weekend,' and then moved about his
business."
</item 7>

<item 8>
Taylor Swift-Travis Kelce timeline: Everything to know about rumored romance between pop star, Chiefs TE (CBSSports.com, 2023-10-12T23:21:52+00:00)
We will see what happens in the near future."

Sept. 22: Another bracelet

"Lately [he's] been dressing for revenge." - Vigilante Sh--

Kelce was spotted at his nonprofit charity event wearing a bracelet that says "Vigilante Sh--," the name of one of Swift's songs.

Sept. 24: Swift attends Chiefs game

"Dating the boy on the football team." - Fifteen

Swift took Kelce up on his offer and attended the Chiefs' Week 3 game against the Chicago Bears. Swift was dressed on theme and was watching intently, cheering
and even chest bumping someone during big plays. She was especially thrilled to see Kelce get in the end zone, a three-yard catch that put the Chiefs up 41-0.
Kansas City went on to win 41-10 to improve to 2-1.

Sept. 24: Swift mingles with Donna Kelce



"Talkin' over a football team."
</item 8>

<item 9>
Taylor Swift-Travis Kelce timeline: Everything to know about rumored romance between pop star, Chiefs TE (CBSSports.com, 2023-10-12T23:21:52+00:00)
But before he could even answer, the answer was given for him.

Sept. 14: Rumors aren't denied

"The rumors are terrible and cruel, but honey, most of them are true." - New Romantics

Of course people were going to ask Jason, who is very close with his brother, whether rumors of the two hanging out were true. He did not confirm them, but he
also did not deny them.

Sept. 17: The references begin

"I watched it begin again." - Begin Again

The rumors began to pick up, and now there were a lot of people, commentators included, making Swift references around Kelce.

When the tight end caught a touchdown pass from Patrick Mahomes, CBS Sports' Ian Eagle said, "Kelce finds a blank space for the score."

This was only the start to many, many, Swift lyrics being referenced during NFL games.

Sept. 21: Kelce sends Swift an invite

"So inviting, I almost jump in."
</item 9>

<item 10>
Taylor Swift reveals secret start to her relationship with Travis Kelce (The Independent - Life and Style, 2023-12-06T14:23:01+00:00)
The “All Too Well” singer’s interview with Time marks her first time publicly speaking out about Kelce, who she’s been romantically linked to since this summer,
when the football tight end first confessed to attempting to give her his number.

While Swift has attended a few of Kelce’s games since their relationship started, he’s also gone on to support her career. In November, he made his way to
Buenos Aires, Argentina, to attend one of her Eras Tour shows. During the concert, Swift gave a shout-out to the Chiefs player when she changed her “Karma”
lyrics to reflect their relationship. “Karma is the guy on the Chiefs, coming straight home to me,” she sang, letting out a laugh as fans in the audience
screamed in response.

After the show, the “Style” singer was seen running into Kelce’s arms and kissing him, before he whisked her away backstage.
</item 10>

<item 11>
Travis Kelce faces backlash after comments about ‘finding a breeder’ resurface amid Taylor Swift romance (The Independent - Life and Style,
2023-10-30T20:56:51+00:00)
“Well I know for DAMN SURE that s*** about ‘having kids within the next year’ is NOT coming from Taylor’s side,” one claimed. “She’s got the Eras Tour booked
all through next year, and she’d rather die than cancel a show. But we do know who’s been public about looking for a ‘BREEDER.’”

The comments about Travis and Swift’s romance came days after they seemingly confirmed their relationship. On 22 October, Swift made her return to Arrowhead
Stadium to watch as the Kansas City Chiefs won their game against the Los Angeles Chargers. The occasion marked Swift’s fourth time attending one of Kelce’s
games, amid ongoing reports that they are dating.

Following the event, Chariah Gordon, the partner of Kelce’s teammate Mecole Hardman Jr, took to Instagram to share a snap of the singer and NFL player’s sweet
PDA moment. In the picture, Swift could be seen standing next to Kelce, reaching to kiss his cheek.
</item 11>

<item 12>
Taylor Swift-Travis Kelce timeline: Everything to know about rumored romance between pop star, Chiefs TE (CBSSports.com, 2023-10-12T23:21:52+00:00)
Oct. 15: Kelce, Swift make 'SNL' cameos

The two were spotted out in NYC together and then had a surprise appearance on "Saturday Night Live." Kelce had a cameo in a skit about how much coverage Swift
was getting during games, while the singer introduced the musical act of the night, Ice Spice.

Oct. 22: Swift attends fifth Chiefs game

Swift has now attended games broadcast across all five networks (CBS, NBC, Fox, ESPN and Amazon). This week, she is watching the Chiefs face the Chargers and
will be shown on CBS. She is sporting a Chiefs sweatshirt and is seated next to Kansas City quarterback Patrick Mahomes' wife, Brittany.
</item 12>

<item 13>
Fans spot Travis Kelce wearing Taylor Swift-themed friendship bracelet before she attended his game (The Independent - Life and Style,
2023-09-26T22:05:34+00:00)
Travis’ teammates also showed up to the after-party, as well as his mom, Donna Kelce. The party lasted until 2 am.”

The late-night outing came days after the NFL star officially addressed ongoing dating rumours about him and the “Anti-Hero” singer. During an episode of The
Pat McAfee Show on 21 September, he mocked his brother Jason Kelce for joking that he “100 per cent” believed his younger sibling was dating Swift.

“It’s hilarious how much traction this has actually gotten, right now it’s like a game of telephone,” Kelce said. “Where everybody’s whispering in each other’s
ears and hearing random stuff. No one actually knows what’s going on, especially when you got Jason Kelce on live television telling people just both sides - he
doesn’t know, it’s true, this and that.”

He also noted that he was still open to pursuing a relationship with Swift, revealing that he subtly asked if she’d want to attend one of his football games.
</item 13>

<item 14>
Taylor Swift-Travis Kelce timeline: Everything to know about rumored romance between pop star, Chiefs TE (CBSSports.com, 2023-10-12T23:21:52+00:00)
21: Kelce sends Swift an invite

"So inviting, I almost jump in." - Gold Rush

While on the Pat McAfee show, Kelce discussed the rumors and sounded somehow surprised people are talking about it.

"It is hilarious how much traction this has actually got," he said of all the talk. "It's like that old game in school called telephone where everybody is just
whispering in each others ear, just hearing some random stuff. Especially when you got Jason Kelce on live television telling people just both sides."

He then admitted that he invited Swift to one of his games, joking that they would compete to see who put on the bigger show at Arrowhead Stadium, home of the
Chiefs.

"I threw it out there, I threw the ball in her court and I told her you know, I've seen you rock the stage in Arrowhead you might have to come see me rock the
stage in Arrowhead and see which one is a little more lit. We will see what happens in the near future."
</item 14>

<item 15>
Taylor Swift reveals secret start to her relationship with Travis Kelce (The Independent - Life and Style, 2023-12-06T14:23:01+00:00)
Elsewhere in the conversation with Time, the “Anti-Hero” singer said that she tries not to let the paparazzi get to her, even though so many of her outings with
pals go on to make headlines. “Over the years, I’ve learned I don’t have the time or bandwidth to get pressed about things that don’t matter. Yes, if I go out
to dinner, there’s going to be a whole chaotic situation outside the restaurant. But I still want to go to dinner with my friends,” she said.

She emphasised how her perspective about the media has changed over the years, adding: “Life is short. Have adventures. Me locking myself away in my house for a
lot of years - I’ll never get that time back. I’m more trusting now than I was six years ago.”

With his love life in the public eye, Kelce has also opened up about his relationship with Swift.
</item 15>

<item 16>
Travis Kelce faces backlash after comments about ‘finding a breeder’ resurface amid Taylor Swift romance (The Independent - Life and Style,
2023-10-30T20:56:51+00:00)
In the picture, Swift could be seen standing next to Kelce, reaching to kiss his cheek.

In addition to the kiss, the football star also had the perfect reaction to a fan calling him Swift’s boyfriend. After a fan shouted: “Let’s go Taylor’s
boyfriend” while Kelce was waiting for an elevator, he responded with a small smile on his face and by pumping his fist into the air.

Aside from their appearances at sports events, Swift and Kelce have also been photographed together on other occasions, including a date earlier this month in
New York City after their surprise appearances on the season 49 premiere of Saturday Night Live. At the time, sources alleged to Entertainment Tonight that,
during the date, Kelce got a little too close to Swift’s security personnel and allegedly told her bodyguard to step aside.

However, during an episode of his podcast, Kelce explained what he said to the security, and clarified that he didn’t push the bodyguard.
</item 16>

<item 17>
Taylor Swift-Travis Kelce timeline: Everything to know about rumored romance between pop star, Chiefs TE (CBSSports.com, 2023-10-12T23:21:52+00:00)
Some found issue with these changes, but the NFL said this is a "pop cultural moment" and they are acting as such.

"We frequently change our bios and profile imagery based on what's happening in and around our games, as well as culturally," the NFL wrote in a statement. "The
Taylor Swift and Travis Kelce news has been a pop cultural moment we've leaned into in real time, as it's an intersection of sport and entertainment, and we've
seen an incredible amount of positivity around the sport."

Oct. 6: Travis and Jason think the NFL is doing too much

The NFL is really digging into the Swift-Kelce coverage, and not everyone is a fan, including the people involved.

"I think it's fun when they show who's at the game," Travis said. "I think it brings a little bit more to the atmosphere, brings a little bit more to what
you're watching. But at the same time, they're overdoing it a little bit."
</item 17>

<item 18>
Travis Kelce faces backlash after comments about ‘finding a breeder’ resurface amid Taylor Swift romance (The Independent - Life and Style,
2023-10-30T20:56:51+00:00)
Stay ahead of the trend in fashion and beyond with our free weekly Lifestyle Edit newsletter Stay ahead of the trend in fashion and beyond with our free weekly
Lifestyle Edit newsletter Please enter a valid email address Please enter a valid email address SIGN UP I would like to be emailed about offers, events and
updates from The Independent. Read our privacy notice Thanks for signing up to the

Lifestyle Edit email {{ #verifyErrors }} {{ message }} {{ /verifyErrors }} {{ ^verifyErrors }} Something went wrong. Please try again later {{ /verifyErrors }}

Travis Kelce’s previous comments about having children have sparked controversy amid his ongoing romance with Taylor Swift.

The Kansas City Chiefs player, 34, first made the remarks about starting his own family during a February 2023 episode of his and his brother Jason Kelce’s
podcast, New Heights. The episode resurfaced after a source claimed to Yahoo Entertainment that Swift and Travis Kelce “have talked about kids” within the next
year.
</item 18>

<item 19>
Taylor Swift-Travis Kelce timeline: Everything to know about rumored romance between pop star, Chiefs TE (CBSSports.com, 2023-10-12T23:21:52+00:00)
Head coach Andy Reid was also asked about his star tight end's supposed relationship, to which he quipped, "I set them up."

Sept. 24: Kelce and Swift leave stadium together

"You know when it's time to go." - It's Time To Go

This was the first time the two were photographed together. They walked side by side, Kelce fittingly wearing an outfit with "1989" in the title and Swift
wearing a Chiefs zip-up.

Sept. 24: The two leave in a convertible

"Nothing good starts in the getaway car." - Getaway Car

Swift and Swift were spotted in a convertible, with the tight end driving, after the Chiefs' win. Kelce reportedly rented out a restaurant for a private party
with Swift and his teammates.

"Taylor Swift likes that Travis Kelce 'pursued her' and the two are having a 'great time getting to know each other,'" a source told Entertainment Tonight.
</item 19>

<item 20>
Taylor Swift-Travis Kelce timeline: Everything to know about rumored romance between pop star, Chiefs TE (CBSSports.com, 2023-10-12T23:21:52+00:00)
"'Taylor wants a guy that's into his career, does his own thing, but is also supportive and understands her,' the source added, "and Travis fits those
qualities.'"

Sept. 25: Kelce jersey sales skyrocket

Call it what you want; I'll call it the Taylor Swift effect. Since Swift attended the Chiefs game, there was a 400% increase in Kelce jersey sales, his podcast
is No. 1 overall on Apple, he gained over 380,000 Instagram followers, and 24.3 million people watched the game, making it the No. 1 game last week, according
to Front Office Sports.

There has also been triple the "Chiefs" searches on web and triple the Chiefs ticket sales on Stubhub.

Sept. 25: Belichick weighs in on the matter



New England Patriots head coach does not always comment on situations that take place in the game, but he was fully ready with a sound bite on the Kelce-Swift
situation.
</item 20>
=== END ITEM id=44dedc4d346c2faf ===
