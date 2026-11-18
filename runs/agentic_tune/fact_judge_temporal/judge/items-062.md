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

This file has 2 items.

=== ITEM id=fde3d53674a71408 ===
Gold sentences and facts:
Sentence 0: Cardholders can get 20% back as a statement credit when they purchase eligible pre-purchased meals, in-flight food purchases, alcoholic beverages, and audio headsets on Delta flights.
  Fact 0.0: Cardholders can get 20% back as a statement credit when they purchase eligible pre-purchased meals on Delta flights.
  Fact 0.1: Cardholders can get 20% back as a statement credit when they purchase eligible in-flight food purchases on Delta flights.
  Fact 0.2: Cardholders can get 20% back as a statement credit when they purchase eligible alcoholic beverages on Delta flights.
  Fact 0.3: Cardholders can get 20% back as a statement credit when they purchase eligible audio headsets on Delta flights.
Sentence 1: And for Black Friday, it's on sale for over $400 off, knocking the price down to a super impressive $297.
  Fact 1.0: For Black Friday, the item is on sale for over $400 off.
  Fact 1.1: For Black Friday, the item's price is knocked down to $297.

Retrieved items:
<item 1>
Fly with Delta frequently? Small business owners can score a solid welcome bonus and other perks with the Delta SkyMiles Gold Business Card (Fortune,
2023-12-12T21:01:43+00:00)
Cardholders can get 20% back as a statement credit when they purchase eligible pre-purchased meals, in-flight food purchases, alcoholic beverages, and audio
headsets on Delta flights. Baggage insurance. Cardholders who purchase plane, train, ship or bus ride tickets with their card are eligible for baggage insurance
coverage—which kicks in if your baggage is stolen, lost, or damaged.

Cardholders who purchase plane, train, ship or bus ride tickets with their card are eligible for baggage insurance coverage—which kicks in if your baggage is
stolen, lost, or damaged. Car Rental Loss and Damage Insurance. A cardholder can get damage and theft insurance on their car rental. Cardholders must use their
card to book the rental and decline the collision damage waiver at the agency to receive it.
</item 1>

<item 2>
Fly with Delta frequently? Small business owners can score a solid welcome bonus and other perks with the Delta SkyMiles Gold Business Card (Fortune,
2023-12-12T21:01:43+00:00)
For $99 a year, cardholders receive benefits like priority boarding, the first checked bag benefit (and for guests), discounts on in-flight purchases, and
baggage insurance. The card, however, doesn’t provide access to Delta Sky Club or Delta’s airport lounges.

The rewards structure on this card is decent, but it’s nothing to write home about—the card offers 2x miles on just a handful of categories like advertising,
shipping, and dining.

A big benefit of the Delta SkyMiles Gold Business Card is the welcome bonus: 75,000 Bonus Miles after you spend $6,000 in purchases on your new Card in your
first 6 months of card opening. And cardholders who use their miles to purchase flights get even more benefits—customers receive a 15% discount when they use
miles to buy Delta flights on the Fly Delta app or the Delta website.

Graph facts:
- Sen. Robert Menendez, facing bribery charges, stumps bank CEOs by asking how much their firms paid in settlements for allegedly ripping off consumers
-PUBLISHED_BY-> Fortune
- Buckle up for more ‘turbulence’ in the housing market, BofA says—it’s a housing recession, 1980s style -PUBLISHED_BY-> Fortune
- Israel's blockade of Gaza means that a region the size of Philadelphia faces a real risk of starvation, says Peace Studies scholar -PUBLISHED_BY-> Fortune
- Gen Z just saw the worst war in Israel in 50 years break out all over the internet, but they were raised on 'disturbing images' -PUBLISHED_BY-> Fortune
- Is Sam Bankman-Fried a bad ‘man’ or a good ‘boy’? Lawyers swap opening statements before first witnesses take the stand -PUBLISHED_BY-> Fortune

Entities: Fortune, Delta, Delta Sky Club, Fly Delta app, Delta website, Fly with Delta frequently? Small business owners can score a solid welcome bonus and
other perks with the Delta SkyMiles Gold Business Card, Delta SkyMiles Gold Business Card
</item 2>

<item 3>
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

Graph facts:
- Sen. Robert Menendez, facing bribery charges, stumps bank CEOs by asking how much their firms paid in settlements for allegedly ripping off consumers
-PUBLISHED_BY-> Fortune
- Buckle up for more ‘turbulence’ in the housing market, BofA says—it’s a housing recession, 1980s style -PUBLISHED_BY-> Fortune
- Israel's blockade of Gaza means that a region the size of Philadelphia faces a real risk of starvation, says Peace Studies scholar -PUBLISHED_BY-> Fortune
- Gen Z just saw the worst war in Israel in 50 years break out all over the internet, but they were raised on 'disturbing images' -PUBLISHED_BY-> Fortune
- Is Sam Bankman-Fried a bad ‘man’ or a good ‘boy’? Lawyers swap opening statements before first witnesses take the stand -PUBLISHED_BY-> Fortune

Entities: Fortune, American Express, Delta, Fly with Delta frequently?, Delta SkyMiles Gold Business Card
</item 3>

<item 4>
Fly with Delta frequently? Small business owners can score a solid welcome bonus and other perks with the Delta SkyMiles Gold Business Card (Fortune,
2023-12-12T21:01:43+00:00)
Delta has more than 20 airline partners, so cardholders can book through airlines like Air France and Virgin Atlantic—though the 15% discount only applies to
Delta and Delta Connection® flights.

This discount is a reason to hold onto the card long-term as you could save thousands of miles over the course of a year if you are a heavy Delta user. The best
bargains on award flights are found through Delta SkyMiles Award Deals. Recent deals from Seattle (SEA) have included Chicago (ORD) for 8,000 miles or Auckland,
New Zealand (AKL) for 64,000 one way in economy plus taxes and fees. Flash sales come and go so it’s a good idea to keep an eye on the site.

Other options include booking travel through Delta Vacations, using miles for upgrades or even buying champagne at the Delta Sky Club. That said, your best
value will most likely be on award tickets.

Graph facts:
- NFL Fantasy Football Week 5 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game -MENTIONS-> Chicago
- Browns survive Bears Hail Mary on wild, near-catch by Darnell Mooney: 'Damn near s— myself' -MENTIONS-> Chicago
- Sam Bankman-Fried didn’t ask where the $8 billion went -MENTIONS-> Chicago
- He’s Hockey’s Brightest Young Star. This Is What Makes His Shot So Special. -MENTIONS-> Chicago
- Chicago Fire vs Inter Miami live score, result, updates, highlights from Lionel Messi in MLS match -MENTIONS-> Chicago

Entities: Chicago, Fortune, Auckland, Seattle, New Zealand, Delta, Fly with Delta frequently? Small business owners can score a solid welcome bonus and other
perks with the Delta SkyMiles Gold Business Card, Air France, Virgin Atlantic, Delta Connection, Delta SkyMiles Award Deals, Delta Vacations, Delta Sky Club,
Delta SkyMiles Gold Business Card
</item 4>

<item 5>
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

Graph facts:
- Sen. Robert Menendez, facing bribery charges, stumps bank CEOs by asking how much their firms paid in settlements for allegedly ripping off consumers
-PUBLISHED_BY-> Fortune
- Buckle up for more ‘turbulence’ in the housing market, BofA says—it’s a housing recession, 1980s style -PUBLISHED_BY-> Fortune
- Israel's blockade of Gaza means that a region the size of Philadelphia faces a real risk of starvation, says Peace Studies scholar -PUBLISHED_BY-> Fortune
- Gen Z just saw the worst war in Israel in 50 years break out all over the internet, but they were raised on 'disturbing images' -PUBLISHED_BY-> Fortune
- Is Sam Bankman-Fried a bad ‘man’ or a good ‘boy’? Lawyers swap opening statements before first witnesses take the stand -PUBLISHED_BY-> Fortune

Entities: Fortune, American Express, Delta SkyMiles® Gold Business American Express Card, Delta, Fly with Delta frequently? Small business owners can score a
solid welcome bonus and other perks with the Delta SkyMiles Gold Business Card
</item 5>

<item 6>
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

Graph facts:
- Sen. Robert Menendez, facing bribery charges, stumps bank CEOs by asking how much their firms paid in settlements for allegedly ripping off consumers
-PUBLISHED_BY-> Fortune
- Buckle up for more ‘turbulence’ in the housing market, BofA says—it’s a housing recession, 1980s style -PUBLISHED_BY-> Fortune
- Israel's blockade of Gaza means that a region the size of Philadelphia faces a real risk of starvation, says Peace Studies scholar -PUBLISHED_BY-> Fortune
- Gen Z just saw the worst war in Israel in 50 years break out all over the internet, but they were raised on 'disturbing images' -PUBLISHED_BY-> Fortune
- Is Sam Bankman-Fried a bad ‘man’ or a good ‘boy’? Lawyers swap opening statements before first witnesses take the stand -PUBLISHED_BY-> Fortune

Entities: Fortune, American Express, Delta, Fly with Delta frequently? Small business owners can score a solid welcome bonus and other perks with the Delta
SkyMiles Gold Business Card, Delta SkyMiles Gold Business Card, Delta SkyMiles Gold Card, FICO credit score, Amex Assurance Company, americanexpress.com,
americanexpress.com/benefitsguide, income, rates, fees, and benefits, American Express benefits and offers
</item 6>

<item 7>
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

Graph facts:
- Sen. Robert Menendez, facing bribery charges, stumps bank CEOs by asking how much their firms paid in settlements for allegedly ripping off consumers
-PUBLISHED_BY-> Fortune
- Buckle up for more ‘turbulence’ in the housing market, BofA says—it’s a housing recession, 1980s style -PUBLISHED_BY-> Fortune
- Israel's blockade of Gaza means that a region the size of Philadelphia faces a real risk of starvation, says Peace Studies scholar -PUBLISHED_BY-> Fortune
- Gen Z just saw the worst war in Israel in 50 years break out all over the internet, but they were raised on 'disturbing images' -PUBLISHED_BY-> Fortune
- Is Sam Bankman-Fried a bad ‘man’ or a good ‘boy’? Lawyers swap opening statements before first witnesses take the stand -PUBLISHED_BY-> Fortune

Entities: Fortune, Delta, Fly with Delta frequently? Small business owners can score a solid welcome bonus and other perks with the Delta SkyMiles Gold Business
Card, Delta SkyMiles Gold Business Card
</item 7>

<item 8>
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

Graph facts:
- Sen. Robert Menendez, facing bribery charges, stumps bank CEOs by asking how much their firms paid in settlements for allegedly ripping off consumers
-PUBLISHED_BY-> Fortune
- Buckle up for more ‘turbulence’ in the housing market, BofA says—it’s a housing recession, 1980s style -PUBLISHED_BY-> Fortune
- Israel's blockade of Gaza means that a region the size of Philadelphia faces a real risk of starvation, says Peace Studies scholar -PUBLISHED_BY-> Fortune
- Gen Z just saw the worst war in Israel in 50 years break out all over the internet, but they were raised on 'disturbing images' -PUBLISHED_BY-> Fortune
- Is Sam Bankman-Fried a bad ‘man’ or a good ‘boy’? Lawyers swap opening statements before first witnesses take the stand -PUBLISHED_BY-> Fortune

Entities: Fortune, American Express, U.S., Delta, Fly Delta app, Delta SkyMiles account, Takeoff15, Fly with Delta frequently? Small business owners can score a
solid welcome bonus and other perks with the Delta SkyMiles Gold Business Card, Delta SkyMiles Gold Business Card
</item 8>

<item 9>
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

Graph facts:
- Sen. Robert Menendez, facing bribery charges, stumps bank CEOs by asking how much their firms paid in settlements for allegedly ripping off consumers
-PUBLISHED_BY-> Fortune
- Buckle up for more ‘turbulence’ in the housing market, BofA says—it’s a housing recession, 1980s style -PUBLISHED_BY-> Fortune
- Israel's blockade of Gaza means that a region the size of Philadelphia faces a real risk of starvation, says Peace Studies scholar -PUBLISHED_BY-> Fortune
- Gen Z just saw the worst war in Israel in 50 years break out all over the internet, but they were raised on 'disturbing images' -PUBLISHED_BY-> Fortune
- Is Sam Bankman-Fried a bad ‘man’ or a good ‘boy’? Lawyers swap opening statements before first witnesses take the stand -PUBLISHED_BY-> Fortune

Entities: Fortune, Delta, Fly with Delta frequently? Small business owners can score a solid welcome bonus and other perks with the Delta SkyMiles Gold Business
Card, Virgin Atlantic, Delta SkyMiles Gold Business Card, Chase Ink Business Preferred® Credit Card, Chase, Ultimate Rewards, Flying Blue
</item 9>

<item 10>
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

Graph facts:
- Sen. Robert Menendez, facing bribery charges, stumps bank CEOs by asking how much their firms paid in settlements for allegedly ripping off consumers
-PUBLISHED_BY-> Fortune
- Buckle up for more ‘turbulence’ in the housing market, BofA says—it’s a housing recession, 1980s style -PUBLISHED_BY-> Fortune
- Israel's blockade of Gaza means that a region the size of Philadelphia faces a real risk of starvation, says Peace Studies scholar -PUBLISHED_BY-> Fortune
- Gen Z just saw the worst war in Israel in 50 years break out all over the internet, but they were raised on 'disturbing images' -PUBLISHED_BY-> Fortune
- Is Sam Bankman-Fried a bad ‘man’ or a good ‘boy’? Lawyers swap opening statements before first witnesses take the stand -PUBLISHED_BY-> Fortune

Entities: Fortune, Delta, Fly with Delta frequently? Small business owners can score a solid welcome bonus and other perks with the Delta SkyMiles Gold Business
Card, SkyMiles, Delta eCredit, Delta flights, priority boarding, flight credit, in-flight purchases, Delta SkyMiles Gold Business Card
</item 10>
=== END ITEM id=fde3d53674a71408 ===

=== ITEM id=47f8a62dfdd4058f ===
Gold sentences and facts:
Sentence 0: After learning of Diana’s death from his father at the royal residence of Balmoral in Scotland, William (Rufus Kampa) is left heartbroken with grief and angry at his family’s response to it.
  Fact 0.0: William is identified as Rufus Kampa.
  Fact 0.1: William learns of Diana's death from his father at the royal residence of Balmoral in Scotland.
  Fact 0.2: William is left heartbroken with grief after learning of Diana's death.
  Fact 0.3: William is angry at his family's response to Diana's death.
Sentence 1: The latest installment takes place between the late 1990s to mid-2000s, and includes the lead-up to Princess Diana’s fatal Paris car crash in 1997.
  Fact 1.0: The latest installment takes place between the late 1990s and mid-2000s.
  Fact 1.1: The latest installment includes the lead-up to Princess Diana's fatal Paris car crash.
  Fact 1.2: Princess Diana's fatal Paris car crash happened in 1997.

Retrieved items:
<item 1>
The Crown: What is fact and what is fiction on season 6? (The Independent - Life and Style, 2023-11-17T19:32:16+00:00)
He is not located, but later walks back to the house on his own accord, drenched in rain. “14 hours, that poor boy was gone,” the Queen later says.

William (Kampa) goes missing after his mother’s death (Keith Bernstein)

It’s true that William and Harry initially stayed in Balmoral with their grandmother following the death of Diana. While it’s not known whether he went missing,
during a visit to Scotland in 2021, William recalled how that experience was one of the “saddest” moments of his life. “I was in Balmoral when I was told that
my mother had died. Still in shock, I found sanctuary in the service at Crathie Kirk that very morning,” he recalled. “And in the dark days of grief that
followed, I found comfort and solace in the Scottish outdoors. As a result, the connection I feel to Scotland will forever run deep.”

The Crown season six part one is on Netflix now, with part two arriving on 14 December.

Graph facts:
- The Crown: What is fact and what is fiction on season 6? -PUBLISHED_BY-> The Independent - Life and Style
- The Crown: What is fact and what is fiction on season 6? -MENTIONS-> France
- The Crown: What is fact and what is fiction on season 6? -MENTIONS-> Charles
- The Crown: What is fact and what is fiction on season 6? -MENTIONS-> Elizabeth
- The Crown: What is fact and what is fiction on season 6? -MENTIONS-> Paris

Entities: The Crown: What is fact and what is fiction on season 6?, Diana, William, Harry, Queen, Kampa, Keith Bernstein, Balmoral, Crathie Kirk, The Crown,
Netflix, The Crown season six part one release, The Crown season six part two arrival, The Independent, Scotland
</item 1>

<item 2>
The Crown: What is fact and what is fiction on season 6? (The Independent - Life and Style, 2023-11-17T19:32:16+00:00)
It was later believed to have been delivered to their room before they went to Dodi’s flat, where a ring bearing the words “Dis-moi Oui” (“Tell me Yes”) – also
the name of episode three – was later recovered, alongside a receipt for a “bague de fiançaille” (engagement ring). Speaking at the inquest, his father Mohamed
claimed that the pair had met the jeweller in Monte Carlo during the holiday on the yacht, and that he believed Dodi planned to propose that night.

Prince William goes missing in Balmoral after Diana’s death

After learning of Diana’s death from his father at the royal residence of Balmoral in Scotland, William (Rufus Kampa) is left heartbroken with grief and angry
at his family’s response to it. At one point, Charles is informed that the prince is not in his room and that “no one can find him”, prompting Charles and
Prince Harry to look for him and a subsequent hunt across the grounds for the young royal.

Graph facts:
- West Indies v England: fifth and deciding men’s T20 cricket international – live -MENTIONS-> Charles
- The Apple Watch’s double tap gesture points at a new way to use wearables -MENTIONS-> Charles
- Charles -PARTICIPATED_IN-> West Indies v England: fifth and deciding men’s T20 cricket international
- Charles -PARTICIPATED_IN-> West Indies v England fifth and deciding men’s T20 cricket international
- Charles -PARTICIPATED_IN-> fifth and deciding men’s T20 cricket international

Entities: Charles, The Crown: What is fact and what is fiction on season 6?, Diana, Mohamed, Dodi, yacht, Monte Carlo, Dis-moi Oui, Prince William, Rufus Kampa,
Prince Harry, the inquest, episode three, Balmoral, The Crown
</item 2>

<item 3>
Fans react to most ‘heartbreaking’ scene in The Crown (The Independent - Life and Style, 2023-11-17T09:35:32+00:00)
Stay ahead of the trend in fashion and beyond with our free weekly Lifestyle Edit newsletter Stay ahead of the trend in fashion and beyond with our free weekly
Lifestyle Edit newsletter Please enter a valid email address Please enter a valid email address SIGN UP I would like to be emailed about offers, events and
updates from The Independent. Read our privacy notice Thanks for signing up to the

Lifestyle Edit email {{ #verifyErrors }} {{ message }} {{ /verifyErrors }} {{ ^verifyErrors }} Something went wrong. Please try again later {{ /verifyErrors }}

The first four episodes of The Crown season six are officially here, but many viewers believe one scene has depicted the most devastating moment throughout the
entire series.

On 16 November, Netflix dropped the first four episodes of The Crown’s final season, followed by the last six episodes on 14 December. The latest installment
takes place between the late 1990s to mid-2000s, and includes the lead-up to Princess Diana’s fatal Paris car crash in 1997.

Graph facts:
- What to know about Fortnite maker Epic Games’ antitrust battle with Google, starting today -MENTIONS-> Netflix
- Video games in 2023: Acquisitions, layoffs, unions -MENTIONS-> Netflix
- The future of DC movies in 2022 and beyond -MENTIONS-> Netflix
- A comprehensive list of 2023 tech layoffs -MENTIONS-> Netflix
- Castlevania: Nocturne is a revolutionary epic that does the franchise’s best games justice -MENTIONS-> Netflix

Entities: Netflix, Paris, Princess Diana, The Crown, Netflix dropped the first four episodes of The Crown season six, The last six episodes of The Crown season
six released, Princess Diana’s fatal Paris car crash, Lifestyle Edit newsletter, Fans react to most ‘heartbreaking’ scene in The Crown, The Independent
</item 3>

<item 4>
Graph query row:
article: The Crown: What is fact and what is fiction on season 6?
published: 2023-11-17T19:32:16+00:00
publisher: The Independent
descriptions: ['The Crown: What is fact and what is fiction on season 6? (The Independent - Life and Style, 2023-11-17T19:32:16+00:00)\nSign up to our free
IndyArts newsletter for all the latest entertainment news and reviews Sign up to our free IndyArts newsletter Please enter a valid email address Please enter a
valid email address SIGN UP I would like to be emailed about offers, events and updates from The Independent. Read our privacy notice Thanks for signing up to
the\n\nIndyArts email {{ #verifyErrors }} {{ message }} {{ /verifyErrors }} {{ ^verifyErrors }} Something went wrong. Please try again later {{ /verifyErrors
}}\n\nAs Netflix’s acclaimed royal drama moves closer to the present day, the line between fact and fiction feels ever more blurred.\n\nWhile Peter Morgan’s
show has always presented a dramatised take on real-life events, many have argued that viewers struggle to tell the difference between the real royal family and
the events on the show. When season five arrived last year, there were calls for the show to include a “fictional explainer” – something Netflix did not do.',
'The Crown: What is fact and what is fiction on season 6? (The Independent - Life and Style, 2023-11-17T19:32:16+00:00)\nWith the show heading into the late
Nineties and early Noughties in its final season, which arrived on Netflix on Thursday (16 November), events are being depicted on screen that many of the
public will remember well.\n\nThe first part of season six takes place in the summer of 1997, and sees Prince Charles (Dominic West) building on his
relationship with Camilla Parker Bowles (Olivia Williams), while Princess Diana (Elizabeth Debicki) sparks up a romance with film producer Dodi Fayed (Khalid
Abdalla). The show also features the pair’s death in a car crash in Paris.\n\nAs the new season arrives, and fans wonder what in The Crown is based in reality,
here’s everything you need to know.\n\nDiana’s role with Tony Blair’s government\n\nEpisode one sees Diana visiting Tony (Bertie Carvel) and Cherie Blair (Lydia
Leonard) with William after asking for a meeting with him.', 'The Crown: What is fact and what is fiction on season 6? (The Independent - Life and Style,
2023-11-17T19:32:16+00:00)\nRecalling their discussion to the Queen, Blair says that Diana feels she “still has a lot to offer the country as a public servant”.
He says that she had asked if there was a way to work with the government on “a more formal basis” and that any “official role” would be appreciated.\n\nBertie
Carvel as Blair (Netflix)\n\nFollowing Diana’s death, Downing Street confirmed that Diana had visited Blair at Chequers with her son a few weeks before, where
they discussed a special role for her as an overseas “ambassador” for Britain. Diana told a journalist at the time that the prime minister had recognised her
skills and asked her to undertake “missions” abroad for Britain.\n\nQueen Elizabeth’s reluctance to accept Camilla\n\nAs The Crown season six begins, Charles is
frustrated at the public for demonising Camilla long after his split from Diana.', 'The Crown: What is fact and what is fiction on season 6? (The Independent -
Life and Style, 2023-11-17T19:32:16+00:00)\nAsks the Queen if she’d received the invitation to Camilla’s 50th birthday, to which she says she has, but cannot
attend as she’s in Derbyshire. When Charles points out that plans “can always be changed”, she replies: “Why would we want to change it?” She doesn’t attend,
but later tells Prince Philip that she doesn’t want for Camilla to be considered “wicked, because she’s not” or to be unkind to her.\n\nCharles (West) hosts
Camilla’s (Williams) birthday (Netflix)\n\nIt took a while for Queen Elizabeth and Camilla to grow close. According to historian Robert Lacey, she was reluctant
to accept Charles’s request to be more accommodating to Camilla. Unlike in The Crown, she is said to have once referred to Camilla as “that wicked woman” to
Charles’s face. However, she grew to become closer to Camilla.', 'The Crown: What is fact and what is fiction on season 6? (The Independent - Life and Style,
2023-11-17T19:32:16+00:00)\nHowever, she grew to become closer to Camilla. Despite announcing in 2005 that Camilla would become the Princess Consort rather than
Queen Consort, it was announced that Camilla would be queen seven months before Elizabeth’s death.\n\nAccess unlimited streaming of movies and TV shows with
Amazon Prime Video Sign up now for a 30-day free trial Sign up\n\nAccess unlimited streaming of movies and TV shows with Amazon Prime Video Sign up now for a
30-day free trial Sign up\n\nDodi Fayed’s engagement to an American model\n\nWhen viewers first meet Dodi in The Crown, he’s in Paris with a woman with short,
blonde hair. However, it’s not Diana, but an American woman who he says he’s picking out fabric for a Malibu home with and says he’s marrying in three weeks.
His father Mohamed is unimpressed, calling him to Saint Tropez to meet a “special guest” and leave his “gold-digger” fiance – who he also calls “Madame Bikini”
– behind.', 'The Crown: What is fact and what is fiction on season 6? (The Independent - Life and Style, 2023-11-17T19:32:16+00:00)\nThe model later confronts
Dodi, asking if he was with Diana when he leaves her to go see his father. Diana also alludes to a “lawsuit” between Dodi and his ex.\n\nDiana (Debicki) and
Dodi (Abdalla) (PA)\n\nThis storyline comes to close to the truth as told by model Kelly Fisher, who claimed to have been engaged to Dodi when he met Diana.
Fisher had previously modelled for brands such as Victoria’s Secret and magazines including Elle and Marie Claire. As in The Crown, Fisher – who met Fayed in
July 1996 – was on another Fayed yacht when Dodi and Diana were first photographed. She staged a press conference where she announced that she was filing a
breach-of-contract lawsuit against Dodi, after he allegedly asked her to stop modelling as much and prioritise the relationship.', 'The Crown: What is fact and
what is fiction on season 6? (The Independent - Life and Style, 2023-11-17T19:32:16+00:00)\nThrough tears, she claimed that he had proposed to her (something
his family denied) and promised to buy her a house in Malibu. However, she dropped the lawsuit following Dodi’s death.\n\nDiana’s close relationship with
paparazzi on France yacht trip\n\nWhen Dodi and Diana first spend time together on his father’s yacht in France, the paparazzi are never far away. However, when
William refuses to go outside until the photographers leave them alone, she takes a boat out to visit them. Wearing a swimming costume, she asks them: “We’re
having a lovely time, apart from one little thing: you lot.” “Don’t be like that, you love us really,” they reply. Diana asks them to leave her alone as they’re
“freaking out” her sons, before telling one photographer – who she knows by name – that if they go away, “you’re going to get a big surprise with the next thing
I do”.', 'The Crown: What is fact and what is fiction on season 6? (The Independent - Life and Style, 2023-11-17T19:32:16+00:00)\nShe then poses for them in her
swimsuit, but complains in a later episode that they can “never relax” with the press “constantly” around.\n\nDiana (Debicki) on the Fayed yacht (Daniel
Escale/Netflix)\n\nFormer Vanity Fair editor Tina Brown, who wrote the controversial 2007 book Diana Chronicles, claimed that Diana would often tip off the
press and “ couldn’t resist giving them the images they wanted”, including on the boat. Photographer Daniel Pirrie, who ended up selling the photo of the kiss,
claimed that Diana herself invited him to photograph her on holiday with Dodi in France that summer. “She told me she’d be on the yacht somewhere in the
Mediterranean,” he told The Daily Mail in 2013. “She let me know she was going to be on a tender going out to the yacht. I think overall, she was happy with all
the pictures taken that summer… I knew that Diana had wanted [the kiss photo] to be taken.”', 'The Crown: What is fact and what is fiction on season 6? (The
Independent - Life and Style, 2023-11-17T19:32:16+00:00)\nMohamed al Fayed’s role in securing kiss photo\n\nThroughout season six, Mohamed al Fayed (Salim Daw)
is shown as heavily invested in his son Dodi’s relationship with Diana, even ringing up the maid to ask if they have been “intimate”. In Arabic, he then asks
his secretary if they can find him an Arabic photographer – “not just any idiot with a long lens”, but “the best photographer on the Meditarranean”.\n\nSalim
Daw as Mohamed al Fayed (Netflix)\n\nIt’s not known whether Mohamed was so instrumental in securing the photo of Dodi and Diana kissing for the first time.
However, following the Harrods owner’s death this year, biographer Tom Bower recalled in The Times how Al-Fayed “gleefully” showed him the photographs of Dodi
and Diana and allegedly “revealed how he had rapidly bought a yacht suitable for his son’s seduction of her”.', 'The Crown: What is fact and what is fiction on
season 6? (The Independent - Life and Style, 2023-11-17T19:32:16+00:00)\nDodi’s proposal to Diana on the night they died\n\nAfter bursting into tears going for
dinner at the Ritz, move upstairs, he proposes to her with a ring she had previously pointed out when they sheltered from crowds in a Monte Carlo jewellery
shop. She tells him to get up off his knee “I know the whole world is wondering if we’re going to get married, but that’s not a reason to actually do it.”
However, they share a drink and bittersweet conversation about how they need to live their lives going forward - advice\n\nDodi is shown proposing to Diana
before they died (Daniel Escale/Netflix)\n\nIn real life, Dodi did not propose to Diana on that fateful evening in Paris. However, at the inquest into the death
in 2007, the jury were shown CCTV footage of him purchasing an engagement ring worth £11,600 in a jewellers across the square from the Ritz on the afternoon of
the crash.', 'The Crown: What is fact and what is fiction on season 6? (The Independent - Life and Style, 2023-11-17T19:32:16+00:00)\nIt was later believed to
have been delivered to their room before they went to Dodi’s flat, where a ring bearing the words “Dis-moi Oui” (“Tell me Yes”) – also the name of episode three
– was later recovered, alongside a receipt for a “bague de fiançaille” (engagement ring). Speaking at the inquest, his father Mohamed claimed that the pair had
met the jeweller in Monte Carlo during the holiday on the yacht, and that he believed Dodi planned to propose that night.\n\nPrince William goes missing in
Balmoral after Diana’s death\n\nAfter learning of Diana’s death from his father at the royal residence of Balmoral in Scotland, William (Rufus Kampa) is left
heartbroken with grief and angry at his family’s response to it. At one point, Charles is informed that the prince is not in his room and that “no one can find
him”, prompting Charles and Prince Harry to look for him and a subsequent hunt across the grounds for the young royal.', 'The Crown: What is fact and what is
fiction on season 6? (The Independent - Life and Style, 2023-11-17T19:32:16+00:00)\nHe is not located, but later walks back to the house on his own accord,
drenched in rain. “14 hours, that poor boy was gone,” the Queen later says.\n\nWilliam (Kampa) goes missing after his mother’s death (Keith Bernstein)\n\nIt’s
true that William and Harry initially stayed in Balmoral with their grandmother following the death of Diana. While it’s not known whether he went missing,
during a visit to Scotland in 2021, William recalled how that experience was one of the “saddest” moments of his life. “I was in Balmoral when I was told that
my mother had died. Still in shock, I found sanctuary in the service at Crathie Kirk that very morning,” he recalled. “And in the dark days of grief that
followed, I found comfort and solace in the Scottish outdoors. As a result, the connection I feel to Scotland will forever run deep.”\n\nThe Crown season six
part one is on Netflix now, with part two arriving on 14 December.']
connections: [{'entity': 'Olivia Williams', 'relationship': 'MENTIONS'}, {'entity': 'Princess Diana', 'relationship': 'MENTIONS'}, {'entity': 'Diana',
'relationship': 'MENTIONS'}, {'entity': 'The Crown', 'relationship': 'MENTIONS'}, {'entity': 'Diana’s death', 'relationship': 'MENTIONS'}, {'entity': 'Dodi and
Diana first photographed', 'relationship': 'MENTIONS'}, {'entity': 'Diana Chronicles', 'relationship': 'MENTIONS'}, {'entity': 'Dodi’s proposal to Diana',
'relationship': 'MENTIONS'}, {'entity': 'Prince William', 'relationship': 'MENTIONS'}, {'entity': 'William', 'relationship': 'MENTIONS'}, {'entity': 'The Crown
season six part one release', 'relationship': 'MENTIONS'}, {'entity': 'The Crown season six part two arrival', 'relationship': 'MENTIONS'}]
</item 4>

<item 5>
The Crown: What is fact and what is fiction on season 6? (The Independent - Life and Style, 2023-11-17T19:32:16+00:00)
With the show heading into the late Nineties and early Noughties in its final season, which arrived on Netflix on Thursday (16 November), events are being
depicted on screen that many of the public will remember well.

The first part of season six takes place in the summer of 1997, and sees Prince Charles (Dominic West) building on his relationship with Camilla Parker Bowles
(Olivia Williams), while Princess Diana (Elizabeth Debicki) sparks up a romance with film producer Dodi Fayed (Khalid Abdalla). The show also features the
pair’s death in a car crash in Paris.

As the new season arrives, and fans wonder what in The Crown is based in reality, here’s everything you need to know.

Diana’s role with Tony Blair’s government

Episode one sees Diana visiting Tony (Bertie Carvel) and Cherie Blair (Lydia Leonard) with William after asking for a meeting with him.

Graph facts:
- Is Sam Bankman-Fried a bad ‘man’ or a good ‘boy’? Lawyers swap opening statements before first witnesses take the stand -MENTIONS-> Paris
- There’s something going on with AI startups in France -MENTIONS-> Paris
- The best movies of 2023 so far -MENTIONS-> Paris
- How U.S. Women’s Soccer Team Lured Emma Hayes Away From Chelsea -MENTIONS-> Paris
- ‘It happens’: All Blacks defend star centre after spray as Cheika heaps pressure on ‘favourites’ New Zealand -MENTIONS-> Paris

Entities: Paris, Tony Blair, The Crown: What is fact and what is fiction on season 6?, Prince Charles, Dominic West, Camilla Parker Bowles, Olivia Williams,
Princess Diana, Elizabeth Debicki, Dodi Fayed, Khalid Abdalla, car crash, Bertie Carvel, Cherie Blair, Lydia Leonard
</item 5>

<item 6>
The Crown: What is fact and what is fiction on season 6? (The Independent - Life and Style, 2023-11-17T19:32:16+00:00)
Through tears, she claimed that he had proposed to her (something his family denied) and promised to buy her a house in Malibu. However, she dropped the lawsuit
following Dodi’s death.

Diana’s close relationship with paparazzi on France yacht trip

When Dodi and Diana first spend time together on his father’s yacht in France, the paparazzi are never far away. However, when William refuses to go outside
until the photographers leave them alone, she takes a boat out to visit them. Wearing a swimming costume, she asks them: “We’re having a lovely time, apart from
one little thing: you lot.” “Don’t be like that, you love us really,” they reply. Diana asks them to leave her alone as they’re “freaking out” her sons, before
telling one photographer – who she knows by name – that if they go away, “you’re going to get a big surprise with the next thing I do”.

Graph facts:
- More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo -MENTIONS-> France
- THE GREATEST! Boks beat All Blacks in instant classic to claim record FOURTH RWC in controversial, dramatic final -MENTIONS-> France
- There’s something going on with AI startups in France -MENTIONS-> France
- Amazon’s iRobot purchase sucks up formal competition concerns in EU -MENTIONS-> France
- ASX set to drop as Wall Street’s September slump deepens -MENTIONS-> France

Entities: France, The Crown: What is fact and what is fiction on season 6?, Diana, Malibu, Dodi, paparazzi, France yacht trip, yacht, William, The Crown,
lawsuit, The Independent - Life and Style
</item 6>

<item 7>
The Crown: What is fact and what is fiction on season 6? (The Independent - Life and Style, 2023-11-17T19:32:16+00:00)
Dodi’s proposal to Diana on the night they died

After bursting into tears going for dinner at the Ritz, move upstairs, he proposes to her with a ring she had previously pointed out when they sheltered from
crowds in a Monte Carlo jewellery shop. She tells him to get up off his knee “I know the whole world is wondering if we’re going to get married, but that’s not
a reason to actually do it.” However, they share a drink and bittersweet conversation about how they need to live their lives going forward - advice

Dodi is shown proposing to Diana before they died (Daniel Escale/Netflix)

In real life, Dodi did not propose to Diana on that fateful evening in Paris. However, at the inquest into the death in 2007, the jury were shown CCTV footage
of him purchasing an engagement ring worth £11,600 in a jewellers across the square from the Ritz on the afternoon of the crash.

Graph facts:
- Is Sam Bankman-Fried a bad ‘man’ or a good ‘boy’? Lawyers swap opening statements before first witnesses take the stand -MENTIONS-> Paris
- There’s something going on with AI startups in France -MENTIONS-> Paris
- The best movies of 2023 so far -MENTIONS-> Paris
- How U.S. Women’s Soccer Team Lured Emma Hayes Away From Chelsea -MENTIONS-> Paris
- ‘It happens’: All Blacks defend star centre after spray as Cheika heaps pressure on ‘favourites’ New Zealand -MENTIONS-> Paris

Entities: Paris, The Crown: What is fact and what is fiction on season 6?, Diana, Dodi, Daniel Escale, the Ritz, Monte Carlo, inquest into the death, the crash,
engagement ring, Dodi’s proposal to Diana, The Crown, The Independent - Life and Style
</item 7>

<item 8>
The Crown: What is fact and what is fiction on season 6? (The Independent - Life and Style, 2023-11-17T19:32:16+00:00)
Recalling their discussion to the Queen, Blair says that Diana feels she “still has a lot to offer the country as a public servant”. He says that she had asked
if there was a way to work with the government on “a more formal basis” and that any “official role” would be appreciated.

Bertie Carvel as Blair (Netflix)

Following Diana’s death, Downing Street confirmed that Diana had visited Blair at Chequers with her son a few weeks before, where they discussed a special role
for her as an overseas “ambassador” for Britain. Diana told a journalist at the time that the prime minister had recognised her skills and asked her to
undertake “missions” abroad for Britain.

Queen Elizabeth’s reluctance to accept Camilla

As The Crown season six begins, Charles is frustrated at the public for demonising Camilla long after his split from Diana.

Graph facts:
- West Indies v England: fifth and deciding men’s T20 cricket international – live -MENTIONS-> Charles
- The Apple Watch’s double tap gesture points at a new way to use wearables -MENTIONS-> Charles
- Charles -PARTICIPATED_IN-> West Indies v England: fifth and deciding men’s T20 cricket international
- Charles -PARTICIPATED_IN-> West Indies v England fifth and deciding men’s T20 cricket international
- Charles -PARTICIPATED_IN-> fifth and deciding men’s T20 cricket international

Entities: Charles, The Crown: What is fact and what is fiction on season 6?, Bertie Carvel, Blair, Diana, Camilla, Downing Street, Chequers, The Crown, Diana’s
death, Queen Elizabeth, Netflix, government, Britain, The Independent
</item 8>

<item 9>
Fans react to most ‘heartbreaking’ scene in The Crown (The Independent - Life and Style, 2023-11-17T09:35:32+00:00)
“The last call between Harry, William and Diana is heartbreaking,” a third user commented.

In both the series and in real-life, Diana and Dodi - along with their chauffeur, Henri Paul - were killed on 31 August 1997 when their Mercedes-Benz was
pursued by paparazzi and crashed in the Pont de l’Alma tunnel in Paris. The mother of two was 36 years old.

The fatal car crash was recreated in the very first scene of season six. While the incident is not shown on-screen, a car carrying Diana and Dodi is seen
speeding past a man walking his dog, before entering the tunnel. As the vehicle’s horn blares, he calls emergency services on his cell phone.

Australian actor Elizabeth Debicki, who took over the role of Diana from Emma Corrin in season five, admitted it was “difficult to recreate” the scene -
especially with her character being hounded by the paparazzi. “It was heavy and very manic, and incredibly invasive.

Graph facts:
- Is Sam Bankman-Fried a bad ‘man’ or a good ‘boy’? Lawyers swap opening statements before first witnesses take the stand -MENTIONS-> Paris
- There’s something going on with AI startups in France -MENTIONS-> Paris
- The best movies of 2023 so far -MENTIONS-> Paris
- How U.S. Women’s Soccer Team Lured Emma Hayes Away From Chelsea -MENTIONS-> Paris
- ‘It happens’: All Blacks defend star centre after spray as Cheika heaps pressure on ‘favourites’ New Zealand -MENTIONS-> Paris

Entities: Paris, Elizabeth Debicki, Diana, Dodi, William, Harry, The Crown, Fans react to most ‘heartbreaking’ scene in The Crown, Henri Paul, fatal car crash,
Mercedes-Benz, Pont de l’Alma tunnel, Emma Corrin, paparazzi, The Independent
</item 9>

<item 10>
Fans react to most ‘heartbreaking’ scene in The Crown (The Independent - Life and Style, 2023-11-17T09:35:32+00:00)
The season begins with Princess Diana (played by Elizabeth Debicki) whisking her sons - Prince William (played by Rufus Kampa) and Prince Harry (played by Fflyn
Edwards) - away on summer holiday to Saint-Tropez, France. After staying several days on Mohamed Al Fayed’s yacht, the boys return home to London where their
father, the then-Prince of Wales, accompanies them to Balmoral Castle to vacation with the rest of the royal family in Scotland.

In episode three, Diana’s partner Dodi Fayed (played by Khalid Abdalla) surprises her with a quick trip to Paris, France. However, the princess is keen to
return home to the United Kingdom, and is eager to speak to her sons over the phone. A mob of paparazzi following the couple delays Diana’s scheduled phone call
with William and Harry, until she’s able to catch them again later that evening.

It is in this scene that The Crown depicts the doting mother’s final conversation with her sons.

Graph facts:
- More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo -MENTIONS-> France
- THE GREATEST! Boks beat All Blacks in instant classic to claim record FOURTH RWC in controversial, dramatic final -MENTIONS-> France
- There’s something going on with AI startups in France -MENTIONS-> France
- Amazon’s iRobot purchase sucks up formal competition concerns in EU -MENTIONS-> France
- ASX set to drop as Wall Street’s September slump deepens -MENTIONS-> France

Entities: France, Paris, royal family, Princess Diana, Elizabeth Debicki, Dodi Fayed, Khalid Abdalla, paparazzi, Mohamed al Fayed, Prince William, Rufus Kampa,
Prince Harry, The Crown, Fans react to most ‘heartbreaking’ scene in The Crown, Fflyn Edwards
</item 10>
=== END ITEM id=47f8a62dfdd4058f ===
