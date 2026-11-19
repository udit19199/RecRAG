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

This file has 5 items.

=== ITEM id=5056d8005cbb96ae ===
Gold sentences and facts:
Sentence 0: Epic will also suggest that Google hid items from discovery by allowing execs and employees to have “off the record” chats, but Google will counter this as well, saying it has provided the court with thousands of chat logs and millions of documents.
  Fact 0.0: Epic will also suggest that Google hid items from discovery.
  Fact 0.1: Epic will suggest that Google hid items from discovery by allowing execs and employees to have "off the record" chats.
  Fact 0.2: Google will counter this, saying it has provided the court with thousands of chat logs.
  Fact 0.3: Google will counter this, saying it has provided the court with millions of documents.
Sentence 1: She admits that she may have misinterpreted the evidence, but maintains that Google manipulates Search to maximise ad revenue.
  Fact 1.0: The woman described admits she may have misinterpreted the evidence.
  Fact 1.1: The woman described maintains that Google manipulates Search to maximise ad revenue.
Sentence 2: The case, filed by Arkansas-based publisher Helena World Chronicle, argues that Google “siphons off” news publishers’ content, their readers and ad revenue through anticompetitive means.
  Fact 2.0: The case was filed by the Arkansas-based publisher Helena World Chronicle.
  Fact 2.1: The case argues that Google "siphons off" news publishers' content.
  Fact 2.2: The case argues that Google siphons off news publishers' readers.
  Fact 2.3: The case argues that Google siphons off news publishers' ad revenue.
  Fact 2.4: The case argues that Google does this through anticompetitive means.

Retrieved items:
<item 1>
News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line (TechCrunch, 2023-12-15T17:56:02+00:00)
“When a user searches for information on a topic, Google displays a ‘Knowledge Panel’ to the right of the search results. This panel contains a summary of
content drawn from the Knowledge Graph database,” the complaint states. “Google compiled this massive database by extracting information from Publishers’
websites — what Google calls ‘materials shared across the web’ —and from ‘open source and licensed databases,'” it says.

By 2020, the Knowledge Graph had grown to 500 billion facts about 5 billion entities. But much of the “collective intelligence” that Google tapped into was
content “misappropriated from Publishers,” the complaint alleges.

Other Google technologies, like “Featured Snippets” where Google algorithmically extracts answers from webpages, were also cited as shifting traffic away from
publishers’ websites.

More importantly, perhaps, is the suit’s tackling of how AI will impact publishers’ businesses. The problem was recently detailed in a report on Thursday by The
Wall Street Journal, which led with a shocking statistic.
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
News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line (TechCrunch, 2023-12-15T17:56:02+00:00)
A Google spokesperson offered a statement on the lawsuit, saying “This lawsuit is meritless. People have many ways to access information and news content today
– through publishers’ websites, dedicated apps, social media platforms, print papers and more. Google links people to publishers’ websites more than 24 billion
times each month – at no cost to them.”

The complaint is available below.

Helena World Chronicle, LLC v. Google LLC and Alphabet Inc by TechCrunch on Scribd

Editor’s note: This post was updated after publication with a statement provided by Google on Friday evening.
</item 3>

<item 4>
Is Google Search better than the rest? And is that fair? (The Age, 2023-10-22T18:00:00+00:00)
The question remains, why do users prefer Google? And if the answer, as Walker says, is that Google is simply better, the question becomes whether that’s down
to its scale, its longevity, its ubiquity and all the user data it sucks up, which no other company could possibly match.

For the US, which has to prove that Google somehow broke the law to achieve the status quo, this legal case could be an uphill battle. But for the rest of us,
it offers a rare opportunity for some insight into what the tech giant does with its enormous market share. Is it still focused on being the best? Or, as has
been alleged, does it use its ubiquity to squeeze us for money even at the expense of product quality? How ‘organic’ are Google’s results? This month, Wired
posted an opinion article by lawyer and privacy advocate Megan Gray, which alleged that Google had accidentally revealed during the trial that it manipulates
people’s search queries for maximum ad revenue.
</item 4>

<item 5>
News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line (TechCrunch, 2023-12-15T17:56:02+00:00)
When online magazine The Atlantic modeled what would happen if Google integrated AI into search, it found that 75% of the time the AI would answer the user’s
query without requiring a click-through to its website, losing it traffic. This could have a major impact on publishers’ traffic going forward, as Google today
drives nearly 40% of their traffic, according to data from Similarweb.

Some publishers are now trying to get ahead of the problem. For example, Axel Springer just this week inked a deal with OpenAI to license its news for AI model
training. But overall, publishers believe they’ll lose somewhere between 20-40% of their website traffic when Google’s AI products fully roll out, The WSJ’s
report noted.

The lawsuit reiterates this concern, claiming that Google’s recent advances in AI-based search were implemented with “the goal of discouraging end-users from
visiting the websites of Class members who are part of the digital news and publishing line of commerce.”
</item 5>

<item 6>
Is Google Search better than the rest? And is that fair? (The Age, 2023-10-22T18:00:00+00:00)
the list of blue links in search results that are not sponsored. “Ad keyword matching is a long-standing and well-known process that is designed to connect
people to relevant ads. A separate process, which has nothing to do with ads, is used to match organic results to a query,” he said. “It’s no secret that Google
Search looks beyond the specific words in a query to better understand their meaning ... If you make a spelling mistake, or search for a term that’s not on a
page but where the page has a close synonym, or if you aren’t even sure exactly how to search for something, our meaning matching systems help.” It is true that
some search queries result in more ads than others, and what ads you get can vary wildly depending on your request, which is the result of the keyword matching
Google uses to sell ads. But Google denies pushing users towards the most monetisable subjects.
</item 6>

<item 7>
News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line (TechCrunch, 2023-12-15T17:56:02+00:00)
The case also arrives alongside the filing of the U.S. Justice Department’s lawsuit against Google for monopolizing digital ad technologies, and references the
2020 Justice Department’s civil antitrust suit over search and search advertising (which are different markets from digital ad technologies in the more recent
suit).

“The anticompetitive effects of Google’s scheme cause profound harm to competition, to consumers, to labor, and to a democratic free press,” reads an
announcement posted to the website of the law firm handling the case, Hausfeld.

“Plaintiff Helena World Chronicle, LLC invokes the Sherman Act and Clayton Act to seek class-wide monetary and injunctive relief to restore and ensure
competition for digital news and reference publishing and set up guardrails to preserve a free marketplace of ideas in the new era of artificial intelligence,”
it states.

A Google spokesperson offered a statement on the lawsuit, saying “This lawsuit is meritless.
</item 7>

<item 8>
Is Google Search better than the rest? And is that fair? (The Age, 2023-10-22T18:00:00+00:00)
The example given was replacing a search for “children’s clothing” with “NIKOLAI-brand kidswear”. Loading Google rejected this in very strong terms, saying the
piece was misleading and inaccurate while denying ever altering search terms. Wired removed the article for not meeting its standards, but the degree to which
it was shared on social media and boosted in write-ups at other outlets shows just how willing people are to accept foul play on Google’s part. A suspicion of
privacy invasion and inappropriate data handling follows the company at all times.

Grey herself is a former vice-president at DuckDuckGo, a privacy-focused company founded explicitly to counter giants such as Google. She admits that she may
have misinterpreted the evidence, but maintains that Google manipulates Search to maximise ad revenue. So what does Google say? In a post on X, Google’s
official search liaison, Danny Sullivan, said ad systems do not affect the organic results, i.e. the list of blue links in search results that are not
sponsored.
</item 8>

<item 9>
News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line (TechCrunch, 2023-12-15T17:56:02+00:00)
A new class action lawsuit filed this week in the U.S. District Court in D.C. accuses Google and parent company Alphabet of anticompetitive behavior in
violation of U.S. antitrust law, the Sherman Act, and others, on behalf of news publishers. The case, filed by Arkansas-based publisher Helena World Chronicle,
argues that Google “siphons off” news publishers’ content, their readers and ad revenue through anticompetitive means. It also specifically cites new AI
technologies like Google’s Search Generative Experience (SGE) and Bard AI chatbot as worsening the problem.

In the complaint, Helena World Chronicle, which owns and publishes two weekly newspapers in Arkansas, argues that Google is “starving the free press” by sharing
publishers’ content on Google, losing them “billions of dollars.”

In addition to new AI technologies, the suit points to Google’s older question-and-answer technologies, like the “Knowledge Graph” launched in May 2012, as part
of the problem.
</item 9>

<item 10>
Is Google Search better than the rest? And is that fair? (The Age, 2023-10-22T18:00:00+00:00)
Some critics, including Megan Grey, contend that Google uses semantic keyword matching to make searches less precise on purpose, which widens the pool of
content served and increases auctions for its ads. The quality tests, then, would ensure that the systems were still returning results that were justifiably
related to the input.
</item 10>
=== END ITEM id=5056d8005cbb96ae ===

=== ITEM id=b47d9538d9962352 ===
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
</item 1>

<item 2>
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
</item 2>

<item 3>
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
</item 4>

<item 5>
Fly with Delta frequently? Small business owners can score a solid welcome bonus and other perks with the Delta SkyMiles Gold Business Card (Fortune,
2023-12-12T21:01:43+00:00)
Cardholders can get 20% back as a statement credit when they purchase eligible pre-purchased meals, in-flight food purchases, alcoholic beverages, and audio
headsets on Delta flights. Baggage insurance. Cardholders who purchase plane, train, ship or bus ride tickets with their card are eligible for baggage insurance
coverage—which kicks in if your baggage is stolen, lost, or damaged.

Cardholders who purchase plane, train, ship or bus ride tickets with their card are eligible for baggage insurance coverage—which kicks in if your baggage is
stolen, lost, or damaged. Car Rental Loss and Damage Insurance. A cardholder can get damage and theft insurance on their car rental. Cardholders must use their
card to book the rental and decline the collision damage waiver at the agency to receive it.
</item 5>

<item 6>
Fly with Delta frequently? Small business owners can score a solid welcome bonus and other perks with the Delta SkyMiles Gold Business Card (Fortune,
2023-12-12T21:01:43+00:00)
For $99 a year, cardholders receive benefits like priority boarding, the first checked bag benefit (and for guests), discounts on in-flight purchases, and
baggage insurance. The card, however, doesn’t provide access to Delta Sky Club or Delta’s airport lounges.

The rewards structure on this card is decent, but it’s nothing to write home about—the card offers 2x miles on just a handful of categories like advertising,
shipping, and dining.

A big benefit of the Delta SkyMiles Gold Business Card is the welcome bonus: 75,000 Bonus Miles after you spend $6,000 in purchases on your new Card in your
first 6 months of card opening. And cardholders who use their miles to purchase flights get even more benefits—customers receive a 15% discount when they use
miles to buy Delta flights on the Fly Delta app or the Delta website.
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
</item 7>

<item 8>
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
</item 9>

<item 10>
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
</item 10>
=== END ITEM id=b47d9538d9962352 ===

=== ITEM id=81f50af75b881559 ===
Gold sentences and facts:
Sentence 0: After learning of Diana’s death from his father at the royal residence of Balmoral in Scotland, William (Rufus Kampa) is left heartbroken with grief and angry at his family’s response to it.
  Fact 0.0: William learned of Diana’s death from his father.
  Fact 0.1: William learned of Diana’s death at the royal residence of Balmoral in Scotland.
  Fact 0.2: The character William is played by Rufus Kampa.
  Fact 0.3: William is left heartbroken with grief.
  Fact 0.4: William is angry at his family’s response to Diana’s death.
Sentence 1: The latest installment takes place between the late 1990s to mid-2000s, and includes the lead-up to Princess Diana’s fatal Paris car crash in 1997.
  Fact 1.0: The latest installment takes place between the late 1990s and the mid-2000s.
  Fact 1.1: The latest installment includes the lead-up to Princess Diana’s fatal Paris car crash.
  Fact 1.2: Princess Diana’s fatal Paris car crash happened in 1997.

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
</item 1>

<item 2>
Fans react to most ‘heartbreaking’ scene in The Crown (The Independent - Life and Style, 2023-11-17T09:35:32+00:00)
It is in this scene that The Crown depicts the doting mother’s final conversation with her sons.

William, then 15, and Harry, then 12, inquire over the phone when their mother will be coming back home from Paris. “I’m coming home tomorrow, and I’m seeing
you both tomorrow night,” Diana replies in the episode, which also takes place on the same day as her fatal car crash later that evening. The boys then ask
Diana whether she plans on marrying Dodi, considering their relationship has been plastered all over the tabloids.

Diana’s poignant last words to William and Harry shown in The Crown scene

“Well, you know better than to believe the papers,” she tells William, before adding: “I’m emphatically not going to marry Dodi. To be honest, I can’t wait to
come home.”

When William sweetly asks his mother if she’s doing “okay”, Diana candidly tells her eldest son: “I’m okay. It’s just a bit mad here.
</item 2>

<item 3>
The Crown: What is fact and what is fiction on season 6? (The Independent - Life and Style, 2023-11-17T19:32:16+00:00)
Recalling their discussion to the Queen, Blair says that Diana feels she “still has a lot to offer the country as a public servant”. He says that she had asked
if there was a way to work with the government on “a more formal basis” and that any “official role” would be appreciated.

Bertie Carvel as Blair (Netflix)

Following Diana’s death, Downing Street confirmed that Diana had visited Blair at Chequers with her son a few weeks before, where they discussed a special role
for her as an overseas “ambassador” for Britain. Diana told a journalist at the time that the prime minister had recognised her skills and asked her to
undertake “missions” abroad for Britain.

Queen Elizabeth’s reluctance to accept Camilla

As The Crown season six begins, Charles is frustrated at the public for demonising Camilla long after his split from Diana.
</item 3>

<item 4>
The Crown: What is fact and what is fiction on season 6? (The Independent - Life and Style, 2023-11-17T19:32:16+00:00)
It was later believed to have been delivered to their room before they went to Dodi’s flat, where a ring bearing the words “Dis-moi Oui” (“Tell me Yes”) – also
the name of episode three – was later recovered, alongside a receipt for a “bague de fiançaille” (engagement ring). Speaking at the inquest, his father Mohamed
claimed that the pair had met the jeweller in Monte Carlo during the holiday on the yacht, and that he believed Dodi planned to propose that night.

Prince William goes missing in Balmoral after Diana’s death

After learning of Diana’s death from his father at the royal residence of Balmoral in Scotland, William (Rufus Kampa) is left heartbroken with grief and angry
at his family’s response to it. At one point, Charles is informed that the prince is not in his room and that “no one can find him”, prompting Charles and
Prince Harry to look for him and a subsequent hunt across the grounds for the young royal.
</item 4>

<item 5>
Fans react to most ‘heartbreaking’ scene in The Crown (The Independent - Life and Style, 2023-11-17T09:35:32+00:00)
“The last call between Harry, William and Diana is heartbreaking,” a third user commented.

In both the series and in real-life, Diana and Dodi - along with their chauffeur, Henri Paul - were killed on 31 August 1997 when their Mercedes-Benz was
pursued by paparazzi and crashed in the Pont de l’Alma tunnel in Paris. The mother of two was 36 years old.

The fatal car crash was recreated in the very first scene of season six. While the incident is not shown on-screen, a car carrying Diana and Dodi is seen
speeding past a man walking his dog, before entering the tunnel. As the vehicle’s horn blares, he calls emergency services on his cell phone.

Australian actor Elizabeth Debicki, who took over the role of Diana from Emma Corrin in season five, admitted it was “difficult to recreate” the scene -
especially with her character being hounded by the paparazzi. “It was heavy and very manic, and incredibly invasive.
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
</item 6>

<item 7>
Fans react to most ‘heartbreaking’ scene in The Crown (The Independent - Life and Style, 2023-11-17T09:35:32+00:00)
It’s just a bit mad here. I don’t really understand how I ended up here. Mummy just needs to make some changes to her life, that’s all. But that’s not your
problem, that’s mine.”

Before saying their goodbyes, Diana reminds William and Harry that she will be seeing them the next day - although that day never comes. “I’ll see you both
tomorrow night,” she says. “Just the three of us.”

The moment unsurprisingly tugged on the heartstrings of many viewers, who took to social media to share their thoughts on the devastating moment.

“The last phone call between Harry, William and Diana broke my heart,” one fan wrote on X, formerly known as Twitter.

“That precious phone call between Diana and her sons,” said someone else. “Don’t mind me, I’m just gonna cry in one corner.”

“The last call between Harry, William and Diana is heartbreaking,” a third user commented.
</item 7>

<item 8>
Fans react to most ‘heartbreaking’ scene in The Crown (The Independent - Life and Style, 2023-11-17T09:35:32+00:00)
Stay ahead of the trend in fashion and beyond with our free weekly Lifestyle Edit newsletter Stay ahead of the trend in fashion and beyond with our free weekly
Lifestyle Edit newsletter Please enter a valid email address Please enter a valid email address SIGN UP I would like to be emailed about offers, events and
updates from The Independent. Read our privacy notice Thanks for signing up to the

Lifestyle Edit email {{ #verifyErrors }} {{ message }} {{ /verifyErrors }} {{ ^verifyErrors }} Something went wrong. Please try again later {{ /verifyErrors }}

The first four episodes of The Crown season six are officially here, but many viewers believe one scene has depicted the most devastating moment throughout the
entire series.

On 16 November, Netflix dropped the first four episodes of The Crown’s final season, followed by the last six episodes on 14 December. The latest installment
takes place between the late 1990s to mid-2000s, and includes the lead-up to Princess Diana’s fatal Paris car crash in 1997.
</item 8>

<item 9>
Fans react to most ‘heartbreaking’ scene in The Crown (The Independent - Life and Style, 2023-11-17T09:35:32+00:00)
The season begins with Princess Diana (played by Elizabeth Debicki) whisking her sons - Prince William (played by Rufus Kampa) and Prince Harry (played by Fflyn
Edwards) - away on summer holiday to Saint-Tropez, France. After staying several days on Mohamed Al Fayed’s yacht, the boys return home to London where their
father, the then-Prince of Wales, accompanies them to Balmoral Castle to vacation with the rest of the royal family in Scotland.

In episode three, Diana’s partner Dodi Fayed (played by Khalid Abdalla) surprises her with a quick trip to Paris, France. However, the princess is keen to
return home to the United Kingdom, and is eager to speak to her sons over the phone. A mob of paparazzi following the couple delays Diana’s scheduled phone call
with William and Harry, until she’s able to catch them again later that evening.

It is in this scene that The Crown depicts the doting mother’s final conversation with her sons.
</item 9>

<item 10>
The Crown: What is fact and what is fiction on season 6? (The Independent - Life and Style, 2023-11-17T19:32:16+00:00)
With the show heading into the late Nineties and early Noughties in its final season, which arrived on Netflix on Thursday (16 November), events are being
depicted on screen that many of the public will remember well.

The first part of season six takes place in the summer of 1997, and sees Prince Charles (Dominic West) building on his relationship with Camilla Parker Bowles
(Olivia Williams), while Princess Diana (Elizabeth Debicki) sparks up a romance with film producer Dodi Fayed (Khalid Abdalla). The show also features the
pair’s death in a car crash in Paris.

As the new season arrives, and fans wonder what in The Crown is based in reality, here’s everything you need to know.

Diana’s role with Tony Blair’s government

Episode one sees Diana visiting Tony (Bertie Carvel) and Cherie Blair (Lydia Leonard) with William after asking for a meeting with him.
</item 10>
=== END ITEM id=81f50af75b881559 ===

=== ITEM id=43ba01a64c469485 ===
Gold sentences and facts:
Sentence 0: He revealed that he made a friendship bracelet, a fan tradition at her concerts, with his number on it to give to her, seemingly expressing his interest in Swift.
  Fact 0.0: He revealed that he made a friendship bracelet.
  Fact 0.1: A friendship bracelet is a fan tradition at Swift's concerts.
  Fact 0.2: The friendship bracelet had his number on it.
  Fact 0.3: He made the friendship bracelet to give to Swift.
  Fact 0.4: The friendship bracelet seemingly expresses his interest in Swift.
Sentence 1: “She’s got the Eras Tour booked all through next year, and she’d rather die than cancel a show.
  Fact 1.0: Swift has the Eras Tour booked all through next year.
  Fact 1.1: Swift would rather die than cancel a show.
Sentence 2: Now, Swift has shared that she has nothing to hide in her new relationship with Kelce.
  Fact 2.0: Swift has now shared that she has nothing to hide.
  Fact 2.1: Swift has a new relationship with Kelce.
  Fact 2.2: Swift has now shared that she has nothing to hide in her new relationship with Kelce.

Retrieved items:
<item 1>
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
</item 1>

<item 2>
Taylor Swift-Travis Kelce timeline: Everything to know about rumored romance between pop star, Chiefs TE (CBSSports.com, 2023-10-12T23:21:52+00:00)
Here's a look at how it all started and where it has gone since:

June 29: Kelce makes friendship bracelets for fellow TEs

Is this where it all begin? Kelce made a friendship bracelet to give to the newcomers at a camp for NFL tight ends. He said he was "taking a page out of Taylor
Swift's book."

July 26: Kelce makes friendship bracelet for Swift

"So make the friendship bracelets, take the moment and taste it." - You're On Your Own Kid

Kelce announced that he went to Swift's highly anticipated concert, but said he did not have the chance to meet the 12-time Grammy winner.
</item 2>

<item 3>
Taylor Swift reveals secret start to her relationship with Travis Kelce (The Independent - Life and Style, 2023-12-06T14:23:01+00:00)
“This all started when Travis very adorably put me on blast on his podcast, which I thought was metal as hell,” she explained. “We started hanging out right
after that. So we actually had a significant amount of time that no one knew, which I’m grateful for, because we got to get to know each other.”

Swift also confirmed that when she famously first attended Kelce’s game at Arrowhead Stadium in September, she was already dating the NFL star.

“By the time I went to that first game, we were a couple. I think some people think that they saw our first date at that game? We would never be psychotic
enough to hard launch a first date,” she said.

The “Blank Space” singer also acknowledged that although her love life continues to make headlines with every Kansas City Chiefs game she attends, she’s focused
on being there for her partner. She specified that she isn’t trying to hide her relationship from the public eye.
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
</item 5>

<item 6>
Taylor Swift-Travis Kelce timeline: Everything to know about rumored romance between pop star, Chiefs TE (CBSSports.com, 2023-10-12T23:21:52+00:00)
Some found issue with these changes, but the NFL said this is a "pop cultural moment" and they are acting as such.

"We frequently change our bios and profile imagery based on what's happening in and around our games, as well as culturally," the NFL wrote in a statement. "The
Taylor Swift and Travis Kelce news has been a pop cultural moment we've leaned into in real time, as it's an intersection of sport and entertainment, and we've
seen an incredible amount of positivity around the sport."

Oct. 6: Travis and Jason think the NFL is doing too much

The NFL is really digging into the Swift-Kelce coverage, and not everyone is a fan, including the people involved.

"I think it's fun when they show who's at the game," Travis said. "I think it brings a little bit more to the atmosphere, brings a little bit more to what
you're watching. But at the same time, they're overdoing it a little bit."
</item 6>

<item 7>
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
</item 7>

<item 8>
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
</item 8>

<item 9>
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
</item 9>

<item 10>
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
</item 10>
=== END ITEM id=43ba01a64c469485 ===

=== ITEM id=5dbb56b9066bdd54 ===
Gold sentences and facts:
Sentence 0: She explained at the time of the 2022 Academy Awards, when Will slapped Chris Rock, they had been separated for six years.
  Fact 0.0: The speaker explained this at the time of the 2022 Academy Awards.
  Fact 0.1: Will slapped Chris Rock at the 2022 Academy Awards.
  Fact 0.2: At the time Will slapped Chris Rock, the speaker and Will had been separated for six years.
Sentence 1: Outsiders are now aware that Jada and Will have been separated since 2016, but the timeline of her relationship with Alsina is still fuzzy.
  Fact 1.0: Outsiders are now aware that Jada and Will have been separated since 2016.
  Fact 1.1: The timeline of Jada's relationship with Alsina is still fuzzy.

Retrieved items:
<item 1>
Jada Pinkett Smith and Will Smith have been separated since 2016 - when should couples just get divorced? (The Independent - Life and Style,
2023-10-16T13:41:30+00:00)
In an interview with People on 10 October, Jada shared that during the 2022 Academy Awards - when Will slapped Chris Rock after he cracked a joke about Jada’s
appearance - the pair had actually been separated for six years. “We’ve been doing some really heavy-duty work together,” she explained, while promoting her new
memoir, Worthy. “We just got deep love for each other and we are going to figure out what that looks like for us.”

She praised her and Will’s two children - Jaden, 25, and Willow, 22 - for being her “little gurus” throughout this time, noting that they’ve taught her “a deep
sense of self-acceptance”. Although the married couple are not living together, she insisted that their marriage has never been an open one, as she and Will
have “eliminated the chance of betrayal” while going on to live their “separate lives”.
</item 1>

<item 2>
Jada Pinkett Smith and Will Smith have been separated since 2016 - when should couples just get divorced? (The Independent - Life and Style,
2023-10-16T13:41:30+00:00)
This isn’t the first time that Jada and Will have taken some time apart from each other. During a 2020 episode of her Facebook Watch series, Red Table Talk,
Jada disclosed that while she and Will were separated amicably, she had a relationship with singer August Alsina, describing their involvement as an
“entanglement”.

Throughout their marriage of 26 years, the A-listers have made it clear that divorce was never in the cards. Jada furthered that belief during a preview for an
interview with Today’s Hoda Kotb, which aired on 13 October. “I made a promise that there will never be a reason for us to get a divorce,” she said. “We will
work through, whatever. I just haven’t been able to break that promise.”

Of course, outsiders aren’t fully aware of the “heavy-duty work” that Jada and Will have done throughout their marriage.
</item 2>

<item 3>
Will Smith speaks to crowd about ‘tumultuous’ relationship with Jada Pinkett Smith (The Independent - Life and Style, 2023-10-19T16:16:20+00:00)
Stay ahead of the trend in fashion and beyond with our free weekly Lifestyle Edit newsletter Stay ahead of the trend in fashion and beyond with our free weekly
Lifestyle Edit newsletter Please enter a valid email address Please enter a valid email address SIGN UP I would like to be emailed about offers, events and
updates from The Independent. Read our privacy notice Thanks for signing up to the

Lifestyle Edit email {{ #verifyErrors }} {{ message }} {{ /verifyErrors }} {{ ^verifyErrors }} Something went wrong. Please try again later {{ /verifyErrors }}

Will Smith has spoken out about his “tumultuous” relationship with his wife, Jada Pinkett Smith, while speaking to a crowd of fans in Baltimore, Maryland.

The 55-year-old actor took the stage at the Enoch Pratt Free Library on 18 October, as he made an appearance at a press event for his wife’s book, Worthy,
alongside their two children – Jaden, 25, and Willow, 22.
</item 3>

<item 4>
Will Smith speaks to crowd about ‘tumultuous’ relationship with Jada Pinkett Smith (The Independent - Life and Style, 2023-10-19T16:16:20+00:00)
Prior to sharing the video, Smith showed his support for his partner and her success in a statement toThe New York Times. During the interview, the outlet noted
that Smith said his wife’s memoir “kind of woke him up” and that he has now realised she is more “resilient, clever and compassionate than he’d understood”.

“When you’ve been with someone for more than half of your life, a sort of emotional blindness sets in, and you can all too easily lose your sensitivity to their
hidden nuances and subtle beauties,” he added.

In an interview with People published on 10 October, Jada first revealed that she and her husband were living separately. “We’ve been doing some really
heavy-duty work together,” she said. “We just got deep love for each other and we are going to figure out what that looks like for us.”
</item 4>

<item 5>
A timeline of Will Smith and Jada Pinkett Smith’s relationship amid separation (The Independent - Life and Style, 2023-10-12T18:10:45+00:00)
he wrote on Facebook at the time. “So, in the interest of redundant, repetitious, over and over-again-ness… Jada and I are… NOT GETTING A DIVORCE!!!!!!!!!!!!!”

2018: Will revealed that he and Jada don’t refer to themselves as married

In an episode of TIDAL’s Rap Radar podcast, Will said he and Jada “refer to ourselves as life partners, where you get into that space where you realise you are
literally with somebody for the rest of your life. There’s no deal breakers. There’s nothing she could do - ever. Nothing that would break our relationship. She
has my support till death and it feels so good to get to that space.”
</item 5>

<item 6>
A timeline of Will Smith and Jada Pinkett Smith’s relationship amid separation (The Independent - Life and Style, 2023-10-12T18:10:45+00:00)
Less than two weeks later, Jada and Will appeared on an episode of Jada’s Red Table Talk where Jada confirmed that she did have a relationship with Alsina,
saying that she and Will were separated at the time.

“We decided that we were going to separate for a period of time and you go figure out how to make yourself happy and I’ll figure out how to make myself happy,”
Jada said at the time.

“From there, as time went on, I got into a different kind of entanglement with August... Yes, it was a relationship absolutely.”

Jada also said that Will didn’t give permission, as the only person who could give her permission was herself.

2021: Jada discusses her and Will’s sex life, Will’s memoir is released

In another episode of Red Table Talk in October 2021, Jada said it “takes work” to maintain a spark in the bedroom.

“It’s hard... the thing Will and I talk about a lot is the journey.
</item 6>

<item 7>
A timeline of Will Smith and Jada Pinkett Smith’s relationship amid separation (The Independent - Life and Style, 2023-10-12T18:10:45+00:00)
Stay ahead of the trend in fashion and beyond with our free weekly Lifestyle Edit newsletter Stay ahead of the trend in fashion and beyond with our free weekly
Lifestyle Edit newsletter Please enter a valid email address Please enter a valid email address SIGN UP I would like to be emailed about offers, events and
updates from The Independent. Read our privacy notice Thanks for signing up to the

Lifestyle Edit email {{ #verifyErrors }} {{ message }} {{ /verifyErrors }} {{ ^verifyErrors }} Something went wrong. Please try again later {{ /verifyErrors }}

Jada Pinkett Smith has revealed that she and husband Will Smith have been separated for the last seven years.

The Girls Trip star, 52, shared in a new interview with People published on 11 October that she and the King Richard actor, 55, are “still figuring it out” when
it comes to their marriage of 26 years. She explained at the time of the 2022 Academy Awards, when Will slapped Chris Rock, they had been separated for six
years.
</item 7>

<item 8>
A timeline of Will Smith and Jada Pinkett Smith’s relationship amid separation (The Independent - Life and Style, 2023-10-12T18:10:45+00:00)
Will and Jada with their son Jayden in 2003 (Getty)

2013: Jada denies open marriage rumours (sort of)

After a quiet decade and a half making movies and raising their children, Jada took to Facebook to address rumours she and Will were in an open relationship.

She wrote: “Here is how I will change my statement... Will and I BOTH can do WHATEVER we want, because we TRUST each other to do so. This does NOT mean we have
an open relationship... this means we have a GROWN one.”

2015: Will addresses divorce rumours

Will addresses persistent separation rumours by clarifying the pair were not getting a divorce.

“Under normal circumstances, I don’t usually respond to foolishness. But, so many people have extended [to] me their ‘deepest condolences’ that I figured -
‘What the hell… I can be foolish, too!’” he wrote on Facebook at the time.
</item 8>

<item 9>
Jada Pinkett Smith and Will Smith have been separated since 2016 - when should couples just get divorced? (The Independent - Life and Style,
2023-10-16T13:41:30+00:00)
Although they’ve made their dedication to each other quite clear, there have been some grey areas over the last seven years about the terms of Will and Jada’s
separation. In 2020, Jada made headlines when she revealed in a Red Table Talk episode with her husband that she had a relationship with Alsina. She explained
that she and her husband were “separated amicably” when the romance occurred, which had not been disclosed to the press at the time. “We decided that we were
going to separate for a period of time and you go figure out how to make yourself happy and I’ll figure out how to make myself happy,” Jada said. As the Girls
Trip star recalled that she “got into a different kind of entanglement with August,” she was prompted by Will to confirm it “was a relationship, absolutely.”
</item 9>

<item 10>
Jada Pinkett Smith and Will Smith have been separated since 2016 - when should couples just get divorced? (The Independent - Life and Style,
2023-10-16T13:41:30+00:00)
Stay ahead of the trend in fashion and beyond with our free weekly Lifestyle Edit newsletter Stay ahead of the trend in fashion and beyond with our free weekly
Lifestyle Edit newsletter Please enter a valid email address Please enter a valid email address SIGN UP I would like to be emailed about offers, events and
updates from The Independent. Read our privacy notice Thanks for signing up to the

Lifestyle Edit email {{ #verifyErrors }} {{ message }} {{ /verifyErrors }} {{ ^verifyErrors }} Something went wrong. Please try again later {{ /verifyErrors }}

Longtime Hollywood couple Will Smith and Jada Pinkett Smith sent shockwaves around the world this week when the Red Table Talk host revealed that she and the
actor have been separated since 2016. Both Jada and Will have previously doubled down on their stance against getting a divorce, but her recent reveal about
their lengthy separation poses a new question: When should couples just get a divorce?
</item 10>
=== END ITEM id=5dbb56b9066bdd54 ===
