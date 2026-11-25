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

This file has 1 items.

=== ITEM id=8b0542150af918b6 ===
Gold sentences and facts:
Sentence 0: The first of these is the fact that tech companies continue to see huge pressure on their valuations.
  Fact 0.0: The first of these is that tech companies continue to see huge pressure on their valuations.
Sentence 1: With a 10% surge in the stock this year, Seeking Alpha's Quant Rating system suggests a Hold, expressing concerns about valuation.
  Fact 1.0: The stock surged 10% this year.
  Fact 1.1: Seeking Alpha's Quant Rating system suggests a Hold.
  Fact 1.2: The Quant Rating system expresses concerns about valuation.

Retrieved items:
<item 1>
A comprehensive list of 2023 tech layoffs (TechCrunch, 2023-09-28T20:04:20+00:00)
But tracking these layoffs helps us to understand the impact on innovation, which companies are facing tough pressures and who is available to hire for the
businesses lucky to be growing right now. Unfortunately, it also serves as a reminder of the deeply human impact of layoffs and how risk profiles could evolve
from here.

Below you’ll find a comprehensive list of all the known layoffs in tech that have occurred in 2023, to be updated monthly. If you have a tip on a layoff,
contact us here. If you prefer to remain anonymous, you can contact us here.

The running total of layoffs for 2023 based on full months to date is 224,503, according to Layoffs.fyi. Tech layoffs conducted to date this year currently
exceed the total number of tech layoffs in 2022, according to the data in the tracker.

Data visualization by Miranda Halpern, created with Flourish

December 2023

ShareChat

Is cutting 15% of its workforce in a “strategic restructuring” effort for 2024.
</item 1>

<item 2>
A comprehensive list of 2023 tech layoffs (TechCrunch, 2023-09-28T20:04:20+00:00)
The exact number of layoffs is unclear. Netflix is scheduled to report Q1 2023 results on April 18.

Roku

Announced March 30 that it is letting go of about 200 employees, or 6% of its workforce. The company had laid off 200 U.S. employees back in November.

Unacademy

Announced March 30 that it has laid off more than 350 roles, or 12% of its workforce — just over four months after cutting about 350 roles in November.

Shift Technologies

Announced March 29 that it laid off 30% of its workforce in Q1 2023.

Lucid

Announced March 28 that it is laying off 1,300 employees, or 18% of its workforce, to be completed by the end of Q2 2023.

GitHub

Announced on March 28 that it has eliminated over 100 jobs in the South Asian market, laying off virtually its entire engineering team in India.
</item 2>

<item 3>
Hipgnosis Songs Fund’s board is concerned over Citrin Cooperman’s latest valuation of the company. What’s it going to do about it? (Music Business Worldwide,
2023-12-19T23:02:01+00:00)
Worth remembering: The HSF board has known about Citrin Cooperman’s latest valuation (as of September 30) for at least the past eight days.

We know this because, as mentioned, on December 11, HSF announced that the $23.1 million price of an agreed HSF ‘non-core’ asset sale represented a 14.2%
discount vs. Citrin Cooperman’s latest valuation of the portfolio (i.e. the HSF board had seen this latest valuation).

Yet it took until today (December 19) for HSF to alert its investors to its dissatisfaction with this valuation.

WHAT does JP Morgan make of it all?

One of the closest watchers of Hipgnosis Songs Fund in recent years has been JP Morgan and its analyst Christopher Brown.
</item 3>

<item 4>
OpenAI hosts a dev day, TechCrunch reviews the M3 iMac and MacBook Pro, and Bumble gets a new CEO (TechCrunch, 2023-11-11T21:15:53+00:00)
So far, its plan to forgo the gigafactory for local microfactories has proved anything but, writes Harri — thanks to missed production targets, low cash
reserves, layoffs and a pivot.

Audio

It’s winter, it ain’t getting warmer (at least here in NYC), and I’d argue that there’s no better place to be than snuggled up indoors with a podcast for
company. If you’re in need of material, TechCrunch has a few that should definitely be on your radar.

This week on Equity, the crew dove deep into the encouraging signs from the fintech startup market, starting with Klarna’s Q3 results. From there, they looked
at buy now, pay later consumer behavior and fintech fundraising results with a 2021 flavor.

Meanwhile, Found featured Nasrat Khalid of Aseel, which started as an e-commerce company making it possible for local artisans in Afghanistan to sell to
customers across the world.
</item 4>

<item 5>
A comprehensive list of 2023 tech layoffs (TechCrunch, 2023-09-28T20:04:20+00:00)
100 Thieves

Is reducing its workforce by 20%, CEO John Robinson announced November 2. The esports brand is spinning out its game studio as well as its energy drink brand,
Juvee.

OpenSpace

Conducted a round of layoffs, CEO Jeevan Kalanithi announced November 2. The number of employees impacted by the layoffs has yet to be reported.

Informatica

Is laying off 10% of its global workforce, the cloud data management company announced in its Q3 results November 1.

Splunk

Is set to cut around 500 employees CEO Gary Steele said in a message to employees November 1 ahead of its planned acquisition by Cisco.

October 2023

Bungie

Is laying off about 100 employees, CEO Pete Parsons announced 10/30. The Sony-owned game studio is also delaying two of its upcoming titles as a result.

Karat

Reportedly cut an estimated 10% of its workforce October 27 after the financial startup for creators raised a $70 million Series B round earlier this year.
</item 5>

<item 6>
Coinbase rallies more than 60% in same month that FTX and Binance founders brace for prison (Cnbc | World Business News Leader, 2023-11-30T21:08:00+00:00)
The initial boost would come from custody revenue tied to the ETFs. Most of the big asset managers jumping into market, including BlackRock, Franklin Templeton
and WisdomTree, have picked Coinbase for custody services, which involves the storage and safekeeping of the assets. However, the longer-term concern, according
to JPMorgan, is that fewer people will need Coinbase accounts, leading to pricing pressure. "We see many novice investors never going beyond these flagship
tokens and thus never needing the services of a Coinbase," wrote the analysts, who have a neutral rating on the stock and an $80 price target. "We also see the
ETF markets as more transparent, efficient and lower cost to execute and we see the potential for a migration to ETFs for cheaper exposure and trading driving
Coinbase to lower fees." WATCH: Former SEC enforcement chief on 'casualness' in crypto compliance
</item 6>

<item 7>
Google fakes an AI demo, Grand Theft Auto VI goes viral and Spotify cuts jobs (TechCrunch, 2023-12-09T21:16:17+00:00)
Before Rogier launched MasterClass, he worked as a VC, and — through his connections — he received a $500,000 seed round before he even had an idea for a
company.

And on Chain Reaction, Jacquelyn interviewed David Pakman, managing partner and head of venture investments at CoinFund. Before CoinFund, David spent 14 years
at the venture capital firm Venrock. He also led the Series A and B rounds at Dollar Shave Club, which was acquired by Unilever for $1 billion. And, in 1991,
David co-created Apple Music when he was part of Apple’s system software product marketing group.

TechCrunch+

TC+ subscribers get access to in-depth commentary, analysis and surveys — which you know if you’re already a subscriber. If you’re not, consider signing up.
</item 7>

<item 8>
Earnings week ahead: Amazon, Microsoft, Alphabet, Meta, Ford, GM and more (Seeking Alpha, 2023-10-22T12:13:00+00:00)
On a YTD basis, Amazon shares have expanded by almost 50% in value.

"AMZN's cost-cutting initiatives and margin improvements make it a solid buying opportunity, despite concerns such as the FTC lawsuit and a weakening economy."
writes SA Investing Group Leader Danie Jones with a bullish stance on the stock. Meanwhile, fellow Investing Group Leader Sensor Unlimited expects the stock to
continue being range-bound in the near term due to the key challenges on its operating margins, competition in key growth areas, and valuation risks.

Consensus EPS Estimates: $0.60

Consensus Revenue Estimates: $141.621B

Earnings Insight: Amazon has beaten EPS and revenue expectations in half of the past 8 quarters.

Also reporting: Kenvue (KVUE), Intel (INTC), Ford Motor Company (F), Altria Group (MO), Merck & Co.
</item 8>

<item 9>
Earnings week ahead: Amazon, Microsoft, Alphabet, Meta, Ford, GM and more (Seeking Alpha, 2023-10-22T12:13:00+00:00)
Shares of the now-Virginia-based industrial and defense giant have dropped over 7% over the past 6 months, attracting a cautious view of Hold from Seeking
Alpha's Quant Rating system, pointing to valuation concerns. Meanwhile, Wall Street analysts rate the stock a Buy.

During the third quarter, Boeing and its supplier Spirit AeroSystems (SPR) determined that the aft pressure bulkhead on many Boeing 737 MAX airplanes had
erroneously drilled fastening holes. At the 2023 Industrial Conference, CFO Brian West claimed that the current issue will affect 75% of the 220 planes in
inventory, or 165 planes. Despite persistent 737 manufacturing problems, the firm reiterated its ambitions to boost 787 production from three to four per month
by the end of 2023.

The company delivered 27 commercial jets to customers in the month of September, raising its yearly total to 371 planes.
</item 9>

<item 10>
Keep Labs’ smart storage redefines medication management (TechCrunch, 2023-10-04T19:24:25+00:00)
This comprehensive, multi-layered approach to security ensures that patient information remains confidential and secure at all times.”

Wilkins added that its encryption keys are generated on the fly within their production environment and securely stored. No individual has direct access to the
keys. Data is transmitted using TLS 1.3 encryption, while data at rest is secured with AES-256 encryption. User passwords undergo hashing with PBKDF2 utilizing
SHA256. The company uses automated tools such as Scan Hawk and Synk to test its security and provide code vulnerability assessments. He notes that their code is
subject to peer reviews, and the company uses the Coalition Cybersecurity and Stendard to provide a final layer of external review.

It’s worth applauding Keep Lab’s security and privacy measures. The above shows a company that’s proud of its efforts and is comfortable making them public.

Cooling meds is next for Keep

With its first product shipping to consumers, Keep Labs is exploring raising capital at the end of 2023 or early 2024.
</item 10>

<item 11>
Fundamental Investing: The Art of Relative Valuation (Business Line, 2023-10-14T15:38:28+00:00)
But there are many times when company profits are low, but its operational performance ie EBITDA is not bad. In such cases, as revenue and EBITDA continue to
grow, operating leverage will result in PE growing much faster than the above two line items.

For example, take the case of Bharti Airtel: at the start of FY22 (April ‘21) its trailing PE was negative/invalid due to losses in FY21. The stock would have
been avoided if one went by PE alone. However, on trailing EV/EBITDA basis, it was trading at 10 times EV/EBITDA. When this was considered against its estimated
next 2 years EBITDA CAGR of over 20 per cent, the valuation, combined with qualitative factors, made it an interesting buy. Since then, net profits have
improved substantially as operating leverage played out.

At bl.portfolio we had recommended a buy on the stock based on this in April 2021 and the stock has returned 80 per cent since.
</item 11>

<item 12>
A comprehensive list of 2023 tech layoffs (TechCrunch, 2023-09-28T20:04:20+00:00)
Multiverse

Will lay off nearly a third of their U.S. staff. It’s the second round of layoffs at the U.K.-based unicorn company in just over a month.

ONE

Is cutting its workforce by around 25% November 27, citing “market conditions” affecting the EV battery startup.

ByteDance

Started mass layoffs November 27 in its gaming department, Nuverse, after two years of tepid performance. The number of affected employees is unknown.

Veev

Is closing up shop, according to multiple reports, after reaching unicorn status in 2022.

Tulip

Laid off 25 employees, CEO Ali Asaria announced November 24.

Anar

Is shutting down operations and returning capital to investors, CEO Nishank Jain announced November 23.

Alerzo

Cut 100 workers November 21, reportedly due to increases in automation. The recent layoffs come after Alzero cut 15% of its workforce earlier this year and 5%
in 2022.
</item 12>

<item 13>
OpenAI buffs safety team and gives board veto power on risky AI (TechCrunch, 2023-12-18T20:12:25+00:00)
Of course, the result of said drama was the sidelining of two of the more critical voices and the appointment of some money-minded guys (Bret Taylor and Larry
Summers), who are sharp but not AI experts by a long shot.

If a panel of experts makes a recommendation, and the CEO makes decisions based on that information, will this friendly board really feel empowered to
contradict them and hit the brakes? And if they do, will we hear about it? Transparency is not really addressed outside a promise that OpenAI will solicit
audits from independent third parties.

Say a model is developed that warrants a “critical” risk category. OpenAI hasn’t been shy about tooting its horn about this kind of thing in the past — talking
about how wildly powerful their models are, to the point where they decline to release them, is great advertising. But do we have any kind of guarantee this
will happen, if the risks are so real and OpenAI is so concerned about them? Maybe it’s a bad idea. But either way it isn’t really mentioned.
</item 13>

<item 14>
The fate of US venture capital in China teeters on uncertainty (TechCrunch, 2023-11-25T09:32:47+00:00)
VCs fear that their portfolio companies might encounter a fate akin to that of Ant Group, whose colossal initial public offering was called off, and Didi, which
weathered an extensive data security probe that eventually led to its delisting from New York. With China tightening its grip on overseas IPOs, investors who
once relied on taking Chinese firms public in the U.S. are no longer assured of an exit channel.

In the meantime, Washington has stepped up restrictions on the flow of U.S. money into China amid an escalating tech war between the two superpowers. In August,
President Joe Biden signed an executive order barring U.S. investments in three strategically critical sectors in China — artificial intelligence, quantum
computing and semiconductors.

As USD funds in China await further clarity on the scope of the ban, they are practicing more discretion than ever before. They slow down capital deployment
even amidst a global AI fervor that has given rise to a parallel AI universe in China.
</item 14>

<item 15>
A comprehensive list of 2023 tech layoffs (TechCrunch, 2023-09-28T20:04:20+00:00)
September 2023

IronNet

In a regulatory filing published September 29, the cybersecurity startup has ceased all business activities as it prepares for Chapter 7 bankruptcy and has laid
off its remaining staff.

Naughty Dog

Is reportedly cutting at least 25 contract developers, two sources familiar with the situation told Kotaku.

Epic Games

Announced on September 28 that the Fortnite maker is laying off 16% of its workforce, amounting to 870 people.

Talkdesk

An impacted employee confirmed on September 27 that at least 140 people, if not more, were affected by Talkdesk’s third round of layoffs in less than 14 months.

Byju’s

Announced on September 26, the company plans to cut 5,000 jobs due to restructuring efforts. The startup has eliminated more than 10,000 positions in the past
two years.

Roblox

Confirmed to TechCrunch on September 21, Roblox let go of 30 employees in its talent acquisition organization. No other teams are impacted.
</item 15>

<item 16>
Tesla ‘digs its own grave with the Cybertruck,’ Convoy collapses and Rivian scores a win at Rebelle (TechCrunch, 2023-10-22T19:15:29+00:00)
Tesla’s shares, which are priced more as a tech stock than as an automaker’s stock, were down after Q3 earnings. Investors were clearly feeling skittish after
Tesla reported a gross margin of 17.9%, down from 25.1% in the same period last year. That’s also down from Q2’s margins of 18.2%. As a result, profits fell 44%
to $1.85 billion in Q3 from the same year-ago period.

Investors see the falling margins and Tesla’s price cuts as proof that demand is lessening for the vehicles as other EVs take market share and rising interest
rates make it difficult for many buyers to afford big ticket purchases. The company also reported that solar deployments slipped 48% in Q3 from the same period
last year. But the company made up for it by pulling in a 90% spike in energy storage deployments.

Tesla also gave some updates about its long-delayed Cybertruck.
</item 16>

<item 17>
Hipgnosis Songs Fund’s board is concerned over Citrin Cooperman’s latest valuation of the company. What’s it going to do about it? (Music Business Worldwide,
2023-12-19T23:02:01+00:00)
A Final thought…

As mentioned in the opening sentence of this article, there is plenty of drama surrounding HSF right now – from the ‘discontinuation vote’ we saw at the company
in October, through to a recent lawsuit launched against Hipgnosis Songs Fund by a former business partner of Merck Mercuriadis.

Yet, amongst all of the noise, what will likely most interest HSF’s investors (including the likes of Investec Wealth & Investment, Aviva Investors, and
BlackRock) is the ongoing underlying commercial performance of the firm’s copyrights.

How are Hipgnosis Songs Fund’s assets performing vs. the market? How is the company’s ‘premium-only’ acquisition strategy holding up against its promise of
growth “uncorrelated” to other industries? And what does all of this suggest about the ongoing growth potential of the HSF portfolio vs. the multiples that
Merck Mercuriadis paid to acquire it?
</item 17>

<item 18>
The fate of US venture capital in China teeters on uncertainty (TechCrunch, 2023-11-25T09:32:47+00:00)
They soon realized that China had entered a new era, where the regulatory hurdles for running a startup have significantly heightened.

The year 2022 saw just $14.5 billion invested in Chinese companies by U.S.-headquartered VCs, compared to $45.4 billion the year before.

To launch an AI service in China, for example, a company needs to navigate a multitude of complications, which can include obtaining a license for its large
language model, seeking regulatory approval for its algorithms and implementing a costly censorship mechanism to comply with censorship requirements.

“You need to pick sides. You either focus on China or go overseas, otherwise, you end up doing double the workload but with a lot less funding secured from the
last two years,” said one of the five China-based VCs we interviewed for the story. We also spoke to six diaspora Chinese entrepreneurs. Due to the sensitivity
of the topic, all of them have asked to remain anonymous.

Some ambitious, well-funded AI startups want to target both sides.
</item 18>

<item 19>
Earnings week ahead: DocuSign, NIO, Broadcom, GameStop and more (Seeking Alpha, 2023-12-03T13:02:00+00:00)
The first full week of December features several significant companies slated to announce their quarterly results. These earnings releases hold considerable
implications for various sectors, including energy, software, retail, autos and semiconductors.

Among the companies scheduled to release financial figures are NIO (NYSE:NIO), GameStop (NYSE:GME), DocuSign (NASDAQ:DOCU), ChargePoint (NYSE:CHPT), Dollar
General (NYSE:DG), C3.ai (NYSE:AI) and Broadcom (NASDAQ:AVGO). Additionally, reports are expected from Toll Brothers (NYSE:TOL), Campbell Soup (NYSE:CPB),
AutoZone (NYSE:AZO), Lululemon (NASDAQ:LULU), MongoDB (NASDAQ:MDB), Signet (NYSE:SIG) and Chewy (NYSE:CHWY).

Below is a curated list of earnings due in the weeks of December 4 to December 8:

Monday, December 4

GitLab Inc.
</item 19>

<item 20>
6 VCs explain how startups can capture and defend marketshare in the AI era (TechCrunch, 2023-10-13T20:01:40+00:00)
That in and of itself is probably still largely unexplored territory that will see a lot of change and maturing in the next few years. One key aspect there
might be cost monitoring as well since companies like OpenAI charge largely “by the token,” which is a very different metric than most cloud computing.

What can be done to ensure industry-specific startups that tune generative AI models for a specific niche will prove defensible?
</item 20>

<item 21>
If your startup doesn’t seem impossible at first, it’s not hard enough (TechCrunch, 2023-11-03T19:05:30+00:00)
Cowards.

A swan song from the bird sanctuary: X has marked its value at $19 billion, according to internal documents, marking a 56% decrease in value since Elon Musk’s
purchase last year. Bad for shareholders and Musk, but potentially good for executives and employees, Amanda explains.
</item 21>

<item 22>
WTF is going on at OpenAI? We have theories (TechCrunch, 2023-11-18T00:09:53+00:00)
Some CEOs get a pass on things like having kids with direct reports, but not all.

Altman also has three siblings, and his younger sister Annie has publicly accused him of abuse. We have no way of evaluating these allegations, which involve
private matters.

Our read on the board’s language in dismissing Altman, however, is that it was not a legal or personal problem that provoked the action, but a professional or
business one.

We probably won’t know the full truth on this for a long time, as the characters in the drama are likely to be NDA’ed up. Per various whispers and leaks, an
all-hands meeting about the situation this afternoon didn’t produce any revelations beyond banal reassurances that the company is fine and they’ll get a fresh
CEO soon. Expect to hear a lot of rumors before we hear the real thing.
</item 22>

<item 23>
A comprehensive list of 2023 tech layoffs (TechCrunch, 2023-09-28T20:04:20+00:00)
Synapse

Confirmed October 6 that it had laid off 86 people, about 40% of the a16z-backed fintech company. Earlier in the year, Synapse laid off 18% of its employees
after its growth slowed.

Bird

Issued a new round of layoffs after acquiring Spin, according to an email interim CEO Michael Washinushi sent to the company. The layoffs came one week after
the company was delisted from the New York Stock Exchange, and the exact number remains unknown.

Qualtrics

Announced on October 4 that 780 roles are being eliminated and that “several hundred” roles are changing or moving locations in 2024.

Hopper

Is reducing its full-time staff by 30%, amounting to 250 job cuts, the company announced October 3.

Chainalysis

Announced October 2 that it is laying off around 15% of its workforce, affecting approximately 150 employees.
</item 23>

<item 24>
OpenAI hosts a dev day, TechCrunch reviews the M3 iMac and MacBook Pro, and Bumble gets a new CEO (TechCrunch, 2023-11-11T21:15:53+00:00)
as an important early step in its plans for an initial public offering, a Klarna spokesperson tells TechCrunch+.

The unicorn’s legacy isn’t over: It’s been 10 years since Cowboy Ventures’ founder Aileen Lee coined an incredibly catchy nickname for what were very rare
startups at the time: Unicorns. TechCrunch+ spoke with Lee about how she feels about the term 10 years later, now that her venture firm is also a decade old.
</item 24>

<item 25>
A comprehensive list of 2023 tech layoffs (TechCrunch, 2023-09-28T20:04:20+00:00)
Alphabet

Google’s parent company announced laying off 6% of its global workforce on January 21, equating to 12,000 employees. These cuts impact divisions such as Area
120, the Google in-house incubator and Alphabet’s robotics division, Intrinsic.

Fandom

The entertainment company announced an unspecified number of employees impacted across multiple properties on January 20. According to a report by Variety, the
company employs around 500 people, and the layoffs have affected roughly 10% of its staff across different sites.

Swiggy

Announced plans to lay off 380 jobs on January 20 and shut down its meat marketplace.

Sophos

Announced on January 18, 10% of its global workforce, about 450 people were let go.

Microsoft

As announced on January 18, 10,000 employees will be impacted.

GoMechanic

Laid off 70% of its workforce on January 18.

Clearco

Announced on January 17, impacting 30% of staff across all teams.
</item 25>

<item 26>
A comprehensive list of 2023 tech layoffs (TechCrunch, 2023-09-28T20:04:20+00:00)
No other teams are impacted.

Divvy Homes

Announced on September 11 that it is laying off 94 employees.

Sensor Tower

Announced on September 8 that it laid off approximately 40 people out of the 270+ at the company, including C-suite executives.

Roku

Announced September 6 that it plans to lay off approximately 10% of its workforce, representing over 300 employees.

August 2023

Malwarebytes

Announced on August 31 that it laid off 100 employees, mainly impacting corporate employees.

Twiga

Announced on August 21 that it is laying off 283 employees, or 33% of its workforce.

SecureWorks

Announced on August 14 that it will let go of 15% of its workforce, impacting about 300 employees. This is the company’s second round of layoffs this year.

CoinDesk

Announced on August 14 that it is cutting 16% of staff.
</item 26>

<item 27>
WTF is going on at OpenAI? We have theories (TechCrunch, 2023-11-18T00:09:53+00:00)
That doesn’t sound a lot like the “mission” the board wants to advance. Then again, this shift didn’t happen today, and it certainly doesn’t seem like a
plausible reason for abruptly firing the CEO and a few others on a beautiful fall Friday afternoon.

6. What about IP and legal liability?

Altman told me at OpenAI’s Dev Day earlier this month that the company doesn’t want to incur any copyright problems by using (as I had asked about) datasets of
pirated books. But a lot of research I’ve been reading contradicts that, as does pretty much every AI data scientist I talk to. It’s exceedingly hard to imagine
that OpenAI built GPT-3 with the copyrighted books database (as seems to be the case) but not GPT-4 or succeeding models. (I was going to write this up next
week, so thanks OpenAI board for eating my lunch.)
</item 27>

<item 28>
6 VCs explain how startups can capture and defend marketshare in the AI era (TechCrunch, 2023-10-13T20:01:40+00:00)
You cannot escape conversations about AI no matter how far or fast you run. Hyperbole abounds around what current AI tech will be able to do (revolutionize
every industry!) and what current AI tech will be able to do (take over the world!). Closer to the ground, TechCrunch+ is working to understand where startups
might find footholds in the market by levering large language models (LLMs), a recent and impactful new method of creating artificially intelligent software.

How AI will play in startup land is not a new topic of conversation. A few years back, one venture firm asked how AI-focused startups would monetize and whether
they would suffer from impaired margins due to costs relating to running models on behalf of customers. That conversation died down, only to come roaring back
in recent quarters as it became clear that while LLM technology is quickly advancing, it’s hardly cheap to run in its present form.

But costs are only one area where we have unanswered questions.
</item 28>

<item 29>
Check Point sees 'fantastic' year ahead even as rival report disappointing billings (Seeking Alpha, 2023-11-18T13:56:00+00:00)
Unfortunately, it’s because there are more hacks out there than ever before.”

Financials

Sales at Check Point (CHKP) have grown on a yearly basis amid the increase in demand. For the most recently reported quarter, non-GAAP earnings of $2.07 per
share beat the average analyst estimate by 5 cents, while revenue of $596.3M beat by $3.78M.

That resulted in an upgrade late last month at Piper Sandler with signs of recovery on the horizon.

The demand for cybersecurity is strong, but companies are not immune to the realities of the global economy.

This week, rival Palo Alto Networks (NASDAQ:PANW) lowered its billings forecast for the fiscal year, citing the cost of money which has remained a constant
discussion with and focus for customers.

It said it is constantly discussing payments in its annual billings plans, which can create variability on total billings more than before depending on
financing used or the duration of contracts.
</item 29>

<item 30>
A comprehensive list of 2023 tech layoffs (TechCrunch, 2023-09-28T20:04:20+00:00)
Clearco

Announced on January 17, impacting 30% of staff across all teams.

ShareChat

Announced on January 15, ShareChat laid off 20% of its workforce — or over 400 employees — just a month after eliminating more than 100 roles.

SmartNews

Announced on January 12 a 40% reduction of its U.S. and China workforce, or around 120 people.

Intrinsic

Alphabet’s robot software firm, Intrinsic, is laying off 40 employees TechCrunch confirmed on January 12. Amounting to around 20% of the headcount.

Greenlight

The fintech startup offering debit cards to kids laid off 104 employees on January 12, or over 21% of its total headcount of 485 employees.

Career Karma

Learning navigation platform Career Karma laid off another 22 people on January 12 across its global and domestic workforce.

DirectTV

Announced on January 12 plans to lay off about 10% of its management staff on January 20.
</item 30>

<item 31>
Check Point sees 'fantastic' year ahead even as rival report disappointing billings (Seeking Alpha, 2023-11-18T13:56:00+00:00)
“We’ve got to work at it on the go-to-market side, but I feel that all the ingredients are in place and, dare I say it, I think 2024 can be a really fantastic
year for us,” Hollenbeck said, adding the company has increased headcount over the past year including a 25% rise in its salesforce.

Hollenbeck said the company welcomes competition, as they "bring out the best of innovation for customers." The executive spent more than two decades at Intel
(INTC) working in Singapore and Beijing, among other places.

Check Point (CHKP) ended the most recent quarter with $3B in cash, which could open the door to additional acquisitions, though Hollenbeck declined to cite
specifics.

Women

Hollenbeck has led the company for almost two years, one of the few C-suite level women in tech. Nearly 17% of tech companies currently have a woman serving as
CEO, according to AnitaB.org, which supports women in computing and tech.
</item 31>

<item 32>
Hipgnosis Songs Fund’s board is concerned over Citrin Cooperman’s latest valuation of the company. What’s it going to do about it? (Music Business Worldwide,
2023-12-19T23:02:01+00:00)
The reason for the delay? According to UK-listed HSF, it’s all to do with a recently-updated company valuation from HSF’s official independent valuer,
US-headquartered Citrin Cooperman (CC).

The HSF board says this valuation from CC was “materially higher than the valuation implied by proposed and recent transactions in the sector”.

HSF’s board particularly points to two transactions/would-be transactions as evidence:

The proposed sale earlier this year of assets to Blackstone-backed Hipgnosis Songs Capital for USD $440 million (pre-costs) – or $417.5 million (post-costs) – a
proposal that was ultimately rejected by HSF shareholders; The sale of 20,000 “non-core” songs from HSF’s portfolio to an unnamed buyer (possibly Kobalt Music
Group) for $23.1 million, as announced on December 11.
</item 32>

<item 33>
Sam Altman backs teens’ startup, Google unveils the Pixel 8 and TikTok tests an ad-free tier (TechCrunch, 2023-10-07T20:15:26+00:00)
For $4.99, subscribers get an ad-free experience on TikTok — no other strings attached. But don’t look for the option to arrive anytime soon. TikTok says that
it’s piloting the plan in a single, English-speaking market outside the U.S. for now.

LinkedIn leans into AI tools: LinkedIn this week unveiled a string of new AI features spanning its job hunting, marketing and sales products, Ingrid writes.
They include a big update to its Recruiter talent sourcing platform, with AI assistance built into it throughout; an AI-powered LinkedIn Learning coach; and a
new AI-powered tool for marketing campaigns.

Musk comes clean about X’s metrics — maybe: In September, Elon Musk said that X users were generating a lot of content — creating 100 million to 200 million
posts every day, excluding retweets. But speaking at an event this week, X CEO Linda Yaccarino offered a contradictory figure.
</item 33>

<item 34>
If your startup doesn’t seem impossible at first, it’s not hard enough (TechCrunch, 2023-11-03T19:05:30+00:00)
The trend points toward a favorable environment for startups prioritizing growth over cash preservation.

At the same time, we wouldn’t be surprised if WeWork ends up filing for bankruptcy (TC+). The company has been grappling with significant challenges. WeWork’s
stock hit a new low, dropping over 47% after hours to just $1.21, which brings the company’s market cap down to a mere $121 million. This is a drastic fall from
the $47 billion valuation it achieved after its SoftBank-led Series H round in early 2019.

Hope for the best, prepare for the worst: Startup co-founders don’t always make it, and in her fantastic TC+ piece, Rebecca explores what you can do to prepare
today, should a co-founder breakup be on your startup journey’s path.

Bumper banking backing in Brazil: Brazilian startup QI Tech, a banking-as-a-service platform, has raised $200 million in a Series B funding round, marking the
largest raise in Brazil this year across all sectors.
</item 34>

<item 35>
Fundamental Investing: The Art of Relative Valuation (Business Line, 2023-10-14T15:38:28+00:00)
When other assets get more attractive one must assess scope for valuation change in equities as well. For example, interest rates and bond yields in many
developed markets, including the US, are at their highest levels since 2007. This means investors may have to assess current valuations not just against last 10
years’ valuations, but of prior periods as well.

Why? Near zero interest rates and low bond yields made stocks with earnings yield of mere 1-2 per cent attractive for an FPI — for example stocks such as Asian
Paints, Nestle India and ABB India which today trade at PE of 65, 83 and 85 times respectively. This implies earnings yield of 1.5 per cent for Asian Paints and
1.2 per cent for Nestle and ABB. Such earnings yields of these stocks were fine for FPIs to hold when compared to zero interest income in their domestic market.
But now, with US bond yields offering 5 per cent, these may not be so attractive even when considering the growth prospects.
</item 35>

<item 36>
Founders, are events useful? (TechCrunch, 2023-10-13T19:05:37+00:00)
It is aiming to help any software play nice with ChatGPT and other LLM ecosystems.

What big eyes you have: Adobe continues to push for easier image editing, showing off its Project Stardust as a sneak preview of its next-gen AI photo editing
engine.

Holding back the tide: Creatives across industries are strategizing in a campaign that targets potential corporate abuse of AI technology. The conglomeration is
realistic about the ways that musicians and some other creatives could benefit on an individual level from automating parts of their work. The goal is that AI
tools “become ways for individual humans to make more money, work less, and compete with the corporations that exploit them.”

Tech you can touch

A while back I argued that Apple’s new AR headset is a game-changer for startups. It seems that’s likely the case at the high end. But more in the realm of
affordability, Meta Quest 3 takes a step closer to mainstream AR/VR, Brian reports.
</item 36>

<item 37>
Check Point sees 'fantastic' year ahead even as rival report disappointing billings (Seeking Alpha, 2023-11-18T13:56:00+00:00)
According to McKinsey, companies that improved women’s representation outperformed their industry from 2019 to 2022 when compared to other companies with data
from the same time period.

“When there is gender parity of the board, that company tends to perform higher than the average,” she said. The question is “why is it the best kept secret and
why aren’t organizations responding to this faster and better?”

War

Like many companies in Israel, Tel Aviv-based Check Point (CHKP) has been impacted by the war, as Hollenbeck said some of its employees have been killed and
others have been called to serve.

However, she stressed operations are normal and business is continuing as usual. Check Point (CHK) has levels of redundancy around the world and research and
development happening in several locations, she said.

In addition, the company’s supply chain is 100% outside of Israel and tech support centers and data centers are found around the world.
</item 37>

<item 38>
Earnings week ahead: DocuSign, NIO, Broadcom, GameStop and more (Seeking Alpha, 2023-12-03T13:02:00+00:00)
(GTLB) is due to post its Q3 results after the market closes on Monday. With a 10% surge in the stock this year, Seeking Alpha's Quant Rating system suggests a
Hold, expressing concerns about valuation. This contrasts with Wall Street analysts, who advocate a Buy rating.

Recently, Raymond James has rated the stock with a Market Perform rating, citing potential market exuberance and growth potential in fiscal 2025 due to a large
total addressable market, healthy customers, and a value proposition favoring consolidation over fragmentation.

On the other hand, Cantor Fitzgerald started coverage with an Overweight rating and a $55 price target, as the company anticipates benefits from product
consolidation and the generative artificial intelligence wave.

Consensus EPS Estimates: -$0.01

Consensus Revenue Estimates: $141.04M

Earnings Insight: GitLab has beaten EPS and revenue expectations in 8 straight quarters.
</item 38>

<item 39>
Hipgnosis Songs Fund’s board is concerned over Citrin Cooperman’s latest valuation of the company. What’s it going to do about it? (Music Business Worldwide,
2023-12-19T23:02:01+00:00)
In a rather damning note issued today covering the delay in the publication of HSF’s results, Brown wrote: “This is an early blow to the credibility of the new
[Hipgnosis Songs Fund] Board, and casts further doubt over the credibility of the independent valuer, Citrin Cooperman.”

Brown added that he was “surprised” to see that a second valuation expert had not been drafted in to value the HSF portfolio for the latest interim results,
especially as Kroll was appointed to consider the “reasonableness” of Citrin Cooperman’s assumptions for HSF’s previous set of results (to end of March 2023).

Added Brown: “We think the easiest solution now for the [HSF] Board is to apply an additional discount rate to the [Citrin Cooperman] valuation to reflect
current market uncertainty, on the basis that it is probably too late to employ another valuer.”
</item 39>

<item 40>
ASX set to rise ahead of inflation report after big tech lifts Wall Street (The Sydney Morning Herald, 2023-10-24T20:05:06+00:00)
Google, which is also seeking to capitalise on surging demand for AI technology, saw revenue climb 11 per cent to $US76.7 billion, with net income of $US19.7
billion thanks to a recovery in digital advertising revenue. However, its closely watched cloud division, which houses its data storage business and many of its
AI efforts, missed Wall Street expectations. “Investors were disappointed by the relatively weak performance at its Google cloud platform, which is at risk of
falling further behind” competitors’ offerings, Jesse Cohen, a senior analyst at Investing.com, wrote in an email. Loading During trading hours, the S&P 500
slipped off session highs, led by losses in energy shares as oil slid below $US84 a barrel, but once again the index found support above the key 4200 mark.
Verizon Communications, 3M and General Electric climbed on bullish forecasts. Facebook and Instagram owner Meta dropped after being sued by California and dozen
of other US states over harmful youth marketing claims.
</item 40>

<item 41>
The fate of US venture capital in China teeters on uncertainty (TechCrunch, 2023-11-25T09:32:47+00:00)
There are no signs that USD funds’ dealmaking in China will bounce back to its heights in the foreseeable future. The PitchBook report predicts that VC activity
with U.S. participation in China during 2023 will hit a nine-year low, followed by a decade low in 2024. Many Chinese general partners have already been
sourcing capital from the Middle East, which might eventually limit the impact of U.S. investors’ pullback. But the question is whether Chinese tech firms, now
under a new stringent regulatory regime, can deliver the same level of strong growth and returns they experienced in the previous laissez-faire era.

If the U.S. market turns out to be too challenging to penetrate due to competitive and geopolitical factors, there are still other markets to explore. Japan,
known for its openness to new technologies, has been particularly popular among China’s SaaS startups venturing abroad.
</item 41>

<item 42>
6 VCs explain how startups can capture and defend marketshare in the AI era (TechCrunch, 2023-10-13T20:01:40+00:00)
This part of the stack includes cutting-edge capabilities, encompassing model fine-tuning, prompt engineering and agile model orchestration. It’s here that we
anticipate the rise of entities akin to Databricks. Yet, the competitive dynamics of this layer present a unique challenge. Primarily, the emergence of
foundation model providers expanding into middle-layer tools heightens commoditization risks. Additionally, established market leaders venturing into this space
further intensify the competition. Consequently, despite a surge in startups within this domain, clear winners still need to be discovered.

Companies like Datadog are building products to support the expanding AI market, including releasing an LLM observability tool. Will efforts like what Datadog
has built (and similar output from large/incumbent tech powers) curtail the market area where startups can build and compete?

LLM observability falls within the “middle layer” category, acting as a catalyst for specialized business applications to use foundational models.
</item 42>

<item 43>
Sam Altman ousted as OpenAI’s CEO (TechCrunch, 2023-11-17T20:41:13+00:00)
Microsoft has poured $13 billion into OpenAI so far and has what amounts to a 49% stake in the company. The former’s stock price fell more than 1% in the last
30 minutes of trading, after Altman’s departure was announced; Axios reports Microsoft was notified Altman would be leaving “minutes” before the public
announcement.

OpenAI was founded as a nonprofit in 2015, but restructured in 2019 as a “capped-profit” company to raise capital — an acknowledgement of the huge costs
associated with training cutting-edge AI systems. Underlining the point, Altman said in an interview this week with the Financial Times that he “hoped”
Microsoft would increase its investment to help pay for “huge” imminent model training costs.

When contacted via email, OpenAI’s press relations declined to comment beyond the official blog post.
</item 43>

<item 44>
Sam Altman backs teens’ startup, Google unveils the Pixel 8 and TikTok tests an ad-free tier (TechCrunch, 2023-10-07T20:15:26+00:00)
On Equity this week, the crew talked about the SBF trial; deals from VR firms Rainforest, At One Ventures, Section 32 and Greylock, where venture funding has
declined; and how Fearless Fund, a firm founded to invest in women of color, is being barred from awarding grants to Black women founders.

Meanwhile, Found featured Esther Rodriguez-Villegas from Acurable, a medical device company that makes patient-friendly wearable devices to diagnose and manage
respiratory conditions at home. As a career-long academic, Rodriguez-Villegas talks about how she never intended to be a founder until she learned about how the
currently available medical devices make it extremely difficult to detect and treat diseases like sleep apnea and epilepsy.

And over on Chain Reaction, Jacquelyn did a crossover episode with Alex about the SBF trial.
</item 44>

<item 45>
A comprehensive list of 2023 tech layoffs (TechCrunch, 2023-09-28T20:04:20+00:00)
A GitHub spokesperson told TechCrunch that the layoff is part of the streamlining effort the firm had disclosed in February to cut roughly 10% of its workforce
by end of Q1 2023.

Disney

On March 27 in an internal memo to employees, Disney revealed there will be three rounds of layoffs, the first beginning this week. The job cuts will impact
approximately 7,000 employees, which was announced in February.

Salesforce

On March 24, Bloomberg reported that more layoffs could be on the way at Salesforce, quoting chief operating officer Brian Millham, who indicated that the
company could be adding to the ongoing job cutting at the CRM leader and in tech in general. If the layoffs happen, it would come on top of the 10% cut in
January.

Accenture

Announced March 23 that it plans to cut 19,000 jobs, or 2.5% of its workforce.

Indeed

Announced March 22 that it will lay off 2,200 employees, or 15% of its staff.
</item 45>

<item 46>
Hipgnosis Songs Fund’s board is concerned over Citrin Cooperman’s latest valuation of the company. What’s it going to do about it? (Music Business Worldwide,
2023-12-19T23:02:01+00:00)
There are a few clues in HSF’s latest public statement to this end, perhaps including its rare – not to mention conspicuous – mention of HSM being
“majority-owned by funds managed and/or advised by Blackstone“.

In addition, today’s statement is also a clear signal from the HSF board that it has understandable concerns over the now-infamous ‘discount rate’ used by
Citrin Cooperman to value HSF’s portfolio (as MBW covered in depth through here).

Citrin Cooperman continues to stick to an 8.5% discount rate in its valuations, despite interest rate rises over the past 12-18 months – and has long refused to
move this 8.5% figure upwards to, say, 9% or 9.5%.

If Citrin Cooperman ever did raise this discount rate, the independent valuation of Hipgnosis Songs Fund’s portfolio would, in turn, instantly fall.
</item 46>

<item 47>
A comprehensive list of 2023 tech layoffs (TechCrunch, 2023-09-28T20:04:20+00:00)
February 2023

Twitter

Announced on February 26, the company laid off more than 200 employees, including Esther Crawford, Haraldur Thorleifsson and Leah Culver. Since Musk took over
Twitter in October last year, the company’s headcount has fallen by more than 70%.

Poshmark

Announced February 24, Poshmark confirmed with TechCrunch that less than 2% of its workforce was affected, primarily in the U.S. The company employs roughly
800+ employees.

Green Labs

We do not have an exact figure of how many Green Labs plans to lay off its staff. Green Labs confirmed to TechCrunch that it is conducting a round of layoffs
that could impact at least 50% of its workforce.

Chipper Cash

Announced on February 20, the African cross-border payments platform conducted a second round of layoffs just 10 weeks after it cut approximately 12.5% of its
workforce. Chipper Cash relieved almost one-third of its workforce, about 100 employees.
</item 47>

<item 48>
OpenAI hosts a dev day, TechCrunch reviews the M3 iMac and MacBook Pro, and Bumble gets a new CEO (TechCrunch, 2023-11-11T21:15:53+00:00)
WeWork goes bust: As predicted, flexible office-space firm WeWork has filed for Chapter 11 bankruptcy protection, listing over $18.6 billion of debt in a
remarkable collapse for the once high-flying startup co-founded by Adam Neumann and bankrolled by SoftBank, BlackRock and Goldman Sachs.

Slack’s loss, Bumble’s gain: Dating app Bumble announced a doozy this week: It’s replacing founder CEO Whitney Wolfe Herd with Slack CEO Lidiane Jones. Jones
only started as CEO at Slack last year, stepping in for another founder CEO, Stewart Butterfield. Ron and Sarah write that — while Bumble now has a clear line
of succession — the move leaves Slack in a bit of a pickle.

Arrival fails to deliver: Arrival set out eight years ago to make electric vehicle production “radically more efficient.”
</item 48>

<item 49>
If your startup doesn’t seem impossible at first, it’s not hard enough (TechCrunch, 2023-11-03T19:05:30+00:00)
: Shield AI, a startup specializing in autonomous flying systems for the U.S. military, has raised $200 million in a new funding round, reaching a valuation of
$2.7 billion.

Eero grows up, with best-in-class Wi-Fi: Amazon’s Eero has launched its most advanced mesh Wi-Fi router yet, the Eero Max 7.

Boksing claver: Danish startup Soundboks released its fourth-generation Bluetooth speaker that has been a hit in Europe’s festival scene but has yet to gain a
strong foothold in the U.S. market.

Highs and lows in the world of startups

The current bear market in cloud stocks has led to a reversal of revenue multiples from earlier this year, making it difficult for late-stage startups to accrue
value, Alex writes on TC+. However, early-stage startups that are growing rapidly may still be able to raise funds and demand for a better valuation. The trend
points toward a favorable environment for startups prioritizing growth over cash preservation.
</item 49>

<item 50>
Check Point sees 'fantastic' year ahead even as rival report disappointing billings (Seeking Alpha, 2023-11-18T13:56:00+00:00)
"We have the right level of redundancies and business continuity plans in place around the world," Hollenbeck said. "So we're okay."
</item 50>

<item 51>
Hipgnosis Songs Fund’s board is concerned over Citrin Cooperman’s latest valuation of the company. What’s it going to do about it? (Music Business Worldwide,
2023-12-19T23:02:01+00:00)
“[It] is up to the [Hipgnosis Songs Fund] Board to satisfy itself that the independent valuer, whom it hires, is up to the job, rather than try and pin the
blame on [Hipgnosis Songs Management].” Christopher Brown, JP Morgan

Concluded Brown: “[It] is up to the [Hipgnosis Songs Fund] Board to satisfy itself that the independent valuer, whom it hires, is up to the job, rather than try
and pin the blame on [Hipgnosis Songs Management].

“Our view is that a breakdown in the relationship between Board and manager would not be optimal for maximising shareholder value.”

Brown further noted that JP Morgan would “not be surprised to see Citrin Cooperman resign, having been very publicly undermined by the [HSF] Board today”.

JP Morgan retained its ‘overweight’ rating against HSF’s share price “in spite of yet another [HSF] own goal”.
</item 51>

<item 52>
Israel’s startup ecosystem: down but not out (TechCrunch, 2023-10-13T19:10:07+00:00)
Mike Butcher, Frederic Lardinois, Anna Heim and Kyle Wiggers also contributed to this report.
</item 52>

<item 53>
6 VCs explain how startups can capture and defend marketshare in the AI era (TechCrunch, 2023-10-13T20:01:40+00:00)
Startups, he added, can use speed to their advantage as they work to “innovate, iterate and deploy solutions” to customers.

Will that work prove defensible in the long run? Edward Tsai, a managing partner at Alumni Ventures, told us that he had a potentially “controversial opinion
that VCs and startups may want to temporarily reduce their focus on defensibility and increase their focus on products that deliver compelling value and
focusing on speed to market.” Presuming massive TAM, that could work!

Read on for answers to all our questions from:

Rick Grinnell, founder and managing partner, Glasswing Ventures

There are several layers to the emerging LLM stack, including models, pre-training solutions and fine-tuning tools. Do you expect startups to build striated
solutions for individual layers of the LLM stack, or pursue a more vertical approach?
</item 53>

<item 54>
A comprehensive list of 2023 tech layoffs (TechCrunch, 2023-09-28T20:04:20+00:00)
Groupon

Impacting another 500 employees announced on January 31. The company said this new set of layoffs will be spread across the first two quarters of 2023.

Impossible Foods

Reportedly affecting 20% of its staff, over 100 employees, Bloomberg reported first.

PayPal

Announced on January 30, about 2,000 full-time employees, or 7% of its workforce, were affected.

Arrival

Announced on January 30, with a newly appointed CEO, slashing 50% of its workforce — 800 employees globally.

Waymo

The self-driving technology unit under Alphabet quietly laid off workers on January 24, according to The Information and several posts on LinkedIn and Blind.
It’s not yet clear how many of Waymo’s staff will be affected.

Spotify

Announced on January 23, impacting around 6% of its global workforce — around 600 employees.
</item 54>

<item 55>
Earnings week ahead: Amazon, Microsoft, Alphabet, Meta, Ford, GM and more (Seeking Alpha, 2023-10-22T12:13:00+00:00)
Consensus EPS Estimates: $-2.53

Consensus Revenue Estimates: $18.37B

Earnings Insight: Boeing has beaten EPS expectations just once in the past 8 quarters and revenue just twice in that span.

Also reporting: International Business Machines (IBM), QuantumScape Corporation (QS), ServiceNow (NOW), General Dynamics (GD), Automatic Data Processing (ADP),
Baker Hughes Company (BKR), KLA Corporation (KLAC) and more.

Thursday, October 26

Amazon (AMZN)

Ecommerce and cloud giant Amazon (AMZN) is due to release its Q3 earnings on Thursday after the market closes. The quarter will focus on Amazon's cost-cutting
initiatives, supply chain management, and AI product deployment as potential positive factors for attracting investment in equities amidst current market
environments.

Wall Street analysts have a consensus Strong Buy rating, but Seeking Alpha Quant Ratings assesses the company as a Hold. On a YTD basis, Amazon shares have
expanded by almost 50% in value.
</item 55>

<item 56>
Earnings week ahead: Amazon, Microsoft, Alphabet, Meta, Ford, GM and more (Seeking Alpha, 2023-10-22T12:13:00+00:00)
The search-engine giant has outperformed the broader market index with a 55% growth rate this year, surpassing the S&P 500's 12%.

Following the tech behemoth's upbeat Q2 results, investment firm Bank of America said it is expecting "favorable" headlines, due in part to strength in YouTube,
search, and artificial intelligence.

Just a few days before the earnings report, Seeking Alpha's Quant Rating system changed its view from Strong Buy to Hold, while Wall Street analysts maintained
a bullish view on the stock, giving it a Strong Buy rating.

Consensus EPS Estimates: $1.45

Consensus Revenue Estimates: $75.73B

Earnings Insight: Google has topped EPS expectations in 4 of the past 8 quarters, beating revenue estimates in 5 of those reports.

Microsoft (MSFT)

Joining Alphabet on Tuesday, Microsoft (MSFT) is also set to report its Q4 performance after the closing bell.
</item 56>

<item 57>
Coinbase rallies more than 60% in same month that FTX and Binance founders brace for prison (Cnbc | World Business News Leader, 2023-11-30T21:08:00+00:00)
For early holders of the stock, the rebound helps ease the pain of 2022, when Coinbase lost 86% of its value as soaring inflation and rising interest rates
pushed investors out of crypto and high-growth tech companies, and into assets deemed safer in a recession. Tech stocks have roared back this year, particularly
those tied to the artificial intelligence boom and crypto. Coinbase has the added benefit of having survived the so-called crypto winter, while so many of its
rivals disappeared or downsized. The industry fallout came to a head this month, when Sam Bankman-Fried, founder of former Coinbase rival FTX, was found guilty
of seven criminal fraud counts tied to the collapse of his exchange and the theft of customer funds. His conviction landed on Nov. 2 after a monthlong trial.
Less than three weeks later, on Nov.
</item 57>

<item 58>
A comprehensive list of 2023 tech layoffs (TechCrunch, 2023-09-28T20:04:20+00:00)
SiriusXM

Announced on March 6, the company laid off 475 employees, or 8% of its total workforce.

Alerzo

The Nigerian B2B e-commerce platform had a headcount of more than 2,000 before a first round of layoffs in September 2022. Alerzo has laid off 15% of its
full-time workforce, the company confirmed on March 6, leaving about 800 employees at the startup.

Cerebral

Announced March 1, the company is letting go 15% of it’s workforce — roughly 285 employees.

Waymo

Announced March 1, Alphabet’s Waymo issued a second round of layoffs this year. Combined with the initial cuts in January, the self-driving technology company
has let go of 8%, or 209 employees, of its workforce.

Thoughtworks

Announced on March 1, the company laid off about 4% of its global workforce — approximately 500 employees.
</item 58>

<item 59>
Earnings week ahead: Amazon, Microsoft, Alphabet, Meta, Ford, GM and more (Seeking Alpha, 2023-10-22T12:13:00+00:00)
The stock gets a Hold rating from Seeking Alpha's Quant Rating system, a more cautious view than the consensus Buy rating from Wall Street analysts.

"Cleveland-Cliffs is well-positioned to perform well over time, thanks to its strong position when it comes to supplying automobile companies and due to energy
cost advantages compared to competitors from markets such as Europe," says investing group leader Jonathan Weber.

Consensus EPS Estimates: $0.44

Consensus Revenue Estimates: $5.53B

Earnings Insight: The company has beaten EPS expectations in 3 of the past 8 quarters and revenue in 5 of those reports.

Also reporting: Logitech International S.A. (LOGI), Koninklijke Philips N.V. (PHG), Packaging Corporation of America (PKG) and more.

Tuesday, October 24

Alphabet (GOOG) (GOOGL)

Alphabet (GOOG) (GOOGL) will be announcing its Q3 results during Tuesday’s post-market session.
</item 59>

<item 60>
A comprehensive list of 2023 tech layoffs (TechCrunch, 2023-09-28T20:04:20+00:00)
A comprehensive list of 2023 tech layoffs From major layoffs at Google, Amazon and Microsoft to small fintech startups and apps

Last year’s techwide reckoning continues. The tech industry has seen more than 240,000 jobs lost in 2023, a total that’s already 50% higher than last year and
growing. Earlier this year, mass workforce reductions were driven by the biggest names in tech like Google, Amazon, Microsoft, Yahoo, Meta and Zoom. Startups
across many sectors also announced cutbacks through the first half of the year. And while tech layoffs slowed down in the summer and fall, it appears that cuts
are ramping up yet again.

Many economists have cautioned against fears of a recession, which would seem like a reason for optimism. But the momentum for a tech sector rebound has been
slow to build, resulting in tech companies continuing to cut back on their workforces and pivot from a growth mindset to one based on efficiency in the face of
stubborn market conditions.
</item 60>

<item 61>
6 VCs explain how startups can capture and defend marketshare in the AI era (TechCrunch, 2023-10-13T20:01:40+00:00)
But costs are only one area where we have unanswered questions. We are also incredibly curious about how startups should approach building tools for AI
technologies, how defensible startup-focused AI work will prove, and how upstart tech companies should charge for AI-powered tooling.

With the amount of capital flowing to startups working with and building AI today, it’s critical that we understand the market as best we can. So we asked a
number of venture capitalists who are active in the AI investing space to walk us through what they are seeing in the market today.

What we learned from the investing side of the house was useful. Rick Grinnell, founder and managing partner at Glasswing Ventures, said that within the new AI
tech stack, “most of the opportunity lies in the application layer,” where “the best applications will harness their in-house expertise to build specialized
middle-layer tooling and blend them with the appropriate foundational models.”
</item 61>

<item 62>
A comprehensive list of 2023 tech layoffs (TechCrunch, 2023-09-28T20:04:20+00:00)
CoinDesk

Announced on August 14 that it is cutting 16% of staff.

NCC Group

Confirmed on August 10 that it is making more layoffs, its second round of layoffs in just six months. The company declined to say how many employees are
affected by this round of cuts. NCC Group cut 125 employees in February.

Rapid7

Announced on August 9 that it plans to lay off 18% of its workforce, affecting more than 400 employees globally.

Mobile Premier League

Announced on August 8 that it is laying off approximately 50% of its employees or 350 people.

Astra

Announced on August 4 that it has laid off 25% of its workforce since the beginning of the quarter.

Discord

Announced on August 3 that it has let go of nearly 40 employees or 4% of its workforce.

HackerOne

Announced on August 2 that it is laying off 12% of its workforce, approximately 50 employees.
</item 62>

<item 63>
6 VCs explain how startups can capture and defend marketshare in the AI era (TechCrunch, 2023-10-13T20:01:40+00:00)
Incumbents like Datadog, New Relic and Splunk have all produced LLM observability tools and do appear to be putting a lot of R&D dollars behind this, which may
curtail the market area in the short-term.

However, as we have seen before with the inceptions of the internet and cloud computing, incumbents tend to innovate until innovation becomes stagnant. With AI
becoming a household name that finds use cases in every vertical, startups have the chance to come in with innovative solutions that disrupt and reimagine the
work of incumbents. It’s still too early to say with certainty who the winners will be, as every day reveals new gaps in existing AI frameworks. Therein lie
major opportunities for startups.

How much room in the market do the largest tech companies’ services leave for smaller companies and startups tooling for LLM deployment?
</item 63>

<item 64>
8 predictions for AI in 2024 (TechCrunch, 2023-12-19T16:19:34+00:00)
Marketing claims made for machine learning systems that companies adopted in order to not fall behind will receive their quarterly and yearly reviews… and it’s
very likely they will be found wanting.

Expect a considerable customer withdrawal from AI tools as the benefits fail to justify the costs and risks. On the far end of this spectrum, we are likely to
see lawsuits and regulatory action with AI service providers that failed to back up their claims.

While capabilities will continue to grow and advance, 2023’s products will not all survive by a long shot, and there will be a round of consolidation as the
wobblier riders of the wave fall and are consumed.

Apple jumps in

Apple has an established pattern of waiting, watching and learning from other companies’ failures, then blowing in with a refined and polished take that puts
others to shame. The timing is right for Apple to do this in AI, not just because if it waits too long its competition may eat up the market, but because the
tech is ripe for their kind of improvement.
</item 64>

<item 65>
Sam Altman backs teens’ startup, Google unveils the Pixel 8 and TikTok tests an ad-free tier (TechCrunch, 2023-10-07T20:15:26+00:00)
Hands on with the Pixel 8 Pro: Darrell took the newly unveiled Pixel 8 Pro for a whirl, and he liked what he saw. While very similar to last year’s model (the
Pixel 7 Pro), Darrell felt that the improved cameras, brighter screen and enhanced AI-powered features made it enough of an upgrade to (potentially) warrant a
purchase — minus the underutilized temperature sensor. Stay tuned for his full review.

Turmoil at Flexport: Dave Clark, the former Amazon executive who was ousted as CEO of Flexport just a year into the job, fired back at its founder and board,
calling recent reporting on the logistics company “deeply concerning.” Clark made the comments Monday in a lengthy post on social media site X following a
report from CNBC that provided new information about his last days at Flexport, a freight forwarding and customs brokerage startup valued at $8 billion.
</item 65>

<item 66>
OpenAI hosts a dev day, TechCrunch reviews the M3 iMac and MacBook Pro, and Bumble gets a new CEO (TechCrunch, 2023-11-11T21:15:53+00:00)
The idea, O’Hara said, is to rethink social networking from the ground up.

Ford shutters SaaS app for field work: Ford has shut down VIIZR, a software-as-a-service company that, along with Salesforce, built an app to help tradespeople
like plumbers, locksmiths and electricians to schedule field appointments, send invoices and manage customers, Kirsten reports. VIIZR, which was announced in
December 2021, was a separate company majority owned by Ford, with Salesforce as a minority investor.

Apple bets on generative AI: Apple CEO Tim Cook pushed back against the notion that the company was behind in AI on Apple’s Q4 earnings call with investors, as
he highlighted technology developments that Apple had made recently that “would not be possible without AI.” Cook also said that Apple was working on generative
AI technologies, lending credence to reports suggesting the company is on track to spend $1 billion per year on developing generative AI products.
</item 66>

<item 67>
6 VCs explain how startups can capture and defend marketshare in the AI era (TechCrunch, 2023-10-13T20:01:40+00:00)
Do you expect startups to build striated solutions for individual layers of the LLM stack, or pursue a more vertical approach?

While there are startups specializing in parts of the stack (like Pinecone), Valor’s focus is on applied AI, which we define as AI that is solving a customer
problem. Saile.ai is a good example — it uses AI to generate closeable leads for the Fortune 500. Or Funding U using its own trained dataset to create a more
useful credit risk score. Or Allelica, using AI on treatment solutions applied to individual DNA to find the best medical treatment for you personally in a
given situation.

Companies like Datadog are building products to support the expanding AI market, including releasing an LLM observability tool. Will efforts like what Datadog
has built (and similar output from large/incumbent tech powers) curtail the market area where startups can build and compete?

Tools like Datadog can only help the acceptance of AI tools if they succeed in monitoring AI performance bottlenecks.
</item 67>

<item 68>
A comprehensive list of 2023 tech layoffs (TechCrunch, 2023-09-28T20:04:20+00:00)
This comes just a few months after the company announced a significant wave of layoffs.

May 2023

Taxfix

Announced on May 30 that it has laid off 20% of staff—120 employees.

Meta

Announced May 24 that it is laying off about 6,000 people. In total, about 21,000 people have lost their jobs at Meta since November.

JioMart

Reliance Retail’s online shopping platform laid off over 1,000 employees on May 22, and plans to cut as many as 9,900 more roles over the coming weeks.

Krebs Stamos Group

Announced May 18 that the consulting firm laid off six people. In April, the firm had 18 employees including the founders. On May 18, the site shows only 14
team members.

TuSimple

Announced May 18 that it is laying off about 30% of employees.

Prior to the layoff, TuSimple had about 550 employees in the U.S.
</item 68>

<item 69>
WTF is going on at OpenAI? We have theories (TechCrunch, 2023-11-18T00:09:53+00:00)
2. Do they disagree on long-term strategy?

Despite being the hottest tech company in the world right now and everyone talking about ChatGPT, OpenAI isn’t exactly a sound business. It’s shoveling money
into the furnace as fast as it can by serving, by all accounts, a fantastically expensive product at bargain-bin prices.

That’s all well and good for a year or two, but at some point that strategy changes from a growth hack to an existential liability. Could Altman and the board
have had irreconcilable differences on where that point lies?

This doesn’t seem so likely. The company has been very deliberately pursuing this very publicly, confidently and on a long-term basis. Altman and the board seem
to be in sync on this, at least for the present.

3. Do the numbers not add up?

On the other hand, could OpenAI be losing even more money than Altman admitted or projected? It seems impossible, but the costs of running this operation have
no precedent, nor really does the operation itself.
</item 69>

<item 70>
The fate of US venture capital in China teeters on uncertainty (TechCrunch, 2023-11-25T09:32:47+00:00)
At the same time, domestic RMB funds play an increasing role in funding critical tech sectors. Zhipu AI, one of China’s most ambitious challengers to OpenAI,
for instance, raised financing in RMB instead of USD.

Even having the Chinese branches of famed American VCs listed on the cap table might deter U.S. investors from funding Chinese founders in their backyard. Local
investors are now shunning Chinese “links”, of which definition is ever evolving and broadening, at all costs.

The country’s USD investors continue to deploy capital in industries that aren’t in the crosshairs of U.S.-China competition, but overall, their activity has
dwindled significantly due to the changing regulatory and geopolitical currents, coupled with a slowing economy. The year 2022 saw just $14.5 billion invested
in Chinese companies by U.S.-headquartered VCs, compared to $45.4 billion the year before, according to a report from research firm PitchBook.
</item 70>

<item 71>
OpenAI hosts a dev day, TechCrunch reviews the M3 iMac and MacBook Pro, and Bumble gets a new CEO (TechCrunch, 2023-11-11T21:15:53+00:00)
It has evolved into working in humanitarian aid, delivering emergency food supplies to people in need in Afghanistan and Turkey.

TechCrunch+

TC+ subscribers get access to in-depth commentary, analysis and surveys — which you know if you’re already a subscriber. If you’re not, consider signing up.
Here are a few highlights from this week:

Another superconductor disappointment: Tim writes that a new, supposedly room-temperature superconducting material isn’t what the scientific community hoped it
would be. With the Nature-published paper detailing the material facing retraction, the odds of researchers discovering a room-temperature superconductor are
looking even longer.

Klarna inches toward an IPO: Mary Ann and Alex write that Swedish fintech Klarna is taking steps toward an eventual IPO. The company has initiated a process for
a legal entity restructuring to set up a holding company in the U.K.
</item 71>

<item 72>
Google fakes an AI demo, Grand Theft Auto VI goes viral and Spotify cuts jobs (TechCrunch, 2023-12-09T21:16:17+00:00)
If you’re not, consider signing up. Here are a few highlights from this week:

Bitcoin surge: Jacquelyn writes about Bitcoin’s rapid-fire ascent to $44,000, which came on the back of roughly 25% gains in the last week. Her piece for TC+
explores what’s driving Bitcoin’s price ascent and similar value gains among other tokens — and whether the good vibes continue into the new year.

To swap, or not to swap: Tim reports on how consumer EV battery swapping could usher in freedom for a wide range of people, allowing them to participate in the
EV transition in ways that traditional built-in batteries don’t. The challenge is making the unit economics work.

Coinbase and Robin and the future of fintech: Investors are betting that consumer trading of equity and crypto is rebounding and are consequently pushing the
value of some former startups higher, Alex writes. That could spell good news for startups offering consumer trading services directly — or indirectly, for that
matter.
</item 72>

<item 73>
WTF is going on at OpenAI? We have theories (TechCrunch, 2023-11-18T00:09:53+00:00)
But one can never be sure.

4. Could it be a major security or privacy incident?

The idea that the company has experienced a major, perhaps pervasive, security issue is bolstered by the fact that Microsoft reportedly suspended use of ChatGPT
internally a few days ago. OpenAI subsequently stopped allowing new signups. If there was a serious security problem in its biggest product and Altman
downplayed it, that would obviously create distrust with the board.

There is also the potential for misuse at scale with the enormous amount of personal data that travels through OpenAI’s APIs and services.

Working against this theory is the fact that CTO Mira Murati was just elevated to interim CEO in Altman’s place. It seems unlikely that anything
security-related would go through the CEO and not the CTO, or that the two would be at odds to the point where one could be fired like this and the other
swapped in to clean up the mess. As the board’s statement notes, Murati is in charge of product and safety, among other things.
</item 73>

<item 74>
Earnings week ahead: Amazon, Microsoft, Alphabet, Meta, Ford, GM and more (Seeking Alpha, 2023-10-22T12:13:00+00:00)
(TDOC), Dow (DOW), Snap (SNAP), Texas Instruments Incorporated (TXN), Halliburton Company (HAL), Corning Incorporated (GLW), Novartis AG (NVS), Waste Management
(WM), Spotify Technology S.A. (SPOT), Kimberly-Clark (KMB) and more.

Wednesday, October 25

Meta Platforms (META)

Facebook parent Meta Platforms (META) is scheduled to announce its Q3 results after the close on Wednesday. After exceeding expectations in Q2, the social media
giant forecast Q3 revenue in the range of $32B–$34.5B.

SA contributor Dair Sansyzbayev, who is extremely optimistic about the company's near-term prospects, says, "Meta Platforms stock remains undervalued despite a
tripling price since November 2022. Quarterly earnings show revenue growth, profitability, and solid cash flow, while Meta's expansion and large user base
present potential revenue streams."
</item 74>

<item 75>
A comprehensive list of 2023 tech layoffs (TechCrunch, 2023-09-28T20:04:20+00:00)
Informatica

Reported on January 11 that it will lay off 7% of it’s workforce, or 450 staffers globally.

Carta

Announced on January 11, the equity management platform cut 10% of its staff. Judging by LinkedIn data, the layoff could have impacted around 200 employees.

Citizen

Impacting 33 staff members on January 11.

Coinbase

To cut 950 jobs, or about 20% of its workforce, and shut down “several” projects, announced on January 10. This is the second round of major layoffs at the
crypto exchange, which eliminated 18% of its workforce, or nearly 1,100 jobs last June.

SuperRare

The NFT marketplace is cutting 30% of its staff, announced on January 6.

Amazon

Announced on January 5, eliminating more than 18,000 roles. This announcement extends a previously announced round of layoffs in November of 2022. On January
19, the company announced it would end AmazonSmile.
</item 75>

<item 76>
Hipgnosis Songs Fund’s board is concerned over Citrin Cooperman’s latest valuation of the company. What’s it going to do about it? (Music Business Worldwide,
2023-12-19T23:02:01+00:00)
That price, according to HSF’s board, represented a 14.2% discount on Citrin Cooperman’s valuation of this portfolio as of September 30 (something we’ll come
back to).

As part of its announcement this morning, the HSF board took something of a swipe at its investment adviser, Merck Mercuriadis‘ Hipgnosis Song Management.

After seeing Citrin Cooperman’s latest valuation and believing it to be too high, said HSF, it then “sought advice from Hipgnosis Song Management… which is
majority owned by funds managed and/or advised by Blackstone, on their opinion on the independent valuer’s valuation.”

HSF’s board added: “Hipgnosis Song Management… eventually provided an opinion, which was heavily caveated, such that the Board has concerns as to the valuation
of the Company’s assets in its interim results.”
</item 76>

<item 77>
Hipgnosis Songs Fund’s board is concerned over Citrin Cooperman’s latest valuation of the company. What’s it going to do about it? (Music Business Worldwide,
2023-12-19T23:02:01+00:00)
Brown noted that a 0.5% rise in Citrin Cooperman’s discount rate (i.e. raising it from 8.5% to 9.0%) would “reduce the fair value of the [HSF] portfolio by
7.9%, which on a leveraged basis is [approximately a] 10% reduction in NAV [Net Asset Value]”.

Should HSF’s board go down this route, he noted, he expected “the [HSF] shares to weaken… given that this represents another ‘unforced error’”.

“This is an early blow to the credibility of the new [Hipgnosis Songs Fund] Board, and casts further doubt over the credibility of the independent valuer,
Citrin Cooperman.”
</item 77>

<item 78>
Fundamental Investing: The Art of Relative Valuation (Business Line, 2023-10-14T15:38:28+00:00)
Portfolio podcast: The art of valuing stocks Portfolio podcast: The art of valuing stocks

Let there be no doubt, valuation experts like Aswath Damodaran and many successful hedge fund managers believe that absolute valuation (which we discussed in
the Big Story in bl.portfolio edition dated October 8) is the better way to arrive at the value of a stock.

However, due to a combination of complexities involved in building appropriate models, insufficient information, time constraints and also the fact that there
are hundreds of stocks you want to consider, it is just not possible to do a DCF ( discounted cash flow model) for all your stock investments, and hence
relative valuation as an approach to picking stocks too is well accepted and widely prevalent.

When you decide to buy or sell a stock based on its PE or P/B or EV/EBITDA or EV/Revenue or even EV/subscriber ratios, you are engaging in the art of relative
valuation.
</item 78>

<item 79>
6 VCs explain how startups can capture and defend marketshare in the AI era (TechCrunch, 2023-10-13T20:01:40+00:00)
Despite cloud computing remaining critical, AI budgets are now more than double that of cloud computing. Eighty-two percent of business leaders believe the
integration of AI solutions will increase their employee performance and job satisfaction, and startups should expect a high level of desire for and experience
with AI solutions in their future customers.

Finally, we’ve seen consumption, or usage-based priced tech products’ growth slow in recent quarters. Will that fact lead startups building modern AI tools to
pursue more traditional SaaS pricing? (The OpenAI pricing schema based on tokens and usage led us to this question.)

The trajectory of usage-based pricing has organically aligned with the needs of large language models, given that there is significant variation in
prompt/output sizes and resource utilization per user. OpenAI itself racks upward of $700,000 per day on compute, so to achieve profitability, these operation
costs need to be allocated effectively.

Nevertheless, we’ve seen the sentiment that tying all costs to volume is generally unpopular with end users, who prefer predictable systems that allow them to
budget more effectively.
</item 79>

<item 80>
How the OpenAI fiasco could bolster Meta and the ‘open AI’ movement (TechCrunch, 2023-11-21T15:50:50+00:00)
For sure, Meta isn’t the only company flaunting its “open” approach to AI development — notably, the likes of Hugging Face, Mistral AI and 01.AI, which have all
raised sizeable sums at lofty valuations with similar goals in mind. But as a $900 billion juggernaut with a long history of courting developers through open
source endeavors, Meta is perhaps best positioned to capitalize on the mess that OpenAI has created for itself. Its decision to pursue “openness” over
“closedness” seems to be well vindicated right now, and regardless of whether Llama is or isn’t really open source, it’s likely “open enough” for most people.

It’s still too early to make any surefire claims on what impact the OpenAI fallout will have on LLM development and uptake in the future. Altman and Brockman
are undoubtedly steady hands for a commercial AI startup, and they may even return to steward OpenAI.
</item 80>

<item 81>
Earnings week ahead: DocuSign, NIO, Broadcom, GameStop and more (Seeking Alpha, 2023-12-03T13:02:00+00:00)
Also reporting: Lululemon Athletica (LULU), Dollar General Corporation (DG), Ciena Corporation (CIEN), Smartsheet (SMAR), Vail Resorts (MTN), Planet Labs PBC
(PL), Guidewire Software (GWRE), The Cooper Companies (COO), HashiCorp (HCP), Lakeland Industries (LAKE), Greif (GEF), Liquidity Services (LQDT), GMS (GMS),
Methode Electronics (MEI), Hooker Furnishings Corporation (HOFT), DLH Holdings (DLHC), SecureWorks (SCWX), and more.

Friday, December 8

Hello Group (NASDAQ:MOMO) will release its Q3 earnings results before the opening bell on Friday. Following upbeat Q2 results, the company saw a softer Q3 and
anticipated a drop in revenues of 10.3% to 7.2% Y/Y.

The Seeking Alpha's Quant Ratings moved to a Hold recommendation just days before the earnings announcement.
</item 81>

<item 82>
Check Point sees 'fantastic' year ahead even as rival report disappointing billings (Seeking Alpha, 2023-11-18T13:56:00+00:00)
Hollenbeck is a seasoned executive: after Intel (INTC), she spent time at Oracle (ORCL) and is on the board of Neythri Futures Fund, which invests in female
founders tapping into a growing base of South Asian female leaders. Although she’s been in tech for nearly 30 years, it hasn’t gotten easier to be a woman in a
male-dominated industry.

At most tech events, "there's never a line [outside] of the lady's room,” Hollenbeck said. “I think it's gotten materially better, but it's only gotten
materially better because we started from such a low point.”

Hollenbeck is proud that four of the top executives at Check Point (CHKP) are women and three women are on the board.

“That’s pretty amazing,” she said, pointing to data that shows the more women in leadership roles, the better they perform.
</item 82>

<item 83>
WTF is going on at OpenAI? We have theories (TechCrunch, 2023-11-18T00:09:53+00:00)
In perhaps the most unexpected tech news of the year, billionaire and AI evangelist Sam Altman has been ejected from his CEO role at OpenAI by the company’s
board after an apparent vote of no confidence. Its exact wording in a release issued this afternoon: Altman’s “departure follows a deliberative review process
by the board, which concluded that he was not consistently candid in his communications with the board, hindering its ability to exercise its responsibilities.”

What the hell is happening at the most hyped company in the world?! Here are some totally speculative theories that occurred to us and others around the web.

1. Did Altman circumvent the board in a major deal?

Based on the board’s language and the way these giant tech companies work, this is the prevailing theory floating around right now. “Not consistently candid” is
a very diplomatic way of saying Altman lied.
</item 83>

<item 84>
Sam Altman backs teens’ startup, Google unveils the Pixel 8 and TikTok tests an ad-free tier (TechCrunch, 2023-10-07T20:15:26+00:00)
She claimed X was seeing 500 million posts per day on the platform. So who’s right? Beats us.

Former NSA director’s startup shutters: IronNet, a once-promising cybersecurity startup founded by a former NSA director, has shuttered and laid off its
remaining staff following its collapse. The Virginia-based IronNet was founded in 2014 by retired four-star general Keith Alexander and had raised more than
$400 million in funding. But IronNet failed to gain traction after going public in August 2021, and its stock price continued to steeply decline in the wake of
an initial spike.

Audio

On the hunt for a new podcast to listen to while you work out, do the dishes or rake the leaves (now that fall’s arrived)? Look no further than TechCrunch’s
roster, which covers the world of startups, the blockchain and more.
</item 84>

<item 85>
Fundamental Investing: The Art of Relative Valuation (Business Line, 2023-10-14T15:38:28+00:00)
When the risks are higher, you must apply a higher margin of safety before choosing the stock.

Another thing to bear in mind when it comes to EV-based valuation — any change in EV will impact only the equity value. For example, if EV of a company is ₹100
and it has ₹50 in debt, then the value of its equity is ₹50. In this case, if you think EV can increase by 10 per cent, the value of equity will increase by 20
per cent, and the reverse is also true. The higher the leverage, more the impact of change in EV on the value of the stock (both ways).

Keep a weather eye on Ketchup Economics

Given the fact that relative valuation is based on comparison with peer multiples, investors must take measures to check whether the peer multiples are
reasonable as well. An overvalued peer cannot be used to justify expensive valuation in one stock. This is what economist Larry Summers termed as Ketchup
economists/economics.
</item 85>

<item 86>
Israel’s startup ecosystem: down but not out (TechCrunch, 2023-10-13T19:10:07+00:00)
Undoubtedly, those numbers are all going to be significantly lower this year, not least because the disruptions of this war are coming directly on the heels of
a major protest movement.

The tech industry led opposition to the government’s bid for judicial reform. Even before the fighting started, it was voting with its feet, and some companies
and investors are already refusing to do business in the country with the prospect of the reforms getting pushed through. (That effort, in turn, has been
brought to a halt: People on the two sides of that debate are now standing together for what they see as the significantly bigger fight and threat.)

Even before last weekend’s assault, startup investment in Israel was way down from the prior year. As TechCrunch’s Anna Heim reported in July, “With a
provisional tally of $3.2 billion for the first half of the year, funding activity in Israel dropped by 73% compared to the same period in 2022, IVC data
shows.” Other countries and regions have reported similar declines in their private-market capital flows this year.
</item 86>

<item 87>
Earnings week ahead: Amazon, Microsoft, Alphabet, Meta, Ford, GM and more (Seeking Alpha, 2023-10-22T12:13:00+00:00)
Despite both stocks lagging behind the broader market index, Seeking Alpha's Quant Rating system has approached them cautiously, assigning a Hold rating. In
contrast, Wall Street analysts express optimism, giving both stocks a Buy rating.

Given the prevailing market conditions and the upward trend in crude oil prices, analysts anticipate Y/Y growth in earnings per share and revenue for CVX, while
expecting a decline for XOM.

Also reporting: WisdomTree (WT), AbbVie (NYSE:ABBV), Phillips 66 (PSX), Colgate-Palmolive Company (CL), Sanofi (SNY), T. Rowe Price Group (TROW), Stanley Black
& Decker (SWK), Charter Communications (CHTR), Newell Brands (NWL), Xcel Energy (XEL), Aon (AON), AutoNation (AN), CBRE Group (CBRE), and more.

Related stock stories:
</item 87>

<item 88>
Hipgnosis Songs Fund’s board is concerned over Citrin Cooperman’s latest valuation of the company. What’s it going to do about it? (Music Business Worldwide,
2023-12-19T23:02:01+00:00)
WHAT’S HAPPENED?

MBW Explains is a series of analytical features in which we explore the context behind major music industry talking points – and suggest what might happen next.
MBW Explains is supported by JKBX , a technology platform that offers consumers access to music royalties as an asset class.

The drama at Hipgnosis Songs Fund isn’t letting up for the Holidays, it seems.

This morning (December 19), the company’s board – now led by Chairman, Rob Naylor – issued a statement announcing that HSF would no longer, as previously
expected, be issuing its latest financial results today.

Instead, the results (covering the six months to end of September 2023) are now expected to be published sometime before the close of December 31.

The reason for the delay?
</item 88>

<item 89>
Earnings week ahead: Amazon, Microsoft, Alphabet, Meta, Ford, GM and more (Seeking Alpha, 2023-10-22T12:13:00+00:00)
Seeking Alpha's Quant Rating system turned bullish on the stock just a few days before the earnings by upgrading its view to Strong Buy from Hold. Meanwhile,
Wall Street analysts maintain their optimism about the stock with a Strong Buy rating.

After more than a year of rigorous examination by antitrust investigators across the world, Microsoft just completed its $69B acquisition of Activision
Blizzard, the largest deal in video game history. Its benefits, however, will not be reflected in the company's financial report.

Consensus EPS Estimates: $2.65

Consensus Revenue Estimates: $54.54B

Earnings Insight: Microsoft has outperformed EPS forecasts in 7 of the last 8 quarters, and revenue in 6 of those quarters.

Also reporting: General Electric Company (NYSE:GE), Verizon Communications (VZ), Visa Inc. (V), The Coca-Cola Company (KO), 3M Company (MMM), General Motors
Company (GM), Raytheon Technologies Corporation (RTX), Teladoc Health, Inc.
</item 89>

<item 90>
How the OpenAI fiasco could bolster Meta and the ‘open AI’ movement (TechCrunch, 2023-11-21T15:50:50+00:00)
Countless startups and scale-ups have built businesses atop OpenAI’s proprietary GPT-X large language models (LLMs), and over the weekend hundreds of OpenAI
customers reportedly started contacting OpenAI’s rivals, which include Anthropic, Google and Cohere, concerned that their own businesses might be impacted if
OpenAI was to disintegrate overnight.

Over-reliance

The panic has been palpable. But there are precedents from elsewhere in the technology sphere, perhaps most notably that of the cloud computing industry, which
became renowned for the way it locked companies in to centralized, vortex-like silos.

“Part of the frenzy around the future of OpenAI is due to too many startups over-relying on their proprietary models,” Luis Ceze, University of Washington
computer science professor and OctoML CEO, told TechCrunch in an emailed statement. “It’s dangerous to put all your chips in one basket — we saw that in the
early cloud days which led to companies shifting to multi-cloud and hybrid environments.”
</item 90>

<item 91>
The fate of US venture capital in China teeters on uncertainty (TechCrunch, 2023-11-25T09:32:47+00:00)
Driven by a confluence of factors, from China’s crackdown on the tech industry to escalating U.S.-China tensions, some of these investors are now turning their
gaze to opportunities abroad, tracing the footsteps of a new generation of Chinese-founded startups that are expanding overseas.

Between a rock and a hard place

Since their entry into China in the late 1990s, American venture capital firms, led by powerhouses like Sequoia Capital, IDG Capital and GGV, have played a
major role in funding high-risk, high-reward startups in the country’s consumer internet sector. This two-decade-long mutually beneficial relationship, however,
now hangs in the balance as changes at home and abroad diminish the pool of investment opportunities for outside financiers.

In recent years, Beijing’s sweeping tech crackdowns have introduced a new sense of uncertainty to investors.
</item 91>

<item 92>
Fundamental Investing: The Art of Relative Valuation (Business Line, 2023-10-14T15:38:28+00:00)
Any other metric could be wrong and misleading.

For example, price by revenue or price by EBITDA is incorrect. Why? How do you differentiate between two companies A and B — both having same market cap of ₹100
and similar in all means except for debt levels — same revenue (₹100) and same gross margin and same EBITDA margins of 20 per cent and thus same EBITDA of ₹20.
But A has zero debt, while B has debt of ₹10. Since revenue or sales and EBITDA are same, if you do a market cap/revenue or Price /Sales, or marketcap/EBITDA,
both companies will have the same valuation.

However, let’s assume depreciation is ₹10, interest cost is 10 per cent and taxes are zero. This means the net profit of A is ₹10, while that of B is ₹9.
</item 92>

<item 93>
Earnings week ahead: Amazon, Microsoft, Alphabet, Meta, Ford, GM and more (Seeking Alpha, 2023-10-22T12:13:00+00:00)
Meta's cost-cutting actions and a rebound in ad income boosted the stock's performance this year, which has climbed by about 150%. The stock has been rated as a
Strong Buy by sell-side analysts and Seeking Alpha's Quant Rating system.

Recently, Alphabet's unit Google (GOOG) (GOOGL), Meta Platforms (META), and Apple (AAPL) have received approval from U.S. regulators to deploy the new set of
mobile AR/VR devices, including goggles and in-car connections.

Consensus EPS Estimates: $3.60

Consensus Revenue Estimates: $33.43B

Earnings Insight: Meta has topped EPS expectations in 4 of the past 8 quarters, beating revenue estimates in 5 of those reports.

Boeing (BA)

Boeing (BA) is preparing to report Q3 results after the closing bell on Wednesday.
</item 93>

<item 94>
Fundamental Investing: The Art of Relative Valuation (Business Line, 2023-10-14T15:38:28+00:00)
Here are a few pointers.

One, after comparing stocks based on relative valuation, you can do more layers of comparison by assessing how their valuation fares relative to growth
prospects. This apart, many other factors can justify different valuations. For example, TCS has always commanded a valuation premium to peer stocks like
Infosys, Wipro and HCLTech because of its superior margins and much more consistent growth. So just because the other stocks are cheaper on PE basis does not
mean they are cheap compared to TCS. So you can also do one more layer of comparison by assessing each company’s valuation against its long-term (5-10 years)
mean and median valuations and assess how they fare relative to their own history.

Two, equity valuations do not function in a silo. Investors have many options — fixed deposits, bonds, real estate, etc. When other assets get more attractive
one must assess scope for valuation change in equities as well.
</item 94>

<item 95>
Earnings week ahead: DocuSign, NIO, Broadcom, GameStop and more (Seeking Alpha, 2023-12-03T13:02:00+00:00)
Analysts expect losses to narrow Y/Y with relatively flat revenue.

Over the past 5 days, GameStop stock has experienced a surge of over 20% amid the recent volatility in one-time meme stocks. BTIG technical strategist Jonathan
Krinsky attributes this upward trend to renewed buying interest in this group.

Wall Street analysts recommend selling GME stock, whereas Seeking Alpha's Quant rating system advises a Hold.

Seeking Alpha author Josh Arnold writes, "GameStop faces fundamental headwinds, including poor margins and declining sales, making it a structurally
unprofitable company."

Consensus EPS Estimates: -$0.08

Consensus Revenue Estimates: $1.18B

Earnings Insight: GameStop has beaten EPS estimates in only 3 of the past 8 quarters and revenue expectations in 5 of those reports.
</item 95>

<item 96>
Sam Altman backs teens’ startup, Google unveils the Pixel 8 and TikTok tests an ad-free tier (TechCrunch, 2023-10-07T20:15:26+00:00)
Battery-boosting software tech: Tim covers Breathe Battery Technologies, a startup that’s developed a bit of software that can be slipped into just about any
lithium-ion battery in use today — endowing it with either faster charging speeds or greater longevity.

What lies beyond ChatGPT: Anna surveyed 10 investors about the future of AI and what they believe might be the next big thing. Among other topics, they touched
on where startups still stand a chance, where oligopoly dynamics and first-mover advantages are shaping up and the value of proprietary data.
</item 96>

<item 97>
6 VCs explain how startups can capture and defend marketshare in the AI era (TechCrunch, 2023-10-13T20:01:40+00:00)
When considering the landscape of foundational layer model providers like Alphabet/Google’s Bard, Microsoft/OpenAI’s GPT-4, and Anthropic’s Claude, it’s evident
that the more significant players possess inherent advantages regarding data accessibility, talent pool and computational resources. We expect this layer to
settle into an oligopolistic structure like the cloud provider market, albeit with the addition of a strong open source contingency that will drive considerable
third-party adoption.

As we look at the generative AI tech stack, the largest market opportunity lies above the model itself. Companies that introduce AI-powered APIs and operational
layers for specific industries will create brand-new use cases and transform workflows. By embracing this technology to revolutionize workflows, these companies
stand to unlock substantial value.

However, it’s essential to recognize that the market is still far from being crystallized. LLMs are still in their infancy, with adoption at large corporations
and startups lacking full maturity and refinement. We need robust tools and platforms to enable broader utilization among businesses and individuals.
</item 97>

<item 98>
Check Point sees 'fantastic' year ahead even as rival report disappointing billings (Seeking Alpha, 2023-11-18T13:56:00+00:00)
Palo Alto (PANW) said it nonetheless sees demand for its products.

Earlier this month, Fortinet (NASDAQ:FTNT) got a slew of analyst downgrades over a disappointing fourth-quarter outlook.

The near- to medium-term is uncertain after disappointing results, Cantor Fitzgerald said. Stifel, which likes the company, said moving parts will take time to
play out and shares are expected trade sideways in coming quarters.

The future

Check Point (CHKP) has made three recent acquisitions to expand its offerings.

It has also invested in R&D and launched new products, like the Rugged Industrial Control Systems Gateway, which provides artificial intelligence security and
deployments in harsh environments. Other new offerings include Playblocks, which extends the reach of siloed security systems to stop attacks across products,
people and processes.
</item 98>

<item 99>
A comprehensive list of 2023 tech layoffs (TechCrunch, 2023-09-28T20:04:20+00:00)
GitHub

Announced February 9, 10% of its staff will be impacted through the end of the company’s fiscal year. Before this announcement, which was first reported by
Fortune, GitHub had about 3,000 employees.

Yahoo

Announced on February 9, 20% of its staff, impacting 1,600 employees in its adtech business. Yahoo is the parent company to TechCrunch.

GitLab

Announced February 9 that it’s reducing its headcount by 7%. The round of redundancies will impact around 114 people, though that specific figure is dependent
on its actual headcount as of February 9.

Affirm

Announced on February 8 that it is reducing its staff by 19%, or about 500 employees, and shutting down its crypto unit.

Zoom

Announced the cut of 15% of its staff, or 1,300 people on February 7.
</item 99>

<item 100>
Check Point sees 'fantastic' year ahead even as rival report disappointing billings (Seeking Alpha, 2023-11-18T13:56:00+00:00)
The financial damage from cybercrimes will reach an astronomical $10.5 trillion by 2025. As global economies go, that would rank third behind only the U.S. and
China.

But what’s bad for global conglomerates and the average person could be an opportunity for cybersecurity firms like Check Point Software (NASDAQ:CHKP).

Even as rivals like Palo Alto Networks (PANW) and Fortinet (FTNT) cite an uncertain macroeconomic environment for weak billings forecasts, the need for
cybersecurity is clearly growing. Global cybercrime damage costs are expected to increase 15% annually over the next two years, according to Cybersecurity
Ventures.

“There’s never been a better time to be in cyber,” Check Point (CHKP) President Rupal Shah Hollenbeck said in a Zoom interview this week. “It’s fortunate in
that organizations are understanding the business value of cybersecurity in a really big way. Unfortunately, it’s because there are more hacks out there than
ever before.”
</item 100>
=== END ITEM id=8b0542150af918b6 ===
