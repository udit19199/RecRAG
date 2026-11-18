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

This file has 3 items.

=== ITEM id=1253f31d172fc656 ===
Gold sentences and facts:
Sentence 0: The first of these is the fact that tech companies continue to see huge pressure on their valuations.
  Fact 0.0: The first of these is that tech companies continue to see huge pressure on their valuations.
Sentence 1: With a 10% surge in the stock this year, Seeking Alpha's Quant Rating system suggests a Hold, expressing concerns about valuation.
  Fact 1.0: The stock rose 10% this year.
  Fact 1.1: Seeking Alpha's Quant Rating system suggests a Hold rating for the stock.
  Fact 1.2: The Hold suggestion from Seeking Alpha's Quant Rating system expresses concerns about valuation.

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

Graph facts:
- A comprehensive list of 2023 tech layoffs -MENTIONS-> Turkey
- A comprehensive list of 2023 tech layoffs -MENTIONS-> Yahoo
- A comprehensive list of 2023 tech layoffs -MENTIONS-> Chapter 11 bankruptcy
- A comprehensive list of 2023 tech layoffs -MENTIONS-> IronNet
- A comprehensive list of 2023 tech layoffs -MENTIONS-> Mark Zuckerberg

Entities: tech layoffs, Layoffs.fyi, Miranda Halpern, Flourish, ShareChat, cutting 15% of its workforce, A comprehensive list of 2023 tech layoffs, TechCrunch
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

Graph facts:
- There’s something going on with AI startups in France -MENTIONS-> GitHub
- GitHub -SAID-> streamlining effort
- Jason Warner -WORKS_FOR-> GitHub
- GitHub -PARTICIPATED_IN-> GitHub staff reduction announcement
- A comprehensive list of 2023 tech layoffs -MENTIONS-> Chapter 11 bankruptcy

Entities: GitHub, tech layoffs, A comprehensive list of 2023 tech layoffs, Unacademy, Shift Technologies, South Asian market, Netflix, TechCrunch, India, Roku,
Lucid, U.S.
</item 2>

<item 3>
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

Graph facts:
- A comprehensive list of 2023 tech layoffs -MENTIONS-> Alphabet
- News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line -MENTIONS-> Alphabet
- ASX set to rise ahead of inflation report after big tech lifts Wall Street -MENTIONS-> Alphabet
- Amazon’s Tye Brady discusses generative AI, humanoid robots and mobile manipulation -MENTIONS-> Alphabet
- Palestine’s growing tech industry has been literally blown apart by Israel’s war on Hamas -MENTIONS-> Alphabet

Entities: Alphabet, Microsoft, Bank of America, Meta, Google, artificial intelligence, GM, search, Q2 results, Q4 performance report, Earnings week ahead:
Amazon, Microsoft, Alphabet, Meta, Ford, GM and more, Seeking Alpha, S&P 500, YouTube, Ford
</item 3>

<item 4>
Earnings week ahead: Amazon, Microsoft, Alphabet, Meta, Ford, GM and more (Seeking Alpha, 2023-10-22T12:13:00+00:00)
Shares of the now-Virginia-based industrial and defense giant have dropped over 7% over the past 6 months, attracting a cautious view of Hold from Seeking
Alpha's Quant Rating system, pointing to valuation concerns. Meanwhile, Wall Street analysts rate the stock a Buy.

During the third quarter, Boeing and its supplier Spirit AeroSystems (SPR) determined that the aft pressure bulkhead on many Boeing 737 MAX airplanes had
erroneously drilled fastening holes. At the 2023 Industrial Conference, CFO Brian West claimed that the current issue will affect 75% of the 220 planes in
inventory, or 165 planes. Despite persistent 737 manufacturing problems, the firm reiterated its ambitions to boost 787 production from three to four per month
by the end of 2023.

The company delivered 27 commercial jets to customers in the month of September, raising its yearly total to 371 planes.

Graph facts:
- After losing her leg to the flu, Virginia woman urges people to get vaccinated: ‘Don’t waste time’ -MENTIONS-> Virginia
- Biden’s AI EO hailed as broad, but not deep without legislation to match -MENTIONS-> Virginia
- The obsessive tormenter who made professors’ lives miserable -MENTIONS-> Virginia
- Best sportsbook bonus offers for NFL Monday Night Football Eagles vs. Seahawks: Claim over $5,000 in bonuses from Bet365, BetMGM, BetRivers, Caesars
Sportsbook, DraftKings and FanDuel -MENTIONS-> Virginia
- Sam Altman backs teens’ startup, Google unveils the Pixel 8 and TikTok tests an ad-free tier -MENTIONS-> Virginia

Entities: Virginia, Boeing, Spirit AeroSystems, Brian West, 2023 Industrial Conference, Boeing 737 MAX airplanes, 787, aft pressure bulkhead fastening-hole
issue, Earnings week ahead: Amazon, Microsoft, Alphabet, Meta, Ford, GM and more, Seeking Alpha, Seeking Alpha Quant Rating system, Wall Street analysts
</item 4>

<item 5>
Keep Labs’ smart storage redefines medication management (TechCrunch, 2023-10-04T19:24:25+00:00)
This comprehensive, multi-layered approach to security ensures that patient information remains confidential and secure at all times.”

Wilkins added that its encryption keys are generated on the fly within their production environment and securely stored. No individual has direct access to the
keys. Data is transmitted using TLS 1.3 encryption, while data at rest is secured with AES-256 encryption. User passwords undergo hashing with PBKDF2 utilizing
SHA256. The company uses automated tools such as Scan Hawk and Synk to test its security and provide code vulnerability assessments. He notes that their code is
subject to peer reviews, and the company uses the Coalition Cybersecurity and Stendard to provide a final layer of external review.

It’s worth applauding Keep Lab’s security and privacy measures. The above shows a company that’s proud of its efforts and is comfortable making them public.

Cooling meds is next for Keep

With its first product shipping to consumers, Keep Labs is exploring raising capital at the end of 2023 or early 2024.

Graph facts:
- SBF’s trial starts soon, but how did he — and FTX — get here? -PUBLISHED_BY-> TechCrunch
- Sam Altman backs teens’ startup, Google unveils the Pixel 8 and TikTok tests an ad-free tier -PUBLISHED_BY-> TechCrunch
- More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo -PUBLISHED_BY-> TechCrunch
- After warning, the EU opens investigation into X over alleged disinformation -PUBLISHED_BY-> TechCrunch
- SBF Trial: The latest updates from the FTX collapse’s courtroom drama -PUBLISHED_BY-> TechCrunch

Entities: TechCrunch, Keep Labs, Wilkins, smart storage, Keep Labs’ smart storage redefines medication management, Scan Hawk, Synk, Coalition Cybersecurity,
Stendard, raising capital
</item 5>

<item 6>
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

Graph facts:
- Earnings week ahead: Amazon, Microsoft, Alphabet, Meta, Ford, GM and more -PUBLISHED_BY-> Seeking Alpha
- Earnings week ahead: DocuSign, NIO, Broadcom, GameStop and more -PUBLISHED_BY-> Seeking Alpha
- Earnings week ahead: Amazon, Microsoft, Alphabet, Meta, Ford, GM and more -MENTIONS-> Seeking Alpha
- Palo Alto in advanced talks to buy Talon and Dig in a $1B security sweep -MENTIONS-> Palo Alto
- Palo Alto -PARTICIPATED_IN-> advanced talks to buy Talon and Dig

Entities: Seeking Alpha, Palo Alto, Fortinet, Check Point sees 'fantastic' year ahead even as rival report disappointing billings, Check Point, Cantor
Fitzgerald, Stifel, Rugged Industrial Control Systems Gateway, Playblocks, artificial intelligence security, demand for its products, near- to medium-term
uncertainty after disappointing results, shares expected to trade sideways in coming quarters
</item 6>

<item 7>
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

Graph facts:
- Yahoo -COMPETES_WITH-> Google
- The people who ruined the internet -MENTIONS-> Yahoo
- Baker -SAID-> Yahoo
- More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo -MENTIONS-> Yahoo
- Google -COMPETES_WITH-> Yahoo

Entities: Yahoo, GitHub, A comprehensive list of 2023 tech layoffs, GitLab, Affirm, Zoom, GitHub staff reduction announcement, Yahoo staff reduction
announcement, GitLab headcount reduction announcement, Affirm staff reduction announcement, Affirm crypto unit shutdown, Zoom staff reduction announcement,
crypto unit, TechCrunch, Fortune
</item 7>

<item 8>
Graph query row:
article: OpenAI buffs safety team and gives board veto power on risky AI
published_on: 2023-12-18T20:12:25+00:00
publisher: TechCrunch
claims: ['OpenAI buffs safety team and gives board veto power on risky AI (TechCrunch, 2023-12-18T20:12:25+00:00)\nOpenAI is expanding its internal safety
processes to fend off the threat of harmful AI. A new “safety advisory group” will sit above the technical teams and make recommendations to leadership, and the
board has been granted veto power — of course, whether it will actually use it is another question entirely.\n\nNormally the ins and outs of policies like these
don’t necessitate coverage, as in practice they amount to a lot of closed-door meetings with obscure functions and responsibility flows that outsiders will
seldom be privy to. Though that’s likely also true in this case, the recent leadership fracas and evolving AI risk discussion warrant taking a look at how the
world’s leading AI development company is approaching safety considerations.\n\nIn a new document and blog post, OpenAI discusses their updated “Preparedness
Framework,” which one imagines got a bit of a retool after November’s shake-up that removed the board’s two most “decelerationist” members: Ilya Sutskever
(still at the company in a somewhat changed role) and Helen Toner (totally gone).', 'OpenAI buffs safety team and gives board veto power on risky AI
(TechCrunch, 2023-12-18T20:12:25+00:00)\nThe main purpose of the update appears to be to show a clear path for identifying, analyzing, and deciding what do to
about “catastrophic” risks inherent to models they are developing. As they define it:\n\nBy catastrophic risk, we mean any risk which could result in hundreds
of billions of dollars in economic damage or lead to the severe harm or death of many individuals — this includes, but is not limited to, existential
risk.\n\n(Existential risk is the “rise of the machines” type stuff.)\n\nIn-production models are governed by a “safety systems” team; this is for, say,
systematic abuses of ChatGPT that can be mitigated with API restrictions or tuning. Frontier models in development get the “preparedness” team, which tries to
identify and quantify risks before the model is released. And then there’s the “superalignment” team, which is working on theoretical guide rails for
“superintelligent” models, which we may or may not be anywhere near.', 'OpenAI buffs safety team and gives board veto power on risky AI (TechCrunch,
2023-12-18T20:12:25+00:00)\nThe first two categories, being real and not fictional, have a relatively easy-to-understand rubric. Their teams rate each model on
four risk categories: cybersecurity, “persuasion” (e.g., disinfo), model autonomy (i.e., acting on its own), and CBRN (chemical, biological, radiological, and
nuclear threats; e.g., the ability to create novel pathogens).\n\nVarious mitigations are assumed: For instance, a reasonable reticence to describe the process
of making napalm or pipe bombs. After taking into account known mitigations, if a model is still evaluated as having a “high” risk, it cannot be deployed, and
if a model has any “critical” risks, it will not be developed further.\n\nThese risk levels are actually documented in the framework, in case you were wondering
if they are to be left to the discretion of some engineer or product manager.\n\nFor example, in the cybersecurity section, which is the most practical of them,
it is a “medium” risk to “increase the productivity of operators .', 'OpenAI buffs safety team and gives board veto power on risky AI (TechCrunch,
2023-12-18T20:12:25+00:00)\n. . on key cyber operation tasks” by a certain factor. A high-risk model, on the other hand, would “identify and develop
proofs-of-concept for high-value exploits against hardened targets without human intervention.” Critical is “model can devise and execute end-to-end novel
strategies for cyberattacks against hardened targets given only a high level desired goal.” Obviously we don’t want that out there (though it would sell for
quite a sum).\n\nI’ve asked OpenAI for more information on how these categories are defined and refined — for instance, if a new risk like photorealistic fake
video of people goes under “persuasion” or a new category — and will update this post if I hear back.\n\nSo, only medium and high risks are to be tolerated one
way or another. But the people making those models aren’t necessarily the best ones to evaluate them and make recommendations.', 'OpenAI buffs safety team and
gives board veto power on risky AI (TechCrunch, 2023-12-18T20:12:25+00:00)\nBut the people making those models aren’t necessarily the best ones to evaluate them
and make recommendations. For that reason, OpenAI is making a “cross-functional Safety Advisory Group” that will sit on top of the technical side, reviewing the
boffins’ reports and making recommendations inclusive of a higher vantage. Hopefully (they say) this will uncover some “unknown unknowns,” though by their
nature those are fairly difficult to catch.\n\nThe process requires these recommendations to be sent simultaneously to the board and leadership, which we
understand to mean CEO Sam Altman and CTO Mira Murati, plus their lieutenants. Leadership will make the decision on whether to ship it or fridge it, but the
board will be able to reverse those decisions.\n\nThis will hopefully short-circuit anything like what was rumored to have happened before the big drama, a
high-risk product or process getting greenlit without the board’s awareness or approval.']
connections: [{'relationship': 'MENTIONS', 'entity': 'ChatGPT'}, {'relationship': 'MENTIONS', 'entity': 'safety advisory group'}, {'relationship': 'MENTIONS',
'entity': 'technical teams'}, {'relationship': 'MENTIONS', 'entity': 'leadership'}, {'relationship': 'MENTIONS', 'entity': 'Preparedness Framework'},
{'relationship': 'MENTIONS', 'entity': 'November’s shake-up'}, {'relationship': 'MENTIONS', 'entity': 'AI risk'}, {'relationship': 'MENTIONS', 'entity':
'harmful AI'}, {'relationship': 'MENTIONS', 'entity': 'safety systems team'}, {'relationship': 'MENTIONS', 'entity': 'preparedness team'}]
</item 8>

<item 9>
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

Graph facts:
- A comprehensive list of 2023 tech layoffs -MENTIONS-> Alphabet
- News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line -MENTIONS-> Alphabet
- ASX set to rise ahead of inflation report after big tech lifts Wall Street -MENTIONS-> Alphabet
- Amazon’s Tye Brady discusses generative AI, humanoid robots and mobile manipulation -MENTIONS-> Alphabet
- Palestine’s growing tech industry has been literally blown apart by Israel’s war on Hamas -MENTIONS-> Alphabet

Entities: Alphabet, Microsoft, Meta, GM, Cleveland-Cliffs, Jonathan Weber, Logitech International S.A., Koninklijke Philips N.V., Packaging Corporation of
America, automobile companies, Alphabet Q3 results, Earnings week ahead: Amazon, Microsoft, Alphabet, Meta, Ford, GM and more, Seeking Alpha, Europe, Ford
</item 9>

<item 10>
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

Graph facts:
- A comprehensive list of 2023 tech layoffs -MENTIONS-> GitLab
- GitLab -PARTICIPATED_IN-> GitLab headcount reduction announcement
- Earnings week ahead: Amazon, Microsoft, Alphabet, Meta, Ford, GM and more -PUBLISHED_BY-> Seeking Alpha
- Check Point sees 'fantastic' year ahead even as rival report disappointing billings -PUBLISHED_BY-> Seeking Alpha
- Earnings week ahead: Amazon, Microsoft, Alphabet, Meta, Ford, GM and more -MENTIONS-> Seeking Alpha

Entities: GitLab, Seeking Alpha, Cantor Fitzgerald, GameStop, Raymond James, Seeking Alpha Quant Rating system, Q3 results, generative artificial intelligence,
NIO, Earnings week ahead: DocuSign, NIO, Broadcom, GameStop and more, DocuSign, Broadcom, Wall Street analysts
</item 10>
=== END ITEM id=1253f31d172fc656 ===

=== ITEM id=7ee83a4206b1dcac ===
Gold sentences and facts:
Sentence 0: That answer ultimately lies with the jury, which was selected on Wednesday morning before lawyers for the government and then Bankman-Fried swapped two very different stories of the former crypto mogul’s sudden rise and almost instantaneous fall.
  Fact 0.0: The answer ultimately lies with the jury.
  Fact 0.1: The jury was selected on Wednesday morning, before lawyers for the government and Bankman-Fried swapped two very different stories.
  Fact 0.2: Lawyers for the government and Bankman-Fried swapped two very different stories of the former crypto mogul’s sudden rise and almost instantaneous fall.
Sentence 1: The highly anticipated criminal trial for Sam Bankman-Fried, former CEO of bankrupt crypto exchange FTX, started Tuesday to determine whether he’s guilty of seven counts of fraud and conspiracy.
  Fact 1.0: Sam Bankman-Fried is the former CEO of bankrupt crypto exchange FTX.
  Fact 1.1: The highly anticipated criminal trial for Sam Bankman-Fried started on Tuesday.
  Fact 1.2: The criminal trial is to determine whether Sam Bankman-Fried is guilty of seven counts of fraud and conspiracy.
Sentence 2: The prosecution painted Bankman-Fried as someone who knowingly committed fraud to achieve great wealth, power and influence, while the defense countered that the FTX founder acted in good faith, never meant to commit fraud or steal and basically got in over his head.
  Fact 2.0: The prosecution portrayed Bankman-Fried as someone who knowingly committed fraud to achieve great wealth, power and influence.
  Fact 2.1: The defense argued that the FTX founder acted in good faith.
  Fact 2.2: The defense argued that the FTX founder never meant to commit fraud or steal.
  Fact 2.3: The defense argued that the FTX founder basically got in over his head.

Retrieved items:
<item 1>
How is it still getting worse for Sam Bankman-Fried? (The Verge, 2023-10-12T23:08:14+00:00)
The recording was later sent to Drappi, who sent it to the government. Joseph Bankman, the defendant’s father and a senior advisor to FTX’s philanthropic arm,
wasn’t in the room for the recording. Barbara Fried looked unhappy, rubbing her left temple as though she had a tremendous headache.

And in a small miracle of pacing, the government got Zac Prince, the founder and former CEO of crypto lender BlockFi, on the stand just long enough to blame
FTX’s bankruptcy for BlockFi’s subsequent bankruptcy before we broke for the day.

I’ve been asked by some people why the reporting coming out of the trial seems so skewed toward the prosecution. There’s an answer for this: the prosecution has
put together a strong, comprehensive indictment of Bankman-Fried’s behavior at FTX and Alameda. The defense, so far, has managed to do almost nothing in
response.

Graph facts:
- SBF’s trial starts soon, but how did he — and FTX — get here? -MENTIONS-> Alameda
- SBF, riding high on FTX, reportedly offered $5B to Trump to cancel President re-run -MENTIONS-> Alameda
- Is Sam Bankman-Fried a bad ‘man’ or a good ‘boy’? Lawyers swap opening statements before first witnesses take the stand -MENTIONS-> Alameda
- The jury finally hears from Sam Bankman-Fried -MENTIONS-> Alameda
- SBF Trial: The latest updates from the FTX collapse’s courtroom drama -MENTIONS-> Alameda

Entities: Alameda, Barbara Fried, the defense, FTX, the government, Sam Bankman-Fried, How is it still getting worse for Sam Bankman-Fried?, the recording,
Drappi, Joseph Bankman, FTX’s philanthropic arm, Zac Prince, BlockFi, the trial, FTX’s bankruptcy
</item 1>

<item 2>
Is Sam Bankman-Fried a bad ‘man’ or a good ‘boy’? Lawyers swap opening statements before first witnesses take the stand (Fortune, 2023-10-04T23:32:53+00:00)
How much? Almost $100,000, he said. And was he ever able to? “Never,” he told prosecutors.

Shortly afterward, as the trial neared late afternoon, the government called Adam Yedidia to the stand. A fast-talking graduate of MIT, he and Bankman-Fried
were close friends in college, he said. And after Bankman-Fried left Jane Street, the high-frequency trading firm where the former billionaire got his start in
finance after MIT, he persuaded Yedidia to join him as a trader at Alameda and then as a developer at FTX.

When Yedidia first took the stand, Danielle Sassoon, one of the lead prosecutors, said that the college friend of Bankman-Fried had legal immunity during his
testimony. Why did he make such a deal with the government, she asked.

“I was concerned that I had unwittingly written code that contributed to a crime,” he said.

Graph facts:
- Is Sam Bankman-Fried a bad ‘man’ or a good ‘boy’? Lawyers swap opening statements before first witnesses take the stand -MENTIONS-> Matt Huang
- Is Sam Bankman-Fried a bad ‘man’ or a good ‘boy’? Lawyers swap opening statements before first witnesses take the stand -MENTIONS-> Damian Williams
- Is Sam Bankman-Fried a bad ‘man’ or a good ‘boy’? Lawyers swap opening statements before first witnesses take the stand -MENTIONS-> Gary Wang
- Is Sam Bankman-Fried a bad ‘man’ or a good ‘boy’? Lawyers swap opening statements before first witnesses take the stand -MENTIONS-> Caroline Ellison
- Is Sam Bankman-Fried a bad ‘man’ or a good ‘boy’? Lawyers swap opening statements before first witnesses take the stand -MENTIONS-> crypto

Entities: Is Sam Bankman-Fried a bad ‘man’ or a good ‘boy’? Lawyers swap opening statements before first witnesses take the stand, Adam Yedidia, Jane Street,
Danielle Sassoon, Alameda, crime, finance, Fortune, FTX, the government, Sam Bankman-Fried, trial, MIT
</item 2>

<item 3>
Is Sam Bankman-Fried a bad ‘man’ or a good ‘boy’? Lawyers swap opening statements before first witnesses take the stand (Fortune, 2023-10-04T23:32:53+00:00)
“You will see the full picture.”

…or a well-meaning founder?

But Bankman-Fried, whose cheekbones were more prominent after spending about seven weeks in a Brooklyn prison, was no liar, according to Mark Cohen, one of his
lawyers. “Sam didn’t defraud anyone,” he said early on in his opening statement.

What the jury will see is a nerdy startup founder who acted in “good faith,” not the prosecution’s “cartoon of a villain.” (Cohen repeatedly harped on
Bankman-Fried’s allegedly good-faith actions throughout his address to the jury.)

Alameda was not subterranean or shady. It was a successful hedge fund, he said. FTX was no Ponzi scheme. It was a “very innovative, successful company.”

Graph facts:
- SBF’s trial starts soon, but how did he — and FTX — get here? -MENTIONS-> Mark Cohen
- SBF Trial: The latest updates from the FTX collapse’s courtroom drama -MENTIONS-> Mark Cohen
- The jury finally hears from Sam Bankman-Fried -MENTIONS-> Mark Cohen
- Is Sam Bankman-Fried’s defense even trying to win? -MENTIONS-> Mark Cohen
- How is it still getting worse for Sam Bankman-Fried? -MENTIONS-> Mark Cohen

Entities: Mark Cohen, Is Sam Bankman-Fried a bad ‘man’ or a good ‘boy’? Lawyers swap opening statements before first witnesses take the stand, Brooklyn,
Alameda, Fortune, FTX, Sam Bankman-Fried
</item 3>

<item 4>
Graph query row:
title: SBF Trial: The latest updates from the FTX collapse’s courtroom drama
publication_date: 2023-10-06T17:22:02+00:00
passage: SBF Trial: The latest updates from the FTX collapse’s courtroom drama (TechCrunch, 2023-10-06T17:22:02+00:00)
SBF trial: Everything to know from the FTX courtroom ahead of his testimony Sam Bankman-Fried is in court for alleged fraud and money laundering while leading
crypto exchange FTX and sister company Alameda. Here’s our latest coverage as the case nears its end.

The trial of Bankman-Fried, the former CEO and co-founder of collapsed crypto exchange FTX, is entering its next chapter as proceedings resume Thursday. As
confirmed during a teleconference Wednesday, SBF will be taking the stand, along with a handful of other witnesses as the onus shifts toward the defense after
the prosecution brought its case to its conclusion.

As former federal prosecutor Josh Naftalis told TechCrunch, SBF taking to the stand in his own trial is a “Hail Mary” ahead of a verdict and potential
sentencing for seven counts tied to fraud and money laundering.

“Once it goes to cross[-examination], he doesn’t get to say, ‘I’m done,’” Naftalis said.
relationship: MENTIONS
connection: Josh Naftalis
</item 4>

<item 5>
Sam Altman backs teens’ startup, Google unveils the Pixel 8 and TikTok tests an ad-free tier (TechCrunch, 2023-10-07T20:15:26+00:00)
Jacquelyn has been on the ground at the Southern District of New York courthouse, listening in to the trial in the same room as Bankman-Fried, so there was lots
to talk about.

TechCrunch+

TC+ subscribers get access to in-depth commentary, analysis and surveys — which you know if you’re already a subscriber. If you’re not, consider signing up.
Here are a few highlights from this week:

Inside the SBF trial: Rebecca and Jacquelyn report on the second day of the SBF and FTX trial. The prosecution painted Bankman-Fried as someone who knowingly
committed fraud to achieve great wealth, power and influence, while the defense countered that the FTX founder acted in good faith, never meant to commit fraud
or steal and basically got in over his head.

Graph facts:
- Sam Altman backs teens’ startup, Google unveils the Pixel 8 and TikTok tests an ad-free tier -MENTIONS-> SBF Trial
- Sam Altman backs teens’ startup, Google unveils the Pixel 8 and TikTok tests an ad-free tier -MENTIONS-> Google’s hardware event
- Founders, are events useful? -MENTIONS-> Pixel 8
- The 53 best Black Friday deals we could find at Amazon, Walmart, Target and more -MENTIONS-> Pixel 8
- Sam Altman backs teens’ startup, Google unveils the Pixel 8 and TikTok tests an ad-free tier -MENTIONS-> Pixel 8 Pro

Entities: teens’ startup, ad-free tier, Pixel 8, Sam Altman backs teens’ startup, Google unveils the Pixel 8 and TikTok tests an ad-free tier, Jacquelyn,
Southern District of New York courthouse, Bankman-Fried, SBF and FTX trial, TechCrunch, TechCrunch+, the defense, FTX, Google, Rebecca, the prosecution
</item 5>

<item 6>
Is Sam Bankman-Fried’s defense even trying to win? (The Verge, 2023-10-04T23:02:14+00:00)
I have never seen Sam Bankman-Fried so still as he was during the prosecution’s opening statement. The characteristic leg-jiggling was absent. He barely moved
as the prosecutor listed the evidence against him: internal company files, what customers were told, the testimony of his co-conspirators and his own words.

His hair was shorn, the result of a haircut from a fellow prisoner, the Wall Street Journal reported. He wore a suit bought at a discount at Macy’s, per the
Journal; it hung on him. He appeared to have lost some weight.

“All of that was built on lies.”

Bankman-Fried, at this time last year, had a luxury lifestyle as the CEO of crypto exchange FTX, said the assistant US attorney, Thane Rehn, in the cadence of a
high schooler delivering his lines in a student play. Bankman-Fried hung out with Tom Brady. He was on magazine covers, lived in a $30 million penthouse, and
spent time with world politicians.

Graph facts:
- Is Sam Bankman-Fried a bad ‘man’ or a good ‘boy’? Lawyers swap opening statements before first witnesses take the stand -MENTIONS-> Thane Rehn
- SBF Trial: The latest updates from the FTX collapse’s courtroom drama -MENTIONS-> Thane Rehn
- Thane Rehn -SAID-> SBF
- Thane Rehn -SAID-> stealing customer funds
- Thane Rehn -WORKS_FOR-> government

Entities: Thane Rehn, The Wall Street Journal, prosecution’s opening statement, Is Sam Bankman-Fried’s defense even trying to win?, FTX, Macy’s, Sam
Bankman-Fried, Tom Brady, The Verge
</item 6>

<item 7>
Is Sam Bankman-Fried a bad ‘man’ or a good ‘boy’? Lawyers swap opening statements before first witnesses take the stand (Fortune, 2023-10-04T23:32:53+00:00)
Who is Sam Bankman-Fried, the former CEO of crypto exchange FTX? Is he a liar and a fraud? Or did he act in good faith, and like many a startup founder, fly too
close to the sun?

That answer ultimately lies with the jury, which was selected on Wednesday morning before lawyers for the government and then Bankman-Fried swapped two very
different stories of the former crypto mogul’s sudden rise and almost instantaneous fall.

Here’s what happened on the second day of the trial, which featured pointed allegations, a friend from MIT, and an audience replete with big names, including
Bankman-Fried’s professorial parents and Damian Williams, U.S. attorney for the Southern District of New York.

A conscious criminal…

The prosecution’s account of the alleged crimes by Bankman-Fried, who spent most of his day in court staring into a laptop while seated between his two
attorneys, featured a study in contrasts.

Graph facts:
- SBF’s trial starts soon, but how did he — and FTX — get here? -MENTIONS-> Damian Williams
- Damian Williams -SAID-> financial fraud
- Damian Williams -PARTICIPATED_IN-> Press conference
- SBF Trial: The latest updates from the FTX collapse’s courtroom drama -MENTIONS-> crypto
- SBF’s trial starts soon, but how did he — and FTX — get here? -MENTIONS-> crypto

Entities: Damian Williams, crypto, alleged crimes, Is Sam Bankman-Fried a bad ‘man’ or a good ‘boy’? Lawyers swap opening statements before first witnesses take
the stand, Fortune, FTX, the government, Sam Bankman-Fried, the trial, MIT, fraud
</item 7>

<item 8>
The FTX trial is bigger than Sam Bankman-Fried (The Verge, 2023-09-28T12:00:00+00:00)
“We had him going out directly to the internet,” LaVigne says. If he wrote things on Twitter or said things in interviews that weren’t true, that’s more fodder
for the government’s case. “They can point to that and say, ‘This is what he said, this is what actually happened.’”

Other evidence may include Signal messages and testimony from co-conspirators who plead guilty to their own charges. Alameda Research CEO Caroline Ellison, who
was also sometimes Bankman-Fried’s girlfriend, may play an important role — Bankman-Fried leaked her diaries to The New York Times and was consequently jailed
for witness tampering. The government has indicated FTX co-founder Gary Wang and engineering head Nishad Singh will also be among the witnesses called to give
testimony about Bankman-Fried.

Bankman-Fried’s defense can also introduce risks for people who dealt with him

There may be testimony from lenders, venture capitalists, and customers in order to establish the basis for some charges.

Graph facts:
- He’s Hockey’s Brightest Young Star. This Is What Makes His Shot So Special. -PUBLISHED_BY-> The New York Times
- How U.S. Women’s Soccer Team Lured Emma Hayes Away From Chelsea -PUBLISHED_BY-> The New York Times
- After a Stunning Snub, College Football Is Facing an Identity Crisis -PUBLISHED_BY-> The New York Times
- N.F.L. Executives on the Biggest Questions Facing Super Bowl Contenders -PUBLISHED_BY-> The New York Times
- Which M.L.B. Playoff Team Has the Most Daunting Home Field? We Ranked Them. -PUBLISHED_BY-> The New York Times

Entities: The New York Times, Gary Wang, Caroline Ellison, Alameda Research, defense, Nishad Singh, Signal, FTX trial, The FTX trial is bigger than Sam
Bankman-Fried, LaVigne, witness tampering, FTX, Sam Bankman-Fried, government, Twitter
</item 8>

<item 9>
Is Sam Bankman-Fried a bad ‘man’ or a good ‘boy’? Lawyers swap opening statements before first witnesses take the stand (Fortune, 2023-10-04T23:32:53+00:00)
“One year ago, it looked like the defendant was on the top of the world,” began Thane Rehn, a prosecutor for the government, in his opening statement. The
former CEO of FTX oversaw a supposedly thriving crypto exchange, jetted between international locales, and hobnobbed with celebrities like Tom Brady and Larry
David. He repeatedly emphasized to customers that their money was safe and secure.

But “all of that, all of it, was built on lies,” Rehn declared to the jury. “Behind the curtains, he was not what he appeared to be.” What followed was a
roughly 30-minute story that repeatedly emphasized how Bankman-Fried allegedly stole customer funds to facilitate his jet-setting lifestyle, donate millions to
political candidates, and finance risky bets.

The key to his alleged scheme? Alameda Research, a crypto hedge fund he also owned, argued Rehn.

Graph facts:
- Thane Rehn -PARTICIPATED_IN-> SBF Trial
- Is Sam Bankman-Fried’s defense even trying to win? -MENTIONS-> Thane Rehn
- Thane Rehn -SAID-> SBF
- SBF Trial: The latest updates from the FTX collapse’s courtroom drama -MENTIONS-> Thane Rehn
- How is it still getting worse for Sam Bankman-Fried? -MENTIONS-> stealing customer funds

Entities: Thane Rehn, Larry David, stealing customer funds, Is Sam Bankman-Fried a bad ‘man’ or a good ‘boy’? Lawyers swap opening statements before first
witnesses take the stand, Alameda Research, Fortune, FTX, Sam Bankman-Fried, Tom Brady, government
</item 9>

<item 10>
Is Sam Bankman-Fried a bad ‘man’ or a good ‘boy’? Lawyers swap opening statements before first witnesses take the stand (Fortune, 2023-10-04T23:32:53+00:00)
Rather, he was willing to give up his personal wealth to make customers whole, Cohen argued.

“In the end, Sam started and built two billion-dollar businesses,” he concluded. “He didn’t steal any money.”

A Frenchman who lives in London testifies in New York

After lawyers from both sides depicted two very different Bankman-Frieds, the prosecution called its first two witnesses to the stand—and they weren’t
blockbuster names or former lieutenants-turned-government-cooperators, like Ellison.

The first was a victim: Marc-Antoine Julliard, a Paris-born cocoa trader who lives in London. In 2021, Julliard, who had coiffed hair and spoke with a strong
French accent, decided to invest in crypto and landed on FTX as his exchange of choice, where he traded cryptocurrencies like Bitcoin and Dogecoin.

On Nov. 8, in the crypto exchange’s final few days, he tried to pull out his cash and crypto.

Graph facts:
- SBF Trial: The latest updates from the FTX collapse’s courtroom drama -MENTIONS-> crypto
- SBF, riding high on FTX, reportedly offered $5B to Trump to cancel President re-run -MENTIONS-> crypto
- SBF’s trial starts soon, but how did he — and FTX — get here? -MENTIONS-> crypto
- Sam Bankman-Fried was a terrible boyfriend -MENTIONS-> crypto
- Is Sam Bankman-Fried’s defense even trying to win? -MENTIONS-> crypto

Entities: crypto, Is Sam Bankman-Fried a bad ‘man’ or a good ‘boy’? Lawyers swap opening statements before first witnesses take the stand, Bitcoin, Dogecoin,
attempted withdrawal of cash and crypto, FTX, Paris, Ellison, New York, Cohen, Marc-Antoine Julliard, Fortune, Sam Bankman-Fried, London
</item 10>
=== END ITEM id=7ee83a4206b1dcac ===

=== ITEM id=48bacf7b63472651 ===
Gold sentences and facts:
Sentence 0: He immediately left the game and proceeded to the locker room following the hit, and it was confirmed that he would not return with a head injury.
  Fact 0.0: He immediately left the game following the hit.
  Fact 0.1: He proceeded to the locker room following the hit.
  Fact 0.2: It was confirmed that he would not return with a head injury.
Sentence 1: The former All-Pro, who battled a lingering hamstring injury earlier this year, received attention on the sidelines after limping off, then left the game for further evaluation in the locker room.
  Fact 1.0: The former All-Pro battled a lingering hamstring injury earlier this year.
  Fact 1.1: The former All-Pro received attention on the sidelines after limping off.
  Fact 1.2: The former All-Pro then left the game for further evaluation in the locker room.

Retrieved items:
<item 1>
2023 NFL Week 11 injury roundup: Rams WR Cooper Kupp, Seahawks RB Kenneth Walker III among stars to exit (CBSSports.com, 2023-11-19T23:51:49+00:00)
Smith did return, however, for the final drive of the game in the losing effort. Head coach Pete Carroll said that Smith will "make it back." When asked if that
means for Thursday's matchup against San Francisco, the coach said, "I would think so."

Sam Howell (Commanders) received medical attention on the sidelines after enduring several hard hits to the upper body on a touchdown run against the Giants. He
was quickly cleared of a potential concussion, however, and remained in the game. Veteran journeyman Jacoby Brissett is Washington's backup.

RB

Aaron Jones (Packers) suited up against the Chargers despite a lingering hamstring issue, only to manage just five carries before suffering a knee injury. The
former Pro Bowler needed help to leave the field and was then carted to the locker room, replaced by A.J. Dillon.
</item 1>

<item 2>
Graph query row:
person: Cooper Kupp
article: 2023 NFL Week 11 injury roundup: Rams WR Cooper Kupp, Seahawks RB Kenneth Walker III among stars to exit
published_on: 2023-11-19T23:51:49+00:00
publisher: CBSSports.com
connected_person: Derrick Henry
connection: Fantasy Football Week 13 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game
first_relationship: MENTIONS
second_relationship: MENTIONS
passage: 2023 NFL Week 11 injury roundup: Rams WR Cooper Kupp, Seahawks RB Kenneth Walker III among stars to exit (CBSSports.com, 2023-11-19T23:51:49+00:00)
Week 11 of the 2023 NFL season has provided plenty of drama, from the Bears hanging with the Lions to the Giants getting a rebound from emergency quarterback
Tommy DeVito. But the latest round of regular-season action has also brought another wave of unfortunate injuries. Here's a roundup of the latest big names to
go down around the league:

QB

C.J. Stroud (Texans) left for a single play against the Cardinals after absorbing a hard hit, with backup Davis Mills entering in relief. Stroud did not receive
an injury designation, however, and returned after a quick check-in from trainers.

Geno Smith (Seahawks) suffered an injury to his throwing arm after guiding a 16-7 lead over the Rams, showing frustration while trying to stretch the arm after
a hit from Aaron Donald. Seattle quickly ruled him questionable to return with an elbow issue, and backup Drew Lock took over under center with 18 seconds left
in the third quarter.
</item 2>

<item 3>
2023 NFL Week 11 injury roundup: Rams WR Cooper Kupp, Seahawks RB Kenneth Walker III among stars to exit (CBSSports.com, 2023-11-19T23:51:49+00:00)
Dillon. Green Bay ruled him questionable to return, then downgraded him to out, but coach Matt LaFleur said after the Packers' win over Los Angeles that he
doesn't expect the injury to be a long-term issue.

De'Von Achane (Dolphins) returned from injured reserve -- and a four-game absence -- but limped to the sidelines after his second touch of the first quarter
against the Raiders. Miami ruled him questionable to return with a knee injury, and while he later returned to the sidelines from a locker-room visit, he was
replaced on the field indefinitely by Raheem Mostert. Head coach Mike McDaniel told reporters on Monday that Achane is day-to-day. While he hasn't been ruled
out of Friday's matchup with the Jets, he's not been given the green light either, so his status is worth monitoring.

Kenneth Walker III (Seahawks) took four carries against the Rams before leaving with an oblique injury.
</item 3>

<item 4>
Fantasy Football Week 13 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-11-30T22:10:58+00:00)
If they're ailing, the Rams would be smart to try and use the run game early and often. Expect a low-scoring game.

STARTS: Kyren Williams, Amari Cooper, Puka Nacua, David Njoku, Jerome Ford,

Kyren Williams, Amari Cooper, Puka Nacua, David Njoku, Jerome Ford, SITS: Cooper Kupp (flex at best), Kareem Hunt, Tutu Atwell, Tyler Higbee, Royce Freeman,
both QBs, both DSTs (low-end options)

Cooper Kupp (flex at best), Kareem Hunt, Tutu Atwell, Tyler Higbee, Royce Freeman, both QBs, both DSTs (low-end options) SLEEPERS: Elijah Moore (PPR flex)

Start Him (Lineup Decisions) Projections powered by Sportsline Amari Cooper WR CLE Cleveland • #2

Age: 29 • Experience: 9 yrs.
</item 4>

<item 5>
2023 NFL Week 11 injury roundup: Rams WR Cooper Kupp, Seahawks RB Kenneth Walker III among stars to exit (CBSSports.com, 2023-11-19T23:51:49+00:00)
Week 11 of the 2023 NFL season has provided plenty of drama, from the Bears hanging with the Lions to the Giants getting a rebound from emergency quarterback
Tommy DeVito. But the latest round of regular-season action has also brought another wave of unfortunate injuries. Here's a roundup of the latest big names to
go down around the league:

QB

C.J. Stroud (Texans) left for a single play against the Cardinals after absorbing a hard hit, with backup Davis Mills entering in relief. Stroud did not receive
an injury designation, however, and returned after a quick check-in from trainers.

Geno Smith (Seahawks) suffered an injury to his throwing arm after guiding a 16-7 lead over the Rams, showing frustration while trying to stretch the arm after
a hit from Aaron Donald. Seattle quickly ruled him questionable to return with an elbow issue, and backup Drew Lock took over under center with 18 seconds left
in the third quarter.

Graph facts:
- Christmas Day preview: 49ers, Ravens square off in potential Super Bowl sneak peek -MENTIONS-> NFL
- NFL DFS picks Week 15: Best sleepers, value players for FanDuel, DraftKings, SuperDraft lineups -MENTIONS-> Tommy DeVito
- Fantasy Football QB Rankings Week 14: Who to start, best sleepers at quarterback -MENTIONS-> Tommy DeVito
- Christmas Day preview: 49ers, Ravens square off in potential Super Bowl sneak peek -MENTIONS-> Tommy DeVito
- Brock Purdy -MENTIONS-> NFL

Entities: Tommy DeVito, NFL, CBSSports.com, 2023 NFL Week 11 injury roundup: Rams WR Cooper Kupp, Seahawks RB Kenneth Walker III among stars to exit, Davis
Mills, Cooper Kupp, Texans, Cardinals, C.J. Stroud, Lions, Rams, Geno Smith, Drew Lock, Seahawks, Kenneth Walker III
</item 5>

<item 6>
2023 NFL Week 11 injury roundup: Rams WR Cooper Kupp, Seahawks RB Kenneth Walker III among stars to exit (CBSSports.com, 2023-11-19T23:51:49+00:00)
DB

Talanoa Hufanga (49ers), who emerged as a ballhawk for San Francisco's secondary in 2022, left early against the Buccaneers with a knee injury. Coach Kyle
Shanahan confirmed to reporters on Monday that the All-Pro suffered a season-ending ACL tear.

Graph facts:
- Fantasy Football Week 13 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game -PUBLISHED_BY-> CBSSports.com
- Taylor Swift-Travis Kelce timeline: Everything to know about rumored romance between pop star, Chiefs TE -PUBLISHED_BY-> CBSSports.com
- 2023 Kentucky online sports betting sites: Best legal sportsbooks, promos, bonuses, mobile apps, how to bet -PUBLISHED_BY-> CBSSports.com
- NFL Fantasy Football Week 5 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game -PUBLISHED_BY-> CBSSports.com
- Day of Reckoning -- featuring Anthony Joshua, Deontay Wilder: Predictions, undercard, odds, picks, start time -PUBLISHED_BY-> CBSSports.com

Entities: CBSSports.com, 2023 NFL Week 11 injury roundup: Rams WR Cooper Kupp, Seahawks RB Kenneth Walker III among stars to exit, Cooper Kupp, Talanoa Hufanga,
ACL tear, knee injury, Rams, Seahawks, Kenneth Walker III, San Francisco, Kyle Shanahan, Buccaneers, 49ers
</item 6>

<item 7>
2023 NFL Week 11 injury roundup: Rams WR Cooper Kupp, Seahawks RB Kenneth Walker III among stars to exit (CBSSports.com, 2023-11-19T23:51:49+00:00)
I agree to receive the "Pick Six Newsletter" and marketing communications, updates, special offers (including partner offers), and other information from CBS
Sports and the Paramount family of companies. By pressing sign up, I confirm that I have read and agree to the Terms of Use and acknowledge Paramount's Privacy
Policy See All Newsletters Please check the opt-in box to acknowledge that you would like to subscribe. Thanks for signing up! Keep an eye on your inbox. Sorry!
There was an error processing your subscription.

WR

Tyreek Hill (Dolphins) briefly left Miami's game against the Raiders while nursing a hand injury and was ruled questionable to return, but he took the field
again soon afterward, securing seven catches for 80 yards and a score before halftime. Hill has left multiple games this year, only to return in all of them.

Cooper Kupp (Rams) had just a single catch against the Seahawks before suffering an ankle injury while blocking.

Graph facts:
- Christmas Day preview: 49ers, Ravens square off in potential Super Bowl sneak peek -MENTIONS-> Raiders
- Kelce -MENTIONS-> Raiders
- NFL Fantasy Football Week 5 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game -MENTIONS-> Raiders
- Raiders vs. Lions live score, updates, highlights from NFL 'Monday Night Football' game -MENTIONS-> Raiders
- Fantasy Football WR PPR Rankings Week 13: Who to start, best sleepers at wide receiver -MENTIONS-> Raiders

Entities: Raiders, hand injury, game against the Seahawks, CBSSports.com, 2023 NFL Week 11 injury roundup: Rams WR Cooper Kupp, Seahawks RB Kenneth Walker III
among stars to exit, Cooper Kupp, Miami's game against the Raiders, Dolphins, ankle injury, NFL, Rams, Seahawks, Tyreek Hill
</item 7>

<item 8>
2023 NFL Week 11 injury roundup: Rams WR Cooper Kupp, Seahawks RB Kenneth Walker III among stars to exit (CBSSports.com, 2023-11-19T23:51:49+00:00)
He was seen walking on the sidelines shortly afterward but ruled doubtful to return. Rookie Zach Charbonnet, who entered Week 11 with just over 200 rushing
yards on the year, replaced Walker as Seattle's top ball carrier. Following the game, Seahawks head coach Pete Carroll told reporters Walker suffered an
"oblique strain that was legit" and could miss time.

D'Onta Foreman (Bears) remained a headliner of Chicago's backfield against the Lions, even with Khalil Herbert back from injured reserve, but managed just 14
yards on the ground before suffering an ankle injury. He walked off with a limp and attempted to run on the sidelines but did not return to the field after the
Bears ruled him questionable.

Pick Six Newsletter Crafted By The Best NFL Experts Get the day's big stories + fun stuff you love like mock drafts, picks and power rankings.

Graph facts:
- Fantasy Football Week 13 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game -PUBLISHED_BY-> CBSSports.com
- Taylor Swift-Travis Kelce timeline: Everything to know about rumored romance between pop star, Chiefs TE -PUBLISHED_BY-> CBSSports.com
- 2023 Kentucky online sports betting sites: Best legal sportsbooks, promos, bonuses, mobile apps, how to bet -PUBLISHED_BY-> CBSSports.com
- NFL Fantasy Football Week 5 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game -PUBLISHED_BY-> CBSSports.com
- Day of Reckoning -- featuring Anthony Joshua, Deontay Wilder: Predictions, undercard, odds, picks, start time -PUBLISHED_BY-> CBSSports.com

Entities: CBSSports.com, 2023 NFL Week 11 injury roundup: Rams WR Cooper Kupp, Seahawks RB Kenneth Walker III among stars to exit, Pete Carroll, oblique strain,
Pick Six Newsletter, Cooper Kupp, ankle injury, Lions, Rams, Seahawks, Kenneth Walker III, D'Onta Foreman, Zach Charbonnet, Bears, Khalil Herbert
</item 8>

<item 9>
Fantasy Football Week 13 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-11-30T22:10:58+00:00)
Sit Him (Lineup Decisions) Projections powered by Sportsline Cooper Kupp WR LAR L.A. Rams • #10

Age: 30 • Experience: 7 yrs. Weekly Breakdown MATCHUP vs CLE LAR -3.5 O/U 40 OPP VS WR 14th PROJ PTS 13.8 WR RNK 25th YTD Stats REC 27 TAR 51 REYDS 393 TD 1
FPTS/G 10.6 You're never supposed to sit your studs, but what if your stud is playing through something that made him look like a dud last week? That was Kupp,
who in Week 12 ran a ton of short routes and couldn't separate and didn't have much sudden movement. If he's going to come through this week it'll likely come
on a ton of short targets and/or some targets inside the 10-yard line.

Graph facts:
- Fantasy Football Week 13 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game -MENTIONS-> Brock Purdy
- Fantasy Football Week 13 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game -MENTIONS-> Mark Andrews
- Fantasy Football Week 13 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game -MENTIONS-> Alex Anzalone
- Fantasy Football Week 13 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game -MENTIONS-> NFL
- Fantasy Football Week 13 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game -MENTIONS-> Jordan Love

Entities: Fantasy Football Week 13 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game, L.A. Rams, Lineup Decisions, CBSSports.com, Cooper
Kupp, CLE, fantasy football, SportsLine
</item 9>

<item 10>
2023 NFL Week 11 injury roundup: Rams WR Cooper Kupp, Seahawks RB Kenneth Walker III among stars to exit (CBSSports.com, 2023-11-19T23:51:49+00:00)
The former All-Pro, who battled a lingering hamstring injury earlier this year, received attention on the sidelines after limping off, then left the game for
further evaluation in the locker room. He was ruled questionable to return. NFL Media reported on Monday that Kupp suffered a low ankle sprain and will be
evaluated going forward.

Darius Slayton (Giants) had a big impact against the Commanders, hauling in a deep score from Tommy DeVito, before exiting with an arm injury. He was ruled out
for the remainder of the contest halfway through the third quarter.

EDGE

Joey Bosa (Chargers) appeared to injure his right leg or foot while held in a block during a first-quarter pass-rushing attempt against the Packers. He limped
to the sidelines and was carted to the locker room, then ruled questionable to return.

Graph facts:
- Christmas Day preview: 49ers, Ravens square off in potential Super Bowl sneak peek -MENTIONS-> NFL
- NFL DFS picks Week 15: Best sleepers, value players for FanDuel, DraftKings, SuperDraft lineups -MENTIONS-> Tommy DeVito
- Fantasy Football QB Rankings Week 14: Who to start, best sleepers at quarterback -MENTIONS-> Tommy DeVito
- Christmas Day preview: 49ers, Ravens square off in potential Super Bowl sneak peek -MENTIONS-> Tommy DeVito
- Brock Purdy -MENTIONS-> NFL

Entities: Tommy DeVito, NFL, Packers, CBSSports.com, 2023 NFL Week 11 injury roundup: Rams WR Cooper Kupp, Seahawks RB Kenneth Walker III among stars to exit,
Cooper Kupp, Joey Bosa, NFL Media, lingering hamstring injury, low ankle sprain, arm injury, right leg or foot injury, Darius Slayton, Rams, Seahawks
</item 10>
=== END ITEM id=48bacf7b63472651 ===
