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

=== ITEM id=548306f2e0a844a3 ===
Gold sentences and facts:
Sentence 0: Using Caroline Ellison, his on-again off-again girlfriend and the CEO of Alameda, as a front, Bankman-Fried had “secret access” to customer money—both cash and crypto—the government claimed.
  Fact 0.0: Caroline Ellison was Bankman-Fried's on-again off-again girlfriend.
  Fact 0.1: Caroline Ellison was the CEO of Alameda.
  Fact 0.2: The government claimed Bankman-Fried used Caroline Ellison as a front.
  Fact 0.3: The government claimed Bankman-Fried had 'secret access' to customer money, both cash and crypto.
Sentence 1: The second week of the trial’s standout testimony came from Alameda Research’s former CEO Caroline Ellison, who claimed she took $14 billion from customers to repay debts to lenders, using them as a line of credit under the instruction of SBF.
  Fact 1.0: The second week of the trial's standout testimony came from Caroline Ellison.
  Fact 1.1: Caroline Ellison is Alameda Research's former CEO.
  Fact 1.2: Caroline Ellison claimed she took $14 billion from customers.
  Fact 1.3: Caroline Ellison claimed she took the $14 billion to repay debts to lenders.
  Fact 1.4: Caroline Ellison claimed she used customers as a line of credit.
  Fact 1.5: Caroline Ellison claimed she acted under the instruction of SBF.

Retrieved items:
<item 1>
Sam Bankman-Fried was a terrible boyfriend (The Verge, 2023-10-10T23:50:21+00:00)
Alameda got its war chest — 60 percent to 70 percent of the initial supply — for free, while seed investors got FTT at 10 cents a coin, and FTT first listed at
$1 a coin. Bankman-Fried felt that $1 per coin was psychologically important, Ellison said, and he directed her to buy up FTT using Alameda if its price fell
below a dollar.

FTT was one of several “Sam coins,” a nickname for tokens that Bankman-Fried was heavily involved in and owned a lot of, either personally or through Alameda.
Those coins were almost certainly worth less than the value displayed on the balance sheet because trying to sell them all at once would crater the prices.
Bankman-Fried directed her to put those coins on the balance sheets Alameda showed to lenders, even though she felt it was “somewhat misleading.”

Alameda was also getting loans from outside lenders, such as Genesis, because when FTX started, there weren’t a lot of customer funds to borrow, Ellison
testified.

Graph facts:
- SBF’s trial starts soon, but how did he — and FTX — get here? -MENTIONS-> Alameda
- SBF, riding high on FTX, reportedly offered $5B to Trump to cancel President re-run -MENTIONS-> Alameda
- Is Sam Bankman-Fried a bad ‘man’ or a good ‘boy’? Lawyers swap opening statements before first witnesses take the stand -MENTIONS-> Alameda
- The jury finally hears from Sam Bankman-Fried -MENTIONS-> Alameda
- SBF Trial: The latest updates from the FTX collapse’s courtroom drama -MENTIONS-> Alameda

Entities: Alameda, Sam Bankman-Fried was a terrible boyfriend, FTT, Sam coins, Ellison, FTX, Sam Bankman-Fried, Genesis, The Verge
</item 1>

<item 2>
Graph query row:
outlet: TechCrunch
article: SBF Trial: The latest updates from the FTX collapse’s courtroom drama
page: SBF Trial: The latest updates from the FTX collapse’s courtroom drama (TechCrunch, 2023-10-06T17:22:02+00:00)
passage: SBF Trial: The latest updates from the FTX collapse’s courtroom drama (TechCrunch, 2023-10-06T17:22:02+00:00)
SBF trial: Everything to know from the FTX courtroom ahead of his testimony Sam Bankman-Fried is in court for alleged fraud and money laundering while leading
crypto exchange FTX and sister company Alameda. Here’s our latest coverage as the case nears its end.

The trial of Bankman-Fried, the former CEO and co-founder of collapsed crypto exchange FTX, is entering its next chapter as proceedings resume Thursday. As
confirmed during a teleconference Wednesday, SBF will be taking the stand, along with a handful of other witnesses as the onus shifts toward the defense after
the prosecution brought its case to its conclusion.

As former federal prosecutor Josh Naftalis told TechCrunch, SBF taking to the stand in his own trial is a “Hail Mary” ahead of a verdict and potential
sentencing for seven counts tied to fraud and money laundering.

“Once it goes to cross[-examination], he doesn’t get to say, ‘I’m done,’” Naftalis said.
</item 2>

<item 3>
Sam Bankman-Fried was a terrible boyfriend (The Verge, 2023-10-10T23:50:21+00:00)
That gave Ellison an unusual view of his character. “He was very ambitious,” she said. Besides telling her about his presidential chances, he also told her that
if there was a coin flip where tails destroyed the world and heads made the world twice as good, he’d flip the coin. He called this being “risk-neutral,” which
seems like a fancy way of saying he was a gambling addict.

She was named co-CEO of Alameda with Sam Trabucco in 2021, while she and Bankman-Fried were broken up, and CEO in 2022. The goal, Bankman-Fried told her, was to
“optically” separate Alameda Research and FTX. “The whole time we were dating, he was my boss at work,” she said. They broke up because she wanted more from the
relationship; Bankman-Fried was distant and not paying enough attention to her.

Graph facts:
- SBF’s trial starts soon, but how did he — and FTX — get here? -MENTIONS-> Alameda Research
- SBF, riding high on FTX, reportedly offered $5B to Trump to cancel President re-run -MENTIONS-> Alameda Research
- Is Sam Bankman-Fried a bad ‘man’ or a good ‘boy’? Lawyers swap opening statements before first witnesses take the stand -MENTIONS-> Alameda Research
- The jury finally hears from Sam Bankman-Fried -MENTIONS-> Alameda Research
- SBF Trial: The latest updates from the FTX collapse’s courtroom drama -MENTIONS-> Alameda Research

Entities: Alameda Research, Sam Trabucco, Sam Bankman-Fried was a terrible boyfriend, Ellison, FTX, Sam Bankman-Fried, The Verge
</item 3>

<item 4>
Sam Bankman-Fried was a terrible boyfriend (The Verge, 2023-10-10T23:50:21+00:00)
It was that Ellison’s calculations assumed Alameda could borrow $1.8 billion in normal dollars and $1.5 billion in crypto from FTX. The spreadsheet makes this
clear with a row labeled “FTX borrows,” which Ellison said were customer funds.

Meanwhile, echoing Bankman-Fried and Ellison’s romantic relationship, FTX was keeping cozy private ties with Alameda yet publicly holding it at arm’s length. On
January 14th, 2022, Bankman-Fried tweeted, “We’re launching a $2b venture fund: FTX Ventures!” Those funds came from Alameda, Ellison testified. But
Bankman-Fried didn’t want to go public with the source of the funds. He said he thought Alameda’s brand was less good, and he didn’t want his name associated
with it. Alameda also bought Robinhood shares for Bankman-Fried, who moved them to a vehicle called “Emergent Fidelity Technologies” to avoid association with
Alameda.

Graph facts:
- SBF Trial: The latest updates from the FTX collapse’s courtroom drama -MENTIONS-> crypto
- SBF, riding high on FTX, reportedly offered $5B to Trump to cancel President re-run -MENTIONS-> crypto
- Is Sam Bankman-Fried a bad ‘man’ or a good ‘boy’? Lawyers swap opening statements before first witnesses take the stand -MENTIONS-> crypto
- SBF’s trial starts soon, but how did he — and FTX — get here? -MENTIONS-> crypto
- Is Sam Bankman-Fried’s defense even trying to win? -MENTIONS-> crypto

Entities: crypto, Alameda, Sam Bankman-Fried was a terrible boyfriend, FTX Ventures, Emergent Fidelity Technologies, Ellison, customer funds, FTX, Sam
Bankman-Fried, Robinhood, The Verge
</item 4>

<item 5>
Sam Bankman-Fried was a terrible boyfriend (The Verge, 2023-10-10T23:50:21+00:00)
(Fraud, conspiracy to commit fraud, and money laundering, in case you were wondering.) “We ultimately took about $14 billion, some of which we were not able to
pay back.” She tilted her head down to answer the questions, then lifted her head when she’d finished her answer.

Bankman-Fried called this being “risk-neutral,” which seems like a fancy way of saying he was a gambling addict

In Ellison’s telling, Alameda was troubled from her earliest time there in 2018. “Shortly after I started, I learned the company was in worse shape than I
realized,” Ellison said. Alameda had initially been funded with loans “from acquaintances,” she said, and those loans were recalled a few weeks after she
arrived. (There was a staff revolt within Alameda Research over lost millions and general financial chaos, according to Michael Lewis’ Going Infinite.) Ellison
asked Bankman-Fried why he hadn’t shared the company’s shaky circumstances in the job offer. “He hadn’t known how to tell me,” she said.

Graph facts:
- Michael Lewis -PARTICIPATED_IN-> Michael Lewis interview
- SBF, riding high on FTX, reportedly offered $5B to Trump to cancel President re-run -MENTIONS-> Michael Lewis
- SBF, riding high on FTX, reportedly offered $5B to Trump to cancel President re-run -MENTIONS-> Going Infinite
- Michael Lewis -SAID-> FTX
- SBF Trial: The latest updates from the FTX collapse’s courtroom drama -MENTIONS-> Alameda

Entities: Going Infinite, Michael Lewis, Alameda, Sam Bankman-Fried was a terrible boyfriend, Conspiracy to commit fraud, Risk-neutral, Gambling addict, Staff
revolt within Alameda Research over lost millions and general financial chaos, money laundering, Ellison, Sam Bankman-Fried, fraud, The Verge
</item 5>

<item 6>
Sam Bankman-Fried was a terrible boyfriend (The Verge, 2023-10-10T23:50:21+00:00)
That was the basis of the worst of her testimony — and the spreadsheet from hell.

Ellison said she’d prepared the spreadsheet at Bankman-Fried’s request in the fall of 2021 and shared it with him. The point was risk analysis around paying
back Alameda’s loans if they were abruptly recalled by Genesis, their lender. Bankman-Fried wanted to use $3 billion for venture investments, so Ellison was
ballparking what that would do to Alameda’s risk. In the as-is scenario, if things went south, she figured there was a 30 percent chance they wouldn’t be able
to meet the loan recalls. If Bankman-Fried used $3 billion to make investments, there was a 100 percent chance they couldn’t meet the recalls, even with FTX
customer funds.

Echoing Bankman-Fried and Ellison’s romantic relationship, FTX was keeping cozy private ties with Alameda yet publicly holding it at arm’s length

The problem here wasn’t really the math, which seemed pretty arbitrary.

Graph facts:
- SBF’s trial starts soon, but how did he — and FTX — get here? -MENTIONS-> Alameda
- SBF, riding high on FTX, reportedly offered $5B to Trump to cancel President re-run -MENTIONS-> Alameda
- Is Sam Bankman-Fried a bad ‘man’ or a good ‘boy’? Lawyers swap opening statements before first witnesses take the stand -MENTIONS-> Alameda
- The jury finally hears from Sam Bankman-Fried -MENTIONS-> Alameda
- SBF Trial: The latest updates from the FTX collapse’s courtroom drama -MENTIONS-> Alameda

Entities: Alameda, Sam Bankman-Fried was a terrible boyfriend, Ellison, FTX, Sam Bankman-Fried, Genesis, The Verge
</item 6>

<item 7>
Sam Bankman-Fried was a terrible boyfriend (The Verge, 2023-10-10T23:50:21+00:00)
Alameda’s credit line — which was taken in increments of $100,000 to $10 million at a time — was used for trading. Using the effectively unlimited funds
“allowed us to make profitable trades we couldn’t have made otherwise,” Ellison testified.

Customer funds were also used when Bankman-Fried bought back FTX shares from Binance, an early investor, in the summer of 2021. Bankman-Fried told Ellison it
was “really important,” otherwise “Binance would do things to mess with FTX.” Ellison says she told him Alameda didn’t have the money. So Bankman-Fried took $1
billion of FTX customer funds to buy out Binance, the first time Ellison recalled an amount that large. It was Bankman-Fried’s decision, she said, as he was the
CEO of FTX.

FTT was one of several “Sam coins”

There was also the FTT token, which was created by Bankman-Fried and Wang.

Graph facts:
- SBF’s trial starts soon, but how did he — and FTX — get here? -MENTIONS-> Alameda
- SBF, riding high on FTX, reportedly offered $5B to Trump to cancel President re-run -MENTIONS-> Alameda
- Is Sam Bankman-Fried a bad ‘man’ or a good ‘boy’? Lawyers swap opening statements before first witnesses take the stand -MENTIONS-> Alameda
- The jury finally hears from Sam Bankman-Fried -MENTIONS-> Alameda
- SBF Trial: The latest updates from the FTX collapse’s courtroom drama -MENTIONS-> Alameda

Entities: Alameda, Sam Bankman-Fried was a terrible boyfriend, FTT, Ellison, FTX, Wang, Sam Bankman-Fried, Binance, The Verge
</item 7>

<item 8>
Sam Bankman-Fried was a terrible boyfriend (The Verge, 2023-10-10T23:50:21+00:00)
Using the effectively unlimited funds “allowed us to make profitable trades we couldn’t have made otherwise”

Bankman-Fried didn’t grant Ellison equity, even though she asked; he told her it would be too complicated. Instead, she got a $200,000 salary, even as CEO, and
bonuses twice a year, which ranged from $100,000 to $20 million.

Initially, Alameda and FTX were “very integrated,” Ellison said. They were run by the same team, from the same office. And when Alameda was scrounging for
funds, Bankman-Fried told Ellison that FTX would be a good source of capital. The $65 billion line of credit Alameda Research had meant that it did not have to
post collateral. There was no contract and no written terms, she testified. It also wasn’t visible to FTX’s auditors — she’d raised the question with
Bankman-Fried, and he told her not to worry about it.

Graph facts:
- SBF’s trial starts soon, but how did he — and FTX — get here? -MENTIONS-> Alameda Research
- SBF, riding high on FTX, reportedly offered $5B to Trump to cancel President re-run -MENTIONS-> Alameda Research
- Is Sam Bankman-Fried a bad ‘man’ or a good ‘boy’? Lawyers swap opening statements before first witnesses take the stand -MENTIONS-> Alameda Research
- The jury finally hears from Sam Bankman-Fried -MENTIONS-> Alameda Research
- SBF Trial: The latest updates from the FTX collapse’s courtroom drama -MENTIONS-> Alameda Research

Entities: Alameda Research, Sam Bankman-Fried was a terrible boyfriend, $65 billion line of credit, Ellison, FTX, Sam Bankman-Fried, The Verge
</item 8>

<item 9>
How is it still getting worse for Sam Bankman-Fried? (The Verge, 2023-10-12T23:08:14+00:00)
The testimony set up an argument for the prosecution to bring in those tapes. The judge ruled in favor of the jury hearing the recordings and we briefly
recessed.

That was when Bankman-Fried’s mother approached the defense lawyers.

The late afternoon was short and snappy. Christian Drappi, a former Alameda software engineer who looked like a handsome funeral director in a black suit and
tie, testified briefly to set up the tape. When Changpeng Zhao, the CEO of Binance, announced on Twitter that he intended to acquire FTX, Ellison confessed the
theft of customer funds to him and a few other employees, Drappi said. The all-hands took place the following day, and was secretly recorded by an employee
who’d joined Alameda three days before.

In the recording, Ellison did indeed confess to stealing customer funds with Bankman-Fried’s approval. Drappi said he resigned less than 24 hours after the
meeting. The recording was later sent to Drappi, who sent it to the government.

Graph facts:
- SBF’s trial starts soon, but how did he — and FTX — get here? -MENTIONS-> Alameda
- SBF, riding high on FTX, reportedly offered $5B to Trump to cancel President re-run -MENTIONS-> Alameda
- Is Sam Bankman-Fried a bad ‘man’ or a good ‘boy’? Lawyers swap opening statements before first witnesses take the stand -MENTIONS-> Alameda
- The jury finally hears from Sam Bankman-Fried -MENTIONS-> Alameda
- SBF Trial: The latest updates from the FTX collapse’s courtroom drama -MENTIONS-> Alameda

Entities: Alameda, Ellison, FTX, the government, Sam Bankman-Fried, Christian Drappi, all-hands, recording, theft of customer funds, How is it still getting
worse for Sam Bankman-Fried?, Binance, Changpeng Zhao, Twitter, The Verge
</item 9>

<item 10>
Sam Bankman-Fried didn’t ask where the $8 billion went (The Verge, 2023-10-31T19:52:07+00:00)
Instead, Bankman-Fried directed alleged co-conspirator Caroline Ellison to repay third-party loans and went on making investments.

Bankman-Fried said that he found out about the fiat@ftx account — that’s the one that tracked how much Alameda owed FTX — in June 2022 while his senior staff
was focused on fixing the software bug. He did not, however, discover what the account was for until October. I know this sounds unbelievable, but this is his
actual testimony.

His employees told him “they were busy and I should stop asking questions because it was distracting.”

Simple questions nailed Bankman-Fried to the wall. Had he given his employees at Alameda the direction not to spend FTX customer deposits? Had he put in any
policies to prevent Alameda employees from spending FTX customer money? Did he put in place any measures at Alameda to protect the FTX customer money? No, no,
and no. Oh, but he was testifying to Congress about keeping customer funds safe.

Graph facts:
- SBF Trial: The latest updates from the FTX collapse’s courtroom drama -MENTIONS-> Caroline Ellison
- Is Sam Bankman-Fried a bad ‘man’ or a good ‘boy’? Lawyers swap opening statements before first witnesses take the stand -MENTIONS-> Caroline Ellison
- The jury finally hears from Sam Bankman-Fried -MENTIONS-> Caroline Ellison
- SBF’s trial starts soon, but how did he — and FTX — get here? -MENTIONS-> Caroline Ellison
- Sam Bankman-Fried was a terrible boyfriend -MENTIONS-> Caroline Ellison

Entities: Caroline Ellison, Alameda, fiat@ftx account, testifying to Congress, Sam Bankman-Fried didn’t ask where the $8 billion went, conspiracy, customer
funds, Congress, FTX, Sam Bankman-Fried, The Verge
</item 10>
=== END ITEM id=548306f2e0a844a3 ===

=== ITEM id=e509debca4526018 ===
Gold sentences and facts:
Sentence 0: With this sale, they offer some of the best value in their class from one of the top names in the business.
  Fact 0.0: The seller offers some of the best value in its class.
  Fact 0.1: The seller offers this value from one of the top names in the business.
Sentence 1: “Each actor will have to be responsible to ensure that the market remains reasonable amid rates that are relatively low,” CMA CGM Chief Financial Officer Ramon Fernandez told reporters Friday.
  Fact 1.0: Ramon Fernandez is Chief Financial Officer of CMA CGM.
  Fact 1.1: Ramon Fernandez told reporters on Friday.
  Fact 1.2: Ramon Fernandez said each actor will have to be responsible to ensure that the market remains reasonable amid rates that are relatively low.

Retrieved items:
<item 1>
These Are Our Favorite Walmart Cyber Monday Deals (Wired, 2023-11-27T14:22:26+00:00)
As Sony’s priciest wireless headphones, the WH-1000XM5 had a lot to live up to following the remarkably popular WH-1000XM4. Luckily they deliver the goods with
upgraded noise canceling, a posh new design, and a more refined sound signature that adds up to a great package – especially on sale.

Bose’s aptly named QuietComfort Ultra (9/10, WIRED Recommends) arrived in October with only one notable problem: they’re very pricey. Luckily they’ve already
gotten a discount, which makes it easier to land their class-leading noise canceling, great sound, and luxuriously comfy design that’s loaded with modern
features.

Photograph: Sony

Sony's WH-1000XM4 headphones (9/10, WIRED Recommends) have been a stalwart pair of wireless headphones for years now, and there's no secret to their success.
Offering excellent noise canceling, warm and detailed sound, and a ton of modern features in a comfy design, they're among the best travel headphones you can
buy.

Graph facts:
- Video games in 2023: Acquisitions, layoffs, unions -MENTIONS-> Sony
- Young Taylor Swift fans go viral for struggling to open the star's CD: 'Making me laugh' -MENTIONS-> Sony
- Steam Deck OLED review: It's just better -MENTIONS-> Sony
- Michelle Jubelirer on Ice Spice, The Beatles, and reinventing Capitol Music Group: ‘I don’t think any label group has been able to turn things around as
quickly as my team and I have done in less than two years.’ -MENTIONS-> Sony
- A comprehensive list of 2023 tech layoffs -MENTIONS-> Sony

Entities: Sony, WIRED, Bose, These Are Our Favorite Walmart Cyber Monday Deals, WH-1000XM5, WH-1000XM4, QuietComfort Ultra
</item 1>

<item 2>
The best Black Friday headphone deals from Sony, Apple, Bose, Beats and more (Engadget, 2023-11-24T14:30:56+00:00)
If you prefer this kind of bulkier but easier-to-remove shape, this is an excellent value.

Anker Soundcore Space Q45

If you’re after a wallet-friendly pair of noise-canceling over-ear headphones, Anker’s Soundcore Space Q45 is worth a look at its Black Friday price of $100.
For that amount, this pair’s better-than-average ANC, softly padded design and multi-device pairing make for a strong value. Its V-shaped sound profile isn’t
especially detailed, but it can work well enough for pop music and it’s customizable through Anker’s app. Call quality is mediocre, however. This $50 discount
matches the lowest price we’ve tracked, and it's available at both Amazon and Anker's online store; just use the on-page coupon for the latter.

Audio-Technica ATH-M20xBT

The Audio-Technica ATH-M20xBT is a modest $10 off for Black Friday, bringing the wireless headphones down to $69.

Graph facts:
- Video games in 2023: Acquisitions, layoffs, unions -MENTIONS-> Sony
- Young Taylor Swift fans go viral for struggling to open the star's CD: 'Making me laugh' -MENTIONS-> Sony
- Steam Deck OLED review: It's just better -MENTIONS-> Sony
- Michelle Jubelirer on Ice Spice, The Beatles, and reinventing Capitol Music Group: ‘I don’t think any label group has been able to turn things around as
quickly as my team and I have done in less than two years.’ -MENTIONS-> Sony
- A comprehensive list of 2023 tech layoffs -MENTIONS-> Sony

Entities: Sony, Apple, Bose, Beats, The best Black Friday headphone deals from Sony, Apple, Bose, Beats and more, Soundcore Space Q45, Audio-Technica,
ATH-M20xBT, Black Friday, Anker, Amazon, Engadget
</item 2>

<item 3>
The best Black Friday headphone deals from Sony, Apple, Bose, Beats and more (Engadget, 2023-11-24T14:30:56+00:00)
That's another $50 discount and an all-time low. We gave these in-ears a review score of 88 in September and recently named them the “best for noise
cancellation” pick in our wireless earbuds buying guide. As that title suggests, the main reason to consider this pair over the Sony WF-1000XM5 is its ANC
performance, which is the strongest we’ve tested from a true wireless model. We also enjoy Bose’s spatial audio tech. But the earpieces are on the larger side,
the call quality and battery life (at six-ish hours) are just OK and there’s no support for multi-device pairing.

Beats Fit Pro

The Beats Fit Pro is back down to $159, which isn’t an all-time low but is still roughly $20 below its average street price. This is the “best for workouts”
pick in our wireless earbuds buying guide, as it packs many of the iOS-friendly conveniences of Apple’s AirPods in a sportier and more stable design.

Graph facts:
- Video games in 2023: Acquisitions, layoffs, unions -MENTIONS-> Sony
- Young Taylor Swift fans go viral for struggling to open the star's CD: 'Making me laugh' -MENTIONS-> Sony
- Steam Deck OLED review: It's just better -MENTIONS-> Sony
- Michelle Jubelirer on Ice Spice, The Beatles, and reinventing Capitol Music Group: ‘I don’t think any label group has been able to turn things around as
quickly as my team and I have done in less than two years.’ -MENTIONS-> Sony
- A comprehensive list of 2023 tech layoffs -MENTIONS-> Sony

Entities: Sony, Apple, AirPods, Bose, Beats, Beats Fit Pro, Sony WF-1000XM5, The best Black Friday headphone deals from Sony, Apple, Bose, Beats and more,
wireless earbuds buying guide, Black Friday, Engadget
</item 3>

<item 4>
The best Black Friday headphone deals from Sony, Apple, Bose, Beats and more (Engadget, 2023-11-24T14:30:56+00:00)
Engadget Editor Jon Turi found this pair to deliver more low-end depth than most of its peers, however. It can also last up to 10 hours per charge.

Your Black Friday Shopping Guide: See all of Yahoo’s Black Friday coverage, here. Follow Engadget for Black Friday tech deals. Learn about Black Friday trends
on In The Know. Hear from Autoblog’s experts on the best Black Friday deals for your car, garage, and home, and find Black Friday sales to shop on AOL,
handpicked just for you.

Graph facts:
- Video games in 2023: Acquisitions, layoffs, unions -MENTIONS-> Sony
- Young Taylor Swift fans go viral for struggling to open the star's CD: 'Making me laugh' -MENTIONS-> Sony
- Steam Deck OLED review: It's just better -MENTIONS-> Sony
- Michelle Jubelirer on Ice Spice, The Beatles, and reinventing Capitol Music Group: ‘I don’t think any label group has been able to turn things around as
quickly as my team and I have done in less than two years.’ -MENTIONS-> Sony
- A comprehensive list of 2023 tech layoffs -MENTIONS-> Sony

Entities: Sony, Apple, In The Know, Autoblog, AOL, Yahoo, Black Friday deals, Bose, Beats, The best Black Friday headphone deals from Sony, Apple, Bose, Beats
and more, Jon Turi, Tech deals, Black Friday trends, headphones, Black Friday
</item 4>

<item 5>
These Are Our Favorite Walmart Cyber Monday Deals (Wired, 2023-11-27T14:22:26+00:00)
Photograph: Technics

The Technics EAH-AZ80 (9/10, WIRED Recommends) are some of our favorite top-shelf earbuds thanks to their fantastic sound alongside great features and noise
canceling, and their ability to connect to up to three devices at once. This is one of the few sales we've seen all year, which makes their very high asking
price a lot more palatable.

Soundcore's Life Q30 are already impressive at full price thanks to noise canceling that punches well above its class, so this deal makes them something of a
steal. They add to the package with accessible and adjustable sound quality, good features, and great battery life.

Jump to a Topic: PC and Gaming Deals, Home and Kitchen Deals, TV Deals, Watch and Fitness Tracker Deals, Headphone and Earbud Deals, Smart Home Deals, Speaker
Deals

Smart Home Deals

Want to smarten up your home? See our guides to setting up your smart home, the best smart lighting, and the best mesh routers to get started.

Graph facts:
- The best Apple Black Friday deals on iPads, AirPods, Apple Watches and more -MENTIONS-> Walmart
- The best Prime Day smart home deals -MENTIONS-> Walmart
- The 53 best Black Friday deals we could find at Amazon, Walmart, Target and more -MENTIONS-> Walmart
- Reliance Industries: With the stock having underperformed Nifty 50 in last three years, is it time to buy? -MENTIONS-> Walmart
- The best Apple Cyber Monday deals of 2023: Save on AirPods, iPads, Apple Watches and more -MENTIONS-> Walmart

Entities: Walmart, PC and Gaming Deals, Home and Kitchen Deals, TV Deals, Watch and Fitness Tracker Deals, Headphone and Earbud Deals, Smart Home Deals, Speaker
Deals, WIRED, These Are Our Favorite Walmart Cyber Monday Deals, Technics, Technics EAH-AZ80, Soundcore, Life Q30, mesh routers
</item 5>

<item 6>
These Are Our Favorite Walmart Cyber Monday Deals (Wired, 2023-11-27T14:22:26+00:00)
Sony's WF-1000XM5 (7/10, WIRED Review) source multiple generations of great sound, top features, and excellent noise canceling, all wrapped into their most
compact presentation yet. They're not cheap, so this discount is key if you're looking to be an earlier adopter of Sony's latest flagship buds.

The only issues we had with Sennheiser's HD 660S2 (7/10, WIRED Review) bass-enriched audiophile beauties were their high price and stiff clamping force. At this
price, we can live with a tighter fit, especially since the payoff is gloriously clear and richly detailed sound.

Sony's working-class noise cancelers offer solid performance. clear sound, and impressive battery life at a much more reasonable price than the brand's top
offerings. With this sale, they offer some of the best value in their class from one of the top names in the business.

Graph facts:
- Video games in 2023: Acquisitions, layoffs, unions -MENTIONS-> Sony
- Young Taylor Swift fans go viral for struggling to open the star's CD: 'Making me laugh' -MENTIONS-> Sony
- Steam Deck OLED review: It's just better -MENTIONS-> Sony
- Michelle Jubelirer on Ice Spice, The Beatles, and reinventing Capitol Music Group: ‘I don’t think any label group has been able to turn things around as
quickly as my team and I have done in less than two years.’ -MENTIONS-> Sony
- A comprehensive list of 2023 tech layoffs -MENTIONS-> Sony

Entities: Sony, WF-1000XM5, WIRED, These Are Our Favorite Walmart Cyber Monday Deals, HD 660S2, Sony's working-class noise cancelers, Sennheiser
</item 6>

<item 7>
78 Absolute Best October Prime Day Deals (2023) (Wired, 2023-10-10T14:23:00+00:00)
Anker's Liberty 4 NC earbuds (8/10, WIRED Recommends) are another feature-packed pair of buds with great noise canceling, and are on sale for $80 ($20 off).

The Beats Studio Pro (7/10, WIRED Review) were released just a couple of months ago. We wish the battery life was longer and that the controls and EQ options
were more robust. But this price makes them more worthwhile, and they do have excellent noise canceling.

Photograph: Bose

These noise-cancelling headphones are our favorite for the office. With advanced signal processing and four microphones built in, these will limit any sound
around you, making them great for focusing on work in a busy environment and for Zoom calls. These often fluctuate in price, but we think it's a great deal.

We like Sony's standard LinkBuds (8/10, WIRED Recommends) for the open-ear design that allows you to hear what's going on around you.

Graph facts:
- Video games in 2023: Acquisitions, layoffs, unions -MENTIONS-> Sony
- Young Taylor Swift fans go viral for struggling to open the star's CD: 'Making me laugh' -MENTIONS-> Sony
- Steam Deck OLED review: It's just better -MENTIONS-> Sony
- Michelle Jubelirer on Ice Spice, The Beatles, and reinventing Capitol Music Group: ‘I don’t think any label group has been able to turn things around as
quickly as my team and I have done in less than two years.’ -MENTIONS-> Sony
- A comprehensive list of 2023 tech layoffs -MENTIONS-> Sony

Entities: Sony, WIRED, Bose, Beats, Beats Studio Pro, LinkBuds, Liberty 4 NC earbuds, 78 Absolute Best October Prime Day Deals (2023), Anker
</item 7>

<item 8>
Graph query row:
article: Is Sam Bankman-Fried a bad ‘man’ or a good ‘boy’? Lawyers swap opening statements before first witnesses take the stand
publisher: Fortune
subject: Matt Huang
page: Is Sam Bankman-Fried a bad ‘man’ or a good ‘boy’? Lawyers swap opening statements before first witnesses take the stand (Fortune,
2023-10-04T23:32:53+00:00)
passage: Is Sam Bankman-Fried a bad ‘man’ or a good ‘boy’? Lawyers swap opening statements before first witnesses take the stand (Fortune,
2023-10-04T23:32:53+00:00)
Who is Sam Bankman-Fried, the former CEO of crypto exchange FTX? Is he a liar and a fraud? Or did he act in good faith, and like many a startup founder, fly too
close to the sun?

That answer ultimately lies with the jury, which was selected on Wednesday morning before lawyers for the government and then Bankman-Fried swapped two very
different stories of the former crypto mogul’s sudden rise and almost instantaneous fall.

Here’s what happened on the second day of the trial, which featured pointed allegations, a friend from MIT, and an audience replete with big names, including
Bankman-Fried’s professorial parents and Damian Williams, U.S. attorney for the Southern District of New York.

A conscious criminal…

The prosecution’s account of the alleged crimes by Bankman-Fried, who spent most of his day in court staring into a laptop while seated between his two
attorneys, featured a study in contrasts.
</item 8>

<item 9>
The best Black Friday headphone deals from Sony, Apple, Bose, Beats and more (Engadget, 2023-11-24T14:30:56+00:00)
However, there’s no wireless charging, the battery life (five or so hours per charge) is mediocre and the IPX2 sweat-resistance rating is too weak for the gym.
Amazon’s companion app is rather bloated too. But those shouldn’t be deal-breakers for $35. We gave the Echo Buds a review score of 77 in July.

Anker Soundcore Space A40

The Anker Soundcore Space A40 is one of the best values in wireless earphones at its normal price of $79, but now it’s down to $54 for Black Friday. That ties
the lowest price we’ve tracked. The Space A40 tops our guide to the best budget wireless earbuds because it delivers most of the features we expect from
headphones that cost twice as much. It’s comfortable, its ANC is genuinely capable, its battery lasts up to 10 hours per charge, its tiny case supports wireless
charging and it can pair with two devices at once.

Graph facts:
- Video games in 2023: Acquisitions, layoffs, unions -MENTIONS-> Sony
- Young Taylor Swift fans go viral for struggling to open the star's CD: 'Making me laugh' -MENTIONS-> Sony
- Steam Deck OLED review: It's just better -MENTIONS-> Sony
- Michelle Jubelirer on Ice Spice, The Beatles, and reinventing Capitol Music Group: ‘I don’t think any label group has been able to turn things around as
quickly as my team and I have done in less than two years.’ -MENTIONS-> Sony
- A comprehensive list of 2023 tech layoffs -MENTIONS-> Sony

Entities: Sony, Apple, Bose, Beats, The best Black Friday headphone deals from Sony, Apple, Bose, Beats and more, ANC, Echo Buds, Anker Soundcore Space A40,
wireless earphones, wireless charging, headphones, Black Friday, Amazon, Engadget
</item 9>

<item 10>
The best Black Friday headphone deals from Sony, Apple, Bose, Beats and more (Engadget, 2023-11-24T14:30:56+00:00)
Engadget has been testing and reviewing consumer tech since 2004. Our stories may include affiliate links; if you buy something through a link, we may earn a
commission. Read more about how we evaluate products.

Black Friday is often a boon for deals on headphones and earbuds, and this year is no different. The shopping holiday has brought a plethora of deals on all
kinds of headphones, include several top picks from our guides to the best wireless headphones, wireless earbuds, budget wireless earbuds and gaming headsets.
Other popular models from the likes of Sony, Beats, Bose and Apple are discounted as well. All of this means it's an ideal time to pick up a new set, whether
it's to give as a gift to someone on your holiday shopping list or just to replace your own aging cans. To help you get a good bargain, we've rounded up the
best Black Friday deals on headphones and earbuds we could find for 2023.

Graph facts:
- Video games in 2023: Acquisitions, layoffs, unions -MENTIONS-> Sony
- Young Taylor Swift fans go viral for struggling to open the star's CD: 'Making me laugh' -MENTIONS-> Sony
- Steam Deck OLED review: It's just better -MENTIONS-> Sony
- Michelle Jubelirer on Ice Spice, The Beatles, and reinventing Capitol Music Group: ‘I don’t think any label group has been able to turn things around as
quickly as my team and I have done in less than two years.’ -MENTIONS-> Sony
- A comprehensive list of 2023 tech layoffs -MENTIONS-> Sony

Entities: Sony, Apple, Bose, Beats, headphones, earbuds, wireless headphones, budget wireless earbuds, gaming headsets, The best Black Friday headphone deals
from Sony, Apple, Bose, Beats and more, Black Friday, wireless earbuds, Engadget
</item 10>
=== END ITEM id=e509debca4526018 ===
