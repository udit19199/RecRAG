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

=== ITEM id=70c45314ad43bad4 ===
Gold sentences and facts:
Sentence 0: But “all of that, all of it, was built on lies,” Rehn declared to the jury.
  Fact 0.0: Rehn declared to the jury.
  Fact 0.1: Rehn declared to the jury that all of that, all of it, was built on lies.
Sentence 1: Binance has made headlines this past year for a range of reasons, including Zhao’s comments contributing to the collapse of FTX, which was once one of its top competitors.
  Fact 1.0: Binance has made headlines this past year for a range of reasons.
  Fact 1.1: Binance has made headlines this past year including because of Zhao's comments.
  Fact 1.2: Zhao's comments contributed to the collapse of FTX.
  Fact 1.3: FTX was once one of Binance's top competitors.

Retrieved items:
<item 1>
SBF’s trial starts soon, but how did he — and FTX — get here? (TechCrunch, 2023-10-01T14:00:29+00:00)
SBF’s trial has started, this is how he and FTX got here

The highly anticipated criminal trial for Sam Bankman-Fried, former CEO of bankrupt crypto exchange FTX, started Tuesday to determine whether he’s guilty of
seven counts of fraud and conspiracy. And as one former federal prosecutor put it: “The odds seem to be stacked against him at this point.”

The 31-year-old co-founded FTX in 2019; within a few years the once third-largest crypto exchange’s valuation hit $32 billion at its peak. It’s now trying to
claw back any funds to distribute to creditors.

But how did the once third-largest crypto exchange get here?

Before FTX, Bankman-Fried co-founded crypto-trading firm Alameda Research in 2017. He co-founded FTX in 2019 as a complement to Alameda, to help bring in
revenue and liquidity for the trading arm.
</item 1>

<item 2>
Sam Bankman-Fried didn’t ask where the $8 billion went (The Verge, 2023-10-31T19:52:07+00:00)
Oh, but he was testifying to Congress about keeping customer funds safe.

Sam Bankman-Fried, the founder of failed cryptocurrency exchange FTX, is on trial for seven counts of wire fraud and conspiracy. FTX was a fraud “from the
start,” the Securities and Exchange Commission alleges — with a “multi-billion-dollar deficiency caused by his own misappropriation of customer funds.” Follow
along for all the latest news and regular updates from the trial.

Who was making decisions to spend $8 billion of customer funds? Bankman-Fried couldn’t recall knowing anything about it. Were there rules or requirements for
how money borrowed from FTX would be returned? Were there rules for risk management? “I was concerned with overall risk management,” Bankman-Fried said.

But it was the testimony about June 2022 that resonated the most to me. Didn’t Bankman-Fried ask what “fiat@ftx” was? He did.
</item 2>

<item 3>
SBF’s trial starts soon, but how did he — and FTX — get here? (TechCrunch, 2023-10-01T14:00:29+00:00)
That means it purported to have more FTT tokens on its balance sheet than what existed in the world.

Around the same time it was exposed, the world’s largest crypto exchange, Binance, started pulling out its remaining $2.1 billion equivalent of cash in BUSD and
FTT. (It had an equity position in FTX from 2019 to 2021.) This essentially triggered a bank run on FTX.

FTX and Alameda filed for Chapter 11 bankruptcy in the U.S. mid-November 2022. Bankman-Fried resigned, and John J. Ray III, the Enron turnaround veteran, was
appointed its new CEO.

Bankman-Fried, however, maintained his innocence. At The New York Times’ DealBook Summit, he appeared virtually from the Bahamas, saying “I didn’t ever try to
commit fraud on anyone; I was shocked by what happened this month.”
</item 3>

<item 4>
The jury finally hears from Sam Bankman-Fried (The Verge, 2023-10-28T00:12:41+00:00)
Bankman-Fried said he worked 12 to 22 hours a day, and took one day off every couple of months. Because FTX had grown so much, he could no longer run both
companies, he said. Bankman-Fried handed the company off to Caroline Ellison and Sam Trabucco, who immediately after being named co-CEO promptly drifted away to
early retirement. (Quiet quitting king!) Bankman-Fried did remain involved in hedging and risk at Alameda, though.

About that $1 billion of revenue in 2021: Bankman-Fried definitely did not know that Singh, his employee, had backdated interest payments to get FTX “over the
line” to $1 billion. See, he’d just asked his employees to check and see if there was any source of funds that was missing to get to $1 billion. This testimony
was especially rambling.

Oh, also that MobileCoin loss? The one Wang said Alameda took to keep off FTX’s balance sheet?
</item 4>

<item 5>
SBF’s trial starts soon, but how did he — and FTX — get here? (TechCrunch, 2023-10-01T14:00:29+00:00)
regulators and government officials, many of whom he donated to.

Bankman-Fried was even compared to Warren Buffet and many called him the white horse of crypto (TechCrunch never did, for what it’s worth).

But in early November 2022, that all changed.

FTX’s collapse

Concerns surrounding FTX’s liquidity grew after CoinDesk published a copy of Alameda’s balance sheet, showing the firm held $14.6 billion in assets and $8
billion in liabilities as of June 30, 2022.

But there was a problem: The report showed Alameda’s largest asset was $3.66 billion of “unlocked FTT” and $2.16 billion of “FTT collateral.” FTT was the token
behind FTX.

The balance sheet showed that the $5.82 billion in FTT tokens that Alameda owned was 193% higher than the total FTT market cap, which was about $3 billion at
the time.
</item 5>

<item 6>
Sex, Signal messages, and sabotaging FTX's code: SBF execs and Bahamas roommates tell all in court (Cnbc | World Business News Leader,
2023-10-06T20:17:00+00:00)
"FTX defrauded all its customers," he said.

Matt Huang, co-founder of Paradigm Operations LP, right, arrives at court in New York, US, on Thursday, Oct. 5, 2023. Former FTX Co-Founder Sam Bankman-Fried is
charged with seven counts of fraud and money laundering following the collapse of his cryptocurrency empire last year. Photographer: Yuki Iwamura/Bloomberg via
Getty Images Yuki Iwamura | Bloomberg | Getty Images

Investment to zero

The third witness to take the stand was Matt Huang, co-founder and managing partner of Paradigm, a crypto venture capital firm that invested over $275 million
in FTX. That stake was wiped out. Huang testified about his firm's due diligence on FTX, and he told the court that Bankman-Fried assured him that funds would
be used for FTX and not Alameda.
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
</item 7>

<item 8>
Sex, Signal messages, and sabotaging FTX's code: SBF execs and Bahamas roommates tell all in court (Cnbc | World Business News Leader,
2023-10-06T20:17:00+00:00)
Witness Adam Yedidia answers questions during Sam Bankman-Fried fraud trial over the collapse of FTX, the bankrupt cryptocurrency exchange, at Federal Court in
New York City, October 5, 2023, in this courtroom sketch. Jane Rosenberg | Reuters

Two of Sam Bankman-Fried's former friends from MIT, who also worked at crypto exchange FTX while living with the company's founder in the Bahamas, took the
stand in a Manhattan courtroom this week to testify against their former classmate, confidant, and boss — a man who allegedly ran a crypto empire that defrauded
thousands of customers out of billions of dollars. Gary Wang, the lesser-known co-founder of FTX, was asked by Assistant U.S. Attorney Nicolas Roos on Thursday,
"Did you commit financial crimes while working at FTX?" "Yes," responded Wang.
</item 8>

<item 9>
Sam Bankman-Fried was a terrible boyfriend (The Verge, 2023-10-10T23:50:21+00:00)
FTX was a fraud “from the start,” the Securities and Exchange Commission alleges — with a “multi-billion-dollar deficiency caused by his own misappropriation of
customer funds.” Follow along for all the latest news and regular updates from the trial.

The day started off promisingly for the defense as it cross-examined Gary Wang, the chief technology officer of FTX and co-owner of both FTX and Alameda.
Christian Everdell, one of Bankman-Fried’s defense attorneys, couldn’t undo the damage of last week’s code review. But he managed to shake the rust off long
enough to make Wang sound less reliable, drowning the jury in confusing technicalities.

Last week, Wang testified that Alameda got access to a special credit line and an option to take its balance into the negative without triggering liquidation —
something he alleged other accounts at FTX didn’t get. Everdell tried to undermine this claim by talking about the spot margin program, which let users lend
each other assets for margin trading.
</item 9>

<item 10>
The FTX trial is bigger than Sam Bankman-Fried (The Verge, 2023-09-28T12:00:00+00:00)
Bankman-Fried is accused of using “billions of dollars in misappropriated FTX customer deposits” to help buy more than $200 million of real estate for himself,
make billions of dollars of investments for his own interest, and repay Alameda’s lenders, according to the indictment, Additionally, Bankman-Fried allegedly
used more than $100 million of customer funds to make political contributions; prosecutors can show evidence of those contributions in this trial, even though
they aren’t part of the charges brought. A second trial is scheduled for March 2024, with additional charges.

Before his fall, Bankman-Fried made himself out to be the Good Boy of crypto — the trustworthy face of a sometimes-shady industry. He was also very interested
in publicity, sitting for many interviews both before and after the fall of FTX. The quick rise of FTX as an industry force was at least in part due to
Bankman-Fried’s appetite for attention.
</item 10>
=== END ITEM id=70c45314ad43bad4 ===

=== ITEM id=22e69387f8e46892 ===
Gold sentences and facts:
Sentence 0: The Justice Department is focused on the deals Google makes — with Apple but also with Samsung and Mozilla and many others — to ensure it is the default search engine on practically every platform.
  Fact 0.0: The Justice Department is focused on the deals Google makes.
  Fact 0.1: Google makes deals with Apple.
  Fact 0.2: Google makes deals with Samsung.
  Fact 0.3: Google makes deals with Mozilla.
  Fact 0.4: Google makes deals with many others.
  Fact 0.5: The Justice Department focuses on these deals to ensure Google is the default search engine on practically every platform.
Sentence 1: In our last roundup, we learned how Google spent $26.3 billion in 2021 making itself the default search engine across platforms and how Google tried to have Chrome preinstalled on iPhones.
  Fact 1.0: In the last roundup, the writers learned that Google spent $26.3 billion in 2021 making itself the default search engine across platforms.
  Fact 1.1: In the last roundup, the writers learned that Google tried to have Chrome preinstalled on iPhones.

Retrieved items:
<item 1>
5 things we learned so far about the Google antitrust case (TechCrunch, 2023-10-31T02:30:32+00:00)
Google CEO Sundar Pichai testified Monday in the U.S. government’s antitrust trial against the company. The executive defended Google’s business tactics,
including its deal with Apple and other partners to make Google the default search engine.

The lawsuit stems from a 2020 antitrust claim over Google’s dominance in the online search market. The claim is separate from one filed in January by the
Department of Justice, along with eight states including New York, California and Colorado, that aims to “halt Google’s anticompetitive scheme, unwind Google’s
monopolistic grip on the market and restore competition to digital advertising.”

The government has argued that Google uses its platforms and deals with partners to block out any competition in search or advertising, thus hindering
competitors from accessing the data they’d need to improve their products.

Google argues that it’s just doing business. Everybody wants Google as the default engine because it’s the best. That doesn’t make its actions illegal, the
company says.
</item 1>

<item 2>
Is Google Search better than the rest? And is that fair? (The Age, 2023-10-22T18:00:00+00:00)
The example given was replacing a search for “children’s clothing” with “NIKOLAI-brand kidswear”. Loading Google rejected this in very strong terms, saying the
piece was misleading and inaccurate while denying ever altering search terms. Wired removed the article for not meeting its standards, but the degree to which
it was shared on social media and boosted in write-ups at other outlets shows just how willing people are to accept foul play on Google’s part. A suspicion of
privacy invasion and inappropriate data handling follows the company at all times.

Grey herself is a former vice-president at DuckDuckGo, a privacy-focused company founded explicitly to counter giants such as Google. She admits that she may
have misinterpreted the evidence, but maintains that Google manipulates Search to maximise ad revenue. So what does Google say? In a post on X, Google’s
official search liaison, Danny Sullivan, said ad systems do not affect the organic results, i.e. the list of blue links in search results that are not
sponsored.
</item 2>

<item 3>
More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo (TechCrunch, 2023-11-13T14:37:44+00:00)
He said that the competition between Google and Apple is “as intense as it gets,” reports Bloomberg.

Rosenberg explained how Google gets manufacturers to sign a mobile app distribution agreement (MADA) that requires Android smartphone makers (like Samsung or
Oppo) to pre-load a bundle of 11 Google apps on the device, including Search, Chrome and Play. They don’t have to be the default choices, he said.

Google also has revenue share agreements (RSAs) with smartphone makers and wireless carriers (like Verizon) that require them to set Google search and Chrome
web browser as defaults. Rosenberg defended the move and said it was because Google apps [like Search] are “best in class.” The RSAs also motivate other
companies to make or sell more Android devices, he said.
</item 3>

<item 4>
5 things we learned so far about the Google antitrust case (TechCrunch, 2023-10-31T02:30:32+00:00)
Google pays big bucks to be everywhere so that it can take in all the data and be the best search engine across different companies’ devices, said Pichai,
according to The Verge.

Google understood the value of defaults very early on. U.S. Justice Department lawyer Meagan Bellshaw showed Pichai a 2007 email from a Google product strategy
meeting containing data showing that when people changed their browser homepage to Google, they did 15% more Google searches. When they switched away, their
Google searches dropped 27%.

“Nitin argues that focusing on homepage market share is one of the most effective things we can do to make gains in search market share,” read an email that
summarized the meeting and was sent to Pichai, as well as other Google leaders, according to The Verge.

The amount that Google spent on homepage market share has been a fixing point in the trial.
</item 4>

<item 5>
5 things we learned so far about the Google antitrust case (TechCrunch, 2023-10-31T02:30:32+00:00)
That doesn’t make its actions illegal, the company says.

Monday’s trial hearing revealed plenty of juicy tidbits, including the $26.3 billion Google spent making itself the default search engine across platforms in
2021, how Google tried to take it further and have Chrome preinstalled on iPhones and more.

Google paid $26 billion in 2021 to be everyone’s default search engine

When Google’s search head Prabhakar Raghavan testified in court on October 28, he revealed that the tech giant had paid $26.3 billion in 2021 to multiple
browsers, phones and platforms, from companies including Apple, Samsung and Mozilla, The Verge reports.

About $18 billion of that total amount went directly to Apple, according to a New York Times report published earlier this month. Google has had its deal with
Apple in place since 2003.

When questioned on the amount of money Google spends to get first pick of search engines, Pichai said that the decision was made with the consumer in mind.
</item 5>

<item 6>
5 things we learned so far about the Google antitrust case (TechCrunch, 2023-10-31T02:30:32+00:00)
Bellshaw pulled up a message exchange in 2021 where Pichai asked for history to be turned off in a group chat. Pichai responded that he wanted to discuss a
personnel matter and the subject had nothing to do with the litigation hold, which he said he takes great care to comply with.

A moment of nostalgia for Internet Explorer

Cast your memories back to 2005, when Microsoft’s Internet Explorer became the default search engine. Back then, Google’s legal chief David Drummond sent
Microsoft an angry letter, saying that making Internet Explorer the search default was anticompetitive. Oh, how the tables have turned.

After establishing that Google understands the inherent value of defaults, Bellshaw brought up Drummond’s letter to establish the hypocrisy of Google today. The
letter declared that problems with a default setting are made worse by how changes to defaults are handled, and that most end users “do not change defaults.”
</item 6>

<item 7>
Apple defends Google Search deal in court: ‘There wasn’t a valid alternative’ (The Verge, 2023-09-26T17:01:22+00:00)
But the questions put to Cue were the same ones the DOJ is going to keep asking: is Google really the best search engine, or is it just the one writing the
biggest checks? And if those checks went away, what would the search engine market look like? Cue said Apple’s never really thought about it. Google said Apple
would be silly to do so. And the Justice Department thinks it’s about time Apple starts doing so.
</item 7>

<item 8>
Apple defends Google Search deal in court: ‘There wasn’t a valid alternative’ (The Verge, 2023-09-26T17:01:22+00:00)
He said there still isn’t one.

That question — whether Apple picked Google because it’s the most lucrative choice or the best product — was a key part of Cue’s testimony and, in fact, a key
part of the DOJ’s entire case against Google. The Justice Department is focused on the deals Google makes — with Apple but also with Samsung and Mozilla and
many others — to ensure it is the default search engine on practically every platform.

Bellshaw asked Cue a number of questions about the iPhone setup process. Those three screenshots showed the Appearance screen that shows up when you first boot
up your iPhone so you can pick font sizes; the location-tracking prompt that appears when you open Maps; and the App Tracking Transparency pop-up that tells you
when an app wants to collect your data. Cue objected to all these things being considered part of setup, but Bellshaw’s point was that Apple offers its users a
choice about lots of things, big and small, and that search could be one of them.
</item 8>

<item 9>
Is Google Search better than the rest? And is that fair? (The Age, 2023-10-22T18:00:00+00:00)
and it replied with more options that were all over $300. Another observation from this test is that I tend to use the Google Search bar as much for returning
to familiar sites as I do for searching for information, which means switching to another provider could be temporarily annoying simply because it wouldn’t
remember all the places I’ve been. Is Google as the default unfair?

In a response to the trial on Google’s blog, president of global affairs Kent Walker said the company’s search product was the favourite of users, browsers and
device makers simply because it was the best in the business. And being default was not determinative, he said; changing from Google Search to any other search
provider was easy, and the company competes with many others on phones and PCs. “People don’t use Google because they have to, they use it because they want
to,” he said. “This lawsuit simply ignores how intensely competitive and dynamic the technology industry is today.”
</item 9>

<item 10>
Is Google Search better than the rest? And is that fair? (The Age, 2023-10-22T18:00:00+00:00)
“This lawsuit simply ignores how intensely competitive and dynamic the technology industry is today.” Walker also points out that, when faced with a situation
where Google is not the default, users tend to spend some effort putting Google back in charge. When Mozilla made Yahoo! the default on Firefox, most users
changed it to Google. And when people set up a Windows device for the first time they frequently sidestep the Microsoft default — “Google” was the number one
search query on Bing in 2021 — despite Microsoft making that very annoying to do. This is a compelling point, and many of us will be well accustomed to dodging
all of Microsoft’s pleading while trying to get Chrome and Google set as defaults whenever we set up a new PC. But all of these arguments do ignore the fact
that Google and its competitors are not on an even playing field. The question remains, why do users prefer Google?
</item 10>
=== END ITEM id=22e69387f8e46892 ===

=== ITEM id=95b3a0bbfb208c42 ===
Gold sentences and facts:
Sentence 0: The second week of the trial’s standout testimony came from Alameda Research’s former CEO Caroline Ellison, who claimed she took $14 billion from customers to repay debts to lenders, using them as a line of credit under the instruction of SBF.
  Fact 0.0: The second week of the trial's standout testimony came from Caroline Ellison.
  Fact 0.1: Caroline Ellison is Alameda Research's former CEO.
  Fact 0.2: Caroline Ellison claimed she took $14 billion from customers.
  Fact 0.3: Caroline Ellison claimed the $14 billion was used to repay debts to lenders.
  Fact 0.4: Caroline Ellison claimed she used customers as a line of credit.
  Fact 0.5: Caroline Ellison claimed she did this under the instruction of SBF.
Sentence 1: Using Caroline Ellison, his on-again off-again girlfriend and the CEO of Alameda, as a front, Bankman-Fried had “secret access” to customer money—both cash and crypto—the government claimed.
  Fact 1.0: The government claimed that Bankman-Fried used Caroline Ellison as a front.
  Fact 1.1: Caroline Ellison is Bankman-Fried's on-again off-again girlfriend.
  Fact 1.2: Caroline Ellison is the CEO of Alameda.
  Fact 1.3: The government claimed that Bankman-Fried had 'secret access' to customer money, both cash and crypto.
Sentence 2: The prosecution painted Bankman-Fried as someone who knowingly committed fraud to achieve great wealth, power and influence, while the defense countered that the FTX founder acted in good faith, never meant to commit fraud or steal and basically got in over his head.
  Fact 2.0: The prosecution portrayed Bankman-Fried as someone who knowingly committed fraud to achieve great wealth, power and influence.
  Fact 2.1: The defense countered that the FTX founder acted in good faith.
  Fact 2.2: The defense countered that the FTX founder never meant to commit fraud or steal.
  Fact 2.3: The defense countered that the FTX founder basically got in over his head.

Retrieved items:
<item 1>
Sam Bankman-Fried was a terrible boyfriend (The Verge, 2023-10-10T23:50:21+00:00)
Using the effectively unlimited funds “allowed us to make profitable trades we couldn’t have made otherwise”

Bankman-Fried didn’t grant Ellison equity, even though she asked; he told her it would be too complicated. Instead, she got a $200,000 salary, even as CEO, and
bonuses twice a year, which ranged from $100,000 to $20 million.

Initially, Alameda and FTX were “very integrated,” Ellison said. They were run by the same team, from the same office. And when Alameda was scrounging for
funds, Bankman-Fried told Ellison that FTX would be a good source of capital. The $65 billion line of credit Alameda Research had meant that it did not have to
post collateral. There was no contract and no written terms, she testified. It also wasn’t visible to FTX’s auditors — she’d raised the question with
Bankman-Fried, and he told her not to worry about it.
</item 1>

<item 2>
Sam Bankman-Fried was a terrible boyfriend (The Verge, 2023-10-10T23:50:21+00:00)
Alameda’s credit line — which was taken in increments of $100,000 to $10 million at a time — was used for trading. Using the effectively unlimited funds
“allowed us to make profitable trades we couldn’t have made otherwise,” Ellison testified.

Customer funds were also used when Bankman-Fried bought back FTX shares from Binance, an early investor, in the summer of 2021. Bankman-Fried told Ellison it
was “really important,” otherwise “Binance would do things to mess with FTX.” Ellison says she told him Alameda didn’t have the money. So Bankman-Fried took $1
billion of FTX customer funds to buy out Binance, the first time Ellison recalled an amount that large. It was Bankman-Fried’s decision, she said, as he was the
CEO of FTX.

FTT was one of several “Sam coins”

There was also the FTT token, which was created by Bankman-Fried and Wang.
</item 2>

<item 3>
The jury finally hears from Sam Bankman-Fried (The Verge, 2023-10-28T00:12:41+00:00)
Then, Nishad Singh, about a month after founding Alameda Research. Finally, Caroline Ellison.

Not having a risk team, when you are any kind of financial anything, is certainly a choice

Though Bankman-Fried was the CEO, and also the majority owner, he wanted to be clear: he did not supervise Wang’s direct work. Anyway, after a bunch of wildly
successful arbitrage — 50 percent to 100 percent annualized returns, per his testimony — he decided to found a cryptocurrency exchange, FTX. He figured he’d
fail; that there was only a 20 percent chance of success. Bankman-Fried did not define what he expected the time period to be on this estimate, but arguably 20
percent was a much higher chance of success than FTX would enjoy once Alameda dipped into the customer deposits.

By the way, because he was such a good guy, Bankman-Fried made a point of “periodically” handling support tickets himself.
</item 3>

<item 4>
Sam Bankman-Fried was a terrible boyfriend (The Verge, 2023-10-10T23:50:21+00:00)
That was the basis of the worst of her testimony — and the spreadsheet from hell.

Ellison said she’d prepared the spreadsheet at Bankman-Fried’s request in the fall of 2021 and shared it with him. The point was risk analysis around paying
back Alameda’s loans if they were abruptly recalled by Genesis, their lender. Bankman-Fried wanted to use $3 billion for venture investments, so Ellison was
ballparking what that would do to Alameda’s risk. In the as-is scenario, if things went south, she figured there was a 30 percent chance they wouldn’t be able
to meet the loan recalls. If Bankman-Fried used $3 billion to make investments, there was a 100 percent chance they couldn’t meet the recalls, even with FTX
customer funds.

Echoing Bankman-Fried and Ellison’s romantic relationship, FTX was keeping cozy private ties with Alameda yet publicly holding it at arm’s length

The problem here wasn’t really the math, which seemed pretty arbitrary.
</item 4>

<item 5>
Sam Bankman-Fried was a terrible boyfriend (The Verge, 2023-10-10T23:50:21+00:00)
Alameda got its war chest — 60 percent to 70 percent of the initial supply — for free, while seed investors got FTT at 10 cents a coin, and FTT first listed at
$1 a coin. Bankman-Fried felt that $1 per coin was psychologically important, Ellison said, and he directed her to buy up FTT using Alameda if its price fell
below a dollar.

FTT was one of several “Sam coins,” a nickname for tokens that Bankman-Fried was heavily involved in and owned a lot of, either personally or through Alameda.
Those coins were almost certainly worth less than the value displayed on the balance sheet because trying to sell them all at once would crater the prices.
Bankman-Fried directed her to put those coins on the balance sheets Alameda showed to lenders, even though she felt it was “somewhat misleading.”

Alameda was also getting loans from outside lenders, such as Genesis, because when FTX started, there weren’t a lot of customer funds to borrow, Ellison
testified.
</item 5>

<item 6>
Sam Bankman-Fried was a terrible boyfriend (The Verge, 2023-10-10T23:50:21+00:00)
That gave Ellison an unusual view of his character. “He was very ambitious,” she said. Besides telling her about his presidential chances, he also told her that
if there was a coin flip where tails destroyed the world and heads made the world twice as good, he’d flip the coin. He called this being “risk-neutral,” which
seems like a fancy way of saying he was a gambling addict.

She was named co-CEO of Alameda with Sam Trabucco in 2021, while she and Bankman-Fried were broken up, and CEO in 2022. The goal, Bankman-Fried told her, was to
“optically” separate Alameda Research and FTX. “The whole time we were dating, he was my boss at work,” she said. They broke up because she wanted more from the
relationship; Bankman-Fried was distant and not paying enough attention to her.
</item 6>

<item 7>
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
</item 7>

<item 8>
Sam Bankman-Fried was a terrible boyfriend (The Verge, 2023-10-10T23:50:21+00:00)
It was that Ellison’s calculations assumed Alameda could borrow $1.8 billion in normal dollars and $1.5 billion in crypto from FTX. The spreadsheet makes this
clear with a row labeled “FTX borrows,” which Ellison said were customer funds.

Meanwhile, echoing Bankman-Fried and Ellison’s romantic relationship, FTX was keeping cozy private ties with Alameda yet publicly holding it at arm’s length. On
January 14th, 2022, Bankman-Fried tweeted, “We’re launching a $2b venture fund: FTX Ventures!” Those funds came from Alameda, Ellison testified. But
Bankman-Fried didn’t want to go public with the source of the funds. He said he thought Alameda’s brand was less good, and he didn’t want his name associated
with it. Alameda also bought Robinhood shares for Bankman-Fried, who moved them to a vehicle called “Emergent Fidelity Technologies” to avoid association with
Alameda.
</item 8>

<item 9>
SBF Trial: The latest updates from the FTX collapse’s courtroom drama (TechCrunch, 2023-10-06T17:22:02+00:00)
Much of said funding came from customers, and Easton testified that he found that user deposits were used by both FTX and Alameda to purchase real estate, make
investments or direct funding toward political causes and charities.

SBF trial: What we learned in week two

The second week of the trial’s standout testimony came from Alameda Research’s former CEO Caroline Ellison, who claimed she took $14 billion from customers to
repay debts to lenders, using them as a line of credit under the instruction of SBF. Ellison also went into detail about how a $2 billion venture fund, FTX
Ventures, was funded by money that had previously been allocated to Alameda from third-party lenders.

And though there are no charges of bribery in the case, Ellison also testified to paying Chinese officials $150 million to restore access to $1 billion in
frozen trading accounts.

SBF trial: What we learned in the week one

As expected, the opening statements in the SBF trial were spicy.
</item 9>

<item 10>
Sam Bankman-Fried was a terrible boyfriend (The Verge, 2023-10-10T23:50:21+00:00)
(Fraud, conspiracy to commit fraud, and money laundering, in case you were wondering.) “We ultimately took about $14 billion, some of which we were not able to
pay back.” She tilted her head down to answer the questions, then lifted her head when she’d finished her answer.

Bankman-Fried called this being “risk-neutral,” which seems like a fancy way of saying he was a gambling addict

In Ellison’s telling, Alameda was troubled from her earliest time there in 2018. “Shortly after I started, I learned the company was in worse shape than I
realized,” Ellison said. Alameda had initially been funded with loans “from acquaintances,” she said, and those loans were recalled a few weeks after she
arrived. (There was a staff revolt within Alameda Research over lost millions and general financial chaos, according to Michael Lewis’ Going Infinite.) Ellison
asked Bankman-Fried why he hadn’t shared the company’s shaky circumstances in the job offer. “He hadn’t known how to tell me,” she said.
</item 10>
=== END ITEM id=95b3a0bbfb208c42 ===

=== ITEM id=4d5b63c798fcabbf ===
Gold sentences and facts:
Sentence 0: Both PMS and mutual funds provide expert guidance in navigating volatile markets and seizing investment opportunities that help create wealth.
  Fact 0.0: PMS provides expert guidance in navigating volatile markets.
  Fact 0.1: PMS provides expert guidance in seizing investment opportunities.
  Fact 0.2: PMS helps create wealth.
  Fact 0.3: Mutual funds provide expert guidance in navigating volatile markets.
  Fact 0.4: Mutual funds provide expert guidance in seizing investment opportunities.
  Fact 0.5: Mutual funds help create wealth.
Sentence 1: Brown added that he was “surprised” to see that a second valuation expert had not been drafted in to value the HSF portfolio for the latest interim results, especially as Kroll was appointed to consider the “reasonableness” of Citrin Cooperman’s assumptions for HSF’s previous set of results (to end of March 2023).
  Fact 1.0: Brown said he was surprised.
  Fact 1.1: Brown was surprised that a second valuation expert had not been drafted in to value the HSF portfolio for the latest interim results.
  Fact 1.2: Kroll was appointed to consider the reasonableness of Citrin Cooperman's assumptions for HSF's previous set of results.
  Fact 1.3: HSF's previous set of results runs to end of March 2023.

Retrieved items:
<item 1>
Hipgnosis Songs Fund’s board is concerned over Citrin Cooperman’s latest valuation of the company. What’s it going to do about it? (Music Business Worldwide,
2023-12-19T23:02:01+00:00)
In a rather damning note issued today covering the delay in the publication of HSF’s results, Brown wrote: “This is an early blow to the credibility of the new
[Hipgnosis Songs Fund] Board, and casts further doubt over the credibility of the independent valuer, Citrin Cooperman.”

Brown added that he was “surprised” to see that a second valuation expert had not been drafted in to value the HSF portfolio for the latest interim results,
especially as Kroll was appointed to consider the “reasonableness” of Citrin Cooperman’s assumptions for HSF’s previous set of results (to end of March 2023).

Added Brown: “We think the easiest solution now for the [HSF] Board is to apply an additional discount rate to the [Citrin Cooperman] valuation to reflect
current market uncertainty, on the basis that it is probably too late to employ another valuer.”
</item 1>

<item 2>
Hipgnosis Songs Fund’s board is concerned over Citrin Cooperman’s latest valuation of the company. What’s it going to do about it? (Music Business Worldwide,
2023-12-19T23:02:01+00:00)
Not to mention this line in various HSF company reports to date: “The [HSF] Board is ultimately and solely responsible for overseeing the valuation of the
Company’s investments in music catalogues and has appointed [Citrin Cooperman] to perform this specialist work.”

WHAT’S really going on?

It’s a fair bet that Hipgnosis Songs Fund’s board – which last week appointed activist investor Christopher Mills as a non-exec Director – is making something
of a ‘hidden’ statement to shareholders with today’s announcement.

It may be a coded attempt by HSF’s board to gnash teeth towards HSM/Mercuriadis’ ‘call option’ – which, as MBW has pointed out before, leaves the exec with the
exclusive power to sweep in and acquire HSF’s portfolio under several different circumstances.
</item 2>

<item 3>
Investment options for high networth investors (Business Line, 2023-10-28T14:42:28+00:00)
Portfolio Management Services (PMS)

PMS guidelines in India date back to 1993, but this investment avenue has grown in popularity with HNIs recently. The growing financialisation of savings, the
formalisation of the economy, and the growing prosperity of tier 2 and 3 cities have resulted in an increase in AUM (excluding EPFO/PF), which stood at ₹5.28
lakh crore as of July 2023. Because the minimum investment is ₹50 lakh, this option is also only available to HNIs. Both PMS and mutual funds provide expert
guidance in navigating volatile markets and seizing investment opportunities that help create wealth. Because mutual funds cater to retail investors, their
guidelines are relatively stringent, and compliance is rigorous.

Regarding PMS, the investment guidelines are somewhat more flexible, allowing portfolio managers to outperform mutual funds. Another feature that appeals to
HNIs is that the beneficial ownership of securities is that of the investors, as opposed to mutual funds, where the ownership is that of the fund.
</item 3>

<item 4>
Hipgnosis Songs Fund’s board is concerned over Citrin Cooperman’s latest valuation of the company. What’s it going to do about it? (Music Business Worldwide,
2023-12-19T23:02:01+00:00)
“[It] is up to the [Hipgnosis Songs Fund] Board to satisfy itself that the independent valuer, whom it hires, is up to the job, rather than try and pin the
blame on [Hipgnosis Songs Management].” Christopher Brown, JP Morgan

Concluded Brown: “[It] is up to the [Hipgnosis Songs Fund] Board to satisfy itself that the independent valuer, whom it hires, is up to the job, rather than try
and pin the blame on [Hipgnosis Songs Management].

“Our view is that a breakdown in the relationship between Board and manager would not be optimal for maximising shareholder value.”

Brown further noted that JP Morgan would “not be surprised to see Citrin Cooperman resign, having been very publicly undermined by the [HSF] Board today”.

JP Morgan retained its ‘overweight’ rating against HSF’s share price “in spite of yet another [HSF] own goal”.
</item 4>

<item 5>
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
</item 5>

<item 6>
Hipgnosis Songs Fund’s board is concerned over Citrin Cooperman’s latest valuation of the company. What’s it going to do about it? (Music Business Worldwide,
2023-12-19T23:02:01+00:00)
That price, according to HSF’s board, represented a 14.2% discount on Citrin Cooperman’s valuation of this portfolio as of September 30 (something we’ll come
back to).

As part of its announcement this morning, the HSF board took something of a swipe at its investment adviser, Merck Mercuriadis‘ Hipgnosis Song Management.

After seeing Citrin Cooperman’s latest valuation and believing it to be too high, said HSF, it then “sought advice from Hipgnosis Song Management… which is
majority owned by funds managed and/or advised by Blackstone, on their opinion on the independent valuer’s valuation.”

HSF’s board added: “Hipgnosis Song Management… eventually provided an opinion, which was heavily caveated, such that the Board has concerns as to the valuation
of the Company’s assets in its interim results.”
</item 6>

<item 7>
Hipgnosis Songs Fund’s board is concerned over Citrin Cooperman’s latest valuation of the company. What’s it going to do about it? (Music Business Worldwide,
2023-12-19T23:02:01+00:00)
Worth remembering: The HSF board has known about Citrin Cooperman’s latest valuation (as of September 30) for at least the past eight days.

We know this because, as mentioned, on December 11, HSF announced that the $23.1 million price of an agreed HSF ‘non-core’ asset sale represented a 14.2%
discount vs. Citrin Cooperman’s latest valuation of the portfolio (i.e. the HSF board had seen this latest valuation).

Yet it took until today (December 19) for HSF to alert its investors to its dissatisfaction with this valuation.

WHAT does JP Morgan make of it all?

One of the closest watchers of Hipgnosis Songs Fund in recent years has been JP Morgan and its analyst Christopher Brown.
</item 7>

<item 8>
Hipgnosis Songs Fund’s board is concerned over Citrin Cooperman’s latest valuation of the company. What’s it going to do about it? (Music Business Worldwide,
2023-12-19T23:02:01+00:00)
In response, Hipgnosis Song Management issued its own statement today: “Hipgnosis Song Management has fulfilled its duties to [HSF] with respect to both the
independent valuation and preparation of the interim results in a timely and efficient manner.

“Notwithstanding the Board’s decision to delay publication of the interim financial statements, [we] will continue to work in a constructive manner to support
the interests of the Company and its shareholders.”

The oddest thing about all of this? Citrin Cooperman’s valuation of Hipgnosis Songs Fund is, according to HSF’s own company documentation, appointed and
overseen… by the HSF board.

See this line, in every annual and interim report from HSF so far (bolding MBW’s own): “Portfolio Independent Valuer… Citrin Cooperman Advisors LLC, formerly
Massarsky Consulting, Inc., [is] appointed by the Board to independently value the Company’s Catalogues within the Portfolio.”
</item 8>

<item 9>
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
</item 9>

<item 10>
Hipgnosis Songs Fund’s board is concerned over Citrin Cooperman’s latest valuation of the company. What’s it going to do about it? (Music Business Worldwide,
2023-12-19T23:02:01+00:00)
the multiples that Merck Mercuriadis paid to acquire it?

Such questions will only be answered when Hipgnosis Songs Fund’s board elects to release the company’s six-monthly financial results to end of September 30.

The most important KPI within those results, in MBW’s view: the company’s Pro-Forma Revenue, and how it compares, like-for-like, vs. the previous year’s
equivalent period.

In the meantime, back in the ‘real world’ of the music biz, HSF is actually enjoying quite a fanciable Q4.

The company is benefitting from its cuts on hit albums including Taylor Swift’s 1989 (Taylor’s Version), via its acquired Jack Antonoff catalog, as well as
Swift’s Lover (via its acquired Joel Little catalog).
</item 10>
=== END ITEM id=4d5b63c798fcabbf ===

=== ITEM id=ee8f4c3e09bc593f ===
Gold sentences and facts:
Sentence 0: From Netflix: Jess and Bill Walker are doing their best to keep their family connected as their children grow older, more independent, and more distant.
  Fact 0.0: The Netflix description states that Jess and Bill Walker are doing their best to keep their family connected.
  Fact 0.1: Jess and Bill Walker are doing their best to keep their family connected as their children grow older.
  Fact 0.2: Jess and Bill Walker's children are growing more independent.
  Fact 0.3: Jess and Bill Walker's children are growing more distant.
Sentence 1: Speaking of the Christmas season, McLean — born in New Rochelle, New York — said that "the holidays are a time for families to come together.
  Fact 1.0: McLean spoke about the Christmas season.
  Fact 1.1: McLean was born in New Rochelle, New York.
  Fact 1.2: McLean said that the holidays are a time for families to come together.

Retrieved items:
<item 1>
Woman switched at birth spends holidays with birth mom after decades-long search: 'Feel at peace' (FOX News - Lifestyle, 2023-12-15T09:30:54+00:00)
While she has found a place with her birth parents, Bazella knows she may never solve the mystery of how she was switched at birth.

"I lost 39 years," Bazella said. "There’s a lot of grieving with that."

Still, Bazella said she believes she's found unconditional love with her mother.

"It's overwhelming for me coming from where I have been," she added.

"I always felt like I was kind of in survival mode and alone. I finally feel at peace with everything, which is really nice."

CLICK HERE TO GET THE FOX NEWS APP

As for Geerts, she said she hopes to make up for lost time.

"One of the main dreams is that we stay healthy and I live long enough, like a hundred years at least — so we can make up some of this time," Geerts said.
</item 1>

<item 2>
Should you introduce your partner to your family during the holidays? Relationship experts weigh in (The Independent - Life and Style,
2023-12-12T16:02:44+00:00)
“I’d recommend pulling your partner aside and saying: ‘How are we doing?’ Like a temperature check,” she added. “Those are points where the two of you can come
and remind each other: ‘We are a team in this.’ Or, the two of you could go on a walk and separate from the family, just for a little bit of time to reset
within the relationship.”

During the holiday season, your partner may be seeing how you interact with your family for the first time. But that’s not necessarily a bad thing, according to
the relationship experts. “I think introducing a significant other to family is a really vulnerable act, and that vulnerability is going to be what allows for
greater intimacy,” Lurie said. “When your significant other sees you with your family, they may see another side of you that they may not have had access to
before.”

(Getty Images)

On the other hand, your partner could learn that your relationship with your parents and extended family members is complex.
</item 2>

<item 3>
Should you introduce your partner to your family during the holidays? Relationship experts weigh in (The Independent - Life and Style,
2023-12-12T16:02:44+00:00)
In a recent poll from the American Psychological Association, 41 per cent of US adults said their stress levels increased from November to January when compared
to other months of the year. Nearly 50 per cent of Americans also described their stress level during the season as “moderate”.

Everyone has different concerns leading up to Christmas, Hanukkah, or New Years - from family drama at the dinner table to a small spats with your mother about
decorations. With what can be stressful events, it only begs the question: do you really want to bring your partner into the mix? While it’s important to
eventually introduce your loved ones to your significant other, relationship experts believe there are some discussions you should have beforehand, especially
during the holidays.

Speaking to The Independent, Rachel Zar - a licensed marriage and family therapist at Spark Chicago Therapy - acknowledged that while every family is different,
there’s a variety of problems that could arise during the holiday season - mainly because we put too much pressure on having a joyful holiday.
</item 3>

<item 4>
Woman switched at birth spends holidays with birth mom after decades-long search: 'Feel at peace' (FOX News - Lifestyle, 2023-12-15T09:30:54+00:00)
"We have our lists and we like to get things done."

CLICK HERE TO SIGN UP FOR OUR LIFESTYLE NEWSLETTER

"I also remember the first time we met — ee were getting ready to go out and she dug in her purse and pulled out all these tubes of lipstick. I said, ‘Oh my
God. You've surpassed me.’ We both constantly put on lipstick. It's so funny."

Bazella continued, "We both like [garden] gnomes and we’re kind of collecting them now. Our mannerisms are similar. We have the same sense of humor."

"One of the main dreams is that we stay healthy and I live long enough … so we can make up some of this time." — Sherri Geerts

Bazella said she has met all four of her siblings — and also has formed a relationship with her birth father, Victor Rebeck, 84.
</item 4>

<item 5>
Woman switched at birth spends holidays with birth mom after decades-long search: 'Feel at peace' (FOX News - Lifestyle, 2023-12-15T09:30:54+00:00)
And Sherri Nordlie Geerts was her mother.

Bazella emailed Geerts and they started corresponding regularly.

"I lost 39 years … There’s a lot of grieving with that." — Diane Bazella

"We were slowly getting to know each other through email," Bazella said. "She had a lot on her plate as a caregiver to her husband, so we were slowly
establishing our mother-daughter relationship."

A LIFE IN PICTURES: FROM BIRTH INTO ADULTHOOD, A FATHER TAKES HIS SON’S PHOTO EVERY DAY FOR DECADES

Finally, on Christmas Day 2021, Bazella and Geerts heard each other’s voices over the phone for the first time.

They met in person in July 2022.

Since then, they have been talking on the phone frequently, sometimes for hours at a time.

‘Same sense of humor’

During their conversations, Geerts opened up about having a baby when she was just 18 years old.
</item 5>

<item 6>
Should you introduce your partner to your family during the holidays? Relationship experts weigh in (The Independent - Life and Style,
2023-12-12T16:02:44+00:00)
“I think anytime that emotions are heightened, with the holidays certainly being one of those times, our expectations are really high,” she said. “We’re
expecting it to be this joyful, happy, perfect occasion. So, we sometimes forget that the holidays come with a lot of stress.”

Despite tensions between relatives at the dinner table, that doesn’t mean your relationship shouldn’t be celebrated. Saba Lurie is a licensed marriage and
family therapist at Take Root Thereapy in California. She explained to The Independent that while there isn’t a right or wrong time to introduce your partner to
your family, the intentions of such an introduction must be clear. Not only should you have already discussed where your relationship is headed, but you should
also tell your partner everything they need to know about your family.

“I think you should tell the person why you’re bringing them home for the holidays, so you’re on the same page about the intentions and what’s motivating the
decision to include,” she said.
</item 6>

<item 7>
Every movie and show coming to Netflix in November (Polygon, 2023-11-01T12:00:00+00:00)
Over the course of two-episodes, discover your new favorite comedian.

Available Nov. 29

American Symphony (Netflix Documentary)

Bad Surgeon: Love Under the Knife (Netflix Documentary)

From Netflix: This shocking docuseries about one of the biggest frauds in modern medical history exposes a thoracic surgeon famous for groundbreaking
achievements in the field of regenerative medicine.

Available Nov. 30

The Bad Guys: A Very Bad Holiday (Netflix Family)

From Netflix: Mr. Wolf and his crew of animal outlaws realize they’ll need to restore the city’s Christmas spirit to keep their annual holiday heist afloat.

Family Switch (Netflix Film)

From Netflix: Jess and Bill Walker are doing their best to keep their family connected as their children grow older, more independent, and more distant. When a
chance encounter with an astrological reader causes the family to wake up to a full body switch, on the morning of the most important day of each of their
lives, can the Walkers unite to land a promotion, college interview, record deal and soccer tryout?
</item 7>

<item 8>
Should you introduce your partner to your family during the holidays? Relationship experts weigh in (The Independent - Life and Style,
2023-12-12T16:02:44+00:00)
That isn’t necessarily a cause for concern, as long as you inform your significant other about the interworkings of your family dynamics.

“I think that we tend to be our best relational self when we’ve really done the work around understanding all of our past relationships,” Zar explained. “You
can tell your partner: ‘I don’t have a close relationship with my brother, and here’s why. Here are the intentional choices that I’ve made around it.’ Or you
can say: ‘Here’s my chosen family. These are the people who are important to me.’ Your significant other will see that you’ve been introspective and have
takeaways around how you’ve navigated your family’s origin.”

As long as you and your partner have discussed your relationship expectations - both before and during the holidays – introducing them to your family may deepen
your relationship and could be a step in the right direction.
</item 8>

<item 9>
The 5 biggest mistakes parents make at Christmastime, according to a parenting expert (FOX News - Lifestyle, 2023-12-09T19:46:55+00:00)
With this in mind, parents should avoid "processing their frustration around the holidays audibly around the children," she said.

"The ‘mom brain’ is wired to process problems rapidly and move on to solutions," she said. "The ‘child brain,’ however, is wired to process a problem internally
and take on the responsibility."

Said Keeley, "Knowing this, make sure you don’t do any ‘grinchy griping’ out loud to the children. You may see it as temporary venting, but they see it as, ‘Mom
is upset and I’m probably the reason.’"

CLICK HERE TO SIGN UP FOR OUR LIFESTYLE NEWSLETTER

Instead, parents should stay calm, find a reason to be grateful and vocalize it, even when frustrations accumulate.

5. Neglecting the ‘give list’

While building a wish list is fun for kids, Keeley emphasized the moral importance of the "give list."
</item 9>

<item 10>
Should you introduce your partner to your family during the holidays? Relationship experts weigh in (The Independent - Life and Style,
2023-12-12T16:02:44+00:00)
“Sometimes it can be helpful for them to know about the dynamics that they’re stepping into and how they can best support you. Frankly, they should be aware if
there is tension or if there is stress within your family.”

(Getty Images)

Communication before the holidays goes both ways, and there are things that your family should know about your significant other too. “If there’s something
about your partner that could be surprising to your family, or that’s a notable difference from who someone you’ve brought home before, tell your relatives
that,” Zar said. “Let’s set your family and partner up for success.”

Before bringing your significant other home for the holidays, you may want to express to your family how you wish for them to feel included - such as
incorporating your partner’s family traditions into your own celebration. However, Zar advised against oversharing information before Christmas. You still want
your family and partner to connect in person, rather than hearing about each other through the grapevine.
</item 10>
=== END ITEM id=ee8f4c3e09bc593f ===
