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

=== ITEM id=829000fb93198852 ===
Gold sentences and facts:
Sentence 0: Unfortunately, she did say that she had conversations with investors as part of their due diligence — and, of course, Alameda was taking on losses from FTX to keep FTX’s balance sheet pristine.
  Fact 0.0: The witness said she had conversations with investors as part of their due diligence.
  Fact 0.1: Alameda was taking on losses from FTX to keep FTX's balance sheet pristine.
Sentence 1: It was not, however, possible for those accounts to avoid liquidation, as Wang testified Alameda could do — or to have an overall negative balance.
  Fact 1.0: It was not possible for those accounts to avoid liquidation.
  Fact 1.1: Wang testified that Alameda could avoid liquidation for those accounts.
  Fact 1.2: It was not possible for those accounts to have an overall negative balance.
Sentence 2: It was a “very innovative, successful company.” And the business practices between the two were reasonable, he argued, claiming that Alameda acted legally as an FTX customer, payment processor, and market maker, or financial entity that acts as a trading partner for customers looking to buy and sell cryptocurrencies.
  Fact 2.0: The argument described Alameda as a 'very innovative, successful company'.
  Fact 2.1: The argument's speaker claimed the business practices between Alameda and FTX were reasonable.
  Fact 2.2: The argument's speaker claimed Alameda acted legally as an FTX customer.
  Fact 2.3: The argument's speaker claimed Alameda acted legally as an FTX payment processor.
  Fact 2.4: The argument's speaker claimed Alameda acted legally as an FTX market maker.
  Fact 2.5: A market maker is a financial entity that acts as a trading partner for customers looking to buy and sell cryptocurrencies.

Retrieved items:
<item 1>
Sex, Signal messages, and sabotaging FTX's code: SBF execs and Bahamas roommates tell all in court (Cnbc | World Business News Leader,
2023-10-06T20:17:00+00:00)
In June 2022, he submitted a report to Bankman-Fried on Signal that showed $8 billion in customer money held in an internal database tracking the cash wired to
an Alameda account called "fiat at ftx.com" was missing. Yedidia said he and Bankman-Fried spoke about it at the pickleball court at the resort in Nassau,
Bahamas. He asked his boss if things were OK. He was concerned because it "seemed like a lot of money" from FTX customers was at risk. "Sam said, we were
bulletproof last year. We aren't bulletproof this year," Yedidia testified. Yedidia said he asked when they would be bulletproof again. Bankman-Fried said he
wasn't sure, but it may be six months to three years. Yedidia said Bankman-Fried appeared "worried or nervous," which he said was atypical.
</item 1>

<item 2>
The FTX trial is bigger than Sam Bankman-Fried (The Verge, 2023-09-28T12:00:00+00:00)
We have some inkling of what the government will be arguing thanks to the filings prosecutors are making in this case. They argue that Bankman-Fried lied about
consumer protection and that Bankman-Fried’s statements that FTX was “avoiding or managing conflicts of interest” and that “as a general principle FTX
segregates customer assets from its own assets across our platforms” were lies, in a superseding indictment filed August 14th.

The indictment also says that there were special features in FTX’s code that “permitted Alameda to spend and withdraw unlimited amounts of money from FTX,”
which were created at Bankman-Fried’s direction. That effectively exempted Alameda from the kinds of risk management other customers faced. At the same time,
the government alleges, “Bankman-Fried publicly and repeatedly asserted that Alameda did not have privileged access to FTX.”
</item 2>

<item 3>
The FTX trial is bigger than Sam Bankman-Fried (The Verge, 2023-09-28T12:00:00+00:00)
“We had him going out directly to the internet,” LaVigne says. If he wrote things on Twitter or said things in interviews that weren’t true, that’s more fodder
for the government’s case. “They can point to that and say, ‘This is what he said, this is what actually happened.’”

Other evidence may include Signal messages and testimony from co-conspirators who plead guilty to their own charges. Alameda Research CEO Caroline Ellison, who
was also sometimes Bankman-Fried’s girlfriend, may play an important role — Bankman-Fried leaked her diaries to The New York Times and was consequently jailed
for witness tampering. The government has indicated FTX co-founder Gary Wang and engineering head Nishad Singh will also be among the witnesses called to give
testimony about Bankman-Fried.

Bankman-Fried’s defense can also introduce risks for people who dealt with him

There may be testimony from lenders, venture capitalists, and customers in order to establish the basis for some charges.
</item 3>

<item 4>
Today the FTX jury suffered through a code review (The Verge, 2023-10-06T22:21:46+00:00)
It was striking, though, that while there was evidence of the computer code that gave Alameda special privileges, there was no evidence of the conversations
Wang recounted. Of course Wang turned himself in quickly; sooner or later, someone would find his code — and Singh’s, too.

It also struck me that all of Bankman-Fried’s vanishing messages might not help him as much as he’d hoped. He said too many things in public to too many people.
Even without the smoking gun of Signal messages, it’s hard to see how he wasn’t involved. Wang was a dev; the money wasn’t his area. And while Bankman-Fried can
try to blame Ellison for some of the decisions, the code shows she couldn’t have made all the calls. After all, she wasn’t FTX’s CEO. And in 2019, she wasn’t
Alameda’s CEO, either.
</item 4>

<item 5>
Sam Bankman-Fried was a terrible boyfriend (The Verge, 2023-10-10T23:50:21+00:00)
Using the effectively unlimited funds “allowed us to make profitable trades we couldn’t have made otherwise”

Bankman-Fried didn’t grant Ellison equity, even though she asked; he told her it would be too complicated. Instead, she got a $200,000 salary, even as CEO, and
bonuses twice a year, which ranged from $100,000 to $20 million.

Initially, Alameda and FTX were “very integrated,” Ellison said. They were run by the same team, from the same office. And when Alameda was scrounging for
funds, Bankman-Fried told Ellison that FTX would be a good source of capital. The $65 billion line of credit Alameda Research had meant that it did not have to
post collateral. There was no contract and no written terms, she testified. It also wasn’t visible to FTX’s auditors — she’d raised the question with
Bankman-Fried, and he told her not to worry about it.
</item 5>

<item 6>
The FTX trial is bigger than Sam Bankman-Fried (The Verge, 2023-09-28T12:00:00+00:00)
First and foremost, they’re trying to prove their client is not guilty. But just in case they don’t get the outcome they want, they’re also laying the
groundwork for appeals and sentencing arguments. Any piece of evidence they want to introduce for those two purposes has to come into play in the trial.

Already, Bankman-Fried’s lawyers are marking arguments that his constitutional rights are being violated by his pretrial incarceration because, among other
things, his internet connection wasn’t good enough for him to plan his defense, notes LaVigne. That could be grounds for an appeal.

“Is he going to throw the entire industry under the bus?”

Bankman-Fried’s behavior after the fall of FTX suggests he’s something of a wild card. He may suggest he was acting on the advice of his lawyers. But he may
also introduce other evidence that could be troublesome — implying, for instance, that he was engaged in standard industry behavior or that everything that
happened was Binance’s fault.
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
The jury finally hears from Sam Bankman-Fried (The Verge, 2023-10-28T00:12:41+00:00)
There was a minimum of word salad today. I don’t know if he was more relaxed, or he’d just been more rehearsed, but I will certainly be watching to see if he
suddenly becomes much less coherent when Sassoon gets him for the cross.

Here is the story of FTX, from his point of view.

Bankman-Fried, who informed us he’s “somewhat introverted, naturally,” gave us a rather prolonged tour of his pre-Alameda Research life, which I will skip. In
2017, during a crypto bull run, he started his cryptocurrency trading firm. He knew “basically nothing” about cryptocurrency at the time, he explained, but he
wanted to do arbitrage on it anyway.

Alameda Research was named for Alameda County in California, which was where its first office was set up. As for its name, here’s what Bankman-Fried said on the
stand:

Effectively, we wanted to be under the radar at that point in time.
</item 8>

<item 9>
The FTX trial is bigger than Sam Bankman-Fried (The Verge, 2023-09-28T12:00:00+00:00)
Here are the hits:

Bankman-Fried gave interviews freely — and quickly rose to public prominence in the industry. Though FTX hadn’t been in the business as long as competing
exchanges such as Coinbase, Kraken, or Gemini, Bankman-Fried positioned himself as an important, boyish face for crypto. (At one point, Bankman-Fried told a
colleague at FTX that “I honestly think it’s negative EV [this may mean “expected value,” as in poker] for me to cut my hair. I think it’s important for people
to think I look crazy.”)

Because he was so successful at this kind of public relations, his fall from grace was another mark against an industry that was already roiled by bankruptcies
and scandals. Some additional trouble for the crypto industry is likely to come from one crucial element of the fraud trial — the part where the government must
prove intent.

The first part of proving the government’s case is pretty simple and a little boring: prosecutors must show that certain transactions took place.
</item 9>

<item 10>
SBF Trial: The latest updates from the FTX collapse’s courtroom drama (TechCrunch, 2023-10-06T17:22:02+00:00)
The prosecution painted a picture of an empire “built on lies” and made a point to show actual pictures of SBF alongside rich and powerful figures to illustrate
the peak of his power and influence prior to FTX’s implosion.

“This man stole billions of dollars from thousands of people,” prosecutor Thane Rehn told the court. “He defrauded sophisticated investors and lenders, and he
emptied the accounts of ordinary customers, too. He bought himself wealth, power and influence.”

The defense, on the other hand, depicted SBF and his colleagues as in over their heads, without ill-will as FTX collapsed around them.

“Sam and his colleagues were building the plane as they were flying it,” defense attorney Mark Cohen said. “They had to figure out how to navigate a world where
they were running FTX, building out its systems, dealing with hacking threats, managing the credit risk of their customers, managing hundreds of employees, all
while building up their actual exchange.”

Read a recap of the prosecution and defense’s full arguments here.
</item 10>

<item 11>
How is it still getting worse for Sam Bankman-Fried? (The Verge, 2023-10-12T23:08:14+00:00)
Prosecutor Danielle Sassoon had run an effective direct examination, creating an easy-to-follow narrative. By contrast, Cohen appeared to be bumbling around,
taking up one topic only to abruptly pivot. Sure, we’re still in the prosecution’s case, but Cohen had all night to prepare his lines of questioning.

Apparently, Alameda had a problem with retaining accountants

We established that Bankman-Fried had a much larger appetite for risk than Ellison. I thought perhaps it might be building to something, but this line of
questioning was quickly dropped. We established that Bankman-Fried and Ellison reacted differently to stress, and that they also had different approaches to
media: namely, that Ellison avoided it while Bankman-Fried sought it out. Okay?

We discovered that there was one accountant at Alameda in 2021, and two more junior accountants were hired in 2022. Apparently, Alameda had a problem with
retaining accountants, which didn’t surprise me much; CEOs generally don’t do the balance sheets for their companies.
</item 11>

<item 12>
The FTX trial is bigger than Sam Bankman-Fried (The Verge, 2023-09-28T12:00:00+00:00)
Bankman-Fried is accused of using “billions of dollars in misappropriated FTX customer deposits” to help buy more than $200 million of real estate for himself,
make billions of dollars of investments for his own interest, and repay Alameda’s lenders, according to the indictment, Additionally, Bankman-Fried allegedly
used more than $100 million of customer funds to make political contributions; prosecutors can show evidence of those contributions in this trial, even though
they aren’t part of the charges brought. A second trial is scheduled for March 2024, with additional charges.

Before his fall, Bankman-Fried made himself out to be the Good Boy of crypto — the trustworthy face of a sometimes-shady industry. He was also very interested
in publicity, sitting for many interviews both before and after the fall of FTX. The quick rise of FTX as an industry force was at least in part due to
Bankman-Fried’s appetite for attention.
</item 12>

<item 13>
Sam Bankman-Fried was a terrible boyfriend (The Verge, 2023-10-10T23:50:21+00:00)
That was the basis of the worst of her testimony — and the spreadsheet from hell.

Ellison said she’d prepared the spreadsheet at Bankman-Fried’s request in the fall of 2021 and shared it with him. The point was risk analysis around paying
back Alameda’s loans if they were abruptly recalled by Genesis, their lender. Bankman-Fried wanted to use $3 billion for venture investments, so Ellison was
ballparking what that would do to Alameda’s risk. In the as-is scenario, if things went south, she figured there was a 30 percent chance they wouldn’t be able
to meet the loan recalls. If Bankman-Fried used $3 billion to make investments, there was a 100 percent chance they couldn’t meet the recalls, even with FTX
customer funds.

Echoing Bankman-Fried and Ellison’s romantic relationship, FTX was keeping cozy private ties with Alameda yet publicly holding it at arm’s length

The problem here wasn’t really the math, which seemed pretty arbitrary.
</item 13>

<item 14>
SBF, riding high on FTX, reportedly offered $5B to Trump to cancel President re-run (TechCrunch, 2023-10-02T13:11:07+00:00)
Yet Lewis’s picture is a little less stark from the looks of the 60 Minutes episode. In fact, I’d say it’s not that clear at all if SBF is getting a kicking or
a coat of polish in this interview when you consider Lewis’s celebration of SBF’s self-proclaimed (yet not quite proven) idealism, and the “Bankman-Fried-shaped
hold in the world” that exists.

You could say that confused message is very apt for cryptocurrency, where the process by which it works is forever murky and often misrepresented, at least to
most people.

If SBF didn’t intentionally aim to cynically defraud a bunch of people, in the long run he became one of the many who disastrously failed to manage the risk
inherent in crypto speculation. That’s ultimately what happened with FTX, as financial discrepancies racked up between the exchange and its sister company
Alameda Research (a trader itself on the platform).
</item 14>

<item 15>
In the end, the FTX trial was about the friends screwed along the way (The Verge, 2023-10-26T12:00:00+00:00)
The general sense I got, from the testimony of Bankman-Fried’s former friends, was that things just kept escalating

During Yedidia’s testimony, we were shown a photograph in court: former President Bill Clinton and former Prime Minister Tony Blair onstage in suits next to
Bankman-Fried, in a rumpled T-shirt and cargo shorts. Sure, dressing down is a staple of Silicon Valley startup culture — and a sign of its swagger. The defense
counsel tried to elicit testimony from Yedidia that the cargo shorts were evidence of Bankman-Fried’s humility, but the photograph seemed to signify just the
opposite: that the man in cargo shorts is so important, he doesn’t need to dress up, even to meet former world leaders.

Now, in court, Bankman-Fried is wearing a suit.

In 2019, just months after FTX was founded, Wang oversaw code called “allow_negative” that let Alameda Research avoid being liquidated by FTX’s risk management.
It was turned on in July.
</item 15>

<item 16>
The FTX trial is bigger than Sam Bankman-Fried (The Verge, 2023-09-28T12:00:00+00:00)
According to the SEC, FTX was a fraud “from the start,” diverting customers’ funds to Alameda.

Sam Bankman-Fried, the founder of failed cryptocurrency exchange FTX, is on trial for seven counts of wire fraud and conspiracy. FTX was a fraud “from the
start,” the Securities and Exchange Commission alleges — with a “multi-billion-dollar deficiency caused by his own misappropriation of customer funds.” Follow
along for all the latest news and regular updates from the trial.

“Really just old-fashioned embezzlement.”

What’s left of FTX is now being led by John J. Ray III — you may remember him as the guy who cleaned up Enron and who also said FTX is worse than Enron. Just
before the trial began, the FTX lawyers filed suit against Bankman-Fried’s parents, saying they should give back millions of dollars they got from their kid.
Ray has also referred to Bankman-Fried’s conduct as “really just old-fashioned embezzlement.”
</item 16>

<item 17>
SBF’s trial starts soon, but how did he — and FTX — get here? (TechCrunch, 2023-10-01T14:00:29+00:00)
SBF’s trial has started, this is how he and FTX got here

The highly anticipated criminal trial for Sam Bankman-Fried, former CEO of bankrupt crypto exchange FTX, started Tuesday to determine whether he’s guilty of
seven counts of fraud and conspiracy. And as one former federal prosecutor put it: “The odds seem to be stacked against him at this point.”

The 31-year-old co-founded FTX in 2019; within a few years the once third-largest crypto exchange’s valuation hit $32 billion at its peak. It’s now trying to
claw back any funds to distribute to creditors.

But how did the once third-largest crypto exchange get here?

Before FTX, Bankman-Fried co-founded crypto-trading firm Alameda Research in 2017. He co-founded FTX in 2019 as a complement to Alameda, to help bring in
revenue and liquidity for the trading arm.
</item 17>

<item 18>
The jury finally hears from Sam Bankman-Fried (The Verge, 2023-10-28T00:12:41+00:00)
“I worried if I didn’t, I would lose touch with the actual concerns of the customers,” he testified. What he did not do was create a risk team, which he is now
characterizing as a “big mistake.”

Risk is an inherent part of a futures exchange, which is even more like a casino than regular cryptocurrency. Not having a risk team, when you are any kind of
financial anything, is certainly a choice. It is especially a choice when you go around telling everyone your crypto exchange is very good and safe.

FTX’s big selling point was its “risk engine,” which was supposed to prevent big losses that would then be spread around all the rest of the customers. But
Bankman-Fried testified that in 2020, the “risk engine was effectively sagging under the weight” of the exchange’s rapid growth. So its time to liquidation went
up — it took minutes to determine which accounts needed to be liquidated.
</item 18>

<item 19>
In the end, the FTX trial was about the friends screwed along the way (The Verge, 2023-10-26T12:00:00+00:00)
A substantial chunk had been spent on various investments, including in the celebrity-connected venture firm K5, trading firm Modulo (co-founded by one of
Bankman-Fried’s ex-girlfriends), and AI company Anthropic.

Another $413 million had gone directly to Bankman-Fried himself, through his wholly-owned company Paper Bird. “Outflows to insiders,” which suggested Singh’s
not-quite-loans were included, was $192 million; real estate was another $228 million; and the donations to Gabe Bankman-Fried’s Guarding Against Pandemics,
combined with the philanthropy from the FTX Foundation, totaled less than $40 million.

It is easy to lie with words and harder to lie with money.
</item 19>

<item 20>
Sam Bankman-Fried was a terrible boyfriend (The Verge, 2023-10-10T23:50:21+00:00)
It was that Ellison’s calculations assumed Alameda could borrow $1.8 billion in normal dollars and $1.5 billion in crypto from FTX. The spreadsheet makes this
clear with a row labeled “FTX borrows,” which Ellison said were customer funds.

Meanwhile, echoing Bankman-Fried and Ellison’s romantic relationship, FTX was keeping cozy private ties with Alameda yet publicly holding it at arm’s length. On
January 14th, 2022, Bankman-Fried tweeted, “We’re launching a $2b venture fund: FTX Ventures!” Those funds came from Alameda, Ellison testified. But
Bankman-Fried didn’t want to go public with the source of the funds. He said he thought Alameda’s brand was less good, and he didn’t want his name associated
with it. Alameda also bought Robinhood shares for Bankman-Fried, who moved them to a vehicle called “Emergent Fidelity Technologies” to avoid association with
Alameda.
</item 20>

<item 21>
Today the FTX jury suffered through a code review (The Verge, 2023-10-06T22:21:46+00:00)
Wang said he trusted Bankman-Fried’s judgment and didn’t pursue the issue further. Later, though, Alameda’s balance was more negative than FTT and the trading
revenue, and Wang had more conversations with Bankman-Fried about it. Wang said he knew at the time that Alameda was using FTX’s customers’ funds and that he
knew it was wrong.

We heard yesterday about the line of credit Alameda had on FTX, a staggering $65 billion. Today we saw it in the database. The line of credit didn’t start that
big — Wang testified that Bankman-Fried had asked him to increase it a few times because Alameda kept running into its limits. First it was “a few hundred
million,” then a billion, then even that wasn’t enough. It wasn’t clear how Wang arrived at $65 billion specifically, but he said he’d discussed the number with
Bankman-Fried. Other customers didn’t have the same privileges.
</item 21>

<item 22>
SBF’s trial starts soon, but how did he — and FTX — get here? (TechCrunch, 2023-10-01T14:00:29+00:00)
U.S. Attorney Damian Williams called Bankman-Fried’s alleged crimes “one of the biggest financial frauds in American history,” in a press conference.

The aftermath

FTX co-founder and former CTO Gary Wang, and Alameda Research’s former CEO, Caroline Ellison, both pleaded guilty in December 2022 to federal criminal charges
in relation to the FTX collapse. They’re also facing civil penalties from the U.S. Securities and Exchange Commission (SEC) and Commodity Futures Trading
Commission (CFTC) alongside the criminal charges. Wang and Ellison plan to cooperate with prosecutors and will be major witnesses in the trial, given their
close ties to Bankman-Fried, FTX and Alameda.

In January, Bankman-Fried pleaded not guilty to all counts, which include wire fraud, conspiracy to commit money laundering and conspiracy to misuse customer
funds. He could face up to 115 years in jail if convicted on all charges.
</item 22>

<item 23>
The jury finally hears from Sam Bankman-Fried (The Verge, 2023-10-28T00:12:41+00:00)
Bankman-Fried said he worked 12 to 22 hours a day, and took one day off every couple of months. Because FTX had grown so much, he could no longer run both
companies, he said. Bankman-Fried handed the company off to Caroline Ellison and Sam Trabucco, who immediately after being named co-CEO promptly drifted away to
early retirement. (Quiet quitting king!) Bankman-Fried did remain involved in hedging and risk at Alameda, though.

About that $1 billion of revenue in 2021: Bankman-Fried definitely did not know that Singh, his employee, had backdated interest payments to get FTX “over the
line” to $1 billion. See, he’d just asked his employees to check and see if there was any source of funds that was missing to get to $1 billion. This testimony
was especially rambling.

Oh, also that MobileCoin loss? The one Wang said Alameda took to keep off FTX’s balance sheet?
</item 23>

<item 24>
The jury finally hears from Sam Bankman-Fried (The Verge, 2023-10-28T00:12:41+00:00)
As a result, at one point the risk engine got stuck in a catastrophic feedback loop that would have created losses in the “trillions of dollars,” Bankman-Fried
testified. As part of that feedback loop, Alameda teetered on the brink of liquidation, which “would have disastrous consequences” for FTX.

“At the time, I wasn’t entirely sure what was happening.”

Because of that experience, Bankman-Fried suggested an “alert” or “delay” that would keep Alameda from being liquidated by a bug. This is the supposed origin
story of “allow_negative,” which Bankman-Fried says was the eventual result of that conversation, and that he says he didn’t know about until very recently.

There is a problem with this story. “Allow_negative” was coded and switched on in 2019. I saw the code in court, and so did Bankman-Fried, who was also there
for the testimony. Perhaps you are wondering, was the jury also there? Reader, it was.
</item 24>

<item 25>
SBF’s trial starts soon, but how did he — and FTX — get here? (TechCrunch, 2023-10-01T14:00:29+00:00)
That means it purported to have more FTT tokens on its balance sheet than what existed in the world.

Around the same time it was exposed, the world’s largest crypto exchange, Binance, started pulling out its remaining $2.1 billion equivalent of cash in BUSD and
FTT. (It had an equity position in FTX from 2019 to 2021.) This essentially triggered a bank run on FTX.

FTX and Alameda filed for Chapter 11 bankruptcy in the U.S. mid-November 2022. Bankman-Fried resigned, and John J. Ray III, the Enron turnaround veteran, was
appointed its new CEO.

Bankman-Fried, however, maintained his innocence. At The New York Times’ DealBook Summit, he appeared virtually from the Bahamas, saying “I didn’t ever try to
commit fraud on anyone; I was shocked by what happened this month.”
</item 25>

<item 26>
How is it still getting worse for Sam Bankman-Fried? (The Verge, 2023-10-12T23:08:14+00:00)
I was ready to hear this pursued further — but then it, too, was dropped.

I think the defense was also trying to suggest that the government had coerced Ellison’s testimony, by suggesting that she had pleaded guilty to a charge of
defrauding investors that she couldn’t have been involved in. After all, she didn’t prepare materials for them. Unfortunately, she did say that she had
conversations with investors as part of their due diligence — and, of course, Alameda was taking on losses from FTX to keep FTX’s balance sheet pristine. This
line of questioning felt like a waste of time.

There were rather a lot of sidebars during the cross-examination, to the point that when one occurred, several jurors looked entertained. There were a few
yesterday, too, including one in which the prosecution complained that Bankman-Fried was visibly scoffing at Ellison’s answers, according to the court
transcript.
</item 26>

<item 27>
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
</item 27>

<item 28>
SBF Trial: The latest updates from the FTX collapse’s courtroom drama (TechCrunch, 2023-10-06T17:22:02+00:00)
Read a recap of the prosecution and defense’s full arguments here.

FTX’s co-founder and CTO Gary Wang testified Thursday on the extent to which Alameda Research was used for whatever purposes he and SBF saw fit. The crypto
trading firm, according to Wang, pulled funding directly from FTX customers, whose transactions would be funneled toward Alameda and then directed elsewhere.

Check here for our rundown of Wang’s testimony so far, resulting from a guilty plea in December 2022.

Thursday also brought in testimony from Matt Huang, co-founder and managing partner at crypto investment firm Paradigm. That firm invested $278 million in FTX
across 2021 and 2022, and is part of a class-action lawsuit accusing it and others of defrauding customers by promoting FTX.

His testimony centered around being left uninformed about FTX’s utilization of customer money to keep the Alameda Research afloat, a practice that Huang says
would have halted his involvement.
</item 28>

<item 29>
Today the FTX jury suffered through a code review (The Verge, 2023-10-06T22:21:46+00:00)
When FTX went down, Wang left himself with no plausible deniability. I cannot for the life of me understand why.

Normal customers — the defense objected to “normal” but was overruled — could be automatically liquidated

Today, the prosecution grilled Wang on a single, damning column in FTX’s databases. Called “allow_negative,” it allowed Alameda to have a negative balance.
Alameda could withdraw money even when its accounts didn’t have any, and it had an enormous line of credit. Wang added that it could place orders faster than
other users, and we already saw evidence that users may have unwittingly deposited money into Alameda rather than FTX.

The alleged motivation for this — as Wang had to explain at painful length to a jury — was avoiding liquidation. Liquidation is a way for FTX to manage risk
when people were engaging in risky bets on its platform. The cartoon version of futures trading is this: a futures exchange is the middleman that lets strangers
bet with each other.
</item 29>

<item 30>
The jury finally hears from Sam Bankman-Fried (The Verge, 2023-10-28T00:12:41+00:00)
I didn’t want to call it Sam’s Crypto Trading Firm or anything like that. We — there are a lot of competitors and people who we didn’t particularly want to know
what we were building out because they would race to do it. “Research” was a sort of generic word, which filled out the company name. And that was — it was far
better than the internal name that we had at that point, which was Wireless Mouse.

I would find this much more believable if I hadn’t already watched a video of Bankman-Fried explaining on a podcast that the name made it easier to get a bank
account. That happened during the first day of Gary Wang’s testimony. Bankman-Fried was there, too. You know who else was there? The jury.

Anyway, Bankman-Fried went on a hiring spree for Alameda. He rounded up his merry gang of alleged co-conspirators. First, Wang, to program the computers. Then,
Nishad Singh, about a month after founding Alameda Research.
</item 30>

<item 31>
SBF’s trial starts soon, but how did he — and FTX — get here? (TechCrunch, 2023-10-01T14:00:29+00:00)
In a published DM exchange with a Vox reporter, he said he regretted filing for bankruptcy and thought that “regulators make everything worse.”

SBF arrested

Bankman-Fried was arrested in December 2022 in the Bahamas, where FTX was based. He was then extradited to the U.S. to face a number of criminal charges. He was
released on a $250 million bail bond, and he remained under house arrest at his parents’ home in Palo Alto. This was revoked in August after he was accused of
intimidating Alameda’s former CEO, Caroline Ellison, by leaking her private diary.

Ray represented the company during a House Financial Services Committee hearing regarding FTX. When asked whether the firm had significant risk management
systems, Ray said at the time that “there were virtually no internal controls and no separateness whatsoever” and added that he did not “trust a single piece of
paper” in the exchange’s organization. U.S.
</item 31>

<item 32>
Is Sam Bankman-Fried’s defense even trying to win? (The Verge, 2023-10-04T23:02:14+00:00)
Except, uh, he also said this: FTX “didn’t have a chief risk officer, which became an issue when the storm hit.”

I couldn’t stop thinking about the missing risk officer

The problem with this metaphor is that if FTX was a plane, it was a plane flying with a key component missing — namely, the risk officer, an executive whose job
it is to, well, manage risk. This is sort of an important thing, as risks can be anything from reputational to regulatory to financial.

FTX was named such as it was because it was a futures exchange, which, to borrow a phrase from Bloomberg’s Matt Levine, “sits between the winners and losers of
bets.” That means FTX can’t pay out what it owes the winners unless the losers pay up. Risk management is a crucial part of the business; risk officers exist to
identify business’ potential risks, monitor, and mitigate them. This is to say nothing of the regulatory risks around crypto.
</item 32>

<item 33>
Is Sam Bankman-Fried’s defense even trying to win? (The Verge, 2023-10-04T23:02:14+00:00)
“Sam acted in good faith and took reasonable business measures” is a pretty hard pill to swallow with that in mind.

Man, it’s no good when your defense lawyer has just made you sound worse than the prosecution already did. And while Cohen tried to make the common white-collar
defense argument that Bankman-Fried, as CEO, was simply too busy to oversee what everyone did every day, he just made me more suspicious. That’s why you hire a
risk officer and delegate! That’s the whole point! I could barely even hear Cohen blaming Caroline Ellison and Changpeng “CZ” Zhao for the debacle over the “no
risk officer” ringing in my ears.

Following the defense’s opening statements, things got still worse for Bankman-Fried. The prosecution called its first witness, Marc-Antoine Julliard, whose
money got stuck on FTX. Juilliard, who was born in Paris and lives in London, testified that he trusted FTX because Bankman-Fried came across as a leading
figure of the industry.
</item 33>

<item 34>
The jury finally hears from Sam Bankman-Fried (The Verge, 2023-10-28T00:12:41+00:00)
The one Wang said Alameda took to keep off FTX’s balance sheet? Yeah, so it was a totally innocent thing where what actually happened was that Bankman-Fried
thought it was appropriate that Alameda take the position as a backstop liquidity provider, that’s all.

In June 2022, Bankman-Fried heard about the account called “fiat@ftx” tracking how much money Alameda owed to FTX, he testified. He did not know what it was and
did not bother to find out. He was busy! That was when Bankman-Fried directed Ellison to repay Alameda’s lenders, because he thought Alameda was good for it. He
also gave BlockFi and Voyager, two crypto lenders, some capital infusions for good measure.

He was “very surprised!”

Remember that testimony Adam Yedidia gave about a conversation with Bankman-Fried in August 2022 about the enormous amount of money Alameda owed FTX? Well,
Bankman-Fried remembers it differently.
</item 34>

<item 35>
Sam Bankman-Fried didn’t ask where the $8 billion went (The Verge, 2023-10-31T19:52:07+00:00)
Instead, Bankman-Fried directed alleged co-conspirator Caroline Ellison to repay third-party loans and went on making investments.

Bankman-Fried said that he found out about the fiat@ftx account — that’s the one that tracked how much Alameda owed FTX — in June 2022 while his senior staff
was focused on fixing the software bug. He did not, however, discover what the account was for until October. I know this sounds unbelievable, but this is his
actual testimony.

His employees told him “they were busy and I should stop asking questions because it was distracting.”

Simple questions nailed Bankman-Fried to the wall. Had he given his employees at Alameda the direction not to spend FTX customer deposits? Had he put in any
policies to prevent Alameda employees from spending FTX customer money? Did he put in place any measures at Alameda to protect the FTX customer money? No, no,
and no. Oh, but he was testifying to Congress about keeping customer funds safe.
</item 35>

<item 36>
Today the FTX jury suffered through a code review (The Verge, 2023-10-06T22:21:46+00:00)
They created a spreadsheet to try to figure out how negative Alameda was. The spreadsheet we saw had several pages of attempts to reconcile. Wang figured that
Alameda owed FTX about $11 billion. FTX’s revenue at the time was about $1.5 billion.

Ellison, Singh, Wang, and Bankman-Fried met in the Bahamas office. Alameda’s lenders wanted their money back, and Bankman-Fried directed the group to pay back
the lenders using FTX customer funds. Bankman-Fried considered shutting down Alameda, and even wrote a long memo about it. But Wang says that in a now-deleted
Signal discussion about the memo, he told Bankman-Fried and Singh that they could not shut down Alameda because they couldn’t repay its debt.

Another empty Signal chat, along with its setting to delete messages after a week, was shown to the court. Its members were Ellison, Singh, Wang, and
Bankman-Fried.

Things happened quickly for Wang in November.
</item 36>

<item 37>
The FTX trial is bigger than Sam Bankman-Fried (The Verge, 2023-09-28T12:00:00+00:00)
If it seems like his peer group supported him, or worse, fawned over him, that’s a real problem for the industry’s reputation.”

After FTX declared bankruptcy, Bankman-Fried went on an extensive media tour, doing interviews with The New York Times’ Andrew Ross Sorkin and Good Morning
America. Even under house arrest, he did interviews, including with The New Yorker. Bankman-Fried seemed to think that if he just explained things, everyone
would understand that what he did was a mistake, not a crime.
</item 37>

<item 38>
In the end, the FTX trial was about the friends screwed along the way (The Verge, 2023-10-26T12:00:00+00:00)
Wang trusted him.

Wang also set Alameda Research up with an enormous line of credit, unmatched by anyone else on FTX. Ultimately, the number was $65 billion — the result of
repeated increases as Alameda Research repeatedly ran up against its credit limit. This line of capital wasn’t visible to auditors, Ellison testified. That
credit line was used primarily for trading because it “allowed us to make profitable trades we couldn’t have made otherwise,” Ellison testified.

“Over time, it was something I became more comfortable with.”

Wang’s testimony revealed how the fraud got bigger, little by little. First, it was okay to dip into FTX’s profits. Then, it was okay to take more. Then, it was
okay to take even more. First, it was a little line of credit just to provide liquidity. Then a bigger one. Then an effectively limitless one.
</item 38>

<item 39>
SBF Trial: The latest updates from the FTX collapse’s courtroom drama (TechCrunch, 2023-10-06T17:22:02+00:00)
He, like many of SBF’s former colleagues, opted to cooperate with authorities and testified that SBF asked him to create a theoretical “legal justification” for
the use of billions in consumer funds just days before FTX filed for bankruptcy.

Robert Boroujerdi, managing director of the hedge fund Third Point, testified about his experience working with SBF and FTX, in a partnership that ultimately
resulted in a $60 million investment and a lack of awareness about the special relationship between Alameda Research and FTX. Had he known of such a
relationship, he testified that Third Point would not have invested in FTX in the first place.

And earlier in the week, a University of Notre Dame professor who helped the government prosecution of Enron and WorldCom, Peter Easton, testified about what he
uncovered after tracing the flow of billions of dollars between Alameda Research and FTX.
</item 39>

<item 40>
Sam Bankman-Fried didn’t ask where the $8 billion went (The Verge, 2023-10-31T19:52:07+00:00)
I have come to believe that if you know the meaning of the word “epistemology,” you absolutely should not testify in your own defense

“So it’s your testimony that your supervisees told you to stop asking questions?” Sassoon asked. She could have been filing her nails, her tone was so level.
Had Bankman-Fried called anyone in to ask who spent $8 billion? “I wasn’t trying to build out blame for it,” he said. He was focused on solutions! Did he fire
anyone? Nope!

We also saw an Alameda balance sheet from June 13th, 2022, that included the money it was borrowing from FTX as “ftx borrows.” Bankman-Fried seemed pretty hazy
on this, too.

If you are wondering how Bankman-Fried’s parents reacted to this, I can’t tell you — they weren’t there. I couldn’t really blame them. I wouldn’t want to watch
my child be vivisected, either.
</item 40>

<item 41>
The FTX trial is bigger than Sam Bankman-Fried (The Verge, 2023-09-28T12:00:00+00:00)
That may be risky, but we already know that Bankman-Fried loves risk.

“Is he going to throw the entire industry under the bus?” Wong asks. “An idea like, ‘Everyone was doing this, it’s not fair I’m the only one who was charged?’”
That may not fly in a court of law, but it could absolutely damage public perception of crypto at large.

For instance, just before he was arrested, Bankman-Fried brought forward messages from a crypto group chat in testimony he planned to give before Congress. In
that testimony, he primarily blames Binance and his lawyers at the firm Sullivan & Cromwell.

It’s possible other messages from group chats may be brought into evidence during the trial. That can potentially be embarrassing for the whole industry, Wong
says.

“You can imagine some of these founders, CEOs, people of that echelon talk to each other somewhat informally about what’s going on,” she says.
</item 41>

<item 42>
In the end, the FTX trial was about the friends screwed along the way (The Verge, 2023-10-26T12:00:00+00:00)
It is easy to lie with words and harder to lie with money. Bankman-Fried went around telling people that he wanted to make a lot of money to give it away, an
idea of the Effective Altruism movement that Bankman-Fried affiliated himself with — but the amount of money he funneled to his Paper Bird vehicle alone was ten
times the size of his philanthropic spending as outlined by Easton. And it was all vastly outweighed by his investments — the places where he could prove he was
very smart by investing in the right companies early.

And in brief, explosive testimony, former FTX lawyer Can Sun demonstrated what Bankman-Fried’s tactic was. During the days in November when the exchange
imploded, Bankman-Fried was trying to raise money from Apollo, the private equity group. Apollo wanted an explanation for what had happened, and so
Bankman-Fried came to Sun to ask him to come up with justifications for the missing funds.
</item 42>

<item 43>
SBF, riding high on FTX, reportedly offered $5B to Trump to cancel President re-run (TechCrunch, 2023-10-02T13:11:07+00:00)
That would be a wonderful detail to get teased out in the trial, though.

And lest you think Biden and the Democrats have their hands clean of SBF… think again: He also looked to back candidates on that side of the aisle. You never
know when you might need a friend!

The hard crash

All of that activity, of course, came to a crashing end: The value of cryptocurrencies on FTX was not banked to much more than speculation and the promise of
higher value, and so when those valuations fell, they fell very, very hard.

What we already knew was that SBF has been denied bail; his name is mud (or much worse) among those who have lost money in Alameda or FTX; and those who might
not have had positions in these still blame him for other crypto woes because he set off a ripple effect.

“He doesn’t come off good at all here. You can tell he had no experience in management or handling such big undertakings.
</item 43>

<item 44>
Sam Bankman-Fried didn’t ask where the $8 billion went (The Verge, 2023-10-31T19:52:07+00:00)
But telling your side of the story opens you to a great many questions you might not want to answer on cross-examination. I have watched a lot of crosses. This
was the nastiest I think I’ve ever seen.

The charges in this case hinge on conspiracy and intentional deceit. Just losing $8 billion is not a crime, though it is very embarrassing. (Even losing $900
million is very embarrassing because people will make jokes about it forever!) But if Bankman-Fried lied to customers and lenders about what he was doing and
how safe FTX was, that’s a crime.

Establishing that would have been enough, but Sassoon also managed to get in some flourishes about Bankman-Fried’s relationships to Bahamian power. For
instance, she asked him if he’d made comments about paying off the Bahamian national debt. (You will be shocked, shocked to discover he didn’t recall.)
</item 44>

<item 45>
Sam Bankman-Fried didn’t ask where the $8 billion went (The Verge, 2023-10-31T19:52:07+00:00)
Oh, but he was testifying to Congress about keeping customer funds safe.

Sam Bankman-Fried, the founder of failed cryptocurrency exchange FTX, is on trial for seven counts of wire fraud and conspiracy. FTX was a fraud “from the
start,” the Securities and Exchange Commission alleges — with a “multi-billion-dollar deficiency caused by his own misappropriation of customer funds.” Follow
along for all the latest news and regular updates from the trial.

Who was making decisions to spend $8 billion of customer funds? Bankman-Fried couldn’t recall knowing anything about it. Were there rules or requirements for
how money borrowed from FTX would be returned? Were there rules for risk management? “I was concerned with overall risk management,” Bankman-Fried said.

But it was the testimony about June 2022 that resonated the most to me. Didn’t Bankman-Fried ask what “fiat@ftx” was? He did.
</item 45>

<item 46>
Sex, Signal messages, and sabotaging FTX's code: SBF execs and Bahamas roommates tell all in court (Cnbc | World Business News Leader,
2023-10-06T20:17:00+00:00)
Ellison is expected to take the stand under a similar arrangement.

U.S. District Judge Lewis Kaplan presides as Gary Wang testifies during the fraud trial of Sam Bankman-Fried over the collapse of FTX, the bankrupt
cryptocurrency exchange, at Federal Court in New York City, U.S., October 6, 2023 in this courtroom sketch. Jane Rosenberg | Reuters

Born in China, Wang moved to the U.S. at age 7, and grew up in Minnesota before going to the Massachusetts Institute of Technology to study math and computer
science. He worked at Google after college. Wang, who first met Bankman-Fried during high school at a summer camp, owned 10% of Alameda, while his boss owned
the other 90%. Wang told the court about the advantages that Alameda received by having code baked into FTX's software that allowed special access to the crypto
exchange.
</item 46>

<item 47>
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
</item 47>

<item 48>
Sam Bankman-Fried was a terrible boyfriend (The Verge, 2023-10-10T23:50:21+00:00)
FTX was a fraud “from the start,” the Securities and Exchange Commission alleges — with a “multi-billion-dollar deficiency caused by his own misappropriation of
customer funds.” Follow along for all the latest news and regular updates from the trial.

The day started off promisingly for the defense as it cross-examined Gary Wang, the chief technology officer of FTX and co-owner of both FTX and Alameda.
Christian Everdell, one of Bankman-Fried’s defense attorneys, couldn’t undo the damage of last week’s code review. But he managed to shake the rust off long
enough to make Wang sound less reliable, drowning the jury in confusing technicalities.

Last week, Wang testified that Alameda got access to a special credit line and an option to take its balance into the negative without triggering liquidation —
something he alleged other accounts at FTX didn’t get. Everdell tried to undermine this claim by talking about the spot margin program, which let users lend
each other assets for margin trading.
</item 48>

<item 49>
Today the FTX jury suffered through a code review (The Verge, 2023-10-06T22:21:46+00:00)
Other customers didn’t have the same privileges.

Alameda also allegedly kept FTX from looking bad by assuming some of its losses. In 2021, FTX was facing a “several hundred million dollar” loss from an exploit
in MobileCoin, Wang said. (This seems to have happened in April 2021, though Wang didn’t give a month.) Bankman-Fried directed him to have Alameda take it on
instead, because FTX’s balance sheets were visible to investors and Alameda’s weren’t.

Yesterday, Paradigm Capital’s Matt Huang showed us the balance sheets he received from Bankman-Fried. They didn’t include the loss, which should have been
included in the second quarter numbers he’d received — they showed $63 million in trading expenses and $63 million in “other expenses” in that quarter, and an
estimated profit of $732 million.

“FTX was not fine and assets were not fine.”

By 2022, Alameda’s negative balance on FTX was a cause for concern for Caroline Ellison, Singh, and Wang.
</item 49>

<item 50>
How is it still getting worse for Sam Bankman-Fried? (The Verge, 2023-10-12T23:08:14+00:00)
The recording was later sent to Drappi, who sent it to the government. Joseph Bankman, the defendant’s father and a senior advisor to FTX’s philanthropic arm,
wasn’t in the room for the recording. Barbara Fried looked unhappy, rubbing her left temple as though she had a tremendous headache.

And in a small miracle of pacing, the government got Zac Prince, the founder and former CEO of crypto lender BlockFi, on the stand just long enough to blame
FTX’s bankruptcy for BlockFi’s subsequent bankruptcy before we broke for the day.

I’ve been asked by some people why the reporting coming out of the trial seems so skewed toward the prosecution. There’s an answer for this: the prosecution has
put together a strong, comprehensive indictment of Bankman-Fried’s behavior at FTX and Alameda. The defense, so far, has managed to do almost nothing in
response.
</item 50>

<item 51>
Is Sam Bankman-Fried’s defense even trying to win? (The Verge, 2023-10-04T23:02:14+00:00)
“All of that was built on lies,” Rehn said.

In his opening statement, Rehn dodged explaining cryptocurrency to the jury. Instead, he punched hard on Bankman-Fried lying and stealing.

Bankman-Fried sat almost motionless, occasionally glancing at Rehn, as the prosecutor told the jury that Bankman-Fried sold stock in FTX and borrowed millions
from lenders by lying.

The story Rehn told is familiar to anyone following the news. In May and June of 2022, Alameda Research — the crypto trading company ostensibly helmed by
Caroline Ellison — didn’t have enough to pay its bills, so it pulled customer money to repay loans. By September, the hole in the FTX balance sheet was so big
that customers could never be repaid.

FTX “didn’t have a chief risk officer, which became an issue when the storm hit.”

When CoinDesk published its article in November 2022, people realized FTX was a house of cards, Rehn said. Meanwhile, Bankman-Fried tweeted.
</item 51>

<item 52>
In the end, the FTX trial was about the friends screwed along the way (The Verge, 2023-10-26T12:00:00+00:00)
Then a bigger one. Then an effectively limitless one.

Ellison testified to something similar, choking back tears: “When I started working at Alameda, I don’t think I would have believed you if you told me I would
be sending false balance sheets to our lenders or taking customer money, but over time, it was something I became more comfortable with.”

Ellison’s tenure as CEO of Alameda Research was an exercise in optics, she testified. Alameda Research was, by then, FTX’s sin-eater. When FTX took a loss in
the hundreds of million dollars because a trader had exploited its system, Wang moved that loss onto Alameda’s balance sheet. FTX’s “insurance fund,” meant to
keep traders from getting stuck with other traders’ losses, was totally made up — Alameda simply picked up the tab.

Even that wasn’t good enough.
</item 52>

<item 53>
Is Sam Bankman-Fried’s defense even trying to win? (The Verge, 2023-10-04T23:02:14+00:00)
This is to say nothing of the regulatory risks around crypto.

As Cohen droned on about airplanes, I couldn’t stop thinking about the missing risk officer. Bringing it up, I thought, was a tremendous mistake. The
prosecution hadn’t mentioned it. Either Bankman-Fried is stupid — unlikely — or he deliberately didn’t hire a risk officer. Was he worried about what one might
find?

Sure, as Cohen put it, Bankman-Fried was a math nerd who didn’t party. That paints a picture of someone who’s pretty deliberate, particularly since he
immediately left MIT and went to work on Wall Street. If he had been a party-hardy trainwreck, I could see overlooking a risk officer in order to do another
line, or a supermodel, or something else important. Why was the defense bringing this up?

But as Cohen tried to tell me that FTX’s and Alameda’s business relationships were “reasonable under the circumstances,” the lack of risk officer kept elbowing
me in the ribs.
</item 53>

<item 54>
Today the FTX jury suffered through a code review (The Verge, 2023-10-06T22:21:46+00:00)
As a result of today’s testimony, I am both more and less confused. I am clearer on what probably happened at FTX. What I don’t understand is why Gary Wang did
it.

Wang was the co-founder and chief technology officer of FTX, as well as the co-founder of Alameda Research. He owned only 10 percent of Alameda and 17 percent
of FTX. He got a $200,000 a year salary, with no performance bonuses. (Okay, sure, at times he was a billionaire on paper.) By contrast, Bankman-Fried owned 90
percent of Alameda and 65 percent of FTX.

Yet Wang’s testimony suggests he took bizarre risks for his own relatively small share. Bankman-Fried didn’t code, so none of his fingerprints are on the
technical changes that let FTX allegedly defraud its customers and investors. Meanwhile, Wang supervised the code behind a key prosecution claim: that Alameda
Research was afforded secret, special privileges with FTX’s funds.
</item 54>

<item 55>
Sex, Signal messages, and sabotaging FTX's code: SBF execs and Bahamas roommates tell all in court (Cnbc | World Business News Leader,
2023-10-06T20:17:00+00:00)
He spoke so fast that U.S. District Judge Lewis Kaplan and the prosecutor both stopped him at points to ask that he slow his pace. Much of Wang's testimony on
Friday focused on the final days at FTX before the entire operation imploded, including reports in the media detailing Alameda's business practices and its
troubling ties to FTX. Wang said that in response to the reporting an emergency meeting was called between Bankman-Fried, Wang and Singh, to discuss shutting
down Alameda. He said they ultimately decided against such a move, because he and Bankman-Fried were aware that Alameda had no way to repay the roughly $14
billion hole in its books. Prosecutors took the jury through a series of tweets, beginning on Nov. 7. Posts came from the company blaming bank hours for slow
withdrawals, while Bankman-Fried tweeted from his personal account, assuring customers that all was fine.
</item 55>

<item 56>
The jury finally hears from Sam Bankman-Fried (The Verge, 2023-10-28T00:12:41+00:00)
Then, Nishad Singh, about a month after founding Alameda Research. Finally, Caroline Ellison.

Not having a risk team, when you are any kind of financial anything, is certainly a choice

Though Bankman-Fried was the CEO, and also the majority owner, he wanted to be clear: he did not supervise Wang’s direct work. Anyway, after a bunch of wildly
successful arbitrage — 50 percent to 100 percent annualized returns, per his testimony — he decided to found a cryptocurrency exchange, FTX. He figured he’d
fail; that there was only a 20 percent chance of success. Bankman-Fried did not define what he expected the time period to be on this estimate, but arguably 20
percent was a much higher chance of success than FTX would enjoy once Alameda dipped into the customer deposits.

By the way, because he was such a good guy, Bankman-Fried made a point of “periodically” handling support tickets himself.
</item 56>

<item 57>
Sex, Signal messages, and sabotaging FTX's code: SBF execs and Bahamas roommates tell all in court (Cnbc | World Business News Leader,
2023-10-06T20:17:00+00:00)
Those privileges ultimately resulted in Alameda owing FTX $8 billion worth of customer deposits. "We gave special privileges on FTX that gave unlimited
withdrawals on the platform to Alameda," Wang said. Alameda was allowed to withdraw and transfer those funds and had a $65 billion line of credit. "When
customers deposited USD, it went to Alameda," he said. "It existed in the computer code. Alameda could have negative balances and unlimited withdrawals." That
"bug" in the code was written by Nishad Singh, who was FTX's director of engineering, and reviewed by Wang. Bankman-Fried was calling the shots, Wang said. Wang
also told the court about a $1 million personal loan he received and a $200 million to $300 million loan in his name from Alameda that was never deposited into
his account, but rather was used to make investments into other companies on behalf of FTX.
</item 57>

<item 58>
The FTX trial is bigger than Sam Bankman-Fried (The Verge, 2023-09-28T12:00:00+00:00)
Whatever records the Southern District of New York has for the transactions will be shown.

“What conversations happened between him and his co-conspirators that are now cooperating against him?”

The second part is where all the drama is likely to come, says Christopher LaVigne, a litigation partner and co-chair of the cryptocurrency practice at the law
firm Withers. Prosecutors have to connect those transactions to Bankman-Fried, show that he knew what he was doing was wrong, and prove that he lied about it
anyway.

“What was he saying to his parents and his other advisors about this?” LaVigne says. “What conversations happened between him and his co-conspirators that are
now cooperating against him?”

To further establish intent, the government can use Bankman-Fried’s own words. The indictment calls Bankman-Fried’s tweets in November 2022 “false and
misleading.”

“We had him going out directly to the internet,” LaVigne says.
</item 58>

<item 59>
Today the FTX jury suffered through a code review (The Verge, 2023-10-06T22:21:46+00:00)
Bankman-Fried and his father met with lawyers while Wang waited outside with liquidators. The US lawyers wanted the remaining FTX assets transferred to the US,
Wang said.

That is not what he did. Bankman-Fried told Wang to stall the US lawyers and transfer the funds to Bahamanian regulators, who were more likely to let him stay
in control of the company, Wang testified.

On November 16th, Wang returned to the US and on November 17th, he met with the US government. Wang hadn’t been arrested or charged; he offered to cooperate
immediately. He pled guilty to various charges a month later. Wang was clear: he was cooperating because he didn’t want to go to jail. He was hoping that his
assistance means that a sentencing judge will be lenient, he said.

This was the first full account of Alameda’s and FTX’s entanglements I heard in the trial, and it came from a co-owner of both companies.
</item 59>

<item 60>
In the end, the FTX trial was about the friends screwed along the way (The Verge, 2023-10-26T12:00:00+00:00)
In it, Bankman-Fried tried to justify the missing funds through the margin loan program. It was such a flimsy excuse that even George Stephanopoulos could see
through it.

Did Bankman-Fried really think he could get away with it all? He must still think so, or he would not be dragging his closest companions’ names through the mud
in a jury trial. Throughout the entire prosecution’s case, the defense has struggled to shake the credibility of witnesses such as Yedidia, Wang, and Ellison.
The hits the defense made on Singh didn’t make me doubt the overall truth of his story — they just made me think that Singh was kind of self-serving and trying
to secure himself a lenient sentence.
</item 60>

<item 61>
SBF Trial: The latest updates from the FTX collapse’s courtroom drama (TechCrunch, 2023-10-06T17:22:02+00:00)
“He can’t just walk out if he doesn’t like how it’s going.”



Naftalis spoke at length about what comes next in the trial, and addressed common misconceptions about the prosecution and defense in the latest episode of
TechCrunch’s Chain Reaction podcast, which you can listen to below, or on Apple Podcasts or Spotify.

The SBF trial is one of the biggest cases of its kind within the crypto space. Our resident crypto expert Jacquelyn Melinek has been on the ground at the trial
since it started, and continues to cover the ins and outs of the proceedings. But the trial has been on pause from October 20 to today, so there’s plenty to
catch up on if you need a refresher ahead of it entering its final few phases.

For those in need of a total refresher on the case, her breakdown of how FTX went from the third-largest crypto exchange valued at a peak of $32 billion to
bankruptcy will get you up to speed.
</item 61>

<item 62>
Today the FTX jury suffered through a code review (The Verge, 2023-10-06T22:21:46+00:00)
In late 2019, Alameda had a negative balance on FTX. Because they sat in an open-plan office, Wang heard a trader ask Bankman-Fried if it was okay to keep
withdrawing money from that account. Bankman-Fried said it was okay as long as it was lower than FTX’s total trading revenue. He was the CEO of both companies
at the time.

Alameda also allegedly kept FTX from looking bad by assuming some of its losses.

The amount that Alameda was allowed to go negative began to creep up. Around 2019 or 2020, Wang checked the database and discovered that Alameda was negative by
about $200 million, which was more than the $150 million FTX made in revenue. That had to mean Alameda was taking customer money. It surprised him and he says
he discussed it with Bankman-Fried, who told him that he just needed to take into account the value of FTT. Wang said he trusted Bankman-Fried’s judgment and
didn’t pursue the issue further.
</item 62>

<item 63>
Is Sam Bankman-Fried a bad ‘man’ or a good ‘boy’? Lawyers swap opening statements before first witnesses take the stand (Fortune, 2023-10-04T23:32:53+00:00)
FTX was no Ponzi scheme. It was a “very innovative, successful company.” And the business practices between the two were reasonable, he argued, claiming that
Alameda acted legally as an FTX customer, payment processor, and market maker, or financial entity that acts as a trading partner for customers looking to buy
and sell cryptocurrencies.

In an analogy he employed throughout his opening statement, he said that “working at a startup is like building a plane as you’re flying it” and that businesses
sometimes fail. In fact, he specifically pointed the finger at Ellison, the former CEO of Alameda, who, he said, did not adequately protect her hedge fund from
the inherent risk of the crypto markets.

When the walls came closing in and the aforementioned plane approached the “eye of the storm,” Bankman-Fried did not act like someone who was guilty. Rather, he
was willing to give up his personal wealth to make customers whole, Cohen argued.
</item 63>

<item 64>
Sam Bankman-Fried was a terrible boyfriend (The Verge, 2023-10-10T23:50:21+00:00)
(Fraud, conspiracy to commit fraud, and money laundering, in case you were wondering.) “We ultimately took about $14 billion, some of which we were not able to
pay back.” She tilted her head down to answer the questions, then lifted her head when she’d finished her answer.

Bankman-Fried called this being “risk-neutral,” which seems like a fancy way of saying he was a gambling addict

In Ellison’s telling, Alameda was troubled from her earliest time there in 2018. “Shortly after I started, I learned the company was in worse shape than I
realized,” Ellison said. Alameda had initially been funded with loans “from acquaintances,” she said, and those loans were recalled a few weeks after she
arrived. (There was a staff revolt within Alameda Research over lost millions and general financial chaos, according to Michael Lewis’ Going Infinite.) Ellison
asked Bankman-Fried why he hadn’t shared the company’s shaky circumstances in the job offer. “He hadn’t known how to tell me,” she said.
</item 64>

<item 65>
The FTX trial is bigger than Sam Bankman-Fried (The Verge, 2023-09-28T12:00:00+00:00)
The trial of Sam Bankman-Fried is likely to be more consequential than just whether the man himself is found guilty. Depending on what evidence is introduced
during the trial, it could be rough for the entire crypto industry.

“How much damage can this trial do to the already beaten-down reputation of the industry at this point?” asks Yesha Yadav, a law professor at Vanderbilt
University. “This trial is going to be an excruciating moment for the industry because no one knows what kind of evidence might come out.”

Bankman-Fried, the founder of FTX and Alameda Research, is facing seven counts of criminal charges: two counts of wire fraud, and five counts of conspiracy
charges. FTX was a failed cryptocurrency exchange founded in 2019. According to a now-deleted profile from FTX investors Sequoia Capital, FTX was founded
because of Bankman-Fried’s frustration with other exchanges when he was running Alameda Research, his crypto trading firm.
</item 65>

<item 66>
In the end, the FTX trial was about the friends screwed along the way (The Verge, 2023-10-26T12:00:00+00:00)
Sure, Sam Bankman-Fried’s trial has revealed that FTX was basically a seething mass of scams. But it has also revealed how Bankman-Fried blew up the lives of
the people who are closest to him.

Of course everyone who had money trapped on the FTX exchange when it collapsed — as the result of Bankman-Fried’s trading firm Alameda Research purloining its
customers’ funds — had their lives altered. Even if they are made whole by the bankruptcy proceedings, that’s still money that they should have had access to at
any time they wanted it. That may mean houses that weren’t purchased, medical procedures that were left undone, and life savings lost.

There is a simultaneous drama of ruined lives playing out in the courtroom

Bankman-Fried is charged with seven counts: two counts of wire fraud and five conspiracy charges. As we near the end of the prosecution’s case, I have been
struck by how bizarre it all is.
</item 66>

<item 67>
Is Sam Bankman-Fried a bad ‘man’ or a good ‘boy’? Lawyers swap opening statements before first witnesses take the stand (Fortune, 2023-10-04T23:32:53+00:00)
Alameda Research, a crypto hedge fund he also owned, argued Rehn. Using Caroline Ellison, his on-again off-again girlfriend and the CEO of Alameda, as a front,
Bankman-Fried had “secret access” to customer money—both cash and crypto—the government claimed.

Moreover, Bankman-Fried allegedly directed employees to conceal the flow of money into FTX’s coffers and forged financial documents distributed to lenders and
investors. “The defendant lied to the world,” Rehn alleged.

And who was this defendant? Not a crypto “boy” genius, as so many in the media (Fortune included) have written, but a “man” who “stole billions of dollars from
thousands of victims,” Rehn said. “You will see the full picture.”

…or a well-meaning founder?
</item 67>

<item 68>
The jury finally hears from Sam Bankman-Fried (The Verge, 2023-10-28T00:12:41+00:00)
When he found it, he was “very surprised!”

Besides the two obvious lies Bankman-Fried told on the stand — about Alameda Research’s name and about “allow_negative” — I have been struck by how little he
seems to know about his own companies. Apparently, Singh, Wang, and Ellison were out there just doing whatever their little hearts desired. Because
Bankman-Fried was a CEO, but definitely not the kind that pays any attention to money at his crypto trading firm and futures exchange.
</item 68>

<item 69>
In the end, the FTX trial was about the friends screwed along the way (The Verge, 2023-10-26T12:00:00+00:00)
All the co-conspirators seem oddly younger than their years, as though they got stuck in the “math camp forever” environment Bankman-Fried created for them and
simply failed to grow up.

The prosecution has made a strong case that Bankman-Fried presided over the various forms of fraud that played out at Alameda Research and FTX, orchestrating
some of it and engaging in some of it directly. In the process of making this case, there is a simultaneous drama of ruined lives playing out in the courtroom.

Everyone who loved Bankman-Fried most in the world is now caught up in the case. We have:

Adam Yedidia, Bankman-Fried’s college roommate, who testified with immunity because he was concerned he may have unknowingly worked on code that contributed to
the fraud. “I love you Sam,” he texted at one point. “I’m not going anywhere.” That was before he realized FTX had stolen customer funds — at which point, he
resigned.
</item 69>

<item 70>
Sex, Signal messages, and sabotaging FTX's code: SBF execs and Bahamas roommates tell all in court (Cnbc | World Business News Leader,
2023-10-06T20:17:00+00:00)
Still, Yedidia said he trusted Bankman-Fried and Ellison to "handle the situation." On cross-examination, Christian Everdell, Bankman-Fried's attorney, focused
on how Yedidia was the one responsible for developing and reviewing the code.



He asked about the long hours employees worked and Yedidia's concern for Wang being near burnout. That resulted in Yedidia instituting a rule to not wake Wang
at night for bug fixes because he needed sleep. Everdell also drilled Yedidia on his high level of compensation in his less than two years at FTX. His base
salary was between $175,000 and $200,000, but he received multiple bonuses of more than $12 million in cash and company equity. Yedidia said he's now teaching
math — geometry and algebra — at a high school.
</item 70>

<item 71>
Today the FTX jury suffered through a code review (The Verge, 2023-10-06T22:21:46+00:00)
Things happened quickly for Wang in November. Wang and Bankman-Fried determined that about $8 billion in customer money was gone, but on November 7th,
Bankman-Fried tweeted “FTX is fine. Assets are fine.” Another tweet in the thread said, “FTX has enough to cover all client holdings.”

“FTX was not fine and assets were not fine because FTX did not have enough assets for customer withdrawals,” Wang said.

While Bankman-Fried can try to blame Ellison for some of the decisions, the code shows she couldn’t have made all the calls

FTX went bankrupt on November 11th. On November 12th, Bankman-Fried directed Wang to send customer funds to Bahamanian regulators, Wang said. At the time,
customers couldn’t take their funds off the exchange because FTX had run out of money and its servers had been shut down. Bankman-Fried and his father met with
lawyers while Wang waited outside with liquidators.
</item 71>

<item 72>
Is Sam Bankman-Fried’s defense even trying to win? (The Verge, 2023-10-04T23:02:14+00:00)
When he was evaluating the exchange, he thought the sheer volume of users was important, too — at the time, FTX was among the top three biggest exchanges. Plus,
major VC firms had invested, and “they don’t commit hundreds of millions without doing due diligence, checking the books, the accountancy of the firm, going
through several compliance process[es], so that was a vote of confidence for me,” Juilliard said. (Evidently he had not paid attention to the Elizabeth Holmes
trial.)

He also noted FTX’s glossy ads — featuring Gisele Bündchen, for instance — suggested a very high budget. It wouldn’t make sense to spend that much money unless
FTX had very strong financials, Juilliard figured. He opened an account, transferred in both regular money and cryptocurrency, and used the exchange to execute
his plan: buying Bitcoin to sell back in five to ten years at higher prices.
</item 72>

<item 73>
The FTX trial is bigger than Sam Bankman-Fried (The Verge, 2023-09-28T12:00:00+00:00)
In the indictment, the prosecutors allege that Bankman-Fried lied to FTX investors. Some of those investors may be called on to testify, which is not the kind
of thing VC big shots generally enjoy — and which might create collateral damage for the industry.

If, for instance, Sequoia Capital did due diligence around its investment in FTX, whatever Bankman-Fried told its partners could be important. Was FTX already
sending Alameda customer funds at that point? Did Bankman-Fried know about it? And did he tell VCs at the time? (If he told them it wasn’t happening, that would
establish intent.) Binance was also an early investor in FTX, and former executives may be called on to testify as well, says Hermine Wong, the former head of
policy at Coinbase and a former SEC regulator.

Bankman-Fried’s defense can also introduce risks for people who dealt with him. Defense lawyers have several simultaneous objectives. First and foremost,
they’re trying to prove their client is not guilty.
</item 73>

<item 74>
In the end, the FTX trial was about the friends screwed along the way (The Verge, 2023-10-26T12:00:00+00:00)
Joe Bankman, Bankman-Fried’s father. He is among the people included in the “small group chat,” the Signal group where the FTX inner circle strategized during
the exchange’s last days.

Barbara Fried’s political action committee, Mind the Gap. It is among the entities that received FTX customer money as political donations. Fried and Bankman
lived together in a Bahamas house purchased for them using FTX investor funds, according to a financial expert called by the prosecution; in a lawsuit filed by
FTX’s bankruptcy estate, Bankman and Fried are alleged to have been given $10 million of FTX customer funds.

Gabe Bankman-Fried’s nonprofit, Guarding Against Pandemics, which got millions from FTX. He was also in a Signal group chat, Donations Processing, that Singh
says was used to coordinate the transfer of FTX and Alameda money to various political causes. In that chat, he can be seen recommending some donations be made
“in the dark” to avoid public relations headaches.
</item 74>

<item 75>
In the end, the FTX trial was about the friends screwed along the way (The Verge, 2023-10-26T12:00:00+00:00)
The defense has struggled to shake the credibility of witnesses such as Yedidia, Wang, and Ellison

That “basically confirmed my suspicions that had been rising all day” that Bankman-Fried had purloined the funds, Sun said. He ran Bankman-Fried through
possible explanations, including a margin loan program, and explained that none of them could fit the actual situation — for instance, Alameda had borrowed more
from FTX than had ever been in the margin loan program. He said Bankman-Fried acknowledged their conversation with a “yup, yup.” “I was expecting a bigger
response, but it was very muted,” Sun said.

After that, Singh spilled the beans: Alameda had withdrawn assets that included the customer funds, Sun testified. He quit the next day. Like Yedidia, Sun
testified with immunity.

In the courtroom, we were treated to a Good Morning America appearance by Bankman-Fried on December 1st, 2022.
</item 75>

<item 76>
Sam Bankman-Fried didn’t ask where the $8 billion went (The Verge, 2023-10-31T19:52:07+00:00)
Let’s say I am the owner of a hedge fund, and one fine June day, my employees come to me and say, “Hey, Liz, we have an accounting problem. We are missing
several billion dollars.” How would I react?

I have been wondering this since Danielle Sassoon walked Sam Bankman-Fried through his reaction to the FTX software bug fixed by Adam Yedidia. In my case, there
would probably be shouting? Like, a lot of shouting. I would also probably have my assistant figure out which law enforcement agency to call immediately.
Misplacing $900 million is a five-alarm fire even for Citibank; misplacing several billion is kicking over a lantern in Chicago in 1871.

Obviously, this is not how Bankman-Fried reacted to the software bug that overestimated the amount Alameda owed to FTX by about $8 billion. Nor is it how he
reacted to finding out that even after fixing the bug, Alameda still owed FTX about $8 billion.
</item 76>

<item 77>
Today the FTX jury suffered through a code review (The Verge, 2023-10-06T22:21:46+00:00)
It lives between the two ends of the bet, paying the winner and collecting from the loser. If the loser doesn’t pay up, the exchange still has to pay the
winner, which is why exchanges require collateral.

Under some circumstances, exchanges will automatically sell collateral to limit their losses. On FTX, this process was automated, Wang testified. It took about
30 seconds to find accounts that needed to be liquidated to minimize FTX’s losses.

Normal customers — the defense objected to “normal” but was overruled — could be automatically liquidated, but not Alameda. Bankman-Fried “told me a few times
to make sure that Alameda’s account is never liquidated on FTX,” Wang said. “The code addition was the result of those conversations.”

It’s not Bankman-Fried’s name attached to the change. We saw a code commit from July 31st, 2019 from Nishad Singh titled “OTC trades and transfers to special
accounts.”
</item 77>

<item 78>
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
</item 78>

<item 79>
Sex, Signal messages, and sabotaging FTX's code: SBF execs and Bahamas roommates tell all in court (Cnbc | World Business News Leader,
2023-10-06T20:17:00+00:00)
Additionally, he was promised that Alameda had no preferential treatment on the FTX platform, even though the hedge fund was one of its top traders. Huang said
he was concerned about FTX's lack of a board of directors, but he eventually invested anyway. During cross-examination, Huang said Paradigm pressed
Bankman-Fried on the board issue and was told he didn't want investors as directors but he did plan on having a board with experts. — CNBC's Dawn Giel
contributed to this report.
</item 79>

<item 80>
Today the FTX jury suffered through a code review (The Verge, 2023-10-06T22:21:46+00:00)
It was written in Python; the upshot was that it added two columns to the account database; the relevant one for our purposes was “allow_negative.” This was a
toggle that, when it was on, let the account go negative. The feature was turned on the same day in a second set of code that also had Singh’s username on it.
Wang said he supervised.

That same day, Bankman-Fried tweeted, “Alameda is a liquidity provider on FTX, but their account is just like everyone else’s.” The tweet was, of course, shown
in court. Never tweet.

It was a tough day for the jury

At first, Bankman-Fried said the change was to pay for expenses for the FTT token, which was issued by FTX, Wang said. (Wang described the FTT token as being
“sort of equity” in FTX, which I think some crypto lawyers would dispute but would probably make the SEC happy.)
</item 80>

<item 81>
Sex, Signal messages, and sabotaging FTX's code: SBF execs and Bahamas roommates tell all in court (Cnbc | World Business News Leader,
2023-10-06T20:17:00+00:00)
Yedidia said he's now teaching math — geometry and algebra — at a high school. He invested most of the millions he earned as bonuses back into FTX, and his
equity stake is now worthless. As FTX was failing, Yedidia said he was by Bankman-Fried's side. He highlighted a Signal exchange in November 2022, during which
he wrote, "I love you Sam. I'm not going anywhere." He said he wrote the message because so many people had left. When asked what changed, Yedidia said he
learned that FTX customer deposits had been used to pay loans to creditors. He said Alameda's actions seemed "flagrantly wrong." Yedidia's testimony ended on a
fiery note, which was later struck from the record. He was asked why he had lost faith in FTX and resigned. "FTX defrauded all its customers," he said.
</item 81>

<item 82>
Sex, Signal messages, and sabotaging FTX's code: SBF execs and Bahamas roommates tell all in court (Cnbc | World Business News Leader,
2023-10-06T20:17:00+00:00)
Sam Bankman-Fried, the founder of bankrupt cryptocurrency exchange FTX, is seen during a hearing as a U.S judge revoked his bail, at a courthouse in New York,
U.S., August 11, 2023 in this courtroom sketch. Jane Rosenberg | Reuters

$65 billion line of credit

For months, Bankman-Fried has known that Wang and Ellison, who were integral members of his personal and professional inner circles, had turned on him. Both
pleaded guilty in December and have since been cooperating with the U.S. attorney's office in Manhattan. Wang's testimony, which stretched into Friday, was
given under a cooperation agreement with the government. Ellison is expected to take the stand under a similar arrangement.

U.S.
</item 82>

<item 83>
In the end, the FTX trial was about the friends screwed along the way (The Verge, 2023-10-26T12:00:00+00:00)
The political donations gave Bankman-Fried influence in the highest echelons of power. Ellison testified to Bankman-Fried’s venture investments, which gave him
more dominance in the tech world. At every turn, the scams at FTX and Alameda led back to making Bankman-Fried more important.

The pie chart that accompanied expert witness Peter Easton’s testimony. Image: The prosecution’s exhibit 1045

This was backed up by expert testimony from Notre Dame accounting professor Peter Easton, who’d also assisted in investigating Enron and WorldCom. In a pie
chart that showed how he’d traced funds for the $11.3 billion liability that Ellison, Singh, Wang, and Bankman-Fried had known about in June 2022, we saw that
$1.2 billion was a loan repayment to crypto lender Genesis.
</item 83>

<item 84>
Today the FTX jury suffered through a code review (The Verge, 2023-10-06T22:21:46+00:00)
But once it was implemented, the ability to go negative was used for other things including trading and unlimited withdrawals of FTX’s trading fees and customer
money.

I am summarizing briefly because Wang had to explain to the jury the difference between a front end and back end, explain futures trading, and read a bunch of
stuff into the record, which took a while. It was a tough day for the jury — and probably for Wang, who spoke quickly, perhaps from nerves.

The prosecution doesn’t just allege that Alameda had special, secret privileges — it claims they were used to obfuscate basic elements of FTX’s operations that
exposed its supposed selling points as a lie.

For instance, FTX advertised that it had a good liquidation system. In many crypto networks, if someone lost enough money, the exchanges could stick other
traders with the losses too. FTX touted its automated liquidation as a way to avoid this, so that one customer going bankrupt wouldn’t affect the others.
</item 84>

<item 85>
In the end, the FTX trial was about the friends screwed along the way (The Verge, 2023-10-26T12:00:00+00:00)
Even that wasn’t good enough. Singh testified to sprucing up FTX’s balance sheet with backdated transactions in December 2021 to push FTX’s revenue “over the
line” to $1 billion. By the time Singh found out about the use of customer money, in 2022, he was already compromised.

At every turn, the scams at FTX and Alameda led back to making Bankman-Fried more important

In Singh’s case, it wasn’t just balance sheets he was fiddling with — it was also the American political system. By coordinating with Gabe Bankman-Fried, Singh
had funneled money to a number of political causes. That money came from “loans,” which weren’t really loans since he hadn’t signed any term sheets.

What was the goal here? What was accomplished? Singh testified to more than a billion dollars spent on celebrity endorsements, which made FTX very famous — even
though the futures platform didn’t operate in the US.
</item 85>

<item 86>
Sam Bankman-Fried was a terrible boyfriend (The Verge, 2023-10-10T23:50:21+00:00)
Alameda got its war chest — 60 percent to 70 percent of the initial supply — for free, while seed investors got FTT at 10 cents a coin, and FTT first listed at
$1 a coin. Bankman-Fried felt that $1 per coin was psychologically important, Ellison said, and he directed her to buy up FTT using Alameda if its price fell
below a dollar.

FTT was one of several “Sam coins,” a nickname for tokens that Bankman-Fried was heavily involved in and owned a lot of, either personally or through Alameda.
Those coins were almost certainly worth less than the value displayed on the balance sheet because trying to sell them all at once would crater the prices.
Bankman-Fried directed her to put those coins on the balance sheets Alameda showed to lenders, even though she felt it was “somewhat misleading.”

Alameda was also getting loans from outside lenders, such as Genesis, because when FTX started, there weren’t a lot of customer funds to borrow, Ellison
testified.
</item 86>

<item 87>
Sex, Signal messages, and sabotaging FTX's code: SBF execs and Bahamas roommates tell all in court (Cnbc | World Business News Leader,
2023-10-06T20:17:00+00:00)
That was all done by Bankman-Fried, he testified. In early 2020, Wang said he discovered for the first time Alameda's negative balance exceeded FTX's revenue,
an indication that Alameda was taking customer funds. Wang said he brought this to Bankman-Fried's attention several times. In late 2021, Wang discovered
Alameda had withdrawn $3 billion from its $65 billion line of credit. Wang's compensation was a base salary of $200,000 per year plus stock. He owned roughly
17% of FTX. Even though they were co-founders, "ultimately it was Sam's decision to make" when there were disagreements, he said.
</item 87>

<item 88>
Sex, Signal messages, and sabotaging FTX's code: SBF execs and Bahamas roommates tell all in court (Cnbc | World Business News Leader,
2023-10-06T20:17:00+00:00)
Assistant United States Attorney Nicolas Roos questions Gary Wang during Sam Bankman-Fried's fraud trial over the collapse of FTX, the bankrupt cryptocurrency
exchange, at Federal Court in New York City, U.S., October 6, 2023, in this courtroom sketch. Jane Rosenberg | Reuters

An $8 billion bug

Adam Yedidia, who was the prosecution's second witness on Wednesday, continued his testimony on Thursday. Yedidia met Bankman-Fried in college at MIT, and the
pair remained close friends. Yedidia, assuming a robotic posture on the stand, worked out of FTX's Hong Kong office from January to October of 2021 and then in
the Bahamas until last year's collapse. In his testimony, he referred to a group Signal thread called "People of the House," referring to Bankman-Fried's $35
million penthouse, where many employees lived.
</item 88>

<item 89>
SBF Trial: The latest updates from the FTX collapse’s courtroom drama (TechCrunch, 2023-10-06T17:22:02+00:00)
The full breakdown of his testimony on his relationship with FTX, stretching back to 2019, can be found right here.

We also got a look at the composition of the SBF trial’s jury, which ranges from a train conductor to a retired investment banker who attended Stanford
University. You can listen to a special joint episode of our podcasts Equity and Chain Reaction to get a better sense of the vibe between the jurors and the
rest of the court.

For a full rundown of our coverage of the SBF trial, check below:
</item 89>

<item 90>
Sex, Signal messages, and sabotaging FTX's code: SBF execs and Bahamas roommates tell all in court (Cnbc | World Business News Leader,
2023-10-06T20:17:00+00:00)
Witness Adam Yedidia answers questions during Sam Bankman-Fried fraud trial over the collapse of FTX, the bankrupt cryptocurrency exchange, at Federal Court in
New York City, October 5, 2023, in this courtroom sketch. Jane Rosenberg | Reuters

Two of Sam Bankman-Fried's former friends from MIT, who also worked at crypto exchange FTX while living with the company's founder in the Bahamas, took the
stand in a Manhattan courtroom this week to testify against their former classmate, confidant, and boss — a man who allegedly ran a crypto empire that defrauded
thousands of customers out of billions of dollars. Gary Wang, the lesser-known co-founder of FTX, was asked by Assistant U.S. Attorney Nicolas Roos on Thursday,
"Did you commit financial crimes while working at FTX?" "Yes," responded Wang.
</item 90>

<item 91>
Sex, Signal messages, and sabotaging FTX's code: SBF execs and Bahamas roommates tell all in court (Cnbc | World Business News Leader,
2023-10-06T20:17:00+00:00)
Zoom In Icon Arrows pointing outwards Exhibit from the prosecution shows Signal thread called "People of the House," referring to Bankman-Fried's $35 million
penthouse, where many employees lived. Source: SDNY

In terms of who was paying the rent, Yedidia recalled Bankman-Fried saying he "assumed it's just Alameda paying for it in the end." Yedidia said Bankman-Fried
had told him, before he began working in the Bahamas in 2019, that he and Ellison had sex. Bankman-Fried asked Yedidia if it was a good idea for them to date,
to which Yedidia said no. Bankman-Fried responded by saying he was expecting that answer. One of Yedidia's responsibilities was fixing the bug in the code that
gave Alameda preferential treatment.
</item 91>

<item 92>
Today the FTX jury suffered through a code review (The Verge, 2023-10-06T22:21:46+00:00)
FTX lied about how much money was in the backstop fund

In the process of liquidating, FTX would try to sell the collateral on the open market, but if it couldn’t finish, then backstop liquidity providers would step
in. These were market makers, including Alameda, which could be compensated for the losses they take from a “backstop fund,” Wang said.

But FTX lied about how much money was in the backstop fund, Wang said. In court we saw the code that generated the fake number published on the website: it took
the daily trading volume on FTX, multiplied it by a random number, divided it by a billion, and added it to the existing number displayed on the site. It had
nothing to do with the actual amount of money in the insurance fund.

And when there wasn’t enough money in the fund, money was moved there from Alameda’s accounts in order to pay the insurance out, Wang said.

In late 2019, Alameda had a negative balance on FTX.
</item 92>

<item 93>
How is it still getting worse for Sam Bankman-Fried? (The Verge, 2023-10-12T23:08:14+00:00)
I have been waiting for a juicy cross-examination, as I live for chaos and drama. I am beginning to think I am not going to get one.

Ellison had given, in her direct testimony, fairly damning evidence tying FTX CEO Sam Bankman-Fried to the conspiracy to take FTX customer funds. There were
fake balance sheets, one of which was sent to crypto lender Genesis. After a Genesis representative received the balance sheet, he texted Ellison to tell her
he’d spoken to Bankman-Fried — strongly suggesting that Bankman-Fried was aware of the contents of the fake balance sheet. Not great!

Sam Bankman-Fried, the founder of failed cryptocurrency exchange FTX, is on trial for seven counts of wire fraud and conspiracy. FTX was a fraud “from the
start,” the Securities and Exchange Commission alleges — with a “multi-billion-dollar deficiency caused by his own misappropriation of customer funds.” Follow
along for all the latest news and regular updates from the trial.
</item 93>

<item 94>
In the end, the FTX trial was about the friends screwed along the way (The Verge, 2023-10-26T12:00:00+00:00)
It was turned on in July. No other customer trading on FTX had “allow_negative” privileges — only Alameda. Wang’s testimony suggests that the fraud at FTX
started very early since the same day that allow_negative was switched on, Bankman-Fried tweeted that Alameda’s account “was like everyone else’s.”

The general sense I got, from the testimony of Bankman-Fried’s former friends, was that things just kept escalating. For instance, Alameda Research first had a
negative balance in late 2019, Wang testified. Wang testified that he heard Bankman-Fried tell another trader that it was okay to continue withdrawing money
from the account as long as the amount was less than FTX’s total trading revenue. A little later, Wang discovered that Alameda was negative by $50 million more
than FTX’s revenue. Bankman-Fried told him it was fine, and it just didn’t account for the FTT token that the two of them had launched. Wang trusted him.
</item 94>

<item 95>
The jury finally hears from Sam Bankman-Fried (The Verge, 2023-10-28T00:12:41+00:00)
Well, Bankman-Fried remembers it differently. See, Yedidia was just asking about Alameda’s risk profile, and Bankman-Fried wasn’t talking about insolvency at
all.

Also, when Singh and Bankman-Fried had the dramatic balcony conversation at their penthouse, it was just that Singh thought Alameda’s liabilities had gotten too
high, and FTX was spending too much money on marketing. But Bankman-Fried still thought that Alameda had more assets than liabilities, so it was all fine, and
besides, if Singh thought he was going to be better at marketing, he could take it over. It didn’t have anything to do with the money Alameda owed FTX at all.

Of course not! Bankman-Fried didn’t learn about the $8 billion liability associated with Alameda until October 2022, he said. And he learned it all by himself,
by looking at a computer database. When he found it, he was “very surprised!”
</item 95>

<item 96>
Sam Bankman-Fried was a terrible boyfriend (The Verge, 2023-10-10T23:50:21+00:00)
That gave Ellison an unusual view of his character. “He was very ambitious,” she said. Besides telling her about his presidential chances, he also told her that
if there was a coin flip where tails destroyed the world and heads made the world twice as good, he’d flip the coin. He called this being “risk-neutral,” which
seems like a fancy way of saying he was a gambling addict.

She was named co-CEO of Alameda with Sam Trabucco in 2021, while she and Bankman-Fried were broken up, and CEO in 2022. The goal, Bankman-Fried told her, was to
“optically” separate Alameda Research and FTX. “The whole time we were dating, he was my boss at work,” she said. They broke up because she wanted more from the
relationship; Bankman-Fried was distant and not paying enough attention to her.
</item 96>

<item 97>
Is Sam Bankman-Fried’s defense even trying to win? (The Verge, 2023-10-04T23:02:14+00:00)
Meanwhile, Bankman-Fried tweeted. “FTX is fine. Assets are fine” and “We don’t invest customer assets even in treasuries.”

Pointing at Bankman-Fried, Rehn said, “This man stole billions of dollars from thousands of people.”

So how was the defense going to follow it up? I was very curious, having learned yesterday that Bankman-Fried had never been offered a plea deal since he and
his attorneys had told the government they wouldn’t negotiate. Surely there would be some manner of evidence, some something, that would have made him so
confident.

There was, instead, a metaphor.

Defense attorney Mark Cohen, with the energy of a patient father telling his obnoxious children a bedtime story, assured us that working at a startup was like
building a plane while flying it, and that FTX the plane had flown right into the perfect storm: the crypto crash.
</item 97>

<item 98>
Sam Bankman-Fried was a terrible boyfriend (The Verge, 2023-10-10T23:50:21+00:00)
Alameda’s credit line — which was taken in increments of $100,000 to $10 million at a time — was used for trading. Using the effectively unlimited funds
“allowed us to make profitable trades we couldn’t have made otherwise,” Ellison testified.

Customer funds were also used when Bankman-Fried bought back FTX shares from Binance, an early investor, in the summer of 2021. Bankman-Fried told Ellison it
was “really important,” otherwise “Binance would do things to mess with FTX.” Ellison says she told him Alameda didn’t have the money. So Bankman-Fried took $1
billion of FTX customer funds to buy out Binance, the first time Ellison recalled an amount that large. It was Bankman-Fried’s decision, she said, as he was the
CEO of FTX.

FTT was one of several “Sam coins”

There was also the FTT token, which was created by Bankman-Fried and Wang.
</item 98>

<item 99>
In the end, the FTX trial was about the friends screwed along the way (The Verge, 2023-10-26T12:00:00+00:00)
That was before he realized FTX had stolen customer funds — at which point, he resigned.

Gary Wang, the co-founder of FTX and Alameda Research and another college roommate. He took the stand after Yedidia and sang like a canary about the secret
privileges he’d created for Alameda in the FTX code.

Nishad Singh, a longtime family friend, who copped to what sounded an awful lot like classic embezzlement. He said he’d backdated transactions on FTX to help
create doctored balance sheets to show to investors. He pleaded guilty to six charges, including wire fraud, money laundering, and conspiracy counts around his
political donations.

Caroline Ellison, Bankman-Fried’s ex-girlfriend. She told us about the seven phony balance sheets she’d created. She was clear: Alameda took customer funds;
Bankman-Fried knew about it and even invested them against her advice.

Joe Bankman, Bankman-Fried’s father.
</item 99>

<item 100>
The jury finally hears from Sam Bankman-Fried (The Verge, 2023-10-28T00:12:41+00:00)
Perhaps you are wondering, was the jury also there? Reader, it was.

Bankman-Fried denied he knew about the effectively infinite line of credit Alameda Research received from FTX. This argument was peculiar; essentially my
take-away was that the CEO of a financial company simply didn’t pay attention to finances.

FTX couldn’t get bank accounts right away. Bankman-Fried anticipated it would take a year or two. Rather than wait, he decided to use Alameda as the “payment
provider” for bank transfers. “My understanding at the time was that there were teams managing the process,” he said. “At the time, I wasn’t entirely sure what
was happening.”

Well, sure, understandable! He’s an introvert!

Bankman-Fried definitely did not know that Singh, his employee, had backdated interest payments to get FTX “over the line” to $1 billion

In 2021, FTX was growing to millions of users, with $1 billion of revenue.
</item 100>
=== END ITEM id=829000fb93198852 ===

=== ITEM id=c2ae14f2242ec4bf ===
Gold sentences and facts:
Sentence 0: For the Swans, however, the victory has them challenging for a top-eight spot alongside the likes of Gold Coast, Geelong, Collingwood, St Kilda and Carlton.
  Fact 0.0: The victory has the Swans challenging for a top-eight spot.
  Fact 0.1: The Swans are challenging for a top-eight spot alongside the likes of Gold Coast.
  Fact 0.2: The Swans are challenging for a top-eight spot alongside the likes of Geelong.
  Fact 0.3: The Swans are challenging for a top-eight spot alongside the likes of Collingwood.
  Fact 0.4: The Swans are challenging for a top-eight spot alongside the likes of St Kilda.
  Fact 0.5: The Swans are challenging for a top-eight spot alongside the likes of Carlton.
Sentence 1: With only 0.4 of a percentage point separating Sydney in ninth and St Kilda in eighth, the win assured the Swans of retaking eighth spot.
  Fact 1.0: Sydney is in ninth place.
  Fact 1.1: St Kilda is in eighth place.
  Fact 1.2: Only 0.4 of a percentage point separates Sydney in ninth and St Kilda in eighth.
  Fact 1.3: The win assured the Swans of retaking eighth spot.
Sentence 2: “Our team has put a lot of work into sourcing some wonderful food offerings, entertainment and activities for kids, while we’ve had a local focus in promoting our games,” she said.
  Fact 2.0: The speaker said that her team has put a lot of work into sourcing some wonderful food offerings.
  Fact 2.1: The speaker said that her team has put a lot of work into sourcing entertainment for kids.
  Fact 2.2: The speaker said that her team has put a lot of work into sourcing activities for kids.
  Fact 2.3: The speaker said that her team has had a local focus in promoting its games.

Retrieved items:

=== END ITEM id=c2ae14f2242ec4bf ===

=== ITEM id=10b892ce02027237 ===
Gold sentences and facts:
Sentence 0: Johnny Cardoso had to pull out of the September camp due to an ankle injury, but he has returned to action with Brazilian club Internacional, indicating a return to fitness, and has been included on the squad.
  Fact 0.0: Johnny Cardoso had to pull out of the September camp due to an ankle injury.
  Fact 0.1: Johnny Cardoso has returned to action with Brazilian club Internacional.
  Fact 0.2: Johnny Cardoso's return to action indicates a return to fitness.
  Fact 0.3: Johnny Cardoso has been included on the squad.
Sentence 1: He and his fresh legs are back just in time as rookie sensation De'Von Achane hit Injured Reserve this week with a knee injury.
  Fact 1.0: Johnny Cardoso and his fresh legs are back just in time.
  Fact 1.1: Rookie sensation De'Von Achane hit Injured Reserve this week.
  Fact 1.2: De'Von Achane hit Injured Reserve with a knee injury.

Retrieved items:
<item 1>
NBA Rookie of the Year Odds - Wembanyama the early favorite (Sporting News, 2023-10-04T22:54:23+00:00)
Season Player Position Team 2023 Paolo Banchero Forward Orlando Magic 2022 Scottie Barnes Forward Toronto Raptors 2021 LaMelo Ball Guard Charlotte Hornets 2020
Ja Morant Guard Memphis Grizzlies 2019 Luka Dončić Guard/Forward Dallas Mavericks 2018 Ben Simmons Forward/Guard Philadelphia 76ers 2017 Malcolm Brogdon Guard
Milwaukee Bucks 2016 Karl-Anthony Towns Center Minnesota Timberwolves 2015 Andrew Wiggins Forward/Guard Minnesota Timberwolves 2014 Michael Carter-Williams
Guard Philadelphia 76ers 2013 Damian Lillard Guard Portland Trail Blazers 2012 Kyrie Irving Guard Cleveland Cavaliers 2011 Blake Griffin Forward Los Angeles
Clippers 2010 Tyreke Evans Guard/Forward Sacramento Kings

What other NBA futures can I bet on?

Rookie of the Year is not the only form of NBA futures betting.
</item 1>

<item 2>
NBA Rookie of the Year Odds - Wembanyama the early favorite (Sporting News, 2023-10-04T22:54:23+00:00)
Rookie of the Year is not the only form of NBA futures betting. Legal online sportsbooks have a wide selection of NBA futures you can bet on, including:

In addition to futures bets, you can wager on many NBA bets during the regular season and postseason. NBA game lines (moneylines, point spreads, and totals) are
the most popular, but you’ll also be able to bet on NBA team and player props, parlays, and more.

The NBA is the second-most popular odds market to bet on in the US, behind NFL betting odds. That means every online betting site always has a great selection
of NBA odds.
</item 2>

<item 3>
NBA Rookie of the Year Odds - Wembanyama the early favorite (Sporting News, 2023-10-04T22:54:23+00:00)
We urge you to read the requirements for any available promotions and offers before accepting them so you’re aware of any restrictions before betting.

You can now place your initial deposit, which is easy at online betting sites thanks to various deposit and withdrawal options.

To bet on the NBA Rookie of the Year, navigate to the basketball section of the sportsbook website, select the NBA, and go to the futures tab.

You’ll find Rookie of the Year odds, NBA MVP odds, and more there. Select the bet you want to make, enter the amount you want to bet, and finalize your bet
slip.

Ensure you’re using only legal online sportsbooks for any type of betting. Offshore sportsbooks advertise widely on the internet, but they are illegal and offer
no protection against fraud and other criminal activities.

Best sportsbooks for NBA Rookie of the Year odds

Here are some of the best sports betting sites where you can bet on the next NBA Rookie of the Year.
</item 3>

<item 4>
NBA Rookie of the Year Odds - Wembanyama the early favorite (Sporting News, 2023-10-04T22:54:23+00:00)
Once the season has started and you get to see how the various rookies are performing, you can gather the relevant data. But sportsbooks are also gathering this
information, and they begin to tighten the NBA Rookie of the Year lines in response.

So, while you can better guess which player will hoist the Wilt Chamberlain Trophy, the odds will be tighter, and your potential return on your wager will be
reduced.

Pros and cons of betting early on the NBA Rookie of the Year:

Pros:

You have an opportunity to find better value.

The betting odds are fresh and haven't been adjusted.

Cons:

There is a much higher risk as a player's performance, health, and team dynamic can change throughout the season.

You’ll find limited information to make an informed bet.

Pros and cons of waiting to bet on the NBA Rookie of the Year:

Pros:

There is a lot more information available to make an informed bet.

You don’t risk betting on a player not remotely in contention.

Cons:

There is a reduced opportunity to find value.
</item 4>

<item 5>
Line Shopping in Sports Betting - How to Find the Best Odds (Sporting News, 2023-11-06T22:21:05+00:00)
Sportsbooks will move the line or change the odds if the public bets heavily on one side.

Every sportsbook’s goal is to even out the betting on both sides of every bet. That allows sportsbooks to earn the profit built into the odds and lines without
worrying about which side wins.

The betting is rarely the same at every sportsbook. While one sportsbook may be forced to move a line or change the odds because of betting, another won’t.
That’s when you’ll spot a difference between sportsbooks. This difference may allow you to book a bet at improved odds or a more favorable line.

The lines and odds may even be different right from the jump. Either way, the differences are why line shopping always makes sense. Shop around, and you can
find opportunities to lay less, win more, and improve your chances.
</item 5>

<item 6>
The Best NBA Betting Sites and Apps for the 2023-24 Season (Sporting News, 2023-10-02T21:58:37+00:00)
What makes NBA betting special or unique?

Basketball odds are known to fluctuate more than those of other sports. The factors that play into those lines shifting can range from having only a few impact
players per team to the sheer frequency of games.

By following a single team, you can track its momentum swings, which players tend to get on hot or cold streaks, or how well they play at home vs. on the road.
You can use that information to your advantage for both pre-game and live wagers.

Similar categories of moneylines, spreads, totals, and NBA prop bets should appear across both formats.

The NBA is also special because of the number of bets available for every game, ranging from pre-game to live betting and parlays.

When you add up the standard bets, game props, player props, in-game betting, and more, there’s a staggering volume of bets for every NBA game at online
sportsbooks.
</item 6>

<item 7>
NBA Rookie of the Year Odds - Wembanyama the early favorite (Sporting News, 2023-10-04T22:54:23+00:00)
Caesars Sportsbook: Caesars knows its way around sports betting when you use its app and online sportsbook. It offers favorable odds for almost every bet type,
including NBA ROTY betting, and offers a nice variety of deposit and withdrawal options. Additionally, Caesars has a competitive welcome bonus for new players
and runs NBA betting promos for existing players.

When is the best time for NBA ROTY odds?

The earlier you bet on NBA ROTY, the better the odds. But you’ll be doing so without much information that could be key in making an educated decision.

Conversely, if you wait for the data you need for an informed bet, the odds will tighten up, and you will see less of a return on your wager.

When the lines are first released for NBA ROTY honors, the season hasn’t even started yet, so there are no statistics, trends, or player news. The pre-season
odds are so favorable because even the oddsmakers don’t have the data to craft more specific lines.
</item 7>

<item 8>
The Best NBA Betting Sites and Apps for the 2023-24 Season (Sporting News, 2023-10-02T21:58:37+00:00)
See more: Best NCAA basketball betting sites

2023-24 NBA Season

October 24 Regular season begins November 3 - December 9 In-season tournament February 16-18 All-Star Weekend 2024 April 14 Regular season ends April 16-19
Play-in tournament April 20 Playoffs begin June 6 NBA Finals Game 1

NBA spreads, money lines, and over/unders

For anyone new to NBA betting, the staple wager types are:

NBA point spreads: Bet on games via margins of victory or loss. Say the New York Knicks are -7.5 favorites over the Miami Heat. That means the Knicks need to
win the game by 8 points to win a spread bet. On the other hand, the Heat can lose by up to 7 points (or win outright).

NBA moneylines: The simplest form of betting. You pick the winner of the game, and if you’re right, you’ll get paid out based on the odds when you finalize your
bet slip.
</item 8>

<item 9>
NBA Prop Bets - Best Team and Player Props 2023/24 Season (Sporting News, 2023-10-02T23:49:35+00:00)
Pay attention to player and team news: Oddsmakers use various information to develop their lines for games and players, and you should, too. Keep tabs on player
news and injuries, when top players are being rested, how they performed in recent games, and more. You can find value in player and team props by gathering
information from multiple sources and tracking how players and their teammates perform. For example, if one of the starting guards is questionable for a game,
you might want to consider the other guard for increased points or assists since they’ll likely handle a heavier load.

Keep an eye on usage and minutes: Even slight increases to a player’s average minutes can make a difference when betting player props, especially later in the
season when aging veteran players tend to rest more. Keep yourself informed on usage and minutes trends for the players you’re considering betting on, and use
them to your advantage.
</item 9>

<item 10>
The Best NBA Betting Sites and Apps for the 2023-24 Season (Sporting News, 2023-10-02T21:58:37+00:00)
Ensure the site has a responsible gaming section.

The overall user experience should be smooth and hassle-free.

Safety and Security:

Top NBA betting sites prioritize users' safety.

Ensure your personal information is secure.

Some sportsbooks excel in ensuring cybersecurity; always go for those with solid reputations.

How to find the best NBA lines

Not all sportsbook betting odds are created equal. If you're willing to put in some legwork, you can shop for the best NBA lines to maximize your potential
return.

Movement in the odds depends on multiple factors, including team news, injuries, public betting, recent play, and even the number of games a team plays in a
row.

Oddsmakers use all the available information to craft their lines, and you can find value in shopping around for the most favorable lines and odds.

You can use our NBA odds page (see link above) to find the best lines for your prospective bet, or you can open accounts at multiple sportsbooks and do the
shopping yourself.
</item 10>
=== END ITEM id=10b892ce02027237 ===

=== ITEM id=78d5834f3a3886b2 ===
Gold sentences and facts:
Sentence 0: In the year since Altman catapulted ChatGPT to global fame, he has become Silicon Valley’s sought-after voice on the promise and potential dangers of artificial intelligence and his sudden and mostly unexplained exit brought uncertainty to the industry’s future.
  Fact 0.0: In the year since Altman catapulted ChatGPT to global fame, he has become Silicon Valley's sought-after voice on the promise and potential dangers of artificial intelligence.
  Fact 0.1: Altman had a sudden and mostly unexplained exit.
  Fact 0.2: Altman's exit brought uncertainty to the industry's future.
Sentence 1: This week, we cover Sam Altman backing a teen’s AI startup, Google’s hardware event (and first impressions of the Pixel 8 Pro), Flexport drama, and the ongoing FTX fallout.
  Fact 1.0: This week's coverage includes Sam Altman backing a teen's AI startup.
  Fact 1.1: This week's coverage includes Google's hardware event.
  Fact 1.2: This week's coverage includes first impressions of the Pixel 8 Pro.
  Fact 1.3: This week's coverage includes Flexport drama.
  Fact 1.4: This week's coverage includes the ongoing FTX fallout.
Sentence 2: Based on the board’s language and the way these giant tech companies work, this is the prevailing theory floating around right now.
  Fact 2.0: The prevailing theory floating around right now is based on the board's language.
  Fact 2.1: The prevailing theory is based on the way these giant tech companies work.

Retrieved items:
<item 1>
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
</item 1>

<item 2>
Fly with Delta frequently? Small business owners can score a solid welcome bonus and other perks with the Delta SkyMiles Gold Business Card (Fortune,
2023-12-12T21:01:43+00:00)
Cardholders can get 20% back as a statement credit when they purchase eligible pre-purchased meals, in-flight food purchases, alcoholic beverages, and audio
headsets on Delta flights. Baggage insurance. Cardholders who purchase plane, train, ship or bus ride tickets with their card are eligible for baggage insurance
coverage—which kicks in if your baggage is stolen, lost, or damaged.

Cardholders who purchase plane, train, ship or bus ride tickets with their card are eligible for baggage insurance coverage—which kicks in if your baggage is
stolen, lost, or damaged. Car Rental Loss and Damage Insurance. A cardholder can get damage and theft insurance on their car rental. Cardholders must use their
card to book the rental and decline the collision damage waiver at the agency to receive it.
</item 2>

<item 3>
Fly with Delta frequently? Small business owners can score a solid welcome bonus and other perks with the Delta SkyMiles Gold Business Card (Fortune,
2023-12-12T21:01:43+00:00)
Delta has more than 20 airline partners, so cardholders can book through airlines like Air France and Virgin Atlantic—though the 15% discount only applies to
Delta and Delta Connection® flights.

This discount is a reason to hold onto the card long-term as you could save thousands of miles over the course of a year if you are a heavy Delta user. The best
bargains on award flights are found through Delta SkyMiles Award Deals. Recent deals from Seattle (SEA) have included Chicago (ORD) for 8,000 miles or Auckland,
New Zealand (AKL) for 64,000 one way in economy plus taxes and fees. Flash sales come and go so it’s a good idea to keep an eye on the site.

Other options include booking travel through Delta Vacations, using miles for upgrades or even buying champagne at the Delta Sky Club. That said, your best
value will most likely be on award tickets.
</item 3>

<item 4>
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
</item 4>

<item 5>
Fly with Delta frequently? Small business owners can score a solid welcome bonus and other perks with the Delta SkyMiles Gold Business Card (Fortune,
2023-12-12T21:01:43+00:00)
For $99 a year, cardholders receive benefits like priority boarding, the first checked bag benefit (and for guests), discounts on in-flight purchases, and
baggage insurance. The card, however, doesn’t provide access to Delta Sky Club or Delta’s airport lounges.

The rewards structure on this card is decent, but it’s nothing to write home about—the card offers 2x miles on just a handful of categories like advertising,
shipping, and dining.

A big benefit of the Delta SkyMiles Gold Business Card is the welcome bonus: 75,000 Bonus Miles after you spend $6,000 in purchases on your new Card in your
first 6 months of card opening. And cardholders who use their miles to purchase flights get even more benefits—customers receive a 15% discount when they use
miles to buy Delta flights on the Fly Delta app or the Delta website.
</item 5>

<item 6>
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
</item 6>

<item 7>
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
</item 7>

<item 8>
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
</item 8>

<item 9>
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
</item 9>

<item 10>
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
</item 10>
=== END ITEM id=78d5834f3a3886b2 ===
