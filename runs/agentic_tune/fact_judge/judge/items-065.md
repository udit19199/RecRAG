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

=== ITEM id=37a6e7391faba682 ===
Gold sentences and facts:
Sentence 0: Unfortunately, she did say that she had conversations with investors as part of their due diligence — and, of course, Alameda was taking on losses from FTX to keep FTX’s balance sheet pristine.
  Fact 0.0: Unfortunately, she said she had conversations with investors as part of their due diligence.
  Fact 0.1: Alameda was taking on losses from FTX.
  Fact 0.2: Alameda was taking on losses from FTX to keep FTX's balance sheet pristine.
Sentence 1: During cross-examination, Huang said Paradigm pressed Bankman-Fried on the board issue and was told he didn't want investors as directors but he did plan on having a board with experts.
  Fact 1.0: During cross-examination, Huang said Paradigm pressed Bankman-Fried on the board issue.
  Fact 1.1: Huang said Paradigm was told Bankman-Fried did not want investors as directors.
  Fact 1.2: Huang said Bankman-Fried did plan on having a board with experts.

Retrieved items:
<item 1>
Sex, Signal messages, and sabotaging FTX's code: SBF execs and Bahamas roommates tell all in court (Cnbc | World Business News Leader,
2023-10-06T20:17:00+00:00)
Additionally, he was promised that Alameda had no preferential treatment on the FTX platform, even though the hedge fund was one of its top traders. Huang said
he was concerned about FTX's lack of a board of directors, but he eventually invested anyway. During cross-examination, Huang said Paradigm pressed
Bankman-Fried on the board issue and was told he didn't want investors as directors but he did plan on having a board with experts. — CNBC's Dawn Giel
contributed to this report.

Graph facts:
- SBF Trial: The latest updates from the FTX collapse’s courtroom drama -MENTIONS-> Paradigm
- Meet Alliance DAO’s latest accelerator cohort of startups building in crypto winter -MENTIONS-> Paradigm
- Matt Huang -WORKS_FOR-> Paradigm
- Paradigm -PARTICIPATED_IN-> class-action lawsuit
- Paradigm -ACCUSED_OF-> defrauding customers

Entities: Paradigm, Bankman-Fried, Alameda, Signal, Bahamas, FTX, Sex, Signal messages, and sabotaging FTX's code: SBF execs and Bahamas roommates tell all in
court, Huang, Dawn Giel, board of directors, CNBC
</item 1>

<item 2>
Today the FTX jury suffered through a code review (The Verge, 2023-10-06T22:21:46+00:00)
In late 2019, Alameda had a negative balance on FTX. Because they sat in an open-plan office, Wang heard a trader ask Bankman-Fried if it was okay to keep
withdrawing money from that account. Bankman-Fried said it was okay as long as it was lower than FTX’s total trading revenue. He was the CEO of both companies
at the time.

Alameda also allegedly kept FTX from looking bad by assuming some of its losses.

The amount that Alameda was allowed to go negative began to creep up. Around 2019 or 2020, Wang checked the database and discovered that Alameda was negative by
about $200 million, which was more than the $150 million FTX made in revenue. That had to mean Alameda was taking customer money. It surprised him and he says
he discussed it with Bankman-Fried, who told him that he just needed to take into account the value of FTT. Wang said he trusted Bankman-Fried’s judgment and
didn’t pursue the issue further.

Graph facts:
- SBF’s trial starts soon, but how did he — and FTX — get here? -MENTIONS-> Bankman-Fried
- In the end, the FTX trial was about the friends screwed along the way -MENTIONS-> Bankman-Fried
- Sex, Signal messages, and sabotaging FTX's code: SBF execs and Bahamas roommates tell all in court -MENTIONS-> Bankman-Fried
- Sam Altman backs teens’ startup, Google unveils the Pixel 8 and TikTok tests an ad-free tier -MENTIONS-> Bankman-Fried
- Bankman-Fried -SAID-> Alameda Research

Entities: Bankman-Fried, Alameda, FTX, Wang, FTT, FTX jury, Today the FTX jury suffered through a code review, The Verge
</item 2>

<item 3>
Sex, Signal messages, and sabotaging FTX's code: SBF execs and Bahamas roommates tell all in court (Cnbc | World Business News Leader,
2023-10-06T20:17:00+00:00)
Those privileges ultimately resulted in Alameda owing FTX $8 billion worth of customer deposits. "We gave special privileges on FTX that gave unlimited
withdrawals on the platform to Alameda," Wang said. Alameda was allowed to withdraw and transfer those funds and had a $65 billion line of credit. "When
customers deposited USD, it went to Alameda," he said. "It existed in the computer code. Alameda could have negative balances and unlimited withdrawals." That
"bug" in the code was written by Nishad Singh, who was FTX's director of engineering, and reviewed by Wang. Bankman-Fried was calling the shots, Wang said. Wang
also told the court about a $1 million personal loan he received and a $200 million to $300 million loan in his name from Alameda that was never deposited into
his account, but rather was used to make investments into other companies on behalf of FTX.

Graph facts:
- SBF’s trial starts soon, but how did he — and FTX — get here? -MENTIONS-> Bankman-Fried
- In the end, the FTX trial was about the friends screwed along the way -MENTIONS-> Bankman-Fried
- Sam Altman backs teens’ startup, Google unveils the Pixel 8 and TikTok tests an ad-free tier -MENTIONS-> Bankman-Fried
- Today the FTX jury suffered through a code review -MENTIONS-> Bankman-Fried
- Bankman-Fried -SAID-> Alameda Research

Entities: Bankman-Fried, Alameda, Nishad Singh, Signal, Bahamas, FTX, Sex, Signal messages, and sabotaging FTX's code: SBF execs and Bahamas roommates tell all
in court, Wang, CNBC, Code
</item 3>

<item 4>
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

Graph facts:
- Matt Huang -WORKS_FOR-> Paradigm
- Sex, Signal messages, and sabotaging FTX's code: SBF execs and Bahamas roommates tell all in court -MENTIONS-> Matt Huang
- Is Sam Bankman-Fried a bad ‘man’ or a good ‘boy’? Lawyers swap opening statements before first witnesses take the stand -MENTIONS-> Matt Huang
- SBF Trial: The latest updates from the FTX collapse’s courtroom drama -MENTIONS-> Matt Huang
- Matt Huang -WORKS_FOR-> Sequoia Capital

Entities: Matt Huang, Caroline Ellison, Bankman-Fried, Alameda, Singh, FTX, Wang, MobileCoin, Paradigm Capital, exploit in MobileCoin, Today the FTX jury
suffered through a code review, The Verge
</item 4>

<item 5>
Sex, Signal messages, and sabotaging FTX's code: SBF execs and Bahamas roommates tell all in court (Cnbc | World Business News Leader,
2023-10-06T20:17:00+00:00)
That was all done by Bankman-Fried, he testified. In early 2020, Wang said he discovered for the first time Alameda's negative balance exceeded FTX's revenue,
an indication that Alameda was taking customer funds. Wang said he brought this to Bankman-Fried's attention several times. In late 2021, Wang discovered
Alameda had withdrawn $3 billion from its $65 billion line of credit. Wang's compensation was a base salary of $200,000 per year plus stock. He owned roughly
17% of FTX. Even though they were co-founders, "ultimately it was Sam's decision to make" when there were disagreements, he said.
</item 5>

<item 6>
The jury finally hears from Sam Bankman-Fried (The Verge, 2023-10-28T00:12:41+00:00)
Well, Bankman-Fried remembers it differently. See, Yedidia was just asking about Alameda’s risk profile, and Bankman-Fried wasn’t talking about insolvency at
all.

Also, when Singh and Bankman-Fried had the dramatic balcony conversation at their penthouse, it was just that Singh thought Alameda’s liabilities had gotten too
high, and FTX was spending too much money on marketing. But Bankman-Fried still thought that Alameda had more assets than liabilities, so it was all fine, and
besides, if Singh thought he was going to be better at marketing, he could take it over. It didn’t have anything to do with the money Alameda owed FTX at all.

Of course not! Bankman-Fried didn’t learn about the $8 billion liability associated with Alameda until October 2022, he said. And he learned it all by himself,
by looking at a computer database. When he found it, he was “very surprised!”

Graph facts:
- The jury finally hears from Sam Bankman-Fried -MENTIONS-> Gary Wang
- The jury finally hears from Sam Bankman-Fried -MENTIONS-> Caroline Ellison
- The jury finally hears from Sam Bankman-Fried -MENTIONS-> Mark Cohen
- The jury finally hears from Sam Bankman-Fried -MENTIONS-> Adam Yedidia
- The jury finally hears from Sam Bankman-Fried -MENTIONS-> Securities and Exchange Commission

Entities: The jury finally hears from Sam Bankman-Fried, Alameda, balcony conversation, penthouse, Singh, FTX, Yedidia, Sam Bankman-Fried, The Verge
</item 6>

<item 7>
The jury finally hears from Sam Bankman-Fried (The Verge, 2023-10-28T00:12:41+00:00)
Bankman-Fried said he worked 12 to 22 hours a day, and took one day off every couple of months. Because FTX had grown so much, he could no longer run both
companies, he said. Bankman-Fried handed the company off to Caroline Ellison and Sam Trabucco, who immediately after being named co-CEO promptly drifted away to
early retirement. (Quiet quitting king!) Bankman-Fried did remain involved in hedging and risk at Alameda, though.

About that $1 billion of revenue in 2021: Bankman-Fried definitely did not know that Singh, his employee, had backdated interest payments to get FTX “over the
line” to $1 billion. See, he’d just asked his employees to check and see if there was any source of funds that was missing to get to $1 billion. This testimony
was especially rambling.

Oh, also that MobileCoin loss? The one Wang said Alameda took to keep off FTX’s balance sheet?

Graph facts:
- SBF Trial: The latest updates from the FTX collapse’s courtroom drama -MENTIONS-> Caroline Ellison
- Is Sam Bankman-Fried a bad ‘man’ or a good ‘boy’? Lawyers swap opening statements before first witnesses take the stand -MENTIONS-> Caroline Ellison
- SBF’s trial starts soon, but how did he — and FTX — get here? -MENTIONS-> Caroline Ellison
- Sam Bankman-Fried didn’t ask where the $8 billion went -MENTIONS-> Caroline Ellison
- Sam Bankman-Fried was a terrible boyfriend -MENTIONS-> Caroline Ellison

Entities: Caroline Ellison, The jury finally hears from Sam Bankman-Fried, Alameda, Sam Trabucco, MobileCoin loss, Singh, FTX, Wang, Sam Bankman-Fried,
MobileCoin, The Verge
</item 7>

<item 8>
Graph query row:
article: The jury finally hears from Sam Bankman-Fried
publisher: The Verge
passages: ['The jury finally hears from Sam Bankman-Fried (The Verge, 2023-10-28T00:12:41+00:00)\nIt is honestly kind of incredible to watch a man torpedo his
own credibility on direct testimony. We’re not even at the cross yet, and the judge has already instructed him to answer the question he’s being asked by his
own lawyer.\n\nThe jury is watching all of this intently.\n\nThe main thing that’s been clear so far from Bankman-Fried’s testimony is that the man really loves
the sound of his own voice. So far, the count of “Objection, narrative!” to Bankman-Fried’s answers, followed by “Sustained” is at three.\n\nAlso, sometimes
when Bankman-Fried says “we,” he only means himself.\n\nSam Bankman-Fried, the founder of failed cryptocurrency exchange FTX, has been found guilty on seven
counts including charges of wire fraud. FTX was a fraud “from the start,” the Securities and Exchange Commission alleged — with a “multi-billion-dollar
deficiency caused by his own misappropriation of customer funds.” Follow along for all the latest news and regular updates from the trial.', 'The jury finally
hears from Sam Bankman-Fried (The Verge, 2023-10-28T00:12:41+00:00)\nFollow along for all the latest news and regular updates from the trial.\n\nYesterday,
during an evidentiary hearing, Bankman-Fried was repeatedly scolded by Judge Lewis Kaplan for not answering prosecutor Danielle Sassoon’s questions on
cross-examination. Today, Bankman-Fried was scolded by Kaplan for not answering his counsel Mark Cohen’s questions on direct examination. Bankman-Fried has also
occasionally interrupted Cohen with “yes” and “yup.”\n\nWe spent a lot of the morning explaining vocabulary. I will spare you the full list, though I will say
that explaining “Amazon Web Services” and “database” was a bit too detail-oriented. Then Bankman-Fried tried to define “market manipulation.” After
Bankman-Fried gave his definition, Kaplan told the jury that he was the final authority on that, thanks.\n\nIn fairness to Bankman-Fried, he has been clearer
and much easier to understand than he was at the evidentiary hearing yesterday. There was a minimum of word salad today.', 'The jury finally hears from Sam
Bankman-Fried (The Verge, 2023-10-28T00:12:41+00:00)\nThere was a minimum of word salad today. I don’t know if he was more relaxed, or he’d just been more
rehearsed, but I will certainly be watching to see if he suddenly becomes much less coherent when Sassoon gets him for the cross.\n\nHere is the story of FTX,
from his point of view.\n\nBankman-Fried, who informed us he’s “somewhat introverted, naturally,” gave us a rather prolonged tour of his pre-Alameda Research
life, which I will skip. In 2017, during a crypto bull run, he started his cryptocurrency trading firm. He knew “basically nothing” about cryptocurrency at the
time, he explained, but he wanted to do arbitrage on it anyway.\n\nAlameda Research was named for Alameda County in California, which was where its first office
was set up. As for its name, here’s what Bankman-Fried said on the stand:\n\nEffectively, we wanted to be under the radar at that point in time.', 'The jury
finally hears from Sam Bankman-Fried (The Verge, 2023-10-28T00:12:41+00:00)\nI didn’t want to call it Sam’s Crypto Trading Firm or anything like that. We —
there are a lot of competitors and people who we didn’t particularly want to know what we were building out because they would race to do it. “Research” was a
sort of generic word, which filled out the company name. And that was — it was far better than the internal name that we had at that point, which was Wireless
Mouse.\n\nI would find this much more believable if I hadn’t already watched a video of Bankman-Fried explaining on a podcast that the name made it easier to
get a bank account. That happened during the first day of Gary Wang’s testimony. Bankman-Fried was there, too. You know who else was there? The jury.\n\nAnyway,
Bankman-Fried went on a hiring spree for Alameda. He rounded up his merry gang of alleged co-conspirators. First, Wang, to program the computers. Then, Nishad
Singh, about a month after founding Alameda Research.', 'The jury finally hears from Sam Bankman-Fried (The Verge, 2023-10-28T00:12:41+00:00)\nThen, Nishad
Singh, about a month after founding Alameda Research. Finally, Caroline Ellison.\n\nNot having a risk team, when you are any kind of financial anything, is
certainly a choice\n\nThough Bankman-Fried was the CEO, and also the majority owner, he wanted to be clear: he did not supervise Wang’s direct work. Anyway,
after a bunch of wildly successful arbitrage — 50 percent to 100 percent annualized returns, per his testimony — he decided to found a cryptocurrency exchange,
FTX. He figured he’d fail; that there was only a 20 percent chance of success. Bankman-Fried did not define what he expected the time period to be on this
estimate, but arguably 20 percent was a much higher chance of success than FTX would enjoy once Alameda dipped into the customer deposits.\n\nBy the way,
because he was such a good guy, Bankman-Fried made a point of “periodically” handling support tickets himself.', 'The jury finally hears from Sam Bankman-Fried
(The Verge, 2023-10-28T00:12:41+00:00)\n“I worried if I didn’t, I would lose touch with the actual concerns of the customers,” he testified. What he did not do
was create a risk team, which he is now characterizing as a “big mistake.”\n\nRisk is an inherent part of a futures exchange, which is even more like a casino
than regular cryptocurrency. Not having a risk team, when you are any kind of financial anything, is certainly a choice. It is especially a choice when you go
around telling everyone your crypto exchange is very good and safe.\n\nFTX’s big selling point was its “risk engine,” which was supposed to prevent big losses
that would then be spread around all the rest of the customers. But Bankman-Fried testified that in 2020, the “risk engine was effectively sagging under the
weight” of the exchange’s rapid growth. So its time to liquidation went up — it took minutes to determine which accounts needed to be liquidated.', 'The jury
finally hears from Sam Bankman-Fried (The Verge, 2023-10-28T00:12:41+00:00)\nAs a result, at one point the risk engine got stuck in a catastrophic feedback loop
that would have created losses in the “trillions of dollars,” Bankman-Fried testified. As part of that feedback loop, Alameda teetered on the brink of
liquidation, which “would have disastrous consequences” for FTX.\n\n“At the time, I wasn’t entirely sure what was happening.”\n\nBecause of that experience,
Bankman-Fried suggested an “alert” or “delay” that would keep Alameda from being liquidated by a bug. This is the supposed origin story of “allow_negative,”
which Bankman-Fried says was the eventual result of that conversation, and that he says he didn’t know about until very recently.\n\nThere is a problem with
this story. “Allow_negative” was coded and switched on in 2019. I saw the code in court, and so did Bankman-Fried, who was also there for the testimony. Perhaps
you are wondering, was the jury also there? Reader, it was.', 'The jury finally hears from Sam Bankman-Fried (The Verge, 2023-10-28T00:12:41+00:00)\nPerhaps you
are wondering, was the jury also there? Reader, it was.\n\nBankman-Fried denied he knew about the effectively infinite line of credit Alameda Research received
from FTX. This argument was peculiar; essentially my take-away was that the CEO of a financial company simply didn’t pay attention to finances.\n\nFTX couldn’t
get bank accounts right away. Bankman-Fried anticipated it would take a year or two. Rather than wait, he decided to use Alameda as the “payment provider” for
bank transfers. “My understanding at the time was that there were teams managing the process,” he said. “At the time, I wasn’t entirely sure what was
happening.”\n\nWell, sure, understandable! He’s an introvert!\n\nBankman-Fried definitely did not know that Singh, his employee, had backdated interest payments
to get FTX “over the line” to $1 billion\n\nIn 2021, FTX was growing to millions of users, with $1 billion of revenue.', 'The jury finally hears from Sam
Bankman-Fried (The Verge, 2023-10-28T00:12:41+00:00)\nBankman-Fried said he worked 12 to 22 hours a day, and took one day off every couple of months. Because
FTX had grown so much, he could no longer run both companies, he said. Bankman-Fried handed the company off to Caroline Ellison and Sam Trabucco, who
immediately after being named co-CEO promptly drifted away to early retirement. (Quiet quitting king!) Bankman-Fried did remain involved in hedging and risk at
Alameda, though.\n\nAbout that $1 billion of revenue in 2021: Bankman-Fried definitely did not know that Singh, his employee, had backdated interest payments to
get FTX “over the line” to $1 billion. See, he’d just asked his employees to check and see if there was any source of funds that was missing to get to $1
billion. This testimony was especially rambling.\n\nOh, also that MobileCoin loss? The one Wang said Alameda took to keep off FTX’s balance sheet?', 'The jury
finally hears from Sam Bankman-Fried (The Verge, 2023-10-28T00:12:41+00:00)\nThe one Wang said Alameda took to keep off FTX’s balance sheet? Yeah, so it was a
totally innocent thing where what actually happened was that Bankman-Fried thought it was appropriate that Alameda take the position as a backstop liquidity
provider, that’s all.\n\nIn June 2022, Bankman-Fried heard about the account called “fiat@ftx” tracking how much money Alameda owed to FTX, he testified. He did
not know what it was and did not bother to find out. He was busy! That was when Bankman-Fried directed Ellison to repay Alameda’s lenders, because he thought
Alameda was good for it. He also gave BlockFi and Voyager, two crypto lenders, some capital infusions for good measure.\n\nHe was “very surprised!”\n\nRemember
that testimony Adam Yedidia gave about a conversation with Bankman-Fried in August 2022 about the enormous amount of money Alameda owed FTX? Well, Bankman-Fried
remembers it differently.']
connections: [{'entity': 'FTX Ventures', 'related': 'SBF Trial: The latest updates from the FTX collapse’s courtroom drama', 'relationship': 'MENTIONS'},
{'entity': 'FTX collapse', 'related': 'SBF Trial: The latest updates from the FTX collapse’s courtroom drama', 'relationship': 'MENTIONS'}, {'entity': 'Matt
Huang', 'related': 'SBF Trial: The latest updates from the FTX collapse’s courtroom drama', 'relationship': 'MENTIONS'}, {'entity': 'Paradigm', 'related': 'SBF
Trial: The latest updates from the FTX collapse’s courtroom drama', 'relationship': 'MENTIONS'}, {'entity': 'Alameda Research', 'related': 'SBF Trial: The
latest updates from the FTX collapse’s courtroom drama', 'relationship': 'MENTIONS'}, {'entity': 'Alameda', 'related': 'SBF Trial: The latest updates from the
FTX collapse’s courtroom drama', 'relationship': 'MENTIONS'}, {'entity': 'FTX', 'related': 'SBF Trial: The latest updates from the FTX collapse’s courtroom
drama', 'relationship': 'MENTIONS'}, {'entity': 'Sam Bankman-Fried', 'related': 'SBF Trial: The latest updates from the FTX collapse’s courtroom drama',
'relationship': 'MENTIONS'}, {'entity': 'SBF Trial: The latest updates from the FTX collapse’s courtroom drama', 'related': 'FTX Ventures', 'relationship':
'MENTIONS'}, {'entity': 'SBF Trial: The latest updates from the FTX collapse’s courtroom drama', 'related': 'FTX collapse', 'relationship': 'MENTIONS'}]
</item 8>

<item 9>
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
</item 9>

<item 10>
The jury finally hears from Sam Bankman-Fried (The Verge, 2023-10-28T00:12:41+00:00)
The one Wang said Alameda took to keep off FTX’s balance sheet? Yeah, so it was a totally innocent thing where what actually happened was that Bankman-Fried
thought it was appropriate that Alameda take the position as a backstop liquidity provider, that’s all.

In June 2022, Bankman-Fried heard about the account called “fiat@ftx” tracking how much money Alameda owed to FTX, he testified. He did not know what it was and
did not bother to find out. He was busy! That was when Bankman-Fried directed Ellison to repay Alameda’s lenders, because he thought Alameda was good for it. He
also gave BlockFi and Voyager, two crypto lenders, some capital infusions for good measure.

He was “very surprised!”

Remember that testimony Adam Yedidia gave about a conversation with Bankman-Fried in August 2022 about the enormous amount of money Alameda owed FTX? Well,
Bankman-Fried remembers it differently.

Graph facts:
- Is Sam Bankman-Fried a bad ‘man’ or a good ‘boy’? Lawyers swap opening statements before first witnesses take the stand -MENTIONS-> Adam Yedidia
- Adam Yedidia -MENTIONS-> MIT
- Sam Bankman-Fried didn’t ask where the $8 billion went -MENTIONS-> Adam Yedidia
- In the end, the FTX trial was about the friends screwed along the way -MENTIONS-> Adam Yedidia
- Is Sam Bankman-Fried’s defense even trying to win? -MENTIONS-> Adam Yedidia

Entities: Adam Yedidia, The jury finally hears from Sam Bankman-Fried, Alameda, Voyager, fiat@ftx, Ellison, FTX, Wang, Sam Bankman-Fried, BlockFi, The Verge
</item 10>
=== END ITEM id=37a6e7391faba682 ===

=== ITEM id=f07c522c027365b4 ===
Gold sentences and facts:
Sentence 0: And so, alongside the big improvements, Valve has made a gaggle of minor physical upgrades with its new iteration.
  Fact 0.0: Alongside the big improvements, Valve has made a gaggle of minor physical upgrades.
  Fact 0.1: Valve's new iteration includes these minor physical upgrades.
Sentence 1: We look at Valve and we see a store that could be both, but they’ve decided to focus exclusively on games.
  Fact 1.0: The authors look at Valve and see a store that could be both.
  Fact 1.1: Valve has decided to focus exclusively on games.
Sentence 2: The Steam Deck OLED will go on sale on November 16th at 1pm ET, and Valve has units ready to ship that same day.
  Fact 2.0: The Steam Deck OLED will go on sale on November 16th at 1pm ET.
  Fact 2.1: Valve has units ready to ship on November 16th.

Retrieved items:
<item 1>
The Steam Deck OLED improves much more than its screen (Polygon, 2023-11-09T18:00:00+00:00)
When we award the Polygon Recommends badge, it’s because we believe the recipient is uniquely thought-provoking, entertaining, inventive, or fun — and worth
fitting into your schedule. If you want curated lists of our favorite media, check out What to Play and What to Watch .

It appears, at first glance, that Valve cribbed from Nintendo’s playbook with the Switch OLED, for which Nintendo upgraded the device’s screen but not its guts.
In terms of raw power, the Steam Deck OLED is identical to the original Steam Deck. Valve says this synchronicity is meant in part to continue to provide
developers with a single goal when optimizing their games — though it promises that the Steam Deck 2 (which the company hopes to release in two to three years)
will have a generational power leap.

For those who haven’t tried a Steam Deck, though, I wouldn’t wait for its sequel. Because while Valve hasn’t upped the horsepower, it has improved practically
everything else.

Graph facts:
- The best video games of the year so far -MENTIONS-> Nintendo
- Microsoft Activision: What does deal mean for gamers? -MENTIONS-> Nintendo
- The 53 best Black Friday deals we could find at Amazon, Walmart, Target and more -MENTIONS-> Nintendo
- The best October Prime Day gaming deals -MENTIONS-> Nintendo
- Super Mario Bros. Wonder review: The joy of pure imagination -MENTIONS-> Nintendo

Entities: Nintendo, Switch OLED, Steam Deck 2, The Steam Deck OLED improves much more than its screen, Valve, Polygon, Steam Deck, Steam Deck OLED, Polygon
Recommends badge, What to Play, What to Watch
</item 1>

<item 2>
The Steam Deck OLED improves much more than its screen (Polygon, 2023-11-09T18:00:00+00:00)
In early 2022, Valve released my favorite video game console. But even I had some gripes. With the Steam Deck, I could — for the first time — play the bulk of
my sprawling collection of video games wherever I’d like. But I tended to bundle it with a huge portable power supply. I also couldn’t quite shirk how much
better games looked on our family’s OLED TV.

And yet, for me, an adult with limited time to play video games alone in a quiet room, nothing could beat the Steam Deck’s freedom. Barring the occasional
diversion, Valve’s handheld PC had served as my preferred gaming option ever since I published our review. Then, last week, the Steam Deck OLED arrived. It’s as
if the Steam Deck team collected a laundry list of complaints about the original, then responded to all of them one by one.

Polygon Recommends is our way of endorsing our favorite games, movies, TV shows, comics, tabletop books, and entertainment experiences.

Graph facts:
- The Steam Deck OLED improves much more than its screen -MENTIONS-> Switch OLED
- The Steam Deck OLED improves much more than its screen -MENTIONS-> battery
- The Steam Deck OLED improves much more than its screen -MENTIONS-> Nintendo
- The Steam Deck OLED improves much more than its screen -MENTIONS-> OLED screen
- The Steam Deck OLED improves much more than its screen -MENTIONS-> Steam Deck 2

Entities: Steam Deck team, The Steam Deck OLED improves much more than its screen, Valve, Polygon Recommends, Polygon, OLED TV, Steam Deck, Steam Deck OLED
</item 2>

<item 3>
The Steam Deck OLED arrives November 16 with an improved screen and longer battery life (Engadget, 2023-11-09T18:00:32+00:00)
Other improvements include support for both Wi-Fi 6E and Bluetooth 5.3, together with a new antenna array that will give a better experience when using both at
the same time. There’s also a reworked power supply (it has a logo now!) with a longer 2.5m cable that will charge the OLED model from 20 to 80 percent in “as
little as 45 minutes.” The thumbsticks have also been made 1mm taller, and the touch pads' haptics are slightly sharper.

Valve

The Steam Deck OLED starts at $549, and will replace all but the entry-level model in the range.

Graph facts:
- Tim Sweeney on Epic’s victory royale over Google -MENTIONS-> Valve
- The best October Prime Day gaming deals -MENTIONS-> Valve
- The Epic v. Google trial may come down to simple v. complicated -MENTIONS-> Valve
- The Steam Deck OLED improves much more than its screen -MENTIONS-> Valve
- Steam Deck OLED review: It's just better -MENTIONS-> Valve

Entities: Valve, Steam Deck OLED, Wi-Fi 6E, Bluetooth 5.3, antenna array, power supply, thumbsticks, touch pads, The Steam Deck OLED arrives November 16 with an
improved screen and longer battery life, Engadget
</item 3>

<item 4>
The Steam Deck OLED improves much more than its screen (Polygon, 2023-11-09T18:00:00+00:00)
Yes. But you might find that the screen and the battery justify the double dip. For someone like me, who puts hundreds of hours into the Steam Deck each year,
the upgrade is inescapable.

And if you don’t have a Steam Deck? How many times can I say it: This is my favorite console ever. And it just got better.

The Steam Deck OLED will be available Nov. 16 to purchase through the Steam storefront. The hardware was reviewed with a unit provided by Valve. Vox Media has
affiliate partnerships. These do not influence editorial content, though Vox Media may earn commissions for products purchased via affiliate links. You can find
additional information about Polygon’s ethics policy here .

Graph facts:
- The Steam Deck OLED improves much more than its screen -MENTIONS-> battery
- The Steam Deck OLED improves much more than its screen -MENTIONS-> Nintendo
- The Steam Deck OLED improves much more than its screen -MENTIONS-> OLED screen
- The Steam Deck OLED improves much more than its screen -MENTIONS-> Steam Deck team
- The Steam Deck OLED improves much more than its screen -MENTIONS-> Switch OLED

Entities: The Steam Deck OLED improves much more than its screen, Valve, Steam storefront, Polygon, Vox Media, Steam Deck, Steam Deck OLED
</item 4>

<item 5>
Steam Deck OLED review: It's just better (Engadget, 2023-11-09T18:00:38+00:00)
The updated display is the highlight of the device, while things like faster charging, improved antennas and smoother haptics are welcome bonuses. More
importantly, this is Valve renewing its commitment to portable PC gaming, and that’s a relief to see. The Steam Deck is the granddaddy of handheld PCs and Valve
has the resources to continue innovating in this space. For now, the OLED model is a half-step toward the Steam Deck 2, which may be the final version of the
device — unless Valve finally learns how to count to three.

Graph facts:
- Steam Deck OLED review: It's just better -MENTIONS-> Sony
- Steam Deck OLED review: It's just better -MENTIONS-> HDR
- Steam Deck OLED review: It's just better -MENTIONS-> Switch OLED upgrade
- Steam Deck OLED review: It's just better -MENTIONS-> OLED screen
- Steam Deck OLED review: It's just better -MENTIONS-> LCD

Entities: Steam Deck OLED review: It's just better, handheld PCs, portable PC gaming, Steam Deck 2, Valve, Steam Deck, Steam Deck OLED, Engadget
</item 5>

<item 6>
Graph query row:
article: Steam Deck OLED review: It's just better
published_on: 2023-11-09T18:00:38+00:00
publisher: Engadget
passage: Steam Deck OLED review: It's just better (Engadget, 2023-11-09T18:00:38+00:00)
The first thing to know about the Steam Deck OLED is that it’s not the Steam Deck 2. The Steam Deck OLED is a mid-cycle refresh from Valve, similar to the
Switch OLED upgrade, but there’s a lot more going on internally here. Its screen is better, its battery life is better, its thumbsticks are better, the haptics
are better, and its chip and thermals are better. One thing that hasn’t changed is its form factor — this is still one of the biggest babies on the handheld
block.

The new Steam Deck’s main upgrade is its OLED screen, which replaces the original LCD and, crucially, has native support for HDR and faster frame rates. The
OLED also has smaller bezels, making the display larger at 7.4 inches, compared with the original 7-inch screen.
</item 6>

<item 7>
The Steam Deck OLED arrives November 16 with an improved screen and longer battery life (Engadget, 2023-11-09T18:00:32+00:00)
Surprise! Valve has announced a new Steam Deck and — double surprise — we’ve already reviewed it. It might not look like a big change from the outside, but the
Steam Deck OLED is a thorough mid-cycle refresh of a machine that’s not even two years old. At the same time, Valve is also adjusting the configuration of its
entry-level model, making it a far more compelling option. Apologies to anyone who literally bought a Steam Deck yesterday.

At an event held for press prior to the device's launch, Greg Coomer, a product designer at Valve, called the new model "the definitive version of the Steam
Deck." He explained that the device contains many things the company wanted to include at the LCD model's launch but wasn't able to, along with some additions
based on community feedback.

As its name suggests, the most noticeable upgrade to the Steam Deck OLED is the display. The original Steam Deck had a 7-inch 800p LCD running at 60Hz.

Graph facts:
- Tim Sweeney on Epic’s victory royale over Google -MENTIONS-> Valve
- The best October Prime Day gaming deals -MENTIONS-> Valve
- The Epic v. Google trial may come down to simple v. complicated -MENTIONS-> Valve
- The Steam Deck OLED improves much more than its screen -MENTIONS-> Valve
- Steam Deck OLED review: It's just better -MENTIONS-> Valve

Entities: Valve, Steam Deck, Steam Deck OLED, Greg Coomer, The Steam Deck OLED arrives November 16 with an improved screen and longer battery life, Engadget
</item 7>

<item 8>
The Steam Deck OLED arrives November 16 with an improved screen and longer battery life (Engadget, 2023-11-09T18:00:32+00:00)
When asked about the potential for future hardware — like a return of the Steam Machine or an updated Steam Controller — there were no firm updates, but there
does seem to be something happening. Griffais agreed that enabling the Steam Deck to work as a hybrid living room console was a "side goal" of the team, and
noted that everything Valve does to improve SteamOS will benefit other devices. Coomer added that Valve is still working internally on "a bunch of controllers
and prototypes." If SteamOS is being used on larger displays, the team would want to augment that experience with better input options. "We don't have any
announcements on that front, but it's why we're building prototypes and thinking about the future there," Coomer said.

The Steam Deck OLED will be available to order on November 16th at 1PM ET. Valve expects units to begin shipping as soon as orders start to roll in. The updated
pricing on the existing LCD models is already in effect.

Graph facts:
- Tim Sweeney on Epic’s victory royale over Google -MENTIONS-> Valve
- The best October Prime Day gaming deals -MENTIONS-> Valve
- The Epic v. Google trial may come down to simple v. complicated -MENTIONS-> Valve
- The Steam Deck OLED improves much more than its screen -MENTIONS-> Valve
- Steam Deck OLED review: It's just better -MENTIONS-> Valve

Entities: Valve, Steam Deck, Steam Deck OLED, The Steam Deck OLED arrives November 16 with an improved screen and longer battery life, Steam Machine, Steam
Controller, Griffais, SteamOS, Coomer, Engadget
</item 8>

<item 9>
Steam Deck OLED review: It's just better (Engadget, 2023-11-09T18:00:38+00:00)
Currently, more than 10,000 titles on Steam fall under the playable or verified labels, with more added every day.

When will the Steam Deck OLED be available?

The Steam Deck OLED will go on sale on November 16th at 1pm ET, and Valve has units ready to ship that same day. The company has lowered the price of its LCD
lineup and will sell through its backstock of 64GB and 512GB models until they’re all gone. It’ll continue selling the 256GB LCD model at a reduced price of
$400, while the 512GB OLED model will cost $550 and the 1TB OLED model will cost $650. There’s also a limited edition 1TB OLED version priced at $680 that comes
with a special carrying case, a translucent gray body and orange accents.

Valve

Wrap-up

According to the developers I spoke with at Valve, this is the definitive version of the Steam Deck.

Graph facts:
- Steam Deck OLED review: It's just better -MENTIONS-> Switch OLED upgrade
- Steam Deck OLED review: It's just better -MENTIONS-> OLED screen
- Steam Deck OLED review: It's just better -MENTIONS-> Sony
- Steam Deck OLED review: It's just better -MENTIONS-> HDR
- Steam Deck OLED review: It's just better -MENTIONS-> LCD

Entities: Steam Deck OLED sale, 64GB Steam Deck LCD model, 512GB Steam Deck LCD model, 256GB Steam Deck LCD model, 512GB Steam Deck OLED model, 1TB Steam Deck
OLED model, Limited edition 1TB OLED version, developers at Valve, Steam Deck OLED review: It's just better, Steam, Valve, Steam Deck OLED, Engadget
</item 9>

<item 10>
The Steam Deck OLED arrives November 16 with an improved screen and longer battery life (Engadget, 2023-11-09T18:00:32+00:00)
The range still tops out at $649, but you’re getting a lot more for your money: All of the hardware improvements, 1TB of storage instead of 512GB and an
improved carry case that can pop out into a slimmer shell that’s smaller and easier to fit in a bag. As mentioned, it also has the same anti-glare etched glass
as the outgoing 512GB LCD model.

Finally, there’s also a Limited Edition version of the 1TB model, with a translucent shell and a customized carry case, which will be sold at $679. We imagine
stocks will not last very long, as it looks pretty rad:

Valve

While there are no concrete updates on future hardware developments, Valve engineer Pierre-Loup Griffais noted that, so long as interest in handheld gaming
remains high, the company has "plans for successive generations of handhelds." For now, Griffais explained, it's important to Valve that developers don't need
to "split their attention" between different performance targets.

Graph facts:
- Tim Sweeney on Epic’s victory royale over Google -MENTIONS-> Valve
- The best October Prime Day gaming deals -MENTIONS-> Valve
- The Epic v. Google trial may come down to simple v. complicated -MENTIONS-> Valve
- The Steam Deck OLED improves much more than its screen -MENTIONS-> Valve
- Steam Deck OLED review: It's just better -MENTIONS-> Valve

Entities: Valve, Steam Deck OLED, The Steam Deck OLED arrives November 16 with an improved screen and longer battery life, Pierre-Loup Griffais, handheld
gaming, handhelds, 1TB model, Limited Edition version, 512GB LCD model, Engadget
</item 10>
=== END ITEM id=f07c522c027365b4 ===

=== ITEM id=5ec52fe90598d936 ===
Gold sentences and facts:
Sentence 0: Argentina, Japan, Chile, Samoa and Fiji have all provided highlights — Los Pumas are also in the semi-finals — and England have beaten all of them.
  Fact 0.0: Argentina, Japan, Chile, Samoa and Fiji have all provided highlights.
  Fact 0.1: Los Pumas, Argentina's rugby team, are also in the semi-finals.
  Fact 0.2: England have beaten Argentina.
  Fact 0.3: England have beaten Japan.
  Fact 0.4: England have beaten Chile.
  Fact 0.5: England have beaten Samoa.
  Fact 0.6: England have beaten Fiji.
Sentence 1: S﻿AINT-ETIENNE: The Wallabies will have to endure a more tortuous Sunday night watch than even the worst rom com as their Rugby World Cup fate is decided by Fiji and Portugal in Toulouse.
  Fact 1.0: The report is datelined Saint-Etienne.
  Fact 1.1: The Wallabies will have to endure a more tortuous Sunday night watch than even the worst rom com.
  Fact 1.2: The Wallabies' Rugby World Cup fate is decided by Fiji and Portugal in Toulouse.

Retrieved items:
<item 1>
Where to watch England vs South Africa: Live stream, TV channel, lineups, odds for 2023 Rugby World Cup semifinal (Sporting News, 2023-10-20T22:00:05+00:00)
In short, this semi-final requires a huge step up in quality for Borthwick's side, and that's why we think that the Springboks will be victorious at the Stade
de France. South Africa have already shown they can match, and beat, the world's best, and we think that they'll do the latter in Saint-Denis.

Graph facts:
- ‘It happens’: All Blacks defend star centre after spray as Cheika heaps pressure on ‘favourites’ New Zealand -MENTIONS-> South Africa
- West Indies v England: fifth and deciding men’s T20 cricket international – live -MENTIONS-> South Africa
- UK View: ‘A bridge too far’ – Boycott, former greats tear into ‘dysfunctional’ Poms with World Cup defence all but dead -MENTIONS-> South Africa
- ICC World Cup 2023: India at Cricket World Cup semi-finals so far -MENTIONS-> South Africa
- THE GREATEST! Boks beat All Blacks in instant classic to claim record FOURTH RWC in controversial, dramatic final -MENTIONS-> South Africa

Entities: South Africa, Stade de France, Saint-Denis, 2023 Rugby World Cup semifinal, Where to watch England vs South Africa: Live stream, TV channel, lineups,
odds for 2023 Rugby World Cup semifinal, England, Borthwick, the world's best, Sporting News
</item 1>

<item 2>
Graph query row:
article: 200+ of the best deals from Amazon's Cyber Monday sale
publisher: None
page: 200+ of the best deals from Amazon's Cyber Monday sale (Mashable, 2023-11-27T08:45:59+00:00)
passage: 200+ of the best deals from Amazon's Cyber Monday sale (Mashable, 2023-11-27T08:45:59+00:00)
Table of Contents Table of Contents Echo, Fire TV, and Kindle deals Apple deals TV deals Laptop deals Headphone and earbud deals Tablet deals Gaming deals
Speaker deals Vacuum deals Kitchen deals Smart home deals Fitness deals Beauty tech deals Drone deals Camera deals Lego deals Gift card deals

UPDATE: Nov. 27, 2023, 5:00 a.m. EST This post has been updated with all of the latest Cyber Monday deals available at Amazon.

Amazon is dragging out the year's biggest shopping holiday(s) into 11 days of deals.

The retail giant began its Black Friday sale in the early morning of Friday, Nov. 17 (a week ahead of schedule) and was on top of making the switch to Cyber
Monday language in the wee hours of Saturday, Nov. 25. Official Cyber Monday mode, which is currently on through Monday, Nov. 27, includes both a ton of deals
carried over from Black Friday plus some new ones.
</item 2>

<item 3>
Where to watch England vs South Africa: Live stream, TV channel, lineups, odds for 2023 Rugby World Cup semifinal (Sporting News, 2023-10-20T22:00:05+00:00)
TV channel Streaming USA — Peacock Canada TSN TSN+ UK ITV ITV X Australia Channel 9 Stan Sport, 9Now New Zealand Sky GO Sky GO India — Fan Code Hong Kong BeIN
Sports BeIN Sports Malaysia BeIN Sports BeIN Sports Singapore BeIN Sports BeIN Sports

MORE: Every single team to have lifted the Rugby World Cup trophy

England vs South Africa lineups, team news

"The start is crucial, the end is vital..."



- Steve Borthwick explains changing 3️⃣ of his England starters for Saturday's Rugby World Cup semi-final, from Liam Heagney 👨‍💻 in Paris. #RWC2023 #ENGvRSA
#EnglandRugby

Graph facts:
- Is Sam Bankman-Fried a bad ‘man’ or a good ‘boy’? Lawyers swap opening statements before first witnesses take the stand -MENTIONS-> Paris
- Carabao Cup fixtures, draws, results, TV channel, and live streams for every round of EFL League Cup 2023/24 -MENTIONS-> beIN Sports
- Carabao Cup fixtures, draws, results, TV channel, and live streams for every round of EFL League Cup 2023/24 -MENTIONS-> Sky Go
- ‘It happens’: All Blacks defend star centre after spray as Cheika heaps pressure on ‘favourites’ New Zealand -MENTIONS-> Paris
- There’s something going on with AI startups in France -MENTIONS-> Paris

Entities: Sky Go, beIN Sports, Paris, India, Malaysia, South Africa, England vs South Africa 2023 Rugby World Cup semifinal, Steve Borthwick, Liam Heagney, TSN,
TSN+, ITV X, Channel 9, 9Now, Fan Code
</item 3>

<item 4>
Where to watch England vs South Africa: Live stream, TV channel, lineups, odds for 2023 Rugby World Cup semifinal (Sporting News, 2023-10-20T22:00:05+00:00)
The Springboks have done it tough, too, having already faced two of the tournament favourites on their way to the semi-finals. Comfortable defeats of Scotland,
Romania and Tonga couldn't mask the disappointment of the 13-8 loss to Ireland, which consigned South Africa to second in Pool B. However, with Ireland sent
home by the All Blacks in the quarters, the Bokke showed their World Cup mettle to edge the hosts 29-28 at the Stade de France, sneaking past Les Bleus in one
of the best games of rugby you're ever likely to see.

Graph facts:
- ‘It happens’: All Blacks defend star centre after spray as Cheika heaps pressure on ‘favourites’ New Zealand -MENTIONS-> South Africa
- West Indies v England: fifth and deciding men’s T20 cricket international – live -MENTIONS-> South Africa
- UK View: ‘A bridge too far’ – Boycott, former greats tear into ‘dysfunctional’ Poms with World Cup defence all but dead -MENTIONS-> South Africa
- ICC World Cup 2023: India at Cricket World Cup semi-finals so far -MENTIONS-> South Africa
- THE GREATEST! Boks beat All Blacks in instant classic to claim record FOURTH RWC in controversial, dramatic final -MENTIONS-> South Africa

Entities: South Africa, Romania, Tonga, Les Bleus, 2023 Rugby World Cup, Stade de France, 2023 Rugby World Cup semifinal, Where to watch England vs South
Africa: Live stream, TV channel, lineups, odds for 2023 Rugby World Cup semifinal, England, Sporting News, Scotland, All Blacks, Ireland
</item 4>

<item 5>
Where to watch England vs South Africa: Live stream, TV channel, lineups, odds for 2023 Rugby World Cup semifinal (Sporting News, 2023-10-20T22:00:05+00:00)
The #Springboks are gearing up for another tough challenge this week with a semi-final clash against England 🇿🇦🏴󠁧󠁢󠁥󠁮󠁧󠁿#StrongerTogether #RWC2023
pic.twitter.com/HLSnnjWggB — Springboks (@Springboks) October 18, 2023

England are the only side still competing who are yet to lose a game, and that will stand them in good stead coming into this semi-final. Dominant wins over
Argentina, Japan and Chile sealed passage to the last eight, and after being given an almighty scare by Samoa in their final pool match, England overcame a
Fijian fightback in Marseille to triumph 30-24. Despite being outscored by three tries to two, captain Owen Farrell kicked 20 points, sending his side into the
last four at the expense of 2023's breakout stars.

Graph facts:
- Liverpool vs Everton begins epic Saturday with World Cup semi-final as well as Chelsea vs Arsenal – and it could end with a bang -MENTIONS-> Owen Farrell
- West Indies v England: fifth and deciding men’s T20 cricket international – live -MENTIONS-> South Africa
- THE GREATEST! Boks beat All Blacks in instant classic to claim record FOURTH RWC in controversial, dramatic final -MENTIONS-> South Africa
- ‘It happens’: All Blacks defend star centre after spray as Cheika heaps pressure on ‘favourites’ New Zealand -MENTIONS-> South Africa
- UK View: ‘A bridge too far’ – Boycott, former greats tear into ‘dysfunctional’ Poms with World Cup defence all but dead -MENTIONS-> South Africa

Entities: Owen Farrell, South Africa, 2023 Rugby World Cup semifinal, Where to watch England vs South Africa: Live stream, TV channel, lineups, odds for 2023
Rugby World Cup semifinal, England, Japan, Chile, Samoa, Marseille, Sporting News, Fiji, Argentina
</item 5>

<item 6>
Where to watch England vs South Africa: Live stream, TV channel, lineups, odds for 2023 Rugby World Cup semifinal (Sporting News, 2023-10-20T22:00:05+00:00)
The 2019 Rugby World Cup final between England and South Africa was one of many firsts: the game itself capped off a sensational tournament played outside of
the sport's traditional strongholds for the first time; it was the first final in which South Africa scored a try, and by full-time, the 32-12 score in the
Springboks' favour sealed a record-equalling third title for the Rainbow Nation.

Fast forward four years, and South Africa once again prepare to take on England in an all-or-nothing match to decide which nation gets a shot at World Cup glory
— but this time around, the pair meet in the semi-finals of the 2023 tournament, rather than the decider.

That won't make an ounce of difference for either camp, though, as these two long-time rivals prepare for a battle that could re-define the definition of a
'physical encounter.'

Graph facts:
- West Indies v England: fifth and deciding men’s T20 cricket international – live -MENTIONS-> South Africa
- UK View: ‘A bridge too far’ – Boycott, former greats tear into ‘dysfunctional’ Poms with World Cup defence all but dead -MENTIONS-> South Africa
- ‘It happens’: All Blacks defend star centre after spray as Cheika heaps pressure on ‘favourites’ New Zealand -MENTIONS-> South Africa
- THE GREATEST! Boks beat All Blacks in instant classic to claim record FOURTH RWC in controversial, dramatic final -MENTIONS-> South Africa
- ICC World Cup 2023: India at Cricket World Cup semi-finals so far -MENTIONS-> South Africa

Entities: 2019 Rugby World Cup final, South Africa, 2023 Rugby World Cup, 2023 Rugby World Cup semifinal, Where to watch England vs South Africa: Live stream,
TV channel, lineups, odds for 2023 Rugby World Cup semifinal, England, Sporting News
</item 6>

<item 7>
Where to watch England vs South Africa: Live stream, TV channel, lineups, odds for 2023 Rugby World Cup semifinal (Sporting News, 2023-10-20T22:00:05+00:00)
England vs. South Africa betting odds, prediction

The last time these two sides met was a tumultuous period to say the least, particularly for English rugby. South Africa's dominant 27-13 victory at Twickenham
proved to be the last game in charge for Eddie Jones, the coach who had taken England to the 2019 World Cup final. In his place, Steve Borthwick was hired, and
he's just one game away from repeating that feat.

England's form at this tournament has been perfect — at face value. Argentina, Japan, Chile, Samoa and Fiji have all provided highlights — Los Pumas are also in
the semi-finals — and England have beaten all of them. However, only Japan and Chile were defeated with bonus points, and the narrow wins over both Samoa and
Fiji in their last two games showed that England are certainly gettable.

Graph facts:
- ‘It happens’: All Blacks defend star centre after spray as Cheika heaps pressure on ‘favourites’ New Zealand -MENTIONS-> South Africa
- West Indies v England: fifth and deciding men’s T20 cricket international – live -MENTIONS-> South Africa
- UK View: ‘A bridge too far’ – Boycott, former greats tear into ‘dysfunctional’ Poms with World Cup defence all but dead -MENTIONS-> South Africa
- ICC World Cup 2023: India at Cricket World Cup semi-finals so far -MENTIONS-> South Africa
- THE GREATEST! Boks beat All Blacks in instant classic to claim record FOURTH RWC in controversial, dramatic final -MENTIONS-> South Africa

Entities: South Africa, Steve Borthwick, 2023 Rugby World Cup semifinal, Where to watch England vs South Africa: Live stream, TV channel, lineups, odds for 2023
Rugby World Cup semifinal, England, Japan, Chile, Samoa, South Africa's 27-13 victory at Twickenham, Twickenham, 2019 World Cup final, Sporting News, Fiji,
Eddie Jones, Argentina
</item 7>

<item 8>
The story behind anguished Wallabies star's photo (Wide World Of Sports, 2023-10-07T22:10:40+00:00)
Oct 8th, 2023

Oct 8th, 2023

S﻿AINT-ETIENNE: The Wallabies will have to endure a more tortuous Sunday night watch than even the worst rom com as their Rugby World Cup fate is decided by
Fiji and Portugal in Toulouse.

Australia will gather together in their plush Saint-Galmier hotel at 9pm local time (6am Monday AEDT) to view what will almost certainly be their World Cup
execution.

The Wallabies are currently sitting second in Pool C but a single bonus point would be enough for Fiji to leapfrog them and advance to the quarter-finals in
Marseille.

Watch all the action from Rugby World Cup 2023 on the home of rugby, Stan Sport.

Graph facts:
- Eddie Jones says he needs to ‘give myself an uppercut’ but ‘100 percent’ committed to coaching Wallabies -MENTIONS-> Wallabies
- ‘Right people in right roles’: QRU breaks silence on RA’s reform hopes, backs review into Wallabies’ RWC flop -MENTIONS-> Wallabies
- Where to watch England vs South Africa: Live stream, TV channel, lineups, odds for 2023 Rugby World Cup semifinal -MENTIONS-> Marseille
- ‘It happens’: All Blacks defend star centre after spray as Cheika heaps pressure on ‘favourites’ New Zealand -MENTIONS-> Wallabies
- RA set to announce local as high-performance director – and how it could help land Schmidt as Wallabies coach -MENTIONS-> Wallabies

Entities: Wide World Of Sports, Rugby World Cup 2023, Toulouse, Marseille, Saint-Etienne, Saint-Galmier, Wallabies, The story behind anguished Wallabies star's
photo, Portugal, Fiji, Stan Sport
</item 8>

<item 9>
Where to watch England vs South Africa: Live stream, TV channel, lineups, odds for 2023 Rugby World Cup semifinal (Sporting News, 2023-10-20T22:00:05+00:00)
England replacements: Theo Dan, Ellis Genge, Kyle Sinckler, Ollie Chessum, Billy Vunipola, Danny Care, George Ford, Ollie Lawrence.

The Springboks are unchanged from that epic quarter-final with France, having all played their part as Jacques Nienaber's side hung on in the second half. The
contribution of backs Kurt-Lee Arendse, Damian de Allende and Cheslin Kolbe saw all three cross for first-half tries, with Kolbe's coming off the back of a
charged-down conversion from France's Thomas Ramos. Preventing those two points proved crucial in the end.

The key contribution, though, came from Eben Etzebeth, who bulldozed his way over to score the only five-pointer of the second half and wrestle the game back
into South African hands.

Graph facts:
- THE GREATEST! Boks beat All Blacks in instant classic to claim record FOURTH RWC in controversial, dramatic final -MENTIONS-> Jacques Nienaber
- Cheslin Kolbe -PARTICIPATED_IN-> RWC final
- THE GREATEST! Boks beat All Blacks in instant classic to claim record FOURTH RWC in controversial, dramatic final -MENTIONS-> Cheslin Kolbe
- Cheslin Kolbe -PARTICIPATED_IN-> Rugby World Cup
- Legacy vs ego: The Springboks are playing for South Africa, the All Blacks are playing for themselves -MENTIONS-> Jacques Nienaber

Entities: Cheslin Kolbe, Jacques Nienaber, Damian de Allende, South Africa, 2023 Rugby World Cup semifinal, quarter-final with France, Theo Dan, Ellis Genge,
Kyle Sinckler, Ollie Chessum, Billy Vunipola, Danny Care, George Ford, Ollie Lawrence, Kurt-Lee Arendse
</item 9>

<item 10>
Where to watch England vs South Africa: Live stream, TV channel, lineups, odds for 2023 Rugby World Cup semifinal (Sporting News, 2023-10-20T22:00:05+00:00)
Click below for more ⤵️ https://t.co/LeUuw4kPfP — RugbyPass (@RugbyPass) October 19, 2023

Marcus Smith hardly put a foot wrong at fullback against Fiji, but the world champions are an entirely different proposition for England to deal with at the
back. The high-ball security and hulking frame of Freddie Steward returns to the No. 15 jersey in one of three changes, while at inside centre, Manu Tuilagi
starts hoping to repeat his heroics of four years ago, when his early try in a dominant semi-final performance took England past the All Blacks.

England's other two changes come in the tight five, with Steve Borthwick making some big calls ahead of this game. Joe Marler starts ahead of vice-captain Ellis
Genge at loosehead prop, and Ollie Chessum is replaced in the second-row by George Martin.

Graph facts:
- ‘It happens’: All Blacks defend star centre after spray as Cheika heaps pressure on ‘favourites’ New Zealand -MENTIONS-> South Africa
- West Indies v England: fifth and deciding men’s T20 cricket international – live -MENTIONS-> South Africa
- UK View: ‘A bridge too far’ – Boycott, former greats tear into ‘dysfunctional’ Poms with World Cup defence all but dead -MENTIONS-> South Africa
- ICC World Cup 2023: India at Cricket World Cup semi-finals so far -MENTIONS-> South Africa
- THE GREATEST! Boks beat All Blacks in instant classic to claim record FOURTH RWC in controversial, dramatic final -MENTIONS-> South Africa

Entities: South Africa, Steve Borthwick, RugbyPass, Marcus Smith, 2023 Rugby World Cup semifinal, Freddie Steward, Manu Tuilagi, Joe Marler, George Martin,
Ellis Genge, Ollie Chessum, Where to watch England vs South Africa: Live stream, TV channel, lineups, odds for 2023 Rugby World Cup semifinal, England, Sporting
News, Fiji
</item 10>
=== END ITEM id=5ec52fe90598d936 ===
