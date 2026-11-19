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

=== ITEM id=01a8d03f4bf56e6f ===
Gold sentences and facts:
Sentence 0: Wang testified that he heard Bankman-Fried tell another trader that it was okay to continue withdrawing money from the account as long as the amount was less than FTX’s total trading revenue.
  Fact 0.0: Wang testified that he heard Bankman-Fried tell another trader that it was okay to continue withdrawing money from the account.
  Fact 0.1: Bankman-Fried said withdrawing money from the account was okay as long as the amount was less than FTX's total trading revenue.
Sentence 1: The highly anticipated criminal trial for Sam Bankman-Fried, former CEO of bankrupt crypto exchange FTX, started Tuesday to determine whether he’s guilty of seven counts of fraud and conspiracy.
  Fact 1.0: The highly anticipated criminal trial for Sam Bankman-Fried started on a Tuesday.
  Fact 1.1: Sam Bankman-Fried is the former CEO of bankrupt crypto exchange FTX.
  Fact 1.2: The trial will determine whether Bankman-Fried is guilty of seven counts of fraud and conspiracy.
Sentence 2: The prosecution painted Bankman-Fried as someone who knowingly committed fraud to achieve great wealth, power and influence, while the defense countered that the FTX founder acted in good faith, never meant to commit fraud or steal and basically got in over his head.
  Fact 2.0: The prosecution portrayed Bankman-Fried as someone who knowingly committed fraud to achieve great wealth, power and influence.
  Fact 2.1: The defense argued that the FTX founder acted in good faith.
  Fact 2.2: The defense argued that the FTX founder never meant to commit fraud or steal.
  Fact 2.3: The defense argued that the FTX founder basically got in over his head.

Retrieved items:
<item 1>
The FTX trial is bigger than Sam Bankman-Fried (The Verge, 2023-09-28T12:00:00+00:00)
Bankman-Fried is accused of using “billions of dollars in misappropriated FTX customer deposits” to help buy more than $200 million of real estate for himself,
make billions of dollars of investments for his own interest, and repay Alameda’s lenders, according to the indictment, Additionally, Bankman-Fried allegedly
used more than $100 million of customer funds to make political contributions; prosecutors can show evidence of those contributions in this trial, even though
they aren’t part of the charges brought. A second trial is scheduled for March 2024, with additional charges.

Before his fall, Bankman-Fried made himself out to be the Good Boy of crypto — the trustworthy face of a sometimes-shady industry. He was also very interested
in publicity, sitting for many interviews both before and after the fall of FTX. The quick rise of FTX as an industry force was at least in part due to
Bankman-Fried’s appetite for attention.

Graph facts:
- SBF’s trial starts soon, but how did he — and FTX — get here? -MENTIONS-> crypto
- SBF, riding high on FTX, reportedly offered $5B to Trump to cancel President re-run -MENTIONS-> crypto
- Is Sam Bankman-Fried a bad ‘man’ or a good ‘boy’? Lawyers swap opening statements before first witnesses take the stand -MENTIONS-> crypto
- Sam Bankman-Fried was a terrible boyfriend -MENTIONS-> crypto
- SBF Trial: The latest updates from the FTX collapse’s courtroom drama -MENTIONS-> crypto

Entities: crypto, Alameda, FTX trial, second trial, misappropriated FTX customer deposits, political contributions, The FTX trial is bigger than Sam
Bankman-Fried, real estate, FTX, Sam Bankman-Fried, The Verge
</item 1>

<item 2>
Graph query row:
article: SBF Trial: The latest updates from the FTX collapse’s courtroom drama
published_on: 2023-10-06T17:22:02+00:00
publisher: TechCrunch
passage: SBF Trial: The latest updates from the FTX collapse’s courtroom drama (TechCrunch, 2023-10-06T17:22:02+00:00)
SBF trial: Everything to know from the FTX courtroom ahead of his testimony Sam Bankman-Fried is in court for alleged fraud and money laundering while leading
crypto exchange FTX and sister company Alameda. Here’s our latest coverage as the case nears its end.

The trial of Bankman-Fried, the former CEO and co-founder of collapsed crypto exchange FTX, is entering its next chapter as proceedings resume Thursday. As
confirmed during a teleconference Wednesday, SBF will be taking the stand, along with a handful of other witnesses as the onus shifts toward the defense after
the prosecution brought its case to its conclusion.

As former federal prosecutor Josh Naftalis told TechCrunch, SBF taking to the stand in his own trial is a “Hail Mary” ahead of a verdict and potential
sentencing for seven counts tied to fraud and money laundering.

“Once it goes to cross[-examination], he doesn’t get to say, ‘I’m done,’” Naftalis said.
article_connections: [{'entity': 'FTX Ventures', 'relationship': 'MENTIONS'}, {'entity': 'FTX collapse', 'relationship': 'MENTIONS'}, {'entity': 'FTX',
'relationship': 'MENTIONS'}, {'entity': 'Sam Bankman-Fried', 'relationship': 'MENTIONS'}]
publisher_connections: [{'entity': 'SBF Trial: The latest updates from the FTX collapse’s courtroom drama', 'relationship': 'PUBLISHED_BY'}, {'entity': 'SBF’s
trial starts soon, but how did he — and FTX — get here?', 'relationship': 'PUBLISHED_BY'}, {'entity': 'SBF, riding high on FTX, reportedly offered $5B to Trump
to cancel President re-run', 'relationship': 'PUBLISHED_BY'}, {'entity': 'SBF Trial: The latest updates from the FTX collapse’s courtroom drama',
'relationship': 'MENTIONS'}]
</item 2>

<item 3>
How is it still getting worse for Sam Bankman-Fried? (The Verge, 2023-10-12T23:08:14+00:00)
I have been waiting for a juicy cross-examination, as I live for chaos and drama. I am beginning to think I am not going to get one.

Ellison had given, in her direct testimony, fairly damning evidence tying FTX CEO Sam Bankman-Fried to the conspiracy to take FTX customer funds. There were
fake balance sheets, one of which was sent to crypto lender Genesis. After a Genesis representative received the balance sheet, he texted Ellison to tell her
he’d spoken to Bankman-Fried — strongly suggesting that Bankman-Fried was aware of the contents of the fake balance sheet. Not great!

Sam Bankman-Fried, the founder of failed cryptocurrency exchange FTX, is on trial for seven counts of wire fraud and conspiracy. FTX was a fraud “from the
start,” the Securities and Exchange Commission alleges — with a “multi-billion-dollar deficiency caused by his own misappropriation of customer funds.” Follow
along for all the latest news and regular updates from the trial.

Graph facts:
- The jury finally hears from Sam Bankman-Fried -MENTIONS-> Securities and Exchange Commission
- Sam Bankman-Fried was a terrible boyfriend -MENTIONS-> Securities and Exchange Commission
- Sam Bankman-Fried didn’t ask where the $8 billion went -MENTIONS-> Securities and Exchange Commission
- Coinbase rallies more than 60% in same month that FTX and Binance founders brace for prison -MENTIONS-> Securities and Exchange Commission
- The jury finally hears from Sam Bankman-Fried -MENTIONS-> misappropriation of customer funds

Entities: Securities and Exchange Commission, misappropriation of customer funds, Sam Bankman-Fried trial, wire fraud, Ellison, FTX, Sam Bankman-Fried,
conspiracy to take FTX customer funds, fake balance sheets, How is it still getting worse for Sam Bankman-Fried?, Genesis, The Verge
</item 3>

<item 4>
The FTX trial is bigger than Sam Bankman-Fried (The Verge, 2023-09-28T12:00:00+00:00)
If it seems like his peer group supported him, or worse, fawned over him, that’s a real problem for the industry’s reputation.”

After FTX declared bankruptcy, Bankman-Fried went on an extensive media tour, doing interviews with The New York Times’ Andrew Ross Sorkin and Good Morning
America. Even under house arrest, he did interviews, including with The New Yorker. Bankman-Fried seemed to think that if he just explained things, everyone
would understand that what he did was a mistake, not a crime.

Graph facts:
- He’s Hockey’s Brightest Young Star. This Is What Makes His Shot So Special. -PUBLISHED_BY-> The New York Times
- How U.S. Women’s Soccer Team Lured Emma Hayes Away From Chelsea -PUBLISHED_BY-> The New York Times
- After a Stunning Snub, College Football Is Facing an Identity Crisis -PUBLISHED_BY-> The New York Times
- N.F.L. Executives on the Biggest Questions Facing Super Bowl Contenders -PUBLISHED_BY-> The New York Times
- Which M.L.B. Playoff Team Has the Most Daunting Home Field? We Ranked Them. -PUBLISHED_BY-> The New York Times

Entities: The New York Times, FTX trial, The FTX trial is bigger than Sam Bankman-Fried, extensive media tour, interviews, Andrew Ross Sorkin, Good Morning
America, The New Yorker, FTX, Sam Bankman-Fried, The Verge
</item 4>

<item 5>
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
</item 5>

<item 6>
The FTX trial is bigger than Sam Bankman-Fried (The Verge, 2023-09-28T12:00:00+00:00)
The trial of Sam Bankman-Fried is likely to be more consequential than just whether the man himself is found guilty. Depending on what evidence is introduced
during the trial, it could be rough for the entire crypto industry.

“How much damage can this trial do to the already beaten-down reputation of the industry at this point?” asks Yesha Yadav, a law professor at Vanderbilt
University. “This trial is going to be an excruciating moment for the industry because no one knows what kind of evidence might come out.”

Bankman-Fried, the founder of FTX and Alameda Research, is facing seven counts of criminal charges: two counts of wire fraud, and five counts of conspiracy
charges. FTX was a failed cryptocurrency exchange founded in 2019. According to a now-deleted profile from FTX investors Sequoia Capital, FTX was founded
because of Bankman-Fried’s frustration with other exchanges when he was running Alameda Research, his crypto trading firm.

Graph facts:
- ChatGPT: Everything you need to know about the AI-powered chatbot -MENTIONS-> Sequoia Capital
- Is Sam Bankman-Fried a bad ‘man’ or a good ‘boy’? Lawyers swap opening statements before first witnesses take the stand -MENTIONS-> Sequoia Capital
- The fate of US venture capital in China teeters on uncertainty -MENTIONS-> Sequoia Capital
- Matt Huang -WORKS_FOR-> Sequoia Capital
- Sequoia Capital -PARTICIPATED_IN-> OpenAI $300M share sale

Entities: Sequoia Capital, Alameda Research, wire fraud, conspiracy charges, FTX trial, Yesha Yadav, Vanderbilt University, The FTX trial is bigger than Sam
Bankman-Fried, FTX, Sam Bankman-Fried, crypto industry, The Verge
</item 6>

<item 7>
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

Graph facts:
- The FTX trial is bigger than Sam Bankman-Fried -MENTIONS-> Caroline Ellison
- The FTX trial is bigger than Sam Bankman-Fried -MENTIONS-> crypto
- The FTX trial is bigger than Sam Bankman-Fried -MENTIONS-> Gary Wang
- The FTX trial is bigger than Sam Bankman-Fried -MENTIONS-> The New York Times
- The FTX trial is bigger than Sam Bankman-Fried -MENTIONS-> Sequoia Capital

Entities: Gemini, fraud trial, The FTX trial is bigger than Sam Bankman-Fried, FTX, Sam Bankman-Fried, Kraken, Coinbase, crypto industry, The Verge
</item 7>

<item 8>
The FTX trial is bigger than Sam Bankman-Fried (The Verge, 2023-09-28T12:00:00+00:00)
In the indictment, the prosecutors allege that Bankman-Fried lied to FTX investors. Some of those investors may be called on to testify, which is not the kind
of thing VC big shots generally enjoy — and which might create collateral damage for the industry.

If, for instance, Sequoia Capital did due diligence around its investment in FTX, whatever Bankman-Fried told its partners could be important. Was FTX already
sending Alameda customer funds at that point? Did Bankman-Fried know about it? And did he tell VCs at the time? (If he told them it wasn’t happening, that would
establish intent.) Binance was also an early investor in FTX, and former executives may be called on to testify as well, says Hermine Wong, the former head of
policy at Coinbase and a former SEC regulator.

Bankman-Fried’s defense can also introduce risks for people who dealt with him. Defense lawyers have several simultaneous objectives. First and foremost,
they’re trying to prove their client is not guilty.

Graph facts:
- ChatGPT: Everything you need to know about the AI-powered chatbot -MENTIONS-> Sequoia Capital
- Is Sam Bankman-Fried a bad ‘man’ or a good ‘boy’? Lawyers swap opening statements before first witnesses take the stand -MENTIONS-> Sequoia Capital
- The fate of US venture capital in China teeters on uncertainty -MENTIONS-> Sequoia Capital
- Matt Huang -WORKS_FOR-> Sequoia Capital
- Sequoia Capital -PARTICIPATED_IN-> OpenAI $300M share sale

Entities: Sequoia Capital, Alameda, FTX trial, The FTX trial is bigger than Sam Bankman-Fried, Hermine Wong, lying to FTX investors, FTX, Sam Bankman-Fried,
Binance, SEC, Coinbase, The Verge
</item 8>

<item 9>
The FTX trial is bigger than Sam Bankman-Fried (The Verge, 2023-09-28T12:00:00+00:00)
According to the SEC, FTX was a fraud “from the start,” diverting customers’ funds to Alameda.

Sam Bankman-Fried, the founder of failed cryptocurrency exchange FTX, is on trial for seven counts of wire fraud and conspiracy. FTX was a fraud “from the
start,” the Securities and Exchange Commission alleges — with a “multi-billion-dollar deficiency caused by his own misappropriation of customer funds.” Follow
along for all the latest news and regular updates from the trial.

“Really just old-fashioned embezzlement.”

What’s left of FTX is now being led by John J. Ray III — you may remember him as the guy who cleaned up Enron and who also said FTX is worse than Enron. Just
before the trial began, the FTX lawyers filed suit against Bankman-Fried’s parents, saying they should give back millions of dollars they got from their kid.
Ray has also referred to Bankman-Fried’s conduct as “really just old-fashioned embezzlement.”
</item 9>

<item 10>
The FTX trial is bigger than Sam Bankman-Fried (The Verge, 2023-09-28T12:00:00+00:00)
We have some inkling of what the government will be arguing thanks to the filings prosecutors are making in this case. They argue that Bankman-Fried lied about
consumer protection and that Bankman-Fried’s statements that FTX was “avoiding or managing conflicts of interest” and that “as a general principle FTX
segregates customer assets from its own assets across our platforms” were lies, in a superseding indictment filed August 14th.

The indictment also says that there were special features in FTX’s code that “permitted Alameda to spend and withdraw unlimited amounts of money from FTX,”
which were created at Bankman-Fried’s direction. That effectively exempted Alameda from the kinds of risk management other customers faced. At the same time,
the government alleges, “Bankman-Fried publicly and repeatedly asserted that Alameda did not have privileged access to FTX.”

Graph facts:
- SBF’s trial starts soon, but how did he — and FTX — get here? -MENTIONS-> Alameda
- SBF, riding high on FTX, reportedly offered $5B to Trump to cancel President re-run -MENTIONS-> Alameda
- Is Sam Bankman-Fried a bad ‘man’ or a good ‘boy’? Lawyers swap opening statements before first witnesses take the stand -MENTIONS-> Alameda
- The jury finally hears from Sam Bankman-Fried -MENTIONS-> Alameda
- SBF Trial: The latest updates from the FTX collapse’s courtroom drama -MENTIONS-> Alameda

Entities: Alameda, risk management, consumer protection, conflicts of interest, superseding indictment, The FTX trial is bigger than Sam Bankman-Fried, FTX, Sam
Bankman-Fried, government, The Verge
</item 10>
=== END ITEM id=01a8d03f4bf56e6f ===

=== ITEM id=428f9c4cc54deb8d ===
Gold sentences and facts:
Sentence 0: Unfortunately, she did say that she had conversations with investors as part of their due diligence — and, of course, Alameda was taking on losses from FTX to keep FTX’s balance sheet pristine.
  Fact 0.0: It was unfortunate that the woman said she had conversations with investors as part of their due diligence.
  Fact 0.1: Alameda was taking on losses from FTX.
  Fact 0.2: Alameda was taking on losses from FTX to keep FTX's balance sheet pristine.
Sentence 1: During cross-examination, Huang said Paradigm pressed Bankman-Fried on the board issue and was told he didn't want investors as directors but he did plan on having a board with experts.
  Fact 1.0: During cross-examination, Huang said Paradigm pressed Bankman-Fried on the board issue.
  Fact 1.1: Huang said Paradigm was told that Bankman-Fried did not want investors as directors.
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
=== END ITEM id=428f9c4cc54deb8d ===

=== ITEM id=1dcd0287790d8838 ===
Gold sentences and facts:
Sentence 0: And so, alongside the big improvements, Valve has made a gaggle of minor physical upgrades with its new iteration.
  Fact 0.0: Valve has made a gaggle of minor physical upgrades with its new iteration.
  Fact 0.1: Valve has made these minor physical upgrades alongside the big improvements.
  Fact 0.2: Valve's new iteration includes big improvements.
Sentence 1: We look at Valve and we see a store that could be both, but they’ve decided to focus exclusively on games.
  Fact 1.0: The writer looks at Valve and sees a store that could be both.
  Fact 1.1: Valve has decided to focus exclusively on games.
Sentence 2: The Steam Deck OLED will go on sale on November 16th at 1pm ET, and Valve has units ready to ship that same day.
  Fact 2.0: The Steam Deck OLED will go on sale on November 16th at 1pm ET.
  Fact 2.1: Valve has units of the Steam Deck OLED ready to ship on that same day.

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
=== END ITEM id=1dcd0287790d8838 ===
