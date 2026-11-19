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

=== ITEM id=301bc60cb518fe2c ===
Gold sentences and facts:
Sentence 0: During cross-examination, Huang said Paradigm pressed Bankman-Fried on the board issue and was told he didn't want investors as directors but he did plan on having a board with experts.
  Fact 0.0: Huang said this during cross-examination.
  Fact 0.1: Huang said Paradigm pressed Bankman-Fried on the board issue.
  Fact 0.2: Huang said Paradigm was told Bankman-Fried did not want investors as directors.
  Fact 0.3: Huang said Bankman-Fried did plan on having a board with experts.
Sentence 1: The second week of the trial’s standout testimony came from Alameda Research’s former CEO Caroline Ellison, who claimed she took $14 billion from customers to repay debts to lenders, using them as a line of credit under the instruction of SBF.
  Fact 1.0: The second week of the trial included standout testimony.
  Fact 1.1: The standout testimony came from Caroline Ellison.
  Fact 1.2: Caroline Ellison is Alameda Research's former CEO.
  Fact 1.3: Caroline Ellison claimed she took $14 billion from customers.
  Fact 1.4: Caroline Ellison claimed she took the $14 billion to repay debts to lenders.
  Fact 1.5: Caroline Ellison claimed she used the money as a line of credit.
  Fact 1.6: Caroline Ellison claimed she did this under the instruction of SBF.
Sentence 2: The prosecution painted Bankman-Fried as someone who knowingly committed fraud to achieve great wealth, power and influence, while the defense countered that the FTX founder acted in good faith, never meant to commit fraud or steal and basically got in over his head.
  Fact 2.0: The prosecution portrayed Bankman-Fried as someone who knowingly committed fraud to achieve great wealth, power and influence.
  Fact 2.1: The defense argued that the FTX founder acted in good faith.
  Fact 2.2: The defense argued that the FTX founder never meant to commit fraud or steal.
  Fact 2.3: The defense argued that the FTX founder basically got in over his head.

Retrieved items:
<item 1>
Sam Bankman-Fried didn’t ask where the $8 billion went (The Verge, 2023-10-31T19:52:07+00:00)
Oh, but he was testifying to Congress about keeping customer funds safe.

Sam Bankman-Fried, the founder of failed cryptocurrency exchange FTX, is on trial for seven counts of wire fraud and conspiracy. FTX was a fraud “from the
start,” the Securities and Exchange Commission alleges — with a “multi-billion-dollar deficiency caused by his own misappropriation of customer funds.” Follow
along for all the latest news and regular updates from the trial.

Who was making decisions to spend $8 billion of customer funds? Bankman-Fried couldn’t recall knowing anything about it. Were there rules or requirements for
how money borrowed from FTX would be returned? Were there rules for risk management? “I was concerned with overall risk management,” Bankman-Fried said.

But it was the testimony about June 2022 that resonated the most to me. Didn’t Bankman-Fried ask what “fiat@ftx” was? He did.

Graph facts:
- The jury finally hears from Sam Bankman-Fried -MENTIONS-> Securities and Exchange Commission
- Sam Bankman-Fried was a terrible boyfriend -MENTIONS-> Securities and Exchange Commission
- How is it still getting worse for Sam Bankman-Fried? -MENTIONS-> Securities and Exchange Commission
- Coinbase rallies more than 60% in same month that FTX and Binance founders brace for prison -MENTIONS-> Securities and Exchange Commission
- The jury finally hears from Sam Bankman-Fried -MENTIONS-> misappropriation of customer funds

Entities: Securities and Exchange Commission, misappropriation of customer funds, Sam Bankman-Fried didn’t ask where the $8 billion went, risk management,
fiat@ftx, Sam Bankman-Fried trial, wire fraud, conspiracy, Congress, FTX, Sam Bankman-Fried, fraud, The Verge
</item 1>

<item 2>
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
</item 2>

<item 3>
Sam Bankman-Fried was a terrible boyfriend (The Verge, 2023-10-10T23:50:21+00:00)
FTX was a fraud “from the start,” the Securities and Exchange Commission alleges — with a “multi-billion-dollar deficiency caused by his own misappropriation of
customer funds.” Follow along for all the latest news and regular updates from the trial.

The day started off promisingly for the defense as it cross-examined Gary Wang, the chief technology officer of FTX and co-owner of both FTX and Alameda.
Christian Everdell, one of Bankman-Fried’s defense attorneys, couldn’t undo the damage of last week’s code review. But he managed to shake the rust off long
enough to make Wang sound less reliable, drowning the jury in confusing technicalities.

Last week, Wang testified that Alameda got access to a special credit line and an option to take its balance into the negative without triggering liquidation —
something he alleged other accounts at FTX didn’t get. Everdell tried to undermine this claim by talking about the spot margin program, which let users lend
each other assets for margin trading.

Graph facts:
- SBF Trial: The latest updates from the FTX collapse’s courtroom drama -MENTIONS-> Gary Wang
- Is Sam Bankman-Fried a bad ‘man’ or a good ‘boy’? Lawyers swap opening statements before first witnesses take the stand -MENTIONS-> Gary Wang
- The jury finally hears from Sam Bankman-Fried -MENTIONS-> Gary Wang
- SBF’s trial starts soon, but how did he — and FTX — get here? -MENTIONS-> Gary Wang
- In the end, the FTX trial was about the friends screwed along the way -MENTIONS-> Gary Wang

Entities: Gary Wang, Securities and Exchange Commission, misappropriation of customer funds, Alameda, Christian Everdell, spot margin program, Sam Bankman-Fried
was a terrible boyfriend, FTX, Sam Bankman-Fried, trial, fraud, The Verge
</item 3>

<item 4>
Is Sam Bankman-Fried a bad ‘man’ or a good ‘boy’? Lawyers swap opening statements before first witnesses take the stand (Fortune, 2023-10-04T23:32:53+00:00)
FTX was no Ponzi scheme. It was a “very innovative, successful company.” And the business practices between the two were reasonable, he argued, claiming that
Alameda acted legally as an FTX customer, payment processor, and market maker, or financial entity that acts as a trading partner for customers looking to buy
and sell cryptocurrencies.

In an analogy he employed throughout his opening statement, he said that “working at a startup is like building a plane as you’re flying it” and that businesses
sometimes fail. In fact, he specifically pointed the finger at Ellison, the former CEO of Alameda, who, he said, did not adequately protect her hedge fund from
the inherent risk of the crypto markets.

When the walls came closing in and the aforementioned plane approached the “eye of the storm,” Bankman-Fried did not act like someone who was guilty. Rather, he
was willing to give up his personal wealth to make customers whole, Cohen argued.

Graph facts:
- Is Sam Bankman-Fried a bad ‘man’ or a good ‘boy’? Lawyers swap opening statements before first witnesses take the stand -MENTIONS-> Damian Williams
- Is Sam Bankman-Fried a bad ‘man’ or a good ‘boy’? Lawyers swap opening statements before first witnesses take the stand -MENTIONS-> Gary Wang
- The best new TV series on every streaming service -MENTIONS-> Ponzi scheme
- Is Sam Bankman-Fried a bad ‘man’ or a good ‘boy’? Lawyers swap opening statements before first witnesses take the stand -MENTIONS-> Matt Huang
- Is Sam Bankman-Fried a bad ‘man’ or a good ‘boy’? Lawyers swap opening statements before first witnesses take the stand -MENTIONS-> Caroline Ellison

Entities: Ponzi scheme, crypto markets, Is Sam Bankman-Fried a bad ‘man’ or a good ‘boy’? Lawyers swap opening statements before first witnesses take the stand,
Alameda, Ellison, Cohen, Fortune, FTX, Sam Bankman-Fried
</item 4>

<item 5>
Sex, Signal messages, and sabotaging FTX's code: SBF execs and Bahamas roommates tell all in court (Cnbc | World Business News Leader,
2023-10-06T20:17:00+00:00)
Witness Adam Yedidia answers questions during Sam Bankman-Fried fraud trial over the collapse of FTX, the bankrupt cryptocurrency exchange, at Federal Court in
New York City, October 5, 2023, in this courtroom sketch. Jane Rosenberg | Reuters

Two of Sam Bankman-Fried's former friends from MIT, who also worked at crypto exchange FTX while living with the company's founder in the Bahamas, took the
stand in a Manhattan courtroom this week to testify against their former classmate, confidant, and boss — a man who allegedly ran a crypto empire that defrauded
thousands of customers out of billions of dollars. Gary Wang, the lesser-known co-founder of FTX, was asked by Assistant U.S. Attorney Nicolas Roos on Thursday,
"Did you commit financial crimes while working at FTX?" "Yes," responded Wang.

Graph facts:
- SBF’s trial starts soon, but how did he — and FTX — get here? -MENTIONS-> Gary Wang
- Is Sam Bankman-Fried a bad ‘man’ or a good ‘boy’? Lawyers swap opening statements before first witnesses take the stand -MENTIONS-> Gary Wang
- The jury finally hears from Sam Bankman-Fried -MENTIONS-> Gary Wang
- Sam Bankman-Fried was a terrible boyfriend -MENTIONS-> Gary Wang
- SBF Trial: The latest updates from the FTX collapse’s courtroom drama -MENTIONS-> Gary Wang

Entities: Gary Wang, Adam Yedidia, New York City, Bahamas, Jane Rosenberg, FTX, Nicolas Roos, Sam Bankman-Fried fraud trial, Federal Court in New York City,
financial crimes, Sex, Signal messages, and sabotaging FTX's code: SBF execs and Bahamas roommates tell all in court, Sam Bankman-Fried, Manhattan, Reuters,
CNBC
</item 5>

<item 6>
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
</item 6>

<item 7>
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

Graph facts:
- SBF Trial: The latest updates from the FTX collapse’s courtroom drama -MENTIONS-> Matt Huang
- Matt Huang -WORKS_FOR-> Sequoia Capital
- Today the FTX jury suffered through a code review -MENTIONS-> Matt Huang
- Is Sam Bankman-Fried a bad ‘man’ or a good ‘boy’? Lawyers swap opening statements before first witnesses take the stand -MENTIONS-> Matt Huang
- SBF Trial: The latest updates from the FTX collapse’s courtroom drama -MENTIONS-> Paradigm

Entities: Matt Huang, Paradigm, Alameda, Getty Images, money laundering, Bahamas, New York, US, FTX, Sex, Signal messages, and sabotaging FTX's code: SBF execs
and Bahamas roommates tell all in court, Yuki Iwamura, Sam Bankman-Fried, CNBC, fraud, Bloomberg
</item 7>

<item 8>
Graph query row:
article_title: SBF Trial: The latest updates from the FTX collapse’s courtroom drama
publisher: TechCrunch
connected_entities: ['Josh Naftalis', 'Naftalis', 'Jacquelyn Melinek', 'Chain Reaction podcast', 'Apple Podcasts', 'bankruptcy', 'Chain Reaction newsletter',
'jury selection', 'Robert Boroujerdi', 'Third Point', 'University of Notre Dame', 'Easton', 'FTX Ventures', 'Chinese officials', 'Paying Chinese officials $150
million to restore access to $1 billion in frozen trading accounts', 'charities', 'SBF Trial', 'FTX collapse', 'SBF', 'Matt Huang', 'Paradigm', 'class-action
lawsuit', 'defrauding customers', 'Chain Reaction', 'SBF trial’s jury', 'Gary Wang', 'Caroline Ellison', 'crypto', 'Thane Rehn', 'Mark Cohen', 'Alameda
Research', 'Alameda', 'bribery', 'TechCrunch', 'Equity', 'TechCrunch+', 'money laundering', 'political causes', 'Peter Easton', 'Can Sun', 'real estate', 'FTX',
'Sam Bankman-Fried', 'investments', 'Spotify', 'fraud', 'Stanford University']
evidence: [{'page': 'SBF Trial: The latest updates from the FTX collapse’s courtroom drama (TechCrunch, 2023-10-06T17:22:02+00:00)', 'passage': 'SBF Trial: The
latest updates from the FTX collapse’s courtroom drama (TechCrunch, 2023-10-06T17:22:02+00:00)\nSBF trial: Everything to know from the FTX courtroom ahead of
his testimony Sam Bankman-Fried is in court for alleged fraud and money laundering while leading crypto exchange FTX and sister company Alameda. Here’s our
latest coverage as the case nears its end.\n\nThe trial of Bankman-Fried, the former CEO and co-founder of collapsed crypto exchange FTX, is entering its next
chapter as proceedings resume Thursday. As confirmed during a teleconference Wednesday, SBF will be taking the stand, along with a handful of other witnesses as
the onus shifts toward the defense after the prosecution brought its case to its conclusion.\n\nAs former federal prosecutor Josh Naftalis told TechCrunch, SBF
taking to the stand in his own trial is a “Hail Mary” ahead of a verdict and potential sentencing for seven counts tied to fraud and money laundering.\n\n“Once
it goes to cross[-examination], he doesn’t get to say, ‘I’m done,’” Naftalis said.'}, {'page': 'SBF Trial: The latest updates from the FTX collapse’s courtroom
drama (TechCrunch, 2023-10-06T17:22:02+00:00)', 'passage': 'SBF Trial: The latest updates from the FTX collapse’s courtroom drama (TechCrunch,
2023-10-06T17:22:02+00:00)\n“He can’t just walk out if he doesn’t like how it’s going.”\n\n\n\nNaftalis spoke at length about what comes next in the trial, and
addressed common misconceptions about the prosecution and defense in the latest episode of TechCrunch’s Chain Reaction podcast, which you can listen to below,
or on Apple Podcasts or Spotify.\n\nThe SBF trial is one of the biggest cases of its kind within the crypto space. Our resident crypto expert Jacquelyn Melinek
has been on the ground at the trial since it started, and continues to cover the ins and outs of the proceedings. But the trial has been on pause from October
20 to today, so there’s plenty to catch up on if you need a refresher ahead of it entering its final few phases.\n\nFor those in need of a total refresher on
the case, her breakdown of how FTX went from the third-largest crypto exchange valued at a peak of $32 billion to bankruptcy will get you up to speed.'},
{'page': 'SBF Trial: The latest updates from the FTX collapse’s courtroom drama (TechCrunch, 2023-10-06T17:22:02+00:00)', 'passage': 'SBF Trial: The latest
updates from the FTX collapse’s courtroom drama (TechCrunch, 2023-10-06T17:22:02+00:00)\nYou can also follow along with her and the TechCrunch team’s coverage
in the Chain Reaction newsletter, which drops Thursdays at 12 p.m. PT, and the Chain Reaction podcast, which will hit your feed every Thursday.\n\nHow to follow
the SBF trial\n\nThe trial kicked off with jury selection October 3. From there, access to the proceedings is limited, as no devices are allowed; reports are
being physically put together within the courtroom (or sent to overflow rooms).\n\nBeyond our daily coverage, additional expertise and commentary will live on
TechCrunch+. There, you can find stories like this in-depth breakdown of what to expect from both sides of the SBF case, where the prosecution and defense could
gain ground or fall short in their arguments and what the takeaways were from the opening arguments.\n\nSBF trial: What we learned in week three\n\nThe third
week of the trial featured a couple of noteworthy moments for the prosecution, including FTX’s former general counsel Can Sun.'}, {'page': 'SBF Trial: The
latest updates from the FTX collapse’s courtroom drama (TechCrunch, 2023-10-06T17:22:02+00:00)', 'passage': 'SBF Trial: The latest updates from the FTX
collapse’s courtroom drama (TechCrunch, 2023-10-06T17:22:02+00:00)\nHe, like many of SBF’s former colleagues, opted to cooperate with authorities and testified
that SBF asked him to create a theoretical “legal justification” for the use of billions in consumer funds just days before FTX filed for bankruptcy.\n\nRobert
Boroujerdi, managing director of the hedge fund Third Point, testified about his experience working with SBF and FTX, in a partnership that ultimately resulted
in a $60 million investment and a lack of awareness about the special relationship between Alameda Research and FTX. Had he known of such a relationship, he
testified that Third Point would not have invested in FTX in the first place.\n\nAnd earlier in the week, a University of Notre Dame professor who helped the
government prosecution of Enron and WorldCom, Peter Easton, testified about what he uncovered after tracing the flow of billions of dollars between Alameda
Research and FTX.'}, {'page': 'SBF Trial: The latest updates from the FTX collapse’s courtroom drama (TechCrunch, 2023-10-06T17:22:02+00:00)', 'passage': 'SBF
Trial: The latest updates from the FTX collapse’s courtroom drama (TechCrunch, 2023-10-06T17:22:02+00:00)\nMuch of said funding came from customers, and Easton
testified that he found that user deposits were used by both FTX and Alameda to purchase real estate, make investments or direct funding toward political causes
and charities.\n\nSBF trial: What we learned in week two\n\nThe second week of the trial’s standout testimony came from Alameda Research’s former CEO Caroline
Ellison, who claimed she took $14 billion from customers to repay debts to lenders, using them as a line of credit under the instruction of SBF. Ellison also
went into detail about how a $2 billion venture fund, FTX Ventures, was funded by money that had previously been allocated to Alameda from third-party
lenders.\n\nAnd though there are no charges of bribery in the case, Ellison also testified to paying Chinese officials $150 million to restore access to $1
billion in frozen trading accounts.\n\nSBF trial: What we learned in the week one\n\nAs expected, the opening statements in the SBF trial were spicy.'},
{'page': 'SBF Trial: The latest updates from the FTX collapse’s courtroom drama (TechCrunch, 2023-10-06T17:22:02+00:00)', 'passage': 'SBF Trial: The latest
updates from the FTX collapse’s courtroom drama (TechCrunch, 2023-10-06T17:22:02+00:00)\nThe prosecution painted a picture of an empire “built on lies” and made
a point to show actual pictures of SBF alongside rich and powerful figures to illustrate the peak of his power and influence prior to FTX’s implosion.\n\n“This
man stole billions of dollars from thousands of people,” prosecutor Thane Rehn told the court. “He defrauded sophisticated investors and lenders, and he emptied
the accounts of ordinary customers, too. He bought himself wealth, power and influence.”\n\nThe defense, on the other hand, depicted SBF and his colleagues as
in over their heads, without ill-will as FTX collapsed around them.\n\n“Sam and his colleagues were building the plane as they were flying it,” defense attorney
Mark Cohen said. “They had to figure out how to navigate a world where they were running FTX, building out its systems, dealing with hacking threats, managing
the credit risk of their customers, managing hundreds of employees, all while building up their actual exchange.”\n\nRead a recap of the prosecution and
defense’s full arguments here.'}, {'page': 'SBF Trial: The latest updates from the FTX collapse’s courtroom drama (TechCrunch, 2023-10-06T17:22:02+00:00)',
'passage': 'SBF Trial: The latest updates from the FTX collapse’s courtroom drama (TechCrunch, 2023-10-06T17:22:02+00:00)\nRead a recap of the prosecution and
defense’s full arguments here.\n\nFTX’s co-founder and CTO Gary Wang testified Thursday on the extent to which Alameda Research was used for whatever purposes
he and SBF saw fit. The crypto trading firm, according to Wang, pulled funding directly from FTX customers, whose transactions would be funneled toward Alameda
and then directed elsewhere.\n\nCheck here for our rundown of Wang’s testimony so far, resulting from a guilty plea in December 2022.\n\nThursday also brought
in testimony from Matt Huang, co-founder and managing partner at crypto investment firm Paradigm. That firm invested $278 million in FTX across 2021 and 2022,
and is part of a class-action lawsuit accusing it and others of defrauding customers by promoting FTX.\n\nHis testimony centered around being left uninformed
about FTX’s utilization of customer money to keep the Alameda Research afloat, a practice that Huang says would have halted his involvement.'}, {'page': 'SBF
Trial: The latest updates from the FTX collapse’s courtroom drama (TechCrunch, 2023-10-06T17:22:02+00:00)', 'passage': 'SBF Trial: The latest updates from the
FTX collapse’s courtroom drama (TechCrunch, 2023-10-06T17:22:02+00:00)\nThe full breakdown of his testimony on his relationship with FTX, stretching back to
2019, can be found right here.\n\nWe also got a look at the composition of the SBF trial’s jury, which ranges from a train conductor to a retired investment
banker who attended Stanford University. You can listen to a special joint episode of our podcasts Equity and Chain Reaction to get a better sense of the vibe
between the jurors and the rest of the court.\n\nFor a full rundown of our coverage of the SBF trial, check below:'}]
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
</item 10>
=== END ITEM id=301bc60cb518fe2c ===
