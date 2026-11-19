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

=== ITEM id=f4175c59f29b4303 ===
Gold sentences and facts:
Sentence 0: During cross-examination, Huang said Paradigm pressed Bankman-Fried on the board issue and was told he didn't want investors as directors but he did plan on having a board with experts.
  Fact 0.0: Huang testified during cross-examination.
  Fact 0.1: Huang said Paradigm pressed Bankman-Fried on the board issue.
  Fact 0.2: Bankman-Fried told Paradigm he did not want investors as directors.
  Fact 0.3: Bankman-Fried planned to have a board with experts.
Sentence 1: It was Bankman-Fried’s decision, she said, as he was the CEO of FTX.
  Fact 1.0: The speaker said the board decision was Bankman-Fried's decision.
  Fact 1.1: Bankman-Fried was the CEO of FTX.
Sentence 2: SBF, as he’s known, has pleaded not-guilty to some seven charges of fraud and conspiracy.
  Fact 2.0: Bankman-Fried, known as SBF, has pleaded not guilty.
  Fact 2.1: Bankman-Fried has pleaded not guilty to some seven charges.
  Fact 2.2: The charges are charges of fraud and conspiracy.
Sentence 3: The prosecution painted Bankman-Fried as someone who knowingly committed fraud to achieve great wealth, power and influence, while the defense countered that the FTX founder acted in good faith, never meant to commit fraud or steal and basically got in over his head.
  Fact 3.0: The prosecution portrayed Bankman-Fried as someone who knowingly committed fraud.
  Fact 3.1: The prosecution said Bankman-Fried committed fraud to achieve great wealth, power and influence.
  Fact 3.2: The defense argued that Bankman-Fried, the FTX founder, acted in good faith.
  Fact 3.3: The defense argued that Bankman-Fried, the FTX founder, never meant to commit fraud or steal.
  Fact 3.4: The defense argued that Bankman-Fried, the FTX founder, basically got in over his head.

Retrieved items:
<item 1>
SBF, riding high on FTX, reportedly offered $5B to Trump to cancel President re-run (TechCrunch, 2023-10-02T13:11:07+00:00)
Come Tuesday, our eyes will be trained on the Manhattan Federal Court for the trial of Sam Bankman-Fried, the disgraced cryptocurrency entrepreneur accused of
orchestrating “one of the biggest financial frauds in American history”: the multibillion-dollar collapse of FTX, the crypto exchange he founded and led.

SBF, as he’s known, has pleaded not-guilty to some seven charges of fraud and conspiracy. But, as we wait to hear both his version and the government’s version
of what happened, there will be another tale recounting how things played out: “Going Infinite,” a book by Michael Lewis chronicling the rise and fall of SBF,
is set to debut on the first day of the trial.

The episode has fired up a lot of people, indignant at what they feel was too-easy treatment for someone who many believe brought down the whole crypto house.

60 Minutes got rightly community noted here.

Graph facts:
- SBF, riding high on FTX, reportedly offered $5B to Trump to cancel President re-run -MENTIONS-> Anna
- Sam Bankman-Fried was a terrible boyfriend -MENTIONS-> Going Infinite
- SBF, riding high on FTX, reportedly offered $5B to Trump to cancel President re-run -MENTIONS-> SBF
- Tennis News: Kyrgios escalates beef with Becker, change ‘warrior’ Novak had to make, WTA criticised for charisma fail -MENTIONS-> 60 Minutes
- SBF, riding high on FTX, reportedly offered $5B to Trump to cancel President re-run -MENTIONS-> 60 Minutes

Entities: Manhattan Federal Court, Going Infinite, 60 Minutes, SBF, riding high on FTX, reportedly offered $5B to Trump to cancel President re-run, Michael
Lewis, Sam Bankman-Fried trial, TechCrunch, conspiracy, FTX, Sam Bankman-Fried, Trump, fraud
</item 1>

<item 2>
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
</item 2>

<item 3>
Sam Bankman-Fried didn’t ask where the $8 billion went (The Verge, 2023-10-31T19:52:07+00:00)
But telling your side of the story opens you to a great many questions you might not want to answer on cross-examination. I have watched a lot of crosses. This
was the nastiest I think I’ve ever seen.

The charges in this case hinge on conspiracy and intentional deceit. Just losing $8 billion is not a crime, though it is very embarrassing. (Even losing $900
million is very embarrassing because people will make jokes about it forever!) But if Bankman-Fried lied to customers and lenders about what he was doing and
how safe FTX was, that’s a crime.

Establishing that would have been enough, but Sassoon also managed to get in some flourishes about Bankman-Fried’s relationships to Bahamian power. For
instance, she asked him if he’d made comments about paying off the Bahamian national debt. (You will be shocked, shocked to discover he didn’t recall.)

Graph facts:
- Sam Bankman-Fried didn’t ask where the $8 billion went -MENTIONS-> Caroline Ellison
- Sam Bankman-Fried didn’t ask where the $8 billion went -MENTIONS-> Adam Yedidia
- Sam Bankman-Fried didn’t ask where the $8 billion went -MENTIONS-> Chicago
- Sam Bankman-Fried didn’t ask where the $8 billion went -MENTIONS-> Securities and Exchange Commission
- Sam Bankman-Fried didn’t ask where the $8 billion went -MENTIONS-> misappropriation of customer funds

Entities: Sam Bankman-Fried didn’t ask where the $8 billion went, intentional deceit, Bahamian national debt, Sassoon, conspiracy, FTX, Sam Bankman-Fried, The
Verge
</item 3>

<item 4>
Sex, Signal messages, and sabotaging FTX's code: SBF execs and Bahamas roommates tell all in court (Cnbc | World Business News Leader,
2023-10-06T20:17:00+00:00)
Sam Bankman-Fried, the founder of bankrupt cryptocurrency exchange FTX, is seen during a hearing as a U.S judge revoked his bail, at a courthouse in New York,
U.S., August 11, 2023 in this courtroom sketch. Jane Rosenberg | Reuters

$65 billion line of credit

For months, Bankman-Fried has known that Wang and Ellison, who were integral members of his personal and professional inner circles, had turned on him. Both
pleaded guilty in December and have since been cooperating with the U.S. attorney's office in Manhattan. Wang's testimony, which stretched into Friday, was
given under a cooperation agreement with the government. Ellison is expected to take the stand under a similar arrangement.

U.S.

Graph facts:
- Is Sam Bankman-Fried a bad ‘man’ or a good ‘boy’? Lawyers swap opening statements before first witnesses take the stand -MENTIONS-> Ellison
- Sam Bankman-Fried was a terrible boyfriend -MENTIONS-> Ellison
- In the end, the FTX trial was about the friends screwed along the way -MENTIONS-> Ellison
- The jury finally hears from Sam Bankman-Fried -MENTIONS-> Ellison
- How is it still getting worse for Sam Bankman-Fried? -MENTIONS-> Ellison

Entities: Ellison, New York, Jane Rosenberg, FTX, U.S. attorney's office, courthouse, pleading guilty, Wang's testimony, Sex, Signal messages, and sabotaging
FTX's code: SBF execs and Bahamas roommates tell all in court, Wang, hearing, Sam Bankman-Fried, Reuters, Manhattan, CNBC
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
SBF’s trial starts soon, but how did he — and FTX — get here? (TechCrunch, 2023-10-01T14:00:29+00:00)
In a published DM exchange with a Vox reporter, he said he regretted filing for bankruptcy and thought that “regulators make everything worse.”

SBF arrested

Bankman-Fried was arrested in December 2022 in the Bahamas, where FTX was based. He was then extradited to the U.S. to face a number of criminal charges. He was
released on a $250 million bail bond, and he remained under house arrest at his parents’ home in Palo Alto. This was revoked in August after he was accused of
intimidating Alameda’s former CEO, Caroline Ellison, by leaking her private diary.

Ray represented the company during a House Financial Services Committee hearing regarding FTX. When asked whether the firm had significant risk management
systems, Ray said at the time that “there were virtually no internal controls and no separateness whatsoever” and added that he did not “trust a single piece of
paper” in the exchange’s organization. U.S.
</item 6>

<item 7>
In the end, the FTX trial was about the friends screwed along the way (The Verge, 2023-10-26T12:00:00+00:00)
It is easy to lie with words and harder to lie with money. Bankman-Fried went around telling people that he wanted to make a lot of money to give it away, an
idea of the Effective Altruism movement that Bankman-Fried affiliated himself with — but the amount of money he funneled to his Paper Bird vehicle alone was ten
times the size of his philanthropic spending as outlined by Easton. And it was all vastly outweighed by his investments — the places where he could prove he was
very smart by investing in the right companies early.

And in brief, explosive testimony, former FTX lawyer Can Sun demonstrated what Bankman-Fried’s tactic was. During the days in November when the exchange
imploded, Bankman-Fried was trying to raise money from Apollo, the private equity group. Apollo wanted an explanation for what had happened, and so
Bankman-Fried came to Sun to ask him to come up with justifications for the missing funds.

Graph facts:
- SBF Trial: The latest updates from the FTX collapse’s courtroom drama -MENTIONS-> Easton
- Easton -PARTICIPATED_IN-> SBF Trial
- SBF’s trial starts soon, but how did he — and FTX — get here? -MENTIONS-> Bankman-Fried
- Sam Altman backs teens’ startup, Google unveils the Pixel 8 and TikTok tests an ad-free tier -MENTIONS-> Bankman-Fried
- Sex, Signal messages, and sabotaging FTX's code: SBF execs and Bahamas roommates tell all in court -MENTIONS-> Bankman-Fried

Entities: Easton, Bankman-Fried, In the end, the FTX trial was about the friends screwed along the way, Paper Bird, Effective Altruism movement, Can Sun,
Apollo, missing funds, FTX trial, FTX, The Verge
</item 7>

<item 8>
Is Sam Bankman-Fried a bad ‘man’ or a good ‘boy’? Lawyers swap opening statements before first witnesses take the stand (Fortune, 2023-10-04T23:32:53+00:00)
Alameda Research, a crypto hedge fund he also owned, argued Rehn. Using Caroline Ellison, his on-again off-again girlfriend and the CEO of Alameda, as a front,
Bankman-Fried had “secret access” to customer money—both cash and crypto—the government claimed.

Moreover, Bankman-Fried allegedly directed employees to conceal the flow of money into FTX’s coffers and forged financial documents distributed to lenders and
investors. “The defendant lied to the world,” Rehn alleged.

And who was this defendant? Not a crypto “boy” genius, as so many in the media (Fortune included) have written, but a “man” who “stole billions of dollars from
thousands of victims,” Rehn said. “You will see the full picture.”

…or a well-meaning founder?

Graph facts:
- SBF’s trial starts soon, but how did he — and FTX — get here? -MENTIONS-> Caroline Ellison
- SBF Trial: The latest updates from the FTX collapse’s courtroom drama -MENTIONS-> Caroline Ellison
- The jury finally hears from Sam Bankman-Fried -MENTIONS-> Caroline Ellison
- Sam Bankman-Fried didn’t ask where the $8 billion went -MENTIONS-> Caroline Ellison
- Sam Bankman-Fried was a terrible boyfriend -MENTIONS-> Caroline Ellison

Entities: Caroline Ellison, theft, Is Sam Bankman-Fried a bad ‘man’ or a good ‘boy’? Lawyers swap opening statements before first witnesses take the stand,
Alameda Research, Rehn, Fortune, FTX, Sam Bankman-Fried, government
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
Graph query row:
person: Sam Bankman-Fried
connections: [{'related': 'SBF Trial: The latest updates from the FTX collapse’s courtroom drama', 'relationship': 'MENTIONS'}, {'related': 'SBF’s trial starts
soon, but how did he — and FTX — get here?', 'relationship': 'MENTIONS'}, {'related': 'SBF, riding high on FTX, reportedly offered $5B to Trump to cancel
President re-run', 'relationship': 'MENTIONS'}, {'related': 'Is Sam Bankman-Fried a bad ‘man’ or a good ‘boy’? Lawyers swap opening statements before first
witnesses take the stand', 'relationship': 'MENTIONS'}, {'related': 'The jury finally hears from Sam Bankman-Fried', 'relationship': 'MENTIONS'}, {'related':
'Sam Bankman-Fried didn’t ask where the $8 billion went', 'relationship': 'MENTIONS'}, {'related': 'Sam Bankman-Fried was a terrible boyfriend', 'relationship':
'MENTIONS'}, {'related': 'In the end, the FTX trial was about the friends screwed along the way', 'relationship': 'MENTIONS'}, {'related': 'Is Sam
Bankman-Fried’s defense even trying to win?', 'relationship': 'MENTIONS'}, {'related': 'The FTX trial is bigger than Sam Bankman-Fried', 'relationship':
'MENTIONS'}, {'related': "Sex, Signal messages, and sabotaging FTX's code: SBF execs and Bahamas roommates tell all in court", 'relationship': 'MENTIONS'},
{'related': 'How is it still getting worse for Sam Bankman-Fried?', 'relationship': 'MENTIONS'}, {'related': 'Coinbase rallies more than 60% in same month that
FTX and Binance founders brace for prison', 'relationship': 'MENTIONS'}, {'related': 'financial fraud', 'relationship': 'SAID'}, {'related': 'market
manipulation', 'relationship': 'SAID'}, {'related': 'Alameda Research', 'relationship': 'SAID'}, {'related': 'risk engine', 'relationship': 'SAID'}, {'related':
'Alameda', 'relationship': 'SAID'}, {'related': 'fiat@ftx account', 'relationship': 'SAID'}, {'related': 'risk management', 'relationship': 'SAID'}, {'related':
'Ryan Salame', 'relationship': 'SAID'}, {'related': 'Bahamian prime minister', 'relationship': 'SAID'}, {'related': 'Risk-neutral', 'relationship': 'SAID'},
{'related': 'FTX Ventures', 'relationship': 'SAID'}, {'related': 'Bahamas', 'relationship': 'SAID'}, {'related': 'allow_negative', 'relationship': 'SAID'},
{'related': 'Ellison', 'relationship': 'SAID'}, {'related': 'customer assets', 'relationship': 'SAID'}, {'related': 'Sullivan & Cromwell', 'relationship':
'SAID'}, {'related': 'FTX', 'relationship': 'SAID'}, {'related': 'Wang', 'relationship': 'SAID'}, {'related': 'Binance', 'relationship': 'SAID'}, {'related':
'fraud', 'relationship': 'SAID'}, {'related': 'Thane Rehn', 'relationship': 'SAID'}, {'related': 'Mark Cohen', 'relationship': 'SAID'}, {'related': 'Securities
and Exchange Commission', 'relationship': 'SAID'}, {'related': 'Rehn', 'relationship': 'SAID'}, {'related': 'Cohen', 'relationship': 'SAID'}, {'related':
'LaVigne', 'relationship': 'SAID'}, {'related': 'government', 'relationship': 'SAID'}, {'related': 'Jane Street', 'relationship': 'WORKS_FOR'}, {'related':
'Alameda Research', 'relationship': 'WORKS_FOR'}, {'related': 'Alameda', 'relationship': 'WORKS_FOR'}, {'related': 'FTX', 'relationship': 'WORKS_FOR'},
{'related': 'SBF Trial', 'relationship': 'PARTICIPATED_IN'}, {'related': 'SBF’s trial', 'relationship': 'PARTICIPATED_IN'}, {'related': 'Bankman-Fried’s trial',
'relationship': 'PARTICIPATED_IN'}, {'related': 'Bankman-Fried pleaded not guilty', 'relationship': 'PARTICIPATED_IN'}, {'related': 'evidentiary hearing',
'relationship': 'PARTICIPATED_IN'}, {'related': "Gary Wang's testimony", 'relationship': 'PARTICIPATED_IN'}, {'related': 'testimony', 'relationship':
'PARTICIPATED_IN'}, {'related': 'jury hears from Sam Bankman-Fried', 'relationship': 'PARTICIPATED_IN'}, {'related': 'balcony conversation', 'relationship':
'PARTICIPATED_IN'}, {'related': 'testifying to Congress', 'relationship': 'PARTICIPATED_IN'}, {'related': 'getting on the stand', 'relationship':
'PARTICIPATED_IN'}, {'related': 'Sam Bankman-Fried trial', 'relationship': 'PARTICIPATED_IN'}, {'related': 'opening withdrawals for Bahamian customers',
'relationship': 'PARTICIPATED_IN'}, {'related': 'criminal fraud trial', 'relationship': 'PARTICIPATED_IN'}, {'related': 'FTX trial', 'relationship':
'PARTICIPATED_IN'}, {'related': 'prosecution’s opening statement', 'relationship': 'PARTICIPATED_IN'}, {'related': 'extensive media tour', 'relationship':
'PARTICIPATED_IN'}, {'related': 'interviews', 'relationship': 'PARTICIPATED_IN'}, {'related': 'Congress', 'relationship': 'PARTICIPATED_IN'}, {'related': 'Sam
Bankman-Fried fraud trial', 'relationship': 'PARTICIPATED_IN'}, {'related': 'summer camp', 'relationship': 'PARTICIPATED_IN'}, {'related': "Sam Bankman-Fried's
fraud trial", 'relationship': 'PARTICIPATED_IN'}, {'related': 'hearing', 'relationship': 'PARTICIPATED_IN'}, {'related': 'trial', 'relationship':
'PARTICIPATED_IN'}, {'related': 'the trial', 'relationship': 'PARTICIPATED_IN'}, {'related': 'monthlong trial', 'relationship': 'PARTICIPATED_IN'}, {'related':
'Sam Bankman-Fried sentencing', 'relationship': 'PARTICIPATED_IN'}, {'related': 'conspiracy to commit money laundering', 'relationship': 'ACCUSED_OF'},
{'related': 'conspiracy to misuse customer funds', 'relationship': 'ACCUSED_OF'}, {'related': 'civil penalties', 'relationship': 'ACCUSED_OF'}, {'related': 'all
charges', 'relationship': 'ACCUSED_OF'}, {'related': 'alleged crimes', 'relationship': 'ACCUSED_OF'}, {'related': 'stealing customer funds', 'relationship':
'ACCUSED_OF'}, {'related': 'theft', 'relationship': 'ACCUSED_OF'}, {'related': 'misappropriation of customer funds', 'relationship': 'ACCUSED_OF'}, {'related':
'intentional deceit', 'relationship': 'ACCUSED_OF'}, {'related': 'wire fraud', 'relationship': 'ACCUSED_OF'}, {'related': 'conspiracy charges', 'relationship':
'ACCUSED_OF'}, {'related': 'money laundering', 'relationship': 'ACCUSED_OF'}, {'related': 'conspiracy', 'relationship': 'ACCUSED_OF'}, {'related': 'lying',
'relationship': 'ACCUSED_OF'}, {'related': 'stealing', 'relationship': 'ACCUSED_OF'}, {'related': 'consumer protection', 'relationship': 'ACCUSED_OF'},
{'related': 'misappropriated FTX customer deposits', 'relationship': 'ACCUSED_OF'}, {'related': 'political contributions', 'relationship': 'ACCUSED_OF'},
{'related': 'lying to FTX investors', 'relationship': 'ACCUSED_OF'}, {'related': 'conspiracy to take FTX customer funds', 'relationship': 'ACCUSED_OF'},
{'related': 'fraud', 'relationship': 'ACCUSED_OF'}, {'related': 'Alameda Research', 'relationship': 'OWNS'}, {'related': 'Alameda Research', 'relationship':
'FOUNDED'}, {'related': 'FTX', 'relationship': 'FOUNDED'}, {'related': 'FTT', 'relationship': 'FOUNDED'}]
articles: [{'passage': 'SBF Trial: The latest updates from the FTX collapse’s courtroom drama (TechCrunch, 2023-10-06T17:22:02+00:00)\nSBF trial: Everything to
know from the FTX courtroom ahead of his testimony Sam Bankman-Fried is in court for alleged fraud and money laundering while leading crypto exchange FTX and
sister company Alameda. Here’s our latest coverage as the case nears its end.\n\nThe trial of Bankman-Fried, the former CEO and co-founder of collapsed crypto
exchange FTX, is entering its next chapter as proceedings resume Thursday. As confirmed during a teleconference Wednesday, SBF will be taking the stand, along
with a handful of other witnesses as the onus shifts toward the defense after the prosecution brought its case to its conclusion.\n\nAs former federal
prosecutor Josh Naftalis told TechCrunch, SBF taking to the stand in his own trial is a “Hail Mary” ahead of a verdict and potential sentencing for seven counts
tied to fraud and money laundering.\n\n“Once it goes to cross[-examination], he doesn’t get to say, ‘I’m done,’” Naftalis said.', 'publisher': 'TechCrunch',
'page': 'SBF Trial: The latest updates from the FTX collapse’s courtroom drama (TechCrunch, 2023-10-06T17:22:02+00:00)', 'article': 'SBF Trial: The latest
updates from the FTX collapse’s courtroom drama'}, {'passage': 'SBF Trial: The latest updates from the FTX collapse’s courtroom drama (TechCrunch,
2023-10-06T17:22:02+00:00)\n“He can’t just walk out if he doesn’t like how it’s going.”\n\n\n\nNaftalis spoke at length about what comes next in the trial, and
addressed common misconceptions about the prosecution and defense in the latest episode of TechCrunch’s Chain Reaction podcast, which you can listen to below,
or on Apple Podcasts or Spotify.\n\nThe SBF trial is one of the biggest cases of its kind within the crypto space. Our resident crypto expert Jacquelyn Melinek
has been on the ground at the trial since it started, and continues to cover the ins and outs of the proceedings. But the trial has been on pause from October
20 to today, so there’s plenty to catch up on if you need a refresher ahead of it entering its final few phases.\n\nFor those in need of a total refresher on
the case, her breakdown of how FTX went from the third-largest crypto exchange valued at a peak of $32 billion to bankruptcy will get you up to speed.',
'publisher': 'TechCrunch', 'page': 'SBF Trial: The latest updates from the FTX collapse’s courtroom drama (TechCrunch, 2023-10-06T17:22:02+00:00)', 'article':
'SBF Trial: The latest updates from the FTX collapse’s courtroom drama'}, {'passage': 'SBF Trial: The latest updates from the FTX collapse’s courtroom drama
(TechCrunch, 2023-10-06T17:22:02+00:00)\nYou can also follow along with her and the TechCrunch team’s coverage in the Chain Reaction newsletter, which drops
Thursdays at 12 p.m. PT, and the Chain Reaction podcast, which will hit your feed every Thursday.\n\nHow to follow the SBF trial\n\nThe trial kicked off with
jury selection October 3. From there, access to the proceedings is limited, as no devices are allowed; reports are being physically put together within the
courtroom (or sent to overflow rooms).\n\nBeyond our daily coverage, additional expertise and commentary will live on TechCrunch+. There, you can find stories
like this in-depth breakdown of what to expect from both sides of the SBF case, where the prosecution and defense could gain ground or fall short in their
arguments and what the takeaways were from the opening arguments.\n\nSBF trial: What we learned in week three\n\nThe third week of the trial featured a couple
of noteworthy moments for the prosecution, including FTX’s former general counsel Can Sun.', 'publisher': 'TechCrunch', 'page': 'SBF Trial: The latest updates
from the FTX collapse’s courtroom drama (TechCrunch, 2023-10-06T17:22:02+00:00)', 'article': 'SBF Trial: The latest updates from the FTX collapse’s courtroom
drama'}, {'passage': 'SBF Trial: The latest updates from the FTX collapse’s courtroom drama (TechCrunch, 2023-10-06T17:22:02+00:00)\nHe, like many of SBF’s
former colleagues, opted to cooperate with authorities and testified that SBF asked him to create a theoretical “legal justification” for the use of billions in
consumer funds just days before FTX filed for bankruptcy.\n\nRobert Boroujerdi, managing director of the hedge fund Third Point, testified about his experience
working with SBF and FTX, in a partnership that ultimately resulted in a $60 million investment and a lack of awareness about the special relationship between
Alameda Research and FTX. Had he known of such a relationship, he testified that Third Point would not have invested in FTX in the first place.\n\nAnd earlier
in the week, a University of Notre Dame professor who helped the government prosecution of Enron and WorldCom, Peter Easton, testified about what he uncovered
after tracing the flow of billions of dollars between Alameda Research and FTX.', 'publisher': 'TechCrunch', 'page': 'SBF Trial: The latest updates from the FTX
collapse’s courtroom drama (TechCrunch, 2023-10-06T17:22:02+00:00)', 'article': 'SBF Trial: The latest updates from the FTX collapse’s courtroom drama'},
{'passage': 'SBF Trial: The latest updates from the FTX collapse’s courtroom drama (TechCrunch, 2023-10-06T17:22:02+00:00)\nMuch of said funding came from
customers, and Easton testified that he found that user deposits were used by both FTX and Alameda to purchase real estate, make investments or direct funding
toward political causes and charities.\n\nSBF trial: What we learned in week two\n\nThe second week of the trial’s standout testimony came from Alameda
Research’s former CEO Caroline Ellison, who claimed she took $14 billion from customers to repay debts to lenders, using them as a line of credit under the
instruction of SBF. Ellison also went into detail about how a $2 billion venture fund, FTX Ventures, was funded by money that had previously been allocated to
Alameda from third-party lenders.\n\nAnd though there are no charges of bribery in the case, Ellison also testified to paying Chinese officials $150 million to
restore access to $1 billion in frozen trading accounts.\n\nSBF trial: What we learned in the week one\n\nAs expected, the opening statements in the SBF trial
were spicy.', 'publisher': 'TechCrunch', 'page': 'SBF Trial: The latest updates from the FTX collapse’s courtroom drama (TechCrunch,
2023-10-06T17:22:02+00:00)', 'article': 'SBF Trial: The latest updates from the FTX collapse’s courtroom drama'}, {'passage': 'SBF Trial: The latest updates
from the FTX collapse’s courtroom drama (TechCrunch, 2023-10-06T17:22:02+00:00)\nThe prosecution painted a picture of an empire “built on lies” and made a point
to show actual pictures of SBF alongside rich and powerful figures to illustrate the peak of his power and influence prior to FTX’s implosion.\n\n“This man
stole billions of dollars from thousands of people,” prosecutor Thane Rehn told the court. “He defrauded sophisticated investors and lenders, and he emptied the
accounts of ordinary customers, too. He bought himself wealth, power and influence.”\n\nThe defense, on the other hand, depicted SBF and his colleagues as in
over their heads, without ill-will as FTX collapsed around them.\n\n“Sam and his colleagues were building the plane as they were flying it,” defense attorney
Mark Cohen said. “They had to figure out how to navigate a world where they were running FTX, building out its systems, dealing with hacking threats, managing
the credit risk of their customers, managing hundreds of employees, all while building up their actual exchange.”\n\nRead a recap of the prosecution and
defense’s full arguments here.', 'publisher': 'TechCrunch', 'page': 'SBF Trial: The latest updates from the FTX collapse’s courtroom drama (TechCrunch,
2023-10-06T17:22:02+00:00)', 'article': 'SBF Trial: The latest updates from the FTX collapse’s courtroom drama'}, {'passage': 'SBF Trial: The latest updates
from the FTX collapse’s courtroom drama (TechCrunch, 2023-10-06T17:22:02+00:00)\nRead a recap of the prosecution and defense’s full arguments here.\n\nFTX’s
co-founder and CTO Gary Wang testified Thursday on the extent to which Alameda Research was used for whatever purposes he and SBF saw fit. The crypto trading
firm, according to Wang, pulled funding directly from FTX customers, whose transactions would be funneled toward Alameda and then directed elsewhere.\n\nCheck
here for our rundown of Wang’s testimony so far, resulting from a guilty plea in December 2022.\n\nThursday also brought in testimony from Matt Huang,
co-founder and managing partner at crypto investment firm Paradigm. That firm invested $278 million in FTX across 2021 and 2022, and is part of a class-action
lawsuit accusing it and others of defrauding customers by promoting FTX.\n\nHis testimony centered around being left uninformed about FTX’s utilization of
customer money to keep the Alameda Research afloat, a practice that Huang says would have halted his involvement.', 'publisher': 'TechCrunch', 'page': 'SBF
Trial: The latest updates from the FTX collapse’s courtroom drama (TechCrunch, 2023-10-06T17:22:02+00:00)', 'article': 'SBF Trial: The latest updates from the
FTX collapse’s courtroom drama'}, {'passage': 'SBF Trial: The latest updates from the FTX collapse’s courtroom drama (TechCrunch,
2023-10-06T17:22:02+00:00)\nThe full breakdown of his testimony on his relationship with FTX, stretching back to 2019, can be found right here.\n\nWe also got a
look at the composition of the SBF trial’s jury, which ranges from a train conductor to a retired investment banker who attended Stanford University. You can
listen to a special joint episode of our podcasts Equity and Chain Reaction to get a better sense of the vibe between the jurors and the rest of the
court.\n\nFor a full rundown of our coverage of the SBF trial, check below:', 'publisher': 'TechCrunch', 'page': 'SBF Trial: The latest updates from the FTX
collapse’s courtroom drama (TechCrunch, 2023-10-06T17:22:02+00:00)', 'article': 'SBF Trial: The latest updates from the FTX collapse’s courtroom drama'},
{'passage': 'SBF’s trial starts soon, but how did he — and FTX — get here? (TechCrunch, 2023-10-01T14:00:29+00:00)\nWithin two years, more than 80 investors
provided about $2 billion in capital to FTX, helping Bankman-Fried propel his vision into a reality. In January 2022, the company raised $400 million in a
Series C round, boosting its valuation to $32 billion. That was its last round of public funding.\n\nThe company gained somewhat mainstream recognition with
branding deals and partnerships. For example, in 2021 it bought the naming rights for the Miami Heat’s home arena. FTX also got its name branded on Major League
Baseball umpires’ polos, and it partnered with celebrities like Tom Brady and his ex-wife, Gisele Bündchen, as well as Steph Curry, Shaquille O’Neal and Naomi
Osaka, among others. He also had close ties to U.S. regulators and government officials, many of whom he donated to.', 'publisher': 'TechCrunch', 'page': 'SBF’s
trial starts soon, but how did he — and FTX — get here? (TechCrunch, 2023-10-01T14:00:29+00:00)', 'article': 'SBF’s trial starts soon, but how did he — and FTX
— get here?'}, {'passage': 'SBF’s trial starts soon, but how did he — and FTX — get here? (TechCrunch, 2023-10-01T14:00:29+00:00)\nregulators and government
officials, many of whom he donated to.\n\nBankman-Fried was even compared to Warren Buffet and many called him the white horse of crypto (TechCrunch never did,
for what it’s worth).\n\nBut in early November 2022, that all changed.\n\nFTX’s collapse\n\nConcerns surrounding FTX’s liquidity grew after CoinDesk published a
copy of Alameda’s balance sheet, showing the firm held $14.6 billion in assets and $8 billion in liabilities as of June 30, 2022.\n\nBut there was a problem:
The report showed Alameda’s largest asset was $3.66 billion of “unlocked FTT” and $2.16 billion of “FTT collateral.” FTT was the token behind FTX.\n\nThe
balance sheet showed that the $5.82 billion in FTT tokens that Alameda owned was 193% higher than the total FTT market cap, which was about $3 billion at the
time.', 'publisher': 'TechCrunch', 'page': 'SBF’s trial starts soon, but how did he — and FTX — get here? (TechCrunch, 2023-10-01T14:00:29+00:00)', 'article':
'SBF’s trial starts soon, but how did he — and FTX — get here?'}, {'passage': 'SBF’s trial starts soon, but how did he — and FTX — get here? (TechCrunch,
2023-10-01T14:00:29+00:00)\nThat means it purported to have more FTT tokens on its balance sheet than what existed in the world.\n\nAround the same time it was
exposed, the world’s largest crypto exchange, Binance, started pulling out its remaining $2.1 billion equivalent of cash in BUSD and FTT. (It had an equity
position in FTX from 2019 to 2021.) This essentially triggered a bank run on FTX.\n\nFTX and Alameda filed for Chapter 11 bankruptcy in the U.S. mid-November
2022. Bankman-Fried resigned, and John J. Ray III, the Enron turnaround veteran, was appointed its new CEO.\n\nBankman-Fried, however, maintained his innocence.
At The New York Times’ DealBook Summit, he appeared virtually from the Bahamas, saying “I didn’t ever try to commit fraud on anyone; I was shocked by what
happened this month.”', 'publisher': 'TechCrunch', 'page': 'SBF’s trial starts soon, but how did he — and FTX — get here? (TechCrunch,
2023-10-01T14:00:29+00:00)', 'article': 'SBF’s trial starts soon, but how did he — and FTX — get here?'}, {'passage': 'SBF’s trial starts soon, but how did he —
and FTX — get here? (TechCrunch, 2023-10-01T14:00:29+00:00)\nU.S. Attorney Damian Williams called Bankman-Fried’s alleged crimes “one of the biggest financial
frauds in American history,” in a press conference.\n\nThe aftermath\n\nFTX co-founder and former CTO Gary Wang, and Alameda Research’s former CEO, Caroline
Ellison, both pleaded guilty in December 2022 to federal criminal charges in relation to the FTX collapse. They’re also facing civil penalties from the U.S.
Securities and Exchange Commission (SEC) and Commodity Futures Trading Commission (CFTC) alongside the criminal charges. Wang and Ellison plan to cooperate with
prosecutors and will be major witnesses in the trial, given their close ties to Bankman-Fried, FTX and Alameda.\n\nIn January, Bankman-Fried pleaded not guilty
to all counts, which include wire fraud, conspiracy to commit money laundering and conspiracy to misuse customer funds. He could face up to 115 years in jail if
convicted on all charges.', 'publisher': 'TechCrunch', 'page': 'SBF’s trial starts soon, but how did he — and FTX — get here? (TechCrunch,
2023-10-01T14:00:29+00:00)', 'article': 'SBF’s trial starts soon, but how did he — and FTX — get here?'}, {'passage': 'SBF’s trial starts soon, but how did he —
and FTX — get here? (TechCrunch, 2023-10-01T14:00:29+00:00)\nHe could face up to 115 years in jail if convicted on all charges.\n\nThe crypto industry as a
whole suffered from FTX’s collapse, which was the first of many. BlockFi filed for Chapter 11 in November 2022, as did Genesis Global Trading in
January.\n\nWhere we are today\n\nBankman-Fried will be represented by Cohen & Gresser, and Mark Cohen, a high-profile defense attorney and former federal
prosecutor, will be the lead attorney. If that name sounds familiar, it might be because he also represented Ghislaine Maxwell in her sex trafficking trial
related to Jeffrey Epstein. He requested an early release for Bankman-Fried but was denied.\n\nWith the trial beginning, we’re witnessing how FTX’s story ends.
But what’s on our mind is what happens to the investors and creditors affected by the collapse? And what happens to the billions in crypto assets tied up in
legal proceedings?', 'publisher': 'TechCrunch', 'page': 'SBF’s trial starts soon, but how did he — and FTX — get here? (TechCrunch, 2023-10-01T14:00:29+00:00)',
'article': 'SBF’s trial starts soon, but how did he — and FTX — get here?'}, {'passage': 'SBF, riding high on FTX, reportedly offered $5B to Trump to cancel
President re-run (TechCrunch, 2023-10-02T13:11:07+00:00)\nCome Tuesday, our eyes will be trained on the Manhattan Federal Court for the trial of Sam
Bankman-Fried, the disgraced cryptocurrency entrepreneur accused of orchestrating “one of the biggest financial frauds in American history”: the
multibillion-dollar collapse of FTX, the crypto exchange he founded and led.\n\nSBF, as he’s known, has pleaded not-guilty to some seven charges of fraud and
conspiracy. But, as we wait to hear both his version and the government’s version of what happened, there will be another tale recounting how things played out:
“Going Infinite,” a book by Michael Lewis chronicling the rise and fall of SBF, is set to debut on the first day of the trial.\n\nThe episode has fired up a lot
of people, indignant at what they feel was too-easy treatment for someone who many believe brought down the whole crypto house.\n\n60 Minutes got rightly
community noted here.', 'publisher': 'TechCrunch', 'page': 'SBF, riding high on FTX, reportedly offered $5B to Trump to cancel President re-run (TechCrunch,
2023-10-02T13:11:07+00:00)', 'article': 'SBF, riding high on FTX, reportedly offered $5B to Trump to cancel President re-run'}, {'passage': 'SBF, riding high on
FTX, reportedly offered $5B to Trump to cancel President re-run (TechCrunch, 2023-10-02T13:11:07+00:00)\n60 Minutes got rightly community noted here. How is it
that SBF is _still_ getting super positive media coverage when the media has nothing good to say about the founders that have actually created things and
brought value into the world? https://t.co/VPV1ORVHqf — tobi lutke (@tobi) October 2, 2023\n\nThe germination of “Going Infinite” sounds as unlikely as the
story of SBF himself.\n\nIn an interview broadcast yesterday on 60 Minutes, Lewis, the famed chronicler of financial adventure and misadventure — his books
include “Liar’s Poker,” “Moneyball” and “The Big Short” — said that he didn’t set out to write a book about SBF. Rather, Lewis claims he was introduced to him
in early 2021 by a friend who wanted to back FTX. Before any financial commitment, that friend asked Lewis to meet the young billionaire — both lived in
Berkeley — to “evaluate his character.”', 'publisher': 'TechCrunch', 'page': 'SBF, riding high on FTX, reportedly offered $5B to Trump to cancel President
re-run (TechCrunch, 2023-10-02T13:11:07+00:00)', 'article': 'SBF, riding high on FTX, reportedly offered $5B to Trump to cancel President re-run'}, {'passage':
'SBF, riding high on FTX, reportedly offered $5B to Trump to cancel President re-run (TechCrunch, 2023-10-02T13:11:07+00:00)\nNo, we don’t find out who that
friend is, nor do we find out if that particular investment was ever made — SBF did raise some $2 billion, including $1 billion in 2021, from backers that
included Sequoia, Temasek, Tiger Global and dozens more.\n\nBut we do find out that another investment, of sorts, was made on that day: Lewis himself was so
taken with the idea of SBF and his apparent brilliance that he came away with the decision to write about him.\n\n“Eighteen months earlier, he had nothing. Now
he had $22.5 billion. He was the richest person in the world under 30 [and] he was going to spend it to save humanity from extinction,” he said. “My jaw was on
the floor.”\n\nThe two proceeded to meet more than 100 times over the next two years. As SBF’s fame and his company’s bank account grew, 60 Minutes itself even
got involved.', 'publisher': 'TechCrunch', 'page': 'SBF, riding high on FTX, reportedly offered $5B to Trump to cancel President re-run (TechCrunch,
2023-10-02T13:11:07+00:00)', 'article': 'SBF, riding high on FTX, reportedly offered $5B to Trump to cancel President re-run'}, {'passage': 'SBF, riding high on
FTX, reportedly offered $5B to Trump to cancel President re-run (TechCrunch, 2023-10-02T13:11:07+00:00)\nThe program shot (and it shows here) one segment with
SBF talking as he shuffles cards with his leg twitching nervously behind his desk (why? we don’t find this out, either) as Lewis, with a flourish of his pen and
a look of concentration on his face, dashes off notes on his yellow legal pad.\n\nLater in the show, you get other behind-the-scenes, unnerving glimpses of how
the wizard operated behind the curtain. For instance, Lewis recounts how he was in the room when SBF made his first television appearance.\n\n“If you watch the
clip, you’ll see his eyes going back and forth,” he said. “It’s because he’s trying to win his video game at the same time he’s on the air.”\n\n“If you watch
the clip, you’ll see his eyes going back and forth. It’s because he’s trying to win his video game at the same time he’s on the air.”', 'publisher':
'TechCrunch', 'page': 'SBF, riding high on FTX, reportedly offered $5B to Trump to cancel President re-run (TechCrunch, 2023-10-02T13:11:07+00:00)', 'article':
'SBF, riding high on FTX, reportedly offered $5B to Trump to cancel President re-run'}, {'passage': 'SBF, riding high on FTX, reportedly offered $5B to Trump to
cancel President re-run (TechCrunch, 2023-10-02T13:11:07+00:00)\nMichael Lewis\n\nFriendships and lucrative marketing deals followed, with sports icons like Tom
Brady (who was paid $55 million) and Steph Curry ($35 million) to “lend FTX legitimacy and edge.” Even Anna Wintour, the icy fashion doyenne, was in contact:
She reached out, over what must have been a hell of a Zoom meeting, to ask SBF — whose preferred get-up is/was wildly curly hair, t-shirts and cargo shorts — to
sponsor her very fancy Met Costume Ball. Desperate times!\n\nAs trades on FTX climbed up to $15 billion/day, SBF, of course, also got tied up in the corridors
of political power.\n\nLewis said that he met with the young crypto king ahead of him seeing Mitch McConnell to discuss how he could fund political candidates
who were looking to offset Trump’s influence in the Republican Party.', 'publisher': 'TechCrunch', 'page': 'SBF, riding high on FTX, reportedly offered $5B to
Trump to cancel President re-run (TechCrunch, 2023-10-02T13:11:07+00:00)', 'article': 'SBF, riding high on FTX, reportedly offered $5B to Trump to cancel
President re-run'}, {'passage': 'SBF, riding high on FTX, reportedly offered $5B to Trump to cancel President re-run (TechCrunch, 2023-10-02T13:11:07+00:00)\nWe
find out that the gravitational pull of cargo shorts was a tough one to pull away from: The suit SBF brought to wear to the McConnell meeting was tightly balled
up under his arm with his dress shoes falling out of the middle of the bundle. We’ve all been there, Sam. (Well, perhaps not Anna.)\n\nLewis also claimed that
SBF had concocted a plan to buy off Trump, to get him not to run again for president.\n\n“Like how much would it take to get an answer?” Lewis recounted. “There
was a number that was kicking around… this was $5 billion.” He said that Sam was never sure whether that figure came directly from Trump. And no, we don’t find
out if he ever did wire anything to camp Trump. That would be a wonderful detail to get teased out in the trial, though.', 'publisher': 'TechCrunch', 'page':
'SBF, riding high on FTX, reportedly offered $5B to Trump to cancel President re-run (TechCrunch, 2023-10-02T13:11:07+00:00)', 'article': 'SBF, riding high on
FTX, reportedly offered $5B to Trump to cancel President re-run'}, {'passage': 'SBF, riding high on FTX, reportedly offered $5B to Trump to cancel President
re-run (TechCrunch, 2023-10-02T13:11:07+00:00)\nThat would be a wonderful detail to get teased out in the trial, though.\n\nAnd lest you think Biden and the
Democrats have their hands clean of SBF… think again: He also looked to back candidates on that side of the aisle. You never know when you might need a
friend!\n\nThe hard crash\n\nAll of that activity, of course, came to a crashing end: The value of cryptocurrencies on FTX was not banked to much more than
speculation and the promise of higher value, and so when those valuations fell, they fell very, very hard.\n\nWhat we already knew was that SBF has been denied
bail; his name is mud (or much worse) among those who have lost money in Alameda or FTX; and those who might not have had positions in these still blame him for
other crypto woes because he set off a ripple effect.\n\n“He doesn’t come off good at all here. You can tell he had no experience in management or handling such
big undertakings.', 'publisher': 'TechCrunch', 'page': 'SBF, riding high on FTX, reportedly offered $5B to Trump to cancel President re-run (TechCrunch,
2023-10-02T13:11:07+00:00)', 'article': 'SBF, riding high on FTX, reportedly offered $5B to Trump to cancel President re-run'}, {'passage': 'SBF, riding high on
FTX, reportedly offered $5B to Trump to cancel President re-run (TechCrunch, 2023-10-02T13:11:07+00:00)\nYou can tell he had no experience in management or
handling such big undertakings. I think he was in way over his head,” Ayelet Noff, the CEO and founder of public relations firm Sliced Brand, which represents a
number of cryptocurrency businesses, told TechCrunch. “I think he does a disservice to the whole community. He pushed back crypto adoption by at least a year or
two. Each time one of these things happens it brings back the whole market. It’s two steps forward and one step back.”\n\nThe Michael Lewis interview is
infuriating. SBF was a scumbag, and now we need to hear about his “tragic fall” in the MSM because Sam told them he was an “effective altruist” and people like
Michael Lewis were so far up his ass they couldn’t tell truth from fiction.', 'publisher': 'TechCrunch', 'page': 'SBF, riding high on FTX, reportedly offered
$5B to Trump to cancel President re-run (TechCrunch, 2023-10-02T13:11:07+00:00)', 'article': 'SBF, riding high on FTX, reportedly offered $5B to Trump to cancel
President re-run'}, {'passage': "SBF, riding high on FTX, reportedly offered $5B to Trump to cancel President re-run (TechCrunch, 2023-10-02T13:11:07+00:00)\n—
Ryan Selkis 🪳 (@twobitidiot) October 2, 2023\n\nShameful that Michael Lewis is defending SBF. “They had a great real business. If no one had cast aspersions on
the business, if there hadn’t been a run on customers deposits, they’d still be making tons of money” – Michael Lewis That's not true. They are not a bank,
they're… pic.twitter.com/vRKBbgzITe — Dan Held (@danheld) October 2, 2023\n\nBut even so, Noff is defensive: She also believes crypto is no worse than the stock
market, which can also be manipulated and mishandled.\n\n“Even when you identify corruption, I don’t think that means we don’t believe in the system in
question,” she added.\n\nYet Lewis’s picture is a little less stark from the looks of the 60 Minutes episode.", 'publisher': 'TechCrunch', 'page': 'SBF, riding
high on FTX, reportedly offered $5B to Trump to cancel President re-run (TechCrunch, 2023-10-02T13:11:07+00:00)', 'article': 'SBF, riding high on FTX,
reportedly offered $5B to Trump to cancel President re-run'}, {'passage': 'SBF, riding high on FTX, reportedly offered $5B to Trump to cancel President re-run
(TechCrunch, 2023-10-02T13:11:07+00:00)\nYet Lewis’s picture is a little less stark from the looks of the 60 Minutes episode. In fact, I’d say it’s not that
clear at all if SBF is getting a kicking or a coat of polish in this interview when you consider Lewis’s celebration of SBF’s self-proclaimed (yet not quite
proven) idealism, and the “Bankman-Fried-shaped hold in the world” that exists.\n\nYou could say that confused message is very apt for cryptocurrency, where the
process by which it works is forever murky and often misrepresented, at least to most people.\n\nIf SBF didn’t intentionally aim to cynically defraud a bunch of
people, in the long run he became one of the many who disastrously failed to manage the risk inherent in crypto speculation. That’s ultimately what happened
with FTX, as financial discrepancies racked up between the exchange and its sister company Alameda Research (a trader itself on the platform).', 'publisher':
'TechCrunch', 'page': 'SBF, riding high on FTX, reportedly offered $5B to Trump to cancel President re-run (TechCrunch, 2023-10-02T13:11:07+00:00)', 'article':
'SBF, riding high on FTX, reportedly offered $5B to Trump to cancel President re-run'}, {'passage': 'SBF, riding high on FTX, reportedly offered $5B to Trump to
cancel President re-run (TechCrunch, 2023-10-02T13:11:07+00:00)\nThe failure is indisputable: The jury will now have to decide how criminal, and how far, his
intentions were around that.', 'publisher': 'TechCrunch', 'page': 'SBF, riding high on FTX, reportedly offered $5B to Trump to cancel President re-run
(TechCrunch, 2023-10-02T13:11:07+00:00)', 'article': 'SBF, riding high on FTX, reportedly offered $5B to Trump to cancel President re-run'}, {'passage': 'Is Sam
Bankman-Fried a bad ‘man’ or a good ‘boy’? Lawyers swap opening statements before first witnesses take the stand (Fortune, 2023-10-04T23:32:53+00:00)\nWho is
Sam Bankman-Fried, the former CEO of crypto exchange FTX? Is he a liar and a fraud? Or did he act in good faith, and like many a startup founder, fly too close
to the sun?\n\nThat answer ultimately lies with the jury, which was selected on Wednesday morning before lawyers for the government and then Bankman-Fried
swapped two very different stories of the former crypto mogul’s sudden rise and almost instantaneous fall.\n\nHere’s what happened on the second day of the
trial, which featured pointed allegations, a friend from MIT, and an audience replete with big names, including Bankman-Fried’s professorial parents and Damian
Williams, U.S. attorney for the Southern District of New York.\n\nA conscious criminal…\n\nThe prosecution’s account of the alleged crimes by Bankman-Fried, who
spent most of his day in court staring into a laptop while seated between his two attorneys, featured a study in contrasts.', 'publisher': 'Fortune', 'page':
'Is Sam Bankman-Fried a bad ‘man’ or a good ‘boy’? Lawyers swap opening statements before first witnesses take the stand (Fortune, 2023-10-04T23:32:53+00:00)',
'article': 'Is Sam Bankman-Fried a bad ‘man’ or a good ‘boy’? Lawyers swap opening statements before first witnesses take the stand'}, {'passage': 'Is Sam
Bankman-Fried a bad ‘man’ or a good ‘boy’? Lawyers swap opening statements before first witnesses take the stand (Fortune, 2023-10-04T23:32:53+00:00)\n“One year
ago, it looked like the defendant was on the top of the world,” began Thane Rehn, a prosecutor for the government, in his opening statement. The former CEO of
FTX oversaw a supposedly thriving crypto exchange, jetted between international locales, and hobnobbed with celebrities like Tom Brady and Larry David. He
repeatedly emphasized to customers that their money was safe and secure.\n\nBut “all of that, all of it, was built on lies,” Rehn declared to the jury. “Behind
the curtains, he was not what he appeared to be.” What followed was a roughly 30-minute story that repeatedly emphasized how Bankman-Fried allegedly stole
customer funds to facilitate his jet-setting lifestyle, donate millions to political candidates, and finance risky bets.\n\nThe key to his alleged scheme?
Alameda Research, a crypto hedge fund he also owned, argued Rehn.', 'publisher': 'Fortune', 'page': 'Is Sam Bankman-Fried a bad ‘man’ or a good ‘boy’? Lawyers
swap opening statements before first witnesses take the stand (Fortune, 2023-10-04T23:32:53+00:00)', 'article': 'Is Sam Bankman-Fried a bad ‘man’ or a good
‘boy’? Lawyers swap opening statements before first witnesses take the stand'}, {'passage': 'Is Sam Bankman-Fried a bad ‘man’ or a good ‘boy’? Lawyers swap
opening statements before first witnesses take the stand (Fortune, 2023-10-04T23:32:53+00:00)\nAlameda Research, a crypto hedge fund he also owned, argued Rehn.
Using Caroline Ellison, his on-again off-again girlfriend and the CEO of Alameda, as a front, Bankman-Fried had “secret access” to customer money—both cash and
crypto—the government claimed.\n\nMoreover, Bankman-Fried allegedly directed employees to conceal the flow of money into FTX’s coffers and forged financial
documents distributed to lenders and investors. “The defendant lied to the world,” Rehn alleged.\n\nAnd who was this defendant? Not a crypto “boy” genius, as so
many in the media (Fortune included) have written, but a “man” who “stole billions of dollars from thousands of victims,” Rehn said. “You will see the full
picture.”\n\n…or a well-meaning founder?', 'publisher': 'Fortune', 'page': 'Is Sam Bankman-Fried a bad ‘man’ or a good ‘boy’? Lawyers swap opening statements
before first witnesses take the stand (Fortune, 2023-10-04T23:32:53+00:00)', 'article': 'Is Sam Bankman-Fried a bad ‘man’ or a good ‘boy’? Lawyers swap opening
statements before first witnesses take the stand'}, {'passage': 'Is Sam Bankman-Fried a bad ‘man’ or a good ‘boy’? Lawyers swap opening statements before first
witnesses take the stand (Fortune, 2023-10-04T23:32:53+00:00)\n“You will see the full picture.”\n\n…or a well-meaning founder?\n\nBut Bankman-Fried, whose
cheekbones were more prominent after spending about seven weeks in a Brooklyn prison, was no liar, according to Mark Cohen, one of his lawyers. “Sam didn’t
defraud anyone,” he said early on in his opening statement.\n\nWhat the jury will see is a nerdy startup founder who acted in “good faith,” not the
prosecution’s “cartoon of a villain.” (Cohen repeatedly harped on Bankman-Fried’s allegedly good-faith actions throughout his address to the jury.)\n\nAlameda
was not subterranean or shady. It was a successful hedge fund, he said. FTX was no Ponzi scheme. It was a “very innovative, successful company.”', 'publisher':
'Fortune', 'page': 'Is Sam Bankman-Fried a bad ‘man’ or a good ‘boy’? Lawyers swap opening statements before first witnesses take the stand (Fortune,
2023-10-04T23:32:53+00:00)', 'article': 'Is Sam Bankman-Fried a bad ‘man’ or a good ‘boy’? Lawyers swap opening statements before first witnesses take the
stand'}, {'passage': 'Is Sam Bankman-Fried a bad ‘man’ or a good ‘boy’? Lawyers swap opening statements before first witnesses take the stand (Fortune,
2023-10-04T23:32:53+00:00)\nFTX was no Ponzi scheme. It was a “very innovative, successful company.” And the business practices between the two were reasonable,
he argued, claiming that Alameda acted legally as an FTX customer, payment processor, and market maker, or financial entity that acts as a trading partner for
customers looking to buy and sell cryptocurrencies.\n\nIn an analogy he employed throughout his opening statement, he said that “working at a startup is like
building a plane as you’re flying it” and that businesses sometimes fail. In fact, he specifically pointed the finger at Ellison, the former CEO of Alameda,
who, he said, did not adequately protect her hedge fund from the inherent risk of the crypto markets.\n\nWhen the walls came closing in and the aforementioned
plane approached the “eye of the storm,” Bankman-Fried did not act like someone who was guilty. Rather, he was willing to give up his personal wealth to make
customers whole, Cohen argued.', 'publisher': 'Fortune', 'page': 'Is Sam Bankman-Fried a bad ‘man’ or a good ‘boy’? Lawyers swap opening statements before first
witnesses take the stand (Fortune, 2023-10-04T23:32:53+00:00)', 'article': 'Is Sam Bankman-Fried a bad ‘man’ or a good ‘boy’? Lawyers swap opening statements
before first witnesses take the stand'}, {'passage': 'Is Sam Bankman-Fried a bad ‘man’ or a good ‘boy’? Lawyers swap opening statements before first witnesses
take the stand (Fortune, 2023-10-04T23:32:53+00:00)\nRather, he was willing to give up his personal wealth to make customers whole, Cohen argued.\n\n“In the
end, Sam started and built two billion-dollar businesses,” he concluded. “He didn’t steal any money.”\n\nA Frenchman who lives in London testifies in New
York\n\nAfter lawyers from both sides depicted two very different Bankman-Frieds, the prosecution called its first two witnesses to the stand—and they weren’t
blockbuster names or former lieutenants-turned-government-cooperators, like Ellison.\n\nThe first was a victim: Marc-Antoine Julliard, a Paris-born cocoa trader
who lives in London. In 2021, Julliard, who had coiffed hair and spoke with a strong French accent, decided to invest in crypto and landed on FTX as his
exchange of choice, where he traded cryptocurrencies like Bitcoin and Dogecoin.\n\nOn Nov. 8, in the crypto exchange’s final few days, he tried to pull out his
cash and crypto.', 'publisher': 'Fortune', 'page': 'Is Sam Bankman-Fried a bad ‘man’ or a good ‘boy’? Lawyers swap opening statements before first witnesses
take the stand (Fortune, 2023-10-04T23:32:53+00:00)', 'article': 'Is Sam Bankman-Fried a bad ‘man’ or a good ‘boy’? Lawyers swap opening statements before first
witnesses take the stand'}, {'passage': 'Is Sam Bankman-Fried a bad ‘man’ or a good ‘boy’? Lawyers swap opening statements before first witnesses take the stand
(Fortune, 2023-10-04T23:32:53+00:00)\nHow much? Almost $100,000, he said. And was he ever able to? “Never,” he told prosecutors.\n\nShortly afterward, as the
trial neared late afternoon, the government called Adam Yedidia to the stand. A fast-talking graduate of MIT, he and Bankman-Fried were close friends in
college, he said. And after Bankman-Fried left Jane Street, the high-frequency trading firm where the former billionaire got his start in finance after MIT, he
persuaded Yedidia to join him as a trader at Alameda and then as a developer at FTX.\n\nWhen Yedidia first took the stand, Danielle Sassoon, one of the lead
prosecutors, said that the college friend of Bankman-Fried had legal immunity during his testimony. Why did he make such a deal with the government, she
asked.\n\n“I was concerned that I had unwittingly written code that contributed to a crime,” he said.', 'publisher': 'Fortune', 'page': 'Is Sam Bankman-Fried a
bad ‘man’ or a good ‘boy’? Lawyers swap opening statements before first witnesses take the stand (Fortune, 2023-10-04T23:32:53+00:00)', 'article': 'Is Sam
Bankman-Fried a bad ‘man’ or a good ‘boy’? Lawyers swap opening statements before first witnesses take the stand'}, {'passage': 'Is Sam Bankman-Fried a bad
‘man’ or a good ‘boy’? Lawyers swap opening statements before first witnesses take the stand (Fortune, 2023-10-04T23:32:53+00:00)\n“I was concerned that I had
unwittingly written code that contributed to a crime,” he said.\n\nSoon, however, the clock neared 4:30 p.m., and court adjourned for the day. Yedidia will
continue his testimony on Thursday, followed by Matt Huang, a former partner at the high-powered venture capital firm Sequoia Capital, and then Gary Wang, a key
Bankman-Fried lieutenant and one of the government’s star witnesses.', 'publisher': 'Fortune', 'page': 'Is Sam Bankman-Fried a bad ‘man’ or a good ‘boy’?
Lawyers swap opening statements before first witnesses take the stand (Fortune, 2023-10-04T23:32:53+00:00)', 'article': 'Is Sam Bankman-Fried a bad ‘man’ or a
good ‘boy’? Lawyers swap opening statements before first witnesses take the stand'}, {'passage': 'The jury finally hears from Sam Bankman-Fried (The Verge,
2023-10-28T00:12:41+00:00)\nIt is honestly kind of incredible to watch a man torpedo his own credibility on direct testimony. We’re not even at the cross yet,
and the judge has already instructed him to answer the question he’s being asked by his own lawyer.\n\nThe jury is watching all of this intently.\n\nThe main
thing that’s been clear so far from Bankman-Fried’s testimony is that the man really loves the sound of his own voice. So far, the count of “Objection,
narrative!” to Bankman-Fried’s answers, followed by “Sustained” is at three.\n\nAlso, sometimes when Bankman-Fried says “we,” he only means himself.\n\nSam
Bankman-Fried, the founder of failed cryptocurrency exchange FTX, has been found guilty on seven counts including charges of wire fraud. FTX was a fraud “from
the start,” the Securities and Exchange Commission alleged — with a “multi-billion-dollar deficiency caused by his own misappropriation of customer funds.”
Follow along for all the latest news and regular updates from the trial.', 'publisher': 'The Verge', 'page': 'The jury finally hears from Sam Bankman-Fried (The
Verge, 2023-10-28T00:12:41+00:00)', 'article': 'The jury finally hears from Sam Bankman-Fried'}, {'passage': 'The jury finally hears from Sam Bankman-Fried (The
Verge, 2023-10-28T00:12:41+00:00)\nFollow along for all the latest news and regular updates from the trial.\n\nYesterday, during an evidentiary hearing,
Bankman-Fried was repeatedly scolded by Judge Lewis Kaplan for not answering prosecutor Danielle Sassoon’s questions on cross-examination. Today, Bankman-Fried
was scolded by Kaplan for not answering his counsel Mark Cohen’s questions on direct examination. Bankman-Fried has also occasionally interrupted Cohen with
“yes” and “yup.”\n\nWe spent a lot of the morning explaining vocabulary. I will spare you the full list, though I will say that explaining “Amazon Web Services”
and “database” was a bit too detail-oriented. Then Bankman-Fried tried to define “market manipulation.” After Bankman-Fried gave his definition, Kaplan told the
jury that he was the final authority on that, thanks.\n\nIn fairness to Bankman-Fried, he has been clearer and much easier to understand than he was at the
evidentiary hearing yesterday. There was a minimum of word salad today.', 'publisher': 'The Verge', 'page': 'The jury finally hears from Sam Bankman-Fried (The
Verge, 2023-10-28T00:12:41+00:00)', 'article': 'The jury finally hears from Sam Bankman-Fried'}, {'passage': 'The jury finally hears from Sam Bankman-Fried (The
Verge, 2023-10-28T00:12:41+00:00)\nThere was a minimum of word salad today. I don’t know if he was more relaxed, or he’d just been more rehearsed, but I will
certainly be watching to see if he suddenly becomes much less coherent when Sassoon gets him for the cross.\n\nHere is the story of FTX, from his point of
view.\n\nBankman-Fried, who informed us he’s “somewhat introverted, naturally,” gave us a rather prolonged tour of his pre-Alameda Research life, which I will
skip. In 2017, during a crypto bull run, he started his cryptocurrency trading firm. He knew “basically nothing” about cryptocurrency at the time, he explained,
but he wanted to do arbitrage on it anyway.\n\nAlameda Research was named for Alameda County in California, which was where its first office was set up. As for
its name, here’s what Bankman-Fried said on the stand:\n\nEffectively, we wanted to be under the radar at that point in time.', 'publisher': 'The Verge',
'page': 'The jury finally hears from Sam Bankman-Fried (The Verge, 2023-10-28T00:12:41+00:00)', 'article': 'The jury finally hears from Sam Bankman-Fried'},
{'passage': 'The jury finally hears from Sam Bankman-Fried (The Verge, 2023-10-28T00:12:41+00:00)\nI didn’t want to call it Sam’s Crypto Trading Firm or
anything like that. We — there are a lot of competitors and people who we didn’t particularly want to know what we were building out because they would race to
do it. “Research” was a sort of generic word, which filled out the company name. And that was — it was far better than the internal name that we had at that
point, which was Wireless Mouse.\n\nI would find this much more believable if I hadn’t already watched a video of Bankman-Fried explaining on a podcast that the
name made it easier to get a bank account. That happened during the first day of Gary Wang’s testimony. Bankman-Fried was there, too. You know who else was
there? The jury.\n\nAnyway, Bankman-Fried went on a hiring spree for Alameda. He rounded up his merry gang of alleged co-conspirators. First, Wang, to program
the computers. Then, Nishad Singh, about a month after founding Alameda Research.', 'publisher': 'The Verge', 'page': 'The jury finally hears from Sam
Bankman-Fried (The Verge, 2023-10-28T00:12:41+00:00)', 'article': 'The jury finally hears from Sam Bankman-Fried'}, {'passage': 'The jury finally hears from Sam
Bankman-Fried (The Verge, 2023-10-28T00:12:41+00:00)\nThen, Nishad Singh, about a month after founding Alameda Research. Finally, Caroline Ellison.\n\nNot
having a risk team, when you are any kind of financial anything, is certainly a choice\n\nThough Bankman-Fried was the CEO, and also the majority owner, he
wanted to be clear: he did not supervise Wang’s direct work. Anyway, after a bunch of wildly successful arbitrage — 50 percent to 100 percent annualized
returns, per his testimony — he decided to found a cryptocurrency exchange, FTX. He figured he’d fail; that there was only a 20 percent chance of success.
Bankman-Fried did not define what he expected the time period to be on this estimate, but arguably 20 percent was a much higher chance of success than FTX would
enjoy once Alameda dipped into the customer deposits.\n\nBy the way, because he was such a good guy, Bankman-Fried made a point of “periodically” handling
support tickets himself.', 'publisher': 'The Verge', 'page': 'The jury finally hears from Sam Bankman-Fried (The Verge, 2023-10-28T00:12:41+00:00)', 'article':
'The jury finally hears from Sam Bankman-Fried'}, {'passage': 'The jury finally hears from Sam Bankman-Fried (The Verge, 2023-10-28T00:12:41+00:00)\n“I worried
if I didn’t, I would lose touch with the actual concerns of the customers,” he testified. What he did not do was create a risk team, which he is now
characterizing as a “big mistake.”\n\nRisk is an inherent part of a futures exchange, which is even more like a casino than regular cryptocurrency. Not having a
risk team, when you are any kind of financial anything, is certainly a choice. It is especially a choice when you go around telling everyone your crypto
exchange is very good and safe.\n\nFTX’s big selling point was its “risk engine,” which was supposed to prevent big losses that would then be spread around all
the rest of the customers. But Bankman-Fried testified that in 2020, the “risk engine was effectively sagging under the weight” of the exchange’s rapid growth.
So its time to liquidation went up — it took minutes to determine which accounts needed to be liquidated.', 'publisher': 'The Verge', 'page': 'The jury finally
hears from Sam Bankman-Fried (The Verge, 2023-10-28T00:12:41+00:00)', 'article': 'The jury finally hears from Sam Bankman-Fried'}, {'passage': 'The jury finally
hears from Sam Bankman-Fried (The Verge, 2023-10-28T00:12:41+00:00)\nAs a result, at one point the risk engine got stuck in a catastrophic feedback loop that
would have created losses in the “trillions of dollars,” Bankman-Fried testified. As part of that feedback loop, Alameda teetered on the brink of liquidation,
which “would have disastrous consequences” for FTX.\n\n“At the time, I wasn’t entirely sure what was happening.”\n\nBecause of that experience, Bankman-Fried
suggested an “alert” or “delay” that would keep Alameda from being liquidated by a bug. This is the supposed origin story of “allow_negative,” which
Bankman-Fried says was the eventual result of that conversation, and that he says he didn’t know about until very recently.\n\nThere is a problem with this
story. “Allow_negative” was coded and switched on in 2019. I saw the code in court, and so did Bankman-Fried, who was also there for the testimony. Perhaps you
are wondering, was the jury also there? Reader, it was.', 'publisher': 'The Verge', 'page': 'The jury finally hears from Sam Bankman-Fried (The Verge,
2023-10-28T00:12:41+00:00)', 'article': 'The jury finally hears from Sam Bankman-Fried'}, {'passage': 'The jury finally hears from Sam Bankman-Fried (The Verge,
2023-10-28T00:12:41+00:00)\nPerhaps you are wondering, was the jury also there? Reader, it was.\n\nBankman-Fried denied he knew about the effectively infinite
line of credit Alameda Research received from FTX. This argument was peculiar; essentially my take-away was that the CEO of a financial company simply didn’t
pay attention to finances.\n\nFTX couldn’t get bank accounts right away. Bankman-Fried anticipated it would take a year or two. Rather than wait, he decided to
use Alameda as the “payment provider” for bank transfers. “My understanding at the time was that there were teams managing the process,” he said. “At the time,
I wasn’t entirely sure what was happening.”\n\nWell, sure, understandable! He’s an introvert!\n\nBankman-Fried definitely did not know that Singh, his employee,
had backdated interest payments to get FTX “over the line” to $1 billion\n\nIn 2021, FTX was growing to millions of users, with $1 billion of revenue.',
'publisher': 'The Verge', 'page': 'The jury finally hears from Sam Bankman-Fried (The Verge, 2023-10-28T00:12:41+00:00)', 'article': 'The jury finally hears
from Sam Bankman-Fried'}, {'passage': 'The jury finally hears from Sam Bankman-Fried (The Verge, 2023-10-28T00:12:41+00:00)\nBankman-Fried said he worked 12 to
22 hours a day, and took one day off every couple of months. Because FTX had grown so much, he could no longer run both companies, he said. Bankman-Fried handed
the company off to Caroline Ellison and Sam Trabucco, who immediately after being named co-CEO promptly drifted away to early retirement. (Quiet quitting king!)
Bankman-Fried did remain involved in hedging and risk at Alameda, though.\n\nAbout that $1 billion of revenue in 2021: Bankman-Fried definitely did not know
that Singh, his employee, had backdated interest payments to get FTX “over the line” to $1 billion. See, he’d just asked his employees to check and see if there
was any source of funds that was missing to get to $1 billion. This testimony was especially rambling.\n\nOh, also that MobileCoin loss? The one Wang said
Alameda took to keep off FTX’s balance sheet?', 'publisher': 'The Verge', 'page': 'The jury finally hears from Sam Bankman-Fried (The Verge,
2023-10-28T00:12:41+00:00)', 'article': 'The jury finally hears from Sam Bankman-Fried'}, {'passage': 'The jury finally hears from Sam Bankman-Fried (The Verge,
2023-10-28T00:12:41+00:00)\nThe one Wang said Alameda took to keep off FTX’s balance sheet? Yeah, so it was a totally innocent thing where what actually
happened was that Bankman-Fried thought it was appropriate that Alameda take the position as a backstop liquidity provider, that’s all.\n\nIn June 2022,
Bankman-Fried heard about the account called “fiat@ftx” tracking how much money Alameda owed to FTX, he testified. He did not know what it was and did not
bother to find out. He was busy! That was when Bankman-Fried directed Ellison to repay Alameda’s lenders, because he thought Alameda was good for it. He also
gave BlockFi and Voyager, two crypto lenders, some capital infusions for good measure.\n\nHe was “very surprised!”\n\nRemember that testimony Adam Yedidia gave
about a conversation with Bankman-Fried in August 2022 about the enormous amount of money Alameda owed FTX? Well, Bankman-Fried remembers it differently.',
'publisher': 'The Verge', 'page': 'The jury finally hears from Sam Bankman-Fried (The Verge, 2023-10-28T00:12:41+00:00)', 'article': 'The jury finally hears
from Sam Bankman-Fried'}, {'passage': 'The jury finally hears from Sam Bankman-Fried (The Verge, 2023-10-28T00:12:41+00:00)\nWell, Bankman-Fried remembers it
differently. See, Yedidia was just asking about Alameda’s risk profile, and Bankman-Fried wasn’t talking about insolvency at all.\n\nAlso, when Singh and
Bankman-Fried had the dramatic balcony conversation at their penthouse, it was just that Singh thought Alameda’s liabilities had gotten too high, and FTX was
spending too much money on marketing. But Bankman-Fried still thought that Alameda had more assets than liabilities, so it was all fine, and besides, if Singh
thought he was going to be better at marketing, he could take it over. It didn’t have anything to do with the money Alameda owed FTX at all.\n\nOf course not!
Bankman-Fried didn’t learn about the $8 billion liability associated with Alameda until October 2022, he said. And he learned it all by himself, by looking at a
computer database. When he found it, he was “very surprised!”', 'publisher': 'The Verge', 'page': 'The jury finally hears from Sam Bankman-Fried (The Verge,
2023-10-28T00:12:41+00:00)', 'article': 'The jury finally hears from Sam Bankman-Fried'}, {'passage': 'The jury finally hears from Sam Bankman-Fried (The Verge,
2023-10-28T00:12:41+00:00)\nWhen he found it, he was “very surprised!”\n\nBesides the two obvious lies Bankman-Fried told on the stand — about Alameda
Research’s name and about “allow_negative” — I have been struck by how little he seems to know about his own companies. Apparently, Singh, Wang, and Ellison
were out there just doing whatever their little hearts desired. Because Bankman-Fried was a CEO, but definitely not the kind that pays any attention to money at
his crypto trading firm and futures exchange.', 'publisher': 'The Verge', 'page': 'The jury finally hears from Sam Bankman-Fried (The Verge,
2023-10-28T00:12:41+00:00)', 'article': 'The jury finally hears from Sam Bankman-Fried'}, {'passage': 'Sam Bankman-Fried didn’t ask where the $8 billion went
(The Verge, 2023-10-31T19:52:07+00:00)\nLet’s say I am the owner of a hedge fund, and one fine June day, my employees come to me and say, “Hey, Liz, we have an
accounting problem. We are missing several billion dollars.” How would I react?\n\nI have been wondering this since Danielle Sassoon walked Sam Bankman-Fried
through his reaction to the FTX software bug fixed by Adam Yedidia. In my case, there would probably be shouting? Like, a lot of shouting. I would also probably
have my assistant figure out which law enforcement agency to call immediately. Misplacing $900 million is a five-alarm fire even for Citibank; misplacing
several billion is kicking over a lantern in Chicago in 1871.\n\nObviously, this is not how Bankman-Fried reacted to the software bug that overestimated the
amount Alameda owed to FTX by about $8 billion. Nor is it how he reacted to finding out that even after fixing the bug, Alameda still owed FTX about $8
billion.', 'publisher': 'The Verge', 'page': 'Sam Bankman-Fried didn’t ask where the $8 billion went (The Verge, 2023-10-31T19:52:07+00:00)', 'article': 'Sam
Bankman-Fried didn’t ask where the $8 billion went'}, {'passage': 'Sam Bankman-Fried didn’t ask where the $8 billion went (The Verge,
2023-10-31T19:52:07+00:00)\nInstead, Bankman-Fried directed alleged co-conspirator Caroline Ellison to repay third-party loans and went on making
investments.\n\nBankman-Fried said that he found out about the fiat@ftx account — that’s the one that tracked how much Alameda owed FTX — in June 2022 while his
senior staff was focused on fixing the software bug. He did not, however, discover what the account was for until October. I know this sounds unbelievable, but
this is his actual testimony.\n\nHis employees told him “they were busy and I should stop asking questions because it was distracting.”\n\nSimple questions
nailed Bankman-Fried to the wall. Had he given his employees at Alameda the direction not to spend FTX customer deposits? Had he put in any policies to prevent
Alameda employees from spending FTX customer money? Did he put in place any measures at Alameda to protect the FTX customer money? No, no, and no. Oh, but he
was testifying to Congress about keeping customer funds safe.', 'publisher': 'The Verge', 'page': 'Sam Bankman-Fried didn’t ask where the $8 billion went (The
Verge, 2023-10-31T19:52:07+00:00)', 'article': 'Sam Bankman-Fried didn’t ask where the $8 billion went'}, {'passage': 'Sam Bankman-Fried didn’t ask where the $8
billion went (The Verge, 2023-10-31T19:52:07+00:00)\nOh, but he was testifying to Congress about keeping customer funds safe.\n\nSam Bankman-Fried, the founder
of failed cryptocurrency exchange FTX, is on trial for seven counts of wire fraud and conspiracy. FTX was a fraud “from the start,” the Securities and Exchange
Commission alleges — with a “multi-billion-dollar deficiency caused by his own misappropriation of customer funds.” Follow along for all the latest news and
regular updates from the trial.\n\nWho was making decisions to spend $8 billion of customer funds? Bankman-Fried couldn’t recall knowing anything about it. Were
there rules or requirements for how money borrowed from FTX would be returned? Were there rules for risk management? “I was concerned with overall risk
management,” Bankman-Fried said.\n\nBut it was the testimony about June 2022 that resonated the most to me. Didn’t Bankman-Fried ask what “fiat@ftx” was? He
did.', 'publisher': 'The Verge', 'page': 'Sam Bankman-Fried didn’t ask where the $8 billion went (The Verge, 2023-10-31T19:52:07+00:00)', 'article': 'Sam
Bankman-Fried didn’t ask where the $8 billion went'}, {'passage': 'Sam Bankman-Fried didn’t ask where the $8 billion went (The Verge,
2023-10-31T19:52:07+00:00)\nHe did. But — I did hear these words uttered aloud in a court of law this morning, I am not creative enough to make this kind of
thing up — his employees told him “they were busy and I should stop asking questions because it was distracting.”\n\nYedidia — Bankman-Fried’s college friend,
Bahamas roommate, and employee at FTX — had testified that he’d asked Bankman-Fried about the $8 billion hole on a padel tennis court in their luxury complex in
June or July. Today, in testimony, Bankman-Fried seemed to be trying to deny that conversation had ever taken place. It was not until Judge Lewis Kaplan
intervened to ask if Bankman-Fried had ever been told by Yedidia about that money, in words or in substance, that Bankman-Fried admitted he’d been told.',
'publisher': 'The Verge', 'page': 'Sam Bankman-Fried didn’t ask where the $8 billion went (The Verge, 2023-10-31T19:52:07+00:00)', 'article': 'Sam Bankman-Fried
didn’t ask where the $8 billion went'}, {'passage': 'Sam Bankman-Fried didn’t ask where the $8 billion went (The Verge, 2023-10-31T19:52:07+00:00)\nI wouldn’t
want to watch my child be vivisected, either. The jurors, however, watched the operation attentively. I suppose for most of us, $8 billion has a way of focusing
the mind.\n\nLook, uttering phrases like “hole isn’t really the word I would use” and responding to a question by saying you wanted “a few more qualifiers and
scoping on it” do not, as a general rule, bode well for your believability. Yes, this will win certain kinds of nerd arguments. But this is a courtroom, and I
have come to believe that if you know the meaning of the word “epistemology,” you absolutely should not testify in your own defense.\n\n“His name is Ryan Salame
:p”\n\nBankman-Fried, like Theranos’ Elizabeth Holmes before him, got on the stand to speak directly to his state of mind in a way no other witness can.',
'publisher': 'The Verge', 'page': 'Sam Bankman-Fried didn’t ask where the $8 billion went (The Verge, 2023-10-31T19:52:07+00:00)', 'article': 'Sam Bankman-Fried
didn’t ask where the $8 billion went'}, {'passage': 'Sam Bankman-Fried didn’t ask where the $8 billion went (The Verge, 2023-10-31T19:52:07+00:00)\nBut telling
your side of the story opens you to a great many questions you might not want to answer on cross-examination. I have watched a lot of crosses. This was the
nastiest I think I’ve ever seen.\n\nThe charges in this case hinge on conspiracy and intentional deceit. Just losing $8 billion is not a crime, though it is
very embarrassing. (Even losing $900 million is very embarrassing because people will make jokes about it forever!) But if Bankman-Fried lied to customers and
lenders about what he was doing and how safe FTX was, that’s a crime.\n\nEstablishing that would have been enough, but Sassoon also managed to get in some
flourishes about Bankman-Fried’s relationships to Bahamian power. For instance, she asked him if he’d made comments about paying off the Bahamian national debt.
(You will be shocked, shocked to discover he didn’t recall.)', 'publisher': 'The Verge', 'page': 'Sam Bankman-Fried didn’t ask where the $8 billion went (The
Verge, 2023-10-31T19:52:07+00:00)', 'article': 'Sam Bankman-Fried didn’t ask where the $8 billion went'}, {'passage': 'Sam Bankman-Fried didn’t ask where the $8
billion went (The Verge, 2023-10-31T19:52:07+00:00)\n(You will be shocked, shocked to discover he didn’t recall.) We then saw an internal FTX group chat called
“Project Chinchilla Chatter” in which another member asked who in the Bahamian government they needed to talk to for the project. “His name is Ryan Salame :p,”
Bankman-Fried replied. Salame was one of FTX’s executives.\n\nIn another part of the chat, Bankman-Fried noted that the Bahamian prime minister was at the FTX
arena in FTX seats with his wife.\n\nWe then saw a November 9th email from Bankman-Fried to Ryan Pinder, attorney general of the Bahamas, that said “we are
deeply grateful” for what the Bahamas had done for FTX. As a token of that gratitude, Bankman-Fried wrote:\n\nWe would be more than happy to open up withdrawals
for all Bahamian customers on FTX, so that they can, tomorrow, fully withdraw all of their assets, making them fully whole.', 'publisher': 'The Verge', 'page':
'Sam Bankman-Fried didn’t ask where the $8 billion went (The Verge, 2023-10-31T19:52:07+00:00)', 'article': 'Sam Bankman-Fried didn’t ask where the $8 billion
went'}, {'passage': 'Sam Bankman-Fried didn’t ask where the $8 billion went (The Verge, 2023-10-31T19:52:07+00:00)\nIt’s your call whether you want us to do
this, but we are more than happy to and would consider it the very least of our duty to the country, and could open it up immediately if you reply saying you
want us to. If we don’t hear back from you, we are going to go ahead and do it tomorrow.\n\nBankman-Fried did indeed open withdrawals for Bahamian customers.
The upshot of this testimony seemed to be that Bankman-Fried had a cozy, perhaps even inappropriately cozy, relationship with the Bahamian government — which
isn’t what he’s on trial for but probably doesn’t make him look any better to a jury.\n\nSassoon successfully established yesterday that Bankman-Fried has a
long history of dishonesty. Today, through a set of questions about what Bankman-Fried did and didn’t do, she established that the story he told on direct
examination was absurd.', 'publisher': 'The Verge', 'page': 'Sam Bankman-Fried didn’t ask where the $8 billion went (The Verge, 2023-10-31T19:52:07+00:00)',
'article': 'Sam Bankman-Fried didn’t ask where the $8 billion went'}, {'passage': 'Sam Bankman-Fried didn’t ask where the $8 billion went (The Verge,
2023-10-31T19:52:07+00:00)\nAfter a brief redirect examination by Bankman-Fried’s own lawyers, which was resplendent with word salad, Bankman-Fried stepped
down, and the defense rested their case.', 'publisher': 'The Verge', 'page': 'Sam Bankman-Fried didn’t ask where the $8 billion went (The Verge,
2023-10-31T19:52:07+00:00)', 'article': 'Sam Bankman-Fried didn’t ask where the $8 billion went'}, {'passage': 'Sam Bankman-Fried was a terrible boyfriend (The
Verge, 2023-10-10T23:50:21+00:00)\nI’ve got some shitty ex-boyfriends, but none of them made me the CEO of their sin-eater hedge fund while refusing to give me
equity and bragging about how there was a 5 percent chance they’d become the president of the United States, you know? Absolutely counting my blessings after
Caroline Ellison’s first day on the stand. I wonder how many of the nine women on the jury are doing the same.\n\nEllison was the head of Alameda Research, the
aforementioned hedge fund, during the implosion of it and FTX. She’s already pleaded guilty to criminal charges stemming from one of the worst romantic
relationships I’ve ever heard of, and her testimony was widely anticipated before the trial. Today, that took the form of discussing a damning spreadsheet — one
she prepared for her ex and boss Sam Bankman-Fried, now the defendant in a criminal fraud trial.\n\nSam Bankman-Fried, the founder of failed cryptocurrency
exchange FTX, is on trial for seven counts of wire fraud and conspiracy.', 'publisher': 'The Verge', 'page': 'Sam Bankman-Fried was a terrible boyfriend (The
Verge, 2023-10-10T23:50:21+00:00)', 'article': 'Sam Bankman-Fried was a terrible boyfriend'}, {'passage': 'Sam Bankman-Fried was a terrible boyfriend (The
Verge, 2023-10-10T23:50:21+00:00)\nFTX was a fraud “from the start,” the Securities and Exchange Commission alleges — with a “multi-billion-dollar deficiency
caused by his own misappropriation of customer funds.” Follow along for all the latest news and regular updates from the trial.\n\nThe day started off
promisingly for the defense as it cross-examined Gary Wang, the chief technology officer of FTX and co-owner of both FTX and Alameda. Christian Everdell, one of
Bankman-Fried’s defense attorneys, couldn’t undo the damage of last week’s code review. But he managed to shake the rust off long enough to make Wang sound less
reliable, drowning the jury in confusing technicalities.\n\nLast week, Wang testified that Alameda got access to a special credit line and an option to take its
balance into the negative without triggering liquidation — something he alleged other accounts at FTX didn’t get. Everdell tried to undermine this claim by
talking about the spot margin program, which let users lend each other assets for margin trading.', 'publisher': 'The Verge', 'page': 'Sam Bankman-Fried was a
terrible boyfriend (The Verge, 2023-10-10T23:50:21+00:00)', 'article': 'Sam Bankman-Fried was a terrible boyfriend'}, {'passage': 'Sam Bankman-Fried was a
terrible boyfriend (The Verge, 2023-10-10T23:50:21+00:00)\nIn those cases, it was possible to have a negative balance in a specific coin. It was not, however,
possible for those accounts to avoid liquidation, as Wang testified Alameda could do — or to have an overall negative balance. But I’m betting the defense is
hoping the jurors will throw up their hands in confusion thinking about this.\n\n“We ultimately took about $14 billion, some of which we were not able to pay
back.”\n\nWang didn’t exactly help himself out, either. Apparently, what Wang said in court contradicted something he’d said in earlier interviews with the
government about market making. I say “apparently” because Everdell was probably giving him his previous testimony to refresh his recollection, but Wang was
insisting he didn’t remember. In any event, whatever Wang was shown wasn’t submitted as evidence or shown to the court. I got the gist, though, and I bet the
jury did, too — probably the strongest work the defense has done so far.', 'publisher': 'The Verge', 'page': 'Sam Bankman-Fried was a terrible boyfriend (The
Verge, 2023-10-10T23:50:21+00:00)', 'article': 'Sam Bankman-Fried was a terrible boyfriend'}, {'passage': 'Sam Bankman-Fried was a terrible boyfriend (The
Verge, 2023-10-10T23:50:21+00:00)\nBut by the end of the day, that all seemed like a sideshow. Bankman-Fried had been vibrating slightly during Wang’s
testimony. During Ellison’s testimony, his bouncing became more noticeable.\n\nEllison was hunched in on herself as she walked into the courtroom, wearing a
dusty rose dress with a gray blazer over it, looking less like an executive than like a girl who’s borrowed her boyfriend’s coat because she’s cold. When the
prosecution asked her to identify Bankman-Fried, she had trouble finding him and gazed around the courtroom for more than 20 seconds — apparently he was
incognito with his new haircut. After she did spot him, she was asked to identify him, which she did by identifying him as wearing a suit. This got chuckles
from the rest of the defense table, also all in suits.\n\nShe listed off the crimes she’d already pleaded guilty to and added that Bankman-Fried “directed me to
commit these crimes,” Ellison said.', 'publisher': 'The Verge', 'page': 'Sam Bankman-Fried was a terrible boyfriend (The Verge, 2023-10-10T23:50:21+00:00)',
'article': 'Sam Bankman-Fried was a terrible boyfriend'}, {'passage': 'Sam Bankman-Fried was a terrible boyfriend (The Verge,
2023-10-10T23:50:21+00:00)\n(Fraud, conspiracy to commit fraud, and money laundering, in case you were wondering.) “We ultimately took about $14 billion, some
of which we were not able to pay back.” She tilted her head down to answer the questions, then lifted her head when she’d finished her answer.\n\nBankman-Fried
called this being “risk-neutral,” which seems like a fancy way of saying he was a gambling addict\n\nIn Ellison’s telling, Alameda was troubled from her
earliest time there in 2018. “Shortly after I started, I learned the company was in worse shape than I realized,” Ellison said. Alameda had initially been
funded with loans “from acquaintances,” she said, and those loans were recalled a few weeks after she arrived. (There was a staff revolt within Alameda Research
over lost millions and general financial chaos, according to Michael Lewis’ Going Infinite.) Ellison asked Bankman-Fried why he hadn’t shared the company’s
shaky circumstances in the job offer. “He hadn’t known how to tell me,” she said.', 'publisher': 'The Verge', 'page': 'Sam Bankman-Fried was a terrible
boyfriend (The Verge, 2023-10-10T23:50:21+00:00)', 'article': 'Sam Bankman-Fried was a terrible boyfriend'}, {'passage': 'Sam Bankman-Fried was a terrible
boyfriend (The Verge, 2023-10-10T23:50:21+00:00)\n“He hadn’t known how to tell me,” she said.\n\nEllison was also, of course, in a more personal relationship
with Bankman-Fried. (A juror who’d been asleep for a discussion of the FTT token woke up when she started discussing it.) The two started sleeping together in
the fall of 2018, on and off. At the time, she was a trader and Bankman-Fried was the CEO. They didn’t date until later — twice. Their first relationship
stretched from the summer of 2020 through the summer of 2021; they agreed to keep it secret. (Some people found out, as they usually do.) The second time, from
the fall of 2021 until the spring of 2022, they lived together.\n\nThat gave Ellison an unusual view of his character. “He was very ambitious,” she said.',
'publisher': 'The Verge', 'page': 'Sam Bankman-Fried was a terrible boyfriend (The Verge, 2023-10-10T23:50:21+00:00)', 'article': 'Sam Bankman-Fried was a
terrible boyfriend'}, {'passage': 'Sam Bankman-Fried was a terrible boyfriend (The Verge, 2023-10-10T23:50:21+00:00)\nThat gave Ellison an unusual view of his
character. “He was very ambitious,” she said. Besides telling her about his presidential chances, he also told her that if there was a coin flip where tails
destroyed the world and heads made the world twice as good, he’d flip the coin. He called this being “risk-neutral,” which seems like a fancy way of saying he
was a gambling addict.\n\nShe was named co-CEO of Alameda with Sam Trabucco in 2021, while she and Bankman-Fried were broken up, and CEO in 2022. The goal,
Bankman-Fried told her, was to “optically” separate Alameda Research and FTX. “The whole time we were dating, he was my boss at work,” she said. They broke up
because she wanted more from the relationship; Bankman-Fried was distant and not paying enough attention to her.', 'publisher': 'The Verge', 'page': 'Sam
Bankman-Fried was a terrible boyfriend (The Verge, 2023-10-10T23:50:21+00:00)', 'article': 'Sam Bankman-Fried was a terrible boyfriend'}, {'passage': 'Sam
Bankman-Fried was a terrible boyfriend (The Verge, 2023-10-10T23:50:21+00:00)\nUsing the effectively unlimited funds “allowed us to make profitable trades we
couldn’t have made otherwise”\n\nBankman-Fried didn’t grant Ellison equity, even though she asked; he told her it would be too complicated. Instead, she got a
$200,000 salary, even as CEO, and bonuses twice a year, which ranged from $100,000 to $20 million.\n\nInitially, Alameda and FTX were “very integrated,” Ellison
said. They were run by the same team, from the same office. And when Alameda was scrounging for funds, Bankman-Fried told Ellison that FTX would be a good
source of capital. The $65 billion line of credit Alameda Research had meant that it did not have to post collateral. There was no contract and no written
terms, she testified. It also wasn’t visible to FTX’s auditors — she’d raised the question with Bankman-Fried, and he told her not to worry about it.',
'publisher': 'The Verge', 'page': 'Sam Bankman-Fried was a terrible boyfriend (The Verge, 2023-10-10T23:50:21+00:00)', 'article': 'Sam Bankman-Fried was a
terrible boyfriend'}, {'passage': 'Sam Bankman-Fried was a terrible boyfriend (The Verge, 2023-10-10T23:50:21+00:00)\nAlameda’s credit line — which was taken in
increments of $100,000 to $10 million at a time — was used for trading. Using the effectively unlimited funds “allowed us to make profitable trades we couldn’t
have made otherwise,” Ellison testified.\n\nCustomer funds were also used when Bankman-Fried bought back FTX shares from Binance, an early investor, in the
summer of 2021. Bankman-Fried told Ellison it was “really important,” otherwise “Binance would do things to mess with FTX.” Ellison says she told him Alameda
didn’t have the money. So Bankman-Fried took $1 billion of FTX customer funds to buy out Binance, the first time Ellison recalled an amount that large. It was
Bankman-Fried’s decision, she said, as he was the CEO of FTX.\n\nFTT was one of several “Sam coins”\n\nThere was also the FTT token, which was created by
Bankman-Fried and Wang.', 'publisher': 'The Verge', 'page': 'Sam Bankman-Fried was a terrible boyfriend (The Verge, 2023-10-10T23:50:21+00:00)', 'article': 'Sam
Bankman-Fried was a terrible boyfriend'}, {'passage': 'Sam Bankman-Fried was a terrible boyfriend (The Verge, 2023-10-10T23:50:21+00:00)\nAlameda got its war
chest — 60 percent to 70 percent of the initial supply — for free, while seed investors got FTT at 10 cents a coin, and FTT first listed at $1 a coin.
Bankman-Fried felt that $1 per coin was psychologically important, Ellison said, and he directed her to buy up FTT using Alameda if its price fell below a
dollar.\n\nFTT was one of several “Sam coins,” a nickname for tokens that Bankman-Fried was heavily involved in and owned a lot of, either personally or through
Alameda. Those coins were almost certainly worth less than the value displayed on the balance sheet because trying to sell them all at once would crater the
prices. Bankman-Fried directed her to put those coins on the balance sheets Alameda showed to lenders, even though she felt it was “somewhat
misleading.”\n\nAlameda was also getting loans from outside lenders, such as Genesis, because when FTX started, there weren’t a lot of customer funds to borrow,
Ellison testified.', 'publisher': 'The Verge', 'page': 'Sam Bankman-Fried was a terrible boyfriend (The Verge, 2023-10-10T23:50:21+00:00)', 'article': 'Sam
Bankman-Fried was a terrible boyfriend'}, {'passage': 'Sam Bankman-Fried was a terrible boyfriend (The Verge, 2023-10-10T23:50:21+00:00)\nThat was the basis of
the worst of her testimony — and the spreadsheet from hell.\n\nEllison said she’d prepared the spreadsheet at Bankman-Fried’s request in the fall of 2021 and
shared it with him. The point was risk analysis around paying back Alameda’s loans if they were abruptly recalled by Genesis, their lender. Bankman-Fried wanted
to use $3 billion for venture investments, so Ellison was ballparking what that would do to Alameda’s risk. In the as-is scenario, if things went south, she
figured there was a 30 percent chance they wouldn’t be able to meet the loan recalls. If Bankman-Fried used $3 billion to make investments, there was a 100
percent chance they couldn’t meet the recalls, even with FTX customer funds.\n\nEchoing Bankman-Fried and Ellison’s romantic relationship, FTX was keeping cozy
private ties with Alameda yet publicly holding it at arm’s length\n\nThe problem here wasn’t really the math, which seemed pretty arbitrary.', 'publisher': 'The
Verge', 'page': 'Sam Bankman-Fried was a terrible boyfriend (The Verge, 2023-10-10T23:50:21+00:00)', 'article': 'Sam Bankman-Fried was a terrible boyfriend'},
{'passage': 'Sam Bankman-Fried was a terrible boyfriend (The Verge, 2023-10-10T23:50:21+00:00)\nIt was that Ellison’s calculations assumed Alameda could borrow
$1.8 billion in normal dollars and $1.5 billion in crypto from FTX. The spreadsheet makes this clear with a row labeled “FTX borrows,” which Ellison said were
customer funds.\n\nMeanwhile, echoing Bankman-Fried and Ellison’s romantic relationship, FTX was keeping cozy private ties with Alameda yet publicly holding it
at arm’s length. On January 14th, 2022, Bankman-Fried tweeted, “We’re launching a $2b venture fund: FTX Ventures!” Those funds came from Alameda, Ellison
testified. But Bankman-Fried didn’t want to go public with the source of the funds. He said he thought Alameda’s brand was less good, and he didn’t want his
name associated with it. Alameda also bought Robinhood shares for Bankman-Fried, who moved them to a vehicle called “Emergent Fidelity Technologies” to avoid
association with Alameda.', 'publisher': 'The Verge', 'page': 'Sam Bankman-Fried was a terrible boyfriend (The Verge, 2023-10-10T23:50:21+00:00)', 'article':
'Sam Bankman-Fried was a terrible boyfriend'}, {'passage': 'Sam Bankman-Fried was a terrible boyfriend (The Verge, 2023-10-10T23:50:21+00:00)\nThe day ended
with a document that had been shared between Ellison and Bankman-Fried — with his comments appearing in bubbles along the main text. Ellison wrote she was
worried about “both actual leverage and presenting on our balance sheet.” Bankman-Fried responded with a note: “Yup, and could also get worse.”', 'publisher':
'The Verge', 'page': 'Sam Bankman-Fried was a terrible boyfriend (The Verge, 2023-10-10T23:50:21+00:00)', 'article': 'Sam Bankman-Fried was a terrible
boyfriend'}, {'passage': 'In the end, the FTX trial was about the friends screwed along the way (The Verge, 2023-10-26T12:00:00+00:00)\nSure, Sam
Bankman-Fried’s trial has revealed that FTX was basically a seething mass of scams. But it has also revealed how Bankman-Fried blew up the lives of the people
who are closest to him.\n\nOf course everyone who had money trapped on the FTX exchange when it collapsed — as the result of Bankman-Fried’s trading firm
Alameda Research purloining its customers’ funds — had their lives altered. Even if they are made whole by the bankruptcy proceedings, that’s still money that
they should have had access to at any time they wanted it. That may mean houses that weren’t purchased, medical procedures that were left undone, and life
savings lost.\n\nThere is a simultaneous drama of ruined lives playing out in the courtroom\n\nBankman-Fried is charged with seven counts: two counts of wire
fraud and five conspiracy charges. As we near the end of the prosecution’s case, I have been struck by how bizarre it all is.', 'publisher': 'The Verge',
'page': 'In the end, the FTX trial was about the friends screwed along the way (The Verge, 2023-10-26T12:00:00+00:00)', 'article': 'In the end, the FTX trial
was about the friends screwed along the way'}, {'passage': 'In the end, the FTX trial was about the friends screwed along the way (The Verge,
2023-10-26T12:00:00+00:00)\nAll the co-conspirators seem oddly younger than their years, as though they got stuck in the “math camp forever” environment
Bankman-Fried created for them and simply failed to grow up.\n\nThe prosecution has made a strong case that Bankman-Fried presided over the various forms of
fraud that played out at Alameda Research and FTX, orchestrating some of it and engaging in some of it directly. In the process of making this case, there is a
simultaneous drama of ruined lives playing out in the courtroom.\n\nEveryone who loved Bankman-Fried most in the world is now caught up in the case. We
have:\n\nAdam Yedidia, Bankman-Fried’s college roommate, who testified with immunity because he was concerned he may have unknowingly worked on code that
contributed to the fraud. “I love you Sam,” he texted at one point. “I’m not going anywhere.” That was before he realized FTX had stolen customer funds — at
which point, he resigned.', 'publisher': 'The Verge', 'page': 'In the end, the FTX trial was about the friends screwed along the way (The Verge,
2023-10-26T12:00:00+00:00)', 'article': 'In the end, the FTX trial was about the friends screwed along the way'}, {'passage': 'In the end, the FTX trial was
about the friends screwed along the way (The Verge, 2023-10-26T12:00:00+00:00)\nThat was before he realized FTX had stolen customer funds — at which point, he
resigned.\n\nGary Wang, the co-founder of FTX and Alameda Research and another college roommate. He took the stand after Yedidia and sang like a canary about
the secret privileges he’d created for Alameda in the FTX code.\n\nNishad Singh, a longtime family friend, who copped to what sounded an awful lot like classic
embezzlement. He said he’d backdated transactions on FTX to help create doctored balance sheets to show to investors. He pleaded guilty to six charges,
including wire fraud, money laundering, and conspiracy counts around his political donations.\n\nCaroline Ellison, Bankman-Fried’s ex-girlfriend. She told us
about the seven phony balance sheets she’d created. She was clear: Alameda took customer funds; Bankman-Fried knew about it and even invested them against her
advice.\n\nJoe Bankman, Bankman-Fried’s father.', 'publisher': 'The Verge', 'page': 'In the end, the FTX trial was about the friends screwed along the way (The
Verge, 2023-10-26T12:00:00+00:00)', 'article': 'In the end, the FTX trial was about the friends screwed along the way'}, {'passage': 'In the end, the FTX trial
was about the friends screwed along the way (The Verge, 2023-10-26T12:00:00+00:00)\nJoe Bankman, Bankman-Fried’s father. He is among the people included in the
“small group chat,” the Signal group where the FTX inner circle strategized during the exchange’s last days.\n\nBarbara Fried’s political action committee, Mind
the Gap. It is among the entities that received FTX customer money as political donations. Fried and Bankman lived together in a Bahamas house purchased for
them using FTX investor funds, according to a financial expert called by the prosecution; in a lawsuit filed by FTX’s bankruptcy estate, Bankman and Fried are
alleged to have been given $10 million of FTX customer funds.\n\nGabe Bankman-Fried’s nonprofit, Guarding Against Pandemics, which got millions from FTX. He was
also in a Signal group chat, Donations Processing, that Singh says was used to coordinate the transfer of FTX and Alameda money to various political causes. In
that chat, he can be seen recommending some donations be made “in the dark” to avoid public relations headaches.', 'publisher': 'The Verge', 'page': 'In the
end, the FTX trial was about the friends screwed along the way (The Verge, 2023-10-26T12:00:00+00:00)', 'article': 'In the end, the FTX trial was about the
friends screwed along the way'}, {'passage': 'In the end, the FTX trial was about the friends screwed along the way (The Verge,
2023-10-26T12:00:00+00:00)\nSingh also said Gabe had an assistant fly to the Bahamas with a stack of blank checks for Singh to sign.\n\nWang, Singh, and Ellison
are now felons. Even if they never do time — and they may very well be imprisoned or subjected to house arrest — their lives are irrevocably changed, not least
by whatever restitution requirements they are subjected to. They are barred from taking certain kinds of jobs. And of course, there is the publicity. Bankman,
Fried, and Gabe Bankman-Fried’s reputations are tarnished; in the political arenas where they once operated, they are suddenly undesirable.\n\nIt is a Greek
tragedy, precipitated by one man’s character flaw: the arrogance of Sam Bankman-Fried.', 'publisher': 'The Verge', 'page': 'In the end, the FTX trial was about
the friends screwed along the way (The Verge, 2023-10-26T12:00:00+00:00)', 'article': 'In the end, the FTX trial was about the friends screwed along the way'},
{'passage': 'In the end, the FTX trial was about the friends screwed along the way (The Verge, 2023-10-26T12:00:00+00:00)\nThe general sense I got, from the
testimony of Bankman-Fried’s former friends, was that things just kept escalating\n\nDuring Yedidia’s testimony, we were shown a photograph in court: former
President Bill Clinton and former Prime Minister Tony Blair onstage in suits next to Bankman-Fried, in a rumpled T-shirt and cargo shorts. Sure, dressing down
is a staple of Silicon Valley startup culture — and a sign of its swagger. The defense counsel tried to elicit testimony from Yedidia that the cargo shorts were
evidence of Bankman-Fried’s humility, but the photograph seemed to signify just the opposite: that the man in cargo shorts is so important, he doesn’t need to
dress up, even to meet former world leaders.\n\nNow, in court, Bankman-Fried is wearing a suit.\n\nIn 2019, just months after FTX was founded, Wang oversaw code
called “allow_negative” that let Alameda Research avoid being liquidated by FTX’s risk management. It was turned on in July.', 'publisher': 'The Verge', 'page':
'In the end, the FTX trial was about the friends screwed along the way (The Verge, 2023-10-26T12:00:00+00:00)', 'article': 'In the end, the FTX trial was about
the friends screwed along the way'}, {'passage': 'In the end, the FTX trial was about the friends screwed along the way (The Verge,
2023-10-26T12:00:00+00:00)\nIt was turned on in July. No other customer trading on FTX had “allow_negative” privileges — only Alameda. Wang’s testimony suggests
that the fraud at FTX started very early since the same day that allow_negative was switched on, Bankman-Fried tweeted that Alameda’s account “was like everyone
else’s.”\n\nThe general sense I got, from the testimony of Bankman-Fried’s former friends, was that things just kept escalating. For instance, Alameda Research
first had a negative balance in late 2019, Wang testified. Wang testified that he heard Bankman-Fried tell another trader that it was okay to continue
withdrawing money from the account as long as the amount was less than FTX’s total trading revenue. A little later, Wang discovered that Alameda was negative by
$50 million more than FTX’s revenue. Bankman-Fried told him it was fine, and it just didn’t account for the FTT token that the two of them had launched. Wang
trusted him.', 'publisher': 'The Verge', 'page': 'In the end, the FTX trial was about the friends screwed along the way (The Verge, 2023-10-26T12:00:00+00:00)',
'article': 'In the end, the FTX trial was about the friends screwed along the way'}, {'passage': 'In the end, the FTX trial was about the friends screwed along
the way (The Verge, 2023-10-26T12:00:00+00:00)\nWang trusted him.\n\nWang also set Alameda Research up with an enormous line of credit, unmatched by anyone else
on FTX. Ultimately, the number was $65 billion — the result of repeated increases as Alameda Research repeatedly ran up against its credit limit. This line of
capital wasn’t visible to auditors, Ellison testified. That credit line was used primarily for trading because it “allowed us to make profitable trades we
couldn’t have made otherwise,” Ellison testified.\n\n“Over time, it was something I became more comfortable with.”\n\nWang’s testimony revealed how the fraud
got bigger, little by little. First, it was okay to dip into FTX’s profits. Then, it was okay to take more. Then, it was okay to take even more. First, it was a
little line of credit just to provide liquidity. Then a bigger one. Then an effectively limitless one.', 'publisher': 'The Verge', 'page': 'In the end, the FTX
trial was about the friends screwed along the way (The Verge, 2023-10-26T12:00:00+00:00)', 'article': 'In the end, the FTX trial was about the friends screwed
along the way'}, {'passage': 'In the end, the FTX trial was about the friends screwed along the way (The Verge, 2023-10-26T12:00:00+00:00)\nThen a bigger one.
Then an effectively limitless one.\n\nEllison testified to something similar, choking back tears: “When I started working at Alameda, I don’t think I would have
believed you if you told me I would be sending false balance sheets to our lenders or taking customer money, but over time, it was something I became more
comfortable with.”\n\nEllison’s tenure as CEO of Alameda Research was an exercise in optics, she testified. Alameda Research was, by then, FTX’s sin-eater. When
FTX took a loss in the hundreds of million dollars because a trader had exploited its system, Wang moved that loss onto Alameda’s balance sheet. FTX’s
“insurance fund,” meant to keep traders from getting stuck with other traders’ losses, was totally made up — Alameda simply picked up the tab.\n\nEven that
wasn’t good enough.', 'publisher': 'The Verge', 'page': 'In the end, the FTX trial was about the friends screwed along the way (The Verge,
2023-10-26T12:00:00+00:00)', 'article': 'In the end, the FTX trial was about the friends screwed along the way'}, {'passage': 'In the end, the FTX trial was
about the friends screwed along the way (The Verge, 2023-10-26T12:00:00+00:00)\nEven that wasn’t good enough. Singh testified to sprucing up FTX’s balance sheet
with backdated transactions in December 2021 to push FTX’s revenue “over the line” to $1 billion. By the time Singh found out about the use of customer money,
in 2022, he was already compromised.\n\nAt every turn, the scams at FTX and Alameda led back to making Bankman-Fried more important\n\nIn Singh’s case, it
wasn’t just balance sheets he was fiddling with — it was also the American political system. By coordinating with Gabe Bankman-Fried, Singh had funneled money
to a number of political causes. That money came from “loans,” which weren’t really loans since he hadn’t signed any term sheets.\n\nWhat was the goal here?
What was accomplished? Singh testified to more than a billion dollars spent on celebrity endorsements, which made FTX very famous — even though the futures
platform didn’t operate in the US.', 'publisher': 'The Verge', 'page': 'In the end, the FTX trial was about the friends screwed along the way (The Verge,
2023-10-26T12:00:00+00:00)', 'article': 'In the end, the FTX trial was about the friends screwed along the way'}, {'passage': 'In the end, the FTX trial was
about the friends screwed along the way (The Verge, 2023-10-26T12:00:00+00:00)\nThe political donations gave Bankman-Fried influence in the highest echelons of
power. Ellison testified to Bankman-Fried’s venture investments, which gave him more dominance in the tech world. At every turn, the scams at FTX and Alameda
led back to making Bankman-Fried more important.\n\nThe pie chart that accompanied expert witness Peter Easton’s testimony. Image: The prosecution’s exhibit
1045\n\nThis was backed up by expert testimony from Notre Dame accounting professor Peter Easton, who’d also assisted in investigating Enron and WorldCom. In a
pie chart that showed how he’d traced funds for the $11.3 billion liability that Ellison, Singh, Wang, and Bankman-Fried had known about in June 2022, we saw
that $1.2 billion was a loan repayment to crypto lender Genesis.', 'publisher': 'The Verge', 'page': 'In the end, the FTX trial was about the friends screwed
along the way (The Verge, 2023-10-26T12:00:00+00:00)', 'article': 'In the end, the FTX trial was about the friends screwed along the way'}, {'passage': 'In the
end, the FTX trial was about the friends screwed along the way (The Verge, 2023-10-26T12:00:00+00:00)\nA substantial chunk had been spent on various
investments, including in the celebrity-connected venture firm K5, trading firm Modulo (co-founded by one of Bankman-Fried’s ex-girlfriends), and AI company
Anthropic.\n\nAnother $413 million had gone directly to Bankman-Fried himself, through his wholly-owned company Paper Bird. “Outflows to insiders,” which
suggested Singh’s not-quite-loans were included, was $192 million; real estate was another $228 million; and the donations to Gabe Bankman-Fried’s Guarding
Against Pandemics, combined with the philanthropy from the FTX Foundation, totaled less than $40 million.\n\nIt is easy to lie with words and harder to lie with
money.', 'publisher': 'The Verge', 'page': 'In the end, the FTX trial was about the friends screwed along the way (The Verge, 2023-10-26T12:00:00+00:00)',
'article': 'In the end, the FTX trial was about the friends screwed along the way'}, {'passage': 'In the end, the FTX trial was about the friends screwed along
the way (The Verge, 2023-10-26T12:00:00+00:00)\nIt is easy to lie with words and harder to lie with money. Bankman-Fried went around telling people that he
wanted to make a lot of money to give it away, an idea of the Effective Altruism movement that Bankman-Fried affiliated himself with — but the amount of money
he funneled to his Paper Bird vehicle alone was ten times the size of his philanthropic spending as outlined by Easton. And it was all vastly outweighed by his
investments — the places where he could prove he was very smart by investing in the right companies early.\n\nAnd in brief, explosive testimony, former FTX
lawyer Can Sun demonstrated what Bankman-Fried’s tactic was. During the days in November when the exchange imploded, Bankman-Fried was trying to raise money
from Apollo, the private equity group. Apollo wanted an explanation for what had happened, and so Bankman-Fried came to Sun to ask him to come up with
justifications for the missing funds.', 'publisher': 'The Verge', 'page': 'In the end, the FTX trial was about the friends screwed along the way (The Verge,
2023-10-26T12:00:00+00:00)', 'article': 'In the end, the FTX trial was about the friends screwed along the way'}, {'passage': 'In the end, the FTX trial was
about the friends screwed along the way (The Verge, 2023-10-26T12:00:00+00:00)\nThe defense has struggled to shake the credibility of witnesses such as Yedidia,
Wang, and Ellison\n\nThat “basically confirmed my suspicions that had been rising all day” that Bankman-Fried had purloined the funds, Sun said. He ran
Bankman-Fried through possible explanations, including a margin loan program, and explained that none of them could fit the actual situation — for instance,
Alameda had borrowed more from FTX than had ever been in the margin loan program. He said Bankman-Fried acknowledged their conversation with a “yup, yup.” “I
was expecting a bigger response, but it was very muted,” Sun said.\n\nAfter that, Singh spilled the beans: Alameda had withdrawn assets that included the
customer funds, Sun testified. He quit the next day. Like Yedidia, Sun testified with immunity.\n\nIn the courtroom, we were treated to a Good Morning America
appearance by Bankman-Fried on December 1st, 2022.', 'publisher': 'The Verge', 'page': 'In the end, the FTX trial was about the friends screwed along the way
(The Verge, 2023-10-26T12:00:00+00:00)', 'article': 'In the end, the FTX trial was about the friends screwed along the way'}, {'passage': 'In the end, the FTX
trial was about the friends screwed along the way (The Verge, 2023-10-26T12:00:00+00:00)\nIn it, Bankman-Fried tried to justify the missing funds through the
margin loan program. It was such a flimsy excuse that even George Stephanopoulos could see through it.\n\nDid Bankman-Fried really think he could get away with
it all? He must still think so, or he would not be dragging his closest companions’ names through the mud in a jury trial. Throughout the entire prosecution’s
case, the defense has struggled to shake the credibility of witnesses such as Yedidia, Wang, and Ellison. The hits the defense made on Singh didn’t make me
doubt the overall truth of his story — they just made me think that Singh was kind of self-serving and trying to secure himself a lenient sentence.',
'publisher': 'The Verge', 'page': 'In the end, the FTX trial was about the friends screwed along the way (The Verge, 2023-10-26T12:00:00+00:00)', 'article': 'In
the end, the FTX trial was about the friends screwed along the way'}, {'passage': 'Is Sam Bankman-Fried’s defense even trying to win? (The Verge,
2023-10-04T23:02:14+00:00)\nI have never seen Sam Bankman-Fried so still as he was during the prosecution’s opening statement. The characteristic leg-jiggling
was absent. He barely moved as the prosecutor listed the evidence against him: internal company files, what customers were told, the testimony of his
co-conspirators and his own words.\n\nHis hair was shorn, the result of a haircut from a fellow prisoner, the Wall Street Journal reported. He wore a suit
bought at a discount at Macy’s, per the Journal; it hung on him. He appeared to have lost some weight.\n\n“All of that was built on lies.”\n\nBankman-Fried, at
this time last year, had a luxury lifestyle as the CEO of crypto exchange FTX, said the assistant US attorney, Thane Rehn, in the cadence of a high schooler
delivering his lines in a student play. Bankman-Fried hung out with Tom Brady. He was on magazine covers, lived in a $30 million penthouse, and spent time with
world politicians.', 'publisher': 'The Verge', 'page': 'Is Sam Bankman-Fried’s defense even trying to win? (The Verge, 2023-10-04T23:02:14+00:00)', 'article':
'Is Sam Bankman-Fried’s defense even trying to win?'}, {'passage': 'Is Sam Bankman-Fried’s defense even trying to win? (The Verge,
2023-10-04T23:02:14+00:00)\n“All of that was built on lies,” Rehn said.\n\nIn his opening statement, Rehn dodged explaining cryptocurrency to the jury. Instead,
he punched hard on Bankman-Fried lying and stealing.\n\nBankman-Fried sat almost motionless, occasionally glancing at Rehn, as the prosecutor told the jury that
Bankman-Fried sold stock in FTX and borrowed millions from lenders by lying.\n\nThe story Rehn told is familiar to anyone following the news. In May and June of
2022, Alameda Research — the crypto trading company ostensibly helmed by Caroline Ellison — didn’t have enough to pay its bills, so it pulled customer money to
repay loans. By September, the hole in the FTX balance sheet was so big that customers could never be repaid.\n\nFTX “didn’t have a chief risk officer, which
became an issue when the storm hit.”\n\nWhen CoinDesk published its article in November 2022, people realized FTX was a house of cards, Rehn said. Meanwhile,
Bankman-Fried tweeted.', 'publisher': 'The Verge', 'page': 'Is Sam Bankman-Fried’s defense even trying to win? (The Verge, 2023-10-04T23:02:14+00:00)',
'article': 'Is Sam Bankman-Fried’s defense even trying to win?'}, {'passage': 'Is Sam Bankman-Fried’s defense even trying to win? (The Verge,
2023-10-04T23:02:14+00:00)\nMeanwhile, Bankman-Fried tweeted. “FTX is fine. Assets are fine” and “We don’t invest customer assets even in
treasuries.”\n\nPointing at Bankman-Fried, Rehn said, “This man stole billions of dollars from thousands of people.”\n\nSo how was the defense going to follow
it up? I was very curious, having learned yesterday that Bankman-Fried had never been offered a plea deal since he and his attorneys had told the government
they wouldn’t negotiate. Surely there would be some manner of evidence, some something, that would have made him so confident.\n\nThere was, instead, a
metaphor.\n\nDefense attorney Mark Cohen, with the energy of a patient father telling his obnoxious children a bedtime story, assured us that working at a
startup was like building a plane while flying it, and that FTX the plane had flown right into the perfect storm: the crypto crash.', 'publisher': 'The Verge',
'page': 'Is Sam Bankman-Fried’s defense even trying to win? (The Verge, 2023-10-04T23:02:14+00:00)', 'article': 'Is Sam Bankman-Fried’s defense even trying to
win?'}, {'passage': 'Is Sam Bankman-Fried’s defense even trying to win? (The Verge, 2023-10-04T23:02:14+00:00)\nExcept, uh, he also said this: FTX “didn’t have
a chief risk officer, which became an issue when the storm hit.”\n\nI couldn’t stop thinking about the missing risk officer\n\nThe problem with this metaphor is
that if FTX was a plane, it was a plane flying with a key component missing — namely, the risk officer, an executive whose job it is to, well, manage risk. This
is sort of an important thing, as risks can be anything from reputational to regulatory to financial.\n\nFTX was named such as it was because it was a futures
exchange, which, to borrow a phrase from Bloomberg’s Matt Levine, “sits between the winners and losers of bets.” That means FTX can’t pay out what it owes the
winners unless the losers pay up. Risk management is a crucial part of the business; risk officers exist to identify business’ potential risks, monitor, and
mitigate them. This is to say nothing of the regulatory risks around crypto.', 'publisher': 'The Verge', 'page': 'Is Sam Bankman-Fried’s defense even trying to
win? (The Verge, 2023-10-04T23:02:14+00:00)', 'article': 'Is Sam Bankman-Fried’s defense even trying to win?'}, {'passage': 'Is Sam Bankman-Fried’s defense even
trying to win? (The Verge, 2023-10-04T23:02:14+00:00)\nThis is to say nothing of the regulatory risks around crypto.\n\nAs Cohen droned on about airplanes, I
couldn’t stop thinking about the missing risk officer. Bringing it up, I thought, was a tremendous mistake. The prosecution hadn’t mentioned it. Either
Bankman-Fried is stupid — unlikely — or he deliberately didn’t hire a risk officer. Was he worried about what one might find?\n\nSure, as Cohen put it,
Bankman-Fried was a math nerd who didn’t party. That paints a picture of someone who’s pretty deliberate, particularly since he immediately left MIT and went to
work on Wall Street. If he had been a party-hardy trainwreck, I could see overlooking a risk officer in order to do another line, or a supermodel, or something
else important. Why was the defense bringing this up?\n\nBut as Cohen tried to tell me that FTX’s and Alameda’s business relationships were “reasonable under
the circumstances,” the lack of risk officer kept elbowing me in the ribs.', 'publisher': 'The Verge', 'page': 'Is Sam Bankman-Fried’s defense even trying to
win? (The Verge, 2023-10-04T23:02:14+00:00)', 'article': 'Is Sam Bankman-Fried’s defense even trying to win?'}, {'passage': 'Is Sam Bankman-Fried’s defense even
trying to win? (The Verge, 2023-10-04T23:02:14+00:00)\n“Sam acted in good faith and took reasonable business measures” is a pretty hard pill to swallow with
that in mind.\n\nMan, it’s no good when your defense lawyer has just made you sound worse than the prosecution already did. And while Cohen tried to make the
common white-collar defense argument that Bankman-Fried, as CEO, was simply too busy to oversee what everyone did every day, he just made me more suspicious.
That’s why you hire a risk officer and delegate! That’s the whole point! I could barely even hear Cohen blaming Caroline Ellison and Changpeng “CZ” Zhao for the
debacle over the “no risk officer” ringing in my ears.\n\nFollowing the defense’s opening statements, things got still worse for Bankman-Fried. The prosecution
called its first witness, Marc-Antoine Julliard, whose money got stuck on FTX. Juilliard, who was born in Paris and lives in London, testified that he trusted
FTX because Bankman-Fried came across as a leading figure of the industry.', 'publisher': 'The Verge', 'page': 'Is Sam Bankman-Fried’s defense even trying to
win? (The Verge, 2023-10-04T23:02:14+00:00)', 'article': 'Is Sam Bankman-Fried’s defense even trying to win?'}, {'passage': 'Is Sam Bankman-Fried’s defense even
trying to win? (The Verge, 2023-10-04T23:02:14+00:00)\nWhen he was evaluating the exchange, he thought the sheer volume of users was important, too — at the
time, FTX was among the top three biggest exchanges. Plus, major VC firms had invested, and “they don’t commit hundreds of millions without doing due diligence,
checking the books, the accountancy of the firm, going through several compliance process[es], so that was a vote of confidence for me,” Juilliard said.
(Evidently he had not paid attention to the Elizabeth Holmes trial.)\n\nHe also noted FTX’s glossy ads — featuring Gisele Bündchen, for instance — suggested a
very high budget. It wouldn’t make sense to spend that much money unless FTX had very strong financials, Juilliard figured. He opened an account, transferred in
both regular money and cryptocurrency, and used the exchange to execute his plan: buying Bitcoin to sell back in five to ten years at higher prices.',
'publisher': 'The Verge', 'page': 'Is Sam Bankman-Fried’s defense even trying to win? (The Verge, 2023-10-04T23:02:14+00:00)', 'article': 'Is Sam
Bankman-Fried’s defense even trying to win?'}, {'passage': 'Is Sam Bankman-Fried’s defense even trying to win? (The Verge, 2023-10-04T23:02:14+00:00)\nIt is a
thankless task to cross-examine a customer whose money is gone\n\nIn November 2022, things went bad for Julliard. He followed Bankman-Fried on Twitter, and read
aloud the “FTX is fine. Assets are fine” tweets, along with “FTX has enough to cover all client holdings. We don’t invest client assets” and a few others, which
gave Julliard the impression that his money was there — the problem might have been technical (anti-spam measures) or regulatory. When he tried to get his money
out on November 8th, it was too late. We saw screenshots of his withdrawal attempts: $20,000 USD and about 4 Bitcoin, which were worth about $20,000 at the
time: about $100,000 money, inaccessible.\n\nIt is a thankless task to cross-examine a customer whose money is gone, but Cohen tried anyway.', 'publisher': 'The
Verge', 'page': 'Is Sam Bankman-Fried’s defense even trying to win? (The Verge, 2023-10-04T23:02:14+00:00)', 'article': 'Is Sam Bankman-Fried’s defense even
trying to win?'}, {'passage': 'Is Sam Bankman-Fried’s defense even trying to win? (The Verge, 2023-10-04T23:02:14+00:00)\nHe noted that Julliard was a licensed
commodities broker, who was trading in crypto because he didn’t have to disclose it; that Julliard knew that crypto was new and risky, and that Julliard didn’t
review the terms of service agreement he’d assented to when making his FTX account.\n\nWell, sure, but so what?\n\nThe next witness called was Bankman-Fried’s
former college (and FTX) roommate, Adam Yedidia, about whom I expect I will have much more to say tomorrow.', 'publisher': 'The Verge', 'page': 'Is Sam
Bankman-Fried’s defense even trying to win? (The Verge, 2023-10-04T23:02:14+00:00)', 'article': 'Is Sam Bankman-Fried’s defense even trying to win?'},
{'passage': 'The FTX trial is bigger than Sam Bankman-Fried (The Verge, 2023-09-28T12:00:00+00:00)\nThe trial of Sam Bankman-Fried is likely to be more
consequential than just whether the man himself is found guilty. Depending on what evidence is introduced during the trial, it could be rough for the entire
crypto industry.\n\n“How much damage can this trial do to the already beaten-down reputation of the industry at this point?” asks Yesha Yadav, a law professor
at Vanderbilt University. “This trial is going to be an excruciating moment for the industry because no one knows what kind of evidence might come
out.”\n\nBankman-Fried, the founder of FTX and Alameda Research, is facing seven counts of criminal charges: two counts of wire fraud, and five counts of
conspiracy charges. FTX was a failed cryptocurrency exchange founded in 2019. According to a now-deleted profile from FTX investors Sequoia Capital, FTX was
founded because of Bankman-Fried’s frustration with other exchanges when he was running Alameda Research, his crypto trading firm.', 'publisher': 'The Verge',
'page': 'The FTX trial is bigger than Sam Bankman-Fried (The Verge, 2023-09-28T12:00:00+00:00)', 'article': 'The FTX trial is bigger than Sam Bankman-Fried'},
{'passage': 'The FTX trial is bigger than Sam Bankman-Fried (The Verge, 2023-09-28T12:00:00+00:00)\nWe have some inkling of what the government will be arguing
thanks to the filings prosecutors are making in this case. They argue that Bankman-Fried lied about consumer protection and that Bankman-Fried’s statements that
FTX was “avoiding or managing conflicts of interest” and that “as a general principle FTX segregates customer assets from its own assets across our platforms”
were lies, in a superseding indictment filed August 14th.\n\nThe indictment also says that there were special features in FTX’s code that “permitted Alameda to
spend and withdraw unlimited amounts of money from FTX,” which were created at Bankman-Fried’s direction. That effectively exempted Alameda from the kinds of
risk management other customers faced. At the same time, the government alleges, “Bankman-Fried publicly and repeatedly asserted that Alameda did not have
privileged access to FTX.”', 'publisher': 'The Verge', 'page': 'The FTX trial is bigger than Sam Bankman-Fried (The Verge, 2023-09-28T12:00:00+00:00)',
'article': 'The FTX trial is bigger than Sam Bankman-Fried'}, {'passage': 'The FTX trial is bigger than Sam Bankman-Fried (The Verge,
2023-09-28T12:00:00+00:00)\nBankman-Fried is accused of using “billions of dollars in misappropriated FTX customer deposits” to help buy more than $200 million
of real estate for himself, make billions of dollars of investments for his own interest, and repay Alameda’s lenders, according to the indictment,
Additionally, Bankman-Fried allegedly used more than $100 million of customer funds to make political contributions; prosecutors can show evidence of those
contributions in this trial, even though they aren’t part of the charges brought. A second trial is scheduled for March 2024, with additional charges.\n\nBefore
his fall, Bankman-Fried made himself out to be the Good Boy of crypto — the trustworthy face of a sometimes-shady industry. He was also very interested in
publicity, sitting for many interviews both before and after the fall of FTX. The quick rise of FTX as an industry force was at least in part due to
Bankman-Fried’s appetite for attention.', 'publisher': 'The Verge', 'page': 'The FTX trial is bigger than Sam Bankman-Fried (The Verge,
2023-09-28T12:00:00+00:00)', 'article': 'The FTX trial is bigger than Sam Bankman-Fried'}, {'passage': 'The FTX trial is bigger than Sam Bankman-Fried (The
Verge, 2023-09-28T12:00:00+00:00)\nHere are the hits:\n\nBankman-Fried gave interviews freely — and quickly rose to public prominence in the industry. Though
FTX hadn’t been in the business as long as competing exchanges such as Coinbase, Kraken, or Gemini, Bankman-Fried positioned himself as an important, boyish
face for crypto. (At one point, Bankman-Fried told a colleague at FTX that “I honestly think it’s negative EV [this may mean “expected value,” as in poker] for
me to cut my hair. I think it’s important for people to think I look crazy.”)\n\nBecause he was so successful at this kind of public relations, his fall from
grace was another mark against an industry that was already roiled by bankruptcies and scandals. Some additional trouble for the crypto industry is likely to
come from one crucial element of the fraud trial — the part where the government must prove intent.\n\nThe first part of proving the government’s case is pretty
simple and a little boring: prosecutors must show that certain transactions took place.', 'publisher': 'The Verge', 'page': 'The FTX trial is bigger than Sam
Bankman-Fried (The Verge, 2023-09-28T12:00:00+00:00)', 'article': 'The FTX trial is bigger than Sam Bankman-Fried'}, {'passage': 'The FTX trial is bigger than
Sam Bankman-Fried (The Verge, 2023-09-28T12:00:00+00:00)\nWhatever records the Southern District of New York has for the transactions will be shown.\n\n“What
conversations happened between him and his co-conspirators that are now cooperating against him?”\n\nThe second part is where all the drama is likely to come,
says Christopher LaVigne, a litigation partner and co-chair of the cryptocurrency practice at the law firm Withers. Prosecutors have to connect those
transactions to Bankman-Fried, show that he knew what he was doing was wrong, and prove that he lied about it anyway.\n\n“What was he saying to his parents and
his other advisors about this?” LaVigne says. “What conversations happened between him and his co-conspirators that are now cooperating against him?”\n\nTo
further establish intent, the government can use Bankman-Fried’s own words. The indictment calls Bankman-Fried’s tweets in November 2022 “false and
misleading.”\n\n“We had him going out directly to the internet,” LaVigne says.', 'publisher': 'The Verge', 'page': 'The FTX trial is bigger than Sam
Bankman-Fried (The Verge, 2023-09-28T12:00:00+00:00)', 'article': 'The FTX trial is bigger than Sam Bankman-Fried'}, {'passage': 'The FTX trial is bigger than
Sam Bankman-Fried (The Verge, 2023-09-28T12:00:00+00:00)\n“We had him going out directly to the internet,” LaVigne says. If he wrote things on Twitter or said
things in interviews that weren’t true, that’s more fodder for the government’s case. “They can point to that and say, ‘This is what he said, this is what
actually happened.’”\n\nOther evidence may include Signal messages and testimony from co-conspirators who plead guilty to their own charges. Alameda Research
CEO Caroline Ellison, who was also sometimes Bankman-Fried’s girlfriend, may play an important role — Bankman-Fried leaked her diaries to The New York Times and
was consequently jailed for witness tampering. The government has indicated FTX co-founder Gary Wang and engineering head Nishad Singh will also be among the
witnesses called to give testimony about Bankman-Fried.\n\nBankman-Fried’s defense can also introduce risks for people who dealt with him\n\nThere may be
testimony from lenders, venture capitalists, and customers in order to establish the basis for some charges.', 'publisher': 'The Verge', 'page': 'The FTX trial
is bigger than Sam Bankman-Fried (The Verge, 2023-09-28T12:00:00+00:00)', 'article': 'The FTX trial is bigger than Sam Bankman-Fried'}, {'passage': 'The FTX
trial is bigger than Sam Bankman-Fried (The Verge, 2023-09-28T12:00:00+00:00)\nIn the indictment, the prosecutors allege that Bankman-Fried lied to FTX
investors. Some of those investors may be called on to testify, which is not the kind of thing VC big shots generally enjoy — and which might create collateral
damage for the industry.\n\nIf, for instance, Sequoia Capital did due diligence around its investment in FTX, whatever Bankman-Fried told its partners could be
important. Was FTX already sending Alameda customer funds at that point? Did Bankman-Fried know about it? And did he tell VCs at the time? (If he told them it
wasn’t happening, that would establish intent.) Binance was also an early investor in FTX, and former executives may be called on to testify as well, says
Hermine Wong, the former head of policy at Coinbase and a former SEC regulator.\n\nBankman-Fried’s defense can also introduce risks for people who dealt with
him. Defense lawyers have several simultaneous objectives. First and foremost, they’re trying to prove their client is not guilty.', 'publisher': 'The Verge',
'page': 'The FTX trial is bigger than Sam Bankman-Fried (The Verge, 2023-09-28T12:00:00+00:00)', 'article': 'The FTX trial is bigger than Sam Bankman-Fried'},
{'passage': 'The FTX trial is bigger than Sam Bankman-Fried (The Verge, 2023-09-28T12:00:00+00:00)\nFirst and foremost, they’re trying to prove their client is
not guilty. But just in case they don’t get the outcome they want, they’re also laying the groundwork for appeals and sentencing arguments. Any piece of
evidence they want to introduce for those two purposes has to come into play in the trial.\n\nAlready, Bankman-Fried’s lawyers are marking arguments that his
constitutional rights are being violated by his pretrial incarceration because, among other things, his internet connection wasn’t good enough for him to plan
his defense, notes LaVigne. That could be grounds for an appeal.\n\n“Is he going to throw the entire industry under the bus?”\n\nBankman-Fried’s behavior after
the fall of FTX suggests he’s something of a wild card. He may suggest he was acting on the advice of his lawyers. But he may also introduce other evidence that
could be troublesome — implying, for instance, that he was engaged in standard industry behavior or that everything that happened was Binance’s fault.',
'publisher': 'The Verge', 'page': 'The FTX trial is bigger than Sam Bankman-Fried (The Verge, 2023-09-28T12:00:00+00:00)', 'article': 'The FTX trial is bigger
than Sam Bankman-Fried'}, {'passage': 'The FTX trial is bigger than Sam Bankman-Fried (The Verge, 2023-09-28T12:00:00+00:00)\nThat may be risky, but we already
know that Bankman-Fried loves risk.\n\n“Is he going to throw the entire industry under the bus?” Wong asks. “An idea like, ‘Everyone was doing this, it’s not
fair I’m the only one who was charged?’” That may not fly in a court of law, but it could absolutely damage public perception of crypto at large.\n\nFor
instance, just before he was arrested, Bankman-Fried brought forward messages from a crypto group chat in testimony he planned to give before Congress. In that
testimony, he primarily blames Binance and his lawyers at the firm Sullivan & Cromwell.\n\nIt’s possible other messages from group chats may be brought into
evidence during the trial. That can potentially be embarrassing for the whole industry, Wong says.\n\n“You can imagine some of these founders, CEOs, people of
that echelon talk to each other somewhat informally about what’s going on,” she says.', 'publisher': 'The Verge', 'page': 'The FTX trial is bigger than Sam
Bankman-Fried (The Verge, 2023-09-28T12:00:00+00:00)', 'article': 'The FTX trial is bigger than Sam Bankman-Fried'}, {'passage': 'The FTX trial is bigger than
Sam Bankman-Fried (The Verge, 2023-09-28T12:00:00+00:00)\nIf it seems like his peer group supported him, or worse, fawned over him, that’s a real problem for
the industry’s reputation.”\n\nAfter FTX declared bankruptcy, Bankman-Fried went on an extensive media tour, doing interviews with The New York Times’ Andrew
Ross Sorkin and Good Morning America. Even under house arrest, he did interviews, including with The New Yorker. Bankman-Fried seemed to think that if he just
explained things, everyone would understand that what he did was a mistake, not a crime.', 'publisher': 'The Verge', 'page': 'The FTX trial is bigger than Sam
Bankman-Fried (The Verge, 2023-09-28T12:00:00+00:00)', 'article': 'The FTX trial is bigger than Sam Bankman-Fried'}, {'passage': 'Sex, Signal messages, and
sabotaging FTX\'s code: SBF execs and Bahamas roommates tell all in court (Cnbc | World Business News Leader, 2023-10-06T20:17:00+00:00)\nWitness Adam Yedidia
answers questions during Sam Bankman-Fried fraud trial over the collapse of FTX, the bankrupt cryptocurrency exchange, at Federal Court in New York City,
October 5, 2023, in this courtroom sketch. Jane Rosenberg | Reuters\n\nTwo of Sam Bankman-Fried\'s former friends from MIT, who also worked at crypto exchange
FTX while living with the company\'s founder in the Bahamas, took the stand in a Manhattan courtroom this week to testify against their former classmate,
confidant, and boss — a man who allegedly ran a crypto empire that defrauded thousands of customers out of billions of dollars. Gary Wang, the lesser-known
co-founder of FTX, was asked by Assistant U.S. Attorney Nicolas Roos on Thursday, "Did you commit financial crimes while working at FTX?" "Yes," responded
Wang.', 'publisher': 'CNBC', 'page': "Sex, Signal messages, and sabotaging FTX's code: SBF execs and Bahamas roommates tell all in court (Cnbc | World Business
News Leader, 2023-10-06T20:17:00+00:00)", 'article': "Sex, Signal messages, and sabotaging FTX's code: SBF execs and Bahamas roommates tell all in court"},
{'passage': "Sex, Signal messages, and sabotaging FTX's code: SBF execs and Bahamas roommates tell all in court (Cnbc | World Business News Leader,
2023-10-06T20:17:00+00:00)\nHe spoke so fast that U.S. District Judge Lewis Kaplan and the prosecutor both stopped him at points to ask that he slow his pace.
Much of Wang's testimony on Friday focused on the final days at FTX before the entire operation imploded, including reports in the media detailing Alameda's
business practices and its troubling ties to FTX. Wang said that in response to the reporting an emergency meeting was called between Bankman-Fried, Wang and
Singh, to discuss shutting down Alameda. He said they ultimately decided against such a move, because he and Bankman-Fried were aware that Alameda had no way to
repay the roughly $14 billion hole in its books. Prosecutors took the jury through a series of tweets, beginning on Nov. 7. Posts came from the company blaming
bank hours for slow withdrawals, while Bankman-Fried tweeted from his personal account, assuring customers that all was fine.", 'publisher': 'CNBC', 'page':
"Sex, Signal messages, and sabotaging FTX's code: SBF execs and Bahamas roommates tell all in court (Cnbc | World Business News Leader,
2023-10-06T20:17:00+00:00)", 'article': "Sex, Signal messages, and sabotaging FTX's code: SBF execs and Bahamas roommates tell all in court"}, {'passage': 'Sex,
Signal messages, and sabotaging FTX\'s code: SBF execs and Bahamas roommates tell all in court (Cnbc | World Business News Leader,
2023-10-06T20:17:00+00:00)\n"FTX was not fine and assets were not fine," Wang testified. On Nov. 12, after FTX declared bankruptcy, Bankman-Fried asked Wang to
drive with him to the Bahamas Securities Commission for a meeting. On the drive, Bankman-Fried told Wang to transfer assets to Bahamian liquidators because he
believed they would allow him to maintain control of the company. Wang said he wasn\'t in the meeting with the securities authority, though Bankman-Fried\'s dad
was present.\n\n\n\nWang said he returned to the U.S. and met with prosecutors the next day. He faces up to 50 years in prison when he faces a judge for
sentencing following this trial. He told jurors he signed a six-page cooperation agreement that requires him to meet with prosecutors, answer their questions
truthfully and turn over evidence.', 'publisher': 'CNBC', 'page': "Sex, Signal messages, and sabotaging FTX's code: SBF execs and Bahamas roommates tell all in
court (Cnbc | World Business News Leader, 2023-10-06T20:17:00+00:00)", 'article': "Sex, Signal messages, and sabotaging FTX's code: SBF execs and Bahamas
roommates tell all in court"}, {'passage': "Sex, Signal messages, and sabotaging FTX's code: SBF execs and Bahamas roommates tell all in court (Cnbc | World
Business News Leader, 2023-10-06T20:17:00+00:00)\nSam Bankman-Fried, the founder of bankrupt cryptocurrency exchange FTX, is seen during a hearing as a U.S
judge revoked his bail, at a courthouse in New York, U.S., August 11, 2023 in this courtroom sketch. Jane Rosenberg | Reuters\n\n$65 billion line of
credit\n\nFor months, Bankman-Fried has known that Wang and Ellison, who were integral members of his personal and professional inner circles, had turned on
him. Both pleaded guilty in December and have since been cooperating with the U.S. attorney's office in Manhattan. Wang's testimony, which stretched into
Friday, was given under a cooperation agreement with the government. Ellison is expected to take the stand under a similar arrangement.\n\nU.S.", 'publisher':
'CNBC', 'page': "Sex, Signal messages, and sabotaging FTX's code: SBF execs and Bahamas roommates tell all in court (Cnbc | World Business News Leader,
2023-10-06T20:17:00+00:00)", 'article': "Sex, Signal messages, and sabotaging FTX's code: SBF execs and Bahamas roommates tell all in court"}, {'passage': "Sex,
Signal messages, and sabotaging FTX's code: SBF execs and Bahamas roommates tell all in court (Cnbc | World Business News Leader,
2023-10-06T20:17:00+00:00)\nEllison is expected to take the stand under a similar arrangement.\n\nU.S. District Judge Lewis Kaplan presides as Gary Wang
testifies during the fraud trial of Sam Bankman-Fried over the collapse of FTX, the bankrupt cryptocurrency exchange, at Federal Court in New York City, U.S.,
October 6, 2023 in this courtroom sketch. Jane Rosenberg | Reuters\n\nBorn in China, Wang moved to the U.S. at age 7, and grew up in Minnesota before going to
the Massachusetts Institute of Technology to study math and computer science. He worked at Google after college. Wang, who first met Bankman-Fried during high
school at a summer camp, owned 10% of Alameda, while his boss owned the other 90%. Wang told the court about the advantages that Alameda received by having code
baked into FTX's software that allowed special access to the crypto exchange.", 'publisher': 'CNBC', 'page': "Sex, Signal messages, and sabotaging FTX's code:
SBF execs and Bahamas roommates tell all in court (Cnbc | World Business News Leader, 2023-10-06T20:17:00+00:00)", 'article': "Sex, Signal messages, and
sabotaging FTX's code: SBF execs and Bahamas roommates tell all in court"}, {'passage': 'Sex, Signal messages, and sabotaging FTX\'s code: SBF execs and Bahamas
roommates tell all in court (Cnbc | World Business News Leader, 2023-10-06T20:17:00+00:00)\nThose privileges ultimately resulted in Alameda owing FTX $8 billion
worth of customer deposits. "We gave special privileges on FTX that gave unlimited withdrawals on the platform to Alameda," Wang said. Alameda was allowed to
withdraw and transfer those funds and had a $65 billion line of credit. "When customers deposited USD, it went to Alameda," he said. "It existed in the computer
code. Alameda could have negative balances and unlimited withdrawals." That "bug" in the code was written by Nishad Singh, who was FTX\'s director of
engineering, and reviewed by Wang. Bankman-Fried was calling the shots, Wang said. Wang also told the court about a $1 million personal loan he received and a
$200 million to $300 million loan in his name from Alameda that was never deposited into his account, but rather was used to make investments into other
companies on behalf of FTX.', 'publisher': 'CNBC', 'page': "Sex, Signal messages, and sabotaging FTX's code: SBF execs and Bahamas roommates tell all in court
(Cnbc | World Business News Leader, 2023-10-06T20:17:00+00:00)", 'article': "Sex, Signal messages, and sabotaging FTX's code: SBF execs and Bahamas roommates
tell all in court"}, {'passage': 'Sex, Signal messages, and sabotaging FTX\'s code: SBF execs and Bahamas roommates tell all in court (Cnbc | World Business
News Leader, 2023-10-06T20:17:00+00:00)\nAssistant United States Attorney Nicolas Roos questions Gary Wang during Sam Bankman-Fried\'s fraud trial over the
collapse of FTX, the bankrupt cryptocurrency exchange, at Federal Court in New York City, U.S., October 6, 2023, in this courtroom sketch. Jane Rosenberg |
Reuters\n\nAn $8 billion bug\n\nAdam Yedidia, who was the prosecution\'s second witness on Wednesday, continued his testimony on Thursday. Yedidia met
Bankman-Fried in college at MIT, and the pair remained close friends. Yedidia, assuming a robotic posture on the stand, worked out of FTX\'s Hong Kong office
from January to October of 2021 and then in the Bahamas until last year\'s collapse. In his testimony, he referred to a group Signal thread called "People of
the House," referring to Bankman-Fried\'s $35 million penthouse, where many employees lived.', 'publisher': 'CNBC', 'page': "Sex, Signal messages, and
sabotaging FTX's code: SBF execs and Bahamas roommates tell all in court (Cnbc | World Business News Leader, 2023-10-06T20:17:00+00:00)", 'article': "Sex,
Signal messages, and sabotaging FTX's code: SBF execs and Bahamas roommates tell all in court"}, {'passage': 'Sex, Signal messages, and sabotaging FTX\'s code:
SBF execs and Bahamas roommates tell all in court (Cnbc | World Business News Leader, 2023-10-06T20:17:00+00:00)\nZoom In Icon Arrows pointing outwards Exhibit
from the prosecution shows Signal thread called "People of the House," referring to Bankman-Fried\'s $35 million penthouse, where many employees lived. Source:
SDNY\n\nIn terms of who was paying the rent, Yedidia recalled Bankman-Fried saying he "assumed it\'s just Alameda paying for it in the end." Yedidia said
Bankman-Fried had told him, before he began working in the Bahamas in 2019, that he and Ellison had sex. Bankman-Fried asked Yedidia if it was a good idea for
them to date, to which Yedidia said no. Bankman-Fried responded by saying he was expecting that answer. One of Yedidia\'s responsibilities was fixing the bug in
the code that gave Alameda preferential treatment.', 'publisher': 'CNBC', 'page': "Sex, Signal messages, and sabotaging FTX's code: SBF execs and Bahamas
roommates tell all in court (Cnbc | World Business News Leader, 2023-10-06T20:17:00+00:00)", 'article': "Sex, Signal messages, and sabotaging FTX's code: SBF
execs and Bahamas roommates tell all in court"}, {'passage': 'Sex, Signal messages, and sabotaging FTX\'s code: SBF execs and Bahamas roommates tell all in
court (Cnbc | World Business News Leader, 2023-10-06T20:17:00+00:00)\nIn June 2022, he submitted a report to Bankman-Fried on Signal that showed $8 billion in
customer money held in an internal database tracking the cash wired to an Alameda account called "fiat at ftx.com" was missing. Yedidia said he and
Bankman-Fried spoke about it at the pickleball court at the resort in Nassau, Bahamas. He asked his boss if things were OK. He was concerned because it "seemed
like a lot of money" from FTX customers was at risk. "Sam said, we were bulletproof last year. We aren\'t bulletproof this year," Yedidia testified. Yedidia
said he asked when they would be bulletproof again. Bankman-Fried said he wasn\'t sure, but it may be six months to three years. Yedidia said Bankman-Fried
appeared "worried or nervous," which he said was atypical.', 'publisher': 'CNBC', 'page': "Sex, Signal messages, and sabotaging FTX's code: SBF execs and
Bahamas roommates tell all in court (Cnbc | World Business News Leader, 2023-10-06T20:17:00+00:00)", 'article': "Sex, Signal messages, and sabotaging FTX's
code: SBF execs and Bahamas roommates tell all in court"}, {'passage': 'Sex, Signal messages, and sabotaging FTX\'s code: SBF execs and Bahamas roommates tell
all in court (Cnbc | World Business News Leader, 2023-10-06T20:17:00+00:00)\nStill, Yedidia said he trusted Bankman-Fried and Ellison to "handle the situation."
On cross-examination, Christian Everdell, Bankman-Fried\'s attorney, focused on how Yedidia was the one responsible for developing and reviewing the
code.\n\n\n\nHe asked about the long hours employees worked and Yedidia\'s concern for Wang being near burnout. That resulted in Yedidia instituting a rule to
not wake Wang at night for bug fixes because he needed sleep. Everdell also drilled Yedidia on his high level of compensation in his less than two years at FTX.
His base salary was between $175,000 and $200,000, but he received multiple bonuses of more than $12 million in cash and company equity. Yedidia said he\'s now
teaching math — geometry and algebra — at a high school.', 'publisher': 'CNBC', 'page': "Sex, Signal messages, and sabotaging FTX's code: SBF execs and Bahamas
roommates tell all in court (Cnbc | World Business News Leader, 2023-10-06T20:17:00+00:00)", 'article': "Sex, Signal messages, and sabotaging FTX's code: SBF
execs and Bahamas roommates tell all in court"}, {'passage': 'Sex, Signal messages, and sabotaging FTX\'s code: SBF execs and Bahamas roommates tell all in
court (Cnbc | World Business News Leader, 2023-10-06T20:17:00+00:00)\nYedidia said he\'s now teaching math — geometry and algebra — at a high school. He
invested most of the millions he earned as bonuses back into FTX, and his equity stake is now worthless. As FTX was failing, Yedidia said he was by
Bankman-Fried\'s side. He highlighted a Signal exchange in November 2022, during which he wrote, "I love you Sam. I\'m not going anywhere." He said he wrote the
message because so many people had left. When asked what changed, Yedidia said he learned that FTX customer deposits had been used to pay loans to creditors. He
said Alameda\'s actions seemed "flagrantly wrong." Yedidia\'s testimony ended on a fiery note, which was later struck from the record. He was asked why he had
lost faith in FTX and resigned. "FTX defrauded all its customers," he said.', 'publisher': 'CNBC', 'page': "Sex, Signal messages, and sabotaging FTX's code: SBF
execs and Bahamas roommates tell all in court (Cnbc | World Business News Leader, 2023-10-06T20:17:00+00:00)", 'article': "Sex, Signal messages, and sabotaging
FTX's code: SBF execs and Bahamas roommates tell all in court"}, {'passage': 'Sex, Signal messages, and sabotaging FTX\'s code: SBF execs and Bahamas roommates
tell all in court (Cnbc | World Business News Leader, 2023-10-06T20:17:00+00:00)\n"FTX defrauded all its customers," he said.\n\nMatt Huang, co-founder of
Paradigm Operations LP, right, arrives at court in New York, US, on Thursday, Oct. 5, 2023. Former FTX Co-Founder Sam Bankman-Fried is charged with seven counts
of fraud and money laundering following the collapse of his cryptocurrency empire last year. Photographer: Yuki Iwamura/Bloomberg via Getty Images Yuki Iwamura
| Bloomberg | Getty Images\n\nInvestment to zero\n\nThe third witness to take the stand was Matt Huang, co-founder and managing partner of Paradigm, a crypto
venture capital firm that invested over $275 million in FTX. That stake was wiped out. Huang testified about his firm\'s due diligence on FTX, and he told the
court that Bankman-Fried assured him that funds would be used for FTX and not Alameda.', 'publisher': 'CNBC', 'page': "Sex, Signal messages, and sabotaging
FTX's code: SBF execs and Bahamas roommates tell all in court (Cnbc | World Business News Leader, 2023-10-06T20:17:00+00:00)", 'article': "Sex, Signal messages,
and sabotaging FTX's code: SBF execs and Bahamas roommates tell all in court"}, {'passage': "Sex, Signal messages, and sabotaging FTX's code: SBF execs and
Bahamas roommates tell all in court (Cnbc | World Business News Leader, 2023-10-06T20:17:00+00:00)\nAdditionally, he was promised that Alameda had no
preferential treatment on the FTX platform, even though the hedge fund was one of its top traders. Huang said he was concerned about FTX's lack of a board of
directors, but he eventually invested anyway. During cross-examination, Huang said Paradigm pressed Bankman-Fried on the board issue and was told he didn't want
investors as directors but he did plan on having a board with experts. — CNBC's Dawn Giel contributed to this report.", 'publisher': 'CNBC', 'page': "Sex,
Signal messages, and sabotaging FTX's code: SBF execs and Bahamas roommates tell all in court (Cnbc | World Business News Leader, 2023-10-06T20:17:00+00:00)",
'article': "Sex, Signal messages, and sabotaging FTX's code: SBF execs and Bahamas roommates tell all in court"}, {'passage': 'How is it still getting worse for
Sam Bankman-Fried? (The Verge, 2023-10-12T23:08:14+00:00)\nIn the break after Caroline Ellison stepped down from the stand, Barbara Fried engaged defense lawyer
Christian Everdell in an animated conversation. Fried, the defendant’s mother, was gesticulating and clearly had a strong opinion about something. Everdell
walked off, and Mark Cohen talked to her for a bit after that.\n\nFried seemed frustrated, and I couldn’t blame her. The defense absolutely biffed the
cross-examination of Ellison and, to make matters worse was unable to keep a recording of an all-hands meeting where Ellison confessed to taking customer funds
from being played for the jury. Is this really the best the defense can do?\n\nIn Cohen’s disorganized cross-examination, he mostly bored the jury\n\nBefore
this case, I had been told that Everdell and Cohen were “workman-like,” which I took to mean that they were unshowy but competent. I now believe that comment
was an insult. I have been waiting for a juicy cross-examination, as I live for chaos and drama.', 'publisher': 'The Verge', 'page': 'How is it still getting
worse for Sam Bankman-Fried? (The Verge, 2023-10-12T23:08:14+00:00)', 'article': 'How is it still getting worse for Sam Bankman-Fried?'}, {'passage': 'How is it
still getting worse for Sam Bankman-Fried? (The Verge, 2023-10-12T23:08:14+00:00)\nI have been waiting for a juicy cross-examination, as I live for chaos and
drama. I am beginning to think I am not going to get one.\n\nEllison had given, in her direct testimony, fairly damning evidence tying FTX CEO Sam Bankman-Fried
to the conspiracy to take FTX customer funds. There were fake balance sheets, one of which was sent to crypto lender Genesis. After a Genesis representative
received the balance sheet, he texted Ellison to tell her he’d spoken to Bankman-Fried — strongly suggesting that Bankman-Fried was aware of the contents of the
fake balance sheet. Not great!\n\nSam Bankman-Fried, the founder of failed cryptocurrency exchange FTX, is on trial for seven counts of wire fraud and
conspiracy. FTX was a fraud “from the start,” the Securities and Exchange Commission alleges — with a “multi-billion-dollar deficiency caused by his own
misappropriation of customer funds.” Follow along for all the latest news and regular updates from the trial.', 'publisher': 'The Verge', 'page': 'How is it
still getting worse for Sam Bankman-Fried? (The Verge, 2023-10-12T23:08:14+00:00)', 'article': 'How is it still getting worse for Sam Bankman-Fried?'},
{'passage': 'How is it still getting worse for Sam Bankman-Fried? (The Verge, 2023-10-12T23:08:14+00:00)\nFollow along for all the latest news and regular
updates from the trial.\n\nBut a lot of testimony relied on Ellison recounting conversations she’d had in person, or on auto-deleting text messaging platforms.
This gave the defense an opportunity to try to make her sound unreliable. After all, she had an incentive to flip on Bankman-Fried: the possibility of leniency
in her sentencing. Given her fun tweets about speed, the fact that she was Bankman-Fried’s ex-girlfriend, and that she’d apparently written a bunch of stuff
down, I was expecting fireworks. For the first time in this trial, maybe the defense had an opening.\n\nInstead, I got a sad trombone. In Cohen’s disorganized
cross-examination, he mostly bored the jury. At one point, two different jurors appeared to be asleep.\n\nMidway through the morning I began wondering if there
was a mercy rule for cross-examinations. Prosecutor Danielle Sassoon had run an effective direct examination, creating an easy-to-follow narrative.',
'publisher': 'The Verge', 'page': 'How is it still getting worse for Sam Bankman-Fried? (The Verge, 2023-10-12T23:08:14+00:00)', 'article': 'How is it still
getting worse for Sam Bankman-Fried?'}, {'passage': 'How is it still getting worse for Sam Bankman-Fried? (The Verge, 2023-10-12T23:08:14+00:00)\nProsecutor
Danielle Sassoon had run an effective direct examination, creating an easy-to-follow narrative. By contrast, Cohen appeared to be bumbling around, taking up one
topic only to abruptly pivot. Sure, we’re still in the prosecution’s case, but Cohen had all night to prepare his lines of questioning.\n\nApparently, Alameda
had a problem with retaining accountants\n\nWe established that Bankman-Fried had a much larger appetite for risk than Ellison. I thought perhaps it might be
building to something, but this line of questioning was quickly dropped. We established that Bankman-Fried and Ellison reacted differently to stress, and that
they also had different approaches to media: namely, that Ellison avoided it while Bankman-Fried sought it out. Okay?\n\nWe discovered that there was one
accountant at Alameda in 2021, and two more junior accountants were hired in 2022. Apparently, Alameda had a problem with retaining accountants, which didn’t
surprise me much; CEOs generally don’t do the balance sheets for their companies.', 'publisher': 'The Verge', 'page': 'How is it still getting worse for Sam
Bankman-Fried? (The Verge, 2023-10-12T23:08:14+00:00)', 'article': 'How is it still getting worse for Sam Bankman-Fried?'}, {'passage': 'How is it still getting
worse for Sam Bankman-Fried? (The Verge, 2023-10-12T23:08:14+00:00)\nI was ready to hear this pursued further — but then it, too, was dropped.\n\nI think the
defense was also trying to suggest that the government had coerced Ellison’s testimony, by suggesting that she had pleaded guilty to a charge of defrauding
investors that she couldn’t have been involved in. After all, she didn’t prepare materials for them. Unfortunately, she did say that she had conversations with
investors as part of their due diligence — and, of course, Alameda was taking on losses from FTX to keep FTX’s balance sheet pristine. This line of questioning
felt like a waste of time.\n\nThere were rather a lot of sidebars during the cross-examination, to the point that when one occurred, several jurors looked
entertained. There were a few yesterday, too, including one in which the prosecution complained that Bankman-Fried was visibly scoffing at Ellison’s answers,
according to the court transcript.', 'publisher': 'The Verge', 'page': 'How is it still getting worse for Sam Bankman-Fried? (The Verge,
2023-10-12T23:08:14+00:00)', 'article': 'How is it still getting worse for Sam Bankman-Fried?'}, {'passage': 'How is it still getting worse for Sam
Bankman-Fried? (The Verge, 2023-10-12T23:08:14+00:00)\n(I did observe him occasionally shaking his head, and sometimes quivering at points during her testimony,
but didn’t have a view of his face.)\n\nAt one point, Ellison appeared to even be toying with Cohen\n\nAs we were approaching lunch, several jurors looked
annoyed, and Cohen looked clueless. He asked Ellison to define what “buy on the way down” meant, as though it were a term of art. (It means what you think it
means, to purchase an asset that’s losing value.) This seemed to puzzle her. At other times, Cohen seemed to forget what she had testified to, bringing up
things she hadn’t said. I don’t know if this was an attempt to trap her in a lie or just poor preparation, but much like FTX employee Adam Yedidia before her,
Ellison was fastidious about making sure a question was clear and her answer was precise.\n\nAt one point, Ellison appeared to even be toying with Cohen. She’d
testified on direct about Luna, a cryptocurrency token.', 'publisher': 'The Verge', 'page': 'How is it still getting worse for Sam Bankman-Fried? (The Verge,
2023-10-12T23:08:14+00:00)', 'article': 'How is it still getting worse for Sam Bankman-Fried?'}, {'passage': 'How is it still getting worse for Sam
Bankman-Fried? (The Verge, 2023-10-12T23:08:14+00:00)\nShe’d testified on direct about Luna, a cryptocurrency token. It had a sister token, Terra, that was a
paired algorithmic stablecoin. (If you don’t understand what that means, it doesn’t matter, because they were both nonsense.) Cohen asked her about Terra/Luna
and she pointed out she’d only spoken about Luna, leaving him to fumble about how to explain the relationship between the two tokens. She kept a straight face
on the stand while I chuckled from the press seats.\n\nDuring the opening statement, Cohen had blamed Ellison for not taking out a hedge on some of the risks
Alameda was trading. We heard more about this hedge in cross-examination, and friends, it was stupid. Forget evaluating the trade itself (hedging being long
crypto by selling Nasdaq futures). Was she supposed to have taken more customer funds to put on the hedge? Was that the defense, that she didn’t take enough of
them? Was it that she should have taken them sooner? What the fuck?', 'publisher': 'The Verge', 'page': 'How is it still getting worse for Sam Bankman-Fried?
(The Verge, 2023-10-12T23:08:14+00:00)', 'article': 'How is it still getting worse for Sam Bankman-Fried?'}, {'passage': 'How is it still getting worse for Sam
Bankman-Fried? (The Verge, 2023-10-12T23:08:14+00:00)\nWas it that she should have taken them sooner? What the fuck?\n\nWhen Sassoon got up for a quick
redirect, she demolished any points Cohen had attempted to make. But I didn’t really appreciate her cleverness until after Ellison left the stand, and the
jurors left the room. She’d managed to set a neat little trap for Cohen.\n\nIn the recording, Ellison did indeed confess to stealing customer funds with
Bankman-Fried’s approval\n\nOn the direct examination, near the end, Sassoon asked about an Alameda all-hands meeting, without bringing up many specifics.
During the cross, Cohen asked Ellison what topics were covered in the meeting, while avoiding details. That opened the door for Sasson on redirect to work in
that Ellison had confessed to stealing billions of FTX customers’ money, at Bankman-Fried’s direction.\n\nThere had been an open question of whether jurors
would hear the tapes of Ellison’s remarks. The testimony set up an argument for the prosecution to bring in those tapes.', 'publisher': 'The Verge', 'page':
'How is it still getting worse for Sam Bankman-Fried? (The Verge, 2023-10-12T23:08:14+00:00)', 'article': 'How is it still getting worse for Sam
Bankman-Fried?'}, {'passage': 'How is it still getting worse for Sam Bankman-Fried? (The Verge, 2023-10-12T23:08:14+00:00)\nThe testimony set up an argument for
the prosecution to bring in those tapes. The judge ruled in favor of the jury hearing the recordings and we briefly recessed.\n\nThat was when Bankman-Fried’s
mother approached the defense lawyers.\n\nThe late afternoon was short and snappy. Christian Drappi, a former Alameda software engineer who looked like a
handsome funeral director in a black suit and tie, testified briefly to set up the tape. When Changpeng Zhao, the CEO of Binance, announced on Twitter that he
intended to acquire FTX, Ellison confessed the theft of customer funds to him and a few other employees, Drappi said. The all-hands took place the following
day, and was secretly recorded by an employee who’d joined Alameda three days before.\n\nIn the recording, Ellison did indeed confess to stealing customer funds
with Bankman-Fried’s approval. Drappi said he resigned less than 24 hours after the meeting. The recording was later sent to Drappi, who sent it to the
government.', 'publisher': 'The Verge', 'page': 'How is it still getting worse for Sam Bankman-Fried? (The Verge, 2023-10-12T23:08:14+00:00)', 'article': 'How
is it still getting worse for Sam Bankman-Fried?'}, {'passage': 'How is it still getting worse for Sam Bankman-Fried? (The Verge,
2023-10-12T23:08:14+00:00)\nThe recording was later sent to Drappi, who sent it to the government. Joseph Bankman, the defendant’s father and a senior advisor
to FTX’s philanthropic arm, wasn’t in the room for the recording. Barbara Fried looked unhappy, rubbing her left temple as though she had a tremendous
headache.\n\nAnd in a small miracle of pacing, the government got Zac Prince, the founder and former CEO of crypto lender BlockFi, on the stand just long enough
to blame FTX’s bankruptcy for BlockFi’s subsequent bankruptcy before we broke for the day.\n\nI’ve been asked by some people why the reporting coming out of the
trial seems so skewed toward the prosecution. There’s an answer for this: the prosecution has put together a strong, comprehensive indictment of Bankman-Fried’s
behavior at FTX and Alameda. The defense, so far, has managed to do almost nothing in response.', 'publisher': 'The Verge', 'page': 'How is it still getting
worse for Sam Bankman-Fried? (The Verge, 2023-10-12T23:08:14+00:00)', 'article': 'How is it still getting worse for Sam Bankman-Fried?'}, {'passage': 'How is it
still getting worse for Sam Bankman-Fried? (The Verge, 2023-10-12T23:08:14+00:00)\nThe defense, so far, has managed to do almost nothing in response. I don’t
know if Cohen and Everdell have bad facts, a bad client, or are simply untalented themselves (or some combination of all three?). But I do know that I haven’t
yet seen any good reason to doubt the very convincing story I’m hearing from the prosecution.', 'publisher': 'The Verge', 'page': 'How is it still getting worse
for Sam Bankman-Fried? (The Verge, 2023-10-12T23:08:14+00:00)', 'article': 'How is it still getting worse for Sam Bankman-Fried?'}, {'passage': "Coinbase
rallies more than 60% in same month that FTX and Binance founders brace for prison (Cnbc | World Business News Leader, 2023-11-30T21:08:00+00:00)\nIn this
article COIN Follow your favorite stocks CREATE FREE ACCOUNT\n\nBrian Armstrong, CEO of Coinbase, slammed the U.S. Securities and Exchange Commission. He also
said the cryptocurrency exchange is looking to invest more outside of the U.S. Carlos Jasso | Bloomberg | Getty Images\n\nIn a month that saw two of the crypto
industry's leading figures headed on the path to prison, Coinbase shares rocketed more than 60%, their second-best monthly performance since the cryptocurrency
exchange went public in 2021. Bolstered by rallies in bitcoin and ether as well as crises at key competitors, Coinbase has been one of Wall Street's best bets
all year, climbing more than 250% in the first 11 months of 2023.", 'publisher': 'CNBC', 'page': 'Coinbase rallies more than 60% in same month that FTX and
Binance founders brace for prison (Cnbc | World Business News Leader, 2023-11-30T21:08:00+00:00)', 'article': 'Coinbase rallies more than 60% in same month that
FTX and Binance founders brace for prison'}, {'passage': 'Coinbase rallies more than 60% in same month that FTX and Binance founders brace for prison (Cnbc |
World Business News Leader, 2023-11-30T21:08:00+00:00)\nFor early holders of the stock, the rebound helps ease the pain of 2022, when Coinbase lost 86% of its
value as soaring inflation and rising interest rates pushed investors out of crypto and high-growth tech companies, and into assets deemed safer in a recession.
Tech stocks have roared back this year, particularly those tied to the artificial intelligence boom and crypto. Coinbase has the added benefit of having
survived the so-called crypto winter, while so many of its rivals disappeared or downsized. The industry fallout came to a head this month, when Sam
Bankman-Fried, founder of former Coinbase rival FTX, was found guilty of seven criminal fraud counts tied to the collapse of his exchange and the theft of
customer funds. His conviction landed on Nov. 2 after a monthlong trial. Less than three weeks later, on Nov.', 'publisher': 'CNBC', 'page': 'Coinbase rallies
more than 60% in same month that FTX and Binance founders brace for prison (Cnbc | World Business News Leader, 2023-11-30T21:08:00+00:00)', 'article': 'Coinbase
rallies more than 60% in same month that FTX and Binance founders brace for prison'}, {'passage': "Coinbase rallies more than 60% in same month that FTX and
Binance founders brace for prison (Cnbc | World Business News Leader, 2023-11-30T21:08:00+00:00)\n2 after a monthlong trial. Less than three weeks later, on
Nov. 21, Binance founder Changpeng Zhao pleaded guilty to violations of the Bank Secrecy Act for failing to implement an effective anti-money laundering program
and for willfully violating U.S. economic sanctions.\n\nCombination showing Former FTX CEO, Sam Bankman-Fried (L) and Zhao Changpeng (R), founder and chief
executive officer of Binance. Getty Images | Reuters\n\nBankman-Fried, who faces potential life behind bars, is scheduled to be sentenced in March. Zhao's
sentencing is set for February. While guidelines suggest a sentence of 12 to 18 months, the Justice Department could push for a lengthier punishment for the
Binance founder. Unlike FTX, which filed for bankruptcy in late 2022, Binance is still standing, though now without Zhao, who agreed to step down as CEO as part
of the plea deal.", 'publisher': 'CNBC', 'page': 'Coinbase rallies more than 60% in same month that FTX and Binance founders brace for prison (Cnbc | World
Business News Leader, 2023-11-30T21:08:00+00:00)', 'article': 'Coinbase rallies more than 60% in same month that FTX and Binance founders brace for prison'},
{'passage': 'Coinbase rallies more than 60% in same month that FTX and Binance founders brace for prison (Cnbc | World Business News Leader,
2023-11-30T21:08:00+00:00)\nEven before that, the company was seeing a plunge in trading, with volume down by two-thirds between the first and third quarters of
the year, according to crypto analyst site CoinGecko. With assets of more than $65 billion on the platform, Binance remains the world\'s largest crypto exchange
globally. But its market share fell from over 60% in February to under 50% in September, "an indication that the exchange may be losing its grip on the industry
as regulators continue to pressure it," CoinGecko said. In the first 24 hours after the Justice Department announced its $4.3 billion settlement with Binance,
customers pulled more than $1 billion from the exchange. Liquidity also dropped 25% in the immediate aftermath of the announcement as market makers pulled back
their positions, according to data provider Kaiko. A Binance spokesperson told CNBC in a statement that Zhao appeared in court "to protect our users and to
ensure the longevity of our company."', 'publisher': 'CNBC', 'page': 'Coinbase rallies more than 60% in same month that FTX and Binance founders brace for
prison (Cnbc | World Business News Leader, 2023-11-30T21:08:00+00:00)', 'article': 'Coinbase rallies more than 60% in same month that FTX and Binance founders
brace for prison'}, {'passage': 'Coinbase rallies more than 60% in same month that FTX and Binance founders brace for prison (Cnbc | World Business News Leader,
2023-11-30T21:08:00+00:00)\n"Binance\'s resilience has been tested unlike any other exchange around today," the spokesperson said. "Yet, we continue to operate
the world\'s largest cryptocurrency exchange by volume. In fact, we currently see a climbing percentage of institutional user transactions." Coinbase is the
fourth-biggest global exchange by daily volume, according to CoinGecko. It\'s the only one that\'s publicly traded in the U.S. and has a market cap of close $30
billion. In a report to clients on Wednesday, analysts at Mizuho noted that Coinbase shares are up about 20% since Zhao\'s settlement, a rally that\'s likely
"in anticipation of potential share gains for COIN in wake of outflows from Binance, the industry\'s largest exchange," they wrote. Coinbase shares fell 2.4% to
$124.72 on Thursday, wiping out some of their recent gains.', 'publisher': 'CNBC', 'page': 'Coinbase rallies more than 60% in same month that FTX and Binance
founders brace for prison (Cnbc | World Business News Leader, 2023-11-30T21:08:00+00:00)', 'article': 'Coinbase rallies more than 60% in same month that FTX and
Binance founders brace for prison'}, {'passage': 'Coinbase rallies more than 60% in same month that FTX and Binance founders brace for prison (Cnbc | World
Business News Leader, 2023-11-30T21:08:00+00:00)\nMizuho raised its price target on the stock to $35 from $31, while keeping its underperform rating, which
it\'s maintained since December.\n\n\'Turn the page\'\n\nA Coinbase spokesperson declined to comment for this story, but CEO Brian Armstrong told CNBC\'s
Joumanna Bercetche earlier this week that the Binance settlement allows the crypto industry to move past a spate of scandals. "The enforcement action against
Binance, that\'s allowing us to kind of turn the page on that and hopefully close that chapter of history," Armstrong said. "I think that regulatory clarity is
going to help bring in more investment, especially from institutions."\n\nwatch now\n\nBoth Coinbase and Binance still face legal battles with the Securities
and Exchange Commission, which was noticeably absent from the Binance settlement. Meanwhile, Coinbase executives have floated the idea of leaving the U.S.
altogether for a jurisdiction with hard-and-fast rules on crypto, should the company be unable to come to a resolution with the SEC.', 'publisher': 'CNBC',
'page': 'Coinbase rallies more than 60% in same month that FTX and Binance founders brace for prison (Cnbc | World Business News Leader,
2023-11-30T21:08:00+00:00)', 'article': 'Coinbase rallies more than 60% in same month that FTX and Binance founders brace for prison'}, {'passage': 'Coinbase
rallies more than 60% in same month that FTX and Binance founders brace for prison (Cnbc | World Business News Leader, 2023-11-30T21:08:00+00:00)\nWall Street
appears to be shrugging off that concern. Analysts at Needham, who recommend buying Coinbase shares, wrote in a report on Nov. 21 that the company "exited the
crypto \'winter\' better positioned than in the prior up cycle." They also noted that in addition to FTX\'s failure and Binance\'s retreat, crypto trading
platform Bittrex has also exited the market. Bittrex said on Nov. 20, that effective Dec. 4, "all trading activity on Bittrex Global will be disabled," and it
encouraged customers "to log into their account and withdraw assets as soon as possible." In April, the SEC charged Bittrex and its ex-CEO with operating an
unregistered exchange. Yet there may be a new competitive threat on the horizon. U.S. regulators are expected to soon approve the first U.S.', 'publisher':
'CNBC', 'page': 'Coinbase rallies more than 60% in same month that FTX and Binance founders brace for prison (Cnbc | World Business News Leader,
2023-11-30T21:08:00+00:00)', 'article': 'Coinbase rallies more than 60% in same month that FTX and Binance founders brace for prison'}, {'passage': 'Coinbase
rallies more than 60% in same month that FTX and Binance founders brace for prison (Cnbc | World Business News Leader, 2023-11-30T21:08:00+00:00)\nU.S.
regulators are expected to soon approve the first U.S. spot bitcoin exchange-traded funds, which would allow investors to buy into digital currency directly
through the same mechanism they use to buy stock and bond ETFs. Top asset managers, including BlackRock , WisdomTree and Invesco, have filed applications with
the SEC. Regulatory approval would open up many more avenues for people to buy bitcoin. While Coinbase allows investors to buy a variety of cryptocurrencies,
bitcoin accounted for 38% of transaction volume in the third quarter and almost the same percentage of revenue. For casual investors who just want some exposure
to bitcoin, there will potentially be additional ways to buy, including through their primary online brokerage. JPMorgan Chase analysts wrote last week that
crypto ETFs would likely be good for Coinbase in the short term but more problematic as time passes. The initial boost would come from custody revenue tied to
the ETFs.', 'publisher': 'CNBC', 'page': 'Coinbase rallies more than 60% in same month that FTX and Binance founders brace for prison (Cnbc | World Business
News Leader, 2023-11-30T21:08:00+00:00)', 'article': 'Coinbase rallies more than 60% in same month that FTX and Binance founders brace for prison'}, {'passage':
'Coinbase rallies more than 60% in same month that FTX and Binance founders brace for prison (Cnbc | World Business News Leader, 2023-11-30T21:08:00+00:00)\nThe
initial boost would come from custody revenue tied to the ETFs. Most of the big asset managers jumping into market, including BlackRock, Franklin Templeton and
WisdomTree, have picked Coinbase for custody services, which involves the storage and safekeeping of the assets. However, the longer-term concern, according to
JPMorgan, is that fewer people will need Coinbase accounts, leading to pricing pressure. "We see many novice investors never going beyond these flagship tokens
and thus never needing the services of a Coinbase," wrote the analysts, who have a neutral rating on the stock and an $80 price target. "We also see the ETF
markets as more transparent, efficient and lower cost to execute and we see the potential for a migration to ETFs for cheaper exposure and trading driving
Coinbase to lower fees." WATCH: Former SEC enforcement chief on \'casualness\' in crypto compliance', 'publisher': 'CNBC', 'page': 'Coinbase rallies more than
60% in same month that FTX and Binance founders brace for prison (Cnbc | World Business News Leader, 2023-11-30T21:08:00+00:00)', 'article': 'Coinbase rallies
more than 60% in same month that FTX and Binance founders brace for prison'}]
passages: [{'passage': 'SBF Trial: The latest updates from the FTX collapse’s courtroom drama (TechCrunch, 2023-10-06T17:22:02+00:00)\nSBF trial: Everything to
know from the FTX courtroom ahead of his testimony Sam Bankman-Fried is in court for alleged fraud and money laundering while leading crypto exchange FTX and
sister company Alameda. Here’s our latest coverage as the case nears its end.\n\nThe trial of Bankman-Fried, the former CEO and co-founder of collapsed crypto
exchange FTX, is entering its next chapter as proceedings resume Thursday. As confirmed during a teleconference Wednesday, SBF will be taking the stand, along
with a handful of other witnesses as the onus shifts toward the defense after the prosecution brought its case to its conclusion.\n\nAs former federal
prosecutor Josh Naftalis told TechCrunch, SBF taking to the stand in his own trial is a “Hail Mary” ahead of a verdict and potential sentencing for seven counts
tied to fraud and money laundering.\n\n“Once it goes to cross[-examination], he doesn’t get to say, ‘I’m done,’” Naftalis said.', 'page': 'SBF Trial: The latest
updates from the FTX collapse’s courtroom drama (TechCrunch, 2023-10-06T17:22:02+00:00)'}, {'passage': 'SBF’s trial starts soon, but how did he — and FTX — get
here? (TechCrunch, 2023-10-01T14:00:29+00:00)\nThat means it purported to have more FTT tokens on its balance sheet than what existed in the world.\n\nAround
the same time it was exposed, the world’s largest crypto exchange, Binance, started pulling out its remaining $2.1 billion equivalent of cash in BUSD and FTT.
(It had an equity position in FTX from 2019 to 2021.) This essentially triggered a bank run on FTX.\n\nFTX and Alameda filed for Chapter 11 bankruptcy in the
U.S. mid-November 2022. Bankman-Fried resigned, and John J. Ray III, the Enron turnaround veteran, was appointed its new CEO.\n\nBankman-Fried, however,
maintained his innocence. At The New York Times’ DealBook Summit, he appeared virtually from the Bahamas, saying “I didn’t ever try to commit fraud on anyone; I
was shocked by what happened this month.”', 'page': 'SBF’s trial starts soon, but how did he — and FTX — get here? (TechCrunch, 2023-10-01T14:00:29+00:00)'},
{'passage': 'SBF’s trial starts soon, but how did he — and FTX — get here? (TechCrunch, 2023-10-01T14:00:29+00:00)\nU.S. Attorney Damian Williams called
Bankman-Fried’s alleged crimes “one of the biggest financial frauds in American history,” in a press conference.\n\nThe aftermath\n\nFTX co-founder and former
CTO Gary Wang, and Alameda Research’s former CEO, Caroline Ellison, both pleaded guilty in December 2022 to federal criminal charges in relation to the FTX
collapse. They’re also facing civil penalties from the U.S. Securities and Exchange Commission (SEC) and Commodity Futures Trading Commission (CFTC) alongside
the criminal charges. Wang and Ellison plan to cooperate with prosecutors and will be major witnesses in the trial, given their close ties to Bankman-Fried, FTX
and Alameda.\n\nIn January, Bankman-Fried pleaded not guilty to all counts, which include wire fraud, conspiracy to commit money laundering and conspiracy to
misuse customer funds. He could face up to 115 years in jail if convicted on all charges.', 'page': 'SBF’s trial starts soon, but how did he — and FTX — get
here? (TechCrunch, 2023-10-01T14:00:29+00:00)'}, {'passage': 'SBF’s trial starts soon, but how did he — and FTX — get here? (TechCrunch,
2023-10-01T14:00:29+00:00)\nHe could face up to 115 years in jail if convicted on all charges.\n\nThe crypto industry as a whole suffered from FTX’s collapse,
which was the first of many. BlockFi filed for Chapter 11 in November 2022, as did Genesis Global Trading in January.\n\nWhere we are today\n\nBankman-Fried
will be represented by Cohen & Gresser, and Mark Cohen, a high-profile defense attorney and former federal prosecutor, will be the lead attorney. If that name
sounds familiar, it might be because he also represented Ghislaine Maxwell in her sex trafficking trial related to Jeffrey Epstein. He requested an early
release for Bankman-Fried but was denied.\n\nWith the trial beginning, we’re witnessing how FTX’s story ends. But what’s on our mind is what happens to the
investors and creditors affected by the collapse? And what happens to the billions in crypto assets tied up in legal proceedings?', 'page': 'SBF’s trial starts
soon, but how did he — and FTX — get here? (TechCrunch, 2023-10-01T14:00:29+00:00)'}, {'passage': 'SBF, riding high on FTX, reportedly offered $5B to Trump to
cancel President re-run (TechCrunch, 2023-10-02T13:11:07+00:00)\nCome Tuesday, our eyes will be trained on the Manhattan Federal Court for the trial of Sam
Bankman-Fried, the disgraced cryptocurrency entrepreneur accused of orchestrating “one of the biggest financial frauds in American history”: the
multibillion-dollar collapse of FTX, the crypto exchange he founded and led.\n\nSBF, as he’s known, has pleaded not-guilty to some seven charges of fraud and
conspiracy. But, as we wait to hear both his version and the government’s version of what happened, there will be another tale recounting how things played out:
“Going Infinite,” a book by Michael Lewis chronicling the rise and fall of SBF, is set to debut on the first day of the trial.\n\nThe episode has fired up a lot
of people, indignant at what they feel was too-easy treatment for someone who many believe brought down the whole crypto house.\n\n60 Minutes got rightly
community noted here.', 'page': 'SBF, riding high on FTX, reportedly offered $5B to Trump to cancel President re-run (TechCrunch, 2023-10-02T13:11:07+00:00)'},
{'passage': 'Is Sam Bankman-Fried a bad ‘man’ or a good ‘boy’? Lawyers swap opening statements before first witnesses take the stand (Fortune,
2023-10-04T23:32:53+00:00)\nWho is Sam Bankman-Fried, the former CEO of crypto exchange FTX? Is he a liar and a fraud? Or did he act in good faith, and like
many a startup founder, fly too close to the sun?\n\nThat answer ultimately lies with the jury, which was selected on Wednesday morning before lawyers for the
government and then Bankman-Fried swapped two very different stories of the former crypto mogul’s sudden rise and almost instantaneous fall.\n\nHere’s what
happened on the second day of the trial, which featured pointed allegations, a friend from MIT, and an audience replete with big names, including
Bankman-Fried’s professorial parents and Damian Williams, U.S. attorney for the Southern District of New York.\n\nA conscious criminal…\n\nThe prosecution’s
account of the alleged crimes by Bankman-Fried, who spent most of his day in court staring into a laptop while seated between his two attorneys, featured a
study in contrasts.', 'page': 'Is Sam Bankman-Fried a bad ‘man’ or a good ‘boy’? Lawyers swap opening statements before first witnesses take the stand (Fortune,
2023-10-04T23:32:53+00:00)'}, {'passage': 'Is Sam Bankman-Fried a bad ‘man’ or a good ‘boy’? Lawyers swap opening statements before first witnesses take the
stand (Fortune, 2023-10-04T23:32:53+00:00)\n“One year ago, it looked like the defendant was on the top of the world,” began Thane Rehn, a prosecutor for the
government, in his opening statement. The former CEO of FTX oversaw a supposedly thriving crypto exchange, jetted between international locales, and hobnobbed
with celebrities like Tom Brady and Larry David. He repeatedly emphasized to customers that their money was safe and secure.\n\nBut “all of that, all of it, was
built on lies,” Rehn declared to the jury. “Behind the curtains, he was not what he appeared to be.” What followed was a roughly 30-minute story that repeatedly
emphasized how Bankman-Fried allegedly stole customer funds to facilitate his jet-setting lifestyle, donate millions to political candidates, and finance risky
bets.\n\nThe key to his alleged scheme? Alameda Research, a crypto hedge fund he also owned, argued Rehn.', 'page': 'Is Sam Bankman-Fried a bad ‘man’ or a good
‘boy’? Lawyers swap opening statements before first witnesses take the stand (Fortune, 2023-10-04T23:32:53+00:00)'}, {'passage': 'Is Sam Bankman-Fried a bad
‘man’ or a good ‘boy’? Lawyers swap opening statements before first witnesses take the stand (Fortune, 2023-10-04T23:32:53+00:00)\nAlameda Research, a crypto
hedge fund he also owned, argued Rehn. Using Caroline Ellison, his on-again off-again girlfriend and the CEO of Alameda, as a front, Bankman-Fried had “secret
access” to customer money—both cash and crypto—the government claimed.\n\nMoreover, Bankman-Fried allegedly directed employees to conceal the flow of money into
FTX’s coffers and forged financial documents distributed to lenders and investors. “The defendant lied to the world,” Rehn alleged.\n\nAnd who was this
defendant? Not a crypto “boy” genius, as so many in the media (Fortune included) have written, but a “man” who “stole billions of dollars from thousands of
victims,” Rehn said. “You will see the full picture.”\n\n…or a well-meaning founder?', 'page': 'Is Sam Bankman-Fried a bad ‘man’ or a good ‘boy’? Lawyers swap
opening statements before first witnesses take the stand (Fortune, 2023-10-04T23:32:53+00:00)'}, {'passage': 'Is Sam Bankman-Fried a bad ‘man’ or a good ‘boy’?
Lawyers swap opening statements before first witnesses take the stand (Fortune, 2023-10-04T23:32:53+00:00)\n“You will see the full picture.”\n\n…or a
well-meaning founder?\n\nBut Bankman-Fried, whose cheekbones were more prominent after spending about seven weeks in a Brooklyn prison, was no liar, according
to Mark Cohen, one of his lawyers. “Sam didn’t defraud anyone,” he said early on in his opening statement.\n\nWhat the jury will see is a nerdy startup founder
who acted in “good faith,” not the prosecution’s “cartoon of a villain.” (Cohen repeatedly harped on Bankman-Fried’s allegedly good-faith actions throughout his
address to the jury.)\n\nAlameda was not subterranean or shady. It was a successful hedge fund, he said. FTX was no Ponzi scheme. It was a “very innovative,
successful company.”', 'page': 'Is Sam Bankman-Fried a bad ‘man’ or a good ‘boy’? Lawyers swap opening statements before first witnesses take the stand
(Fortune, 2023-10-04T23:32:53+00:00)'}, {'passage': 'Is Sam Bankman-Fried a bad ‘man’ or a good ‘boy’? Lawyers swap opening statements before first witnesses
take the stand (Fortune, 2023-10-04T23:32:53+00:00)\nFTX was no Ponzi scheme. It was a “very innovative, successful company.” And the business practices between
the two were reasonable, he argued, claiming that Alameda acted legally as an FTX customer, payment processor, and market maker, or financial entity that acts
as a trading partner for customers looking to buy and sell cryptocurrencies.\n\nIn an analogy he employed throughout his opening statement, he said that
“working at a startup is like building a plane as you’re flying it” and that businesses sometimes fail. In fact, he specifically pointed the finger at Ellison,
the former CEO of Alameda, who, he said, did not adequately protect her hedge fund from the inherent risk of the crypto markets.\n\nWhen the walls came closing
in and the aforementioned plane approached the “eye of the storm,” Bankman-Fried did not act like someone who was guilty. Rather, he was willing to give up his
personal wealth to make customers whole, Cohen argued.', 'page': 'Is Sam Bankman-Fried a bad ‘man’ or a good ‘boy’? Lawyers swap opening statements before first
witnesses take the stand (Fortune, 2023-10-04T23:32:53+00:00)'}, {'passage': 'Is Sam Bankman-Fried a bad ‘man’ or a good ‘boy’? Lawyers swap opening statements
before first witnesses take the stand (Fortune, 2023-10-04T23:32:53+00:00)\nRather, he was willing to give up his personal wealth to make customers whole, Cohen
argued.\n\n“In the end, Sam started and built two billion-dollar businesses,” he concluded. “He didn’t steal any money.”\n\nA Frenchman who lives in London
testifies in New York\n\nAfter lawyers from both sides depicted two very different Bankman-Frieds, the prosecution called its first two witnesses to the
stand—and they weren’t blockbuster names or former lieutenants-turned-government-cooperators, like Ellison.\n\nThe first was a victim: Marc-Antoine Julliard, a
Paris-born cocoa trader who lives in London. In 2021, Julliard, who had coiffed hair and spoke with a strong French accent, decided to invest in crypto and
landed on FTX as his exchange of choice, where he traded cryptocurrencies like Bitcoin and Dogecoin.\n\nOn Nov. 8, in the crypto exchange’s final few days, he
tried to pull out his cash and crypto.', 'page': 'Is Sam Bankman-Fried a bad ‘man’ or a good ‘boy’? Lawyers swap opening statements before first witnesses take
the stand (Fortune, 2023-10-04T23:32:53+00:00)'}, {'passage': 'Is Sam Bankman-Fried a bad ‘man’ or a good ‘boy’? Lawyers swap opening statements before first
witnesses take the stand (Fortune, 2023-10-04T23:32:53+00:00)\nHow much? Almost $100,000, he said. And was he ever able to? “Never,” he told
prosecutors.\n\nShortly afterward, as the trial neared late afternoon, the government called Adam Yedidia to the stand. A fast-talking graduate of MIT, he and
Bankman-Fried were close friends in college, he said. And after Bankman-Fried left Jane Street, the high-frequency trading firm where the former billionaire got
his start in finance after MIT, he persuaded Yedidia to join him as a trader at Alameda and then as a developer at FTX.\n\nWhen Yedidia first took the stand,
Danielle Sassoon, one of the lead prosecutors, said that the college friend of Bankman-Fried had legal immunity during his testimony. Why did he make such a
deal with the government, she asked.\n\n“I was concerned that I had unwittingly written code that contributed to a crime,” he said.', 'page': 'Is Sam
Bankman-Fried a bad ‘man’ or a good ‘boy’? Lawyers swap opening statements before first witnesses take the stand (Fortune, 2023-10-04T23:32:53+00:00)'},
{'passage': 'Is Sam Bankman-Fried a bad ‘man’ or a good ‘boy’? Lawyers swap opening statements before first witnesses take the stand (Fortune,
2023-10-04T23:32:53+00:00)\n“I was concerned that I had unwittingly written code that contributed to a crime,” he said.\n\nSoon, however, the clock neared 4:30
p.m., and court adjourned for the day. Yedidia will continue his testimony on Thursday, followed by Matt Huang, a former partner at the high-powered venture
capital firm Sequoia Capital, and then Gary Wang, a key Bankman-Fried lieutenant and one of the government’s star witnesses.', 'page': 'Is Sam Bankman-Fried a
bad ‘man’ or a good ‘boy’? Lawyers swap opening statements before first witnesses take the stand (Fortune, 2023-10-04T23:32:53+00:00)'}, {'passage': 'The jury
finally hears from Sam Bankman-Fried (The Verge, 2023-10-28T00:12:41+00:00)\nIt is honestly kind of incredible to watch a man torpedo his own credibility on
direct testimony. We’re not even at the cross yet, and the judge has already instructed him to answer the question he’s being asked by his own lawyer.\n\nThe
jury is watching all of this intently.\n\nThe main thing that’s been clear so far from Bankman-Fried’s testimony is that the man really loves the sound of his
own voice. So far, the count of “Objection, narrative!” to Bankman-Fried’s answers, followed by “Sustained” is at three.\n\nAlso, sometimes when Bankman-Fried
says “we,” he only means himself.\n\nSam Bankman-Fried, the founder of failed cryptocurrency exchange FTX, has been found guilty on seven counts including
charges of wire fraud. FTX was a fraud “from the start,” the Securities and Exchange Commission alleged — with a “multi-billion-dollar deficiency caused by his
own misappropriation of customer funds.” Follow along for all the latest news and regular updates from the trial.', 'page': 'The jury finally hears from Sam
Bankman-Fried (The Verge, 2023-10-28T00:12:41+00:00)'}, {'passage': 'The jury finally hears from Sam Bankman-Fried (The Verge,
2023-10-28T00:12:41+00:00)\nFollow along for all the latest news and regular updates from the trial.\n\nYesterday, during an evidentiary hearing, Bankman-Fried
was repeatedly scolded by Judge Lewis Kaplan for not answering prosecutor Danielle Sassoon’s questions on cross-examination. Today, Bankman-Fried was scolded by
Kaplan for not answering his counsel Mark Cohen’s questions on direct examination. Bankman-Fried has also occasionally interrupted Cohen with “yes” and
“yup.”\n\nWe spent a lot of the morning explaining vocabulary. I will spare you the full list, though I will say that explaining “Amazon Web Services” and
“database” was a bit too detail-oriented. Then Bankman-Fried tried to define “market manipulation.” After Bankman-Fried gave his definition, Kaplan told the
jury that he was the final authority on that, thanks.\n\nIn fairness to Bankman-Fried, he has been clearer and much easier to understand than he was at the
evidentiary hearing yesterday. There was a minimum of word salad today.', 'page': 'The jury finally hears from Sam Bankman-Fried (The Verge,
2023-10-28T00:12:41+00:00)'}, {'passage': 'The jury finally hears from Sam Bankman-Fried (The Verge, 2023-10-28T00:12:41+00:00)\nThere was a minimum of word
salad today. I don’t know if he was more relaxed, or he’d just been more rehearsed, but I will certainly be watching to see if he suddenly becomes much less
coherent when Sassoon gets him for the cross.\n\nHere is the story of FTX, from his point of view.\n\nBankman-Fried, who informed us he’s “somewhat introverted,
naturally,” gave us a rather prolonged tour of his pre-Alameda Research life, which I will skip. In 2017, during a crypto bull run, he started his
cryptocurrency trading firm. He knew “basically nothing” about cryptocurrency at the time, he explained, but he wanted to do arbitrage on it anyway.\n\nAlameda
Research was named for Alameda County in California, which was where its first office was set up. As for its name, here’s what Bankman-Fried said on the
stand:\n\nEffectively, we wanted to be under the radar at that point in time.', 'page': 'The jury finally hears from Sam Bankman-Fried (The Verge,
2023-10-28T00:12:41+00:00)'}, {'passage': 'The jury finally hears from Sam Bankman-Fried (The Verge, 2023-10-28T00:12:41+00:00)\nI didn’t want to call it Sam’s
Crypto Trading Firm or anything like that. We — there are a lot of competitors and people who we didn’t particularly want to know what we were building out
because they would race to do it. “Research” was a sort of generic word, which filled out the company name. And that was — it was far better than the internal
name that we had at that point, which was Wireless Mouse.\n\nI would find this much more believable if I hadn’t already watched a video of Bankman-Fried
explaining on a podcast that the name made it easier to get a bank account. That happened during the first day of Gary Wang’s testimony. Bankman-Fried was
there, too. You know who else was there? The jury.\n\nAnyway, Bankman-Fried went on a hiring spree for Alameda. He rounded up his merry gang of alleged
co-conspirators. First, Wang, to program the computers. Then, Nishad Singh, about a month after founding Alameda Research.', 'page': 'The jury finally hears
from Sam Bankman-Fried (The Verge, 2023-10-28T00:12:41+00:00)'}, {'passage': 'The jury finally hears from Sam Bankman-Fried (The Verge,
2023-10-28T00:12:41+00:00)\nThen, Nishad Singh, about a month after founding Alameda Research. Finally, Caroline Ellison.\n\nNot having a risk team, when you
are any kind of financial anything, is certainly a choice\n\nThough Bankman-Fried was the CEO, and also the majority owner, he wanted to be clear: he did not
supervise Wang’s direct work. Anyway, after a bunch of wildly successful arbitrage — 50 percent to 100 percent annualized returns, per his testimony — he
decided to found a cryptocurrency exchange, FTX. He figured he’d fail; that there was only a 20 percent chance of success. Bankman-Fried did not define what he
expected the time period to be on this estimate, but arguably 20 percent was a much higher chance of success than FTX would enjoy once Alameda dipped into the
customer deposits.\n\nBy the way, because he was such a good guy, Bankman-Fried made a point of “periodically” handling support tickets himself.', 'page': 'The
jury finally hears from Sam Bankman-Fried (The Verge, 2023-10-28T00:12:41+00:00)'}, {'passage': 'The jury finally hears from Sam Bankman-Fried (The Verge,
2023-10-28T00:12:41+00:00)\n“I worried if I didn’t, I would lose touch with the actual concerns of the customers,” he testified. What he did not do was create a
risk team, which he is now characterizing as a “big mistake.”\n\nRisk is an inherent part of a futures exchange, which is even more like a casino than regular
cryptocurrency. Not having a risk team, when you are any kind of financial anything, is certainly a choice. It is especially a choice when you go around telling
everyone your crypto exchange is very good and safe.\n\nFTX’s big selling point was its “risk engine,” which was supposed to prevent big losses that would then
be spread around all the rest of the customers. But Bankman-Fried testified that in 2020, the “risk engine was effectively sagging under the weight” of the
exchange’s rapid growth. So its time to liquidation went up — it took minutes to determine which accounts needed to be liquidated.', 'page': 'The jury finally
hears from Sam Bankman-Fried (The Verge, 2023-10-28T00:12:41+00:00)'}, {'passage': 'The jury finally hears from Sam Bankman-Fried (The Verge,
2023-10-28T00:12:41+00:00)\nAs a result, at one point the risk engine got stuck in a catastrophic feedback loop that would have created losses in the “trillions
of dollars,” Bankman-Fried testified. As part of that feedback loop, Alameda teetered on the brink of liquidation, which “would have disastrous consequences”
for FTX.\n\n“At the time, I wasn’t entirely sure what was happening.”\n\nBecause of that experience, Bankman-Fried suggested an “alert” or “delay” that would
keep Alameda from being liquidated by a bug. This is the supposed origin story of “allow_negative,” which Bankman-Fried says was the eventual result of that
conversation, and that he says he didn’t know about until very recently.\n\nThere is a problem with this story. “Allow_negative” was coded and switched on in
2019. I saw the code in court, and so did Bankman-Fried, who was also there for the testimony. Perhaps you are wondering, was the jury also there? Reader, it
was.', 'page': 'The jury finally hears from Sam Bankman-Fried (The Verge, 2023-10-28T00:12:41+00:00)'}, {'passage': 'The jury finally hears from Sam
Bankman-Fried (The Verge, 2023-10-28T00:12:41+00:00)\nPerhaps you are wondering, was the jury also there? Reader, it was.\n\nBankman-Fried denied he knew about
the effectively infinite line of credit Alameda Research received from FTX. This argument was peculiar; essentially my take-away was that the CEO of a financial
company simply didn’t pay attention to finances.\n\nFTX couldn’t get bank accounts right away. Bankman-Fried anticipated it would take a year or two. Rather
than wait, he decided to use Alameda as the “payment provider” for bank transfers. “My understanding at the time was that there were teams managing the
process,” he said. “At the time, I wasn’t entirely sure what was happening.”\n\nWell, sure, understandable! He’s an introvert!\n\nBankman-Fried definitely did
not know that Singh, his employee, had backdated interest payments to get FTX “over the line” to $1 billion\n\nIn 2021, FTX was growing to millions of users,
with $1 billion of revenue.', 'page': 'The jury finally hears from Sam Bankman-Fried (The Verge, 2023-10-28T00:12:41+00:00)'}, {'passage': 'The jury finally
hears from Sam Bankman-Fried (The Verge, 2023-10-28T00:12:41+00:00)\nBankman-Fried said he worked 12 to 22 hours a day, and took one day off every couple of
months. Because FTX had grown so much, he could no longer run both companies, he said. Bankman-Fried handed the company off to Caroline Ellison and Sam
Trabucco, who immediately after being named co-CEO promptly drifted away to early retirement. (Quiet quitting king!) Bankman-Fried did remain involved in
hedging and risk at Alameda, though.\n\nAbout that $1 billion of revenue in 2021: Bankman-Fried definitely did not know that Singh, his employee, had backdated
interest payments to get FTX “over the line” to $1 billion. See, he’d just asked his employees to check and see if there was any source of funds that was
missing to get to $1 billion. This testimony was especially rambling.\n\nOh, also that MobileCoin loss? The one Wang said Alameda took to keep off FTX’s balance
sheet?', 'page': 'The jury finally hears from Sam Bankman-Fried (The Verge, 2023-10-28T00:12:41+00:00)'}, {'passage': 'The jury finally hears from Sam
Bankman-Fried (The Verge, 2023-10-28T00:12:41+00:00)\nThe one Wang said Alameda took to keep off FTX’s balance sheet? Yeah, so it was a totally innocent thing
where what actually happened was that Bankman-Fried thought it was appropriate that Alameda take the position as a backstop liquidity provider, that’s
all.\n\nIn June 2022, Bankman-Fried heard about the account called “fiat@ftx” tracking how much money Alameda owed to FTX, he testified. He did not know what it
was and did not bother to find out. He was busy! That was when Bankman-Fried directed Ellison to repay Alameda’s lenders, because he thought Alameda was good
for it. He also gave BlockFi and Voyager, two crypto lenders, some capital infusions for good measure.\n\nHe was “very surprised!”\n\nRemember that testimony
Adam Yedidia gave about a conversation with Bankman-Fried in August 2022 about the enormous amount of money Alameda owed FTX? Well, Bankman-Fried remembers it
differently.', 'page': 'The jury finally hears from Sam Bankman-Fried (The Verge, 2023-10-28T00:12:41+00:00)'}, {'passage': 'The jury finally hears from Sam
Bankman-Fried (The Verge, 2023-10-28T00:12:41+00:00)\nWell, Bankman-Fried remembers it differently. See, Yedidia was just asking about Alameda’s risk profile,
and Bankman-Fried wasn’t talking about insolvency at all.\n\nAlso, when Singh and Bankman-Fried had the dramatic balcony conversation at their penthouse, it was
just that Singh thought Alameda’s liabilities had gotten too high, and FTX was spending too much money on marketing. But Bankman-Fried still thought that
Alameda had more assets than liabilities, so it was all fine, and besides, if Singh thought he was going to be better at marketing, he could take it over. It
didn’t have anything to do with the money Alameda owed FTX at all.\n\nOf course not! Bankman-Fried didn’t learn about the $8 billion liability associated with
Alameda until October 2022, he said. And he learned it all by himself, by looking at a computer database. When he found it, he was “very surprised!”', 'page':
'The jury finally hears from Sam Bankman-Fried (The Verge, 2023-10-28T00:12:41+00:00)'}, {'passage': 'The jury finally hears from Sam Bankman-Fried (The Verge,
2023-10-28T00:12:41+00:00)\nWhen he found it, he was “very surprised!”\n\nBesides the two obvious lies Bankman-Fried told on the stand — about Alameda
Research’s name and about “allow_negative” — I have been struck by how little he seems to know about his own companies. Apparently, Singh, Wang, and Ellison
were out there just doing whatever their little hearts desired. Because Bankman-Fried was a CEO, but definitely not the kind that pays any attention to money at
his crypto trading firm and futures exchange.', 'page': 'The jury finally hears from Sam Bankman-Fried (The Verge, 2023-10-28T00:12:41+00:00)'}, {'passage':
'Sam Bankman-Fried didn’t ask where the $8 billion went (The Verge, 2023-10-31T19:52:07+00:00)\nLet’s say I am the owner of a hedge fund, and one fine June day,
my employees come to me and say, “Hey, Liz, we have an accounting problem. We are missing several billion dollars.” How would I react?\n\nI have been wondering
this since Danielle Sassoon walked Sam Bankman-Fried through his reaction to the FTX software bug fixed by Adam Yedidia. In my case, there would probably be
shouting? Like, a lot of shouting. I would also probably have my assistant figure out which law enforcement agency to call immediately. Misplacing $900 million
is a five-alarm fire even for Citibank; misplacing several billion is kicking over a lantern in Chicago in 1871.\n\nObviously, this is not how Bankman-Fried
reacted to the software bug that overestimated the amount Alameda owed to FTX by about $8 billion. Nor is it how he reacted to finding out that even after
fixing the bug, Alameda still owed FTX about $8 billion.', 'page': 'Sam Bankman-Fried didn’t ask where the $8 billion went (The Verge,
2023-10-31T19:52:07+00:00)'}, {'passage': 'Sam Bankman-Fried didn’t ask where the $8 billion went (The Verge, 2023-10-31T19:52:07+00:00)\nInstead, Bankman-Fried
directed alleged co-conspirator Caroline Ellison to repay third-party loans and went on making investments.\n\nBankman-Fried said that he found out about the
fiat@ftx account — that’s the one that tracked how much Alameda owed FTX — in June 2022 while his senior staff was focused on fixing the software bug. He did
not, however, discover what the account was for until October. I know this sounds unbelievable, but this is his actual testimony.\n\nHis employees told him
“they were busy and I should stop asking questions because it was distracting.”\n\nSimple questions nailed Bankman-Fried to the wall. Had he given his employees
at Alameda the direction not to spend FTX customer deposits? Had he put in any policies to prevent Alameda employees from spending FTX customer money? Did he
put in place any measures at Alameda to protect the FTX customer money? No, no, and no. Oh, but he was testifying to Congress about keeping customer funds
safe.', 'page': 'Sam Bankman-Fried didn’t ask where the $8 billion went (The Verge, 2023-10-31T19:52:07+00:00)'}, {'passage': 'Sam Bankman-Fried didn’t ask
where the $8 billion went (The Verge, 2023-10-31T19:52:07+00:00)\nOh, but he was testifying to Congress about keeping customer funds safe.\n\nSam Bankman-Fried,
the founder of failed cryptocurrency exchange FTX, is on trial for seven counts of wire fraud and conspiracy. FTX was a fraud “from the start,” the Securities
and Exchange Commission alleges — with a “multi-billion-dollar deficiency caused by his own misappropriation of customer funds.” Follow along for all the latest
news and regular updates from the trial.\n\nWho was making decisions to spend $8 billion of customer funds? Bankman-Fried couldn’t recall knowing anything about
it. Were there rules or requirements for how money borrowed from FTX would be returned? Were there rules for risk management? “I was concerned with overall risk
management,” Bankman-Fried said.\n\nBut it was the testimony about June 2022 that resonated the most to me. Didn’t Bankman-Fried ask what “fiat@ftx” was? He
did.', 'page': 'Sam Bankman-Fried didn’t ask where the $8 billion went (The Verge, 2023-10-31T19:52:07+00:00)'}, {'passage': 'Sam Bankman-Fried didn’t ask where
the $8 billion went (The Verge, 2023-10-31T19:52:07+00:00)\nHe did. But — I did hear these words uttered aloud in a court of law this morning, I am not creative
enough to make this kind of thing up — his employees told him “they were busy and I should stop asking questions because it was distracting.”\n\nYedidia —
Bankman-Fried’s college friend, Bahamas roommate, and employee at FTX — had testified that he’d asked Bankman-Fried about the $8 billion hole on a padel tennis
court in their luxury complex in June or July. Today, in testimony, Bankman-Fried seemed to be trying to deny that conversation had ever taken place. It was not
until Judge Lewis Kaplan intervened to ask if Bankman-Fried had ever been told by Yedidia about that money, in words or in substance, that Bankman-Fried
admitted he’d been told.', 'page': 'Sam Bankman-Fried didn’t ask where the $8 billion went (The Verge, 2023-10-31T19:52:07+00:00)'}, {'passage': 'Sam
Bankman-Fried didn’t ask where the $8 billion went (The Verge, 2023-10-31T19:52:07+00:00)\nI wouldn’t want to watch my child be vivisected, either. The jurors,
however, watched the operation attentively. I suppose for most of us, $8 billion has a way of focusing the mind.\n\nLook, uttering phrases like “hole isn’t
really the word I would use” and responding to a question by saying you wanted “a few more qualifiers and scoping on it” do not, as a general rule, bode well
for your believability. Yes, this will win certain kinds of nerd arguments. But this is a courtroom, and I have come to believe that if you know the meaning of
the word “epistemology,” you absolutely should not testify in your own defense.\n\n“His name is Ryan Salame :p”\n\nBankman-Fried, like Theranos’ Elizabeth
Holmes before him, got on the stand to speak directly to his state of mind in a way no other witness can.', 'page': 'Sam Bankman-Fried didn’t ask where the $8
billion went (The Verge, 2023-10-31T19:52:07+00:00)'}, {'passage': 'Sam Bankman-Fried didn’t ask where the $8 billion went (The Verge,
2023-10-31T19:52:07+00:00)\nBut telling your side of the story opens you to a great many questions you might not want to answer on cross-examination. I have
watched a lot of crosses. This was the nastiest I think I’ve ever seen.\n\nThe charges in this case hinge on conspiracy and intentional deceit. Just losing $8
billion is not a crime, though it is very embarrassing. (Even losing $900 million is very embarrassing because people will make jokes about it forever!) But if
Bankman-Fried lied to customers and lenders about what he was doing and how safe FTX was, that’s a crime.\n\nEstablishing that would have been enough, but
Sassoon also managed to get in some flourishes about Bankman-Fried’s relationships to Bahamian power. For instance, she asked him if he’d made comments about
paying off the Bahamian national debt. (You will be shocked, shocked to discover he didn’t recall.)', 'page': 'Sam Bankman-Fried didn’t ask where the $8 billion
went (The Verge, 2023-10-31T19:52:07+00:00)'}, {'passage': 'Sam Bankman-Fried didn’t ask where the $8 billion went (The Verge, 2023-10-31T19:52:07+00:00)\n(You
will be shocked, shocked to discover he didn’t recall.) We then saw an internal FTX group chat called “Project Chinchilla Chatter” in which another member asked
who in the Bahamian government they needed to talk to for the project. “His name is Ryan Salame :p,” Bankman-Fried replied. Salame was one of FTX’s
executives.\n\nIn another part of the chat, Bankman-Fried noted that the Bahamian prime minister was at the FTX arena in FTX seats with his wife.\n\nWe then saw
a November 9th email from Bankman-Fried to Ryan Pinder, attorney general of the Bahamas, that said “we are deeply grateful” for what the Bahamas had done for
FTX. As a token of that gratitude, Bankman-Fried wrote:\n\nWe would be more than happy to open up withdrawals for all Bahamian customers on FTX, so that they
can, tomorrow, fully withdraw all of their assets, making them fully whole.', 'page': 'Sam Bankman-Fried didn’t ask where the $8 billion went (The Verge,
2023-10-31T19:52:07+00:00)'}, {'passage': 'Sam Bankman-Fried didn’t ask where the $8 billion went (The Verge, 2023-10-31T19:52:07+00:00)\nIt’s your call whether
you want us to do this, but we are more than happy to and would consider it the very least of our duty to the country, and could open it up immediately if you
reply saying you want us to. If we don’t hear back from you, we are going to go ahead and do it tomorrow.\n\nBankman-Fried did indeed open withdrawals for
Bahamian customers. The upshot of this testimony seemed to be that Bankman-Fried had a cozy, perhaps even inappropriately cozy, relationship with the Bahamian
government — which isn’t what he’s on trial for but probably doesn’t make him look any better to a jury.\n\nSassoon successfully established yesterday that
Bankman-Fried has a long history of dishonesty. Today, through a set of questions about what Bankman-Fried did and didn’t do, she established that the story he
told on direct examination was absurd.', 'page': 'Sam Bankman-Fried didn’t ask where the $8 billion went (The Verge, 2023-10-31T19:52:07+00:00)'}, {'passage':
'Sam Bankman-Fried didn’t ask where the $8 billion went (The Verge, 2023-10-31T19:52:07+00:00)\nAfter a brief redirect examination by Bankman-Fried’s own
lawyers, which was resplendent with word salad, Bankman-Fried stepped down, and the defense rested their case.', 'page': 'Sam Bankman-Fried didn’t ask where the
$8 billion went (The Verge, 2023-10-31T19:52:07+00:00)'}, {'passage': 'Sam Bankman-Fried was a terrible boyfriend (The Verge, 2023-10-10T23:50:21+00:00)\nI’ve
got some shitty ex-boyfriends, but none of them made me the CEO of their sin-eater hedge fund while refusing to give me equity and bragging about how there was
a 5 percent chance they’d become the president of the United States, you know? Absolutely counting my blessings after Caroline Ellison’s first day on the stand.
I wonder how many of the nine women on the jury are doing the same.\n\nEllison was the head of Alameda Research, the aforementioned hedge fund, during the
implosion of it and FTX. She’s already pleaded guilty to criminal charges stemming from one of the worst romantic relationships I’ve ever heard of, and her
testimony was widely anticipated before the trial. Today, that took the form of discussing a damning spreadsheet — one she prepared for her ex and boss Sam
Bankman-Fried, now the defendant in a criminal fraud trial.\n\nSam Bankman-Fried, the founder of failed cryptocurrency exchange FTX, is on trial for seven
counts of wire fraud and conspiracy.', 'page': 'Sam Bankman-Fried was a terrible boyfriend (The Verge, 2023-10-10T23:50:21+00:00)'}, {'passage': 'Sam
Bankman-Fried was a terrible boyfriend (The Verge, 2023-10-10T23:50:21+00:00)\nFTX was a fraud “from the start,” the Securities and Exchange Commission alleges
— with a “multi-billion-dollar deficiency caused by his own misappropriation of customer funds.” Follow along for all the latest news and regular updates from
the trial.\n\nThe day started off promisingly for the defense as it cross-examined Gary Wang, the chief technology officer of FTX and co-owner of both FTX and
Alameda. Christian Everdell, one of Bankman-Fried’s defense attorneys, couldn’t undo the damage of last week’s code review. But he managed to shake the rust off
long enough to make Wang sound less reliable, drowning the jury in confusing technicalities.\n\nLast week, Wang testified that Alameda got access to a special
credit line and an option to take its balance into the negative without triggering liquidation — something he alleged other accounts at FTX didn’t get. Everdell
tried to undermine this claim by talking about the spot margin program, which let users lend each other assets for margin trading.', 'page': 'Sam Bankman-Fried
was a terrible boyfriend (The Verge, 2023-10-10T23:50:21+00:00)'}, {'passage': 'Sam Bankman-Fried was a terrible boyfriend (The Verge,
2023-10-10T23:50:21+00:00)\nIn those cases, it was possible to have a negative balance in a specific coin. It was not, however, possible for those accounts to
avoid liquidation, as Wang testified Alameda could do — or to have an overall negative balance. But I’m betting the defense is hoping the jurors will throw up
their hands in confusion thinking about this.\n\n“We ultimately took about $14 billion, some of which we were not able to pay back.”\n\nWang didn’t exactly help
himself out, either. Apparently, what Wang said in court contradicted something he’d said in earlier interviews with the government about market making. I say
“apparently” because Everdell was probably giving him his previous testimony to refresh his recollection, but Wang was insisting he didn’t remember. In any
event, whatever Wang was shown wasn’t submitted as evidence or shown to the court. I got the gist, though, and I bet the jury did, too — probably the strongest
work the defense has done so far.', 'page': 'Sam Bankman-Fried was a terrible boyfriend (The Verge, 2023-10-10T23:50:21+00:00)'}, {'passage': 'Sam Bankman-Fried
was a terrible boyfriend (The Verge, 2023-10-10T23:50:21+00:00)\nBut by the end of the day, that all seemed like a sideshow. Bankman-Fried had been vibrating
slightly during Wang’s testimony. During Ellison’s testimony, his bouncing became more noticeable.\n\nEllison was hunched in on herself as she walked into the
courtroom, wearing a dusty rose dress with a gray blazer over it, looking less like an executive than like a girl who’s borrowed her boyfriend’s coat because
she’s cold. When the prosecution asked her to identify Bankman-Fried, she had trouble finding him and gazed around the courtroom for more than 20 seconds —
apparently he was incognito with his new haircut. After she did spot him, she was asked to identify him, which she did by identifying him as wearing a suit.
This got chuckles from the rest of the defense table, also all in suits.\n\nShe listed off the crimes she’d already pleaded guilty to and added that
Bankman-Fried “directed me to commit these crimes,” Ellison said.', 'page': 'Sam Bankman-Fried was a terrible boyfriend (The Verge,
2023-10-10T23:50:21+00:00)'}, {'passage': 'Sam Bankman-Fried was a terrible boyfriend (The Verge, 2023-10-10T23:50:21+00:00)\n(Fraud, conspiracy to commit
fraud, and money laundering, in case you were wondering.) “We ultimately took about $14 billion, some of which we were not able to pay back.” She tilted her
head down to answer the questions, then lifted her head when she’d finished her answer.\n\nBankman-Fried called this being “risk-neutral,” which seems like a
fancy way of saying he was a gambling addict\n\nIn Ellison’s telling, Alameda was troubled from her earliest time there in 2018. “Shortly after I started, I
learned the company was in worse shape than I realized,” Ellison said. Alameda had initially been funded with loans “from acquaintances,” she said, and those
loans were recalled a few weeks after she arrived. (There was a staff revolt within Alameda Research over lost millions and general financial chaos, according
to Michael Lewis’ Going Infinite.) Ellison asked Bankman-Fried why he hadn’t shared the company’s shaky circumstances in the job offer. “He hadn’t known how to
tell me,” she said.', 'page': 'Sam Bankman-Fried was a terrible boyfriend (The Verge, 2023-10-10T23:50:21+00:00)'}, {'passage': 'Sam Bankman-Fried was a
terrible boyfriend (The Verge, 2023-10-10T23:50:21+00:00)\n“He hadn’t known how to tell me,” she said.\n\nEllison was also, of course, in a more personal
relationship with Bankman-Fried. (A juror who’d been asleep for a discussion of the FTT token woke up when she started discussing it.) The two started sleeping
together in the fall of 2018, on and off. At the time, she was a trader and Bankman-Fried was the CEO. They didn’t date until later — twice. Their first
relationship stretched from the summer of 2020 through the summer of 2021; they agreed to keep it secret. (Some people found out, as they usually do.) The
second time, from the fall of 2021 until the spring of 2022, they lived together.\n\nThat gave Ellison an unusual view of his character. “He was very
ambitious,” she said.', 'page': 'Sam Bankman-Fried was a terrible boyfriend (The Verge, 2023-10-10T23:50:21+00:00)'}, {'passage': 'Sam Bankman-Fried was a
terrible boyfriend (The Verge, 2023-10-10T23:50:21+00:00)\nThat gave Ellison an unusual view of his character. “He was very ambitious,” she said. Besides
telling her about his presidential chances, he also told her that if there was a coin flip where tails destroyed the world and heads made the world twice as
good, he’d flip the coin. He called this being “risk-neutral,” which seems like a fancy way of saying he was a gambling addict.\n\nShe was named co-CEO of
Alameda with Sam Trabucco in 2021, while she and Bankman-Fried were broken up, and CEO in 2022. The goal, Bankman-Fried told her, was to “optically” separate
Alameda Research and FTX. “The whole time we were dating, he was my boss at work,” she said. They broke up because she wanted more from the relationship;
Bankman-Fried was distant and not paying enough attention to her.', 'page': 'Sam Bankman-Fried was a terrible boyfriend (The Verge,
2023-10-10T23:50:21+00:00)'}, {'passage': 'Sam Bankman-Fried was a terrible boyfriend (The Verge, 2023-10-10T23:50:21+00:00)\nUsing the effectively unlimited
funds “allowed us to make profitable trades we couldn’t have made otherwise”\n\nBankman-Fried didn’t grant Ellison equity, even though she asked; he told her it
would be too complicated. Instead, she got a $200,000 salary, even as CEO, and bonuses twice a year, which ranged from $100,000 to $20 million.\n\nInitially,
Alameda and FTX were “very integrated,” Ellison said. They were run by the same team, from the same office. And when Alameda was scrounging for funds,
Bankman-Fried told Ellison that FTX would be a good source of capital. The $65 billion line of credit Alameda Research had meant that it did not have to post
collateral. There was no contract and no written terms, she testified. It also wasn’t visible to FTX’s auditors — she’d raised the question with Bankman-Fried,
and he told her not to worry about it.', 'page': 'Sam Bankman-Fried was a terrible boyfriend (The Verge, 2023-10-10T23:50:21+00:00)'}, {'passage': 'Sam
Bankman-Fried was a terrible boyfriend (The Verge, 2023-10-10T23:50:21+00:00)\nAlameda’s credit line — which was taken in increments of $100,000 to $10 million
at a time — was used for trading. Using the effectively unlimited funds “allowed us to make profitable trades we couldn’t have made otherwise,” Ellison
testified.\n\nCustomer funds were also used when Bankman-Fried bought back FTX shares from Binance, an early investor, in the summer of 2021. Bankman-Fried told
Ellison it was “really important,” otherwise “Binance would do things to mess with FTX.” Ellison says she told him Alameda didn’t have the money. So
Bankman-Fried took $1 billion of FTX customer funds to buy out Binance, the first time Ellison recalled an amount that large. It was Bankman-Fried’s decision,
she said, as he was the CEO of FTX.\n\nFTT was one of several “Sam coins”\n\nThere was also the FTT token, which was created by Bankman-Fried and Wang.',
'page': 'Sam Bankman-Fried was a terrible boyfriend (The Verge, 2023-10-10T23:50:21+00:00)'}, {'passage': 'Sam Bankman-Fried was a terrible boyfriend (The
Verge, 2023-10-10T23:50:21+00:00)\nAlameda got its war chest — 60 percent to 70 percent of the initial supply — for free, while seed investors got FTT at 10
cents a coin, and FTT first listed at $1 a coin. Bankman-Fried felt that $1 per coin was psychologically important, Ellison said, and he directed her to buy up
FTT using Alameda if its price fell below a dollar.\n\nFTT was one of several “Sam coins,” a nickname for tokens that Bankman-Fried was heavily involved in and
owned a lot of, either personally or through Alameda. Those coins were almost certainly worth less than the value displayed on the balance sheet because trying
to sell them all at once would crater the prices. Bankman-Fried directed her to put those coins on the balance sheets Alameda showed to lenders, even though she
felt it was “somewhat misleading.”\n\nAlameda was also getting loans from outside lenders, such as Genesis, because when FTX started, there weren’t a lot of
customer funds to borrow, Ellison testified.', 'page': 'Sam Bankman-Fried was a terrible boyfriend (The Verge, 2023-10-10T23:50:21+00:00)'}, {'passage': 'Sam
Bankman-Fried was a terrible boyfriend (The Verge, 2023-10-10T23:50:21+00:00)\nThat was the basis of the worst of her testimony — and the spreadsheet from
hell.\n\nEllison said she’d prepared the spreadsheet at Bankman-Fried’s request in the fall of 2021 and shared it with him. The point was risk analysis around
paying back Alameda’s loans if they were abruptly recalled by Genesis, their lender. Bankman-Fried wanted to use $3 billion for venture investments, so Ellison
was ballparking what that would do to Alameda’s risk. In the as-is scenario, if things went south, she figured there was a 30 percent chance they wouldn’t be
able to meet the loan recalls. If Bankman-Fried used $3 billion to make investments, there was a 100 percent chance they couldn’t meet the recalls, even with
FTX customer funds.\n\nEchoing Bankman-Fried and Ellison’s romantic relationship, FTX was keeping cozy private ties with Alameda yet publicly holding it at
arm’s length\n\nThe problem here wasn’t really the math, which seemed pretty arbitrary.', 'page': 'Sam Bankman-Fried was a terrible boyfriend (The Verge,
2023-10-10T23:50:21+00:00)'}, {'passage': 'Sam Bankman-Fried was a terrible boyfriend (The Verge, 2023-10-10T23:50:21+00:00)\nIt was that Ellison’s calculations
assumed Alameda could borrow $1.8 billion in normal dollars and $1.5 billion in crypto from FTX. The spreadsheet makes this clear with a row labeled “FTX
borrows,” which Ellison said were customer funds.\n\nMeanwhile, echoing Bankman-Fried and Ellison’s romantic relationship, FTX was keeping cozy private ties
with Alameda yet publicly holding it at arm’s length. On January 14th, 2022, Bankman-Fried tweeted, “We’re launching a $2b venture fund: FTX Ventures!” Those
funds came from Alameda, Ellison testified. But Bankman-Fried didn’t want to go public with the source of the funds. He said he thought Alameda’s brand was less
good, and he didn’t want his name associated with it. Alameda also bought Robinhood shares for Bankman-Fried, who moved them to a vehicle called “Emergent
Fidelity Technologies” to avoid association with Alameda.', 'page': 'Sam Bankman-Fried was a terrible boyfriend (The Verge, 2023-10-10T23:50:21+00:00)'},
{'passage': 'Sam Bankman-Fried was a terrible boyfriend (The Verge, 2023-10-10T23:50:21+00:00)\nThe day ended with a document that had been shared between
Ellison and Bankman-Fried — with his comments appearing in bubbles along the main text. Ellison wrote she was worried about “both actual leverage and presenting
on our balance sheet.” Bankman-Fried responded with a note: “Yup, and could also get worse.”', 'page': 'Sam Bankman-Fried was a terrible boyfriend (The Verge,
2023-10-10T23:50:21+00:00)'}, {'passage': 'In the end, the FTX trial was about the friends screwed along the way (The Verge, 2023-10-26T12:00:00+00:00)\nSure,
Sam Bankman-Fried’s trial has revealed that FTX was basically a seething mass of scams. But it has also revealed how Bankman-Fried blew up the lives of the
people who are closest to him.\n\nOf course everyone who had money trapped on the FTX exchange when it collapsed — as the result of Bankman-Fried’s trading firm
Alameda Research purloining its customers’ funds — had their lives altered. Even if they are made whole by the bankruptcy proceedings, that’s still money that
they should have had access to at any time they wanted it. That may mean houses that weren’t purchased, medical procedures that were left undone, and life
savings lost.\n\nThere is a simultaneous drama of ruined lives playing out in the courtroom\n\nBankman-Fried is charged with seven counts: two counts of wire
fraud and five conspiracy charges. As we near the end of the prosecution’s case, I have been struck by how bizarre it all is.', 'page': 'In the end, the FTX
trial was about the friends screwed along the way (The Verge, 2023-10-26T12:00:00+00:00)'}, {'passage': 'In the end, the FTX trial was about the friends screwed
along the way (The Verge, 2023-10-26T12:00:00+00:00)\nSingh also said Gabe had an assistant fly to the Bahamas with a stack of blank checks for Singh to
sign.\n\nWang, Singh, and Ellison are now felons. Even if they never do time — and they may very well be imprisoned or subjected to house arrest — their lives
are irrevocably changed, not least by whatever restitution requirements they are subjected to. They are barred from taking certain kinds of jobs. And of course,
there is the publicity. Bankman, Fried, and Gabe Bankman-Fried’s reputations are tarnished; in the political arenas where they once operated, they are suddenly
undesirable.\n\nIt is a Greek tragedy, precipitated by one man’s character flaw: the arrogance of Sam Bankman-Fried.', 'page': 'In the end, the FTX trial was
about the friends screwed along the way (The Verge, 2023-10-26T12:00:00+00:00)'}, {'passage': 'Is Sam Bankman-Fried’s defense even trying to win? (The Verge,
2023-10-04T23:02:14+00:00)\nI have never seen Sam Bankman-Fried so still as he was during the prosecution’s opening statement. The characteristic leg-jiggling
was absent. He barely moved as the prosecutor listed the evidence against him: internal company files, what customers were told, the testimony of his
co-conspirators and his own words.\n\nHis hair was shorn, the result of a haircut from a fellow prisoner, the Wall Street Journal reported. He wore a suit
bought at a discount at Macy’s, per the Journal; it hung on him. He appeared to have lost some weight.\n\n“All of that was built on lies.”\n\nBankman-Fried, at
this time last year, had a luxury lifestyle as the CEO of crypto exchange FTX, said the assistant US attorney, Thane Rehn, in the cadence of a high schooler
delivering his lines in a student play. Bankman-Fried hung out with Tom Brady. He was on magazine covers, lived in a $30 million penthouse, and spent time with
world politicians.', 'page': 'Is Sam Bankman-Fried’s defense even trying to win? (The Verge, 2023-10-04T23:02:14+00:00)'}, {'passage': 'Is Sam Bankman-Fried’s
defense even trying to win? (The Verge, 2023-10-04T23:02:14+00:00)\n“All of that was built on lies,” Rehn said.\n\nIn his opening statement, Rehn dodged
explaining cryptocurrency to the jury. Instead, he punched hard on Bankman-Fried lying and stealing.\n\nBankman-Fried sat almost motionless, occasionally
glancing at Rehn, as the prosecutor told the jury that Bankman-Fried sold stock in FTX and borrowed millions from lenders by lying.\n\nThe story Rehn told is
familiar to anyone following the news. In May and June of 2022, Alameda Research — the crypto trading company ostensibly helmed by Caroline Ellison — didn’t
have enough to pay its bills, so it pulled customer money to repay loans. By September, the hole in the FTX balance sheet was so big that customers could never
be repaid.\n\nFTX “didn’t have a chief risk officer, which became an issue when the storm hit.”\n\nWhen CoinDesk published its article in November 2022, people
realized FTX was a house of cards, Rehn said. Meanwhile, Bankman-Fried tweeted.', 'page': 'Is Sam Bankman-Fried’s defense even trying to win? (The Verge,
2023-10-04T23:02:14+00:00)'}, {'passage': 'Is Sam Bankman-Fried’s defense even trying to win? (The Verge, 2023-10-04T23:02:14+00:00)\nMeanwhile, Bankman-Fried
tweeted. “FTX is fine. Assets are fine” and “We don’t invest customer assets even in treasuries.”\n\nPointing at Bankman-Fried, Rehn said, “This man stole
billions of dollars from thousands of people.”\n\nSo how was the defense going to follow it up? I was very curious, having learned yesterday that Bankman-Fried
had never been offered a plea deal since he and his attorneys had told the government they wouldn’t negotiate. Surely there would be some manner of evidence,
some something, that would have made him so confident.\n\nThere was, instead, a metaphor.\n\nDefense attorney Mark Cohen, with the energy of a patient father
telling his obnoxious children a bedtime story, assured us that working at a startup was like building a plane while flying it, and that FTX the plane had flown
right into the perfect storm: the crypto crash.', 'page': 'Is Sam Bankman-Fried’s defense even trying to win? (The Verge, 2023-10-04T23:02:14+00:00)'},
{'passage': 'Is Sam Bankman-Fried’s defense even trying to win? (The Verge, 2023-10-04T23:02:14+00:00)\nExcept, uh, he also said this: FTX “didn’t have a chief
risk officer, which became an issue when the storm hit.”\n\nI couldn’t stop thinking about the missing risk officer\n\nThe problem with this metaphor is that if
FTX was a plane, it was a plane flying with a key component missing — namely, the risk officer, an executive whose job it is to, well, manage risk. This is sort
of an important thing, as risks can be anything from reputational to regulatory to financial.\n\nFTX was named such as it was because it was a futures exchange,
which, to borrow a phrase from Bloomberg’s Matt Levine, “sits between the winners and losers of bets.” That means FTX can’t pay out what it owes the winners
unless the losers pay up. Risk management is a crucial part of the business; risk officers exist to identify business’ potential risks, monitor, and mitigate
them. This is to say nothing of the regulatory risks around crypto.', 'page': 'Is Sam Bankman-Fried’s defense even trying to win? (The Verge,
2023-10-04T23:02:14+00:00)'}, {'passage': 'Is Sam Bankman-Fried’s defense even trying to win? (The Verge, 2023-10-04T23:02:14+00:00)\nThis is to say nothing of
the regulatory risks around crypto.\n\nAs Cohen droned on about airplanes, I couldn’t stop thinking about the missing risk officer. Bringing it up, I thought,
was a tremendous mistake. The prosecution hadn’t mentioned it. Either Bankman-Fried is stupid — unlikely — or he deliberately didn’t hire a risk officer. Was he
worried about what one might find?\n\nSure, as Cohen put it, Bankman-Fried was a math nerd who didn’t party. That paints a picture of someone who’s pretty
deliberate, particularly since he immediately left MIT and went to work on Wall Street. If he had been a party-hardy trainwreck, I could see overlooking a risk
officer in order to do another line, or a supermodel, or something else important. Why was the defense bringing this up?\n\nBut as Cohen tried to tell me that
FTX’s and Alameda’s business relationships were “reasonable under the circumstances,” the lack of risk officer kept elbowing me in the ribs.', 'page': 'Is Sam
Bankman-Fried’s defense even trying to win? (The Verge, 2023-10-04T23:02:14+00:00)'}, {'passage': 'Is Sam Bankman-Fried’s defense even trying to win? (The
Verge, 2023-10-04T23:02:14+00:00)\n“Sam acted in good faith and took reasonable business measures” is a pretty hard pill to swallow with that in mind.\n\nMan,
it’s no good when your defense lawyer has just made you sound worse than the prosecution already did. And while Cohen tried to make the common white-collar
defense argument that Bankman-Fried, as CEO, was simply too busy to oversee what everyone did every day, he just made me more suspicious. That’s why you hire a
risk officer and delegate! That’s the whole point! I could barely even hear Cohen blaming Caroline Ellison and Changpeng “CZ” Zhao for the debacle over the “no
risk officer” ringing in my ears.\n\nFollowing the defense’s opening statements, things got still worse for Bankman-Fried. The prosecution called its first
witness, Marc-Antoine Julliard, whose money got stuck on FTX. Juilliard, who was born in Paris and lives in London, testified that he trusted FTX because
Bankman-Fried came across as a leading figure of the industry.', 'page': 'Is Sam Bankman-Fried’s defense even trying to win? (The Verge,
2023-10-04T23:02:14+00:00)'}, {'passage': 'Is Sam Bankman-Fried’s defense even trying to win? (The Verge, 2023-10-04T23:02:14+00:00)\nWhen he was evaluating the
exchange, he thought the sheer volume of users was important, too — at the time, FTX was among the top three biggest exchanges. Plus, major VC firms had
invested, and “they don’t commit hundreds of millions without doing due diligence, checking the books, the accountancy of the firm, going through several
compliance process[es], so that was a vote of confidence for me,” Juilliard said. (Evidently he had not paid attention to the Elizabeth Holmes trial.)\n\nHe
also noted FTX’s glossy ads — featuring Gisele Bündchen, for instance — suggested a very high budget. It wouldn’t make sense to spend that much money unless FTX
had very strong financials, Juilliard figured. He opened an account, transferred in both regular money and cryptocurrency, and used the exchange to execute his
plan: buying Bitcoin to sell back in five to ten years at higher prices.', 'page': 'Is Sam Bankman-Fried’s defense even trying to win? (The Verge,
2023-10-04T23:02:14+00:00)'}, {'passage': 'Is Sam Bankman-Fried’s defense even trying to win? (The Verge, 2023-10-04T23:02:14+00:00)\nIt is a thankless task to
cross-examine a customer whose money is gone\n\nIn November 2022, things went bad for Julliard. He followed Bankman-Fried on Twitter, and read aloud the “FTX is
fine. Assets are fine” tweets, along with “FTX has enough to cover all client holdings. We don’t invest client assets” and a few others, which gave Julliard the
impression that his money was there — the problem might have been technical (anti-spam measures) or regulatory. When he tried to get his money out on November
8th, it was too late. We saw screenshots of his withdrawal attempts: $20,000 USD and about 4 Bitcoin, which were worth about $20,000 at the time: about $100,000
money, inaccessible.\n\nIt is a thankless task to cross-examine a customer whose money is gone, but Cohen tried anyway.', 'page': 'Is Sam Bankman-Fried’s
defense even trying to win? (The Verge, 2023-10-04T23:02:14+00:00)'}, {'passage': 'Is Sam Bankman-Fried’s defense even trying to win? (The Verge,
2023-10-04T23:02:14+00:00)\nHe noted that Julliard was a licensed commodities broker, who was trading in crypto because he didn’t have to disclose it; that
Julliard knew that crypto was new and risky, and that Julliard didn’t review the terms of service agreement he’d assented to when making his FTX
account.\n\nWell, sure, but so what?\n\nThe next witness called was Bankman-Fried’s former college (and FTX) roommate, Adam Yedidia, about whom I expect I will
have much more to say tomorrow.', 'page': 'Is Sam Bankman-Fried’s defense even trying to win? (The Verge, 2023-10-04T23:02:14+00:00)'}, {'passage': 'The FTX
trial is bigger than Sam Bankman-Fried (The Verge, 2023-09-28T12:00:00+00:00)\nThe trial of Sam Bankman-Fried is likely to be more consequential than just
whether the man himself is found guilty. Depending on what evidence is introduced during the trial, it could be rough for the entire crypto industry.\n\n“How
much damage can this trial do to the already beaten-down reputation of the industry at this point?” asks Yesha Yadav, a law professor at Vanderbilt University.
“This trial is going to be an excruciating moment for the industry because no one knows what kind of evidence might come out.”\n\nBankman-Fried, the founder of
FTX and Alameda Research, is facing seven counts of criminal charges: two counts of wire fraud, and five counts of conspiracy charges. FTX was a failed
cryptocurrency exchange founded in 2019. According to a now-deleted profile from FTX investors Sequoia Capital, FTX was founded because of Bankman-Fried’s
frustration with other exchanges when he was running Alameda Research, his crypto trading firm.', 'page': 'The FTX trial is bigger than Sam Bankman-Fried (The
Verge, 2023-09-28T12:00:00+00:00)'}, {'passage': 'The FTX trial is bigger than Sam Bankman-Fried (The Verge, 2023-09-28T12:00:00+00:00)\nWe have some inkling of
what the government will be arguing thanks to the filings prosecutors are making in this case. They argue that Bankman-Fried lied about consumer protection and
that Bankman-Fried’s statements that FTX was “avoiding or managing conflicts of interest” and that “as a general principle FTX segregates customer assets from
its own assets across our platforms” were lies, in a superseding indictment filed August 14th.\n\nThe indictment also says that there were special features in
FTX’s code that “permitted Alameda to spend and withdraw unlimited amounts of money from FTX,” which were created at Bankman-Fried’s direction. That effectively
exempted Alameda from the kinds of risk management other customers faced. At the same time, the government alleges, “Bankman-Fried publicly and repeatedly
asserted that Alameda did not have privileged access to FTX.”', 'page': 'The FTX trial is bigger than Sam Bankman-Fried (The Verge,
2023-09-28T12:00:00+00:00)'}, {'passage': 'The FTX trial is bigger than Sam Bankman-Fried (The Verge, 2023-09-28T12:00:00+00:00)\nBankman-Fried is accused of
using “billions of dollars in misappropriated FTX customer deposits” to help buy more than $200 million of real estate for himself, make billions of dollars of
investments for his own interest, and repay Alameda’s lenders, according to the indictment, Additionally, Bankman-Fried allegedly used more than $100 million of
customer funds to make political contributions; prosecutors can show evidence of those contributions in this trial, even though they aren’t part of the charges
brought. A second trial is scheduled for March 2024, with additional charges.\n\nBefore his fall, Bankman-Fried made himself out to be the Good Boy of crypto —
the trustworthy face of a sometimes-shady industry. He was also very interested in publicity, sitting for many interviews both before and after the fall of FTX.
The quick rise of FTX as an industry force was at least in part due to Bankman-Fried’s appetite for attention.', 'page': 'The FTX trial is bigger than Sam
Bankman-Fried (The Verge, 2023-09-28T12:00:00+00:00)'}, {'passage': 'The FTX trial is bigger than Sam Bankman-Fried (The Verge, 2023-09-28T12:00:00+00:00)\nHere
are the hits:\n\nBankman-Fried gave interviews freely — and quickly rose to public prominence in the industry. Though FTX hadn’t been in the business as long as
competing exchanges such as Coinbase, Kraken, or Gemini, Bankman-Fried positioned himself as an important, boyish face for crypto. (At one point, Bankman-Fried
told a colleague at FTX that “I honestly think it’s negative EV [this may mean “expected value,” as in poker] for me to cut my hair. I think it’s important for
people to think I look crazy.”)\n\nBecause he was so successful at this kind of public relations, his fall from grace was another mark against an industry that
was already roiled by bankruptcies and scandals. Some additional trouble for the crypto industry is likely to come from one crucial element of the fraud trial —
the part where the government must prove intent.\n\nThe first part of proving the government’s case is pretty simple and a little boring: prosecutors must show
that certain transactions took place.', 'page': 'The FTX trial is bigger than Sam Bankman-Fried (The Verge, 2023-09-28T12:00:00+00:00)'}, {'passage': 'The FTX
trial is bigger than Sam Bankman-Fried (The Verge, 2023-09-28T12:00:00+00:00)\nWhatever records the Southern District of New York has for the transactions will
be shown.\n\n“What conversations happened between him and his co-conspirators that are now cooperating against him?”\n\nThe second part is where all the drama
is likely to come, says Christopher LaVigne, a litigation partner and co-chair of the cryptocurrency practice at the law firm Withers. Prosecutors have to
connect those transactions to Bankman-Fried, show that he knew what he was doing was wrong, and prove that he lied about it anyway.\n\n“What was he saying to
his parents and his other advisors about this?” LaVigne says. “What conversations happened between him and his co-conspirators that are now cooperating against
him?”\n\nTo further establish intent, the government can use Bankman-Fried’s own words. The indictment calls Bankman-Fried’s tweets in November 2022 “false and
misleading.”\n\n“We had him going out directly to the internet,” LaVigne says.', 'page': 'The FTX trial is bigger than Sam Bankman-Fried (The Verge,
2023-09-28T12:00:00+00:00)'}, {'passage': 'The FTX trial is bigger than Sam Bankman-Fried (The Verge, 2023-09-28T12:00:00+00:00)\n“We had him going out directly
to the internet,” LaVigne says. If he wrote things on Twitter or said things in interviews that weren’t true, that’s more fodder for the government’s case.
“They can point to that and say, ‘This is what he said, this is what actually happened.’”\n\nOther evidence may include Signal messages and testimony from
co-conspirators who plead guilty to their own charges. Alameda Research CEO Caroline Ellison, who was also sometimes Bankman-Fried’s girlfriend, may play an
important role — Bankman-Fried leaked her diaries to The New York Times and was consequently jailed for witness tampering. The government has indicated FTX
co-founder Gary Wang and engineering head Nishad Singh will also be among the witnesses called to give testimony about Bankman-Fried.\n\nBankman-Fried’s defense
can also introduce risks for people who dealt with him\n\nThere may be testimony from lenders, venture capitalists, and customers in order to establish the
basis for some charges.', 'page': 'The FTX trial is bigger than Sam Bankman-Fried (The Verge, 2023-09-28T12:00:00+00:00)'}, {'passage': 'The FTX trial is bigger
than Sam Bankman-Fried (The Verge, 2023-09-28T12:00:00+00:00)\nIn the indictment, the prosecutors allege that Bankman-Fried lied to FTX investors. Some of those
investors may be called on to testify, which is not the kind of thing VC big shots generally enjoy — and which might create collateral damage for the
industry.\n\nIf, for instance, Sequoia Capital did due diligence around its investment in FTX, whatever Bankman-Fried told its partners could be important. Was
FTX already sending Alameda customer funds at that point? Did Bankman-Fried know about it? And did he tell VCs at the time? (If he told them it wasn’t
happening, that would establish intent.) Binance was also an early investor in FTX, and former executives may be called on to testify as well, says Hermine
Wong, the former head of policy at Coinbase and a former SEC regulator.\n\nBankman-Fried’s defense can also introduce risks for people who dealt with him.
Defense lawyers have several simultaneous objectives. First and foremost, they’re trying to prove their client is not guilty.', 'page': 'The FTX trial is bigger
than Sam Bankman-Fried (The Verge, 2023-09-28T12:00:00+00:00)'}, {'passage': 'The FTX trial is bigger than Sam Bankman-Fried (The Verge,
2023-09-28T12:00:00+00:00)\nFirst and foremost, they’re trying to prove their client is not guilty. But just in case they don’t get the outcome they want,
they’re also laying the groundwork for appeals and sentencing arguments. Any piece of evidence they want to introduce for those two purposes has to come into
play in the trial.\n\nAlready, Bankman-Fried’s lawyers are marking arguments that his constitutional rights are being violated by his pretrial incarceration
because, among other things, his internet connection wasn’t good enough for him to plan his defense, notes LaVigne. That could be grounds for an appeal.\n\n“Is
he going to throw the entire industry under the bus?”\n\nBankman-Fried’s behavior after the fall of FTX suggests he’s something of a wild card. He may suggest
he was acting on the advice of his lawyers. But he may also introduce other evidence that could be troublesome — implying, for instance, that he was engaged in
standard industry behavior or that everything that happened was Binance’s fault.', 'page': 'The FTX trial is bigger than Sam Bankman-Fried (The Verge,
2023-09-28T12:00:00+00:00)'}, {'passage': 'The FTX trial is bigger than Sam Bankman-Fried (The Verge, 2023-09-28T12:00:00+00:00)\nThat may be risky, but we
already know that Bankman-Fried loves risk.\n\n“Is he going to throw the entire industry under the bus?” Wong asks. “An idea like, ‘Everyone was doing this,
it’s not fair I’m the only one who was charged?’” That may not fly in a court of law, but it could absolutely damage public perception of crypto at
large.\n\nFor instance, just before he was arrested, Bankman-Fried brought forward messages from a crypto group chat in testimony he planned to give before
Congress. In that testimony, he primarily blames Binance and his lawyers at the firm Sullivan & Cromwell.\n\nIt’s possible other messages from group chats may
be brought into evidence during the trial. That can potentially be embarrassing for the whole industry, Wong says.\n\n“You can imagine some of these founders,
CEOs, people of that echelon talk to each other somewhat informally about what’s going on,” she says.', 'page': 'The FTX trial is bigger than Sam Bankman-Fried
(The Verge, 2023-09-28T12:00:00+00:00)'}, {'passage': 'The FTX trial is bigger than Sam Bankman-Fried (The Verge, 2023-09-28T12:00:00+00:00)\nIf it seems like
his peer group supported him, or worse, fawned over him, that’s a real problem for the industry’s reputation.”\n\nAfter FTX declared bankruptcy, Bankman-Fried
went on an extensive media tour, doing interviews with The New York Times’ Andrew Ross Sorkin and Good Morning America. Even under house arrest, he did
interviews, including with The New Yorker. Bankman-Fried seemed to think that if he just explained things, everyone would understand that what he did was a
mistake, not a crime.', 'page': 'The FTX trial is bigger than Sam Bankman-Fried (The Verge, 2023-09-28T12:00:00+00:00)'}, {'passage': 'Sex, Signal messages, and
sabotaging FTX\'s code: SBF execs and Bahamas roommates tell all in court (Cnbc | World Business News Leader, 2023-10-06T20:17:00+00:00)\nWitness Adam Yedidia
answers questions during Sam Bankman-Fried fraud trial over the collapse of FTX, the bankrupt cryptocurrency exchange, at Federal Court in New York City,
October 5, 2023, in this courtroom sketch. Jane Rosenberg | Reuters\n\nTwo of Sam Bankman-Fried\'s former friends from MIT, who also worked at crypto exchange
FTX while living with the company\'s founder in the Bahamas, took the stand in a Manhattan courtroom this week to testify against their former classmate,
confidant, and boss — a man who allegedly ran a crypto empire that defrauded thousands of customers out of billions of dollars. Gary Wang, the lesser-known
co-founder of FTX, was asked by Assistant U.S. Attorney Nicolas Roos on Thursday, "Did you commit financial crimes while working at FTX?" "Yes," responded
Wang.', 'page': "Sex, Signal messages, and sabotaging FTX's code: SBF execs and Bahamas roommates tell all in court (Cnbc | World Business News Leader,
2023-10-06T20:17:00+00:00)"}, {'passage': "Sex, Signal messages, and sabotaging FTX's code: SBF execs and Bahamas roommates tell all in court (Cnbc | World
Business News Leader, 2023-10-06T20:17:00+00:00)\nSam Bankman-Fried, the founder of bankrupt cryptocurrency exchange FTX, is seen during a hearing as a U.S
judge revoked his bail, at a courthouse in New York, U.S., August 11, 2023 in this courtroom sketch. Jane Rosenberg | Reuters\n\n$65 billion line of
credit\n\nFor months, Bankman-Fried has known that Wang and Ellison, who were integral members of his personal and professional inner circles, had turned on
him. Both pleaded guilty in December and have since been cooperating with the U.S. attorney's office in Manhattan. Wang's testimony, which stretched into
Friday, was given under a cooperation agreement with the government. Ellison is expected to take the stand under a similar arrangement.\n\nU.S.", 'page': "Sex,
Signal messages, and sabotaging FTX's code: SBF execs and Bahamas roommates tell all in court (Cnbc | World Business News Leader, 2023-10-06T20:17:00+00:00)"},
{'passage': "Sex, Signal messages, and sabotaging FTX's code: SBF execs and Bahamas roommates tell all in court (Cnbc | World Business News Leader,
2023-10-06T20:17:00+00:00)\nEllison is expected to take the stand under a similar arrangement.\n\nU.S. District Judge Lewis Kaplan presides as Gary Wang
testifies during the fraud trial of Sam Bankman-Fried over the collapse of FTX, the bankrupt cryptocurrency exchange, at Federal Court in New York City, U.S.,
October 6, 2023 in this courtroom sketch. Jane Rosenberg | Reuters\n\nBorn in China, Wang moved to the U.S. at age 7, and grew up in Minnesota before going to
the Massachusetts Institute of Technology to study math and computer science. He worked at Google after college. Wang, who first met Bankman-Fried during high
school at a summer camp, owned 10% of Alameda, while his boss owned the other 90%. Wang told the court about the advantages that Alameda received by having code
baked into FTX's software that allowed special access to the crypto exchange.", 'page': "Sex, Signal messages, and sabotaging FTX's code: SBF execs and Bahamas
roommates tell all in court (Cnbc | World Business News Leader, 2023-10-06T20:17:00+00:00)"}, {'passage': 'Sex, Signal messages, and sabotaging FTX\'s code: SBF
execs and Bahamas roommates tell all in court (Cnbc | World Business News Leader, 2023-10-06T20:17:00+00:00)\nAssistant United States Attorney Nicolas Roos
questions Gary Wang during Sam Bankman-Fried\'s fraud trial over the collapse of FTX, the bankrupt cryptocurrency exchange, at Federal Court in New York City,
U.S., October 6, 2023, in this courtroom sketch. Jane Rosenberg | Reuters\n\nAn $8 billion bug\n\nAdam Yedidia, who was the prosecution\'s second witness on
Wednesday, continued his testimony on Thursday. Yedidia met Bankman-Fried in college at MIT, and the pair remained close friends. Yedidia, assuming a robotic
posture on the stand, worked out of FTX\'s Hong Kong office from January to October of 2021 and then in the Bahamas until last year\'s collapse. In his
testimony, he referred to a group Signal thread called "People of the House," referring to Bankman-Fried\'s $35 million penthouse, where many employees lived.',
'page': "Sex, Signal messages, and sabotaging FTX's code: SBF execs and Bahamas roommates tell all in court (Cnbc | World Business News Leader,
2023-10-06T20:17:00+00:00)"}, {'passage': 'Sex, Signal messages, and sabotaging FTX\'s code: SBF execs and Bahamas roommates tell all in court (Cnbc | World
Business News Leader, 2023-10-06T20:17:00+00:00)\nZoom In Icon Arrows pointing outwards Exhibit from the prosecution shows Signal thread called "People of the
House," referring to Bankman-Fried\'s $35 million penthouse, where many employees lived. Source: SDNY\n\nIn terms of who was paying the rent, Yedidia recalled
Bankman-Fried saying he "assumed it\'s just Alameda paying for it in the end." Yedidia said Bankman-Fried had told him, before he began working in the Bahamas
in 2019, that he and Ellison had sex. Bankman-Fried asked Yedidia if it was a good idea for them to date, to which Yedidia said no. Bankman-Fried responded by
saying he was expecting that answer. One of Yedidia\'s responsibilities was fixing the bug in the code that gave Alameda preferential treatment.', 'page': "Sex,
Signal messages, and sabotaging FTX's code: SBF execs and Bahamas roommates tell all in court (Cnbc | World Business News Leader, 2023-10-06T20:17:00+00:00)"},
{'passage': 'Sex, Signal messages, and sabotaging FTX\'s code: SBF execs and Bahamas roommates tell all in court (Cnbc | World Business News Leader,
2023-10-06T20:17:00+00:00)\nIn June 2022, he submitted a report to Bankman-Fried on Signal that showed $8 billion in customer money held in an internal database
tracking the cash wired to an Alameda account called "fiat at ftx.com" was missing. Yedidia said he and Bankman-Fried spoke about it at the pickleball court at
the resort in Nassau, Bahamas. He asked his boss if things were OK. He was concerned because it "seemed like a lot of money" from FTX customers was at risk.
"Sam said, we were bulletproof last year. We aren\'t bulletproof this year," Yedidia testified. Yedidia said he asked when they would be bulletproof again.
Bankman-Fried said he wasn\'t sure, but it may be six months to three years. Yedidia said Bankman-Fried appeared "worried or nervous," which he said was
atypical.', 'page': "Sex, Signal messages, and sabotaging FTX's code: SBF execs and Bahamas roommates tell all in court (Cnbc | World Business News Leader,
2023-10-06T20:17:00+00:00)"}, {'passage': 'Sex, Signal messages, and sabotaging FTX\'s code: SBF execs and Bahamas roommates tell all in court (Cnbc | World
Business News Leader, 2023-10-06T20:17:00+00:00)\nYedidia said he\'s now teaching math — geometry and algebra — at a high school. He invested most of the
millions he earned as bonuses back into FTX, and his equity stake is now worthless. As FTX was failing, Yedidia said he was by Bankman-Fried\'s side. He
highlighted a Signal exchange in November 2022, during which he wrote, "I love you Sam. I\'m not going anywhere." He said he wrote the message because so many
people had left. When asked what changed, Yedidia said he learned that FTX customer deposits had been used to pay loans to creditors. He said Alameda\'s actions
seemed "flagrantly wrong." Yedidia\'s testimony ended on a fiery note, which was later struck from the record. He was asked why he had lost faith in FTX and
resigned. "FTX defrauded all its customers," he said.', 'page': "Sex, Signal messages, and sabotaging FTX's code: SBF execs and Bahamas roommates tell all in
court (Cnbc | World Business News Leader, 2023-10-06T20:17:00+00:00)"}, {'passage': 'Sex, Signal messages, and sabotaging FTX\'s code: SBF execs and Bahamas
roommates tell all in court (Cnbc | World Business News Leader, 2023-10-06T20:17:00+00:00)\n"FTX defrauded all its customers," he said.\n\nMatt Huang,
co-founder of Paradigm Operations LP, right, arrives at court in New York, US, on Thursday, Oct. 5, 2023. Former FTX Co-Founder Sam Bankman-Fried is charged
with seven counts of fraud and money laundering following the collapse of his cryptocurrency empire last year. Photographer: Yuki Iwamura/Bloomberg via Getty
Images Yuki Iwamura | Bloomberg | Getty Images\n\nInvestment to zero\n\nThe third witness to take the stand was Matt Huang, co-founder and managing partner of
Paradigm, a crypto venture capital firm that invested over $275 million in FTX. That stake was wiped out. Huang testified about his firm\'s due diligence on
FTX, and he told the court that Bankman-Fried assured him that funds would be used for FTX and not Alameda.', 'page': "Sex, Signal messages, and sabotaging
FTX's code: SBF execs and Bahamas roommates tell all in court (Cnbc | World Business News Leader, 2023-10-06T20:17:00+00:00)"}, {'passage': 'How is it still
getting worse for Sam Bankman-Fried? (The Verge, 2023-10-12T23:08:14+00:00)\nIn the break after Caroline Ellison stepped down from the stand, Barbara Fried
engaged defense lawyer Christian Everdell in an animated conversation. Fried, the defendant’s mother, was gesticulating and clearly had a strong opinion about
something. Everdell walked off, and Mark Cohen talked to her for a bit after that.\n\nFried seemed frustrated, and I couldn’t blame her. The defense absolutely
biffed the cross-examination of Ellison and, to make matters worse was unable to keep a recording of an all-hands meeting where Ellison confessed to taking
customer funds from being played for the jury. Is this really the best the defense can do?\n\nIn Cohen’s disorganized cross-examination, he mostly bored the
jury\n\nBefore this case, I had been told that Everdell and Cohen were “workman-like,” which I took to mean that they were unshowy but competent. I now believe
that comment was an insult. I have been waiting for a juicy cross-examination, as I live for chaos and drama.', 'page': 'How is it still getting worse for Sam
Bankman-Fried? (The Verge, 2023-10-12T23:08:14+00:00)'}, {'passage': 'How is it still getting worse for Sam Bankman-Fried? (The Verge,
2023-10-12T23:08:14+00:00)\nI have been waiting for a juicy cross-examination, as I live for chaos and drama. I am beginning to think I am not going to get
one.\n\nEllison had given, in her direct testimony, fairly damning evidence tying FTX CEO Sam Bankman-Fried to the conspiracy to take FTX customer funds. There
were fake balance sheets, one of which was sent to crypto lender Genesis. After a Genesis representative received the balance sheet, he texted Ellison to tell
her he’d spoken to Bankman-Fried — strongly suggesting that Bankman-Fried was aware of the contents of the fake balance sheet. Not great!\n\nSam Bankman-Fried,
the founder of failed cryptocurrency exchange FTX, is on trial for seven counts of wire fraud and conspiracy. FTX was a fraud “from the start,” the Securities
and Exchange Commission alleges — with a “multi-billion-dollar deficiency caused by his own misappropriation of customer funds.” Follow along for all the latest
news and regular updates from the trial.', 'page': 'How is it still getting worse for Sam Bankman-Fried? (The Verge, 2023-10-12T23:08:14+00:00)'}, {'passage':
'How is it still getting worse for Sam Bankman-Fried? (The Verge, 2023-10-12T23:08:14+00:00)\nFollow along for all the latest news and regular updates from the
trial.\n\nBut a lot of testimony relied on Ellison recounting conversations she’d had in person, or on auto-deleting text messaging platforms. This gave the
defense an opportunity to try to make her sound unreliable. After all, she had an incentive to flip on Bankman-Fried: the possibility of leniency in her
sentencing. Given her fun tweets about speed, the fact that she was Bankman-Fried’s ex-girlfriend, and that she’d apparently written a bunch of stuff down, I
was expecting fireworks. For the first time in this trial, maybe the defense had an opening.\n\nInstead, I got a sad trombone. In Cohen’s disorganized
cross-examination, he mostly bored the jury. At one point, two different jurors appeared to be asleep.\n\nMidway through the morning I began wondering if there
was a mercy rule for cross-examinations. Prosecutor Danielle Sassoon had run an effective direct examination, creating an easy-to-follow narrative.', 'page':
'How is it still getting worse for Sam Bankman-Fried? (The Verge, 2023-10-12T23:08:14+00:00)'}, {'passage': 'How is it still getting worse for Sam
Bankman-Fried? (The Verge, 2023-10-12T23:08:14+00:00)\nProsecutor Danielle Sassoon had run an effective direct examination, creating an easy-to-follow
narrative. By contrast, Cohen appeared to be bumbling around, taking up one topic only to abruptly pivot. Sure, we’re still in the prosecution’s case, but Cohen
had all night to prepare his lines of questioning.\n\nApparently, Alameda had a problem with retaining accountants\n\nWe established that Bankman-Fried had a
much larger appetite for risk than Ellison. I thought perhaps it might be building to something, but this line of questioning was quickly dropped. We
established that Bankman-Fried and Ellison reacted differently to stress, and that they also had different approaches to media: namely, that Ellison avoided it
while Bankman-Fried sought it out. Okay?\n\nWe discovered that there was one accountant at Alameda in 2021, and two more junior accountants were hired in 2022.
Apparently, Alameda had a problem with retaining accountants, which didn’t surprise me much; CEOs generally don’t do the balance sheets for their companies.',
'page': 'How is it still getting worse for Sam Bankman-Fried? (The Verge, 2023-10-12T23:08:14+00:00)'}, {'passage': 'How is it still getting worse for Sam
Bankman-Fried? (The Verge, 2023-10-12T23:08:14+00:00)\nI was ready to hear this pursued further — but then it, too, was dropped.\n\nI think the defense was also
trying to suggest that the government had coerced Ellison’s testimony, by suggesting that she had pleaded guilty to a charge of defrauding investors that she
couldn’t have been involved in. After all, she didn’t prepare materials for them. Unfortunately, she did say that she had conversations with investors as part
of their due diligence — and, of course, Alameda was taking on losses from FTX to keep FTX’s balance sheet pristine. This line of questioning felt like a waste
of time.\n\nThere were rather a lot of sidebars during the cross-examination, to the point that when one occurred, several jurors looked entertained. There were
a few yesterday, too, including one in which the prosecution complained that Bankman-Fried was visibly scoffing at Ellison’s answers, according to the court
transcript.', 'page': 'How is it still getting worse for Sam Bankman-Fried? (The Verge, 2023-10-12T23:08:14+00:00)'}, {'passage': 'How is it still getting worse
for Sam Bankman-Fried? (The Verge, 2023-10-12T23:08:14+00:00)\n(I did observe him occasionally shaking his head, and sometimes quivering at points during her
testimony, but didn’t have a view of his face.)\n\nAt one point, Ellison appeared to even be toying with Cohen\n\nAs we were approaching lunch, several jurors
looked annoyed, and Cohen looked clueless. He asked Ellison to define what “buy on the way down” meant, as though it were a term of art. (It means what you
think it means, to purchase an asset that’s losing value.) This seemed to puzzle her. At other times, Cohen seemed to forget what she had testified to, bringing
up things she hadn’t said. I don’t know if this was an attempt to trap her in a lie or just poor preparation, but much like FTX employee Adam Yedidia before
her, Ellison was fastidious about making sure a question was clear and her answer was precise.\n\nAt one point, Ellison appeared to even be toying with Cohen.
She’d testified on direct about Luna, a cryptocurrency token.', 'page': 'How is it still getting worse for Sam Bankman-Fried? (The Verge,
2023-10-12T23:08:14+00:00)'}, {'passage': 'How is it still getting worse for Sam Bankman-Fried? (The Verge, 2023-10-12T23:08:14+00:00)\nShe’d testified on
direct about Luna, a cryptocurrency token. It had a sister token, Terra, that was a paired algorithmic stablecoin. (If you don’t understand what that means, it
doesn’t matter, because they were both nonsense.) Cohen asked her about Terra/Luna and she pointed out she’d only spoken about Luna, leaving him to fumble about
how to explain the relationship between the two tokens. She kept a straight face on the stand while I chuckled from the press seats.\n\nDuring the opening
statement, Cohen had blamed Ellison for not taking out a hedge on some of the risks Alameda was trading. We heard more about this hedge in cross-examination,
and friends, it was stupid. Forget evaluating the trade itself (hedging being long crypto by selling Nasdaq futures). Was she supposed to have taken more
customer funds to put on the hedge? Was that the defense, that she didn’t take enough of them? Was it that she should have taken them sooner? What the fuck?',
'page': 'How is it still getting worse for Sam Bankman-Fried? (The Verge, 2023-10-12T23:08:14+00:00)'}, {'passage': 'How is it still getting worse for Sam
Bankman-Fried? (The Verge, 2023-10-12T23:08:14+00:00)\nWas it that she should have taken them sooner? What the fuck?\n\nWhen Sassoon got up for a quick
redirect, she demolished any points Cohen had attempted to make. But I didn’t really appreciate her cleverness until after Ellison left the stand, and the
jurors left the room. She’d managed to set a neat little trap for Cohen.\n\nIn the recording, Ellison did indeed confess to stealing customer funds with
Bankman-Fried’s approval\n\nOn the direct examination, near the end, Sassoon asked about an Alameda all-hands meeting, without bringing up many specifics.
During the cross, Cohen asked Ellison what topics were covered in the meeting, while avoiding details. That opened the door for Sasson on redirect to work in
that Ellison had confessed to stealing billions of FTX customers’ money, at Bankman-Fried’s direction.\n\nThere had been an open question of whether jurors
would hear the tapes of Ellison’s remarks. The testimony set up an argument for the prosecution to bring in those tapes.', 'page': 'How is it still getting
worse for Sam Bankman-Fried? (The Verge, 2023-10-12T23:08:14+00:00)'}, {'passage': 'How is it still getting worse for Sam Bankman-Fried? (The Verge,
2023-10-12T23:08:14+00:00)\nThe testimony set up an argument for the prosecution to bring in those tapes. The judge ruled in favor of the jury hearing the
recordings and we briefly recessed.\n\nThat was when Bankman-Fried’s mother approached the defense lawyers.\n\nThe late afternoon was short and snappy.
Christian Drappi, a former Alameda software engineer who looked like a handsome funeral director in a black suit and tie, testified briefly to set up the tape.
When Changpeng Zhao, the CEO of Binance, announced on Twitter that he intended to acquire FTX, Ellison confessed the theft of customer funds to him and a few
other employees, Drappi said. The all-hands took place the following day, and was secretly recorded by an employee who’d joined Alameda three days before.\n\nIn
the recording, Ellison did indeed confess to stealing customer funds with Bankman-Fried’s approval. Drappi said he resigned less than 24 hours after the
meeting. The recording was later sent to Drappi, who sent it to the government.', 'page': 'How is it still getting worse for Sam Bankman-Fried? (The Verge,
2023-10-12T23:08:14+00:00)'}, {'passage': 'How is it still getting worse for Sam Bankman-Fried? (The Verge, 2023-10-12T23:08:14+00:00)\nThe recording was later
sent to Drappi, who sent it to the government. Joseph Bankman, the defendant’s father and a senior advisor to FTX’s philanthropic arm, wasn’t in the room for
the recording. Barbara Fried looked unhappy, rubbing her left temple as though she had a tremendous headache.\n\nAnd in a small miracle of pacing, the
government got Zac Prince, the founder and former CEO of crypto lender BlockFi, on the stand just long enough to blame FTX’s bankruptcy for BlockFi’s subsequent
bankruptcy before we broke for the day.\n\nI’ve been asked by some people why the reporting coming out of the trial seems so skewed toward the prosecution.
There’s an answer for this: the prosecution has put together a strong, comprehensive indictment of Bankman-Fried’s behavior at FTX and Alameda. The defense, so
far, has managed to do almost nothing in response.', 'page': 'How is it still getting worse for Sam Bankman-Fried? (The Verge, 2023-10-12T23:08:14+00:00)'},
{'passage': 'How is it still getting worse for Sam Bankman-Fried? (The Verge, 2023-10-12T23:08:14+00:00)\nThe defense, so far, has managed to do almost nothing
in response. I don’t know if Cohen and Everdell have bad facts, a bad client, or are simply untalented themselves (or some combination of all three?). But I do
know that I haven’t yet seen any good reason to doubt the very convincing story I’m hearing from the prosecution.', 'page': 'How is it still getting worse for
Sam Bankman-Fried? (The Verge, 2023-10-12T23:08:14+00:00)'}, {'passage': 'Coinbase rallies more than 60% in same month that FTX and Binance founders brace for
prison (Cnbc | World Business News Leader, 2023-11-30T21:08:00+00:00)\nFor early holders of the stock, the rebound helps ease the pain of 2022, when Coinbase
lost 86% of its value as soaring inflation and rising interest rates pushed investors out of crypto and high-growth tech companies, and into assets deemed safer
in a recession. Tech stocks have roared back this year, particularly those tied to the artificial intelligence boom and crypto. Coinbase has the added benefit
of having survived the so-called crypto winter, while so many of its rivals disappeared or downsized. The industry fallout came to a head this month, when Sam
Bankman-Fried, founder of former Coinbase rival FTX, was found guilty of seven criminal fraud counts tied to the collapse of his exchange and the theft of
customer funds. His conviction landed on Nov. 2 after a monthlong trial. Less than three weeks later, on Nov.', 'page': 'Coinbase rallies more than 60% in same
month that FTX and Binance founders brace for prison (Cnbc | World Business News Leader, 2023-11-30T21:08:00+00:00)'}, {'passage': "Coinbase rallies more than
60% in same month that FTX and Binance founders brace for prison (Cnbc | World Business News Leader, 2023-11-30T21:08:00+00:00)\n2 after a monthlong trial. Less
than three weeks later, on Nov. 21, Binance founder Changpeng Zhao pleaded guilty to violations of the Bank Secrecy Act for failing to implement an effective
anti-money laundering program and for willfully violating U.S. economic sanctions.\n\nCombination showing Former FTX CEO, Sam Bankman-Fried (L) and Zhao
Changpeng (R), founder and chief executive officer of Binance. Getty Images | Reuters\n\nBankman-Fried, who faces potential life behind bars, is scheduled to be
sentenced in March. Zhao's sentencing is set for February. While guidelines suggest a sentence of 12 to 18 months, the Justice Department could push for a
lengthier punishment for the Binance founder. Unlike FTX, which filed for bankruptcy in late 2022, Binance is still standing, though now without Zhao, who
agreed to step down as CEO as part of the plea deal.", 'page': 'Coinbase rallies more than 60% in same month that FTX and Binance founders brace for prison
(Cnbc | World Business News Leader, 2023-11-30T21:08:00+00:00)'}]
</item 10>
=== END ITEM id=f4175c59f29b4303 ===
