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

=== ITEM id=202c5c833da15018 ===
Gold sentences and facts:
Sentence 0: But “all of that, all of it, was built on lies,” Rehn declared to the jury.
  Fact 0.0: Rehn declared to the jury that all of that was built on lies.
  Fact 0.1: Rehn declared to the jury that all of it was built on lies.
Sentence 1: Binance has made headlines this past year for a range of reasons, including Zhao’s comments contributing to the collapse of FTX, which was once one of its top competitors.
  Fact 1.0: Binance has made headlines this past year for a range of reasons.
  Fact 1.1: Zhao's comments contributed to the collapse of FTX.
  Fact 1.2: FTX was once one of Binance's top competitors.

Retrieved items:
<item 1>
SBF’s trial starts soon, but how did he — and FTX — get here? (TechCrunch, 2023-10-01T14:00:29+00:00)
That means it purported to have more FTT tokens on its balance sheet than what existed in the world.

Around the same time it was exposed, the world’s largest crypto exchange, Binance, started pulling out its remaining $2.1 billion equivalent of cash in BUSD and
FTT. (It had an equity position in FTX from 2019 to 2021.) This essentially triggered a bank run on FTX.

FTX and Alameda filed for Chapter 11 bankruptcy in the U.S. mid-November 2022. Bankman-Fried resigned, and John J. Ray III, the Enron turnaround veteran, was
appointed its new CEO.

Bankman-Fried, however, maintained his innocence. At The New York Times’ DealBook Summit, he appeared virtually from the Bahamas, saying “I didn’t ever try to
commit fraud on anyone; I was shocked by what happened this month.”

Graph facts:
- He’s Hockey’s Brightest Young Star. This Is What Makes His Shot So Special. -PUBLISHED_BY-> The New York Times
- How U.S. Women’s Soccer Team Lured Emma Hayes Away From Chelsea -PUBLISHED_BY-> The New York Times
- After a Stunning Snub, College Football Is Facing an Identity Crisis -PUBLISHED_BY-> The New York Times
- N.F.L. Executives on the Biggest Questions Facing Super Bowl Contenders -PUBLISHED_BY-> The New York Times
- Which M.L.B. Playoff Team Has the Most Daunting Home Field? We Ranked Them. -PUBLISHED_BY-> The New York Times

Entities: The New York Times, SBF’s trial, FTT tokens, BUSD, Chapter 11 bankruptcy, bank run on FTX, John J. Ray III, DealBook Summit, SBF’s trial starts soon,
but how did he — and FTX — get here?, Alameda, TechCrunch, Bahamas, Enron, FTX, Sam Bankman-Fried
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

Graph facts:
- SBF’s trial starts soon, but how did he — and FTX — get here? -MENTIONS-> Alameda
- SBF, riding high on FTX, reportedly offered $5B to Trump to cancel President re-run -MENTIONS-> Alameda
- Is Sam Bankman-Fried a bad ‘man’ or a good ‘boy’? Lawyers swap opening statements before first witnesses take the stand -MENTIONS-> Alameda
- The jury finally hears from Sam Bankman-Fried -MENTIONS-> Alameda
- SBF Trial: The latest updates from the FTX collapse’s courtroom drama -MENTIONS-> Alameda

Entities: Alameda, Sam Bankman-Fried was a terrible boyfriend, FTT, Ellison, FTX, Wang, Sam Bankman-Fried, Binance, The Verge
</item 2>

<item 3>
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
</item 3>

<item 4>
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
</item 4>

<item 5>
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
</item 5>

<item 6>
SBF’s trial starts soon, but how did he — and FTX — get here? (TechCrunch, 2023-10-01T14:00:29+00:00)
SBF’s trial has started, this is how he and FTX got here

The highly anticipated criminal trial for Sam Bankman-Fried, former CEO of bankrupt crypto exchange FTX, started Tuesday to determine whether he’s guilty of
seven counts of fraud and conspiracy. And as one former federal prosecutor put it: “The odds seem to be stacked against him at this point.”

The 31-year-old co-founded FTX in 2019; within a few years the once third-largest crypto exchange’s valuation hit $32 billion at its peak. It’s now trying to
claw back any funds to distribute to creditors.

But how did the once third-largest crypto exchange get here?

Before FTX, Bankman-Fried co-founded crypto-trading firm Alameda Research in 2017. He co-founded FTX in 2019 as a complement to Alameda, to help bring in
revenue and liquidity for the trading arm.
</item 6>

<item 7>
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
</item 7>

<item 8>
In the end, the FTX trial was about the friends screwed along the way (The Verge, 2023-10-26T12:00:00+00:00)
Even that wasn’t good enough. Singh testified to sprucing up FTX’s balance sheet with backdated transactions in December 2021 to push FTX’s revenue “over the
line” to $1 billion. By the time Singh found out about the use of customer money, in 2022, he was already compromised.

At every turn, the scams at FTX and Alameda led back to making Bankman-Fried more important

In Singh’s case, it wasn’t just balance sheets he was fiddling with — it was also the American political system. By coordinating with Gabe Bankman-Fried, Singh
had funneled money to a number of political causes. That money came from “loans,” which weren’t really loans since he hadn’t signed any term sheets.

What was the goal here? What was accomplished? Singh testified to more than a billion dollars spent on celebrity endorsements, which made FTX very famous — even
though the futures platform didn’t operate in the US.

Graph facts:
- Sam Altman backs teens’ startup, Google unveils the Pixel 8 and TikTok tests an ad-free tier -MENTIONS-> Bankman-Fried
- SBF’s trial starts soon, but how did he — and FTX — get here? -MENTIONS-> Bankman-Fried
- Sex, Signal messages, and sabotaging FTX's code: SBF execs and Bahamas roommates tell all in court -MENTIONS-> Bankman-Fried
- Today the FTX jury suffered through a code review -MENTIONS-> Bankman-Fried
- Bankman-Fried -SAID-> Alameda Research

Entities: Bankman-Fried, Alameda, In the end, the FTX trial was about the friends screwed along the way, political causes, Gabe Bankman-Fried, Singh, futures
platform, American political system, celebrity endorsements, FTX trial, FTX, The Verge
</item 8>

<item 9>
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

Graph facts:
- SBF’s trial starts soon, but how did he — and FTX — get here? -MENTIONS-> The New York Times
- Sam Bankman-Fried -PARTICIPATED_IN-> SBF’s trial
- SBF -PARTICIPATED_IN-> SBF’s trial
- Is Sam Bankman-Fried’s defense even trying to win? -MENTIONS-> CoinDesk
- SBF’s trial starts soon, but how did he — and FTX — get here? -MENTIONS-> FTX collapse

Entities: SBF’s trial, Warren Buffet, CoinDesk, FTX’s collapse, liquidity, SBF’s trial starts soon, but how did he — and FTX — get here?, Bankman-Fried, crypto,
Alameda, FTT, TechCrunch, FTX
</item 9>

<item 10>
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
</item 10>
=== END ITEM id=202c5c833da15018 ===

=== ITEM id=56053c574fa7402a ===
Gold sentences and facts:
Sentence 0: His career has only blossomed in the years since he took that trip, and while some artists might feel trepidation at the attainment of so much success from such an early age, Mitchell is clear-eyed.
  Fact 0.0: Mitchell's career has only blossomed in the years since he took that trip.
  Fact 0.1: Some artists might feel trepidation at attaining so much success at such an early age.
  Fact 0.2: Mitchell is clear-eyed.
Sentence 1: Since then he’s been linked to club jobs in Europe but opted to stick around with the national team.
  Fact 1.0: Since then, he has been linked to club jobs in Europe.
  Fact 1.1: He opted to stick around with the national team.

Retrieved items:
<item 1>
‘Bleeds green and gold’: Ballsy career-saving call that’s delivered Arnie a record-breaking moment (The Roar | Sports Writers Blog, 2023-11-13T23:01:20+00:00)
“I was able to put lessons from that time into practice, develop and eventually return to the Socceroos much more prepared for the role. It’s been a privilege
to be part of the national team again with the benefit of that experience, and I’m immensely proud of what this group has been able to achieve in the last four
years.

“In my opinion, there’s no greater honour than representing your nation on the international stage. I look at the young players coming through the Socceroos
system now full of pride in the work they’ve put in to get where they are, and the journey that’s ahead of them.

“I’ve always loved being a part of the Socceroos culture and my desire to see Australia fulfil its footballing potential is still driving me to take this group
to further success.”

Australia starts the road to north America in Melbourne at AAMI Park on Thursday night, before travelling to the Middle East to play Palestine next Wednesday
(1.00am AEDT).

Graph facts:
- ASX set to open higher as Wall Street rebounds; $A rises -MENTIONS-> Australia
- A comprehensive list of 2023 tech layoffs -MENTIONS-> Australia
- ASX to open lower; Wall Street falls as the clamp tightens from rising bond yields -MENTIONS-> Australia
- Israel's defense undone by reliance on technology -MENTIONS-> Australia
- ASX set to drop as Wall Street’s September slump deepens -MENTIONS-> Australia

Entities: Australia, ‘Bleeds green and gold’: Ballsy career-saving call that’s delivered Arnie a record-breaking moment, Socceroos, Arnie, AAMI Park, Australia
starts the road to North America, Play Palestine, Melbourne, Middle East, North America, The Roar, Palestine
</item 1>

<item 2>
‘Bleeds green and gold’: Ballsy career-saving call that’s delivered Arnie a record-breaking moment (The Roar | Sports Writers Blog, 2023-11-13T23:01:20+00:00)
In March last year Graham Arnold seemed a dead man walking – the Socceroos World Cup qualification campaign was lurching towards failure and the manager was
being battered from all sides.

Arnold’s team had finished seven points off automatic qualification for Qatar after a shocking run of one win in seven games left them facing a Asian playoff
against United Arab Emirates and then another against Peru to make the finals.

While rumours swirled and ex-players and fans bayed for his blood, Football Australia took the pragmatic course and stuck by Arnold. It likely had more to do
with the lateness of the hour and financial considerations than any overwhelming faith in Arnie to run around the fortunes.

Facebook Twitter Whatsapp Reddit Email Share

But he did just that with qualification – and an excellent run at the finals – bowing out to the eventual champions Argentina and after nearly taking them to
extra time.

Graph facts:
- Arnold -PARTICIPATED_IN-> playoff against Peru
- ‘Bleeds green and gold’: Ballsy career-saving call that’s delivered Arnie a record-breaking moment -MENTIONS-> Australia
- ‘Bleeds green and gold’: Ballsy career-saving call that’s delivered Arnie a record-breaking moment -PUBLISHED_BY-> The Roar | Sports Writers Blog
- Arnold -WORKS_FOR-> Socceroos
- ChatGPT: Everything you need to know about the AI-powered chatbot -MENTIONS-> Peru

Entities: ‘Bleeds green and gold’: Ballsy career-saving call that’s delivered Arnie a record-breaking moment, Graham Arnold, Socceroos, Socceroos World Cup
qualification campaign, Asian playoff against United Arab Emirates, playoff against Peru, Peru, World Cup, Football Australia, World Cup finals, Qatar,
Argentina, The Roar, United Arab Emirates
</item 2>

<item 3>
‘Bleeds green and gold’: Ballsy career-saving call that’s delivered Arnie a record-breaking moment (The Roar | Sports Writers Blog, 2023-11-13T23:01:20+00:00)
With moments left in extra time against the South Americans, Arnold dragged Mat Ryan and replaced him with Andrew Redmayne – who saved the decisive spot kick
(and also threw his opponents water bottle with cheat sheet attached away from the goal).

Advertisement

Ryan might well have gotten the Socceroos to Qatar, but if Australia had not advanced there’s little chance Arnold would still be their coach today.

And there’s virtually no chance he’ll have to endure a similar level of stress through to the next finals – with FIFA expanding the tournament from 36 to 48
teams meaning eight direct Asian confederation qualifiers will proceed directly instead of four.

Arnold, in his second stint as national team manager, will prowl the technical area for a 59th time on Thursday, taking him clear of Frank Farina.

Graph facts:
- ASX set to open higher as Wall Street rebounds; $A rises -MENTIONS-> Australia
- A comprehensive list of 2023 tech layoffs -MENTIONS-> Australia
- ASX to open lower; Wall Street falls as the clamp tightens from rising bond yields -MENTIONS-> Australia
- Israel's defense undone by reliance on technology -MENTIONS-> Australia
- ASX set to drop as Wall Street’s September slump deepens -MENTIONS-> Australia

Entities: Australia, ‘Bleeds green and gold’: Ballsy career-saving call that’s delivered Arnie a record-breaking moment, Socceroos, Arnold, Mat Ryan, Andrew
Redmayne, South Americans, the tournament, Asian confederation, Frank Farina, Qatar, FIFA, The Roar
</item 3>

<item 4>
Graph query row:
article: How shooting Beyoncé and Kamala Harris made this photographer famous too
published_on: 2023-11-27T23:00:00+00:00
document_title: How shooting Beyoncé and Kamala Harris made this photographer famous too (The Age, 2023-11-27T23:00:00+00:00)
publisher: The Age
publisher_relationship: PUBLISHED_BY
person: Tyler Mitchell
chunk_index: 2482
evidence: How shooting Beyoncé and Kamala Harris made this photographer famous too (The Age, 2023-11-27T23:00:00+00:00)
Save Log in , register or subscribe to save articles for later. Save articles for later Add articles to your saved list and come back to them any time. Got it
Normal text size Larger text size Very large text size Tyler Mitchell was just 23 when he became the first black photographer to shoot the cover of American
Vogue. His subject was Beyoncé, and his dreamy, delicately hued portraits spoke of a precocious visual talent. In the five years since, he has exhibited around
the world, become an Instagram darling, with more than 500,00 followers, and shot what would become the most controversial vice-presidential portrait in living
memory. When Mitchell’s portrait of Kamala Harris dressed in sneakers and casual clothing was chosen for the cover of Vogue for February 2021, the social media
backlash was swift. Critics questioned why his alternative image – in which Harris looked more, well, vice-presidential – hadn’t been used instead.
</item 4>

<item 5>
Who’s the next Gawn, Grundy or Dusty? AFL’s talent guru names top 40 draft prospects and compares them to current stars (The Roar | Sports Writers Blog,
2023-11-15T23:48:27+00:00)
Was also impressive for the Murray Bushrangers in the Coates Talent League, averaging 16 disposals and 3.7 tackles across 10 matches and booted nine goals. His
high natural leap and pace were both prominent at the national Draft Combine with his standing vertical leap of 78cm ranked second overall and his time of 2.96
seconds on the 20m sprint ranked in the top ten.

Weapon: Elusiveness and creativity

AFL player comparison: Tom Papley (Sydney Swans)

Will Graham

State: Queensland

State League Club: Gold Coast Suns Academy

Community Club: Palm Beach Currumbin

Date of Birth: 27/07/05

Height: 186cm

Adapted seamlessly to playing in defence for the Allies after predominantly playing midfield for the Gold Coast Suns Academy.

Graph facts:
- Craft beer and cheese boxes: Swans tap into something ‘uniquely Sydney’ to draw AFLW crowds -MENTIONS-> Sydney Swans
- Loretta Johns -WORKS_FOR-> Sydney Swans
- Brodie Grundy -WORKS_FOR-> Sydney Swans
- Sydney Swans -PARTICIPATED_IN-> marquee pride match
- Sydney Swans -COMPETES_WITH-> Collingwood

Entities: Sydney Swans, Allies, Will Graham, Tom Papley, Murray Bushrangers, Coates Talent League, Gold Coast Suns Academy, Palm Beach Currumbin, AFL, Who’s the
next Gawn, Grundy or Dusty? AFL’s talent guru names top 40 draft prospects and compares them to current stars, National Draft Combine, Queensland, The Roar
</item 5>

<item 6>
‘Bleeds green and gold’: Ballsy career-saving call that’s delivered Arnie a record-breaking moment (The Roar | Sports Writers Blog, 2023-11-13T23:01:20+00:00)
Arnold – who also represented the Socceroos 56 times as a player – has been involved in national team coaching ranks since 2000, first as an assistant to Farina
and then Guus Hiddink, before a brief period as caretaker coach in 2006.

“It’s certainly a proud moment and an achievement that I’ll reflect on when the time is right,” Arnold said.

“I look at some of the names on that list – Farina, Arok, Thompson, Rasic – guys I played under or assisted in the early part of my coaching career, and it
gives you some perspective on what a journey it’s been so far.

Advertisement

“I think back to that caretaker period and how much I learnt about coaching; it was a challenging experience and one that on reflection I probably wasn’t ready
for. I’m grateful for the opportunity I was given and what it was able to teach me ahead of my pathway into club coaching.

Graph facts:
- ‘Bleeds green and gold’: Ballsy career-saving call that’s delivered Arnie a record-breaking moment -MENTIONS-> Socceroos World Cup qualification campaign
- ‘Bleeds green and gold’: Ballsy career-saving call that’s delivered Arnie a record-breaking moment -MENTIONS-> Australia
- ‘Bleeds green and gold’: Ballsy career-saving call that’s delivered Arnie a record-breaking moment -MENTIONS-> Graham Arnold
- ‘Bleeds green and gold’: Ballsy career-saving call that’s delivered Arnie a record-breaking moment -PUBLISHED_BY-> The Roar | Sports Writers Blog
- ‘Bleeds green and gold’: Ballsy career-saving call that’s delivered Arnie a record-breaking moment -MENTIONS-> Asian playoff against United Arab Emirates

Entities: ‘Bleeds green and gold’: Ballsy career-saving call that’s delivered Arnie a record-breaking moment, Socceroos, Arnold, Farina, Guus Hiddink, Arok,
Rasic, caretaker period, Thompson, The Roar
</item 6>

<item 7>
‘Bleeds green and gold’: Ballsy career-saving call that’s delivered Arnie a record-breaking moment (The Roar | Sports Writers Blog, 2023-11-13T23:01:20+00:00)
He got to answer critics like Mark Bosnich and Craig Foster as well – bluntly rejecting their negativity with the dismissive “who cares?”. That would have
tasted sweet.

On Tuesday, FA CEO James Johnson, who held firm on Arnold 20 months ago, got to congratulate the coach on his impending record-breaking match against Bangladesh
as 2026 World Cup qualifying starts on Thursday.

Advertisement

“Graham is one of the greatest contributors to the Socceroos,” Johnson said in a statement.

“He’s been a constant in the Australian football landscape for more than 30 years as both a player and coach, being a part of some of the Socceroos most
memorable moments.

“Anyone who knows Arnie will tell you he bleeds green and gold. His love for the Socceroos and football in Australia is both infectious and inspiring.

Graph facts:
- ASX set to open higher as Wall Street rebounds; $A rises -MENTIONS-> Australia
- A comprehensive list of 2023 tech layoffs -MENTIONS-> Australia
- ASX to open lower; Wall Street falls as the clamp tightens from rising bond yields -MENTIONS-> Australia
- Israel's defense undone by reliance on technology -MENTIONS-> Australia
- ASX set to drop as Wall Street’s September slump deepens -MENTIONS-> Australia

Entities: Australia, ‘Bleeds green and gold’: Ballsy career-saving call that’s delivered Arnie a record-breaking moment, Socceroos, Mark Bosnich, Craig Foster,
James Johnson, FA, record-breaking match against Bangladesh, 2026 World Cup qualifying, negativity, Arnold, Bangladesh, football, The Roar | Sports Writers Blog
</item 7>

<item 8>
‘Bleeds green and gold’: Ballsy career-saving call that’s delivered Arnie a record-breaking moment (The Roar | Sports Writers Blog, 2023-11-13T23:01:20+00:00)
His love for the Socceroos and football in Australia is both infectious and inspiring. That energy and passion is a huge element to the way he’s able to lead
and inspire our nation’s best male footballers in some of our sport’s most challenging arenas.

“On behalf of Football Australia, I’d like to congratulate Arnie on everything he’s achieved in reaching this milestone and we look forward to him continuing to
build on what is already an exceptional legacy.”

In a remarkable few weeks last year Arnold went from on the nose to a man revived. Since then he’s been linked to club jobs in Europe but opted to stick around
with the national team.

While national team coaches make important calls every game, Arnold’s record-breaking achievement is conceivably down to one big ballsy call in the playoff
against Peru.

Graph facts:
- ASX set to open higher as Wall Street rebounds; $A rises -MENTIONS-> Australia
- A comprehensive list of 2023 tech layoffs -MENTIONS-> Australia
- ASX to open lower; Wall Street falls as the clamp tightens from rising bond yields -MENTIONS-> Australia
- Israel's defense undone by reliance on technology -MENTIONS-> Australia
- ASX set to drop as Wall Street’s September slump deepens -MENTIONS-> Australia

Entities: Australia, ‘Bleeds green and gold’: Ballsy career-saving call that’s delivered Arnie a record-breaking moment, Socceroos, playoff against Peru, Peru,
Arnold, Football Australia, Europe, The Roar
</item 8>

<item 9>
How shooting Beyoncé and Kamala Harris made this photographer famous too (The Age, 2023-11-27T23:00:00+00:00)
Tyler Mitchell: “I’ve never been to Australia so it’s very exciting to sort of have a global or new conversation on a new continent with my work.″⁣ Credit:
Courtesy of the artist and Jack Shainman Gallery, New York The theme of the Triennial is “Magic, Matter, Memory”, making Mitchell’s pieces a logical inclusion.
“A lot of my work meditates on being between a heightened reality, or potentially even a magical reality and an actual reality,” he says.

Advertisement

“I’m creating what I call lightly staged moments. And so that means I’m setting up scenarios where these families are enjoying real moments of leisure in
outdoor spaces, usually open pastures and fields, and those moments are meant to stand in for a meditation on what it means to enjoy leisurely outdoor nature as
a black person.” Tyler Mitchell, Vastness, 2021.

Graph facts:
- Taylor Swift-Travis Kelce timeline: Everything to know about rumored romance between pop star, Chiefs TE -MENTIONS-> New York
- 2023 Kentucky online sports betting sites: Best legal sportsbooks, promos, bonuses, mobile apps, how to bet -MENTIONS-> New York
- How the conspiracy-fueled Epoch Times went mainstream and made millions -MENTIONS-> New York
- AfroFuture’s vision is to bring African art, cuisine, fashion, and music to a global audience. -MENTIONS-> New York
- Is Sam Bankman-Fried a bad ‘man’ or a good ‘boy’? Lawyers swap opening statements before first witnesses take the stand -MENTIONS-> New York

Entities: New York, Australia, Tyler Mitchell, How shooting Beyoncé and Kamala Harris made this photographer famous too, Triennial, Vastness, Magic, Matter,
Memory, Jack Shainman Gallery, leisurely outdoor nature, The Age, Kamala Harris, Beyoncé
</item 9>

<item 10>
The story behind anguished Wallabies star's photo (Wide World Of Sports, 2023-10-07T22:10:40+00:00)
"Last week 45,000 fans (versus Portugal), last night (60,000 in Lyon). Rugby's such a small aspect in terms of Australians but our fans have turned up here,
we've seen them and spoken to them and their unconditional love for us, the fact that they came 20,000km to get here and support us, has been awesome.

"My message is just to understand that we're working extremely hard, there's no stone unturned, especially the youth in this team, to leave the jersey in a
better place and just to stick with us.

"I know the results haven't been coming our way the last couple of years but when it does turn, that's what we're building towards.﻿"

Graph facts:
- ‘It happens’: All Blacks defend star centre after spray as Cheika heaps pressure on ‘favourites’ New Zealand -MENTIONS-> Wallabies
- RA set to announce local as high-performance director – and how it could help land Schmidt as Wallabies coach -MENTIONS-> Wallabies
- Eddie Jones says he needs to ‘give myself an uppercut’ but ‘100 percent’ committed to coaching Wallabies -MENTIONS-> Wallabies
- ‘Right people in right roles’: QRU breaks silence on RA’s reform hopes, backs review into Wallabies’ RWC flop -MENTIONS-> Wallabies
- Samu Kerevi -SAID-> Wallabies

Entities: Wide World Of Sports, Wallabies, The story behind anguished Wallabies star's photo, Lyon, rugby, Portugal
</item 10>
=== END ITEM id=56053c574fa7402a ===

=== ITEM id=50472cf3d210eb0f ===
Gold sentences and facts:
Sentence 0: The present Board of Trustees of the Fund which consists of the Secretary of Education as Chairman with the Director-General of the National Economic and Development Authority, the three Presidents of the Association of Christian Schools, Colleges and Universities (ACSCU), the Catholic Educational Association of the Philippines (CEAP) and the Philippine Association of Colleges and Universities (PACU) shall also constitute the proposed Board of Trustees of the PhilEd.
  Fact 0.0: The present Board of Trustees of the Fund consists of the Secretary of Education as Chairman.
  Fact 0.1: The present Board of Trustees of the Fund includes the Director-General of the National Economic and Development Authority.
  Fact 0.2: The present Board of Trustees of the Fund includes the President of the Association of Christian Schools, Colleges and Universities (ACSCU).
  Fact 0.3: The present Board of Trustees of the Fund includes the President of the Catholic Educational Association of the Philippines (CEAP).
  Fact 0.4: The present Board of Trustees of the Fund includes the President of the Philippine Association of Colleges and Universities (PACU).
  Fact 0.5: The present Board of Trustees of the Fund shall also constitute the proposed Board of Trustees of the PhilEd.
Sentence 1: In addition, Andrew Wilkinson, 72, has informed the HSF board of his intention to retire as a board director, which will reduce the company’s board to five directors.
  Fact 1.0: Andrew Wilkinson is 72 years old.
  Fact 1.1: Andrew Wilkinson has informed the HSF board of his intention to retire as a board director.
  Fact 1.2: Andrew Wilkinson's retirement will reduce the HSF board to five directors.

Retrieved items:
<item 1>
Hipgnosis Songs Fund Chair, Andrew Sutch, to step down; all-important ‘continuation vote’ likely to take place on October 26 (Music Business Worldwide,
2023-09-28T23:36:54+00:00)
its share price).

For example: On September 14, HSF’s market cap stood at approximately GBP £1.125 billion, which represented a 40% discount on the firm’s operative NAV of USD
$2.316 billion as of the end of March this year.

(Said ‘operative NAV’ was calculated by an independent valuer – Citrin Cooperman – using a discount rate of 8.5%.)

This is where the importance of that potential $440 million transaction comes in.

As MBW reported earlier this month, Hipgnosis Songs Fund’s shareholders are currently considering a proposal to sell 29 of the company’s catalogs to another
Hipgnosis company – the Blackstone-backed Hipgnosis Songs Capital (HSC) – for USD $440 million.

Graph facts:
- Andrew Wilkinson -WORKS_FOR-> Hipgnosis Songs Fund
- Hipgnosis Songs Fund -SAID-> valuation
- Christopher Brown -SAID-> Hipgnosis Songs Fund
- Hipgnosis Songs Fund’s board is concerned over Citrin Cooperman’s latest valuation of the company. What’s it going to do about it? -MENTIONS-> Hipgnosis Songs
Fund
- Hipgnosis Songs Fund -PARTICIPATED_IN-> Proposed sale of assets to Hipgnosis Songs Capital for USD $440 million (pre-costs) or $417.5 million (post-costs)

Entities: Hipgnosis Songs Fund, Hipgnosis Songs Capital, Citrin Cooperman, Blackstone, Music Business Worldwide, Andrew Sutch, continuation vote, operative NAV,
proposal to sell 29 catalogs, Andrew Sutch stepping down as Chair, Hipgnosis Songs Fund Chair, Andrew Sutch, to step down; all-important ‘continuation vote’
likely to take place on October 26
</item 1>

<item 2>
Hipgnosis Songs Fund Chair, Andrew Sutch, to step down; all-important ‘continuation vote’ likely to take place on October 26 (Music Business Worldwide,
2023-09-28T23:36:54+00:00)
This has been evidenced by the 44% total return, including right to income, transaction fees and expected taxes, realised since acquisition on the [$440
million] Transaction.

“Furthermore, the Board and the Investment Adviser are committed to ensuring that this value is achieved for shareholders by a re-rating of the share price and
are determined to deliver on the ongoing opportunity of the Company.”Music Business Worldwide

Graph facts:
- Andrew Wilkinson -WORKS_FOR-> Hipgnosis Songs Fund
- Hipgnosis Songs Fund -SAID-> valuation
- Christopher Brown -SAID-> Hipgnosis Songs Fund
- Hipgnosis Songs Fund’s board is concerned over Citrin Cooperman’s latest valuation of the company. What’s it going to do about it? -MENTIONS-> Hipgnosis Songs
Fund
- Hipgnosis Songs Fund -PARTICIPATED_IN-> Proposed sale of assets to Hipgnosis Songs Capital for USD $440 million (pre-costs) or $417.5 million (post-costs)

Entities: Hipgnosis Songs Fund, Music Business Worldwide, Andrew Sutch, continuation vote, share price, Hipgnosis Songs Fund Chair, Andrew Sutch, to step down;
all-important ‘continuation vote’ likely to take place on October 26, Investment Adviser, Board, shareholders, Transaction
</item 2>

<item 3>
Hipgnosis Songs Fund Chair, Andrew Sutch, to step down; all-important ‘continuation vote’ likely to take place on October 26 (Music Business Worldwide,
2023-09-28T23:36:54+00:00)
That potential transaction is being managed for both HSF and HSC by two separate teams within their mutual investment advisor, Hipgnosis Song Management (HSM),
which is run and part-owned by Merck Mercuriadis.

When it announced its intention to explore the $440 million catalog sale to HSC two weeks ago, Hipgnosis Songs Fund confirmed that one of its motivating factors
was that the transaction could potentially act “as a catalyst for a re-rating of [HSF’s] share price”.

Aka: Increasing that share price to bring it up closer to the current operative NAV of the company.

HSC’s $440 million offer to acquire the 29 catalogs, HSF confirmed earlier this month, equates to a 26% increase on the amount HSF paid for those catalogs to
acquire them in the past.

Graph facts:
- Andrew Wilkinson -WORKS_FOR-> Hipgnosis Songs Fund
- Hipgnosis Songs Fund -SAID-> valuation
- Christopher Brown -SAID-> Hipgnosis Songs Fund
- Hipgnosis Songs Fund’s board is concerned over Citrin Cooperman’s latest valuation of the company. What’s it going to do about it? -MENTIONS-> Hipgnosis Songs
Fund
- Hipgnosis Songs Fund -PARTICIPATED_IN-> Proposed sale of assets to Hipgnosis Songs Capital for USD $440 million (pre-costs) or $417.5 million (post-costs)

Entities: Hipgnosis Songs Fund, Music Business Worldwide, Andrew Sutch, continuation vote, operative NAV, HSC, Hipgnosis Song Management, Merck Mercuriadis,
$440 million catalog sale offer, 29 catalogs, share price, Hipgnosis Songs Fund Chair, Andrew Sutch, to step down; all-important ‘continuation vote’ likely to
take place on October 26
</item 3>

<item 4>
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

Graph facts:
- Hipgnosis Songs Fund Chair, Andrew Sutch, to step down; all-important ‘continuation vote’ likely to take place on October 26 -MENTIONS-> Hipgnosis Songs Fund
- Andrew Wilkinson -WORKS_FOR-> Hipgnosis Songs Fund
- Christopher Brown -SAID-> Hipgnosis Songs Fund
- Andrew Sutch -WORKS_FOR-> Hipgnosis Songs Fund
- Hipgnosis Songs Fund -PARTICIPATED_IN-> Proposed sale of assets to Hipgnosis Songs Capital for USD $440 million (pre-costs) or $417.5 million (post-costs)

Entities: Hipgnosis Songs Fund, HSF board, valuation, Citrin Cooperman Advisors LLC, Hipgnosis Songs Fund’s board is concerned over Citrin Cooperman’s latest
valuation of the company. What’s it going to do about it?, Music Business Worldwide, Hipgnosis Song Management
</item 4>

<item 5>
Hipgnosis Songs Fund Chair, Andrew Sutch, to step down; all-important ‘continuation vote’ likely to take place on October 26 (Music Business Worldwide,
2023-09-28T23:36:54+00:00)
The $440 million price also reflects a +51% premium on the value of the 29 catalogs implied by Hipgnosis Songs Fund’s public market cap for the 30 days ending
September 13.

What happened today?

Amongst the news confirmed by HSF today (September 28) is that the firm’s Chair of its board since 2018, Andrew Sutch, 73, is stepping down, both as HSF board
Chair and as a director of the company.

HSF says it’s begun a process to recruit a new Chair; Sutch (pictured inset) will leave his post as and when this person is hired, or in any case before HSF’s
AGM next year (in Q4 2024).

In addition, Andrew Wilkinson, 72, has informed the HSF board of his intention to retire as a board director, which will reduce the company’s board to five
directors.

Graph facts:
- Hipgnosis Songs Fund -PARTICIPATED_IN-> Proposed sale of assets to Hipgnosis Songs Capital for USD $440 million (pre-costs) or $417.5 million (post-costs)
- Hipgnosis Songs Fund -SAID-> valuation
- Christopher Brown -SAID-> Hipgnosis Songs Fund
- Hipgnosis Songs Fund’s board is concerned over Citrin Cooperman’s latest valuation of the company. What’s it going to do about it? -MENTIONS-> Hipgnosis Songs
Fund
- Hipgnosis Songs Fund -PARTICIPATED_IN-> Sale of 20,000 non-core songs from HSF’s portfolio for $23.1 million

Entities: Hipgnosis Songs Fund, Music Business Worldwide, Andrew Sutch, continuation vote, Andrew Wilkinson, AGM, Hipgnosis Songs Fund Chair, Andrew Sutch, to
step down; all-important ‘continuation vote’ likely to take place on October 26
</item 5>

<item 6>
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

Graph facts:
- Christopher Brown -SAID-> Hipgnosis Songs Fund
- Andrew Sutch -WORKS_FOR-> Hipgnosis Songs Fund
- Hipgnosis Songs Fund Chair, Andrew Sutch, to step down; all-important ‘continuation vote’ likely to take place on October 26 -MENTIONS-> Hipgnosis Songs Fund
- Hipgnosis Songs Fund -SAID-> valuation
- Andrew Wilkinson -WORKS_FOR-> Hipgnosis Songs Fund

Entities: Rob Naylor, Hipgnosis Songs Fund, Hipgnosis Songs Fund board, Citrin Cooperman, Hipgnosis Songs Fund’s board is concerned over Citrin Cooperman’s
latest valuation of the company. What’s it going to do about it?, music royalties, Music Business Worldwide, JKBX
</item 6>

<item 7>
Hipgnosis Songs Fund Chair, Andrew Sutch, to step down; all-important ‘continuation vote’ likely to take place on October 26 (Music Business Worldwide,
2023-09-28T23:36:54+00:00)
Hipgnosis Songs Fund (HSF), the UK-listed music rights fund, looks likely to hold its all-important 2023 ‘continuation vote’ on Thursday, October 26, according
to an announcement from the firm today.

Why’s that ‘likely’ rather than ‘definitely’? Because there’s still a possibility it might get bumped into November – the reasons for which we’ll get into later
in this article.

The ‘continuation vote’ will take place at HSF’s Annual General Meeting (AGM), and will see the firm’s shareholders vote on whether to keep the entity going as
a closed-ended investment entity i.e. a publicly-traded bundle of music rights.

How will that vote go? Obviously we don’t know yet – and there’s the small matter of a $440 million transaction looming that could potentially influence things.

Graph facts:
- Andrew Wilkinson -WORKS_FOR-> Hipgnosis Songs Fund
- Hipgnosis Songs Fund -SAID-> valuation
- Christopher Brown -SAID-> Hipgnosis Songs Fund
- Hipgnosis Songs Fund’s board is concerned over Citrin Cooperman’s latest valuation of the company. What’s it going to do about it? -MENTIONS-> Hipgnosis Songs
Fund
- Hipgnosis Songs Fund -PARTICIPATED_IN-> Proposed sale of assets to Hipgnosis Songs Capital for USD $440 million (pre-costs) or $417.5 million (post-costs)

Entities: Hipgnosis Songs Fund, Music Business Worldwide, Andrew Sutch, continuation vote, Annual General Meeting, $440 million transaction, Hipgnosis Songs
Fund Chair, Andrew Sutch, to step down; all-important ‘continuation vote’ likely to take place on October 26
</item 7>

<item 8>
Hipgnosis Songs Fund Chair, Andrew Sutch, to step down; all-important ‘continuation vote’ likely to take place on October 26 (Music Business Worldwide,
2023-09-28T23:36:54+00:00)
HSF also confirmed a number of other new measures in a ‘circular’ to its shareholders today.

These measures included the fact that, if Hipgnosis Songs Fund shareholders vote to remain a publicly-traded fund (via that 2023 ‘continuation vote’) at the
firm’s next AGM, then another ‘continuation vote’ will be put in front of HSF shareholders at an Extraordinary General Meeting in January 2026.

(The ‘continuation vote’ that will take place in Q4 2023 has been scheduled ever since HSF went public in 2018 – i.e. five years after it floated. The 2026
‘Continuation Vote’, as newly-agreed, means another one is now being scheduled for around two years’ time.)

In addition, HSF today confirmed some new amendments to its ‘Investment Advisory Agreement’ (IAA) with HSM.

Graph facts:
- Andrew Wilkinson -WORKS_FOR-> Hipgnosis Songs Fund
- Hipgnosis Songs Fund -SAID-> valuation
- Christopher Brown -SAID-> Hipgnosis Songs Fund
- Hipgnosis Songs Fund’s board is concerned over Citrin Cooperman’s latest valuation of the company. What’s it going to do about it? -MENTIONS-> Hipgnosis Songs
Fund
- Hipgnosis Songs Fund -PARTICIPATED_IN-> Proposed sale of assets to Hipgnosis Songs Capital for USD $440 million (pre-costs) or $417.5 million (post-costs)

Entities: Hipgnosis Songs Fund, HSM, Music Business Worldwide, Andrew Sutch, AGM, 2023 continuation vote, 2026 Continuation Vote, Extraordinary General Meeting,
Investment Advisory Agreement, Hipgnosis Songs Fund Chair, Andrew Sutch, to step down; all-important ‘continuation vote’ likely to take place on October 26
</item 8>

<item 9>
Hipgnosis Songs Fund Chair, Andrew Sutch, to step down; all-important ‘continuation vote’ likely to take place on October 26 (Music Business Worldwide,
2023-09-28T23:36:54+00:00)
If a ‘superior proposal’ (versus the $440 million offer from HSC) is received during the ‘go-shop’ period, said HSF today – whether or not HSC uses its
‘matching right’ to beat it – it may be necessary to adjourn the HSF AGM (and therefore the 2023 ‘continuation vote’) into November.

In its note to shareholders today, the HSF board reiterated its belief – and Hipgnosis Song Management’s belief – that HSF owns “a unique portfolio of iconic,
culturally significant songs that will deliver strong long-term value as they benefit from the structural tailwinds in the music industry”.

The note added: “Furthermore, the [HSF] Board believes that the Investment Adviser’s approach to Song Management should enable the Company to outperform the
wider music market.

Graph facts:
- Andrew Wilkinson -WORKS_FOR-> Hipgnosis Songs Fund
- Hipgnosis Songs Fund -SAID-> valuation
- Christopher Brown -SAID-> Hipgnosis Songs Fund
- Hipgnosis Songs Fund’s board is concerned over Citrin Cooperman’s latest valuation of the company. What’s it going to do about it? -MENTIONS-> Hipgnosis Songs
Fund
- Hipgnosis Songs Fund -PARTICIPATED_IN-> Proposed sale of assets to Hipgnosis Songs Capital for USD $440 million (pre-costs) or $417.5 million (post-costs)

Entities: Hipgnosis Songs Fund, HSF board, Music Business Worldwide, music industry, Andrew Sutch, HSC, Hipgnosis Song Management, 2023 continuation vote,
Hipgnosis Songs Fund Chair, Andrew Sutch, to step down; all-important ‘continuation vote’ likely to take place on October 26, Investment Adviser, HSF AGM, $440
million offer from HSC, Song Management, music market
</item 9>

<item 10>
Graph query row:
article: 200+ of the best deals from Amazon's Cyber Monday sale
published_on: 2023-11-27T08:45:59+00:00
publisher: Mashable
relevant_relationships: [{'entity': 'Echo', 'relationship': 'MENTIONS'}, {'entity': 'Cyber Weekend', 'relationship': 'MENTIONS'}, {'entity': 'Echo Show',
'relationship': 'MENTIONS'}, {'entity': 'smart screen', 'relationship': 'MENTIONS'}, {'entity': 'smart lights', 'relationship': 'MENTIONS'}, {'entity': 'Audible
Premium Plus', 'relationship': 'MENTIONS'}, {'entity': 'Echo smart displays', 'relationship': 'MENTIONS'}, {'entity': 'Echo smart speakers', 'relationship':
'MENTIONS'}, {'entity': 'Fire TVs', 'relationship': 'MENTIONS'}, {'entity': 'MGM+', 'relationship': 'MENTIONS'}, {'entity': 'Liquid Retina display',
'relationship': 'MENTIONS'}, {'entity': 'A14 Bionic chip', 'relationship': 'MENTIONS'}, {'entity': 'USB-C charging', 'relationship': 'MENTIONS'}, {'entity':
'Microsoft Surface Laptop Go 3', 'relationship': 'MENTIONS'}, {'entity': 'Surface Go 2', 'relationship': 'MENTIONS'}, {'entity': 'Go 3', 'relationship':
'MENTIONS'}, {'entity': 'QuietComfort 45s', 'relationship': 'MENTIONS'}, {'entity': "Amazon's Cyber Monday sale", 'relationship': 'MENTIONS'}, {'entity':
'META50', 'relationship': 'MENTIONS'}, {'entity': "Mashable's daily Deals newsletter", 'relationship': 'MENTIONS'}, {'entity': 'Terms of Use', 'relationship':
'MENTIONS'}, {'entity': 'Tablet deals', 'relationship': 'MENTIONS'}, {'entity': 'Gaming keyboards', 'relationship': 'MENTIONS'}, {'entity': 'VR headsets',
'relationship': 'MENTIONS'}, {'entity': 'Vacuum deals', 'relationship': 'MENTIONS'}, {'entity': 'Cordless vacuums', 'relationship': 'MENTIONS'}, {'entity':
'Robot vacuums and vacuum/mop hybrids', 'relationship': 'MENTIONS'}, {'entity': 'Kitchen deals', 'relationship': 'MENTIONS'}, {'entity': 'Pizza ovens',
'relationship': 'MENTIONS'}, {'entity': 'Sparkling water makers', 'relationship': 'MENTIONS'}, {'entity': 'Toaster ovens', 'relationship': 'MENTIONS'},
{'entity': 'Amazon device and service deals', 'relationship': 'MENTIONS'}, {'entity': 'Blink', 'relationship': 'MENTIONS'}, {'entity': 'Chromecast',
'relationship': 'MENTIONS'}, {'entity': 'Google Nest', 'relationship': 'MENTIONS'}, {'entity': 'Fitness deals', 'relationship': 'MENTIONS'}, {'entity':
'Smartwatches and fitness trackers', 'relationship': 'MENTIONS'}, {'entity': 'Apple deals', 'relationship': 'MENTIONS'}, {'entity': 'Beauty tech deals',
'relationship': 'MENTIONS'}, {'entity': 'Drone deals', 'relationship': 'MENTIONS'}, {'entity': 'Camera deals', 'relationship': 'MENTIONS'}, {'entity': 'GoPro',
'relationship': 'MENTIONS'}, {'entity': 'Lego deals', 'relationship': 'MENTIONS'}, {'entity': 'Gift card deals', 'relationship': 'MENTIONS'}, {'entity': 'Mac',
'relationship': 'MENTIONS'}, {'entity': 'Microsoft', 'relationship': 'MENTIONS'}, {'entity': 'Apple', 'relationship': 'MENTIONS'}, {'entity': 'AirPods',
'relationship': 'MENTIONS'}, {'entity': 'AirPods Max', 'relationship': 'MENTIONS'}, {'entity': 'Echo devices', 'relationship': 'MENTIONS'}, {'entity': 'Alexa',
'relationship': 'MENTIONS'}, {'entity': 'privacy policy', 'relationship': 'MENTIONS'}, {'entity': 'iPad', 'relationship': 'MENTIONS'}, {'entity': 'Eero',
'relationship': 'MENTIONS'}, {'entity': 'Meta Quest 2', 'relationship': 'MENTIONS'}, {'entity': 'Apple Watch', 'relationship': 'MENTIONS'}, {'entity': 'Smart
Home Deals', 'relationship': 'MENTIONS'}, {'entity': 'Speaker Deals', 'relationship': 'MENTIONS'}, {'entity': 'Bose', 'relationship': 'MENTIONS'}, {'entity':
'Earbuds', 'relationship': 'MENTIONS'}, {'entity': 'MacBooks', 'relationship': 'MENTIONS'}, {'entity': 'MacBook Air', 'relationship': 'MENTIONS'}, {'entity':
'ANC', 'relationship': 'MENTIONS'}, {'entity': 'Echo Buds', 'relationship': 'MENTIONS'}, {'entity': 'MacBook', 'relationship': 'MENTIONS'}, {'entity':
'headphones', 'relationship': 'MENTIONS'}, {'entity': 'Fire TV', 'relationship': 'MENTIONS'}, {'entity': 'Black Friday', 'relationship': 'MENTIONS'}, {'entity':
'Kindle', 'relationship': 'MENTIONS'}, {'entity': 'Prime', 'relationship': 'MENTIONS'}, {'entity': 'Gaming mice', 'relationship': 'MENTIONS'}, {'entity': 'Black
Friday sale', 'relationship': 'MENTIONS'}, {'entity': 'Intel', 'relationship': 'MENTIONS'}, {'entity': 'Amazon', 'relationship': 'MENTIONS'}, {'entity': 'Cyber
Monday sale', 'relationship': 'MENTIONS'}, {'entity': 'Intel Core i5 processor', 'relationship': 'MENTIONS'}, {'entity': 'Fire tablets', 'relationship':
'MENTIONS'}, {'entity': 'gaming deals', 'relationship': 'MENTIONS'}, {'entity': 'Amazon credit', 'relationship': 'MENTIONS'}, {'entity': 'gaming headsets',
'relationship': 'MENTIONS'}, {'entity': 'Apple Watches', 'relationship': 'MENTIONS'}]
page: None
passage: None
</item 10>
=== END ITEM id=50472cf3d210eb0f ===
