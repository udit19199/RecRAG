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

This file has 6 items.

=== ITEM id=acc7bb0919fa97ca ===
Gold sentences and facts:
Sentence 0: Tyreek Hill had two touchdowns among his 157 receiving yards to help the Miami Dolphins rout the Washington Commanders (4-9).
  Fact 0.0: Tyreek Hill had two touchdowns in this game.
  Fact 0.1: Tyreek Hill had 157 receiving yards in this game.
  Fact 0.2: Tyreek Hill helped the Miami Dolphins rout the Washington Commanders.
  Fact 0.3: The Washington Commanders were 4-9 at this point.
Sentence 1: • Six catches for 163 yards and a touchdown against Carolina left the Dolphins’ Tyreek Hill with 812 yards through the first six games.
  Fact 1.0: Tyreek Hill made six catches for 163 yards and a touchdown against Carolina.
  Fact 1.1: Tyreek Hill had 812 yards through the first six games.
Sentence 2: Of Miami's five remaining opponents, four have pass defenses ranked in the league's top eight.
  Fact 2.0: Miami has five remaining opponents.
  Fact 2.1: Four of Miami's five remaining opponents have pass defenses ranked in the league's top eight.

Retrieved items:
<item 1>
N.F.L. Executives on the Biggest Questions Facing Super Bowl Contenders (The New York Times, 2023-10-16T23:58:53+00:00)
Six of the 17 won anyway. The Dolphins were the only one of those 223 teams to win and cover the spread, which was 13.5 for this game.

• Six catches for 163 yards and a touchdown against Carolina left the Dolphins’ Tyreek Hill with 812 yards through the first six games. That ranks second in NFL
history to Hall of Famer Don Hutson, who had 819 yards through six with Green Bay in 1942.
</item 1>

<item 2>
NFL awards odds: Six longshots bettors should consider for MVP, OPOY, DPOY and more (Sporting News, 2023-12-05T20:45:25+00:00)
1 seed in the conference.

Compared to the 49ers, Cowboys and Eagles, Goff and the Lions may have the easiest schedule to finish the season.

But bettors should also pay attention to C.J. Stroud. He leads the league in passing yards, is one of four quarterbacks eho have started the whole season with a
passer rating over 100, and his touchdown to interception ratio is the best in the NFL (4 to 1)

As for the value of his play, Stroud has taken one of the worst teams in the NFL (11 total wins the last three seasons) and turned it into a playoff contender
and potential division champion.

Offensive Player of the Year

Current favorite: Tyreek Hill

Longshot: CeeDee Lamb

There's a good reason why Tyreek Hill is such a heavy favorite to win OPOY. The Dolphins wide receiver is on track to become the first player with 2,000-plus
receiving yards in a single season.
</item 2>

<item 3>
Fantasy Football Week 16 Wide Receiver Preview: Waiver adds, DFS plays, and more (CBSSports.com, 2023-12-19T20:51:32+00:00)
Weekly Breakdown MATCHUP @ MIA MIA -1.5 O/U 51 OPP VS WR 15th PROJ PTS 19.4 WR RNK 2nd YTD Stats REC 103 TAR 141 REYDS 1306 TD 10 FPTS/G 21.6 The choice between
Lamb and Tyreek Hill will not be an easy one. The tiebreaker for me is that Lamb is $900 cheaper and Hill may still not be 100% by the time the Cowboys and
Dolphins kick off. Last week was the first time since Week 9 that Lamb failed to reach the end zone and the second time since Week 5 that he didn't score at
least 17 PPR Fantasy points.

Contrarian DFS Play (WR Preview) Projections powered by Sportsline Jaylen Waddle WR MIA Miami • #17

Age: 25 • Experience: 3 yrs.
</item 3>

<item 4>
NFL roundup: Pats hit new low as Lions reach 9-3 for first time since 1962 (The Guardian, 2023-12-03T22:11:26+00:00)
Keenan Allen came back from a thigh injury to catch five passes for 58 yards. But the only scoring the Chargers could manage was a pair of 38-yard field goals
in the second quarter. New England made a quarterback switch, benching Mac Jones in favor of Bailey Zappe. But the Patriots (2-10) still lost their fifth
straight game. For the first time in franchise history, the Patriots have been shut out at home twice in a season. Zappe was sacked five times – two by Khalil
Mack, giving him a career-high 15 for the season and 99 in his career. Zappe was 13 for 25 for 141 yards; he did avoid an interception for the first time in
three games, the previous two in relief of Jones.

Tyreek Hill had two touchdowns among his 157 yards receiving to help the Miami Dolphins rout the Washington Commanders 45-15.
</item 4>

<item 5>
MVPs for all 16 AFC teams at the midpoint (Yardbarker, 2023-11-09T21:51:02+00:00)
MIAMI DOLPHINS | WR Tyreek Hill: Arguably the best wide receiver in the NFL, Hill has somehow reached a new level of dominance in his age-29 season. The
seven-time Pro Bowler leads the NFL in receiving yards (1,076) and TD catches (eight). As long as the Dolphins maintain their success on offense down the
stretch, Hill should earn Offensive Player of the Year honors. He's on pace to break Calvin Johnson's season record for receiving yards (1,964).



NEW YORK JETS | LB Quincy Williams: While his brother, Quinnen, may be the better-known Jets defender in the family, Williams — the AFC Defensive Player of the
Month for October — is also making a name for himself. Aside from being the fifth-highest-rated linebacker in the NFL by Pro Football Focus, Williams is tied
for 17th in tackles (74) and tied for ninth in TFLs (nine) among all defenders.
</item 5>

<item 6>
Dolphins vs. Eagles live score, updates, highlights from NFL 'Sunday Night Football' game (Sporting News, 2023-10-22T23:28:04+00:00)
— Oh yeah, Julio Jones is back in the NFL.

What a group 🤌 pic.twitter.com/fV9WgGOpRY — Philadelphia Eagles (@Eagles) October 22, 2023

7:41 p.m. — Believe it or not, Tyreek Hill has a chance to break 1,000 yards tonight. He would need 186, but with the way he's played this season, that at least
seems to be within reach.

7:29 p.m. — Don't forget the Eagles will be back in Kelly Green tonight. This will be a fascinating uniform matchup.

7:20 p.m. — It's 55 degrees and dropping in Philadelphia. That's not unusual for the Eagles, but will Tua Tagovailoa be able to rise to the occasion in cooler
weather?

What channel is Dolphins vs. Eagles on today?

Game: Dolphins vs. Eagles

Dolphins vs.
</item 6>

<item 7>
Dolphins vs. Eagles live score, updates, highlights from NFL 'Sunday Night Football' game (Sporting News, 2023-10-22T23:28:04+00:00)
That's a pick six, and the Dolphins have tied the game just like that after the no-call.

10:22 p.m. — This looks more obvious than a certain penalty on James Bradberry from February.

Looks like the refs missed another facemask, this one on James Bradberry pic.twitter.com/fYYeWnEWaGhttps://t.co/54oa1jfeHf — Dov Kleiman (@NFL_DovKleiman)
October 23, 2023

10:20 p.m. — James Bradberry appears to get ahold of Cedrick Wilson's face mask on 4th & 3, but no flag flies. The Dolphins turn it over on downs.

10:19 p.m. — Tyreek Hill just can't hang onto what would've been a big-time catch. It's not a fumble, but it is a fourth down for the Dolphins. They'll go for
it.

10:16 p.m.
</item 7>

<item 8>
N.F.L. Executives on the Biggest Questions Facing Super Bowl Contenders (The New York Times, 2023-10-16T23:58:53+00:00)
Receiving Yards Through Six NFL Games Rk Player Season Yds 1 Don Hutson 1942 819 2 Tyreek Hill 2023 812 3 Wes Welker 2011 785 4 Elroy Hirsch 1951 766 5 Santana
Moss 2005 743 6 DeAndre Hopkins 2015 726 7 Raymond Berry 1960 725 8 Jerry Rice 1986 723 9 Bob Boyd 1954 718 10 Bobby Mitchell 1962 714

• Jared Goff is enjoying the second-best start to a season for his career from a statistical standpoint, behind only his 2018 Super Bowl season with the Rams.
With the Lions at 5-1 and Goff heading toward a likely contract extension, could another team come after his injured backup, former University of Tennessee
quarterback Hendon Hooker?
</item 8>

<item 9>
Dolphins vs. Eagles live score, updates, highlights from NFL 'Sunday Night Football' game (Sporting News, 2023-10-22T23:28:04+00:00)
9:43 p.m. — Hill is up to 65 yards on eight catches. No other Dolphins receiver has more than two catches, and Hill now has 879 yards on the season.

Eagles 17, Dolphins 10

9:39 p.m. TOUCHDOWN — Tagovailoa hits Hill in the end zone for a touchdown! Hill was facing double coverage but simply beat both defenders. Miami has the
deficit back to one possession.

9:38 p.m. — Rookie Nolan Smith sacks Tagovailoa with 44 seconds remaining in the half.

9:37 p.m. — The Dolphins are down to the Eagles' 29-yard line after catches by Braxton Berrios and Tyreek Hill. They have a chance to score on back-to-back
possessions, as they get the ball out of halftime.

9:33 p.m. TWO-MINUTE WARNING — Woah!
</item 9>

<item 10>
NFL roundup: Pats hit new low as Lions reach 9-3 for first time since 1962 (The Guardian, 2023-12-03T22:11:26+00:00)
The Titans blew a 17-7 lead. Nick Folk missed his first extra point this season with 5:26 left in regulation, which would have put Tennessee up 26-25.

skip past newsletter promotion Sign up to Soccer with Jonathan Wilson Free weekly newsletter Jonathan Wilson brings expert analysis on the biggest stories from
European soccer Privacy Notice: Newsletters may contain info about charities, online ads, and content funded by outside parties. For more information see our
Newsletters may contain info about charities, online ads, and content funded by outside parties. For more information see our Privacy Policy . We use Google
reCaptcha to protect our website and the Google Privacy Policy and Terms of Service apply. after newsletter promotion

Miami Dolphins 45–15 Washington Commanders

Tyreek Hill had two touchdowns among his 157 receiving yards to help the Miami Dolphins rout the Washington Commanders (4-9). The AFC East-leading Dolphins are
9-3 for the first time since 2001.
</item 10>
=== END ITEM id=acc7bb0919fa97ca ===

=== ITEM id=652bb1c69fefe5af ===
Gold sentences and facts:
Sentence 0: But “all of that, all of it, was built on lies,” Rehn declared to the jury.
  Fact 0.0: Rehn declared to the jury that 'all of that, all of it, was built on lies.'
Sentence 1: Using Caroline Ellison, his on-again off-again girlfriend and the CEO of Alameda, as a front, Bankman-Fried had “secret access” to customer money—both cash and crypto—the government claimed.
  Fact 1.0: Bankman-Fried used Caroline Ellison as a front.
  Fact 1.1: Caroline Ellison is Bankman-Fried's on-again off-again girlfriend.
  Fact 1.2: Caroline Ellison is the CEO of Alameda.
  Fact 1.3: The government claimed Bankman-Fried had 'secret access' to customer money.
  Fact 1.4: The government claimed the customer money was both cash and crypto.
Sentence 2: The prosecution painted Bankman-Fried as someone who knowingly committed fraud to achieve great wealth, power and influence, while the defense countered that the FTX founder acted in good faith, never meant to commit fraud or steal and basically got in over his head.
  Fact 2.0: The prosecution portrayed Bankman-Fried as someone who knowingly committed fraud.
  Fact 2.1: The prosecution said Bankman-Fried committed fraud to achieve great wealth.
  Fact 2.2: The prosecution said Bankman-Fried committed fraud to achieve great power.
  Fact 2.3: The prosecution said Bankman-Fried committed fraud to achieve great influence.
  Fact 2.4: The defense countered that the FTX founder acted in good faith.
  Fact 2.5: The defense countered that the FTX founder never meant to commit fraud or steal.
  Fact 2.6: The defense countered that the FTX founder basically got in over his head.

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
Sam Bankman-Fried was a terrible boyfriend (The Verge, 2023-10-10T23:50:21+00:00)
That gave Ellison an unusual view of his character. “He was very ambitious,” she said. Besides telling her about his presidential chances, he also told her that
if there was a coin flip where tails destroyed the world and heads made the world twice as good, he’d flip the coin. He called this being “risk-neutral,” which
seems like a fancy way of saying he was a gambling addict.

She was named co-CEO of Alameda with Sam Trabucco in 2021, while she and Bankman-Fried were broken up, and CEO in 2022. The goal, Bankman-Fried told her, was to
“optically” separate Alameda Research and FTX. “The whole time we were dating, he was my boss at work,” she said. They broke up because she wanted more from the
relationship; Bankman-Fried was distant and not paying enough attention to her.
</item 2>

<item 3>
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
</item 3>

<item 4>
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
</item 4>

<item 5>
SBF Trial: The latest updates from the FTX collapse’s courtroom drama (TechCrunch, 2023-10-06T17:22:02+00:00)
SBF trial: Everything to know from the FTX courtroom ahead of his testimony Sam Bankman-Fried is in court for alleged fraud and money laundering while leading
crypto exchange FTX and sister company Alameda. Here’s our latest coverage as the case nears its end.

The trial of Bankman-Fried, the former CEO and co-founder of collapsed crypto exchange FTX, is entering its next chapter as proceedings resume Thursday. As
confirmed during a teleconference Wednesday, SBF will be taking the stand, along with a handful of other witnesses as the onus shifts toward the defense after
the prosecution brought its case to its conclusion.

As former federal prosecutor Josh Naftalis told TechCrunch, SBF taking to the stand in his own trial is a “Hail Mary” ahead of a verdict and potential
sentencing for seven counts tied to fraud and money laundering.

“Once it goes to cross[-examination], he doesn’t get to say, ‘I’m done,’” Naftalis said.
</item 5>

<item 6>
SBF’s trial starts soon, but how did he — and FTX — get here? (TechCrunch, 2023-10-01T14:00:29+00:00)
U.S. Attorney Damian Williams called Bankman-Fried’s alleged crimes “one of the biggest financial frauds in American history,” in a press conference.

The aftermath

FTX co-founder and former CTO Gary Wang, and Alameda Research’s former CEO, Caroline Ellison, both pleaded guilty in December 2022 to federal criminal charges
in relation to the FTX collapse. They’re also facing civil penalties from the U.S. Securities and Exchange Commission (SEC) and Commodity Futures Trading
Commission (CFTC) alongside the criminal charges. Wang and Ellison plan to cooperate with prosecutors and will be major witnesses in the trial, given their
close ties to Bankman-Fried, FTX and Alameda.

In January, Bankman-Fried pleaded not guilty to all counts, which include wire fraud, conspiracy to commit money laundering and conspiracy to misuse customer
funds. He could face up to 115 years in jail if convicted on all charges.
</item 6>

<item 7>
SBF, riding high on FTX, reportedly offered $5B to Trump to cancel President re-run (TechCrunch, 2023-10-02T13:11:07+00:00)
Come Tuesday, our eyes will be trained on the Manhattan Federal Court for the trial of Sam Bankman-Fried, the disgraced cryptocurrency entrepreneur accused of
orchestrating “one of the biggest financial frauds in American history”: the multibillion-dollar collapse of FTX, the crypto exchange he founded and led.

SBF, as he’s known, has pleaded not-guilty to some seven charges of fraud and conspiracy. But, as we wait to hear both his version and the government’s version
of what happened, there will be another tale recounting how things played out: “Going Infinite,” a book by Michael Lewis chronicling the rise and fall of SBF,
is set to debut on the first day of the trial.

The episode has fired up a lot of people, indignant at what they feel was too-easy treatment for someone who many believe brought down the whole crypto house.

60 Minutes got rightly community noted here.
</item 7>

<item 8>
Sam Bankman-Fried was a terrible boyfriend (The Verge, 2023-10-10T23:50:21+00:00)
(Fraud, conspiracy to commit fraud, and money laundering, in case you were wondering.) “We ultimately took about $14 billion, some of which we were not able to
pay back.” She tilted her head down to answer the questions, then lifted her head when she’d finished her answer.

Bankman-Fried called this being “risk-neutral,” which seems like a fancy way of saying he was a gambling addict

In Ellison’s telling, Alameda was troubled from her earliest time there in 2018. “Shortly after I started, I learned the company was in worse shape than I
realized,” Ellison said. Alameda had initially been funded with loans “from acquaintances,” she said, and those loans were recalled a few weeks after she
arrived. (There was a staff revolt within Alameda Research over lost millions and general financial chaos, according to Michael Lewis’ Going Infinite.) Ellison
asked Bankman-Fried why he hadn’t shared the company’s shaky circumstances in the job offer. “He hadn’t known how to tell me,” she said.
</item 8>

<item 9>
SBF’s trial starts soon, but how did he — and FTX — get here? (TechCrunch, 2023-10-01T14:00:29+00:00)
In a published DM exchange with a Vox reporter, he said he regretted filing for bankruptcy and thought that “regulators make everything worse.”

SBF arrested

Bankman-Fried was arrested in December 2022 in the Bahamas, where FTX was based. He was then extradited to the U.S. to face a number of criminal charges. He was
released on a $250 million bail bond, and he remained under house arrest at his parents’ home in Palo Alto. This was revoked in August after he was accused of
intimidating Alameda’s former CEO, Caroline Ellison, by leaking her private diary.

Ray represented the company during a House Financial Services Committee hearing regarding FTX. When asked whether the firm had significant risk management
systems, Ray said at the time that “there were virtually no internal controls and no separateness whatsoever” and added that he did not “trust a single piece of
paper” in the exchange’s organization. U.S.
</item 9>

<item 10>
SBF Trial: The latest updates from the FTX collapse’s courtroom drama (TechCrunch, 2023-10-06T17:22:02+00:00)
He, like many of SBF’s former colleagues, opted to cooperate with authorities and testified that SBF asked him to create a theoretical “legal justification” for
the use of billions in consumer funds just days before FTX filed for bankruptcy.

Robert Boroujerdi, managing director of the hedge fund Third Point, testified about his experience working with SBF and FTX, in a partnership that ultimately
resulted in a $60 million investment and a lack of awareness about the special relationship between Alameda Research and FTX. Had he known of such a
relationship, he testified that Third Point would not have invested in FTX in the first place.

And earlier in the week, a University of Notre Dame professor who helped the government prosecution of Enron and WorldCom, Peter Easton, testified about what he
uncovered after tracing the flow of billions of dollars between Alameda Research and FTX.
</item 10>
=== END ITEM id=652bb1c69fefe5af ===

=== ITEM id=7d86fa4de522320a ===
Gold sentences and facts:
Sentence 0: For the masses of NFL fans across the U.S., Thursday's clash between the Lions and Packers will be aired on Amazon Prime Video.
  Fact 0.0: Thursday's game between the Lions and Packers will be aired on Amazon Prime Video.
  Fact 0.1: The broadcast is for NFL fans across the U.S.
Sentence 1: For the masses of NFL fans across the U.S., Thursday's clash between the Bears and Panthers will be aired on Amazon Prime Video.
  Fact 1.0: Thursday's game between the Bears and Panthers will be aired on Amazon Prime Video.
  Fact 1.1: The broadcast is for NFL fans across the U.S.

Retrieved items:
<item 1>
Lions vs. Packers live score, updates, highlights from NFL 'Thursday Night Football' game (Sporting News, 2023-09-28T23:54:07+00:00)
11:11 p.m.: FIELD GOAL! Carlson's try from distance makes the deficit two possessions.

11:08 p.m.: Not so fast. ... A quick three-and-out gives Green Bay a glimmer. That glimmer grows even brighter after an impressive return by Keisean Nixon.
Let's see what they can do with 3:30 left.

11:04 p.m.: INTERCEPTION! That's wraps. Love misses his target, instead finding the mitts of Jacobs for his second pick of the day. Strong showing by this
Detroit defense. And that should put things to bed.

10:54 p.m.: TOUCHDOWN! Walker's penalty proves vital as Montgomery finds pay dirt from a couple yards out on fourth down. Gutsy call by Campbell. And one that
pays off in spades for Detroit, which extends its lead to 17.

Lions 34, Packers 17

10:50 p.m.
</item 1>

<item 2>
Bears vs. Panthers live score, updates, highlights from NFL 'Thursday Night Football' game (Sporting News, 2023-11-09T23:04:33+00:00)
Bears vs. Panthers start time

Date: Thursday, Nov. 9

Thursday, Nov. 9 Start time: 8:15 p.m. ET

Thursday night's game between the Bears and Panthers is set to kick off at 8:15 p.m. ET (7:15 p.m. local time) from Soldier Field in Chicago.

What channel is Bears vs. Panthers on today?

Game: Chicago Bears vs. Carolina Panthers

Chicago Bears vs. Carolina Panthers Date: Thursday, Nov. 9

Thursday, Nov. 9 TV channel (Chicago) : WFLD (Channel 32)

: WFLD (Channel 32) TV channel (Charlotte) : WSOC (Channel 9)

: WSOC (Channel 9) Live stream: Amazon Prime Video | DAZN (in Canada)

For the masses of NFL fans across the U.S., Thursday's clash between the Bears and Panthers will be aired on Amazon Prime Video.
</item 2>

<item 3>
Lions vs. Packers live score, updates, highlights from NFL 'Thursday Night Football' game (Sporting News, 2023-09-28T23:54:07+00:00)
NFL WEEK 4 PICKS: Straight up | Against the spread | Projection model

The Sporting News tracked live updates from Thursday's clash between the Lions and the Packers. Scroll below to check out highlights and live results as two of
the NFC's oldest sides faced off for NFC North glory.

Lions vs. Packers score

1 2 3 4 F Lions 14 13 0 7 34 Packers 3 0 8 9 20

Lions vs. Packers live updates, highlights from Thursday Night Football

(All times Eastern)

End of fourth quarter: Lions 34, Packers 20

11:19 p.m.: Lions complete yet another fourth down, officially putting the game on ice.

11:12 p.m.: Onside kick spills out of bounds, and it's looking like that might be that. Packers have one timeout remaining but Montgomery's fiery run through
the tackles should make that obsolete.

11:11 p.m.: FIELD GOAL!
</item 3>

<item 4>
Lions vs. Packers live score, updates, highlights from NFL 'Thursday Night Football' game (Sporting News, 2023-09-28T23:54:07+00:00)
Sideline reporter Kaylee Hartung will offer updates throughout the affair.

Fans in the Detroit and Green Bay markets can watch the Amazon broadcast over the air on local channels.

Those in Canada can stream it on DAZN, which carries every NFL game all season.
</item 4>

<item 5>
Bears vs. Panthers live score, updates, highlights from NFL 'Thursday Night Football' game (Sporting News, 2023-11-09T23:04:33+00:00)
The broadcast will be spearheaded by play-by-play announcer Al Michaels and color commentator Kirk Herbstreit. Sideline reporter Kaylee Hartung will offer
sideline updates throughout the game.

Fans in the Chicago and Charlotte markets can watch the Amazon broadcast over the air on local channels.

Those in Canada can stream it on DAZN, which carries every NFL game all season.
</item 5>

<item 6>
Bears vs. Panthers live score, updates, highlights from NFL 'Thursday Night Football' game (Sporting News, 2023-11-09T23:04:33+00:00)
8:09 p.m. — Perhaps the most interesting storyline tonight will be whether Al Michaels can keep a brave face throughout what might not be the most appealing
game.

8:00 p.m. — We're going to keep spirits high tonight despite a matchup of teams with a combined three wins. All football is good football! Let's see how long
this mentality lasts...

7:46 p.m. — It's 52 degrees and clear in Chicago right now, so certainly not as cold as it could be in the Windy City this time of year. Still, it's a rare
colder-weather game for Bryce Young.

7:30 p.m. — The Bears will be donning orange tonight, including an alternate orange helmet.

7:10 p.m. — D.J. Moore is back facing his former team tonight, eight months after he was traded to the Bears in the deal that sent the No. 1 pick to Carolina.

6:50 p.m.
</item 6>

<item 7>
Thursday Night Football FanDuel Picks: NFL DFS lineup advice for Week 4 Lions-Packers single-game tournaments (Sporting News, 2023-09-26T21:53:00+00:00)
Aaron Rodgers is no longer in Green Bay, and a new love interest is now under center, Jordan Love. It was these Lions that ended the Rodgers era and swept Green
Bay last season. Now they meet again in prime time on one of the league's most hallowed grounds in a battle for NFC North supremacy. The tundra won't be frozen
at Lambeau Field at this point in the season, but the game sure is important, and for a game like this, there's no way we could pass up the chance to win some
cash with our FanDuel single-game DFS lineup.

MORE THURSDAY NIGHT FOOTBALL:

DK lineup | Best props | Betting preview | Start-Sit

Before we break down our picks and strategy, here are the most notable scoring rules for FanDuel contests: four-point passing TDs and half-point PPR.
</item 7>

<item 8>
Thursday Night Football FanDuel Picks: NFL DFS lineup advice for Week 4 Lions-Packers single-game tournaments (Sporting News, 2023-09-26T21:53:00+00:00)
For Detroit, it's all about courage. For a team that was once down and out, much like their city, they are now once again on the rise. The Lions are supposed to
be the kings of the jungle and hope to become the NFL's version in 2023. In terms of the food chain, a basic concept taught in science classes across the
country, lions are the hunter rather than the hunted, but it's been the other way around for far too long, as 31 other teams have feasted on a once proud
franchise.

Their motors have been restored and the team has been rebuilt with an attitude and moxie that would make even the most "football guy" of "football guys" blush.
It's a combination of quiet confidence and a downright arrogant confidence that draws people in to root for -- or against -- an underdog.

While the Packers have packed up the Lions for years, that all began to change in 2022.
</item 8>

<item 9>
Lions vs. Packers live score, updates, highlights from NFL 'Thursday Night Football' game (Sporting News, 2023-09-28T23:54:07+00:00)
ET.

What channel is Lions vs. Packers on today?

Game: Detroit Lions at Green Bay Packers

Detroit Lions at Green Bay Packers Date: Thursday, September 28

Thursday, September 28 TV channel (Detroit) : WJBK (Channel 2)

: WJBK (Channel 2) TV channel (Green Bay) : WITI (Channel 6) or WGBA (Channel 26)

: WITI (Channel 6) or WGBA (Channel 26) Live stream: Amazon Prime Video | DAZN (in Canada)

For the masses of NFL fans across the U.S., Thursday's clash between the Lions and Packers will be aired on Amazon Prime Video. The broadcast will be
spearheaded by legendary play-by-play announcer Al Michaels. He'll be joined by longtime ESPN college football commentator Kirk Herbstreit, who will offer
occasional musings as an analyst. Sideline reporter Kaylee Hartung will offer updates throughout the affair.
</item 9>

<item 10>
Lions vs. Packers live score, updates, highlights from NFL 'Thursday Night Football' game (Sporting News, 2023-09-28T23:54:07+00:00)
: Jordan Love and the Packers heading onto the field ahed of tonight's game! We're about 45 minutes from kickoff at the Pantheon of football excellence, Lambeau
Field.

Lions vs. Packers start time

Date: Thursday, Sept. 28

Thursday, Sept. 28 Start time: 8:15 p.m. ET

Thursday night's game between the Lions and Packers will feature in the primetime slot of 8:15 p.m. ET.

What channel is Lions vs. Packers on today?
</item 10>
=== END ITEM id=7d86fa4de522320a ===

=== ITEM id=87c525c96d1ff1b7 ===
Gold sentences and facts:
Sentence 0: Not only has he produced at a high level, but his play has been instrumental in the Vikings turning things around after a 1-4 start.
  Fact 0.0: He has produced at a high level.
  Fact 0.1: His play has been instrumental in the Vikings turning things around.
  Fact 0.2: The Vikings had a 1-4 start.
Sentence 1: But they've also seen the fourth-lowest ADOT from tight ends (5.0 yards per target), almost as if teams know they won't be able to succeed on downfield shots against them (zero completions of 20-plus yards from tight ends).
  Fact 1.0: The team under discussion has seen the fourth-lowest ADOT from tight ends.
  Fact 1.1: The ADOT from tight ends is 5.0 yards per target.
  Fact 1.2: Teams seem to know they will not be able to succeed on downfield shots against the team under discussion.
  Fact 1.3: The team under discussion has had zero completions of 20-plus yards from tight ends.

Retrieved items:
<item 1>
NFL Fantasy Football Week 6 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-12T21:23:16+00:00)
In an effort not to overwork their backs, who won't have it easy against the Vikings, Kmet could handle more short-area throws than normal. His ADOT over the
past two games has been 7.64, which isn't far from his season average of 6.69.

VIKINGS: Allow the fewest yards per catch to rival tigh ends (5.9 yards), even keeping a lid on Travis Kelce last week (6.7 yards per catch). But they've also
seen the fourth-lowest ADOT from tight ends (5.0 yards per target), almost as if teams know they won't be able to succeed on downfield shots against them (zero
completions of 20-plus yards from tight ends). These numbers are tough but if Kmet gets the target volume we've seen him have over the past couple of weeks he
can still put up palatable numbers. And, tight ends have scored three times in the Vikings past three games.
</item 1>

<item 2>
NFL Fantasy Football Week 5 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-04T21:56:25+00:00)
LAST WEEK: Total disaster. In a gamescript that actually gave Minnesota a chance to -- gasp! -- run the ball, Kirk Cousins barely threw the ball and Addison
went catchless on one target. There was also a major reduction in snaps played, down to 58%. His season-low was 56% from Week 1.

MATCHUP: With the Vikings capable of running the ball, the offense went with heavier formations. Backup tight end Josh Oliver played about as many snaps as he
did in Week 1 and third tight end Johnny Mundt saw a season-high 17% of snaps. Week 4 was the first time all year the Vikings DIDN'T throw the ball on at least
69% of their snaps.

THIRD MAN: Last week did affirm that Addison may actually be the No. 3 wide receiver in the offense behind Justin Jefferson and K.J.
</item 2>

<item 3>
NFL awards odds: Six longshots bettors should consider for MVP, OPOY, DPOY and more (Sporting News, 2023-12-05T20:45:25+00:00)
But there is a solid chance he does not hit that mark. Of Miami's five remaining opponents, four have pass defenses ranked in the league's top eight. If he
slows down, someone like CeeDee Lamb could catch up.

Based on his current season average, Lamb is on track for close to 1,600 yards, but based on his production over the last seven games, he could finish with
closer to 1,800 yards.

He will probably need Hill to fade down the stretch to have a shot. But if Hill does and he continues to produce, Lamb could steal OPOY from Hill.

Defensive Player of the Year

Current favorite(s): Micah Parsons , Myles Garrett , and T.J. Watt

, Myles Garrett , and T.J. Watt Longshot: Danielle Hunter

Hunter leads the league in tackles for a loss and is third in sacks with 13.5 (behind Khalil Mack and Watt).
</item 3>

<item 4>
NFL power rankings: Lions, 49ers level up; Texans jump Bengals, Bills; Jets, Patriots sink to bottom for Week 11 (Sporting News, 2023-11-13T21:54:00+00:00)
MORE: Inside the Cowboys historic offensive explosion vs. Giants

11. Seattle Seahawks 6-3 (13)

The Seahawks needed every ounce of their offense around Geno Smith to rally to beat the Commanders, but the get-well performance worked well for him, Kenneth
Walker and the wide receivers. The defense could be more consistent.

12. Minnesota Vikings 6-4 (14)

The new man on the Vikings, Josh Dobbs, keeps producing the same results they were enjoying offensively with Kirk Cousins, thanks to more help from the running
game for Kevin O'Connell and the defense overall under Brian Flores. Their 0-3 and 1-4 starts are distant memories as the Vikings look for a sixth straight win
in Week 11 vs. Denver.

13. Houston Texans 5-4 (16)

The Texans are looking great with C.J. Stroud leading their offense and now they've found some running game, too.
</item 4>

<item 5>
NFL awards odds: Six longshots bettors should consider for MVP, OPOY, DPOY and more (Sporting News, 2023-12-05T20:45:25+00:00)
Not only has he produced at a high level, but his play has been instrumental in the Vikings turning things around after a 1-4 start.

He may not have the name recognition that Parsons, Garrett and Watt have, but he has played as well, if not better, this season. If he can continue to produce,
he at least deserves to be in the conversation, if not win.

Coach of the Year

Current favorite: Dan Campbell +200 (via DraftKings)

Longshot: Sean Payton +2200 (via DraftKings)

Campbell has done a tremendous job with the Lions, but it is fair to wonder if he's the betting favorite because of how well he has done his job this season or
over the last two. This year, the team benefited from the groundwork laid last season.

If there were a reward for the best coach over two seasons, he would deserve it. As for this season, not so much.
</item 5>

<item 6>
NFL awards odds: Six longshots bettors should consider for MVP, OPOY, DPOY and more (Sporting News, 2023-12-05T20:45:25+00:00)
1 seed in the conference.

Compared to the 49ers, Cowboys and Eagles, Goff and the Lions may have the easiest schedule to finish the season.

But bettors should also pay attention to C.J. Stroud. He leads the league in passing yards, is one of four quarterbacks eho have started the whole season with a
passer rating over 100, and his touchdown to interception ratio is the best in the NFL (4 to 1)

As for the value of his play, Stroud has taken one of the worst teams in the NFL (11 total wins the last three seasons) and turned it into a playoff contender
and potential division champion.

Offensive Player of the Year

Current favorite: Tyreek Hill

Longshot: CeeDee Lamb

There's a good reason why Tyreek Hill is such a heavy favorite to win OPOY. The Dolphins wide receiver is on track to become the first player with 2,000-plus
receiving yards in a single season.
</item 6>

<item 7>
NFL awards odds: Six longshots bettors should consider for MVP, OPOY, DPOY and more (Sporting News, 2023-12-05T20:45:25+00:00)
But rather than lead the team back to prominence, Wilson was terrible and so was Denver.

This season, Wilson has played well even when the Broncos have not. He is not producing mind-blowing stat lines, but he is keeping his team competitive, playing
efficiently and not making many mistakes.

Until his three-interception day last week, he had the best touchdown-to-interception ratio in the league. But even with those picks, his is still second-best.
</item 7>

<item 8>
NFL awards odds: Six longshots bettors should consider for MVP, OPOY, DPOY and more (Sporting News, 2023-12-05T20:45:25+00:00)
NFL award longshots can be easy to overlook since each has a favorite(s). The longshot may be playing well, but other players have commanded the spotlight,
making it easier for them to go unnoticed.

Whatever the reason, they have not gotten the credit they deserve. However, with five weeks left in the regular season, there is plenty of time for a longshot
to become a contender.

But if you wait too long before placing a bet, your payout will be a lot less than if you were to roll the dice on a longshot right now.

Longshots to consider for NFL MVP, OPOY, DPOY, Coach of the Year and Comeback POY

There is always a reason why a player or several players have relatively short odds — they've been the best at their position throughout the season.
Consequently, there is just as good a reason why longshots have odds as long as they do.
</item 8>

<item 9>
NFL awards odds: Six longshots bettors should consider for MVP, OPOY, DPOY and more (Sporting News, 2023-12-05T20:45:25+00:00)
So, keep your bets small, or use a bonus bet (if you have one available), on longshots. If they do go on to win, smile all the way to the bank while you tell
everyone, "I told you so."

RELATED: Best NFL Betting sites | How to bet on the NFL

NFL MVP

Current favorite(s): Brock Purdy , Dak Prescott , Jalen Hurts

, Dak Prescott , Jalen Hurts Longshot(s): C.J. Stroud , Jared Goff

The MVP often ends up being someone, typically the quarterback, from one of the top-seeded teams. Hence, Brock Purdy, Dak Prescott and Jalen Hurts sit atop
betting boards after Week 13.

But, by that logic, Jared Goff should be in the conversation. He has played well this season — his Lions are 9-3, poised to win the NFC South and in contention
for the No. 1 seed in the conference.
</item 9>

<item 10>
NFL awards odds: Six longshots bettors should consider for MVP, OPOY, DPOY and more (Sporting News, 2023-12-05T20:45:25+00:00)
As for this season, not so much.

Sean Payton should be at the forefront of the conversation. It's one thing to turn around a team that was dreadful last season. But he has taken one that got
off to a horrible start and turned things around in-season.

Oh — and he has resurrected Russell Wilson's career.

That alone should put him at the top of the list, but he may need to get Denver into the playoffs to convince voters.

Comeback Player of the Year

Current favorite: Demar Hamlin

Longshot: Russell Wilson

Coming back from a life-threatening injury like a cardiac arrest makes Hamlin a great candidate to win CPOY. However, it would be understandable if voters were
to look elsewhere since Hamlin has seldom been active on game day and rarely seen the field.

It will depend on how vital voters deem playing to be.

Wilson was supposed to be the next great franchise quarterback for the Broncos when they acquired him prior to last season.
</item 10>
=== END ITEM id=87c525c96d1ff1b7 ===

=== ITEM id=a41b22928b286b3c ===
Gold sentences and facts:
Sentence 0: Wired removed the article for not meeting its standards, but the degree to which it was shared on social media and boosted in write-ups at other outlets shows just how willing people are to accept foul play on Google’s part.
  Fact 0.0: Wired removed the article for not meeting its standards.
  Fact 0.1: The article was shared widely on social media.
  Fact 0.2: The article was boosted in write-ups at other outlets.
  Fact 0.3: The degree to which the article was shared and boosted shows how willing people are to accept foul play on Google's part.
Sentence 1: When Google’s search head Prabhakar Raghavan testified in court on October 28, he revealed that the tech giant had paid $26.3 billion in 2021 to multiple browsers, phones and platforms, from companies including Apple, Samsung and Mozilla, The Verge reports.
  Fact 1.0: Google's search head Prabhakar Raghavan testified in court on October 28.
  Fact 1.1: Prabhakar Raghavan revealed that Google paid $26.3 billion in 2021 to multiple browsers, phones and platforms.
  Fact 1.2: The payments went to companies including Apple, Samsung and Mozilla.
  Fact 1.3: The Verge reports this.
Sentence 2: The case, filed by Arkansas-based publisher Helena World Chronicle, argues that Google “siphons off” news publishers’ content, their readers and ad revenue through anticompetitive means.
  Fact 2.0: The case was filed by Arkansas-based publisher Helena World Chronicle.
  Fact 2.1: The case argues that Google siphons off news publishers' content, their readers and ad revenue through anticompetitive means.

Retrieved items:
<item 1>
News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line (TechCrunch, 2023-12-15T17:56:02+00:00)
A Google spokesperson offered a statement on the lawsuit, saying “This lawsuit is meritless. People have many ways to access information and news content today
– through publishers’ websites, dedicated apps, social media platforms, print papers and more. Google links people to publishers’ websites more than 24 billion
times each month – at no cost to them.”

The complaint is available below.

Helena World Chronicle, LLC v. Google LLC and Alphabet Inc by TechCrunch on Scribd

Editor’s note: This post was updated after publication with a statement provided by Google on Friday evening.
</item 1>

<item 2>
News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line (TechCrunch, 2023-12-15T17:56:02+00:00)
A new class action lawsuit filed this week in the U.S. District Court in D.C. accuses Google and parent company Alphabet of anticompetitive behavior in
violation of U.S. antitrust law, the Sherman Act, and others, on behalf of news publishers. The case, filed by Arkansas-based publisher Helena World Chronicle,
argues that Google “siphons off” news publishers’ content, their readers and ad revenue through anticompetitive means. It also specifically cites new AI
technologies like Google’s Search Generative Experience (SGE) and Bard AI chatbot as worsening the problem.

In the complaint, Helena World Chronicle, which owns and publishes two weekly newspapers in Arkansas, argues that Google is “starving the free press” by sharing
publishers’ content on Google, losing them “billions of dollars.”

In addition to new AI technologies, the suit points to Google’s older question-and-answer technologies, like the “Knowledge Graph” launched in May 2012, as part
of the problem.
</item 2>

<item 3>
Is Google Search better than the rest? And is that fair? (The Age, 2023-10-22T18:00:00+00:00)
The question remains, why do users prefer Google? And if the answer, as Walker says, is that Google is simply better, the question becomes whether that’s down
to its scale, its longevity, its ubiquity and all the user data it sucks up, which no other company could possibly match.

For the US, which has to prove that Google somehow broke the law to achieve the status quo, this legal case could be an uphill battle. But for the rest of us,
it offers a rare opportunity for some insight into what the tech giant does with its enormous market share. Is it still focused on being the best? Or, as has
been alleged, does it use its ubiquity to squeeze us for money even at the expense of product quality? How ‘organic’ are Google’s results? This month, Wired
posted an opinion article by lawyer and privacy advocate Megan Gray, which alleged that Google had accidentally revealed during the trial that it manipulates
people’s search queries for maximum ad revenue.
</item 3>

<item 4>
Is Google Search better than the rest? And is that fair? (The Age, 2023-10-22T18:00:00+00:00)
The example given was replacing a search for “children’s clothing” with “NIKOLAI-brand kidswear”. Loading Google rejected this in very strong terms, saying the
piece was misleading and inaccurate while denying ever altering search terms. Wired removed the article for not meeting its standards, but the degree to which
it was shared on social media and boosted in write-ups at other outlets shows just how willing people are to accept foul play on Google’s part. A suspicion of
privacy invasion and inappropriate data handling follows the company at all times.

Grey herself is a former vice-president at DuckDuckGo, a privacy-focused company founded explicitly to counter giants such as Google. She admits that she may
have misinterpreted the evidence, but maintains that Google manipulates Search to maximise ad revenue. So what does Google say? In a post on X, Google’s
official search liaison, Danny Sullivan, said ad systems do not affect the organic results, i.e. the list of blue links in search results that are not
sponsored.
</item 4>

<item 5>
News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line (TechCrunch, 2023-12-15T17:56:02+00:00)
SGE, it argues, offers web searchers a way to seek information in a conversational mode, but ultimately keeps users in Google’s “walled garden” as it
“plagiarizes” their content. Publishers also can’t block SGE because it uses the same web crawler as Google’s general search service, GoogleBot.

Plus, it says Google’s Bard AI was trained on a dataset that included “news, magazine and digital publications,” citing both a 2023 report from the News Media
Alliance and a Washington Post article about AI training data for reference. (The Post, which worked with researchers at the Allen Institute for AI, had found
that News and Media sites were the third largest category of AI training data.)

The case points to other concerns, too, like changing AdSense rates and evidence of improper spoliation of evidence on Google’s part, by its destruction of chat
messages — an issue raised in the recent Epic Games lawsuit against Google over app store antitrust issues, which Epic won.
</item 5>

<item 6>
News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line (TechCrunch, 2023-12-15T17:56:02+00:00)
In addition to damages, the suit is asking for an injunction that would require Google to obtain consent from publishers to use their website data to train its
general artificial intelligence products including Google’s own and those of rivals. It also asks Google to allow publishers who opt out of SGE to still show up
in Google search results, among other things.

The U.S. lawsuit follows an agreement Google reached last month with the Canadian government which would see the search giant paying Canadian media for use of
their content. Under the terms of the deal, Google will provide $73.5 million (100 million Canadian dollars) every year to news organizations in the country,
with funds distributed based on the news outlets’ headcount. Negotiations with Meta are still unresolved, though Meta began blocking news in Canada in August,
in light of the pressure to pay for the content under the new Canadian law.

The case also arrives alongside the filing of the U.S.
</item 6>

<item 7>
News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line (TechCrunch, 2023-12-15T17:56:02+00:00)
“When a user searches for information on a topic, Google displays a ‘Knowledge Panel’ to the right of the search results. This panel contains a summary of
content drawn from the Knowledge Graph database,” the complaint states. “Google compiled this massive database by extracting information from Publishers’
websites — what Google calls ‘materials shared across the web’ —and from ‘open source and licensed databases,'” it says.

By 2020, the Knowledge Graph had grown to 500 billion facts about 5 billion entities. But much of the “collective intelligence” that Google tapped into was
content “misappropriated from Publishers,” the complaint alleges.

Other Google technologies, like “Featured Snippets” where Google algorithmically extracts answers from webpages, were also cited as shifting traffic away from
publishers’ websites.

More importantly, perhaps, is the suit’s tackling of how AI will impact publishers’ businesses. The problem was recently detailed in a report on Thursday by The
Wall Street Journal, which led with a shocking statistic.
</item 7>

<item 8>
Is Google Search better than the rest? And is that fair? (The Age, 2023-10-22T18:00:00+00:00)
Some critics, including Megan Grey, contend that Google uses semantic keyword matching to make searches less precise on purpose, which widens the pool of
content served and increases auctions for its ads. The quality tests, then, would ensure that the systems were still returning results that were justifiably
related to the input.
</item 8>

<item 9>
News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line (TechCrunch, 2023-12-15T17:56:02+00:00)
The case also arrives alongside the filing of the U.S. Justice Department’s lawsuit against Google for monopolizing digital ad technologies, and references the
2020 Justice Department’s civil antitrust suit over search and search advertising (which are different markets from digital ad technologies in the more recent
suit).

“The anticompetitive effects of Google’s scheme cause profound harm to competition, to consumers, to labor, and to a democratic free press,” reads an
announcement posted to the website of the law firm handling the case, Hausfeld.

“Plaintiff Helena World Chronicle, LLC invokes the Sherman Act and Clayton Act to seek class-wide monetary and injunctive relief to restore and ensure
competition for digital news and reference publishing and set up guardrails to preserve a free marketplace of ideas in the new era of artificial intelligence,”
it states.

A Google spokesperson offered a statement on the lawsuit, saying “This lawsuit is meritless.
</item 9>

<item 10>
News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line (TechCrunch, 2023-12-15T17:56:02+00:00)
When online magazine The Atlantic modeled what would happen if Google integrated AI into search, it found that 75% of the time the AI would answer the user’s
query without requiring a click-through to its website, losing it traffic. This could have a major impact on publishers’ traffic going forward, as Google today
drives nearly 40% of their traffic, according to data from Similarweb.

Some publishers are now trying to get ahead of the problem. For example, Axel Springer just this week inked a deal with OpenAI to license its news for AI model
training. But overall, publishers believe they’ll lose somewhere between 20-40% of their website traffic when Google’s AI products fully roll out, The WSJ’s
report noted.

The lawsuit reiterates this concern, claiming that Google’s recent advances in AI-based search were implemented with “the goal of discouraging end-users from
visiting the websites of Class members who are part of the digital news and publishing line of commerce.”
</item 10>
=== END ITEM id=a41b22928b286b3c ===

=== ITEM id=02640dc51bef9f59 ===
Gold sentences and facts:
Sentence 0: Instead, Marshall will start freshman Cole Pennington, who is the son of former Marshall star QB Chad Pennington.
  Fact 0.0: Instead, Marshall will start freshman Cole Pennington.
  Fact 0.1: Cole Pennington is a freshman.
  Fact 0.2: Cole Pennington is the son of former Marshall star QB Chad Pennington.
  Fact 0.3: Chad Pennington is a former Marshall star QB.
Sentence 1: Geno Smith's night appears to be done as Drew Lock is back in for the Seahawks.
  Fact 1.0: Geno Smith's night appears to be done.
  Fact 1.1: Drew Lock is back in for the Seahawks.

Retrieved items:
<item 1>
UTSA vs. Marshall odds, props, predictions: Roadrunners lay big spread in Frisco Bowl (Sporting News, 2023-12-18T22:29:26+00:00)
Marshall prediction: Frank Harris leads Roadrunners past the Herd

Even with the transfer portal hitting this game hard, the UTSA offense will be too much for Marshall to keep up with. The Roadrunners push the pace in a way
that Marshall can't replicate, and the Thundering Herd aren't efficient enough to make the most of what few scoring opportunities they'll have.

The Thundering Herd could find themselves down several scores by the end of the first frame if Pennington turns the ball over, as he's prone to do. UTSA is a
well-oiled machine against teams with similar or worse talent, and Marshall is in that latter category.

It's a big number to lay, but we're on the favorite Tuesday night.

Pick: UTSA
</item 1>

<item 2>
UTSA vs. Marshall odds, props, predictions: Roadrunners lay big spread in Frisco Bowl (Sporting News, 2023-12-18T22:29:26+00:00)
Pennington has thrown six interceptions and zero touchdowns on only 79 passing attempts this season.

Marshall's offense will continue to run through star RB Rasheen Ali. The senior speedster carried the offense even when Fancher was under center, totaling 1,256
yards and 15 touchdowns on the season. This offense has struggled despite Ali's efforts, ranking 93rd in scoring and 89th in yards per game.

On the season, Marshall is 4-8 ATS and 6-6 on the O/U board. The Herd won only three of their eight conference games on the season.

Marshall Thundering Herd opting out of Scooter's Coffee Frisco Bowl

While UTSA has the biggest individual loss for this game, the Herd's depth has taken a huge blow thanks to transfer portal entrants.
</item 2>

<item 3>
Seahawks vs. Giants live score, updates, highlights from NFL 'Monday Night Football' game (Sporting News, 2023-10-02T22:00:04+00:00)
9:30 p.m.: At the two-minute warning, Drew Lock has gotten the Seahawks to their own 48-yard line. Lock will continue to lead the Seahawks' two-minute drill.

Geno Smith is out of the medical tent and headed to the locker room for further testing on his knee. His status for the remainder of the game is questionable.

The Seahawks will look to add some cushion to their lead as they will also receive the ball to open the second half.

9:26 p.m.: Geno Smith's displeasure with the tackle from earlier is beginning to make sense. Seattle's QB has been in the medical tent with an undetermined
injury.

Drew Lock will be taking snaps until there is more clarity on Smith's status moving forward.

9:23 p.m.: FIELD GOAL! Graham Gano connects on a 55-yarder to get the Giants on the board.
</item 3>

<item 4>
UTSA vs. Marshall odds, props, predictions: Roadrunners lay big spread in Frisco Bowl (Sporting News, 2023-12-18T22:29:26+00:00)
Chase Harrison, QB (transfer portal)

Brandon McElroy, DL (transfer portal)

Caleb Coombs, WR (transfer portal)

Zion Odoms, WR (transfer portal)

Jayven Hall, LB (transfer portal)

Cam Fancher, QB (transfer portal)

Thomas Lane, DL (transfer portal)

Chinazo Obobi, OT (transfer portal)

Maurice Jones, RB (transfer portal)

Sean Sallis, TE (transfer portal)

Myles Bell, DB (transfer portal)

Trent Holler, iOL (transfer portal)

Joshua McTier, CB (transfer portal)

Jabarrek Hopkins, DL (transfer portal)

CK Obobi, OL (transfer portal)

Sean Ellis, TE (transfer portal)

TyShawn Hurst, OT (transfer portal)

Elijah Russell, EDGE (transfer portal)

UTSA vs.
</item 4>

<item 5>
Fantasy Football QB Rankings Week 14: Who to start, best sleepers at quarterback (Sporting News, 2023-12-07T22:06:00+00:00)
WEEK 14 DFS:

DK lineup | FD lineup | Best values | Best stacks

What are the biggest QB injuries in Week 14?

Leading into Week 13, Dorian Thompson-Robinson (concussion) was the lone QB injury situation we were closely monitoring. Even if Thompson-Robinson is able to
clear concussion protocol in time for Week 14, we expect Flacco to make his second start as a Brown.

This week, we unfortunately saw Kenny Pickett (ankle) and Derek Carr (head, shoulder, back) exit their respective contests with injuries, complicating their
statuses for Week 14. We've replaced both QBs with backups Mitchell Trubisky and Jameis Winston, respectively, as early indications point toward both missing
time.

UPDATE: Trevor Lawrence suffered an ankle injury on Monday, which will likely keep him out for at least a week and open the door for C.J. Beathard (vs. Browns)
to start.
</item 5>

<item 6>
Seahawks vs. Giants live score, updates, highlights from NFL 'Monday Night Football' game (Sporting News, 2023-10-02T22:00:04+00:00)
New York has been dominated at home twice this season now, as this performance follows a 40-0 blowout in Week 1 on "Sunday Night Football." Daniel Jones' career
mark in primetime falls to 1-13 and 0-3 this season.

11:09 p.m.: TURNOVER ON DOWNS! Daniel Jones is sacked for an 11th time as Seattle takes back over at the New York 35-yard line.

Geno Smith's night appears to be done as Drew Lock is back in for the Seahawks.

11:06 p.m: Daniel Jones is back under center for New York, which will look to drive and score to at least lessen the damage on the scoreboard.

11:04 p.m.: FIELD GOAL! The turnover leads to three points for the Seahawks, who get a 34-yard field goal from Jason Myers. It's a three-touchdown lead for
Seattle now with just over five minutes to play.
</item 6>

<item 7>
Monday Night Football DraftKings Picks: NFL DFS lineup advice for Week 15 Eagles-Seahawks Showdown tournaments (Sporting News, 2023-12-18T22:47:00+00:00)
The receiver has been prone to being undisciplined on the field, however, resulting in an ejection against the 49ers last week. That is always a concern when
trusting Metcalf, but we take the good with the bad since the ceiling is so high. No matter who ends up starting at QB, the WR is in line for another dominant
showing against one of the league's most generous secondaries.

FLEX QB Geno Smith, Seahawks ($9,400)

Metcalf won't be throwing the ball to himself and he's not one to get opportunities in the running game, meaning that someone will be setting him up for
success. And since Smith appears ready to play, he finds himself with a spot in our lineup. In the event that he doesn't play, however, we will turn to Drew
Lock ($9,200) in his place. Regardless, the Seattle quarterbacks are set up for success and should be able to light up the skies on Monday night.
</item 7>

<item 8>
Fantasy Football QB Rankings Week 14: Who to start, best sleepers at quarterback (Sporting News, 2023-12-07T22:06:00+00:00)
Broncos

12 Russell Wilson, Broncos @ Chargers

13 Justin Fields, Bears vs. Lions

14 Gardner Minshew, Colts @ Bengals

15 Matthew Stafford, Rams @ Ravens

16 Trevor Lawrence, Jaguars @ Browns

17 Geno Smith, Seahawks @ 49ers

18 Derek Carr, Saints vs. Panthers

19 Joe Flacco, Browns @ Jaguars

20 Baker Mayfield, Buccaneers @ Falcons

21 Josh Dobbs, Vikings @ Raiders

22 Jake Browning, Bengals vs. Colts

23 Bryce Young, Panthers @ Saints

24 Aidan O'Connell, Raiders vs. Vikings

25 Desmond Ridder, Falcons vs. Buccaneers

26 Will Levis, Titans @ Dolphins

27 Bailey Zappe, Patriots @ Steelers

28 Mitch Trubisky, Steelers @ Patriots

29 Tommy DeVito, Giants vs. Packers

30 Zach Wilson, Jets vs. Texans
</item 8>

<item 9>
UTSA vs. Marshall odds, props, predictions: Roadrunners lay big spread in Frisco Bowl (Sporting News, 2023-12-18T22:29:26+00:00)
UTSA is 5-6-1 ATS and 4-7-1 to the OVER.

UTSA Roadrunners opting out of Scooter's Coffee Frisco Bowl

The Roadrunners will be without their best defensive player, Trey Moore, as he looks to jump to a Power 5 school. Moore led the conference in sacks, so he's a
star-level absence for the unit.

Here are USTA players who won't play in the Frisco Bowl.

Trey Moore, EDGE (transfer portal)

Syrus Dumas, DB (transfer portal)

Xavier Spencer, CB (transfer portal)

Kyle Eaves, WR (transfer portal)

Marshall Thundering Herd betting news

The toughest blow for Marshall in this game is being without second-year starting QB Cam Fancher (transfer portal). Instead, Marshall will start freshman Cole
Pennington, who is the son of former Marshall star QB Chad Pennington.
</item 9>

<item 10>
Fantasy Football QB Rankings Week 14: Who to start, best sleepers at quarterback (Sporting News, 2023-12-07T22:06:00+00:00)
Matthew Stafford (@ Ravens), Geno Smith (@ 49ers), and Josh Dobbs (@ Raiders) are all risky against some of the league's better pass defenses. Stafford faces a
sound Ravens stop unit that entered their Week 13 bye allowing the second-fewest FPPG to QBs.

There's a chance the Vikings opt to start fellow QB Nick Mullens over Dobbs after his debacle last Monday night against the Bears, but if Minnesota elects to
stick with Dobbs, his rocky Week 12 outing concerns us against a Raiders defense that's been one of the more efficient pass defenses since Week 5, sitting
eighth in dropback EPA.

On the other hand, after an elite Week 13 effort in Seattle's one-possession loss to Dallas, fantasy owners might go looking to chase points with Smith, but
replicating last week's showing against the 49ers is a completely different ask for a QB who's underwhelmed this season.
</item 10>
=== END ITEM id=02640dc51bef9f59 ===
