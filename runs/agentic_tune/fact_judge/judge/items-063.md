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

=== ITEM id=55df217ebd1c3943 ===
Gold sentences and facts:
Sentence 0: Instead, Marshall will start freshman Cole Pennington, who is the son of former Marshall star QB Chad Pennington.
  Fact 0.0: Marshall will start freshman Cole Pennington instead.
  Fact 0.1: Cole Pennington is a freshman.
  Fact 0.2: Cole Pennington is the son of former Marshall star QB Chad Pennington.
Sentence 1: Geno Smith's night appears to be done as Drew Lock is back in for the Seahawks.
  Fact 1.0: Geno Smith's night appears to be done.
  Fact 1.1: Drew Lock is back in for the Seahawks.

Retrieved items:
<item 1>
Monday Night Football DraftKings Picks: NFL DFS lineup advice for Week 15 Eagles-Seahawks Showdown tournaments (Sporting News, 2023-12-18T22:47:00+00:00)
The receiver has been prone to being undisciplined on the field, however, resulting in an ejection against the 49ers last week. That is always a concern when
trusting Metcalf, but we take the good with the bad since the ceiling is so high. No matter who ends up starting at QB, the WR is in line for another dominant
showing against one of the league's most generous secondaries.

FLEX QB Geno Smith, Seahawks ($9,400)

Metcalf won't be throwing the ball to himself and he's not one to get opportunities in the running game, meaning that someone will be setting him up for
success. And since Smith appears ready to play, he finds himself with a spot in our lineup. In the event that he doesn't play, however, we will turn to Drew
Lock ($9,200) in his place. Regardless, the Seattle quarterbacks are set up for success and should be able to light up the skies on Monday night.

Graph facts:
- Brock Purdy -MENTIONS-> NFL
- Raiders vs. Lions live score, updates, highlights from NFL 'Monday Night Football' game -MENTIONS-> NFL
- Fans spot Travis Kelce wearing Taylor Swift-themed friendship bracelet before she attended his game -MENTIONS-> NFL
- Chiefs vs. Packers live score, updates, highlights from NFL 'Sunday Night Football' game -MENTIONS-> NFL
- Christmas Day preview: 49ers, Ravens square off in potential Super Bowl sneak peek -MENTIONS-> NFL

Entities: NFL, Eagles, Sporting News, Geno Smith, Drew Lock, Seahawks, DraftKings, Metcalf, ejection against the 49ers last week, Monday Night Football
DraftKings Picks: NFL DFS lineup advice for Week 15 Eagles-Seahawks Showdown tournaments, 49ers, NFL DFS
</item 1>

<item 2>
Graph query row:
article: UTSA vs. Marshall odds, props, predictions: Roadrunners lay big spread in Frisco Bowl
player1: Cam Fancher
player2: Cole Pennington
passage: UTSA vs. Marshall odds, props, predictions: Roadrunners lay big spread in Frisco Bowl (Sporting News, 2023-12-18T22:29:26+00:00)
The UTSA Roadrunners (8-4) take on the Marshall Thundering Herd (6-6) on Tuesday night in the Scooter's Coffee Frisco Bowl (9 p.m ET, ESPN). The matchup sees
the Roadrunners as large betting favorites on odds boards. Currently, USTA is favored by on the point spread, with Marshall available for on the moneyline.

The Roadrunners are in their 13th season as a program but have yet to win a bowl game. This is the fourth bowl game they've played in five years. Marshall is
competing in a bowl game for the seventh straight year.

These teams have been on different streaks as the year has progressed. UTSA has won seven of its last eight games, while Marshall has lost six of its last
eight.

Here are the live odds, team betting news, player props and our betting prediction for UTSA vs. Marshall in Tuesday's Scooter's Coffee Frisco Bowl.
</item 2>

<item 3>
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
</item 3>

<item 4>
NFL Fantasy Football Week 5 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-04T21:56:25+00:00)
On the season he's totaled 28 rushes for 134 yards and a score.

Sneaky Sleeper (Lineup Decisions) Projections powered by Sportsline Sam Howell QB WAS Washington • #14

Age: 23 • Experience: 2 yrs. Weekly Breakdown MATCHUP vs CHI WAS -6 O/U 44.5 OPP VS QB 31st PROJ PTS 21.3 QB RNK 14th YTD Stats PAYDS 961 RUYDS 82 TD 5 INT 5
FPTS/G 16.1 The Commanders like limiting Howell's pass attempts in general -- he's only thrown over 31 times in high-scoring shootouts like in Weeks 2 and 4.
While there's potential for that on Thursday, it's not a certainty. Howell could just as easily put up nominal stats to help the Commanders bring home a win.

Graph facts:
- NFL Fantasy Football Week 6 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game -MENTIONS-> NFL Fantasy Football
- NFL Fantasy Football Week 5 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game -MENTIONS-> Jimmy Garoppolo
- NFL Fantasy Football Week 5 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game -MENTIONS-> Brock Purdy
- NFL Fantasy Football Week 5 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game -MENTIONS-> Lamar Jackson
- NFL Fantasy Football Week 5 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game -MENTIONS-> Jakobi Meyers

Entities: NFL Fantasy Football, NFL Fantasy Football Week 5 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game, Sam Howell, CBSSports.com,
Commanders, SportsLine
</item 4>

<item 5>
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

Graph facts:
- UTSA vs. Marshall odds, props, predictions: Roadrunners lay big spread in Frisco Bowl -MENTIONS-> West Virginia
- UTSA vs. Marshall odds, props, predictions: Roadrunners lay big spread in Frisco Bowl -MENTIONS-> Duke's Mayo Bowl
- UTSA -PARTICIPATED_IN-> Scooter's Coffee Frisco Bowl
- Marshall -PARTICIPATED_IN-> Scooter's Coffee Frisco Bowl
- UTSA vs. Marshall odds, props, predictions: Roadrunners lay big spread in Frisco Bowl -MENTIONS-> UTSA vs. Tulane game

Entities: UTSA Roadrunners, Marshall Thundering Herd, Scooter's Coffee Frisco Bowl, Scooter's Coffee, Trey Moore, Syrus Dumas, Xavier Spencer, Kyle Eaves, Cam
Fancher, Cole Pennington, Chad Pennington, UTSA vs. Marshall odds, props, predictions: Roadrunners lay big spread in Frisco Bowl, Sporting News
</item 5>

<item 6>
Fantasy Football Week 13 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-11-30T22:10:58+00:00)
Stroud's Houdini act has helped him beat most of the defenses he's faced, and this one isn't worthy of being feared, but there is some worry Stroud have a
monster game without some massive yards after the catch (YAC) from his receivers. Fortunately he ranks ninth-best on the season among qualifying QBs in YAC per
completion (5.37).

STARTS: C.J. Stroud, Tank Dell, Nico Collins, Courtland Sutton, Devin Singletary, Javonte Williams, Russell Wilson

C.J. Stroud, Tank Dell, Nico Collins, Courtland Sutton, Devin Singletary, Javonte Williams, Russell Wilson SITS: Jerry Jeudy, both DSTs; Dalton Schultz is OUT

Start Him (Lineup Decisions) Projections powered by Sportsline Russell Wilson QB DEN Denver • #3

Age: 35 • Experience: 12 yrs.

Graph facts:
- Vikings vs. Broncos live score, updates, highlights from NFL 'Sunday Night Football' game -MENTIONS-> Russell Wilson
- NFL Fantasy Football Week 5 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game -MENTIONS-> Russell Wilson
- NFL power rankings: Lions, 49ers level up; Texans jump Bengals, Bills; Jets, Patriots sink to bottom for Week 11 -MENTIONS-> Russell Wilson
- NFL roundup: Pats hit new low as Lions reach 9-3 for first time since 1962 -MENTIONS-> Russell Wilson
- NFL Best Bets Week 9: Colts clobber Panthers, Dolphins drop Chiefs in Germany -MENTIONS-> Russell Wilson

Entities: Russell Wilson, YAC, Fantasy Football Week 13 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game, Tank Dell, CBSSports.com, C.J.
Stroud, Nico Collins, fantasy football, Courtland Sutton, Jerry Jeudy, Javonte Williams, Denver, Devin Singletary, Dalton Schultz, SportsLine
</item 6>

<item 7>
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

Graph facts:
- Raiders vs. Lions live score, updates, highlights from NFL 'Monday Night Football' game -MENTIONS-> Aidan O'Connell
- NFL DFS picks Week 15: Best sleepers, value players for FanDuel, DraftKings, SuperDraft lineups -MENTIONS-> Tommy DeVito
- 2023 NFL Week 11 injury roundup: Rams WR Cooper Kupp, Seahawks RB Kenneth Walker III among stars to exit -MENTIONS-> Tommy DeVito
- Christmas Day preview: 49ers, Ravens square off in potential Super Bowl sneak peek -MENTIONS-> Tommy DeVito
- NFL Fantasy Football Week 6 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game -MENTIONS-> Aidan O'Connell

Entities: Tommy DeVito, Aidan O'Connell, Raiders, Jaguars, Trevor Lawrence, Vikings, Russell Wilson, Saints, Titans, Bengals, Packers, Texans, Josh Dobbs,
Dolphins, Derek Carr
</item 7>

<item 8>
Fantasy Football QB Rankings Week 14: Who to start, best sleepers at quarterback (Sporting News, 2023-12-07T22:06:00+00:00)
Matthew Stafford (@ Ravens), Geno Smith (@ 49ers), and Josh Dobbs (@ Raiders) are all risky against some of the league's better pass defenses. Stafford faces a
sound Ravens stop unit that entered their Week 13 bye allowing the second-fewest FPPG to QBs.

There's a chance the Vikings opt to start fellow QB Nick Mullens over Dobbs after his debacle last Monday night against the Bears, but if Minnesota elects to
stick with Dobbs, his rocky Week 12 outing concerns us against a Raiders defense that's been one of the more efficient pass defenses since Week 5, sitting
eighth in dropback EPA.

On the other hand, after an elite Week 13 effort in Seattle's one-possession loss to Dallas, fantasy owners might go looking to chase points with Smith, but
replicating last week's showing against the 49ers is a completely different ask for a QB who's underwhelmed this season.

Graph facts:
- Raiders vs. Lions live score, updates, highlights from NFL 'Monday Night Football' game -MENTIONS-> Raiders
- Kelce -MENTIONS-> Raiders
- NFL Fantasy Football Week 5 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game -MENTIONS-> Raiders
- 2023 NFL Week 11 injury roundup: Rams WR Cooper Kupp, Seahawks RB Kenneth Walker III among stars to exit -MENTIONS-> Raiders
- Christmas Day preview: 49ers, Ravens square off in potential Super Bowl sneak peek -MENTIONS-> Raiders

Entities: Raiders, Vikings, Josh Dobbs, Dallas, fantasy football, Ravens, Sporting News, Geno Smith, Bears, 49ers, Matthew Stafford, Nick Mullens, Seattle,
Seattle's one-possession loss to Dallas, Pass defenses
</item 8>

<item 9>
UTSA vs. Marshall odds, props, predictions: Roadrunners lay big spread in Frisco Bowl (Sporting News, 2023-12-18T22:29:26+00:00)
Pennington has thrown six interceptions and zero touchdowns on only 79 passing attempts this season.

Marshall's offense will continue to run through star RB Rasheen Ali. The senior speedster carried the offense even when Fancher was under center, totaling 1,256
yards and 15 touchdowns on the season. This offense has struggled despite Ali's efforts, ranking 93rd in scoring and 89th in yards per game.

On the season, Marshall is 4-8 ATS and 6-6 on the O/U board. The Herd won only three of their eight conference games on the season.

Marshall Thundering Herd opting out of Scooter's Coffee Frisco Bowl

While UTSA has the biggest individual loss for this game, the Herd's depth has taken a huge blow thanks to transfer portal entrants.

Graph facts:
- North Texas vs. SMU odds, props, predictions: Red hot Mustangs could overpower not-so-Mean Green -MENTIONS-> UTSA
- UTSA Roadrunners -COMPETES_WITH-> Marshall Thundering Herd
- Marshall -PARTICIPATED_IN-> Scooter's Coffee Frisco Bowl
- UTSA Roadrunners -PARTICIPATED_IN-> Scooter's Coffee Frisco Bowl
- College Bowl Odds, Spreads, Money Lines and Totals - NCAAF 2023 Bowl Season -MENTIONS-> UTSA

Entities: Marshall Thundering Herd, Scooter's Coffee Frisco Bowl, UTSA, Pennington, Rasheen Ali, Fancher, UTSA vs. Marshall odds, props, predictions:
Roadrunners lay big spread in Frisco Bowl, Sporting News
</item 9>

<item 10>
2023 NFL Week 11 injury roundup: Rams WR Cooper Kupp, Seahawks RB Kenneth Walker III among stars to exit (CBSSports.com, 2023-11-19T23:51:49+00:00)
Smith did return, however, for the final drive of the game in the losing effort. Head coach Pete Carroll said that Smith will "make it back." When asked if that
means for Thursday's matchup against San Francisco, the coach said, "I would think so."

Sam Howell (Commanders) received medical attention on the sidelines after enduring several hard hits to the upper body on a touchdown run against the Giants. He
was quickly cleared of a potential concussion, however, and remained in the game. Veteran journeyman Jacoby Brissett is Washington's backup.

RB

Aaron Jones (Packers) suited up against the Chargers despite a lingering hamstring issue, only to manage just five carries before suffering a knee injury. The
former Pro Bowler needed help to leave the field and was then carted to the locker room, replaced by A.J. Dillon.
</item 10>
=== END ITEM id=55df217ebd1c3943 ===

=== ITEM id=4c58454aefcec71b ===
Gold sentences and facts:
Sentence 0: In the third quarter, Uber generated $394 million in operating income and $219 million worth of net income.
  Fact 0.0: In the third quarter, Uber generated $394 million in operating income.
  Fact 0.1: In the third quarter, Uber generated $219 million worth of net income.
Sentence 1: The ride-hail company claims that the rate of sexual assault reported on the app decreased 38% between its first and second reports.
  Fact 1.0: The ride-hail company claims that the rate of sexual assault reported on its app decreased 38%.
  Fact 1.1: The 38% decrease is measured between the company's first and second reports.

Retrieved items:
<item 1>
Uber’s Q3 numbers include impressive profitability gains, slower-than-expected growth (TechCrunch, 2023-11-07T17:03:41+00:00)
The combined impact of those two items was 8 percentage points of growth.

The profit perspective

From bookings to revenues to profits: How did the main portions of Uber’s business generate black ink?

Turning to what Uber calls “segment adjusted EBITDA,” it’s not hard to see how the company managed to rack up more income in its most recent quarter.
Ride-hailing adjusted profit rose to $1.29 billion, up 43% from $898 million in the year-ago quarter, while delivery saw its own profitability skyrocket from
$181 million in Q3 2022 to $413 million in its most recent fiscal period.

Uber did spend more during Q3 2023 when compared to its Q3 2022 result, but the 5% gain in “Corporate G&A and Platform R&D” costs to $595 million was far less
than the gains we saw above. So, Uber’s adjusted EBITDA rose from $516 million in aggregate to $1.09 billion in the third quarter of this year.

Graph facts:
- SBF’s trial starts soon, but how did he — and FTX — get here? -PUBLISHED_BY-> TechCrunch
- Sam Altman backs teens’ startup, Google unveils the Pixel 8 and TikTok tests an ad-free tier -PUBLISHED_BY-> TechCrunch
- More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo -PUBLISHED_BY-> TechCrunch
- After warning, the EU opens investigation into X over alleged disinformation -PUBLISHED_BY-> TechCrunch
- SBF Trial: The latest updates from the FTX collapse’s courtroom drama -PUBLISHED_BY-> TechCrunch

Entities: TechCrunch, Uber’s Q3 numbers include impressive profitability gains, slower-than-expected growth, ride-hailing, delivery, Corporate G&A and Platform
R&D, segment adjusted EBITDA, Uber
</item 1>

<item 2>
Uber’s Q3 numbers include impressive profitability gains, slower-than-expected growth (TechCrunch, 2023-11-07T17:03:41+00:00)
After reporting, shares of Uber are up 1.6% in trading, after an up-and-down morning of shares exchange.

Against a backdrop of macroeconomic uncertainty, the former startup darling’s results can be viewed as an indication that its business model has matured and is
now in a stable, profitable and cash-generating position. On the other hand, under-performing demand results from a company as global as Uber could indicate
that consumer spend is coming in lighter than anticipated.

For startups in the transport and on-demand sectors, Uber’s earnings are a regular informational zeitgeist. So let’s work to understand where Uber’s revenue
came from in the third quarter, and how each of its top-line sources converted — or not — to bottom-line results.

Where did the money come from?

In the third quarter, Uber saw total bookings rise from $29.1 billion to $35.3 billion, a gain of around 21%.

Graph facts:
- SBF’s trial starts soon, but how did he — and FTX — get here? -PUBLISHED_BY-> TechCrunch
- Sam Altman backs teens’ startup, Google unveils the Pixel 8 and TikTok tests an ad-free tier -PUBLISHED_BY-> TechCrunch
- More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo -PUBLISHED_BY-> TechCrunch
- After warning, the EU opens investigation into X over alleged disinformation -PUBLISHED_BY-> TechCrunch
- SBF Trial: The latest updates from the FTX collapse’s courtroom drama -PUBLISHED_BY-> TechCrunch

Entities: TechCrunch, transport and on-demand sectors, macroeconomic uncertainty, consumer spend, Uber’s Q3 numbers include impressive profitability gains,
slower-than-expected growth, Uber
</item 2>

<item 3>
Uber’s Q3 numbers include impressive profitability gains, slower-than-expected growth (TechCrunch, 2023-11-07T17:03:41+00:00)
In business segment terms, Uber generated $17.9 billion in ride-hailing bookings (+31% year over year), and $16.1 billion worth of delivery bookings (+18%).
Those key business groups at the company generated $5.1 billion and $2.9 billion in revenue, respectively, during the September quarter.

There’s nuance to the revenue figures that we need to consider as the rule-changes in question do affect startups that operate in related categories. On the
ride-hailing front, Uber told its investors that its Q3 2023 revenue was “negatively impacted by business model changes in some countries that classified
certain sales and marketing costs as contra revenue by $161 million.” That was not the only legal change that led to Uber’s results changing shape. Under its
delivery business results, the company added that its revenue result was “negatively impacted by business model changes that classified certain sales and
marketing costs as contra revenue by $360 million.”

The combined impact of those two items was 8 percentage points of growth.

Graph facts:
- SBF’s trial starts soon, but how did he — and FTX — get here? -PUBLISHED_BY-> TechCrunch
- Sam Altman backs teens’ startup, Google unveils the Pixel 8 and TikTok tests an ad-free tier -PUBLISHED_BY-> TechCrunch
- More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo -PUBLISHED_BY-> TechCrunch
- After warning, the EU opens investigation into X over alleged disinformation -PUBLISHED_BY-> TechCrunch
- SBF Trial: The latest updates from the FTX collapse’s courtroom drama -PUBLISHED_BY-> TechCrunch

Entities: TechCrunch, Uber’s Q3 numbers include impressive profitability gains, slower-than-expected growth, ride-hailing, delivery, business model changes,
contra revenue, Uber, startups
</item 3>

<item 4>
Uber’s Q3 numbers include impressive profitability gains, slower-than-expected growth (TechCrunch, 2023-11-07T17:03:41+00:00)
Of course, adjusted EBITDA is to profit as hidden heels in men’s shoes are to height, so we’ll want some harder figures as well. In the third quarter, Uber
generated $394 million in operating income and $219 million worth of net income. Mix in $966 million worth of positive operating cash flow, and Uber looks very
healthy, given that all of the figures in this paragraph were improvements on its year-ago results.

But while food delivery and scooting humans about town were profitable for Uber in the quarter, the final major portion of its business had a more lackluster
quarter.

Freight falls

One area that continues to drag on Uber is its freight business.

While Uber’s ride-hailing and delivery business saw an uptick in gross bookings in the third quarter, Uber Freight experienced a 27% drop year-over-year.
Revenue, as a result, had a similar fall.

The business unit reported revenue of $1.3 billion in the third quarter, a 27% drop from the same period last year.

Graph facts:
- SBF’s trial starts soon, but how did he — and FTX — get here? -PUBLISHED_BY-> TechCrunch
- Sam Altman backs teens’ startup, Google unveils the Pixel 8 and TikTok tests an ad-free tier -PUBLISHED_BY-> TechCrunch
- More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo -PUBLISHED_BY-> TechCrunch
- After warning, the EU opens investigation into X over alleged disinformation -PUBLISHED_BY-> TechCrunch
- SBF Trial: The latest updates from the FTX collapse’s courtroom drama -PUBLISHED_BY-> TechCrunch

Entities: TechCrunch, Uber’s Q3 numbers include impressive profitability gains, slower-than-expected growth, Uber Freight, Uber
</item 4>

<item 5>
Uber sexual assault survivors call for in-car cameras, tech upgrades (TechCrunch, 2023-10-13T19:26:48+00:00)
Uber hasn’t released safety data for 2021 and 2022, and won’t say when it plans to do so. Older data backs up some of Abrams’ claims.

Uber has published two safety reports — one containing data from 2017 to 2018 and another with data from 2019 to 2020. The ride-hail company claims that the
rate of sexual assault reported on the app decreased 38% between its first and second reports.

While the rate decrease is positive, not all the data delivers the same message. Over that same time period, the total number of sexual assault reports across
five categories fell from 5,981 to 3,824. That drop could be explained by the decrease of trips from 1.4 billion in 2019 to 650 million in 2020.

And while the total number of incidents as a percentage of total number of rides does decrease through the years, the number of incidents in certain categories
actually increases.

Graph facts:
- SBF’s trial starts soon, but how did he — and FTX — get here? -PUBLISHED_BY-> TechCrunch
- Sam Altman backs teens’ startup, Google unveils the Pixel 8 and TikTok tests an ad-free tier -PUBLISHED_BY-> TechCrunch
- More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo -PUBLISHED_BY-> TechCrunch
- After warning, the EU opens investigation into X over alleged disinformation -PUBLISHED_BY-> TechCrunch
- SBF Trial: The latest updates from the FTX collapse’s courtroom drama -PUBLISHED_BY-> TechCrunch

Entities: TechCrunch, Uber, Abrams, Uber sexual assault survivors call for in-car cameras, tech upgrades, Uber safety reports, sexual assault
</item 5>

<item 6>
Uber’s Q3 numbers include impressive profitability gains, slower-than-expected growth (TechCrunch, 2023-11-07T17:03:41+00:00)
The question is whether Uber believes in the long-term income potential of freight.

So what?

When discussing Uber’s results internally, our vibe was that it is a very profitable and healthy company today, albeit one that is not growing as fast as the
market had hoped. The fact that Uber’s share price is up as we write this indicates that investors are content thus far to excuse the small Q3 misses and keep
their eyes more focused on its year-over-year improvements and forward guidance. More when we get Lyft’s numbers.

Graph facts:
- SBF’s trial starts soon, but how did he — and FTX — get here? -PUBLISHED_BY-> TechCrunch
- Sam Altman backs teens’ startup, Google unveils the Pixel 8 and TikTok tests an ad-free tier -PUBLISHED_BY-> TechCrunch
- More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo -PUBLISHED_BY-> TechCrunch
- After warning, the EU opens investigation into X over alleged disinformation -PUBLISHED_BY-> TechCrunch
- SBF Trial: The latest updates from the FTX collapse’s courtroom drama -PUBLISHED_BY-> TechCrunch

Entities: TechCrunch, Uber’s Q3 numbers include impressive profitability gains, slower-than-expected growth, freight, Uber, Lyft
</item 6>

<item 7>
Uber’s Q3 numbers include impressive profitability gains, slower-than-expected growth (TechCrunch, 2023-11-07T17:03:41+00:00)
Uber reported third-quarter earnings on Tuesday that show a profitable ride-hail and delivery company that’s chugging along in spite of slowing growth in some
sectors.

The company reported revenue of $9.3 billion, an 11% increase year over year. Investors had expected Uber to report revenues of around $9.5 billion (FactSet,
Refinitiv), meaning that despite the company’s growth, it fell short of estimates. Turning to profitability, Uber reported net income of $221 million in the
third quarter, or 10 cents per share, compared with a net loss of $1.2 billion, or 61 cents per share, in the same quarter last year. Again, the company fell
short of expectations that it would generate 12 cents in per-share profit.

Looking ahead, Uber anticipates gross bookings of $36.5 billion to $37.5 billion, up just 6% at the top-end from its Q3 result.

Graph facts:
- SBF’s trial starts soon, but how did he — and FTX — get here? -PUBLISHED_BY-> TechCrunch
- Sam Altman backs teens’ startup, Google unveils the Pixel 8 and TikTok tests an ad-free tier -PUBLISHED_BY-> TechCrunch
- More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo -PUBLISHED_BY-> TechCrunch
- After warning, the EU opens investigation into X over alleged disinformation -PUBLISHED_BY-> TechCrunch
- SBF Trial: The latest updates from the FTX collapse’s courtroom drama -PUBLISHED_BY-> TechCrunch

Entities: TechCrunch, FactSet, Refinitiv, Uber’s Q3 numbers include impressive profitability gains, slower-than-expected growth, Uber
</item 7>

<item 8>
Graph query row:
article: Uber’s Q3 numbers include impressive profitability gains, slower-than-expected growth
publisher: TechCrunch
organization: Uber Freight
chunk: 4570
passage: Uber’s Q3 numbers include impressive profitability gains, slower-than-expected growth (TechCrunch, 2023-11-07T17:03:41+00:00)
Uber reported third-quarter earnings on Tuesday that show a profitable ride-hail and delivery company that’s chugging along in spite of slowing growth in some
sectors.

The company reported revenue of $9.3 billion, an 11% increase year over year. Investors had expected Uber to report revenues of around $9.5 billion (FactSet,
Refinitiv), meaning that despite the company’s growth, it fell short of estimates. Turning to profitability, Uber reported net income of $221 million in the
third quarter, or 10 cents per share, compared with a net loss of $1.2 billion, or 61 cents per share, in the same quarter last year. Again, the company fell
short of expectations that it would generate 12 cents in per-share profit.

Looking ahead, Uber anticipates gross bookings of $36.5 billion to $37.5 billion, up just 6% at the top-end from its Q3 result.
</item 8>

<item 9>
Uber’s Q3 numbers include impressive profitability gains, slower-than-expected growth (TechCrunch, 2023-11-07T17:03:41+00:00)
On a quarter-over-quarter basis, Uber Freight had a 1% gain in revenue.

The results don’t get any better once we turn to net income. On an adjusted basis, Uber Freight lost $13 million in the third quarter compared to a $1 million
profit in the same quarter last year.

Uber said the dismal year-over-year revenue results for its freight business was driven by lower revenue per load and volume. Both are consequences of the
challenging freight market cycle.

Uber Freight isn’t alone. Other more recent entrants to the freight and logistics industry such as Flexport and Convoy have struggled this year. In Convoy’s
case, the business was forced to shut down with its assets gobbled up by Flexport.

Uber Freight continues to plug along, despite these economic headwinds. The question is whether Uber believes in the long-term income potential of freight.

So what?

Graph facts:
- SBF’s trial starts soon, but how did he — and FTX — get here? -PUBLISHED_BY-> TechCrunch
- Sam Altman backs teens’ startup, Google unveils the Pixel 8 and TikTok tests an ad-free tier -PUBLISHED_BY-> TechCrunch
- More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo -PUBLISHED_BY-> TechCrunch
- After warning, the EU opens investigation into X over alleged disinformation -PUBLISHED_BY-> TechCrunch
- SBF Trial: The latest updates from the FTX collapse’s courtroom drama -PUBLISHED_BY-> TechCrunch

Entities: TechCrunch, Uber’s Q3 numbers include impressive profitability gains, slower-than-expected growth, Uber Freight, challenging freight market cycle,
lower revenue per load and volume, Uber, Convoy, Flexport
</item 9>

<item 10>
Tesla ‘digs its own grave with the Cybertruck,’ Convoy collapses and Rivian scores a win at Rebelle (TechCrunch, 2023-10-22T19:15:29+00:00)
Tesla’s shares, which are priced more as a tech stock than as an automaker’s stock, were down after Q3 earnings. Investors were clearly feeling skittish after
Tesla reported a gross margin of 17.9%, down from 25.1% in the same period last year. That’s also down from Q2’s margins of 18.2%. As a result, profits fell 44%
to $1.85 billion in Q3 from the same year-ago period.

Investors see the falling margins and Tesla’s price cuts as proof that demand is lessening for the vehicles as other EVs take market share and rising interest
rates make it difficult for many buyers to afford big ticket purchases. The company also reported that solar deployments slipped 48% in Q3 from the same period
last year. But the company made up for it by pulling in a 90% spike in energy storage deployments.

Tesla also gave some updates about its long-delayed Cybertruck.

Graph facts:
- SBF’s trial starts soon, but how did he — and FTX — get here? -PUBLISHED_BY-> TechCrunch
- Sam Altman backs teens’ startup, Google unveils the Pixel 8 and TikTok tests an ad-free tier -PUBLISHED_BY-> TechCrunch
- More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo -PUBLISHED_BY-> TechCrunch
- After warning, the EU opens investigation into X over alleged disinformation -PUBLISHED_BY-> TechCrunch
- SBF Trial: The latest updates from the FTX collapse’s courtroom drama -PUBLISHED_BY-> TechCrunch

Entities: TechCrunch, Rivian, Rebelle, Convoy, Tesla ‘digs its own grave with the Cybertruck,’ Convoy collapses and Rivian scores a win at Rebelle, Cybertruck,
Tesla
</item 10>
=== END ITEM id=4c58454aefcec71b ===

=== ITEM id=9d88fe68b3a216e9 ===
Gold sentences and facts:
Sentence 0: 10:59 p.m. — The Cowboys are moving the ball now, when it's a bit too late.
  Fact 0.0: At 10:59 p.m., the Cowboys are moving the ball now.
  Fact 0.1: The Cowboys are moving the ball when it's a bit too late.
Sentence 1: Mixon has been forced to step up without Burrow under center, doubling as a runner and receiver for Cincinnati.
  Fact 1.0: Mixon has been forced to step up without Burrow under center.
  Fact 1.1: Mixon doubles as a runner for Cincinnati.
  Fact 1.2: Mixon doubles as a receiver for Cincinnati.

Retrieved items:
<item 1>
NFL Fantasy Football Week 6 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-12T21:23:16+00:00)
D.J. Turner is expected to replace Awuzie.

GENO SMITH: Has a 47.8% completion rate and a lowly 4.1 yards per attempt average against man coverage so far this season. He's also barely thrown deep (4.3% of
his attempts went 20-plus Air Yards). A huge reason for the low numbers is because his pass protection has been dreadful. The anticipated return of left tackle
Charles Cross should help him have more time, which in turn helps his receivers.

OUTSIDE WRs: Have caught just 50% of their targets against the Bengals this season, third-lowest in football. However, when those receivers have caught the ball
they've averaged 17.9 yards per catch, which is the third-most in the league. The Bengals defense has really only let up big games to receivers who catch
touchdowns.

Graph facts:
- Brock Purdy -MENTIONS-> NFL
- Raiders vs. Lions live score, updates, highlights from NFL 'Monday Night Football' game -MENTIONS-> NFL
- Fans spot Travis Kelce wearing Taylor Swift-themed friendship bracelet before she attended his game -MENTIONS-> NFL
- Chiefs vs. Packers live score, updates, highlights from NFL 'Sunday Night Football' game -MENTIONS-> NFL
- Christmas Day preview: 49ers, Ravens square off in potential Super Bowl sneak peek -MENTIONS-> NFL

Entities: NFL, Bengals, CBSSports.com, D.J. Turner, Awuzie, NFL Fantasy Football Week 6 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game,
fantasy football, Geno Smith, Charles Cross
</item 1>

<item 2>
Bills vs. Bengals live score, updates, highlights from NFL 'Sunday Night Football' game (Sporting News, 2023-11-05T23:47:00+00:00)
: Mixon gets five yards on a run outside to the right, he's pushed out of bounds, but that will do it. The Bengals can kneel the game out and will win 24-18.

11:19 p.m.: Mixon rushes for four yards to bring up a third-and-4 at the Bills' 37, and bring up the two-minute warning. A first down will win the game for the
Bengals, since Buffalo cannot stop the clock. A stop on third will bring up an interesting decision for Zac Taylor and Cincinnati.

11:17 p.m.: Mixon rushes for two yards after the big gain, and the Bills use their final timeout with 2:37 left.

11:17 p.m.: Feels like a time to run the ball, right? Well, not to the Bengals, who trust Burrow to air out a deep pass to Boyd for 32 yards to the Bills 43.

Graph facts:
- Brock Purdy -MENTIONS-> NFL
- Raiders vs. Lions live score, updates, highlights from NFL 'Monday Night Football' game -MENTIONS-> NFL
- Fans spot Travis Kelce wearing Taylor Swift-themed friendship bracelet before she attended his game -MENTIONS-> NFL
- Chiefs vs. Packers live score, updates, highlights from NFL 'Sunday Night Football' game -MENTIONS-> NFL
- Christmas Day preview: 49ers, Ravens square off in potential Super Bowl sneak peek -MENTIONS-> NFL

Entities: NFL, Bills, Bengals, Sporting News, Burrow, Mixon, Zac Taylor, Boyd, Bills vs. Bengals live score, updates, highlights from NFL 'Sunday Night
Football' game, NFL 'Sunday Night Football' game
</item 2>

<item 3>
NFL Fantasy Football Week 5 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-04T21:56:25+00:00)
But just three players have averaged over 4.0 yards per carry: Joe Mixon (4.5 yards per run, didn't have 100 total yards, didn't score), Zack Moss (4.1 yards
per run, ran for 122 yards on 30 carries and caught a touchdown) and Pierre Strong (9.8 yards per carry, most of it coming in garbage time last week).

HISTORY: Harris had over 120 total yards and a touchdown in a Week 17 win in Baltimore last year. He also scored on them with exactly 50 total yards against
them in at home in Week 14. He has posted at least 10 non-PPR points in 3 of 4 career games.

Graph facts:
- Biden’s AI EO hailed as broad, but not deep without legislation to match -MENTIONS-> Harris
- Bryce Harper 'Atta Boy' controversy, explained: Why Orlando Arcia's comments led to backlash, Alanna Rizzo rant -MENTIONS-> Harris
- UTSA vs. Marshall odds, props, predictions: Roadrunners lay big spread in Frisco Bowl -MENTIONS-> Harris
- NFL Fantasy Football Week 6 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game -MENTIONS-> NFL Fantasy Football
- Harris -PARTICIPATED_IN-> Harris’s remarks

Entities: NFL Fantasy Football, Pierre Strong, Harris, Week 17 win, Week 14 game, NFL Fantasy Football Week 5 Lineup Decisions: Starts, Sits, Sleepers, Busts to
know for every game, CBSSports.com, Zack Moss, Joe Mixon, Baltimore
</item 3>

<item 4>
Graph query row:
article: Cowboys vs. 49ers live score, updates, highlights from NFL 'Sunday Night Football' game
publisher: The Sporting News
passage: Cowboys vs. 49ers live score, updates, highlights from NFL 'Sunday Night Football' game (Sporting News, 2023-10-08T22:47:03+00:00)
Jerry Jones said earlier in the week that Sunday night's game would show him how the Cowboys stack up against the NFL's best.

The answer? Not very well.

Everything that could go wrong for Dallas on Sunday did go wrong, in a game that confirmed a 49ers team that ended the Cowboys' season in each of the last two
years remains a level above its NFC counterpart.

The 49ers carved up Dallas' defense on the opening drive, and that was a sign of what was to come. Brock Purdy threw four touchdowns in the 42-10 rout,
including three to George Kittle. Christian McCaffrey scored a touchdown for a 14th consecutive game as all of San Francisco's offensive weapons got involved.

MORE: Comparing stats for NFC stars Nick Bosa, Micah Parsons

Defensively, the 49ers were relentless. Nobody had a larger impact than Fred Warner, who finished the night with an interception, sack and forced fumble in a
swarming performance.
relationship: MENTIONS
connected_entity: Brock Purdy
</item 4>

<item 5>
NFL Fantasy Football Week 5 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-04T21:56:25+00:00)
Or Denver. Or Chicago. You get the gist. The Cardinals pass rush shouldn't be too dangerous and Burrow, now over a month into his season, figures to eventually
show some signs of improvement with his health. In no way is he a must start, and there's plenty of good reason to go with Trevor Lawrence, Russell Wilson, C.J.
Stroud and even Joshua Dobbs over him, but there's some mild appeal that should at least keep him in lineups over Dak Prescott and Daniel Jones.

LAST WEEK: Looked like a quarterback playing on one leg. Burrow didn't get much into his throws, didn't challenge downfield much, got the ball out as fast as he
could and was sensitive to pass rush pressure. Even when the game was in garbage time Burrow couldn't get much going.

Graph facts:
- Brock Purdy -MENTIONS-> NFL
- Raiders vs. Lions live score, updates, highlights from NFL 'Monday Night Football' game -MENTIONS-> NFL
- Fans spot Travis Kelce wearing Taylor Swift-themed friendship bracelet before she attended his game -MENTIONS-> NFL
- Chiefs vs. Packers live score, updates, highlights from NFL 'Sunday Night Football' game -MENTIONS-> NFL
- Christmas Day preview: 49ers, Ravens square off in potential Super Bowl sneak peek -MENTIONS-> NFL

Entities: NFL, Trevor Lawrence, Russell Wilson, Joshua Dobbs, Chicago, NFL Fantasy Football Week 5 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for
every game, CBSSports.com, Cardinals, C.J. Stroud, Dak Prescott, Daniel Jones, Burrow, Denver
</item 5>

<item 6>
NFL Fantasy Football Week 5 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-04T21:56:25+00:00)
Only Joshua Dobbs, who ran for 55 yards, has scored double-digit Fantasy points against the Niners (18 points). Not that the Cowboys have been tested much --
Daniel Jones, Zach Wilson and the Patriots QBs are the other guys they've man-handled.

O-LINE: San Francisco has allowed a pass rush pressure on 36.4% of their snaps, which sounds like a lot but it's just a shade worse than league average. That
number is weighted by the Giants' decision to blitz on 84.6% of Purdy's dropbacks in Week 2, creating an inflated pass rush pressure rate. Despite the Giants'
efforts, Purdy still threw for 310 yards and two touchdowns in a game he didn't have Brandon Aiyuk in.

COWBOYS: Have become one of the most aggressive defenses in the league, ranking third from the bottom in zone coverage snaps.

Graph facts:
- Vikings vs. Broncos live score, updates, highlights from NFL 'Sunday Night Football' game -MENTIONS-> Joshua Dobbs
- Bears vs. Vikings live score, updates, highlights from NFL 'Monday Night Football' game -MENTIONS-> Joshua Dobbs
- 2024 NFL Mock Draft: Bengals' Joe Burrow gets another playmaker; four QBs go in Round 1, but Pats opt for WR -MENTIONS-> Joshua Dobbs
- Joshua Dobbs -WORKS_FOR-> Vikings
- Joshua Dobbs -PARTICIPATED_IN-> Vikings vs. Broncos game

Entities: Joshua Dobbs, NFL Fantasy Football Week 5 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game, Niners, Week 2, CBSSports.com,
fantasy football, Zach Wilson, Purdy, Daniel Jones, Cowboys, Giants, Patriots, Brandon Aiyuk
</item 6>

<item 7>
Bills vs. Bengals live score, updates, highlights from NFL 'Sunday Night Football' game (Sporting News, 2023-11-05T23:47:00+00:00)
Bengals 24, Bills 10

10:59 p.m.: Another toss play outside to Mixon goes for a loss of three, but Burrow finds Smith, who picks up eight yards to the Bills' 2 before he steps out of
bounds. Third-and-goal at the 2, with the Bengals knocking on the door.

10:57 p.m.: Burrow to Higgins has been the top combination all night, and it is again on the third down play. They connect for 18 yards, with Higgins catching a
slant and races it to the Buffalo 7. First-and-goal with less than 10 minutes to play.

10:56 p.m.: Mixon picks up four yards on a run up the middle, but Burrow throws the ball away on the second-and-6. Bengals use their third and final timeout
with 10:19 left before a third-and-6 play.

10:53 p.m.

Graph facts:
- Brock Purdy -MENTIONS-> NFL
- Raiders vs. Lions live score, updates, highlights from NFL 'Monday Night Football' game -MENTIONS-> NFL
- Fans spot Travis Kelce wearing Taylor Swift-themed friendship bracelet before she attended his game -MENTIONS-> NFL
- Chiefs vs. Packers live score, updates, highlights from NFL 'Sunday Night Football' game -MENTIONS-> NFL
- Christmas Day preview: 49ers, Ravens square off in potential Super Bowl sneak peek -MENTIONS-> NFL

Entities: NFL, Smith, Bills, Bengals, Sporting News, Burrow, Mixon, Higgins, Bills vs. Bengals live score, updates, highlights from NFL 'Sunday Night Football'
game, Bills vs. Bengals NFL 'Sunday Night Football' game, Buffalo
</item 7>

<item 8>
NFL Fantasy Football Week 5 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-04T21:56:25+00:00)
Even when the game was in garbage time Burrow couldn't get much going.

AIR YARDS LAST WEEK: Burrow was 1 for 4 on throws of 10-plus Air Yards and 0 for 2 on throws of 15-plus Air Yards.

AIR YARDS THIS SEASON: Burrow has been 11 for 38 on throws of 10-plus Air Yards and 2 for 22 on throws of 15-plus Air Yards.

TOUCHDOWNS: Burrow has thrown just two scores this season. Both came against Baltimore in Week 2, both were throws inside the five-yard line, both were to Tee
Higgins. That's it. Even his red-zone stats are unfathomable -- he's completed 7 of 15 passes in the red zone. At least 9 of those 15 throws went into the end
zone.

Graph facts:
- NFL Fantasy Football Week 5 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game -MENTIONS-> Brock Purdy
- NFL Fantasy Football Week 5 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game -MENTIONS-> Lamar Jackson
- NFL Fantasy Football Week 5 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game -MENTIONS-> Jimmy Garoppolo
- NFL Fantasy Football Week 5 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game -MENTIONS-> Jakobi Meyers
- NFL Fantasy Football Week 5 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game -MENTIONS-> Goff

Entities: NFL Fantasy Football Week 5 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game, CBSSports.com, Tee Higgins, Burrow, Baltimore
</item 8>

<item 9>
NFL Fantasy Football Week 5 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-04T21:56:25+00:00)
At least 9 of those 15 throws went into the end zone.

PASS RUSH: Bengals opponents have started to sniff out Burrow's issues and have come after him. The Browns and Titans had a pass rush pressure rate between 38&
and 42% in their wins over Cincy with high blitz rates (at least 31% each). Compare that to the Rams and Ravens, who barely brought pressure or blitzes and
either lost or nearly lost to the Bengals.

CARDINALS: Are dead-last in blitz rate this season (14%) and third-to-worst in pass rush pressure rate (28%). They actually had nine sacks after Weeks 1 and 2
but have had three since against the Cowboys and 49ers. Arizona seems to prefer using just four rushers to affect the quarterback, something that should be
helpful to Burrow this week.

Graph facts:
- Brock Purdy -MENTIONS-> NFL
- Raiders vs. Lions live score, updates, highlights from NFL 'Monday Night Football' game -MENTIONS-> NFL
- Fans spot Travis Kelce wearing Taylor Swift-themed friendship bracelet before she attended his game -MENTIONS-> NFL
- Chiefs vs. Packers live score, updates, highlights from NFL 'Sunday Night Football' game -MENTIONS-> NFL
- Christmas Day preview: 49ers, Ravens square off in potential Super Bowl sneak peek -MENTIONS-> NFL

Entities: NFL, pass rush, NFL Fantasy Football Week 5 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game, Titans, Bengals, Blitz Rate, Pass
Rush Pressure Rate, Sacks, CBSSports.com, Cardinals, fantasy football, Ravens, Browns, Rams, Burrow
</item 9>

<item 10>
FanDuel Picks Week 14: Best NFL DFS lineup advice for daily fantasy football tournaments (Sporting News, 2023-12-07T22:17:00+00:00)
RB Joe Mixon, Bengals vs. Colts ($7,200). Mixon has been forced to step up without Burrow under center, doubling as a runner and receiver for Cincinnati. Jake
Browning has relied heavily upon the veteran RB, evidenced by Mixon's two touchdowns on Monday night against the Jaguars. Burrow's injury has made Mixon
arguably the Bengals' top fantasy option now. We'll gladly insert the RB into the lineup in a good matchup against a Indianapolis team that's had some issues
defending RBs, notably getting torched by Derrick Henry and Tyjae Spears one week ago.

WEEK 14 DFS:

DK lineup | Best values | Best stacks

RB Breece Hall, Jets vs. Texans ($6,400). It's Week 14 and there is a Jets player in this DFS lineup. How bad can it get? The good news is that it can always
get worse, and Hall's role in this offense makes him a worthy dart throw in this lineup.

Graph facts:
- NFL standings: Updated AFC, NFC playoff picture for Week 13 of 2023 season -MENTIONS-> Jaguars
- When pop culture and sport collide: A timeline of Taylor Swift’s NFL takeover -MENTIONS-> Jaguars
- Fantasy Football Week 13 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game -MENTIONS-> Jaguars
- NFL Fantasy Football Week 5 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game -MENTIONS-> Jaguars
- Jaguars vs. Saints live score, updates, highlights from NFL 'Thursday Night Football' game -MENTIONS-> Jaguars

Entities: Jaguars, Bengals, Cincinnati, Texans, fantasy football, NFL, Jets, Breece Hall, Sporting News, daily fantasy football, FanDuel Picks Week 14: Best NFL
DFS lineup advice for daily fantasy football tournaments, Burrow, Tyjae Spears, Joe Mixon, Derrick Henry
</item 10>
=== END ITEM id=9d88fe68b3a216e9 ===
