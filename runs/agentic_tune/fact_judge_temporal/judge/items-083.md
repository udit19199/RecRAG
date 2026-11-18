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

=== ITEM id=999e21577a30d57f ===
Gold sentences and facts:
Sentence 0: You're probably starting him if Kenneth Walker III remains out with his oblique injury, but consider other options if you have the depth.
  Fact 0.0: Starting him is probably advised if Kenneth Walker III remains out with his oblique injury.
  Fact 0.1: Other options should be considered if you have the depth.
Sentence 1: With Kenneth Walker III's oblique injury comes opportunity for the rookie, who already looked strong in the month leading up to Seattle's narrow Week 11 loss to the Rams.
  Fact 1.0: Kenneth Walker III has an oblique injury.
  Fact 1.1: Kenneth Walker III's oblique injury creates opportunity for the rookie.
  Fact 1.2: The rookie already looked strong in the month leading up to Seattle's narrow Week 11 loss to the Rams.
Sentence 2: Kenneth Walker III (Seahawks) took four carries against the Rams before leaving with an oblique injury.
  Fact 2.0: Kenneth Walker III plays for the Seahawks.
  Fact 2.1: Kenneth Walker III took four carries against the Rams.
  Fact 2.2: Kenneth Walker III left the game against the Rams with an oblique injury.

Retrieved items:
<item 1>
2023 NFL Week 11 injury roundup: Rams WR Cooper Kupp, Seahawks RB Kenneth Walker III among stars to exit (CBSSports.com, 2023-11-19T23:51:49+00:00)
Smith did return, however, for the final drive of the game in the losing effort. Head coach Pete Carroll said that Smith will "make it back." When asked if that
means for Thursday's matchup against San Francisco, the coach said, "I would think so."

Sam Howell (Commanders) received medical attention on the sidelines after enduring several hard hits to the upper body on a touchdown run against the Giants. He
was quickly cleared of a potential concussion, however, and remained in the game. Veteran journeyman Jacoby Brissett is Washington's backup.

RB

Aaron Jones (Packers) suited up against the Chargers despite a lingering hamstring issue, only to manage just five carries before suffering a knee injury. The
former Pro Bowler needed help to leave the field and was then carted to the locker room, replaced by A.J. Dillon.
</item 1>

<item 2>
2023 NFL Week 11 injury roundup: Rams WR Cooper Kupp, Seahawks RB Kenneth Walker III among stars to exit (CBSSports.com, 2023-11-19T23:51:49+00:00)
He was seen walking on the sidelines shortly afterward but ruled doubtful to return. Rookie Zach Charbonnet, who entered Week 11 with just over 200 rushing
yards on the year, replaced Walker as Seattle's top ball carrier. Following the game, Seahawks head coach Pete Carroll told reporters Walker suffered an
"oblique strain that was legit" and could miss time.

D'Onta Foreman (Bears) remained a headliner of Chicago's backfield against the Lions, even with Khalil Herbert back from injured reserve, but managed just 14
yards on the ground before suffering an ankle injury. He walked off with a limp and attempted to run on the sidelines but did not return to the field after the
Bears ruled him questionable.

Pick Six Newsletter Crafted By The Best NFL Experts Get the day's big stories + fun stuff you love like mock drafts, picks and power rankings.

Graph facts:
- Fantasy Football Week 13 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game -PUBLISHED_BY-> CBSSports.com
- Taylor Swift-Travis Kelce timeline: Everything to know about rumored romance between pop star, Chiefs TE -PUBLISHED_BY-> CBSSports.com
- 2023 Kentucky online sports betting sites: Best legal sportsbooks, promos, bonuses, mobile apps, how to bet -PUBLISHED_BY-> CBSSports.com
- NFL Fantasy Football Week 5 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game -PUBLISHED_BY-> CBSSports.com
- Day of Reckoning -- featuring Anthony Joshua, Deontay Wilder: Predictions, undercard, odds, picks, start time -PUBLISHED_BY-> CBSSports.com

Entities: CBSSports.com, 2023 NFL Week 11 injury roundup: Rams WR Cooper Kupp, Seahawks RB Kenneth Walker III among stars to exit, Pete Carroll, oblique strain,
Pick Six Newsletter, Cooper Kupp, ankle injury, Lions, Rams, Seahawks, Kenneth Walker III, D'Onta Foreman, Zach Charbonnet, Bears, Khalil Herbert
</item 2>

<item 3>
Fantasy Football WR PPR Rankings Week 13: Who to start, best sleepers at wide receiver (Sporting News, 2023-11-29T20:27:00+00:00)
Geno Smith's struggles complicate their fantasy prospects, as well.

What are the biggest WR injuries in Week 13?

Entering Week 12, Tee Higgins (hamstring), Noah Brown (knee), Treylon Burks (concussion), and Kadarius Toney (ankle, hip) all failed to suit up due to nagging
injuries. For now, we've placed Higgins and Brown in our Week 13 list and omitted Burks and Toney. We'll also be eyeing the Chargers' Josh Palmer (knee), who's
eligible to come off the IR this week but probably won't suit up against New England.

Last week, the most notable WRs to exit their respective games with injuries include Chris Olave (concussion), Amari Cooper (ribs), Demario Douglas
(concussion), and Rashid Shaheed (thigh). We've left all four WRs out of our initial Week 13 list but will closely monitor their teams' injury reports to gauge
their playing prospects.

Graph facts:
- Fantasy Football Week 13 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game -MENTIONS-> Chris Olave
- Jaguars vs. Saints live score, updates, highlights from NFL 'Thursday Night Football' game -MENTIONS-> Chris Olave
- Fantasy Football WR PPR Rankings Week 14: Who to start, best sleepers at wide receiver -MENTIONS-> Chris Olave
- Chris Olave -PARTICIPATED_IN-> Jaguars vs. Saints on Week 7 "Thursday Night Football"
- Chris Olave -PARTICIPATED_IN-> Jaguars vs. Saints game

Entities: Chris Olave, New England, Josh Palmer, Fantasy Football WR PPR Rankings Week 13: Who to start, best sleepers at wide receiver, Demario Douglas, Rashid
Shaheed, Amari Cooper, Noah Brown, Treylon Burks, Kadarius Toney, Sporting News, Tee Higgins, Geno Smith, Chargers
</item 3>

<item 4>
2023 NFL Week 11 injury roundup: Rams WR Cooper Kupp, Seahawks RB Kenneth Walker III among stars to exit (CBSSports.com, 2023-11-19T23:51:49+00:00)
The former All-Pro, who battled a lingering hamstring injury earlier this year, received attention on the sidelines after limping off, then left the game for
further evaluation in the locker room. He was ruled questionable to return. NFL Media reported on Monday that Kupp suffered a low ankle sprain and will be
evaluated going forward.

Darius Slayton (Giants) had a big impact against the Commanders, hauling in a deep score from Tommy DeVito, before exiting with an arm injury. He was ruled out
for the remainder of the contest halfway through the third quarter.

EDGE

Joey Bosa (Chargers) appeared to injure his right leg or foot while held in a block during a first-quarter pass-rushing attempt against the Packers. He limped
to the sidelines and was carted to the locker room, then ruled questionable to return.

Graph facts:
- Christmas Day preview: 49ers, Ravens square off in potential Super Bowl sneak peek -MENTIONS-> NFL
- NFL DFS picks Week 15: Best sleepers, value players for FanDuel, DraftKings, SuperDraft lineups -MENTIONS-> Tommy DeVito
- Fantasy Football QB Rankings Week 14: Who to start, best sleepers at quarterback -MENTIONS-> Tommy DeVito
- Christmas Day preview: 49ers, Ravens square off in potential Super Bowl sneak peek -MENTIONS-> Tommy DeVito
- Brock Purdy -MENTIONS-> NFL

Entities: Tommy DeVito, NFL, Packers, CBSSports.com, 2023 NFL Week 11 injury roundup: Rams WR Cooper Kupp, Seahawks RB Kenneth Walker III among stars to exit,
Cooper Kupp, Joey Bosa, NFL Media, lingering hamstring injury, low ankle sprain, arm injury, right leg or foot injury, Darius Slayton, Rams, Seahawks
</item 4>

<item 5>
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
</item 5>

<item 6>
Fantasy Football RB Rankings Week 12: Who to start, best sleepers at running back (Sporting News, 2023-11-21T21:33:00+00:00)
They weren't awful against Dallas last week on a per-touch basis, but neither finished with more than 12 touches. Even worse, no Carolina back has enjoyed a
60-yard rushing game since Oct. 15. Bad offensive line, negative game scripts, awful coaching — it's all bad, man!

What are the biggest RB injuries in Week 12?

Second-year Seahawks back Kenneth Walker III suffered an oblique injury in Week 11 that the ever-hip Pete Carroll referred to as "legit." KW3 has a better
chance of missing multiple weeks than he does at suiting up for Seattle's Thanksgiving Day game against Fred Warner and the 49ers. If Zach Charbonnet somehow
finds himself on your waiver wire, an early Merry Christmas to you if you have the FAAB or a high waiver priority.

The hits just keep on coming for the Packers' Aaron Jones, who suffered a knee injury but seems to have at least caught a break in the sense that he didn't tear
his ACL.

Graph facts:
- NFL standings: Updated AFC, NFC playoff picture for Week 13 of 2023 season -MENTIONS-> Packers
- FanDuel Picks Week 15: Best NFL DFS lineup advice for daily fantasy football tournaments -MENTIONS-> Packers
- Fantasy Football Week 13 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game -MENTIONS-> Packers
- NFL Fantasy Football Week 5 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game -MENTIONS-> Packers
- Chiefs vs. Packers live score, updates, highlights from NFL 'Sunday Night Football' game -MENTIONS-> Packers

Entities: Packers, Pete Carroll, Sporting News, knee injury, Seahawks, Kenneth Walker III, Aaron Jones, Zach Charbonnet, oblique injury, ACL, Fantasy Football
RB Rankings Week 12: Who to start, best sleepers at running back, 49ers, Fred Warner
</item 6>

<item 7>
Fantasy Football RB PPR Rankings Week 14: Who to start, best sleepers at running back (Sporting News, 2023-12-07T22:09:00+00:00)
Tampa Bay has a pretty good run defense, so Allgeier can't be trusted outside of a desperate flex play with his nine average touches over the past two weeks.

What are the biggest RB injuries in Week 14?

Derrick Henry (head/neck), Rhamondre Stevenson (ankle), Kenneth Walker III (oblique), Aaron Jones (knee), D'Onta Foreman (ankle), and D'Andre Swift
(undisclosed) are all "questionable," at best, to suit up this week. We know for sure Stevenson will be out, but we'll have to monitor the injury reports
throughout the week for the rest.

Jonathan Taylor (thumb) underwent surgery last week and will miss the next few weeks, opening the door for Zack Moss to resume his role as Indy's super-sub lead
back.

We'll be updating these RB PPR rankings all week, so check back for the latest player movement.

Graph facts:
- Fantasy Football Week 13 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game -MENTIONS-> Rhamondre Stevenson
- NFL Fantasy Football Week 6 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game -MENTIONS-> Rhamondre Stevenson
- Fantasy Football RB Rankings Week 14: Who to start, best sleepers at running back -MENTIONS-> Rhamondre Stevenson
- NFL Fantasy Football Week 5 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game -MENTIONS-> Rhamondre Stevenson
- Patriots vs. Broncos live score, updates, highlights from NFL Sunday night football game -MENTIONS-> Rhamondre Stevenson

Entities: Rhamondre Stevenson, Tampa Bay, Week 14, Sporting News, Allgeier, Zack Moss, Indy, RB injuries, Fantasy Football RB PPR Rankings Week 14: Who to
start, best sleepers at running back, Derrick Henry, D'Andre Swift, Kenneth Walker III, Aaron Jones, D'Onta Foreman, Jonathan Taylor
</item 7>

<item 8>
Fantasy Football RB Rankings Week 14: Who to start, best sleepers at running back (Sporting News, 2023-12-07T22:07:00+00:00)
The injury highlights the importance of handcuffs, as Zack Moss has put up multiple top-10 performances with and without Taylor this season.

Seattle's Kenneth Walker III (oblique) missed Week 12 and 13, and we're not confident in his ability to suit up against the hard-hitting 49ers coming off a big
win in Philly. Backup Zach Charbonnet exited early because of a knee injury against the Cowboys, putting his status in doubt, too. If both are out, DeeJay
Dallas would start and lead the backfield in touches.

Green Bay's Aaron Jones has a sprained MCL, and sources have revealed that the Packers continue to consider him week-to-week. He only logged four carries before
sustaining the injury in Week 11, then he sat for the Packers' Thanksgiving Day tilt with the Lions and their Sunday Night Football win over Kansas City. Why
rush him back if Green Bay is already winning games?

Graph facts:
- NFL standings: Updated AFC, NFC playoff picture for Week 13 of 2023 season -MENTIONS-> Week 13
- NFL awards odds: Six longshots bettors should consider for MVP, OPOY, DPOY and more -MENTIONS-> Week 13
- Fantasy Football WR PPR Rankings Week 14: Who to start, best sleepers at wide receiver -MENTIONS-> Week 13
- Fantasy Football WR PPR Rankings Week 13: Who to start, best sleepers at wide receiver -MENTIONS-> Week 13
- NFL Week 13 injuries: Browns to start Joe Flacco with Dorian Thompson-Robinson out, Saints' Olave questionable -MENTIONS-> Week 13

Entities: Week 13, Week 12, Kansas City, Packers, fantasy football, Week 14, running back, DeeJay Dallas, Philly, Week 11, Thanksgiving Day tilt with the Lions,
Sunday Night Football win over Kansas City, Fantasy Football RB Rankings Week 14: Who to start, best sleepers at running back, Lions, Sporting News
</item 8>

<item 9>
2023 NFL Week 11 injury roundup: Rams WR Cooper Kupp, Seahawks RB Kenneth Walker III among stars to exit (CBSSports.com, 2023-11-19T23:51:49+00:00)
Dillon. Green Bay ruled him questionable to return, then downgraded him to out, but coach Matt LaFleur said after the Packers' win over Los Angeles that he
doesn't expect the injury to be a long-term issue.

De'Von Achane (Dolphins) returned from injured reserve -- and a four-game absence -- but limped to the sidelines after his second touch of the first quarter
against the Raiders. Miami ruled him questionable to return with a knee injury, and while he later returned to the sidelines from a locker-room visit, he was
replaced on the field indefinitely by Raheem Mostert. Head coach Mike McDaniel told reporters on Monday that Achane is day-to-day. While he hasn't been ruled
out of Friday's matchup with the Jets, he's not been given the green light either, so his status is worth monitoring.

Kenneth Walker III (Seahawks) took four carries against the Rams before leaving with an oblique injury.
</item 9>

<item 10>
Graph query row:
article: Fantasy Football RB Rankings Week 14: Who to start, best sleepers at running back
published_on: 2023-12-07T22:07:00+00:00
publisher: Sporting News
relationship: MENTIONS
related: DeeJay Dallas
</item 10>
=== END ITEM id=999e21577a30d57f ===

=== ITEM id=b0666735da1a8974 ===
Gold sentences and facts:
Sentence 0: Even so, Kelce's 81.3 average receiving yards per game against the Raiders since 2019 suggests the All-Pro tight end is in line for a productive outing and could potentially snap his four-game scoring drought.
  Fact 0.0: Kelce averages 81.3 receiving yards per game against the Raiders since 2019.
  Fact 0.1: Kelce is an All-Pro tight end.
  Fact 0.2: Kelce's average suggests he is in line for a productive outing.
  Fact 0.3: Kelce could potentially snap his four-game scoring drought.
Sentence 1: The first link between Swift and Kelce was established in 2016, when Kelce – who helped bring his team to Super Bowl victory in both 2020 and 2023 – played the classic game “Kiss, Marry, Kill”.
  Fact 1.0: The first link between Swift and Kelce was established in 2016.
  Fact 1.1: The first link between Swift and Kelce came when Kelce played the classic game 'Kiss, Marry, Kill' in 2016.
  Fact 1.2: Kelce helped bring his team to a Super Bowl victory in 2020.
  Fact 1.3: Kelce helped bring his team to a Super Bowl victory in 2023.

Retrieved items:
<item 1>
Is the Coverage on Taylor Swift and Travis Kelce’s Romance Nothing but Hype or a Pivotal Shift in NFL’s Century-Old Game Plan Towards Global Recognition?
(Essentially Sports, 2023-10-25T21:30:41+00:00)
From Colin Kaepernick taking a knee during the national anthem to protest against racism, to domestic violence by players, to the Deflategate and Bountygate
scandals. But there have been bigger crises as well. The Union of Concerned Scientists accused the NFL of trying to intimidate scientists studying the link
between CTE and pro football. The NFL is already way past other leagues in terms of revenue, and now with the pivotal shift in its strategy, “America’s favorite
pastime” is on its way to world domination.

ADVERTISEMENT Article continues below this ad

Editorial Disclaimer: The views expressed are those of the author and do not necessarily reflect the views of EssentiallySports.

Watch This Story: Will Patrick Mahomes Eclipse Tom Brady as the NFL Goat? Travis Kelce Responds

Graph facts:
- Brock Purdy -MENTIONS-> NFL
- Raiders vs. Lions live score, updates, highlights from NFL 'Monday Night Football' game -MENTIONS-> NFL
- Fans spot Travis Kelce wearing Taylor Swift-themed friendship bracelet before she attended his game -MENTIONS-> NFL
- Chiefs vs. Packers live score, updates, highlights from NFL 'Sunday Night Football' game -MENTIONS-> NFL
- Christmas Day preview: 49ers, Ravens square off in potential Super Bowl sneak peek -MENTIONS-> NFL

Entities: NFL, Taylor Swift, Travis Kelce, Patrick Mahomes, Is the Coverage on Taylor Swift and Travis Kelce’s Romance Nothing but Hype or a Pivotal Shift in
NFL’s Century-Old Game Plan Towards Global Recognition?, Global recognition, EssentiallySports, Colin Kaepernick, Union of Concerned Scientists, racism,
domestic violence by players, Deflategate, Bountygate, CTE, pro football
</item 1>

<item 2>
N.F.L. Executives on the Biggest Questions Facing Super Bowl Contenders (The New York Times, 2023-10-16T23:58:53+00:00)
That figure was 7.8 in Mahomes’ first year as the starter and has climbed every season since. It’s a testament to the Chiefs’ adaptability, but also a
reflection of plummeting explosive play rates.

Exec comment: “Yeah, it looks harder for them on offense until Mahomes finds (Travis) Kelce on every major third down, every major two-minute play, every major
touchdown. Kansas City has the luxury of their best players being their toughest players. Chris Jones is a very tough player. Kelce is a very tough player.
Mahomes is a very tough player. You wonder why they win the games? Because the games are won by the toughest teams most of the time.”

• Philadelphia Eagles: Has Jalen Hurts regressed? Where are the explosive pass plays?

What I think: Explosive passing is down through most of the league, including for the Eagles.

Graph facts:
- NFL standings: Updated AFC, NFC playoff picture for Week 13 of 2023 season -MENTIONS-> Philadelphia Eagles
- Travis Kelce says he tried to delete his ‘nonsense’ tweets before they went viral -MENTIONS-> Philadelphia Eagles
- Taylor Swift-Travis Kelce timeline: Everything to know about rumored romance between pop star, Chiefs TE -MENTIONS-> Philadelphia Eagles
- NFL power rankings: Lions, 49ers level up; Texans jump Bengals, Bills; Jets, Patriots sink to bottom for Week 11 -MENTIONS-> Philadelphia Eagles
- Christmas Day preview: 49ers, Ravens square off in potential Super Bowl sneak peek -MENTIONS-> Philadelphia Eagles

Entities: Philadelphia Eagles, The New York Times, Travis Kelce, Mahomes, Jalen Hurts, Super Bowl, Chiefs, N.F.L., Chris Jones, explosive play rates, explosive
passing, N.F.L. Executives on the Biggest Questions Facing Super Bowl Contenders
</item 2>

<item 3>
Is the Coverage on Taylor Swift and Travis Kelce’s Romance Nothing but Hype or a Pivotal Shift in NFL’s Century-Old Game Plan Towards Global Recognition?
(Essentially Sports, 2023-10-25T21:30:41+00:00)
$28 billion. And just like that, their target will be met.

USA Today via Reuters NFL, American Football – Taylor Swift reacts while sitting next to Donna Kelce watching the Kansas City Chiefs vs Chicago Bears game
during the first half at GEHA Field at Arrowhead Stadium, Kansas City, Missouri, USA | Image Credits: Denny Medley-USA TODAY Sports/Reuters

And while analysts like Stephen A. Smith may go around telling people they should be embarrassed of themselves for getting so invested in someone else’s
relationship, it is good business for the NFL. TMZ reported that #87’s jersey sales skyrocketed 400% and his Instagram followers by 276%. Yahoo Finance reported
a 53% rise in viewership among teen girls. Who would have thought in their wildest dreams that the NFL would manage to rope in this part of the population? The
October 1 matchup between the Chiefs and the Jets drew 27 million viewers.

Graph facts:
- Brock Purdy -MENTIONS-> NFL
- Raiders vs. Lions live score, updates, highlights from NFL 'Monday Night Football' game -MENTIONS-> NFL
- Fans spot Travis Kelce wearing Taylor Swift-themed friendship bracelet before she attended his game -MENTIONS-> NFL
- Chiefs vs. Packers live score, updates, highlights from NFL 'Sunday Night Football' game -MENTIONS-> NFL
- Christmas Day preview: 49ers, Ravens square off in potential Super Bowl sneak peek -MENTIONS-> NFL

Entities: NFL, Kansas City, Missouri, Chicago Bears, Taylor Swift, Travis Kelce, New York Jets, Reuters, Essentially Sports, Stephen A. Smith, TMZ, Yahoo
Finance, Denny Medley, Chiefs and Jets matchup, Kansas City Chiefs vs Chicago Bears game
</item 3>

<item 4>
Taylor Swift-Travis Kelce timeline: Everything to know about rumored romance between pop star, Chiefs TE (CBSSports.com, 2023-10-12T23:21:52+00:00)
"'Taylor wants a guy that's into his career, does his own thing, but is also supportive and understands her,' the source added, "and Travis fits those
qualities.'"

Sept. 25: Kelce jersey sales skyrocket

Call it what you want; I'll call it the Taylor Swift effect. Since Swift attended the Chiefs game, there was a 400% increase in Kelce jersey sales, his podcast
is No. 1 overall on Apple, he gained over 380,000 Instagram followers, and 24.3 million people watched the game, making it the No. 1 game last week, according
to Front Office Sports.

There has also been triple the "Chiefs" searches on web and triple the Chiefs ticket sales on Stubhub.

Sept. 25: Belichick weighs in on the matter



New England Patriots head coach does not always comment on situations that take place in the game, but he was fully ready with a sound bite on the Kelce-Swift
situation.

Graph facts:
- Taylor Swift -MENTIONS-> jersey sales
- Christmas Day preview: 49ers, Ravens square off in potential Super Bowl sneak peek -MENTIONS-> Taylor Swift
- Country music couples Tim McGraw, Faith Hill and Garth Brooks, Trisha Yearwood share tips to lasting marriages -MENTIONS-> Taylor Swift
- Fans spot Travis Kelce wearing Taylor Swift-themed friendship bracelet before she attended his game -MENTIONS-> Taylor Swift
- Chiefs vs. Packers live score, updates, highlights from NFL 'Sunday Night Football' game -MENTIONS-> Taylor Swift

Entities: Taylor Swift, rumored romance, Travis Kelce, Belichick, Front Office Sports, Apple, Stubhub, Taylor Swift-Travis Kelce timeline: Everything to know
about rumored romance between pop star, Chiefs TE, Chiefs game, CBSSports.com, Chiefs, Instagram, New England Patriots
</item 4>

<item 5>
Is the Coverage on Taylor Swift and Travis Kelce’s Romance Nothing but Hype or a Pivotal Shift in NFL’s Century-Old Game Plan Towards Global Recognition?
(Essentially Sports, 2023-10-25T21:30:41+00:00)
The October 1 matchup between the Chiefs and the Jets drew 27 million viewers. And became the most-watched Sunday TV show since the last Super Bowl! And the
last Super Bowl had an actual concert by Rihanna.

Revealed: The Dark Side Of Pat McAfee & Why Rob Gronkowski Is In The Firing Line

Kaepernick, who? Deflategate, what? In the media frenzy surrounding the two lovebirds, other issues get thrown out of the window. How can fans forget that the
NFL is also organizing games outside the American borders with a view to garnering international appeal? The NFL has scheduled 5 games abroad this season. It is
also addressing the concussion controversies. It recently added both new concussion rules and a new $100 million initiative called “Play Smart, Play Safe,” to
make the sport safer for players. And all the effort is showing.

Graph facts:
- Brock Purdy -MENTIONS-> NFL
- Raiders vs. Lions live score, updates, highlights from NFL 'Monday Night Football' game -MENTIONS-> NFL
- Fans spot Travis Kelce wearing Taylor Swift-themed friendship bracelet before she attended his game -MENTIONS-> NFL
- Chiefs vs. Packers live score, updates, highlights from NFL 'Sunday Night Football' game -MENTIONS-> NFL
- Christmas Day preview: 49ers, Ravens square off in potential Super Bowl sneak peek -MENTIONS-> NFL

Entities: NFL, Pat McAfee, Taylor Swift, Travis Kelce, Jets, Chiefs, Essentially Sports, October 1 matchup, last Super Bowl, Rihanna, concert at the last Super
Bowl, Rob Gronkowski, Kaepernick, Deflategate, concussion controversies
</item 5>

<item 6>
Christmas Day preview: 49ers, Ravens square off in potential Super Bowl sneak peek (Yardbarker, 2023-12-24T23:34:39+00:00)
Can Raiders QB Aidan O’Connell earn next season’s starting job?: After leading the offense on seven touchdown scoring drives in last week's 63-21 win over the
Chargers, the rookie fourth-rounder aims to further establish himself as a serious contender for Las Vegas' starting job next season. Despite the encouraging
performance, O'Connell has been inconsistent across six games as the full-time starter, throwing for 1,300 yards, seven touchdowns, and five interceptions.

However, a stellar showing against the NFL's fifth-ranked pass defense (202.5 yards per game allowed) would benefit his bid to be the starter in 2024.

Chiefs TE Travis Kelce: Yes, his girlfriend will be in town for the holiday. Kelce is averaging 89.6 yards per game with global popstar Taylor Swift in
attendance this season but is coming off his worst outing in her presence, catching five passes for 28 yards.

Graph facts:
- Christmas Day preview: 49ers, Ravens square off in potential Super Bowl sneak peek -MENTIONS-> New York Giants
- Christmas Day preview: 49ers, Ravens square off in potential Super Bowl sneak peek -MENTIONS-> Philadelphia Eagles
- Christmas Day preview: 49ers, Ravens square off in potential Super Bowl sneak peek -MENTIONS-> Las Vegas Raiders at Kansas City Chiefs
- Christmas Day preview: 49ers, Ravens square off in potential Super Bowl sneak peek -MENTIONS-> 49ers, Ravens square off in potential Super Bowl sneak peek
- Christmas Day preview: 49ers, Ravens square off in potential Super Bowl sneak peek -MENTIONS-> New York Giants at Philadelphia Eagles

Entities: Christmas Day preview: 49ers, Ravens square off in potential Super Bowl sneak peek, Aidan O’Connell, 63-21 win over the Chargers, pass defense,
Raiders, NFL, Taylor Swift, Travis Kelce, Super Bowl, Ravens, Chiefs, Christmas Day, Chargers, 49ers, Yardbarker
</item 6>

<item 7>
Graph query row:
article: Christmas Day preview: 49ers, Ravens square off in potential Super Bowl sneak peek
published_on: 2023-12-24T23:34:39+00:00
publisher: Yardbarker
person: Travis Kelce
passage: Christmas Day preview: 49ers, Ravens square off in potential Super Bowl sneak peek (Yardbarker, 2023-12-24T23:34:39+00:00)
Christmas Day isn't just for the NBA, as the NFL is set to air an exciting triple-header featuring fierce division clashes and a highly anticipated matchup we
could see again on Super Bowl Sunday. Here are the top players and storylines to monitor for Monday's slate.

Las Vegas Raiders (6-8) at Kansas City Chiefs (9-5) | 1 p.m. ET

New chapter in Patrick Mahomes-Maxx Crosby rivalry: This hasn't been as much of a rivalry between the teams as of late, with Kansas City holding a 10-1 record
over the Raiders since Mahomes became its starting quarterback. Yet it has sparked an exciting feud between the two-time Super Bowl MVP and the two-time Pro
Bowler Crosby, who has five sacks in nine career games against the Chiefs.

Mahomes and Crosby might not like facing one another, though it's clear there's mutual respect between the two, which only makes their rivalry more intriguing
to watch.

Can Raiders QB Aidan O’Connell earn next season’s starting job?
</item 7>

<item 8>
Travis Kelce admits the NFL is overdoing it with their Taylor Swift coverage (The Independent - Life and Style, 2023-10-04T19:58:00+00:00)
Stay ahead of the trend in fashion and beyond with our free weekly Lifestyle Edit newsletter Stay ahead of the trend in fashion and beyond with our free weekly
Lifestyle Edit newsletter Please enter a valid email address Please enter a valid email address SIGN UP I would like to be emailed about offers, events and
updates from The Independent. Read our privacy notice Thanks for signing up to the

Lifestyle Edit email {{ #verifyErrors }} {{ message }} {{ /verifyErrors }} {{ ^verifyErrors }} Something went wrong. Please try again later {{ /verifyErrors }}

Travis Kelce believes the National Football League’s coverage of his rumoured girlfriend Taylor Swift is a bit too much.

Following his game against the New York Jets on 1 October, the 33-year-old Kansas City Chiefs tight end opened up about the NFL broadcast on the most recent
episode of his podcast, New Heights with Jason and Travis Kelce.

Graph facts:
- Travis Kelce says he tried to delete his ‘nonsense’ tweets before they went viral -MENTIONS-> Jason Kelce
- Taylor Swift-Travis Kelce timeline: Everything to know about rumored romance between pop star, Chiefs TE -MENTIONS-> Jason Kelce
- Travis Kelce faces backlash after comments about ‘finding a breeder’ resurface amid Taylor Swift romance -MENTIONS-> Jason Kelce
- Bears vs. Panthers live score, updates, highlights from NFL 'Thursday Night Football' game -MENTIONS-> Jason Kelce
- Fans spot Travis Kelce wearing Taylor Swift-themed friendship bracelet before she attended his game -MENTIONS-> Jason Kelce

Entities: Jason Kelce, Taylor Swift, Travis Kelce, New York Jets, The Independent, Kansas City Chiefs, National Football League, game against the New York Jets,
New Heights with Jason and Travis Kelce, Travis Kelce admits the NFL is overdoing it with their Taylor Swift coverage
</item 8>

<item 9>
Travis Kelce admits the NFL is overdoing it with their Taylor Swift coverage (The Independent - Life and Style, 2023-10-04T19:58:00+00:00)
“I think everybody is just, like, overwhelmed,” the Chiefs tight end replied.

“Take away your feelings for Taylor,” Jason added. “What is your honest opinion on how the NFL is treating celebrities at the games?”

Seemingly beating around the bush, Kelce remarked: “I think it brings a little more to the atmosphere, it brings a little bit more to what you’re watching. But
at the same time, I think...”

Before his younger brother could continue his thought, the Philadelphia Eagles player chimed in: “They’re overdoing it.”

“They’re overdoing it a little bit, for sure,” Kelce agreed. “Especially my situation. I think they’re just trying to have fun with it.”

While sports lovers may not be too pleased with the NFL’s focus on Swift, the football league’s supposed obsession with ongoing Swift and Kelce dating rumours
has paid off in terms of ratings.

Graph facts:
- NFL standings: Updated AFC, NFC playoff picture for Week 13 of 2023 season -MENTIONS-> Philadelphia Eagles
- Travis Kelce says he tried to delete his ‘nonsense’ tweets before they went viral -MENTIONS-> Philadelphia Eagles
- Taylor Swift-Travis Kelce timeline: Everything to know about rumored romance between pop star, Chiefs TE -MENTIONS-> Philadelphia Eagles
- NFL power rankings: Lions, 49ers level up; Texans jump Bengals, Bills; Jets, Patriots sink to bottom for Week 11 -MENTIONS-> Philadelphia Eagles
- Christmas Day preview: 49ers, Ravens square off in potential Super Bowl sneak peek -MENTIONS-> Philadelphia Eagles

Entities: Philadelphia Eagles, NFL, Jason, Taylor Swift, Travis Kelce, ratings, Chiefs, The Independent, Travis Kelce admits the NFL is overdoing it with their
Taylor Swift coverage, dating rumours
</item 9>

<item 10>
Travis Kelce admits the NFL is overdoing it with their Taylor Swift coverage (The Independent - Life and Style, 2023-10-04T19:58:00+00:00)
“The Taylor Swift and Travis Kelce news has been a pop cultural moment we’ve leaned into in real time, as it’s an intersection of sport and entertainment, and
we’ve seen an incredible amount of positivity around the sport.”

“The vast majority of our content has remained focused on the game, our players and variety of other initiatives, including our Toy Story Funday Football
alt-cast, the international games and more,” they continued.

Graph facts:
- Brock Purdy -MENTIONS-> NFL
- Raiders vs. Lions live score, updates, highlights from NFL 'Monday Night Football' game -MENTIONS-> NFL
- Fans spot Travis Kelce wearing Taylor Swift-themed friendship bracelet before she attended his game -MENTIONS-> NFL
- Chiefs vs. Packers live score, updates, highlights from NFL 'Sunday Night Football' game -MENTIONS-> NFL
- Christmas Day preview: 49ers, Ravens square off in potential Super Bowl sneak peek -MENTIONS-> NFL

Entities: NFL, Taylor Swift, Travis Kelce, The Independent, Travis Kelce admits the NFL is overdoing it with their Taylor Swift coverage, Toy Story Funday
Football
</item 10>
=== END ITEM id=b0666735da1a8974 ===

=== ITEM id=392a18c3b46ce9c8 ===
Gold sentences and facts:
Sentence 0: Depending on which one you get, it might come with USB ports so you can charge multiple devices in one outlet.
  Fact 0.0: Depending on which one a reader gets, it might come with USB ports.
  Fact 0.1: The USB ports let multiple devices be charged in one outlet.
Sentence 1: With a 10,000-mAh capacity, WIRED contributor Simon Hill says it charged his iPhone 14 Pro once and then delivered an additional 64 percent the second time before tapping out.
  Fact 1.0: The device has a 10,000-mAh capacity.
  Fact 1.1: WIRED contributor Simon Hill says the device charged his iPhone 14 Pro once.
  Fact 1.2: Simon Hill says the device delivered an additional 64 percent the second time before tapping out.

Retrieved items:
<item 1>
Some of our favorite devices are on sale for Black Friday. (The Verge, 2023-11-24T15:42:21+00:00)
A car-friendly charger

Sean Hollister, senior editor

Maybe someday Apple will realize that it wasn’t the brightest idea to artificially limit the reach of its MagSafe charging ecosystem and you’ll be able to plop
your phone on a magical minimalist disc that charges it at high speeds. In the meanwhile, a standard Qi charger is about the best you’ll get — and the bulky but
practical iOttie Easy One Touch Qi does it with the satisfying snap of springs. When you push your phone into its waiting jaws, it depresses a button that
causes those jaws to firmly snap closed on either side of your device. When you want to remove it, you pinch a pair of levers with your finger and thumb to
release as you grab your slab. It’s wide enough to fit practically anything on the market, save an opened Samsung Z Fold. I’ve used one for years with Android
and Apple phones alike, including newer MagSafe handsets.

Graph facts:
- What to know about Fortnite maker Epic Games’ antitrust battle with Google, starting today -MENTIONS-> Android
- ChatGPT’s mobile app hits record $4.58M in revenue in September -MENTIONS-> Android
- ChatGPT: Everything you need to know about the AI-powered chatbot -MENTIONS-> Android
- The best Black Friday deals you can get for under $25 -MENTIONS-> Android
- More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo -MENTIONS-> Android

Entities: Android, Apple, Samsung, Black Friday, Some of our favorite devices are on sale for Black Friday., Sean Hollister, MagSafe charging ecosystem,
standard Qi charger, iOttie Easy One Touch Qi, Samsung Z Fold, The Verge
</item 1>

<item 2>
Some of our favorite devices are on sale for Black Friday. (The Verge, 2023-11-24T15:42:21+00:00)
And while I’ve long been a proponent of MPOWERD’s solar-powered Luci lanterns, I recently picked up the company’s like-minded string lights for car camping and
overnight jaunts in the backcountry when I don’t mind toting a little extra weight with me.

The 18-foot string is certainly not the brightest you can buy — it packs a series of 100-lumen LEDs, whereas your average headlamp might offer 400 — but it can
swap between six different colors and features a 2,000mAh battery for when your phone needs some emergency juice. Best of all, you can charge the lights via USB
or solar, meaning you can spend less time worrying about how to keep them going and more time taking in the vibes.

Memory card holder and reader

Becca Farsace, senior producer

As a video person who is constantly on the move, there is nothing better than the consolidation of gear — especially when it pertains to dongles.

Graph facts:
- The Ray-Ban Meta smart glasses actually make the future look cool -MENTIONS-> Becca Farsace
- 200+ of the best deals from Amazon's Cyber Monday sale -MENTIONS-> Black Friday
- The best Black Friday deals you can get for under $25 -MENTIONS-> Black Friday
- The best Apple Cyber Monday deals of 2023: Save on AirPods, iPads, Apple Watches and more -MENTIONS-> Black Friday
- The best Apple Black Friday deals on iPads, AirPods, Apple Watches and more -MENTIONS-> Black Friday

Entities: Becca Farsace, Black Friday, Some of our favorite devices are on sale for Black Friday., MPOWERD, Luci lanterns, string lights, headlamp, Memory card
holder and reader, dongles, The Verge
</item 2>

<item 3>
78 Absolute Best October Prime Day Deals (2023) (Wired, 2023-10-10T14:23:00+00:00)
The performance is fast, the recharging is fast. It even looks fast. It has a bright, 120-Hz AMOLED screen, great speakers, and surprisingly good battery life
for all that. Our main gripe is the IP64 water- and dust-resistance rating, which is not nearly as good as other flagship phones.

Ugreen's charger is, as the name suggests, a 145-watt charger with a 25,000-mAh battery. It's surprisingly compact for the power it provides, although at 1.1
pounds, it's definitely not ultralight. There are two USB-C ports and one USB-A port. What sets the Ugreen apart is that you can actually draw 145 watts while
charging. That works out to one USB-C port at 100W and the other at 45W. Very few other batteries we've tested are capable of that feat.

Photograph: Anker

The Anker Nano is one of our favorite portable power banks, particularly for phones.

Graph facts:
- The Israel-Hamas War Is Drowning X in Disinformation -PUBLISHED_BY-> WIRED
- These Are Our Favorite Walmart Cyber Monday Deals -PUBLISHED_BY-> WIRED
- Wired opinion article -PUBLISHED_BY-> WIRED
- 14 Best Deals: TVs, Keyboards, Mice, Chargers, and Earbuds -PUBLISHED_BY-> WIRED
- 54 Best Apple Black Friday Deals (2023): iPad, Apple Watch, AirPods -PUBLISHED_BY-> WIRED

Entities: WIRED, Ugreen, Ugreen charger, Anker Nano, 78 Absolute Best October Prime Day Deals (2023), Anker
</item 3>

<item 4>
Some of our favorite devices are on sale for Black Friday. (The Verge, 2023-11-24T15:42:21+00:00)
Older buildings especially suffer from what I can only describe as loose socket syndrome, where those very convenient modern boxy socket extenders with five
different USB and USB-C charging ports simply cannot stay in place and fall right out of the wall because they’re too heavy. After one (totally pleasant)
vacation where I had to charge my phone, watch, AirPods, and laptop in a weird corner of my room with the plug-in charging hub propped up on a strategically
balanced mountain of books and sham pillows, I bought this Anker combination extension cord / power strip. It’s not a full power strip — just a cube with a few
sockets along with USB and USB-C charging ports at the end of a five-foot cable. I’ve brought it on a few trips since then. It takes up extra space in my
suitcase but each time has left me feeling vindicated about the purchase.

The three prongs at the end of the cable are static, rather than folding flat for easy packing.

Graph facts:
- The best Apple Cyber Monday deals of 2023: Save on AirPods, iPads, Apple Watches and more -MENTIONS-> AirPods
- The best Apple Black Friday deals on iPads, AirPods, Apple Watches and more -MENTIONS-> AirPods
- The 53 best Black Friday deals we could find at Amazon, Walmart, Target and more -MENTIONS-> AirPods
- Here are over 40 last-minute gifts you can still nab before Christmas -MENTIONS-> AirPods
- 200+ of the best deals from Amazon's Cyber Monday sale -MENTIONS-> AirPods

Entities: AirPods, Anker, Black Friday, Some of our favorite devices are on sale for Black Friday., Anker combination extension cord / power strip, The Verge
</item 4>

<item 5>
54 Best Apple Black Friday Deals (2023): iPad, Apple Watch, AirPods (Wired, 2023-11-25T00:00:29+00:00)
Flip over to the back, and you'll find two USB-A ports, two USB-C ports, and three AC outlets. It's a compact and tidy charger, and our unit has been going
strong for more than two years. It fluctuates in price very often, and it's hit as low as $60 in the past, but we still think this is a good deal.

This power bank is one of our favorites if you're looking for an upgrade. It has an absolutely gargantuan 24,000-mAh capacity, and it can be charged up in an
hour. It can output up to 140W, making it powerful enough to charge laptops and tablets, as well as charging your phone super quick.

This is a great charger if you travel often with an iPhone and Apple Watch. Instead of packing two separate chargers, the Duo offers the luxury of both all
within a compact, foldable form factor.

Graph facts:
- The best Apple Cyber Monday deals of 2023: Save on AirPods, iPads, Apple Watches and more -MENTIONS-> AirPods
- The best Apple Black Friday deals on iPads, AirPods, Apple Watches and more -MENTIONS-> AirPods
- The 53 best Black Friday deals we could find at Amazon, Walmart, Target and more -MENTIONS-> AirPods
- Here are over 40 last-minute gifts you can still nab before Christmas -MENTIONS-> AirPods
- 200+ of the best deals from Amazon's Cyber Monday sale -MENTIONS-> AirPods

Entities: AirPods, iPad, Black Friday deals, Apple Watch, WIRED, phone, power bank, tablets, laptops, iPhone, Duo, 54 Best Apple Black Friday Deals (2023):
iPad, Apple Watch, AirPods, charger
</item 5>

<item 6>
78 Absolute Best October Prime Day Deals (2023) (Wired, 2023-10-10T14:23:00+00:00)
There's even a built-in display to monitor stats like temperature and remaining battery percentage.

This heavy metal brick is a good option for charging laptops and smaller gadgets. It packs 20,000 mAh and supports a wide variety of fast charging standards.
There is one USB-C PD port rated at 65 watts, two USB-A QC ports at 30 watts apiece, and a micro-USB input (though you are best using the USB-C to recharge it).
We haven't seen it go on sale very often.

This travel kit from ESR works well if you want something that can prop your iPhone in portrait or landscape orientation. It can also display your Apple Watch
in Nightstand mode, so you can peek and see how long before you must get out of bed. The main body folds open with a MagSafe charging pad for your iPhone and a
slot behind for your AirPods. We have seen this dip a little lower, but this is still a solid deal.

Graph facts:
- The best Apple Cyber Monday deals of 2023: Save on AirPods, iPads, Apple Watches and more -MENTIONS-> AirPods
- The best Apple Black Friday deals on iPads, AirPods, Apple Watches and more -MENTIONS-> AirPods
- The 53 best Black Friday deals we could find at Amazon, Walmart, Target and more -MENTIONS-> AirPods
- Here are over 40 last-minute gifts you can still nab before Christmas -MENTIONS-> AirPods
- 200+ of the best deals from Amazon's Cyber Monday sale -MENTIONS-> AirPods

Entities: AirPods, Apple Watch, WIRED, October Prime Day, iPhone, heavy metal brick, MagSafe charging pad, 78 Absolute Best October Prime Day Deals (2023), ESR,
travel kit
</item 6>

<item 7>
Some of our favorite devices are on sale for Black Friday. (The Verge, 2023-11-24T15:42:21+00:00)
But that’s just life while traveling with a dog anyway.

An extension cord for awkward situations

Sarah Jeong, deputy features editor

Anker 321 Power Strip $ 15 $ 26 42 % off $ 15 $ 15 $ 26 42 % off This all-in-one 20W USB-C power cube boasts three AC outlets, two USB-A ports, and one USB-C
port. $15 at Amazon

Nobody wants to carry a power strip or an extension cord with them on their vacation. It’s probably unnecessary if you’re staying in relatively modern buildings
and definitely unnecessary if you’re camping. But sometimes you want to stay in a charming historical hotel or a lovely cabin in the woods, and it’s only when
you go to charge your devices at night that you realize that the only electrical socket in the bedroom is in the corner farthest away from the bed and there’s
already two lamps plugged into it.

Graph facts:
- The best Black Friday deals you can get for under $25 -MENTIONS-> Black Friday
- The best Apple Cyber Monday deals of 2023: Save on AirPods, iPads, Apple Watches and more -MENTIONS-> Black Friday
- The best Apple Black Friday deals on iPads, AirPods, Apple Watches and more -MENTIONS-> Black Friday
- The 53 best Black Friday deals we could find at Amazon, Walmart, Target and more -MENTIONS-> Black Friday
- 200+ of the best deals from Amazon's Cyber Monday sale -MENTIONS-> Black Friday

Entities: Black Friday, Amazon, Some of our favorite devices are on sale for Black Friday., Sarah Jeong, Anker 321 Power Strip, The Verge
</item 7>

<item 8>
78 Absolute Best October Prime Day Deals (2023) (Wired, 2023-10-10T14:23:00+00:00)
This 2.5-inch cube from Anker is a great compact charger. It comes with a MagSafe pad on top (that charges at up to 15 watts), a top section that hinges to a
60-degree angle to reveal a charging surface for your AirPods, and a shelf on the side that has a built-in Apple Watch charger (which can comfortably
accommodate any Apple Watch including the Ultra). You'll also get a 5-foot cable and a 30-watt charger in the box. It supports fast charging too.

Anker 737 Power Bank Photograph: Anker

We just added this power bank as our top upgrade pick in our guide to the Best Portable Chargers. It's pricey, but today's deal makes it more accessible. It
charges from zero to completely full in an hour and boasts a whopping 24,000-mAh capacity. And it's powerful enough to charge laptops and tablets as well as
phones and other gadgets. There's even a built-in display to monitor stats like temperature and remaining battery percentage.

Graph facts:
- The best Apple Cyber Monday deals of 2023: Save on AirPods, iPads, Apple Watches and more -MENTIONS-> AirPods
- The best Apple Black Friday deals on iPads, AirPods, Apple Watches and more -MENTIONS-> AirPods
- The 53 best Black Friday deals we could find at Amazon, Walmart, Target and more -MENTIONS-> AirPods
- Here are over 40 last-minute gifts you can still nab before Christmas -MENTIONS-> AirPods
- 200+ of the best deals from Amazon's Cyber Monday sale -MENTIONS-> AirPods

Entities: AirPods, Apple Watch, WIRED, MagSafe, 2.5-inch cube charger, Apple Watch Ultra, Anker 737 Power Bank, Best Portable Chargers, 78 Absolute Best October
Prime Day Deals (2023), Anker
</item 8>

<item 9>
Some of our favorite devices are on sale for Black Friday. (The Verge, 2023-11-24T15:42:21+00:00)
Universal travel adapter

Victoria Song, senior reviewer

Epicka universal travel adapter $ 20 $ 25 20 % off $ 20 $ 20 $ 25 20 % off Epicka’s universal travel adapter is an all-in-one adapter that includes four
different plugs that cover over 150 countries. $20 at Amazon

In my youth, I forgot to pack plug adapters for international trips one too many times. Buying them once you’ve landed in another country isn’t always easy,
either. And if you’re like me, your relatives in rural Korea don’t always have more than one plug for your American devices — in which case, you’ll have to
share with your six other cousins. Nope. Absolutely not. Which is why I never leave this country without a universal travel adapter.

Basically, it’s six plug adaptors in one. Depending on which one you get, it might come with USB ports so you can charge multiple devices in one outlet.

Graph facts:
- The best Black Friday deals you can get for under $25 -MENTIONS-> Epicka universal travel adapter
- ChatGPT: Everything you need to know about the AI-powered chatbot -MENTIONS-> Korea
- 9 great movies to watch on Tubi for free right now -MENTIONS-> Korea
- 10 scorching-hot discoveries made about the sun in 2023 -MENTIONS-> Korea
- MLB free agency tracker: Live updates on news, rumors, signings and trades after 2023 Winter Meetings -MENTIONS-> Korea

Entities: Epicka universal travel adapter, Korea, Black Friday, Amazon, Victoria Song, Some of our favorite devices are on sale for Black Friday., The Verge
</item 9>

<item 10>
Some of our favorite devices are on sale for Black Friday. (The Verge, 2023-11-24T15:42:21+00:00)
Granted, it’s bulkier than buying one or two specialized adapter plugs, but if you’ve got a multi-continent itinerary, it’s a game-changer. What I like about
this one from Epicka is that it comes with a spare fuse in case things go sideways with voltage.

The only caveat is that, although it says “universal,” it’s technically only the four most common types of plugs. That’ll get you by in most countries, but it’s
not a guarantee in places like Brazil, South Africa, or India. Even so, I’ll take this over price gouging at airport electronics shops or having to take time
out of my schedule to visit a local hardware store.

Lovely lights

Brandon Widder, senior commerce editor

I’m a big fan of ambient lighting, even when I’m 50 miles from the nearest outlet.

Graph facts:
- ChatGPT: Everything you need to know about the AI-powered chatbot -MENTIONS-> Brazil
- Here’s how Rainforest, a budding Stripe rival, aims to win over software companies -MENTIONS-> Brazil
- Watch Argentina vs Paraguay live stream, TV channel, lineups, betting odds for World Cup qualifying match -MENTIONS-> Brazil
- If your startup doesn’t seem impossible at first, it’s not hard enough -MENTIONS-> Brazil
- Taylor Swift is Time’s Person of the Year -MENTIONS-> Brazil

Entities: Brazil, India, South Africa, Black Friday, Some of our favorite devices are on sale for Black Friday., Brandon Widder, Epicka, ambient lighting, The
Verge
</item 10>
=== END ITEM id=392a18c3b46ce9c8 ===
