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

=== ITEM id=1e6d6eaf10ef8f51 ===
Gold sentences and facts:
Sentence 0: The league's leading receiver, Tyreek Hill (vs. Titans in Week 14), stands as the unquestioned WR1 for Week 14 after torching the Commanders to the tune of five catches, 157 yards, and two TDs.
  Fact 0.0: Tyreek Hill is the league's leading receiver.
  Fact 0.1: Tyreek Hill played against the Titans in Week 14.
  Fact 0.2: Tyreek Hill is the unquestioned WR1 for Week 14.
  Fact 0.3: Tyreek Hill torched the Commanders.
  Fact 0.4: Tyreek Hill had five catches against the Commanders.
  Fact 0.5: Tyreek Hill had 157 yards against the Commanders.
  Fact 0.6: Tyreek Hill had two TDs against the Commanders.
  Fact 0.7: Tyreek Hill's WR1 status for Week 14 came after his game against the Commanders.
Sentence 1: 152.67 -- After missing last week's game, Tyreek Hill now needs to average almost 153 yards per game in his final three to reach his stated goal of 2,000 receiving yards this season.
  Fact 1.0: The figure given at the start of the sentence is 152.67.
  Fact 1.1: Tyreek Hill missed last week's game.
  Fact 1.2: Tyreek Hill needs to average almost 153 yards per game.
  Fact 1.3: Tyreek Hill needs to average almost 153 yards per game in his final three games.
  Fact 1.4: Tyreek Hill's stated goal is 2,000 receiving yards this season.
  Fact 1.5: Tyreek Hill needs to reach 2,000 receiving yards this season to meet his goal.
Sentence 2: Of Miami's five remaining opponents, four have pass defenses ranked in the league's top eight.
  Fact 2.0: Miami has five remaining opponents.
  Fact 2.1: Four of Miami's five remaining opponents have pass defenses ranked in the league's top eight.

Retrieved items:
<item 1>
NFL Fantasy Football Week 5 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-04T21:56:25+00:00)
He got open on a handful of third-down plays when he didn't get a target last week.

TANNEHILL: Is the top problem Hopkins has right now. Hopkins' 38-yard deep-ball catch last week could have been as many as 50 yards if Tannehill hit him in
stride, and the gimmick-play touchdown would have obviously been a massive point-getter had Tannehill been slightly more on-target there. There was even a
target into the end zone that was a miscommunication between Hopkins and Tannehill. On the season Tannehill has a 13% off-target rate but over the past two
games that number has ballooned to 18%, which is ... yikes.

TITANS: The second problem Hopkins has right now is a Titans offense that's stayed true to their run-first preference -- at least when they win.
</item 1>

<item 2>
NFL Fantasy Football Week 5 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-04T21:56:25+00:00)
3 wide receiver in the offense behind Justin Jefferson and K.J. Osborn, but even with the donut Addison still has a five-catch lead on Osborn on the season with
68 more yards gained.

THIS WEEK: You can bank on the Vikings reversing course against the Chiefs. Two of the past three offenses against Kansas City wound up throwing at least 71% of
their snaps. The team that didn't? Chicago, who was so badly blown out by halftime that they ran the ball 55% of their second-half snaps just to get out of
Arrowhead.

CHIEFS: Have been stout against receivers all season long, ranking top-five in catch rate allowed (53.4%) and YAC/reception allowed (2.96), and sixth-best in
yards per catch given up (11.1).
</item 2>

<item 3>
Fantasy Football WR PPR Rankings Week 14: Who to start, best sleepers at wide receiver (Sporting News, 2023-12-07T22:11:00+00:00)
Bills

65 Tutu Atwell, Rams @ Ravens

66 Treylon Burks, Titans @ Dolphins

67 Josh Reynolds, Lions @ Bears

68 Nick Westbrook-Ikhine, Titans @ Dolphins

69 Cedric Tillman, Browns vs. Jaguars

70 Marquez Valdes-Scantling, Chiefs vs. Bills

71 Dontayvion Wicks, Packers @ Giants

72 Hunter Renfrow, Raiders vs. Vikings

73 Darnell Mooney, Bears vs. Lions

74 Skyy Moore, Chiefs vs. Bills

75 Kalif Raymond, Lions @ Bears

76 A.T. Perry, Saints vs. Panthers

77 Cedrick Wilson, Dolphins vs. Titans

78 Trenton Irwin, Bengals vs. Colts

79 Michael Gallup, Cowboys vs.
</item 3>

<item 4>
NFL Fantasy Football Week 6 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-12T21:23:16+00:00)
That included in the second half when he saw one target and couldn't catch it. He definitely wasn't shadowed by anyone and was almost never double-teamed,
though there were a couple of routes where a safety was in proximity. It could have had more to do with the Texans attacking the middle of the field against the
Falcons' zone coverage.

SAINTS: Have played the fifth-fewest snaps of zone coverage this season and have ignored tendencies some quarterbacks have had through the early portion of the
season. It's just what they like to do. Stud cornerback Marshon Lattimore has not followed receivers much this season, playing 87% of his snaps at right
cornerback (to the left of the quarterback). Their other primary outside corner, Paulson Adebo, has allowed a 50% catch rate on the season with minimal YAC.
</item 4>

<item 5>
Fantasy Football Week 16 Wide Receiver Preview: Waiver adds, DFS plays, and more (CBSSports.com, 2023-12-19T20:51:32+00:00)
Weekly Breakdown MATCHUP vs DAL MIA -1.5 O/U 51 OPP VS WR 7th PROJ PTS 14.2 WR RNK 16th YTD Stats REC 71 TAR 99 REYDS 964 TD 4 FPTS/G 14.8 We expect Hill to
return this week, which should drive people off of Waddle. But there is at least a chance that Hill won't be 100% and Waddle will still be Tua Tagovailoa's top
target. Waddle has been the quintessential DFS tournament play as of late with three of seven games over 19 PPR Fantasy points, topping 26 twice.

WR Preview Heath's Projections

My full set of Week 16 Fantasy Football projections for every position are now available on SportsLine. Find out which of my favorite plays are projected to
score higher than consensus rankings and which don't live up to their draft hype, at least in Week 16.
</item 5>

<item 6>
Fantasy Football WR PPR Rankings Week 14: Who to start, best sleepers at wide receiver (Sporting News, 2023-12-07T22:11:00+00:00)
WEEK 14 DFS:

DK lineup | FD lineup | Best values | Best stacks

Who are the biggest potential WR busts in Week 14?

It's tough to trust Calvin Ridley (@ Browns), Drake London (vs. Buccaneers), Jordan Addison (@ Raiders), and Jayden Reed (@ Giants) this week. Although Ridley
has produced of late, a brutal matchup against a Browns coverage unit that entered Week 13 allowing the second-fewest fantasy points per game (FPPG) to WRs
raises a red flag, and it's only compounded by the loss of Trevor Lawrence (ankle). London's week-to-week output is way too unpredictable, and with the fantasy
season on the line for some of you, there's no reason to roll London out.

Addison is likely to take a step back with Jefferson back in the fold, while the Packers' receiving situation is too dispersed to put faith into Reed in PPR
formats, though he might get a boost with Christian Watson (hamstring) banged up.
</item 6>

<item 7>
FanDuel Picks Week 15: Best NFL DFS lineup advice for daily fantasy football tournaments (Sporting News, 2023-12-13T20:28:00+00:00)
Nacua checks in at $200 less and is arguably better than Kupp in fantasy at this stage. In a game where every dollar counts, pocketing that extra $200 can
really pay off at another roster spot.

WR Rashee Rice, Chiefs @ Patriots ($6,500). There are plenty of issues with the Chiefs' receivers, but Rice is not one of them. The rookie has been very
impressive through the first 14 weeks of his NFL career and hauled in yet another touchdown on Sunday to post a respectable fantasy showing. He's been targeted
29 times over the past three games and scored twice. We can expect New England will try to take away Travis Kelce, meaning that the opportunities could be
coming even more often for Rice in Week 15.

WEEK 15 STANDARD RANKINGS:

QBs | RBs | WRs | TEs | D/STs | Kickers

TE Chig Okonkwo, Titans vs. Texans ($5,000).
</item 7>

<item 8>
Fantasy Football WR PPR Rankings Week 13: Who to start, best sleepers at wide receiver (Sporting News, 2023-11-29T20:27:00+00:00)
Chiefs

31 Diontae Johnson, Steelers vs. Cardinals

32 Tee Higgins, Bengals @ Jaguars

33 George Pickens, Steelers vs. Cardinals

34 Jerry Jeudy, Broncos @ Texans

35 Tyler Lockett, Seahawks @ Cowboys

36 Drake London, Falcons @ Jets

37 Noah Brown, Texans vs. Broncos

38 Josh Downs, Colts @ Titans

39 Brandin Cooks, Cowboys vs. Seahawks

40 DeVante Parker, Patriots vs. Chargers

41 Jahan Dotson, Commanders vs. Dolphins

42 Jayden Reed, Packers vs. Chiefs

43 Amari Cooper, Browns @ Rams

44 Jaxon Smith-Njigba, Seahawks @ Cowboys

45 Curtis Samuel, Commanders vs. Dolphins

46 Elijah Moore, Browns @ Rams

47 Rondale Moore, Cardinals @ Steelers

48 Tyler Boyd, Bengals vs. Jaguars

49 Robert Woods, Texans vs.
</item 8>

<item 9>
NFL Fantasy Football Week 5 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-04T21:56:25+00:00)
It was also his third game out of four with four or fewer targets.

BOOM/BUST INDEED: Davis has a massive ADOT of 16.8 yards (top-10 among 96 receivers with at least 10 targets this season), but he also has a minuscule target
share of 13.6% and a target per route run rate of 13.4% (86th out of those 96 receivers). He's clearly a role player with a big-time skill-set that isn't leaned
on too much.

JAGUARS: Statistically are top-three in catch rate allowed to wideouts on throws of 15-plus Air Yards (29.4%) with seven passes defensed, but dead-last in yards
per catch (43.8!) and YAC/reception (15.0!) on those passes that are completed. They have done a nice job staying close in coverage on those throws.
</item 9>

<item 10>
NFL Fantasy Football Week 6 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-12T21:23:16+00:00)
UNCATCHABLE: Of the 43 targets Garrett Wilson has seen from Zach Wilson, 14 of them (33%) have been uncatchable.

EAGLES: Have let up a slew of receiving numbers by virtue of seeing so many passes thrown against them (27.4 targets per game just to wide receivers). So
naturally they rank in the bottom five in catches allowed, YAC allowed and touchdowns allowed. But on per-target metrics they're doing all right. A 60.6%
completion rate allowed is 10th-best, 11.4 yards per catch allowed is seventh-best and 4.06 YAC/reception is on par with the league average.

PASS RUSH: As overworked as the Eagles secondary has been, their pass rush has been solid. Against soft O-lines in Washington and L.A., the Eagles have put
pressure on at a rate of 37.6% and have blitzed 33.3% of snaps.
</item 10>

<item 11>
Fantasy Football WR PPR Rankings Week 14: Who to start, best sleepers at wide receiver (Sporting News, 2023-12-07T22:11:00+00:00)
Colts

79 Michael Gallup, Cowboys vs. Eagles

80 Brandon Powell, Vikings @ Raiders

81 Chris Moore, Titans @ Dolphins

82 Rashod Bateman, Ravens vs. Rams
</item 11>

<item 12>
NFL Fantasy Football Week 5 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-04T21:56:25+00:00)
STEVENSON: The Patriots front five has gone through all kinds of issues and as a result have contributed to Stevenson averaging 0.98 yards before contact per
rush this season. That ranks 29th out of 44 qualifying running backs. And if you think that's bad then consider his yards after contact per rush average of
1.75, which is dead-last among those same 44 qualifiers.

EXPLOSIVE RUNS: Stevenson has one (1) explosive carry of 12 or more yards. It was against Miami. It went 12 yards. This is a big change from 2022 when he had 21
explosive runs and an explosive run rate of 10%, which was eighth-best.

RECEIVING: Is averaging 4.0 targets per game and catching 75% of them but is getting just 7.3 yards per catch and has one (1) explosive catch of 16 or more
yards.
</item 12>

<item 13>
N.F.L. Executives on the Biggest Questions Facing Super Bowl Contenders (The New York Times, 2023-10-16T23:58:53+00:00)
Especially when you face a defensive coordinator who says, ‘You show me you can beat me throwing the ball outside the numbers — best of luck, and to do that,
you are going to have to take a five-step drop and I’m going to have Myles Garrett chasing you every play.'”

Advertisement

• Kansas City Chiefs: Is the explosive passing game gone for good?

What I think: The Chiefs have never scored less per game on offense through six games or a full season since Patrick Mahomes became their quarterback. They have
also never enjoyed larger average point differentials (9.8 per game), because their defense is so much better. I think the pass offense will become slightly
more explosive as Mahomes gets reps with his receivers, but I’m not expecting a huge shift.

The Chiefs are averaging 9.5 plays per touchdown drive when starting at least 70 yards from the opponent’s goal line, per TruMedia.
</item 13>

<item 14>
NFL Fantasy Football Week 6 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-12T21:23:16+00:00)
TITANS PASS DEFENSE: Rank in the bottom-five in catch rate allowed to outside wide receivers (75.8%) but handle them well otherwise (12.5 yards per catch, 3.34
YAC/reception are both top-10). They're equally adept versus hitch routes and WR screens, Flowers' two most-run routes. Tennessee has allowed exactly 50% of
pass attempts of 15-plus Air Yards to all wide receivers be completed for a bottom-five yards per catch allowed rate, but they're league average in
YAC/reception and touchdowns allowed on those throws, and the numbers are a little bit better for them if you discount four hiccups in Week 1.
</item 14>

<item 15>
Fantasy Football Week 16 Wide Receiver Preview: Waiver adds, DFS plays, and more (CBSSports.com, 2023-12-19T20:51:32+00:00)
Weekly Breakdown MATCHUP @ CAR GB -5 O/U 36.5 OPP VS WR 12th PROJ PTS 10.6 WR RNK 37th YTD Stats REC 52 TAR 84 REYDS 567 TD 7 FPTS/G 10.8 Jordan Addison WR MIN
Minnesota • #3

Age: 21 • Experience: Rookie Weekly Breakdown MATCHUP vs DET DET -3 O/U 47 OPP VS WR 7th PROJ PTS 13.7 WR RNK 29th YTD Stats REC 62 TAR 90 REYDS 824 TD 9 FPTS/G
14.2 Chris Godwin WR TB Tampa Bay • #14

Age: 27 • Experience: 7 yrs.
</item 15>

<item 16>
Fantasy Football Week 13 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-11-30T22:10:58+00:00)
Weekly Breakdown MATCHUP @ DAL DAL -9 O/U 47.5 OPP VS WR 10th PROJ PTS 10.6 WR RNK 31st YTD Stats REC 54 TAR 79 REYDS 575 TD 4 FPTS/G 12.5 Lockett has
seven-plus targets in four of his past five, but he still doesn't have as many as Metcalf's gotten in those games. He also doesn't see nearly as many deep
throws as Metcalf -- literally two targets in his past two games traveled 15 or more Air Yards. Obviously he has potential to always pop off for a big game like
he did against a tough Browns defense five weeks ago, but recent matchups against the Niners, Rams and Ravens weren't kind to Lockett. They were even less kind
to Geno Smith, and this week shouldn't be much different.
</item 16>

<item 17>
N.F.L. Executives on the Biggest Questions Facing Super Bowl Contenders (The New York Times, 2023-10-16T23:58:53+00:00)
Cover 7 | Monday A daily NFL destination that provides in-depth analysis of football’s biggest stories. Each Monday, Mike Sando breaks down the six most
impactful takeaways from the week.

The San Francisco 49ers entered Week 6 riding a 15-game winning streak and looking unstoppable with a young quarterback, Brock Purdy, who was making the game
look easy. They exited with a 19-17 defeat at Cleveland, injuries to key players and, for the first time this season, a question mark or two.

Advertisement

With the previously unbeaten Philadelphia Eagles also losing ugly Sunday, this edition of the Pick Six column addresses key questions facing Super Bowl
favorites with one-third of the 18-week regular season nearly complete.

So far this season, we’ve seen the Miami Dolphins set NFL records for offense, only to lose by four touchdowns in their lone game against a true contender.
We’ve seen the Kansas City Chiefs win with defense, sometimes in spite of their offense. We’ve seen the Buffalo Bills incur devastating injuries.
</item 17>

<item 18>
Fantasy Football WR PPR Rankings Week 14: Who to start, best sleepers at wide receiver (Sporting News, 2023-12-07T22:11:00+00:00)
As fantasy football owners gear up for the last week of the fantasy regular season (in most leagues), our Week 14 fantasy WR PPR rankings are loaded top to
bottom with just two teams on bye. Rolling out a strong receiving corps in a potential must-win week is essential, and we're here to shore up your start 'em,
sit 'em concerns.

The highlight of our rankings is the long-awaited return of Justin Jefferson, who is (finally) set to suit up for the first time since Week 5. It's easy to
forget, but in Jefferson's four full games this year, he averaged just over eight receptions and 135.8 yards per game while scoring three total touchdowns. Few
receivers can dominate as he can, so we're happy to welcome him back.

Of course, Jefferson's return only helps one owner in each league, so let's break down the rest of the WR rankings to help you finish your season with a win.

Which WRs are on bye in Week 14?
</item 18>

<item 19>
Fantasy Football WR PPR Rankings Week 14: Who to start, best sleepers at wide receiver (Sporting News, 2023-12-07T22:11:00+00:00)
Rams), Jonathan Mingo (@ Saints), and Xavier Gipson (vs. Texans) are our preferred waiver-wire pickups in PPR formats for Week 14.

Moore (vs. Jaguars) just saw 12 targets from Joe Flacco and is in line for another high-usage day with Amari Cooper (concussion) potentially sidelined. Brown (@
Jets) should also see a steady target share after Tank Dell's (leg) season-ending injury.

Beckham Jr. (vs. Rams) is also likely to see increased usage with Mark Andrews (leg, ankle) done for the regular season. Mingo (@ Saints) has seen six-plus
targets in his past four games, as his rapport with Bryce Young is strengthening by the week. Gipson led the Jets in receiving this past week (five receptions,
77 yards), and with New York routinely playing from behind, similar stat lines could be in the works.
</item 19>

<item 20>
NFL Fantasy Football Week 6 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-12T21:23:16+00:00)
His high-value touch share is the icing on the cake and hopefully can lead to him scoring just over 12 PPR points for the second week in a row. I'd take my
chances with him over low-upside running backs like Gus Edwards, Antonio Gibson and Jonathan Taylor (for now he's low-upside).

LAST WEEK: I surmised that Flowers would be targeted more if the Ravens run game couldn't get going. Sure enough, it didn't, and Flowers saw a season-high 11
targets at Pittsburgh. Unfortunately, Flowers underwhelmingly caught just five balls for 73 yards with two drops that would have pushed his yardage over 100 had
he held on to both targets.

AIR YARDS: Flowers was targeted downfield more last week than in any game this season with a 15.27 Average Depth of Target (ADOT). It was the first time he had
a double-digit ADOT and more than six targets.
</item 20>

<item 21>
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
</item 21>

<item 22>
Fantasy Football WR PPR Rankings Week 13: Who to start, best sleepers at wide receiver (Sporting News, 2023-11-29T20:27:00+00:00)
Bowden Jr., alongside A.T. Perry and Keith Kirkwood, are the only healthy Saints WRs right now, and a favorable matchup against Detroit gives each some flex
value in deep leagues. Mingo also draws a plus matchup against a reeling Buccaneers secondary, as a potential negative game script should boost his Week 13
fantasy ceiling.

Tillman is another talented rookie who's getting more run as the season progresses, and although P.J. Walker's struggled under center, Tillman is set up for a
high-usage day against the Rams with Amari Cooper (ribs) banged up.

WEEK 13 DFS:

DK lineup | FD lineup | Best stacks | Best values

Who are the biggest potential WR busts in Week 13?

Beware of Terry McLaurin (vs. Dolphins), Puka Nacua and Tutu Atwell (vs.
</item 22>

<item 23>
Fantasy Football WR PPR Rankings Week 14: Who to start, best sleepers at wide receiver (Sporting News, 2023-12-07T22:11:00+00:00)
Fantasy WR PPR Rankings Week 14: Who to start, sit at wide receiver

Rankings based on full-point PPR scoring formats

Rank Player

1 Tyreek Hill, Dolphins vs. Titans

2 CeeDee Lamb, Cowboys vs. Eagles

3 Amon-Ra St. Brown, Lions @ Bears

4 A.J. Brown, Eagles @ Cowboys

5 Keenan Allen, Chargers vs. Broncos

6 Justin Jefferson, Vikings @ Raiders

7 Stefon Diggs, Bills @ Chiefs

8 Mike Evans, Buccaneers @ Falcons

9 Michael Pittman Jr., Colts @ Bengals

10 Davante Adams, Raiders vs. Vikings

11 Brandon Aiyuk, 49ers vs. Seahawks

12 Nico Collins, Texans @ Jets

13 Ja'Marr Chase, Bengals vs. Colts

14 Cooper Kupp, Rams @ Ravens

15 Chris Olave, Saints vs.
</item 23>

<item 24>
Fantasy Football Week 13 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-11-30T22:10:58+00:00)
Weekly Breakdown MATCHUP @ DAL DAL -9 O/U 47.5 OPP VS WR 17th PROJ PTS 11.1 WR RNK 22nd YTD Stats REC 43 TAR 80 REYDS 678 TD 3 FPTS/G 12.9 Metcalf has at least
nine targets in each of his past three games and in five of his past six. That kind of usage, especially in a game where the Seahawks figure to pass a bunch,
makes him a borderline must-start. But on the 12 targets he's seen that have traveled at least 15 Air Yards over the past four weeks, Metcalf has caught just
three. More alarming: Five of the targets were knocked away by defenders who were right on him when the ball arrived. He's not really zooming past anyone these
days, and he's not winning contested catches regularly either.
</item 24>

<item 25>
Fantasy Football WR PPR Rankings Week 13: Who to start, best sleepers at wide receiver (Sporting News, 2023-11-29T20:27:00+00:00)
Geno Smith's struggles complicate their fantasy prospects, as well.

What are the biggest WR injuries in Week 13?

Entering Week 12, Tee Higgins (hamstring), Noah Brown (knee), Treylon Burks (concussion), and Kadarius Toney (ankle, hip) all failed to suit up due to nagging
injuries. For now, we've placed Higgins and Brown in our Week 13 list and omitted Burks and Toney. We'll also be eyeing the Chargers' Josh Palmer (knee), who's
eligible to come off the IR this week but probably won't suit up against New England.

Last week, the most notable WRs to exit their respective games with injuries include Chris Olave (concussion), Amari Cooper (ribs), Demario Douglas
(concussion), and Rashid Shaheed (thigh). We've left all four WRs out of our initial Week 13 list but will closely monitor their teams' injury reports to gauge
their playing prospects.
</item 25>

<item 26>
Fantasy Football WR PPR Rankings Week 13: Who to start, best sleepers at wide receiver (Sporting News, 2023-11-29T20:27:00+00:00)
A.J. Brown (vs. 49ers) faces a sound San Francisco coverage unit in Sunday's NFC Championship rematch, but his ability to dominate in the red zone still makes
him a top-10 PPR play. Ja'Marr Chase (@ Jaguars) averaged over 20 yards per reception in Week 12, and although the drop-off from Joe Burrow to Jake Browning is
massive, Chase still has WR1 upside given his usage.

Michael Pittman Jr. (@ Texans) faces a Texans' pass defense that's been shredded the past few weeks, so another 100-yard receiving day could be in store. Jaylen
Waddle (@ Commanders) is also looking to post his second consecutive 100-plus yard day, which could come to fruition against a struggling Commanders' coverage
unit. Tank Dell (vs. Broncos) continues to rise up our PPR rankings, rounding out our top 10 as a trusted option of C.J. Stroud's.
</item 26>

<item 27>
NFL Fantasy Football Week 6 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-12T21:23:16+00:00)
PITTS: Continues to harvest a high ADOT -- last week it was 11.36 yards and on the season it's 11.72, the highest among all qualifying tight ends. He also has
the lowest catch rate (56.3%) and worst YAC/reception average (1.33 yards) among all qualifying tight ends. Of his 208 yards, just 24 have come after the catch.

COMMANDERS: Play zone coverage at a slightly lower-than-league-average rate, typically turning to man coverage as soon as games get sideways (like in the second
quarter last week or the fourth quarter against the Bills). Opponents average 32.0 pass attempts per game against them, ninth-fewest in football, and against
tight ends that line up in the slot or out wide like Pitts, the Commanders rank among the top-five in yards per catch, YAC/reception and have yet to allow a
reception of 20-plus yards.
</item 27>

<item 28>
Fantasy Football Week 13 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-11-30T22:10:58+00:00)
I'd expect that to be out the window now -- Flacco wasn't shy about chucking the rock deep with the Jets, attempting passes of 15-plus Air Yards on 20 percent
of his throws, and he had no problem focusing on one primary receiver (Garrett Wilson had 8, 14 and 11 targets in Flacco's starts). The hope is that both Flacco
and Cooper are experienced and capable enough to develop rhythm to salvage the Browns passing game. It's the only thing to buy into here as the Rams have
allowed the 12th fewest Fantasy points to wide receivers on the season and happen to have done well against deep throws against wide receivers (40% completion
rate allowed, two touchdowns, both top six on the season). Cooper's volume should hopefully push him to a decent Fantasy stat line. I'd give him a whirl over
Terry McLaurin, Marquise Brown and Diontae Johnson.
</item 28>

<item 29>
Fantasy Football WR PPR Rankings Week 14: Who to start, best sleepers at wide receiver (Sporting News, 2023-12-07T22:11:00+00:00)
Broncos) caught just five-of-nine targets last week, but as L.A.'s only reliable receiving option, a get-right game feels likely this week. Justin Jefferson (@
Raiders) is set to suit up for the first time since Week 5, and as we alluded to earlier, he's a must-start whenever he's active.

Stefon Diggs (@ Chiefs) hasn't totaled 100-plus receiving yards in any of his past six games, but he has every chance to get back on track against a Chiefs
coverage unit that showed some leaks in Sunday's loss to the Packers.

WEEK 14 FANTASY ADVICE:

Sleepers | Busts | Projections | Start-sit

Who are the best fantasy WR sleepers, waiver pickups in Week 14?

Elijah Moore (vs. Jaguars), Noah Brown (@ Jets), Odell Beckham Jr. (vs. Rams), Jonathan Mingo (@ Saints), and Xavier Gipson (vs.
</item 29>

<item 30>
Fantasy Football Week 16 Wide Receiver Preview: Waiver adds, DFS plays, and more (CBSSports.com, 2023-12-19T20:51:32+00:00)
I'm hoping for a bounce back in Week 16. Dontayvion Wicks WR GB Green Bay • #13

Age: 22 • Experience: Rookie Weekly Breakdown MATCHUP @ CAR GB -5 O/U 37.5 OPP VS WR 22nd WR RNK 35th ROSTERED 15% YTD Stats REC 31 TAR 49 REYDS 491 TD 1 FPTS/G
6.5 If Christian Watson and Jayden Reed remain out then Wicks might just be the best option available. He should certainly be started in more leagues than he is
currently rostered in. His seven targets in Week 15 were a career high and part of a trend. His snap share has increased for three weeks in a row.

WR Preview DFS Plays

Top DFS Play (WR Preview) Projections powered by Sportsline CeeDee Lamb WR DAL Dallas • #88

Age: 24 • Experience: 4 yrs.
</item 30>

<item 31>
Fantasy Football WR PPR Rankings Week 13: Who to start, best sleepers at wide receiver (Sporting News, 2023-11-29T20:27:00+00:00)
Dolphins), Puka Nacua and Tutu Atwell (vs. Browns), Drake London (@ Jets), and Tyler Lockett and Jaxon Smith-Njigba (@ Cowboys) this week.

McLaurin's averaging just 42 receiving yards over his past three games, and a heavy dosage of Jalen Ramsey hinders his Week 13 fantasy value.

Nacua and Atwell see a Cleveland coverage unit who entered Week 12 allowing the fewest fantasy points per game to WRs. Heck, even Cooper Kupp, who hasn't looked
like himself of late, could underperform again.

We can't trust London and the rest of the Falcons passing offense against a well-rounded Jets defense. To make matters worse, he'll be matched up against Sauce
Gardner for most of the day, too.

Lockett and Smith-Njigba might be better suited for the bench in PPR leagues against a lockdown Dallas coverage unit.
</item 31>

<item 32>
Fantasy Football WR PPR Rankings Week 14: Who to start, best sleepers at wide receiver (Sporting News, 2023-12-07T22:11:00+00:00)
The league's leading receiver, Tyreek Hill (vs. Titans in Week 14), stands as the unquestioned WR1 for Week 14 after torching the Commanders to the tune of five
catches, 157 yards, and two TDs. CeeDee Lamb (vs. Eagles) and the Cowboys look to avenge their road loss to the Eagles in Week 9, as Dallas' WR1 routinely has
double-digit reception potential.

Amon-Ra St. Brown (@ Bears) was relatively quiet this past week, but he's likely to rebound against a Bears defense that he totaled nine catches for 95 yards
against in Week 11. A.J. Brown (@ Cowboys) reached the century mark for the first time in four games in Week 13, and another high-usage day looks to be in store
for a potential shootout.

Keenan Allen (vs. Broncos) caught just five-of-nine targets last week, but as L.A.
</item 32>

<item 33>
Fantasy Football TE PPR Rankings Week 14: Who to start, best sleepers at tight end (Sporting News, 2023-12-07T22:14:00+00:00)
Jaguars), and Dalton Kincaid (@ Chiefs) aren't far behind. LaPorta and Njoku both have great matchups against defenses that entered last week in the top 10 in
fantasy points per game (FPPG) allowed to TEs, and we know what Kelce, Hockenson, and Kincaid do every week in PPR formats.

George Kittle (vs. Seahawks) doesn't have a great matchup -- especially when you remember that he had just three catches for 19 yards against Seattle in Week 12
-- but he remains a must-start given his high ceiling in any matchup.

WEEK 14 FANTASY ADVICE:

Sleepers | Busts | Projections | Start-sit

Who are the best fantasy TE sleepers, pickups, & streamers in Week 14?

It can be tough to trust Taysom Hill (vs.
</item 33>

<item 34>
NFL Fantasy Football Week 6 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-12T21:23:16+00:00)
But they float around league average in catch rate allowed (77.8%, below league average) and yards per catch (8.9, better than league average). However, no team
has allowed more first-down receptions to tight ends (17). Keep in mind, this defense hasn't seen tight ends better than Evan Engram, Sam LaPorta or Dalton
Schultz.

COMMANDERS: Throw to their tight ends on 26.3% of their throws, fourth-most in football. But that hasn't always meant a huge target share for Thomas. In Weeks 1
and 5 he saw at least 22.9% of Sam Howell's targets. He left Week 2 after suffering a concussion and was absent in Week 3; then was not part of the game plan
much in Week 4 (7.7% target share against Philadelphia). Perhaps it's all health related and Thomas' volume from last week will stick.
</item 34>

<item 35>
MVPs for all 16 AFC teams at the midpoint (Yardbarker, 2023-11-09T21:51:02+00:00)
He’s third in the NFL in passing yards (2,442), fourth in TD passes (17) and third in QBR (73). The 28-year-old might capture his third MVP, as oddschecker
notes he’s favored to win the award again.

LAS VEGAS RAIDERS | Defensive end Maxx Crosby: Despite a weak supporting cast, Crosby thrives. He has 9.5 of the Raiders’ 24 sacks and is tied for the league
lead in tackles for loss (13). If the two-time Pro Bowler keeps dominating, he will generate more Defensive Player of the Year buzz.

LOS ANGELES CHARGERS | WR Keenan Allen: The 31-year-old is the centerpiece of the offense and leads the team in yards receiving (720) and receptions (62).
Through Week 9, Pro Football Focus gave him a receiving grade of 84.6, eighth among WRs.
</item 35>

<item 36>
Fantasy Football WR PPR Rankings Week 13: Who to start, best sleepers at wide receiver (Sporting News, 2023-11-29T20:27:00+00:00)
Falcons

13 Calvin Ridley, Jaguars vs. Bengals

14 Cooper Kupp, Rams vs. Browns

15 DeVonta Smith, Eagles vs. 49ers

16 DK Metcalf, Seahawks @ Cowboys

17 Chris Olave, Saints vs. Lions

18 Deebo Samuel, 49ers @ Eagles

19 DeAndre Hopkins, Titans vs. Colts

20 Courtland Sutton, Broncos @ Texans

21 Christian Kirk, Jaguars vs. Bengals

22 Terry McLaurin, Commanders vs. Dolphins

23 Nico Collins, Texans vs. Broncos

24 Rashee Rice, Chiefs @ Packers

25 Puka Nacua, Rams vs. Browns

26 Chris Godwin, Buccaneers vs. Panthers

27 Marquise Brown, Cardinals @ Steelers

28 Christian Watson, Packers vs. Chiefs

29 Adam Thielen, Panthers @ Buccaneers

30 Romeo Doubs, Packers vs.
</item 36>

<item 37>
N.F.L. Executives on the Biggest Questions Facing Super Bowl Contenders (The New York Times, 2023-10-16T23:58:53+00:00)
Receiving Yards Through Six NFL Games Rk Player Season Yds 1 Don Hutson 1942 819 2 Tyreek Hill 2023 812 3 Wes Welker 2011 785 4 Elroy Hirsch 1951 766 5 Santana
Moss 2005 743 6 DeAndre Hopkins 2015 726 7 Raymond Berry 1960 725 8 Jerry Rice 1986 723 9 Bob Boyd 1954 718 10 Bobby Mitchell 1962 714

• Jared Goff is enjoying the second-best start to a season for his career from a statistical standpoint, behind only his 2018 Super Bowl season with the Rams.
With the Lions at 5-1 and Goff heading toward a likely contract extension, could another team come after his injured backup, former University of Tennessee
quarterback Hendon Hooker?
</item 37>

<item 38>
Fantasy Football WR PPR Rankings Week 13: Who to start, best sleepers at wide receiver (Sporting News, 2023-11-29T20:27:00+00:00)
Tyreek Hill (@ Commanders in Week 13) did his thing despite a tough matchup in Week 12, resulting in Miami's WR1 returning to the top spot against a putrid
Washington stop unit. CeeDee Lamb (vs. Seahawks) isn't far behind, as Dallas' WR1 aims to maintain his efficient play against Seattle.

Amon-Ra St. Brown (@ Saints) has been a target hog the past several weeks, seeing at least nine looks in each of his past six games. Don't expect his workload
to decrease in a bounce-back game for the Lions. Keenan Allen (@ Patriots) looks to continue his dominant ways, as Justin Herbert's go-to target routinely has
double-digit target potential.

Mike Evans (vs. Panthers) sits in our top five for Week 13, totaling four TDs over the past three weeks. A.J. Brown (vs.
</item 38>

<item 39>
Fantasy Football RB PPR Rankings Week 14: Who to start, best sleepers at running back (Sporting News, 2023-12-07T22:09:00+00:00)
Rams) and think he has the potential to smash Los Angeles at home this week. The speedster has quietly racked up over 300 yards and two TDs over the Ravens'
past four games, and with Lamar Jackson facing Aaron Donald and a still-talented pass-rush, John Harbaugh's squad might turn to the rookie to move the chains
between the 20s. All we need is one or two big chunk gains to make him a worthwhile start — he can run friggin' 22 miles per hour!

WEEK 14 DFS:

DK lineup | FD lineup | Best values | Best stacks

Who are the biggest potential fantasy RB busts in Week 14?

While he just barely cracks our top 25 thanks to his passing-down usage, James Cook (@ Chiefs) doesn't thrill us at Arrowhead against the pissed-off defending
champs.
</item 39>

<item 40>
Fantasy Football WR PPR Rankings Week 13: Who to start, best sleepers at wide receiver (Sporting News, 2023-11-29T20:27:00+00:00)
Stroud's.

WEEK 13 FANTASY ADVICE: Sleepers | Busts | Projections | Start-sit

Who are the best fantasy WR sleepers, waiver pickups in Week 13?

Curtis Samuel (vs. Dolphins), Jayden Reed (vs. Chiefs), Lynn Bowden Jr. (vs. Lions), Jonathan Mingo (@ Buccaneers), Cedric Tillman (@ Rams), and DeVante Parker
(vs. Chargers) profile as this week's crop of sleepers and waiver-wire adds.

Samuel saw a heavy dosage of targets in Week 12, and with Washington passing at a high clip, another high-usage day could be in store. Reed's scored in three
straight games and while he has a higher ceiling in standard formats, he still has flex appeal in a potentially negative game script.

Bowden Jr., alongside A.T.
</item 40>

<item 41>
NFL Fantasy Football Week 6 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-12T21:23:16+00:00)
Jets • #17

Age: 23 • Experience: 2 yrs. Weekly Breakdown MATCHUP vs PHI PHI -7 O/U 41 OPP VS WR 31st PROJ PTS 11.7 WR RNK 29th YTD Stats REC 24 TAR 43 REYDS 279 TD 2
FPTS/G 12.8 Wilson's situation continues to hold him back from the potential he so clearly has. If he couldn't put up a monster game against the Broncos, it's a
tough sell that he'll do so against the Eagles. I do think volume will be in his favor, so counting on him for close to 16 PPR points (half that in non-PPR) is
fair. That gives him flex appeal over Nico Collins and Tyler Lockett, but not over Jakobi Meyers, Zay Flowers or Gabe Davis.
</item 41>

<item 42>
Fantasy Football WR PPR Rankings Week 13: Who to start, best sleepers at wide receiver (Sporting News, 2023-11-29T20:27:00+00:00)
To see Draft Sharks' Week 13 3D floor-to-ceiling projections for wide receivers -- plus get access to all of Draft Sharks' industry-leading tools -- become a
Draft Sharks' member today at the link below!

TRY DRAFT SHARKS FREE: Dominate with exclusive tools, including Free-Agent Finder & Trade Navigator!
</item 42>

<item 43>
Fantasy Football Week 13 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-11-30T22:10:58+00:00)
0-7.00 6.99 - 5.01 5.00 - 0 Sit 'em

The Lions need an answer for their ugly pass defense -- since their Week 9 bye they've been mostly great against the run (3.1 yards per carry allowed, one RB to
get 15-plus PPR points) but they rank bottom-six in completion rate allowed (68.4%), yards per attempt allowed (8.0), passing ADOT against (8.84 yards) and
defensive Expected Points Added (EPA) per dropback (-0.40, which is league-worst). And of their second-most eight passing touchdowns given up in three games,
five have come on passes of 15-plus Air Yards, which is tied for most in the league. They should catch a break with field stretcher Rashid Shaheed (thigh)
missing the matchup for New Orleans; Chris Olave looks like he will play.
</item 43>

<item 44>
Fantasy Football Week 16 Wide Receiver Preview: Waiver adds, DFS plays, and more (CBSSports.com, 2023-12-19T20:51:32+00:00)
Weekly Breakdown MATCHUP @ MIA MIA -1.5 O/U 51 OPP VS WR 15th PROJ PTS 19.4 WR RNK 2nd YTD Stats REC 103 TAR 141 REYDS 1306 TD 10 FPTS/G 21.6 The choice between
Lamb and Tyreek Hill will not be an easy one. The tiebreaker for me is that Lamb is $900 cheaper and Hill may still not be 100% by the time the Cowboys and
Dolphins kick off. Last week was the first time since Week 9 that Lamb failed to reach the end zone and the second time since Week 5 that he didn't score at
least 17 PPR Fantasy points.

Contrarian DFS Play (WR Preview) Projections powered by Sportsline Jaylen Waddle WR MIA Miami • #17

Age: 25 • Experience: 3 yrs.
</item 44>

<item 45>
Fantasy Football Week 16 Wide Receiver Preview: Waiver adds, DFS plays, and more (CBSSports.com, 2023-12-19T20:51:32+00:00)
There will be many difficult decisions to be made in Week 16. It is the Fantasy Football semifinals in most leagues and there is a certain finality to every
decision. There's no shortage of things to stress over as a Fantasy manager. Whether or not you start Rashee Rice shouldn't be one of them. While Rice is just
WR29 on the season, that's mostly due to a slow start, and his hot start has pushed him into must-start, borderline top-12 territory.

The Chiefs' star rookie has four straight games with at least nine targets and he's scored at least 14 PPR Fantasy points in all of those games. Since Week 12,
only Deebo Samuel and CeeDee Lamb have scored more PPR Fantasy points that Rice.

On paper, the matchup doesn't look great, because the Raiders have been above average in terms of limiting the Fantasy production of wide receivers.
</item 45>

<item 46>
NFL Fantasy Football Week 5 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-04T21:56:25+00:00)
MATCHUPS: Each top-target earning wide receiver that plays mostly outside (like McLaurin) has found at least 10 PPR points in every game against the Bears. That
includes Rashee Rice in Week 3 (7 targets, 5-59-0), Courtland Sutton in Week 4 (5 targets, 3-27-1), Romeo Doubs in Week 1 (5 targets, 4-26-2) and Mike Evans in
Week 2 (8 targets, 6-171-1). Note that these receivers combined to catch 18 of 25 targets (72%) against Chicago.
</item 46>

<item 47>
NFL Fantasy Football Week 6 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-12T21:23:16+00:00)
Brissett didn't throw downfield as much as Walker, contributing to a higher ADOT for Walker, but not by much (8.8 to 8.3). Both quarterbacks were right around
6.5 yards per pass attempt.

49ERS: Oh yeah, the matchup isn't great. No team allows fewer yards per catch to wideouts than the Niners (9.7), and they're thrown at a ton by offenses that
have to play from behind. Only three receivers -- Puka Nacua (20 targets), Marquise Brown (10 targets) and Michael Wilson (two touchdowns) have had more than 15
PPR points against them. And only one other receiver -- Tutu Atwell (nine target) -- had more than 12 PPR points.
</item 47>

<item 48>
Fantasy Football Week 13 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-11-30T22:10:58+00:00)
Weekly Breakdown MATCHUP vs CAR TB -5.5 O/U 37 OPP VS WR 15th PROJ PTS 12.5 WR RNK 32nd YTD Stats REC 53 TAR 82 REYDS 606 TD 1 FPTS/G 11.2 Remember when Chris
Godwin was known for being a super-reliable SLOT receiver? Those were the days. This season he's lined up wide on 68% of his snaps. Can't help but think that's
caused the massive change in his production this year -- that and his quarterback isn't a serial slot-targeter (or a Hall of Famer). Godwin's been under 60
yards in five straight, hasn't had a game with even 10 PPR points since Week 8 and has one touchdown on the year despite being tied for 12th among all NFL
players in end-zone targets with nine.
</item 48>

<item 49>
Fantasy Football Week 13 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-11-30T22:10:58+00:00)
In their past three games, they gave up 4.1 yards per rush to the Ravens (just 6.3% of their runs went 10-plus yards), 7.8 yards per rush to the Steelers thanks
to a long TD run by Jaylen Warren (23.8% of their runs went 10-plus yards) and 4.6 yards per rush to the Broncos (18.5% of their runs went 10-plus yards).
Playing more snaps because the Browns offense has been inept isn't the reason for it either because they struggled with the run in the first half against all
three teams (over 5.3 yards per carry allowed against each). It's silly to say the Rams passing game is struggling when Matthew Stafford just threw four
touchdowns, but Cooper Kupp looked like he played hurt last week and Puka Nacua didn't look that much better.
</item 49>

<item 50>
NFL DFS picks Week 15: Best sleepers, value players for FanDuel, DraftKings, SuperDraft lineups (Sporting News, 2023-12-13T20:32:00+00:00)
6x); Tyler Allgeier, Falcons at Panthers (DK: $4,400 | FD: $5,700 | SD: 1.6x)

WEEK 15 PPR RANKINGS:

QBs | RBs | WRs | TEs | D/STs | Kickers

NFL DFS Picks Week 15: WR sleepers, values for DraftKings, FanDuel, SuperDraft

Rashee Rice, Chiefs at Patriots (DK: $6,100 | FD: $6,500 | SD: 1.35x)

With Isiah Pacheco (shoulder) banged up, Rice seems like Kansas City's best skill-position player this side of Travis Kelce. He's certainly the most sure-handed
wideout on the Chiefs (not saying much, but still), and he has electric after-the-catch playmaking ability (which is good, considering Patrick Mahomes has
struggled with deep-ball accuracy).
</item 50>

<item 51>
Fantasy Football WR PPR Rankings Week 14: Who to start, best sleepers at wide receiver (Sporting News, 2023-12-07T22:11:00+00:00)
Bills

33 Tee Higgins, Bengals vs. Colts

34 Jerry Jeudy, Broncos @ Chargers

35 Brandin Cooks, Cowboys vs. Eagles

36 Diontae Johnson, Steelers vs. Patriots

37 Elijah Moore, Browns vs. Jaguars

38 Adam Thielen, Panthers @ Saints

39 Tyler Lockett, Seahawks @ 49ers

40 Josh Downs, Colts @ Bengals

41 George Pickens, Steelers vs. Patriots

42 Jaxon Smith-Njigba, Seahawks @ 49ers

43 Drake London, Falcons vs. Buccaneers

44 Zay Jones, Jaguars @ Browns

45 Jordan Addison, Vikings @ Raiders

46 Jayden Reed, Packers @ Giants

47 Robert Woods, Texans @ Jets

48 Gabe Davis, Bills @ Chiefs

49 Tyler Boyd, Bengals vs. Colts

50 Odell Beckham Jr., Ravens vs.
</item 51>

<item 52>
NFL Fantasy Football Week 6 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-12T21:23:16+00:00)
COLLINS: It's minor but Collins does see a higher target per route run rate against man coverage (25%) than zone (22.2%). He also has a higher catch rate
(87.5%) and higher receiving average (23.3) against man than zone (65.4% and 17.8). However, he's run nearly four times as many routes against zone coverage
compared to man.

MATCHUPS: It's also worth pointing out that Collins' two really big games came in Week 2 (vs. Indianapolis) and Week 4 (vs. Pittsburgh), both teams that rank in
the bottom-10 in Fantasy points allowed to wide receivers. The Saints are in the top half.

STROUD: Has been outstanding against man coverage all year, completing a higher rate of passes (67.6%) for 3.5 more yards per attempt (11.1) than versus zone
(7.6).
</item 52>

<item 53>
Fantasy Football WR PPR Rankings Week 14: Who to start, best sleepers at wide receiver (Sporting News, 2023-12-07T22:11:00+00:00)
Panthers

16 DK Metcalf, Seahawks @ 49ers

17 Garrett Wilson, Jets vs. Texans

18 DeVonta Smith, Eagles @ Cowboys

19 DJ Moore, Bears vs. Lions

20 Puka Nacua, Rams @ Ravens

21 Deebo Samuel, 49ers vs. Seahawks

22 Jaylen Waddle, Dolphins vs. Titans

23 DeAndre Hopkins, Titans @ Dolphins

24 Amari Cooper, Browns vs. Jaguars

25 Calvin Ridley, Jaguars @ Browns

26 Zay Flowers, Ravens vs. Rams

27 Noah Brown, Texans @ Jets

28 Courtland Sutton, Broncos @ Chargers

29 Romeo Doubs, Packers @ Giants

30 Jakobi Meyers, Raiders vs. Vikings

31 Chris Godwin, Buccaneers @ Falcons

32 Rashee Rice, Chiefs vs. Bills

33 Tee Higgins, Bengals vs.
</item 53>

<item 54>
FanDuel Picks Week 15: Best NFL DFS lineup advice for daily fantasy football tournaments (Sporting News, 2023-12-13T20:28:00+00:00)
He shouldn't have any problems continuing that run against the Cardinals on Sunday.

WEEK 15 PPR RANKINGS:

QBs | RBs | WRs | TEs | D/STs | Kickers

WR Puka Nacua, Rams vs. Commanders ($8,000). Usually, one of Nacua or Kupp has a big day, but this week it could both. Either way, we're on the Nacua train
heading into Sunday since the rookie has been by far the L.A.'s most consistent weapon all season. Even with Kupp back in the fold, Stafford continues to trust
Nacua. He's not seeing the 15-20 targets a game like earlier in the year, but the rookie is making the most of his seven-plus opportunities a game. Against a
terrible defense, both receivers should see plenty of chances to make their mark on the game.
</item 54>

<item 55>
MVPs for all 16 AFC teams at the midpoint (Yardbarker, 2023-11-09T21:51:02+00:00)
MIAMI DOLPHINS | WR Tyreek Hill: Arguably the best wide receiver in the NFL, Hill has somehow reached a new level of dominance in his age-29 season. The
seven-time Pro Bowler leads the NFL in receiving yards (1,076) and TD catches (eight). As long as the Dolphins maintain their success on offense down the
stretch, Hill should earn Offensive Player of the Year honors. He's on pace to break Calvin Johnson's season record for receiving yards (1,964).



NEW YORK JETS | LB Quincy Williams: While his brother, Quinnen, may be the better-known Jets defender in the family, Williams — the AFC Defensive Player of the
Month for October — is also making a name for himself. Aside from being the fifth-highest-rated linebacker in the NFL by Pro Football Focus, Williams is tied
for 17th in tackles (74) and tied for ninth in TFLs (nine) among all defenders.
</item 55>

<item 56>
Fantasy Football Week 13 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-11-30T22:10:58+00:00)
Weekly Breakdown MATCHUP @ HOU HOU -3.5 O/U 47 OPP VS QB 10th PROJ PTS 17.8 QB RNK 10th YTD Stats PAYDS 2199 RUYDS 266 TD 21 INT 4 FPTS/G 20.4 There's no
denying that he is now among the most efficient quarterbacks in the league. He's top five among qualifying quarterbacks in completion rate at 68.3%, he's first
in TD rate at 6.3%, he's fourth in off-target rate (7.8%) and he's clearly worked out his issues with his offensive line because he's landed the fourth-longest
time-to-throw average (3.14 seconds). And yet he's done this without firing downfield a ton (32nd in passing ADOT at 6.7 yards)!
</item 56>

<item 57>
Fantasy Football Week 16 Wide Receiver Preview: Waiver adds, DFS plays, and more (CBSSports.com, 2023-12-19T20:51:32+00:00)
Weekly Breakdown MATCHUP vs JAC TB -1 O/U 43 OPP VS WR 25th PROJ PTS 13.4 WR RNK 24th YTD Stats REC 68 TAR 108 REYDS 814 TD 2 FPTS/G 11.9 Calvin Ridley WR JAC
Jacksonville

Age: 29 • Experience: 5 yrs. Weekly Breakdown MATCHUP @ TB TB -1 O/U 43 OPP VS WR 16th PROJ PTS 11.6 WR RNK 26th YTD Stats REC 60 TAR 109 REYDS 781 TD 5 FPTS/G
12.4

WR Preview Waiver Wire Targets

Week 16 Adds (WR Preview) Curtis Samuel WR WAS Washington • #4

Age: 27 • Experience: 7 yrs.
</item 57>

<item 58>
NFL Fantasy Football Week 6 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-12T21:23:16+00:00)
But I would rather start Christian Kirk, Marquise Brown, Michael Pittman and Jakobi Meyers as WR2 options this week.

Sneaky Sleeper in PPR (Lineup Decisions) Projections powered by Sportsline Justice Hill RB BAL Baltimore • #43

Age: 26 • Experience: 5 yrs. Weekly Breakdown MATCHUP @ TEN BAL -4 O/U 41 OPP VS RB 14th PROJ PTS 7.5 RB RNK 29th YTD Stats RUYDS 115 REC 7 REYDS 25 TD 3 FPTS/G
9.3 It's a little gutsy to go with Hill, who has been inefficient for much of the year. We're counting on the Ravens outsmarting the big size of the Titans
defensive line and utilizing Hill for his speed along the edges and in the passing game.
</item 58>

<item 59>
Christmas Day preview: 49ers, Ravens square off in potential Super Bowl sneak peek (Yardbarker, 2023-12-24T23:34:39+00:00)
McCaffrey, who seems poised to win the Offensive Player of the Year Award, leads the NFL in rushing (1,292 yards) and total yards per game (128.6) this season
and is tied for first in touchdowns (20).

Ravens TE Isaiah Likely thriving in the starting role: The Ravens offense hasn't lost a step since losing one of the team's top weapons in tight end Mark
Andrews (ankle) in Week 11, largely due to Likely's emergence.

The second-year tight end has averaged 64.3 yards over his last three games, which is particularly noteworthy since Andrews averaged 54.4 yards per game before
sustaining his injury.

Should we expect a defensive battle?: While both teams boast offenses that rank in the top five in yardage and scoring, this may not turn into a shootout.
</item 59>

<item 60>
Is the Coverage on Taylor Swift and Travis Kelce’s Romance Nothing but Hype or a Pivotal Shift in NFL’s Century-Old Game Plan Towards Global Recognition?
(Essentially Sports, 2023-10-25T21:30:41+00:00)
From Colin Kaepernick taking a knee during the national anthem to protest against racism, to domestic violence by players, to the Deflategate and Bountygate
scandals. But there have been bigger crises as well. The Union of Concerned Scientists accused the NFL of trying to intimidate scientists studying the link
between CTE and pro football. The NFL is already way past other leagues in terms of revenue, and now with the pivotal shift in its strategy, “America’s favorite
pastime” is on its way to world domination.

ADVERTISEMENT Article continues below this ad

Editorial Disclaimer: The views expressed are those of the author and do not necessarily reflect the views of EssentiallySports.

Watch This Story: Will Patrick Mahomes Eclipse Tom Brady as the NFL Goat? Travis Kelce Responds
</item 60>

<item 61>
Christmas Day preview: 49ers, Ravens square off in potential Super Bowl sneak peek (Yardbarker, 2023-12-24T23:34:39+00:00)
Can Raiders QB Aidan O’Connell earn next season’s starting job?: After leading the offense on seven touchdown scoring drives in last week's 63-21 win over the
Chargers, the rookie fourth-rounder aims to further establish himself as a serious contender for Las Vegas' starting job next season. Despite the encouraging
performance, O'Connell has been inconsistent across six games as the full-time starter, throwing for 1,300 yards, seven touchdowns, and five interceptions.

However, a stellar showing against the NFL's fifth-ranked pass defense (202.5 yards per game allowed) would benefit his bid to be the starter in 2024.

Chiefs TE Travis Kelce: Yes, his girlfriend will be in town for the holiday. Kelce is averaging 89.6 yards per game with global popstar Taylor Swift in
attendance this season but is coming off his worst outing in her presence, catching five passes for 28 yards.
</item 61>

<item 62>
Fantasy Football Week 13 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-11-30T22:10:58+00:00)
The fact that he has 13 total targets in his past three games combined makes it tough to buy into him suddenly getting 10-plus targets this week. The matchup is
also a problem -- the Browns have the defensive backs to hang tight with a slower-than-normal Kupp in man-to-man coverage, which they play in a lot. The Browns
also have the pass rush to force Matthew Stafford to get the ball out quickly and without much distance. I'm not that nervous to bench Kupp for Rashee Rice,
Courtland Sutton and both of the top Seattle WRs.
</item 62>

<item 63>
Fantasy Football Week 13 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-11-30T22:10:58+00:00)
Tua Tagovailoa, Terry McLaurin (low-end WR2), De'Von Achane (low-end RB2), Dolphins DST

Tyreek Hill, Jaylen Waddle, Raheem Mostert, Tua Tagovailoa, Terry McLaurin (low-end WR2), De'Von Achane (low-end RB2), Dolphins DST SITS: Sam Howell (low-end
starter at best), Brian Robinson (desperation RB starter), Logan Thomas (PPR bye-week replacement), Antonio Gibson (PPR bye-week replacement), Commanders DST

Sam Howell (low-end starter at best), Brian Robinson (desperation RB starter), Logan Thomas (PPR bye-week replacement), Antonio Gibson (PPR bye-week
replacement), Commanders DST SLEEPERS: Curtis Samuel (especially PPR), Jahan Dotson (non-PPR only)

Start 'em 10.0-7.00 6.99 - 5.01 5.
</item 63>

<item 64>
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
</item 64>

<item 65>
Fantasy Football WR PPR Rankings Week 14: Who to start, best sleepers at wide receiver (Sporting News, 2023-12-07T22:11:00+00:00)
What are the biggest WR injuries in Week 14?

In Week 13, Demario Douglas (concussion) and Rashid Shaheed (thigh) missed game action, and Tank Dell (leg), Amari Cooper (concussion), Christian Watson
(hamstring), and Marquise Brown (heel) exited early from their respective games.

UPDATE: Christian Kirk suffered a groin injury on Monday night, and he figures to be more doubtful than questionable for this week's game against Cleveland. If
he's out, Zay Jones and Tim Jones will see more targets, but with Trevor Lawrence (ankle) also hurt, it's tough to count on either.

UPDATE: Douglas (concussion) has been ruled OUT for Week 14.

As usual, we'll update these WR PPR rankings throughout the week, so check back for the latest player movement and analysis.
</item 65>

<item 66>
Fantasy Football Week 16 Wide Receiver Preview: Waiver adds, DFS plays, and more (CBSSports.com, 2023-12-19T20:51:32+00:00)
The Texans offense looks like one to avoid as long as C.J. Stroud is out. Ja'Marr Chase WR CIN Cincinnati • #1

Age: 23 • Experience: 3 yrs. Tee Higgins is a No. 2 receiver and Tyler Boyd is a fine flex.

WR Preview Numbers to Know

69.9% -- Nearly 70% of Rashee Rice's receiving yards have come after the catch, that's tops amongst wide receivers with at least 50 targets.

-- Nearly 70% of Rashee Rice's receiving yards have come after the catch, that's tops amongst wide receivers with at least 50 targets. 11.4 -- Deebo Samuel is
on pace to become just the fourth player since 1992 to average more than 11 yards per target with at least 60 targets in multiple seasons. The other three are
Tyreek Hill, Tyler Lockett, and DeSean Jackson.
</item 66>

<item 67>
Fantasy Football WR PPR Rankings Week 14: Who to start, best sleepers at wide receiver (Sporting News, 2023-12-07T22:11:00+00:00)
Which WRs are on bye in Week 14?

The final set of bye weeks sees only the Commanders and Cardinals idle, resulting in the losses of Terry McLaurin, Marquise Brown, Jahan Dotson, Curtis Samuel,
Rondale Moore, and Greg Dortch.

WEEK 14 PPR RANKINGS:

QBs | RBs | TEs | D/STs | Kickers

Even worse, Washington and Arizona sit as two of the weakest pass defenses in the league, so fantasy owners won't be able to count on big stats from their
receivers going against the Commanders and Cardinals.

WEEK 14 STANDARD RANKINGS:

QBs | RBs | WRs | TEs | D/STs | Kickers

Who are the best fantasy WRs in Week 14?

The league's leading receiver, Tyreek Hill (vs.
</item 67>

<item 68>
Fantasy Football RB Rankings Week 14: Who to start, best sleepers at running back (Sporting News, 2023-12-07T22:07:00+00:00)
Good luck, check back often, and enjoy the remainder of the first week of December!

Which RBs are on bye in Week 14?

After a six-team byepocalypse in Week 13, only the Cardinals and Commanders will be off this coming weekend. That takes lead backs Brian Robinson Jr. and James
Conner off the table, plus Antonio Gibson for those in deep leagues.

WEEK 14 STANDARD RANKINGS:

QBs | WRs | TEs | D/STs | Kickers

One thing that stinks for fantasy is that both Washington and Arizona have been defensive turnstiles. Many of the stud running backs have tough matchups in Week
14, but it's only right that many of the elite backs must beat the best defenses to lift fantasy squads to the big dance.

WEEK 14 PPR RANKINGS:

QBs | RBs | WRs | TEs | D/STs | Kickers

Who are the best fantasy RBs in Week 14?
</item 68>

<item 69>
Fantasy Football Week 13 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-11-30T22:10:58+00:00)
Steve Spagnuolo's defense has been blitzing its tails off over the past two weeks (nearly 50% of their snaps against the Eagles and Raiders), so the hunch is
they'll apply maximum pressure to that O-line in an effort to make Love more off-target than he's been and, hopefully, get some turnovers. Only one receiver has
had more than 15 PPR points against the Chiefs in their past four games (Jakobi Meyers last week), making it tough to expect grand games from Love's favorite
targets. I'd expect shutdown cornerback L'Jarius Sneed to line up across from Christian Watson the most -- and Jayden Reed the least.
</item 69>

<item 70>
Fantasy Football WR PPR Rankings Week 13: Who to start, best sleepers at wide receiver (Sporting News, 2023-11-29T20:27:00+00:00)
Seahawks

66 JuJu Smith-Schuster, Patriots vs. Chargers

67 Kalif Raymond, Lions @ Saints

68 DJ Chark, Panthers @ Buccaneers

69 Skyy Moore, Chiefs @ Packers

70 Dontayvion Wicks, Packers vs. Chiefs

71 Trenton Irwin, Bengals @ Jaguars

72 Nick Westbrook-Ikhine, Titans vs. Colts

73 Cedrick Wilson, Dolphins @ Commanders

74 Marquez Valdes-Scantling, Chiefs @ Packers

75 Alec Pierce, Colts @ Titans

76 Chris Moore, Titans vs. Colts

Week 13 WR PPR fantasy projections, rankings from Draft Sharks

Draft Sharks projections for the new week will update every Tuesday and be constantly adjusted throughout the week

We're proud to offer SN Fantasy readers an exclusive look at Draft Sharks’ award-winning projections and rankings every week.
</item 70>

<item 71>
2023 NFL Week 11 injury roundup: Rams WR Cooper Kupp, Seahawks RB Kenneth Walker III among stars to exit (CBSSports.com, 2023-11-19T23:51:49+00:00)
I agree to receive the "Pick Six Newsletter" and marketing communications, updates, special offers (including partner offers), and other information from CBS
Sports and the Paramount family of companies. By pressing sign up, I confirm that I have read and agree to the Terms of Use and acknowledge Paramount's Privacy
Policy See All Newsletters Please check the opt-in box to acknowledge that you would like to subscribe. Thanks for signing up! Keep an eye on your inbox. Sorry!
There was an error processing your subscription.

WR

Tyreek Hill (Dolphins) briefly left Miami's game against the Raiders while nursing a hand injury and was ruled questionable to return, but he took the field
again soon afterward, securing seven catches for 80 yards and a score before halftime. Hill has left multiple games this year, only to return in all of them.

Cooper Kupp (Rams) had just a single catch against the Seahawks before suffering an ankle injury while blocking.
</item 71>

<item 72>
Fantasy Football Week 13 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-11-30T22:10:58+00:00)
The Commanders offensive line has had some ugly showings lately against the Cowboys (four sacks, 49% pass rush pressure rate allowed), the Giants (four sacks,
32.7% pressure rate allowed) and the Seahawks (three sacks, 46.8% pressure rate allowed). Somehow these results are better than what they were allowing through
much of the first two months of the year. Sam Howell getting the ball out faster has been a positive in this regard but a negative for his numbers in three of
his past four games (the one game he did really well in was sort of fluky). The matchup is a good test for the Dolphins pass rush post-Phillips, one that it
should pass without giving up much in the way of deep completions.

STARTS: Tyreek Hill, Jaylen Waddle, Raheem Mostert, Tua Tagovailoa, Terry McLaurin (low-end WR2),
</item 72>

<item 73>
NFL Fantasy Football Week 5 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-04T21:56:25+00:00)
Wilson ranks top-10 among quarterbacks in completion rate (50%) and yards per attempt (17.2) with the second-lowest off-target rate (12.5%) on those long throws
but is below league-average in overall attempts. His numbers are also inflated because of that Commanders game and the Hail Mary he hit on.

VS. OUTSIDE WRs: The Jets allow the second-highest catch rate to perimeter guys (79.4%), but they're top-five in yards per catch (10.7), ADOT (10.38) and
completions allowed of 20-plus yards (three, which is best in NFL). Outside guys have also seen the third-fewest targets because teams know not to attack there.

WILSON: 50 of his 132 passes (38%) have gone to outside receivers this year. Seven of his nine touchdowns have headed that way too. This is a problem for
Wilson.
</item 73>

<item 74>
Fantasy Football Week 16 Wide Receiver Preview: Waiver adds, DFS plays, and more (CBSSports.com, 2023-12-19T20:51:32+00:00)
Weekly Breakdown MATCHUP vs IND ATL -1 O/U 44.5 OPP VS WR 9th PROJ PTS 10.7 WR RNK 19th YTD Stats REC 58 TAR 86 REYDS 769 TD 2 FPTS/G 11.5 Amari Cooper WR CLE
Cleveland • #2

Age: 29 • Experience: 9 yrs. Weekly Breakdown MATCHUP @ HOU HOU -2.5 O/U 42.5 OPP VS WR 21st PROJ PTS 13.6 WR RNK 20th YTD Stats REC 61 TAR 113 REYDS 985 TD 3
FPTS/G 12.5 Romeo Doubs WR GB Green Bay • #87

Age: 23 • Experience: 2 yrs.
</item 74>

<item 75>
NFL Fantasy Football Week 6 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-12T21:23:16+00:00)
STEVENSON: Nothing has changed in terms of his efficiency. We can blame the offensive line to some degree but Stevenson has averaged 0.97 yards before contact
per rush this season. That ranks 30th out of 41 qualifying running backs. And if you think that's bad then consider his yards after contact per rush average of
1.79, which is dead last among those same 41 qualifiers.

EXPLOSIVE RUNS: Stevenson has one (1) explosive carry of 12 or more yards. It was against Miami. It went 12 yards. This is a big change from 2022 when he had 21
explosive runs and an explosive run rate of 10%, which was eighth-best.

RECEIVING: Is averaging 3.6 targets per game and catching 67% of them but is getting just 7.3 yards per catch and has one (1) explosive catch of 16 or more
yards.
</item 75>

<item 76>
Fantasy Football Week 16 Wide Receiver Preview: Waiver adds, DFS plays, and more (CBSSports.com, 2023-12-19T20:51:32+00:00)
Weekly Breakdown MATCHUP @ NYJ NYJ -3 O/U 37 OPP VS WR 14th WR RNK 30th ROSTERED 58% YTD Stats REC 56 TAR 77 REYDS 549 TD 5 FPTS/G 11 Samuel was the top waiver
wire add at wide receiver last week as well, but he still sits at just 57% rostered. He has three straight games with double-digit Fantasy points and he's
topped 19 in two of those games. The matchup isn't great but it's easier for him than it is for Terry McLaurin and Jahan Dotson.
</item 76>

<item 77>
Fantasy Football WR PPR Rankings Week 13: Who to start, best sleepers at wide receiver (Sporting News, 2023-11-29T20:27:00+00:00)
With the Ravens, Bills, Vikings, Raiders, Giants, and Bears idle in Week 13, that means Davante Adams, Stefon Diggs, Justin Jefferson, DJ Moore, Zay Flowers,
Jakobi Meyers, and Jordan Addison are among the most notable wideouts omitted from our rankings. Deep-league owners will also be without Odell Beckham Jr., K.J.
Osborn, Gabe Davis, Jalin Hyatt, and Khalil Shakir.

WEEK 13 PPR RANKINGS:

QBs | RBs | TEs | D/STs | Kickers

Unfortunately, we also won't get to pick on the poor WR defenses of the Giants and Vikings, but it's a relief to see the Ravens' shut-down secondary on the
sidelines.

WEEK 13 STANDARD RANKINGS:

QBs | RBs | WRs | TEs | D/STs | Kickers

Who are the best fantasy WRs in Week 13?
</item 77>

<item 78>
NFL Fantasy Football Week 5 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-04T21:56:25+00:00)
I think he'll continue to see high-value targets near the end zone and, hopefully, a small uptick in targets. I'd start him as a No. 2 receiver over Terry
McLaurin, any Broncos wideouts and any Packers wideouts.

LAST WEEK: Was so close to having a monster game. He was the designed target on a trick play and saw the ball fall just past his outstretched hands in the end
zone. He also had two targets inside the 10-yard line (his first two of the season) and two end-zone targets including the aforementioned trick play (his first
two of the season). The heavy dose of targets inside the 10 and into the end zone are positive signs that the Titans are trying to get him going as a playmaker.

FILM: Hopkins might be 31 years old but he moved really, really well, creating separation with outstanding footwork and having just enough speed to stay a step
ahead of defenders.
</item 78>

<item 79>
N.F.L. Executives on the Biggest Questions Facing Super Bowl Contenders (The New York Times, 2023-10-16T23:58:53+00:00)
Six of the 17 won anyway. The Dolphins were the only one of those 223 teams to win and cover the spread, which was 13.5 for this game.

• Six catches for 163 yards and a touchdown against Carolina left the Dolphins’ Tyreek Hill with 812 yards through the first six games. That ranks second in NFL
history to Hall of Famer Don Hutson, who had 819 yards through six with Green Bay in 1942.
</item 79>

<item 80>
Fantasy Football RB Rankings Week 14: Who to start, best sleepers at running back (Sporting News, 2023-12-07T22:07:00+00:00)
It's a tough matchup, so temper your expectations, but there's some high-floor flex appeal considering Alvin Kamara and Taysom Hill averaged 5.2 yards per touch
and found the end zone three times via the ground against Detroit last week.

WEEK 14 DFS:

DK lineup | FD lineup | Best values | Best stacks

Who are the biggest potential RB busts in Week 14?

We're a tad worried about Buffalo second-year back James Cook (@ Chiefs), as Kansas City's front-seven tends to be stout against speedsters but below-average
against power backs. You probably aren't in a position to bench Cook, but don't expect massive numbers from the guy who averaged just 2.7 yards per carry
against the Eagles his last time out.

Zach Charbonnet (@ 49ers) faces a tough matchup and has a bruise on his knee, so we don't expect very fruitful results from the rookie if he starts in Santa
Clara this weekend.
</item 80>

<item 81>
Fantasy Football WR PPR Rankings Week 13: Who to start, best sleepers at wide receiver (Sporting News, 2023-11-29T20:27:00+00:00)
In the final weeks of the fantasy football regular season, rolling out a strong receiver corps comprised of wideouts set up for high usage days is every fantasy
owner's goal. Having those WRs in plus matchups makes it even more likely you'll end the week in the win column. Our Week 13 fantasy WR PPR rankings can help
pinpoint all the studs and sleepers at fantasy's deepest position.

There aren't as many studs or sleepers this week, as six byes and some inopportune injuries thin out our rankings. Nonetheless, we still have to fill out the
same number of starting spots in our lineups, so those in 12-team leagues might have to dig deep to find decent WR3s and flexes. That's where we come in.

Now that we know the backdrop to this pivotal week, let's break down our Week 13 WR rankings.

MORE: Watch NFL RedZone live with Sling (50% off first month)

Which WRs are on bye in Week 13?
</item 81>

<item 82>
FanDuel Picks Week 14: Best NFL DFS lineup advice for daily fantasy football tournaments (Sporting News, 2023-12-07T22:17:00+00:00)
The incredible target share gives us no reason to think that won't continue.

WEEK 14 PPR RANKINGS:

QBs | RBs | WRs | TEs | D/STs | Kickers

WR Michael Pittman Jr., Colts @ Bengals ($8,000). Imagine saying back in the summer that the Minshew-Pittman stack would be one of the best in fantasy this
season. It's a wild reality, but also one that continues to fly under the radar more often than not. Pittman is coming off back-to-back games of at least 100
yards, and he just put up a season-high 16 against the Titans. That volume doesn't grow on trees, and there are no signs of the WR slowing down anytime soon as
the Colts push for a playoff spot.

WR Deebo Samuel, 49ers vs. Seahawks ($7,500).
</item 82>

<item 83>
N.F.L. Executives on the Biggest Questions Facing Super Bowl Contenders (The New York Times, 2023-10-16T23:58:53+00:00)
That figure was 7.8 in Mahomes’ first year as the starter and has climbed every season since. It’s a testament to the Chiefs’ adaptability, but also a
reflection of plummeting explosive play rates.

Exec comment: “Yeah, it looks harder for them on offense until Mahomes finds (Travis) Kelce on every major third down, every major two-minute play, every major
touchdown. Kansas City has the luxury of their best players being their toughest players. Chris Jones is a very tough player. Kelce is a very tough player.
Mahomes is a very tough player. You wonder why they win the games? Because the games are won by the toughest teams most of the time.”

• Philadelphia Eagles: Has Jalen Hurts regressed? Where are the explosive pass plays?

What I think: Explosive passing is down through most of the league, including for the Eagles.
</item 83>

<item 84>
NFL Fantasy Football Week 5 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-04T21:56:25+00:00)
That one went for 32 yards against the Eagles in Week 1. Since then his longest reception has been for seven yards.

SAINTS: Are sixth-best in rushing average to RBs (3.6) and are top-10 in basically every single run defense metric you could think of, including missed tackles
with just nine on the year. The only rusher to exceed 10 Fantasy points against them, PPR or not, was Derrick Henry in Week 1. He had 13 PPR points. No one's
run for a touchdown on them either.

SAINTS: Aren't quite as impressive against running backs through the air, but they're not awful. Figure them as league average across the board except in
YAC/reception (8.53), but that was due to a massive catch-and-run by Henry in Week 1.
</item 84>

<item 85>
Fantasy Football WR PPR Rankings Week 13: Who to start, best sleepers at wide receiver (Sporting News, 2023-11-29T20:27:00+00:00)
As usual, we'll update these WR PPR rankings throughout the week, so check back for the latest player movement and analysis.

Fantasy WR PPR Rankings Week 13: Who to start, sit at wide receiver

Rankings based on PPR scoring formats

Rank Player

1 Tyreek Hill, Dolphins @ Commanders

2 CeeDee Lamb, Cowboys vs. Seahawks

3 Amon-Ra St. Brown, Lions @ Saints

4 Keenan Allen, Chargers @ Patriots

5 Mike Evans, Buccaneers vs. Panthers

6 A.J. Brown, Eagles vs. 49ers

7 Michael Pittman Jr., Colts @ Titans

8 Ja'Marr Chase, Bengals @ Jaguars

9 Jaylen Waddle, Dolphins @ Commanders

10 Tank Dell, Texans vs. Broncos

11 Brandon Aiyuk, 49ers @ Eagles

12 Garrett Wilson, Jets vs. Falcons

13 Calvin Ridley, Jaguars vs.
</item 85>

<item 86>
Fantasy Football TE PPR Rankings Week 14: Who to start, best sleepers at tight end (Sporting News, 2023-12-07T22:14:00+00:00)
Who are the TEs on bye in Week 14?

Few would've expected Week 14's two-team bye to be as problematic as it is, but losing Logan Thomas and Trey McBride this late in the season could have some
fantasy owners scrambling.

WEEK 14 PPR RANKINGS:

QBs | RBs | WRs | D/STs | Kickers

Fortunately, both the Commanders and Cardinals have above-average TE defenses, so taking those off the board only helps the position this week.

WEEK 14 STANDARD RANKINGS:

QBs | RBs | WRs | TEs | D/STs | Kickers

Who are the best fantasy TEs in Week 14?

As usual, Travis Kelce (vs. Bills in Week 14) tops our rankings, but T.J. Hockenson (@ Raiders), Sam LaPorta (@ Bears), David Njoku (vs. Jaguars), and Dalton
Kincaid (@ Chiefs) aren't far behind.
</item 86>

<item 87>
Fantasy Football TE PPR Rankings Week 14: Who to start, best sleepers at tight end (Sporting News, 2023-12-07T22:14:00+00:00)
It can be tough to trust Taysom Hill (vs. Panthers) in full-point PPR leagues, but a matchup against the league's second-worst run defense gives him a high
ceiling regardless of format. He's been more involved as a receiver this year, so he'll get a few catches to go along with the rushing stats, too. With Derek
Carr (concussion) banged up, Hill might throw more this week, too.

Gerald Everett (vs. Broncos) and Isaiah Likely (vs. Rams) have matchups against bottom-six TE defenses, and Cole Kmet (vs. Lions), Jake Ferguson (vs. Eagles),
and Kyle Pitts (vs. Buccaneers) also have above-average matchups that give them higher floors and higher ceilings than usual.

For those in deeper leagues, Tyler Conklin (vs.
</item 87>

<item 88>
Fantasy Football Week 16 Wide Receiver Preview: Waiver adds, DFS plays, and more (CBSSports.com, 2023-12-19T20:51:32+00:00)
But this hot stretch started against the Raiders, when Rice scored 24.7 PPR Fantasy points against them in Week 12. They are playing a style of defense that
limits big plays but allows throws underneath which fits Rice and his 4.3 ADOT perfectly. While the Raiders have surrendered the 11th-fewest passing yards this
season, they've given up the fourth-most yards after the catch.

If Rice is on your roster, he's in your starting lineup. You have plenty of other things to stress about.

Heath's Week 16 Previews: QB | RB | WR TE



You can find more in my weekly projections over at SportsLine. Now here is the rest of the Week 16 WR Preview:

WR Preview Who's Out

The following players are not being projected to play Week 16 at this time. Here's what it means:

Nico Collins WR HOU Houston • #12

Age: 24 • Experience: 3 yrs.
</item 88>

<item 89>
NFL Fantasy Football Week 6 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-12T21:23:16+00:00)
The only rush defense metrics they're not top-five in are Defensive Rush EPA (just below league average), yards before contact per rush (0.88 yards, they're
sixth), yards after contact per rush (2.80, just above league average). They even lead the NFL in fewest missed tackles. Tony Pollard had an 8-29-0 rushing stat
line against them last week.

WHAT ABOUT THROUGH THE AIR? Even though the 49ers see the fifth-most RB targets per game, they're better than league average in catch rate allowed (75%) and
top-10 in both yards per catch (5.7) and YAC/reception (6.10) allowed to running backs. Pollard caught 4 of 5 passes for 35 yards last week.

P.J. WALKER: Has thrown 17.2% of his career pass attempts to running backs.
</item 89>

<item 90>
Fantasy Football RB PPR Rankings Week 14: Who to start, best sleepers at running back (Sporting News, 2023-12-07T22:09:00+00:00)
WEEK 14 STANDARD RANKINGS:

QBs | RBs | WRs | TEs | D/STs | Kickers

Who are the best fantasy RBs in Week 14?

Christian McCaffrey (vs. Seahawks in Week 14) has given us no reason to stop believing in him as the best overall player in the NFL, so he stays atop this list.
Austin Ekeler (vs. Broncos) has started to give us some pause as of late, as he's averaged fewer than 3.0 yards per carry since coming off IR eight games ago,
but a juicy home matchup against Denver should be his get-right game. Clear skies ahead, Ek owners — the Broncos have surrendered the most PPR points in the NFL
to RBs, and L.A. faces them twice between now and the fantasy title week.

With Jacksonville playing very well over the past month, we're all-in on Travis Etienne Jr. (@ Browns) even in a tough matchup.
</item 90>

<item 91>
Fantasy Football RB PPR Rankings Week 14: Who to start, best sleepers at running back (Sporting News, 2023-12-07T22:09:00+00:00)
As we do every week, we're deep-diving the running back position and analyzing all the stats, matchups, and injuries before breaking down our rankings. So,
let's get to it and survey the RB landscape for the final week of the fantasy regular season.

Which RBs are on bye in Week 14?

After a whopping six teams were on bye last week, only the Commanders and Cardinals will be off in Week 14. That takes Brian Robinson Jr. and James Conner off
the table, but it doesn't hurt us much beyond that.

WEEK 14 PPR RANKINGS:

QBs | WRs | TEs | D/STs | Kickers

It does hurt a bit to have two below-average run defenses off the table, as Washington and Arizona have both ranked among the 10 most generous teams to RBs this
season.
</item 91>

<item 92>
Fantasy Football Week 16 Wide Receiver Preview: Waiver adds, DFS plays, and more (CBSSports.com, 2023-12-19T20:51:32+00:00)
The other three are Tyreek Hill, Tyler Lockett, and DeSean Jackson.

-- Deebo Samuel is on pace to become just the fourth player since 1992 to average more than 11 yards per target with at least 60 targets in multiple seasons.
The other three are Tyreek Hill, Tyler Lockett, and DeSean Jackson. 18 -- Puka Nacua needs 18 more catches to set the rookie record for most catches in a
season.

-- Puka Nacua needs 18 more catches to set the rookie record for most catches in a season. 152.67 -- After missing last week's game, Tyreek Hill now needs to
average almost 153 yards per game in his final three to reach his stated goal of 2,000 receiving yards this season.

-- After missing last week's game, Tyreek Hill now needs to average almost 153 yards per game in his final three to reach his stated goal of 2,000 receiving
yards this season.
</item 92>

<item 93>
Fantasy Football Week 13 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-11-30T22:10:58+00:00)
Sit Him (Lineup Decisions) Projections powered by Sportsline Cooper Kupp WR LAR L.A. Rams • #10

Age: 30 • Experience: 7 yrs. Weekly Breakdown MATCHUP vs CLE LAR -3.5 O/U 40 OPP VS WR 14th PROJ PTS 13.8 WR RNK 25th YTD Stats REC 27 TAR 51 REYDS 393 TD 1
FPTS/G 10.6 You're never supposed to sit your studs, but what if your stud is playing through something that made him look like a dud last week? That was Kupp,
who in Week 12 ran a ton of short routes and couldn't separate and didn't have much sudden movement. If he's going to come through this week it'll likely come
on a ton of short targets and/or some targets inside the 10-yard line.
</item 93>

<item 94>
Fantasy Football WR PPR Rankings Week 14: Who to start, best sleepers at wide receiver (Sporting News, 2023-12-07T22:11:00+00:00)
Colts

50 Odell Beckham Jr., Ravens vs. Rams

51 Jonathan Mingo, Panthers @ Saints

52 Jameson Williams, Lions @ Bears

53 Xavier Gipson, Jets vs. Texans

54 Jalen Guyton, Chargers vs. Broncos

55 Darius Slayton, Giants vs. Packers

56 Quentin Johnston, Chargers vs. Broncos

57 Trey Palmer, Buccaneers @ Falcons

58 K.J. Osborn, Vikings @ Raiders

59 JuJu Smith-Schuster, Patriots @ Steelers

60 DJ Chark, Panthers @ Saints

61 Wan'Dale Robinson, Giants vs. Packers

62 Khalil Shakir, Bills @ Chiefs

63 Alec Pierce, Colts @ Bengals

64 Justin Watson, Chiefs vs.
</item 94>

<item 95>
Fantasy Football WR PPR Rankings Week 13: Who to start, best sleepers at wide receiver (Sporting News, 2023-11-29T20:27:00+00:00)
Jaguars

49 Robert Woods, Texans vs. Broncos

50 Jonathan Mingo, Panthers @ Buccaneers

51 Jameson Williams, Lions @ Saints

52 Zay Jones, Jaguars vs. Bengals

53 Tutu Atwell, Rams vs. Browns

54 Greg Dortch, Cardinals @ Steelers

55 Jalen Guyton, Chargers @ Patriots

56 Lynn Bowden Jr., Saints vs. Lions

57 Quentin Johnston, Chargers @ Patriots

58 Justin Watson, Chiefs @ Packers

59 Treylon Burks, Titans vs. Colts

60 Josh Reynolds, Lions @ Saints

61 Keith Kirkwood, Saints vs. Lions

62 Cedric Tillman, Browns @ Rams

63 Trey Palmer, Buccaneers vs. Panthers

64 A.T. Perry, Saints vs. Lions

65 Michael Gallup, Cowboys vs. Seahawks

66 JuJu Smith-Schuster, Patriots vs.
</item 95>

<item 96>
NFL Fantasy Football Week 5 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-04T21:56:25+00:00)
Even if you take out the Bears game the Chiefs have yielded a 56.8% catch rate and 2.93 YAC/reception with a 10.9 receiving average to the Lions, Jaguars and
Jets. Drill down even further and they're great against outside receivers like Addison in terms of catch rate (45.8%) and YAC/reception (2.86), but they have
allowed 15.0 yards per catch specifically to outside receivers. It's the only saving grace Addison has.

Start 'em 10.0-7.00 6.99 - 5.01 5.00 - 0 Sit 'em

Start Him (Lineup Decisions) Projections powered by Sportsline Garrett Wilson WR NYJ N.Y. Jets • #17

Age: 23 • Experience: 2 yrs.
</item 96>

<item 97>
NFL Fantasy Football Week 5 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-04T21:56:25+00:00)
I'd rather start Breece Hall, Rhamondre Stevenson, Dameon Pierce and Jaleel McLaughlin (if Javonte Williams sits out).

LAST WEEK: Harris had his first game with over 100 total yards against the Texans and averaged 5.1 yards per rush, just shy of the season-high he technically
had on six carries against the 49ers in Week 1. A 32-yard reception helped him cross the triple-digit barrier.

HARRIS: This might shock you, but Harris is averaging 4.3 yards per run this year with a 12.2% explosive run rate. Would you believe the explosive run rate,
which reflects the frequency of his 12-plus-yard runs, is fifth-best among 44 qualifying running backs?! He's also 13th in avoided tackle rate and top-10 in
yards after contact per carry at 3.53.

TOUCHDOWNS: Harris has zero.
</item 97>

<item 98>
Fantasy Football TE PPR Rankings Week 14: Who to start, best sleepers at tight end (Sporting News, 2023-12-07T22:14:00+00:00)
This week might not matter for many fantasy football owners. Some have secured their spot in the postseason; others are well out of contention. However, for
those in must-win situations, this week really matters. It's not the time to have a hole in your lineup at tight end, but if you do, we're here to help with our
Week 14 fantasy TE PPR rankings.

Fortunately, there are several solid sleepers and streaming options despite two tough byes and some big injuries to notable TEs. It's always tough to know who
to trust at this volatile position, so we break down which tight ends have higher floors for those favored to win comfortably and which have higher ceilings for
those who need to really shoot for big points.

Let's get to it and check out the TE landscape for Week 14.

Who are the TEs on bye in Week 14?
</item 98>

<item 99>
Fantasy Football Week 16 Wide Receiver Preview: Waiver adds, DFS plays, and more (CBSSports.com, 2023-12-19T20:51:32+00:00)
Demario Douglas WR NE New England • #81

Age: 23 • Experience: Rookie Weekly Breakdown MATCHUP @ DEN DEN -6.5 O/U 34.5 OPP VS WR 14th WR RNK 46th ROSTERED 40% YTD Stats REC 39 TAR 60 REYDS 443 TD 0
FPTS/G 7.7 Douglas' first game back was a disappointment but he still has nine targets in two of his last three games and seven targets in four of his last five
games. I'm hoping for a bounce back in Week 16.
</item 99>

<item 100>
Fantasy Football Week 16 Wide Receiver Preview: Waiver adds, DFS plays, and more (CBSSports.com, 2023-12-19T20:51:32+00:00)
Projected stats for all starting quarterbacks are available at Sportsline.
</item 100>
=== END ITEM id=1e6d6eaf10ef8f51 ===
