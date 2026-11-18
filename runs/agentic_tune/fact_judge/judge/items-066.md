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

=== ITEM id=b02c1530894205b7 ===
Gold sentences and facts:
Sentence 0: After turning in three limited practices this week, we'll trust that Hockenson is ready to go, especially since he continued to play through the ailment in Week 9 as a big part of the offense with Josh Dobbs under center.
  Fact 0.0: Hockenson turned in three limited practices this week.
  Fact 0.1: The author will trust that Hockenson is ready to go.
  Fact 0.2: The author's trust is especially supported because Hockenson continued to play through the ailment in Week 9.
  Fact 0.3: Hockenson was a big part of the offense in Week 9.
  Fact 0.4: Josh Dobbs was under center in Week 9.
Sentence 1: Kincaid was a limited participant in practice on Wednesday due to a shoulder injury, but it’s been reported he’s likely to suit up on Saturday at less than 100%.
  Fact 1.0: Kincaid was a limited participant in practice on Wednesday.
  Fact 1.1: Kincaid's limited participation was due to a shoulder injury.
  Fact 1.2: It has been reported that Kincaid is likely to suit up on Saturday.
  Fact 1.3: It has been reported that Kincaid is likely to play at less than 100%.

Retrieved items:
<item 1>
Fantasy Injury Updates Week 10: News on Justin Jefferson, Ja'Marr Chase, T.J. Hockenson & Curtis Samuel (Sporting News, 2023-11-11T21:17:00+00:00)
ET, according to ESPN's Jeremy Fowler. The receiver was limited in practice thanks to the toe injury that has plagued him since leaving Week 8's game against
the Eagles, but barring a setback, he will return to action this weekend.

The receiver was working his way into flex territory with a couple of good performances before picking up the injury. However, we wouldn't consider him to be
worthy of a fantasy start this week unless you're in a bind. It's unclear whether Samuel will regain the same workload or if he'll be playing in limited action
as he works his way back. Jamison Crowder emerged in Samuel's absence and Jahan Dotson has seen his production drastically increase, as well. The wait-and-see
approach is the preferred one heading into Week 10.

Graph facts:
- Which M.L.B. Playoff Team Has the Most Daunting Home Field? We Ranked Them. -MENTIONS-> ESPN
- Taylor Swift-Travis Kelce timeline: Everything to know about rumored romance between pop star, Chiefs TE -MENTIONS-> ESPN
- Michigan sign-stealing scandal winners and losers, from Jim Harbaugh to Tony Petitti -MENTIONS-> ESPN
- Carabao Cup fixtures, draws, results, TV channel, and live streams for every round of EFL League Cup 2023/24 -MENTIONS-> ESPN
- UTSA vs. Marshall odds, props, predictions: Roadrunners lay big spread in Frisco Bowl -MENTIONS-> ESPN

Entities: ESPN, Curtis Samuel, Eagles, Sporting News, Justin Jefferson, Fantasy Injury Updates Week 10: News on Justin Jefferson, Ja'Marr Chase, T.J. Hockenson
& Curtis Samuel, Ja'Marr Chase, Jeremy Fowler, Week 10, Jamison Crowder, Week 8's game against the Eagles, toe injury, Jahan Dotson, T.J. Hockenson
</item 1>

<item 2>
Fantasy Injury Updates Week 10: News on Justin Jefferson, Ja'Marr Chase, T.J. Hockenson & Curtis Samuel (Sporting News, 2023-11-11T21:17:00+00:00)
Jefferson (hamstring) is officially listed as "questionable" for Sunday's game against the Saints. Kickoff is slated for 1 p.m. ET, but fantasy owners won't
have to wait long for news on the star receiver. In order to be eligible to play, Jefferson will need to be activated by 4 p.m. ET on Saturday. Earlier in the
week head coach Kevin O'Connell said that it would be "a little aggressive" for Jefferson to get back on the field this week. However, the receiver's three
straight limited practices were enough for the team to at least weigh the decision, but thankfully, an early answer awaits.

WEEK 10 FANTASY ADVICE:

Sleepers | Busts | Projections | Start 'em, sit 'em

Jefferson's fantasy outlook is always through the roof, but it will be interesting to see if that changes with Josh Dobbs under center instead of Kirk Cousins.

Graph facts:
- NFL standings: Updated AFC, NFC playoff picture for Week 13 of 2023 season -MENTIONS-> Saints
- FanDuel Picks Week 15: Best NFL DFS lineup advice for daily fantasy football tournaments -MENTIONS-> Saints
- Fantasy Football Week 13 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game -MENTIONS-> Saints
- NFL Fantasy Football Week 5 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game -MENTIONS-> Saints
- Jaguars vs. Saints live score, updates, highlights from NFL 'Thursday Night Football' game -MENTIONS-> Saints

Entities: Saints, Curtis Samuel, Josh Dobbs, Kevin O'Connell, Sporting News, Justin Jefferson, Sunday's game against the Saints, Fantasy Injury Updates Week 10:
News on Justin Jefferson, Ja'Marr Chase, T.J. Hockenson & Curtis Samuel, Ja'Marr Chase, Kirk Cousins, T.J. Hockenson, fantasy
</item 2>

<item 3>
Fantasy Injury Updates Week 10: News on Justin Jefferson, Ja'Marr Chase, T.J. Hockenson & Curtis Samuel (Sporting News, 2023-11-11T21:17:00+00:00)
Chase put a scare into the fantasy community on Wednesday when he was listed as a DNP, but after logging back-to-back limited practices, there is a lot more
optimism heading into Sunday. With Tee Higgins (hamstring) already ruled out, Cincinnati will be hoping Chase can feel healthy enough to play when they host the
Texans at 1 p.m. ET on Sunday.

The Bengals, despite getting a win over the Bills last Sunday, are a beaten-down bunch. Their defense will also be without a prominent edge rusher, Sam Hubbard,
complicating matters for a team hoping to stack wins as they compete for an AFC North crown. On the bright side, Joe Burrow appears to be finally healthy and
should have no problems targeting Tyler Boyd, Trenton Irwin, and Andrei Iosivas in the passing game. If Chase does end up suiting up, fantasy owners should
enjoy an even larger workload than the receiver already had.

Graph facts:
- NFL Fantasy Football Week 5 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game -MENTIONS-> Joe Burrow
- 2023 Kentucky online sports betting sites: Best legal sportsbooks, promos, bonuses, mobile apps, how to bet -MENTIONS-> Joe Burrow
- Fantasy Football WR PPR Rankings Week 13: Who to start, best sleepers at wide receiver -MENTIONS-> Joe Burrow
- 2024 NFL Mock Draft: Bengals' Joe Burrow gets another playmaker; four QBs go in Round 1, but Pats opt for WR -MENTIONS-> Joe Burrow
- NFL standings: Updated AFC, NFC playoff picture for Week 13 of 2023 season -MENTIONS-> Joe Burrow

Entities: Joe Burrow, Bills, Bengals, Curtis Samuel, Texans, Trenton Irwin, Tyler Boyd, Sporting News, Justin Jefferson, Fantasy Injury Updates Week 10: News on
Justin Jefferson, Ja'Marr Chase, T.J. Hockenson & Curtis Samuel, Ja'Marr Chase, Bengals win over the Bills, Bengals host the Texans, Tee Higgins, Sam Hubbard
</item 3>

<item 4>
Fantasy Injury Updates Week 10: News on Justin Jefferson, Ja'Marr Chase, T.J. Hockenson & Curtis Samuel (Sporting News, 2023-11-11T21:17:00+00:00)
After turning back the clocks last week, it now gets darker earlier for the foreseeable future. That darkness allows us to see the stars on a clear night, so
it's only natural to wonder if the pass-catching stars will shine bright in Week 10 of the NFL season. We already know Tee Higgins and Nico Collins are OUT,
while Justin Jefferson, Ja'Marr Chase, and T.J. Hockenson are among the other star pass-catchers beaten up heading into the week. Can we expect them on the
field when the action kicks off Sunday?

Below, we'll give you the latest news on these ailing pass-catchers and break down how their injuries could potentially affect the Week 10 fantasy landscape.

WEEK 10 INJURY UPDATES: Early WRs | RBs

Is Justin Jefferson playing Week 10?

UPDATE: The Vikings will not activate Jefferson before Sunday's game and have officially ruled him OUT, according to NFL Network's Tom Pelissero.
</item 4>

<item 5>
Fantasy Injury Updates Week 10: News on Justin Jefferson, Ja'Marr Chase, T.J. Hockenson & Curtis Samuel (Sporting News, 2023-11-11T21:17:00+00:00)
If he's out, Boyd is a must-start, and you can try one of Cincinnati's secondary receivers in deep leagues.

WEEK 10 STANDARD RANKINGS:

QBs | RBs | WRs | TEs | D/STs | Kickers

Is T.J. Hockenson playing Week 10?

UPDATE: NFL Network's Ian Rapoport reported Sunday morning that Hockenson is expected to play against New Orleans.

UPDATE 2: Hockenson is officially ACTIVE.

Hockenson (ribs) is listed as "questionable" for Week 10, but he's already said he plans to play on Sunday. The tight end has dealt with his fair share of
injury woes this season, however, he's yet to miss a game.

Graph facts:
- Jaguars vs. Saints live score, updates, highlights from NFL 'Thursday Night Football' game -MENTIONS-> New Orleans
- Fantasy Football Week 13 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game -MENTIONS-> New Orleans
- The Halloween Countdown: 31 days of horror to watch -MENTIONS-> New Orleans
- NFL standings: Updated AFC, NFC playoff picture for Week 13 of 2023 season -MENTIONS-> New Orleans
- DraftKings Picks Week 14: Best NFL DFS lineup advice for daily fantasy football tournaments -MENTIONS-> New Orleans

Entities: New Orleans, NFL Network, Curtis Samuel, fantasy football, Sporting News, Justin Jefferson, Fantasy Injury Updates Week 10: News on Justin Jefferson,
Ja'Marr Chase, T.J. Hockenson & Curtis Samuel, Ja'Marr Chase, Ian Rapoport, Week 10, Boyd, Cincinnati, injury, T.J. Hockenson
</item 5>

<item 6>
Fantasy Football TE PPR Rankings Week 14: Who to start, best sleepers at tight end (Sporting News, 2023-12-07T22:14:00+00:00)
Jaguars), and Dalton Kincaid (@ Chiefs) aren't far behind. LaPorta and Njoku both have great matchups against defenses that entered last week in the top 10 in
fantasy points per game (FPPG) allowed to TEs, and we know what Kelce, Hockenson, and Kincaid do every week in PPR formats.

George Kittle (vs. Seahawks) doesn't have a great matchup -- especially when you remember that he had just three catches for 19 yards against Seattle in Week 12
-- but he remains a must-start given his high ceiling in any matchup.

WEEK 14 FANTASY ADVICE:

Sleepers | Busts | Projections | Start-sit

Who are the best fantasy TE sleepers, pickups, & streamers in Week 14?

It can be tough to trust Taysom Hill (vs.

Graph facts:
- Raiders vs. Lions live score, updates, highlights from NFL 'Monday Night Football' game -MENTIONS-> LaPorta
- LaPorta -PARTICIPATED_IN-> NFL 'Monday Night Football' game
- LaPorta -PARTICIPATED_IN-> Raiders vs. Lions game
- Jaguars vs. Saints live score, updates, highlights from NFL 'Thursday Night Football' game -MENTIONS-> Taysom Hill
- Fantasy Football Week 13 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game -MENTIONS-> Taysom Hill

Entities: LaPorta, Taysom Hill, Jaguars, Week 12, Kelce, Njoku, Hockenson, fantasy football, Week 14, fantasy points per game (FPPG), Chiefs, Sporting News,
Seahawks, tight end, PPR formats
</item 6>

<item 7>
Graph query row:
article: Fantasy Football TE PPR Rankings Week 14: Who to start, best sleepers at tight end
publisher: Sporting News
player1: T.J. Hockenson
player2: Dalton Kincaid
saturday_game: None
passage: Fantasy Football TE PPR Rankings Week 14: Who to start, best sleepers at tight end (Sporting News, 2023-12-07T22:14:00+00:00)
This week might not matter for many fantasy football owners. Some have secured their spot in the postseason; others are well out of contention. However, for
those in must-win situations, this week really matters. It's not the time to have a hole in your lineup at tight end, but if you do, we're here to help with our
Week 14 fantasy TE PPR rankings.

Fortunately, there are several solid sleepers and streaming options despite two tough byes and some big injuries to notable TEs. It's always tough to know who
to trust at this volatile position, so we break down which tight ends have higher floors for those favored to win comfortably and which have higher ceilings for
those who need to really shoot for big points.

Let's get to it and check out the TE landscape for Week 14.

Who are the TEs on bye in Week 14?
</item 7>

<item 8>
Fantasy Injury Updates Week 10: News on Justin Jefferson, Ja'Marr Chase, T.J. Hockenson & Curtis Samuel (Sporting News, 2023-11-11T21:17:00+00:00)
In addition, K.J. Osborn (concussion) is also listed as "questionable," though his status steadily improved throughout the week from DNP on Wednesday to full
practice on Friday. If Jefferson is back, he's a must-start. If Jefferson is out but Osborn returns, both Osborn and Jordan Addison are worth using.

WEEK 10 PPR RANKINGS:

QBs | RBs | WRs | TEs | D/STs | Kickers

Is Ja'Marr Chase playing Week 10?

UPDATE: ESPN's Jeremy Fowler reported late Saturday that the Bengals are "hopeful" that Chase will play against Houston, but they won't know for sure until
after a pre-game workout.

UPDATE 2: Chase is officially ACTIVE.

Chase (back) appears ready to take his status down to kickoff. According to The Athletic's Dianna Russini, the Bengals' star receiver will work out on Sunday
morning and his status will be determined afterward.

Graph facts:
- Which M.L.B. Playoff Team Has the Most Daunting Home Field? We Ranked Them. -MENTIONS-> The Athletic
- Michigan sign-stealing scandal winners and losers, from Jim Harbaugh to Tony Petitti -MENTIONS-> The Athletic
- Browns survive Bears Hail Mary on wild, near-catch by Darnell Mooney: 'Damn near s— myself' -MENTIONS-> The Athletic
- He’s Hockey’s Brightest Young Star. This Is What Makes His Shot So Special. -MENTIONS-> The Athletic
- How Did the Houston Astros Cheat in 2017? Everything to Know About a Dark Moment in MLB History -MENTIONS-> The Athletic

Entities: The Athletic, Bengals, Jordan Addison, Curtis Samuel, PPR rankings, K.J. Osborn, ESPN, Sporting News, Justin Jefferson, Fantasy Injury Updates Week
10: News on Justin Jefferson, Ja'Marr Chase, T.J. Hockenson & Curtis Samuel, Ja'Marr Chase, Jeremy Fowler, Dianna Russini, pre-game workout, Week 10
</item 8>

<item 9>
Fantasy Injury Updates Week 10: News on Justin Jefferson, Ja'Marr Chase, T.J. Hockenson & Curtis Samuel (Sporting News, 2023-11-11T21:17:00+00:00)
After turning in three limited practices this week, we'll trust that Hockenson is ready to go, especially since he continued to play through the ailment in Week
9 as a big part of the offense with Josh Dobbs under center.

As long as Hockenson is playing, he's not being benched in any fantasy format. The tight end projects as a top option at the position and remains the Vikings'
leader in targets with 80 on the season. Dobbs enjoyed targeting the tight ends in Arizona and continued that trend in his Minnesota debut. The floor for
Hockenson is equivalent to that of a rooftop bar.

WEEK 10 DFS: DK lineup | FD lineup | Best values | Best stacks

Is Curtis Samuel playing Week 10?

UPDATE: Samuel is officially ACTIVE.

Samuel (toe) is expected to play on Sunday when the Commanders take on the Seahawks at 4:25 p.m. ET, according to ESPN's Jeremy Fowler.

Graph facts:
- Vikings vs. Broncos live score, updates, highlights from NFL 'Sunday Night Football' game -MENTIONS-> Vikings
- NFL standings: Updated AFC, NFC playoff picture for Week 13 of 2023 season -MENTIONS-> Vikings
- Fantasy Football Week 13 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game -MENTIONS-> Vikings
- NFL Fantasy Football Week 5 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game -MENTIONS-> Vikings
- Vikings -MENTIONS-> Minn.

Entities: Vikings, Minnesota, ESPN, Curtis Samuel, Josh Dobbs, Sporting News, Justin Jefferson, Fantasy Injury Updates Week 10: News on Justin Jefferson,
Ja'Marr Chase, T.J. Hockenson & Curtis Samuel, Ja'Marr Chase, Jeremy Fowler, Week 10, Seahawks, Arizona, Commanders, Week 9
</item 9>

<item 10>
NFL Week 13 injuries: Browns to start Joe Flacco with Dorian Thompson-Robinson out, Saints' Olave questionable (CBSSports.com, 2023-12-01T22:52:36+00:00)
The wideout still has another hurdle to pass to be cleared to play on Sunday so his status is still very much up in the air.

Falcons (-2) at Jets

Atlanta is relatively healthy coming into this matchup. Hollins, who injured his ankle in Week 9, participated on a limited basis every day this week, which is
encouraging for his prospects to play on Sunday. As for Hughes, he practiced fully all week after suffering a hand injury last weekend against the Saints.
Linebacker Nate Landman (thigh) missed practice on Wednesday but will play vs. New York.

Rodgers was once again limited in practice but, as expected, will not be activated this week and will be sidelined as he continues his recovery from a torn
Achilles just a few months ago. Meanwhile, Cook popped up on the injury report due to a shoulder injury, but head coach Robert Saleh told reporters Friday that
the back should be fine.

Graph facts:
- NFL standings: Updated AFC, NFC playoff picture for Week 13 of 2023 season -MENTIONS-> Saints
- FanDuel Picks Week 15: Best NFL DFS lineup advice for daily fantasy football tournaments -MENTIONS-> Saints
- Fantasy Football Week 13 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game -MENTIONS-> Saints
- NFL Fantasy Football Week 5 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game -MENTIONS-> Saints
- Jaguars vs. Saints live score, updates, highlights from NFL 'Thursday Night Football' game -MENTIONS-> Saints

Entities: Saints, hand injury, torn Achilles, CBSSports.com, Atlanta, New York, thigh injury, ankle injury, Jets, Hughes, injuries, Hollins, Nate Landman,
Olave, NFL Week 13 injuries: Browns to start Joe Flacco with Dorian Thompson-Robinson out, Saints' Olave questionable
</item 10>
=== END ITEM id=b02c1530894205b7 ===

=== ITEM id=2dd6f0eb2dd00470 ===
Gold sentences and facts:
Sentence 0: You can find value in player and team props by gathering information from multiple sources and tracking how players and their teammates perform.
  Fact 0.0: You can find value in player and team props by gathering information from multiple sources.
  Fact 0.1: You can find value in player and team props by tracking how players and their teammates perform.
Sentence 1: You can also bet on team props, such as whether a team will lead at the end of a certain quarter or an over/under bet on the total number of points the team will score in the game.
  Fact 1.0: You can also bet on team props.
  Fact 1.1: A team prop can be whether a team will lead at the end of a certain quarter.
  Fact 1.2: A team prop can be an over/under bet on the total number of points the team will score in the game.

Retrieved items:
<item 1>
Graph query row:
article: Prop Betting - What is a Prop Bet?
publisher: Sporting News
nba_props: None
relationship: None
passage: Prop Betting - What is a Prop Bet? (Sporting News, 2023-11-06T21:23:22+00:00)
Prop bets are easy to understand and add a twist to most sporting events as you can bet on in-game events not tied to the outcome.

Prop bets are popular throughout the year and are posted for every major sport, but the pinnacle for prop betting is the Super Bowl. It’s the most popular day
of the year for prop bets, which range from the typical to the exotic.

But you don’t have to wait for the Super Bowl to get involved in prop betting. See below for more on props, including the pros, cons, and prop bet examples from
all the major sports.

What is a prop bet?

A prop bet is a “side bet” with a sportsbook on a specific event or statistical accomplishment within the game. These rarely have anything to do with the
outcome of a game itself.

For example, you might bet on:

LeBron James scoring over 20 points in a single game

Dallas Cowboys gaining over 300 receiving yards in a game

Pros and cons of prop bets

Every bet has positives and negatives, and prop bets are no different.
</item 1>

<item 2>
Prop Betting - What is a Prop Bet? (Sporting News, 2023-11-06T21:23:22+00:00)
They vary from player to team bets and lean heavily on the statistical side of the game. Some NBA prop examples include:

How many points a player will score

Number of assists or three-pointers a player will have

Which player will score the first basket of a game

You can also bet on team props, such as whether a team will lead at the end of a certain quarter or an over/under bet on the total number of points the team
will score in the game.

See more: NBA prop bets

NFL prop bets

The National Football League is the king of sports betting, and NFL betting sites have no qualms about showcasing every kind of possible bet type for it.

Bettors will find many prop bet opportunities, especially for the Super Bowl. During the regular season, one of the most popular prop bet options includes the
“anytime touchdown scorer” bet, where you can wager on a player to score a TD at any point in the game. Other prop bets can be focused on the team or even both
teams.

Graph facts:
- Moneyline Betting - What is the Moneyline and How Does it Work? -MENTIONS-> Super Bowl
- Vermont Sportsbook Promos and Sports Betting Launch News -MENTIONS-> Super Bowl
- NFL roundup: Pats hit new low as Lions reach 9-3 for first time since 1962 -MENTIONS-> Super Bowl
- Travis Kelce faces backlash after comments about ‘finding a breeder’ resurface amid Taylor Swift romance -MENTIONS-> Super Bowl
- Christmas Day preview: 49ers, Ravens square off in potential Super Bowl sneak peek -MENTIONS-> Super Bowl

Entities: Super Bowl, prop betting, Prop Betting - What is a Prop Bet?, Sporting News, NBA, sports betting, National Football League
</item 2>

<item 3>
NBA Prop Bets - Best Team and Player Props 2023/24 Season (Sporting News, 2023-10-02T23:49:35+00:00)
Positive numbers, like +145, show how much you stand to win for every $100 you wager. If you were to bet $50 on Giannis Antetokounmpo to have over 7.5 rebounds
and the odds were +130 at the time you finalized your bet slip, you would win $65 in profit if your bet wins.

Positive numbers also indicate which side of the bet is the underdog in the eyes of the oddsmakers.

Remember that you can bet as little or as much as you want, and your payout will be calculated based on your wager amount and the odds at the time you make your
bet.

Types of NBA props

You can make two main prop bets on the National Basketball Association:

Team props

Player props

Team props are typically focused on statistical accomplishments by a particular team and rarely on the game’s result.

Graph facts:
- UTSA vs. Marshall odds, props, predictions: Roadrunners lay big spread in Frisco Bowl -PUBLISHED_BY-> Sporting News
- North Texas vs. SMU odds, props, predictions: Red hot Mustangs could overpower not-so-Mean Green -PUBLISHED_BY-> Sporting News
- Chiefs vs. Packers live score, updates, highlights from NFL 'Sunday Night Football' game -PUBLISHED_BY-> Sporting News
- Jaguars vs. Saints live score, updates, highlights from NFL 'Thursday Night Football' game -PUBLISHED_BY-> Sporting News
- Raiders vs. Lions live score, updates, highlights from NFL 'Monday Night Football' game -PUBLISHED_BY-> Sporting News

Entities: Sporting News, player props, team props, National Basketball Association, NBA Prop Bets - Best Team and Player Props 2023/24 Season, Giannis
Antetokounmpo
</item 3>

<item 4>
Prop Betting - What is a Prop Bet? (Sporting News, 2023-11-06T21:23:22+00:00)
Use most of your budget on more stable bet types, such as moneylines, spreads, or totals. Focus only a small percentage of your bankroll on props.

We wouldn’t recommend using prop bets as your main bet type when building your strategy. Use most of your budget on more stable bet types, such as moneylines,
spreads, or totals. Focus only a small percentage of your bankroll on props. Exotic props are a guessing game: When large events like the Super Bowl roll
around, you’ll be inundated with exotic and entertaining prop bets. These are almost impossible to research, and pure guessing isn’t good for anyone’s bankroll
or betting strategy.

Prop bet examples for major sports

Every major sport with odds at the best sports betting apps will have prop bets. Below, we give examples of prop bets you might find when betting on major
American sports.

NBA prop bets

The NBA offers a wide range of prop bets. They vary from player to team bets and lean heavily on the statistical side of the game.

Graph facts:
- Moneyline Betting - What is the Moneyline and How Does it Work? -MENTIONS-> Super Bowl
- Vermont Sportsbook Promos and Sports Betting Launch News -MENTIONS-> Super Bowl
- NFL roundup: Pats hit new low as Lions reach 9-3 for first time since 1962 -MENTIONS-> Super Bowl
- Travis Kelce faces backlash after comments about ‘finding a breeder’ resurface amid Taylor Swift romance -MENTIONS-> Super Bowl
- Christmas Day preview: 49ers, Ravens square off in potential Super Bowl sneak peek -MENTIONS-> Super Bowl

Entities: Super Bowl, Bankroll, Betting strategy, Exotic props, Prop Betting - What is a Prop Bet?, Sporting News, spreads, totals, NBA, prop bets, moneylines
</item 4>

<item 5>
NBA Prop Bets - Best Team and Player Props 2023/24 Season (Sporting News, 2023-10-02T23:49:35+00:00)
You’ll often come across NBA team prop bets that are based on a team’s points, such as the first team to reach 20 points, the first team to score in a game, or
whether the total points by both teams in a contest will end up being odd or even.

Team props, also known as “game props,” don’t rely on individual players but rather on the team as a whole. Player props, on the other hand, are driven by
individual performance.

These individual props are focused on statistical milestones during a game. The most popular bets typically focus on points, rebounds, assists, and three-point
field goals. You can also bet on combination wagers, such as points and rebounds together.

In most cases, you can bet on traditional statistics for the five starters on each team.

See more:

Tips for betting on NBA props

Every bettor should develop their strategies, but we do have three tips you can use for NBA props.

Graph facts:
- UTSA vs. Marshall odds, props, predictions: Roadrunners lay big spread in Frisco Bowl -PUBLISHED_BY-> Sporting News
- North Texas vs. SMU odds, props, predictions: Red hot Mustangs could overpower not-so-Mean Green -PUBLISHED_BY-> Sporting News
- Chiefs vs. Packers live score, updates, highlights from NFL 'Sunday Night Football' game -PUBLISHED_BY-> Sporting News
- Jaguars vs. Saints live score, updates, highlights from NFL 'Thursday Night Football' game -PUBLISHED_BY-> Sporting News
- Raiders vs. Lions live score, updates, highlights from NFL 'Monday Night Football' game -PUBLISHED_BY-> Sporting News

Entities: Sporting News, NBA, player props, NBA Prop Bets - Best Team and Player Props 2023/24 Season, team prop bets, statistical milestones, points, rebounds,
assists, three-point field goals, combination wagers, traditional statistics, NBA props
</item 5>

<item 6>
NBA Prop Bets - Best Team and Player Props 2023/24 Season (Sporting News, 2023-10-02T23:49:35+00:00)
Pay attention to player and team news: Oddsmakers use various information to develop their lines for games and players, and you should, too. Keep tabs on player
news and injuries, when top players are being rested, how they performed in recent games, and more. You can find value in player and team props by gathering
information from multiple sources and tracking how players and their teammates perform. For example, if one of the starting guards is questionable for a game,
you might want to consider the other guard for increased points or assists since they’ll likely handle a heavier load.

Keep an eye on usage and minutes: Even slight increases to a player’s average minutes can make a difference when betting player props, especially later in the
season when aging veteran players tend to rest more. Keep yourself informed on usage and minutes trends for the players you’re considering betting on, and use
them to your advantage.

Graph facts:
- UTSA vs. Marshall odds, props, predictions: Roadrunners lay big spread in Frisco Bowl -PUBLISHED_BY-> Sporting News
- North Texas vs. SMU odds, props, predictions: Red hot Mustangs could overpower not-so-Mean Green -PUBLISHED_BY-> Sporting News
- Chiefs vs. Packers live score, updates, highlights from NFL 'Sunday Night Football' game -PUBLISHED_BY-> Sporting News
- Jaguars vs. Saints live score, updates, highlights from NFL 'Thursday Night Football' game -PUBLISHED_BY-> Sporting News
- Raiders vs. Lions live score, updates, highlights from NFL 'Monday Night Football' game -PUBLISHED_BY-> Sporting News

Entities: Sporting News, NBA, player props, team props, NBA Prop Bets - Best Team and Player Props 2023/24 Season, usage and minutes, betting
</item 6>

<item 7>
NBA Prop Bets - Best Team and Player Props 2023/24 Season (Sporting News, 2023-10-02T23:49:35+00:00)
Remember that some of the best players in the NBA were once on the bench for most games until small increases in their playing time gave them a chance to shine.

Shop for the best lines: This tip applies to every bet you make. Using multiple sportsbook accounts lets you find the best odds for every NBA prop bet. Even the
smallest differences, like one sportsbook having odds at +125 and another at +135, can make a difference in your profit. By shopping for the best lines, you can
make smarter bets and find the best value.

Related Pages:

Graph facts:
- UTSA vs. Marshall odds, props, predictions: Roadrunners lay big spread in Frisco Bowl -PUBLISHED_BY-> Sporting News
- North Texas vs. SMU odds, props, predictions: Red hot Mustangs could overpower not-so-Mean Green -PUBLISHED_BY-> Sporting News
- Chiefs vs. Packers live score, updates, highlights from NFL 'Sunday Night Football' game -PUBLISHED_BY-> Sporting News
- Jaguars vs. Saints live score, updates, highlights from NFL 'Thursday Night Football' game -PUBLISHED_BY-> Sporting News
- Raiders vs. Lions live score, updates, highlights from NFL 'Monday Night Football' game -PUBLISHED_BY-> Sporting News

Entities: Sporting News, NBA, NBA prop bets, NBA Prop Bets - Best Team and Player Props 2023/24 Season
</item 7>

<item 8>
NBA Prop Bets - Best Team and Player Props 2023/24 Season (Sporting News, 2023-10-02T23:49:35+00:00)
In the NBA, prop bets deal with both players and teams.

Some examples of NBA player props are:

How many points a player will score

Number of rebounds a player will have

How many assists a player will get

Rebounds and assists combined

Points, rebounds, and assists combined

How many steals a player will get

Number of three-pointers made

Some examples of NBA team props include:

Which team will reach 10 points first

Which team will have the highest-scoring period

Which team will score first

Which team will win the first half

Which team will win the second half

Will there be a fourth-quarter comeback victory

All of the best sports betting sites have their selection of prop bets for each NBA game, so shop around to find the sportsbooks that best suit your needs.

How to bet on NBA props

The first step to betting on NBA props is opening an account at an online sportsbook. The process is simple, free, and doesn’t take very long.

Graph facts:
- UTSA vs. Marshall odds, props, predictions: Roadrunners lay big spread in Frisco Bowl -PUBLISHED_BY-> Sporting News
- North Texas vs. SMU odds, props, predictions: Red hot Mustangs could overpower not-so-Mean Green -PUBLISHED_BY-> Sporting News
- Chiefs vs. Packers live score, updates, highlights from NFL 'Sunday Night Football' game -PUBLISHED_BY-> Sporting News
- Jaguars vs. Saints live score, updates, highlights from NFL 'Thursday Night Football' game -PUBLISHED_BY-> Sporting News
- Raiders vs. Lions live score, updates, highlights from NFL 'Monday Night Football' game -PUBLISHED_BY-> Sporting News

Entities: Sporting News, NBA, prop bets, sports betting, sportsbook, NBA game, NBA Prop Bets - Best Team and Player Props 2023/24 Season
</item 8>

<item 9>
NBA Prop Bets - Best Team and Player Props 2023/24 Season (Sporting News, 2023-10-02T23:49:35+00:00)
Start betting: Go to the basketball section of your sportsbook and select the NBA. From there, you can choose individual games and make bets like money lines,
point spreads, totals, props, futures, and more.

How to read NBA props odds

At first glance, all the positive and negative numbers at US sportsbooks can be daunting. The good news is they are easy to read once you understand what you
see.

Negative numbers, such as -110, represent how much money you need to wager to win $100. In many cases, such as a moneyline bet, the negative number also
indicates which side of the bet is the favorite to win.

For example, if you were to bet on LeBron James scoring over 15.5 points in a contest at -125 odds, you’d need to wager $125 to win $100. If you did and your
bet was correct, then you’d take home $225, which is your original $125 stake returned to you plus your $100 in profit.

Graph facts:
- ASX set for bright start despite Wall Street slip; $A drops -MENTIONS-> US
- ASX set to open higher as Wall Street rebounds; $A rises -MENTIONS-> US
- Is Google Search better than the rest? And is that fair? -MENTIONS-> US
- ChatGPT: Everything you need to know about the AI-powered chatbot -MENTIONS-> US
- ASX set to drop as Wall Street’s September slump deepens -MENTIONS-> US

Entities: US, Sporting News, NBA, basketball, NBA Prop Bets - Best Team and Player Props 2023/24 Season, LeBron James
</item 9>

<item 10>
NBA Prop Bets - Best Team and Player Props 2023/24 Season (Sporting News, 2023-10-02T23:49:35+00:00)
The NBA is one of the most popular sports to bet on in the US. NBA prop bets are an easy way to jump into basketball betting.

Proposition bets, or props, are easy to understand, numerous, and cover both player props and team props to give you plenty of variety.

Below, we look at betting NBA props at the best NBA betting sites and share some key information to make informed bets.

What are NBA props?

A prop bet is a side wager on an aspect of a game that usually doesn’t tie directly to the result. In the NBA, prop bets deal with both players and teams.

Graph facts:
- ASX set for bright start despite Wall Street slip; $A drops -MENTIONS-> US
- ASX set to open higher as Wall Street rebounds; $A rises -MENTIONS-> US
- Is Google Search better than the rest? And is that fair? -MENTIONS-> US
- ChatGPT: Everything you need to know about the AI-powered chatbot -MENTIONS-> US
- ASX set to drop as Wall Street’s September slump deepens -MENTIONS-> US

Entities: US, Sporting News, NBA, basketball betting, proposition bets, player props, team props, NBA betting sites, NBA Prop Bets - Best Team and Player Props
2023/24 Season
</item 10>
=== END ITEM id=2dd6f0eb2dd00470 ===

=== ITEM id=ff2b6bc1c24d368e ===
Gold sentences and facts:
Sentence 0: As such, they’re never going to replace the need for reducing emissions of CO 2 and other greenhouse gases.
  Fact 0.0: They are never going to replace the need for reducing emissions of CO2.
  Fact 0.1: They are never going to replace the need for reducing emissions of other greenhouse gases.
Sentence 1: He stopped flying, became a vegetarian and ditched gasoline-powered cars (he drives a Tesla), cutting his personal emissions by about 90%, according to his math.
  Fact 1.0: The person stopped flying.
  Fact 1.1: The person became a vegetarian.
  Fact 1.2: The person ditched gasoline-powered cars.
  Fact 1.3: The person drives a Tesla.
  Fact 1.4: The person cut his personal emissions by about 90%, according to his math.

Retrieved items:
<item 1>
New ultrathin materials can pull climate-warming CO2 from the air (Science News For Students, 2023-12-15T11:30:00+00:00)
This diagram depicts a process to convert MAX phase materials into thin sheets of MXenes. An acid etches out all but the M layers (blue) and X layers (purple)
to yield thick MXene sheets. The far right panel shows a scanning electron microscope image of the MXene. M. Ozkan et al/Chem. Oct. 4, 2023 ( CC BY 4.0 DEED

MXenes have “probably the largest diversity and tunability any 2-D material can offer,” says Michael Naguib. By tunable, this engineering physicist means that
MXenes can be tweaked to take on different properties. Naguib works at Tulane University in New Orleans, La. He made the world’s first MXene in 2011, a titanium
carbide.

Scientists have now made more than 50 MXenes. New ones arise when scientists mix and match different M and X atoms. Each MXene has slightly different
properties, from how well it conducts electricity to how strong it is.

Graph facts:
- NFL standings: Updated AFC, NFC playoff picture for Week 13 of 2023 season -MENTIONS-> New Orleans
- Fantasy Football Week 13 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game -MENTIONS-> New Orleans
- The Halloween Countdown: 31 days of horror to watch -MENTIONS-> New Orleans
- Fantasy Injury Updates Week 10: News on Justin Jefferson, Ja'Marr Chase, T.J. Hockenson & Curtis Samuel -MENTIONS-> New Orleans
- Jaguars vs. Saints live score, updates, highlights from NFL 'Thursday Night Football' game -MENTIONS-> New Orleans

Entities: New Orleans, Science News For Students, New ultrathin materials can pull climate-warming CO2 from the air, MXenes, Michael Naguib, Tulane University,
Louisiana, MAX phase materials, titanium carbide, M. Ozkan, CO2
</item 1>

<item 2>
New ultrathin materials can pull climate-warming CO2 from the air (Science News For Students, 2023-12-15T11:30:00+00:00)
Otherwise, that would defeat their use in carbon capture.

Educators and Parents, Sign Up for The Cheat Sheet Weekly updates to help you use Science News Explores in the learning environment Client key* E-mail Address*
Go Thank you for signing up! There was a problem signing you up.

MXenes also cost a lot to make. The high cost of making these materials would be less important if they could withstand thousands of CO 2 capture-and-release
cycles. That way, scientists wouldn’t have to keep making new MXenes.

So far, no one has tested MXenes’ durability for stored CO 2 removal and reuse. Still, it helps that MXenes are mechanically rugged. Research has shown that
certain types can be as strong as steel.

The Riverside group’s new paper also points to a related family of nanosheets that may prove good CO 2 sponges. Made from MXenes, they’re known as MBenes,
because the “X” layer is made up of boron atoms.

Graph facts:
- New ultrathin materials can pull climate-warming CO2 from the air -MENTIONS-> Per Persson
- A new hydrogel could help pull drinking water from the air -PUBLISHED_BY-> Science News For Students
- Creation of quantum dots wins 2023 chemistry Nobel -PUBLISHED_BY-> Science News For Students
- These jellyfish can learn without brains -PUBLISHED_BY-> Science News For Students
- New ultrathin materials can pull climate-warming CO2 from the air -MENTIONS-> Linköping University

Entities: Science News For Students, carbon capture, New ultrathin materials can pull climate-warming CO2 from the air, MXenes, CO2, MBenes, Riverside group,
steel, boron, Science News Explores
</item 2>

<item 3>
New ultrathin materials can pull climate-warming CO2 from the air (Science News For Students, 2023-12-15T11:30:00+00:00)
Each MXene has slightly different properties, from how well it conducts electricity to how strong it is.

Scientists can even modify existing MXenes by treating them with different chemicals. This involves covering the MXene’s surface with new atoms, such as oxygen
and fluorine, to change its behavior. For more than a decade, scientists have been tailoring different MXenes for use in batteries and other energy storage
devices. But tweaking MXenes’ chemistry can also alter their ability to collect CO 2 .

In an October 4 paper in Chem, engineers at the University of California, Riverside, reviewed the potential widespread use of MXenes. In a statement, lead
author Mihri Ozkan noted, “Their unique properties make them excellent candidates for capturing CO 2 .”

Painting on a CO 2 -absorber

Freshly prepared MXenes appear as a dark-colored powder. Each fleck contains many MXene sheets. Under a microscope, they look like loose sheets of paper.

Graph facts:
- These jellyfish can learn without brains -PUBLISHED_BY-> Science News For Students
- A new hydrogel could help pull drinking water from the air -PUBLISHED_BY-> Science News For Students
- Creation of quantum dots wins 2023 chemistry Nobel -PUBLISHED_BY-> Science News For Students
- New ultrathin materials can pull climate-warming CO2 from the air -MENTIONS-> Per Persson
- New ultrathin materials can pull climate-warming CO2 from the air -MENTIONS-> Linköping University

Entities: Science News For Students, New ultrathin materials can pull climate-warming CO2 from the air, MXenes, Chem, Mihri Ozkan, energy storage devices,
fluorine, University of California, Riverside, CO2, batteries, oxygen
</item 3>

<item 4>
New ultrathin materials can pull climate-warming CO2 from the air (Science News For Students, 2023-12-15T11:30:00+00:00)
For comparison, a sheet of paper is 100,000 times thicker.

MXenes normally come as stacks of these individual sheets. The many separate sheets inside a stack add up to a huge overall surface area. It’s like how the
combined surface area of all the pages in a book is much larger than that of just the covers.

In fact, the total surface area contained in one gram (1/400 ounce) of MXene “can cover a football field,” notes Per Persson. He’s a materials scientist at
Linköping University in Sweden.

Atoms across MXenes’ large surface areas can interact with their environment. That includes grabbing carbon dioxide, or CO 2, out of the air around them — a
process called carbon capture.

The white exhaust leaving smokestacks at this coal-fired power plant in Arizona is largely steam. Not visible is the carbon dioxide gas also leaving those
stacks.

Graph facts:
- New ultrathin materials can pull climate-warming CO2 from the air -MENTIONS-> carbon dioxide
- A new hydrogel could help pull drinking water from the air -PUBLISHED_BY-> Science News For Students
- Creation of quantum dots wins 2023 chemistry Nobel -PUBLISHED_BY-> Science News For Students
- These jellyfish can learn without brains -PUBLISHED_BY-> Science News For Students
- New ultrathin materials can pull climate-warming CO2 from the air -MENTIONS-> Persson

Entities: Science News For Students, Per Persson, Linköping University, carbon dioxide, carbon capture, coal-fired power plant, New ultrathin materials can pull
climate-warming CO2 from the air, MXenes, Arizona, Sweden
</item 4>

<item 5>
New ultrathin materials can pull climate-warming CO2 from the air (Science News For Students, 2023-12-15T11:30:00+00:00)
Several research groups have been exploring MXenes’ potential to capture CO 2 from the air. In the Linköping experiments, “CO 2 stuck to the MXene surface like
refrigerator magnets,” Persson says. “It was absolutely amazing.” His group has shown that MXenes can sponge up as much as half their weight in CO 2 .

One large family

MXenes don’t exist in nature. Scientists have to make them.

This is freshly produced MXene, which normally forms a dark powder when dry. This MXene contains alternating layers of vanadium (M) and carbon (X) atoms.
Karamullah Eisawi/Tulane Univ.

They do this using other labmade compounds called MAX phases. Like MXenes, MAX phases are layered crystals. But MAX phases have extra sheets of metal “A” that
sandwich between their M and X layers.

To turn MAX phases into MXenes, scientists must eat away the A layers using a strong acid.

Graph facts:
- Persson -SAID-> carbon dioxide
- A new hydrogel could help pull drinking water from the air -PUBLISHED_BY-> Science News For Students
- Creation of quantum dots wins 2023 chemistry Nobel -PUBLISHED_BY-> Science News For Students
- These jellyfish can learn without brains -PUBLISHED_BY-> Science News For Students
- Persson -SAID-> MXenes

Entities: Science News For Students, Persson, Linköping experiments, Linköping, Karamullah Eisawi, Tulane Univ, vanadium, carbon, strong acid, New ultrathin
materials can pull climate-warming CO2 from the air, MXenes, MAX phases, CO2
</item 5>

<item 6>
New ultrathin materials can pull climate-warming CO2 from the air (Science News For Students, 2023-12-15T11:30:00+00:00)
MXenes could be heated up to release their CO 2 for long-term storage somewhere else. Flushing MXenes with a different kind of gas, such as hydrogen, might also
remove the CO 2 from them. Then the MXenes could be reused for a new round of carbon capture.

MXenes could also help transform the CO 2 they capture into other useful compounds, Stavila says. That’s because MXenes are catalytic. That means they can help
drive other chemicals, such as CO 2 , to react. MXenes also conduct electricity. Since they like sharing electrons with CO 2 , they can ferry those electrons to
other compounds.

As a result, “you can do [transformations of CO 2 ] that may not be otherwise possible,” Stavila says. For instance, with a boost from MXenes, CO 2 can react
with hydrogen to form methanol or formic acid.

Graph facts:
- New ultrathin materials can pull climate-warming CO2 from the air -MENTIONS-> Per Persson
- A new hydrogel could help pull drinking water from the air -PUBLISHED_BY-> Science News For Students
- Creation of quantum dots wins 2023 chemistry Nobel -PUBLISHED_BY-> Science News For Students
- These jellyfish can learn without brains -PUBLISHED_BY-> Science News For Students
- New ultrathin materials can pull climate-warming CO2 from the air -MENTIONS-> Linköping University

Entities: Science News For Students, carbon capture, New ultrathin materials can pull climate-warming CO2 from the air, CO2, MXenes, long-term storage,
hydrogen, methanol, formic acid, Stavila, transformations of CO2
</item 6>

<item 7>
Graph query row:
article: A scientist reckons with climate grief
publisher: Yahoo News
subject: Peter Kalmus
climate_or_carbon_connection: climate anxiety
</item 7>

<item 8>
New ultrathin materials can pull climate-warming CO2 from the air (Science News For Students, 2023-12-15T11:30:00+00:00)
So water and oxygen make MXenes less reactive. But scientists think they can overcome this problem by tweaking MXenes’ structure. Capping the materials’ edges
with bulky salts, for instance, could shield MXenes from damage by oxygen.

It’s also hard, right now, to make large amounts of MXenes. In the lab, researchers can make only several kilograms (pounds) in a single batch. Several
companies are working out how to boost batch sizes.

But Thakur points out one irony: Making MXenes for carbon capture can have a big carbon footprint. It takes a lot of electricity to get the heat needed to make
them, she notes. Much of that electricity is made using CO 2 -spewing fossil fuels. It’s important to make sure that the energy used to make these materials
doesn’t spew more CO 2 than the MXenes can later sop up. Otherwise, that would defeat their use in carbon capture.

Graph facts:
- New ultrathin materials can pull climate-warming CO2 from the air -MENTIONS-> Per Persson
- A new hydrogel could help pull drinking water from the air -PUBLISHED_BY-> Science News For Students
- Creation of quantum dots wins 2023 chemistry Nobel -PUBLISHED_BY-> Science News For Students
- These jellyfish can learn without brains -PUBLISHED_BY-> Science News For Students
- New ultrathin materials can pull climate-warming CO2 from the air -MENTIONS-> Linköping University

Entities: Science News For Students, carbon capture, New ultrathin materials can pull climate-warming CO2 from the air, MXenes, Thakur, CO2, oxygen, carbon
footprint, water, bulky salts, electricity, fossil fuels
</item 8>

<item 9>
New ultrathin materials can pull climate-warming CO2 from the air (Science News For Students, 2023-12-15T11:30:00+00:00)
These materials are harder to make than MXenes. So most research on them has been done with math or computer models. But MBenes could have unique mechanical and
electrical traits, Ozkan’s team finds. And those just might help them last through many, many cycles of CO 2 capture and release.

Each year, human activities spew tens of billions of tons of CO 2 into the air. That’s equivalent to the weight of tens of billions of elephants. Carbon
absorbers, including MXenes or MBenes, can’t catch it all. As such, they’re never going to replace the need for reducing emissions of CO 2 and other greenhouse
gases.

Still, given the urgency of climate change, even a quick or partial fix — such as carbon capture — can’t be ignored. “At this point in time,” says Persson, “we
need to look at all the possible solutions.” And MXene-like materials are emerging as potentially important candidates.

Graph facts:
- Persson -SAID-> carbon dioxide
- A new hydrogel could help pull drinking water from the air -PUBLISHED_BY-> Science News For Students
- Creation of quantum dots wins 2023 chemistry Nobel -PUBLISHED_BY-> Science News For Students
- These jellyfish can learn without brains -PUBLISHED_BY-> Science News For Students
- Persson -SAID-> MXenes

Entities: Science News For Students, carbon capture, Persson, New ultrathin materials can pull climate-warming CO2 from the air, MXenes, CO2, Ozkan, Ozkan’s
team, MBenes, greenhouse gases, all the possible solutions, climate change
</item 9>

<item 10>
New ultrathin materials can pull climate-warming CO2 from the air (Science News For Students, 2023-12-15T11:30:00+00:00)
This is another in our series of stories identifying new technologies and actions that can slow climate change, reduce its impacts or help communities cope with
a rapidly changing world.

One up-and-coming class of tiny materials could help solve the huge problem of climate change. How? By catching climate-warming carbon dioxide out of the air.

The nano-scale materials are called MXenes (MAX-eenz). Their name comes from their structure, which is made up of alternating layers of M and X atoms. “M”
layers consist of some transition metal. (Those atoms are bulky and can shed a variable number of electrons.) “X” layers contain atoms of another element, such
as carbon or nitrogen.

A single MXene sheet contains a few of these alternating M and X layers chemically bound to each other. It may look like MXM or MXMXM. Each sheet is typically
only about 1 nanometer thick. For comparison, a sheet of paper is 100,000 times thicker.

Graph facts:
- These jellyfish can learn without brains -PUBLISHED_BY-> Science News For Students
- A new hydrogel could help pull drinking water from the air -PUBLISHED_BY-> Science News For Students
- Creation of quantum dots wins 2023 chemistry Nobel -PUBLISHED_BY-> Science News For Students
- Patterns of Surface Warming Matter for Climate Sensitivity -MENTIONS-> carbon dioxide
- 'They seemed primed to take over': How the Great Dying doomed the 'beast tooth' and set the stage for the dawn of the dinosaurs -MENTIONS-> carbon dioxide

Entities: Science News For Students, carbon dioxide, New ultrathin materials can pull climate-warming CO2 from the air, MXenes, climate change
</item 10>
=== END ITEM id=ff2b6bc1c24d368e ===
