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

=== ITEM id=15a01ff08f139f0d ===
Gold sentences and facts:
Sentence 0: These delays may lead to bets being refunded depending on the circumstances and the specific rules of your sportsbook.
  Fact 0.0: These delays may lead to bets being refunded.
  Fact 0.1: Whether bets are refunded depends on the circumstances.
  Fact 0.2: Whether bets are refunded depends on the specific rules of the sportsbook.
Sentence 1: Every sportsbook jumps on board when a line moves or odds change for a legitimate reason, like an injury or roster move.
  Fact 1.0: Every sportsbook jumps on board when a line moves or odds change for a legitimate reason.
  Fact 1.1: An injury or roster move is an example of a legitimate reason for a line or odds change.
Sentence 2: But sportsbooks are also gathering this information, and they begin to tighten the NBA Rookie of the Year lines in response.
  Fact 2.0: Sportsbooks are also gathering this information.
  Fact 2.1: Sportsbooks begin to tighten the NBA Rookie of the Year lines in response.

Retrieved items:
<item 1>
Moneyline Betting - What is the Moneyline and How Does it Work? (Sporting News, 2023-11-01T21:25:48+00:00)
New Information: Changes in factors like player injuries, player substitutions, or even changes in weather conditions can affect the perceived outcome of an
event. When such information becomes public, sportsbooks may adjust the odds accordingly.

Changes in factors like player injuries, player substitutions, or even changes in weather conditions can affect the perceived outcome of an event. When such
information becomes public, sportsbooks may adjust the odds accordingly. Professional Bettors: Large wagers from sharp bettors or professional gamblers can
cause the lines to shift. These individuals often have insights or information that can influence the odds.

Large wagers from sharp bettors or professional gamblers can cause the lines to shift. These individuals often have insights or information that can influence
the odds. Initial Miscalculation: Sometimes, the initial odds set by oddsmakers may be perceived as offering value to bettors. In such cases, heavy betting on
one side might indicate that the initial odds were potentially "off," prompting an adjustment.

Graph facts:
- Moneyline Betting - What is the Moneyline and How Does it Work? -MENTIONS-> MLS
- Moneyline Betting - What is the Moneyline and How Does it Work? -MENTIONS-> New York Giants
- Moneyline Betting - What is the Moneyline and How Does it Work? -MENTIONS-> NFL
- Moneyline Betting - What is the Moneyline and How Does it Work? -MENTIONS-> Dallas Cowboys
- Moneyline Betting - What is the Moneyline and How Does it Work? -MENTIONS-> moneyline bet

Entities: moneyline betting, Moneyline Betting - What is the Moneyline and How Does it Work?, Player injuries, Player substitutions, Professional bettors,
Professional gamblers, Lines, Oddsmakers, Initial miscalculation, Sporting News, sportsbooks, betting, Odds, Weather Conditions
</item 1>

<item 2>
Point Spread Betting - What is a Point Spread Bet? (Sporting News, 2023-11-01T22:50:52+00:00)
Do point spread odds change?

Yes, point spread odds can change, and these shifts are commonly referred to as "line movement."

Lines move when the point spread or the odds associated with a particular bet change after sportsbooks have initially set them. There are several reasons for
these adjustments.

One of the most common causes is an imbalance in the amount of money being wagered on one side of the bet. If a significant majority of bettors are placing
their money on one team to cover the spread, the top sports betting apps will adjust the point spread to make the other side more appealing, thereby balancing
the betting action.

Another factor that can affect the spread odds is the occurrence of unexpected events, such as injuries to key players, changes in team strategies, or even
significant weather events that might impact a game's outcome.

For instance, if a star quarterback is announced as injured just days before a big game, the point spread might shift in favor of the opposing team.

Moreover, information and public perception play a role.

Graph facts:
- UTSA vs. Marshall odds, props, predictions: Roadrunners lay big spread in Frisco Bowl -PUBLISHED_BY-> Sporting News
- North Texas vs. SMU odds, props, predictions: Red hot Mustangs could overpower not-so-Mean Green -PUBLISHED_BY-> Sporting News
- Chiefs vs. Packers live score, updates, highlights from NFL 'Sunday Night Football' game -PUBLISHED_BY-> Sporting News
- Jaguars vs. Saints live score, updates, highlights from NFL 'Thursday Night Football' game -PUBLISHED_BY-> Sporting News
- Raiders vs. Lions live score, updates, highlights from NFL 'Monday Night Football' game -PUBLISHED_BY-> Sporting News

Entities: Sporting News, injuries, point spread odds, sportsbooks, sports betting apps, team strategies, weather events, game outcome, star quarterback, big
game, public perception, information, Point Spread Betting - What is a Point Spread Bet?, betting, line movement
</item 2>

<item 3>
MLB Betting Sites & Apps - The Best Baseball Sportsbooks 2023 (Sporting News, 2023-10-24T22:32:27+00:00)
By doing this research, you’ll find the best odds for the bets you were already planning on making, which means you can maximize your potential return on your
wager.

This can be a time-consuming process, but it also helps you boost your bankroll and give yourself the largest possible profit.

MLB betting rules at online sportsbooks

Every sportsbook has terms and conditions for all bets, MLB included. While those rules may be specific to each sportsbook, there are some general rules that
should apply everywhere.

For example, MLB First 5 Innings bets are usually considered actionable after five innings of play, or 4.5 innings in the case of the home team having the lead.

Another rule many sportsbooks follow for baseball bets is weather-related, such as if a game is suspended due to rain or other weather conditions. These delays
may lead to bets being refunded depending on the circumstances and the specific rules of your sportsbook.

Here are some other rules found at online sportsbooks for MLB betting.

Wagers include extra innings unless specified otherwise.

Graph facts:
- MLB Betting Sites & Apps - The Best Baseball Sportsbooks 2023 -MENTIONS-> MLB
- MLB Betting Sites & Apps - The Best Baseball Sportsbooks 2023 -MENTIONS-> odds and lines
- MLB Betting Sites & Apps - The Best Baseball Sportsbooks 2023 -MENTIONS-> totals bets
- MLB Betting Sites & Apps - The Best Baseball Sportsbooks 2023 -MENTIONS-> Caesars
- MLB Betting Sites & Apps - The Best Baseball Sportsbooks 2023 -MENTIONS-> bet365 app

Entities: MLB Betting Sites & Apps - The Best Baseball Sportsbooks 2023, MLB betting, online sportsbooks, MLB First 5 Innings bets, weather-related betting
rules, wagers, Sporting News, baseball
</item 3>

<item 4>
Moneyline Betting - What is the Moneyline and How Does it Work? (Sporting News, 2023-11-01T21:25:48+00:00)
This includes your initial $10 wager, along with a profit of $15.30.

It's essential to note that this calculation gives you the total payout, not just your earnings. If you're keen on tracking your profits from betting, remember
to subtract your original stake from the payout to get an accurate figure.

Why do moneyline odds change?

Lines at the best sports betting apps change quickly due to two primary reasons:

Public betting

New information

Before setting odds for an event, oddsmakers gather extensive data. Once they finalize the odds, they release them to the public.

As a bettor, you can bet immediately or wait for potential shifts in the odds, although predicting these shifts is challenging.

For instance, when examining odds for the next Super Bowl champion released shortly after the previous Super Bowl, these odds are based mostly on the recently
concluded season.

As the new season approaches and teams make changes, odds adjust based on perceived championship chances. As games progress, lines continue to evolve based on
team performances.

Public betting can also sway lines.

Graph facts:
- Moneyline Betting - What is the Moneyline and How Does it Work? -MENTIONS-> Dallas Cowboys
- Moneyline Betting - What is the Moneyline and How Does it Work? -MENTIONS-> MLS
- Moneyline Betting - What is the Moneyline and How Does it Work? -MENTIONS-> New York Giants
- Moneyline Betting - What is the Moneyline and How Does it Work? -MENTIONS-> NFL
- Moneyline Betting - What is the Moneyline and How Does it Work? -MENTIONS-> moneyline bet

Entities: moneyline betting, Moneyline odds, Public betting, New information, Moneyline Betting - What is the Moneyline and How Does it Work?, Super Bowl,
Sporting News, sports betting apps
</item 4>

<item 5>
MLB Betting Sites & Apps - The Best Baseball Sportsbooks 2023 (Sporting News, 2023-10-24T22:32:27+00:00)
Wagers include extra innings unless specified otherwise.

The game must start on the scheduled day (local stadium time) for bets to have action.

The game must go 9 innings or 8½ if the home team is winning for the run line to have action.

On ‘team to score first’ wagers, once a run is scored, the prop will be graded, regardless of how long the game lasts.

Wagers on postseason baseball series are based on the team to win the series, regardless of the number of games involved.

For player props, the player must be in the starting lineup for wagers to have action.

It’s important to read through the rules for baseball betting before placing your MLB bets, and you can find them in the terms and conditions section of your
sportsbook’s website.

Related Pages:
</item 5>

<item 6>
Totals Betting - What is an Over/Under Bet? (Sporting News, 2023-11-01T22:08:31+00:00)
Bookmakers adjust the line to balance the action and protect themselves from potential losses.

One of the most common reasons for line movement is a significant amount of money being wagered on one side (either over or under). Bookmakers adjust the line
to balance the action and protect themselves from potential losses. Injury Updates: If a key player is injured or unable to play, this can influence the
expected total score, leading bookmakers to adjust the line accordingly.

If a key player is injured or unable to play, this can influence the expected total score, leading bookmakers to adjust the line accordingly. Weather
Conditions: In outdoor sports, weather can play a significant role. For example, heavy rain or snow in a football game might lead to expectations of a
lower-scoring game, causing the line to move down.

In outdoor sports, weather can play a significant role. For example, heavy rain or snow in a football game might lead to expectations of a lower-scoring game,
causing the line to move down.

Graph facts:
- UTSA vs. Marshall odds, props, predictions: Roadrunners lay big spread in Frisco Bowl -PUBLISHED_BY-> Sporting News
- North Texas vs. SMU odds, props, predictions: Red hot Mustangs could overpower not-so-Mean Green -PUBLISHED_BY-> Sporting News
- Chiefs vs. Packers live score, updates, highlights from NFL 'Sunday Night Football' game -PUBLISHED_BY-> Sporting News
- Jaguars vs. Saints live score, updates, highlights from NFL 'Thursday Night Football' game -PUBLISHED_BY-> Sporting News
- Raiders vs. Lions live score, updates, highlights from NFL 'Monday Night Football' game -PUBLISHED_BY-> Sporting News

Entities: Sporting News, totals betting, over/under bet, line movement, Injury Updates, Weather Conditions, Totals Betting - What is an Over/Under Bet?,
football, outdoor sports
</item 6>

<item 7>
Moneyline Betting - What is the Moneyline and How Does it Work? (Sporting News, 2023-11-01T21:25:48+00:00)
Sometimes, the initial odds set by oddsmakers may be perceived as offering value to bettors. In such cases, heavy betting on one side might indicate that the
initial odds were potentially "off," prompting an adjustment. Event Dynamics: For live betting, where bets are placed while the event is in progress, the
evolving dynamics of the match can lead to frequent changes in the odds.

Sportsbooks adjust moneyline odds to ensure they remain profitable, balance their book, and reflect the most current information about the event.

Related Pages:

Graph facts:
- Moneyline Betting - What is the Moneyline and How Does it Work? -MENTIONS-> MLS
- Moneyline Betting - What is the Moneyline and How Does it Work? -MENTIONS-> New York Giants
- Moneyline Betting - What is the Moneyline and How Does it Work? -MENTIONS-> NFL
- Moneyline Betting - What is the Moneyline and How Does it Work? -MENTIONS-> Dallas Cowboys
- Moneyline Betting - What is the Moneyline and How Does it Work? -MENTIONS-> moneyline bet

Entities: moneyline betting, Moneyline Betting - What is the Moneyline and How Does it Work?, Oddsmakers, Bettors, Sporting News, sportsbooks, live betting,
Odds
</item 7>

<item 8>
Point Spread Betting - What is a Point Spread Bet? (Sporting News, 2023-11-01T22:50:52+00:00)
Moreover, information and public perception play a role. If a piece of news or a prevailing sentiment starts to gain traction about a particular team's
increased chances of winning, this can influence how people bet, leading sportsbooks to adjust their lines accordingly.

Point spread odds can and do change based on a combination of actual betting patterns, unexpected events, and shifts in public perception. Savvy bettors keep a
close eye on these line moves, as they can offer insights into where the value lies in a particular bet.

Graph facts:
- UTSA vs. Marshall odds, props, predictions: Roadrunners lay big spread in Frisco Bowl -PUBLISHED_BY-> Sporting News
- North Texas vs. SMU odds, props, predictions: Red hot Mustangs could overpower not-so-Mean Green -PUBLISHED_BY-> Sporting News
- Chiefs vs. Packers live score, updates, highlights from NFL 'Sunday Night Football' game -PUBLISHED_BY-> Sporting News
- Jaguars vs. Saints live score, updates, highlights from NFL 'Thursday Night Football' game -PUBLISHED_BY-> Sporting News
- Raiders vs. Lions live score, updates, highlights from NFL 'Monday Night Football' game -PUBLISHED_BY-> Sporting News

Entities: Sporting News, point spread betting, point spread bet, point spread odds, sportsbooks, public perception, Betting patterns, bettors, Point Spread
Betting - What is a Point Spread Bet?
</item 8>

<item 9>
Graph query row:
article_title: Line Shopping in Sports Betting - How to Find the Best Odds
subject_relationship: MENTIONS
subject: sportsbook apps
mechanism_relationship: MENTIONS
mechanism: odds and lines
cause_relationship: MENTIONS
cause: injury
passage: Line Shopping in Sports Betting - How to Find the Best Odds (Sporting News, 2023-11-06T22:21:05+00:00)
Line shopping is the key that unlocks the best odds and lines at top sports betting sites.

The process is as simple as checking the lines and odds for your prospective bet at multiple online sports betting sites. The result is quite often better odds
and improved lines.

What is line shopping, and why is it worth your time? Find the answers below in our complete guide to line shopping.

How do sportsbooks set odds and lines?

The best sportsbooks employ oddsmakers and oddsmaking services to set their lines and odds.

Other sports betting apps simply copy those lines. The oddsmakers and oddsmaking services use computer models and algorithms that consider analytics,
statistics, injury and roster news, performance, matchup data, and more to develop the lines and odds.

Betting lines and odds are then subject to change based on the betting public’s reaction and any additional information that comes to light ahead of the game.
Sportsbooks will move the line or change the odds if the public bets heavily on one side.
</item 9>

<item 10>
Line Shopping in Sports Betting - How to Find the Best Odds (Sporting News, 2023-11-06T22:21:05+00:00)
Sportsbooks will move the line or change the odds if the public bets heavily on one side.

Every sportsbook’s goal is to even out the betting on both sides of every bet. That allows sportsbooks to earn the profit built into the odds and lines without
worrying about which side wins.

The betting is rarely the same at every sportsbook. While one sportsbook may be forced to move a line or change the odds because of betting, another won’t.
That’s when you’ll spot a difference between sportsbooks. This difference may allow you to book a bet at improved odds or a more favorable line.

The lines and odds may even be different right from the jump. Either way, the differences are why line shopping always makes sense. Shop around, and you can
find opportunities to lay less, win more, and improve your chances.

Graph facts:
- Line Shopping in Sports Betting - How to Find the Best Odds -MENTIONS-> live odds feed
- Line Shopping in Sports Betting - How to Find the Best Odds -MENTIONS-> NFL
- Line Shopping in Sports Betting - How to Find the Best Odds -MENTIONS-> odds and lines
- Line Shopping in Sports Betting - How to Find the Best Odds -MENTIONS-> How to bet on sports in the US
- Line Shopping in Sports Betting - How to Find the Best Odds -MENTIONS-> sportsbook apps

Entities: line shopping, Line Shopping in Sports Betting - How to Find the Best Odds, Sporting News, sportsbooks, sports betting
</item 10>
=== END ITEM id=15a01ff08f139f0d ===

=== ITEM id=f774d97ccbb00117 ===
Gold sentences and facts:
Sentence 0: Harbaugh reportedly watched the game from the team hotel, and now the leading piece of drama returns.
  Fact 0.0: Harbaugh reportedly watched the game from the team hotel.
  Fact 0.1: The leading piece of drama now returns.
Sentence 1: The Big Ten suspended coach Jim Harbaugh for three games on Friday with the team on a plane en route to Penn State, and although Michigan had hoped to block the suspension, the Wolverines will be without their coach on the sidelines in Happy Valley.
  Fact 1.0: The Big Ten suspended coach Jim Harbaugh for three games.
  Fact 1.1: The Big Ten suspended Harbaugh on Friday.
  Fact 1.2: The team was on a plane en route to Penn State when the suspension happened.
  Fact 1.3: Michigan had hoped to block the suspension.
  Fact 1.4: The Wolverines will be without their coach on the sidelines in Happy Valley.

Retrieved items:
<item 1>
Michigan sign-stealing scandal news: Latest updates on possible Jim Harbaugh suspension from NCAA cheating investigation (Sporting News,
2023-11-06T23:58:04+00:00)
Within it, the school warned Big Ten commissioner Tony Pettiti against levying "premature" disciplinary action against Jim Harbaugh and alleged that in-person
scouting and sign-stealing are commonplace across the NCAA.

"The conference should act cautiously when setting precedent given the reality that in-person scouting, collusion among opponents, and other questionable
practices may well be far more prevalent than believed," the letter read.

Michigan also included with the letter "two attached documents of Michigan's offensive and defensive signs allegedly stolen and shared by other Big Ten teams,"
per Yahoo Sports.

Conference awaits Michigan response

The Big Ten expects a formal response from Michigan to the notice of discipline by the end of the day Wednesday, ESPN's Pete Thamel reports. His colleague Adam
Rittenberg adds that a ruling is expected to come down Thursday.

Sources: The Big Ten expects a response from Michigan on the notice of discipline by the end of day on Wednesday. Don’t expect any ruling by the Big Ten on
Wednesday, as they are expected to take time to absorb Michigan’s response.

Graph facts:
- Michigan sign-stealing scandal winners and losers, from Jim Harbaugh to Tony Petitti -MENTIONS-> Michigan
- After a Stunning Snub, College Football Is Facing an Identity Crisis -MENTIONS-> Michigan
- 2024 NFL Mock Draft: Bengals' Joe Burrow gets another playmaker; four QBs go in Round 1, but Pats opt for WR -MENTIONS-> Michigan
- USC basketball futures odds to win Pac-12, NCAA tourney ahead of Bronny James collegiate debut -MENTIONS-> Michigan
- Michigan conquers Ohio State again, but the story is far from over for 2023 Wolverines -MENTIONS-> Michigan

Entities: Michigan, in-person scouting, Michigan sign-stealing scandal, Big Ten, Yahoo Sports, Jim Harbaugh, NCAA cheating investigation, Adam Rittenberg,
Michigan sign-stealing scandal news: Latest updates on possible Jim Harbaugh suspension from NCAA cheating investigation, Tony Pettiti, collusion among
opponents, questionable practices, Pete Thamel, ESPN, Sporting News
</item 1>

<item 2>
Michigan sign-stealing scandal news: Latest updates on possible Jim Harbaugh suspension from NCAA cheating investigation (Sporting News,
2023-11-06T23:58:04+00:00)
17

Michigan fires coach connected to destroying evidence in investigation

Yahoo Sports' Ross Dellenger and Dan Wetzel reported on Friday that the NCAA's investigation into the sign-stealing and advanced scouting operation has found
that Stalions was at least partially funded by a Michigan booster, named as "Uncle T," and that linebackers coach Chris Partridge, who was fired by Michigan on
Friday, is alleged to have destroyed evidence on a computer pertaining to the sign-stealing allegations.

The report states Partridge is not believed to have had prior knowledge of the sign-stealing operation, but then later worked to cover it up. In a statement on
his firing, Michigan appeared to indicate a connection between Partridge and the NCAA's investigation, which Dellenger and Wetzel reported is still in the early
stage:

From the outset, our focus has been on seeking due process and allowing the NCAA to conduct a fair and deliberate investigation.

Graph facts:
- Michigan sign-stealing scandal winners and losers, from Jim Harbaugh to Tony Petitti -MENTIONS-> Michigan
- After a Stunning Snub, College Football Is Facing an Identity Crisis -MENTIONS-> Michigan
- 2024 NFL Mock Draft: Bengals' Joe Burrow gets another playmaker; four QBs go in Round 1, but Pats opt for WR -MENTIONS-> Michigan
- USC basketball futures odds to win Pac-12, NCAA tourney ahead of Bronny James collegiate debut -MENTIONS-> Michigan
- Michigan conquers Ohio State again, but the story is far from over for 2023 Wolverines -MENTIONS-> Michigan

Entities: Michigan, Yahoo Sports, Ross Dellenger, Dan Wetzel, sign-stealing and advanced scouting operation, Stalions, Uncle T, Chris Partridge, destroying
evidence, NCAA investigation into the sign-stealing and advanced scouting operation, cover-up, Jim Harbaugh, Michigan sign-stealing scandal news: Latest updates
on possible Jim Harbaugh suspension from NCAA cheating investigation, Sporting News, NCAA
</item 2>

<item 3>
Michigan sign-stealing scandal news: Latest updates on possible Jim Harbaugh suspension from NCAA cheating investigation (Sporting News,
2023-11-06T23:58:04+00:00)
— Nicole Auerbach (@NicoleAuerbach) November 11, 2023

Harbaugh hearing coming down to wire before Penn State game

ESPN's Pete Thamel reported it's unclear Harbaugh will be granted a hearing in time to be able to coach against Penn State. With kickoff scheduled for noon ET,
Thamel said "the drama's going to go right up 'till noon."

Latest from ⁦@CollegeGameDay⁩ on Jim Harbaugh’s TRO legal filing. Will he coach Michigan at Penn State today? pic.twitter.com/bz3R1eYxI9 — Pete Thamel
(@PeteThamel) November 11, 2023

The case is reportedly in the hands of Judge Carol Kunhke, a Michigan alumnus.

Harbaugh was not on the Michigan team bus with other coaches and players when it arrived at Beaver Stadium Saturday morning.

Michigan players and coaches arrive. No Jim Harbaugh on the bus.

Graph facts:
- Michigan sign-stealing scandal winners and losers, from Jim Harbaugh to Tony Petitti -MENTIONS-> Michigan
- After a Stunning Snub, College Football Is Facing an Identity Crisis -MENTIONS-> Michigan
- 2024 NFL Mock Draft: Bengals' Joe Burrow gets another playmaker; four QBs go in Round 1, but Pats opt for WR -MENTIONS-> Michigan
- USC basketball futures odds to win Pac-12, NCAA tourney ahead of Bronny James collegiate debut -MENTIONS-> Michigan
- Michigan conquers Ohio State again, but the story is far from over for 2023 Wolverines -MENTIONS-> Michigan

Entities: Michigan, Nicole Auerbach, Penn State, Michigan sign-stealing scandal, Beaver Stadium, ESPN, Jim Harbaugh, NCAA cheating investigation, Carol Kunhke,
Harbaugh hearing, TRO legal filing, Penn State game, Michigan sign-stealing scandal news: Latest updates on possible Jim Harbaugh suspension from NCAA cheating
investigation, Pete Thamel, Sporting News
</item 3>

<item 4>
Michigan sign-stealing scandal news: Latest updates on possible Jim Harbaugh suspension from NCAA cheating investigation (Sporting News,
2023-11-06T23:58:04+00:00)
— Pete Thamel (@PeteThamel) November 8, 2023

Tuesday, Nov. 7

Michigan says three other Big Ten schools shared decoded Wolverines signals

While Michigan tries to navigate the storm around its program, the Wolverines are also trying to make one thing clear: they're hardly alone in trying to
decipher opponents' signals.

Michigan has shared evidence with the Big Ten and NCAA showing staff members at Ohio State, Purdue and Rutgers shared intelligence on the Wolverines' signals
prior to last year's Michigan-Purdue Big Ten Championship Game, The Athletic reported.

Documents breaking down Michigan's signals for numerous plays were compiled by Purdue staffers ahead of the game based on information provided by Ohio State and
Rutgers football personnel, the report says. The person who supplied the documents told The Athletic he spent 10 to 12 hours per week trying to decode signals
based on game video and said the practice of sharing decoded signals with other schools is common.

Graph facts:
- Which M.L.B. Playoff Team Has the Most Daunting Home Field? We Ranked Them. -MENTIONS-> The Athletic
- Michigan sign-stealing scandal winners and losers, from Jim Harbaugh to Tony Petitti -MENTIONS-> The Athletic
- Browns survive Bears Hail Mary on wild, near-catch by Darnell Mooney: 'Damn near s— myself' -MENTIONS-> The Athletic
- He’s Hockey’s Brightest Young Star. This Is What Makes His Shot So Special. -MENTIONS-> The Athletic
- How Did the Houston Astros Cheat in 2017? Everything to Know About a Dark Moment in MLB History -MENTIONS-> The Athletic

Entities: The Athletic, Michigan, Rutgers, Big Ten, Jim Harbaugh, NCAA cheating investigation, Michigan sign-stealing scandal news: Latest updates on possible
Jim Harbaugh suspension from NCAA cheating investigation, Michigan-Purdue Big Ten Championship Game, decoded signals, Pete Thamel, Ohio State, Sporting News,
NCAA, Purdue
</item 4>

<item 5>
Michigan sign-stealing scandal news: Latest updates on possible Jim Harbaugh suspension from NCAA cheating investigation (Sporting News,
2023-11-06T23:58:04+00:00)
Michigan noted that regarding evidence, there is nothing it is aware of to claim athletics personnel at Michigan committed the allegations, and said the rule
pertaining to electronic recordings of games is limited to the use of field equipment used by the institution for the game it is currently playing. It also said
Michigan has not seen much evidence, but rather has just heard a summary of second- and third-hand information.

The letter also said given the use of sign decoding by other universities, there is little reason to believe Stalions' efforts proved meaningful, and pointed
out the teams' average margin of victory since Stalions was suspended. It said the information obtained is "at most a reflection of information that teams can
and easily do obtain through public and permissible sources," and because of that it is "highly dubious" that Stalions' observations would have a "material
effect on the integrity of the competition."

Graph facts:
- Michigan sign-stealing scandal winners and losers, from Jim Harbaugh to Tony Petitti -MENTIONS-> Michigan
- After a Stunning Snub, College Football Is Facing an Identity Crisis -MENTIONS-> Michigan
- 2024 NFL Mock Draft: Bengals' Joe Burrow gets another playmaker; four QBs go in Round 1, but Pats opt for WR -MENTIONS-> Michigan
- USC basketball futures odds to win Pac-12, NCAA tourney ahead of Bronny James collegiate debut -MENTIONS-> Michigan
- Michigan conquers Ohio State again, but the story is far from over for 2023 Wolverines -MENTIONS-> Michigan

Entities: Michigan, sign-stealing scandal, Stalions, Jim Harbaugh, NCAA cheating investigation, sign decoding, electronic recordings of games, Michigan
sign-stealing scandal news: Latest updates on possible Jim Harbaugh suspension from NCAA cheating investigation, Sporting News, NCAA
</item 5>

<item 6>
Michigan sign-stealing scandal news: Latest updates on possible Jim Harbaugh suspension from NCAA cheating investigation (Sporting News,
2023-11-06T23:58:04+00:00)
Public reprimands/fines are common penalties under B1G sportsmanship policy. — Adam Rittenberg (@ESPNRittenberg) November 9, 2023

Michigan's 10-page letter to Big Ten released

Yahoo Sports' Dan Wetzel posted Michigan's full 10-page letter to the Big Ten on X (formerly Twitter). The letter outlines Michigan's belief that the Big Ten
would be acting too hastily in suspending Harbaugh or levying any punishment toward Michigan before the completion of a full NCAA investigation.

In the letter, Michigan said the university has not had the chance to review all the evidence and respond, which oversteps the due process allowed for Michigan.
It said Pettiti should not have unilateral authority to punish violations without first having a completed investigation and then a response from the Compliance
and Reinstatement Committee, which would recommend penalties for rules violations.

Graph facts:
- Michigan sign-stealing scandal winners and losers, from Jim Harbaugh to Tony Petitti -MENTIONS-> Michigan
- After a Stunning Snub, College Football Is Facing an Identity Crisis -MENTIONS-> Michigan
- 2024 NFL Mock Draft: Bengals' Joe Burrow gets another playmaker; four QBs go in Round 1, but Pats opt for WR -MENTIONS-> Michigan
- USC basketball futures odds to win Pac-12, NCAA tourney ahead of Bronny James collegiate debut -MENTIONS-> Michigan
- Michigan conquers Ohio State again, but the story is far from over for 2023 Wolverines -MENTIONS-> Michigan

Entities: Michigan, Big Ten, X, Dan Wetzel, Jim Harbaugh, due process, Adam Rittenberg, Yahoo Sports, Pettiti, Compliance and Reinstatement Committee, NCAA
cheating investigation, rules violations, Michigan's 10-page letter to the Big Ten, Michigan sign-stealing scandal news: Latest updates on possible Jim Harbaugh
suspension from NCAA cheating investigation, Sporting News
</item 6>

<item 7>
Michigan sign-stealing scandal news: Latest updates on possible Jim Harbaugh suspension from NCAA cheating investigation (Sporting News,
2023-11-06T23:58:04+00:00)
🗣️ "Bet" pic.twitter.com/ENB7O8MyPv — The Sporting News (@sportingnews) November 10, 2023

Michigan announces intention to dispute suspension

Michigan has released a statement expressing frustration with the Big Ten's process and says it plans to seek a court order to block Harbaugh's suspension from
taking effect.

NEWS: Michigan says it intends to seek a court order with Jim Harbaugh preventing disciplinary action from taking effect.



Full statement: pic.twitter.com/p92yUCLsTX — Matt Fortuna (@Matt_Fortuna) November 10, 2023

Big Ten claims no evidence Harbaugh was aware of scheme

While the Big Ten says it's "proven" that an illegal sign-stealing scheme existed, the conference could not prove that Harbaugh knew about it. So, why was
Harbaugh suspended? The Big Ten asserts that, because he is the head coach, he has a responsibility for preventing this kind of scheme from taking place.

Graph facts:
- Michigan sign-stealing scandal winners and losers, from Jim Harbaugh to Tony Petitti -MENTIONS-> Michigan
- After a Stunning Snub, College Football Is Facing an Identity Crisis -MENTIONS-> Michigan
- 2024 NFL Mock Draft: Bengals' Joe Burrow gets another playmaker; four QBs go in Round 1, but Pats opt for WR -MENTIONS-> Michigan
- USC basketball futures odds to win Pac-12, NCAA tourney ahead of Bronny James collegiate debut -MENTIONS-> Michigan
- Michigan conquers Ohio State again, but the story is far from over for 2023 Wolverines -MENTIONS-> Michigan

Entities: Michigan, Big Ten, Jim Harbaugh suspension, Jim Harbaugh, Matt Fortuna, court order, cheating investigation, Michigan sign-stealing scandal news:
Latest updates on possible Jim Harbaugh suspension from NCAA cheating investigation, Sporting News, sign-stealing scheme, NCAA
</item 7>

<item 8>
Michigan sign-stealing scandal news: Latest updates on possible Jim Harbaugh suspension from NCAA cheating investigation (Sporting News,
2023-11-06T23:58:04+00:00)
On the field, Michigan is in the midst of what could be a historic season. Off the field, the the focus has rarely been on the team's success.

An investigation into Michigan's alleged sign-stealing scheme, centered on now-fired staffer Connor Stalions, continues to evolve by the hour, with one update
after another adding another wrinkle to the saga that began in October. The Big Ten suspended coach Jim Harbaugh for three games on Friday with the team on a
plane en route to Penn State, and although Michigan had hoped to block the suspension, the Wolverines will be without their coach on the sidelines in Happy
Valley.

The Sporting News is tracking the latest updates from the Michigan sign-stealing scandal. Follow along with the newest developments below.

MORE ON MICHIGAN SIGN-STEALING:

Michigan sign-stealing scandal updates

Friday, Nov.

Graph facts:
- Michigan sign-stealing scandal winners and losers, from Jim Harbaugh to Tony Petitti -MENTIONS-> Michigan
- After a Stunning Snub, College Football Is Facing an Identity Crisis -MENTIONS-> Michigan
- 2024 NFL Mock Draft: Bengals' Joe Burrow gets another playmaker; four QBs go in Round 1, but Pats opt for WR -MENTIONS-> Michigan
- USC basketball futures odds to win Pac-12, NCAA tourney ahead of Bronny James collegiate debut -MENTIONS-> Michigan
- Michigan conquers Ohio State again, but the story is far from over for 2023 Wolverines -MENTIONS-> Michigan

Entities: Michigan, Penn State, Big Ten, Happy Valley, Michigan sign-stealing scheme investigation, Jim Harbaugh suspension, Jim Harbaugh, Michigan
sign-stealing scandal news: Latest updates on possible Jim Harbaugh suspension from NCAA cheating investigation, Connor Stalions, Sporting News, sign-stealing
scheme, NCAA
</item 8>

<item 9>
Graph query row:
article: Cowboys vs. 49ers live score, updates, highlights from NFL 'Sunday Night Football' game
published_on: 2023-10-08T22:47:03+00:00
game: None
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
harbaugh_game_involvement: not recorded as an event participant
</item 9>

<item 10>
Michigan sign-stealing scandal winners and losers, from Jim Harbaugh to Tony Petitti (Sporting News, 2023-11-11T22:50:13+00:00)
Will Harbaugh finally take the parachute to the NFL? Those are questions bound to come up at the end of the 2023 season.

BENDER: Michigan will need Jim Harbaugh against Ohio State

Big Ten

This is the fourth football scandal in less than a calendar year.

Three former Iowa football players and a student manager were charged in an illegal sports wagering scandal. Several student-athletes face allegations of
placing bets on games in which they played in. Northwestern coach Pat Fitzgerald was fired on July 10 after a hazing investigation into the program. Fitzgerald
is suing the university for wrongful termination. Michigan State fired Mel Tucker on Sept. 27 after allegations of sexual misconduct from Brenda Tracy, a
speaker on sexual assault awareness.

Now, Michigan has a scandal on the field, and they countered with allegations that Ohio State and Rutgers leaked Michigan's signs to Purdue in last year's Big
Ten championship game.

Graph facts:
- Brock Purdy -MENTIONS-> NFL
- Raiders vs. Lions live score, updates, highlights from NFL 'Monday Night Football' game -MENTIONS-> NFL
- Fans spot Travis Kelce wearing Taylor Swift-themed friendship bracelet before she attended his game -MENTIONS-> NFL
- Chiefs vs. Packers live score, updates, highlights from NFL 'Sunday Night Football' game -MENTIONS-> NFL
- Christmas Day preview: 49ers, Ravens square off in potential Super Bowl sneak peek -MENTIONS-> NFL

Entities: NFL, Michigan, Michigan sign-stealing scandal, Michigan sign-stealing scandal winners and losers, from Jim Harbaugh to Tony Petitti, Pat Fitzgerald,
Northwestern, Brenda Tracy, Mel Tucker, Rutgers, illegal sports wagering scandal, hazing investigation, wrongful termination, sexual assault awareness, leaking
Michigan's signs, Big Ten
</item 10>
=== END ITEM id=f774d97ccbb00117 ===

=== ITEM id=db8ba354ff27afd1 ===
Gold sentences and facts:
Sentence 0: If you've got your gift list handy, now's a good time to start checking it off so you won't be pressed for time or funds when Black Friday and Cyber Monday come around next month.
  Fact 0.0: If you have your gift list handy, now is a good time to start checking it off.
  Fact 0.1: Starting to check off the gift list now avoids being pressed for time or funds when Black Friday and Cyber Monday come around next month.
  Fact 0.2: Black Friday and Cyber Monday come around next month.
Sentence 1: “One of the hardest things is resentment can build up if you feel like you need to buy gifts for your family and you can barely make your student loan payments, so bring things up,” Alev said.
  Fact 1.0: Alev said one of the hardest things is that resentment can build up.
  Fact 1.1: Resentment can build up if you feel you need to buy gifts for your family and can barely make your student loan payments.
  Fact 1.2: Alev advised bringing things up.

Retrieved items:
<item 1>
The 5 biggest mistakes parents make at Christmastime, according to a parenting expert (FOX News - Lifestyle, 2023-12-09T19:46:55+00:00)
Pressures are high for parents and caregivers around the holidays.

That means there’s room for error, according to parenting expert Hannah Keeley, who offered up some of the biggest mistakes parents tend to make at
Christmastime.

In an email she sent to Fox News Digital, Virginia-based Keeley identified the top five holiday blunders that parents often make.

HOME FOR THE HOLIDAYS? HERE'S HOW TO AVOID STRESS AROUND FAMILY MEMBERS, ACCORDING TO EXPERTS

Here are the mistakes Keeley thinks parents should try to avoid.

Diving right in …

1. Creating unrealistic expectations

Parents who make a "must-do" Christmas list that’s a "mile long" at the "first sniff of peppermint mochas" are perhaps setting themselves up for failure — and
Keeley said she empathized with this instinct.

Doing the most could lead to a season of disappointment, she warned.

CHRISTMAS TREE SHOPPING?

Graph facts:
- After losing her leg to the flu, Virginia woman urges people to get vaccinated: ‘Don’t waste time’ -MENTIONS-> Virginia
- Biden’s AI EO hailed as broad, but not deep without legislation to match -MENTIONS-> Virginia
- The obsessive tormenter who made professors’ lives miserable -MENTIONS-> Virginia
- Best sportsbook bonus offers for NFL Monday Night Football Eagles vs. Seahawks: Claim over $5,000 in bonuses from Bet365, BetMGM, BetRivers, Caesars
Sportsbook, DraftKings and FanDuel -MENTIONS-> Virginia
- Sam Altman backs teens’ startup, Google unveils the Pixel 8 and TikTok tests an ad-free tier -MENTIONS-> Virginia

Entities: Virginia, Hannah Keeley, Christmastime, unrealistic expectations, holiday blunders, The 5 biggest mistakes parents make at Christmastime, according to
a parenting expert, Fox News, Fox News Digital, parenting
</item 1>

<item 2>
26-year-old with $145,000 in student debt can't afford Christmas. She's giving 'affordable gifts or combining gifts with my older sister' (Fortune,
2023-12-23T15:13:12+00:00)
She’s been looking into gifting a used book to her dad and because she loves painting and already has all the tools, she’s working on a painting for her mom.

Other affordable alternatives can be sending hand written notes to your loved ones, buying from thrift stores, hosting a cookie bake off or simply spending
quality time with your family.

___

The Associated Press receives support from Charles Schwab Foundation for educational and explanatory reporting to improve financial literacy. The independent
foundation is separate from Charles Schwab and Co. Inc. The AP is solely responsible for its journalism.

Graph facts:
- Sen. Robert Menendez, facing bribery charges, stumps bank CEOs by asking how much their firms paid in settlements for allegedly ripping off consumers
-PUBLISHED_BY-> Fortune
- Buckle up for more ‘turbulence’ in the housing market, BofA says—it’s a housing recession, 1980s style -PUBLISHED_BY-> Fortune
- Israel's blockade of Gaza means that a region the size of Philadelphia faces a real risk of starvation, says Peace Studies scholar -PUBLISHED_BY-> Fortune
- Gen Z just saw the worst war in Israel in 50 years break out all over the internet, but they were raised on 'disturbing images' -PUBLISHED_BY-> Fortune
- Is Sam Bankman-Fried a bad ‘man’ or a good ‘boy’? Lawyers swap opening statements before first witnesses take the stand -PUBLISHED_BY-> Fortune

Entities: Fortune, Christmas, 26-year-old with $145,000 in student debt can't afford Christmas. She's giving 'affordable gifts or combining gifts with my older
sister', older sister, 26-year-old woman, Charles Schwab Foundation, Charles Schwab and Co. Inc., Associated Press, student debt, Dad, Mom
</item 2>

<item 3>
26-year-old with $145,000 in student debt can't afford Christmas. She's giving 'affordable gifts or combining gifts with my older sister' (Fortune,
2023-12-23T15:13:12+00:00)
If you are looking for a budget template, Microsoft Office offers a holiday spending-focused version.

COMMUNICATE WITH YOUR LOVED ONES

If you’re experiencing a lot of financial pressure this holiday season, Alev recommends that you try to communicate your feelings to your loved ones.

“One of the hardest things is resentment can build up if you feel like you need to buy gifts for your family and you can barely make your student loan payments,
so bring things up,” Alev said.

Sharing your financial struggles can be hard. If you want to talk but don’t want to share every detail of your financial struggles, that’s okay. Ultimately,
what you want is for your loved ones to understand how you’re feeling.

Plauché just recently started sharing her money worries with her family.

“I have been open with my family, just that I’m not going to be able to give gifts that year,” she said.

EXPERIENCES VS.

Graph facts:
- Sen. Robert Menendez, facing bribery charges, stumps bank CEOs by asking how much their firms paid in settlements for allegedly ripping off consumers
-PUBLISHED_BY-> Fortune
- Buckle up for more ‘turbulence’ in the housing market, BofA says—it’s a housing recession, 1980s style -PUBLISHED_BY-> Fortune
- Israel's blockade of Gaza means that a region the size of Philadelphia faces a real risk of starvation, says Peace Studies scholar -PUBLISHED_BY-> Fortune
- Gen Z just saw the worst war in Israel in 50 years break out all over the internet, but they were raised on 'disturbing images' -PUBLISHED_BY-> Fortune
- Is Sam Bankman-Fried a bad ‘man’ or a good ‘boy’? Lawyers swap opening statements before first witnesses take the stand -PUBLISHED_BY-> Fortune

Entities: Fortune, 26-year-old with $145,000 in student debt can't afford Christmas. She's giving 'affordable gifts or combining gifts with my older sister',
Microsoft Office, holiday spending-focused budget template, Alev, Plauché, communicating feelings to loved ones, being unable to give gifts that year
</item 3>

<item 4>
26-year-old with $145,000 in student debt can't afford Christmas. She's giving 'affordable gifts or combining gifts with my older sister' (Fortune,
2023-12-23T15:13:12+00:00)
If you can’t pay your balance in full every month, it’s important that you understand when you payment is due and have a plan to avoid paying a lot of interest.

If you’re planning to use “ buy now, pay later services ”, it’s best to be very intentional, Alev said.

“It’s easy to get caught up. It’s easy to use it a lot and have several different payments that will come down the road,” she said.

AFFORDABLE ALTERNATIVES

There is more than one way to find the “perfect” gift for your family and it doesn’t always involve spending a lot. In her quest to find gifts that don’t cause
her financial stress, Johnson has been getting creative.

“I’ve definitely been trying to save or just make gifts because I don’t really have the disposable income to buy other people gifts or even buy myself a
Christmas gift,” Johnson said.

Graph facts:
- The 5 biggest mistakes parents make at Christmastime, according to a parenting expert -MENTIONS-> Christmas
- This Christmas let your home be the centre of the celebrations with these festive ideas from The White Company -MENTIONS-> Christmas
- Christmas proves that love triumphs over hate — even amid war, suffering, tragedy and turmoil -MENTIONS-> Christmas
- Are NBA Christmas Day Games a Better Tradition Than NFL Thanksgiving Day Games? -MENTIONS-> Christmas
- From a Turkey trot to a movie marathon, Friendsgiving celebrations that aren’t a sit-down dinner -MENTIONS-> Christmas

Entities: Christmas, Fortune, 26-year-old with $145,000 in student debt can't afford Christmas. She's giving 'affordable gifts or combining gifts with my older
sister', Johnson, Alev, buy now, pay later services, gifts, student debt
</item 4>

<item 5>
26-year-old with $145,000 in student debt can't afford Christmas. She's giving 'affordable gifts or combining gifts with my older sister' (Fortune,
2023-12-23T15:13:12+00:00)
EXPERIENCES VS. GIFTS

Another great way to save money is to skip material gifts and instead, grant your time. In many cases, your family would rather spend quality time with you.

Alev recommended planning low-cost activities such as a hike with your family, or attending a free event in your city. Other options can be a holiday movie
marathon with your friends, a free museum with your parents or a volunteering day with your family.

USE CREDIT RESPONSIBLY

Since Plauché is already paying her student loan debt every month, she is avoiding using her credit card this holiday season to avoid carrying more debt.

If using your credit card is your only option, it’s important that you use it wisely.

If you struggle with overspending during the holidays, Alev also recommends setting a budget for each of your credit cards.

Graph facts:
- Sen. Robert Menendez, facing bribery charges, stumps bank CEOs by asking how much their firms paid in settlements for allegedly ripping off consumers
-PUBLISHED_BY-> Fortune
- Buckle up for more ‘turbulence’ in the housing market, BofA says—it’s a housing recession, 1980s style -PUBLISHED_BY-> Fortune
- Israel's blockade of Gaza means that a region the size of Philadelphia faces a real risk of starvation, says Peace Studies scholar -PUBLISHED_BY-> Fortune
- Gen Z just saw the worst war in Israel in 50 years break out all over the internet, but they were raised on 'disturbing images' -PUBLISHED_BY-> Fortune
- Is Sam Bankman-Fried a bad ‘man’ or a good ‘boy’? Lawyers swap opening statements before first witnesses take the stand -PUBLISHED_BY-> Fortune

Entities: Fortune, Christmas, 26-year-old with $145,000 in student debt can't afford Christmas. She's giving 'affordable gifts or combining gifts with my older
sister', student loan debt, Alev, Plauché, older sister, credit card
</item 5>

<item 6>
26-year-old with $145,000 in student debt can't afford Christmas. She's giving 'affordable gifts or combining gifts with my older sister' (Fortune,
2023-12-23T15:13:12+00:00)
BUDGET

Alev is a big proponent of the 50-30-20 budgeting method, where 50% of your income is allocated for necessities like food and rent, 30% for things you want, and
20% for savings and debt repayment.

To find space in your budget for gifts, Alev recommends that you plan ahead as much as possible and only take money from the 30% portion of your budget.

If the 50-30-20 is not the right format for you, there are plenty of other budgeting methods that you can use. Whether is recording your spending on an Excel
spreadsheet, using an app or writing it down on a notebook, keeping a budget is most effective when you do it in a format that works for you.

If you are looking for a budget template, Microsoft Office offers a holiday spending-focused version.
</item 6>

<item 7>
26-year-old with $145,000 in student debt can't afford Christmas. She's giving 'affordable gifts or combining gifts with my older sister' (Fortune,
2023-12-23T15:13:12+00:00)
“I’ve mostly been trying to think of affordable gifts or combining gifts with my older sister,” said Johnson, a non-profit worker in Citrus Heights, California.

Plauché and Johnson are just two of the millions of people who had to start paying back student loans after the Biden administration ended the three-year
payment pause.

“We know that the holidays are often a very joyful time of the year and an expensive one every year,” said Courtney Alev, consumer financial advocate at Credit
Karma. “With federal student loan payments just resuming, it’s going to add another often very significant line item into holiday budgets.”

If student loan payments are cutting into what used to be your holiday budget, here are some recommendations:

THE PRESSURE IS REAL

From family expectations to social media and TV ads showing lavish gifts, many people feel that the only way they can show love is to spend a lot of money.

Graph facts:
- Sen. Robert Menendez, facing bribery charges, stumps bank CEOs by asking how much their firms paid in settlements for allegedly ripping off consumers
-PUBLISHED_BY-> Fortune
- Buckle up for more ‘turbulence’ in the housing market, BofA says—it’s a housing recession, 1980s style -PUBLISHED_BY-> Fortune
- Israel's blockade of Gaza means that a region the size of Philadelphia faces a real risk of starvation, says Peace Studies scholar -PUBLISHED_BY-> Fortune
- Gen Z just saw the worst war in Israel in 50 years break out all over the internet, but they were raised on 'disturbing images' -PUBLISHED_BY-> Fortune
- Is Sam Bankman-Fried a bad ‘man’ or a good ‘boy’? Lawyers swap opening statements before first witnesses take the stand -PUBLISHED_BY-> Fortune

Entities: Fortune, California, 26-year-old with $145,000 in student debt can't afford Christmas. She's giving 'affordable gifts or combining gifts with my older
sister', Johnson, three-year payment pause, Courtney Alev, Credit Karma, Citrus Heights, Plauché, Biden administration, student loans, holidays
</item 7>

<item 8>
Adobe: Thanksgiving US online sales nudge up to $5.6B; Salesforce: $31.7B spent globally (TechCrunch, 2023-11-24T15:43:51+00:00)
Today’s 5.5% rate is definitely an improvement on last year’s 2.9%. But it’s nothing compared to the years preceding COVID-19, such as 2017, where we were
seeing growth of 18% or more.

Inflation is making an impact but not as much as the worry around consumer spending, said Adobe, which said that spending is being led actually by more
discounts to encourage buying, rather than less buying of more expensive products. That leads one to wonder what kind of impact that’s having on retailers’
margins.

“Cyber Week is off to a strong start with Thanksgiving driving a record $5.6 billion in online spend as consumers took advantage of strong discounts and
continued their shopping plans, virtually,” said Vivek Pandya, lead analyst, Adobe Digital Insights, in a statement. “Mobile shopping hit an all-time high, as
shoppers took to their smartphones to get the best deals during holiday gatherings, further solidifying mobile’s growing importance in e-commerce.”

Graph facts:
- SBF’s trial starts soon, but how did he — and FTX — get here? -PUBLISHED_BY-> TechCrunch
- Sam Altman backs teens’ startup, Google unveils the Pixel 8 and TikTok tests an ad-free tier -PUBLISHED_BY-> TechCrunch
- More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo -PUBLISHED_BY-> TechCrunch
- After warning, the EU opens investigation into X over alleged disinformation -PUBLISHED_BY-> TechCrunch
- SBF Trial: The latest updates from the FTX collapse’s courtroom drama -PUBLISHED_BY-> TechCrunch

Entities: TechCrunch, Thanksgiving, US, COVID-19, online sales, Adobe: Thanksgiving US online sales nudge up to $5.6B; Salesforce: $31.7B spent globally, Adobe,
consumer spending, Cyber Week, Vivek Pandya, Adobe Digital Insights, mobile shopping, retailers' margins, discounts, inflation
</item 8>

<item 9>
26-year-old with $145,000 in student debt can't afford Christmas. She's giving 'affordable gifts or combining gifts with my older sister' (Fortune,
2023-12-23T15:13:12+00:00)
Between being a mom and a new grad with a salary to match, Plauché says she can barely manage her daily expenses. Still, she feels a responsibility to give her
family gifts because that’s how they have traditionally shown each other love.

“I don’t want to give my family this message that they’re not loved and it definitely adds a lot of pressure,” she said.

Johnson feels guilty because the holidays are the time she feels she can show her family how grateful she is for them. She has always prided herself on being a
great gift giver who not only finds something nice but knows how to spot unique and special gifts.

“I think I’m a little embarrassed to tell them that this year I might not be able to be a good gift giver,” said Johnson.

Acknowledging that you’re feeling pressure is a good first step towards making a plan to set expectations and be realistic about what you can spend.

Graph facts:
- The 5 biggest mistakes parents make at Christmastime, according to a parenting expert -MENTIONS-> Christmas
- This Christmas let your home be the centre of the celebrations with these festive ideas from The White Company -MENTIONS-> Christmas
- Christmas proves that love triumphs over hate — even amid war, suffering, tragedy and turmoil -MENTIONS-> Christmas
- Are NBA Christmas Day Games a Better Tradition Than NFL Thanksgiving Day Games? -MENTIONS-> Christmas
- From a Turkey trot to a movie marathon, Friendsgiving celebrations that aren’t a sit-down dinner -MENTIONS-> Christmas

Entities: Christmas, Fortune, 26-year-old with $145,000 in student debt can't afford Christmas. She's giving 'affordable gifts or combining gifts with my older
sister', gift giving, Johnson, daily expenses, family gifts, Plauché, student debt, holidays
</item 9>

<item 10>
26-year-old with $145,000 in student debt can't afford Christmas. She's giving 'affordable gifts or combining gifts with my older sister' (Fortune,
2023-12-23T15:13:12+00:00)
For Nicole Plauché and her family, Christmas is the time of the year where they can use their preferred love language: gift giving. This year, however, Plauché
is worried about how much she can afford to spend on gifts. Her main struggle? Student loan payments.

“It just doesn’t leave much room for anything outside of just basic necessities,” said Plauché, a 23-year old tech sales manager from Dallas, Texas.

Marisa Johnson, 26, has approximately $145,000 in student loan debt from both her bachelor’s and master’s degrees. Since the COVID-era payment pause ended in
October, she’s been paying back $300 a month while she waits to see if she’ll be enrolled in a new income-driven repayment plan.

Because all of her money is going to necessities and paying back her student loans, Johnson hasn’t been able to spend as much on gifts as in previous years.

Graph facts:
- Sen. Robert Menendez, facing bribery charges, stumps bank CEOs by asking how much their firms paid in settlements for allegedly ripping off consumers
-PUBLISHED_BY-> Fortune
- Buckle up for more ‘turbulence’ in the housing market, BofA says—it’s a housing recession, 1980s style -PUBLISHED_BY-> Fortune
- Israel's blockade of Gaza means that a region the size of Philadelphia faces a real risk of starvation, says Peace Studies scholar -PUBLISHED_BY-> Fortune
- Gen Z just saw the worst war in Israel in 50 years break out all over the internet, but they were raised on 'disturbing images' -PUBLISHED_BY-> Fortune
- Is Sam Bankman-Fried a bad ‘man’ or a good ‘boy’? Lawyers swap opening statements before first witnesses take the stand -PUBLISHED_BY-> Fortune

Entities: Fortune, Christmas, 26-year-old with $145,000 in student debt can't afford Christmas. She's giving 'affordable gifts or combining gifts with my older
sister', Nicole Plauché, Marisa Johnson, student loan debt, gift giving, COVID-era payment pause, income-driven repayment plan, basic necessities, Texas, Dallas
</item 10>
=== END ITEM id=db8ba354ff27afd1 ===
