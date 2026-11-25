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

=== ITEM id=a1876c8822e63e22 ===
Gold sentences and facts:
Sentence 0: The Cowboys opened Week 13 by rallying to beat the Seahawks on Thursday night, keeping the heat on the losing Eagles ahead of them before hosting that team in Dallas in Week 14.
  Fact 0.0: The Cowboys opened Week 13 by rallying to beat the Seahawks on Thursday night.
  Fact 0.1: The Cowboys' win kept the heat on the Eagles.
  Fact 0.2: The Eagles were a losing team.
  Fact 0.3: The Eagles were ahead of the Cowboys.
  Fact 0.4: The Cowboys will host the Eagles in Dallas in Week 14.
Sentence 1: Fueled by a stout defensive showing, a gritty run game and a bit of fairy dust from the likes of St. Brown, Sam LaPorta and Josh Reynolds, the Lions cruised to a 34-20 win over their hated NFC North rivals, Green Bay.
  Fact 1.0: The Lions' win was fueled by a stout defensive showing.
  Fact 1.1: The Lions' win was fueled by a gritty run game.
  Fact 1.2: The Lions' win was fueled by a bit of fairy dust from St. Brown.
  Fact 1.3: The Lions' win was fueled by a bit of fairy dust from Sam LaPorta.
  Fact 1.4: The Lions' win was fueled by a bit of fairy dust from Josh Reynolds.
  Fact 1.5: The Lions cruised to a 34-20 win.
  Fact 1.6: The Lions beat Green Bay 34-20.
  Fact 1.7: Green Bay is the Lions' hated NFC North rival.

Retrieved items:
<item 1>
Lions vs. Packers live score, updates, highlights from NFL 'Thursday Night Football' game (Sporting News, 2023-09-28T23:54:07+00:00)
8:13 p.m.: Lions starting the game with the ball. Packers defense looking stout as ever through the opening two plays.

8:12 p.m.: Weezy F Baby leading the Packers out onto the field! Wasn't expecting that!

7:53 p.m.: Marshawn Lynch kicking it with Barry Sanders in a pre-game segment. This is what you love to see. It looks like two legends can co-exist.

7:40 p.m.: Aaron Jones' return should offer Love a much-needed reprieve. Green Bay's run game hasn't been all that effective thus far — the Pack are averaging
just 3.4 yards per carry so far this year. But Jones is a dynamic force coming out of the backfield, someone capable of carrying an offense on his back for
large portions of a game. That's a young QB's dream.

7:30 p.m.: Jordan Love and the Packers heading onto the field ahed of tonight's game!
</item 1>

<item 2>
NFL power rankings: Lions, 49ers level up; Texans jump Bengals, Bills; Jets, Patriots sink to bottom for Week 11 (Sporting News, 2023-11-13T21:54:00+00:00)
Their defense should be getting a little better to support their often explosive offense. They should feel good about a Week 11 rebound vs. the Raiders.

7. Pittsburgh Steelers 6-3 (9)

The Steelers are playing the way Mike Tomlin wants, with the running game rolling to big totals with Jaylen Warren and Najee Harris and the defense doing what
it needs, more so vs. pass than run, in key moments. You can't argue with the overachieving results.

8. Cleveland Browns 6-3 (10)

The Browns proved they can win games with defense in another way, with big plays. They also have more offensive pop when Deshaun Watson plays well and the
running game doesn't slump. Their playoff hopes look much better after their stunning comeback vs. the Ravens.

9.
</item 2>

<item 3>
NFL roundup: Pats hit new low as Lions reach 9-3 for first time since 1962 (The Guardian, 2023-12-03T22:11:26+00:00)
The AFC East-leading Dolphins are 9-3 for the first time since 2001. Tua Tagovailoa was turnover-free after giving the ball away four times over the past two
games. De’Von Achane ran for two touchdowns in his return after missing five of the past six with a knee injury. The Commanders have lost four in a row.
Washington’s Sam Howell threw a pick-6 for a third consecutive game.

Detroit Lions 33–28 New Orleans Saints

Sam LaPorta had career highs of nine catches for 140 yards, including an early touchdown and a crucial first-down snag in the final minutes, and the Detroit
Lions beat the reeling New Orleans Saints. Detroit improved to 9-3, their best record through 12 games since they went 10-2 in 1962. The Saints lost quarterback
Derek Carr in the fourth quarter to back, shoulder and head injuries. Carr was hurt on a penalized hit by Bruce Irvin.
</item 3>

<item 4>
NFL power rankings: Lions, 49ers level up; Texans jump Bengals, Bills; Jets, Patriots sink to bottom for Week 11 (Sporting News, 2023-11-13T21:54:00+00:00)
Stroud leading their offense and now they've found some running game, too. The defense is just making enough plays to put them in position to win high-scoring
games. Houston is a legitimate playoff contender under rookie coach DeMeco Ryans.

14. Cincinnati Bengals 5-4 (7)

The Bengals are battling some key injuries, leaving them much less room for error. Unfortunately, Joe Burrow made key mistakes and the defense wilted against
run and pass at home in the loss to the Texans.

15. Buffalo Bills 5-5 (12)

The Bills fell right into the trap game against the Broncos at home, losing because of more mistakes from Josh Allen. The defense can't tolerate those mistakes
with all of its injuries.

16.
</item 4>

<item 5>
Seahawks vs. Giants live score, updates, highlights from NFL 'Monday Night Football' game (Sporting News, 2023-10-02T22:00:04+00:00)
Inactive for the Seahawks: CB Coby Bryant, CB Tre Brown, CB Artie Burns, LB Nick Bellore, G McClendon Curtis, T Charles Cross and G Ben Brown.

6:25 p.m.: With less than two hours remaining until kickoff, there's still no concrete update on the availability of Giants RB Saquon Barkley, who is currently
listed as doubtful with a high ankle sprain.

Barring a last-minute change, it doesn't appear as though No. 26 will be suiting up for New York.

6 p.m.: Jamal Adams is back!

For the first time since Week 1 of the 2022 NFL season, the three-time All-Pro safety will take the field for the Seahawks. Almost poetically, Adams makes his
return to the field in the place he called home for the first three years of his career as a member of the Jets.

What channel is Seahawks vs. Giants on today?
</item 5>

<item 6>
Seahawks vs. Giants live score, updates, highlights from NFL 'Monday Night Football' game (Sporting News, 2023-10-02T22:00:04+00:00)
: The Giants just pulled off one of their biggest plays of the night.

On fourth-and-1 from Seattle's 16-yard line, Brian Daboll elects to keep the offense on the field instead of electing for three points. Out of play action,
Daniel Jones rolls out and gets 11 yards to bring New York within five yards of its first touchdown of the night.

10:22 p.m.: A 53-yard field goal attempt from Jason Myers is no good!

Seattle fails to add to its lead and New York gets another chance to cut into its deficit, this time, with field position that's slightly more favorable. The
Giants will take over at their on 34-yard line.

10:16 p.m.: New York's drive ends about as quickly as it began as it is three-and-out for the Giants.

Daniel Jones is sacked for a sixth time this evening. This time it's eight-time Pro Bowler Bobby Wagner.
</item 6>

<item 7>
Cowboys vs. 49ers live score, updates, highlights from NFL 'Sunday Night Football' game (Sporting News, 2023-10-08T22:47:03+00:00)
11:24 p.m. — These teams are trading possessions as the clock winds down. There was a brief skirmish officials had to break up as frustrations mount for Dallas.

11:14 p.m. — A sack by Kevin Givens ends Rush's first drive of the night and sets up a Bryan Anger punt. Backups are in nearly all around for the 49ers.

11:09 p.m. — Cooper Rush is in the game for the Cowboys...

11:07 p.m. — Mitch Wishnowsky with an excellent punt for the 49ers, who are just trying to get through the rest of the game injury-free.

11:04 p.m. — Sam Darnold has entered the game with 49ers. So, that's where we are with 12 minutes still on the clock.

11:00 p.m. INTERCEPTION — So much for that.
</item 7>

<item 8>
Raiders vs. Lions live score, updates, highlights from NFL 'Monday Night Football' game (Sporting News, 2023-10-30T22:20:03+00:00)
Adams was seen slamming his helmet down on the sideline in the fourth quarter, and it's tough to blame the star receiver after an 11-yard performance that could
have been exponentially better with better quarterback play.

The story of the night was Lions rookie Jahmyr Gibbs, who finally broke through after entering the season with sky-high expectations. Tasked with leading the
backfield in David Montgomery's absence, Gibbs exploded for 189 scrimmage yards and a touchdown in the win. It was the first 100-yard rushing performance of his
career.

Meanwhile, Amon-Ra St. Brown notched his third consecutive 100-yard game despite entering Monday as questionable with an illness.

MORE: Why Davante Adams is unlikely to be traded before deadline

The Lions enter their bye in sole possession of the No. 2 seed in the NFC, while the Raiders drop their second consecutive game against an NFC North opponent
and have what might be a must-win game against the Giants in Week 9 if they want any chance to linger in the AFC playoff race.
</item 8>

<item 9>
NFL Fantasy Football Week 5 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-04T21:56:25+00:00)
PURDY: Has hit at least 20 Fantasy points in 8 of his 9 regular-season starts, and it really should be 9 for 9 because of that first touchdown that was ruled a
lateral back in Week 2. If you include the two playoff games he completed, Purdy has delivered at least 20 in 9 of 11 games. And in 5 of those 9 he hit at least
23 Fantasy points including each of his past two games.

COWBOYS: Dallas' defense lost top cornerback Trevon Diggs and they're still red hot. They lead the league in pass rush pressure rate this season (55.6%) and
they're in the top-10 in blitz rate (31.9%). It's that pressure that's helped the Cowboys rank in the top-five in completion rate allowed (57.4%), passing
touchdowns allowed (two) and missed tackles on pass plays (just five, that's best in football).
</item 9>

<item 10>
NFL roundup: Pats hit new low as Lions reach 9-3 for first time since 1962 (The Guardian, 2023-12-03T22:11:26+00:00)
The 49ers spent an offseason chirping at the Birds – Samuel called one Eagles cornerback “trash” – and then entered the Linc as a rare betting favorite against
a 10-win team at home.

The Eagles (10-2) suffered a scare when Hurts jogged off the field in the fourth and went straight to the locker room. He was cleared to play and returned to
action with about 10 minutes left. It was not immediately clear when Hurts was injured.

By the time last season’s NFL MVP runner-up returned, the Eagles trailed 35-13 and were on their way toward their first home loss of the season. Hurts did hit
DeVonta Smith for a two-yard TD on his return drive. He finished 26-of-45 passing for 298 yards. Purdy put the finishing touches on the win with a 46-yard TD to
Samuel.

Perhaps all that braggadocio paid off for the 49ers.
</item 10>

<item 11>
Chiefs vs. Packers live score, updates, highlights from NFL 'Sunday Night Football' game (Sporting News, 2023-12-03T23:00:04+00:00)
Yikes...

Packers 14. Chiefs 6

9:00 p.m.: TOUCHDOWN! Green Bay's second drive of the game ends much like the first, with Love firing an arrow to Watson to add to Green Bay's early advantage.

Packers 14, Chiefs 3

End of first quarter: Packers 7, Chiefs 3

8:50 p.m.: FIELD GOAL! Harrison Butler sinks his field goal after a pair of sacks relegate Kansas City to bringing its special teams onto the field.

Packers 7, Chiefs 3

8:48 p.m.: Isiah Pacheco gallops on the outside to grab a heap of yardage. Kansas City is on the brink of a score.

8:47 p.m.: Great composure by Mahomes and Marquez Valdez-Scantling, both of whom combine to keep Kansas City in the ascendancy.

8:45 p.m.
</item 11>

<item 12>
Thursday Night Football FanDuel Picks: NFL DFS lineup advice for Week 4 Lions-Packers single-game tournaments (Sporting News, 2023-09-26T21:53:00+00:00)
Aaron Rodgers is no longer in Green Bay, and a new love interest is now under center, Jordan Love. It was these Lions that ended the Rodgers era and swept Green
Bay last season. Now they meet again in prime time on one of the league's most hallowed grounds in a battle for NFC North supremacy. The tundra won't be frozen
at Lambeau Field at this point in the season, but the game sure is important, and for a game like this, there's no way we could pass up the chance to win some
cash with our FanDuel single-game DFS lineup.

MORE THURSDAY NIGHT FOOTBALL:

DK lineup | Best props | Betting preview | Start-Sit

Before we break down our picks and strategy, here are the most notable scoring rules for FanDuel contests: four-point passing TDs and half-point PPR.
</item 12>

<item 13>
NFL Best Bets Week 9: Colts clobber Panthers, Dolphins drop Chiefs in Germany (Sporting News, 2023-11-03T20:56:00+00:00)
The Panthers have allowed the most rushing TDs (14), the third-most points per game (28.4), and the fourth-most rushing yards per game (139.4) during their 1-7
start. The Colts will cruise in this one.

FINAL SCORE PREDICTION: Colts 27, Panthers 17

Seahawks (+6) at Ravens

Baltimore's defense has been incredible of late, and the Ravens seem to be flying under the radar (so to speak) at 6-2. However, the sportsbooks have clearly
taken note of their awesome run, giving the NFC West-leading Seahawks a full touchdown on the road. This seems like a great opportunity for us to rock with the
'dog, considering Seattle has won two straight contests and gone 2-1 on the road (including a big win over the high-flying Lions in Detroit).
</item 13>

<item 14>
2023 NFL Week 11 injury roundup: Rams WR Cooper Kupp, Seahawks RB Kenneth Walker III among stars to exit (CBSSports.com, 2023-11-19T23:51:49+00:00)
The former All-Pro, who battled a lingering hamstring injury earlier this year, received attention on the sidelines after limping off, then left the game for
further evaluation in the locker room. He was ruled questionable to return. NFL Media reported on Monday that Kupp suffered a low ankle sprain and will be
evaluated going forward.

Darius Slayton (Giants) had a big impact against the Commanders, hauling in a deep score from Tommy DeVito, before exiting with an arm injury. He was ruled out
for the remainder of the contest halfway through the third quarter.

EDGE

Joey Bosa (Chargers) appeared to injure his right leg or foot while held in a block during a first-quarter pass-rushing attempt against the Packers. He limped
to the sidelines and was carted to the locker room, then ruled questionable to return.
</item 14>

<item 15>
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
</item 15>

<item 16>
NFL standings: Updated AFC, NFC playoff picture for Week 13 of 2023 season (Sporting News, 2023-12-03T21:49:04+00:00)
They got to .500 and increased their lead to a full game by beating the Jets in Week 13.

5. Dallas Cowboys (9-3), second place, NFC East. The Cowboys opened Week 13 by rallying to beat the Seahawks on Thursday night, keeping the heat on the losing
Eagles ahead of them before hosting that team in Dallas in Week 14. They can finish No. 1, No, 2 or No. 3 if they win the division and are locked into No. 5
again otherwise.

6. Minnesota Vikings (6-6), second place, NFC North. The Vikings jumped up a spot during their bye with the Seahawks losing on Thursday night to match their
record on Thursday night. They have the tiebreaker over the three other .500 NFC teams based on conference record, but right behind them is the team in third
place in the division.

7. Green Bay Packers (6-6), third place, NFC North.
</item 16>

<item 17>
N.F.L. Executives on the Biggest Questions Facing Super Bowl Contenders (The New York Times, 2023-10-16T23:58:53+00:00)
But they also haven’t really beaten anyone. They were down 14 points to Carolina. Do you think they are coming back from 13 down against Philly? In my opinion,
no.”

Advertisement

• Dallas Cowboys: Is this team any better than previous Dallas teams that fell short?

What I think: It’s a tough time to evaluate the Cowboys fairly because they were so bad against San Francisco on a big stage in their most recent game. I just
don’t see where Dallas is better or going to get considerably better, especially on offense. And every time they enjoy success, they seem to enjoy it a little
too much. That is a reflection of their owner.

Exec comment: “Run defense and consistency on defense, and then offensively, what is their identity? They just are not very good. Their best games are when they
play great defense and get turnovers, but that is not going to happen enough, especially against the good teams.”

• Detroit Lions: Have they fixed their defense?
</item 17>

<item 18>
Cowboys vs. 49ers live score, updates, highlights from NFL 'Sunday Night Football' game (Sporting News, 2023-10-08T22:47:03+00:00)
This 49ers defense has picked up right where it left off against the Cowboys in January.

8:44 p.m. — Dallas' second drive starts out with Tony Pollard getting dropped for a loss.

8:40 p.m. — Big-time stop by the Cowboys' defense as Johnathan Hankins deflects Purdy's third down pass with his helmet and forces a punt.

8:36 p.m. — The 49ers' defense was all over Dallas on that drive. Three plays, two incompletions, and now the Cowboys will punt it right back.

49ers 7, Cowboys 0

8:30 p.m. TOUCHDOWN — Purdy finds a wide open Kittle in the end zone for the score! That was a well-scripted opening drive by Kyle Shanahan, and San Francisco
has an early 7-0 lead.

8:27 p.m.
</item 18>

<item 19>
Cowboys vs. 49ers live score, updates, highlights from NFL 'Sunday Night Football' game (Sporting News, 2023-10-08T22:47:03+00:00)
9:37 p.m. — Deebo Samuel takes some big hits but gets the 49ers down to the one-yard line. San Francisco is knocking on the door again.

9:35 p.m. — The Cowboys had done a pretty nice job of containing McCaffrey to this point, but the former Pro-Bowler takes a screen pass for 19 yards to push the
Niners to the edge of the red zone.

9:31 p.m. — Aiyuk with a quick 15-yard catch to start the 49ers' drive. He's up to 35 yards on three catches.

49ers 14, Cowboys 7

9:27 p.m. TOUCHDOWN — KaVontae Turpin flashes his speed, gets separation, and hauls in Prescott's pass in the end zone! A perfectly drawn-up 26-yard play puts
Dallas right back in this game.
</item 19>

<item 20>
Fantasy Football Week 13 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-11-30T22:10:58+00:00)
Easy peasy!

Start 'em 10.0-7.00 6.99 - 5.01 5.00 - 0 Sit 'em

Both teams had a week of rest, so don't treat this like a typical Thursday game. But do expect the Cowboys defense to keep the pressure on Geno Smith and a
Seattle offense that's struggled to gain consistency over the last month. Smith actually did OK completing passes last week when pressured (60% completion rate)
but not at an efficient level (5.4 yards per attempt). Just two QBs have exceeded 250 yards and four QBs have thrown multiple touchdowns against Dallas this
year; I don't like Smith's chances to do either, which hurts any upside case for any Seahawks receiver. By the way, Curtis Samuel was the first receiver to get
100 yards on the Cowboys this year, and it took him 12 targets and 9 catches to get there.
</item 20>

<item 21>
NFL roundup: Pats hit new low as Lions reach 9-3 for first time since 1962 (The Guardian, 2023-12-03T22:11:26+00:00)
The AFC East-leading Dolphins are 9-3 for the first time since 2001. Tua Tagovailoa was turnover-free after giving the ball away four times over the past two
games. De’Von Achane ran for two touchdowns in his return after missing five of the past six with a knee injury. The Commanders have lost four in a row.
Washington’s Sam Howell threw a pick-six for a third consecutive game.

Atlanta Falcons 13-8 New York Jets

Desmond Ridder threw a 20-yard touchdown pass to MyCole Pruitt and the Atlanta Falcons took over sole possession of first place in the NFC South by slogging
their way to an ugly victory over the punchless New York Jets.

The Falcons (6-6) have back-to-back wins for the first time since opening the season with victories over Carolina and Green Bay. Atlanta entered tied atop the
division with New Orleans, who lost to Detroit.
</item 21>

<item 22>
Seahawks vs. Giants live score, updates, highlights from NFL 'Monday Night Football' game (Sporting News, 2023-10-02T22:00:04+00:00)
This time it's eight-time Pro Bowler Bobby Wagner. The excitement at MetLife Stadium is gone once again as New York will punt from its own 12.

10:12 p.m.: TURNOVER ON DOWNS! New York gives its fans something to be excited about as its defense holds up on fourth down thanks to a major PBU from LB Micah
McFadden.

The Giants keep the Seahawks off the board and take over at their own 15-yard line.

10:03 p.m.: The second half is underway and Geno Smith is back on the field for the Seahawks.

Seattle fans can breathe a sigh of relief as its top signal-caller will be back under center and avoided serious injury.

Halftime: Seahawks 14, Giants 3

9:47 p.m.: The half closes with New York committing its third and fourth special teams penalties of the night, giving Seattle the ball back after it intended to
punt on fourth-and-4.
</item 22>

<item 23>
Cowboys vs. 49ers live score, updates, highlights from NFL 'Sunday Night Football' game (Sporting News, 2023-10-08T22:47:03+00:00)
The Cowboys' offense looked very much like it did in last season's playoff exit -- uncomfortable and out of sorts. Dallas was sluggish through the first half of
the game, and the night unraveled quickly when Dak Prescott threw three interceptions in the second half. It got out of hand so quickly that Sam Darnold entered
the game for the 49ers with more than 12 minutes remaining.

For Prescott and the Cowboys, questions about whether they can get over the hump against the class of the NFC will persist. While their wins have been dominant,
Dallas already sits two games behind the Eagles in the NFC East.

The 49ers are rolling right along, with all of their major offensive weapons now healthy and the defense living up to its billing as well. San Francisco, now
5-0, doesn't face a team with a winning record until Week 10, though the Browns, Vikings and Bengals are far from automatic wins for Kyle Shanahan's team.
</item 23>

<item 24>
Raiders vs. Lions live score, updates, highlights from NFL 'Monday Night Football' game (Sporting News, 2023-10-30T22:20:03+00:00)
The Lions just needed to get themselves back in the win column after a blowout loss in Baltimore, and they did just that in front of their home fans on Monday
night.

Detroit rolled to a 26-14 victory over the Raiders in a game that felt much more one-sided than the score indicates.

The Lions thoroughly outplayed the Raiders, out-gaining Las Vegas by 329 yards, but critical mistakes by Detroit left points on the board. The offense struggled
to turn red zone opportunities into touchdowns in the first half, and two turnovers deep in Raiders territory -- including a pick-six by Jared Goff -- kept Las
Vegas in the game in the second half.

MORE: What to know about Lions' retro 'Honolulu blue' helmets

Nonetheless, the Lions' defense dominated. The Raiders were abysmal on the offensive side of the ball, outside of a second-quarter touchdown drive led by Josh
Jacobs. Jimmy Garoppolo completed just 10 passes, badly missing a wide-open Davante Adams on multiple occasions.
</item 24>

<item 25>
Thursday Night Football FanDuel Picks: NFL DFS lineup advice for Week 4 Lions-Packers single-game tournaments (Sporting News, 2023-09-26T21:53:00+00:00)
For Detroit, it's all about courage. For a team that was once down and out, much like their city, they are now once again on the rise. The Lions are supposed to
be the kings of the jungle and hope to become the NFL's version in 2023. In terms of the food chain, a basic concept taught in science classes across the
country, lions are the hunter rather than the hunted, but it's been the other way around for far too long, as 31 other teams have feasted on a once proud
franchise.

Their motors have been restored and the team has been rebuilt with an attitude and moxie that would make even the most "football guy" of "football guys" blush.
It's a combination of quiet confidence and a downright arrogant confidence that draws people in to root for -- or against -- an underdog.

While the Packers have packed up the Lions for years, that all began to change in 2022.
</item 25>

<item 26>
Cowboys vs. 49ers live score, updates, highlights from NFL 'Sunday Night Football' game (Sporting News, 2023-10-08T22:47:03+00:00)
The crew is entering their second season together after Tirico replaced play-by-play legend Al Michaels while Stark took over for Michele Tafoya.

Fans can also stream the game via Peacock or NBCSports.com with a subscription, along with Fubo, which offers a free trial.

Viewers in Canada can stream every NFL game via DAZN.

Cowboys vs. 49ers start time

Date: Sunday, Oct. 8

Sunday, Oct. 8 Start time: 8:20 p.m. ET (5:20 p.m. CT)

Sunday night's game between the Cowboys and 49ers is set to kick off at 8:20 p.m. ET (5:20 p.m. local time) from Levi's Stadium in Santa Clara, Calif.

The 49ers and Cowboys have a long history with one another, but San Francisco has gotten the better of Dallas in each of their last two meetings.
</item 26>

<item 27>
Seahawks vs. Giants live score, updates, highlights from NFL 'Monday Night Football' game (Sporting News, 2023-10-02T22:00:04+00:00)
Seahawks vs. Giants final score

1 2 3 4 Final Seahawks 7 7 7 3 24 Giants 0 3 0 0 3

Seahawks vs. Giants live updates, highlights from 'Monday Night Football'

(All times Eastern)

Team stats

Seahawks Giants First downs 13 17 Third downs 3-12 6-16 Fourth downs 0-2 2-4 Total yards 281 248 Passing 15-26, 160 yards 27-34, 136 yards Rushing att-yards
23-121 29-112 Penalties-yards 8-74 6-45 Turnovers 0 3 Time of possession 24:00 36:00

Final: Seahawks 24, Giants 3

11:18 p.m.: It's a wrap at MetLife Stadium. The Seahawks advance to 3-1 on the year while the Giants fall to 1-3.
</item 27>

<item 28>
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
</item 28>

<item 29>
NFL power rankings: Lions, 49ers level up; Texans jump Bengals, Bills; Jets, Patriots sink to bottom for Week 11 (Sporting News, 2023-11-13T21:54:00+00:00)
the Patriots. At least Jonathan Taylor is doing his normal things and the defense is playing well again while Gardner Minshew looks meh.

18. Los Angeles Chargers 4-5 (17)

The Chargers got well offensively with Justin Herbert, Keenan Allen and Austin Ekeler against the Lions. But the defense went back to crumbling vs. big pass
plays and added wilting against the run. The pressure is right back on Brandon Staley with a trip to Green Bay next.

20. Tampa Bay Buccaneers 4-5 (23)

Baker Mayfield and the offense is back on track to move the ball well after a brief midseason slump. They're still not lighting it up, but the Bucs are back to
playing smart complementary football with their defense rebounding and still might have found a long-term QB answer.

21.
</item 29>

<item 30>
Seahawks vs. Giants live score, updates, highlights from NFL 'Monday Night Football' game (Sporting News, 2023-10-02T22:00:04+00:00)
As the Seahawks get the ball back, they run the clock out as boos begin to rain down on the Giants from their home crowd.

Making matters worse, Seattle is set to receive the second-half kickoff.

9:37 p.m: TOUCHDOWN! RB Kenneth Walker III punches it in from one yard out. It's his second touchdown of the 2023 season.

The play doesn't come without extracurriculars, however, as it is followed by a scrum between linemen. The result is offsetting penalties from Seahawks G
Anthony Bradford and Giants DE Leonard Williams.

Seahawks 14, Giants 3.

9:35 p.m.: Drew Lock is balling.

Lock connects with Noah Fant for a 51-yard completion that has the Seahawks on the 1-yard line. Credit to Fant, as he used his footwork to toe the sideline and
turn the completion to a big play.

9:30 p.m.
</item 30>

<item 31>
2023 NFL Week 11 injury roundup: Rams WR Cooper Kupp, Seahawks RB Kenneth Walker III among stars to exit (CBSSports.com, 2023-11-19T23:51:49+00:00)
Smith did return, however, for the final drive of the game in the losing effort. Head coach Pete Carroll said that Smith will "make it back." When asked if that
means for Thursday's matchup against San Francisco, the coach said, "I would think so."

Sam Howell (Commanders) received medical attention on the sidelines after enduring several hard hits to the upper body on a touchdown run against the Giants. He
was quickly cleared of a potential concussion, however, and remained in the game. Veteran journeyman Jacoby Brissett is Washington's backup.

RB

Aaron Jones (Packers) suited up against the Chargers despite a lingering hamstring issue, only to manage just five carries before suffering a knee injury. The
former Pro Bowler needed help to leave the field and was then carted to the locker room, replaced by A.J. Dillon.
</item 31>

<item 32>
Cowboys vs. 49ers live score, updates, highlights from NFL 'Sunday Night Football' game (Sporting News, 2023-10-08T22:47:03+00:00)
Not allowing even a field goal is a huge win for Dallas.

8:56 p.m. — The 49ers are excelling on third down early in this game, and Purdy flips it to Kyle Juszczyk to keep the Niners' drive alive on 3rd & 1.

8:54 p.m. — Now Kearse lines up offsides... That extends the drive for the 49ers, who aren't moving as freely as they were on that opening drive.

8:50 p.m. — Jayron Kearse comes through with the Cowboys' first sack of the night, but Deebo Samuel makes up 11 of the 12 yards San Francisco needs on the next
play.

8:46 p.m. — There's the first catch of the night for Lamb, but he doesn't get the first down and Dak Prescott is sacked by Nick Bosa to quickly end the drive.
This 49ers defense has picked up right where it left off against the Cowboys in January.
</item 32>

<item 33>
NFL power rankings: Lions, 49ers level up; Texans jump Bengals, Bills; Jets, Patriots sink to bottom for Week 11 (Sporting News, 2023-11-13T21:54:00+00:00)
They can win with offense or defense. Dan Campbell is coach of the year.

4. San Francisco 49ers 6-3 (8)

The 49ers had a surprise offensive get-well game in Jacksonville with Brock Purdy leading the way and all his key supporting weapons doing damage, even without
Christian McCaffrey scoring. The biggest development was the defense going back to dominating a good offense.

5. Baltimore Ravens 7-3 (2)

The Ravens' offense is just fine with the running and passing of Lamar Jackson, but he still can't afford to make big mistakes, which he did vs. the Browns. The
defense needs to pick up the pieces, too, stat, ahead of facing the Bengals on a short week.

6. Miami Dolphins 6-3 (6)

The Dolphins got a much-needed bye to figure out how to beat the better teams in the league.
</item 33>

<item 34>
Thursday Night Football FanDuel Picks: NFL DFS lineup advice for Week 4 Lions-Packers single-game tournaments (Sporting News, 2023-09-26T21:53:00+00:00)
TNF INJURY UPDATES: Montgomery | Jones | Watson

Lions-Packers FanDuel Picks Week 4: NFL DFS lineup for GPP tournaments

Lions-Packers TNF, $60,000 budget

MVP (1.5x points) QB Jared Goff, Lions ($15,500). The Lions might not be the Super Bowl contenders we thought they could be, but that doesn't mean we aren't in
for a treat on Thursday night at Lambeau Field. Detroit is finding its footing in 2023 and should be able to have some success against the Packers. On the
surface, Green Bay's pass defense is playing around league average, allowing 207.3 passing yards per game. That slots them in at No. 18, while Pro Football
Focus grades them the No. 8 team in coverage with an 81.5. Let's look beneath the surface a little more. The Packers' first three games have come against the
No.
</item 34>

<item 35>
Cowboys vs. 49ers live score, updates, highlights from NFL 'Sunday Night Football' game (Sporting News, 2023-10-08T22:47:03+00:00)
MORE: Why Brock Purdy remains one of the NFL's cheapest starting quarterbacks

The Sporting News tracked live scoring updates and highlights from "Sunday Night Football" between the Cowboys and 49ers. Check out all of the key moments you
might have missed.

Cowboys vs. 49ers final score

1 2 3 4 F Cowboys 0 7 3 0 10 49ers 7 14 7 14 42

Cowboys vs. 49ers results, highlights from 'Sunday Night Football'

(All times Eastern)

Final: 49ers 42, Cowboys 10

11:26 p.m. — That'll do it! The 49ers trounce the Cowboys and reassert themselves as one of the NFL's most dominant teams. Through five games, San Francisco is
unbeaten.

11:24 p.m. — These teams are trading possessions as the clock winds down.
</item 35>

<item 36>
NFL power rankings: Lions, 49ers level up; Texans jump Bengals, Bills; Jets, Patriots sink to bottom for Week 11 (Sporting News, 2023-11-13T21:54:00+00:00)
32. Carolina Panthers 1-8 (31)

The Panthers found better defense and special teams to keep them in the game in Chicago, but Bryce Young still can't unlock a limited passing game and the
offensive line and running game hasn't help nearly enough.
</item 36>

<item 37>
Monday Night Football DraftKings Picks: NFL DFS lineup advice for Week 15 Eagles-Seahawks Showdown tournaments (Sporting News, 2023-12-18T22:47:00+00:00)
The Seahawks are just about hanging on by a thread. At 6-7, the team is on the outside looking in when it comes to the playoff picture. A win on Monday night
against the suddenly reeling Eagles could be a big boost in their chances going forward -- a matchup that is actually a pretty good one after looking beneath
the surface. The Eagles come in as losers of two straight, losing in blowout fashion both times no less. Suddenly the sky is falling as the Cowboys have taken
over the NFC East division lead, complicating matters as the regular season winds down. The sense of urgency for both sides means we are looking at two teams
with plenty to lose, setting up for a fun Monday night under the lights at Lumen Field.
</item 37>

<item 38>
Prisco's Week 12 NFL picks: Jaguars beat Texans in AFC South battle, Saints get key divisional win vs. Falcons (CBSSports.com, 2023-11-23T21:30:38+00:00)
Dak Prescott plays well again. Cowboys roll.

Pick: Cowboys 32, Commanders 16 (Result: Cowboys 45-10)

Thursday, 8:20 p.m. ET (NBC, fubo)

This is a huge division game for two teams that have had a good rivalry over the past decade. The Seahawks are coming off a bad loss to the Rams, while the
49ers are coming off two dominant victories. Geno Smith is banged up, so his health is key. The 49ers have the look of a Super team again. They show it here.

Pick: 49ers 28, Seahawks 19 (Result: 49ers 31-13)

Friday, 3 p.m. ET (Amazon Prime Video)

This Friday game the day after Thanksgiving isn't what it was expected to be with the Jets being an offensive disaster.
</item 38>

<item 39>
Best sportsbook bonus offers for NFL Monday Night Football Eagles vs. Seahawks: Claim over $5,000 in bonuses from Bet365, BetMGM, BetRivers, Caesars Sportsbook,
DraftKings and FanDuel  (Sporting News, 2023-12-18T21:00:03+00:00)
After tonight’s game, the Seahawks’ schedule eases up a bit, with winnable games against the Titans, Steelers and Cardinals. But they could really use a victory
to give them some confidence going into the final stretch.

Picks to consider
</item 39>

<item 40>
Chiefs vs. Packers live score, updates, highlights from NFL 'Sunday Night Football' game (Sporting News, 2023-12-03T23:00:04+00:00)
Should've been a pass interference. It wasn't, though. And now, Kansas City could be on the brink of losing.

11:22 p.m.: Whole lot of laundry left back on the field. Looked like Rashee Rice may have fumbled the rock while scampering down the field. However, he was
ruled down with contact. At which point Pacheco threw a punch at Nixon, prompting an ejection. Nice...

11:18 p.m.: Referees make a baffling unnecessary roughness call on Jonathan Owens, gifting the Chiefs an extra 20 yards. That could prove vital.

11:14 p.m.: FIELD GOAL! And it's good! Right down the middle! What a kick by Carlson. Packers lead grows back to eight.

Packers 27, Chiefs 19

11:12 p.m.
</item 40>

<item 41>
Lions vs. Packers live score, updates, highlights from NFL 'Thursday Night Football' game (Sporting News, 2023-09-28T23:54:07+00:00)
Sideline reporter Kaylee Hartung will offer updates throughout the affair.

Fans in the Detroit and Green Bay markets can watch the Amazon broadcast over the air on local channels.

Those in Canada can stream it on DAZN, which carries every NFL game all season.
</item 41>

<item 42>
Cowboys vs. 49ers live score, updates, highlights from NFL 'Sunday Night Football' game (Sporting News, 2023-10-08T22:47:03+00:00)
— The Purdy-to-Aiyuk connection continues to get stronger. Aiyuk goes for 23 yards to give the 49ers some breathing room after opening their drive with a false
start.

49ers 21, Cowboys 10

10:17 p.m. FIELD GOAL — The 49ers' defense tightens the screws and limits Dallas to a field goal. Brandon Aubrey drills it from 50 yards out, and it's down to a
21-10 lead for San Francisco.

10:13 p.m. — Prescott hits CeeDee Lamb for a big 29-yard gain. That quickly takes the Cowboys into 49ers territory to start the second half.

End of first half: 49ers 21, Cowboys 7

9:58 p.m. — San Francisco's defense is all over the Cowboys yet again. It's Dre Greenlaw's turn to sack Prescott, and Dallas is waving the white flag on the
half.
</item 42>

<item 43>
Lions vs. Packers live score, updates, highlights from NFL 'Thursday Night Football' game (Sporting News, 2023-09-28T23:54:07+00:00)
9:15 p.m.: FIELD GOAL! The good times keep on rolling for Detroit, which adds to its lead after another standout drive spearheaded by Goff and Montgomery. The
Lions are rolling early.

Lions 17, Packers 3

End of first quarter: Lions 14, Pacers 3

8:51 p.m.: Impressive play-calling for the Lions, who continue to carve their way up the field.

8:50 p.m.: Love has been unable to find a rhythm in his first few possessions. Another gives Detroit another chance to make the Lions' life a living hell.

8:43 p.m.: TOUCHDOWN! Montgomery goes into his right tackle before finding the end zone. Detroit's office os cooking with gan.

Lions 14, Packers 3

8:42 p.m.: Decent scamper by Goff brings Detroit right to the cusp of pay first. Lions are closed to adding a second score.
</item 43>

<item 44>
Lions vs. Packers live score, updates, highlights from NFL 'Thursday Night Football' game (Sporting News, 2023-09-28T23:54:07+00:00)
Amon-Ra St. Brown was in the midst of a tempest, shrouded in a sea of green with few options around him. The Lions WR had done the hard part, stretching beyond
an exasperated Packers defense to trot into the end zone.

But he had a choice to make: rollick in the joy of his sapphire-clad teammates. Or take a leap of faith into the Lambeau Field abyss.

He opted for the latter. St. Brown hurled himself into the green lagoon. Then, he smiled.

St. Brown's impersonation of the Lambeau Leap was one of many high points in a memorable night for Detroit. Fueled by a stout defensive showing, a gritty run
game and a bit of fairy dust from the likes of St. Brown, Sam LaPorta and Josh Reynolds, the Lions cruised to a 34-20 win over their hated NFC North rivals,
Green Bay.
</item 44>

<item 45>
Seahawks vs. Giants live score, updates, highlights from NFL 'Monday Night Football' game (Sporting News, 2023-10-02T22:00:04+00:00)
: Jamal Adams' return to the field may end up being short-lived.

Adams was on the wrong end of a knee to the helmet and appeared to be shaken up as he needed some assistance getting off the field. More updates to come on
Adams' status.

8:15 p.m.: Seattle wins the coin toss and defers to the second half.

New York elects for a fair catch on the kickoff, meaning Daniel Jones and company will get their night started on the 25-yard line.

Pregame

7:50 p.m.: It's officially official.

Jamal Adams is back on the field warming up for the Seahawks, surely a welcome sight for the 12s.

6:55 p.m.: Saquon Barkley has officially been ruled inactive for the Giants.

Joining Barkley on the inactive list are S Bobby McCain, S Gervarrius Owens, T Andrew Thomas and DL Jordon Riley.
</item 45>

<item 46>
Seahawks vs. Giants live score, updates, highlights from NFL 'Monday Night Football' game (Sporting News, 2023-10-02T22:00:04+00:00)
It was all Seahawks on “Monday Night Football” as the team turned in a dominant defensive effort en route to a 24-3 win over the Giants to wrap up Week 4 of the
NFL season.

In addition to sacking Daniel Jones 11 times, Seattle’s defense forced New York’s QB to commit three turnovers — a fumble and two interceptions, none more
notable than rookie Devon Witherspoon's 97-yard pick-6 that turned the game on its ear.

With the win, the Seahawks move to 3-1 on the season, including a perfect 2-0 record away from home. The Giants drop to 1-3 on the season and have been
outscored 64-3 in two primetime home losses.

The Sporting News tracked score updates and highlights from throughout Monday night's meeting between New York and Seattle.

Seahawks vs.
</item 46>

<item 47>
Chiefs vs. Packers live score, updates, highlights from NFL 'Sunday Night Football' game (Sporting News, 2023-12-03T23:00:04+00:00)
They're back in the postseason race, all on the back of their lithe young hurler.

Miss any of the game? The Sporting News tracked live updates from Sunday's clash under the lights at Lambeau Field below.

Chiefs vs. Packers final score

Q1 Q2 Q3 Q4 Final Chiefs 3 3 6 7 19 Packers 7 7 7 6 27

Chiefs vs. Packers results, highlights from 'Sunday Night Football'

(All times Eastern)

11:28 p.m.: Game. Set. Match. Mahomes' Hail Mary falls harmlessly to the turf, giving the Packers a much-needed win. What a game!

11:25 p.m.: Proper make-up call by the referees there, as Valdez-Scantling was almost certainly impeded illegally while streaking down the field. Should've been
a pass interference. It wasn't, though.
</item 47>

<item 48>
N.F.L. Executives on the Biggest Questions Facing Super Bowl Contenders (The New York Times, 2023-10-16T23:58:53+00:00)
• Detroit Lions: Have they fixed their defense?

What I think: The Lions are 11.3 EPA per game better on defense from last season to this season, the second-largest gain for any team since last season. The
schedule has something to do with that, but the 2022 defense was so bad, it couldn’t stop Carolina. Detroit will win the NFC North and return to the playoffs,
which will make this season successful no matter what happens from there. But we won’t find out until the postseason just how much better the defense might be,
because the remaining schedule features so many weak offenses. That includes Chicago twice, Green Bay, Denver and Las Vegas.

Exec comment: “Yes, I do buy them. This is a league full of teams who do not really know who they are week to week. Detroit knows who they are, and they play to
their identity. Anytime you know who you are and play to that style, you have a chance to be dangerous.”
</item 48>

<item 49>
Chiefs vs. Packers live score, updates, highlights from NFL 'Sunday Night Football' game (Sporting News, 2023-12-03T23:00:04+00:00)
For 59 minutes, the story of Week 13's 'Sunday Night Football' clash was Jordan Love. The first-year starter continued to showcase his evolution as a big-game
player, dicing up Kansas City's All-World defense with incredible efficiency.

Then came the blown calls.

The Packers edged past the Chiefs on Sunday, beneficiaries of another Love masterpiece — an increasingly common sight in the 2023 season. But it was the game's
final moments that drew the most intrigue, mostly because of their incredulous nature.

Jonathan Owens was whistled for an unnecessary roughness that never was on the game's final drive, penalized for crashing into a scrambling Patrick Mahomes on
the field of play. Soon after, Marquez Valdes-Scantling was seemingly impeded by Green Bay cornerback Carrington Valentine on a streak play, an unabashed pass
interference call in most circumstances. The umpires fell silent.
</item 49>

<item 50>
NFL roundup: Pats hit new low as Lions reach 9-3 for first time since 1962 (The Guardian, 2023-12-03T22:11:26+00:00)
Atlanta entered tied atop the division with New Orleans, who lost to Detroit.

But this game appealed only to fans of sloppy play, punting and penalty flags. The Jets (4-8) got an early safety but couldn’t get anything going – again – on
offense in losing their fifth straight game. New York even pulled quarterback Tim Boyle and replaced him with Trevor Siemian in the fourth quarter to try to
provide a spark, but it was more of the same misery in what appeared a half-filled MetLife Stadium.

Cleveland Browns 19–36 Los Angeles Rams

Matthew Stafford threw three touchdown passes, Puka Nacua became the first Rams rookie wide receiver to reach 1,000 yards in a season, and Los Angeles extended
their winning streak to three games. The Rams have won three straight for the first time since winning the Super Bowl after the 2021 season. At 6-6, they are
back in the playoff race after dropping three straight before their bye week.
</item 50>

<item 51>
Chiefs vs. Packers live score, updates, highlights from NFL 'Sunday Night Football' game (Sporting News, 2023-12-03T23:00:04+00:00)
📺: #KCvsGB on NBC

📱: Stream on #NFLPlus https://t.co/CQH1cx3gZS pic.twitter.com/SFNXGRVAga — NFL (@NFL) December 4, 2023

Packers 21, Chiefs 19

End of third quarter: Packers 21, Chiefs 12

10:23 p.m.: Pacheco is as hard-nosed as they come. Just carried an entire pile on his back to take the ball to the goal line. What a showing by the second-year
ball-carrier, who is up over the century mark today.

Isiah Pacheco with a little help from his friends 😮



📺: #KCvsGB on NBC

📱: Stream on #NFLPlus https://t.co/CQH1cx3gZS pic.twitter.com/nAohFvTg0S — NFL (@NFL) December 4, 2023

10:11 p.m.
</item 51>

<item 52>
NFL power rankings: Lions, 49ers level up; Texans jump Bengals, Bills; Jets, Patriots sink to bottom for Week 11 (Sporting News, 2023-11-13T21:54:00+00:00)
Week 10 in the NFL had some of the league's best teams not playing. But that didn't stop the excitement, with five teams winning on walk-off field goals and
more games coming right down to the wire.

With that, there were some surprising results and expected domination. Some favorites dug deep to win. Some underdogs turned in shocking upsets.

Putting the latest wins and losses into a blender, here's Sporting News' updated look at how all teams stack up against each other, 1-32 going into Week 11:

NFL WEEK 11 PICKS: Straight up | Against the spread

NFL power rankings Week 11

1. Philadelphia Eagles 8-1 (previous week: 1)

The Eagles enjoyed a bye trying to get a little healthier for the more difficult stretch run in a top-heavy NFC as they have tough games at the Chiefs (the
Super Bowl 57 rematch), vs. the Bills, vs.
</item 52>

<item 53>
NFL power rankings: Lions, 49ers level up; Texans jump Bengals, Bills; Jets, Patriots sink to bottom for Week 11 (Sporting News, 2023-11-13T21:54:00+00:00)
Their playoff hopes look much better after their stunning comeback vs. the Ravens.

9. Jacksonville Jaguars 6-3 (4)

The Jaguars got an NFC wake-up call against the 49ers, who were much more prepared after a Week 9 bye than they were. Jacksonville needs to forget that debacle
and get back on track, stat. The team gets a division double vs. Tennessee and Houston before facing Cincinnati.

10. Dallas Cowboys 6-3 (11)

The Cowboys took advantage of their "no contest" vs. the Giants at home with Dak Prescott continuing to ball as a passer and runner. They need to keep the
defense and running game success, too, in bigger games going forward.

MORE: Inside the Cowboys historic offensive explosion vs. Giants

11.
</item 53>

<item 54>
NFL power rankings: Lions, 49ers level up; Texans jump Bengals, Bills; Jets, Patriots sink to bottom for Week 11 (Sporting News, 2023-11-13T21:54:00+00:00)
But the reeling defense and pass protection continue to be un-winnable assets.

24. Atlanta Falcons 4-5 (20)

The Falcons were forced into another QB change because of injury. They found a few more answers offensively but that position still made it an overall mess. The
defense also doesn't seem to be well prepared to different styles of offenses.

25. Green Bay Packers 3-6 (25)

The Packers saw some improved work from Jordan Love and his young weapons as they spread the ball well in the passing game, but the run defense woes and key
holes in coverage have all but doomed their playoff hopes.

26. Los Angeles Rams 3-6 (26)

The Rams are hoping to find a few more answers during a bye, as Sean McVay has seen his good coaching job earlier in the season go by the wayside with key
offensive injuries.
</item 54>

<item 55>
NFL power rankings: Lions, 49ers level up; Texans jump Bengals, Bills; Jets, Patriots sink to bottom for Week 11 (Sporting News, 2023-11-13T21:54:00+00:00)
21. New York Jets 4-5 (18)

Aaron Rodgers can't come back soon enough, and going for a December miracle might be too late after two straight key losses to the Chargers and Raiders. Zach
Wilson will always cap the potential of a talented offense that keeps making mistakes. The Bills and Dolphins in back-to-back division games next might be their
season.

22. Denver Broncos 4-5 (24)

The Broncos got another big win against a AFC power, sandwiching their Week 9 bye with upsets over the Chiefs and Bills. The offense has hit a balanced grooved
with Russell Wilson and the defense is making a lot more plays, too.

23. Washington Commanders 4-5 (21)

The Commanders should be pleased about Sam Howell and the offense showing more fight with additional weapons and the running backs look better all-around again.
But the reeling defense and pass protection continue to be un-winnable assets.

24.
</item 55>

<item 56>
Lions vs. Packers live score, updates, highlights from NFL 'Thursday Night Football' game (Sporting News, 2023-09-28T23:54:07+00:00)
But Green Bay is firmly back in it.

Lions 27, Packers 17

End of third quarter: Lions 27, Packers 11

10:31 p.m.: Impressive pitch-and-catch by Love and Jayden Reed. Packers hurler drops it right into Reed's bread basket for the 44 yard gain as the quarter comes
to an end.

10:24 p.m.: Things slow down for Green Bay on Drive No. 2 of the second half. Punt incoming. Packers need less of that as they attempt their second-straight
rally in as many weeks.

10:15 p.m.: Green Bay defense firms up, forcing yet another punt. Can Love and the Packers bring things back?

10:10 p.m.: TOUCHDOWN! Love finds Watson for the TD, giving Packers fans a much-needed boost of adrenaline. Good drive by Green Bay, which finally shows signs
of life after a putrid first half.
</item 56>

<item 57>
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
</item 57>

<item 58>
NFL roundup: Pats hit new low as Lions reach 9-3 for first time since 1962 (The Guardian, 2023-12-03T22:11:26+00:00)
Brock Purdy threw for 314 yards and four touchdowns, Deebo Samuel scored three TDs and Eagles quarterback Jalen Hurts was checked for a concussion in the San
Francisco 49ers’ 42-19 win over Philadelphia on Sunday.

The fiery NFC championship game rematch also featured 49ers’ touchdowns on six straight possessions and a pair of ejections that showed this year’s title game
may not run again through Philly, after all.

49ers' Dre Greenlaw has been disqualified from tonight's game after this incident on the Eagles sideline. pic.twitter.com/2N4W17N8u9 — FOX Sports: NFL
(@NFLonFOX) December 3, 2023

The 49ers (9-3) left Philly 10 months ago battered in the NFC championship game behind injuries to Purdy and his backup, and they never stood a chance in a 31-7
loss.
</item 58>

<item 59>
Jets vs. Raiders live score, updates, highlights from NFL 'Sunday Night Football' game (Sporting News, 2023-11-12T23:30:03+00:00)
For the Jets, it marked a second straight game without a touchdown, and gives them their second consecutive loss before hitting the road to face the Bills in
Week 11.

MORE: Revisiting Antonio Pierce's NFL playing career

Sporting News tracked all the scores and highlights from Jets vs. Raiders on "Sunday Night Football." Check out all of the key moments you might have missed.

Jets vs. Raiders final score

1 2 3 4 F Jets 6 3 0 3 12 Raiders 3 3 3 7 16

Jets vs. Raiders results, highlights from 'Sunday Night Football'

(All times Eastern)

Fourth quarter

11:26 p.m.: Wilson's final pass after being chased out of the pocket is batted down, and the Raiders hang on to win 16-12.

11:24 p.m.: It's time for Hail Marys, and the first to Gipson sails out of bounds.
</item 59>

<item 60>
Cowboys vs. 49ers live score, updates, highlights from NFL 'Sunday Night Football' game (Sporting News, 2023-10-08T22:47:03+00:00)
A perfectly drawn-up 26-yard play puts Dallas right back in this game.

KaVontae Turpin shows off the wheels and gets in the end zone 💨 @KaVontaeTurpin



📺: #DALvsSF on NBC

📱: Stream on #NFLPlus https://t.co/gzzkvIRUdj pic.twitter.com/1bwEuaj5pN — NFL (@NFL) October 9, 2023

9:24 p.m. — Prescott hits Jake Ferguson for a big third down conversion, and then some. That's 20 yards.

9:23 p.m. — Tony Pollard has had a tough night so far, with just 10 yards on six touches.

9:22 p.m. — Prescott fires a nice pass to Cooks to give the Cowboys another first down.

9:20 p.m. — The Cowboys have a first down! And it only took 17+ minutes.
</item 60>

<item 61>
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
</item 61>

<item 62>
Prisco's Week 12 NFL picks: Jaguars beat Texans in AFC South battle, Saints get key divisional win vs. Falcons (CBSSports.com, 2023-11-23T21:30:38+00:00)
Thursday, 12:30 p.m. ET (Fox, fubo)

This is a big game in the division for both teams, a survival test for Green Bay and a big one for the Lions if they want to try and get home-field advantage in
the conference. The Lions have had defensive issues as of late and Jordan Love has played well the past two weeks. That will keep the Packers in this game, but
the Lions win it.

Pick: Lions 31, Packers 27 (Result: Packers 29, Lions 22)

Thursday, 4:30 p.m. ET (CBS, Paramount+)

The Commanders are coming off a dud last week against the Giants, while Dallas is flying high. The Cowboys are playing as well as any team in the league.
Washington's offensive line is an issue, which will lead to the Cowboys having a field day with their defensive front. Dak Prescott plays well again. Cowboys
roll.
</item 62>

<item 63>
NFL standings: Updated AFC, NFC playoff picture for Week 13 of 2023 season (Sporting News, 2023-12-03T21:49:04+00:00)
7. Green Bay Packers (6-6), third place, NFC North. The Packers moved into the final playoff spot thanks to having the second-best conference record after the
Vikings, to whom they already lost but face again in Minnesota in Week 17. The Packers also hold key wins over the Rams and Saints in the wild-card race.

Who's in the hunt?

8. Los Angeles Rams (6-6)

9. Seattle Seahawks (6-6)

10. Tampa Bay Buccaneers (5-7)

11. New Orleans Saints (5-7)

12. New York Giants (4-8)

13. Chicago Bears (4-8)

14. Washington Commanders (4-9)

The Rams and Seahawks are the best wild-card bets here now, while the Saints and Buccaneers will still have their sights on making the playoffs via still
winning the weak South over the Falcons.
</item 63>

<item 64>
Cowboys vs. 49ers live score, updates, highlights from NFL 'Sunday Night Football' game (Sporting News, 2023-10-08T22:47:03+00:00)
The 49ers knocked the Cowboys out of the playoffs in both of the last two seasons, so a win for Dallas on Sunday could be helpful just from a psychological
standpoint.
</item 64>

<item 65>
Seahawks vs. Giants live score, updates, highlights from NFL 'Monday Night Football' game (Sporting News, 2023-10-02T22:00:04+00:00)
It's a three-touchdown lead for Seattle now with just over five minutes to play.

Seahawks 24, Giants 3

10:59 p.m.: INTERCEPTION! The nightmarish game for Daniel Jones continues as he commits his third turnover of the game, a pass intercepted by Seahawks safety
Quandre Diggs.

After being sacked nine times and committing three turnovers, Jones may not have much more time on the field tonight.

10:56 p.m.: New York forces a three-and-out to get the ball back and on the first offensive play, Daniel Jones is sacked again.

Jones has now been sacked nine times in the game. It's a rough showing considering Seattle entered the week with just five sacks on the season.

10:48 p.m.: …And ANOTHER sack.

Daniel Jones has now been sacked eight times in the game and New York will be forced to punt as it faces fourth-and-31.

Fans are starting to leave the Meadowlands as it's been all Seahawks tonight.
</item 65>

<item 66>
Seahawks vs. Giants live score, updates, highlights from NFL 'Monday Night Football' game (Sporting News, 2023-10-02T22:00:04+00:00)
The extra point is good! And Seattle turns a turnover into seven points.

Seahawks 7, Giants 0

8:54 p.m.: FUMBLE! After evading one sack, Daniel Jones is sacked at the Giants 15-yard line by Mario Edwards, who also forces the ball out.

Seattle recovers and returns the fumble to New York's 7-yard line, where it will look to score for the first time tonight.

8:47 p.m.: Flags… flags everywhere.

Four straight plays with a penalty for the Seahawks means their drive officially starts with a first-and-20. As the game approaches the two-minute mark of the
first quarter, neither team has been able to strike offensively.

8:35 p.m.: Jamal Adams has been ruled out for the remainder of the game.

8:34 p.m.: And a few more bizarre plays. It's a three-and-out for Seattle, which sees its drive end courtesy of a sack from Kayvon Thibodeaux.
</item 66>

<item 67>
Seahawks-Cowboys Fantasy Start 'Em Sit 'Em Week 13: Should you start Zach Charbonnet, Jake Ferguson on TNF? (Sporting News, 2023-11-30T21:31:00+00:00)
WEEK 13 DFS:

DK lineup | FD lineup | Best stacks | Best values

Cowboys: Who to start, sit in fantasy football on TNF

Start: QB Dak Prescott

Guess the only two quarterbacks who have been better than Rayne Dakota in fantasy this season? Yep, after his sizzling seven-game stretch, Prescott has ascended
behind just Josh Allen and Jalen Hurts. The Seahawks haven't given up many big games to QBs, but you need to keep rolling with Dak on his heater, especially at
home. One can't be surprised, however, if it's more of a 250-and-2 kind of night.

Start: RB Tony Pollard | Sit: RB Rico Dowdle

Pollard and the Cowboys' defense is a reason why Prescott may not need to go nuts passing. The Seahawks have now joined the bottom five teams in giving up
fantasy points to running backs.
</item 67>

<item 68>
Cowboys vs. 49ers live score, updates, highlights from NFL 'Sunday Night Football' game (Sporting News, 2023-10-08T22:47:03+00:00)
8:27 p.m. — A little bit of chaos as Deebo Samuel drops a ball that was pretty close to being a fumble, but it's all a non-issue as Dallas is hit with an
offsides penalty. The 49ers are right about in field goal range.

8:25 p.m. — After a facemask penalty and a nine-yard pass to Brandon Aiyuk, the 49ers are already over midfield.

8:23 p.m. — Kickoff is a bit on the later end, but we are about to get underway. The 49ers will start with the ball.

8:10 p.m. — Jerry Jones said this week that this game will be a test of how the Cowboys match up with the NFL's best, as if there wasn't already pressure on
Dallas to show something after the 49ers knocked them out of the playoffs in back-to-back years.

7:56 p.m. — Frank Gore and Vernon Davis are in the house!
</item 68>

<item 69>
Raiders vs. Lions live score, updates, highlights from NFL 'Monday Night Football' game (Sporting News, 2023-10-30T22:20:03+00:00)
The Sporting News tracked live updates and highlights throughout the "Monday Night Football" matchup between the Raiders and Lions. Check out all of the key
moments you might have missed.

Raiders vs. Lions final score

1 2 3 4 Final Raiders 0 7 7 0 14 Lions 3 13 7 3 26

Raiders vs. Lions results, highlights from 'Monday Night Football'

(All times Eastern)

Final: Lions 26, Raiders 14

11:15 p.m. — That'll do it! The Lions bounce back and move to 6-2 with the win over Las Vegas. Detroit left points on the board, but the Raiders were thoroughly
outplayed in this one.

11:09 p.m. — The Lions get the ball back and will try to ice the game at this point.

11:08 p.m. — Adding insult to injury, Garoppolo is sacked for the sixth time tonight.
</item 69>

<item 70>
NFL power rankings: Lions, 49ers level up; Texans jump Bengals, Bills; Jets, Patriots sink to bottom for Week 11 (Sporting News, 2023-11-13T21:54:00+00:00)
the Bills, vs. 49ers and at the Cowboys in the next four. They still look like the conference team to beat as they have proved they still win in many ways
around Jalen Hurts.

2. Kansas City Chiefs 7-2 (3)

The Chiefs had a pretty good week off given they jumped back into the AFC's top seed and home-field advantage spot with the Ravens losing. They already are
rolling defensively and expect them to figure out a lot more around Patrick Mahomes after the key Andy Reid bye.

3. Detroit Lions 7-2 (5)

The Lions are becoming like the Eagles, and that's what makes them the second-best team in the NFC. They can win shootouts with Jared Goff, Amon-Ra St. Brown
and their passing game. They also can dominate with the running game. They can win with offense or defense. Dan Campbell is coach of the year.

4.
</item 70>

<item 71>
Lions vs. Packers live score, updates, highlights from NFL 'Thursday Night Football' game (Sporting News, 2023-09-28T23:54:07+00:00)
Thursday's game appeared to be destined to be a showcase of quarterbacking excellence; Jared Goff and Jordan Love have proved themselves to be formidable
options under center since taking the reins of their respective offenses.

But it was the supporting cast that proved so pivotal to Thursday's result for Dan Campbell's side. David Montgomery was a nuisance, slipping through crevices
to puncture the heart of the Packers' defense. When it wasn't him, it was that receiving corp, made up of unheralded youngsters who have all come up trumps in
recent weeks.

As for Love and Co., Thursday's affair was another indication of just how much work there is to be done before they compete for the divisional crown. There were
certainly some flits of delight, throws that highlighted just how much talent Love possesses in his golden arm.

But there were low points, too, misshapen efforts that fell harmlessly to the turf, or harmfully into the opposition's hands.
</item 71>

<item 72>
Cowboys vs. 49ers live score, updates, highlights from NFL 'Sunday Night Football' game (Sporting News, 2023-10-08T22:47:03+00:00)
1 seed is undoubtedly in the back of their minds.

6:45 p.m. — The run of great prime-time weather continues, as it's set to be in the high 70s and sunny in Santa Clara tonight. No excuses for these powerful
offenses.

What channel is Cowboys vs. 49ers on today?

Game: Cowboys vs. 49ers

Cowboys vs. 49ers Date: Sunday, Oct. 8

Sunday, Oct. 8 TV channel: NBC

NBC Live stream: Peacock | NBCSports.com | Fubo (U.S.) | DAZN (Canada)

Cowboys vs, 49ers can be watched on NBC, which has long held the broadcast rights to "Sunday Night Football."

Mike Tirico (play-by-play) and Cris Collinsworth (color analyst) will be on the call from Levi's Stadium while Melissa Stark will serve as the sideline
reporter.
</item 72>

<item 73>
NFL power rankings: Lions, 49ers level up; Texans jump Bengals, Bills; Jets, Patriots sink to bottom for Week 11 (Sporting News, 2023-11-13T21:54:00+00:00)
They also seemed invigorated for Jonathan Gannon on defense and special teams. Look for the Cardinals to finish strong and play a whole lot of spoiler.

30. New England Patriots 2-8 (28)

The Patriots tried to win a game by hiding Mac Jones and their quarterbacks in general. They almost pulled it off with the running game dominating and
short-handed defense making a lot of plays, but they didn't see the results in Germany ahead of a tough bye for Bill Belichick.

31. New York Giants 2-8 (30)

The Giants can just flush away this season without Daniel Jones and Tyrod Taylor. The injuries go to their offense early and remained relentless, and now their
defense is falling apart, too. The question is, will Brian Daboll get a massive mulligan after his rookie playoff leadership?

32.
</item 73>

<item 74>
Seahawks vs. Giants live score, updates, highlights from NFL 'Monday Night Football' game (Sporting News, 2023-10-02T22:00:04+00:00)
New York has been dominated at home twice this season now, as this performance follows a 40-0 blowout in Week 1 on "Sunday Night Football." Daniel Jones' career
mark in primetime falls to 1-13 and 0-3 this season.

11:09 p.m.: TURNOVER ON DOWNS! Daniel Jones is sacked for an 11th time as Seattle takes back over at the New York 35-yard line.

Geno Smith's night appears to be done as Drew Lock is back in for the Seahawks.

11:06 p.m: Daniel Jones is back under center for New York, which will look to drive and score to at least lessen the damage on the scoreboard.

11:04 p.m.: FIELD GOAL! The turnover leads to three points for the Seahawks, who get a 34-yard field goal from Jason Myers. It's a three-touchdown lead for
Seattle now with just over five minutes to play.
</item 74>

<item 75>
NFL power rankings: Lions, 49ers level up; Texans jump Bengals, Bills; Jets, Patriots sink to bottom for Week 11 (Sporting News, 2023-11-13T21:54:00+00:00)
They need Matthew Stafford to help save face the rest of the way or the season will look a lot like 2022.

27. Tennessee Titans 3-6 (27)

Will Levis is going through rookie lumps after his dazzling debut as the brutal offensive line woes exacerbated by injuries are getting to him and the entire
Titans offense. The Titans still need to keep starting him, because they can't overcome those systemic problems or those in overall pass defense.

28. Chicago Bears 3-7 (29)

The Bears don't care how they win, but they should be pleased their running game and defense bailed out their limited passing game with Tyson Bagent to secure
another win in Week 10. They will hope to finish strong with Justin Fields soon.

29. Arizona Cardinals 2-8 (32)

The Cardinals got an immediate spark for their offense with Kyler Murray and James Conner back.
</item 75>

<item 76>
Cowboys vs. 49ers live score, updates, highlights from NFL 'Sunday Night Football' game (Sporting News, 2023-10-08T22:47:03+00:00)
49ers 35, Cowboys 10

10:47 p.m. TOUCHDOWN — Purdy hits Kyle Juszczyk for the 49ers' fifth touchdown of the night. That's four touchdowns for Purdy alone, and San Francisco is
pouring it all on the Cowboys.

End of third quarter: 49ers 28, Cowboys 10

10:43 p.m. — San Francisco is moving the ball inch-by-inch now, and the 'Niners are a yard away from making this a 25-point game.

10:38 p.m. — The 49ers are on the verge of entering the red zone after a 19-yard catch by Jauan Jennings. Purdy is up to 242 yards to go along with his three
touchdowns.

10:32 p.m. INTERCEPTION — Prescott heaves it downfield under pressure and it's intercepted by Tashaun Gipson.
</item 76>

<item 77>
Seahawks-Cowboys Fantasy Start 'Em Sit 'Em Week 13: Should you start Zach Charbonnet, Jake Ferguson on TNF? (Sporting News, 2023-11-30T21:31:00+00:00)
He should be good for a couple of field goals to complement the Cowboys' touchdowns.

Seahawks vs. Cowboys: DraftKings Showdown lineup

Captain (1.5X): Cowboys RB Tony Pollard $14,700

FLEX: Cowboys QB Dak Prescott $11,000

FLEX: Seahawks WR Tyler Lockett $7,200

FLEX: Cowboys TE Jake Ferguson $6,200

FLEX: Cowboys D/ST $5,600

FLEX: Seahawks WR Jaxon Smith-Njigba $5,200

Based on the analysis, it's fine to invest plenty in the Cowboys' running and intermediate passing games. Pollard would be a good correlated stack with the
Cowboys' defense. Here's hoping that Lockett and JSN will be the busiest Seahawks in second-half catchup mode.

Seahawks vs.
</item 77>

<item 78>
Seahawks vs. Giants live score, updates, highlights from NFL 'Monday Night Football' game (Sporting News, 2023-10-02T22:00:04+00:00)
The Giants muff the punt from Michael Dixon but avoid a crisis by recovering it. New York will take the floor for its second offensive drive from Seattle's
34-yard line.

8:28 p.m.: We've got our first bizarre play of the night.

Seahawks RB Kenneth Walker III runs for a touchdown after he isn't ruled down but the play is called back. Instead of a 73-yard touchdown run, it's a two-yard
loss.

8:25 p.m.: TURNOVER ON DOWNS! On fourth-and-1 from the Seahawks' 27-yard line, Daniel Jones is stopped in his tracks as he attempts a QB sneak.

After an impressive defensive stand, Geno Smith and the Seattle offense will take the field for the first time tonight.

8:22 p.m.: Jamal Adams' return to the field may end up being short-lived.
</item 78>

<item 79>
Seahawks-Cowboys Fantasy Start 'Em Sit 'Em Week 13: Should you start Zach Charbonnet, Jake Ferguson on TNF? (Sporting News, 2023-11-30T21:31:00+00:00)
Jourdan Lewis. JSN can mirror some of the production Curtis Samuel gave the Commanders last week in Dallas, but forget 100 yards. Around 50 yards seems like the
ceiling with a big boost from garbage time.

Sit: TEs Noah Fant and Will Dissly

Noah Gray is a better fantasy player than Noah Fant this season. Will Dissly is barely a better fantasy player than Will Mallory this season.

WEEK 13 STANDARD RANKINGS:

QBs | RBs | WRs | TEs | D/STs | Kickers

Sit: Seahawks D/ST

The Seahawks are getting better defensively against the pass, but that's not enough to trust them here, as they're getting gashed on the ground, too.

Sit: K Jason Myers

This is actually not a good spot for one of the most reliable fantasy kickers, given the Cowboys don't give up many field-goal attempts.
</item 79>

<item 80>
NFL standings: Updated AFC, NFC playoff picture for Week 13 of 2023 season (Sporting News, 2023-12-03T21:49:04+00:00)
2. San Francisco 49ers (9-3), first place, NFC West. The 49ers' red-hot ways since getting fully healthy around Brock Purdy after their bye continue with a
dominant offensive performance against the Eagles. They made a statement that they are the new team to beat in the conference, wherever they play. Up three
games on the Rams and Seahawks, another division title is coming, too.

3. Detroit Lions (9-3), first place, NFC North. The Lions are still in great shape to win the division with a 3-game lead over the Vikings and the Packers after
beating the Saints in New Orleans in Week 13. But they're looking more like the No. 2 seed will be their ceiling.

4. Atlanta Falcons (6-6), first place, NFC South. The Falcons are in control of the NFL's weakest division thanks to beating the Saints and earning the
head-to-head tiebreaker in Week 12.
</item 80>

<item 81>
Lions vs. Packers live score, updates, highlights from NFL 'Thursday Night Football' game (Sporting News, 2023-09-28T23:54:07+00:00)
Lions are closed to adding a second score.

8:40 p.m.: Since the interception, it's felt as if Goff and the Lions can get anything they want. Gibbs is lightning in a bottle, while Montgomery brings the
thunder. This drive just continues to get lengthier and lengthier.

8:34 p.m.: Lions are looking menacing as can be in the trenches. Isaiah Buggs sends Love spiraling with a crunching hit behind the line. The Packers are giving
the ball back to the Lions on a three-and-out.

8:29 p.m.: TOUCHDOWN! Amon-Ra St. Brown leaps, stretches and reels in Goff's feather towards the corner of the end zone! Detroit showing just what its offense
can do when its clicking.

Lions 7, Packers 3

8:27 p.m.: Sam LaPorta gets beyond his marker with a strong move and plods 35 yards down the field.
</item 81>

<item 82>
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
</item 82>

<item 83>
Cowboys vs. 49ers live score, updates, highlights from NFL 'Sunday Night Football' game (Sporting News, 2023-10-08T22:47:03+00:00)
Jerry Jones said earlier in the week that Sunday night's game would show him how the Cowboys stack up against the NFL's best.

The answer? Not very well.

Everything that could go wrong for Dallas on Sunday did go wrong, in a game that confirmed a 49ers team that ended the Cowboys' season in each of the last two
years remains a level above its NFC counterpart.

The 49ers carved up Dallas' defense on the opening drive, and that was a sign of what was to come. Brock Purdy threw four touchdowns in the 42-10 rout,
including three to George Kittle. Christian McCaffrey scored a touchdown for a 14th consecutive game as all of San Francisco's offensive weapons got involved.

MORE: Comparing stats for NFC stars Nick Bosa, Micah Parsons

Defensively, the 49ers were relentless. Nobody had a larger impact than Fred Warner, who finished the night with an interception, sack and forced fumble in a
swarming performance.
</item 83>

<item 84>
Chiefs vs. Packers live score, updates, highlights from NFL 'Sunday Night Football' game (Sporting News, 2023-12-03T23:00:04+00:00)
He's unable to find the target again on the two-point conversion, however, keeping the Packers in control (just barely).

Packers 14, Chiefs 12

9:54 p.m.: Mahomes does it again! Delicate touch pass between the Magic Man and Kelce results in yet another first down on third-and-long. Kelce's up to 81
yards on four grabs. Kansas City is threatening to breach into the red zone.

9:47 p.m.: Mahomes hits Kelce with a floated effort as the Chiefs open the first half with a flourish. Need more of that going forward.

End of second quarter: Packers 14, Chiefs 6

9:10 p.m.: FIELD GOAL! Kansas City's explosive offense has been held in check thus far. Despite showing flashes of what they can accomplish, the Chiefs were
held to another field goal in this one. Yikes...

Packers 14. Chiefs 6

9:00 p.m.
</item 84>

<item 85>
Cowboys vs. 49ers live score, updates, highlights from NFL 'Sunday Night Football' game (Sporting News, 2023-10-08T22:47:03+00:00)
For a Cowboys team that has embarrassed a few opponents this year, they might be on the bad end of an ugly game here.

49ers 28, Cowboys 10

10:27 p.m. TOUCHDOWN — Not one, not two, THREE touchdowns for George Kittle. The 49ers have everything working offensively tonight, and any Kittle fantasy
owners are thrilled.

10:25 p.m. — Aiyuk's loss is Deebo Samuel's gain. Samuel goes 43 yards on the very next play, which tells you how automatic the 49ers' offense is right now. San
Francisco is in the red zone looking to make this a three-score game.

10:23 p.m. — An offensive holding call wipes out a 40-yard gain from Aiyuk. Wow. That's a momentum swing. Spencer Burford is the culprit.

10:21 p.m. — The Purdy-to-Aiyuk connection continues to get stronger.
</item 85>

<item 86>
Lions vs. Packers live score, updates, highlights from NFL 'Thursday Night Football' game (Sporting News, 2023-09-28T23:54:07+00:00)
The rookie tight end has been producing at a historic clip so far. And it looks like he's on pace for another standout showing this week.

8:26 p.m.: Jahmyr Gibbs and David Montgomery look a formidable duo in the backfield. Just collaborated to push the Lions forward as they move the chains for the
first time today.

8:21 p.m.: FIELD GOAL! Packers open the scoring courtesy of an Anders Carlson field goal. Lions defense held tough, all things considered. Aidan Hutchinson
burst through the trenches to send Love tumbling to precipitate the stop.

Packers 3, Lions 0

8:14 p.m.: INTERCEPTION! Rudy Ford swipes Jared Goff's errant pass from the sky and takes it back 16 yards to put the Packers on the cusp of the end zone!
That's Goff's third interception in as many games!

8:13 p.m.: Lions starting the game with the ball.
</item 86>

<item 87>
NFL power rankings: Lions, 49ers level up; Texans jump Bengals, Bills; Jets, Patriots sink to bottom for Week 11 (Sporting News, 2023-11-13T21:54:00+00:00)
The defense can't tolerate those mistakes with all of its injuries.

16. Las Vegas Raiders 5-5 (22)

The Raiders are hot, having won two consecutive games for interim coach Antonio Pierce, bringing up flashbacks to the playoff push they made with Rich Bisaccia
in 2021. They are right back in the AFC wild-card race with the Dolphins, Chiefs, Vikings ahead.

17. New Orleans Saints 5-5 (15)

The Saints will go into a bye not knowing what do at quarterback. with Derek Carr further hurting his right shoulder and needed to be checked for a concussion.
Jameis Winston did provide a spark, too little too late at Minnesota. But the division lead is precarious with poor QB play overall.

19. Indianapolis Colts 5-5 (19)

The Colts stopped scoring at least 20 points in every game during their ugly, run-heavy victory in Germany vs. the Patriots.
</item 87>

<item 88>
Seahawks vs. Giants live score, updates, highlights from NFL 'Monday Night Football' game (Sporting News, 2023-10-02T22:00:04+00:00)
Fans are starting to leave the Meadowlands as it's been all Seahawks tonight.

10:45 p.m.: Daniel Jones has now been sacked seven times in the game.

Bobby Wagner gets Jones again, setting up second-and-28 for the Giants, who were already at a disadvantage thanks to a penalty on first down.

End of third quarter: Seahawks 21, Giants 3

10:33 p.m.: PICK SIX! In the worst possible outcome for the Giants, Devon Witherspoon sits on an underneath route and intercepts Daniel Jones, returning the
pick 97 yards for a touchdown. The PAT is good!

What could have been a one-possession game is now a three-score game. And Daniel Jones now has two turnovers in this one.

Seahawks 21, Giants 3

10:30 p.m.: The Giants just pulled off one of their biggest plays of the night.
</item 88>

<item 89>
NFL roundup: Pats hit new low as Lions reach 9-3 for first time since 1962 (The Guardian, 2023-12-03T22:11:26+00:00)
Carr was hurt on a penalized hit by Bruce Irvin. New Orleans fell to 5-7, behind first-place Atlanta in the NFC South.

Kansas City Chiefs 19-27 Green Bay Packers

The Green Bay Packers picked up their fourth win in five games as they beat the defending Super Bowl champions. The 6-6 Packers are now in the NFC wildcard
places after Sunday night’s win.

Denver Broncos 17-22 Houston Texans

Nico Collins had a career-high 191 yards receiving and a fourth-quarter touchdown, and Jimmie Ward intercepted Denver’s Russell Wilson in the end zone with nine
seconds left to left to help the Houston (7-5) Texans hold on for a win over the Broncos. Ward leapt in front of intended receiver Lucas Krull to secure the
fourth victory in five games for Houston and snap a five-game winning streak for Denver (6-6). Wilson was intercepted a season-high three times, all of them
after halftime.
</item 89>

<item 90>
Lions vs. Packers live score, updates, highlights from NFL 'Thursday Night Football' game (Sporting News, 2023-09-28T23:54:07+00:00)
Lions 34, Packers 17

10:50 p.m.: Quay Walker's leaps over a blocker on Riley Patterson's field goal attempt, prompting a unsportsmanlike penalty. That will put Detroit on the cusp
of another score, pending a Packers defensive rally.

10:43 p.m.: Lions beginning to salt the game away with strong running through the tackles. Montgomery and Gibbs creeping through the crevices to push Detroit
down the field. And an impressive toss from Goff into the outstretched hands of Reynolds, as well. The Lions are on the doorstep of the red zone.

10:36 p.m.: TOUCHDOWN! Love scampers into the end zone from short distance to cut that Detroit lead down even more. He got held short of the end zone on the
ensuing two-point effort. But Green Bay is firmly back in it.
</item 90>

<item 91>
Jets vs. Raiders live score, updates, highlights from NFL 'Sunday Night Football' game (Sporting News, 2023-11-12T23:30:03+00:00)
If you tuned into "Sunday Night Football" expecting to see some offense, you tuned in on the wrong weekend.

The Jets and Raiders were locked into an old-school, defense-heavy primetime clash that featured nothing but six field goals through the first three quarters of
the game. It wasn't until 50 seconds into the fourth quarter that Las Vegas finally picked up the first touchdown, with Michael Mayer making a leaping grab in
the end zone to put the Raiders up 16-9.

In such a low-scoring game, that touchdown felt, even as early as it was, that it could be a game-winner. And indeed, it was as the Raiders held the Jets to
just a field goal the rest of the game and picked off Zach Wilson late to stymy New York in a 16-12 win.

MORE: What to know about Raiders 'SNF' hero Robert Spillane

It wasn't exactly a quarterback showdown that many might have expected in the preseason when this game appeared to be Aaron Rodgers vs.
</item 91>

<item 92>
Cowboys vs. 49ers live score, updates, highlights from NFL 'Sunday Night Football' game (Sporting News, 2023-10-08T22:47:03+00:00)
— The Cowboys have a first down! And it only took 17+ minutes. Dallas is still operating in its own territory, but it feels like points are needed on this drive
to keep the game within reach.

49ers 14, Cowboys 0

9:14 p.m. TOUCHDOWN — That was quick! Purdy hits Kittle after some trickery, and it's a two-score night for the tight end. 49ers lead, 14-0.

9:13 p.m. — The Cowboys are forced to punt out of their own end zone after Fred Warner sacks Prescott. Warner now has a sack and a fumble. The 49ers will have
strong field position.

9:10 p.m. FUMBLE — Oh no again! They're trading fumbles! Jourdan Lewis takes the ball right out of McCaffrey's arms and spoils the 49ers' hopes of making this a
14-0 game.
</item 92>

<item 93>
Seahawks-Cowboys Fantasy Start 'Em Sit 'Em Week 13: Should you start Zach Charbonnet, Jake Ferguson on TNF? (Sporting News, 2023-11-30T21:31:00+00:00)
When the Seahawks and Cowboys meet in Dallas on "Thursday Night Football" to open NFL Week 13 (8:15 p.m. ET, Amazon Prime Video), it will be a prime-time
matchup featuring two potential playoff teams and plenty of key fantasy football start 'em, sit 'em question marks.

With a game total of 47.5 and the Cowboys favored by three field goals, it may be another rout for the home team against a weaker opponent. Dallas has some
red-hot fantasy assets, while Seattle's struggling offensive skill players might be best suited for fantasy benches.
</item 93>

<item 94>
Seahawks-Cowboys Fantasy Start 'Em Sit 'Em Week 13: Should you start Zach Charbonnet, Jake Ferguson on TNF? (Sporting News, 2023-11-30T21:31:00+00:00)
The Seahawks have now joined the bottom five teams in giving up fantasy points to running backs. Pollard has come on the past two weeks against the Panthers and
Commanders after sliding into RB2 range. Pollard scoots to something like 80 and a score. Dowdle is a stretch to get mop-up production, however,

Start: WR CeeDee Lamb

Lamb can eat again in the slot against Jamal Adams, as he avoids the Seahawks' better cornerbacks on the outside. Again, don't expect a massive game because the
running game, defense, and special teams will have plenty of success in another big victory.

Sit: WR Brandin Cooks

Cooks is the one Cowboy to fade this week vs. rookie Devon Witherspoon, who's quickly emerging as the Seahawks' best corner. The nature of the zone defense is
to limit big plays.
</item 94>

<item 95>
Cowboys vs. 49ers live score, updates, highlights from NFL 'Sunday Night Football' game (Sporting News, 2023-10-08T22:47:03+00:00)
11:00 p.m. INTERCEPTION — So much for that. Oren Burks gets the 49ers' third interception of the night as Prescott continues to struggle against this San
Francisco defense.

10:59 p.m. — The Cowboys are moving the ball now, when it's a bit too late. Tony Pollard rips off a 26-yard run to finally break through.

49ers 42, Cowboys 10

10:53 p.m. TOUCHDOWN — And just like that, Jordan Mason runs in for another 49ers touchdown! This is ugly on all levels for the Cowboys, and Leighton Vander
Esch is now down as well. 42-10, 49ers.

10:52 p.m. INTERCEPTION — Fred Warner now has a sack, forced fumble and an interception tonight. A pass from Prescott is deflected into his hands, and the 49ers
have another instant scoring opportunity.
</item 95>

<item 96>
Seahawks vs. Giants live score, updates, highlights from NFL 'Monday Night Football' game (Sporting News, 2023-10-02T22:00:04+00:00)
Graham Gano connects on a 55-yarder to get the Giants on the board. Since the beginning of last season, Gano is 10 of 11 on attempts that are 50 yards or
longer.

Seahawks 7, Giants 3

9:11 p.m.: We've got some fireworks on the field.

Geno Smith catches his own batted pass and on the ensuing run, is pulled down out of bounds by Giants LB Isaiah Simmons. Smith, clearly taking exception to the
hit, comes back on the field and has words for S Xavier McKinney.

New York almost forces a turnover on third down but still gets off the field. Smith appears to be slightly hobbled after the hit.

End of first quarter: Seahawks 7, Giants 0

8:56 p.m.: TOUCHDOWN! As time expires in the first quarter, Geno Smith rolls out and finds DK Metcalf, who hauls in his second touchdown of the year.

The extra point is good!
</item 96>

<item 97>
Cowboys vs. 49ers live score, updates, highlights from NFL 'Sunday Night Football' game (Sporting News, 2023-10-08T22:47:03+00:00)
End of first quarter: 49ers 7, Cowboys 0

9:04 p.m. — Confirmed: 49ers ball. San Francisco has an instant opportunity here.

9:01 p.m. FUMBLE — Oh no! Pollard gets the ball knocked out deep in the Cowboys' own territory, and it looks like it was recovered by the 49ers. Officials are
checking to make sure nobody was out of bounds, but that ball was very much out of Pollard's control. The 49ers could be set up in the red zone.

8:59 p.m. — The drive completely falls apart for the 49ers. A delay of game penalty sets San Francisco back, and Christian McCaffrey is dropped for a five-yard
loss on 3rd & 15. Not allowing even a field goal is a huge win for Dallas.

8:56 p.m.
</item 97>

<item 98>
Best sportsbook bonus offers for NFL Monday Night Football Eagles vs. Seahawks: Claim over $5,000 in bonuses from Bet365, BetMGM, BetRivers, Caesars Sportsbook,
DraftKings and FanDuel  (Sporting News, 2023-12-18T21:00:03+00:00)
For a good stretch of the season, the Eagles appeared unstoppable and have been a favorite to return to the Super Bowl. After a surprising loss to the Jets in
Week 6 (their first of the season), Philadelphia went on a five-game winning streak, impressively taking down top teams like the Dolphins, Cowboys and Chiefs.
Most of those games were gritty, close wins, but then the Eagles dropped back-to-back games in blowout fashion to the 49ers and Cowboys, raising alarm bells
amongst the Philly fanbase.

The Eagles are looking to rebound against a Seahawks team that has been reeling even worse, losing four straight. Sitting four games behind the 49ers in the NFC
West, despite a disappointing season, Seattle still has a shot at a wild card playoff spot.
</item 98>

<item 99>
Prisco's Week 12 NFL picks: Jaguars beat Texans in AFC South battle, Saints get key divisional win vs. Falcons (CBSSports.com, 2023-11-23T21:30:38+00:00)
The Raiders have looked so much better the past three games, but this is a tough test against a wounded Super Bowl champ. The Chiefs get back on track here.

Pick: Chiefs 27, Raiders 17

Sunday, 4:25 p.m. ET (CBS, Paramount+)

The Eagles are playing on a short week after a road victory over the Chiefs on Monday night. The Bills are coming off an impressive victory over the Jets. Josh
Allen looked good in that game, and he will in this one. He keeps it close, but the Eagles find a way late.

Pick: Eagles 31, Bills 30

Sunday, 8:20 p.m. ET (NBC, fubo)

The Chargers are having major issues right now and face one of the best teams in the league in this one. The Ravens offense will miss tight end Mark Andrews,
but the Chargers defense is perfect for any team throwing the football.
</item 99>

<item 100>
Lions vs. Packers live score, updates, highlights from NFL 'Thursday Night Football' game (Sporting News, 2023-09-28T23:54:07+00:00)
Good drive by Green Bay, which finally shows signs of life after a putrid first half.

Lions 27, Packers 11

10:05 p.m.: Great throw by Love, who threads the needle to find Romeo Doubs to keep the chains moving on fourth down. Packers on the brink of a much-needed
touchdown.

9:58 p.m.: This Green Bay offense has to come to life sooner rather than later. If not, this scoreboard could get even more crooked.

Good start to the third frame by the Pack, who find themselves marauding down the field with relative ease in the quarter's opening possession. Every journey
began with a single step...

9:20 p.m.: INTERCEPTION! Love's errant pass gets snagged by Jerry Jacobs, who plods 20 yards back to put Detroit on the brink of the promised land. It's been a
putrid start for Love and the Green Bay offense so far.

9:15 p.m.: FIELD GOAL!
</item 100>
=== END ITEM id=a1876c8822e63e22 ===
