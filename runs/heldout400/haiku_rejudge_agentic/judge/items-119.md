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

=== ITEM id=53c1ee76492bad02 ===
Gold sentences and facts:
Sentence 0: 10:59 p.m. — The Cowboys are moving the ball now, when it's a bit too late.
  Fact 0.0: At 10:59 p.m., the Cowboys are moving the ball.
  Fact 0.1: The Cowboys are moving the ball when it is a bit too late.
Sentence 1: Mixon has been forced to step up without Burrow under center, doubling as a runner and receiver for Cincinnati.
  Fact 1.0: Mixon has been forced to step up.
  Fact 1.1: Burrow is not under center.
  Fact 1.2: Mixon is forced to step up without Burrow under center.
  Fact 1.3: Mixon doubles as a runner for Cincinnati.
  Fact 1.4: Mixon doubles as a receiver for Cincinnati.

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
=== END ITEM id=53c1ee76492bad02 ===

=== ITEM id=c64e9e679c8a55ef ===
Gold sentences and facts:
Sentence 0: “It was the millionaires of the Premier League against the youngsters of the Championship,” reflected the former Denmark forward.
  Fact 0.0: The speaker is a former Denmark forward.
  Fact 0.1: The speaker reflected that it was the millionaires of the Premier League against the youngsters of the Championship.
Sentence 1: The Commanders are coming off a dud last week against the Giants, while Dallas is flying high.
  Fact 1.0: The Commanders are coming off a dud last week against the Giants.
  Fact 1.1: Dallas is flying high.

Retrieved items:
<item 1>
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

Graph facts:
- Which M.L.B. Playoff Team Has the Most Daunting Home Field? We Ranked Them. -PUBLISHED_BY-> The New York Times
- How U.S. Women’s Soccer Team Lured Emma Hayes Away From Chelsea -PUBLISHED_BY-> The New York Times
- After a Stunning Snub, College Football Is Facing an Identity Crisis -PUBLISHED_BY-> The New York Times
- He’s Hockey’s Brightest Young Star. This Is What Makes His Shot So Special. -PUBLISHED_BY-> The New York Times
- Alcoholism, Ayahuasca and the Enlightenment of an N.F.L. Player -PUBLISHED_BY-> The New York Times

Entities: The New York Times, Dallas Cowboys, Carolina, Super Bowl, Detroit Lions, N.F.L., Philly, N.F.L. Executives on the Biggest Questions Facing Super Bowl
Contenders, San Francisco
</item 1>

<item 2>
Is the Coverage on Taylor Swift and Travis Kelce’s Romance Nothing but Hype or a Pivotal Shift in NFL’s Century-Old Game Plan Towards Global Recognition?
(Essentially Sports, 2023-10-25T21:30:41+00:00)
Revisiting the NBA caricature back in time, the New York Knicks became the first all-Black team in 1979. But their fans made their disapproval very apparent.
“It wasn’t so much New York attitudes, but fans around the league let you know how they felt about you as an opponent and as a Black man as well,” Knicks star
Earl Monroe told Newsday.

USA Today via Reuters May 15, 2021; San Antonio, Texas, USA; Phoenix Suns forward Torrey Craig (12) and San Antonio Spurs forward Keldon Johnson (3) fight for
loose ball as Suns guards Devin Booker (1) and Chris Paul (3) look on in the third quarterat AT&T Center. Mandatory Credit: Scott Wachter-USA TODAY Sports

We could list out all the scandals, fights, and controversies here, but you get the idea.

Graph facts:
- Brock Purdy -MENTIONS-> NFL
- Raiders vs. Lions live score, updates, highlights from NFL 'Monday Night Football' game -MENTIONS-> NFL
- Fans spot Travis Kelce wearing Taylor Swift-themed friendship bracelet before she attended his game -MENTIONS-> NFL
- Chiefs vs. Packers live score, updates, highlights from NFL 'Sunday Night Football' game -MENTIONS-> NFL
- Christmas Day preview: 49ers, Ravens square off in potential Super Bowl sneak peek -MENTIONS-> NFL

Entities: NFL, Taylor Swift, Travis Kelce, Reuters, Essentially Sports, romance, Is the Coverage on Taylor Swift and Travis Kelce’s Romance Nothing but Hype or
a Pivotal Shift in NFL’s Century-Old Game Plan Towards Global Recognition?, New York Knicks, Earl Monroe, Newsday, Phoenix Suns, San Antonio Spurs, Torrey
Craig, Keldon Johnson, Devin Booker
</item 2>

<item 3>
Is the Coverage on Taylor Swift and Travis Kelce’s Romance Nothing but Hype or a Pivotal Shift in NFL’s Century-Old Game Plan Towards Global Recognition?
(Essentially Sports, 2023-10-25T21:30:41+00:00)
However, the NFL does not just want to stop at $18 billion. It wants to become a $27 billion league in 4 years. But it must address a few skeletons in its
closet before reaching that goal.

If you were around in the 1970s (that would make you at least 50 years old today), you would know that the NBA did not have the mass appeal that it enjoys now.
With average game attendance hovering around 8,000 a game, it was one of the darkest periods of the basketball league. Drug abuse, brawls on the hardwood court
that sometimes spilled over onto coaches and players – it wasn’t a pretty sight. As reported in the Washington Post, even though there are no reliable figures
on cocaine use by players, “estimates by people in the game range from 40 percent to 75 percent.”

Graph facts:
- Brock Purdy -MENTIONS-> NFL
- Raiders vs. Lions live score, updates, highlights from NFL 'Monday Night Football' game -MENTIONS-> NFL
- Fans spot Travis Kelce wearing Taylor Swift-themed friendship bracelet before she attended his game -MENTIONS-> NFL
- Chiefs vs. Packers live score, updates, highlights from NFL 'Sunday Night Football' game -MENTIONS-> NFL
- Christmas Day preview: 49ers, Ravens square off in potential Super Bowl sneak peek -MENTIONS-> NFL

Entities: NFL, Washington Post, Taylor Swift, Travis Kelce, cocaine use, Essentially Sports, drug abuse, Is the Coverage on Taylor Swift and Travis Kelce’s
Romance Nothing but Hype or a Pivotal Shift in NFL’s Century-Old Game Plan Towards Global Recognition?, NBA, basketball
</item 3>

<item 4>
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

Graph facts:
- Christmas Day preview: 49ers, Ravens square off in potential Super Bowl sneak peek -MENTIONS-> Brock Purdy
- NFL standings: Updated AFC, NFC playoff picture for Week 13 of 2023 season -MENTIONS-> Brock Purdy
- Fantasy Football Week 13 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game -MENTIONS-> Brock Purdy
- NFL Fantasy Football Week 5 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game -MENTIONS-> Brock Purdy
- NFL power rankings: Lions, 49ers level up; Texans jump Bengals, Bills; Jets, Patriots sink to bottom for Week 11 -MENTIONS-> Brock Purdy

Entities: Brock Purdy, NFL, Jerry Jones, NFC, Sporting News, Cowboys vs. 49ers game, Nick Bosa, Cowboys vs. 49ers live score, updates, highlights from NFL
'Sunday Night Football' game, Micah Parsons, Cowboys, Christian McCaffrey, 49ers, Fred Warner, George Kittle
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
Which M.L.B. Playoff Team Has the Most Daunting Home Field? We Ranked Them. (The New York Times, 2023-10-08T21:56:00+00:00)
Hard to rank it any higher until we know what we’re dealing with.

Largest crowd ever for a game at Globe Life Field. #StraightUpTX pic.twitter.com/4I7NIB5eF8 — Chris Halicke (@ChrisHalicke) May 1, 2022

The old Rangers ballpark could drum up a Texas-sized atmosphere worthy of Stevie Ray Vaughan and Pat Green, but the new place — at least in the regular season —
feels very much like a multi-events space that happens to be hosting a ballgame. Maybe that changes in the playoffs? The Rangers haven’t been home since Sept.
24 when they won their last five games in Arlington, the last three of which went a long way toward keeping the Mariners out of the postseason.

With a closed roof containing all the noise, Globe Life is sure to get awfully loud, and nothing turns a glorified convention center into a real ballpark quite
like a late-inning rally in October.

No.

Graph facts:
- Which M.L.B. Playoff Team Has the Most Daunting Home Field? We Ranked Them. -MENTIONS-> Detroit
- Which M.L.B. Playoff Team Has the Most Daunting Home Field? We Ranked Them. -MENTIONS-> Houston Astros
- Which M.L.B. Playoff Team Has the Most Daunting Home Field? We Ranked Them. -MENTIONS-> The Athletic
- Which M.L.B. Playoff Team Has the Most Daunting Home Field? We Ranked Them. -MENTIONS-> MLB
- Which M.L.B. Playoff Team Has the Most Daunting Home Field? We Ranked Them. -MENTIONS-> Chad Jennings

Entities: Which M.L.B. Playoff Team Has the Most Daunting Home Field? We Ranked Them., The New York Times, Globe Life Field, Chris Halicke, Stevie Ray Vaughan,
Pat Green, Arlington, Mariners, Rangers, playoffs, Texas, postseason, M.L.B.
</item 6>

<item 7>
Is the Coverage on Taylor Swift and Travis Kelce’s Romance Nothing but Hype or a Pivotal Shift in NFL’s Century-Old Game Plan Towards Global Recognition?
(Essentially Sports, 2023-10-25T21:30:41+00:00)
The hardcore football fans are pretty much done with the NFL harping about Travis Kelce and Taylor Swift. Reality TV star Rachel Lindsay, a longtime NFL fan,
recently slammed the “obnoxious” coverage, highlighting the “exhausting” uptick in Swifties obsessing over football. But that’s just one example of how fans are
finding this fixation a little too disrespectful towards the sport.

America’s Favorite Video Today

From the league’s humble beginnings in 1920 to the $18 billion juggernaut that it is today, the ride has been rocky with more than just a few small bumps along
the road. But with the ambitious target that NFL Commissioner, Roger Goodell, has set for 2027, is there a new strategy in the 104-year-old league’s master plan
to dominate the world sporting scene?

Why the ‘Traylor’ saga will be the key to Roger Goodell’s $27 billion target?

Graph facts:
- Brock Purdy -MENTIONS-> NFL
- Raiders vs. Lions live score, updates, highlights from NFL 'Monday Night Football' game -MENTIONS-> NFL
- Fans spot Travis Kelce wearing Taylor Swift-themed friendship bracelet before she attended his game -MENTIONS-> NFL
- Chiefs vs. Packers live score, updates, highlights from NFL 'Sunday Night Football' game -MENTIONS-> NFL
- Christmas Day preview: 49ers, Ravens square off in potential Super Bowl sneak peek -MENTIONS-> NFL

Entities: NFL, Taylor Swift, Travis Kelce, Essentially Sports, Swifties, Rachel Lindsay, Roger Goodell, Is the Coverage on Taylor Swift and Travis Kelce’s
Romance Nothing but Hype or a Pivotal Shift in NFL’s Century-Old Game Plan Towards Global Recognition?
</item 7>

<item 8>
Is the Coverage on Taylor Swift and Travis Kelce’s Romance Nothing but Hype or a Pivotal Shift in NFL’s Century-Old Game Plan Towards Global Recognition?
(Essentially Sports, 2023-10-25T21:30:41+00:00)
Billionaire entrepreneur Mark Cuban, who paid $1.8 million (approx.) in penalties on behalf of his team, the Dallas Mavericks, fondly remembered the former NBA
commissioner David Stern, on his death in January 2020. Stern came and instituted an anti-drug agreement and even banned All-Star player Michal Richardson for
life for drug abuse. The revitalized image of the NBA helped rake in the moolah from sponsors and made it more appealing to fans. He even went against the tide
to open up the Olympic basketball competition to NBA players resulting in the famous “Dream Team” of 1992. America has never failed to win a medal in the 19
Olympic basketball games that it has played since then.

ADVERTISEMENT Article continues below this ad

The NFL has had its fair share of, um, let’s say, “bumps” down the road as well.

Graph facts:
- Brock Purdy -MENTIONS-> NFL
- Raiders vs. Lions live score, updates, highlights from NFL 'Monday Night Football' game -MENTIONS-> NFL
- Fans spot Travis Kelce wearing Taylor Swift-themed friendship bracelet before she attended his game -MENTIONS-> NFL
- Chiefs vs. Packers live score, updates, highlights from NFL 'Sunday Night Football' game -MENTIONS-> NFL
- Christmas Day preview: 49ers, Ravens square off in potential Super Bowl sneak peek -MENTIONS-> NFL

Entities: NFL, Taylor Swift, Travis Kelce, Dallas Mavericks, Essentially Sports, drug abuse, Is the Coverage on Taylor Swift and Travis Kelce’s Romance Nothing
but Hype or a Pivotal Shift in NFL’s Century-Old Game Plan Towards Global Recognition?, Mark Cuban, David Stern, anti-drug agreement, Michal Richardson, Olympic
basketball competition, NBA players, Dream Team, Olympic basketball games
</item 8>

<item 9>
Is the Coverage on Taylor Swift and Travis Kelce’s Romance Nothing but Hype or a Pivotal Shift in NFL’s Century-Old Game Plan Towards Global Recognition?
(Essentially Sports, 2023-10-25T21:30:41+00:00)
And all the effort is showing.

According to Joe Pompliano, “The NFL will bring in $20 billion in revenue this year, more than the Premier League, La Liga, Bundesliga, Serie A, and Ligue 1
combined.” Coming to the strategy part, a curious mind would perhaps think can they take a leaf out of the NBA’s playbook and reinvigorate its appeal as it aims
to conquer international audiences?

Can the NFL take a leaf out of the NBA’s grisly past and glorious recovery?

The NFL is already the world’s richest league. Calling it an incredible feat would be a gross understatement. Consider this: the English Premier League, which
is broadcast in over 212 territories, has a revenue of just $6 billion, while football games are primarily viewed in America only. However, the NFL does not
just want to stop at $18 billion.
</item 9>

<item 10>
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
</item 10>
=== END ITEM id=c64e9e679c8a55ef ===

=== ITEM id=edd357635e11549e ===
Gold sentences and facts:
Sentence 0: Love Story This finally brings us to the fated sighting of Swift at Arrowhead Stadium on Monday, cheering on No.
  Fact 0.0: This sentence follows the heading 'Love Story'.
  Fact 0.1: This finally brings us to the fated sighting of Swift at Arrowhead Stadium on Monday.
  Fact 0.2: Swift was cheering on No.
Sentence 1: The game is taking place at Arrowhead Stadium, where Swift has performed herself and attended a game before.
  Fact 1.0: The game is taking place at Arrowhead Stadium.
  Fact 1.1: Swift has performed at Arrowhead Stadium.
  Fact 1.2: Swift has attended a game at Arrowhead Stadium before.
Sentence 2: Now, Swift has shared that she has nothing to hide in her new relationship with Kelce.
  Fact 2.0: Swift has shared that she has nothing to hide.
  Fact 2.1: Swift has a new relationship with Kelce.
  Fact 2.2: Swift has said she has nothing to hide in her new relationship with Kelce.

Retrieved items:
<item 1>
When pop culture and sport collide: A timeline of Taylor Swift’s NFL takeover (The Age, 2023-09-26T05:26:05+00:00)
Suddenly, football fans and Swifties were living through this surprising collision. Swift lyrics were infiltrating NFL game commentary, and her fans were
furiously becoming football experts. All that needed to happen was for Swift herself to infiltrate a game, determining once and for all whether there was
anything to these rumours or not. This Is Me Trying

On September 21, Kelce announced on The Pat McAfee Show he had invited Swift to watch one of his games. “I told her, ‘I’ve seen you rock the stage in Arrowhead,
you might have to come see me rock a stage at Arrowhead and we can see which one’s a little more lit,’” he said. “We’ll see what happens in the near future.”
Were they finally about to hard-launch their friendship/potential romance to the world, laying all speculation to rest? Love Story This finally brings us to the
fated sighting of Swift at Arrowhead Stadium on Monday, cheering on No. 87 with gusto.

Graph facts:
- Brock Purdy -MENTIONS-> NFL
- Raiders vs. Lions live score, updates, highlights from NFL 'Monday Night Football' game -MENTIONS-> NFL
- Fans spot Travis Kelce wearing Taylor Swift-themed friendship bracelet before she attended his game -MENTIONS-> NFL
- Chiefs vs. Packers live score, updates, highlights from NFL 'Sunday Night Football' game -MENTIONS-> NFL
- Christmas Day preview: 49ers, Ravens square off in potential Super Bowl sneak peek -MENTIONS-> NFL

Entities: NFL, Arrowhead Stadium, When pop culture and sport collide: A timeline of Taylor Swift’s NFL takeover, Pat McAfee, The Pat McAfee Show, game at
Arrowhead Stadium, Taylor Swift, Travis Kelce, The Age, football
</item 1>

<item 2>
Taylor Swift-Travis Kelce timeline: Everything to know about rumored romance between pop star, Chiefs TE (CBSSports.com, 2023-10-12T23:21:52+00:00)
24: Swift mingles with Donna Kelce



"Talkin' over a football team." - Mean

Swift already seems to be making friends with the family. Swift was sitting next to Donna in the box and the two were seen talking and laughing as the game went
along.

Sept. 24: Postgame interviews turn into Swiftie talk

The postgame press conference seemed more about Swift and Kelce than it did about the actual game. The media wanted any insight it could get into the
relationship and how Swift ended up at Arrowhead.

"He told me at the last minute," Mahomes said when asked if he knew Swift was coming. "There's some things with Trav where he kind of just says it and you don't
know if it's true or not. He says it so calmly. ... Friday he was like, 'Yeah, I think she's coming to the game this weekend,' and then moved about his
business."

Graph facts:
- Taylor Swift -MENTIONS-> Instagram
- Christmas Day preview: 49ers, Ravens square off in potential Super Bowl sneak peek -MENTIONS-> Taylor Swift
- Country music couples Tim McGraw, Faith Hill and Garth Brooks, Trisha Yearwood share tips to lasting marriages -MENTIONS-> Taylor Swift
- Fans spot Travis Kelce wearing Taylor Swift-themed friendship bracelet before she attended his game -MENTIONS-> Taylor Swift
- Taylor Swift -MENTIONS-> jersey sales

Entities: Taylor Swift, Arrowhead, rumored romance, Travis Kelce, Mahomes, postgame press conference, Swiftie talk, Taylor Swift-Travis Kelce timeline:
Everything to know about rumored romance between pop star, Chiefs TE, CBSSports.com, Chiefs, game, Donna Kelce
</item 2>

<item 3>
Taylor Swift-Travis Kelce timeline: Everything to know about rumored romance between pop star, Chiefs TE (CBSSports.com, 2023-10-12T23:21:52+00:00)
Oct. 15: Kelce, Swift make 'SNL' cameos

The two were spotted out in NYC together and then had a surprise appearance on "Saturday Night Live." Kelce had a cameo in a skit about how much coverage Swift
was getting during games, while the singer introduced the musical act of the night, Ice Spice.

Oct. 22: Swift attends fifth Chiefs game

Swift has now attended games broadcast across all five networks (CBS, NBC, Fox, ESPN and Amazon). This week, she is watching the Chiefs face the Chargers and
will be shown on CBS. She is sporting a Chiefs sweatshirt and is seated next to Kansas City quarterback Patrick Mahomes' wife, Brittany.

Graph facts:
- Fans spot Travis Kelce wearing Taylor Swift-themed friendship bracelet before she attended his game -MENTIONS-> Kansas City
- When pop culture and sport collide: A timeline of Taylor Swift’s NFL takeover -MENTIONS-> Kansas City
- Chiefs vs. Packers live score, updates, highlights from NFL 'Sunday Night Football' game -MENTIONS-> Kansas City
- Beyoncé tops box office with $21 million debut for concert film ‘Renaissance’ -MENTIONS-> Kansas City
- Fantasy Football RB Rankings Week 14: Who to start, best sleepers at running back -MENTIONS-> Kansas City

Entities: Kansas City, CBS, Taylor Swift, Fox, Travis Kelce, Taylor Swift-Travis Kelce timeline: Everything to know about rumored romance between pop star,
Chiefs TE, SNL cameos, Chiefs game, Patrick Mahomes, Brittany, CBSSports.com, ESPN, Saturday Night Live, Chiefs, Ice Spice
</item 3>

<item 4>
Taylor Swift reveals secret start to her relationship with Travis Kelce (The Independent - Life and Style, 2023-12-06T14:23:01+00:00)
She explained that she doesn’t necessarily pay attention to the cameras and fans around her in those stadiums because she is watching her boyfriend play.

(Denver Post via Getty Images)

“I’m just there to support Travis,” she said. “I have no awareness of if I’m being shown too much and pissing off a few dads, Brads, and Chads.”

When discussing how many Swifties have now started watching NFL games since she started dating Kelce, she quipped about how much she likes the sport. “Football
is awesome, it turns out,” she joked. “I’ve been missing out my whole life.”

Swift’s effect on the NFL didn’t go unnoticed in October, as one game between the Chiefs and New York Jets - which she attended with some of her famous friends
- became the most-watched Sunday NFL show since the Super Bowl LVII in February 2023.

Graph facts:
- Brock Purdy -MENTIONS-> NFL
- Raiders vs. Lions live score, updates, highlights from NFL 'Monday Night Football' game -MENTIONS-> NFL
- Fans spot Travis Kelce wearing Taylor Swift-themed friendship bracelet before she attended his game -MENTIONS-> NFL
- Chiefs vs. Packers live score, updates, highlights from NFL 'Sunday Night Football' game -MENTIONS-> NFL
- Christmas Day preview: 49ers, Ravens square off in potential Super Bowl sneak peek -MENTIONS-> NFL

Entities: NFL, Taylor Swift, Travis Kelce, Getty Images, New York Jets, Chiefs, Taylor Swift reveals secret start to her relationship with Travis Kelce, game
between the Chiefs and New York Jets, Denver Post, Super Bowl LVII, The Independent, football
</item 4>

<item 5>
Taylor Swift-Travis Kelce timeline: Everything to know about rumored romance between pop star, Chiefs TE (CBSSports.com, 2023-10-12T23:21:52+00:00)
Singer Taylor Swift has a blank space, and she's writing Travis Kelce's name. There were rumors the two were dating for weeks, and those talks only intensified
with swift attending five Chiefs games so far this season, including Sunday's game agains the Chargers.

The Era's Tour singer was seen chatting with the tight end's mom, Donna Kelce, and cheering emphatically when No. 87 scored a touchdown in the first game she
attended last month. Last week, she was in attendance to watch the Chiefs as they defeated the Denver Broncos in Week 6 on "Thursday Night Football."

She is once again at Arrowhead Stadium, this time to watch the Chiefs take on another AFC West opponent, the Los Angeles Chargers.

For those wondering how this apparent relationship came to be, wonder no longer. We have created an extensive timeline of the high-profile couple that is
currently the talk of the league, and even added some Swift lyrics to go along with some of the highlights.
</item 5>

<item 6>
Graph query row:
article: When pop culture and sport collide: A timeline of Taylor Swift’s NFL takeover
publisher: The Age
connected_entity: Arrowhead Stadium
relationship: MENTIONS
passage: When pop culture and sport collide: A timeline of Taylor Swift’s NFL takeover (The Age, 2023-09-26T05:26:05+00:00)
The first link between Swift and Kelce was established in 2016, when Kelce – who helped bring his team to Super Bowl victory in both 2020 and 2023 – played the
classic game “Kiss, Marry, Kill”. Speaking with AfterBuzz TV, Kelce was given the options of Swift, Katy Perry and Ariana Grande. He said he’d ultimately kill
Grande, marry Perry and kiss Swift. That was about six years ago, but it has more recently set tongues wagging as fans wondered how long the plan for them to
meet has been in the works. Cruel Summer Fast-forward seven years, and things started to heat up. In July, speaking on the podcast he shares with his brother,
New Heights, Kelce said he intended to give Swift a friendship bracelet with his number on it during her Eras Tour concert in Kansas City. Unfortunately, the
bracelet never made it to her wrist or his digits to her phone.
</item 6>

<item 7>
Taylor Swift reveals secret start to her relationship with Travis Kelce (The Independent - Life and Style, 2023-12-06T14:23:01+00:00)
“This all started when Travis very adorably put me on blast on his podcast, which I thought was metal as hell,” she explained. “We started hanging out right
after that. So we actually had a significant amount of time that no one knew, which I’m grateful for, because we got to get to know each other.”

Swift also confirmed that when she famously first attended Kelce’s game at Arrowhead Stadium in September, she was already dating the NFL star.

“By the time I went to that first game, we were a couple. I think some people think that they saw our first date at that game? We would never be psychotic
enough to hard launch a first date,” she said.

The “Blank Space” singer also acknowledged that although her love life continues to make headlines with every Kansas City Chiefs game she attends, she’s focused
on being there for her partner. She specified that she isn’t trying to hide her relationship from the public eye.

Graph facts:
- When pop culture and sport collide: A timeline of Taylor Swift’s NFL takeover -MENTIONS-> Arrowhead Stadium
- Travis Kelce says he tried to delete his ‘nonsense’ tweets before they went viral -MENTIONS-> Arrowhead Stadium
- Taylor Swift-Travis Kelce timeline: Everything to know about rumored romance between pop star, Chiefs TE -MENTIONS-> Arrowhead Stadium
- Travis Kelce faces backlash after comments about ‘finding a breeder’ resurface amid Taylor Swift romance -MENTIONS-> Arrowhead Stadium
- Fans spot Travis Kelce wearing Taylor Swift-themed friendship bracelet before she attended his game -MENTIONS-> Arrowhead Stadium

Entities: Arrowhead Stadium, Taylor Swift, Travis Kelce, NFL, first game, Taylor Swift reveals secret start to her relationship with Travis Kelce, The
Independent, Kansas City Chiefs
</item 7>

<item 8>
Fans spot Travis Kelce wearing Taylor Swift-themed friendship bracelet before she attended his game (The Independent - Life and Style,
2023-09-26T22:05:34+00:00)
Stay ahead of the trend in fashion and beyond with our free weekly Lifestyle Edit newsletter Stay ahead of the trend in fashion and beyond with our free weekly
Lifestyle Edit newsletter Please enter a valid email address Please enter a valid email address SIGN UP I would like to be emailed about offers, events and
updates from The Independent. Read our privacy notice Thanks for signing up to the

Lifestyle Edit email {{ #verifyErrors }} {{ message }} {{ /verifyErrors }} {{ ^verifyErrors }} Something went wrong. Please try again later {{ /verifyErrors }}

Fans have spotted Travis Kelce wearing a friendship bracelet with Taylor Swift lyrics on it, just days before she attended his Kansas City Chiefs game.

In a video posted to TikTok on 25 September, a fan shared a photo of Kelce posing with the piece of jewellery on his wrist. The post came after Swift was seen
enthusiastically cheering him on in the box seats at Arrowhead Stadium, fuelling speculation that she and the athlete are dating.

Graph facts:
- Taylor Swift-Travis Kelce timeline: Everything to know about rumored romance between pop star, Chiefs TE -MENTIONS-> Arrowhead Stadium
- Travis Kelce faces backlash after comments about ‘finding a breeder’ resurface amid Taylor Swift romance -MENTIONS-> Arrowhead Stadium
- When pop culture and sport collide: A timeline of Taylor Swift’s NFL takeover -MENTIONS-> Arrowhead Stadium
- Travis Kelce says he tried to delete his ‘nonsense’ tweets before they went viral -MENTIONS-> Arrowhead Stadium
- Taylor Swift reveals secret start to her relationship with Travis Kelce -MENTIONS-> Arrowhead Stadium

Entities: Kansas City Chiefs game, Arrowhead Stadium, Fans spot Travis Kelce wearing Taylor Swift-themed friendship bracelet before she attended his game,
friendship bracelet, Taylor Swift, Travis Kelce, TikTok, The Independent, dating, Kansas City Chiefs
</item 8>

<item 9>
Fans spot Travis Kelce wearing Taylor Swift-themed friendship bracelet before she attended his game (The Independent - Life and Style,
2023-09-26T22:05:34+00:00)
After the game, which saw the Chiefs beat the Chicago Bears 41-10, Kelce was seen leaving the stadium in a blue and white splatter-paint denim suit. Eagle-eyed
fans were quick to believe that Kelce’s outfit paid homage to Swift’s forthcoming album, 1989 (Taylor’s Version), which is due for re-release on 27 October.

After the two escaped the crowd at Arrowhead Stadium in Kelce’s convertible, according to photos and videos posted online, the football star reportedly rented
out Prime Social Rooftop in Kansas City for a post-game celebration with his family, his teammates, and Swift.

“Travis bought out the restaurant for his family and team. Taylor arrived, wearing a denim dress, and was seen snacking, having some cocktails and dancing
alongside Travis,” the witness told Entertainment Tonight. “The two were very affectionate with one another but kept things fun and lighthearted.

Graph facts:
- When pop culture and sport collide: A timeline of Taylor Swift’s NFL takeover -MENTIONS-> Arrowhead Stadium
- Travis Kelce says he tried to delete his ‘nonsense’ tweets before they went viral -MENTIONS-> Arrowhead Stadium
- Taylor Swift-Travis Kelce timeline: Everything to know about rumored romance between pop star, Chiefs TE -MENTIONS-> Arrowhead Stadium
- Travis Kelce faces backlash after comments about ‘finding a breeder’ resurface amid Taylor Swift romance -MENTIONS-> Arrowhead Stadium
- Taylor Swift reveals secret start to her relationship with Travis Kelce -MENTIONS-> Arrowhead Stadium

Entities: Arrowhead Stadium, Fans spot Travis Kelce wearing Taylor Swift-themed friendship bracelet before she attended his game, Kansas City, Chicago Bears,
Chiefs–Chicago Bears game, post-game celebration, Prime Social Rooftop, 1989 (Taylor’s Version), Taylor Swift, Travis Kelce, Chiefs, Entertainment Tonight, The
Independent
</item 9>

<item 10>
Taylor Swift-Travis Kelce timeline: Everything to know about rumored romance between pop star, Chiefs TE (CBSSports.com, 2023-10-12T23:21:52+00:00)
But at the same time, they're overdoing it a little bit."

Jason also chimed in on the matter, saying, "I just think the NFL is not used to celebrities coming to the games. Like basketball has it all figured out.
They're all courtside, they're sitting there. They show 'em once or twice, but then they get back to the game."

Oct. 12: Swift attends Chiefs-Broncos

"Now it's too late for you and your white horse *Bronco, to come around" - White Horse

One day after her movie premiere, Swift is in the house for the Chiefs' "Thursday Night Football" divisional matchup against the Broncos. The game is taking
place at Arrowhead Stadium, where Swift has performed herself and attended a game before. According to Al Michaels, the game will have minimal Swift coverage,
though so far there has already been a mention of the superstar.

Oct.

Graph facts:
- Brock Purdy -MENTIONS-> NFL
- Raiders vs. Lions live score, updates, highlights from NFL 'Monday Night Football' game -MENTIONS-> NFL
- Fans spot Travis Kelce wearing Taylor Swift-themed friendship bracelet before she attended his game -MENTIONS-> NFL
- Chiefs vs. Packers live score, updates, highlights from NFL 'Sunday Night Football' game -MENTIONS-> NFL
- Christmas Day preview: 49ers, Ravens square off in potential Super Bowl sneak peek -MENTIONS-> NFL

Entities: NFL, Arrowhead Stadium, Thursday Night Football, Jason, Taylor Swift, rumored romance, Travis Kelce, Al Michaels, Chiefs-Broncos, White Horse,
celebrities, Taylor Swift-Travis Kelce timeline: Everything to know about rumored romance between pop star, Chiefs TE, CBSSports.com, Chiefs, Broncos
</item 10>
=== END ITEM id=edd357635e11549e ===
