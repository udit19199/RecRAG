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

=== ITEM id=f3c97b11901a591b ===
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
Bills vs. Bengals live score, updates, highlights from NFL 'Sunday Night Football' game (Sporting News, 2023-11-05T23:47:00+00:00)
Bills will have 1:37 with one timeout left in the half. Bengals 21, Bills 7

9:41 p.m.: Burrow dances out of traffic in the pocket and throws a perfect deep shot to Boyd, but he just loses the handle on it at the 2-yard line. Third-and-7
for Cincinnati.

9:40 p.m.: Bills looking to keep some time on the clock. Mixon picks up three yards to the 21, and the Bills use their second timeout right away.

9:37 p.m.: Bengals decide to go for it, and it works out. Chase sneaks past the defender, catches a pass from Burrow and takes it 11 yards to the Bills' 25.
That takes this game to the two-minute warning.

9:36 p.m.
</item 1>

<item 2>
NFL Fantasy Football Week 5 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-04T21:56:25+00:00)
Or Denver. Or Chicago. You get the gist. The Cardinals pass rush shouldn't be too dangerous and Burrow, now over a month into his season, figures to eventually
show some signs of improvement with his health. In no way is he a must start, and there's plenty of good reason to go with Trevor Lawrence, Russell Wilson, C.J.
Stroud and even Joshua Dobbs over him, but there's some mild appeal that should at least keep him in lineups over Dak Prescott and Daniel Jones.

LAST WEEK: Looked like a quarterback playing on one leg. Burrow didn't get much into his throws, didn't challenge downfield much, got the ball out as fast as he
could and was sensitive to pass rush pressure. Even when the game was in garbage time Burrow couldn't get much going.
</item 2>

<item 3>
Bills vs. Bengals live score, updates, highlights from NFL 'Sunday Night Football' game (Sporting News, 2023-11-05T23:47:00+00:00)
He's the lone analyst that picks the Bills to beat the Bengals.

Almost a Bengals sweep... 👀 pic.twitter.com/KDh3awFV2y — Sunday Night Football on NBC (@SNFonNBC) November 6, 2023

7:56 p.m.: Before the game, former punter Kevin Huber and former long-snapper Clark Harris are recognized on the jumbotron. The two are Cincinnati legends, with
Huber having played in the most games as a Bengal (216 games) and Harris ranking fourth (202 games). The two are the rulers of the jungle for the game.

7:45 p.m.: Rasul Douglas will be playing, but The Athletic's Joe Buscaglia is reporting Dane Jackson is getting more of the first-team snaps during warmups.

6:50 p.m.: As expected, Hamlin is officially listed as inactive for the game.

6:47 p.m.: Hamlin running out onto the field before the game.

6:29 p.m.
</item 3>

<item 4>
N.F.L. Executives on the Biggest Questions Facing Super Bowl Contenders (The New York Times, 2023-10-16T23:58:53+00:00)
They are gaining more than 15 yards on just 11 percent of pass plays, down from 21 percent through six games last season and 19 percent for 2022 overall. Jalen
Hurts, A.J. Brown, DeVonta Smith and Dallas Goedert are still on the team, so I’m inclined to trust the production will return. But there’s also a lingering
fear defenses have caught up to aspects of the Eagles’ scheme. Does Philly have a counterpunch?

Advertisement

Exec comment: “People are going to crush Hurts and the offense. They are just churning the clock so much. It just feels like they are imposing their will on
teams through the run game and through their short passing and it just hasn’t taken off from an explosive passing standpoint, so when they play a team that can
move the ball on them through the air, it presents some challenges.”
</item 4>

<item 5>
Fantasy Football Week 13 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-11-30T22:10:58+00:00)
Start 'em 10.0-7.00 6.99 - 5.01 5.00 - 0 Sit 'em

In three games since getting Chase Young, the Niners have allowed 30 points total, yielded just 3.3 yards per rush to RBs, 5.9 yards per attempt to QBs,
pressured quarterbacks at a 43.2% rate (without blitzing much!), accumulated 15 sacks and picked off four interceptions. As many problems as the Eagles present
with their offense, the Niners figure to be just as menacing with their collective defense. That will make things tough for the Eagles, whose defense played
over 90 snaps last Sunday and could be without a couple of starters in the front seven, and whose offense could be forced into an uncomfortable game script if
the Niners build a lead.
</item 5>

<item 6>
Cowboys vs. 49ers live score, updates, highlights from NFL 'Sunday Night Football' game (Sporting News, 2023-10-08T22:47:03+00:00)
11:24 p.m. — These teams are trading possessions as the clock winds down. There was a brief skirmish officials had to break up as frustrations mount for Dallas.

11:14 p.m. — A sack by Kevin Givens ends Rush's first drive of the night and sets up a Bryan Anger punt. Backups are in nearly all around for the 49ers.

11:09 p.m. — Cooper Rush is in the game for the Cowboys...

11:07 p.m. — Mitch Wishnowsky with an excellent punt for the 49ers, who are just trying to get through the rest of the game injury-free.

11:04 p.m. — Sam Darnold has entered the game with 49ers. So, that's where we are with 12 minutes still on the clock.

11:00 p.m. INTERCEPTION — So much for that.
</item 6>

<item 7>
Fantasy Football RB Rankings Week 12: Who to start, best sleepers at running back (Sporting News, 2023-11-21T21:33:00+00:00)
49ers

26 Javonte Williams, Broncos vs. Browns

27 AJ Dillon, Packers @ Lions

28 Jerome Ford, Browns @ Broncos

29 Devin Singletary, Texans vs. Jaguars

30 Kareem Hunt, Browns @ Broncos

31 James Cook, Bills @ Eagles

32 Jaylen Warren, Steelers @ Bengals

33 Najee Harris, Steelers @ Bengals

34 Khalil Herbert, Bears @ Vikings

35 Tyler Allgeier, Falcons vs. Saints

36 Chuba Hubbard, Panthers @ Titans

37 Ezekiel Elliott, Patriots @ Giants

38 Latavius Murray, Bills @ Eagles

39 Roschon Johnson, Bears @ Vikings

40 Joshua Kelley, Chargers vs. Ravens

41 Keaton Mitchell, Ravens @ Chargers

42 Ty Chandler, Vikings vs. Bears

43 Dameon Pierce, Texans vs.
</item 7>

<item 8>
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
</item 8>

<item 9>
Cowboys vs. 49ers live score, updates, highlights from NFL 'Sunday Night Football' game (Sporting News, 2023-10-08T22:47:03+00:00)
The Cowboys' offense looked very much like it did in last season's playoff exit -- uncomfortable and out of sorts. Dallas was sluggish through the first half of
the game, and the night unraveled quickly when Dak Prescott threw three interceptions in the second half. It got out of hand so quickly that Sam Darnold entered
the game for the 49ers with more than 12 minutes remaining.

For Prescott and the Cowboys, questions about whether they can get over the hump against the class of the NFC will persist. While their wins have been dominant,
Dallas already sits two games behind the Eagles in the NFC East.

The 49ers are rolling right along, with all of their major offensive weapons now healthy and the defense living up to its billing as well. San Francisco, now
5-0, doesn't face a team with a winning record until Week 10, though the Browns, Vikings and Bengals are far from automatic wins for Kyle Shanahan's team.
</item 9>

<item 10>
NFL Fantasy Football Week 6 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-12T21:23:16+00:00)
O-LINE: It's possible that injuries that go as far back as to Week 2 along with some declining play from Tyler Biadasz and Terence Steele could be contributing
factors for Prescott's low ADOT. He doesn't want to get hit, so he's getting the ball out as fast as he can. Even against the 49ers he sped up his motion and
overshot those targets. He's just not as comfortable and the offense isn't adjusting well enough.

CHARGERS: Traditionally play a good amount of zone coverage but have changed it up more often this season and used more man-to-man, including versus Miami in
Week 1. But despite some headliners in the front seven and 16 sacks in their past three games, the Chargers rank fourth-worst in pass rush pressure rate at
29.5% this season. They also blitz at about the league average rate.
</item 10>

<item 11>
FanDuel Picks Week 14: Best NFL DFS lineup advice for daily fantasy football tournaments (Sporting News, 2023-12-07T22:17:00+00:00)
WEEK 14 FANTASY ADVICE:

Sleepers | Busts | Projections | Start-sit

FanDuel Picks Week 14: NFL DFS lineup for GPP tournaments

Sunday main slate, $60,000 budget

QB Gardner Minshew, Colts @ Bengals ($6,900). Minshew Mania is eyeing a postseason spot and has the Colts with a 7-5 record heading into the final few weeks of
the season. Now he gets a Bengals team that's limping to the finish thanks to Joe Burrow's thumb injury that cost him the season. The Colts are a motivated
bunch in a tight AFC South, while evidence continues to mount that they knocked the Shane Steichen hire out of the park. Minshew is a capable fantasy starter,
as shown by his 312 yards and two TDs last week, and he's certainly worth another DFS flier in a week that offers very few worthwhile value options.

RB Joe Mixon, Bengals vs.
</item 11>

<item 12>
Bills vs. Bengals live score, updates, highlights from NFL 'Sunday Night Football' game (Sporting News, 2023-11-05T23:47:00+00:00)
Burrow completed 31-of-44 passes for 348 yards with a pair of touchdowns, while Allen went 26-for-38 with 258 passing yards, one passing touchdown and an
interception. He also rushed eight times for 44 yards and another score.

With the win, the Bengals move into the final wild-card spot behind the Steelers and Browns, while the Bills are bumped down to the outside-looking-in of the
AFC playoff race.

Sporting News tracked live scores and highlights from the "Sunday Night Football" matchup. Follow for updates throughout the Bills-Bengals matchup.

Bills vs. Bengals score

1 2 3 4 F Bills 7 0 3 8 18 Bengals 14 7 0 3 24

Bills vs. Bengals live updates, highlights from 'Sunday Night Football'

(All times Eastern)

Fourth quarter

11:21 p.m.
</item 12>

<item 13>
NFL Fantasy Football Week 5 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-04T21:56:25+00:00)
At least 9 of those 15 throws went into the end zone.

PASS RUSH: Bengals opponents have started to sniff out Burrow's issues and have come after him. The Browns and Titans had a pass rush pressure rate between 38&
and 42% in their wins over Cincy with high blitz rates (at least 31% each). Compare that to the Rams and Ravens, who barely brought pressure or blitzes and
either lost or nearly lost to the Bengals.

CARDINALS: Are dead-last in blitz rate this season (14%) and third-to-worst in pass rush pressure rate (28%). They actually had nine sacks after Weeks 1 and 2
but have had three since against the Cowboys and 49ers. Arizona seems to prefer using just four rushers to affect the quarterback, something that should be
helpful to Burrow this week.
</item 13>

<item 14>
Bills vs. Bengals live score, updates, highlights from NFL 'Sunday Night Football' game (Sporting News, 2023-11-05T23:47:00+00:00)
Folks, the Bengals are back.

When Cincinnati began the season 1-3, including a pair of blowout losses in which they scored just three points in each game against the Browns and Titans, it
sure seemed Joe Burrow's calf injury would wind up holding the team back in what appeared to be one of their best chances to finally win that elusive Super
Bowl.

Since that loss to Tennessee, it's been the Bengals of old. And that team showed up again on "Sunday Night Football" against the Bills. Cincinnati jumped out to
an first-half, 21-7 lead against Buffalo, and the defense did the rest, limiting the Bills to just 11 points the rest of the game as the Bengals won their
fourth straight, beating the Bills 24-18.

MORE: Is Damar Hamlin playing vs. Bengals on SNF?

Fans expected this to be an offensive showdown against two premier quarterbacks.
</item 14>

<item 15>
NFL Fantasy Football Week 6 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-12T21:23:16+00:00)
The Cowboys wound up paying the price through the air for playing a ton of zone coverage instead. It was suggested by a friend that they didn't want to get
burned by long runs by Christian McCaffrey. I guess that worked. The combination of the 49ers defense running hot and the Browns offense figuring to get stuffed
should make for at least a game script for Purdy to have decent numbers. I'm not making the mistake again of benching Purdy -- I'd start him over Trevor
Lawrence, Geno Smith and Jared Goff.

LAST WEEK: Took on one of the toughest pass rushes in football and slayed them for four touchdowns on just 24 pass attempts. Of his seven incompletions, one was
dropped, two were batted at the line of scrimmage, another was a throwaway and two more were defensed. Purdy was off-target on 0.0% of his throws.
</item 15>

<item 16>
Bills vs. Bengals live score, updates, highlights from NFL 'Sunday Night Football' game (Sporting News, 2023-11-05T23:47:00+00:00)
That takes this game to the two-minute warning.

9:36 p.m.: The Bengals get within a yard, with Burrow firing a short pass over the middle to Boyd, who picks up nine yards to set up a fourth-and-1, and another
decision for Cincinnati at the Bills' 36.

9:32 p.m.: Back-to-back incompletions from Burrow out to Hudson and Chase to start the drive. Third-and-10 coming up.

9:31 p.m.: Irwin returns the punt nine yards to the Bills' 45 yards. Back-to-back Cincinnati drives starting in Buffalo territory.

9:30 p.m.: Huge sack by the Bengals as BJ Hill gets to Allen at the Bills' 8. Buffalo will have to punt. Huge three-and-out after a disappointing Bengals'
offensive possession.

9:28 p.m.: Cook rushes for six to start the drive, then picks up two yards to the 17 to set up a third-and-2.
</item 16>

<item 17>
NFL Fantasy Football Week 6 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-12T21:23:16+00:00)
D.J. Turner is expected to replace Awuzie.

GENO SMITH: Has a 47.8% completion rate and a lowly 4.1 yards per attempt average against man coverage so far this season. He's also barely thrown deep (4.3% of
his attempts went 20-plus Air Yards). A huge reason for the low numbers is because his pass protection has been dreadful. The anticipated return of left tackle
Charles Cross should help him have more time, which in turn helps his receivers.

OUTSIDE WRs: Have caught just 50% of their targets against the Bengals this season, third-lowest in football. However, when those receivers have caught the ball
they've averaged 17.9 yards per catch, which is the third-most in the league. The Bengals defense has really only let up big games to receivers who catch
touchdowns.
</item 17>

<item 18>
Bills vs. Bengals live score, updates, highlights from NFL 'Sunday Night Football' game (Sporting News, 2023-11-05T23:47:00+00:00)
Third-and-6 coming up. Bills wind up using their first timeout with 5:54 left in the half and the play clock down to three seconds.

10:22 p.m.: Burrow again looks Chase's way, and again comes up empty. He has just three catches for nine yards on seven targets. Robbins' punt spots the Bills
at their own 9.

10:21 p.m.: Burrow goes for the deep shot to Chase, but the throw is just a bit short, and the contact with Dane Jackson breaks up the pass. Mixon then catches
a dump-off pass, but is brought down for a loss of a yard. Third-and-11 coming up.

10:20 p.m.: Higgins catches a short pass, but makes a few players miss, moving all the way up to the Cincinnati 43.

10:19 p.m.
</item 18>

<item 19>
N.F.L. Executives on the Biggest Questions Facing Super Bowl Contenders (The New York Times, 2023-10-16T23:58:53+00:00)
This team is relatively young. They are scrappy. The fact that they have been there before gives them a chance. I think the quarterback is growing.”

• Cincinnati Bengals: Can Joe Burrow hold up for a full season?

What I think: Burrow is looking healthier over the past two weeks. The bye week comes at a great time for Burrow to get even healthier, but check out the
schedule on the other side: San Francisco, Buffalo, Houston, Baltimore, Pittsburgh and Jacksonville. Cincy still has a second game with the Steelers, plus games
against Kansas City and Cleveland. I’m worried Burrow will wear down against these defenses.

Exec comment: “You cannot always expect Joe Burrow to create magic with a bottom-half offensive line. The line is my concern all the way.”

Advertisement

2. Are teams sometimes going too far playing to win? The Eagles come to mind after falling 20-14 to the Jets.
</item 19>

<item 20>
FanDuel Picks Week 14: Best NFL DFS lineup advice for daily fantasy football tournaments (Sporting News, 2023-12-07T22:17:00+00:00)
RB Joe Mixon, Bengals vs. Colts ($7,200). Mixon has been forced to step up without Burrow under center, doubling as a runner and receiver for Cincinnati. Jake
Browning has relied heavily upon the veteran RB, evidenced by Mixon's two touchdowns on Monday night against the Jaguars. Burrow's injury has made Mixon
arguably the Bengals' top fantasy option now. We'll gladly insert the RB into the lineup in a good matchup against a Indianapolis team that's had some issues
defending RBs, notably getting torched by Derrick Henry and Tyjae Spears one week ago.

WEEK 14 DFS:

DK lineup | Best values | Best stacks

RB Breece Hall, Jets vs. Texans ($6,400). It's Week 14 and there is a Jets player in this DFS lineup. How bad can it get? The good news is that it can always
get worse, and Hall's role in this offense makes him a worthy dart throw in this lineup.
</item 20>

<item 21>
Bills vs. Bengals live score, updates, highlights from NFL 'Sunday Night Football' game (Sporting News, 2023-11-05T23:47:00+00:00)
Bengals 24, Bills 10

10:59 p.m.: Another toss play outside to Mixon goes for a loss of three, but Burrow finds Smith, who picks up eight yards to the Bills' 2 before he steps out of
bounds. Third-and-goal at the 2, with the Bengals knocking on the door.

10:57 p.m.: Burrow to Higgins has been the top combination all night, and it is again on the third down play. They connect for 18 yards, with Higgins catching a
slant and races it to the Buffalo 7. First-and-goal with less than 10 minutes to play.

10:56 p.m.: Mixon picks up four yards on a run up the middle, but Burrow throws the ball away on the second-and-6. Bengals use their third and final timeout
with 10:19 left before a third-and-6 play.

10:53 p.m.
</item 21>

<item 22>
N.F.L. Executives on the Biggest Questions Facing Super Bowl Contenders (The New York Times, 2023-10-16T23:58:53+00:00)
We’ve seen the Buffalo Bills incur devastating injuries. Here we’ll address 10 teams with the shortest Super Bowl odds, running through what I think and what
NFL insiders have to say.

The full Pick Six menu this week:

• Unanswered questions for contenders

• When “playing to win” goes too far

• Championship roster, rookie kicker

• Do they make wristbands for coaches?

• Giants, Vikings and winning too soon

• Two-minute drill: Wackiest win in a while

GO DEEPER The Eagles' 'tush push' play is taking over the NFL. But is it working?

1. Where do the top 10 current Super Bowl contenders stand one-third of the way through the season? Let’s hit key questions facing the Chiefs, Eagles, 49ers,
Bills, Bengals, Cowboys, Ravens, Lions, Dolphins and Jaguars.

You can scroll through the latest Super Bowl future odds from BetMGM below.

• San Francisco 49ers: What do they have in Purdy?
</item 22>

<item 23>
Bills vs. Bengals live score, updates, highlights from NFL 'Sunday Night Football' game (Sporting News, 2023-11-05T23:47:00+00:00)
But Burrow to Higgins again, this time for nine yards, to set up a third-and-5 from the Bills' 37.

10:33 p.m.: Mixon takes a checkdown for five yards, then Higgins hauls in an 11-yard pass to move up to the Bills' 42.

10:28 p.m.: Irwin fair catches the punt at the Bengals' 42. Could be a pivotal drive for both teams, with the Bengals leading by two scores with 5:06 left in
the third.

10:27 p.m.: Allen is flushed out of the pocket and tries to pick up yardage, but he's brought down by Trey Hendrickson for only a yard. Bills will have to punt
the ball away.

10:26 p.m.: Allen throws a quick pass out to Cooks for four, then misses a quick strike to Diggs on the left side. Third-and-6 coming up.
</item 23>

<item 24>
NFL Fantasy Football Week 5 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-04T21:56:25+00:00)
PURDY: Has hit at least 20 Fantasy points in 8 of his 9 regular-season starts, and it really should be 9 for 9 because of that first touchdown that was ruled a
lateral back in Week 2. If you include the two playoff games he completed, Purdy has delivered at least 20 in 9 of 11 games. And in 5 of those 9 he hit at least
23 Fantasy points including each of his past two games.

COWBOYS: Dallas' defense lost top cornerback Trevon Diggs and they're still red hot. They lead the league in pass rush pressure rate this season (55.6%) and
they're in the top-10 in blitz rate (31.9%). It's that pressure that's helped the Cowboys rank in the top-five in completion rate allowed (57.4%), passing
touchdowns allowed (two) and missed tackles on pass plays (just five, that's best in football).
</item 24>

<item 25>
Fantasy Football RB Rankings Week 14: Who to start, best sleepers at running back (Sporting News, 2023-12-07T22:07:00+00:00)
The injury highlights the importance of handcuffs, as Zack Moss has put up multiple top-10 performances with and without Taylor this season.

Seattle's Kenneth Walker III (oblique) missed Week 12 and 13, and we're not confident in his ability to suit up against the hard-hitting 49ers coming off a big
win in Philly. Backup Zach Charbonnet exited early because of a knee injury against the Cowboys, putting his status in doubt, too. If both are out, DeeJay
Dallas would start and lead the backfield in touches.

Green Bay's Aaron Jones has a sprained MCL, and sources have revealed that the Packers continue to consider him week-to-week. He only logged four carries before
sustaining the injury in Week 11, then he sat for the Packers' Thanksgiving Day tilt with the Lions and their Sunday Night Football win over Kansas City. Why
rush him back if Green Bay is already winning games?
</item 25>

<item 26>
Bills vs. Bengals live score, updates, highlights from NFL 'Sunday Night Football' game (Sporting News, 2023-11-05T23:47:00+00:00)
: Much ado was made about the tight end position last week when Irv Smith Jr. fumbled in the red zone. Today, the first tight end to get involved in the game is
Tanner Hudson, who catches a six-yard pass and gets the first down.

8:26 p.m.: Williams gets the first two carries of the game by a Bengal, and he rushes right up the middle for seven yards on first down and then a pickup of one
trying to bounce outside.

8:24 p.m.: Burrow starts the game with a quick checkdown pass to Joe Mixon, who takes it up the field eight yards. He then catches another short pass and takes
it up 15 yards to the Bengals' 47.

8:23 p.m.: Trayveon Williams returns the ball out to the 24, where the Bengals will start on offense.

8:20 p.m.: The Bengals won the toss, and decided to take the ball on offense to start the game.

8:13 p.m.
</item 26>

<item 27>
NFL Fantasy Football Week 6 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-12T21:23:16+00:00)
Plus the Browns just saw the Cowboys play 90% of their snaps in zone coverage and Purdy cooked them like BBQ chicken.

PURDY: Has been sensational against both man and zone coverages this season. His completion rate and yards per attempt are noticeably higher against zone, but
five of his seven passing touchdowns have come versus man coverage. However, Purdy's been at his worst when pressured (like most quarterbacks), completing 50%
of his throws for 6.7 yards per attempt with a gaudy 15.9% off-target rate. These stats are actually better than they were last week.

BROWNS: Rank third in pass rush pressure rate at 41.5% and blitz at an above-average 29.3% rate. Granted, the Cowboys ranked higher than that last week and got
beat anyway.
</item 27>

<item 28>
NFL Fantasy Football Week 6 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-12T21:23:16+00:00)
Then Prescott HAD to throw deep to try and climb back into the game and got sloppy on two of his interceptions (a third bounced off of Michael Gallup's hands).
He didn't even play the last 12 minutes of the game and the Cowboys got blown out.

THIS SEASON: Prescott's completion rate is fine (69.4%) and his off-target rate is low (8.1%). So what's the problem? Prescott's ADOT is 6.5, ranking 28th among
33 qualifying quarterbacks. He's actually second-to-last in pass attempts of 10-plus Air Yards; only Daniel Jones has thrown fewer. And the scary part is that
his ADOT was over a yard lower (5.4) going into last week's game, so the Cowboys tried to let him air it out and the plan completely backfired.
</item 28>

<item 29>
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
</item 29>

<item 30>
Bills vs. Bengals live score, updates, highlights from NFL 'Sunday Night Football' game (Sporting News, 2023-11-05T23:47:00+00:00)
8:49 p.m.: Bengals just miss out on another touchdown. Burrow escapes the pocket, then gets the ball out to Higgins who dives for the pylon. However, Higgins is
ruled out at the 2.

8:48 p.m.: Make it back-to-back firsts for the Bengals, as Hudson gets his second catch of the day, this one for 13 yards to the Bills' 14.

8:47 p.m.: Bengals stay perfect on third downs, with Tyler Boyd catching a pass in front of the sideline for 15 yards up to the Bills' 27.

8:46 p.m.: Bengals have their longest third-down attempt of the day. Mixon rushes for no gain, and Burrow can't connect with Smith after he gets flushed out of
the pocket. Third-and-10 coming up.

8:44 p.m.: Andrei Iosivas is getting involved in the passing game after back-to-back weeks with a touchdown.
</item 30>

<item 31>
NFL Fantasy Football Week 5 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-04T21:56:25+00:00)
LAST WEEK: Was the benefactor of Dak Prescott taking what the Patriots defense gave him, mostly utilizing Ferguson as a short-area target to fire the ball to
when the pressure was quick off the snap. But there were a couple of throws where Prescott read the defense well and found Ferguson for pretty good gains. All
told Ferguson led the Cowboys with seven targets and seven catches for 77 yards in three quarters of work.

THIS WEEK: The 49ers figure to unleash its pass rush on Prescott with zone coverage behind it, which only makes sense given how Prescott has handled the pass
rush pressure this year. It should mean more short throws for Ferguson.

49ERS: Are tops in football in catch rate allowed to tight ends (48.6%) and YAC/reception allowed (1.29 yards). They're also top-five in yards per catch (7.1).
</item 31>

<item 32>
NFL Fantasy Football Week 6 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-12T21:23:16+00:00)
The only rush defense metrics they're not top-five in are Defensive Rush EPA (just below league average), yards before contact per rush (0.88 yards, they're
sixth), yards after contact per rush (2.80, just above league average). They even lead the NFL in fewest missed tackles. Tony Pollard had an 8-29-0 rushing stat
line against them last week.

WHAT ABOUT THROUGH THE AIR? Even though the 49ers see the fifth-most RB targets per game, they're better than league average in catch rate allowed (75%) and
top-10 in both yards per catch (5.7) and YAC/reception (6.10) allowed to running backs. Pollard caught 4 of 5 passes for 35 yards last week.

P.J. WALKER: Has thrown 17.2% of his career pass attempts to running backs.
</item 32>

<item 33>
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
</item 33>

<item 34>
Bills vs. Bengals live score, updates, highlights from NFL 'Sunday Night Football' game (Sporting News, 2023-11-05T23:47:00+00:00)
: Mixon gets five yards on a run outside to the right, he's pushed out of bounds, but that will do it. The Bengals can kneel the game out and will win 24-18.

11:19 p.m.: Mixon rushes for four yards to bring up a third-and-4 at the Bills' 37, and bring up the two-minute warning. A first down will win the game for the
Bengals, since Buffalo cannot stop the clock. A stop on third will bring up an interesting decision for Zac Taylor and Cincinnati.

11:17 p.m.: Mixon rushes for two yards after the big gain, and the Bills use their final timeout with 2:37 left.

11:17 p.m.: Feels like a time to run the ball, right? Well, not to the Bengals, who trust Burrow to air out a deep pass to Boyd for 32 yards to the Bills 43.
</item 34>

<item 35>
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
</item 35>

<item 36>
Bills vs. Bengals live score, updates, highlights from NFL 'Sunday Night Football' game (Sporting News, 2023-11-05T23:47:00+00:00)
: Burrow clearly wanted to go for it, but the Bengals decide to punt. Brad Robbins boots it up, and Hardy fair catches the ball at the Bills' 12.

9:10 p.m.: Brutal false start penalty by Alex Cappa backs up the Bengals, and while Burrow connects with Drew Sample for six, the Bengals are still two yards
shy of the marker. Fourth-and-2 coming up, and a decision for Cincinnati at the Bills' 48.

9:08 p.m.: Bengals keep leaning on the running game with Mixon picking up two, then Williams rushing for five to bring up a third-and-3 at the Bills' 49.

9:07 p.m.: Chase is tackled for a loss of two, but Mixon makes up for it right away, following Orlando Brown up the middle for a pickup of 12 yards for a first
down up to the Bengals' 44.

9:05 p.m.: The Bengals' offense picks up right where it left off.
</item 36>

<item 37>
Cowboys vs. 49ers live score, updates, highlights from NFL 'Sunday Night Football' game (Sporting News, 2023-10-08T22:47:03+00:00)
For a Cowboys team that has embarrassed a few opponents this year, they might be on the bad end of an ugly game here.

49ers 28, Cowboys 10

10:27 p.m. TOUCHDOWN — Not one, not two, THREE touchdowns for George Kittle. The 49ers have everything working offensively tonight, and any Kittle fantasy
owners are thrilled.

10:25 p.m. — Aiyuk's loss is Deebo Samuel's gain. Samuel goes 43 yards on the very next play, which tells you how automatic the 49ers' offense is right now. San
Francisco is in the red zone looking to make this a three-score game.

10:23 p.m. — An offensive holding call wipes out a 40-yard gain from Aiyuk. Wow. That's a momentum swing. Spencer Burford is the culprit.

10:21 p.m. — The Purdy-to-Aiyuk connection continues to get stronger.
</item 37>

<item 38>
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
</item 38>

<item 39>
NFL Fantasy Football Week 5 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-04T21:56:25+00:00)
Achane is clearly far ahead of Mostert.

LAST WEEK: Before he fumbled the ball away twice, Mostert was splitting reps with De'Von Achane in a way that was unfavorable for him. Before the Dolphins got
inside the 10-yard line Mostert had a 16-to-9 edge, then Achane played every single snap inside the 10. Then after Mostert's second fumble of the half, Mostert
played just nine of the remaining 35 snaps including nothing near the end zone. Coach Mike McDaniel preached the importance of his offense not having
"self-inflicted wounds" immediately after the game.

LAST TWO WEEKS: Achane played all but two snaps inside the 10-yard line for Miami, so that seems like a big part of his role that will stick. As for third
downs, Mostert has a one-snap edge over Achane at 6 to 5, which means they're splitting those valuable touches.
</item 39>

<item 40>
Bills vs. Bengals live score, updates, highlights from NFL 'Sunday Night Football' game (Sporting News, 2023-11-05T23:47:00+00:00)
10:53 p.m.: Burrow finally connects on the big shot to Chase. His star wide receiver finds some open space, and he catches a 32-yard pass up to the Bills' 29.

10:52 p.m.: Why run it on third-and-1, when Burrow can call a play action then connect with Hudson for 17 yards up to the Cincinnati 39.

Tanner Hudson had 4 total catches this season entering today.



He has 4 in this game alone 😱



📺: #BUFvsCIN on NBC

📱: Stream on #NFLPlus https://t.co/T6GLEfxpQv pic.twitter.com/HnhhsUL8In — NFL (@NFL) November 6, 2023

10:50 p.m.: Irv Smith drops a pass on first down, but who else but Higgins makes a leaping grab for nine yards over the first down marker.
</item 40>

<item 41>
Cowboys vs. 49ers live score, updates, highlights from NFL 'Sunday Night Football' game (Sporting News, 2023-10-08T22:47:03+00:00)
Not allowing even a field goal is a huge win for Dallas.

8:56 p.m. — The 49ers are excelling on third down early in this game, and Purdy flips it to Kyle Juszczyk to keep the Niners' drive alive on 3rd & 1.

8:54 p.m. — Now Kearse lines up offsides... That extends the drive for the 49ers, who aren't moving as freely as they were on that opening drive.

8:50 p.m. — Jayron Kearse comes through with the Cowboys' first sack of the night, but Deebo Samuel makes up 11 of the 12 yards San Francisco needs on the next
play.

8:46 p.m. — There's the first catch of the night for Lamb, but he doesn't get the first down and Dak Prescott is sacked by Nick Bosa to quickly end the drive.
This 49ers defense has picked up right where it left off against the Cowboys in January.
</item 41>

<item 42>
Fantasy Injury Updates Week 10: News on Justin Jefferson, Ja'Marr Chase, T.J. Hockenson & Curtis Samuel (Sporting News, 2023-11-11T21:17:00+00:00)
Chase put a scare into the fantasy community on Wednesday when he was listed as a DNP, but after logging back-to-back limited practices, there is a lot more
optimism heading into Sunday. With Tee Higgins (hamstring) already ruled out, Cincinnati will be hoping Chase can feel healthy enough to play when they host the
Texans at 1 p.m. ET on Sunday.

The Bengals, despite getting a win over the Bills last Sunday, are a beaten-down bunch. Their defense will also be without a prominent edge rusher, Sam Hubbard,
complicating matters for a team hoping to stack wins as they compete for an AFC North crown. On the bright side, Joe Burrow appears to be finally healthy and
should have no problems targeting Tyler Boyd, Trenton Irwin, and Andrei Iosivas in the passing game. If Chase does end up suiting up, fantasy owners should
enjoy an even larger workload than the receiver already had.
</item 42>

<item 43>
Cowboys vs. 49ers live score, updates, highlights from NFL 'Sunday Night Football' game (Sporting News, 2023-10-08T22:47:03+00:00)
49ers 35, Cowboys 10

10:47 p.m. TOUCHDOWN — Purdy hits Kyle Juszczyk for the 49ers' fifth touchdown of the night. That's four touchdowns for Purdy alone, and San Francisco is
pouring it all on the Cowboys.

End of third quarter: 49ers 28, Cowboys 10

10:43 p.m. — San Francisco is moving the ball inch-by-inch now, and the 'Niners are a yard away from making this a 25-point game.

10:38 p.m. — The 49ers are on the verge of entering the red zone after a 19-yard catch by Jauan Jennings. Purdy is up to 242 yards to go along with his three
touchdowns.

10:32 p.m. INTERCEPTION — Prescott heaves it downfield under pressure and it's intercepted by Tashaun Gipson.
</item 43>

<item 44>
Cowboys vs. 49ers live score, updates, highlights from NFL 'Sunday Night Football' game (Sporting News, 2023-10-08T22:47:03+00:00)
This 49ers defense has picked up right where it left off against the Cowboys in January.

8:44 p.m. — Dallas' second drive starts out with Tony Pollard getting dropped for a loss.

8:40 p.m. — Big-time stop by the Cowboys' defense as Johnathan Hankins deflects Purdy's third down pass with his helmet and forces a punt.

8:36 p.m. — The 49ers' defense was all over Dallas on that drive. Three plays, two incompletions, and now the Cowboys will punt it right back.

49ers 7, Cowboys 0

8:30 p.m. TOUCHDOWN — Purdy finds a wide open Kittle in the end zone for the score! That was a well-scripted opening drive by Kyle Shanahan, and San Francisco
has an early 7-0 lead.

8:27 p.m.
</item 44>

<item 45>
Bills vs. Bengals live score, updates, highlights from NFL 'Sunday Night Football' game (Sporting News, 2023-11-05T23:47:00+00:00)
That will move the Bills up 17 yards to the Bills' 44.

First quarter

8:56 p.m.: James Cook gets his first rush of the game, picking up two yards. That will bring this quarter to an end.

8:55 p.m.: Bills will take the ball at the 25 on the fair catch at the 2. Just 0:35 left in a high-scoring first quarter.

8:52 p.m. Bengals scoring play: Hand-off to Mixon, and he bulldozes his way up the middle for the 2-yard touchdown to give Cincinnati the lead again. Bengals
14, Bills 7

8:51 p.m.: Cincinnati tries twice to get it to Chase, first with Chase being tackled at the 2 on a wide receiver screen and then a leaping grab for Chase in the
end zone, where he can't quite hold on after the catch.

8:49 p.m.
</item 45>

<item 46>
NFL Fantasy Football Week 6 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-12T21:23:16+00:00)
SINCE CHUBB GOT HURT: Ford has averaged 4.3 yards per rush and 7.7 yards per catch with three total touchdowns (including the catch he had on the play after the
Chubb injury). His 69-yard cross-field run at Pittsburgh has certainly helped his numbers out; without it he has zero touches of 20-plus yards and his only run
for more than 10 yards came with a holding penalty.

SNAPS: Ford has yet to dominate snaps at running back for the Browns, playing right around 60% in Weeks 3 and 4. In fact, Ford does not have more than 14
touches in a single game since taking over for Chubb, but it's been just two games.

HERE COME THE 49ERS: On the year they've held enemy running backs to 3.7 yards per carry with an explosive run rate of 1.7% (which somehow is only second-best
in football).
</item 46>

<item 47>
Bills vs. Bengals live score, updates, highlights from NFL 'Sunday Night Football' game (Sporting News, 2023-11-05T23:47:00+00:00)
9:25 p.m.: Brown is called for a false start on the field goal try, which was good from 55. The Bengals will have to punt. Bills will start from their own 9.

9:24 p.m.: Burrow is sacked by Leonard Floyd as he tried to escape at the Bills 37. Offensive holding is declined, and the Bengals will try for a field goal to
make it a two-score game.

9:23 p.m.: Bengals get five yards back on an offsides by the Bills, which they'll accept despite an 8-yard grab by the Irwin. Third-and-10 coming up.

9:19 p.m.: Cincinnati keeps heading in the wrong direction. Burrow scrambles for a first down, but it's coming back on offensive holding. Now a third-and-15
from the Bills' 37.

9:18 p.m.: Bengals try for a quarterback sneak, but he's stopped just short.
</item 47>

<item 48>
NFL Fantasy Football Week 5 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-04T21:56:25+00:00)
Even when the game was in garbage time Burrow couldn't get much going.

AIR YARDS LAST WEEK: Burrow was 1 for 4 on throws of 10-plus Air Yards and 0 for 2 on throws of 15-plus Air Yards.

AIR YARDS THIS SEASON: Burrow has been 11 for 38 on throws of 10-plus Air Yards and 2 for 22 on throws of 15-plus Air Yards.

TOUCHDOWNS: Burrow has thrown just two scores this season. Both came against Baltimore in Week 2, both were throws inside the five-yard line, both were to Tee
Higgins. That's it. Even his red-zone stats are unfathomable -- he's completed 7 of 15 passes in the red zone. At least 9 of those 15 throws went into the end
zone.
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
</item 50>

<item 51>
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
</item 51>

<item 52>
Cowboys vs. 49ers live score, updates, highlights from NFL 'Sunday Night Football' game (Sporting News, 2023-10-08T22:47:03+00:00)
They'll start the second half with the ball.

9:56 p.m. — 49ers edge rusher Drake Jackson is walking off after requiring some medical attention on the field.

9:52 p.m. — The 49ers tried to put something together, but a third down pass to Aiyuk was too low. Dallas has a minute to try and get something before halftime.

9:45 p.m. — The Cowboys looked better on third down on their last drive, but Prescott's third down pass to Ferguson is short of the line to gain and ends this
drive quickly. The 49ers will have a couple minutes to blow this game open before halftime.

49ers 21, Cowboys 7

9:38 p.m. TOUCHDOWN — McCaffrey pushes through for the touchdown. That's his 14th consecutive game finding the end zone, and it puts the 'Niners back up two
scores.

9:37 p.m.
</item 52>

<item 53>
Bills vs. Bengals live score, updates, highlights from NFL 'Sunday Night Football' game (Sporting News, 2023-11-05T23:47:00+00:00)
Burrow and Boyd get the @Bengals past midfield on the first play of the drive



📺: #BUFvsCIN on NBC

📱: Stream on #NFLPlus https://t.co/T6GLEfxpQv pic.twitter.com/SOTyFnyzp7 — NFL (@NFL) November 6, 2023

11:16 p.m.: Bengals start the drive at the 25.

11:13 p.m. Bills scoring play: There's the big strike Bills fans have been waiting for. Allen fires a deep shot to Diggs, who beat rookie corner DJ Turner for
the 17-yard touchdown. The Bills convert on the two-point conversion, throwing again to Diggs to make it a six-point game. Bengals 24, Bills 18

11:12 p.m.: Bills move quickly into the red zone, as Allen connects with Shakir for three yards then a short pass to Murray for six yards.
</item 53>

<item 54>
NFL Fantasy Football Week 5 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-04T21:56:25+00:00)
LAST WEEK: The Texans didn't need him! Houston built a nice lead thanks to its defense and held on without having to throw the ball a ton.

FILM: The Steelers really emphasized not getting beat deep by Dell, often using a safety over the top to his side of the field to deter C.J. Stroud from even
thinking about throwing at him. It worked out for the Steelers because Dell had zero deep targets, and it worked for the Texans because Nico Collins beat soft
coverage a bunch including on a 24-yard deep ball (Dell drew triple coverage!) and on his long touchdown score (Dell got a safety to cheat over to his side).

FALCONS: Should be able to deploy the same type of defensive coverage with a safety cheating over to Dell's side of the field. They could also stick outside
cornerback A.J. Terrell on Collins and make things a little harder on the Texans.
</item 54>

<item 55>
Bills vs. Bengals live score, updates, highlights from NFL 'Sunday Night Football' game (Sporting News, 2023-11-05T23:47:00+00:00)
Buffalo picked up a field goal on the first drive of the second half and held the Bengals to just three points in the second half, but it wasn't until 3:32 left
that Allen finally got the Bills back into the end zone, with a 17-yard strike to Diggs and a subsequent 2-point try to make it a six-point game.

MORE: Watch 'Sunday Night Football' live with Fubo (free trial)

The Bengals had the ball back with 3:32 left, but put the trust in No. 9, having him start the put-away drive with a 32-yard deep shot to Tyler Boyd to put the
Bengals quickly in Buffalo territory. Joe Mixon ran three times to pick up the final first down needed to allow Burrow to kneel out the contest.

In a highly anticipated rematch between Allen and Burrow, it was the Bengals' signal-caller that once again came out the better in both the win column and the
statsheet.
</item 55>

<item 56>
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
</item 56>

<item 57>
NFL Fantasy Football Week 6 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-12T21:23:16+00:00)
You guessed it -- McLaughlin (6.1 yards per carry).

CHIEFS: Tend to build leads on their opponents. They also have given up just four rushing plays of 12-plus yards this season, and that includes a fake punt run
by the Vikings last week. Kansas City has given up one rushing TD to a running back this season. Impressive numbers considering they're giving up 4.5 yards per
carry and rank in the bottom-10 in yards before contact and yards after contact. The secret? Teams don't run the ball that much against them (16.4 attempts per
game, fourth-fewest) because ... they tend to build leads on their opponents.

CHIEFS: David Montgomery and Alexander Mattison are the only two running backs to tally at least 10 Fantasy points (PPR or non-PPR) against Kansas City this
year. Both scored.
</item 57>

<item 58>
Bills vs. Bengals live score, updates, highlights from NFL 'Sunday Night Football' game (Sporting News, 2023-11-05T23:47:00+00:00)
: Bengals try for a quarterback sneak, but he's stopped just short. Cincinnati then takes a delay of game, and Cincinnati is now backed up to a third-and-5.

9:16 p.m.: Hudson remains active in the passing game, catching a nine-yard pass from Burrow to start the drive. The Bills use their first timeout with 8:17 left
as the Bengals prepare for a second-and-1 from the Bills' 23.

9:15 p.m.: Allen makes his first mistake of the game. Taylor-Britt leaps in front of Gabe Davis and picks up his third interception of the season (Allen's ninth
of the year). That will give the Bengals excellent field position at the Bills' 32.

9:14 p.m.: Allen starts the drive trying again to connect with Harty on a deep shot, but it's overthrown.

9:11 p.m.: Burrow clearly wanted to go for it, but the Bengals decide to punt.
</item 58>

<item 59>
NFL Fantasy Football Week 5 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-04T21:56:25+00:00)
Overall the targets have favored Mostert only because he had more of them in Week 3 against Denver (7 to 4).

GIANTS: Are about to get run over. Their 4.3-yard rushing average they've allowed is a shade worse than league average but they rate poorly in rushing
touchdowns allowed (five), rushes of 10-plus yards allowed (14) and missed tackles on runs (24, fourth-worst in football).

DOLPHINS: Have split the workload pretty evenly despite the snap shares morphing in Achane's favor. Last week Achane had 11 touches to Mostert's 10. The week
before Achane had 22 to Mostert's 20. The week before that ... Achane wasn't a thing and Mostert had 19 touches to Salvon Ahmed's six (Achane had none).
</item 59>

<item 60>
NFL power rankings: Lions, 49ers level up; Texans jump Bengals, Bills; Jets, Patriots sink to bottom for Week 11 (Sporting News, 2023-11-13T21:54:00+00:00)
Their defense should be getting a little better to support their often explosive offense. They should feel good about a Week 11 rebound vs. the Raiders.

7. Pittsburgh Steelers 6-3 (9)

The Steelers are playing the way Mike Tomlin wants, with the running game rolling to big totals with Jaylen Warren and Najee Harris and the defense doing what
it needs, more so vs. pass than run, in key moments. You can't argue with the overachieving results.

8. Cleveland Browns 6-3 (10)

The Browns proved they can win games with defense in another way, with big plays. They also have more offensive pop when Deshaun Watson plays well and the
running game doesn't slump. Their playoff hopes look much better after their stunning comeback vs. the Ravens.

9.
</item 60>

<item 61>
NFL Fantasy Football Week 6 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-12T21:23:16+00:00)
BILLS: Got their first test against tight ends last week in Evan Engram and swarmed him -- he caught 4 of 8 targets for 28 yards. He is used similarly to
Waller, which makes Buffalo's game plan for slowing Waller down easy to carry over this week. Their pass rush should also impact the Giants passing game and may
force Waller to run shorter routes so he gets open quicker.

Start 'em 10.0-7.00 6.99 - 5.01 5.00 - 0 Sit 'em

Sit Him (Lineup Decisions) Projections powered by Sportsline Dak Prescott QB DAL Dallas • #4

Age: 30 • Experience: 8 yrs.
</item 61>

<item 62>
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
</item 62>

<item 63>
Fantasy Football Week 13 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-11-30T22:10:58+00:00)
While it's true that the Broncos have given up just one rushing touchdown in their past six games, it's also true they've let up 5.4 yards per carry in those
same six games. Bank on the Texans giving Singletary more work on the ground to pace their offense. I'd start him over Jerome Ford, Javonte Williams and Joe
Mixon.

Start 'em 10.0-7.00 6.99 - 5.01 5.00 - 0 Sit 'em

The Patriots have seemingly backed off of being blitz-heavy over their past three games, perhaps because they realized they can get similar payoffs without it
(only three of their past nine sacks have come on blitzes). They also have an inflated sack total because they recorded six against the Giants -- they have no
more than three in any other game this year including zero against the Colts in Week 10.
</item 63>

<item 64>
Bills vs. Bengals live score, updates, highlights from NFL 'Sunday Night Football' game (Sporting News, 2023-11-05T23:47:00+00:00)
Latavius Murray gets his first carry and picks up two and the first down to the Bengals' 48.

11:04 p.m.: Allen goes right back to his trusted tight end, even after the fumble, with Kincaid catching a nine-yard pass to start the drive. Allen then runs
right, dodging a tackle from Sam Hubbard, to pick up seven and the first at the Bills' 41.

11:03 p.m.: Bills will take the touchback, and start the drive from the 25.

11 p.m. Bengals scoring play: Burrow misses Mixon trying to get him on a swing pass. Cincinnati opts for the field goal at the 2-yard line to make it a
two-touchdown advantage with 8:08 left to play. Bengals 24, Bills 10

10:59 p.m.
</item 64>

<item 65>
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
</item 65>

<item 66>
Fantasy Football Week 13 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-11-30T22:10:58+00:00)
If that doesn't make you feel good about Conner, then you'll especially dislike that he's averaged 0.42 yards before contact on his runs since Murray's return,
good for 44th out of 47 running backs (Breece Hall and Derrick Henry had it worse). Pittsburgh's unit held the Bengals and Browns in consecutive weeks to 0.5
yards before contact or less. If you start Conner, don't expect an efficient day and pray to Ye Olde Fantasy Gods for a garbage-time touchdown. I'd rather trust
Jaylen Warren, Chuba Hubbard and A.J. Dillon.

Start 'em 10.0-7.00 6.99 - 5.01 5.00 - 0 Sit 'em

Expect plenty of rushing.
</item 66>

<item 67>
Cowboys vs. 49ers live score, updates, highlights from NFL 'Sunday Night Football' game (Sporting News, 2023-10-08T22:47:03+00:00)
End of first quarter: 49ers 7, Cowboys 0

9:04 p.m. — Confirmed: 49ers ball. San Francisco has an instant opportunity here.

9:01 p.m. FUMBLE — Oh no! Pollard gets the ball knocked out deep in the Cowboys' own territory, and it looks like it was recovered by the 49ers. Officials are
checking to make sure nobody was out of bounds, but that ball was very much out of Pollard's control. The 49ers could be set up in the red zone.

8:59 p.m. — The drive completely falls apart for the 49ers. A delay of game penalty sets San Francisco back, and Christian McCaffrey is dropped for a five-yard
loss on 3rd & 15. Not allowing even a field goal is a huge win for Dallas.

8:56 p.m.
</item 67>

<item 68>
Cowboys vs. 49ers live score, updates, highlights from NFL 'Sunday Night Football' game (Sporting News, 2023-10-08T22:47:03+00:00)
9:37 p.m. — Deebo Samuel takes some big hits but gets the 49ers down to the one-yard line. San Francisco is knocking on the door again.

9:35 p.m. — The Cowboys had done a pretty nice job of containing McCaffrey to this point, but the former Pro-Bowler takes a screen pass for 19 yards to push the
Niners to the edge of the red zone.

9:31 p.m. — Aiyuk with a quick 15-yard catch to start the 49ers' drive. He's up to 35 yards on three catches.

49ers 14, Cowboys 7

9:27 p.m. TOUCHDOWN — KaVontae Turpin flashes his speed, gets separation, and hauls in Prescott's pass in the end zone! A perfectly drawn-up 26-yard play puts
Dallas right back in this game.
</item 68>

<item 69>
NFL Fantasy Football Week 6 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-12T21:23:16+00:00)
Consider Lockett more of a top-30 receiver who you should sit for Zay Flowers and Gabe Davis but start over Chris Godwin, Nico Collins, Garrett Wilson and Drake
London.

BENGALS: Consistently tailor their defensive coverage to the weaknesses of the offenses they face. So when a quarterback struggles against zone coverage,
they'll play more zone. This worked in their favor in all but one game this year (Ravens, Week 2). Cincinnati also has turned on the juice for their pass rush,
getting a pressure rate of over 40% in each of their past three games, all versus teams with suspect offensive lines.

BENGALS: Are expected to be without top cornerback Chidobe Awuzie, and they also are rotating one of their safety spots between Nick Scott and Jordan Battle.
D.J. Turner is expected to replace Awuzie.
</item 69>

<item 70>
Cowboys vs. 49ers live score, updates, highlights from NFL 'Sunday Night Football' game (Sporting News, 2023-10-08T22:47:03+00:00)
— The Cowboys have a first down! And it only took 17+ minutes. Dallas is still operating in its own territory, but it feels like points are needed on this drive
to keep the game within reach.

49ers 14, Cowboys 0

9:14 p.m. TOUCHDOWN — That was quick! Purdy hits Kittle after some trickery, and it's a two-score night for the tight end. 49ers lead, 14-0.

9:13 p.m. — The Cowboys are forced to punt out of their own end zone after Fred Warner sacks Prescott. Warner now has a sack and a fumble. The 49ers will have
strong field position.

9:10 p.m. FUMBLE — Oh no again! They're trading fumbles! Jourdan Lewis takes the ball right out of McCaffrey's arms and spoils the 49ers' hopes of making this a
14-0 game.
</item 70>

<item 71>
Cowboys vs. 49ers live score, updates, highlights from NFL 'Sunday Night Football' game (Sporting News, 2023-10-08T22:47:03+00:00)
8:27 p.m. — A little bit of chaos as Deebo Samuel drops a ball that was pretty close to being a fumble, but it's all a non-issue as Dallas is hit with an
offsides penalty. The 49ers are right about in field goal range.

8:25 p.m. — After a facemask penalty and a nine-yard pass to Brandon Aiyuk, the 49ers are already over midfield.

8:23 p.m. — Kickoff is a bit on the later end, but we are about to get underway. The 49ers will start with the ball.

8:10 p.m. — Jerry Jones said this week that this game will be a test of how the Cowboys match up with the NFL's best, as if there wasn't already pressure on
Dallas to show something after the 49ers knocked them out of the playoffs in back-to-back years.

7:56 p.m. — Frank Gore and Vernon Davis are in the house!
</item 71>

<item 72>
Bills vs. Bengals live score, updates, highlights from NFL 'Sunday Night Football' game (Sporting News, 2023-11-05T23:47:00+00:00)
#RuleTheJungle



via @NFL pic.twitter.com/DKh9g4zugl — Sunday Night Football on NBC (@SNFonNBC) November 6, 2023

8:28 p.m.: Mixon gets his first rush of the game for three yards up the middle, then Burrow gets Smith involved, throwing a short pass to him. He takes it up to
the Bills' 7 for 11 yards, and the Bengals are at a first-and-goal.

8:28 p.m.: Burrow dodges a sack in the backfield, rolls out to his left and flips a pass to Tee Higgins for 18 yards up to the Bills' 21. Bengals are rolling
early.

Joe Burrow came ready for primetime! 💯

📺: NBC & Peacock



(Via: NFL)pic.twitter.com/bBnw5i4CvV — Sunday Night Football on NBC (@SNFonNBC) November 6, 2023

8:27 p.m.
</item 72>

<item 73>
Bills vs. Bengals live score, updates, highlights from NFL 'Sunday Night Football' game (Sporting News, 2023-11-05T23:47:00+00:00)
: Hamlin running out onto the field before the game.

6:29 p.m.: Damar Hamlin won't be playing against the Bengals, but he is working out on the field in Cincinnati.

Damar Hamlin is not expected to be active tonight, but is out on the field for a workout, taking in his return to Paycor Stadium. pic.twitter.com/nYzR62iTWl —
Lindsey Thiry (@LindseyThiry) November 5, 2023

6:19 p.m.: Joe Burrow arriving to the locker room with diamond headphones ahead of tonight's game.

Busted out the diamond striped headphones for SNF 🥶 pic.twitter.com/G7DXUAEpPo — Cincinnati Bengals (@Bengals) November 5, 2023

What channel is Bills vs. Bengals on today?

Game: Bills at Bengals

Bills at Bengals Date: Sunday, Nov. 5

Sunday, Nov.
</item 73>

<item 74>
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
</item 74>

<item 75>
Fantasy Football Week 13 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-11-30T22:10:58+00:00)
Easy peasy!

Start 'em 10.0-7.00 6.99 - 5.01 5.00 - 0 Sit 'em

Both teams had a week of rest, so don't treat this like a typical Thursday game. But do expect the Cowboys defense to keep the pressure on Geno Smith and a
Seattle offense that's struggled to gain consistency over the last month. Smith actually did OK completing passes last week when pressured (60% completion rate)
but not at an efficient level (5.4 yards per attempt). Just two QBs have exceeded 250 yards and four QBs have thrown multiple touchdowns against Dallas this
year; I don't like Smith's chances to do either, which hurts any upside case for any Seahawks receiver. By the way, Curtis Samuel was the first receiver to get
100 yards on the Cowboys this year, and it took him 12 targets and 9 catches to get there.
</item 75>

<item 76>
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
</item 76>

<item 77>
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
</item 77>

<item 78>
Bills vs. Bengals live score, updates, highlights from NFL 'Sunday Night Football' game (Sporting News, 2023-11-05T23:47:00+00:00)
He catches a nine-yard pass after Burrow rolls out to his right. Mixon then picks up the first by rushing for three yards to the Bills' 42.

8:44 p.m.: Burrow tries to connect with Ja'Marr Chase, but the throw is broken up. However, Burrow is roughed on the play, and the Bengals move up 15 yards.

8:43 p.m.: Williams with a solid return, moving up to the Cincinnati 31 to start the drive.

8:39 p.m. Bills scoring play: Allen rushes in for two yards, and is flagged for taunting as he runs in through the right side with 6:03 left in the first
quarter. All offense so far. Bengals 7, Bills 7

Two drives. Two touchdowns.



Strap in folks!
</item 78>

<item 79>
Bills vs. Bengals live score, updates, highlights from NFL 'Sunday Night Football' game (Sporting News, 2023-11-05T23:47:00+00:00)
10:19 p.m.: A healthy Burrow can be a game-changer for the Bengals, and he shows it on third-and-4, scrambling out of the pocket and picking up seven yards and
the first to keep the drive going.

10:18 p.m.: Bengals feeding Mixon to start the drive, with him rushing for three then catching a pass for three to set up a third-and-4.

10:17 p.m.: Williams returns the kickoff to the 16, but a penalty pushes the spot back to the Cincinnati 8.

10:13 p.m. Bills scoring play: Allen, on a third-and-9, tries to connect with Davis in the end zone, but the ball falls incomplete. Tyler Bass drills the
34-yard field goal to at least get the Bills a few points. Bengals 21, Bills 10

10:12 p.m.: Pressure to Allen has limited him on each of the past two passes.
</item 79>

<item 80>
NFL power rankings: Lions, 49ers level up; Texans jump Bengals, Bills; Jets, Patriots sink to bottom for Week 11 (Sporting News, 2023-11-13T21:54:00+00:00)
the Patriots. At least Jonathan Taylor is doing his normal things and the defense is playing well again while Gardner Minshew looks meh.

18. Los Angeles Chargers 4-5 (17)

The Chargers got well offensively with Justin Herbert, Keenan Allen and Austin Ekeler against the Lions. But the defense went back to crumbling vs. big pass
plays and added wilting against the run. The pressure is right back on Brandon Staley with a trip to Green Bay next.

20. Tampa Bay Buccaneers 4-5 (23)

Baker Mayfield and the offense is back on track to move the ball well after a brief midseason slump. They're still not lighting it up, but the Bucs are back to
playing smart complementary football with their defense rebounding and still might have found a long-term QB answer.

21.
</item 80>

<item 81>
Fantasy Football RB PPR Rankings Week 14: Who to start, best sleepers at running back (Sporting News, 2023-12-07T22:09:00+00:00)
Spears contributed 20 yards on the game-tying drive in the fourth quarter and another 27 yards in OT, but Mike Vrabel's squad ultimately lost. In what should be
a high-scoring game against the Super Bowl-hopeful 'Fins, we like Spears to be busy as rookie QB Will Levis's top check-down option -- assuming Henry
(head/neck) is out, of course.

Ezekiel Elliott (@ Steelers) is another potential fill-in who should produce at least flex-worthy stats this week. With Rhamondre Stevenson (ankle) on the
shelf, Elliott could see 20 touches. Similarly, Kenneth Gainwell (@ Cowboys) and Roschon Johnson (vs. Lions) could see big upticks in touches if D'Andre Swift
and D'Onta Foreman can't suit up, but we're not as confident those backs can produce given their matchups.
</item 81>

<item 82>
NFL Fantasy Football Week 5 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-04T21:56:25+00:00)
He's been so consistent and the 49ers are so loaded, but this is just one of those rare matchups where there's too much downside to buy into Purdy. After
ranking him high early in the week I've moved him behind Anthony Richardson, Jordan Love and Trevor Lawrence, but still ahead of Joe Burrow, Dak Prescott and
Daniel Jones.

LAST WEEK: Purdy's been a find ever since the 49ers put him on the field but last week he executed their offense flawlessly. Seriously -- he missed on one
throw. And for the second time this season, Purdy threw a pass that was ruled a lateral for a touchdown, robbing his Fantasy managers of precious points. Of
course, it helped that he played Arizona, a defense that has taken a step back in pressuring quarterbacks.
</item 82>

<item 83>
NFL Fantasy Football Week 5 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-04T21:56:25+00:00)
Only Joshua Dobbs, who ran for 55 yards, has scored double-digit Fantasy points against the Niners (18 points). Not that the Cowboys have been tested much --
Daniel Jones, Zach Wilson and the Patriots QBs are the other guys they've man-handled.

O-LINE: San Francisco has allowed a pass rush pressure on 36.4% of their snaps, which sounds like a lot but it's just a shade worse than league average. That
number is weighted by the Giants' decision to blitz on 84.6% of Purdy's dropbacks in Week 2, creating an inflated pass rush pressure rate. Despite the Giants'
efforts, Purdy still threw for 310 yards and two touchdowns in a game he didn't have Brandon Aiyuk in.

COWBOYS: Have become one of the most aggressive defenses in the league, ranking third from the bottom in zone coverage snaps.
</item 83>

<item 84>
NFL Fantasy Football Week 5 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-04T21:56:25+00:00)
LAST WEEK: Total disaster. In a gamescript that actually gave Minnesota a chance to -- gasp! -- run the ball, Kirk Cousins barely threw the ball and Addison
went catchless on one target. There was also a major reduction in snaps played, down to 58%. His season-low was 56% from Week 1.

MATCHUP: With the Vikings capable of running the ball, the offense went with heavier formations. Backup tight end Josh Oliver played about as many snaps as he
did in Week 1 and third tight end Johnny Mundt saw a season-high 17% of snaps. Week 4 was the first time all year the Vikings DIDN'T throw the ball on at least
69% of their snaps.

THIRD MAN: Last week did affirm that Addison may actually be the No. 3 wide receiver in the offense behind Justin Jefferson and K.J.
</item 84>

<item 85>
Cowboys vs. 49ers live score, updates, highlights from NFL 'Sunday Night Football' game (Sporting News, 2023-10-08T22:47:03+00:00)
11:00 p.m. INTERCEPTION — So much for that. Oren Burks gets the 49ers' third interception of the night as Prescott continues to struggle against this San
Francisco defense.

10:59 p.m. — The Cowboys are moving the ball now, when it's a bit too late. Tony Pollard rips off a 26-yard run to finally break through.

49ers 42, Cowboys 10

10:53 p.m. TOUCHDOWN — And just like that, Jordan Mason runs in for another 49ers touchdown! This is ugly on all levels for the Cowboys, and Leighton Vander
Esch is now down as well. 42-10, 49ers.

10:52 p.m. INTERCEPTION — Fred Warner now has a sack, forced fumble and an interception tonight. A pass from Prescott is deflected into his hands, and the 49ers
have another instant scoring opportunity.
</item 85>

<item 86>
Cowboys vs. 49ers live score, updates, highlights from NFL 'Sunday Night Football' game (Sporting News, 2023-10-08T22:47:03+00:00)
A perfectly drawn-up 26-yard play puts Dallas right back in this game.

KaVontae Turpin shows off the wheels and gets in the end zone 💨 @KaVontaeTurpin



📺: #DALvsSF on NBC

📱: Stream on #NFLPlus https://t.co/gzzkvIRUdj pic.twitter.com/1bwEuaj5pN — NFL (@NFL) October 9, 2023

9:24 p.m. — Prescott hits Jake Ferguson for a big third down conversion, and then some. That's 20 yards.

9:23 p.m. — Tony Pollard has had a tough night so far, with just 10 yards on six touches.

9:22 p.m. — Prescott fires a nice pass to Cooks to give the Cowboys another first down.

9:20 p.m. — The Cowboys have a first down! And it only took 17+ minutes.
</item 86>

<item 87>
NFL Fantasy Football Week 5 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-04T21:56:25+00:00)
And truthfully, the only quarters they've actually played heavy zone have been at the end of blowout wins over the Giants and Jets. If they keep that up they'd
look completely different than in their playoff meeting against Purdy in January when they played heavy zone for nearly 86% of their snaps.

PURDY: Has been sensational against both man and zone coverages this season. His completion rate and yards per attempt are noticeably higher against zone, but
four of his five passing touchdowns have come versus man coverage. However, Purdy's been at his worst when pressured (like most quarterbacks), completing 48.6%
of his throws for 5.6 yards per attempt with a gaudy 18.9% off-target rate.

LAST YEAR IN THE PLAYOFFS: The Cowboys pressured Purdy on 48.5% of his drop backs despite blitzing just 21.2% of the time.
</item 87>

<item 88>
Bills vs. Bengals live score, updates, highlights from NFL 'Sunday Night Football' game (Sporting News, 2023-11-05T23:47:00+00:00)
Allen then sneaks to the left side to pick up three and the first down to bring the third quarter to an end.

10:39 p.m.: Allen is putting on the superhero cape to start the drive, powering his way for a 10-yard run to start the drive then hitting Kincaid for 11 yards
to quickly get the Bills back into Cincinnati territory at the 44.

10:36 p.m.: The Bengals try for it on fourth down, but while Burrow connects with Sample, he's brought down three yards shy of the marker. The Bills take over
at their own 35.

10:34 p.m.: Burrow can't connect with Boyd, and it's now fourth-and-5. The Bengals keep the offense out there, but use a timeout with 2:21 left in the third.

10:33 p.m.: Jordan Poyer blows up a pitch to Mixon, bringing him down for a loss of four.
</item 88>

<item 89>
Fantasy Football Week 13 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-11-30T22:10:58+00:00)
Now they'll take on a Bengals offense that was borderline lucky to even score 10 points last week against the Steelers. Two of Ja'Marr Chase's four receptions
were tipped by Steelers first, then the Bengals' longest pass play was on a designed dump-off pass to Joe Mixon for 39 yards. Jake Browning had his moments
evading the pass rush but his arm strength is just not very good. Expect the Jaguars to bring plenty of heat to Browning, then lean into Travis Etienne on the
ground to bring home a win.
</item 89>

<item 90>
NFL Fantasy Football Week 5 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-04T21:56:25+00:00)
Tannehill could have more time to throw accurately.

Start 'em 10.0-7.00 6.99 - 5.01 5.00 - 0 Sit 'em

Sneaky Sleeper (Lineup Decisions) Projections powered by Sportsline Joe Burrow QB CIN Cincinnati • #9

Age: 27 • Experience: 4 yrs. Weekly Breakdown MATCHUP @ ARI CIN -3 O/U 44.5 OPP VS QB 25th PROJ PTS 20.8 QB RNK 19th YTD Stats PAYDS 728 RUYDS 3 TD 2 INT 2
FPTS/G 8.9 Look, if there's a matchup Burrow could piece together a decent stat line in, this would be it. Or Denver. Or Chicago. You get the gist.
</item 90>

<item 91>
NFL Fantasy Football Week 5 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-04T21:56:25+00:00)
But just three players have averaged over 4.0 yards per carry: Joe Mixon (4.5 yards per run, didn't have 100 total yards, didn't score), Zack Moss (4.1 yards
per run, ran for 122 yards on 30 carries and caught a touchdown) and Pierre Strong (9.8 yards per carry, most of it coming in garbage time last week).

HISTORY: Harris had over 120 total yards and a touchdown in a Week 17 win in Baltimore last year. He also scored on them with exactly 50 total yards against
them in at home in Week 14. He has posted at least 10 non-PPR points in 3 of 4 career games.
</item 91>

<item 92>
Bills vs. Bengals live score, updates, highlights from NFL 'Sunday Night Football' game (Sporting News, 2023-11-05T23:47:00+00:00)
Fans expected this to be an offensive showdown against two premier quarterbacks. And after the first three drives of the game, it sure appeared that's how it
would go. The Bengals scored an opening-drive touchdown on just nine plays, matched quickly by a Bills' scoring drive of seven plays and 3:15. The Bengals
re-took the lead on a 5:28 scoring drive to wrap up the first.

From there, however, "Mad Scientist" Lou Anarumo's defense kept Josh Allen and the Bills off balance offensively for most of the remainder of the game. The next
four drives of the half went punt, interception, punt, end of half.

Burrow and the Bengals, meanwhile, hit a few bumps in Buffalo territory, but scored a touchdown on their final drive of of the first half to take a two-score
lead into the locker room.
</item 92>

<item 93>
NFL Fantasy Football Week 6 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-12T21:23:16+00:00)
Both scored.

BRONCOS: When McLaughlin, Williams and Perine were all part of the Broncos game plan in their Week 3 blowout loss, Williams had the most work (13 touches) and
the most PPR points (8.5). The only time a Denver back had 15-plus touches came in Week 1 when Williams had 17 against the Raiders (and had 9.7 PPR points, a
season-high).
</item 93>

<item 94>
NFL power rankings: Lions, 49ers level up; Texans jump Bengals, Bills; Jets, Patriots sink to bottom for Week 11 (Sporting News, 2023-11-13T21:54:00+00:00)
They also seemed invigorated for Jonathan Gannon on defense and special teams. Look for the Cardinals to finish strong and play a whole lot of spoiler.

30. New England Patriots 2-8 (28)

The Patriots tried to win a game by hiding Mac Jones and their quarterbacks in general. They almost pulled it off with the running game dominating and
short-handed defense making a lot of plays, but they didn't see the results in Germany ahead of a tough bye for Bill Belichick.

31. New York Giants 2-8 (30)

The Giants can just flush away this season without Daniel Jones and Tyrod Taylor. The injuries go to their offense early and remained relentless, and now their
defense is falling apart, too. The question is, will Brian Daboll get a massive mulligan after his rookie playoff leadership?

32.
</item 94>

<item 95>
Cowboys vs. 49ers live score, updates, highlights from NFL 'Sunday Night Football' game (Sporting News, 2023-10-08T22:47:03+00:00)
The 49ers knocked the Cowboys out of the playoffs in both of the last two seasons, so a win for Dallas on Sunday could be helpful just from a psychological
standpoint.
</item 95>

<item 96>
N.F.L. Executives on the Biggest Questions Facing Super Bowl Contenders (The New York Times, 2023-10-16T23:58:53+00:00)
That figure was 7.8 in Mahomes’ first year as the starter and has climbed every season since. It’s a testament to the Chiefs’ adaptability, but also a
reflection of plummeting explosive play rates.

Exec comment: “Yeah, it looks harder for them on offense until Mahomes finds (Travis) Kelce on every major third down, every major two-minute play, every major
touchdown. Kansas City has the luxury of their best players being their toughest players. Chris Jones is a very tough player. Kelce is a very tough player.
Mahomes is a very tough player. You wonder why they win the games? Because the games are won by the toughest teams most of the time.”

• Philadelphia Eagles: Has Jalen Hurts regressed? Where are the explosive pass plays?

What I think: Explosive passing is down through most of the league, including for the Eagles.
</item 96>

<item 97>
NFL Fantasy Football Week 6 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-12T21:23:16+00:00)
But the biggest detriment is that he can't consistently overcome bad blocking, which has been a consistent problem for the Denver O-line all year and certainly
something that impacted Javonte Williams in his first handful of games.

WHEN TRAILING: When the Jets were up by eight or more points in the second half, McLauglhin played 4 of 15 snaps and had two carries. That's in line with a
trend we've seen from Denver all year: Of the 117 snaps they've played when down by eight or more points, McLaughlin has played 26 snaps (13 touches, seven at
Chicago), Williams has played 27 snaps (15 touches, almost all against Miami) and Samaje Perine has played 58 snaps (13 touches) -- more than both backs
combined. The only guy who's averaged a strong number on the ground in this specific situation? You guessed it -- McLaughlin (6.1 yards per carry).
</item 97>

<item 98>
Fantasy Football RB Rankings Week 12: Who to start, best sleepers at running back (Sporting News, 2023-11-21T21:33:00+00:00)
We have the Cowboys back just outside our top five with a juicy matchup against a Washington D that just got lit up by Saquon and the G-men.

Raheem Mostert (@ Jets) should continue to eat for Miami, with rookie De'Von Achane (knee) unfortunately getting hurt almost immediately after returning from IR
(more on that later). Gang Green hasn't been nearly as good against the run as it has been against the pass, as Jets opponents have averaged 139 rushing yards
per game over the past two weeks. 'Fins backup Salvon Ahmed also landed on the IR with a foot injury, so consider stashing Jeff Wilson Jr. as a handcuff.

Rounding out the top 10, the trio of Travis Etienne (@ Texans), Josh Jacobs (vs. Chiefs), and Breece Hall (vs. Dolphins) will be looking to bounce back from
disappointing Week 11 outings on the ground.
</item 98>

<item 99>
Bills vs. Bengals live score, updates, highlights from NFL 'Sunday Night Football' game (Sporting News, 2023-11-05T23:47:00+00:00)
That's a huge penalty as it backs the Bills out of field-goal range back to the Bengals' 48. NBC's Terry McAulay says that Allen threw it well past Davis, which
is what made it intentional grounding.

9:47 p.m.: Two more completions to Kincaid, first for four then another for no gain. But Taylor-Britt is injured on the second pass, and the Bengals have to use
a timeout to look him over on the field.

9:45 p.m.: Bills start at the 25, and Allen wastes no time getting Buffalo up the field. He finds Shakir for 22 yards to the 47, then hits Kincaid for 11 to get
Buffalo in Cincinnati territory at the 42.

9:44 p.m. Bengals scoring play: Sample lines up in the backfield, darts out, catches a short pass and races it to the end zone to add to Cincinnati's lead.
</item 99>

<item 100>
Fantasy Football RB PPR Rankings Week 14: Who to start, best sleepers at running back (Sporting News, 2023-12-07T22:09:00+00:00)
Eagles

15 Kyren Williams, Rams @ Ravens

16 Joe Mixon, Bengals vs. Colts

17 Javonte Williams, Broncos @ Chargers

18 Derrick Henry, Titans @ Dolphins

19 D'Andre Swift, Eagles @ Cowboys

20 Jaylen Warren, Steelers vs. Patriots

21 Alexander Mattison, Vikings @ Raiders

22 Jerome Ford, Browns vs. Jaguars

23 Kenneth Walker III, Seahawks

24 Najee Harris, Steelers vs. Patriots

25 James Cook, Bills @ Chiefs

26 Zack Moss, Colts @ Bengals

27 Keaton Mitchell, Ravens vs. Rams

28 AJ Dillon, Packers @ Giants

29 Ezekiel Elliott, Patriots @ Steelers

30 Aaron Jones, Packers @ Giants

31 Jerick McKinnon, Chiefs @ Bills

32 Tyjae Spears, Titans @ Dolphins

33 Gus Edwards, Ravens vs. Rams

34 Dameon Pierce, Texans @ Jets

35 Roschon Johnson, Bears vs.
</item 100>
=== END ITEM id=f3c97b11901a591b ===
