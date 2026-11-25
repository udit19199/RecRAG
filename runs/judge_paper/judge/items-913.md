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

=== ITEM id=8b7e6c6509f688ef ===
Gold sentences and facts:
Sentence 0: As an added bonus, Harris is also the preferred goal-line back, scoring in three of those games.
  Fact 0.0: Harris is also the preferred goal-line back.
  Fact 0.1: Harris scored in three of those games.
Sentence 1: The Wildcats on their first drive put a touchdown on the board a seven play, 61 yard drive, thanks to a 25-yard dash before diving into the end zone by running back Jonah Coleman.
  Fact 1.0: The Wildcats scored a touchdown on their first drive.
  Fact 1.1: The Wildcats' first drive was a seven play drive.
  Fact 1.2: The Wildcats' first drive covered 61 yards.
  Fact 1.3: The Wildcats' first-drive touchdown came thanks to a 25-yard dash.
  Fact 1.4: The 25-yard dash was followed by a dive into the end zone.
  Fact 1.5: Running back Jonah Coleman dove into the end zone.

Retrieved items:
<item 1>
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
</item 1>

<item 2>
Fantasy Football WR PPR Rankings Week 13: Who to start, best sleepers at wide receiver (Sporting News, 2023-11-29T20:27:00+00:00)
Tyreek Hill (@ Commanders in Week 13) did his thing despite a tough matchup in Week 12, resulting in Miami's WR1 returning to the top spot against a putrid
Washington stop unit. CeeDee Lamb (vs. Seahawks) isn't far behind, as Dallas' WR1 aims to maintain his efficient play against Seattle.

Amon-Ra St. Brown (@ Saints) has been a target hog the past several weeks, seeing at least nine looks in each of his past six games. Don't expect his workload
to decrease in a bounce-back game for the Lions. Keenan Allen (@ Patriots) looks to continue his dominant ways, as Justin Herbert's go-to target routinely has
double-digit target potential.

Mike Evans (vs. Panthers) sits in our top five for Week 13, totaling four TDs over the past three weeks. A.J. Brown (vs.
</item 2>

<item 3>
MVPs for all 16 AFC teams at the midpoint (Yardbarker, 2023-11-09T21:51:02+00:00)
Stroud: Stroud is having one of the best statistical rookie seasons of all time. He ranks in the top 10 in passing yards (2,270), touchdown passes (14) and
passer rating (102.9), and has only thrown one interception. The Texans (4-4) aren’t close to a .500 team without him.

INDIANAPOLIS COLTS | RB Zack Moss: With Jonathan Taylor missing the first four weeks of the season, Moss stepped up in a big way. After not playing in Week 1,
he rushed for 445 yards and scored four touchdowns over his next four games to help Indy win three of four and stay in the thick of the AFC South. Moss is
second in the NFL in rushing, 37 yards behind San Francisco's Christian McCaffrey.
</item 3>

<item 4>
Fantasy Football RB PPR Rankings Week 14: Who to start, best sleepers at running back (Sporting News, 2023-12-07T22:09:00+00:00)
Entering Week 13, only two teams had allowed more PPR points than the Panthers (28.9 FPPG).

We also love Saquon Barkley (vs. Packers) against the Packers' leaky run defense and Breece Hall (vs. Texans) in what should be a bounce-back game. Entering
Week 13, Houston had allowed 112.9 scrimmage yards per game and nine total TDs to RBs on the year.

WEEK 14 FANTASY ADVICE:

Sleepers | Busts | Projections | Start-sit

Who are the best fantasy RB sleepers, waiver pickups Week 14?

Welcome to the top 25, Tyjae Spears (@ Dolphins)! We've loved the way this kid has run all season. He averaged 5.1 yards per carry entering last weekend, and he
stepped up in a major way for Tennessee when Derrick Henry sustained a game-ending hit to the head area against Indy.
</item 4>

<item 5>
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
</item 5>

<item 6>
Michigan didn't need Jim Harbaugh against Penn State, but Ohio State is a different game (Sporting News, 2023-11-11T21:32:44+00:00)
Will Harbaugh be on the field before the end of the regular season? The Big Ten made a Friday ruling that will prohibit Harbaugh from being on the field for the
final three regular-season games as a result of the in-person scouting and sign-stealing scandal that led to the resignation of staffer Conor Stallions. ESPN
reported there will be a temporary restraining order hearing on Nov. 17 that will answer that question.

On the field, Michigan answered the question about its College Football Playoff. Sort of. The Wolverines proved they could beat Penn State (8-2, 5-2) in a
20-yard box. Neither team completed a pass of 20 yards or more. Michigan trailed 3-0 after the first quarter. The Wolverines had eight carries for 10 yards, and
Penn State's Abdul Carter set the tone with a jarring hit on Blake Corum that nearly resulted in a fumble.

Michigan took control with the running game.
</item 6>

<item 7>
Fantasy Injury Updates Week 10: News on Justin Jefferson, Ja'Marr Chase, T.J. Hockenson & Curtis Samuel (Sporting News, 2023-11-11T21:17:00+00:00)
Chase put a scare into the fantasy community on Wednesday when he was listed as a DNP, but after logging back-to-back limited practices, there is a lot more
optimism heading into Sunday. With Tee Higgins (hamstring) already ruled out, Cincinnati will be hoping Chase can feel healthy enough to play when they host the
Texans at 1 p.m. ET on Sunday.

The Bengals, despite getting a win over the Bills last Sunday, are a beaten-down bunch. Their defense will also be without a prominent edge rusher, Sam Hubbard,
complicating matters for a team hoping to stack wins as they compete for an AFC North crown. On the bright side, Joe Burrow appears to be finally healthy and
should have no problems targeting Tyler Boyd, Trenton Irwin, and Andrei Iosivas in the passing game. If Chase does end up suiting up, fantasy owners should
enjoy an even larger workload than the receiver already had.
</item 7>

<item 8>
Fantasy Football Week 13 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-11-30T22:10:58+00:00)
Now they'll take on a Bengals offense that was borderline lucky to even score 10 points last week against the Steelers. Two of Ja'Marr Chase's four receptions
were tipped by Steelers first, then the Bengals' longest pass play was on a designed dump-off pass to Joe Mixon for 39 yards. Jake Browning had his moments
evading the pass rush but his arm strength is just not very good. Expect the Jaguars to bring plenty of heat to Browning, then lean into Travis Etienne on the
ground to bring home a win.
</item 8>

<item 9>
Fantasy Football RB Rankings Week 12: Who to start, best sleepers at running back (Sporting News, 2023-11-21T21:33:00+00:00)
We have the Cowboys back just outside our top five with a juicy matchup against a Washington D that just got lit up by Saquon and the G-men.

Raheem Mostert (@ Jets) should continue to eat for Miami, with rookie De'Von Achane (knee) unfortunately getting hurt almost immediately after returning from IR
(more on that later). Gang Green hasn't been nearly as good against the run as it has been against the pass, as Jets opponents have averaged 139 rushing yards
per game over the past two weeks. 'Fins backup Salvon Ahmed also landed on the IR with a foot injury, so consider stashing Jeff Wilson Jr. as a handcuff.

Rounding out the top 10, the trio of Travis Etienne (@ Texans), Josh Jacobs (vs. Chiefs), and Breece Hall (vs. Dolphins) will be looking to bounce back from
disappointing Week 11 outings on the ground.
</item 9>

<item 10>
NFL DFS picks Week 15: Best sleepers, value players for FanDuel, DraftKings, SuperDraft lineups (Sporting News, 2023-12-13T20:32:00+00:00)
Stroud in the league's concussion protocol and Nico Collins (calf) joining Tank Dell (fibula) on the Texans' injury list, Houston suddenly finds itself without
many impact skill-position players. Singletary, however, has been pretty damn good and showed us last week that he's the main man over Dameon Pierce in DeMeco
Ryans' backfield. He had 13 carries for 65 yards and a TD against the Jets, while Pierce totaled just five touches for seven yards. Don't be scared of
Tennessee's run D because it ain't what it used to be under Mike Vrabel. Raheem Mostert and De'Von Achane just posted 35.1 PPR points against the Titans, Chuba
Hubbard had 20.2 a few weeks ago, and Rachaad White had 17.8 against them in Week 10.
</item 10>

<item 11>
Fantasy Football RB Rankings Week 14: Who to start, best sleepers at running back (Sporting News, 2023-12-07T22:07:00+00:00)
The injury highlights the importance of handcuffs, as Zack Moss has put up multiple top-10 performances with and without Taylor this season.

Seattle's Kenneth Walker III (oblique) missed Week 12 and 13, and we're not confident in his ability to suit up against the hard-hitting 49ers coming off a big
win in Philly. Backup Zach Charbonnet exited early because of a knee injury against the Cowboys, putting his status in doubt, too. If both are out, DeeJay
Dallas would start and lead the backfield in touches.

Green Bay's Aaron Jones has a sprained MCL, and sources have revealed that the Packers continue to consider him week-to-week. He only logged four carries before
sustaining the injury in Week 11, then he sat for the Packers' Thanksgiving Day tilt with the Lions and their Sunday Night Football win over Kansas City. Why
rush him back if Green Bay is already winning games?
</item 11>

<item 12>
Fantasy Football Week 13 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-11-30T22:10:58+00:00)
While it's true that the Broncos have given up just one rushing touchdown in their past six games, it's also true they've let up 5.4 yards per carry in those
same six games. Bank on the Texans giving Singletary more work on the ground to pace their offense. I'd start him over Jerome Ford, Javonte Williams and Joe
Mixon.

Start 'em 10.0-7.00 6.99 - 5.01 5.00 - 0 Sit 'em

The Patriots have seemingly backed off of being blitz-heavy over their past three games, perhaps because they realized they can get similar payoffs without it
(only three of their past nine sacks have come on blitzes). They also have an inflated sack total because they recorded six against the Giants -- they have no
more than three in any other game this year including zero against the Colts in Week 10.
</item 12>

<item 13>
Fantasy Football RB Rankings Week 12: Who to start, best sleepers at running back (Sporting News, 2023-11-21T21:33:00+00:00)
As we trot into Turkey Day week, only three more regular-season weeks remain until the fantasy football playoffs. Every decision — from the tough starts to the
difficult sits; the savvy waiver adds to the costly drops — can ultimately decide whether your team is a contender or a pretender. Our Week 12 fantasy RB
rankings for standard leagues will help guide you to the right moves and hopefully land you a big "W" for which to be thankful.

Trade deadlines have passed in most leagues, so the majority of fantasy owners must soldier on toward the home stretch with the main core they have built to
this point. On a positive note, we have no byes to worry about this — just plenty of short turnarounds, some potential weather concerns, and of course, the
dreaded injury bug. Fortunately for us, we have been ahead of the curve with sleepers and breakouts all season, so we're more prepared to weather any storm than
the average fantasy Joe.
</item 13>

<item 14>
No. 17 Arizona dominates No. 22 Utah, 42-18 (Rivals, 2023-11-18T23:45:32+00:00)
Following the trickery, another huge roar of the Wildcat faithful ensued following a blocked punt and score by Anthony Ward, the first Arizona blocked punt and
score since Nov. 13, 2021.

The Wildcats’ momentum was rolling like a well-oiled machine on both ends of the ball in the first half, scoring offensive touchdowns on three of four drives,
led by Michael Wiley tallying two of them, and outgaining the Utes by 98 yards on the ground, as well as forcing the Utes to two three-and-outs, three punts and
a red zone interception, headlined by captain linebacker Jacob Manu, the first Arizona linebacker interception since Anthony Pandy against USC in 2021. With
rain starting to pour late in the first half in a rare Arizona “rain game”, the overcast didn’t hinder the Zona Zoo from leaving the stadium, nor the Wildcats
from letting up their progression, resulting in the UA taking staggering 28-7 lead at halftime.
</item 14>

<item 15>
NFL Fantasy Football Week 6 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-12T21:23:16+00:00)
TITANS PAST TWO: Against the Bengals and Colts, the Titans allowed at least 2.79 yards before contact per rush in each game and were decimated by zone-scheme
runs (at least 6.0 rush yards per carry by RBs).

RAVENS RUN GAME: Have called zone runs half as many times as power runs even though they average a half-yard better per carry on zone runs. They'd be smart to
do more of that this week against Tennessee. Hill has the team's best rushing average and highest rush EPA on zone runs.

HILL: In the three games he's played the most snaps in (Weeks 1, 2 and 5), he's played 24 of a possible 43 third-down snaps (56%) and 14 of 20 snaps inside the
10. He's smaller than Gus Edwards, but faster and more versatile, and it's given him extended opportunities for high-value touches.
</item 15>

<item 16>
NFL Fantasy Football Week 5 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-04T21:56:25+00:00)
I'd rather start Breece Hall, Rhamondre Stevenson, Dameon Pierce and Jaleel McLaughlin (if Javonte Williams sits out).

LAST WEEK: Harris had his first game with over 100 total yards against the Texans and averaged 5.1 yards per rush, just shy of the season-high he technically
had on six carries against the 49ers in Week 1. A 32-yard reception helped him cross the triple-digit barrier.

HARRIS: This might shock you, but Harris is averaging 4.3 yards per run this year with a 12.2% explosive run rate. Would you believe the explosive run rate,
which reflects the frequency of his 12-plus-yard runs, is fifth-best among 44 qualifying running backs?! He's also 13th in avoided tackle rate and top-10 in
yards after contact per carry at 3.53.

TOUCHDOWNS: Harris has zero.
</item 16>

<item 17>
NFL Fantasy Football Week 5 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-04T21:56:25+00:00)
STEVENSON: The Patriots front five has gone through all kinds of issues and as a result have contributed to Stevenson averaging 0.98 yards before contact per
rush this season. That ranks 29th out of 44 qualifying running backs. And if you think that's bad then consider his yards after contact per rush average of
1.75, which is dead-last among those same 44 qualifiers.

EXPLOSIVE RUNS: Stevenson has one (1) explosive carry of 12 or more yards. It was against Miami. It went 12 yards. This is a big change from 2022 when he had 21
explosive runs and an explosive run rate of 10%, which was eighth-best.

RECEIVING: Is averaging 4.0 targets per game and catching 75% of them but is getting just 7.3 yards per catch and has one (1) explosive catch of 16 or more
yards.
</item 17>

<item 18>
Fantasy Football RB Rankings Week 12: Who to start, best sleepers at running back (Sporting News, 2023-11-21T21:33:00+00:00)
We'll be updating these standard RB rankings all week, so check back for the latest player movement.

Fantasy RB Rankings Week 12: Who to start, sit at running back

Rankings based on standard, non-PPR scoring

Rank Player

1 Christian McCaffrey, 49ers @ Seahawks

2 Austin Ekeler, Chargers vs. Ravens

3 Saquon Barkley, Giants vs. Patriots

4 David Montgomery, Lions vs. Packers

5 Jahmyr Gibbs, Lions vs. Packers

6 Tony Pollard, Cowboys vs. Commanders

7 Raheem Mostert, Dolphins @ Jets

8 Travis Etienne, Jaguars @ Texans

9 Josh Jacobs, Raiders vs. Chiefs

10 Breece Hall, Jets vs.
</item 18>

<item 19>
Fantasy Football RB Rankings Week 14: Who to start, best sleepers at running back (Sporting News, 2023-12-07T22:07:00+00:00)
We'll talk more about Derrick Henry (concussion) in the injury section, but we can say with confidence here and now that Tyjae Spears (@ Dolphins) should be the
hottest waiver-wire pickup of Week 14. Tennessee is a run-first offense and King Henry might be sidelined with a head injury, so Spears could very well pick up
where he left off against Indy last weekend. The young back averaged 4.7 yards per carry (just shy of his 5.1 YPC entering the week) en route to 88 yards.

Another backup getting a boost this week is the Patriots' Ezekiel Elliott (@ Steelers), who will assume lead-back duties after Rhamondre Stevenson went down
with a pretty bad ankle injury in Week 13. Zeke might put in a vintage performance considering James Conner just went back to Pittsburgh and hung 105 yards and
two TDs on his old squad.

Could this be the weekend that Keaton Mitchell (vs.
</item 19>

<item 20>
Fantasy Football RB Rankings Week 12: Who to start, best sleepers at running back (Sporting News, 2023-11-21T21:33:00+00:00)
Don't worry — we'll steer you in the right direction and keep you apprised of all the information you'll need throughout the week.

WEEK 12 PPR RANKINGS:

QBs | RBs | WRs | TEs | D/STs | Kickers

Who are the best fantasy RBs in Week 12?

As always, Christian McCaffrey (@ Seahawks in Week 12) heads the list as the most matchup-proof running back in the NFL. Right behind him, Austin Ekeler (vs.
Ravens) will look to bounce back after an uncharacteristically rough showing in Green Bay last week.

Against the Commanders last week, Saquon Barkley (vs. Patriots) looked as good as we've seen him this season (140 total yards, two TDs).
</item 20>

<item 21>
2024 NFL Mock Draft: Bengals' Joe Burrow gets another playmaker; four QBs go in Round 1, but Pats opt for WR (CBSSports.com, 2023-11-17T20:07:09+00:00)
Buffalo adds him in the first round to fill out its secondary.

Round 1 - Pick 18 Keon Coleman WR Florida State • Jr • 6'4" / 215 lbs Projected Team Cincinnati PROSPECT RNK 16th POSITION RNK 3rd Tee Higgins and Cincinnati
will likely part ways after the season, which creates a need in that offense. Keon Coleman is a big-bodied receiver who can win jump balls down the field.

Round 1 - Pick 19 JT Tuimoloau EDGE Ohio State • Jr • 6'4" / 270 lbs Projected Team New Orleans PROSPECT RNK 15th POSITION RNK 4th New Orleans adds a powerful
rusher to its repertoire. Cam Jordan is 34 years old and has two years remaining on his contract. The Saints have a quality rotation of rushers until JT
Tuimoloau eventually takes over.
</item 21>

<item 22>
Fantasy Football RB PPR Rankings Week 14: Who to start, best sleepers at running back (Sporting News, 2023-12-07T22:09:00+00:00)
Spears contributed 20 yards on the game-tying drive in the fourth quarter and another 27 yards in OT, but Mike Vrabel's squad ultimately lost. In what should be
a high-scoring game against the Super Bowl-hopeful 'Fins, we like Spears to be busy as rookie QB Will Levis's top check-down option -- assuming Henry
(head/neck) is out, of course.

Ezekiel Elliott (@ Steelers) is another potential fill-in who should produce at least flex-worthy stats this week. With Rhamondre Stevenson (ankle) on the
shelf, Elliott could see 20 touches. Similarly, Kenneth Gainwell (@ Cowboys) and Roschon Johnson (vs. Lions) could see big upticks in touches if D'Andre Swift
and D'Onta Foreman can't suit up, but we're not as confident those backs can produce given their matchups.
</item 22>

<item 23>
NFL roundup: Pats hit new low as Lions reach 9-3 for first time since 1962 (The Guardian, 2023-12-03T22:11:26+00:00)
Joe Flacco was 23 of 44 for 254 yards with two touchdowns and an interception in his Browns debut. Cleveland fell to 7-5 and occupy the final playoff spot in
the AFC.

Arizona Cardinals 24–10 Pittsburgh Steelers

James Conner ran for 105 yards and a pair of touchdowns against his old team as the Arizona Cardinals beat the Pittsburgh Steelers. It was the Cardinals’ first
win at Pittsburgh since 1969. Arizona took control late in the first half with a 99-yard touchdown drive that ended with a five-yard strike from Kyler Murray to
Trey McBride. The Cardinals led the rest of the way, enduring a pair of lengthy delays due to severe weather. The Steelers lost starting quarterback Kenny
Pickett to an ankle injury late in the first half and scored their only touchdown in garbage time.
</item 23>

<item 24>
2023 NFL Week 11 injury roundup: Rams WR Cooper Kupp, Seahawks RB Kenneth Walker III among stars to exit (CBSSports.com, 2023-11-19T23:51:49+00:00)
Dillon. Green Bay ruled him questionable to return, then downgraded him to out, but coach Matt LaFleur said after the Packers' win over Los Angeles that he
doesn't expect the injury to be a long-term issue.

De'Von Achane (Dolphins) returned from injured reserve -- and a four-game absence -- but limped to the sidelines after his second touch of the first quarter
against the Raiders. Miami ruled him questionable to return with a knee injury, and while he later returned to the sidelines from a locker-room visit, he was
replaced on the field indefinitely by Raheem Mostert. Head coach Mike McDaniel told reporters on Monday that Achane is day-to-day. While he hasn't been ruled
out of Friday's matchup with the Jets, he's not been given the green light either, so his status is worth monitoring.

Kenneth Walker III (Seahawks) took four carries against the Rams before leaving with an oblique injury.
</item 24>

<item 25>
Fantasy Football RB Rankings Week 12: Who to start, best sleepers at running back (Sporting News, 2023-11-21T21:33:00+00:00)
Keep rolling with Alexander Mattison (vs. Bears), too, against another below-average run defense.

Welcome to the top 25, Zach Charbonnet (vs. 49ers)! With Kenneth Walker III's oblique injury comes opportunity for the rookie, who already looked strong in the
month leading up to Seattle's narrow Week 11 loss to the Rams. Sure, Charbs' 69-yard effort in Los Angeles last week was a bit underwhelming, but we love his
upside with a full game as the Seahawks' bell-cow.

Speaking of bell-cow backs, things are looking very promising for the short-term workload of Green Bay's AJ Dillon (@ Lions). Veteran Aaron Jones (knee) and
rookie Emanuel Wilson (shoulder) both got hurt in Week 11, so Dillon should be the main man in Motown for the Pack.
</item 25>

<item 26>
Fantasy Football RB PPR Rankings Week 14: Who to start, best sleepers at running back (Sporting News, 2023-12-07T22:09:00+00:00)
We still like Jaylen Warren better overall, but you can confidently take Najee Harris (vs. Patriots) out of the doghouse for good. Harris has been running with
purpose, and his role in the passing game continues to make him a high-ceiling RB2 in a run-first offense that will be without Kenny Pickett (ankle) this week.
New England might be banged up, but it's still a tough D to get chunk yardage against through the air. Look for Harris to see lots of screens and dump-downs in
a rough-and-tumble Thursday night game in Pittsburgh.

We love Baltimore's Keaton Mitchell (vs. Rams) and think he has the potential to smash Los Angeles at home this week.
</item 26>

<item 27>
Fantasy Football RB Rankings Week 14: Who to start, best sleepers at running back (Sporting News, 2023-12-07T22:07:00+00:00)
Titans

6 De'Von Achane, Dolphins vs. Titans

7 Breece Hall, Jets vs. Texans

8 Jahmyr Gibbs, Lions @ Bears

9 Alvin Kamara, Saints vs. Panthers

10 Saquon Barkley, Giants vs. Packers

11 Tony Pollard, Cowboys vs. Eagles

12 Bijan Robinson, Falcons vs. Buccaneers

13 Derrick Henry, Titans @ Dolphins

14 Kyren Williams, Rams @ Ravens

15 Josh Jacobs, Raiders vs. VIkings

16 Rachaad White, Buccaneers @ Falcons

17 Joe Mixon, Bengals vs. Colts

18 D'Andre Swift, Eagles @ Cowboys

19 Javonte Williams, Broncos @ Chargers

20 Kenneth Walker III, Seahawks @ 49ers

21 Jaylen Warren, Steelers vs. Patriots

22 Jerome Ford, Browns vs.
</item 27>

<item 28>
Fantasy Football RB Rankings Week 12: Who to start, best sleepers at running back (Sporting News, 2023-11-21T21:33:00+00:00)
However, even a moderate MCL sprain could mean a multi-week absence for the injury-prone veteran back, especially in a lost season for the 4-6 Pack. Rookie
Emanuel Wilson also hurt his shoulder in Week 11, so AJ Dillon could be getting the keys to Green Bay's backfield castle (although it's more of an oversized
shack, at this point).

Dolphins rookie De'Von Achane, who looked like the runaway Offensive Rookie of the Year earlier this season, looked to have aggravated his MCL injury upon his
Week 11 return against the Raiders. Achane has a chance to return for Miami's Week 12 tilt with the Jets on Friday, but we wouldn't be surprised if the 7-3
'Fins play it safe and keep him out until he's back to 100 percent. Meanwhile, Miami placed backup Salvon Ahmed on IR with a foot injury, so Jeff Wilson Jr.
could be back in play as a handcuff stash.
</item 28>

<item 29>
Fantasy Football Week 13 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-11-30T22:10:58+00:00)
Weekly Breakdown MATCHUP vs DEN HOU -3.5 O/U 47 OPP VS RB 22nd PROJ PTS 8.7 RB RNK 19th YTD Stats RUYDS 489 REC 18 REYDS 113 TD 3 FPTS/G 8.8 Singletary
disappointed because he didn't get many carries last week, but that's uncharacteristic of the Texans offense. Some of that was made up for with receptions,
giving him weak yardage totals without a touchdown. Rest assured, Singletary is still clearly Houston's lead back -- he played 81% of their snaps and dominated
third downs and snaps inside the 5, even though it appeared Pierce rotated drives with him starting with the third one. The matchup is pretty good as Denver's
run defense is the only element that hasn't improved much this season.
</item 29>

<item 30>
No. 17 Arizona dominates No. 22 Utah, 42-18 (Rivals, 2023-11-18T23:45:32+00:00)
"Great win for the program," Arizona head coach Jedd Fisch said following the win. "Great win for our team, great win for our seniors, great win for Tucson,
appreciate Tucson coming out in full force, selling out the stadium. ... I think you might as well put the season tickets on sale for next year on Monday and
see if we can get six sell outs."

Leading the Wildcats was junior safety Dalton Johnson, logging 11 tackles as well as two tackles for loss in the UA’s routing.

"Dalton is a fantastic tackler," Fisch said. "When he's around the ball, he gets them down."

After failing continuously on trick plays involving receiver Tetairoa McMillan passing behind the scrimmage this season, Arizona head coach Jedd Fisch designed
it again — this time to perfection — with McMillan finding a wide-open fifth-year senior Michael Wiley to score a touchdown in his last game in Tucson.
</item 30>

<item 31>
Patriots vs. Steelers FanDuel Picks: NFL DFS lineup advice for Week 14 TNF single-game tournaments (Sporting News, 2023-12-07T21:26:00+00:00)
Expect the RB to be fed a ton. Whether that results in a touchdown remains to be seen, but if anyone is scoring a TD on the Pats roster Thursday night, it's
Elliott.

FLEX RB Najee Harris, Steelers ($14,000). Much like the Patriots, this Steelers offense will want to establish the ground game, and that starts with Harris. It
wasn't long ago that the RB was the subject of major criticism for looking slow and ineffective, but that narrative has flipped in a big way. With an average of
15 carries per game in the past five, we can expect at least that and maybe more from a Pittsburgh team that is starting Mitch Trubisky on Thursday night. As an
added bonus, Harris is also the preferred goal-line back, scoring in three of those games. The Patriots' defense doesn't give up much, making Harris our best
bet for success in an old-fashioned defensive struggle between two of the worst offenses the NFL has to offer.
</item 31>

<item 32>
NFL Fantasy Football Week 5 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-04T21:56:25+00:00)
I'd rather start Jahmyr Gibbs, Dameon Pierce and Najee Harris.

LAST WEEK: His playing time hit a season-low 43% of the snaps as he continued losing third-down work to Chuba Hubbard. It hasn't crushed his reception totals at
all but it's notable that his only game with more than four grabs came with Andy Dalton under center. Sanders also played just 31% of the Panthers' second-half
snaps, a big-time change after his second-half snaps were a lot closer to Hubbard's in Weeks 2 and 3.

SANDERS: His yards per carry has tumbled like a bowling ball downstairs this season, going from 4.0 against the Falcons to 3.1 against the Saints to 2.7 against
the Seahawks to 1.5 against the Vikings. Tough matchups? Sure, even including the Vikings. Doesn't mean it's not a good look.
</item 32>

<item 33>
Fantasy Football RB Rankings Week 12: Who to start, best sleepers at running back (Sporting News, 2023-11-21T21:33:00+00:00)
Chiefs

10 Breece Hall, Jets vs. Dolphins

11 Rhamondre Stevenson, Patriots @ Giants

12 Alvin Kamara, Saints @ Falcons

13 Isiah Pacheco, Chiefs @ Raiders

14 Brian Robinson Jr., Commanders @ Cowboys

15 Bijan Robinson, Falcons vs. Saints

16 Jonathan Taylor, Colts vs. Buccaneers

17 James Conner, Cardinals vs. Rams

18 D'Andre Swift, Eagles vs. Bills

19 Joe Mixon, Bengals vs. Steelers

20 Derrick Henry, Titans vs. Panthers

21 Rachaad White, Buccaneers @ Colts

22 Kyren Williams, Rams @ Cardinals

23 Alexander Mattison, Vikings vs. Bears

24 Gus Edwards, Ravens @ Chargers

25 Zach Charbonnet, Seahawks vs. 49ers

26 Javonte Williams, Broncos vs.
</item 33>

<item 34>
Fantasy Football RB PPR Rankings Week 14: Who to start, best sleepers at running back (Sporting News, 2023-12-07T22:09:00+00:00)
(@ Browns) even in a tough matchup. Trevor Lawrence and his high-flying offense should dominate the time of possession against the Joe Flacco-led Browns,
leaving ETN plenty of opportunities to add to his per-game PPR average of 17.4 points (third best in the NFL entering Week 13).

Guess who's back? Back again? Achane's back — tell a friend! Stud rookie De'Von Achane (vs. Titans) went right back to roughing up the opposition in his Week 13
return from a knee injury that sidelined him one week prior. The potential league-winner casually put up 103 yards and two TDs on the Commanders, and we can't
help but place him and running mate Raheem Mostert in the top five this week despite a somewhat-tough matchup with Tennessee.
</item 34>

<item 35>
Fantasy Football RB PPR Rankings Week 14: Who to start, best sleepers at running back (Sporting News, 2023-12-07T22:09:00+00:00)
Kansas City's front-seven has been tough against speedsters and more vulnerable to power backs, and the 5-11, 190-pound Cook 100 percent falls into the former
category. Temper your expectations.

We feel the same way about Zach Charbonnet (@ 49ers). It's great that the Week 13 injury he sustained on Thursday night was just a bruise, but it's not great
that he now faces a 49ers defense that already stifled him on Thanksgiving. You're probably starting him if Kenneth Walker III remains out with his oblique
injury, but consider other options if you have the depth. The floor and ceiling remain low against this uber-confident Niners D.

Tyler Allgeier (vs. Buccaneers) continues to see his role diminish for the NFC South-leading Falcons, with dum-dum coach Arthur Smith finally realizing that
Bijan Robinson gives the ATL the best chance at continued success.
</item 35>

<item 36>
Michigan conquers Ohio State again, but the story is far from over for 2023 Wolverines (Sporting News, 2023-11-25T23:20:56+00:00)
That set the Wolverines up at the 7-yard line. Corum slammed into the line of scrimmage three times before a fourth-and-1.

Corum heard Moore scream, "Smash." Tight end A.J. Barner did his best Triple H impersonation in the huddle: "It's time to play the game."

Michigan goes for it on 4th and goal and Blake Corum punches it in for The Game's first touchdown.



That's Corum's 21st touchdown run of the season 😲pic.twitter.com/TrF5pIpni3 — 247Sports (@247Sports) November 25, 2023

"I gotta score," Corum said. "Run through the hole and I'm about to run somebody over. That makes the hairs on my arm stick up."

Corum scored on a one-yard touchdown for a 7-0 lead. Michigan never played from behind against the Buckeyes. What was Moore thinking in that situation?
</item 36>

<item 37>
Fantasy Football RB Rankings Week 12: Who to start, best sleepers at running back (Sporting News, 2023-11-21T21:33:00+00:00)
Both Darrell Henderson and Royce Freeman (@ Cardinals) would be decent dart throws in a week filled with byes, but their floors in a zero-bye week seem too low
to trust them if Kyren Williams is back and at 100 percent. The matchup is juicy, but we have no idea what to expect from a touch-share perspective, so these
guys are no better than desperation, TD-dependent flexes.

Speaking of guys coming back from injury, we saw Khalil Herbert (@ Vikings) return to the field for Chicago against the Lions last week. However, his
performance left a lot to be desired. He finished with just 35 yards on 16 carries and wound up vastly overshadowed by Justin Field's 100-yard rushing effort.
Even worse, D'Onta Foreman scored the Chi's lone rushing TD on the afternoon.

I don't care what your team looks like — you shouldn't be starting Chuba Hubbard or Miles Sanders (@ Titans).
</item 37>

<item 38>
Fantasy Football RB PPR Rankings Week 14: Who to start, best sleepers at running back (Sporting News, 2023-12-07T22:09:00+00:00)
Rams) and think he has the potential to smash Los Angeles at home this week. The speedster has quietly racked up over 300 yards and two TDs over the Ravens'
past four games, and with Lamar Jackson facing Aaron Donald and a still-talented pass-rush, John Harbaugh's squad might turn to the rookie to move the chains
between the 20s. All we need is one or two big chunk gains to make him a worthwhile start — he can run friggin' 22 miles per hour!

WEEK 14 DFS:

DK lineup | FD lineup | Best values | Best stacks

Who are the biggest potential fantasy RB busts in Week 14?

While he just barely cracks our top 25 thanks to his passing-down usage, James Cook (@ Chiefs) doesn't thrill us at Arrowhead against the pissed-off defending
champs.
</item 38>

<item 39>
Fantasy Football RB Rankings Week 14: Who to start, best sleepers at running back (Sporting News, 2023-12-07T22:07:00+00:00)
Why rush him back if Green Bay is already winning games?

D'Onta Foreman (ankle) missed Chicago's game before the bye, and his status remains unclear heading into Week 14. If he's healthy, he'll likely start in
Chicago's backfield, but it's anyone's guess how carries will be divvied up after Roschon Johnson surprisingly led the way in Week 12.

We'll be updating these standard RB rankings all week, so check back for the latest player movement.

Fantasy RB Rankings Week 14: Who to start, sit at running back

Rankings based on standard, non-PPR scoring

Rank Player

1 Christian McCaffrey, 49ers vs. Seahawks

2 Austin Ekeler, Chargers vs. Broncos

3 Travis Etienne, Jaguars @ Browns

4 David Montgomery, Lions @ Bears

5 Raheem Mostert, Dolphins vs. Titans

6 De'Von Achane, Dolphins vs.
</item 39>

<item 40>
Fantasy Football RB Rankings Week 12: Who to start, best sleepers at running back (Sporting News, 2023-11-21T21:33:00+00:00)
Power backs have enjoyed the most success against the Lions, as evidenced last week by Chicago's Roschon Johnson averaging 5.0 yards per carry and D'Onta
Foreman finding paydirt.

WEEK 12 DFS:

DK lineup | FD lineup | Best values | Best stacks

Who are the biggest potential RB busts in Week 12?

We don't quite have the intestinal fortitude to bench Derrick Henry (vs. Panthers), but we must absolutely temper our expectations with the dude at this point.
Over the past two weeks, he has tallied just 64 total yards on 23 touches (2.8 yards per touch). Carolina served as Tony Pollard's get-right game last week, so
fire King Henry up if you have him. If he falters against the Panthers, you better have a backup plan moving forward (picking up Tyjae Spears as insurance is a
good place to start).
</item 40>

<item 41>
NFL Fantasy Football Week 5 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-04T21:56:25+00:00)
Overall the targets have favored Mostert only because he had more of them in Week 3 against Denver (7 to 4).

GIANTS: Are about to get run over. Their 4.3-yard rushing average they've allowed is a shade worse than league average but they rate poorly in rushing
touchdowns allowed (five), rushes of 10-plus yards allowed (14) and missed tackles on runs (24, fourth-worst in football).

DOLPHINS: Have split the workload pretty evenly despite the snap shares morphing in Achane's favor. Last week Achane had 11 touches to Mostert's 10. The week
before Achane had 22 to Mostert's 20. The week before that ... Achane wasn't a thing and Mostert had 19 touches to Salvon Ahmed's six (Achane had none).
</item 41>

<item 42>
Colorado vs. UCLA live score, updates, highlights from Week 9 college football game (Sporting News, 2023-10-28T22:08:03+00:00)
Collin Schlee wasn't going to be kept out of the end zone 💪@UCLAFootball makes it 28-9 pic.twitter.com/MtTdXmvF6B — FOX College Football (@CFBONFOX) October 29,
2023

10:43 p.m. – Harden going the distance with 25-yards and a hurdle.

Colorado 9, UCLA 21

10:23 p.m. TOUCHDOWN – Too easy for TJ Harden. He caps off an 80-yard drive with a short run for a touchdown. UCLA extends its lead to 12 with 14 minutes on the
clock. Can the Buffaloes finally connect for a touchdown before its too late?
</item 42>

<item 43>
Fantasy Injury Updates Week 10: News on Justin Jefferson, Ja'Marr Chase, T.J. Hockenson & Curtis Samuel (Sporting News, 2023-11-11T21:17:00+00:00)
Jefferson (hamstring) is officially listed as "questionable" for Sunday's game against the Saints. Kickoff is slated for 1 p.m. ET, but fantasy owners won't
have to wait long for news on the star receiver. In order to be eligible to play, Jefferson will need to be activated by 4 p.m. ET on Saturday. Earlier in the
week head coach Kevin O'Connell said that it would be "a little aggressive" for Jefferson to get back on the field this week. However, the receiver's three
straight limited practices were enough for the team to at least weigh the decision, but thankfully, an early answer awaits.

WEEK 10 FANTASY ADVICE:

Sleepers | Busts | Projections | Start 'em, sit 'em

Jefferson's fantasy outlook is always through the roof, but it will be interesting to see if that changes with Josh Dobbs under center instead of Kirk Cousins.
</item 43>

<item 44>
Fantasy Football Week 13 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-11-30T22:10:58+00:00)
STARTS: Bijan Robinson, Garrett Wilson, Breece Hall, Drake London, Jets DST, Falcons DST

Bijan Robinson, Garrett Wilson, Breece Hall, Drake London, Jets DST, Falcons DST SITS: Kyle Pitts, Tyler Conklin, Tyler Allgeier, both QBs

Start 'em 10.0-7.00 6.99 - 5.01 5.00 - 0 Sit 'em

Pittsburgh's biggest change in offensive scheme last week involved Kenny Pickett throwing over the middle much more than before (Pat Freiermuth benefitted). But
they also ran the ball effectively at Cincinnati -- Najee Harris had one of his best games this year. Expect more of the same against the Cardinals this week.
Arizona has allowed five touchdowns to tight ends over its past five games but actually has been relatively efficient defending them otherwise.
</item 44>

<item 45>
2023 NFL Week 11 injury roundup: Rams WR Cooper Kupp, Seahawks RB Kenneth Walker III among stars to exit (CBSSports.com, 2023-11-19T23:51:49+00:00)
He was seen walking on the sidelines shortly afterward but ruled doubtful to return. Rookie Zach Charbonnet, who entered Week 11 with just over 200 rushing
yards on the year, replaced Walker as Seattle's top ball carrier. Following the game, Seahawks head coach Pete Carroll told reporters Walker suffered an
"oblique strain that was legit" and could miss time.

D'Onta Foreman (Bears) remained a headliner of Chicago's backfield against the Lions, even with Khalil Herbert back from injured reserve, but managed just 14
yards on the ground before suffering an ankle injury. He walked off with a limp and attempted to run on the sidelines but did not return to the field after the
Bears ruled him questionable.

Pick Six Newsletter Crafted By The Best NFL Experts Get the day's big stories + fun stuff you love like mock drafts, picks and power rankings.
</item 45>

<item 46>
Fantasy Football Week 13 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-11-30T22:10:58+00:00)
Losing Jonathan Taylor stings for the Colts but Zack Moss was unreal in his efficiency between Weeks 2 through 5 when Taylor was either out or barely played
(5.0 yards per carry is a generic example), and he's only been a little bit worse (4.4 yards per carry) since Taylor took a larger chunk of the workload
starting in Week 6. Meanwhile, Derrick Henry was back in a familiar spot of handling plenty of carries against an inferior opponent, scoring twice and tallying
76 rush yards. Anytime he's in a close game or a Titans win, he tends to do well. The Colts rank in the bottom-five in Fantasy points allowed per game this
season while the Titans are closer to league-average, but both are in the top-six in RB rush attempts per game against (at least 24.0 each) in their past four
games.
</item 46>

<item 47>
NFL Fantasy Football Week 5 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-04T21:56:25+00:00)
TOUCHDOWNS: Harris has zero. Somehow it gets worse: He has one carry inside the 15-yard line. The Steelers have run the second-fewest red-zone plays (18, one
more than New England) and second-fewest plays inside the 5-yard line (two, one more than New England). The Steelers have played four games! This is nuts.

RAVENS: Have yet to allow a rushing score to a running back this year. They've been otherwise solid against RBs, holding them to 4.0 yards per carry on the
season and roughly league-average in other metrics including missed tackles (15).
</item 47>

<item 48>
NFL Fantasy Football Week 6 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-12T21:23:16+00:00)
He played just 16% of the snaps to Zack Moss' 80%, and Moss handled nearly every third down, fourth down and snap inside the 10-yard line. Of note, Taylor
played just three second-half snaps.

FILM: Taylor looked as big and strong as I remember, and he wasn't shy about contact at all. I liked how he fought for yardage on pretty much every carry. He
did seem rusty when it came to processing his blocking and figuring out where to run -- it felt more like he ran up the backs of his linemen more than scan for
space to make cuts and elude defenders. It was a good first step.

FILM: Moss, meanwhile, had consistently great vision and used his combination of speed and power to crash through the Titans' front, especially when they
stacked the box against him on his 56-yard touchdown run.
</item 48>

<item 49>
2023 NFL Week 11 injury roundup: Rams WR Cooper Kupp, Seahawks RB Kenneth Walker III among stars to exit (CBSSports.com, 2023-11-19T23:51:49+00:00)
Smith did return, however, for the final drive of the game in the losing effort. Head coach Pete Carroll said that Smith will "make it back." When asked if that
means for Thursday's matchup against San Francisco, the coach said, "I would think so."

Sam Howell (Commanders) received medical attention on the sidelines after enduring several hard hits to the upper body on a touchdown run against the Giants. He
was quickly cleared of a potential concussion, however, and remained in the game. Veteran journeyman Jacoby Brissett is Washington's backup.

RB

Aaron Jones (Packers) suited up against the Chargers despite a lingering hamstring issue, only to manage just five carries before suffering a knee injury. The
former Pro Bowler needed help to leave the field and was then carted to the locker room, replaced by A.J. Dillon.
</item 49>

<item 50>
Fantasy Football RB PPR Rankings Week 14: Who to start, best sleepers at running back (Sporting News, 2023-12-07T22:09:00+00:00)
Buccaneers

54 Justice Hill, Ravens vs. Rams

55 Rico Dowdle, Cowboys vs. Eagles

56 Kenneth Gainwell, Eagles @ Cowboys

57 Khalil Herbert, Bears vs. Lions

58 Tank Bigsby, Jaguars @ Browns

59 Samaje Perine, Broncos @ Chargers

60 Chase Edmonds, Buccaneers @ Falcons

61 Boston Scott, Eagles @ Cowboys

62 Matt Breida, Giants vs. Packers

63 Jamaal Williams, Saints vs. Panthers
</item 50>

<item 51>
Fantasy Football Week 13 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-11-30T22:10:58+00:00)
In their past three games, they gave up 4.1 yards per rush to the Ravens (just 6.3% of their runs went 10-plus yards), 7.8 yards per rush to the Steelers thanks
to a long TD run by Jaylen Warren (23.8% of their runs went 10-plus yards) and 4.6 yards per rush to the Broncos (18.5% of their runs went 10-plus yards).
Playing more snaps because the Browns offense has been inept isn't the reason for it either because they struggled with the run in the first half against all
three teams (over 5.3 yards per carry allowed against each). It's silly to say the Rams passing game is struggling when Matthew Stafford just threw four
touchdowns, but Cooper Kupp looked like he played hurt last week and Puka Nacua didn't look that much better.
</item 51>

<item 52>
NFL Fantasy Football Week 6 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-12T21:23:16+00:00)
Maybe the Patriots will turn to another running back in hopes of finding anything that can help take pressure off of their depleted passing game. I can't
recommend Stevenson as a start until we see proof of him playing well -- a favorable road matchup alone isn't enough. I'd rather start Jerome Ford, Zack Moss
and D'Onta Foreman.

LAST WEEK: Was the first time Stevenson did not have 16 or more touches in a game. He had eight carries and no receptions. He also played a season-low 50% of
the snaps but did handle 10 of 16 snaps on third/fourth downs. Stevenson started and played the first two series, then Elliott started off the next two series,
but they rotated in and out for each other the whole game. Elliott continued to be the Patriots RB of choice in the two-minute offense to end the first half.

STEVENSON: Nothing has changed in terms of his efficiency.
</item 52>

<item 53>
Fantasy Football RB Rankings Week 14: Who to start, best sleepers at running back (Sporting News, 2023-12-07T22:07:00+00:00)
It wouldn't make much sense for a four-win team to rush back its franchise player.

Patriots dual-purpose back Rhamondre Stevenson (ankle) suffered an injury on a nasty hip-drop tackle in New England's Week 13 loss to the Chargers. Bill
Belichick and company already noted that 'Mondre would miss multiple weeks, so we're bumping Ezekiel Elliott way up against the Steelers.

D'Andre Swift took a hard hit and exited Philadelphia's game late against the 49ers, but it's tough to know if he's actually hurt or was just kept out because
of the lopsided score. If Swift can't go in Week 14 against Dallas, Kenneth Gainwell and Boston Scott will split touches, with Gainwell likely garnering flex
value.

Jonathan Taylor underwent thumb surgery last week and is expected to miss a few weeks. Tough for Taylor owners to ride the home stretch of the regular season
without JT, but at least they could get him back for the end of the fantasy playoffs.
</item 53>

<item 54>
Fantasy Football RB PPR Rankings Week 14: Who to start, best sleepers at running back (Sporting News, 2023-12-07T22:09:00+00:00)
Mostert has more PPR points than everyone but CMC, while Achane ranks just two spots behind CMC at third in PPR points per game (20.6).

Another stud duo, Detroit's Jahmyr Gibbs and David Montgomery (@ Bears), also makes our top 10. They ravaged Chicago when they last met a few weeks ago,
combining for a total of 193 yards and two TDs, and we have no reason to think the Lions will slow down in Week 13. Dan Campbell's squad just put 142 rushing
yards and 33 points on the Saints in the Bayou this past weekend.

Speaking of New Orleans, old PPR reliable Alvin Kamara (vs. Panthers) has been a source of light for an otherwise-dismal Saints offense. With Derek Carr
suffering a brutal blow in the Week 13 loss to Detroit, Kamara will likely shoulder the load against Carolina and its atrocious run defense.
</item 54>

<item 55>
Fantasy Football RB Rankings Week 12: Who to start, best sleepers at running back (Sporting News, 2023-11-21T21:33:00+00:00)
Bears

43 Dameon Pierce, Texans vs. Jaguars

44 Royce Freeman, Rams @ Cardinals

45 Jeff Wilson Jr., Dolphins @ Jets

46 Rico Dowdle, Cowboys vs. Commanders

47 Antonio Gibson, Commanders @ Cowboys

48 Miles Sanders, Panthers @ Titans

49 Tyjae Spears, Titans vs. Panthers

50 Clyde Edwards-Helaire, Chiefs @ Raiders

51 Dalvin Cook, Jets vs. Dolphins

52 Zack Moss, Colts vs. Buccaneers

53 Kenneth Gainwell, Eagles vs. Bills

54 Tony Jones Jr., Cardinals vs. Rams

55 Elijah Mitchell, 49ers @ Seahawks

56 Jaleel McLaughlin, Broncos vs. Browns

57 Justice Hill, Ravens @ Chargers

58 Chase Edmonds, Buccaneers @ Colts

59 Boston Scott, Eagles vs.
</item 55>

<item 56>
Fantasy Football RB PPR Rankings Week 14: Who to start, best sleepers at running back (Sporting News, 2023-12-07T22:09:00+00:00)
WEEK 14 STANDARD RANKINGS:

QBs | RBs | WRs | TEs | D/STs | Kickers

Who are the best fantasy RBs in Week 14?

Christian McCaffrey (vs. Seahawks in Week 14) has given us no reason to stop believing in him as the best overall player in the NFL, so he stays atop this list.
Austin Ekeler (vs. Broncos) has started to give us some pause as of late, as he's averaged fewer than 3.0 yards per carry since coming off IR eight games ago,
but a juicy home matchup against Denver should be his get-right game. Clear skies ahead, Ek owners — the Broncos have surrendered the most PPR points in the NFL
to RBs, and L.A. faces them twice between now and the fantasy title week.

With Jacksonville playing very well over the past month, we're all-in on Travis Etienne Jr. (@ Browns) even in a tough matchup.
</item 56>

<item 57>
Fantasy Football RB Rankings Week 14: Who to start, best sleepers at running back (Sporting News, 2023-12-07T22:07:00+00:00)
Lions

41 Devin Singletary, Texans @ Jets

42 Dalvin Cook, Jets vs. Texans

43 Zach Charbonnet, Seahawks @ 49ers

44 Chase Brown, Bengals vs. Colts

45 D'Ernest Johnson, Jaguars @ Browns

46 Miles Sanders, Panthers @ Saints

47 Joshua Kelley, Chargers vs. Broncos

48 Royce Freeman, Rams @ Ravens

49 Jaleel McLaughlin, Broncos @ Chargers

50 Jeff Wilson Jr., Dolphins vs. Titans

51 Ty Chandler, Vikings @ Raiders

52 Justice Hill, Ravens vs. Rams

53 Khalil Herbert, Bears vs. Lions

54 Latavius Murray, Bills @ Chiefs

55 Cordarrelle Patterson, Falcons vs. Buccaneers

56 Kenneth Gainwell, Eagles @ Cowboys

57 Tank Bigsby, Jaguars @ Browns

58 Rico Dowdle, Cowboys vs.
</item 57>

<item 58>
Fantasy Football RB Rankings Week 14: Who to start, best sleepers at running back (Sporting News, 2023-12-07T22:07:00+00:00)
Texans), and we think they will be rewarded for their patience. Houston has surrendered 111.8 scrimmage yards per game to RBs this season, and DeMeco Ryans'
unit has also allowed 14 total rushing TDs on the year. We like Hall's upside even more than we like stud dual-purpose backs Alvin Kamara (vs. Panthers) and
Saquon Barkley (vs. Packers), but just barely. Kamara and Barkley both have strong matchups, with Carolina and Green Bay both sitting in the bottom 10 in the
NFL in standard fantasy points allowed to RBs.

WEEK 14 FANTASY ADVICE:

Sleepers | Busts | Projections | Start-sit

Who are the best fantasy RB sleepers, waiver pickups in Week 14?
</item 58>

<item 59>
MVPs for all 16 AFC teams at the midpoint (Yardbarker, 2023-11-09T21:51:02+00:00)
He appears fully recovered now, though, as the Bengals (5-3) have won four straight. During their win streak, Burrow has thrown for 1,133 yards, 10 touchdowns
and just two interceptions.

PITTSBURGH STEELERS | Linebacker T.J. Watt: The Steelers are the first team in NFL history to be outgained by their opponents in each of their first eight games
and have a winning record (5-3). It hasn’t been pretty to watch, but Pittsburgh always makes plays when needed. The player who seems to make the most is Watt.
The 2021 NFL Defensive Player of the Year is tied for second-most sacks (9.5) in the league. He also has two forced fumbles, three fumble recoveries and one
interception. — Aaron Becker

AFC South

HOUSTON TEXANS | QB C.J. Stroud: Stroud is having one of the best statistical rookie seasons of all time.
</item 59>

<item 60>
Fantasy Football Week 13 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-11-30T22:10:58+00:00)
With Pickett nursing an ankle injury we could see the Steelers run the ball aplenty.

STARTS: Najee Harris, Kyler Murray, Trey McBride, Marquise Brown, Pat Freiermuth, Diontae Johnson (PPR flex), Jaylen Warren (PPR only), Steelers DST

Najee Harris, Kyler Murray, Trey McBride, Marquise Brown, Pat Freiermuth, Diontae Johnson (PPR flex), Jaylen Warren (PPR only), Steelers DST SITS: Kenny
Pickett, James Conner, Rondale Moore, Cardinals DST

Kenny Pickett, James Conner, Rondale Moore, Cardinals DST SLEEPERS: Greg Dortch (PPR only), George Pickens

Sit Him (Lineup Decisions) Projections powered by Sportsline James Conner RB ARI Arizona • #6

Age: 28 • Experience: 7 yrs.
</item 60>

<item 61>
Fantasy Football RB Rankings Week 12: Who to start, best sleepers at running back (Sporting News, 2023-11-21T21:33:00+00:00)
They weren't awful against Dallas last week on a per-touch basis, but neither finished with more than 12 touches. Even worse, no Carolina back has enjoyed a
60-yard rushing game since Oct. 15. Bad offensive line, negative game scripts, awful coaching — it's all bad, man!

What are the biggest RB injuries in Week 12?

Second-year Seahawks back Kenneth Walker III suffered an oblique injury in Week 11 that the ever-hip Pete Carroll referred to as "legit." KW3 has a better
chance of missing multiple weeks than he does at suiting up for Seattle's Thanksgiving Day game against Fred Warner and the 49ers. If Zach Charbonnet somehow
finds himself on your waiver wire, an early Merry Christmas to you if you have the FAAB or a high waiver priority.

The hits just keep on coming for the Packers' Aaron Jones, who suffered a knee injury but seems to have at least caught a break in the sense that he didn't tear
his ACL.
</item 61>

<item 62>
Colorado vs. UCLA live score, updates, highlights from Week 9 college football game (Sporting News, 2023-10-28T22:08:03+00:00)
It didn't help that the Colorado running game was absent in the second half. Sanders handed the ball off only four times through the final two quarters, with
the first one coming halfway into the final frame; the Buffaloes scored their only touchdown on that drive. They finished with 38 rushing yards, the lowest in a
game this season.

Safety Shilo Sanders missed much of the game after being ejected late in the first half for targeting, but Travis Hunter shined in his second game back on the
field after injury. He recorded the game's two picks.

Colorado's quest for bowl eligibility continues after dropping to 4-4. Two of the Buffs' final four opponents -- Oregon State and Utah, are ranked.

MORE: Watch Colorado vs. UCLA live with Fubo (free trial)

The Sporting News tracked highlights and updates from the Colorado-UCLA game.

Colorado vs.
</item 62>

<item 63>
Fantasy Football RB Rankings Week 14: Who to start, best sleepers at running back (Sporting News, 2023-12-07T22:07:00+00:00)
Eagles

59 Samaje Perine, Broncos @ Chargers

60 Matt Breida, Giants vs. Packers

61 Chase Edmonds, Buccaneers @ Falcons

62 Boston Scott, Eagles @ Cowboys

63 Jamaal Williams, Saints vs. Panthers
</item 63>

<item 64>
Fantasy Football RB Rankings Week 12: Who to start, best sleepers at running back (Sporting News, 2023-11-21T21:33:00+00:00)
Bills

60 Tank Bigsby, Jaguars @ Texans

61 Jamaal Williams, Saints @ Falcons

62 Samaje Perine, Broncos vs. Browns

63 Matt Breida, Giants vs. Patriots

Week 12 RB fantasy projections, rankings from Draft Sharks

Draft Sharks projections for the new week will update every Tuesday and be constantly adjusted throughout the week

We're proud to offer SN Fantasy readers an exclusive look at Draft Sharks’ award-winning projections and rankings every week. To see Draft Sharks' Week 12 3D
floor-to-ceiling projections for running backs -- plus get access to all of Draft Sharks' industry-leading tools -- become a Draft Sharks' member today at the
link below!

TRY DRAFT SHARKS FREE: Dominate with exclusive tools, including Free-Agent Finder & Trade Navigator!
</item 64>

<item 65>
Jets vs. Raiders live score, updates, highlights from NFL 'Sunday Night Football' game (Sporting News, 2023-11-12T23:30:03+00:00)
Jimmy Garoppolo. Instead, Wilson went 23-for-39 with 263 yards, no touchdowns and a pick (rushing four times for 54 yards) while Aidan O'Connell completed
16-of-27 passes for 153 yards, a touchdown and an interception.

The Raiders largely turned to running back Josh Jacobs, who set season-best marks with 27 carries for 116 yards, while the Jets' running back duo of Breece Hall
and Dalvin Cook was held to just 17 rushes for 54 yards.

The win for the Raiders already marks their second in as many games since the firing of Josh McDaniels, with interim coach Antonio Pierce getting off to an
undefeated 2-0 start to his career. It also vaults the Raiders into a deep AFC playoff race, with Las Vegas joining the now five AFC wild-card contenders with
at least five wins.
</item 65>

<item 66>
No. 17 Arizona dominates No. 22 Utah, 42-18 (Rivals, 2023-11-18T23:45:32+00:00)
"I don't think the rain effected us at all," Fisch said.

With the rain starting to clear up to start the second half, overtime it was clear that the Wildcats’ defense was holding a cloud over the Utes’ heads. Arizona
got off to a slow start but picked up much of its thunderous momentum after junior safety Gunner Maldonado broke up a pass on fourth and three. Since the
fourth-down stop, the UA logged an interception and another turnover on downs.

Although the defense was surging, the Wildcats not scoring since the first drive of the second quarter — their offense seemed to have lost its touch.

But then “Swag Surfin” by Fast Life Yungstaz came on, signaling the fourth quarter, the quarter in which going into the game, Arizona was outscoring opponents
90-37. The Wildcats on their first drive put a touchdown on the board a seven play, 61 yard drive, thanks to a 25-yard dash before diving into the end zone by
running back Jonah Coleman.
</item 66>

<item 67>
NFL Fantasy Football Week 5 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-04T21:56:25+00:00)
Since Week 2 their YAC/reception has been 5.94, sixth best in that span with zero explosive pass plays allowed to RBs. So they're pretty good.

Start 'em 10.0-7.00 6.99 - 5.01 5.00 - 0 Sit 'em

Bust Candidate (Lineup Decisions) Projections powered by Sportsline Raheem Mostert RB MIA Miami • #31

Age: 31 • Experience: 9 yrs.
</item 67>

<item 68>
NFL Fantasy Football Week 6 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-12T21:23:16+00:00)
That one went for 32 yards against the Eagles in Week 1. Since then his longest reception has been for seven yards.

PATRIOTS: Have run one (1) play inside the opponents' five-yard line this entire season. That should help explain the lack of touchdowns for Stevenson.

RAIDERS: Rank in the bottom 10 versus RBs in yards per carry allowed (4.3) and yards after contact per rush allowed (3.04). But they're also in the top-10 in
most rush attempts by RBs faced (24.4 per game) and they've given up just three rushing touchdowns and an explosive run on 4.9% of their snaps (that's actually
10th best). And would you believe only one running back has scored more than 15 PPR points against the Raiders this season? It's James Cook.
</item 68>

<item 69>
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
</item 69>

<item 70>
Fantasy Football RB Rankings Week 12: Who to start, best sleepers at running back (Sporting News, 2023-11-21T21:33:00+00:00)
Let's get right to our Week 12 RB rankings so you can fully prepare for Thanksgiving week and put out a lineup with backs who will feast like the rest of us.

MORE: Watch NFL RedZone live with Sling (50% off first month)

Which RBs are on bye in Week 12?

None! It's a Thanksgiving miracle! Not only do we not have any byes to worry about this week, but we also have NFL action on four different days (Thursday,
Friday, Sunday, Monday).

WEEK 12 STANDARD RANKINGS:

QBs | WRs | TEs | D/STs | Kickers

In some ways, however, a bye-free week at this stage of the season presents more challenges than ever. Whereas byes make some start 'em, sit 'em decisions
no-brainers, 16-game weeks make it a lot easier to get lineup decisions wrong.
</item 70>

<item 71>
Fantasy Football RB Rankings Week 14: Who to start, best sleepers at running back (Sporting News, 2023-12-07T22:07:00+00:00)
It's make-or-break time in fantasy land. Only one week stands between now and the fantasy football playoffs, so every decision we make could be the difference
between an optimal finish and a womp-womp ending. Our Week 14 fantasy RB rankings for standard leagues will help steer your fantasy ship to the smoothest
possible sailing, guiding you to the running back studs, sleepers, busts, and injuries of note going into the most important week of the regular season.

The rollercoaster that the 2023 season has been rolled on through Week 13, with yet another fresh batch of injuries. Luckily for us, we have strong stomachs
(and we're not above taking offers from potential sponsors, TUMS and Kaopectate!). So, like always, we will navigate through the injuries to provide you with
the most up-to-date information and analysis.

Let's get to our Week 14 RB rankings and hopefully help you close out the regular season in style.
</item 71>

<item 72>
NFL Fantasy Football Week 6 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-12T21:23:16+00:00)
McDANIEL: Prefers to use two running backs from game to game, and in Miami's past three games there's been a relatively close split in carries between two backs
(18 and 13 in Week 3, 8 and 7 in Week 4, 11 and 10 in Week 5).

LAST YEAR: Wilson was efficient in his eight games in Miami, averaging 4.7 yards per carry with three touchdowns and just over 60 total yards per matchup. That
includes two games he barely played in (under 20 snaps) -- take those out and he averaged 78.2 total yards per game.

SALVON AHMED? Was a healthy scratch last week and figures to be no better than a special-teams player this week. In fact, in the seven games Raheem Mostert,
Wilson and Ahmed were all active for with the Dolphins last year, Ahmed played 18 offensive snaps and 25 special-teams snaps.
</item 72>

<item 73>
Fantasy Football RB Rankings Week 14: Who to start, best sleepers at running back (Sporting News, 2023-12-07T22:07:00+00:00)
Could this be the weekend that Keaton Mitchell (vs. Rams) gets back into RB2 range for the Ravens? Los Angeles might find itself spying Lamar Jackson and Gus
Edwards so often that the speedy rookie busts a couple of big runs. Mitchell has racked up 330 yards and two TDs since he broke out four games ago, but the
majority of that damage (134 yards and a TD) came in the initial breakout against Seattle. He has still yet to see 12 touches in a game, something we can't
imagine will be the case by the end of the regular season.

We also think Roschon Johnson (vs. Lions) could have a chance to take over lead-back duties in Chicago, as both Khalil Herbert and D'Onta Foreman have battled
injuries and inconsistent touches this season.
</item 73>

<item 74>
NFL Fantasy Football Week 5 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-04T21:56:25+00:00)
Weekly Breakdown MATCHUP vs NYG MIA -11 O/U 48.5 OPP VS RB 23rd PROJ PTS 16 RB RNK 17th YTD Stats RUYDS 249 REC 13 REYDS 115 TD 7 FPTS/G 22.4 If you're starting
Mostert, you're hoping he hasn't fumbled away his half of the touches he's had over the past two weeks AND can have the chance at scoring. Miami is capable of
giving Mostert more short-yardage work out of the blue, but their actions over the past two weeks suggest Achane is their preferred back near the goal line.
Understand the risks in starting Mostert as a No. 2 RB, who could be dog-house bound with another fumble. He's behind Isiah Pacheco and Breece Hall in my
rankings but ahead of Jahmyr Gibbs, Dameon Pierce and Najee Harris.
</item 74>

<item 75>
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
</item 75>

<item 76>
Fantasy Football RB Rankings Week 14: Who to start, best sleepers at running back (Sporting News, 2023-12-07T22:07:00+00:00)
Christian McCaffrey (vs. Seahawks in Week 14) retains his spot at No. 1, especially considering he's playing at home against a Seattle team he just roughed up
for 139 yards and two TDs on Thanksgiving. We're also sticking with Austin Ekeler (vs. Broncos) even though he has managed just 2.99 yards per carry since
returning from injury just eight games ago. If Denver's abysmal run D doesn't cure what ails him, Ek's getting demoted for the fantasy playoffs.

Travis Etienne Jr. (@ Browns) draws a tough matchup on paper, but the Rams' Kyren Williams (@ Ravens) might beg to differ. Williams put 112 total yards and a TD
on Cleveland in Week 13, and with Etienne's usage rate and the Jags hitting their stride offensively as of late, we're confident in the third-year back's
ranking at No. 3.
</item 76>

<item 77>
NFL Fantasy Football Week 5 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-04T21:56:25+00:00)
But just three players have averaged over 4.0 yards per carry: Joe Mixon (4.5 yards per run, didn't have 100 total yards, didn't score), Zack Moss (4.1 yards
per run, ran for 122 yards on 30 carries and caught a touchdown) and Pierre Strong (9.8 yards per carry, most of it coming in garbage time last week).

HISTORY: Harris had over 120 total yards and a touchdown in a Week 17 win in Baltimore last year. He also scored on them with exactly 50 total yards against
them in at home in Week 14. He has posted at least 10 non-PPR points in 3 of 4 career games.
</item 77>

<item 78>
Fantasy Football RB Rankings Week 14: Who to start, best sleepers at running back (Sporting News, 2023-12-07T22:07:00+00:00)
Patriots

22 Jerome Ford, Browns vs. Jaguars

23 Alexander Mattison, Vikings @ Raiders

24 Najee Harris, Steelers vs. Patriots

25 Ezekiel Elliott, Patriots @ Steelers

26 Gus Edwards, Ravens vs. Rams

27 James Cook, Bills @ Chiefs

28 Zack Moss, Colts @ Bengals

29 Keaton Mitchell, Ravens vs. Rams

30 AJ Dillon, Packers @ Giants

31 Dameon Pierce, Texans @ Jets

32 Aaron Jones, Packers @ Giants

33 Clyde Edwards-Helaire, Chiefs vs. BIlls

34 Jerick McKinnon, Chiefs vs. Bills

35 Roschon Johnson, Bears vs. Lions

36 Tyjae Spears, Titans @ Dolphins

37 Kareem Hunt, Browns vs. Jaguars

38 Tyler Allgeier, Falcons vs. Buccaneers

39 Chuba Hubbard, Panthers @ Saints

40 D'Onta Foreman, Bears vs.
</item 78>

<item 79>
NFL Fantasy Football Week 6 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-12T21:23:16+00:00)
The only rush defense metrics they're not top-five in are Defensive Rush EPA (just below league average), yards before contact per rush (0.88 yards, they're
sixth), yards after contact per rush (2.80, just above league average). They even lead the NFL in fewest missed tackles. Tony Pollard had an 8-29-0 rushing stat
line against them last week.

WHAT ABOUT THROUGH THE AIR? Even though the 49ers see the fifth-most RB targets per game, they're better than league average in catch rate allowed (75%) and
top-10 in both yards per catch (5.7) and YAC/reception (6.10) allowed to running backs. Pollard caught 4 of 5 passes for 35 yards last week.

P.J. WALKER: Has thrown 17.2% of his career pass attempts to running backs.
</item 79>

<item 80>
Fantasy Football RB Rankings Week 12: Who to start, best sleepers at running back (Sporting News, 2023-11-21T21:33:00+00:00)
The Pats have surrendered 116 scrimmage yards per game and nine total TDs to RBs this season, so we like Barkley to pick up where he left off this Sunday (drive
up that free-agency value, 'Quon!).

You don't see running mates cracking the top five very often, but we had to make the exception for David Montgomery and Jahmyr Gibbs (vs. Packers). Entering
Week 11, Green Bay had surrendered nearly 140 all-purpose yards per game to RBs. Between Detroit's o-line, the Lions' high-flying offensive attack, and the
ruthless 1-2 punch DMont and Gibbs have been this season, the feast could be on very early for Dan Campbell's guys.

Take a collective sigh of relief, Tony Pollard (vs. Commanders) owners! TP looked solid against Carolina last week, averaging 5.1 yards per carry and finally
getting into the end zone for the first time since Week 1.
</item 80>

<item 81>
Fantasy Football Week 13 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-11-30T22:10:58+00:00)
Can't say the same against the run as the Redbirds rank in the bottom five in pretty much every single run defense metric except yards after contact per rush
(where they're league average). A running back has scored 17 or more PPR points in four of their past five against the Cardinals, and eight in 12 games have hit
at least 15 PPR points. With Pickett nursing an ankle injury we could see the Steelers run the ball aplenty.
</item 81>

<item 82>
NFL Fantasy Football Week 5 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-04T21:56:25+00:00)
He's still a decent start ahead of Najee Harris, Khalil Herbert and Jahmyr Gibbs but I'd rather chance my lineup with someone who has upside like Jaleel
McLaughlin (if Javonte Williams sits out) or Alexander Mattison.

FILM: The numbers don't quite match the talent -- Stevenson is a capable runner with great vision, balance and power to go with good speed. He struggled on
lateral runs last week and his line failed him plenty on north-south runs. Game script kept him from finding a lot of work last week (one fourth-quarter touch).

THIS SEASON: Stevenson has at least 16 touches in every game, which is good. Unfortunately he has yet to total 100 yards and has been under 70 yards three
times, which is definitely the opposite of good.
</item 82>

<item 83>
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
</item 83>

<item 84>
Fantasy Football RB Rankings Week 12: Who to start, best sleepers at running back (Sporting News, 2023-11-21T21:33:00+00:00)
could be back in play as a handcuff stash.

Bears veteran D'Onta Foreman injured his ankle in Week 11. He could miss some time now that Khalil Herbert and Roschon Johnson are both back in action for
Chicago. The best runner on the Bears, however, is QB Justin Fields when he's active.

Texans back Dameon Pierce returned to practice late last week, but he remained out with his ankle injury against the Cardinals. We could see him return to the
field this weekend, but for now, we're keeping him off our rankings until we hear more positive news. Either way, it seems Devin Singletary is the new RB1 in
this breakout offense.

We'll be updating these standard RB rankings all week, so check back for the latest player movement.
</item 84>

<item 85>
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
</item 85>

<item 86>
NFL power rankings: Lions, 49ers level up; Texans jump Bengals, Bills; Jets, Patriots sink to bottom for Week 11 (Sporting News, 2023-11-13T21:54:00+00:00)
Their defense should be getting a little better to support their often explosive offense. They should feel good about a Week 11 rebound vs. the Raiders.

7. Pittsburgh Steelers 6-3 (9)

The Steelers are playing the way Mike Tomlin wants, with the running game rolling to big totals with Jaylen Warren and Najee Harris and the defense doing what
it needs, more so vs. pass than run, in key moments. You can't argue with the overachieving results.

8. Cleveland Browns 6-3 (10)

The Browns proved they can win games with defense in another way, with big plays. They also have more offensive pop when Deshaun Watson plays well and the
running game doesn't slump. Their playoff hopes look much better after their stunning comeback vs. the Ravens.

9.
</item 86>

<item 87>
Fantasy Football RB PPR Rankings Week 14: Who to start, best sleepers at running back (Sporting News, 2023-12-07T22:09:00+00:00)
Happy December, fantasy friends! We have somehow reached the final week of the fantasy football regular season, so the running back decisions we make in our PPR
leagues this week could very well impact our playoff fate. Our Week 14 fantasy RB PPR rankings will help you optimize your lineup at RB1, RB2, and flex heading
into this crucial weekend.

It hasn't been all holly, jolly this season — in fact, it has often been a bumpy sleigh ride. We've suffered a plethora of injuries to stud RBs the past few
weeks, but we keep rolling with the punches and always do our best to stay two steps ahead of our opponents. One major positive is that only two teams will be
on bye this week, a welcome relief after the six-team byepocalypse that was Week 13. Still, the injury-ravaged teams will have trouble replacing their studs and
the deep squads will have difficult start 'em, sit 'em decisions to make.
</item 87>

<item 88>
NFL Fantasy Football Week 5 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-04T21:56:25+00:00)
At least all of them went to his wideouts. All were in the red zone, none were on deep throws and none were on WR screens.

Sit Him (Lineup Decisions) Projections powered by Sportsline Najee Harris RB PIT Pittsburgh • #22

Age: 25 • Experience: 3 yrs. Weekly Breakdown MATCHUP vs BAL BAL -4 O/U 38 OPP VS RB 12th PROJ PTS 9.3 RB RNK 30th YTD Stats RUYDS 210 REC 4 REYDS 34 TD 0
FPTS/G 7.1 The Steelers' lack of scoring opportunities really crushes the upside for Harris, who admittedly has improved this season and could be on his way to
being a reliable RB2. The Ravens have been pretty good against the run but they can give up some numbers. Harris is a volume-based back who is at best a low-end
starting option given everything that's going on.
</item 88>

<item 89>
Fantasy Football RB Rankings Week 14: Who to start, best sleepers at running back (Sporting News, 2023-12-07T22:07:00+00:00)
We also don't love Jerome Ford or Kareem Hunt (vs. Jaguars), as Jacksonville entered Week 12 allowing just 12.3 standard fantasy points per game to RBs (third
lowest in the NFL).

We say it almost every week, but the song remains the same: don't start Carolina backs Chuba Hubbard or Miles Sanders (@ Saints). It's even worse than usual
this week, as New Orleans has a pretty hard-hitting front-seven and the Bayou is no place for below-average offenses.

What are the biggest RB injuries in Week 14?

Two-time rushing champ Derrick Henry (head) suffered a huge blow to the head and neck area in Tennessee's Week 13 loss to the Colts, so don't be surprised if
the king gives up his throne to Tyjae Spears this week. He immediately left the game and proceeded to the locker room following the hit, and it was confirmed
that he would not return with a head injury.
</item 89>

<item 90>
Fantasy Football RB Rankings Week 14: Who to start, best sleepers at running back (Sporting News, 2023-12-07T22:07:00+00:00)
3.

We have two sets of teammates in our top 10 this week: Detroit's David Montgomery and Jahmyr Gibbs (@ Bears) and Miami's Raheem Mostert and De'Von Achane (vs.
Titans). Both duos have such solid workloads in high-flying, run-first offenses — and such juicy matchups — that we can't imagine any of these studs finishing
outside the elite eight. DMont and Gibbs combined for 193 yards and two TDs when they faced Chicago in Week 11. Tennessee just surrendered 31 points to Indy, so
we know Miami will be dominating the time of possession and running early and often against Will Levis and company.

Jets fans will be hoping for some Breece Lightning from stud second-year back Breece Hall (vs. Texans), and we think they will be rewarded for their patience.
</item 90>

<item 91>
MVPs for all 16 AFC teams at the midpoint (Yardbarker, 2023-11-09T21:51:02+00:00)
JACKSONVILLE JAGUARS | RB Travis Etienne: QB Trevor Lawrence hasn’t quite taken the step forward everyone thought he would, but fortunately for the Jags,
Etienne is emerging as one of the top RBs in the league. He has rushed for multiple touchdowns in three of his past four games and scored a touchdown in five of
the team’s six wins. Etienne ranks third the NFL in rushing touchdowns (seven) and sixth in rushing yards (583).

TENNESSEE TITANS | DT Jeffery Simmons: There haven’t been too many bright spots for the Titans, but Simmons continues to be a wrecking ball on the defensive
line. He leads the team in tackles for loss (six) and ranks second in sacks (4.5). If not for Simmons, Tennessee’s defense would rank a heck of a lot lower than
20th in yards allowed per game (337.8). — Michael Gallagher
</item 91>

<item 92>
Patriots vs. Steelers FanDuel Picks: NFL DFS lineup advice for Week 14 TNF single-game tournaments (Sporting News, 2023-12-07T21:26:00+00:00)
MORE SUPERDRAFT DFS: How to play SuperDraft

Champion (+50% points): RB Najee Harris, Steelers (Multiplier 1.9x)

Co-Champ (+25% points): RB Ezekiel Elliott, Patriots (Multiplier 1.4x)

Super FLEX: QB Mitch Trubisky, Steelers (Multiplier 1.15x)

Super FLEX: TE Pat Freiermuth, Steelers (Multiplier 1.35x)

Super FLEX: K Chris Boswell, Steelers (Multiplier 1.8x)

Super FLEX: K Chad Ryland, Patriots (Multiplier 1.8x)

This week's SuperDraft lineup features our first "Co-Champ," which gives us 25 percent more points for our player of choice. With the ground game being a point
of emphasis, we're focused on the RBs for those two top spots, while filling in the lineup with both kickers alongside a Trubisky-Freiermuth stack.
</item 92>

<item 93>
Fantasy Football Week 13 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-11-30T22:10:58+00:00)
If that doesn't make you feel good about Conner, then you'll especially dislike that he's averaged 0.42 yards before contact on his runs since Murray's return,
good for 44th out of 47 running backs (Breece Hall and Derrick Henry had it worse). Pittsburgh's unit held the Bengals and Browns in consecutive weeks to 0.5
yards before contact or less. If you start Conner, don't expect an efficient day and pray to Ye Olde Fantasy Gods for a garbage-time touchdown. I'd rather trust
Jaylen Warren, Chuba Hubbard and A.J. Dillon.

Start 'em 10.0-7.00 6.99 - 5.01 5.00 - 0 Sit 'em

Expect plenty of rushing.
</item 93>

<item 94>
Fantasy Football RB PPR Rankings Week 14: Who to start, best sleepers at running back (Sporting News, 2023-12-07T22:09:00+00:00)
Lions

36 Clyde Edwards-Helaire, Chiefs @ Bills

37 Kareem Hunt, Browns vs. Jaguars

38 Tyler Allgeier, Falcons vs. Buccaneers

39 Chuba Hubbard, Panthers @ Saints

40 Devin Singletary, Texans @ Jets

41 D'Onta Foreman, Bears vs. Lions

42 Chase Brown, Bengals vs. Colts

43 D'Ernest Johnson, Jaguars @ Browns

44 Jaleel McLaughlin, Broncos @ Chargers

45 Miles Sanders, Panthers @ Saints

46 Zach Charbonnet, Seahawks @ 49ers

47 Dalvin Cook, Jets vs. Texans

48 Joshua Kelley, Chargers vs. Broncos

49 Royce Freeman, Rams @ Ravens

50 Jeff Wilson Jr., Dolphins vs. Titans

51 Ty Chandler, Vikings @ Raiders

52 Latavius Murray, Bills @ Chiefs

53 Cordarrelle Patterson, Falcons vs.
</item 94>

<item 95>
NFL Fantasy Football Week 6 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-12T21:23:16+00:00)
STEVENSON: Nothing has changed in terms of his efficiency. We can blame the offensive line to some degree but Stevenson has averaged 0.97 yards before contact
per rush this season. That ranks 30th out of 41 qualifying running backs. And if you think that's bad then consider his yards after contact per rush average of
1.79, which is dead last among those same 41 qualifiers.

EXPLOSIVE RUNS: Stevenson has one (1) explosive carry of 12 or more yards. It was against Miami. It went 12 yards. This is a big change from 2022 when he had 21
explosive runs and an explosive run rate of 10%, which was eighth-best.

RECEIVING: Is averaging 3.6 targets per game and catching 67% of them but is getting just 7.3 yards per catch and has one (1) explosive catch of 16 or more
yards.
</item 95>

<item 96>
Fantasy Football RB PPR Rankings Week 14: Who to start, best sleepers at running back (Sporting News, 2023-12-07T22:09:00+00:00)
Tampa Bay has a pretty good run defense, so Allgeier can't be trusted outside of a desperate flex play with his nine average touches over the past two weeks.

What are the biggest RB injuries in Week 14?

Derrick Henry (head/neck), Rhamondre Stevenson (ankle), Kenneth Walker III (oblique), Aaron Jones (knee), D'Onta Foreman (ankle), and D'Andre Swift
(undisclosed) are all "questionable," at best, to suit up this week. We know for sure Stevenson will be out, but we'll have to monitor the injury reports
throughout the week for the rest.

Jonathan Taylor (thumb) underwent surgery last week and will miss the next few weeks, opening the door for Zack Moss to resume his role as Indy's super-sub lead
back.

We'll be updating these RB PPR rankings all week, so check back for the latest player movement.
</item 96>

<item 97>
Fantasy Football RB Rankings Week 12: Who to start, best sleepers at running back (Sporting News, 2023-11-21T21:33:00+00:00)
All three of these backs have been mostly solid, and none have much to worry about in terms of backfield competition.

WEEK 12 FANTASY ADVICE:

Sleepers | Busts | Projections | Start-sit

Who are the best fantasy RB sleepers, waiver pickups in Week 12?

Sleep on Rhamondre Stevenson (@ Giants) at your own peril. 'Mondre has accumulated 231 yards from scrimmage and a TD over New England's past two games, and the
Pats are coming off bye. Washington just carved the G-men up like a turkey on the ground last week, surrendering a whopping 174 yards and a TD. Gobble gobble.

Get Kyren Williams (@ Cardinals) right back into your lineup the moment Los Angeles activates him from IR. Arizona just made Devin Singletary look like Barry
Sanders, so Williams (ankle) should enjoy a triumphant return to the field. Keep rolling with Alexander Mattison (vs.
</item 97>

<item 98>
Fantasy Football RB PPR Rankings Week 14: Who to start, best sleepers at running back (Sporting News, 2023-12-07T22:09:00+00:00)
Fantasy RB PPR Rankings Week 14: Who to start, sit at running back

Rankings based on full-point PPR scoring formats

Rank Player

1 Christian McCaffrey, 49ers vs. Seahawks

2 Austin Ekeler, Chargers vs. Broncos

3 Travis Etienne, Jaguars @ Browns

4 De'Von Achane, Dolphins vs. Titans

5 Raheem Mostert, Dolphins vs. Titans

6 Jahmyr Gibbs, Lions @ Bears

7 Alvin Kamara, Saints vs. Panthers

8 Saquon Barkley, Giants vs. Packers

9 David Montgomery, Lions @ Bears

10 Breece Hall, Jets vs. Texans

11 Bijan Robinson, Falcons vs. Buccaneers

12 Josh Jacobs, Raiders vs. Vikings

13 Rachaad White, Buccaneers @ Falcons

14 Tony Pollard, Cowboys vs.
</item 98>

<item 99>
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
</item 99>

<item 100>
Fantasy Football Week 13 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-11-30T22:10:58+00:00)
Weekly Breakdown MATCHUP @ PIT PIT -5.5 O/U 41 OPP VS RB 8th PROJ PTS 10.1 RB RNK 27th YTD Stats RUYDS 526 REC 14 REYDS 36 TD 2 FPTS/G 10.3 Kyler Murray's
return has changed the math on Conner's expectations. In three games, Arizona has become pass-heavy (63.7% pass rate) -- a stark contrast from Conner's five
games before injury (55.6% pass rate). And while Conner has had some red-zone carries in those games with Murray, his O-line hasn't kept him clean, especially
near the goal line. Meanwhile Murray has a rushing touchdown in three straight games. Conner will face his former team, one that has held all but four running
backs to less than 15 PPR points this year, and the ones who notched 15-plus scored a touchdown.
</item 100>
=== END ITEM id=8b7e6c6509f688ef ===
