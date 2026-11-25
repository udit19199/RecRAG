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

=== ITEM id=bc5a7a42782f72b9 ===
Gold sentences and facts:
Sentence 0: Weekly Breakdown MATCHUP @ ATL ATL -2.5 O/U 42.5 OPP VS WR 6th PROJ PTS 12.1 WR RNK 37th YTD Stats REC 25 TAR 31 REYDS 261 TD 1 FPTS/G 11.4 The only two games McLaurin posted good numbers in were Washington's 35-33 win over Denver in Week 2 and Washington's 34-31 OT loss to Philadelphia in Week 4.
  Fact 0.0: The matchup is at ATL, and the betting line is ATL -2.5.
  Fact 0.1: The over/under is 42.5.
  Fact 0.2: The opponent's ranking against wide receivers is 6th.
  Fact 0.3: The projected points are 12.1.
  Fact 0.4: The WR rank is 37th.
  Fact 0.5: Year-to-date receptions are 25.
  Fact 0.6: Year-to-date targets are 31.
  Fact 0.7: Year-to-date receiving yards are 261.
  Fact 0.8: Year-to-date touchdowns are 1.
  Fact 0.9: Fantasy points per game are 11.4.
  Fact 0.10: The only two games McLaurin posted good numbers in were Washington's 35-33 win over Denver in Week 2.
  Fact 0.11: The only two games McLaurin posted good numbers in were Washington's 34-31 overtime loss to Philadelphia in Week 4.
Sentence 1: Again, don't expect a massive game because the running game, defense, and special teams will have plenty of success in another big victory.
  Fact 1.0: The writer says readers should not expect a massive game.
  Fact 1.1: The running game will have plenty of success in another big victory.
  Fact 1.2: The defense will have plenty of success in another big victory.
  Fact 1.3: The special teams will have plenty of success in another big victory.

Retrieved items:
<item 1>
Fantasy Football WR PPR Rankings Week 13: Who to start, best sleepers at wide receiver (Sporting News, 2023-11-29T20:27:00+00:00)
Dolphins), Puka Nacua and Tutu Atwell (vs. Browns), Drake London (@ Jets), and Tyler Lockett and Jaxon Smith-Njigba (@ Cowboys) this week.

McLaurin's averaging just 42 receiving yards over his past three games, and a heavy dosage of Jalen Ramsey hinders his Week 13 fantasy value.

Nacua and Atwell see a Cleveland coverage unit who entered Week 12 allowing the fewest fantasy points per game to WRs. Heck, even Cooper Kupp, who hasn't looked
like himself of late, could underperform again.

We can't trust London and the rest of the Falcons passing offense against a well-rounded Jets defense. To make matters worse, he'll be matched up against Sauce
Gardner for most of the day, too.

Lockett and Smith-Njigba might be better suited for the bench in PPR leagues against a lockdown Dallas coverage unit.
</item 1>

<item 2>
NFL Fantasy Football Week 6 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-12T21:23:16+00:00)
COLLINS: It's minor but Collins does see a higher target per route run rate against man coverage (25%) than zone (22.2%). He also has a higher catch rate
(87.5%) and higher receiving average (23.3) against man than zone (65.4% and 17.8). However, he's run nearly four times as many routes against zone coverage
compared to man.

MATCHUPS: It's also worth pointing out that Collins' two really big games came in Week 2 (vs. Indianapolis) and Week 4 (vs. Pittsburgh), both teams that rank in
the bottom-10 in Fantasy points allowed to wide receivers. The Saints are in the top half.

STROUD: Has been outstanding against man coverage all year, completing a higher rate of passes (67.6%) for 3.5 more yards per attempt (11.1) than versus zone
(7.6).
</item 2>

<item 3>
Fantasy Football WR PPR Rankings Week 13: Who to start, best sleepers at wide receiver (Sporting News, 2023-11-29T20:27:00+00:00)
A.J. Brown (vs. 49ers) faces a sound San Francisco coverage unit in Sunday's NFC Championship rematch, but his ability to dominate in the red zone still makes
him a top-10 PPR play. Ja'Marr Chase (@ Jaguars) averaged over 20 yards per reception in Week 12, and although the drop-off from Joe Burrow to Jake Browning is
massive, Chase still has WR1 upside given his usage.

Michael Pittman Jr. (@ Texans) faces a Texans' pass defense that's been shredded the past few weeks, so another 100-yard receiving day could be in store. Jaylen
Waddle (@ Commanders) is also looking to post his second consecutive 100-plus yard day, which could come to fruition against a struggling Commanders' coverage
unit. Tank Dell (vs. Broncos) continues to rise up our PPR rankings, rounding out our top 10 as a trusted option of C.J. Stroud's.
</item 3>

<item 4>
NFL Fantasy Football Week 5 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-04T21:56:25+00:00)
Weekly Breakdown MATCHUP vs JAC BUF -5.5 O/U 48.5 OPP VS WR 11th PROJ PTS 12.6 WR RNK 31st YTD Stats REC 12 TAR 18 REYDS 220 TD 3 FPTS/G 13 Until his role
changes, Davis will be the same kind of Fantasy option every week -- one big play and he's a stud, one or two misses and he's a dud. The Jaguars have done well
against deep-ball receivers but not enough to the point where benching Davis is a good idea. He's toward the top of the flex mix ahead of Tank Dell, Tutu
Atwell, Christian Watson and Jerry Jeudy.

LAST WEEK: Kept playing up to his boom/bust ceiling by catching a long touchdown. That's three straight games with a score. It was also his third game out of
four with four or fewer targets.
</item 4>

<item 5>
Fantasy Football RB PPR Rankings Week 14: Who to start, best sleepers at running back (Sporting News, 2023-12-07T22:09:00+00:00)
WEEK 14 STANDARD RANKINGS:

QBs | RBs | WRs | TEs | D/STs | Kickers

Who are the best fantasy RBs in Week 14?

Christian McCaffrey (vs. Seahawks in Week 14) has given us no reason to stop believing in him as the best overall player in the NFL, so he stays atop this list.
Austin Ekeler (vs. Broncos) has started to give us some pause as of late, as he's averaged fewer than 3.0 yards per carry since coming off IR eight games ago,
but a juicy home matchup against Denver should be his get-right game. Clear skies ahead, Ek owners — the Broncos have surrendered the most PPR points in the NFL
to RBs, and L.A. faces them twice between now and the fantasy title week.

With Jacksonville playing very well over the past month, we're all-in on Travis Etienne Jr. (@ Browns) even in a tough matchup.
</item 5>

<item 6>
NFL Fantasy Football Week 6 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-12T21:23:16+00:00)
Weekly Breakdown MATCHUP @ JAC JAC -4 O/U 44.5 OPP VS RB 7th PROJ PTS 8.8 RB RNK 20th YTD Stats RUYDS 445 REC 8 REYDS 72 TD 4 FPTS/G 21.4 The Colts have an edge
with their offensive line getting a step on the Jaguars' front; Moss already averages 2.13 yards before contact and the Jags give up 1.35 yards before contact
(and 1.94 yards before contact on power-scheme runs). It's purely a hunch but because Taylor is still ramping up and it takes as many as four weeks to "get
right," Moss should lead the way and could see as many as 15 touches. I'd start Moss as an RB2 over Rhamondre Stevenson, Emari Demercado and Taylor.
</item 6>

<item 7>
NFL Fantasy Football Week 6 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-12T21:23:16+00:00)
By the end of the game McLaurin's body language seemed dispirited.

COMMANDERS: Throw to their wideouts on 58.7% of their targets, which is below league average (but not as bad as, say, the Falcons at 36.4%). What makes that
number worse is that McLaurin has just one more target (31) than Jahan Dotson (30), who has three more targets than Curtis Samuel (27). They're all nearly even
after five games and each of them has had a turn leading in targets at least once.

FALCONS: Are expected to play a heavy dose of zone coverage, if only because the Commanders have struggled to be explosive against zone coverage this year.
Atlanta also gave a full dose of snaps last week to outside cornerback Jeff Okudah, who wasn't flawless but does offer another sizable body to help against
stronger perimeter receivers. Between Okudah and A.J.
</item 7>

<item 8>
NFL Fantasy Football Week 6 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-12T21:23:16+00:00)
BILLS: Got their first test against tight ends last week in Evan Engram and swarmed him -- he caught 4 of 8 targets for 28 yards. He is used similarly to
Waller, which makes Buffalo's game plan for slowing Waller down easy to carry over this week. Their pass rush should also impact the Giants passing game and may
force Waller to run shorter routes so he gets open quicker.

Start 'em 10.0-7.00 6.99 - 5.01 5.00 - 0 Sit 'em

Sit Him (Lineup Decisions) Projections powered by Sportsline Dak Prescott QB DAL Dallas • #4

Age: 30 • Experience: 8 yrs.
</item 8>

<item 9>
NFL Fantasy Football Week 5 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-04T21:56:25+00:00)
MATCHUPS: Each top-target earning wide receiver that plays mostly outside (like McLaurin) has found at least 10 PPR points in every game against the Bears. That
includes Rashee Rice in Week 3 (7 targets, 5-59-0), Courtland Sutton in Week 4 (5 targets, 3-27-1), Romeo Doubs in Week 1 (5 targets, 4-26-2) and Mike Evans in
Week 2 (8 targets, 6-171-1). Note that these receivers combined to catch 18 of 25 targets (72%) against Chicago.
</item 9>

<item 10>
NFL DFS picks Week 15: Best sleepers, value players for FanDuel, DraftKings, SuperDraft lineups (Sporting News, 2023-12-13T20:32:00+00:00)
Fire him up in what should be a pass-heavy matchup at Lambeau.

Demarcus Robinson, Rams vs. Commanders (DK: $3,600 | FD: $5,600 | SD: 1.7x)

Robinson has become one of Matthew Stafford's most trusted receivers, drawing a 21.8-percent target share over the past two weeks. And the veteran has made the
most of his opportunities, collecting seven catches for 101 yards and two TDs during that span. The Commanders are terrible at defending the pass — even worse
than they were earlier in the season since they shipped out multiple impact pass-rushers. Washington has surrendered the most FanDuel points and second-most
DraftKings points to wide receivers this season. Run DMC into your DFS lineups this weekend.

More solid Week 15 WR values: Amari Cooper, Browns vs.
</item 10>

<item 11>
N.F.L. Executives on the Biggest Questions Facing Super Bowl Contenders (The New York Times, 2023-10-16T23:58:53+00:00)
• Baltimore Ravens: Are the Ravens better off shifting to more of a pass-oriented offense?

What I think: Baltimore has never scored fewer points or amassed less offensive EPA through the first six games of a season with Lamar Jackson in the lineup.
The schedule and injuries have played roles, but with the Ravens shifting toward a more conventional pass offense, are the explosive quarterback runs mostly in
the past? Jackson has one rush longer than 20 yards (he’s had four or five by now previously). He’s gaining at least eight yards on 15 percent of carries, half
his previous rate. I’m not expecting a consistent precision passing game to suddenly materialize. Will this new offense be tougher to defend?

Advertisement

Exec comment: “The defense keeps them in it, but I don’t necessarily think this offensive change is going to make them any better when it counts. Lamar Jackson
does not appear as dynamic as a runner, but in this offense, he does not need to be.
</item 11>

<item 12>
Fantasy Football Week 13 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-11-30T22:10:58+00:00)
I'd expect that to be out the window now -- Flacco wasn't shy about chucking the rock deep with the Jets, attempting passes of 15-plus Air Yards on 20 percent
of his throws, and he had no problem focusing on one primary receiver (Garrett Wilson had 8, 14 and 11 targets in Flacco's starts). The hope is that both Flacco
and Cooper are experienced and capable enough to develop rhythm to salvage the Browns passing game. It's the only thing to buy into here as the Rams have
allowed the 12th fewest Fantasy points to wide receivers on the season and happen to have done well against deep throws against wide receivers (40% completion
rate allowed, two touchdowns, both top six on the season). Cooper's volume should hopefully push him to a decent Fantasy stat line. I'd give him a whirl over
Terry McLaurin, Marquise Brown and Diontae Johnson.
</item 12>

<item 13>
NFL Fantasy Football Week 6 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-12T21:23:16+00:00)
Tack on the ascending secondary the Falcons seem to have and the Commanders' diffusion of targets and McLaurin is at best a low-end flex. I'd rather roll the
dice on Drake London, Tyler Lockett or Garrett Wilson.

LAST WEEK: Hopeful that his target edge over his teammates would keep growing, and even in a favorable game script where the Commanders were trailing by three
scores before halftime, McLaurin only saw five targets against the Bears' heavy zone coverage.

FILM: On the 10 third-down routes McLaurin ran last week, he drew extra coverage on four of them, got open on two of them (Howell threw elsewhere) and was a
decoy on Curtis Samuel's touchdown. What happened? Sam Howell was getting rid of the ball quickly and to shorter distances as the Bears were getting some
pressure on him. By the end of the game McLaurin's body language seemed dispirited.
</item 13>

<item 14>
NFL Fantasy Football Week 6 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-12T21:23:16+00:00)
Weekly Breakdown MATCHUP @ LAC DAL -2.5 O/U 51 OPP VS QB 21st PROJ PTS 18.9 QB RNK 15th YTD Stats PAYDS 1061 RUYDS 45 TD 5 INT 4 FPTS/G 13.8 Trusting Prescott
is a tall task, even in what looks like a favorable matchup. The Cowboys might pivot to a slower approach with Tony Pollard handling the run game more than he
has been over the past couple of games. That could take pressure off of Prescott, but it doesn't mean he'll unload statistically. I'd rather start Sam Howell,
Jared Goff and Matthew Stafford.

LAST WEEK: A nightmare scenario as Prescott tried to get aggressive on downfield throws early and missed until he hit KaVontae Turpin for a 26-yard touchdown.
</item 14>

<item 15>
N.F.L. Executives on the Biggest Questions Facing Super Bowl Contenders (The New York Times, 2023-10-16T23:58:53+00:00)
GO DEEPER Behind the AI magic that lets Amazon's Prime Vision show the NFL like never before

• Buffalo Bills: Are the defensive injuries too much to overcome?

What I think: The rest of the conference is the key variable. The Bills are diminished. They have imperfections. But Buffalo already defeated the Dolphins. The
other AFC teams aren’t exactly peaking. Kansas City and Cincinnati appear easier to defend this season than in the recent past. We’ll see how well the Dolphins
hold up. This can still be Buffalo’s season.

Exec comment: “Matt Milano and Tre’Davious White, those are huge injuries for that football team. The Bills are not as dominant as they felt going into it with
so many injuries on defense. But the division they are in, they will still be able to slug it out and get to where they need to go. Nobody has really emerged in
the AFC to unseat these teams at the top.”
</item 15>

<item 16>
NFL Fantasy Football Week 6 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-12T21:23:16+00:00)
Start 'em 10.0-7.00 6.99 - 5.01 5.00 - 0 Sit 'em

Sit Him (Lineup Decisions) Projections powered by Sportsline Terry McLaurin WR WAS Washington • #17

Age: 28 • Experience: 5 yrs. Weekly Breakdown MATCHUP @ ATL ATL -2.5 O/U 42.5 OPP VS WR 6th PROJ PTS 12.1 WR RNK 37th YTD Stats REC 25 TAR 31 REYDS 261 TD 1
FPTS/G 11.4 The only two games McLaurin posted good numbers in were Washington's 35-33 win over Denver in Week 2 and Washington's 34-31 OT loss to Philadelphia
in Week 4. This matchup has an over/under of 42.5 points, tied for fourth-lowest of the week.
</item 16>

<item 17>
NFL Fantasy Football Week 6 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-12T21:23:16+00:00)
He has more upside than Jerry Jeudy, Jahan Dotson and Tutu Atwell but I'd rather take a shot on Drake London, Terry McLaurin or Chris Godwin.

LAST WEEK: Over 19 snaps without Justin Jefferson last week, Osborn ran 18 routes and caught both of his two targets for 30 total yards. The receiving average
was delightful, but the small-sample target share (12.5%) was saddening. What's funny is that Osborn's target share was much higher when Jefferson was playing
(23.3%) but his receiving average tanked (6.3).

FILM: Osborn didn't beat coverage with his speed last week and his burst and agility were just OK. His 22-yard catch came on a Kirk Cousins scramble where he
waited in a spot and Cousins hit him, then Osborn ran inside instead of outside where he might have been able to score.
</item 17>

<item 18>
Fantasy Football WR PPR Rankings Week 14: Who to start, best sleepers at wide receiver (Sporting News, 2023-12-07T22:11:00+00:00)
The league's leading receiver, Tyreek Hill (vs. Titans in Week 14), stands as the unquestioned WR1 for Week 14 after torching the Commanders to the tune of five
catches, 157 yards, and two TDs. CeeDee Lamb (vs. Eagles) and the Cowboys look to avenge their road loss to the Eagles in Week 9, as Dallas' WR1 routinely has
double-digit reception potential.

Amon-Ra St. Brown (@ Bears) was relatively quiet this past week, but he's likely to rebound against a Bears defense that he totaled nine catches for 95 yards
against in Week 11. A.J. Brown (@ Cowboys) reached the century mark for the first time in four games in Week 13, and another high-usage day looks to be in store
for a potential shootout.

Keenan Allen (vs. Broncos) caught just five-of-nine targets last week, but as L.A.
</item 18>

<item 19>
NFL Fantasy Football Week 5 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-04T21:56:25+00:00)
I think he'll continue to see high-value targets near the end zone and, hopefully, a small uptick in targets. I'd start him as a No. 2 receiver over Terry
McLaurin, any Broncos wideouts and any Packers wideouts.

LAST WEEK: Was so close to having a monster game. He was the designed target on a trick play and saw the ball fall just past his outstretched hands in the end
zone. He also had two targets inside the 10-yard line (his first two of the season) and two end-zone targets including the aforementioned trick play (his first
two of the season). The heavy dose of targets inside the 10 and into the end zone are positive signs that the Titans are trying to get him going as a playmaker.

FILM: Hopkins might be 31 years old but he moved really, really well, creating separation with outstanding footwork and having just enough speed to stay a step
ahead of defenders.
</item 19>

<item 20>
Fantasy Football Week 13 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-11-30T22:10:58+00:00)
The Commanders offensive line has had some ugly showings lately against the Cowboys (four sacks, 49% pass rush pressure rate allowed), the Giants (four sacks,
32.7% pressure rate allowed) and the Seahawks (three sacks, 46.8% pressure rate allowed). Somehow these results are better than what they were allowing through
much of the first two months of the year. Sam Howell getting the ball out faster has been a positive in this regard but a negative for his numbers in three of
his past four games (the one game he did really well in was sort of fluky). The matchup is a good test for the Dolphins pass rush post-Phillips, one that it
should pass without giving up much in the way of deep completions.

STARTS: Tyreek Hill, Jaylen Waddle, Raheem Mostert, Tua Tagovailoa, Terry McLaurin (low-end WR2),
</item 20>

<item 21>
Fantasy Football RB PPR Rankings Week 14: Who to start, best sleepers at running back (Sporting News, 2023-12-07T22:09:00+00:00)
Spears contributed 20 yards on the game-tying drive in the fourth quarter and another 27 yards in OT, but Mike Vrabel's squad ultimately lost. In what should be
a high-scoring game against the Super Bowl-hopeful 'Fins, we like Spears to be busy as rookie QB Will Levis's top check-down option -- assuming Henry
(head/neck) is out, of course.

Ezekiel Elliott (@ Steelers) is another potential fill-in who should produce at least flex-worthy stats this week. With Rhamondre Stevenson (ankle) on the
shelf, Elliott could see 20 touches. Similarly, Kenneth Gainwell (@ Cowboys) and Roschon Johnson (vs. Lions) could see big upticks in touches if D'Andre Swift
and D'Onta Foreman can't suit up, but we're not as confident those backs can produce given their matchups.
</item 21>

<item 22>
NFL Fantasy Football Week 6 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-12T21:23:16+00:00)
Plus the Browns just saw the Cowboys play 90% of their snaps in zone coverage and Purdy cooked them like BBQ chicken.

PURDY: Has been sensational against both man and zone coverages this season. His completion rate and yards per attempt are noticeably higher against zone, but
five of his seven passing touchdowns have come versus man coverage. However, Purdy's been at his worst when pressured (like most quarterbacks), completing 50%
of his throws for 6.7 yards per attempt with a gaudy 15.9% off-target rate. These stats are actually better than they were last week.

BROWNS: Rank third in pass rush pressure rate at 41.5% and blitz at an above-average 29.3% rate. Granted, the Cowboys ranked higher than that last week and got
beat anyway.
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
NFL Fantasy Football Week 5 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-04T21:56:25+00:00)
Only Joshua Dobbs, who ran for 55 yards, has scored double-digit Fantasy points against the Niners (18 points). Not that the Cowboys have been tested much --
Daniel Jones, Zach Wilson and the Patriots QBs are the other guys they've man-handled.

O-LINE: San Francisco has allowed a pass rush pressure on 36.4% of their snaps, which sounds like a lot but it's just a shade worse than league average. That
number is weighted by the Giants' decision to blitz on 84.6% of Purdy's dropbacks in Week 2, creating an inflated pass rush pressure rate. Despite the Giants'
efforts, Purdy still threw for 310 yards and two touchdowns in a game he didn't have Brandon Aiyuk in.

COWBOYS: Have become one of the most aggressive defenses in the league, ranking third from the bottom in zone coverage snaps.
</item 24>

<item 25>
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
</item 25>

<item 26>
NFL Fantasy Football Week 6 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-12T21:23:16+00:00)
The only rush defense metrics they're not top-five in are Defensive Rush EPA (just below league average), yards before contact per rush (0.88 yards, they're
sixth), yards after contact per rush (2.80, just above league average). They even lead the NFL in fewest missed tackles. Tony Pollard had an 8-29-0 rushing stat
line against them last week.

WHAT ABOUT THROUGH THE AIR? Even though the 49ers see the fifth-most RB targets per game, they're better than league average in catch rate allowed (75%) and
top-10 in both yards per catch (5.7) and YAC/reception (6.10) allowed to running backs. Pollard caught 4 of 5 passes for 35 yards last week.

P.J. WALKER: Has thrown 17.2% of his career pass attempts to running backs.
</item 26>

<item 27>
N.F.L. Executives on the Biggest Questions Facing Super Bowl Contenders (The New York Times, 2023-10-16T23:58:53+00:00)
They are gaining more than 15 yards on just 11 percent of pass plays, down from 21 percent through six games last season and 19 percent for 2022 overall. Jalen
Hurts, A.J. Brown, DeVonta Smith and Dallas Goedert are still on the team, so I’m inclined to trust the production will return. But there’s also a lingering
fear defenses have caught up to aspects of the Eagles’ scheme. Does Philly have a counterpunch?

Advertisement

Exec comment: “People are going to crush Hurts and the offense. They are just churning the clock so much. It just feels like they are imposing their will on
teams through the run game and through their short passing and it just hasn’t taken off from an explosive passing standpoint, so when they play a team that can
move the ball on them through the air, it presents some challenges.”
</item 27>

<item 28>
Fantasy Football Week 13 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-11-30T22:10:58+00:00)
They were even less kind to Geno Smith, and this week shouldn't be much different. Tack on the tough assignment in Stephon Gilmore in coverage (unless the
Cowboys use shadow coverage for the first time this year), and Lockett could be in for another lean night. I don't mind him as a WR3 over Drake London, DeAndre
Hopkins and Diontae Johnson, but I'd start Rashee Rice and Adam Thielen over Lockett.

Sit Him (Lineup Decisions) Projections powered by Sportsline Jake Ferguson TE DAL Dallas • #87

Age: 24 • Experience: 2 yrs.
</item 28>

<item 29>
Christmas Day preview: 49ers, Ravens square off in potential Super Bowl sneak peek (Yardbarker, 2023-12-24T23:34:39+00:00)
The 23-year-old ranks first in the NFL in QBR (76) and touchdown passes (29) while ranking third in passing yards (3,795), all of which are especially
impressive considering he's coming off offseason elbow surgery.

Jackson might not be posting the ridiculous numbers he did during his 2019 MVP campaign, but the adjustment in his play style in his first season under OC Todd
Monken has led to sustainable, winning football. Nevertheless, Baltimore needs an upset win for Jackson to capture another MVP Award, as he has thrown for 3,105
yards and 17 touchdowns while adding 741 yards on the ground with five scores this season.

49ers RB Christian McCaffrey: If Purdy wasn't playing at such a high level — and the MVP wasn't a quarterback-driven award — it'd be difficult to argue against
McCaffrey taking home the honors.
</item 29>

<item 30>
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
</item 30>

<item 31>
Fantasy Football Week 13 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-11-30T22:10:58+00:00)
Weekly Breakdown MATCHUP @ DAL DAL -9 O/U 47.5 OPP VS WR 10th PROJ PTS 10.6 WR RNK 31st YTD Stats REC 54 TAR 79 REYDS 575 TD 4 FPTS/G 12.5 Lockett has
seven-plus targets in four of his past five, but he still doesn't have as many as Metcalf's gotten in those games. He also doesn't see nearly as many deep
throws as Metcalf -- literally two targets in his past two games traveled 15 or more Air Yards. Obviously he has potential to always pop off for a big game like
he did against a tough Browns defense five weeks ago, but recent matchups against the Niners, Rams and Ravens weren't kind to Lockett. They were even less kind
to Geno Smith, and this week shouldn't be much different.
</item 31>

<item 32>
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
</item 32>

<item 33>
Fantasy Football Week 13 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-11-30T22:10:58+00:00)
Steve Spagnuolo's defense has been blitzing its tails off over the past two weeks (nearly 50% of their snaps against the Eagles and Raiders), so the hunch is
they'll apply maximum pressure to that O-line in an effort to make Love more off-target than he's been and, hopefully, get some turnovers. Only one receiver has
had more than 15 PPR points against the Chiefs in their past four games (Jakobi Meyers last week), making it tough to expect grand games from Love's favorite
targets. I'd expect shutdown cornerback L'Jarius Sneed to line up across from Christian Watson the most -- and Jayden Reed the least.
</item 33>

<item 34>
Fantasy Football QB Rankings Week 14: Who to start, best sleepers at quarterback (Sporting News, 2023-12-07T22:06:00+00:00)
Not far behind sits Tua Tagovailoa (vs. Titans), who shredded the Commanders' defense to the tune of 280 yards and two TDs in Sunday's blowout. Another juicy
matchup awaits against a lackluster Tennessee coverage unit. Dak Prescott (vs. Eagles) remains a top-three play in Sunday's rematch versus the Eagles, a team he
torched for 374 yards and three TDs in a narrow 28-23 loss in Week 9.

Jalen Hurts (@ Cowboys) and the Eagles look to get back on track in a pivotal NFC East clash in Dallas, facing a Cowboys defense that let Geno Smith post
top-end QB1 numbers against them last week. Patrick Mahomes (vs. Bills) is in line to put up strong numbers in a potential shootout against Buffalo, and
although his receiving corps has its issues, the reigning MVP is still a top-five play this week.

Lamar Jackson (vs.
</item 34>

<item 35>
Fantasy Football Week 13 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-11-30T22:10:58+00:00)
The uncertainty of how the Panthers will operate moving forward with Brown basically calling the offense without any oversight keeps these players are risky
Fantasy propositions, though it helps a lot that the Buccaneers are dealing with injuries that hurt their run defense (Lavonte David and Devin White will miss
the game) and keep their pass defense from improving (Jamel Dean will miss the game).
</item 35>

<item 36>
NFL Fantasy Football Week 5 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-04T21:56:25+00:00)
On the flip side, three of the four quarterbacks Chicago faced found three touchdowns on their limited throws.

Start Him (Lineup Decisions) Projections powered by Sportsline Terry McLaurin WR WAS Washington • #17

Age: 28 • Experience: 5 yrs. Weekly Breakdown MATCHUP vs CHI WAS -6 O/U 44.5 OPP VS WR 19th PROJ PTS 15.2 WR RNK 21st YTD Stats REC 21 TAR 26 REYDS 212 TD 1
FPTS/G 12.1 The Commanders may wish to win this one with a big dose of the run game (Brian Robinson is a must-start), but if the Bears put up any offense at
all, McLaurin will be in position to put up nice numbers. He's a low-end WR2 worth the nod over George Pickens, Zay Flowers, Drake London and Jahan Dotson.
</item 36>

<item 37>
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
</item 37>

<item 38>
NFL Fantasy Football Week 5 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-04T21:56:25+00:00)
BILLS: Lost top cornerback Tre'Davious White to a torn Achilles last week which means some combination of Christian Benford, Dane Jackson and Kaiir Elam (who
had been benched for the first four games) will work at outside cornerback with Taron Johnson in the slot. That looks good, but the Bills pass rush is among the
fiercest in football with a top-10 pass rush pressure rate (38.2%) without blitzing much (15.3% blitz rate is second-lowest in football). Unfortunately, pass
rush terror Greg Rousseau won't play for the Bills, lowering their dominance in this area.

Flex Starter (Lineup Decisions) Projections powered by Sportsline Gabe Davis WR BUF Buffalo • #13

Age: 24 • Experience: 4 yrs.
</item 38>

<item 39>
Christmas Day preview: 49ers, Ravens square off in potential Super Bowl sneak peek (Yardbarker, 2023-12-24T23:34:39+00:00)
Can QB Jalen Hurts and the Eagles offense finally bounce back?: Philadelphia's defense isn't the only unit that needs to rebound, as the offense has struggled
mightily over the last three games, with Hurts' ineffectiveness playing a significant role. After averaging the third-most points in the NFL over the first 11
games (28.2), the Eagles have scored four offensive touchdowns during their current skid, three of which came via the "tush push."

The Eagles may have difficulty throwing against a solid Giants pass defense, though they could alleviate some of their offensive woes by running the ball
efficiently. New York gives up 131.6 rushing yards per game, the fourth-most in the league.

Does Giants QB Tommy DeVito have any magic left?: The Giants may have a terrible offense, but this unit has operated at its best under DeVito, an undrafted
rookie and native of Cedar Grove, New Jersey.
</item 39>

<item 40>
NFL Fantasy Football Week 5 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-04T21:56:25+00:00)
He's still a decent start ahead of Najee Harris, Khalil Herbert and Jahmyr Gibbs but I'd rather chance my lineup with someone who has upside like Jaleel
McLaughlin (if Javonte Williams sits out) or Alexander Mattison.

FILM: The numbers don't quite match the talent -- Stevenson is a capable runner with great vision, balance and power to go with good speed. He struggled on
lateral runs last week and his line failed him plenty on north-south runs. Game script kept him from finding a lot of work last week (one fourth-quarter touch).

THIS SEASON: Stevenson has at least 16 touches in every game, which is good. Unfortunately he has yet to total 100 yards and has been under 70 yards three
times, which is definitely the opposite of good.
</item 40>

<item 41>
NFL Fantasy Football Week 6 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-12T21:23:16+00:00)
Between Okudah and A.J. Terrell there isn't a matchup that's great for McLaurin.

McLAURIN: Has an impressive 91.3% catch rate versus zone coverage this year, but is averaging just 9.3 yards per catch with it on a sub-9.0 ADOT. His target
share is below 20% regardless of the kind of coverage he sees.

Start Him (Lineup Decisions) Projections powered by Sportsline Logan Thomas TE WAS Washington • #82

Age: 32 • Experience: 9 yrs. Weekly Breakdown MATCHUP @ ATL ATL -2.5 O/U 42.5 OPP VS TE 31st PROJ PTS 8.4 TE RNK 11th YTD Stats REC 18 TAR 25 REYDS 183 TD 2
FPTS/G 11.6 Streaming Thomas is worth the risk considering how the Falcons have let up numbers to tight ends.
</item 41>

<item 42>
Prisco's Week 12 NFL picks: Jaguars beat Texans in AFC South battle, Saints get key divisional win vs. Falcons (CBSSports.com, 2023-11-23T21:30:38+00:00)
Dak Prescott plays well again. Cowboys roll.

Pick: Cowboys 32, Commanders 16 (Result: Cowboys 45-10)

Thursday, 8:20 p.m. ET (NBC, fubo)

This is a huge division game for two teams that have had a good rivalry over the past decade. The Seahawks are coming off a bad loss to the Rams, while the
49ers are coming off two dominant victories. Geno Smith is banged up, so his health is key. The 49ers have the look of a Super team again. They show it here.

Pick: 49ers 28, Seahawks 19 (Result: 49ers 31-13)

Friday, 3 p.m. ET (Amazon Prime Video)

This Friday game the day after Thanksgiving isn't what it was expected to be with the Jets being an offensive disaster.
</item 42>

<item 43>
NFL Fantasy Football Week 6 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-12T21:23:16+00:00)
O-LINE: It's possible that injuries that go as far back as to Week 2 along with some declining play from Tyler Biadasz and Terence Steele could be contributing
factors for Prescott's low ADOT. He doesn't want to get hit, so he's getting the ball out as fast as he can. Even against the 49ers he sped up his motion and
overshot those targets. He's just not as comfortable and the offense isn't adjusting well enough.

CHARGERS: Traditionally play a good amount of zone coverage but have changed it up more often this season and used more man-to-man, including versus Miami in
Week 1. But despite some headliners in the front seven and 16 sacks in their past three games, the Chargers rank fourth-worst in pass rush pressure rate at
29.5% this season. They also blitz at about the league average rate.
</item 43>

<item 44>
Fantasy Football Week 13 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-11-30T22:10:58+00:00)
It gets worse: The Panthers offense can't score and their run defense is so bad, leaving pass-negative game scripts for their opponents. Just two teams (the
Saints and Dolphins) have rostered multiple receivers with 10-plus PPR points against the Panthers in 2023. I'd sit Godwin for Josh Downs, Brandin Cooks,
Christian Watson and Curtis Samuel (especially if PPR).

Start 'em 10.0-7.00 6.99 - 5.01 5.00 - 0 Sit 'em

I wouldn't expect the Rams offensive line to handle the Browns' defensive front like they did the Cardinals' last week, but it must be mentioned that the Browns
run defense isn't anywhere near as dominant as it was earlier this year.
</item 44>

<item 45>
Fantasy Football Week 13 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-11-30T22:10:58+00:00)
No one else has had more than 86.

STARTS: Dak Prescott, CeeDee Lamb, Tony Pollard, DK Metcalf (low-end WR2), Jake Ferguson, Cowboys DST

Dak Prescott, CeeDee Lamb, Tony Pollard, DK Metcalf (low-end WR2), Jake Ferguson, Cowboys DST SITS: Tyler Lockett (flex at best), Geno Smith, Jaxon
Smith-Njigba, Zach Charbonnet, Michael Gallup, Seahawks DST

Tyler Lockett (flex at best), Geno Smith, Jaxon Smith-Njigba, Zach Charbonnet, Michael Gallup, Seahawks DST SLEEPERS: Brandin Cooks

Start Him (Lineup Decisions) Projections powered by Sportsline DK Metcalf WR SEA Seattle • #14

Age: 26 • Experience: 5 yrs.
</item 45>

<item 46>
Fantasy Football Week 13 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-11-30T22:10:58+00:00)
0-7.00 6.99 - 5.01 5.00 - 0 Sit 'em

The Lions need an answer for their ugly pass defense -- since their Week 9 bye they've been mostly great against the run (3.1 yards per carry allowed, one RB to
get 15-plus PPR points) but they rank bottom-six in completion rate allowed (68.4%), yards per attempt allowed (8.0), passing ADOT against (8.84 yards) and
defensive Expected Points Added (EPA) per dropback (-0.40, which is league-worst). And of their second-most eight passing touchdowns given up in three games,
five have come on passes of 15-plus Air Yards, which is tied for most in the league. They should catch a break with field stretcher Rashid Shaheed (thigh)
missing the matchup for New Orleans; Chris Olave looks like he will play.
</item 46>

<item 47>
NFL DFS picks Week 15: Best sleepers, value players for FanDuel, DraftKings, SuperDraft lineups (Sporting News, 2023-12-13T20:32:00+00:00)
Buccaneers (DK: $3,000 | FD: $4,700 | SD: 1.8x)

NFL DFS Picks Week 15: D/ST sleepers, values for DraftKings, FanDuel

Los Angeles Rams vs. Commanders (DK: $3,100 | FD: $4,300)

The Rams got destroyed by MVP candidate Lamar Jackson and the high-flying Ravens in Week 14, but push that out of your memory banks. Before that game, Sean
McVay's squad had surrendered 20 or fewer points in four straight contests. Aaron Donald and company also have 17 sacks and six takeaways over their past six
games, so they'll be licking their chops looking across the line at Sam Howell this weekend. No QB has more interceptions than Howell, who has also taken 10
more sacks than any other signal-caller this season.
</item 47>

<item 48>
NFL Fantasy Football Week 6 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-12T21:23:16+00:00)
The Cowboys wound up paying the price through the air for playing a ton of zone coverage instead. It was suggested by a friend that they didn't want to get
burned by long runs by Christian McCaffrey. I guess that worked. The combination of the 49ers defense running hot and the Browns offense figuring to get stuffed
should make for at least a game script for Purdy to have decent numbers. I'm not making the mistake again of benching Purdy -- I'd start him over Trevor
Lawrence, Geno Smith and Jared Goff.

LAST WEEK: Took on one of the toughest pass rushes in football and slayed them for four touchdowns on just 24 pass attempts. Of his seven incompletions, one was
dropped, two were batted at the line of scrimmage, another was a throwaway and two more were defensed. Purdy was off-target on 0.0% of his throws.
</item 48>

<item 49>
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
</item 49>

<item 50>
Fantasy Football Week 13 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-11-30T22:10:58+00:00)
Easy peasy!

Start 'em 10.0-7.00 6.99 - 5.01 5.00 - 0 Sit 'em

Both teams had a week of rest, so don't treat this like a typical Thursday game. But do expect the Cowboys defense to keep the pressure on Geno Smith and a
Seattle offense that's struggled to gain consistency over the last month. Smith actually did OK completing passes last week when pressured (60% completion rate)
but not at an efficient level (5.4 yards per attempt). Just two QBs have exceeded 250 yards and four QBs have thrown multiple touchdowns against Dallas this
year; I don't like Smith's chances to do either, which hurts any upside case for any Seahawks receiver. By the way, Curtis Samuel was the first receiver to get
100 yards on the Cowboys this year, and it took him 12 targets and 9 catches to get there.
</item 50>

<item 51>
Seahawks-Cowboys Fantasy Start 'Em Sit 'Em Week 13: Should you start Zach Charbonnet, Jake Ferguson on TNF? (Sporting News, 2023-11-30T21:31:00+00:00)
The Seahawks have now joined the bottom five teams in giving up fantasy points to running backs. Pollard has come on the past two weeks against the Panthers and
Commanders after sliding into RB2 range. Pollard scoots to something like 80 and a score. Dowdle is a stretch to get mop-up production, however,

Start: WR CeeDee Lamb

Lamb can eat again in the slot against Jamal Adams, as he avoids the Seahawks' better cornerbacks on the outside. Again, don't expect a massive game because the
running game, defense, and special teams will have plenty of success in another big victory.

Sit: WR Brandin Cooks

Cooks is the one Cowboy to fade this week vs. rookie Devon Witherspoon, who's quickly emerging as the Seahawks' best corner. The nature of the zone defense is
to limit big plays.
</item 51>

<item 52>
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
</item 52>

<item 53>
NFL Fantasy Football Week 5 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-04T21:56:25+00:00)
LAST WEEK: Was the benefactor of Dak Prescott taking what the Patriots defense gave him, mostly utilizing Ferguson as a short-area target to fire the ball to
when the pressure was quick off the snap. But there were a couple of throws where Prescott read the defense well and found Ferguson for pretty good gains. All
told Ferguson led the Cowboys with seven targets and seven catches for 77 yards in three quarters of work.

THIS WEEK: The 49ers figure to unleash its pass rush on Prescott with zone coverage behind it, which only makes sense given how Prescott has handled the pass
rush pressure this year. It should mean more short throws for Ferguson.

49ERS: Are tops in football in catch rate allowed to tight ends (48.6%) and YAC/reception allowed (1.29 yards). They're also top-five in yards per catch (7.1).
</item 53>

<item 54>
NFL Fantasy Football Week 5 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-04T21:56:25+00:00)
On the season he's been a full yard worse on outside runs and about the same on inside runs, so perhaps he and the Texans are improving on those edge plays.

FALCONS: Have run hot and cold against the run but they've ultimately been good by virtue of their 3.8 rushing average and zero rushing touchdowns allowed to
running backs this season. They gave up just three runs of 10-plus yards to RBs in their past three games (all against Jahmyr Gibbs, none to Travis Etienne or
A.J. Dillon). They've also held backs to 2.7 yards per carry on edge runs while yielding 4.3 yards on carries between the tackles.

FALCONS: Have not allowed more than 8 non-PPR/11 PPR points to any running back so far this season.
</item 54>

<item 55>
NFL Best Bets Week 9: Colts clobber Panthers, Dolphins drop Chiefs in Germany (Sporting News, 2023-11-03T20:56:00+00:00)
Expect more of the same from QB Justin Herbert and his stud back against the Jets, who have allowed 46 receiving yards per game to RBs (sixth-most in the NFL).
They'll try to shut him down this Sunday, but it would feel so empty without AE!

Jahan Dotson, WR, Commanders @ Patriots — OVER 45.5 receiving yards

We've come full circle with Dotson, who made our preseason sleepers column then faded into early-season obscurity only to emerge again over the past two weeks.
Curtis Samuel was clearly not right last week, and the second-year wideout more than made up for it. Dotson caught 8-of-10 targets for 108 yards and a TD, a
stellar encore to his Week 7 line of five catches for 43 yards. The banged-up Patriots defense will be primarily focused on stopping stud Commanders receiver
Terry McLaurin, which should open up plenty of opportunities for Dotson.
</item 55>

<item 56>
NFL Fantasy Football Week 5 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-04T21:56:25+00:00)
PAST THREE WEEKS: McLaurin has led all Commanders wide receivers in targets and all Commanders players in receptions. He's the No. 1 guy, and his big Week 4
game (8-86-0 with an offensive fumble recovery for a touchdown) would have been even bigger if Sam Howell had better ball placement on a deep pass McLaurin was
open for.

BEARS: Are expected to play again without top outside cornerback Jaylon Johnson and safety Eddie Jackson. Not that those two absences led to a bonanza from
Broncos receivers; Russell Wilson spread the ball around. But if McLaurin remains the target leader then he'll have the best shot at a big game. Rookies Tyrique
Stevenson and Terell Smith figure to line up across from McLaurin the most.
</item 56>

<item 57>
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
</item 57>

<item 58>
NFL Fantasy Football Week 5 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-04T21:56:25+00:00)
Or Denver. Or Chicago. You get the gist. The Cardinals pass rush shouldn't be too dangerous and Burrow, now over a month into his season, figures to eventually
show some signs of improvement with his health. In no way is he a must start, and there's plenty of good reason to go with Trevor Lawrence, Russell Wilson, C.J.
Stroud and even Joshua Dobbs over him, but there's some mild appeal that should at least keep him in lineups over Dak Prescott and Daniel Jones.

LAST WEEK: Looked like a quarterback playing on one leg. Burrow didn't get much into his throws, didn't challenge downfield much, got the ball out as fast as he
could and was sensitive to pass rush pressure. Even when the game was in garbage time Burrow couldn't get much going.
</item 58>

<item 59>
Fantasy Football Week 13 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-11-30T22:10:58+00:00)
Weekly Breakdown MATCHUP @ LAR LAR -3.5 O/U 40.5 OPP VS WR 3rd PROJ PTS 11.4 WR RNK 23rd YTD Stats REC 47 TAR 86 REYDS 765 TD 2 FPTS/G 12.3 It appears Cooper
will play through a rib injury he suffered last week, and it also appears he'll catch passes from new Browns quarterback Joe Flacco. This is actually a great
thing for Cooper; over his past two weeks his ADOT sat at 7.57 yards, which was roughly half of what it was in his first nine games at 15.42 yards. Clearly, the
Browns wanted Cooper to run shorter routes and make things easier on rookie QB Dorian Thompson-Robinson.
</item 59>

<item 60>
NFL Fantasy Football Week 5 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-04T21:56:25+00:00)
Bringing back guard Teven Jenkins might help a little, but they're still a struggling unit.

DEEP BALL: Washington ranks bottom-five in attempts, yards per catch and touchdowns allowed on pass plays of 15-plus Air Yards over the past three weeks. This
is an area Fields can exploit -- he's tied for the most touchdowns on throws of 15-plus Air Yards this season than Fields with four and is around league-average
in completion rate, attempts per game and Expected Points Added (EPA) per drop back.

LAST SEASON: Washington got a season-best 57.1% pass rush pressure rate on Fields and blitzed him on 40.5% of his drop backs. Fields had one passing touchdown
and one interception and still had nearly 20 Fantasy points because he ran for 88 yards.

RUSHING: Fields followed up a Week 3 game with seven designed runs with just two designed runs in Week 4 for two total yards.
</item 60>

<item 61>
NFL Fantasy Football Week 6 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-12T21:23:16+00:00)
LAST WEEK: Despite splitting first-half snaps almost evenly and losing out on second-half snaps considerably, McLaughlin made three splash plays of 21, 22 and
38 yards including a touchdown on a tipped-then-secured designed screen that pierced through the middle of the Jets defense. Those three touches were his best
-- his other nine touches each failed to exceed 3 yards. And as mentioned two sentences ago, he played just 10 snaps in the second half when the Broncos were
trailing.

FILM: McLaughlin has good vision and speed, both of which were essential in reading the Jets defense at the first and second levels and choosing the correct
paths to run on his big plays. His lateral agility was a differentiator on his touchdown catch. McLaughlin's worst asset is his power as he just won't finish
runs with physicality like his backfield mates do.
</item 61>

<item 62>
DraftKings Picks Week 14: Best NFL DFS lineup advice for daily fantasy football tournaments (Sporting News, 2023-12-07T22:18:00+00:00)
Because we splurged for McCaffrey and Kamara, we had to sacrifice salary elsewhere, and London is a high-upside option even in an offense that doesn't throw the
ball with efficiency.

WR Jaxon Smith-Njigba, Seahawks @ 49ers ($4,100). Is a takeover underway in the Seahawks offense? Smith-Njigba is coming off a game against the Cowboys that saw
him collect seven-of-11 targets for 62 yards. It's a step up from a disappointing outing against San Francisco, but Tyler Lockett's involvement seems to be
lessening as of late with the rookie stepping up. It wasn't a great day for the Seattle offense on Thanksgiving against the Niners, but the Seahawks should put
up a better effort against a divisional opponent they know so well. If they're trailing, JSN should see action with a negative game script.
</item 62>

<item 63>
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
</item 63>

<item 64>
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
</item 64>

<item 65>
NFL Fantasy Football Week 6 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-12T21:23:16+00:00)
STEVENSON: Nothing has changed in terms of his efficiency. We can blame the offensive line to some degree but Stevenson has averaged 0.97 yards before contact
per rush this season. That ranks 30th out of 41 qualifying running backs. And if you think that's bad then consider his yards after contact per rush average of
1.79, which is dead last among those same 41 qualifiers.

EXPLOSIVE RUNS: Stevenson has one (1) explosive carry of 12 or more yards. It was against Miami. It went 12 yards. This is a big change from 2022 when he had 21
explosive runs and an explosive run rate of 10%, which was eighth-best.

RECEIVING: Is averaging 3.6 targets per game and catching 67% of them but is getting just 7.3 yards per catch and has one (1) explosive catch of 16 or more
yards.
</item 65>

<item 66>
NFL Fantasy Football Week 6 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-12T21:23:16+00:00)
That included in the second half when he saw one target and couldn't catch it. He definitely wasn't shadowed by anyone and was almost never double-teamed,
though there were a couple of routes where a safety was in proximity. It could have had more to do with the Texans attacking the middle of the field against the
Falcons' zone coverage.

SAINTS: Have played the fifth-fewest snaps of zone coverage this season and have ignored tendencies some quarterbacks have had through the early portion of the
season. It's just what they like to do. Stud cornerback Marshon Lattimore has not followed receivers much this season, playing 87% of his snaps at right
cornerback (to the left of the quarterback). Their other primary outside corner, Paulson Adebo, has allowed a 50% catch rate on the season with minimal YAC.
</item 66>

<item 67>
NFL Fantasy Football Week 6 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-12T21:23:16+00:00)
Then Prescott HAD to throw deep to try and climb back into the game and got sloppy on two of his interceptions (a third bounced off of Michael Gallup's hands).
He didn't even play the last 12 minutes of the game and the Cowboys got blown out.

THIS SEASON: Prescott's completion rate is fine (69.4%) and his off-target rate is low (8.1%). So what's the problem? Prescott's ADOT is 6.5, ranking 28th among
33 qualifying quarterbacks. He's actually second-to-last in pass attempts of 10-plus Air Yards; only Daniel Jones has thrown fewer. And the scary part is that
his ADOT was over a yard lower (5.4) going into last week's game, so the Cowboys tried to let him air it out and the plan completely backfired.
</item 67>

<item 68>
Fantasy Football RB Rankings Week 14: Who to start, best sleepers at running back (Sporting News, 2023-12-07T22:07:00+00:00)
The injury highlights the importance of handcuffs, as Zack Moss has put up multiple top-10 performances with and without Taylor this season.

Seattle's Kenneth Walker III (oblique) missed Week 12 and 13, and we're not confident in his ability to suit up against the hard-hitting 49ers coming off a big
win in Philly. Backup Zach Charbonnet exited early because of a knee injury against the Cowboys, putting his status in doubt, too. If both are out, DeeJay
Dallas would start and lead the backfield in touches.

Green Bay's Aaron Jones has a sprained MCL, and sources have revealed that the Packers continue to consider him week-to-week. He only logged four carries before
sustaining the injury in Week 11, then he sat for the Packers' Thanksgiving Day tilt with the Lions and their Sunday Night Football win over Kansas City. Why
rush him back if Green Bay is already winning games?
</item 68>

<item 69>
Fantasy Football Week 13 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-11-30T22:10:58+00:00)
Brown, Jahmyr Gibbs, Alvin Kamara, Chris Olave (if he passes concussion protocol), David Montgomery, Sam LaPorta, Taysom Hill SITS: Derek Carr, both DSTs

Derek Carr, both DSTs SLEEPERS: Juwan Johnson

Start 'em 10.0-7.00 6.99 - 5.01 5.00 - 0 Sit 'em

The Falcons defense finally had a strong showing against the Saints last week, but they got a little lucky because so many players from New Orleans left with
injuries. In the three games prior their defense had a knack for making mediocre offenses look amazing (Will Levis' breakout game, Josh Dobbs' first game with
the Vikings, Kyler Murray's first game with the Cardinals).
</item 69>

<item 70>
Fantasy Football WR PPR Rankings Week 14: Who to start, best sleepers at wide receiver (Sporting News, 2023-12-07T22:11:00+00:00)
What are the biggest WR injuries in Week 14?

In Week 13, Demario Douglas (concussion) and Rashid Shaheed (thigh) missed game action, and Tank Dell (leg), Amari Cooper (concussion), Christian Watson
(hamstring), and Marquise Brown (heel) exited early from their respective games.

UPDATE: Christian Kirk suffered a groin injury on Monday night, and he figures to be more doubtful than questionable for this week's game against Cleveland. If
he's out, Zay Jones and Tim Jones will see more targets, but with Trevor Lawrence (ankle) also hurt, it's tough to count on either.

UPDATE: Douglas (concussion) has been ruled OUT for Week 14.

As usual, we'll update these WR PPR rankings throughout the week, so check back for the latest player movement and analysis.
</item 70>

<item 71>
NFL Fantasy Football Week 6 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-12T21:23:16+00:00)
But the biggest detriment is that he can't consistently overcome bad blocking, which has been a consistent problem for the Denver O-line all year and certainly
something that impacted Javonte Williams in his first handful of games.

WHEN TRAILING: When the Jets were up by eight or more points in the second half, McLauglhin played 4 of 15 snaps and had two carries. That's in line with a
trend we've seen from Denver all year: Of the 117 snaps they've played when down by eight or more points, McLaughlin has played 26 snaps (13 touches, seven at
Chicago), Williams has played 27 snaps (15 touches, almost all against Miami) and Samaje Perine has played 58 snaps (13 touches) -- more than both backs
combined. The only guy who's averaged a strong number on the ground in this specific situation? You guessed it -- McLaughlin (6.1 yards per carry).
</item 71>

<item 72>
NFL Fantasy Football Week 5 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-04T21:56:25+00:00)
LAST WEEK: The Texans didn't need him! Houston built a nice lead thanks to its defense and held on without having to throw the ball a ton.

FILM: The Steelers really emphasized not getting beat deep by Dell, often using a safety over the top to his side of the field to deter C.J. Stroud from even
thinking about throwing at him. It worked out for the Steelers because Dell had zero deep targets, and it worked for the Texans because Nico Collins beat soft
coverage a bunch including on a 24-yard deep ball (Dell drew triple coverage!) and on his long touchdown score (Dell got a safety to cheat over to his side).

FALCONS: Should be able to deploy the same type of defensive coverage with a safety cheating over to Dell's side of the field. They could also stick outside
cornerback A.J. Terrell on Collins and make things a little harder on the Texans.
</item 72>

<item 73>
NFL Fantasy Football Week 5 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-04T21:56:25+00:00)
On the season he's totaled 28 rushes for 134 yards and a score.

Sneaky Sleeper (Lineup Decisions) Projections powered by Sportsline Sam Howell QB WAS Washington • #14

Age: 23 • Experience: 2 yrs. Weekly Breakdown MATCHUP vs CHI WAS -6 O/U 44.5 OPP VS QB 31st PROJ PTS 21.3 QB RNK 14th YTD Stats PAYDS 961 RUYDS 82 TD 5 INT 5
FPTS/G 16.1 The Commanders like limiting Howell's pass attempts in general -- he's only thrown over 31 times in high-scoring shootouts like in Weeks 2 and 4.
While there's potential for that on Thursday, it's not a certainty. Howell could just as easily put up nominal stats to help the Commanders bring home a win.
</item 73>

<item 74>
NFL Fantasy Football Week 5 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-04T21:56:25+00:00)
His off-target rate in the fourth was 9.1%, which was closer to what he had averaged in his first three games (13.6%).

BRONCOS: Have allowed season-best Fantasy numbers to each of the past three quarterbacks they've played this season. They're a terrific matchup for anybody
right now. Remember this when we talk about the Jets.

COMMANDERS: Sport a slightly below-average pass rush pressure rate thanks to only one game all year with a rate over 30% (52.7% at Denver, and Wilson still had
a great game even if you take away a Hail Mary touchdown catch). They rank seventh in sacks (13) and 13h in pressures (53), which again floats around league
average.

BEARS: Rank league-worst in pass rush pressure rate allowed at 47.4% of their snaps -- nearly one out of every two drop backs is pressured.
</item 74>

<item 75>
NFL Fantasy Football Week 6 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-12T21:23:16+00:00)
They also blitz at about the league average rate. In theory this should not be a difficult matchup for Prescott, but

KELLEN MOORE: Prescott's QB coach and/or playcaller for the prior five seasons will be coaching the Chargers offense on Sunday, but he's sure to have spilled
the beans on Prescott's tendencies. We've seen offensive coordinators come back to wreck their old teams all the time and could see it again on Sunday.

FANTASY: The only quarterback to not get at least 22 Fantasy points against the Chargers was Raiders rookie Aidan O'Connell.
</item 75>

<item 76>
NFL Fantasy Football Week 5 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-04T21:56:25+00:00)
Sure, even including the Vikings. Doesn't mean it's not a good look.

NOT SURPRISINGLY: Sanders has two runs of 12-plus yards through four games -- and neither went more than 15 yards. It's not all on him -- the offensive line for
the Panthers barely gave him any daylight last week (and for much of the year) and defenses aren't yet scared of getting beat by Bryce Young's arm.

LIONS: Are a tough matchup, giving up 2.9 yards per rush to running backs on the season. That ranks top-three, as does their yards before contact allowed (0.49)
and runs of 10-plus yards allowed (three). Forgive them for the 2.38 yards after contact per rush allowed -- that's only fifth-best in the league. Only Kenneth
Walker, on the strength of two touchdowns, has had more than five non-PPR and nine PPR points against these Lions.
</item 76>

<item 77>
NFL Fantasy Football Week 6 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-12T21:23:16+00:00)
You guessed it -- McLaughlin (6.1 yards per carry).

CHIEFS: Tend to build leads on their opponents. They also have given up just four rushing plays of 12-plus yards this season, and that includes a fake punt run
by the Vikings last week. Kansas City has given up one rushing TD to a running back this season. Impressive numbers considering they're giving up 4.5 yards per
carry and rank in the bottom-10 in yards before contact and yards after contact. The secret? Teams don't run the ball that much against them (16.4 attempts per
game, fourth-fewest) because ... they tend to build leads on their opponents.

CHIEFS: David Montgomery and Alexander Mattison are the only two running backs to tally at least 10 Fantasy points (PPR or non-PPR) against Kansas City this
year. Both scored.
</item 77>

<item 78>
Fantasy Football Week 16 Wide Receiver Preview: Waiver adds, DFS plays, and more (CBSSports.com, 2023-12-19T20:51:32+00:00)
I'm hoping for a bounce back in Week 16. Dontayvion Wicks WR GB Green Bay • #13

Age: 22 • Experience: Rookie Weekly Breakdown MATCHUP @ CAR GB -5 O/U 37.5 OPP VS WR 22nd WR RNK 35th ROSTERED 15% YTD Stats REC 31 TAR 49 REYDS 491 TD 1 FPTS/G
6.5 If Christian Watson and Jayden Reed remain out then Wicks might just be the best option available. He should certainly be started in more leagues than he is
currently rostered in. His seven targets in Week 15 were a career high and part of a trend. His snap share has increased for three weeks in a row.

WR Preview DFS Plays

Top DFS Play (WR Preview) Projections powered by Sportsline CeeDee Lamb WR DAL Dallas • #88

Age: 24 • Experience: 4 yrs.
</item 78>

<item 79>
NFL Fantasy Football Week 6 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-12T21:23:16+00:00)
There's nothing to suggest Moss is struggling, which means his playing time doesn't figure to dwindle.

THIS WEEK: Colts coach Shane Steichen said he's ramping up Taylor's workload in practice, and the official team website believes Taylor's snap share will
increase both this week and as the season goes on. But Moss still figures to be the lead back and handle a majority of high-value touches.

JAGUARS: Last week's showing against a Bills run defense that was disinterested in running the ball helped push their run defense numbers to new heights. On the
season they've given up one touchdown on the ground, 3.6 yards per carry, a 6.0% explosive run rate and 2.27 yards after contact per rush to running backs. Only
the explosive run rate doesn't register among the top-six across the league.
</item 79>

<item 80>
NFL Fantasy Football Week 5 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-04T21:56:25+00:00)
I'd rather start Breece Hall, Rhamondre Stevenson, Dameon Pierce and Jaleel McLaughlin (if Javonte Williams sits out).

LAST WEEK: Harris had his first game with over 100 total yards against the Texans and averaged 5.1 yards per rush, just shy of the season-high he technically
had on six carries against the 49ers in Week 1. A 32-yard reception helped him cross the triple-digit barrier.

HARRIS: This might shock you, but Harris is averaging 4.3 yards per run this year with a 12.2% explosive run rate. Would you believe the explosive run rate,
which reflects the frequency of his 12-plus-yard runs, is fifth-best among 44 qualifying running backs?! He's also 13th in avoided tackle rate and top-10 in
yards after contact per carry at 3.53.

TOUCHDOWNS: Harris has zero.
</item 80>

<item 81>
NFL Fantasy Football Week 6 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-12T21:23:16+00:00)
Both scored.

BRONCOS: When McLaughlin, Williams and Perine were all part of the Broncos game plan in their Week 3 blowout loss, Williams had the most work (13 touches) and
the most PPR points (8.5). The only time a Denver back had 15-plus touches came in Week 1 when Williams had 17 against the Raiders (and had 9.7 PPR points, a
season-high).
</item 81>

<item 82>
Seahawks-Cowboys Fantasy Start 'Em Sit 'Em Week 13: Should you start Zach Charbonnet, Jake Ferguson on TNF? (Sporting News, 2023-11-30T21:31:00+00:00)
This is the good advice that you should take: Be wary of anyone who's rushed for just 47 yards two straight weeks despite getting significant duty with Kenneth
Walker III (oblique) on the shelf. Charbonnet's also not doing much in the passing game, and this is as bad of a matchup as San Francisco was last week. You
should do better, even in a big bye.

WEEK 13 FANTASY ADVICE:

Sleepers | Busts | Projections | Start-sit

Start: WR DK Metcalf

Metcalf can flash big plays and stay involved in games...until he disappears when Smith can't get the ball to him. Smith might be judicious about throwing his
way with second-year corner sensation DaRon Bland in coverage. That said, you simply can't sit DeKaylin Zacharius against anyone because of his big-game
ability.
</item 82>

<item 83>
N.F.L. Executives on the Biggest Questions Facing Super Bowl Contenders (The New York Times, 2023-10-16T23:58:53+00:00)
• Detroit Lions: Have they fixed their defense?

What I think: The Lions are 11.3 EPA per game better on defense from last season to this season, the second-largest gain for any team since last season. The
schedule has something to do with that, but the 2022 defense was so bad, it couldn’t stop Carolina. Detroit will win the NFC North and return to the playoffs,
which will make this season successful no matter what happens from there. But we won’t find out until the postseason just how much better the defense might be,
because the remaining schedule features so many weak offenses. That includes Chicago twice, Green Bay, Denver and Las Vegas.

Exec comment: “Yes, I do buy them. This is a league full of teams who do not really know who they are week to week. Detroit knows who they are, and they play to
their identity. Anytime you know who you are and play to that style, you have a chance to be dangerous.”
</item 83>

<item 84>
Fantasy Football Week 13 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-11-30T22:10:58+00:00)
In their past three games, they gave up 4.1 yards per rush to the Ravens (just 6.3% of their runs went 10-plus yards), 7.8 yards per rush to the Steelers thanks
to a long TD run by Jaylen Warren (23.8% of their runs went 10-plus yards) and 4.6 yards per rush to the Broncos (18.5% of their runs went 10-plus yards).
Playing more snaps because the Browns offense has been inept isn't the reason for it either because they struggled with the run in the first half against all
three teams (over 5.3 yards per carry allowed against each). It's silly to say the Rams passing game is struggling when Matthew Stafford just threw four
touchdowns, but Cooper Kupp looked like he played hurt last week and Puka Nacua didn't look that much better.
</item 84>

<item 85>
Fantasy Football Week 13 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-11-30T22:10:58+00:00)
The fact that he has 13 total targets in his past three games combined makes it tough to buy into him suddenly getting 10-plus targets this week. The matchup is
also a problem -- the Browns have the defensive backs to hang tight with a slower-than-normal Kupp in man-to-man coverage, which they play in a lot. The Browns
also have the pass rush to force Matthew Stafford to get the ball out quickly and without much distance. I'm not that nervous to bench Kupp for Rashee Rice,
Courtland Sutton and both of the top Seattle WRs.
</item 85>

<item 86>
Christmas Day preview: 49ers, Ravens square off in potential Super Bowl sneak peek (Yardbarker, 2023-12-24T23:34:39+00:00)
McCaffrey, who seems poised to win the Offensive Player of the Year Award, leads the NFL in rushing (1,292 yards) and total yards per game (128.6) this season
and is tied for first in touchdowns (20).

Ravens TE Isaiah Likely thriving in the starting role: The Ravens offense hasn't lost a step since losing one of the team's top weapons in tight end Mark
Andrews (ankle) in Week 11, largely due to Likely's emergence.

The second-year tight end has averaged 64.3 yards over his last three games, which is particularly noteworthy since Andrews averaged 54.4 yards per game before
sustaining his injury.

Should we expect a defensive battle?: While both teams boast offenses that rank in the top five in yardage and scoring, this may not turn into a shootout.
</item 86>

<item 87>
NFL Fantasy Football Week 6 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-12T21:23:16+00:00)
Brissett didn't throw downfield as much as Walker, contributing to a higher ADOT for Walker, but not by much (8.8 to 8.3). Both quarterbacks were right around
6.5 yards per pass attempt.

49ERS: Oh yeah, the matchup isn't great. No team allows fewer yards per catch to wideouts than the Niners (9.7), and they're thrown at a ton by offenses that
have to play from behind. Only three receivers -- Puka Nacua (20 targets), Marquise Brown (10 targets) and Michael Wilson (two touchdowns) have had more than 15
PPR points against them. And only one other receiver -- Tutu Atwell (nine target) -- had more than 12 PPR points.
</item 87>

<item 88>
NFL Fantasy Football Week 5 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-04T21:56:25+00:00)
And truthfully, the only quarters they've actually played heavy zone have been at the end of blowout wins over the Giants and Jets. If they keep that up they'd
look completely different than in their playoff meeting against Purdy in January when they played heavy zone for nearly 86% of their snaps.

PURDY: Has been sensational against both man and zone coverages this season. His completion rate and yards per attempt are noticeably higher against zone, but
four of his five passing touchdowns have come versus man coverage. However, Purdy's been at his worst when pressured (like most quarterbacks), completing 48.6%
of his throws for 5.6 yards per attempt with a gaudy 18.9% off-target rate.

LAST YEAR IN THE PLAYOFFS: The Cowboys pressured Purdy on 48.5% of his drop backs despite blitzing just 21.2% of the time.
</item 88>

<item 89>
Fantasy Football Week 13 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-11-30T22:10:58+00:00)
Weekly Breakdown MATCHUP @ DAL DAL -9 O/U 47.5 OPP VS WR 17th PROJ PTS 11.1 WR RNK 22nd YTD Stats REC 43 TAR 80 REYDS 678 TD 3 FPTS/G 12.9 Metcalf has at least
nine targets in each of his past three games and in five of his past six. That kind of usage, especially in a game where the Seahawks figure to pass a bunch,
makes him a borderline must-start. But on the 12 targets he's seen that have traveled at least 15 Air Yards over the past four weeks, Metcalf has caught just
three. More alarming: Five of the targets were knocked away by defenders who were right on him when the ball arrived. He's not really zooming past anyone these
days, and he's not winning contested catches regularly either.
</item 89>

<item 90>
NFL Fantasy Football Week 6 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-12T21:23:16+00:00)
Granted, the Cowboys ranked higher than that last week and got beat anyway.

LAMAR JACKSON: Faced the Browns in their last game, saw a ton of man coverage and was pressured a bunch and he still threw two touchdowns and completed 79% of
his throws for 9.8 yards per attempt. And he did it without his starting left tackle and two of his top receivers. His defense turning over an ill-prepared
Browns offense helped a bunch.

Sit Him (Lineup Decisions) Projections powered by Sportsline Jerome Ford RB CLE Cleveland • #34

Age: 24 • Experience: 2 yrs.
</item 90>

<item 91>
NFL Fantasy Football Week 5 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-04T21:56:25+00:00)
PURDY: Has hit at least 20 Fantasy points in 8 of his 9 regular-season starts, and it really should be 9 for 9 because of that first touchdown that was ruled a
lateral back in Week 2. If you include the two playoff games he completed, Purdy has delivered at least 20 in 9 of 11 games. And in 5 of those 9 he hit at least
23 Fantasy points including each of his past two games.

COWBOYS: Dallas' defense lost top cornerback Trevon Diggs and they're still red hot. They lead the league in pass rush pressure rate this season (55.6%) and
they're in the top-10 in blitz rate (31.9%). It's that pressure that's helped the Cowboys rank in the top-five in completion rate allowed (57.4%), passing
touchdowns allowed (two) and missed tackles on pass plays (just five, that's best in football).
</item 91>

<item 92>
Fantasy Football WR PPR Rankings Week 13: Who to start, best sleepers at wide receiver (Sporting News, 2023-11-29T20:27:00+00:00)
Tyreek Hill (@ Commanders in Week 13) did his thing despite a tough matchup in Week 12, resulting in Miami's WR1 returning to the top spot against a putrid
Washington stop unit. CeeDee Lamb (vs. Seahawks) isn't far behind, as Dallas' WR1 aims to maintain his efficient play against Seattle.

Amon-Ra St. Brown (@ Saints) has been a target hog the past several weeks, seeing at least nine looks in each of his past six games. Don't expect his workload
to decrease in a bounce-back game for the Lions. Keenan Allen (@ Patriots) looks to continue his dominant ways, as Justin Herbert's go-to target routinely has
double-digit target potential.

Mike Evans (vs. Panthers) sits in our top five for Week 13, totaling four TDs over the past three weeks. A.J. Brown (vs.
</item 92>

<item 93>
NFL Fantasy Football Week 6 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-12T21:23:16+00:00)
He also ducked under an end-zone target instead of leaping for it. It's worth noting that a 36-yard crosser he ran against the Chargers flies in the face of
this as he separated from man coverage with speed and technique, then turned upfield and ran outside before racing past a defender and diving for the pylon.

OSBORN: Has a 50% catch rate on the year and that includes whether he faces zone coverage or man coverage. He sees more targets per route run against man
coverage and averages more yards per catch and yards after catch against man coverage. Over his career, targets do seem to come more frequently against man
coverage and he has been more explosive with them.

BEARS: Play zone coverage at the fourth-highest rate of any defense this season (79.2%). And because Cousins is much more explosive of a passer against man
coverage, expect Chicago to keep up their zone defense.
</item 93>

<item 94>
Fantasy Football Week 13 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-11-30T22:10:58+00:00)
Fantasy Football is all about the matchups. Even though you drafted your team with certain hopes and intentions, your weekly lineup decisions shouldn't be
determined by the order you picked your players in. You need to check who your players play and make sure you've got the right guys in -- and the wrong guys
out.

It's too early to be absolutely sure on which matchups will be easy and which ones will be tough, but we can take some educated guesses based on healthy
personnel, defensive schemes, track records and key details of offenses. The things we know can help us minimize the impact of the things we don't know. This
should lead to better decisions being made.

We'll go through every game and highlight the players who aren't obvious starts and sits (because you don't need to be told to start Christian McCaffrey).
</item 94>

<item 95>
NFL Fantasy Football Week 5 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-04T21:56:25+00:00)
Sit Him (Lineup Decision) Projections powered by Sportsline Tank Dell WR HOU Houston • #13

Age: 24 • Experience: Rookie Weekly Breakdown MATCHUP @ ATL ATL -2 O/U 41.5 OPP VS WR 8th PROJ PTS 11 WR RNK 27th YTD Stats REC 16 TAR 24 REYDS 267 TD 2 FPTS/G
14 Dell's upside for one huge catch keeps him in play as a middle-of-the-pack flex, but you should be willing to consider other receivers with safer floors and
nearly as much upside. And if the game is low-scoring or a one-sided win for Houston (don't laugh, they just did it last week), it further lowers Dell's
ceiling. I'd rather start Romeo Doubs, Christian Watson and Adam Thielen.

LAST WEEK: The Texans didn't need him!
</item 95>

<item 96>
Fantasy Football Week 13 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-11-30T22:10:58+00:00)
Weekly Breakdown MATCHUP vs SEA DAL -9 O/U 47.5 OPP VS TE 1st PROJ PTS 8.3 TE RNK 11th YTD Stats REC 40 TAR 58 REYDS 421 TD 4 FPTS/G 9.6 If you start Ferguson,
you're basically hoping for a touchdown grab or for him to see a ton of targets in a come-from-behind effort. The Cowboys are expected to lead this matchup
pretty handily, which can squash the upside for a lot of yardage from Ferguson. As for his chances of scoring, 37% of his targets (7 of 19) over the past five
weeks have come in the red zone. That's good. But the Seahawks have allowed one touchdown to a tight end all season (David Njoku, Week 8)! Mark Andrews had a
great game against Seattle, but even he didn't score.
</item 96>

<item 97>
Fantasy Football RB PPR Rankings Week 14: Who to start, best sleepers at running back (Sporting News, 2023-12-07T22:09:00+00:00)
Rams) and think he has the potential to smash Los Angeles at home this week. The speedster has quietly racked up over 300 yards and two TDs over the Ravens'
past four games, and with Lamar Jackson facing Aaron Donald and a still-talented pass-rush, John Harbaugh's squad might turn to the rookie to move the chains
between the 20s. All we need is one or two big chunk gains to make him a worthwhile start — he can run friggin' 22 miles per hour!

WEEK 14 DFS:

DK lineup | FD lineup | Best values | Best stacks

Who are the biggest potential fantasy RB busts in Week 14?

While he just barely cracks our top 25 thanks to his passing-down usage, James Cook (@ Chiefs) doesn't thrill us at Arrowhead against the pissed-off defending
champs.
</item 97>

<item 98>
Christmas Day preview: 49ers, Ravens square off in potential Super Bowl sneak peek (Yardbarker, 2023-12-24T23:34:39+00:00)
The Ravens defense has allowed the fewest points per game this season (16.1) and second-fewest total yards (4,030), while San Francisco's unit ranks second in
points (16.7) and 10th in yardage (310 per game).

The 49ers defense might be slightly inferior statistically, but it could have more of an impact in this matchup. San Francisco has racked up an NFL-leading 25
turnovers and has given up the second-fewest rushing yards (1,252), which should help slow down the team with the league's most effective running game.
</item 98>

<item 99>
Fantasy Football Week 16 Wide Receiver Preview: Waiver adds, DFS plays, and more (CBSSports.com, 2023-12-19T20:51:32+00:00)
Projected stats for all starting quarterbacks are available at Sportsline.
</item 99>

<item 100>
Fantasy Football WR PPR Rankings Week 13: Who to start, best sleepers at wide receiver (Sporting News, 2023-11-29T20:27:00+00:00)
Geno Smith's struggles complicate their fantasy prospects, as well.

What are the biggest WR injuries in Week 13?

Entering Week 12, Tee Higgins (hamstring), Noah Brown (knee), Treylon Burks (concussion), and Kadarius Toney (ankle, hip) all failed to suit up due to nagging
injuries. For now, we've placed Higgins and Brown in our Week 13 list and omitted Burks and Toney. We'll also be eyeing the Chargers' Josh Palmer (knee), who's
eligible to come off the IR this week but probably won't suit up against New England.

Last week, the most notable WRs to exit their respective games with injuries include Chris Olave (concussion), Amari Cooper (ribs), Demario Douglas
(concussion), and Rashid Shaheed (thigh). We've left all four WRs out of our initial Week 13 list but will closely monitor their teams' injury reports to gauge
their playing prospects.
</item 100>
=== END ITEM id=bc5a7a42782f72b9 ===
