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

This file has 2 items.

=== ITEM id=af203a45a0fe492c ===
Gold sentences and facts:
Sentence 0: After turning in three limited practices this week, we'll trust that Hockenson is ready to go, especially since he continued to play through the ailment in Week 9 as a big part of the offense with Josh Dobbs under center.
  Fact 0.0: Hockenson has had three limited practices this week.
  Fact 0.1: The writer will trust that Hockenson is ready to go.
  Fact 0.2: Hockenson continued to play through the ailment in Week 9.
  Fact 0.3: Hockenson was a big part of the offense in Week 9.
  Fact 0.4: Josh Dobbs was under center for the offense when Hockenson played in Week 9.
Sentence 1: Kincaid was a limited participant in practice on Wednesday due to a shoulder injury, but it’s been reported he’s likely to suit up on Saturday at less than 100%.
  Fact 1.0: Kincaid was a limited participant in practice on Wednesday.
  Fact 1.1: Kincaid's limited practice participation was due to a shoulder injury.
  Fact 1.2: It has been reported that Kincaid is likely to suit up on Saturday.
  Fact 1.3: It has been reported that Kincaid is likely to play on Saturday at less than 100%.

Retrieved items:
<item 1>
NFL Week 13 injuries: Browns to start Joe Flacco with Dorian Thompson-Robinson out, Saints' Olave questionable (CBSSports.com, 2023-12-01T22:52:36+00:00)
The wideout still has another hurdle to pass to be cleared to play on Sunday so his status is still very much up in the air.

Falcons (-2) at Jets

Atlanta is relatively healthy coming into this matchup. Hollins, who injured his ankle in Week 9, participated on a limited basis every day this week, which is
encouraging for his prospects to play on Sunday. As for Hughes, he practiced fully all week after suffering a hand injury last weekend against the Saints.
Linebacker Nate Landman (thigh) missed practice on Wednesday but will play vs. New York.

Rodgers was once again limited in practice but, as expected, will not be activated this week and will be sidelined as he continues his recovery from a torn
Achilles just a few months ago. Meanwhile, Cook popped up on the injury report due to a shoulder injury, but head coach Robert Saleh told reporters Friday that
the back should be fine.
</item 1>

<item 2>
NFL Week 13 injuries: Browns to start Joe Flacco with Dorian Thompson-Robinson out, Saints' Olave questionable (CBSSports.com, 2023-12-01T22:52:36+00:00)
Brown is also questionable but did not practice Friday due to a knee injury.

Panthers at Buccaneers (-5.5)

Carolina may be down two tight ends this weekend as the club has ruled out Hurst and is listing Tremble as doubtful. This is the third straight game Hurst will
miss. The team will also be without Bell and will likely look to rookie Alex Cook to start. Meanwhile, corner CJ Henderson (concussion) was a full participant
on Friday and carries no designation heading into the weekend.

Tampa Bay is pretty banged up coming into Week 13, particularly on the defense side of the ball. The Bucs will be without key pieces David, Dean, and White when
they play host to Carolina. Meanwhile, Godwin did not practice Friday after being added to the injury report on Thursday with a neck injury.

49ers (-3) at Eagles

Each of the 49ers' questionable players were limited Friday and did not practice the previous two days.
</item 2>

<item 3>
NFL Fantasy Football Week 5 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-04T21:56:25+00:00)
Seven of his nine touchdowns have headed that way too. This is a problem for Wilson.

Start 'em 10.0-7.00 6.99 - 5.01 5.00 - 0 Sit 'em

Start Him (Lineup Decisions) Projections powered by Sportsline Jake Ferguson TE DAL Dallas • #87

Age: 24 • Experience: 2 yrs. The only reason to give Ferguson the nod is because he's a safety valve for Prescott to lean on. It's a volume play where you're
counting on Ferguson to catch six or seven balls and average around 10 yards per catch. It makes him much more appealing in PPR than non-PPR, which is why I'm
OK starting him over Dallas Goedert and Kyle Pitts.
</item 3>

<item 4>
NFL Fantasy Football Week 6 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-12T21:23:16+00:00)
But they float around league average in catch rate allowed (77.8%, below league average) and yards per catch (8.9, better than league average). However, no team
has allowed more first-down receptions to tight ends (17). Keep in mind, this defense hasn't seen tight ends better than Evan Engram, Sam LaPorta or Dalton
Schultz.

COMMANDERS: Throw to their tight ends on 26.3% of their throws, fourth-most in football. But that hasn't always meant a huge target share for Thomas. In Weeks 1
and 5 he saw at least 22.9% of Sam Howell's targets. He left Week 2 after suffering a concussion and was absent in Week 3; then was not part of the game plan
much in Week 4 (7.7% target share against Philadelphia). Perhaps it's all health related and Thomas' volume from last week will stick.
</item 4>

<item 5>
Fantasy Football RB Rankings Week 14: Who to start, best sleepers at running back (Sporting News, 2023-12-07T22:07:00+00:00)
The injury highlights the importance of handcuffs, as Zack Moss has put up multiple top-10 performances with and without Taylor this season.

Seattle's Kenneth Walker III (oblique) missed Week 12 and 13, and we're not confident in his ability to suit up against the hard-hitting 49ers coming off a big
win in Philly. Backup Zach Charbonnet exited early because of a knee injury against the Cowboys, putting his status in doubt, too. If both are out, DeeJay
Dallas would start and lead the backfield in touches.

Green Bay's Aaron Jones has a sprained MCL, and sources have revealed that the Packers continue to consider him week-to-week. He only logged four carries before
sustaining the injury in Week 11, then he sat for the Packers' Thanksgiving Day tilt with the Lions and their Sunday Night Football win over Kansas City. Why
rush him back if Green Bay is already winning games?
</item 5>

<item 6>
Fantasy Injury Updates Week 10: News on Justin Jefferson, Ja'Marr Chase, T.J. Hockenson & Curtis Samuel (Sporting News, 2023-11-11T21:17:00+00:00)
ET, according to ESPN's Jeremy Fowler. The receiver was limited in practice thanks to the toe injury that has plagued him since leaving Week 8's game against
the Eagles, but barring a setback, he will return to action this weekend.

The receiver was working his way into flex territory with a couple of good performances before picking up the injury. However, we wouldn't consider him to be
worthy of a fantasy start this week unless you're in a bind. It's unclear whether Samuel will regain the same workload or if he'll be playing in limited action
as he works his way back. Jamison Crowder emerged in Samuel's absence and Jahan Dotson has seen his production drastically increase, as well. The wait-and-see
approach is the preferred one heading into Week 10.
</item 6>

<item 7>
NFL DFS picks Week 15: Best sleepers, value players for FanDuel, DraftKings, SuperDraft lineups (Sporting News, 2023-12-13T20:32:00+00:00)
Stroud in the league's concussion protocol and Nico Collins (calf) joining Tank Dell (fibula) on the Texans' injury list, Houston suddenly finds itself without
many impact skill-position players. Singletary, however, has been pretty damn good and showed us last week that he's the main man over Dameon Pierce in DeMeco
Ryans' backfield. He had 13 carries for 65 yards and a TD against the Jets, while Pierce totaled just five touches for seven yards. Don't be scared of
Tennessee's run D because it ain't what it used to be under Mike Vrabel. Raheem Mostert and De'Von Achane just posted 35.1 PPR points against the Titans, Chuba
Hubbard had 20.2 a few weeks ago, and Rachaad White had 17.8 against them in Week 10.
</item 7>

<item 8>
NFL Best Bets Week 9: Colts clobber Panthers, Dolphins drop Chiefs in Germany (Sporting News, 2023-11-03T20:56:00+00:00)
Expect more of the same from QB Justin Herbert and his stud back against the Jets, who have allowed 46 receiving yards per game to RBs (sixth-most in the NFL).
They'll try to shut him down this Sunday, but it would feel so empty without AE!

Jahan Dotson, WR, Commanders @ Patriots — OVER 45.5 receiving yards

We've come full circle with Dotson, who made our preseason sleepers column then faded into early-season obscurity only to emerge again over the past two weeks.
Curtis Samuel was clearly not right last week, and the second-year wideout more than made up for it. Dotson caught 8-of-10 targets for 108 yards and a TD, a
stellar encore to his Week 7 line of five catches for 43 yards. The banged-up Patriots defense will be primarily focused on stopping stud Commanders receiver
Terry McLaurin, which should open up plenty of opportunities for Dotson.
</item 8>

<item 9>
NFL Fantasy Football Week 5 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-04T21:56:25+00:00)
Terrell on Collins and make things a little harder on the Texans. Terrell shadowed a receiver for the first time this season in Week 4, contributing to Calvin
Ridley's modest day. It ultimately would mean multiple bodies on Dell.

Sit Him (Lineup Decisions) Projections powered by Sportsline Kyle Pitts TE ATL Atlanta • #8

Age: 23 • Experience: 3 yrs. Weekly Breakdown MATCHUP vs HOU ATL -2 O/U 41.5 OPP VS TE 18th PROJ PTS 7.6 TE RNK 13th YTD Stats REC 11 TAR 21 REYDS 121 TD 0
FPTS/G 5.8 If Coach Smith is right, then holding on to Pitts in Fantasy makes sense. But it's hard to see the good in his game so far this season beyond him
finding holes in zone coverage.
</item 9>

<item 10>
NFL Fantasy Football Week 5 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-04T21:56:25+00:00)
THREE WEEKS IN A ROW: With seven or fewer targets, Goedert has failed to get even 50 yards receiving. His target volume dipped from seven in Weeks 2 and 3 to
four in Week 4. Goedert had the same amount of targets as D'Andre Swift.

NICK SIRIANNI: The Eagles coach again said "[w]e need to get him the football," citing that plays designed for him wound up going to other Eagles because of how
defenses played Goedert.

FILM: If there's something teams are doing to take away Goedert, I didn't see it on any of his nine third-down routes last week. Goedert still looked sluggish
in his movements and didn't run a ton of deeper routes, though there were a couple where he did get open but Jalen Hurts had better opportunities elsewhere.
</item 10>

<item 11>
NFL Fantasy Football Week 6 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-12T21:23:16+00:00)
BUCCANEERS: Only one quarterback, Jalen Hurts, has posted over 20 Fantasy points this season on the Bucs. And only one quarterback, Kirk Cousins (on 44
attempts) threw two touchdowns against them.

If the game were in Detroit I wouldn't hesitate to call Goff a must-start. But the Buccaneers defense deserves some credit for holding up against their
opponents. Plus, Goff's two big games came against bad secondaries and suspect pass rushes. This one is a little tougher. I think Goff is good enough to start
ahead of Trevor Lawrence, Sam Howell and Geno Smith, but I would put Matthew Stafford and Brock Purdy ahead of him.

Start 'em 10.0-7.00 6.99 - 5.01 5.00 - 0 Sit 'em

Flex Starter (Lineup Decisions) Projections powered by Sportsline Garrett Wilson WR NYJ N.Y.
</item 11>

<item 12>
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
</item 12>

<item 13>
NFL Fantasy Football Week 6 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-12T21:23:16+00:00)
D.J. Turner is expected to replace Awuzie.

GENO SMITH: Has a 47.8% completion rate and a lowly 4.1 yards per attempt average against man coverage so far this season. He's also barely thrown deep (4.3% of
his attempts went 20-plus Air Yards). A huge reason for the low numbers is because his pass protection has been dreadful. The anticipated return of left tackle
Charles Cross should help him have more time, which in turn helps his receivers.

OUTSIDE WRs: Have caught just 50% of their targets against the Bengals this season, third-lowest in football. However, when those receivers have caught the ball
they've averaged 17.9 yards per catch, which is the third-most in the league. The Bengals defense has really only let up big games to receivers who catch
touchdowns.
</item 13>

<item 14>
NFL Fantasy Football Week 6 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-12T21:23:16+00:00)
DEEP PASSES: The Commanders have allowed the seventh-most receptions of throws that traveled 15-plus Air Yards this season with 19. Of those 19, only two were
caught by tight ends (Cole Kmet, Dallas Goedert) and neither went for more than 23 yards.

Start 'em 10.0-7.00 6.99 - 5.01 5.00 - 0 Sit 'em

Sit Him (Lineup Decisions) Projections powered by Sportsline K.J. Osborn WR MIN Minnesota • #17

Age: 26 • Experience: 4 yrs.
</item 14>

<item 15>
Fantasy Football Week 13 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-11-30T22:10:58+00:00)
Easy peasy!

Start 'em 10.0-7.00 6.99 - 5.01 5.00 - 0 Sit 'em

Both teams had a week of rest, so don't treat this like a typical Thursday game. But do expect the Cowboys defense to keep the pressure on Geno Smith and a
Seattle offense that's struggled to gain consistency over the last month. Smith actually did OK completing passes last week when pressured (60% completion rate)
but not at an efficient level (5.4 yards per attempt). Just two QBs have exceeded 250 yards and four QBs have thrown multiple touchdowns against Dallas this
year; I don't like Smith's chances to do either, which hurts any upside case for any Seahawks receiver. By the way, Curtis Samuel was the first receiver to get
100 yards on the Cowboys this year, and it took him 12 targets and 9 catches to get there.
</item 15>

<item 16>
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
</item 16>

<item 17>
Fantasy Football WR PPR Rankings Week 14: Who to start, best sleepers at wide receiver (Sporting News, 2023-12-07T22:11:00+00:00)
What are the biggest WR injuries in Week 14?

In Week 13, Demario Douglas (concussion) and Rashid Shaheed (thigh) missed game action, and Tank Dell (leg), Amari Cooper (concussion), Christian Watson
(hamstring), and Marquise Brown (heel) exited early from their respective games.

UPDATE: Christian Kirk suffered a groin injury on Monday night, and he figures to be more doubtful than questionable for this week's game against Cleveland. If
he's out, Zay Jones and Tim Jones will see more targets, but with Trevor Lawrence (ankle) also hurt, it's tough to count on either.

UPDATE: Douglas (concussion) has been ruled OUT for Week 14.

As usual, we'll update these WR PPR rankings throughout the week, so check back for the latest player movement and analysis.
</item 17>

<item 18>
NFL Fantasy Football Week 6 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-12T21:23:16+00:00)
Tack on the ascending secondary the Falcons seem to have and the Commanders' diffusion of targets and McLaurin is at best a low-end flex. I'd rather roll the
dice on Drake London, Tyler Lockett or Garrett Wilson.

LAST WEEK: Hopeful that his target edge over his teammates would keep growing, and even in a favorable game script where the Commanders were trailing by three
scores before halftime, McLaurin only saw five targets against the Bears' heavy zone coverage.

FILM: On the 10 third-down routes McLaurin ran last week, he drew extra coverage on four of them, got open on two of them (Howell threw elsewhere) and was a
decoy on Curtis Samuel's touchdown. What happened? Sam Howell was getting rid of the ball quickly and to shorter distances as the Bears were getting some
pressure on him. By the end of the game McLaurin's body language seemed dispirited.
</item 18>

<item 19>
NFL Week 13 injuries: Browns to start Joe Flacco with Dorian Thompson-Robinson out, Saints' Olave questionable (CBSSports.com, 2023-12-01T22:52:36+00:00)
Washington could be without its starting center this weekend as it plays host to Miami. Larsen, who missed Wednesday's practice due to a knee injury, was
limited on Friday and is officially listed as questionable. Meanwhile, the Commanders will play Week 13 without Forbes Jr., who has been sidelined since
suffering an elbow injury against the Giants in Week 11. Head coach Ron Rivera called the injury "significant" earlier this week.

Broncos at Texans (-3)

Denver's lone player with an injury designation for Week 13 is Mathis, who did not practice Friday after being limited on Thursday. Outside of him, the Broncos
are healthy heading into this matchup with Houston. Wideouts Jerry Jeudy (groin) and Marvin Mims Jr. (hamstring) were both full participants on Friday after
being limited on Thursday.

Dell logged back-to-back days as a limited participant in practice to end the week and is officially questionable.
</item 19>

<item 20>
NFL Fantasy Football Week 5 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-04T21:56:25+00:00)
Sure, even including the Vikings. Doesn't mean it's not a good look.

NOT SURPRISINGLY: Sanders has two runs of 12-plus yards through four games -- and neither went more than 15 yards. It's not all on him -- the offensive line for
the Panthers barely gave him any daylight last week (and for much of the year) and defenses aren't yet scared of getting beat by Bryce Young's arm.

LIONS: Are a tough matchup, giving up 2.9 yards per rush to running backs on the season. That ranks top-three, as does their yards before contact allowed (0.49)
and runs of 10-plus yards allowed (three). Forgive them for the 2.38 yards after contact per rush allowed -- that's only fifth-best in the league. Only Kenneth
Walker, on the strength of two touchdowns, has had more than five non-PPR and nine PPR points against these Lions.
</item 20>

<item 21>
NFL Fantasy Football Week 6 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-12T21:23:16+00:00)
Weekly Breakdown MATCHUP vs WAS ATL -2.5 O/U 42.5 OPP VS TE 10th PROJ PTS 10.3 TE RNK 9th YTD Stats REC 18 TAR 32 REYDS 208 TD 0 FPTS/G 7.8 If last week had
been Week 1 then I would have been encouraged by Pitts' performance and would probably keep expectations high. But we've seen four really measly weeks from
Pitts before Week 5, so forgive me for not rushing back to him as a must-start tight end. The truth is that the matchup is a little tougher for him and for
Ridder, plus teams have beaten the Commanders on deep shots to wide receivers, not lumbering tight ends, so I'm not convinced Pitts will maintain his target
share or develop more explosive plays.
</item 21>

<item 22>
DraftKings Picks Week 14: Best NFL DFS lineup advice for daily fantasy football tournaments (Sporting News, 2023-12-07T22:18:00+00:00)
With Zach Wilson reportedly reluctant to play and further destroy a career hanging on by a thread, that leaves us with Tim Boyle, Trevor Siemian, or the lucky
fan from Section 139 to lead the Jets to yet another embarrassing defeat. As the Texans try to make a playoff push, this one could get ugly.

Best SuperDraft NFL DFS Picks for Week 14

This season, we're also listing our favorite Week 14 DFS lineup on SuperDraft, with some of our picks and analysis overlapping.

For those unfamiliar, SuperDraft's DFS scoring involves point multipliers as opposed to traditional salary-capped or tiered contests. A winning lineup will
likely feature a strong mix of stud fantasy contributors with 1x-1.6x boosts, as well as a few matchup-based sleepers with larger multipliers. Each player is
assigned a scoring multiplier based on their value as determined by SuperDraft. Users can roster any player in their lineup with no restrictions.

Win BIG with SuperDraft!
</item 22>

<item 23>
Fantasy Football Week 13 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-11-30T22:10:58+00:00)
Fantasy Football is all about the matchups. Even though you drafted your team with certain hopes and intentions, your weekly lineup decisions shouldn't be
determined by the order you picked your players in. You need to check who your players play and make sure you've got the right guys in -- and the wrong guys
out.

It's too early to be absolutely sure on which matchups will be easy and which ones will be tough, but we can take some educated guesses based on healthy
personnel, defensive schemes, track records and key details of offenses. The things we know can help us minimize the impact of the things we don't know. This
should lead to better decisions being made.

We'll go through every game and highlight the players who aren't obvious starts and sits (because you don't need to be told to start Christian McCaffrey).
</item 23>

<item 24>
Thursday Night Football FanDuel Picks: NFL DFS lineup advice for Week 4 Lions-Packers single-game tournaments (Sporting News, 2023-09-26T21:53:00+00:00)
As Watson eases his way back in, it's not like Doubs will fall off a cliff in terms of production. He remains relevant with that type of connection forming.

FLEX TE Sam LaPorta, Lions ($10,000). Rookie tight ends aren't typically fantasy-relevant right away, but that's where LaPorta finds himself, becoming a
favorite target of Goff. He's been on an upward trajectory, starting with five targets in Week 1, six in Week 2, and 11 in Week 3. He's totaled 39, 63, and 84
yards, respectively, in those games, scoring a touchdown in the most recent contest. The Packers are a middle-of-the-road group against tight ends, but they
still haven't faced one in a passing offense that lives up to LaPorta's level. On a short week with some banged-up players on offense, LaPorta should be a
heavily involved security blanket.
</item 24>

<item 25>
NFL Fantasy Football Week 6 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-12T21:23:16+00:00)
Half of the six targets were uncatchable. In his previous three with Deshaun Watson he had at least seven targets in each, and at least 16 PPR points in the
past two.

SINCE THEN: Watson hasn't practiced with the team. P.J. Walker was named the starter despite being on the team's practice squad and took reps this week in
preparation for their game against the Niners.

WALKER: Was cut by the Bears after the preseason before linking up with Cleveland. Has five career passing touchdowns in 15 games (seven starts). He's had
multiple passing scores in one NFL game. Only twice has a receiver had 100 yards from Walker (both by D.J. Moore, once in 2020, once in 2022). Two other times
has a WR had at least 70 yards (Curtis Samuel in 2020, Terrace Marshall in 2022).
</item 25>

<item 26>
NFL Fantasy Football Week 6 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-12T21:23:16+00:00)
He's still worth trotting out there over Logan Thomas, Dallas Goedert and Cole Kmet, but keep expectations around 70 yards on six catches.

LAST WEEK: Came alive against the Texans' zone-heavy defense with season-highs in targets (11), catches (seven) and yards (87). This was his second game of his
last three where his target share was over 23%. It was nice to see him as a key part of the offense, but I can't help but notice he played a season-low 53% of
the snaps.

FILM: Two things helped Pitts out: One, he looked a little quicker off the snap and a little faster when running straight. He's still lumbering when he changes
directions in his routes but perhaps he's slowly getting healthier. Two, Desmond Ridder seemed very comfortable throwing into tighter windows to Pitts in zone
coverage as the two had some good timing down, mainly on dig and out-breaking routes.
</item 26>

<item 27>
NFL Week 13 injuries: Browns to start Joe Flacco with Dorian Thompson-Robinson out, Saints' Olave questionable (CBSSports.com, 2023-12-01T22:52:36+00:00)
Bengals at Jaguars (-8.5), Monday

Bengals : CB Cam Taylor-Britt (quad/ankle), D'Ante Smith (knee), Jay Tufele (illness) OUT

: CB Cam Taylor-Britt (quad/ankle), D'Ante Smith (knee), Jay Tufele (illness) OUT Jaguars: CB Tyson Campbell (hamstring), RB Travis Etienne (ribs), TE Brenton
Strange (foot), TE Luke Farrell (toe) QUESTIONABLE

The Bengals got good news on the injury front ahead of Monday's game against the Jaguars. Wide receiver Tee Higgins (hamstring/ankle) and linebacker Logan
Wilson (ankle/foot) are set to play, according to head coach Zac Taylor. Higgins and Wilson were full participants in practice. Cornerback Cam Taylor-Britt hurt
his foot in practice and is out.
</item 27>

<item 28>
2023 NFL Week 11 injury roundup: Rams WR Cooper Kupp, Seahawks RB Kenneth Walker III among stars to exit (CBSSports.com, 2023-11-19T23:51:49+00:00)
Smith did return, however, for the final drive of the game in the losing effort. Head coach Pete Carroll said that Smith will "make it back." When asked if that
means for Thursday's matchup against San Francisco, the coach said, "I would think so."

Sam Howell (Commanders) received medical attention on the sidelines after enduring several hard hits to the upper body on a touchdown run against the Giants. He
was quickly cleared of a potential concussion, however, and remained in the game. Veteran journeyman Jacoby Brissett is Washington's backup.

RB

Aaron Jones (Packers) suited up against the Chargers despite a lingering hamstring issue, only to manage just five carries before suffering a knee injury. The
former Pro Bowler needed help to leave the field and was then carted to the locker room, replaced by A.J. Dillon.
</item 28>

<item 29>
NFL Fantasy Football Week 6 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-12T21:23:16+00:00)
HISTORY: In 20 career games with at least five targets, Osborn has scored at least 12 PPR points in 13 of them. Eleven of those 13 involved him catching a
touchdown. Osborn has two career games with over 100 yards -- and one came last January at Chicago.

Start Him (Lineup Decisions) Projections powered by Sportsline Cole Kmet TE CHI Chicago • #85

Age: 24 • Experience: 4 yrs. If you start Kmet, you're buying into his role as Fields' change-up to D.J. Moore -- and in the case of this week, a change-up to
the Bears' depleted run game. I especially like his red-zone role and his touchdown opportunities. I'm ready to chance it with Kmet over Kyle Pitts, George
Kittle and Dallas Goedert.
</item 29>

<item 30>
NFL Week 13 injuries: Browns to start Joe Flacco with Dorian Thompson-Robinson out, Saints' Olave questionable (CBSSports.com, 2023-12-01T22:52:36+00:00)
Cardinals at Steelers (-5.5)

Despite missing the entire week of practice, the Cardinals are holding out hope for Brown to play by listing him as questionable. McBride is also questionable
after being limited on Thursday and Friday, but told reporters that he's "good to go."

Pittsburgh has listed Adams as its lone player with a game designation to wrap up the week. Safety Minkah Fitzpatrick (hamstring) practiced fully throughout the
week, carries no designation, and told reporters Friday that he will play this weekend.

Colts (-1) at Titans

Colts : CB JuJu Brents (quad), RB Jonathan Taylor (thumb) OUT

: CB JuJu Brents (quad), RB Jonathan Taylor (thumb) OUT Titans: WR Treylon Burks (concussion) QUESTIONABLE

Indianapolis ruled out Brents and Taylor after they both missed the entire week of practice.
</item 30>

<item 31>
NFL Fantasy Football Week 6 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-12T21:23:16+00:00)
Weekly Breakdown MATCHUP vs NO NO -1.5 O/U 42.5 OPP VS WR 14th PROJ PTS 12.8 WR RNK 23rd YTD Stats REC 25 TAR 36 REYDS 467 TD 3 FPTS/G 17.9 I'm not sure the
Saints will back off of their aggressive man-to-man coverage even if Stroud's stats suggest they should, and that could open the door for Collins to pick up
some targets. Unfortunately, neither of his matchups are particularly favorable. I could see Collins getting a lot of short targets and for that reason alone
I'd be OK starting him as a flex option in PPR ahead of Tyler Lockett and Chris Godwin, but behind Jakobi Meyers, Jaylen Waddle and Gabe Davis.

LAST WEEK: Even in a game where the Texans lost Tank Dell to a concussion before the half, Collins couldn't get much going.
</item 31>

<item 32>
Fantasy Football Week 13 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-11-30T22:10:58+00:00)
It gets worse: The Panthers offense can't score and their run defense is so bad, leaving pass-negative game scripts for their opponents. Just two teams (the
Saints and Dolphins) have rostered multiple receivers with 10-plus PPR points against the Panthers in 2023. I'd sit Godwin for Josh Downs, Brandin Cooks,
Christian Watson and Curtis Samuel (especially if PPR).

Start 'em 10.0-7.00 6.99 - 5.01 5.00 - 0 Sit 'em

I wouldn't expect the Rams offensive line to handle the Browns' defensive front like they did the Cardinals' last week, but it must be mentioned that the Browns
run defense isn't anywhere near as dominant as it was earlier this year.
</item 32>

<item 33>
2023 NFL Week 11 injury roundup: Rams WR Cooper Kupp, Seahawks RB Kenneth Walker III among stars to exit (CBSSports.com, 2023-11-19T23:51:49+00:00)
The former All-Pro, who battled a lingering hamstring injury earlier this year, received attention on the sidelines after limping off, then left the game for
further evaluation in the locker room. He was ruled questionable to return. NFL Media reported on Monday that Kupp suffered a low ankle sprain and will be
evaluated going forward.

Darius Slayton (Giants) had a big impact against the Commanders, hauling in a deep score from Tommy DeVito, before exiting with an arm injury. He was ruled out
for the remainder of the contest halfway through the third quarter.

EDGE

Joey Bosa (Chargers) appeared to injure his right leg or foot while held in a block during a first-quarter pass-rushing attempt against the Packers. He limped
to the sidelines and was carted to the locker room, then ruled questionable to return.
</item 33>

<item 34>
NFL Fantasy Football Week 5 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-04T21:56:25+00:00)
The hope is we keep seeing that for Smith, especially if the game is competitive like most Falcons games tend to be. He certainly hasn't done anything to lose
his role. I don't mind chancing it with Smith as a streaming tight end, though I just don't have the conviction to start him over Pitts.

LAST WEEK: Was the second time in three weeks where Smith outproduced Pitts, but the first time it was so lopsided. The difference in snaps played and routes
run is marginal -- Pitts did more of both -- but Smith is taking advantage of shorter routes with a quarterback who doesn't like to challenge downfield all that
much. What's more, Smith averaged more yards per catch not just in Week 4 but also in Week 2, and he was behind Pitts by less than a yard in Week 3 when both
had at least eight targets.

FILM ON PITTS: There's no change in his explosiveness whatsoever.
</item 34>

<item 35>
NFL Fantasy Football Week 5 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-04T21:56:25+00:00)
I think he'll continue to see high-value targets near the end zone and, hopefully, a small uptick in targets. I'd start him as a No. 2 receiver over Terry
McLaurin, any Broncos wideouts and any Packers wideouts.

LAST WEEK: Was so close to having a monster game. He was the designed target on a trick play and saw the ball fall just past his outstretched hands in the end
zone. He also had two targets inside the 10-yard line (his first two of the season) and two end-zone targets including the aforementioned trick play (his first
two of the season). The heavy dose of targets inside the 10 and into the end zone are positive signs that the Titans are trying to get him going as a playmaker.

FILM: Hopkins might be 31 years old but he moved really, really well, creating separation with outstanding footwork and having just enough speed to stay a step
ahead of defenders.
</item 35>

<item 36>
Seahawks-Cowboys Fantasy Start 'Em Sit 'Em Week 13: Should you start Zach Charbonnet, Jake Ferguson on TNF? (Sporting News, 2023-11-30T21:31:00+00:00)
Again, you can't expect more than WR3 production, which is where his overall scoring has been in 2023.

Start: WR Tyler Lockett

While Metcalf is WR34 in terms of total points this season in half-point PPR, Lockett is only two notches higher at WR32. They finished as WR13 and WR18,
respectively, last season, so it's a big dropoff. Lockett also has a decent matchup vs. Stephon Gilmore. He, like Metcalf, remains a volume-based WR3.

WEEK 13 PPR RANKINGS:

QBs | RBs | WRs | TEs | D/STs | Kickers

Sit: WR Jaxon Smith-Njigba

Smith-Njigba can be a bit interesting if you are desperate in a deeper league because he has the matchup on par with the other two wideouts vs. Jourdan Lewis.
</item 36>

<item 37>
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
</item 37>

<item 38>
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
</item 38>

<item 39>
NFL Week 13 injuries: Browns to start Joe Flacco with Dorian Thompson-Robinson out, Saints' Olave questionable (CBSSports.com, 2023-12-01T22:52:36+00:00)
Meanwhile, Pipkins was also limited on Friday.

The Patriots will be down a couple of wide receivers as Douglas and Boutte have been ruled out after not practicing on Friday. For Douglas, he missed the entire
week due to a concussion he suffered against the Giants, while Boutte was downgraded on the injury report on Thursday and missed back-to-back days due to an
illness on top of his shoulder injury.

Lions (-4) at Saints

The Lions just designated Hooker for return, who has been rehabbing a torn ACL he suffered at Tennessee last year, this week so the rookie quarterback was not
expected to be activated this week. Meanwhile, Anzalone did not practice at all this week, so his prospects look bleak for Week 13. As for Jackson, he was a
full participant on both Thursday and Friday.

Head coach Dennis Allen told reporters Friday that despite Olave being a full participant in Friday's practice he is still in concussion protocol though has a
chance to play.
</item 39>

<item 40>
NFL DFS picks Week 15: Best sleepers, value players for FanDuel, DraftKings, SuperDraft lineups (Sporting News, 2023-12-13T20:32:00+00:00)
6x); Tyler Allgeier, Falcons at Panthers (DK: $4,400 | FD: $5,700 | SD: 1.6x)

WEEK 15 PPR RANKINGS:

QBs | RBs | WRs | TEs | D/STs | Kickers

NFL DFS Picks Week 15: WR sleepers, values for DraftKings, FanDuel, SuperDraft

Rashee Rice, Chiefs at Patriots (DK: $6,100 | FD: $6,500 | SD: 1.35x)

With Isiah Pacheco (shoulder) banged up, Rice seems like Kansas City's best skill-position player this side of Travis Kelce. He's certainly the most sure-handed
wideout on the Chiefs (not saying much, but still), and he has electric after-the-catch playmaking ability (which is good, considering Patrick Mahomes has
struggled with deep-ball accuracy).
</item 40>

<item 41>
Fantasy Injury Updates Week 10: News on Justin Jefferson, Ja'Marr Chase, T.J. Hockenson & Curtis Samuel (Sporting News, 2023-11-11T21:17:00+00:00)
After turning back the clocks last week, it now gets darker earlier for the foreseeable future. That darkness allows us to see the stars on a clear night, so
it's only natural to wonder if the pass-catching stars will shine bright in Week 10 of the NFL season. We already know Tee Higgins and Nico Collins are OUT,
while Justin Jefferson, Ja'Marr Chase, and T.J. Hockenson are among the other star pass-catchers beaten up heading into the week. Can we expect them on the
field when the action kicks off Sunday?

Below, we'll give you the latest news on these ailing pass-catchers and break down how their injuries could potentially affect the Week 10 fantasy landscape.

WEEK 10 INJURY UPDATES: Early WRs | RBs

Is Justin Jefferson playing Week 10?

UPDATE: The Vikings will not activate Jefferson before Sunday's game and have officially ruled him OUT, according to NFL Network's Tom Pelissero.
</item 41>

<item 42>
N.F.L. Executives on the Biggest Questions Facing Super Bowl Contenders (The New York Times, 2023-10-16T23:58:53+00:00)
Randy Mueller raised the thought last week, recalling examples from his career as a GM when offering a pick one round higher than a prospect’s actual draft
round proved alluring. Detroit used a third-round choice for Hooker, who is recovering from ACL surgery. Hooker would have been valued higher if healthy. Could
that be true again at some point in the future?

• News that Indianapolis Colts rookie quarterback Anthony Richardson might undergo season-ending shoulder surgery could carry a silver lining. Richardson got to
play enough for the Colts to evaluate where he stands, but if he shuts it down for the season, neither he nor the team would have to suffer through a full
season of growing pains. Gardner Minshew’s three-pick game against Jacksonville on Sunday works against the idea Indy is better positioned to compete with the
journeyman in the lineup, but most coaches and execs I’ve spoken with believe that to be the case in the short term.
</item 42>

<item 43>
NFL Fantasy Football Week 5 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-04T21:56:25+00:00)
FILM ON PITTS: There's no change in his explosiveness whatsoever. He's running at a slower-than-expected pace, but the Jaguars still paid him plenty of
attention defensively, creating a double-whammy that left him without many targets. His longest opportunity was a deep ball from Desmond Ridder that was
well-placed and perfectly leapt for by Pitts, but he couldn't secure it to the ground, nor was he separated from cornerback Tyson Campbell at all. Pitts was
best on shorter routes against zone coverage.

ARTHUR SMITH: Said this week that Pitts isn't at 100%, adding " there are certain things, you see him, and he's doing really well right now and there are
certain things that he can't [do well]. It's been a journey back. He's going to get there."
</item 43>

<item 44>
Fantasy Football Week 13 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-11-30T22:10:58+00:00)
If they're ailing, the Rams would be smart to try and use the run game early and often. Expect a low-scoring game.

STARTS: Kyren Williams, Amari Cooper, Puka Nacua, David Njoku, Jerome Ford,

Kyren Williams, Amari Cooper, Puka Nacua, David Njoku, Jerome Ford, SITS: Cooper Kupp (flex at best), Kareem Hunt, Tutu Atwell, Tyler Higbee, Royce Freeman,
both QBs, both DSTs (low-end options)

Cooper Kupp (flex at best), Kareem Hunt, Tutu Atwell, Tyler Higbee, Royce Freeman, both QBs, both DSTs (low-end options) SLEEPERS: Elijah Moore (PPR flex)

Start Him (Lineup Decisions) Projections powered by Sportsline Amari Cooper WR CLE Cleveland • #2

Age: 29 • Experience: 9 yrs.
</item 44>

<item 45>
NFL Week 13 injuries: Browns to start Joe Flacco with Dorian Thompson-Robinson out, Saints' Olave questionable (CBSSports.com, 2023-12-01T22:52:36+00:00)
Cornerback Cam Taylor-Britt hurt his foot in practice and is out.

For the Jaguars, Travis Etienne (ribs) is also expected to play, Doug Pederson told reporters, despite the running back being limited in practice all week.
Strange was a late addition to the injury report, not practicing Saturday.
</item 45>

<item 46>
NFL Fantasy Football Week 6 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-12T21:23:16+00:00)
And, tight ends have scored three times in the Vikings past three games.

Start 'em 10.0-7.00 6.99 - 5.01 5.00 - 0 Sit 'em

Flex Starter (Lineup Decisions) Projections powered by Sportsline Tyler Lockett WR SEA Seattle • #16

Age: 31 • Experience: 9 yrs. Weekly Breakdown MATCHUP @ CIN CIN -2.5 O/U 45 OPP VS WR 11th PROJ PTS 11.9 WR RNK 36th YTD Stats REC 17 TAR 27 REYDS 157 TD 2
FPTS/G 11.7 If you start Lockett, you're doing it for his upside and on the hope that Smith can have a minor resurgence against man coverage. It may also come
to pass that the Seahawks are trailing in the game and thus have to throw.
</item 46>

<item 47>
Fantasy Football WR PPR Rankings Week 13: Who to start, best sleepers at wide receiver (Sporting News, 2023-11-29T20:27:00+00:00)
Geno Smith's struggles complicate their fantasy prospects, as well.

What are the biggest WR injuries in Week 13?

Entering Week 12, Tee Higgins (hamstring), Noah Brown (knee), Treylon Burks (concussion), and Kadarius Toney (ankle, hip) all failed to suit up due to nagging
injuries. For now, we've placed Higgins and Brown in our Week 13 list and omitted Burks and Toney. We'll also be eyeing the Chargers' Josh Palmer (knee), who's
eligible to come off the IR this week but probably won't suit up against New England.

Last week, the most notable WRs to exit their respective games with injuries include Chris Olave (concussion), Amari Cooper (ribs), Demario Douglas
(concussion), and Rashid Shaheed (thigh). We've left all four WRs out of our initial Week 13 list but will closely monitor their teams' injury reports to gauge
their playing prospects.
</item 47>

<item 48>
NFL Fantasy Football Week 6 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-12T21:23:16+00:00)
He has more upside than Jerry Jeudy, Jahan Dotson and Tutu Atwell but I'd rather take a shot on Drake London, Terry McLaurin or Chris Godwin.

LAST WEEK: Over 19 snaps without Justin Jefferson last week, Osborn ran 18 routes and caught both of his two targets for 30 total yards. The receiving average
was delightful, but the small-sample target share (12.5%) was saddening. What's funny is that Osborn's target share was much higher when Jefferson was playing
(23.3%) but his receiving average tanked (6.3).

FILM: Osborn didn't beat coverage with his speed last week and his burst and agility were just OK. His 22-yard catch came on a Kirk Cousins scramble where he
waited in a spot and Cousins hit him, then Osborn ran inside instead of outside where he might have been able to score.
</item 48>

<item 49>
DraftKings Picks Week 14: Best NFL DFS lineup advice for daily fantasy football tournaments (Sporting News, 2023-12-07T22:18:00+00:00)
Because we splurged for McCaffrey and Kamara, we had to sacrifice salary elsewhere, and London is a high-upside option even in an offense that doesn't throw the
ball with efficiency.

WR Jaxon Smith-Njigba, Seahawks @ 49ers ($4,100). Is a takeover underway in the Seahawks offense? Smith-Njigba is coming off a game against the Cowboys that saw
him collect seven-of-11 targets for 62 yards. It's a step up from a disappointing outing against San Francisco, but Tyler Lockett's involvement seems to be
lessening as of late with the rookie stepping up. It wasn't a great day for the Seattle offense on Thanksgiving against the Niners, but the Seahawks should put
up a better effort against a divisional opponent they know so well. If they're trailing, JSN should see action with a negative game script.
</item 49>

<item 50>
NFL Fantasy Football Week 5 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-04T21:56:25+00:00)
Sit Him (Lineup Decision) Projections powered by Sportsline Tank Dell WR HOU Houston • #13

Age: 24 • Experience: Rookie Weekly Breakdown MATCHUP @ ATL ATL -2 O/U 41.5 OPP VS WR 8th PROJ PTS 11 WR RNK 27th YTD Stats REC 16 TAR 24 REYDS 267 TD 2 FPTS/G
14 Dell's upside for one huge catch keeps him in play as a middle-of-the-pack flex, but you should be willing to consider other receivers with safer floors and
nearly as much upside. And if the game is low-scoring or a one-sided win for Houston (don't laugh, they just did it last week), it further lowers Dell's
ceiling. I'd rather start Romeo Doubs, Christian Watson and Adam Thielen.

LAST WEEK: The Texans didn't need him!
</item 50>

<item 51>
Fantasy Football Week 13 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-11-30T22:10:58+00:00)
0-7.00 6.99 - 5.01 5.00 - 0 Sit 'em

The Lions need an answer for their ugly pass defense -- since their Week 9 bye they've been mostly great against the run (3.1 yards per carry allowed, one RB to
get 15-plus PPR points) but they rank bottom-six in completion rate allowed (68.4%), yards per attempt allowed (8.0), passing ADOT against (8.84 yards) and
defensive Expected Points Added (EPA) per dropback (-0.40, which is league-worst). And of their second-most eight passing touchdowns given up in three games,
five have come on passes of 15-plus Air Yards, which is tied for most in the league. They should catch a break with field stretcher Rashid Shaheed (thigh)
missing the matchup for New Orleans; Chris Olave looks like he will play.
</item 51>

<item 52>
NFL Fantasy Football Week 5 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-04T21:56:25+00:00)
's pass stats against tight ends were much better before the Colts laid some numbers on them.

Start 'em 10.0-7.00 6.99 - 5.01 5.00 - 0 Sit 'em

Sit Him (Lineup Decisions) Projections powered by Sportsline Jordan Addison WR MIN Minnesota • #3

Age: 21 • Experience: Rookie Weekly Breakdown MATCHUP vs KC KC -4 O/U 52.5 OPP VS WR 10th PROJ PTS 9.8 WR RNK 34th YTD Stats REC 13 TAR 20 REYDS 185 TD 2 FPTS/G
10.9 While the Vikings are almost guaranteed to throw a ton against the Chiefs, rolling with Addison remains a risk because he's not the definitive No. 2
target, nor is the matchup any good for him. He's a flex at best who I'd sit for Romeo Doubs, Gabe Davis and Christian Watson.
</item 52>

<item 53>
NFL Fantasy Football Week 5 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-04T21:56:25+00:00)
On the season he's totaled 28 rushes for 134 yards and a score.

Sneaky Sleeper (Lineup Decisions) Projections powered by Sportsline Sam Howell QB WAS Washington • #14

Age: 23 • Experience: 2 yrs. Weekly Breakdown MATCHUP vs CHI WAS -6 O/U 44.5 OPP VS QB 31st PROJ PTS 21.3 QB RNK 14th YTD Stats PAYDS 961 RUYDS 82 TD 5 INT 5
FPTS/G 16.1 The Commanders like limiting Howell's pass attempts in general -- he's only thrown over 31 times in high-scoring shootouts like in Weeks 2 and 4.
While there's potential for that on Thursday, it's not a certainty. Howell could just as easily put up nominal stats to help the Commanders bring home a win.
</item 53>

<item 54>
Fantasy Football RB Rankings Week 12: Who to start, best sleepers at running back (Sporting News, 2023-11-21T21:33:00+00:00)
Both Darrell Henderson and Royce Freeman (@ Cardinals) would be decent dart throws in a week filled with byes, but their floors in a zero-bye week seem too low
to trust them if Kyren Williams is back and at 100 percent. The matchup is juicy, but we have no idea what to expect from a touch-share perspective, so these
guys are no better than desperation, TD-dependent flexes.

Speaking of guys coming back from injury, we saw Khalil Herbert (@ Vikings) return to the field for Chicago against the Lions last week. However, his
performance left a lot to be desired. He finished with just 35 yards on 16 carries and wound up vastly overshadowed by Justin Field's 100-yard rushing effort.
Even worse, D'Onta Foreman scored the Chi's lone rushing TD on the afternoon.

I don't care what your team looks like — you shouldn't be starting Chuba Hubbard or Miles Sanders (@ Titans).
</item 54>

<item 55>
DraftKings Picks Week 14: Best NFL DFS lineup advice for daily fantasy football tournaments (Sporting News, 2023-12-07T22:18:00+00:00)
If they're trailing, JSN should see action with a negative game script.

WEEK 14 STANDARD RANKINGS:

QBs | RBs | WRs | TEs | D/STs | Kickers

TE Brevin Jordan, Texans @ Jets ($3,100). Jordan was a great streaming option and fill-in for the crowd that lost Dalton Schultz (hamstring) on Sunday, hauling
in three-of-four targets for 64 yards against the Broncos. If Schultz is out again, Jordan is a serviceable option that has touchdown upside against a reeling
Jets team that is busy booking vacations or plotting how to find themselves on a different roster in 2024. If Schultz does play, we'll pivot to the Jets' Tyler
Conklin, who is $100 less and has plenty of opportunity in an even better matchup.

FLEX WR Jameson Williams, Lions @ Bears ($3,500).
</item 55>

<item 56>
Fantasy Football TE PPR Rankings Week 14: Who to start, best sleepers at tight end (Sporting News, 2023-12-07T22:14:00+00:00)
For those in deeper leagues, Tyler Conklin (vs. Texans), and Cade Otton (@ Falcons) both have matchups against defenses that entered last week in the top five
for FPPG allowed to TEs. Brevin Jordan (@ Jets) and Tanner Hudson (vs. Colts) also have highly favorable matchups and can work as deep-league PPR plays, though
Jordan's outlook is entirely dependent on whether Dalton Schultz (hamstring) is healthy.

WEEK 14 DFS:

DK lineup | FD lineup | Best values | Best stacks

Who are the biggest potential fantasy TE busts in Week 14?

Dallas Goedert (@ Cowboys) isn't a "must-start" if he returns from his forearm injury this week. That said, he's also not a "must-sit." He had three catches and
50 yards against Dallas in Week 9 before exiting early with the forearm injury, so he can clearly produce against this defense.
</item 56>

<item 57>
NFL Fantasy Football Week 5 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-04T21:56:25+00:00)
He got open on a handful of third-down plays when he didn't get a target last week.

TANNEHILL: Is the top problem Hopkins has right now. Hopkins' 38-yard deep-ball catch last week could have been as many as 50 yards if Tannehill hit him in
stride, and the gimmick-play touchdown would have obviously been a massive point-getter had Tannehill been slightly more on-target there. There was even a
target into the end zone that was a miscommunication between Hopkins and Tannehill. On the season Tannehill has a 13% off-target rate but over the past two
games that number has ballooned to 18%, which is ... yikes.

TITANS: The second problem Hopkins has right now is a Titans offense that's stayed true to their run-first preference -- at least when they win.
</item 57>

<item 58>
Fantasy Football QB Rankings Week 14: Who to start, best sleepers at quarterback (Sporting News, 2023-12-07T22:06:00+00:00)
WEEK 14 DFS:

DK lineup | FD lineup | Best values | Best stacks

What are the biggest QB injuries in Week 14?

Leading into Week 13, Dorian Thompson-Robinson (concussion) was the lone QB injury situation we were closely monitoring. Even if Thompson-Robinson is able to
clear concussion protocol in time for Week 14, we expect Flacco to make his second start as a Brown.

This week, we unfortunately saw Kenny Pickett (ankle) and Derek Carr (head, shoulder, back) exit their respective contests with injuries, complicating their
statuses for Week 14. We've replaced both QBs with backups Mitchell Trubisky and Jameis Winston, respectively, as early indications point toward both missing
time.

UPDATE: Trevor Lawrence suffered an ankle injury on Monday, which will likely keep him out for at least a week and open the door for C.J. Beathard (vs. Browns)
to start.
</item 58>

<item 59>
Fantasy Football TE PPR Rankings Week 14: Who to start, best sleepers at tight end (Sporting News, 2023-12-07T22:14:00+00:00)
Packers) always has a high ceiling as the Giants' de facto No. 1 receiver.

Tyler Higbee (neck) was banged up last week, and he's no sure thing to suit up this week against Baltimore. Even if he does, he's not a recommended play against
the Ravens' top-10 TE defense.

We'll be updating these TE PPR rankings throughout the week, so check back for the latest player movement and analysis.

Fantasy TE PPR Rankings Week 14: Who to start, sit at tight end

Rankings based on full-point PPR formats

Rank Player

1 Travis Kelce, Chiefs vs. Bills

2 T.J. Hockenson, Vikings @ Raiders

3 Sam LaPorta, Lions @ Bears

4 David Njoku, Browns vs. Jaguars

5 Dalton Kincaid, Bills @ Chiefs

6 George Kittle, 49ers vs. Seahawks

7 Isaiah Likely, Ravens vs.
</item 59>

<item 60>
NFL Fantasy Football Week 5 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-04T21:56:25+00:00)
Jonnu even has more red-zone targets than Pitts (by one, three to two). The matchup could create some receiving opportunities, so maybe he catches five passes
for 50 yards, but that's not anything to be excited about. I'd rather start Jake Ferguson, Zach Ertz and Dallas Goedert.

Sneaky Sleeper (Lineup Decisions) Projections powered by Sportsline Jonnu Smith TE ATL Atlanta • #81

Age: 28 • Experience: 7 yrs. Weekly Breakdown MATCHUP vs HOU ATL -2 O/U 41.5 OPP VS TE 18th PROJ PTS 5.2 TE RNK 24th YTD Stats REC 15 TAR 20 REYDS 179 TD 0
FPTS/G 8.2 I feel like Smith is doing what Pitts is supposed to do: get open and make plays after the catch.
</item 60>

<item 61>
Fantasy Football Week 13 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-11-30T22:10:58+00:00)
The uncertainty of how the Panthers will operate moving forward with Brown basically calling the offense without any oversight keeps these players are risky
Fantasy propositions, though it helps a lot that the Buccaneers are dealing with injuries that hurt their run defense (Lavonte David and Devin White will miss
the game) and keep their pass defense from improving (Jamel Dean will miss the game).
</item 61>

<item 62>
Fantasy Injury Updates Week 10: News on Justin Jefferson, Ja'Marr Chase, T.J. Hockenson & Curtis Samuel (Sporting News, 2023-11-11T21:17:00+00:00)
Jefferson (hamstring) is officially listed as "questionable" for Sunday's game against the Saints. Kickoff is slated for 1 p.m. ET, but fantasy owners won't
have to wait long for news on the star receiver. In order to be eligible to play, Jefferson will need to be activated by 4 p.m. ET on Saturday. Earlier in the
week head coach Kevin O'Connell said that it would be "a little aggressive" for Jefferson to get back on the field this week. However, the receiver's three
straight limited practices were enough for the team to at least weigh the decision, but thankfully, an early answer awaits.

WEEK 10 FANTASY ADVICE:

Sleepers | Busts | Projections | Start 'em, sit 'em

Jefferson's fantasy outlook is always through the roof, but it will be interesting to see if that changes with Josh Dobbs under center instead of Kirk Cousins.
</item 62>

<item 63>
Fantasy Football Week 13 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-11-30T22:10:58+00:00)
The fact that he has 13 total targets in his past three games combined makes it tough to buy into him suddenly getting 10-plus targets this week. The matchup is
also a problem -- the Browns have the defensive backs to hang tight with a slower-than-normal Kupp in man-to-man coverage, which they play in a lot. The Browns
also have the pass rush to force Matthew Stafford to get the ball out quickly and without much distance. I'm not that nervous to bench Kupp for Rashee Rice,
Courtland Sutton and both of the top Seattle WRs.
</item 63>

<item 64>
NFL Fantasy Football Week 5 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-04T21:56:25+00:00)
LAST WEEK: On the plus side, two of his targets were in the red zone and a third was a deep contested catch for 17 yards. On the downside, he dropped that
fourth target on a short throw.

GOEDERT: Has three red-zone targets and zero end-zone targets this season.

RAMS: Got stung for two touchdowns to tight ends at Indy last week, one on a 35-yard play where Mo Alie-Cox got matched up with DE-turned-OLB Mike Hoecht. The
other was a five-yard toss to Andrew Ogletree, who ran past Hoecht off the snap for an easy touchdown grab. These guys aren't considered primary tight ends like
Goedert, though, so it's unlikely the Rams would cover Goedert the same way as they covered them (which is to say, they barely did).
</item 64>

<item 65>
Fantasy Football TE PPR Rankings Week 14: Who to start, best sleepers at tight end (Sporting News, 2023-12-07T22:14:00+00:00)
Seahawks

7 Isaiah Likely, Ravens vs. Rams

8 Cole Kmet, Bears vs. Lions

9 Jake Ferguson, Cowboys vs. Eagles

10 Dallas Goedert, Eagles @ Cowboys

11 Kyle Pitts, Falcons vs. Buccaneers

12 Gerald Everett, Chargers vs. Broncos

13 Evan Engram, Jaguars @ Browns

14 Brevin Jordan, Texans @ Jets

15 Tyler Conklin, Jets vs. Texans

16 Cade Otton, Buccaneers @ Falcons

17 Tanner Hudson, Bengals vs. Colts

18 Pat Freiermuth, Steelers vs. Patriots

19 Tucker Kraft, Packers @ Giants

20 Chigoziem Okonkwo, Titans @ Dolphins

21 Dawson Knox, Bills @ Chiefs

22 Jonnu Smith, Falcons vs. Buccaneers

23 Juwan Johnson, Saints vs. Panthers

24 Michael Mayer, Raiders vs.
</item 65>

<item 66>
NFL DFS picks Week 15: Best sleepers, value players for FanDuel, DraftKings, SuperDraft lineups (Sporting News, 2023-12-13T20:32:00+00:00)
Bears (DK: $4,700 | FD: $6,200 | SD: 1.3x)

Njoku went nuclear last weekend, catching six of his eight targets from Joe Flacco for 91 yards and two touchdowns. Now he gets a chance for an encore against
the Bears, who have surrendered the 10th-most PPR points to tight ends this season. No joke, dude — you can trust Njoku!

More solid Week 15 TE values: Logan Thomas, Commanders at Rams (DK: $3,700 | FD: $5,100 | SD: 1.5x); Cade Otton, Buccaneers at Packers (DK: $3,200 | FD: $5,200
| SD: 1.65x); Tucker Kraft, Packers vs.
</item 66>

<item 67>
DraftKings Picks Week 14: Best NFL DFS lineup advice for daily fantasy football tournaments (Sporting News, 2023-12-07T22:18:00+00:00)
We're especially fond of the 22 targets with a leaky secondary once again on tap. Moore cooked the Lions to the tune of almost 23 fantasy points in Week 12, and
being well-rested off the bye means we're likely to see more of the same.

WEEK 14 PPR RANKINGS:

QBs | RBs | WRs | TEs | D/STs | Kickers

WR Drake London, Falcons vs. Buccaneers ($4,600). When we see the Buccaneers on the schedule, we immediately look to see what WR1 is going to have a big day. It
wasn't the greatest feeling in the world seeing the Falcons this week, but we're confident in London's ability to make this pick worth our while. He hauled in
six-of-seven targets for 54 yards and almost added a score in Week 7 against Tampa, indicating history is on our side here.
</item 67>

<item 68>
Fantasy Football TE PPR Rankings Week 14: Who to start, best sleepers at tight end (Sporting News, 2023-12-07T22:14:00+00:00)
Jaguars), and Dalton Kincaid (@ Chiefs) aren't far behind. LaPorta and Njoku both have great matchups against defenses that entered last week in the top 10 in
fantasy points per game (FPPG) allowed to TEs, and we know what Kelce, Hockenson, and Kincaid do every week in PPR formats.

George Kittle (vs. Seahawks) doesn't have a great matchup -- especially when you remember that he had just three catches for 19 yards against Seattle in Week 12
-- but he remains a must-start given his high ceiling in any matchup.

WEEK 14 FANTASY ADVICE:

Sleepers | Busts | Projections | Start-sit

Who are the best fantasy TE sleepers, pickups, & streamers in Week 14?

It can be tough to trust Taysom Hill (vs.
</item 68>

<item 69>
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
</item 69>

<item 70>
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
</item 70>

<item 71>
Fantasy Football Week 13 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-11-30T22:10:58+00:00)
I'm pretty certain they're not bad enough to make Tim Boyle look like Tim Tebow or even Tim Hasselbeck, but I do wonder if they're juuust bad enough to give
better-than-normal numbers to Garrett Wilson and Breece Hall. Reinforcements along the O-line should help the Jets' cause against a Falcons defense that's
blitzed at a league-average rate, is eighth-lowest in pass rush pressure rate and is fifth-lowest in sacks (28).
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
NFL Fantasy Football Week 6 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-12T21:23:16+00:00)
Weekly Breakdown MATCHUP @ CHI MIN -2.5 O/U 44.5 OPP VS WR 20th PROJ PTS 10.7 WR RNK 33rd YTD Stats REC 13 TAR 26 REYDS 166 TD 2 FPTS/G 8.3 No doubt there's
some potential here, but some hunches keep me from falling for Osborn this week. One: I think the Vikings will try to get their run game going since they don't
have Justin Jefferson. Two: Jordan Addison is far more explosive and T.J. Hockenson has been a much more relied-upon weapon. They should have more targets than
Osborn. And the expected zone coverage Chicago will play further will limit Osborn unless he sees more targets than expected.
</item 73>

<item 74>
NFL Fantasy Football Week 6 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-12T21:23:16+00:00)
In an effort not to overwork their backs, who won't have it easy against the Vikings, Kmet could handle more short-area throws than normal. His ADOT over the
past two games has been 7.64, which isn't far from his season average of 6.69.

VIKINGS: Allow the fewest yards per catch to rival tigh ends (5.9 yards), even keeping a lid on Travis Kelce last week (6.7 yards per catch). But they've also
seen the fourth-lowest ADOT from tight ends (5.0 yards per target), almost as if teams know they won't be able to succeed on downfield shots against them (zero
completions of 20-plus yards from tight ends). These numbers are tough but if Kmet gets the target volume we've seen him have over the past couple of weeks he
can still put up palatable numbers. And, tight ends have scored three times in the Vikings past three games.
</item 74>

<item 75>
Fantasy Injury Updates Week 10: News on Justin Jefferson, Ja'Marr Chase, T.J. Hockenson & Curtis Samuel (Sporting News, 2023-11-11T21:17:00+00:00)
If he's out, Boyd is a must-start, and you can try one of Cincinnati's secondary receivers in deep leagues.

WEEK 10 STANDARD RANKINGS:

QBs | RBs | WRs | TEs | D/STs | Kickers

Is T.J. Hockenson playing Week 10?

UPDATE: NFL Network's Ian Rapoport reported Sunday morning that Hockenson is expected to play against New Orleans.

UPDATE 2: Hockenson is officially ACTIVE.

Hockenson (ribs) is listed as "questionable" for Week 10, but he's already said he plans to play on Sunday. The tight end has dealt with his fair share of
injury woes this season, however, he's yet to miss a game.
</item 75>

<item 76>
NFL Fantasy Football Week 5 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-04T21:56:25+00:00)
With upside an issue, think of Howell as a sleeper with 20-point potential. I'd rather start Jordan Love, Jared Goff or Justin Fields, but taking Howell over
Dak Prescott and Joe Burrow isn't crazy given their matchups and/or situations.

LAST WEEK: Bounced back from a horrid four-INT loss in Week 3 to complete nearly 71% of his throws for 290 yards with one touchdown and a very clutch 40 rushing
yards at Philly. This was the most efficient game of his young career, but it was also the busiest game of his career as he threw 41 passes and ran six times.
Playing into overtime helped his cause.

HOWELL: Has one game out of five as a pro with multiple passing touchdowns, and two others with one rushing and one passing score.

BEARS: Are one of the easiest matchups for a quarterback.
</item 76>

<item 77>
Fantasy Football TE PPR Rankings Week 14: Who to start, best sleepers at tight end (Sporting News, 2023-12-07T22:14:00+00:00)
This week might not matter for many fantasy football owners. Some have secured their spot in the postseason; others are well out of contention. However, for
those in must-win situations, this week really matters. It's not the time to have a hole in your lineup at tight end, but if you do, we're here to help with our
Week 14 fantasy TE PPR rankings.

Fortunately, there are several solid sleepers and streaming options despite two tough byes and some big injuries to notable TEs. It's always tough to know who
to trust at this volatile position, so we break down which tight ends have higher floors for those favored to win comfortably and which have higher ceilings for
those who need to really shoot for big points.

Let's get to it and check out the TE landscape for Week 14.

Who are the TEs on bye in Week 14?
</item 77>

<item 78>
Fantasy Football Week 13 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-11-30T22:10:58+00:00)
They were even less kind to Geno Smith, and this week shouldn't be much different. Tack on the tough assignment in Stephon Gilmore in coverage (unless the
Cowboys use shadow coverage for the first time this year), and Lockett could be in for another lean night. I don't mind him as a WR3 over Drake London, DeAndre
Hopkins and Diontae Johnson, but I'd start Rashee Rice and Adam Thielen over Lockett.

Sit Him (Lineup Decisions) Projections powered by Sportsline Jake Ferguson TE DAL Dallas • #87

Age: 24 • Experience: 2 yrs.
</item 78>

<item 79>
NFL Fantasy Football Week 5 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-04T21:56:25+00:00)
Weekly Breakdown MATCHUP vs CAR DET -10 O/U 45 OPP VS QB 4th PROJ PTS 20.8 QB RNK 11th YTD Stats PAYDS 1029 RUYDS 12 TD 7 INT 3 FPTS/G 19.6 Any quarterback with
upside is a better pick than Goff, who shouldn't have to do too much to help the Lions pull down a win. There will be temptation on his part to test Jameson
Williams on deep throws, and no doubt there will be opportunities for Sam LaPorta and Josh Reynolds (and Jahmyr Gibbs, hopefully), but it's more likely Goff has
a modest game at home with a couple of scores and maybe 225 yards. I'll give him the nod over Dak Prescott and Joe Burrow, but Jordan Love, Trevor Lawrence,
Josh Dobbs and C.J. Stroud all have more potential to help you win.
</item 79>

<item 80>
Fantasy Football RB PPR Rankings Week 14: Who to start, best sleepers at running back (Sporting News, 2023-12-07T22:09:00+00:00)
Spears contributed 20 yards on the game-tying drive in the fourth quarter and another 27 yards in OT, but Mike Vrabel's squad ultimately lost. In what should be
a high-scoring game against the Super Bowl-hopeful 'Fins, we like Spears to be busy as rookie QB Will Levis's top check-down option -- assuming Henry
(head/neck) is out, of course.

Ezekiel Elliott (@ Steelers) is another potential fill-in who should produce at least flex-worthy stats this week. With Rhamondre Stevenson (ankle) on the
shelf, Elliott could see 20 touches. Similarly, Kenneth Gainwell (@ Cowboys) and Roschon Johnson (vs. Lions) could see big upticks in touches if D'Andre Swift
and D'Onta Foreman can't suit up, but we're not as confident those backs can produce given their matchups.
</item 80>

<item 81>
Seahawks-Cowboys Fantasy Start 'Em Sit 'Em Week 13: Should you start Zach Charbonnet, Jake Ferguson on TNF? (Sporting News, 2023-11-30T21:31:00+00:00)
Here's looking at who to start in seasonal leagues and DFS lineups from the latest edition of TNF:

MORE THURSDAY NIGHT FOOTBALL:

DK lineup | FD lineup | ATS pick | Best props

Seahawks: Who to start, sit in fantasy football on TNF

Sit: QB Geno Smith

Smith has dropped from the top 12 last season to cumulative QB20 in 2023. He's had two great starts and three so-so ones, but the rest has been forgettable. The
Cowboys have allowed only two strong fantasy performances by quarterbacks, (Brock Purdy and Jalen Hurts), and we don't expect Smith to make it three.

Sit: RB Zach Charbonnet

There was a black fly in the Charbonnet against the 49ers last week.
</item 81>

<item 82>
NFL Fantasy Football Week 6 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-12T21:23:16+00:00)
Maybe the Patriots will turn to another running back in hopes of finding anything that can help take pressure off of their depleted passing game. I can't
recommend Stevenson as a start until we see proof of him playing well -- a favorable road matchup alone isn't enough. I'd rather start Jerome Ford, Zack Moss
and D'Onta Foreman.

LAST WEEK: Was the first time Stevenson did not have 16 or more touches in a game. He had eight carries and no receptions. He also played a season-low 50% of
the snaps but did handle 10 of 16 snaps on third/fourth downs. Stevenson started and played the first two series, then Elliott started off the next two series,
but they rotated in and out for each other the whole game. Elliott continued to be the Patriots RB of choice in the two-minute offense to end the first half.

STEVENSON: Nothing has changed in terms of his efficiency.
</item 82>

<item 83>
NFL Fantasy Football Week 5 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-04T21:56:25+00:00)
I'd rather start Jahmyr Gibbs, Dameon Pierce and Najee Harris.

LAST WEEK: His playing time hit a season-low 43% of the snaps as he continued losing third-down work to Chuba Hubbard. It hasn't crushed his reception totals at
all but it's notable that his only game with more than four grabs came with Andy Dalton under center. Sanders also played just 31% of the Panthers' second-half
snaps, a big-time change after his second-half snaps were a lot closer to Hubbard's in Weeks 2 and 3.

SANDERS: His yards per carry has tumbled like a bowling ball downstairs this season, going from 4.0 against the Falcons to 3.1 against the Saints to 2.7 against
the Seahawks to 1.5 against the Vikings. Tough matchups? Sure, even including the Vikings. Doesn't mean it's not a good look.
</item 83>

<item 84>
Fantasy Injury Updates Week 10: News on Justin Jefferson, Ja'Marr Chase, T.J. Hockenson & Curtis Samuel (Sporting News, 2023-11-11T21:17:00+00:00)
Chase put a scare into the fantasy community on Wednesday when he was listed as a DNP, but after logging back-to-back limited practices, there is a lot more
optimism heading into Sunday. With Tee Higgins (hamstring) already ruled out, Cincinnati will be hoping Chase can feel healthy enough to play when they host the
Texans at 1 p.m. ET on Sunday.

The Bengals, despite getting a win over the Bills last Sunday, are a beaten-down bunch. Their defense will also be without a prominent edge rusher, Sam Hubbard,
complicating matters for a team hoping to stack wins as they compete for an AFC North crown. On the bright side, Joe Burrow appears to be finally healthy and
should have no problems targeting Tyler Boyd, Trenton Irwin, and Andrei Iosivas in the passing game. If Chase does end up suiting up, fantasy owners should
enjoy an even larger workload than the receiver already had.
</item 84>

<item 85>
Thursday Night Football FanDuel Picks: NFL DFS lineup advice for Week 4 Lions-Packers single-game tournaments (Sporting News, 2023-09-26T21:53:00+00:00)
He has seven passing touchdowns through the first three weeks and faces a bottom-10 passing defense, per PFF. Love will enjoy a date with Detroit's banged-up
secondary that is just beginning life without C.J. Gardner-Johnson (pec).

FLEX WR Romeo Doubs ($10,500). Christian Watson appears primed for a return on Thursday night, but does it really matter? With him being held out for multiple
weeks with a hamstring injury, we question the type of workload that the former North Dakota State standout is in line for. As a result, we look to Doubs. The
second-year receiver is tied for the team lead in targets and has the most touchdowns with three. Along with Jayden Reed, Doubs has been a favorite of Love's in
the red zone, as the duo has combined for 12-of-16 targets to receivers.
</item 85>

<item 86>
NFL Fantasy Football Week 6 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-12T21:23:16+00:00)
Weekly Breakdown MATCHUP vs CAR MIA -13.5 O/U 48.5 OPP VS RB 31st PROJ PTS 7 RB RNK NR YTD Stats RUYDS 0 REC 0 REYDS 0 TD 0 FPTS/G 0 Once it's clear that Wilson
will play, the matchup combined with the Dolphins' approach to utilize multiple backs should put Wilson in position to have some good opportunities. Remember,
Wilson's injury was to his abdomen, not his legs, so there might not be much of a limit on how much work he'll do (though it's unlikely he'll have a ton of
touches). I'd take my chances with him over Samaje Perine, Jerick McKinnon and Antonio Gibson.

WILSON: Started practicing this week after being sidelined with finger and abdomen injuries since late August.
</item 86>

<item 87>
Fantasy Football Week 13 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-11-30T22:10:58+00:00)
STARTS: Bijan Robinson, Garrett Wilson, Breece Hall, Drake London, Jets DST, Falcons DST

Bijan Robinson, Garrett Wilson, Breece Hall, Drake London, Jets DST, Falcons DST SITS: Kyle Pitts, Tyler Conklin, Tyler Allgeier, both QBs

Start 'em 10.0-7.00 6.99 - 5.01 5.00 - 0 Sit 'em

Pittsburgh's biggest change in offensive scheme last week involved Kenny Pickett throwing over the middle much more than before (Pat Freiermuth benefitted). But
they also ran the ball effectively at Cincinnati -- Najee Harris had one of his best games this year. Expect more of the same against the Cardinals this week.
Arizona has allowed five touchdowns to tight ends over its past five games but actually has been relatively efficient defending them otherwise.
</item 87>

<item 88>
NFL Fantasy Football Week 5 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-04T21:56:25+00:00)
It's been a journey back. He's going to get there."

FILM ON SMITH: From what I see, Smith is running shorter routes and is getting open faster than Pitts. That's earned him some targets from Ridder, who seemed to
have better timing with Smith than Pitts. Smith is not drawing any kind of tight coverage like Pitts either, mostly because he's staying closer to the line of
scrimmage. But after the catch Smith has been a handful, breaking three tackles just last week and picking up extra yards with it. He's not fast, but he's
faster than Pitts, and for at least last week and Week 2 it changed the calculus for how the Falcons divvied up targets.

TEXANS: Have had a bundle of troubles with tight ends, giving up an 84% catch rate and 6.76 YAC/reception to the position, both bottom-six in the league.
</item 88>

<item 89>
Thursday Night Football FanDuel Picks: NFL DFS lineup advice for Week 4 Lions-Packers single-game tournaments (Sporting News, 2023-09-26T21:53:00+00:00)
FLEX TE, Luke Musgrave ($8,000). Much like with the quarterbacks, we also like the other tight end in this game. Musgrave is coming off his best game of the
season, hauling in six catches for 49 yards, and he gets a matchup against the second-worst defense against tight ends. According to FantasyPros, the Lions are
giving up 14.8 fantasy points per game to the position, leaving Musgrave in a great spot to deliver. If Watson does return on Thursday, it could help the
rookie, as Detroit's defense will have to focus their attention elsewhere.

Best SuperDraft NFL DFS Picks for Lions-Packers

This year, we're also listing our favorite weekly DFS plays on SuperDraft.

For those unfamiliar, SuperDraft's DFS scoring involves point multipliers as opposed to traditional salary-capped or tiered contests.
</item 89>

<item 90>
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
</item 90>

<item 91>
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
</item 91>

<item 92>
NFL Fantasy Football Week 6 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-12T21:23:16+00:00)
Perhaps it's all health related and Thomas' volume from last week will stick.

ZONE COVERAGE: Atlanta usually plays a lot of zone coverage early on and if it works they stick with it for the whole game (Week 5 was an example of that). In
Weeks 3 and 4 against the Lions and Jaguars they played much more man coverage, especially when they needed a fourth-quarter turnover. And because Howell is
much less explosive against zone looks, I expect the Falcons to stick with zone much more this week.

Start Him (Lineup Decisions) Projections powered by Sportsline Kyle Pitts TE ATL Atlanta • #8

Age: 23 • Experience: 3 yrs.
</item 92>

<item 93>
Fantasy Football TE PPR Rankings Week 14: Who to start, best sleepers at tight end (Sporting News, 2023-12-07T22:14:00+00:00)
Still, it's smart to check your other options before trusting Goedert.

The same goes for Evan Engram (@ Browns). He's generally more reliable in PPR leagues, but he's facing a Browns' defense that allows the fewest FPPG to TEs.
Both Engram and Pat Freiermuth (vs. Patriots) are in tough spots against top-five TE defenses.

What are the biggest TE injuries in Week 14?

We already mentioned Goedert (forearm) and Schultz (hamstring), but another TE who could be back this week is Darren Waller (hamstring). The Giants really have
no reason to bring back their star TE if they don't feel like he's 100 percent, but it's something to monitor throughout the week, as Waller (vs. Packers)
always has a high ceiling as the Giants' de facto No. 1 receiver.
</item 93>

<item 94>
NFL Week 13 injuries: Browns to start Joe Flacco with Dorian Thompson-Robinson out, Saints' Olave questionable (CBSSports.com, 2023-12-01T22:52:36+00:00)
For Taylor, he underwent surgery on his thumb this week and is expected to miss at least a couple of weeks, which will thrust Zack Moss into a more prominent
role in the Colts backfield.

Burks is the only Titan to have an injury designation entering the weekend, but the wideout was a full participant in Friday's practice. However, he will need
to clear the final stages of concussion protocol in order to play.

Dolphins (-9.5) at Commanders

Every player who the Dolphins listed as questionable for Sunday's game against Washington was able to practice on a limited basis on Friday. That includes
Cotton and Lamm, who both were DNPs earlier in the week. Rookie running back De'Von Achane was a full participant in Friday's practice and head coach Mike
McDaniel said he will play this week after missing time with a knee injury.

Washington could be without its starting center this weekend as it plays host to Miami.
</item 94>

<item 95>
NFL Fantasy Football Week 6 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-12T21:23:16+00:00)
The Commanders clearly value the position in their offense, Thomas sees the most throws from Howell, and if Atlanta plays heavy zone coverage then it should
mean a bunch of underneath throws for Thomas to take in. You're hoping for at least seven targets here, which is possible. He's worth the nod over Jake
Ferguson, Dalton Schultz and Tyler Conklin, and it's close in full PPR over Kyle Pitts.

LAST WEEK: Thomas was a big-time benefactor of the Commanders chasing points against Chicago's heavy zone coverage, seeing 10 of his 11 targets once the Bears
built a 17-point lead in the first half. A 24-yard catch on the final play of the first half against prevent defense didn't hurt either.

FALCONS: See the third-most targets per game to tight ends (9.0) which in turn has made them bottom-five in raw stats like total catches and yards allowed.
</item 95>

<item 96>
NFL Fantasy Football Week 5 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-04T21:56:25+00:00)
BEFORE WEEK 4: The Seahawks, Bengals and 49ers tight ends, including George Kittle, combined for eight non-PPR/19 PPR Fantasy points over three games against
the Rams. L.A.'s pass stats against tight ends were much better before the Colts laid some numbers on them.
</item 96>

<item 97>
N.F.L. Executives on the Biggest Questions Facing Super Bowl Contenders (The New York Times, 2023-10-16T23:58:53+00:00)
This team is relatively young. They are scrappy. The fact that they have been there before gives them a chance. I think the quarterback is growing.”

• Cincinnati Bengals: Can Joe Burrow hold up for a full season?

What I think: Burrow is looking healthier over the past two weeks. The bye week comes at a great time for Burrow to get even healthier, but check out the
schedule on the other side: San Francisco, Buffalo, Houston, Baltimore, Pittsburgh and Jacksonville. Cincy still has a second game with the Steelers, plus games
against Kansas City and Cleveland. I’m worried Burrow will wear down against these defenses.

Exec comment: “You cannot always expect Joe Burrow to create magic with a bottom-half offensive line. The line is my concern all the way.”

Advertisement

2. Are teams sometimes going too far playing to win? The Eagles come to mind after falling 20-14 to the Jets.
</item 97>

<item 98>
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
</item 98>

<item 99>
Fantasy Football TE PPR Rankings Week 14: Who to start, best sleepers at tight end (Sporting News, 2023-12-07T22:14:00+00:00)
It can be tough to trust Taysom Hill (vs. Panthers) in full-point PPR leagues, but a matchup against the league's second-worst run defense gives him a high
ceiling regardless of format. He's been more involved as a receiver this year, so he'll get a few catches to go along with the rushing stats, too. With Derek
Carr (concussion) banged up, Hill might throw more this week, too.

Gerald Everett (vs. Broncos) and Isaiah Likely (vs. Rams) have matchups against bottom-six TE defenses, and Cole Kmet (vs. Lions), Jake Ferguson (vs. Eagles),
and Kyle Pitts (vs. Buccaneers) also have above-average matchups that give them higher floors and higher ceilings than usual.

For those in deeper leagues, Tyler Conklin (vs.
</item 99>

<item 100>
NFL Fantasy Football Week 5 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-04T21:56:25+00:00)
LAST WEEK: Total disaster. In a gamescript that actually gave Minnesota a chance to -- gasp! -- run the ball, Kirk Cousins barely threw the ball and Addison
went catchless on one target. There was also a major reduction in snaps played, down to 58%. His season-low was 56% from Week 1.

MATCHUP: With the Vikings capable of running the ball, the offense went with heavier formations. Backup tight end Josh Oliver played about as many snaps as he
did in Week 1 and third tight end Johnny Mundt saw a season-high 17% of snaps. Week 4 was the first time all year the Vikings DIDN'T throw the ball on at least
69% of their snaps.

THIRD MAN: Last week did affirm that Addison may actually be the No. 3 wide receiver in the offense behind Justin Jefferson and K.J.
</item 100>
=== END ITEM id=af203a45a0fe492c ===

=== ITEM id=a9cbdbbc2fa89867 ===
Gold sentences and facts:
Sentence 0: During cross-examination, Huang said Paradigm pressed Bankman-Fried on the board issue and was told he didn't want investors as directors but he did plan on having a board with experts.
  Fact 0.0: During cross-examination, Huang said Paradigm pressed Bankman-Fried on the board issue.
  Fact 0.1: Huang said Paradigm was told that Bankman-Fried did not want investors as directors.
  Fact 0.2: Huang said Bankman-Fried did plan on having a board with experts.
Sentence 1: The highly anticipated criminal trial for Sam Bankman-Fried, former CEO of bankrupt crypto exchange FTX, started Tuesday to determine whether he’s guilty of seven counts of fraud and conspiracy.
  Fact 1.0: The criminal trial of Sam Bankman-Fried was highly anticipated.
  Fact 1.1: Sam Bankman-Fried is the former CEO of bankrupt crypto exchange FTX.
  Fact 1.2: The criminal trial of Sam Bankman-Fried started on a Tuesday.
  Fact 1.3: The criminal trial determines whether Sam Bankman-Fried is guilty of seven counts of fraud and conspiracy.
Sentence 2: The prosecution painted Bankman-Fried as someone who knowingly committed fraud to achieve great wealth, power and influence, while the defense countered that the FTX founder acted in good faith, never meant to commit fraud or steal and basically got in over his head.
  Fact 2.0: The prosecution portrayed Bankman-Fried as someone who knowingly committed fraud to achieve great wealth, power and influence.
  Fact 2.1: The defense countered that Bankman-Fried acted in good faith.
  Fact 2.2: The defense countered that Bankman-Fried never meant to commit fraud or steal.
  Fact 2.3: The defense countered that Bankman-Fried basically got in over his head.

Retrieved items:
<item 1>
2023 NFL Week 11 injury roundup: Rams WR Cooper Kupp, Seahawks RB Kenneth Walker III among stars to exit (CBSSports.com, 2023-11-19T23:51:49+00:00)
He was seen walking on the sidelines shortly afterward but ruled doubtful to return. Rookie Zach Charbonnet, who entered Week 11 with just over 200 rushing
yards on the year, replaced Walker as Seattle's top ball carrier. Following the game, Seahawks head coach Pete Carroll told reporters Walker suffered an
"oblique strain that was legit" and could miss time.

D'Onta Foreman (Bears) remained a headliner of Chicago's backfield against the Lions, even with Khalil Herbert back from injured reserve, but managed just 14
yards on the ground before suffering an ankle injury. He walked off with a limp and attempted to run on the sidelines but did not return to the field after the
Bears ruled him questionable.

Pick Six Newsletter Crafted By The Best NFL Experts Get the day's big stories + fun stuff you love like mock drafts, picks and power rankings.
</item 1>

<item 2>
2023 NFL Week 11 injury roundup: Rams WR Cooper Kupp, Seahawks RB Kenneth Walker III among stars to exit (CBSSports.com, 2023-11-19T23:51:49+00:00)
I agree to receive the "Pick Six Newsletter" and marketing communications, updates, special offers (including partner offers), and other information from CBS
Sports and the Paramount family of companies. By pressing sign up, I confirm that I have read and agree to the Terms of Use and acknowledge Paramount's Privacy
Policy See All Newsletters Please check the opt-in box to acknowledge that you would like to subscribe. Thanks for signing up! Keep an eye on your inbox. Sorry!
There was an error processing your subscription.

WR

Tyreek Hill (Dolphins) briefly left Miami's game against the Raiders while nursing a hand injury and was ruled questionable to return, but he took the field
again soon afterward, securing seven catches for 80 yards and a score before halftime. Hill has left multiple games this year, only to return in all of them.

Cooper Kupp (Rams) had just a single catch against the Seahawks before suffering an ankle injury while blocking.
</item 2>

<item 3>
Fantasy Injury Updates Week 10: News on Justin Jefferson, Ja'Marr Chase, T.J. Hockenson & Curtis Samuel (Sporting News, 2023-11-11T21:17:00+00:00)
ET, according to ESPN's Jeremy Fowler. The receiver was limited in practice thanks to the toe injury that has plagued him since leaving Week 8's game against
the Eagles, but barring a setback, he will return to action this weekend.

The receiver was working his way into flex territory with a couple of good performances before picking up the injury. However, we wouldn't consider him to be
worthy of a fantasy start this week unless you're in a bind. It's unclear whether Samuel will regain the same workload or if he'll be playing in limited action
as he works his way back. Jamison Crowder emerged in Samuel's absence and Jahan Dotson has seen his production drastically increase, as well. The wait-and-see
approach is the preferred one heading into Week 10.
</item 3>

<item 4>
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
</item 4>

<item 5>
2023 NFL Week 11 injury roundup: Rams WR Cooper Kupp, Seahawks RB Kenneth Walker III among stars to exit (CBSSports.com, 2023-11-19T23:51:49+00:00)
Smith did return, however, for the final drive of the game in the losing effort. Head coach Pete Carroll said that Smith will "make it back." When asked if that
means for Thursday's matchup against San Francisco, the coach said, "I would think so."

Sam Howell (Commanders) received medical attention on the sidelines after enduring several hard hits to the upper body on a touchdown run against the Giants. He
was quickly cleared of a potential concussion, however, and remained in the game. Veteran journeyman Jacoby Brissett is Washington's backup.

RB

Aaron Jones (Packers) suited up against the Chargers despite a lingering hamstring issue, only to manage just five carries before suffering a knee injury. The
former Pro Bowler needed help to leave the field and was then carted to the locker room, replaced by A.J. Dillon.
</item 5>

<item 6>
Fantasy Football Week 13 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-11-30T22:10:58+00:00)
Sit Him (Lineup Decisions) Projections powered by Sportsline Cooper Kupp WR LAR L.A. Rams • #10

Age: 30 • Experience: 7 yrs. Weekly Breakdown MATCHUP vs CLE LAR -3.5 O/U 40 OPP VS WR 14th PROJ PTS 13.8 WR RNK 25th YTD Stats REC 27 TAR 51 REYDS 393 TD 1
FPTS/G 10.6 You're never supposed to sit your studs, but what if your stud is playing through something that made him look like a dud last week? That was Kupp,
who in Week 12 ran a ton of short routes and couldn't separate and didn't have much sudden movement. If he's going to come through this week it'll likely come
on a ton of short targets and/or some targets inside the 10-yard line.
</item 6>

<item 7>
Fantasy Injury Updates Week 10: News on Justin Jefferson, Ja'Marr Chase, T.J. Hockenson & Curtis Samuel (Sporting News, 2023-11-11T21:17:00+00:00)
After turning back the clocks last week, it now gets darker earlier for the foreseeable future. That darkness allows us to see the stars on a clear night, so
it's only natural to wonder if the pass-catching stars will shine bright in Week 10 of the NFL season. We already know Tee Higgins and Nico Collins are OUT,
while Justin Jefferson, Ja'Marr Chase, and T.J. Hockenson are among the other star pass-catchers beaten up heading into the week. Can we expect them on the
field when the action kicks off Sunday?

Below, we'll give you the latest news on these ailing pass-catchers and break down how their injuries could potentially affect the Week 10 fantasy landscape.

WEEK 10 INJURY UPDATES: Early WRs | RBs

Is Justin Jefferson playing Week 10?

UPDATE: The Vikings will not activate Jefferson before Sunday's game and have officially ruled him OUT, according to NFL Network's Tom Pelissero.
</item 7>

<item 8>
2023 NFL Week 11 injury roundup: Rams WR Cooper Kupp, Seahawks RB Kenneth Walker III among stars to exit (CBSSports.com, 2023-11-19T23:51:49+00:00)
Dillon. Green Bay ruled him questionable to return, then downgraded him to out, but coach Matt LaFleur said after the Packers' win over Los Angeles that he
doesn't expect the injury to be a long-term issue.

De'Von Achane (Dolphins) returned from injured reserve -- and a four-game absence -- but limped to the sidelines after his second touch of the first quarter
against the Raiders. Miami ruled him questionable to return with a knee injury, and while he later returned to the sidelines from a locker-room visit, he was
replaced on the field indefinitely by Raheem Mostert. Head coach Mike McDaniel told reporters on Monday that Achane is day-to-day. While he hasn't been ruled
out of Friday's matchup with the Jets, he's not been given the green light either, so his status is worth monitoring.

Kenneth Walker III (Seahawks) took four carries against the Rams before leaving with an oblique injury.
</item 8>

<item 9>
2023 NFL Week 11 injury roundup: Rams WR Cooper Kupp, Seahawks RB Kenneth Walker III among stars to exit (CBSSports.com, 2023-11-19T23:51:49+00:00)
DB

Talanoa Hufanga (49ers), who emerged as a ballhawk for San Francisco's secondary in 2022, left early against the Buccaneers with a knee injury. Coach Kyle
Shanahan confirmed to reporters on Monday that the All-Pro suffered a season-ending ACL tear.
</item 9>

<item 10>
2023 NFL Week 11 injury roundup: Rams WR Cooper Kupp, Seahawks RB Kenneth Walker III among stars to exit (CBSSports.com, 2023-11-19T23:51:49+00:00)
The former All-Pro, who battled a lingering hamstring injury earlier this year, received attention on the sidelines after limping off, then left the game for
further evaluation in the locker room. He was ruled questionable to return. NFL Media reported on Monday that Kupp suffered a low ankle sprain and will be
evaluated going forward.

Darius Slayton (Giants) had a big impact against the Commanders, hauling in a deep score from Tommy DeVito, before exiting with an arm injury. He was ruled out
for the remainder of the contest halfway through the third quarter.

EDGE

Joey Bosa (Chargers) appeared to injure his right leg or foot while held in a block during a first-quarter pass-rushing attempt against the Packers. He limped
to the sidelines and was carted to the locker room, then ruled questionable to return.
</item 10>
=== END ITEM id=a9cbdbbc2fa89867 ===
