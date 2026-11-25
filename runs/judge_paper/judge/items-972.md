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

=== ITEM id=23b448ed4f3d2b27 ===
Gold sentences and facts:
Sentence 0: Of Miami's five remaining opponents, four have pass defenses ranked in the league's top eight.
  Fact 0.0: Miami has five remaining opponents.
  Fact 0.1: Four of Miami's five remaining opponents have pass defenses ranked in the league's top eight.
Sentence 1: -- After missing last week's game, Tyreek Hill now needs to average almost 153 yards per game in his final three to reach his stated goal of 2,000 receiving yards this season.
  Fact 1.0: Tyreek Hill missed last week's game.
  Fact 1.1: Tyreek Hill now needs to average almost 153 yards per game in his final three games.
  Fact 1.2: Tyreek Hill's stated goal is 2,000 receiving yards this season.
  Fact 1.3: Tyreek Hill needs to reach his 2,000-yard goal this season.

Retrieved items:
<item 1>
NFL Fantasy Football Week 6 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-12T21:23:16+00:00)
That included in the second half when he saw one target and couldn't catch it. He definitely wasn't shadowed by anyone and was almost never double-teamed,
though there were a couple of routes where a safety was in proximity. It could have had more to do with the Texans attacking the middle of the field against the
Falcons' zone coverage.

SAINTS: Have played the fifth-fewest snaps of zone coverage this season and have ignored tendencies some quarterbacks have had through the early portion of the
season. It's just what they like to do. Stud cornerback Marshon Lattimore has not followed receivers much this season, playing 87% of his snaps at right
cornerback (to the left of the quarterback). Their other primary outside corner, Paulson Adebo, has allowed a 50% catch rate on the season with minimal YAC.
</item 1>

<item 2>
NFL Fantasy Football Week 6 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-12T21:23:16+00:00)
Jets • #17

Age: 23 • Experience: 2 yrs. Weekly Breakdown MATCHUP vs PHI PHI -7 O/U 41 OPP VS WR 31st PROJ PTS 11.7 WR RNK 29th YTD Stats REC 24 TAR 43 REYDS 279 TD 2
FPTS/G 12.8 Wilson's situation continues to hold him back from the potential he so clearly has. If he couldn't put up a monster game against the Broncos, it's a
tough sell that he'll do so against the Eagles. I do think volume will be in his favor, so counting on him for close to 16 PPR points (half that in non-PPR) is
fair. That gives him flex appeal over Nico Collins and Tyler Lockett, but not over Jakobi Meyers, Zay Flowers or Gabe Davis.
</item 2>

<item 3>
Fantasy Football Week 16 Wide Receiver Preview: Waiver adds, DFS plays, and more (CBSSports.com, 2023-12-19T20:51:32+00:00)
Projected stats for all starting quarterbacks are available at Sportsline.
</item 3>

<item 4>
Fantasy Football Week 16 Wide Receiver Preview: Waiver adds, DFS plays, and more (CBSSports.com, 2023-12-19T20:51:32+00:00)
Weekly Breakdown MATCHUP vs DAL MIA -1.5 O/U 51 OPP VS WR 7th PROJ PTS 14.2 WR RNK 16th YTD Stats REC 71 TAR 99 REYDS 964 TD 4 FPTS/G 14.8 We expect Hill to
return this week, which should drive people off of Waddle. But there is at least a chance that Hill won't be 100% and Waddle will still be Tua Tagovailoa's top
target. Waddle has been the quintessential DFS tournament play as of late with three of seven games over 19 PPR Fantasy points, topping 26 twice.

WR Preview Heath's Projections

My full set of Week 16 Fantasy Football projections for every position are now available on SportsLine. Find out which of my favorite plays are projected to
score higher than consensus rankings and which don't live up to their draft hype, at least in Week 16.
</item 4>

<item 5>
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
</item 5>

<item 6>
Fantasy Football Week 16 Wide Receiver Preview: Waiver adds, DFS plays, and more (CBSSports.com, 2023-12-19T20:51:32+00:00)
Weekly Breakdown MATCHUP @ MIA MIA -1.5 O/U 51 OPP VS WR 15th PROJ PTS 19.4 WR RNK 2nd YTD Stats REC 103 TAR 141 REYDS 1306 TD 10 FPTS/G 21.6 The choice between
Lamb and Tyreek Hill will not be an easy one. The tiebreaker for me is that Lamb is $900 cheaper and Hill may still not be 100% by the time the Cowboys and
Dolphins kick off. Last week was the first time since Week 9 that Lamb failed to reach the end zone and the second time since Week 5 that he didn't score at
least 17 PPR Fantasy points.

Contrarian DFS Play (WR Preview) Projections powered by Sportsline Jaylen Waddle WR MIA Miami • #17

Age: 25 • Experience: 3 yrs.
</item 6>

<item 7>
Fantasy Football TE PPR Rankings Week 14: Who to start, best sleepers at tight end (Sporting News, 2023-12-07T22:14:00+00:00)
It can be tough to trust Taysom Hill (vs. Panthers) in full-point PPR leagues, but a matchup against the league's second-worst run defense gives him a high
ceiling regardless of format. He's been more involved as a receiver this year, so he'll get a few catches to go along with the rushing stats, too. With Derek
Carr (concussion) banged up, Hill might throw more this week, too.

Gerald Everett (vs. Broncos) and Isaiah Likely (vs. Rams) have matchups against bottom-six TE defenses, and Cole Kmet (vs. Lions), Jake Ferguson (vs. Eagles),
and Kyle Pitts (vs. Buccaneers) also have above-average matchups that give them higher floors and higher ceilings than usual.

For those in deeper leagues, Tyler Conklin (vs.
</item 7>

<item 8>
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
</item 8>

<item 9>
N.F.L. Executives on the Biggest Questions Facing Super Bowl Contenders (The New York Times, 2023-10-16T23:58:53+00:00)
That figure was 7.8 in Mahomes’ first year as the starter and has climbed every season since. It’s a testament to the Chiefs’ adaptability, but also a
reflection of plummeting explosive play rates.

Exec comment: “Yeah, it looks harder for them on offense until Mahomes finds (Travis) Kelce on every major third down, every major two-minute play, every major
touchdown. Kansas City has the luxury of their best players being their toughest players. Chris Jones is a very tough player. Kelce is a very tough player.
Mahomes is a very tough player. You wonder why they win the games? Because the games are won by the toughest teams most of the time.”

• Philadelphia Eagles: Has Jalen Hurts regressed? Where are the explosive pass plays?

What I think: Explosive passing is down through most of the league, including for the Eagles.
</item 9>

<item 10>
Christmas Day preview: 49ers, Ravens square off in potential Super Bowl sneak peek (Yardbarker, 2023-12-24T23:34:39+00:00)
Can Raiders QB Aidan O’Connell earn next season’s starting job?: After leading the offense on seven touchdown scoring drives in last week's 63-21 win over the
Chargers, the rookie fourth-rounder aims to further establish himself as a serious contender for Las Vegas' starting job next season. Despite the encouraging
performance, O'Connell has been inconsistent across six games as the full-time starter, throwing for 1,300 yards, seven touchdowns, and five interceptions.

However, a stellar showing against the NFL's fifth-ranked pass defense (202.5 yards per game allowed) would benefit his bid to be the starter in 2024.

Chiefs TE Travis Kelce: Yes, his girlfriend will be in town for the holiday. Kelce is averaging 89.6 yards per game with global popstar Taylor Swift in
attendance this season but is coming off his worst outing in her presence, catching five passes for 28 yards.
</item 10>

<item 11>
Dolphins vs. Eagles live score, updates, highlights from NFL 'Sunday Night Football' game (Sporting News, 2023-10-22T23:28:04+00:00)
Hurts gets two yards on a quarterback sneak to keep the Eagles' drive alive.

End of first quarter: Dolphins 3, Eagles 3

8:57 p.m. — DeVonta Smith has the Eagles into Dolphins territory after an eight-yard grab.

8:56 p.m. — Boston Scott with a really nice return to set up the Eagles at their own 39-yard line.

Dolphins 3, Eagles 3

8:52 p.m. FIELD GOAL — The Dolphins can't capitalize on the opportunity, so Jason Sanders ties the game with a field goal.

8:51 p.m. — A Tyreek Hill touchdown is erased by a holding penalty.

8:50 p.m. — The Eagles have lost yardage on three different plays in this quarter, which doesn't seem sustainable. It'll be 3rd & Goal from the 12.

8:48 p.m.
</item 11>

<item 12>
Fantasy Football Week 13 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-11-30T22:10:58+00:00)
Tua Tagovailoa, Terry McLaurin (low-end WR2), De'Von Achane (low-end RB2), Dolphins DST

Tyreek Hill, Jaylen Waddle, Raheem Mostert, Tua Tagovailoa, Terry McLaurin (low-end WR2), De'Von Achane (low-end RB2), Dolphins DST SITS: Sam Howell (low-end
starter at best), Brian Robinson (desperation RB starter), Logan Thomas (PPR bye-week replacement), Antonio Gibson (PPR bye-week replacement), Commanders DST

Sam Howell (low-end starter at best), Brian Robinson (desperation RB starter), Logan Thomas (PPR bye-week replacement), Antonio Gibson (PPR bye-week
replacement), Commanders DST SLEEPERS: Curtis Samuel (especially PPR), Jahan Dotson (non-PPR only)

Start 'em 10.0-7.00 6.99 - 5.01 5.
</item 12>

<item 13>
Fantasy Football Week 13 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-11-30T22:10:58+00:00)
I'd expect that to be out the window now -- Flacco wasn't shy about chucking the rock deep with the Jets, attempting passes of 15-plus Air Yards on 20 percent
of his throws, and he had no problem focusing on one primary receiver (Garrett Wilson had 8, 14 and 11 targets in Flacco's starts). The hope is that both Flacco
and Cooper are experienced and capable enough to develop rhythm to salvage the Browns passing game. It's the only thing to buy into here as the Rams have
allowed the 12th fewest Fantasy points to wide receivers on the season and happen to have done well against deep throws against wide receivers (40% completion
rate allowed, two touchdowns, both top six on the season). Cooper's volume should hopefully push him to a decent Fantasy stat line. I'd give him a whirl over
Terry McLaurin, Marquise Brown and Diontae Johnson.
</item 13>

<item 14>
NFL Fantasy Football Week 5 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-04T21:56:25+00:00)
That one went for 32 yards against the Eagles in Week 1. Since then his longest reception has been for seven yards.

SAINTS: Are sixth-best in rushing average to RBs (3.6) and are top-10 in basically every single run defense metric you could think of, including missed tackles
with just nine on the year. The only rusher to exceed 10 Fantasy points against them, PPR or not, was Derrick Henry in Week 1. He had 13 PPR points. No one's
run for a touchdown on them either.

SAINTS: Aren't quite as impressive against running backs through the air, but they're not awful. Figure them as league average across the board except in
YAC/reception (8.53), but that was due to a massive catch-and-run by Henry in Week 1.
</item 14>

<item 15>
NFL Best Bets Week 9: Colts clobber Panthers, Dolphins drop Chiefs in Germany (Sporting News, 2023-11-03T20:56:00+00:00)
MORE SUPERDRAFT DFS & FANTASY PROPS: How to play SuperDraft

This one should be a tightly-contested, high-scoring game — with heavy playoff implications, because we all know this could be an AFC Championship preview — so
expect Rice to be very busy through all four frames in Frankfurt. The Dolphins have allowed the fifth-most receptions and the 12th-most yards to wideouts this
season. If a bounce-back week is meant to be, Sir Rashee is our destiny.

Austin Ekeler, RB, Chargers @ Jets — OVER 29.5 receiving yards

Guess who's back? Back again...Ek is back....tell a friend. The Chargers finally got their best player back to full health, with Ekeler exploding for 123 yards
and a TD against the Bears last weekend. A whopping 94 of those 123 yards came through the air, as the PPR god gashed Chicago for chunk yardage a handful of
times.
</item 15>

<item 16>
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
</item 16>

<item 17>
N.F.L. Executives on the Biggest Questions Facing Super Bowl Contenders (The New York Times, 2023-10-16T23:58:53+00:00)
They are gaining more than 15 yards on just 11 percent of pass plays, down from 21 percent through six games last season and 19 percent for 2022 overall. Jalen
Hurts, A.J. Brown, DeVonta Smith and Dallas Goedert are still on the team, so I’m inclined to trust the production will return. But there’s also a lingering
fear defenses have caught up to aspects of the Eagles’ scheme. Does Philly have a counterpunch?

Advertisement

Exec comment: “People are going to crush Hurts and the offense. They are just churning the clock so much. It just feels like they are imposing their will on
teams through the run game and through their short passing and it just hasn’t taken off from an explosive passing standpoint, so when they play a team that can
move the ball on them through the air, it presents some challenges.”
</item 17>

<item 18>
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
</item 18>

<item 19>
Fantasy Football Week 13 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-11-30T22:10:58+00:00)
Weekly Breakdown MATCHUP vs CAR TB -5.5 O/U 37 OPP VS WR 15th PROJ PTS 12.5 WR RNK 32nd YTD Stats REC 53 TAR 82 REYDS 606 TD 1 FPTS/G 11.2 Remember when Chris
Godwin was known for being a super-reliable SLOT receiver? Those were the days. This season he's lined up wide on 68% of his snaps. Can't help but think that's
caused the massive change in his production this year -- that and his quarterback isn't a serial slot-targeter (or a Hall of Famer). Godwin's been under 60
yards in five straight, hasn't had a game with even 10 PPR points since Week 8 and has one touchdown on the year despite being tied for 12th among all NFL
players in end-zone targets with nine.
</item 19>

<item 20>
NFL Best Bets Week 9: Colts clobber Panthers, Dolphins drop Chiefs in Germany (Sporting News, 2023-11-03T20:56:00+00:00)
Bears

MORE BETTING:

Pick'em, Survivor Pool Picks | Chiefs-Dolphins preview | Chargers-Jets MNF preview

Best NFL player prop bets Week 9

Win BIG with SuperDraft! Get a free $10 deposit when you use promo code "TSN"!

Rashee Rice, Chiefs vs. Dolphins in Germany — OVER 45.5 receiving yards

As soon as I opened up my SuperDraft app, Rashee Rice popped out as the first prop of the first game on Sunday morning (9:30 a.m. in Frankfurt). We love this
rookie receiver and the big play ability he has consistently flashed over the past month. During that span, he has averaged over 55 receiving yards per game
despite the Chiefs' games being decided by an average of 11.8 points per game.
</item 20>

<item 21>
MVPs for all 16 AFC teams at the midpoint (Yardbarker, 2023-11-09T21:51:02+00:00)
He’s third in the NFL in passing yards (2,442), fourth in TD passes (17) and third in QBR (73). The 28-year-old might capture his third MVP, as oddschecker
notes he’s favored to win the award again.

LAS VEGAS RAIDERS | Defensive end Maxx Crosby: Despite a weak supporting cast, Crosby thrives. He has 9.5 of the Raiders’ 24 sacks and is tied for the league
lead in tackles for loss (13). If the two-time Pro Bowler keeps dominating, he will generate more Defensive Player of the Year buzz.

LOS ANGELES CHARGERS | WR Keenan Allen: The 31-year-old is the centerpiece of the offense and leads the team in yards receiving (720) and receptions (62).
Through Week 9, Pro Football Focus gave him a receiving grade of 84.6, eighth among WRs.
</item 21>

<item 22>
Fantasy Football Week 16 Wide Receiver Preview: Waiver adds, DFS plays, and more (CBSSports.com, 2023-12-19T20:51:32+00:00)
I'm hoping for a bounce back in Week 16. Dontayvion Wicks WR GB Green Bay • #13

Age: 22 • Experience: Rookie Weekly Breakdown MATCHUP @ CAR GB -5 O/U 37.5 OPP VS WR 22nd WR RNK 35th ROSTERED 15% YTD Stats REC 31 TAR 49 REYDS 491 TD 1 FPTS/G
6.5 If Christian Watson and Jayden Reed remain out then Wicks might just be the best option available. He should certainly be started in more leagues than he is
currently rostered in. His seven targets in Week 15 were a career high and part of a trend. His snap share has increased for three weeks in a row.

WR Preview DFS Plays

Top DFS Play (WR Preview) Projections powered by Sportsline CeeDee Lamb WR DAL Dallas • #88

Age: 24 • Experience: 4 yrs.
</item 22>

<item 23>
FanDuel 'Prop Stars' Picks: Best bets for Travis Kelce & Taylor Swift-inspired props for Chiefs-Jets Week 4 (Sporting News, 2023-09-28T21:41:00+00:00)
Don't expect the Chiefs to start running until it's time to chew the clock.

Our pick: Pass

Seemingly Ranch: Travis Kelce Last TD Scorer (+550)

Ranch is a fine condiment and beloved by many, which makes this wager something that should be seen as an insult to ranch fans everywhere. Unless the Jets make
this a close game or send it to overtime, we have a hard time believing that Kelce will even be on the field to score the last touchdown. Avoid this one like
people with taste buds avoid bleu cheese and treat your nuggets with respect.

Our pick: Pass

Plot Twist: Zach Wilson to have 300+ Passing Yards (+1800)

If you asked the average Jets' fan what is holding this roster back, the overwhelming results would be Wilson. "I'm the problem, it's me," would probably
qualify as the lyrics played in his walk-up song if fans had anything to say about it.
</item 23>

<item 24>
N.F.L. Executives on the Biggest Questions Facing Super Bowl Contenders (The New York Times, 2023-10-16T23:58:53+00:00)
This team is relatively young. They are scrappy. The fact that they have been there before gives them a chance. I think the quarterback is growing.”

• Cincinnati Bengals: Can Joe Burrow hold up for a full season?

What I think: Burrow is looking healthier over the past two weeks. The bye week comes at a great time for Burrow to get even healthier, but check out the
schedule on the other side: San Francisco, Buffalo, Houston, Baltimore, Pittsburgh and Jacksonville. Cincy still has a second game with the Steelers, plus games
against Kansas City and Cleveland. I’m worried Burrow will wear down against these defenses.

Exec comment: “You cannot always expect Joe Burrow to create magic with a bottom-half offensive line. The line is my concern all the way.”

Advertisement

2. Are teams sometimes going too far playing to win? The Eagles come to mind after falling 20-14 to the Jets.
</item 24>

<item 25>
NFL Best Bets Week 9: Colts clobber Panthers, Dolphins drop Chiefs in Germany (Sporting News, 2023-11-03T20:56:00+00:00)
Geno Smith hasn't been the same Pro Bowl QB as he was in 2022, but he has still demonstrated moments of brilliance for Pete Carroll's squad. And with DK Metcalf
and Kenneth Walker III a week closer to full health — and backup RB Zack Charbonnet looking like he's on the verge of a breakout — we like Seattle to at least
make this a game in Baltimore. The 'Hawks defense has quietly been pretty good, and we know what kind of fireworks this offense is capable of when everything's
clicking. Take an extra 1.5 points as insurance if you want, but we're taking the Seahawks to cover the +6.

FINAL SCORE PREDICTION: Ravens 26, Seahawks 23

Other Week 9 spreads we like: Patriots (-3) vs. Commanders; Raiders -1.5 vs. Giants

Best NFL moneyline bets Week 9

Dolphins (+110) vs.
</item 25>

<item 26>
Bills vs. Chargers odds, props, predictions: Buffalo faces struggling L.A. with eyes on AFC playoff picture (Sporting News, 2023-12-21T19:35:54+00:00)
Before Week 15, Cook hadn’t rushed more than 17 times in a game. After carrying 25 times vs. Dallas (and earning AFC Offensive Player of the Week status), BIlls
fans must be extremely hopeful that their offensive attack continues to involve Cook handling more than 20 rushes per game.

Cook’s 179-yard outburst coupled with backup Ty Johnson’s shoulder injury in Week 15 should both contribute to Cook garnering 20-plus touches this week.

Chargers betting news: Los Angeles limps toward the finish line

The Chargers were supposed to be good but are not. It seemed unlikely that coach Brandon Staley was going to be their guy past this year, especially on the
heels of their crushing playoff exit in Jacksonville where they blew a 27-point lead and lost in historic fashion. However, giving up 63 points to a team that
had just been shut out 3-0 four days earlier was enough to cause Los Angeles ownership to hit the shred button.
</item 26>

<item 27>
FanDuel 'Prop Stars' Picks: Best bets for Travis Kelce & Taylor Swift-inspired props for Chiefs-Jets Week 4 (Sporting News, 2023-09-28T21:41:00+00:00)
Wilson has turned into "Checkdown Charlie" as a result of conservative coaching, and we don't expect that to change in the early going. Lean into the narrative
and sprinkle a little cash on Kelce to break the plane first.

Our pick: Bet on it

FriENDZONE: Travis Kelce to NOT score a TD (+120)

Obviously, based on our last recommendation, we're friendzoning this bet and not laying down any action on it.

Our pick: Pass

Running Away Together: Chiefs to have 150+ Total Rushing Yards (+420)

The strength of the Jets roster is along the defensive line, and the Chiefs are a pass-first team that loves to light it up through the air. This is the first
time that Patrick Mahomes is playing in the New York area. He'll want to put on a show, and Andy Reid certainly isn't going to hold him back.
</item 27>

<item 28>
Fantasy Football WR PPR Rankings Week 14: Who to start, best sleepers at wide receiver (Sporting News, 2023-12-07T22:11:00+00:00)
Broncos) caught just five-of-nine targets last week, but as L.A.'s only reliable receiving option, a get-right game feels likely this week. Justin Jefferson (@
Raiders) is set to suit up for the first time since Week 5, and as we alluded to earlier, he's a must-start whenever he's active.

Stefon Diggs (@ Chiefs) hasn't totaled 100-plus receiving yards in any of his past six games, but he has every chance to get back on track against a Chiefs
coverage unit that showed some leaks in Sunday's loss to the Packers.

WEEK 14 FANTASY ADVICE:

Sleepers | Busts | Projections | Start-sit

Who are the best fantasy WR sleepers, waiver pickups in Week 14?

Elijah Moore (vs. Jaguars), Noah Brown (@ Jets), Odell Beckham Jr. (vs. Rams), Jonathan Mingo (@ Saints), and Xavier Gipson (vs.
</item 28>

<item 29>
Fantasy Football Week 13 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-11-30T22:10:58+00:00)
STARTS: Michael Pittman, Derrick Henry, Zack Moss

Michael Pittman, Derrick Henry, Zack Moss SITS: DeAndre Hopkins (PPR flex at best), Josh Downs (PPR flex at best), Will Levis, Titans DST

DeAndre Hopkins (PPR flex at best), Josh Downs (PPR flex at best), Will Levis, Titans DST SLEEPERS: Colts DST, Gardner Minshew

Start 'em 10.0-7.00 6.99 - 5.01 5.00 - 0 Sit 'em

With edge rusher Jaelan Phillips, the Dolphins ranked third in sacks (38) and sixth in pass rush pressure rate (39.9%). Both those numbers were boosted by the
defense's effort against a sagging Jets offense last Friday. Adding veteran Jason Pierre-Paul isn't a cure-all, but he will help the Dolphins from being forced
to blitz a ton week in and week out.
</item 29>

<item 30>
NFL DFS picks Week 15: Best sleepers, value players for FanDuel, DraftKings, SuperDraft lineups (Sporting News, 2023-12-13T20:32:00+00:00)
6x); Tyler Allgeier, Falcons at Panthers (DK: $4,400 | FD: $5,700 | SD: 1.6x)

WEEK 15 PPR RANKINGS:

QBs | RBs | WRs | TEs | D/STs | Kickers

NFL DFS Picks Week 15: WR sleepers, values for DraftKings, FanDuel, SuperDraft

Rashee Rice, Chiefs at Patriots (DK: $6,100 | FD: $6,500 | SD: 1.35x)

With Isiah Pacheco (shoulder) banged up, Rice seems like Kansas City's best skill-position player this side of Travis Kelce. He's certainly the most sure-handed
wideout on the Chiefs (not saying much, but still), and he has electric after-the-catch playmaking ability (which is good, considering Patrick Mahomes has
struggled with deep-ball accuracy).
</item 30>

<item 31>
FanDuel Picks Week 14: Best NFL DFS lineup advice for daily fantasy football tournaments (Sporting News, 2023-12-07T22:17:00+00:00)
He continues to see plenty of carries and a big role in the passing game. While the offense can't score, the volume is there against a Texans team that has
struggled to defend RBs out of the backfield. If the Jets manage to find the end zone for the first time since sliced bread was invented, Hall has a good chance
to do it. Considering it's borderline insane to include a Jets player in any DFS lineup, that means we might be one of the few who actually roster him, which
could help our cause.

WR Mike Evans, Buccaneers @ Falcons ($8,500). Evans hauled in six-of-eight targets for 82 yards and a score last time he saw the Falcons, and he's only gotten
better, putting together another great season for Tampa as Baker Mayfield's top option in the passing game. The incredible target share gives us no reason to
think that won't continue.
</item 31>

<item 32>
NFL Fantasy Football Week 5 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-04T21:56:25+00:00)
Even when the game was in garbage time Burrow couldn't get much going.

AIR YARDS LAST WEEK: Burrow was 1 for 4 on throws of 10-plus Air Yards and 0 for 2 on throws of 15-plus Air Yards.

AIR YARDS THIS SEASON: Burrow has been 11 for 38 on throws of 10-plus Air Yards and 2 for 22 on throws of 15-plus Air Yards.

TOUCHDOWNS: Burrow has thrown just two scores this season. Both came against Baltimore in Week 2, both were throws inside the five-yard line, both were to Tee
Higgins. That's it. Even his red-zone stats are unfathomable -- he's completed 7 of 15 passes in the red zone. At least 9 of those 15 throws went into the end
zone.
</item 32>

<item 33>
MVPs for all 16 AFC teams at the midpoint (Yardbarker, 2023-11-09T21:51:02+00:00)
MIAMI DOLPHINS | WR Tyreek Hill: Arguably the best wide receiver in the NFL, Hill has somehow reached a new level of dominance in his age-29 season. The
seven-time Pro Bowler leads the NFL in receiving yards (1,076) and TD catches (eight). As long as the Dolphins maintain their success on offense down the
stretch, Hill should earn Offensive Player of the Year honors. He's on pace to break Calvin Johnson's season record for receiving yards (1,964).



NEW YORK JETS | LB Quincy Williams: While his brother, Quinnen, may be the better-known Jets defender in the family, Williams — the AFC Defensive Player of the
Month for October — is also making a name for himself. Aside from being the fifth-highest-rated linebacker in the NFL by Pro Football Focus, Williams is tied
for 17th in tackles (74) and tied for ninth in TFLs (nine) among all defenders.
</item 33>

<item 34>
Christmas Day preview: 49ers, Ravens square off in potential Super Bowl sneak peek (Yardbarker, 2023-12-24T23:34:39+00:00)
McCaffrey, who seems poised to win the Offensive Player of the Year Award, leads the NFL in rushing (1,292 yards) and total yards per game (128.6) this season
and is tied for first in touchdowns (20).

Ravens TE Isaiah Likely thriving in the starting role: The Ravens offense hasn't lost a step since losing one of the team's top weapons in tight end Mark
Andrews (ankle) in Week 11, largely due to Likely's emergence.

The second-year tight end has averaged 64.3 yards over his last three games, which is particularly noteworthy since Andrews averaged 54.4 yards per game before
sustaining his injury.

Should we expect a defensive battle?: While both teams boast offenses that rank in the top five in yardage and scoring, this may not turn into a shootout.
</item 34>

<item 35>
Fantasy Football Week 16 Wide Receiver Preview: Waiver adds, DFS plays, and more (CBSSports.com, 2023-12-19T20:51:32+00:00)
There will be many difficult decisions to be made in Week 16. It is the Fantasy Football semifinals in most leagues and there is a certain finality to every
decision. There's no shortage of things to stress over as a Fantasy manager. Whether or not you start Rashee Rice shouldn't be one of them. While Rice is just
WR29 on the season, that's mostly due to a slow start, and his hot start has pushed him into must-start, borderline top-12 territory.

The Chiefs' star rookie has four straight games with at least nine targets and he's scored at least 14 PPR Fantasy points in all of those games. Since Week 12,
only Deebo Samuel and CeeDee Lamb have scored more PPR Fantasy points that Rice.

On paper, the matchup doesn't look great, because the Raiders have been above average in terms of limiting the Fantasy production of wide receivers.
</item 35>

<item 36>
Dolphins vs. Eagles live score, updates, highlights from NFL 'Sunday Night Football' game (Sporting News, 2023-10-22T23:28:04+00:00)
Check out all of the key moments you might have missed.

Dolphins vs. Eagles final score

1 2 3 4 F Dolphins 3 7 7 0 17 Eagles 3 14 7 7 31

Dolphins vs. Eagles results, highlights from 'Sunday Night Football'

(All times Eastern)

Final: Eagles 31, Dolphins 17

11:16 p.m. — And that'll do it. The Eagles finish strong and get back on track with an impressive win over the Dolphins.

11:12 p.m. — D'Andre Swift goes 22 yards for a first down, so the Eagles will simply kneel it the rest of the way.

11:09 p.m. — The Eagles take over on downs and will have the chance to just run out the clock on Miami.

11:07 p.m.
</item 36>

<item 37>
Dolphins vs. Eagles live score, updates, highlights from NFL 'Sunday Night Football' game (Sporting News, 2023-10-22T23:28:04+00:00)
It'll be 3rd & Goal from the 12.

8:48 p.m. — Jaylen Waddle goes airborne to bring down a Tagovailoa pass on third down and set up a 1st & Goal opportunity for Miami.

8:46 p.m. FUMBLE — Jaelan Phillips and Bradley Chubb get in there to knock the ball out of Hurts' hands! The Dolphins have an instant scoring opportunity just
outside the red zone.

Bradley Chubb jars it loose and the Dolphins take over with great field position



📺: #MIAvsPHI on NBC

📱: Stream on #NFLPlus https://t.co/0rIJPFKW6L pic.twitter.com/DDjWGEWofz — NFL (@NFL) October 23, 2023

8:44 p.m. — David Long is all over Hurts, quickly deflecting his second down pass. A key 3rd & 8 here for the Eagles.
</item 37>

<item 38>
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
</item 38>

<item 39>
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
</item 39>

<item 40>
DraftKings Picks Week 14: Best NFL DFS lineup advice for daily fantasy football tournaments (Sporting News, 2023-12-07T22:18:00+00:00)
We're especially fond of the 22 targets with a leaky secondary once again on tap. Moore cooked the Lions to the tune of almost 23 fantasy points in Week 12, and
being well-rested off the bye means we're likely to see more of the same.

WEEK 14 PPR RANKINGS:

QBs | RBs | WRs | TEs | D/STs | Kickers

WR Drake London, Falcons vs. Buccaneers ($4,600). When we see the Buccaneers on the schedule, we immediately look to see what WR1 is going to have a big day. It
wasn't the greatest feeling in the world seeing the Falcons this week, but we're confident in London's ability to make this pick worth our while. He hauled in
six-of-seven targets for 54 yards and almost added a score in Week 7 against Tampa, indicating history is on our side here.
</item 40>

<item 41>
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
</item 41>

<item 42>
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
</item 42>

<item 43>
NFL Fantasy Football Week 6 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-12T21:23:16+00:00)
His high-value touch share is the icing on the cake and hopefully can lead to him scoring just over 12 PPR points for the second week in a row. I'd take my
chances with him over low-upside running backs like Gus Edwards, Antonio Gibson and Jonathan Taylor (for now he's low-upside).

LAST WEEK: I surmised that Flowers would be targeted more if the Ravens run game couldn't get going. Sure enough, it didn't, and Flowers saw a season-high 11
targets at Pittsburgh. Unfortunately, Flowers underwhelmingly caught just five balls for 73 yards with two drops that would have pushed his yardage over 100 had
he held on to both targets.

AIR YARDS: Flowers was targeted downfield more last week than in any game this season with a 15.27 Average Depth of Target (ADOT). It was the first time he had
a double-digit ADOT and more than six targets.
</item 43>

<item 44>
Fantasy Football Week 13 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-11-30T22:10:58+00:00)
Steve Spagnuolo's defense has been blitzing its tails off over the past two weeks (nearly 50% of their snaps against the Eagles and Raiders), so the hunch is
they'll apply maximum pressure to that O-line in an effort to make Love more off-target than he's been and, hopefully, get some turnovers. Only one receiver has
had more than 15 PPR points against the Chiefs in their past four games (Jakobi Meyers last week), making it tough to expect grand games from Love's favorite
targets. I'd expect shutdown cornerback L'Jarius Sneed to line up across from Christian Watson the most -- and Jayden Reed the least.
</item 44>

<item 45>
NFL Best Bets Week 9: Colts clobber Panthers, Dolphins drop Chiefs in Germany (Sporting News, 2023-11-03T20:56:00+00:00)
JUMP TO: Spreads | Moneylines | Over/under | Player props

NFL Best Bets Week 9: Against the spread

All odds courtesy of Caesars Sportsbook

Colts (-2.5) at Panthers

Bryce Young has looked much better now that he has settled into his rookie season with the Panthers, but Carolina still seems like a work in progress. The
Colts, meanwhile, have quietly been operating a pretty decent offense — much better than their three-game losing streak might suggest. Indy has scored 65 points
between Week 7 and 8, 29 more points than Panthers have scored in that span.

In a Gardner Minshew-Bryce Young battle, we'll take the 'Shew — and the backfield committee of Jonathan Taylor and Zack Moss against Carolina's abysmal run
defense just serves as icing on the proverbial cake.
</item 45>

<item 46>
NFL Fantasy Football Week 6 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-12T21:23:16+00:00)
He and his fresh legs are back just in time as rookie sensation De'Von Achane hit Injured Reserve this week with a knee injury.

DOLPHINS: Aren't quite the bastion of rushing that some headlines have made them out to be, but they're just over league average in rush rate at 43.7% and above
league average in RB runs per game at 22.6. All of these numbers are weighted heavily by their 50-point win over Denver in Week 3 when they ran the ball 42
times with their running backs. Their 7.6 yard per carry average is due to Achane's breakout start more than anything else.
</item 46>

<item 47>
Dolphins vs. Eagles live score, updates, highlights from NFL 'Sunday Night Football' game (Sporting News, 2023-10-22T23:28:04+00:00)
A key 3rd & 8 here for the Eagles.

8:40 p.m. — The Eagles quickly force a punt after Tagovailoa's pass on 3rd & 15 falls short. So, neither of these explosive offenses start with a touchdown.

8:39 p.m. — Haason Reddick swallows up Raheem Mostert for a five-yard loss to quickly set the Dolphins back.

8:38 p.m. — Tyreek Hill is on the board with a 15-yard catch on a Tagovailoa pass.

Eagles 3, Dolphins 0

8:33 p.m. FIELD GOAL — Elliott connects from 24 yards out, and the Eagles strike first despite their promising drive fizzling out.

8:32 p.m. — Eagles fans are unhappy after Philadelphia opts for three consecutive runs in the red zone. It'll be a field goal try for Jake Elliott.

8:30 p.m.
</item 47>

<item 48>
Fantasy Football WR PPR Rankings Week 13: Who to start, best sleepers at wide receiver (Sporting News, 2023-11-29T20:27:00+00:00)
Tyreek Hill (@ Commanders in Week 13) did his thing despite a tough matchup in Week 12, resulting in Miami's WR1 returning to the top spot against a putrid
Washington stop unit. CeeDee Lamb (vs. Seahawks) isn't far behind, as Dallas' WR1 aims to maintain his efficient play against Seattle.

Amon-Ra St. Brown (@ Saints) has been a target hog the past several weeks, seeing at least nine looks in each of his past six games. Don't expect his workload
to decrease in a bounce-back game for the Lions. Keenan Allen (@ Patriots) looks to continue his dominant ways, as Justin Herbert's go-to target routinely has
double-digit target potential.

Mike Evans (vs. Panthers) sits in our top five for Week 13, totaling four TDs over the past three weeks. A.J. Brown (vs.
</item 48>

<item 49>
FanDuel 'Prop Stars' Picks: Best bets for Travis Kelce & Taylor Swift-inspired props for Chiefs-Jets Week 4 (Sporting News, 2023-09-28T21:41:00+00:00)
It's not glamorous, but they'll score again. Shutting out NFL teams is no easy task, but the Chiefs defense is much improved and the Jets look like a team that
doesn’t have a quarterback. Don’t consider it likely to hit, but +3400 wagers hardly ever are. Consider this a fun bet, unless you’re a Jets fan, and see how
long K.C. can keep NYC off the scoreboard. They might just do it for the entire game.

Our pick: Bet on it (with a small wager)
</item 49>

<item 50>
NFL roundup: Pats hit new low as Lions reach 9-3 for first time since 1962 (The Guardian, 2023-12-03T22:11:26+00:00)
The Titans blew a 17-7 lead. Nick Folk missed his first extra point this season with 5:26 left in regulation, which would have put Tennessee up 26-25.

skip past newsletter promotion Sign up to Soccer with Jonathan Wilson Free weekly newsletter Jonathan Wilson brings expert analysis on the biggest stories from
European soccer Privacy Notice: Newsletters may contain info about charities, online ads, and content funded by outside parties. For more information see our
Newsletters may contain info about charities, online ads, and content funded by outside parties. For more information see our Privacy Policy . We use Google
reCaptcha to protect our website and the Google Privacy Policy and Terms of Service apply. after newsletter promotion

Miami Dolphins 45–15 Washington Commanders

Tyreek Hill had two touchdowns among his 157 receiving yards to help the Miami Dolphins rout the Washington Commanders (4-9). The AFC East-leading Dolphins are
9-3 for the first time since 2001.
</item 50>

<item 51>
NFL Fantasy Football Week 5 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-04T21:56:25+00:00)
On the year they're throwing the ball 54.6% of the time, 10th-lowest, but in their wins they've thrown the ball 46.4% of their snaps. In losses they've thrown
64.4% of the time. The Colts are one-point home favorites, which means at minimum the game should be competitive, which does not help the pass-run ratio
outlook.

COLTS: Are a frozen yogurt sundae of a dream matchup for Hopkins. On the season outside receivers like Hopkins have caught 72.3% of their targets for 12.7 yards
per catch against the Colts. The catch rate is bottom six, the yardage rate is top-12, but the Colts defense leads the NFL in missed tackles on passing plays
with 10. Indianapolis also lost their top outside corner, Dallis Flowers, to a torn Achilles.
</item 51>

<item 52>
Fantasy Football WR PPR Rankings Week 13: Who to start, best sleepers at wide receiver (Sporting News, 2023-11-29T20:27:00+00:00)
Stroud's.

WEEK 13 FANTASY ADVICE: Sleepers | Busts | Projections | Start-sit

Who are the best fantasy WR sleepers, waiver pickups in Week 13?

Curtis Samuel (vs. Dolphins), Jayden Reed (vs. Chiefs), Lynn Bowden Jr. (vs. Lions), Jonathan Mingo (@ Buccaneers), Cedric Tillman (@ Rams), and DeVante Parker
(vs. Chargers) profile as this week's crop of sleepers and waiver-wire adds.

Samuel saw a heavy dosage of targets in Week 12, and with Washington passing at a high clip, another high-usage day could be in store. Reed's scored in three
straight games and while he has a higher ceiling in standard formats, he still has flex appeal in a potentially negative game script.

Bowden Jr., alongside A.T.
</item 52>

<item 53>
NFL Fantasy Football Week 6 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-12T21:23:16+00:00)
McDANIEL: Prefers to use two running backs from game to game, and in Miami's past three games there's been a relatively close split in carries between two backs
(18 and 13 in Week 3, 8 and 7 in Week 4, 11 and 10 in Week 5).

LAST YEAR: Wilson was efficient in his eight games in Miami, averaging 4.7 yards per carry with three touchdowns and just over 60 total yards per matchup. That
includes two games he barely played in (under 20 snaps) -- take those out and he averaged 78.2 total yards per game.

SALVON AHMED? Was a healthy scratch last week and figures to be no better than a special-teams player this week. In fact, in the seven games Raheem Mostert,
Wilson and Ahmed were all active for with the Dolphins last year, Ahmed played 18 offensive snaps and 25 special-teams snaps.
</item 53>

<item 54>
Dolphins vs. Eagles live score, updates, highlights from NFL 'Sunday Night Football' game (Sporting News, 2023-10-22T23:28:04+00:00)
to stretch the lead back to two possessions.

10:05 p.m. — Miami will punt from just past midfield, after a drive that started with some promise.

10:04 p.m. — The Dolphins face 3rd & 18, largely the result of a holding penalty. Tagovailoa has already converted one of these tonight.

10:01 p.m. — Raheem Mostert has struggled to get going, but he takes the Dolphins into Eagles territory with a solid 7-yard gain.

10:00 p.m. — Hill picks up where he left off and goes 17 yards to give this Dolphins drive an early spark.

End of first half: Eagles 17, Dolphins 10

9:45 p.m. — Any hope the Eagles had of doing something before halftime ends with a Jaelan Phillips sack of Hurts. Philadelphia will go to halftime up 17-10.

9:43 p.m. — Hill is up to 65 yards on eight catches.
</item 54>

<item 55>
Dolphins vs. Eagles live score, updates, highlights from NFL 'Sunday Night Football' game (Sporting News, 2023-10-22T23:28:04+00:00)
That's a pick six, and the Dolphins have tied the game just like that after the no-call.

10:22 p.m. — This looks more obvious than a certain penalty on James Bradberry from February.

Looks like the refs missed another facemask, this one on James Bradberry pic.twitter.com/fYYeWnEWaGhttps://t.co/54oa1jfeHf — Dov Kleiman (@NFL_DovKleiman)
October 23, 2023

10:20 p.m. — James Bradberry appears to get ahold of Cedrick Wilson's face mask on 4th & 3, but no flag flies. The Dolphins turn it over on downs.

10:19 p.m. — Tyreek Hill just can't hang onto what would've been a big-time catch. It's not a fumble, but it is a fourth down for the Dolphins. They'll go for
it.

10:16 p.m.
</item 55>

<item 56>
NFL roundup: Pats hit new low as Lions reach 9-3 for first time since 1962 (The Guardian, 2023-12-03T22:11:26+00:00)
Keenan Allen came back from a thigh injury to catch five passes for 58 yards. But the only scoring the Chargers could manage was a pair of 38-yard field goals
in the second quarter. New England made a quarterback switch, benching Mac Jones in favor of Bailey Zappe. But the Patriots (2-10) still lost their fifth
straight game. For the first time in franchise history, the Patriots have been shut out at home twice in a season. Zappe was sacked five times – two by Khalil
Mack, giving him a career-high 15 for the season and 99 in his career. Zappe was 13 for 25 for 141 yards; he did avoid an interception for the first time in
three games, the previous two in relief of Jones.

Tyreek Hill had two touchdowns among his 157 yards receiving to help the Miami Dolphins rout the Washington Commanders 45-15.
</item 56>

<item 57>
Dolphins vs. Eagles live score, updates, highlights from NFL 'Sunday Night Football' game (Sporting News, 2023-10-22T23:28:04+00:00)
9:43 p.m. — Hill is up to 65 yards on eight catches. No other Dolphins receiver has more than two catches, and Hill now has 879 yards on the season.

Eagles 17, Dolphins 10

9:39 p.m. TOUCHDOWN — Tagovailoa hits Hill in the end zone for a touchdown! Hill was facing double coverage but simply beat both defenders. Miami has the
deficit back to one possession.

9:38 p.m. — Rookie Nolan Smith sacks Tagovailoa with 44 seconds remaining in the half.

9:37 p.m. — The Dolphins are down to the Eagles' 29-yard line after catches by Braxton Berrios and Tyreek Hill. They have a chance to score on back-to-back
possessions, as they get the ball out of halftime.

9:33 p.m. TWO-MINUTE WARNING — Woah!
</item 57>

<item 58>
Fantasy Football WR PPR Rankings Week 13: Who to start, best sleepers at wide receiver (Sporting News, 2023-11-29T20:27:00+00:00)
A.J. Brown (vs. 49ers) faces a sound San Francisco coverage unit in Sunday's NFC Championship rematch, but his ability to dominate in the red zone still makes
him a top-10 PPR play. Ja'Marr Chase (@ Jaguars) averaged over 20 yards per reception in Week 12, and although the drop-off from Joe Burrow to Jake Browning is
massive, Chase still has WR1 upside given his usage.

Michael Pittman Jr. (@ Texans) faces a Texans' pass defense that's been shredded the past few weeks, so another 100-yard receiving day could be in store. Jaylen
Waddle (@ Commanders) is also looking to post his second consecutive 100-plus yard day, which could come to fruition against a struggling Commanders' coverage
unit. Tank Dell (vs. Broncos) continues to rise up our PPR rankings, rounding out our top 10 as a trusted option of C.J. Stroud's.
</item 58>

<item 59>
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
</item 59>

<item 60>
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
</item 60>

<item 61>
Dolphins vs. Eagles live score, updates, highlights from NFL 'Sunday Night Football' game (Sporting News, 2023-10-22T23:28:04+00:00)
After the Eagles' defense clamped down on Tagovailoa and the Dolphins, the offense pieced together a touchdown drive spanning nearly seven minutes to ice the
game.

A.J. Brown overcame a slow start and shined for the Eagles, posting 137 yards and a touchdown in the win. Brown has now ripped off five consecutive games of at
least 125 yards and seems to be on his way to a career-best season.

Hurts, Brown and the Eagles have higher goals than just individual success, and a win over one of the AFC's best teams is one way to make a statement.
Philadelphia moves to 6-1, while the Dolphins miss a chance to open up a two-game lead in the AFC East and fall to 5-2 ahead of a matchup with the Patriots.

MORE: Latest injury updates on Dolphins WR Jaylen Waddle

The Sporting News tracked live scoring updates and highlights from "Sunday Night Football" between the Dolphins and Eagles. Check out all of the key moments you
might have missed.
</item 61>

<item 62>
N.F.L. Executives on the Biggest Questions Facing Super Bowl Contenders (The New York Times, 2023-10-16T23:58:53+00:00)
GO DEEPER Behind the AI magic that lets Amazon's Prime Vision show the NFL like never before

• Buffalo Bills: Are the defensive injuries too much to overcome?

What I think: The rest of the conference is the key variable. The Bills are diminished. They have imperfections. But Buffalo already defeated the Dolphins. The
other AFC teams aren’t exactly peaking. Kansas City and Cincinnati appear easier to defend this season than in the recent past. We’ll see how well the Dolphins
hold up. This can still be Buffalo’s season.

Exec comment: “Matt Milano and Tre’Davious White, those are huge injuries for that football team. The Bills are not as dominant as they felt going into it with
so many injuries on defense. But the division they are in, they will still be able to slug it out and get to where they need to go. Nobody has really emerged in
the AFC to unseat these teams at the top.”
</item 62>

<item 63>
N.F.L. Executives on the Biggest Questions Facing Super Bowl Contenders (The New York Times, 2023-10-16T23:58:53+00:00)
There isn’t a Josh Allen on the Giants or Vikings.

Minnesota in particular finds itself in an interesting spot, with Kirk Cousins approaching free agency and playing well enough, most likely, to keep the Vikings
from climbing too high in the draft order.

Advertisement

“They already gave a contract to (tight end) T.J. Hockenson, and they are going to give one to (receiver) Justin Jefferson,” an exec said. “Now, what? Are you
going into the rookie pool and taking the fourth-best quarterback in the draft? You can look like Atlanta with (Desmond) Ridder pretty quick.”

6. Two-minute drill: There were already many ways to win a game before the Dolphins invented another one.

The Dolphins were one of 223 teams since 2000 to be favored by at least 13.5 points in a regular-season or playoff game. Seventeen of those 223 teams fell
behind by 14 or more. Six of the 17 won anyway.
</item 63>

<item 64>
NFL Fantasy Football Week 6 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-12T21:23:16+00:00)
UNCATCHABLE: Of the 43 targets Garrett Wilson has seen from Zach Wilson, 14 of them (33%) have been uncatchable.

EAGLES: Have let up a slew of receiving numbers by virtue of seeing so many passes thrown against them (27.4 targets per game just to wide receivers). So
naturally they rank in the bottom five in catches allowed, YAC allowed and touchdowns allowed. But on per-target metrics they're doing all right. A 60.6%
completion rate allowed is 10th-best, 11.4 yards per catch allowed is seventh-best and 4.06 YAC/reception is on par with the league average.

PASS RUSH: As overworked as the Eagles secondary has been, their pass rush has been solid. Against soft O-lines in Washington and L.A., the Eagles have put
pressure on at a rate of 37.6% and have blitzed 33.3% of snaps.
</item 64>

<item 65>
FanDuel 'Prop Stars' Picks: Best bets for Travis Kelce & Taylor Swift-inspired props for Chiefs-Jets Week 4 (Sporting News, 2023-09-28T21:41:00+00:00)
87. This is a New York defense that's been very good, although not quite great, over the last year and continues to improve. Robert Saleh's troops have held
opponents in check, giving their anemic offense a fighting chance most weeks, but if there's one thing that Gang Green struggles to defend against, it's the
tight end. They're giving up the fifth-most receiving yards to the position this season (56.3) and just allowed Pharoah Brown to beat them for a long 58-yard
touchdown last Sunday. Getting +140 is a nice price for the best pass-catcher in this game, but there's too much blowout potential to warrant taking a flier
here.

Our pick: Pass

Seeing Red: Chiefs to Win by 22+ (+380)

If this bet ends up cashing, the "Seeing Red" tagline would apply to a lot more than just Chiefs fans.
</item 65>

<item 66>
Dolphins vs. Eagles live score, updates, highlights from NFL 'Sunday Night Football' game (Sporting News, 2023-10-22T23:28:04+00:00)
— Oh yeah, Julio Jones is back in the NFL.

What a group 🤌 pic.twitter.com/fV9WgGOpRY — Philadelphia Eagles (@Eagles) October 22, 2023

7:41 p.m. — Believe it or not, Tyreek Hill has a chance to break 1,000 yards tonight. He would need 186, but with the way he's played this season, that at least
seems to be within reach.

7:29 p.m. — Don't forget the Eagles will be back in Kelly Green tonight. This will be a fascinating uniform matchup.

7:20 p.m. — It's 55 degrees and dropping in Philadelphia. That's not unusual for the Eagles, but will Tua Tagovailoa be able to rise to the occasion in cooler
weather?

What channel is Dolphins vs. Eagles on today?

Game: Dolphins vs. Eagles

Dolphins vs.
</item 66>

<item 67>
NFL Best Bets Week 9: Colts clobber Panthers, Dolphins drop Chiefs in Germany (Sporting News, 2023-11-03T20:56:00+00:00)
Expect more of the same from QB Justin Herbert and his stud back against the Jets, who have allowed 46 receiving yards per game to RBs (sixth-most in the NFL).
They'll try to shut him down this Sunday, but it would feel so empty without AE!

Jahan Dotson, WR, Commanders @ Patriots — OVER 45.5 receiving yards

We've come full circle with Dotson, who made our preseason sleepers column then faded into early-season obscurity only to emerge again over the past two weeks.
Curtis Samuel was clearly not right last week, and the second-year wideout more than made up for it. Dotson caught 8-of-10 targets for 108 yards and a TD, a
stellar encore to his Week 7 line of five catches for 43 yards. The banged-up Patriots defense will be primarily focused on stopping stud Commanders receiver
Terry McLaurin, which should open up plenty of opportunities for Dotson.
</item 67>

<item 68>
NFL Fantasy Football Week 5 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-04T21:56:25+00:00)
I think he'll continue to see high-value targets near the end zone and, hopefully, a small uptick in targets. I'd start him as a No. 2 receiver over Terry
McLaurin, any Broncos wideouts and any Packers wideouts.

LAST WEEK: Was so close to having a monster game. He was the designed target on a trick play and saw the ball fall just past his outstretched hands in the end
zone. He also had two targets inside the 10-yard line (his first two of the season) and two end-zone targets including the aforementioned trick play (his first
two of the season). The heavy dose of targets inside the 10 and into the end zone are positive signs that the Titans are trying to get him going as a playmaker.

FILM: Hopkins might be 31 years old but he moved really, really well, creating separation with outstanding footwork and having just enough speed to stay a step
ahead of defenders.
</item 68>

<item 69>
NFL Fantasy Football Week 6 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-12T21:23:16+00:00)
COLLINS: It's minor but Collins does see a higher target per route run rate against man coverage (25%) than zone (22.2%). He also has a higher catch rate
(87.5%) and higher receiving average (23.3) against man than zone (65.4% and 17.8). However, he's run nearly four times as many routes against zone coverage
compared to man.

MATCHUPS: It's also worth pointing out that Collins' two really big games came in Week 2 (vs. Indianapolis) and Week 4 (vs. Pittsburgh), both teams that rank in
the bottom-10 in Fantasy points allowed to wide receivers. The Saints are in the top half.

STROUD: Has been outstanding against man coverage all year, completing a higher rate of passes (67.6%) for 3.5 more yards per attempt (11.1) than versus zone
(7.6).
</item 69>

<item 70>
Fantasy Football WR PPR Rankings Week 13: Who to start, best sleepers at wide receiver (Sporting News, 2023-11-29T20:27:00+00:00)
Geno Smith's struggles complicate their fantasy prospects, as well.

What are the biggest WR injuries in Week 13?

Entering Week 12, Tee Higgins (hamstring), Noah Brown (knee), Treylon Burks (concussion), and Kadarius Toney (ankle, hip) all failed to suit up due to nagging
injuries. For now, we've placed Higgins and Brown in our Week 13 list and omitted Burks and Toney. We'll also be eyeing the Chargers' Josh Palmer (knee), who's
eligible to come off the IR this week but probably won't suit up against New England.

Last week, the most notable WRs to exit their respective games with injuries include Chris Olave (concussion), Amari Cooper (ribs), Demario Douglas
(concussion), and Rashid Shaheed (thigh). We've left all four WRs out of our initial Week 13 list but will closely monitor their teams' injury reports to gauge
their playing prospects.
</item 70>

<item 71>
Fantasy Football Week 16 Wide Receiver Preview: Waiver adds, DFS plays, and more (CBSSports.com, 2023-12-19T20:51:32+00:00)
Demario Douglas WR NE New England • #81

Age: 23 • Experience: Rookie Weekly Breakdown MATCHUP @ DEN DEN -6.5 O/U 34.5 OPP VS WR 14th WR RNK 46th ROSTERED 40% YTD Stats REC 39 TAR 60 REYDS 443 TD 0
FPTS/G 7.7 Douglas' first game back was a disappointment but he still has nine targets in two of his last three games and seven targets in four of his last five
games. I'm hoping for a bounce back in Week 16.
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
Dolphins vs. Eagles live score, updates, highlights from NFL 'Sunday Night Football' game (Sporting News, 2023-10-22T23:28:04+00:00)
They'll go for it.

10:16 p.m. — Raheem Mostert is running wild. He rips off back-to-back runs of 21 and 15 yards, respectively, and the Dolphins are deep into Eagles territory
just like that.

Raheem Mostert moves the chains in back to back plays pic.twitter.com/cKtaJouqeZ — DIE-HARD 🦅 Fans (@Eaglesfans9) October 23, 2023

10:11 p.m. — The Eagles manage only one first down on the drive, and the Dolphins have a chance to get pretty strong field position after Philadelphia's first
punt of the night.

10:10 p.m. — Zach Sieler gets to Hurts and takes him down for a loss of three yards.

10:06 p.m. — The Eagles will be backed up deep in their own territory to start the next drive, but it's a big opportunity for Hurts and co. to stretch the lead
back to two possessions.
</item 73>

<item 74>
NFL Fantasy Football Week 5 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-04T21:56:25+00:00)
He got open on a handful of third-down plays when he didn't get a target last week.

TANNEHILL: Is the top problem Hopkins has right now. Hopkins' 38-yard deep-ball catch last week could have been as many as 50 yards if Tannehill hit him in
stride, and the gimmick-play touchdown would have obviously been a massive point-getter had Tannehill been slightly more on-target there. There was even a
target into the end zone that was a miscommunication between Hopkins and Tannehill. On the season Tannehill has a 13% off-target rate but over the past two
games that number has ballooned to 18%, which is ... yikes.

TITANS: The second problem Hopkins has right now is a Titans offense that's stayed true to their run-first preference -- at least when they win.
</item 74>

<item 75>
Fantasy Football Week 16 Wide Receiver Preview: Waiver adds, DFS plays, and more (CBSSports.com, 2023-12-19T20:51:32+00:00)
Weekly Breakdown MATCHUP @ NYJ NYJ -3 O/U 37 OPP VS WR 14th WR RNK 30th ROSTERED 58% YTD Stats REC 56 TAR 77 REYDS 549 TD 5 FPTS/G 11 Samuel was the top waiver
wire add at wide receiver last week as well, but he still sits at just 57% rostered. He has three straight games with double-digit Fantasy points and he's
topped 19 in two of those games. The matchup isn't great but it's easier for him than it is for Terry McLaurin and Jahan Dotson.
</item 75>

<item 76>
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
</item 76>

<item 77>
Fantasy Football Week 16 Wide Receiver Preview: Waiver adds, DFS plays, and more (CBSSports.com, 2023-12-19T20:51:32+00:00)
The Texans offense looks like one to avoid as long as C.J. Stroud is out. Ja'Marr Chase WR CIN Cincinnati • #1

Age: 23 • Experience: 3 yrs. Tee Higgins is a No. 2 receiver and Tyler Boyd is a fine flex.

WR Preview Numbers to Know

69.9% -- Nearly 70% of Rashee Rice's receiving yards have come after the catch, that's tops amongst wide receivers with at least 50 targets.

-- Nearly 70% of Rashee Rice's receiving yards have come after the catch, that's tops amongst wide receivers with at least 50 targets. 11.4 -- Deebo Samuel is
on pace to become just the fourth player since 1992 to average more than 11 yards per target with at least 60 targets in multiple seasons. The other three are
Tyreek Hill, Tyler Lockett, and DeSean Jackson.
</item 77>

<item 78>
NFL Best Bets Week 9: Colts clobber Panthers, Dolphins drop Chiefs in Germany (Sporting News, 2023-11-03T20:56:00+00:00)
Chiefs (Germany game)

We get it — the Dolphins have yet to defeat a true contender this season and the Chiefs are the reigning champions of the NFL. However, Kansas City just got its
doors blown off by Russell Wilson and the lowly Broncos, a game in which Patrick Mahomes and company suffered five turnovers and three sacks.

We expect a statement win from Mike McDaniel and the Dolphins on neutral ground. The Dolphins have been in Frankfurt since early this week, having flown
overnight on Monday to give them multiple days to acclimate to the time difference and their surroundings. The Chiefs, meanwhile, waited until after practice on
Thursday to take the 11-hour voyage.

This also happens to be a revenge game for former Chiefs receiver Tyreek Hill, who's on pace for multiple single-season receiving records and poised for his
first Offensive Player of the Year award.
</item 78>

<item 79>
NFL Fantasy Football Week 5 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-04T21:56:25+00:00)
3 wide receiver in the offense behind Justin Jefferson and K.J. Osborn, but even with the donut Addison still has a five-catch lead on Osborn on the season with
68 more yards gained.

THIS WEEK: You can bank on the Vikings reversing course against the Chiefs. Two of the past three offenses against Kansas City wound up throwing at least 71% of
their snaps. The team that didn't? Chicago, who was so badly blown out by halftime that they ran the ball 55% of their second-half snaps just to get out of
Arrowhead.

CHIEFS: Have been stout against receivers all season long, ranking top-five in catch rate allowed (53.4%) and YAC/reception allowed (2.96), and sixth-best in
yards per catch given up (11.1).
</item 79>

<item 80>
MVPs for all 16 AFC teams at the midpoint (Yardbarker, 2023-11-09T21:51:02+00:00)
Stroud: Stroud is having one of the best statistical rookie seasons of all time. He ranks in the top 10 in passing yards (2,270), touchdown passes (14) and
passer rating (102.9), and has only thrown one interception. The Texans (4-4) aren’t close to a .500 team without him.

INDIANAPOLIS COLTS | RB Zack Moss: With Jonathan Taylor missing the first four weeks of the season, Moss stepped up in a big way. After not playing in Week 1,
he rushed for 445 yards and scored four touchdowns over his next four games to help Indy win three of four and stay in the thick of the AFC South. Moss is
second in the NFL in rushing, 37 yards behind San Francisco's Christian McCaffrey.
</item 80>

<item 81>
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
</item 81>

<item 82>
Bills vs. Chargers odds, props, predictions: Buffalo faces struggling L.A. with eyes on AFC playoff picture (Sporting News, 2023-12-21T19:35:54+00:00)
With Buffalo likely to cruise, this isn’t a far-fetched chain of events to target.

Gerald Everett OVER receiving yards

Whether Easton Stick continues to play decently or if he doesn’t, this prop line for receiving yards should be attainable for Chargers TE Gerald Everett in Week
16. Everett has tallied 41, 39, 44 and 43 receiving yards in his last four outings. The Bills are likely to score a lot, necessitating that Los Angeles passes a
lot. Even if this wasn’t a game with such a big spread, it is a small ask to suggest that Everett can net half of what he’s averaged in the past month.

Bills vs. Chargers prediction: Bills continue to roll

The Chargers new coach isn’t going to be able to right all his team’s wrongs overnight, but losing by less than 38 would be a great place to start.
</item 82>

<item 83>
2023 NFL Week 11 injury roundup: Rams WR Cooper Kupp, Seahawks RB Kenneth Walker III among stars to exit (CBSSports.com, 2023-11-19T23:51:49+00:00)
I agree to receive the "Pick Six Newsletter" and marketing communications, updates, special offers (including partner offers), and other information from CBS
Sports and the Paramount family of companies. By pressing sign up, I confirm that I have read and agree to the Terms of Use and acknowledge Paramount's Privacy
Policy See All Newsletters Please check the opt-in box to acknowledge that you would like to subscribe. Thanks for signing up! Keep an eye on your inbox. Sorry!
There was an error processing your subscription.

WR

Tyreek Hill (Dolphins) briefly left Miami's game against the Raiders while nursing a hand injury and was ruled questionable to return, but he took the field
again soon afterward, securing seven catches for 80 yards and a score before halftime. Hill has left multiple games this year, only to return in all of them.

Cooper Kupp (Rams) had just a single catch against the Seahawks before suffering an ankle injury while blocking.
</item 83>

<item 84>
NFL Fantasy Football Week 5 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-04T21:56:25+00:00)
At least we can say over the past two weeks that the Dolphins have used both backs rather evenly in touches, even though WHERE they've gotten them haven't been
as equal.

Start 'em 10.0-7.00 6.99 - 5.01 5.00 - 0 Sit 'em

Start Him (Lineup Decisions) Projections powered by Sportsline DeAndre Hopkins WR TEN Tennessee • #10

Age: 31 • Experience: 10 yrs. Weekly Breakdown MATCHUP @ IND TEN -2 O/U 43 OPP VS WR 25th PROJ PTS 11.6 WR RNK 30th YTD Stats REC 18 TAR 31 REYDS 216 TD 0
FPTS/G 9.9 It's been cool to bench Hopkins so far this season but we're staring at a terrific matchup for him to take advantage of, just as other receivers have
as recently as last week (Puka Nacua).
</item 84>

<item 85>
Fantasy Football Week 13 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-11-30T22:10:58+00:00)
0-7.00 6.99 - 5.01 5.00 - 0 Sit 'em

The Lions need an answer for their ugly pass defense -- since their Week 9 bye they've been mostly great against the run (3.1 yards per carry allowed, one RB to
get 15-plus PPR points) but they rank bottom-six in completion rate allowed (68.4%), yards per attempt allowed (8.0), passing ADOT against (8.84 yards) and
defensive Expected Points Added (EPA) per dropback (-0.40, which is league-worst). And of their second-most eight passing touchdowns given up in three games,
five have come on passes of 15-plus Air Yards, which is tied for most in the league. They should catch a break with field stretcher Rashid Shaheed (thigh)
missing the matchup for New Orleans; Chris Olave looks like he will play.
</item 85>

<item 86>
Christmas Day preview: 49ers, Ravens square off in potential Super Bowl sneak peek (Yardbarker, 2023-12-24T23:34:39+00:00)
The 23-year-old ranks first in the NFL in QBR (76) and touchdown passes (29) while ranking third in passing yards (3,795), all of which are especially
impressive considering he's coming off offseason elbow surgery.

Jackson might not be posting the ridiculous numbers he did during his 2019 MVP campaign, but the adjustment in his play style in his first season under OC Todd
Monken has led to sustainable, winning football. Nevertheless, Baltimore needs an upset win for Jackson to capture another MVP Award, as he has thrown for 3,105
yards and 17 touchdowns while adding 741 yards on the ground with five scores this season.

49ers RB Christian McCaffrey: If Purdy wasn't playing at such a high level — and the MVP wasn't a quarterback-driven award — it'd be difficult to argue against
McCaffrey taking home the honors.
</item 86>

<item 87>
Fantasy Football Week 13 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-11-30T22:10:58+00:00)
Weekly Breakdown MATCHUP @ DAL DAL -9 O/U 47.5 OPP VS WR 17th PROJ PTS 11.1 WR RNK 22nd YTD Stats REC 43 TAR 80 REYDS 678 TD 3 FPTS/G 12.9 Metcalf has at least
nine targets in each of his past three games and in five of his past six. That kind of usage, especially in a game where the Seahawks figure to pass a bunch,
makes him a borderline must-start. But on the 12 targets he's seen that have traveled at least 15 Air Yards over the past four weeks, Metcalf has caught just
three. More alarming: Five of the targets were knocked away by defenders who were right on him when the ball arrived. He's not really zooming past anyone these
days, and he's not winning contested catches regularly either.
</item 87>

<item 88>
Fantasy Football WR PPR Rankings Week 14: Who to start, best sleepers at wide receiver (Sporting News, 2023-12-07T22:11:00+00:00)
Rams), Jonathan Mingo (@ Saints), and Xavier Gipson (vs. Texans) are our preferred waiver-wire pickups in PPR formats for Week 14.

Moore (vs. Jaguars) just saw 12 targets from Joe Flacco and is in line for another high-usage day with Amari Cooper (concussion) potentially sidelined. Brown (@
Jets) should also see a steady target share after Tank Dell's (leg) season-ending injury.

Beckham Jr. (vs. Rams) is also likely to see increased usage with Mark Andrews (leg, ankle) done for the regular season. Mingo (@ Saints) has seen six-plus
targets in his past four games, as his rapport with Bryce Young is strengthening by the week. Gipson led the Jets in receiving this past week (five receptions,
77 yards), and with New York routinely playing from behind, similar stat lines could be in the works.
</item 88>

<item 89>
N.F.L. Executives on the Biggest Questions Facing Super Bowl Contenders (The New York Times, 2023-10-16T23:58:53+00:00)
Nobody has really emerged in the AFC to unseat these teams at the top.”

• Miami Dolphins: Has Tua Tagovailoa proven he’s built to last?

What I think: It was tough watching Tagovailoa last season without worrying for his physical well-being. I’ve never felt that way watching him so far this
season, and will feel even better if he plays a full game against Philadelphia in Week 7. New England, Kansas City, the Jets (twice), Washington, Tennessee,
Dallas, Baltimore and Buffalo will all get their shots at Tagovailoa. I’m cautiously optimistic, but think there will be games when defense muddy the picture
sufficiently to create turnovers.

Exec comment: “Interesting team. They put up 70 on Denver and should have put up 70 on the Giants. No one can keep up with that kind of firepower. But they also
haven’t really beaten anyone. They were down 14 points to Carolina.
</item 89>

<item 90>
Dolphins vs. Eagles live score, updates, highlights from NFL 'Sunday Night Football' game (Sporting News, 2023-10-22T23:28:04+00:00)
The Eagles have a fresh set of downs.

10:52 p.m. — It's 4th & 1, but it looks like the Eagles will go for it deep in their own territory. It's a risk, even with the Brotherly Shove at their
disposal.

10:47 p.m. INTERCEPTION — The Dolphins have been impressive on third down tonight, but not that time. Tagovailoa's pass to Mostert is picked off by Darius Slay,
and Miami's defense suddenly has to meet the moment to keep this game within one possession.

10:44 p.m. — You think the Dolphins are happy to have Waddle back on the field? He rescues Miami's drive with a first down catch on 3rd & 12.

End of third quarter: Eagles 24, Dolphins 17

10:39 p.m. — Jaylen Waddle is back in the game and makes his presence felt with a first down catch to start the Dolphins' drive.
</item 90>

<item 91>
Fantasy Football WR PPR Rankings Week 14: Who to start, best sleepers at wide receiver (Sporting News, 2023-12-07T22:11:00+00:00)
The league's leading receiver, Tyreek Hill (vs. Titans in Week 14), stands as the unquestioned WR1 for Week 14 after torching the Commanders to the tune of five
catches, 157 yards, and two TDs. CeeDee Lamb (vs. Eagles) and the Cowboys look to avenge their road loss to the Eagles in Week 9, as Dallas' WR1 routinely has
double-digit reception potential.

Amon-Ra St. Brown (@ Bears) was relatively quiet this past week, but he's likely to rebound against a Bears defense that he totaled nine catches for 95 yards
against in Week 11. A.J. Brown (@ Cowboys) reached the century mark for the first time in four games in Week 13, and another high-usage day looks to be in store
for a potential shootout.

Keenan Allen (vs. Broncos) caught just five-of-nine targets last week, but as L.A.
</item 91>

<item 92>
N.F.L. Executives on the Biggest Questions Facing Super Bowl Contenders (The New York Times, 2023-10-16T23:58:53+00:00)
Receiving Yards Through Six NFL Games Rk Player Season Yds 1 Don Hutson 1942 819 2 Tyreek Hill 2023 812 3 Wes Welker 2011 785 4 Elroy Hirsch 1951 766 5 Santana
Moss 2005 743 6 DeAndre Hopkins 2015 726 7 Raymond Berry 1960 725 8 Jerry Rice 1986 723 9 Bob Boyd 1954 718 10 Bobby Mitchell 1962 714

• Jared Goff is enjoying the second-best start to a season for his career from a statistical standpoint, behind only his 2018 Super Bowl season with the Rams.
With the Lions at 5-1 and Goff heading toward a likely contract extension, could another team come after his injured backup, former University of Tennessee
quarterback Hendon Hooker?
</item 92>

<item 93>
Fantasy Football WR PPR Rankings Week 13: Who to start, best sleepers at wide receiver (Sporting News, 2023-11-29T20:27:00+00:00)
Dolphins), Puka Nacua and Tutu Atwell (vs. Browns), Drake London (@ Jets), and Tyler Lockett and Jaxon Smith-Njigba (@ Cowboys) this week.

McLaurin's averaging just 42 receiving yards over his past three games, and a heavy dosage of Jalen Ramsey hinders his Week 13 fantasy value.

Nacua and Atwell see a Cleveland coverage unit who entered Week 12 allowing the fewest fantasy points per game to WRs. Heck, even Cooper Kupp, who hasn't looked
like himself of late, could underperform again.

We can't trust London and the rest of the Falcons passing offense against a well-rounded Jets defense. To make matters worse, he'll be matched up against Sauce
Gardner for most of the day, too.

Lockett and Smith-Njigba might be better suited for the bench in PPR leagues against a lockdown Dallas coverage unit.
</item 93>

<item 94>
MVPs for all 16 AFC teams at the midpoint (Yardbarker, 2023-11-09T21:51:02+00:00)
NEW ENGLAND PATRIOTS | S Jabrill Peppers: There hasn't been much to get excited about in New England, but the versatile defensive back is a bright spot and a
respected voice in the locker room. Peppers has 54 tackles — three for loss — one interception, one forced fumble and a team-leading six passes defended. For a
team facing as many uncertainties as the Patriots are, Peppers is a solid piece to build around in the short term. — Colum Dell

AFC West

DENVER BRONCOS | CB Patrick Surtain II: In eight games, the 23-year-old budding star has one interception and eight passes defended, tied for 11th in the NFL.
Don’t be surprised if he earns a second straight Pro Bowl nod.

KANSAS CITY CHIEFS | QB Patrick Mahomes: A shaky receiving corps hasn’t fazed Mahomes.
</item 94>

<item 95>
FanDuel 'Prop Stars' Picks: Best bets for Travis Kelce & Taylor Swift-inspired props for Chiefs-Jets Week 4 (Sporting News, 2023-09-28T21:41:00+00:00)
But Wilson doesn't have time to be the anti-hero, or villain, as traditionalists would say. He needs to flip the script and shake it off. The Jets need to let
him sling the ball around the field instead of worrying about the bad turnovers. They aren't winning this game 14-10, and they aren't winning anything if they
can't let their quarterback be who he is. Are we betting this? Absolutely not, but that would be one heck of a plot twist, right? Perhaps the script writers are
working overtime.

Our pick: Pass

Platinum Performance: Travis Kelce to Score 3+ TDs (+2500)

When it comes to Kelce's greatest hits, he prefers to save them for division rivals like the Raiders and Chargers. In your wildest dreams this bet might hit,
and it might happen in reality too.
</item 95>

<item 96>
NFL Best Bets Week 9: Colts clobber Panthers, Dolphins drop Chiefs in Germany (Sporting News, 2023-11-03T20:56:00+00:00)
Oh, and Raheem Mostert, who leads the NFL with 10 TDs and ranks second behind only Breece Hall in yards per carry (5.4), should be licking his chops against a
Kansas City D that has surrendered 135.6 rushing yards per game over the past three weeks. Fins up in Frankfurt!

FINAL SCORE PREDICTION: Dolphins 26, Chiefs 24

Other Week 9 moneylines we like: Vikings (+170) at Falcons; Bengals (-130) vs. Bills; Cowboys (+140) at Eagles

Best NFL over/under bets Week 9

Raiders vs. Giants: UNDER 37.5

It sounds ludicrous to offer an UNDER of 37.5 as a best bet, but this game takes the cake in a Week 9 schedule loaded with underwhelming matchups.
</item 96>

<item 97>
N.F.L. Executives on the Biggest Questions Facing Super Bowl Contenders (The New York Times, 2023-10-16T23:58:53+00:00)
Six of the 17 won anyway. The Dolphins were the only one of those 223 teams to win and cover the spread, which was 13.5 for this game.

• Six catches for 163 yards and a touchdown against Carolina left the Dolphins’ Tyreek Hill with 812 yards through the first six games. That ranks second in NFL
history to Hall of Famer Don Hutson, who had 819 yards through six with Green Bay in 1942.
</item 97>

<item 98>
N.F.L. Executives on the Biggest Questions Facing Super Bowl Contenders (The New York Times, 2023-10-16T23:58:53+00:00)
Especially when you face a defensive coordinator who says, ‘You show me you can beat me throwing the ball outside the numbers — best of luck, and to do that,
you are going to have to take a five-step drop and I’m going to have Myles Garrett chasing you every play.'”

Advertisement

• Kansas City Chiefs: Is the explosive passing game gone for good?

What I think: The Chiefs have never scored less per game on offense through six games or a full season since Patrick Mahomes became their quarterback. They have
also never enjoyed larger average point differentials (9.8 per game), because their defense is so much better. I think the pass offense will become slightly
more explosive as Mahomes gets reps with his receivers, but I’m not expecting a huge shift.

The Chiefs are averaging 9.5 plays per touchdown drive when starting at least 70 yards from the opponent’s goal line, per TruMedia.
</item 98>

<item 99>
Fantasy Football Week 16 Wide Receiver Preview: Waiver adds, DFS plays, and more (CBSSports.com, 2023-12-19T20:51:32+00:00)
The other three are Tyreek Hill, Tyler Lockett, and DeSean Jackson.

-- Deebo Samuel is on pace to become just the fourth player since 1992 to average more than 11 yards per target with at least 60 targets in multiple seasons.
The other three are Tyreek Hill, Tyler Lockett, and DeSean Jackson. 18 -- Puka Nacua needs 18 more catches to set the rookie record for most catches in a
season.

-- Puka Nacua needs 18 more catches to set the rookie record for most catches in a season. 152.67 -- After missing last week's game, Tyreek Hill now needs to
average almost 153 yards per game in his final three to reach his stated goal of 2,000 receiving yards this season.

-- After missing last week's game, Tyreek Hill now needs to average almost 153 yards per game in his final three to reach his stated goal of 2,000 receiving
yards this season.
</item 99>

<item 100>
Fantasy Football WR PPR Rankings Week 14: Who to start, best sleepers at wide receiver (Sporting News, 2023-12-07T22:11:00+00:00)
What are the biggest WR injuries in Week 14?

In Week 13, Demario Douglas (concussion) and Rashid Shaheed (thigh) missed game action, and Tank Dell (leg), Amari Cooper (concussion), Christian Watson
(hamstring), and Marquise Brown (heel) exited early from their respective games.

UPDATE: Christian Kirk suffered a groin injury on Monday night, and he figures to be more doubtful than questionable for this week's game against Cleveland. If
he's out, Zay Jones and Tim Jones will see more targets, but with Trevor Lawrence (ankle) also hurt, it's tough to count on either.

UPDATE: Douglas (concussion) has been ruled OUT for Week 14.

As usual, we'll update these WR PPR rankings throughout the week, so check back for the latest player movement and analysis.
</item 100>
=== END ITEM id=23b448ed4f3d2b27 ===

=== ITEM id=ab3603ca3695b93a ===
Gold sentences and facts:
Sentence 0: For all the other benefits they get from Android, Google could solve this problem today if they wanted to.
  Fact 0.0: Google gets other benefits from Android.
  Fact 0.1: Google could solve this problem today if it wanted to.
Sentence 1: Epic will also suggest that Google hid items from discovery by allowing execs and employees to have “off the record” chats, but Google will counter this as well, saying it has provided the court with thousands of chat logs and millions of documents.
  Fact 1.0: Epic will suggest that Google hid items from discovery by allowing execs and employees to have 'off the record' chats.
  Fact 1.1: Google will counter that suggestion.
  Fact 1.2: Google will say it has provided the court with thousands of chat logs.
  Fact 1.3: Google will say it has provided the court with millions of documents.
Sentence 2: The case, filed by Arkansas-based publisher Helena World Chronicle, argues that Google “siphons off” news publishers’ content, their readers and ad revenue through anticompetitive means.
  Fact 2.0: The case was filed by Arkansas-based publisher Helena World Chronicle.
  Fact 2.1: The case argues that Google 'siphons off' news publishers' content through anticompetitive means.
  Fact 2.2: The case argues that Google 'siphons off' news publishers' readers through anticompetitive means.
  Fact 2.3: The case argues that Google 'siphons off' news publishers' ad revenue through anticompetitive means.

Retrieved items:
<item 1>
Amazon sellers sound off on the FTC's 'long-overdue' antitrust case (Cnbc | World Business News Leader, 2023-10-06T21:31:00+00:00)
Amazon in recent years made the forum anonymous, but users must have a seller account in order to post. Another post included a screenshot of a message Amazon
sent to sellers the day after the FTC filed its complaint, which said, “As your partners, we know that this news may generate questions for you and our business
together. This lawsuit does not change anything about our relationship with you or how we operate today.” One user called it “BS verbiage,” adding, “Businesses
that sell in their store are indeed customers. And which of us has gotten good customer service?” Another user described their experience in the last 12 months
of selling on Amazon as “being up all night at an effing casino but I'm stuck, the drugs are starting to wear off, but I'm trying to break even on the mortgage
payment I'm using to play. That's how it is selling on Amazon right now to me.” The seller went on to describe the experience as a “race to the bottom.”
</item 1>

<item 2>
Power struggle in one of Canada’s wealthiest clans has sister accusing chairman brother of ‘vendetta’ (Fortune, 2023-10-12T22:22:46+00:00)
The Shaw takeover finally closed on April 3 of this year, and the Rogers siblings engaged in talks “intended to resolve all issues between us,” Rogers-Hixon
stated in an affidavit. “I was hopeful all issues could be resolved without negative publicity.” But those negotiations collapsed last month, she said.

It isn’t clear from the court filings whether the discussions included the possibility that any Rogers family members would sell their stake in Rogers
Communications, leave their roles at the company, or exit the family trust that controls almost all of the voting shares of the public company.

“This matter should be resolved privately,” Rogers Communications spokesperson Sarah Schmidt said by email. “We’ve demonstrated that we will not be distracted
by these actions — we have significant momentum in the market, our merger is tracking ahead of plan, and we remain squarely focused on doing what’s right for
our customers and stakeholders.”
</item 2>

<item 3>
Amazon sellers sound off on the FTC's 'long-overdue' antitrust case (Cnbc | World Business News Leader, 2023-10-06T21:31:00+00:00)
“Just like any store owner who wouldn’t want to promote a bad deal to their customers, we don’t highlight or promote offers that are not competitively priced.”
Zapolsky argued the FTC’s lawsuit could force it to stop highlighting low prices, “a perverse result that would be directly opposed to the goals of antitrust
law.”

“Long overdue” lawsuit

On Amazon’s own forum for merchants, called Seller Central, several users cheered on the FTC and said they hoped it would result in changes to the company’s
business practices. Amazon’s tense relationship with merchants has been well-chronicled over the years, with sellers expressing a range of grievances over
issues like rising fees, an arcane suspensions process, and heightened competition on the marketplace from all sides, including the e-commerce giant. “I think
it’s great, Amazon deserves it,” one person commented, adding, “More should be coming on the way.”
</item 3>

<item 4>
Amazon sellers sound off on the FTC's 'long-overdue' antitrust case (Cnbc | World Business News Leader, 2023-10-06T21:31:00+00:00)
Shopify, for example, gives sellers much more control over how they communicate with customers, Needham said, adding that although Amazon still controls the
communication process, at least there is one. "I wish it was a clear-cut case,” Needham said. “I have a vested interest in the marketplace doing really well, as
a seller and as a service provider. And... this case, it doesn’t make the marketplace better for sellers."

Concerns over Amazon pricing policies, fees

Many sellers have zeroed in on Amazon’s pricing policies and rising fees as rightful areas of concern in the FTC's lawsuit. Molson Hart, whose company Viahart
sells toys on Amazon, has been a longtime critic of Amazon’s pricing policies. Hart complained of how Amazon’s seller fees impact pricing in a 2019 Medium post
and later that year testified about his experience before a House committee.
</item 4>

<item 5>
Power struggle in one of Canada’s wealthiest clans has sister accusing chairman brother of ‘vendetta’ (Fortune, 2023-10-12T22:22:46+00:00)
The documents show that the fractures inside the prominent Toronto family have never healed. Rogers Communications is the country’s largest wireless provider
and the owner of a cable television empire, the Toronto Blue Jays baseball team and other media and sports assets. The family’s controlling stake in the public
company is worth close to C$8 billion ($5.9 billion), according to calculations based on company filings for its 2023 annual meeting.

Shares of Rogers extended losses on news of the lawsuit, closing down 2.2% to C$53.07 in Toronto. They’ve dropped 16% this year.

The children of late founder Ted Rogers have frequently fought one another — sometimes driving down Rogers’ stock price as investors grew worried about the
company’s governance.

The bad blood spilled into full view in October 2021, when Edward Rogers publicly battled his sisters and his mother for control of the board and won in court.
The result was that five Rogers Communications directors were fired, replaced by five allies of Edward Rogers.
</item 5>

<item 6>
Amazon sellers sound off on the FTC's 'long-overdue' antitrust case (Cnbc | World Business News Leader, 2023-10-06T21:31:00+00:00)
A worker sorts out parcels in the outbound dock at the Amazon fulfillment center in Eastvale, California, on Aug. 31, 2021. Watchara Phomicinda | MediaNews
Group | The Riverside Press-Enterprise via Getty Images

It was late in the day on Oct. 27, 2021, when Fred Ruckel received the dreaded automated email from Amazon . Amazon’s software had detected that Ruckel’s
popular cat toy, called the Ripple Rug, was being sold somewhere else for a cheaper price. His product would no longer be shown in Amazon’s all-important buy
box, an area of the listing where shoppers click “Add to Cart.” Ruckel is the sole seller of the Ripple Rug on Amazon, so the move all but ensured his product
would disappear from the website, costing him thousands of dollars per day.
</item 6>

<item 7>
Power struggle in one of Canada’s wealthiest clans has sister accusing chairman brother of ‘vendetta’ (Fortune, 2023-10-12T22:22:46+00:00)
The result was that five Rogers Communications directors were fired, replaced by five allies of Edward Rogers. Soon after, then-CEO Joe Natale was sacked and
replaced by Staffieri, the former chief financial officer. Rogers-Hixon and Martha Rogers had opposed the CEO change.

In early 2022, family members agreed to temporarily shelve their disagreements, according to this week’s court filing, to avoid negative publicity as the
company sought regulators’ approval for the takeover of Shaw Communications Inc. The C$20 billion transaction was the company’s biggest deal ever and among the
largest corporate transactions in Canadian history.

Negotiations Collapse

As part of that truce, Rogers-Hixon and Martha Rogers agreed not to participate in board discussions about the Shaw deal or receive materials on it, because of
an argument about Rogers-Hixon’s personal lawyer, whose firm also does work for Telus Corp., a competitor to Rogers and Shaw.
</item 7>

<item 8>
Power struggle in one of Canada’s wealthiest clans has sister accusing chairman brother of ‘vendetta’ (Fortune, 2023-10-12T22:22:46+00:00)
Two members of the billionaire family that controls Rogers Communications Inc. are suing the company, saying they’ve been frozen out of board meetings and
blocked from getting information, reigniting a bitter power struggle within one of Canada’s wealthiest clans.

Melinda Rogers-Hixon says she’s the target of a “personal vendetta” by her older brother, Rogers Chair Edward Rogers, according to a letter contained in the
court filing. The Rogers siblings were close to an agreement that would have resolved their longstanding differences, but Edward Rogers reneged on it, according
to the Sept. 20 missive by Rogers-Hixon, which was sent to Chief Executive Officer Tony Staffieri.

The court petition was filed by Rogers-Hixon and her sister, Martha Rogers, and it argues the company has been “oppressive” to them.

The documents show that the fractures inside the prominent Toronto family have never healed.
</item 8>

<item 9>
Amazon sellers sound off on the FTC's 'long-overdue' antitrust case (Cnbc | World Business News Leader, 2023-10-06T21:31:00+00:00)
Hart said Amazon sales comprise about 90% of his business, meaning any hit those sales take on Amazon has a considerable impact. He recalled “24 anxious hours”
in September 2022 when a third-party seller of his popular construction toy Brain Flakes listed the toy for a lower price on Target than it was offered on
Amazon.

Molson Hart, CEO of Viahart, an educational toy company that sells on Amazon. Courtesy: Molson Hart
</item 9>

<item 10>
Amazon sellers sound off on the FTC's 'long-overdue' antitrust case (Cnbc | World Business News Leader, 2023-10-06T21:31:00+00:00)
Fred Ruckel

Nearly three years later, Ruckel’s experience hits at the core of a sweeping antitrust lawsuit filed last week by the Federal Trade Commission against Amazon.
The agency accused Amazon of wielding its monopoly power to squeeze merchants and thwart rivals. For consumers, that’s led to artificially inflated prices and a
degraded shopping experience, the agency alleges. In the 172-page suit, the FTC said Amazon relies on an “anti-discounting strategy” and a “massive web-crawling
apparatus that constantly tracks online prices” to stifle competition. The agency said Amazon punishes third-party sellers who offer cheaper products elsewhere
by threatening to disqualify them from appearing in the buy box if it detects a lower price. Losing the buy box is an “existential threat” to sellers’
businesses, the complaint alleges. The end result of these tactics, the FTC argues, is elevated prices across the web.
</item 10>
=== END ITEM id=ab3603ca3695b93a ===
