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

=== ITEM id=d2c51c501f597941 ===
Gold sentences and facts:
Sentence 0: Kenneth Walker III (Seahawks) took four carries against the Rams before leaving with an oblique injury.
  Fact 0.0: Kenneth Walker III plays for the Seahawks.
  Fact 0.1: Kenneth Walker III took four carries against the Rams.
  Fact 0.2: Kenneth Walker III left the game with an oblique injury.
Sentence 1: And Klopp has a new worry: the Reds are without Andy Robertson after the left-back suffered a shoulder injury while on international duty with Scotland.
  Fact 1.0: Klopp has a new worry.
  Fact 1.1: The Reds are without Andy Robertson.
  Fact 1.2: Andy Robertson is a left-back.
  Fact 1.3: Andy Robertson suffered a shoulder injury.
  Fact 1.4: Andy Robertson suffered the shoulder injury while on international duty with Scotland.

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
Man United vs Fulham: Red Devils determined to get back to winning ways in PL (Insidesport, 2023-11-03T21:58:48+00:00)
He will probably remain sidelined for weeks.

Other than Casemiro, Tyrell Malacia and Amad Diallo are suffering from knee issues, Lisandro Martinez is nursing a foot injury, Jadon Sancho is out due to
disciplinary issues, and Luke Shaw is yet to recover from a muscle problem. Adding to that, Victor Lindelof is suffering from a bug that could see him miss this
weekend’s match.

On the other hand, Raphael Varane who missed last day’s match due to an illness of his own, could be back in this match. Aaron Wan-Bissaka was also able to
participate in the last match coming on as a substitute. Hence, he should be able to make a comeback. Marcus Rashford, Rasmus Hojlund, and Bruno Fernandes all
could see themselves in the starting lineup.

Fulham Team News

Fulham’s injury situation has not improved so far.
</item 2>

<item 3>
Liverpool vs Everton begins epic Saturday with World Cup semi-final as well as Chelsea vs Arsenal – and it could end with a bang (TalkSport,
2023-10-20T21:34:28+00:00)
talkSPORT will have updates from all the action in Paris as England take on the Springboks from 8:00pm

9 Owen Farrell will be looking to inspire England to victory against the Springboks Credit: Getty

Over in India, England's Cricket World Cup hopes hang by a thread after their latest shock defeat to Afghanistan.

They will be boosted by the news that Ben Stokes is fit for the match after he picked up a hip injury five days before the tournament.

But victory for England could scarcely be more crucial.

Having also lost to New Zealand in their opener, their chances of finishing in the top four and progressing to the semi-finals are looking dubious.

But South Africa, despite winning their first two games, suffered a shock defeat to the Netherlands in their most recent match - a sign for England that the
Proteas can be beaten.
</item 3>

<item 4>
McRae reveals hidden message he carried under collar of his grand final shirt (The Sydney Morning Herald, 2023-09-30T22:45:39+00:00)
Jeremy Howe was collected by Charlie Cameron late in the game Credit: Getty Images

Cameron crashed into Howe as he cleared the ball from Collingwood’s defence, leaving him gasping on the ground with badly bruised ribs. The Magpies defender is
accustomed to extreme pain, having broken his arm in six places after a sickening midair collision in the opening round of the season. This pain was tolerable
by comparison, and he got up and played out the match, helping Collingwood hold on to a four-point win in one of the great grand finals. “The incident late in
the game with Charlie ... He managed to hit me relatively late, it was probably in play, [but I] got a down-field free kick,” Howe told Channel Nine on Sunday.
Jeremy Howe’s rib injury did not prevent him celebrating Collingwood’s premiership. Credit: Getty

“The ribs are incredibly sore but I think if we lost they’d be sorer. “I’ve got six months to get them right.
</item 4>

<item 5>
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
</item 5>

<item 6>
NFL Week 13 injuries: Browns to start Joe Flacco with Dorian Thompson-Robinson out, Saints' Olave questionable (CBSSports.com, 2023-12-01T22:52:36+00:00)
Week 13 is in full swing and teams are starting to truly jockey for playoff position as we rocket down the stretch run of the regular season. Of course, with it
being so late in the year, injuries continue to be a predominant storyline to monitor heading into each slate and this weekend is no different. Specifically,
the quarterback position in Cleveland has taken an interesting turn with veteran Joe Flacco named the starter against the Rams with Dorian Thompson-Robinson
ruled out with a concussion. And that's just one situation that we'll be monitoring below.

Here's a look at all of the final injury reports for Week 13.

All NFL odds via SportsLine Consensus odds.

Chargers (-5.5) at Patriots

Allen made his return to practice on a limited basis on Friday and is officially questionable for Sunday's matchup in New England due to a quad injury.
Meanwhile, Pipkins was also limited on Friday.
</item 6>

<item 7>
2023 NFL Week 11 injury roundup: Rams WR Cooper Kupp, Seahawks RB Kenneth Walker III among stars to exit (CBSSports.com, 2023-11-19T23:51:49+00:00)
Dillon. Green Bay ruled him questionable to return, then downgraded him to out, but coach Matt LaFleur said after the Packers' win over Los Angeles that he
doesn't expect the injury to be a long-term issue.

De'Von Achane (Dolphins) returned from injured reserve -- and a four-game absence -- but limped to the sidelines after his second touch of the first quarter
against the Raiders. Miami ruled him questionable to return with a knee injury, and while he later returned to the sidelines from a locker-room visit, he was
replaced on the field indefinitely by Raheem Mostert. Head coach Mike McDaniel told reporters on Monday that Achane is day-to-day. While he hasn't been ruled
out of Friday's matchup with the Jets, he's not been given the green light either, so his status is worth monitoring.

Kenneth Walker III (Seahawks) took four carries against the Rams before leaving with an oblique injury.
</item 7>

<item 8>
Inter Miami vs FC Cincinnati live score, result, updates, highlights with Lionel Messi's MLS playoffs on line (Sporting News, 2023-10-07T21:30:00+00:00)
FC Cincinnati are otherwise light on availability concerns, with only a leg injury for Arquimides Ordonez and a long-term injury to young midfielder Stiven
Jimenez to worry about.

Talisman forward Luciano Acosta was partially rested midweek, coming off the bench to score the team's only goal against the Red Bulls in 32 minutes of play, so
he is ready to go as he chases the league's Golden Boot award. Also rested midweek was Alvaro Barreal who leads the team in minutes and also logged just 32 of
them on Wednesday.

FC Cincinnati starting lineup (5-3-2): Celentano (GK) — Arias, Hagglund, Miazga, Murphy, Barreal — Nwobodo, Acosta, Moreno — Vazquez, Santos.

FC Cincinnati subs (9): Kann (GK), Kubo, Mosquera, Angulo, Badji, Pinto, Halsey, Valenzuela, Boupendza.
</item 8>

<item 9>
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
</item 9>

<item 10>
Fantasy Football Week 13 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-11-30T22:10:58+00:00)
In their past three games, they gave up 4.1 yards per rush to the Ravens (just 6.3% of their runs went 10-plus yards), 7.8 yards per rush to the Steelers thanks
to a long TD run by Jaylen Warren (23.8% of their runs went 10-plus yards) and 4.6 yards per rush to the Broncos (18.5% of their runs went 10-plus yards).
Playing more snaps because the Browns offense has been inept isn't the reason for it either because they struggled with the run in the first half against all
three teams (over 5.3 yards per carry allowed against each). It's silly to say the Rams passing game is struggling when Matthew Stafford just threw four
touchdowns, but Cooper Kupp looked like he played hurt last week and Puka Nacua didn't look that much better.
</item 10>

<item 11>
Fantasy Football RB Rankings Week 12: Who to start, best sleepers at running back (Sporting News, 2023-11-21T21:33:00+00:00)
Keep rolling with Alexander Mattison (vs. Bears), too, against another below-average run defense.

Welcome to the top 25, Zach Charbonnet (vs. 49ers)! With Kenneth Walker III's oblique injury comes opportunity for the rookie, who already looked strong in the
month leading up to Seattle's narrow Week 11 loss to the Rams. Sure, Charbs' 69-yard effort in Los Angeles last week was a bit underwhelming, but we love his
upside with a full game as the Seahawks' bell-cow.

Speaking of bell-cow backs, things are looking very promising for the short-term workload of Green Bay's AJ Dillon (@ Lions). Veteran Aaron Jones (knee) and
rookie Emanuel Wilson (shoulder) both got hurt in Week 11, so Dillon should be the main man in Motown for the Pack.
</item 11>

<item 12>
NFL Fantasy Football Week 6 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-12T21:23:16+00:00)
McDANIEL: Prefers to use two running backs from game to game, and in Miami's past three games there's been a relatively close split in carries between two backs
(18 and 13 in Week 3, 8 and 7 in Week 4, 11 and 10 in Week 5).

LAST YEAR: Wilson was efficient in his eight games in Miami, averaging 4.7 yards per carry with three touchdowns and just over 60 total yards per matchup. That
includes two games he barely played in (under 20 snaps) -- take those out and he averaged 78.2 total yards per game.

SALVON AHMED? Was a healthy scratch last week and figures to be no better than a special-teams player this week. In fact, in the seven games Raheem Mostert,
Wilson and Ahmed were all active for with the Dolphins last year, Ahmed played 18 offensive snaps and 25 special-teams snaps.
</item 12>

<item 13>
Fantasy Injury Updates Week 10: News on Justin Jefferson, Ja'Marr Chase, T.J. Hockenson & Curtis Samuel (Sporting News, 2023-11-11T21:17:00+00:00)
If he's out, Boyd is a must-start, and you can try one of Cincinnati's secondary receivers in deep leagues.

WEEK 10 STANDARD RANKINGS:

QBs | RBs | WRs | TEs | D/STs | Kickers

Is T.J. Hockenson playing Week 10?

UPDATE: NFL Network's Ian Rapoport reported Sunday morning that Hockenson is expected to play against New Orleans.

UPDATE 2: Hockenson is officially ACTIVE.

Hockenson (ribs) is listed as "questionable" for Week 10, but he's already said he plans to play on Sunday. The tight end has dealt with his fair share of
injury woes this season, however, he's yet to miss a game.
</item 13>

<item 14>
USA squad for October friendlies 2023: USMNT roster for friendly matches vs Germany, Ghana (Sporting News, 2023-10-04T22:42:00+00:00)
He was initially injured while playing for Leeds United back in March, and while he had finally recovered to make his Bournemouth debut in the EFL Cup in late
September, he played just 20 minutes off the bench before suffering a setback, sidelined again for the foreseeable future.

Gio Reyna has returned from his leg fracture and is called in for the first time under Berhalter since their falling out at the 2022 World Cup. Also included is
defender Cameron Carter-Vickers who only just made his debut for Celtic this season thanks to a hamstring injury. On the heels of that return just before the
international break, he is called in.

Goalkeepers Zack Steffen and Sean Johnson remain out of action due to injuries. Additionally, Josh Sargent is sidelined with a long-term ankle injury, while
Daryl Dike is also out after tearing his Achilles.
</item 14>

<item 15>
2023 NFL Week 11 injury roundup: Rams WR Cooper Kupp, Seahawks RB Kenneth Walker III among stars to exit (CBSSports.com, 2023-11-19T23:51:49+00:00)
The former All-Pro, who battled a lingering hamstring injury earlier this year, received attention on the sidelines after limping off, then left the game for
further evaluation in the locker room. He was ruled questionable to return. NFL Media reported on Monday that Kupp suffered a low ankle sprain and will be
evaluated going forward.

Darius Slayton (Giants) had a big impact against the Commanders, hauling in a deep score from Tommy DeVito, before exiting with an arm injury. He was ruled out
for the remainder of the contest halfway through the third quarter.

EDGE

Joey Bosa (Chargers) appeared to injure his right leg or foot while held in a block during a first-quarter pass-rushing attempt against the Packers. He limped
to the sidelines and was carted to the locker room, then ruled questionable to return.
</item 15>

<item 16>
VAR apology list in Premier League: Updated list of PGMOL apologies for referee mistakes in English football (Sporting News, 2023-09-30T21:27:02+00:00)
It was one of many decisions the Seagulls were not happy about.

Mitoma penalty... ❓

Dunk penalty... ❓

Mitoma handball... ❓

Mac Allister handball... ❓



Dermot Gallagher dissects Spurs vs Brighton 🔍 pic.twitter.com/hw86oA4frs — Sky Sports Premier League (@SkySportsPL) April 10, 2023

Arsenal 1-1 Brentford: February 11, 2023

Arsenal were denied two points in their race for the title after VAR's Lee Mason did not use the right player when checking for an offside in Brentford's goal.

Mikel Arteta was not happy after the game. He said: "That wasn't a human error. That was a big, big, big not conceiving and understanding your job."

Crystal Palace 1-1 Brighton: February 11, 2023

Incredibly, the referees made another wrong offside decision on the same day.
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
Canada vs Jamaica live score, updates, highlights, result from CONCACAF Nations League quarterfinal second leg (Sporting News, 2023-11-21T22:35:30+00:00)
Antonio was replaced by Salernitana striker Trivante Stewart, but he struggled and was himself withdrawn in the 81st minute. Demarai Gray could step in, but he
himself is struggling for fitness, and while he remains with the team he was not part of the first leg matchday squad. Another option is Colorado Switchbacks
forward Romario Williams, who scored 15 goals in 29 USL Championship matches this season.

Other injuries for Jamaica to contend with include Cory Burke and Amari'i Bell, who both miss out on this international window.

Jamaica projected starting lineup (4-3-3): Blake (GK) — Leigh, Bernard, Pinnock, Lembikisa — Latibeaudiere, D. Johnson, Bailey — Reid, Nicholson, R. Williams.
</item 18>

<item 19>
NFL Week 13 injuries: Browns to start Joe Flacco with Dorian Thompson-Robinson out, Saints' Olave questionable (CBSSports.com, 2023-12-01T22:52:36+00:00)
Cornerback Cam Taylor-Britt hurt his foot in practice and is out.

For the Jaguars, Travis Etienne (ribs) is also expected to play, Doug Pederson told reporters, despite the running back being limited in practice all week.
Strange was a late addition to the injury report, not practicing Saturday.
</item 19>

<item 20>
Salah seals emphatic Europa League win for Liverpool against Toulouse (The Guardian, 2023-10-26T21:00:42+00:00)
“I think I said it before – it is really easy to fall in love with this team, there is so much excitement,” said the Liverpool manager after a seventh win in
seven home matches this season. “We have to make massive steps, we have to grow, but a lot of signs are really positive and how the team interacts with each
other is really nice.”

Rivalry was forgotten and friendship remembered as Liverpool paid a warm tribute before kick-off to Bill Kenwright, the Everton chairman who died on Monday. The
Hollies’ He Ain’t Heavy, He’s My Brother – the song Kenwright chose for a Hillsborough tribute at Goodison Park and became a fund-raising single for the justice
campaign – was played to mass applause from the Liverpool fans inside Anfield. Kenwright’s powerful speech at the Hillsborough memorial service in 2013 and
lasting support for the bereaved families will not be forgotten across Stanley Park.
</item 20>

<item 21>
City old boy has final say for Chelsea in eight-goal thriller, supreme Salah leads Liverpool surge (The Roar | Sports Writers Blog, 2023-11-12T20:48:50+00:00)
Stream LIVE, or on-demand after, here ???? https://t.co/tSmadJ11sr#OptusSport #PL pic.twitter.com/gT55yj7loA — Optus Sport (@OptusSport) November 12, 2023

Chelsea’s third goal eventually came from Nicolas Jackson in the 67th, the striker firing in the rebound after Gallagher’s long shot was parried.

Advertisement

Rodri’s powerful effort from the edge of the box took a big deflection off Thiago Silva to send Chelsea goalkeeper Robert Sanchez the wrong way as the ball flew
into the opposite corner, and City looked home and dry until Palmer’s late intervention.

Wow! It's a massive deflection from a Rodri hit and Manchester City have four! ????



Manchester City won't care. Limbs in the away end at Stamford Bridge.
</item 21>

<item 22>
Fantasy Injury Updates Week 10: News on Justin Jefferson, Ja'Marr Chase, T.J. Hockenson & Curtis Samuel (Sporting News, 2023-11-11T21:17:00+00:00)
Jefferson (hamstring) is officially listed as "questionable" for Sunday's game against the Saints. Kickoff is slated for 1 p.m. ET, but fantasy owners won't
have to wait long for news on the star receiver. In order to be eligible to play, Jefferson will need to be activated by 4 p.m. ET on Saturday. Earlier in the
week head coach Kevin O'Connell said that it would be "a little aggressive" for Jefferson to get back on the field this week. However, the receiver's three
straight limited practices were enough for the team to at least weigh the decision, but thankfully, an early answer awaits.

WEEK 10 FANTASY ADVICE:

Sleepers | Busts | Projections | Start 'em, sit 'em

Jefferson's fantasy outlook is always through the roof, but it will be interesting to see if that changes with Josh Dobbs under center instead of Kirk Cousins.
</item 22>

<item 23>
Another cruel comeback sees Newcastle perish in group of death (The Independent - Sports, 2023-12-13T22:16:07+00:00)
Sign up to Miguel Delaney’s Reading the Game newsletter sent straight to your inbox for free Sign up to Miguel’s Delaney’s free weekly newsletter Please enter a
valid email address Please enter a valid email address SIGN UP I would like to be emailed about offers, events and updates from The Independent. Read our
privacy notice Thanks for signing up to the

Football email {{ #verifyErrors }} {{ message }} {{ /verifyErrors }} {{ ^verifyErrors }} Something went wrong. Please try again later {{ /verifyErrors }}

It was the sequel Newcastle United did not want, the second cruel comeback in as many Champions League games. For some 45 minutes, including half-time, they
could savour the sight of a table that showed them in second, permitting them to imagine a date with Real Madrid or Bayern Munich in the knockout stages. And
then, little more than half an hour, they were ejected from Europe altogether.

Christian Pulisic’s equaliser threatened to condemn them to the Europa League.
</item 23>

<item 24>
Fantasy Football Week 16 Wide Receiver Preview: Waiver adds, DFS plays, and more (CBSSports.com, 2023-12-19T20:51:32+00:00)
The Texans offense looks like one to avoid as long as C.J. Stroud is out. Ja'Marr Chase WR CIN Cincinnati • #1

Age: 23 • Experience: 3 yrs. Tee Higgins is a No. 2 receiver and Tyler Boyd is a fine flex.

WR Preview Numbers to Know

69.9% -- Nearly 70% of Rashee Rice's receiving yards have come after the catch, that's tops amongst wide receivers with at least 50 targets.

-- Nearly 70% of Rashee Rice's receiving yards have come after the catch, that's tops amongst wide receivers with at least 50 targets. 11.4 -- Deebo Samuel is
on pace to become just the fourth player since 1992 to average more than 11 yards per target with at least 60 targets in multiple seasons. The other three are
Tyreek Hill, Tyler Lockett, and DeSean Jackson.
</item 24>

<item 25>
USA squad for October friendlies 2023: USMNT roster for friendly matches vs Germany, Ghana (Sporting News, 2023-10-04T22:42:00+00:00)
Also missing is Antonee Robinson, who Gregg Berhalter said is dealing with a sports hernia, and while he played through it last camp and recently for his club,
it's affecting him and therefore rest is the best option here.

Johnny Cardoso had to pull out of the September camp due to an ankle injury, but he has returned to action with Brazilian club Internacional, indicating a
return to fitness, and has been included on the squad.

With Adams injured, there has been some noise on social media calling for Heidenheim midfielder Lennard Maloney to be brought into the squad in October, and he
was given the call-up here. The 23-year-old was born in Berlin to an American father, appearing for the U.S. at the U-20 level in the past and stating his
desire to reach the USMNT senior squad.

MORE: All players in U.S.
</item 25>

<item 26>
NFL Fantasy Football Week 6 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-12T21:23:16+00:00)
In his NFL career, Walker has thrown specifically to wide receivers on 71.2% of his throws, which basically the only good thing you can take away from this
paragraph.

COOPER: In effectively 30 career games without Dak Prescott, Derek Carr or Deshaun Watson on the field (meaning he played with a backup), Cooper has 13
touchdowns (seven from Brissett) and seven with 100-plus yards (four with Brissett). More importantly, Cooper had at least 15 PPR points in 13 of the 30 games
(six with Brissett), so 43% of the time.

BRISSETT VS WALKER: Comparing their play over the past four seasons, Brissett had a considerably higher completion rate (63% to 57.5%) with a lower interception
rate (1.7% to 4.8%) and lower off-target rate (10.3% to 18.9%).
</item 26>

<item 27>
NFL Fantasy Football Week 6 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-12T21:23:16+00:00)
D.J. Turner is expected to replace Awuzie.

GENO SMITH: Has a 47.8% completion rate and a lowly 4.1 yards per attempt average against man coverage so far this season. He's also barely thrown deep (4.3% of
his attempts went 20-plus Air Yards). A huge reason for the low numbers is because his pass protection has been dreadful. The anticipated return of left tackle
Charles Cross should help him have more time, which in turn helps his receivers.

OUTSIDE WRs: Have caught just 50% of their targets against the Bengals this season, third-lowest in football. However, when those receivers have caught the ball
they've averaged 17.9 yards per catch, which is the third-most in the league. The Bengals defense has really only let up big games to receivers who catch
touchdowns.
</item 27>

<item 28>
Liverpool vs Everton begins epic Saturday with World Cup semi-final as well as Chelsea vs Arsenal – and it could end with a bang (TalkSport,
2023-10-20T21:34:28+00:00)
They sit tenth in the table having won four and lost as many from their opening eight games.

But their most recent Premier League victory over Brentford may give the Red Devils hope their season is on the up.

Sheffield United vs Manchester United is live on talkSPORT - kick off is at 8:00pm

9 United got a dramatic win over Brentford before the international break Credit: Getty

An England vs South Africa double header

If you've had your football fix for the day, there's also a packed Saturday of rugby and cricket.

It's crunch time for England in the World Cups as they face South Africa in the semi-final of the Rugby World Cup and in the ODI Cricket World Cup.

After England secured victory over Fiji in the quarter-finals, they now look to overcome defending champions South Africa to advance into their second
consecutive Rugby World Cup final.
</item 28>

<item 29>
Fantasy Football Week 13 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-11-30T22:10:58+00:00)
0-7.00 6.99 - 5.01 5.00 - 0 Sit 'em

When Thomas Brown took over the Panthers playcalling for three games earlier this year, the Panthers averaged a 64% pass rate, Chuba Hubbard dominated touches
(two games with 17-plus touches, none with more than 10 PPR points) and Adam Thielen had a large target share (26.5%) with a tiny ADOT (4.33). In the two games
since, the Panthers pass rate dipped below 60%, Hubbard shared more regularly and Thielen's ADOT nearly doubled. Hubbard had a good game last week, Thielen
hasn't seen monster numbers since before the Week 7 bye.
</item 29>

<item 30>
Breaking down each VAR intervention from wild first half of Tottenham vs Chelsea officiated by Michael Oliver (Sporting News, 2023-11-06T22:06:19+00:00)
Chelsea have a THIRD goal disallowed as Sterling is offside 🤯 pic.twitter.com/brMes6qQQ7 — Mail Sport (@MailSport) November 6, 2023

Reece James avoids red card despite elbow to Destiny Udogie

While two Tottenham defenders escaped an early shower earlier in the match for incidents that could have potentially seen them off, Chelsea also had a player
avoid further punishment just before halftime.

Reece James caught Destiny Udogie in the face with a trailing elbow while challenging for a bouncing ball in the air. Referees have been strict with incidents
such as these, and it was surprising to see James get away with just a yellow card.

The incident was indeed reviewed, but the VAR official did not ultimately advise a check for Michael Oliver.
</item 30>

<item 31>
European roundup: Bellingham helps Real Madrid go top while Bayern draw (The Guardian, 2023-09-30T21:17:47+00:00)
If you already have the Guardian app, make sure you’re on the most recent version.

In the Guardian app, tap the Menu button at the bottom right, then go to Settings (the gear icon), then Notifications.

Turn on sport notifications. Was this helpful? Thank you for your feedback.

“After the break, we were more compact and certainly more energetic, hungrier and more determined,” Milan’s coach, Stefano Pioli, told DAZN. “That is a great
sign when playing every three days and it was a fully deserved victory.”

Milan suffered a potential injury blow as the midfielder Ruben Loftus-Cheek limped off in the first half. “It would be a shame if we lost Ruben, he felt a
twinge near the groin area,” Pioli said. “Unfortunately, injuries tend to come when players are in good shape.”
</item 31>

<item 32>
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
</item 32>

<item 33>
Liverpool enjoy home comforts but Dyche’s Everton offer derby danger | Andy Hunter (The Guardian, 2023-10-20T21:30:28+00:00)
Klopp has lost one of his 17 encounters with Everton, Liverpool one of the past 23 Premier League meetings at Anfield and one of the past 25 league derbies home
or away.

Liverpool’s defensive concerns, heightened by the loss of Andy Robertson for up to three months with a shoulder injury, are offset by the visitors’ troubles in
front of goal. A 3-0 defeat of Bournemouth last time out was only the eighth time Everton have scored more than once in a Premier League match in the past 12
months. The overall improvement in Everton’s performance, however, coupled with three wins in the past four games, raises Sean Dyche hopes of capitalising on
any post-international fatigue in the opposition ranks. Klopp’s compliments towards Dyche and his local rivals were of the back-handed nature.

“Everton are in a good moment,” he said.
</item 33>

<item 34>
Chicago Fire vs Inter Miami live score, result, updates, highlights from Lionel Messi in MLS match (Sporting News, 2023-10-04T23:33:00+00:00)
🤔 pic.twitter.com/Svq79fvbBG — Major League Soccer (@MLS) October 4, 2023

1 hour to kickoff: With lineups not yet released, there is finally a report regarding Lionel Messi's availability. According to Michelle Kaufman of the Miami
Herald, Messi "did not arrive with the team" to Soldier Field. While that's not outright confirmation that he will not take part in the match, it's a very
strong indication that he will not be involved tonight.

1 hour 15 mins to kick: With lineups due imminently, and nothing to indicate if Lionel Messi is involved tonight or not, the Chicago Fire media team has jokes.
</item 34>

<item 35>
Canada vs Jamaica live score, updates, highlights, result from CONCACAF Nations League quarterfinal second leg (Sporting News, 2023-11-21T22:35:30+00:00)
That would allow them to rotate Kamal Miller out, who picked up a yellow card in the first leg, keeping him safe for the semifinals.

In would step Alistair Johnston and Derek Cornelius, with Steven Vitoria retained alongside them.

The only injury absence for Canada at the moment is striker Lucas Cavallini, who withdrew from the squad prior to the two matches, replaced by Jacen
Russell-Rowe.

Canada projected starting lineup (3-5-2): Borjan (GK) — Cornelius, Vitoria, Johnston — Laryea, Eustaquio, Kone, Kaye, Davies — David, Larin.

Jamaica saw star striker Michail Antonio go down with a gutting ACL injury in the first leg, and he now begins a lengthy rehab process. West Ham are reportedly
furious about how it was handled, angered by both the conditions of the pitch in Jamaica and the fact he played another 10 minutes after sustaining the injury.
</item 35>

<item 36>
Theatre producer and Everton chairman Kenwright dies (BBC News - Entertainment & Arts, 2023-10-24T16:32:09+00:00)
Chief executive Denise Barrett-Baxendale, chief finance and strategy officer Grant Ingles and non-executive director Graeme Sharp have all since left their
boardroom roles, but Kenwright remained in his post as chairman until his death.

In September, owner Moshiri agreed to sell his 94% stake to American investment fund 777 Partners.

The club said Kenwright had "worked hard" alongside Moshiri "right up until the day" of his liver operation to help facilitate the proposed takeover.

Moshiri said he was "deeply saddened" by the death of his "great friend".

"Bill was a force of nature and he certainly changed my life nearly 10 years ago when he first spoke to me about getting involved with the club he adored," said
Moshiri.

"He told me about this incredible club, a club that not only has history and heritage but was also a beating heart of our community and for that I will always
be grateful.

"He was a special soul, a man successful in so many different walks of life.
</item 36>

<item 37>
2023 NFL Week 11 injury roundup: Rams WR Cooper Kupp, Seahawks RB Kenneth Walker III among stars to exit (CBSSports.com, 2023-11-19T23:51:49+00:00)
Smith did return, however, for the final drive of the game in the losing effort. Head coach Pete Carroll said that Smith will "make it back." When asked if that
means for Thursday's matchup against San Francisco, the coach said, "I would think so."

Sam Howell (Commanders) received medical attention on the sidelines after enduring several hard hits to the upper body on a touchdown run against the Giants. He
was quickly cleared of a potential concussion, however, and remained in the game. Veteran journeyman Jacoby Brissett is Washington's backup.

RB

Aaron Jones (Packers) suited up against the Chargers despite a lingering hamstring issue, only to manage just five carries before suffering a knee injury. The
former Pro Bowler needed help to leave the field and was then carted to the locker room, replaced by A.J. Dillon.
</item 37>

<item 38>
Salah seals emphatic Europa League win for Liverpool against Toulouse (The Guardian, 2023-10-26T21:00:42+00:00)
Jürgen Klopp’s prediction that Liverpool would not “fly” through the Europa League group stage amounts to their only mis-step so far. Qualification for the
knockout phase is already close at hand with the procession continuing against Toulouse.

Klopp could ring the changes before Nottingham Forest’s Premier League visit on Sunday, give a full debut to 19-year-old left-back Luke Chambers, first
Liverpool outings to Calum Scanlon and James McConnell, and still enjoy a fluent, dominant display. It was polished by goals from Diogo Jota, Wataru Endo,
Darwin Núñez, the highly impressive Ryan Gravenberch and Mohamed Salah. Salah’s strike with the final kick of the game was his 43rd European goal for Liverpool
and took him above Thierry Henry as the most prolific goalscorer for an English club in European competition.
</item 38>

<item 39>
Theatre producer and Everton chairman Kenwright dies (BBC News - Entertainment & Arts, 2023-10-24T16:32:09+00:00)
Leading Everton for almost two decades

Kenwright spent 19 seasons as Everton chairman, overseeing 12 finishes inside the Premier League top eight, including fourth place in 2005, while the Merseyside
club were also FA Cup finalists in 2009.

Everton said he had led the club through "a period of unprecedented change in English football".

In February 2016, Kenwright sold a 49.9% stake in the club to Iranian businessman Farhad Moshiri, who increased his shares to 94% in January 2022.

But with the club fighting against relegation last season, an Everton fans' group called for Kenwright to be sacked and said it had "no confidence" in him as
chairman.

Kenwright and the rest of the board of directors were unable to attend home games last season from January after what the club described as "threatening
correspondence" was received before a game against Southampton.
</item 39>

<item 40>
Fantasy Football RB Rankings Week 14: Who to start, best sleepers at running back (Sporting News, 2023-12-07T22:07:00+00:00)
We'll talk more about Derrick Henry (concussion) in the injury section, but we can say with confidence here and now that Tyjae Spears (@ Dolphins) should be the
hottest waiver-wire pickup of Week 14. Tennessee is a run-first offense and King Henry might be sidelined with a head injury, so Spears could very well pick up
where he left off against Indy last weekend. The young back averaged 4.7 yards per carry (just shy of his 5.1 YPC entering the week) en route to 88 yards.

Another backup getting a boost this week is the Patriots' Ezekiel Elliott (@ Steelers), who will assume lead-back duties after Rhamondre Stevenson went down
with a pretty bad ankle injury in Week 13. Zeke might put in a vintage performance considering James Conner just went back to Pittsburgh and hung 105 yards and
two TDs on his old squad.

Could this be the weekend that Keaton Mitchell (vs.
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
Breaking down each VAR intervention from wild first half of Tottenham vs Chelsea officiated by Michael Oliver (Sporting News, 2023-11-06T22:06:19+00:00)
Chelsea were given a penalty and Tottenham went down to 10 men. #TOTCHE pic.twitter.com/sOzTrxgBzm — NBC Sports Soccer (@NBCSportsSoccer) November 6, 2023

Watch in UK:

"AMID FRENZY AND FURY!" 🤯



Cole Palmer levels for Chelsea as Cristian Romero sees red! 🟥 pic.twitter.com/1oULwk1uOP — Sky Sports Premier League (@SkySportsPL) November 6, 2023

Nicolas Jackson goal disallowed for offside

Chelsea would see a third goal of the half disallowed as Nicolas Jackson put the ball in the net on 37 minutes but Raheem Sterling was offside in the buildup
just past the midfield stripe. There's very little argument to be had here, and the game played on.

While it was a decision that featured little, if any, controversy, it added to the mayhem of the half and contributed to the game's overall narrative.
</item 42>

<item 43>
Fantasy Football RB Rankings Week 12: Who to start, best sleepers at running back (Sporting News, 2023-11-21T21:33:00+00:00)
However, even a moderate MCL sprain could mean a multi-week absence for the injury-prone veteran back, especially in a lost season for the 4-6 Pack. Rookie
Emanuel Wilson also hurt his shoulder in Week 11, so AJ Dillon could be getting the keys to Green Bay's backfield castle (although it's more of an oversized
shack, at this point).

Dolphins rookie De'Von Achane, who looked like the runaway Offensive Rookie of the Year earlier this season, looked to have aggravated his MCL injury upon his
Week 11 return against the Raiders. Achane has a chance to return for Miami's Week 12 tilt with the Jets on Friday, but we wouldn't be surprised if the 7-3
'Fins play it safe and keep him out until he's back to 100 percent. Meanwhile, Miami placed backup Salvon Ahmed on IR with a foot injury, so Jeff Wilson Jr.
could be back in play as a handcuff stash.
</item 43>

<item 44>
Liverpool vs Everton begins epic Saturday with World Cup semi-final as well as Chelsea vs Arsenal – and it could end with a bang (TalkSport,
2023-10-20T21:34:28+00:00)
Liverpool vs Everton begins epic Saturday with World Cup semi-final as well as Chelsea vs Arsenal – and it could end with a bang

As the Premier League returns with a Saturday packed full of huge matches, talkSPORT gets the action started at Anfield.

But that's not all there is, with a Rugby World Cup semi-final, a London derby and a packed UFC card among the other offerings.

9 Merseyside derbies are never dull and talkSPORT has the first one of the season live Credit: Getty

9 Everton are starting to look much better after a rocky start Credit: Getty

Here, talkSPORT runs through what is an epic day of sport.

Merseyside derby

Liverpool are sitting in a comfortable fourth place in the Premier League table having picked up five wins out of their opening eight games.

The bad news for Jurgen Klopp is that the Reds lost to Tottenham and only managed a draw with Brighton just before the international break.
</item 44>

<item 45>
Manchester divided as ten Hag pressure mounts after United Hammered, Ange-ball pays off again as Spurs sink Everton (The Roar | Sports Writers Blog,
2023-12-23T21:19:19+00:00)
Stream LIVE, or on-demand after, here ???? https://t.co/wdM7tdgfb9#OptusSport pic.twitter.com/n7dqm1qnUH — Optus Sport (@OptusSport) December 23, 2023

Advertisement

Arsenal led through defender Gabriel’s fourth-minute header from Martin Odegaard’s free kick on Saturday (local time), but Mohamed Salah beat David Raya at his
near post to level in the 29th.

Liverpool came closest in the second half and hit the woodwork twice, through Trent Alexander-Arnold and substitute Harvey Elliott.

Mikel Arteta’s team go into Christmas a point clear of Liverpool and third-placed Aston Villa.

Liverpool defender Kostas Tsimikas broke his collarbone when he collided with Jurgen Klopp on the sideline.

Tsimikas was challenged by Arsenal forward Bukayo Saka and stumbled into Klopp, who fell on top of him.
</item 45>

<item 46>
Fantasy Football RB Rankings Week 14: Who to start, best sleepers at running back (Sporting News, 2023-12-07T22:07:00+00:00)
We also don't love Jerome Ford or Kareem Hunt (vs. Jaguars), as Jacksonville entered Week 12 allowing just 12.3 standard fantasy points per game to RBs (third
lowest in the NFL).

We say it almost every week, but the song remains the same: don't start Carolina backs Chuba Hubbard or Miles Sanders (@ Saints). It's even worse than usual
this week, as New Orleans has a pretty hard-hitting front-seven and the Bayou is no place for below-average offenses.

What are the biggest RB injuries in Week 14?

Two-time rushing champ Derrick Henry (head) suffered a huge blow to the head and neck area in Tennessee's Week 13 loss to the Colts, so don't be surprised if
the king gives up his throne to Tyjae Spears this week. He immediately left the game and proceeded to the locker room following the hit, and it was confirmed
that he would not return with a head injury.
</item 46>

<item 47>
NFL Week 13 injuries: Browns to start Joe Flacco with Dorian Thompson-Robinson out, Saints' Olave questionable (CBSSports.com, 2023-12-01T22:52:36+00:00)
Chiefs (-6) at Packers

Kansas City is relatively healthy coming into Week 13. McKinnon was a limited participant on both Thursday and Friday, so his prospects of playing on Sunday
appear to be trending in a positive direction. As for Bolton, he is still on IR and working his way back from a wrist injury.

Green Bay will be without Jones as the running back continues to deal with a knee injury. Reed returned to practice Friday after missing both Wednesday and
Thursday due to a chest injury, while Alexander -- who has missed six of the Packers 11 games this year -- was limited throughout the week.
</item 47>

<item 48>
NFL Fantasy Football Week 6 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-12T21:23:16+00:00)
Weekly Breakdown MATCHUP vs CAR MIA -13.5 O/U 48.5 OPP VS RB 31st PROJ PTS 7 RB RNK NR YTD Stats RUYDS 0 REC 0 REYDS 0 TD 0 FPTS/G 0 Once it's clear that Wilson
will play, the matchup combined with the Dolphins' approach to utilize multiple backs should put Wilson in position to have some good opportunities. Remember,
Wilson's injury was to his abdomen, not his legs, so there might not be much of a limit on how much work he'll do (though it's unlikely he'll have a ton of
touches). I'd take my chances with him over Samaje Perine, Jerick McKinnon and Antonio Gibson.

WILSON: Started practicing this week after being sidelined with finger and abdomen injuries since late August.
</item 48>

<item 49>
Theatre producer and Everton chairman Kenwright dies (BBC News - Entertainment & Arts, 2023-10-24T16:32:09+00:00)
Among his current shows, Sir Ian McKellen is starring in Frank and Percy in London, while Twelve Angry Men, Calendar Girls The Musical, Heathers The Musical and
Blood Brothers are all on tour.

Dame Judi Dench, Woody Harrelson, Billie Piper, Rob Lowe and Felicity Kendal have also starred in his productions.

Rooney leads tributes to a 'big inspiration'

Wayne Rooney, former Everton and England forward: "Devastated to hear the sad news about Bill Kenwright. Known Bill since I was young and he's had a huge impact
on me as a person and my career. Great man and a big inspiration. Thoughts are with all Bill's family and friends."

Jamie Carragher, ex-Liverpool and England defender: "Really sad news this. A huge Evertonian who served and loved his club to bits. I'll never forget his and
Everton's support every year around the Hillsborough memorial. RIP Bill."
</item 49>

<item 50>
NFL Fantasy Football Week 6 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-12T21:23:16+00:00)
Brissett didn't throw downfield as much as Walker, contributing to a higher ADOT for Walker, but not by much (8.8 to 8.3). Both quarterbacks were right around
6.5 yards per pass attempt.

49ERS: Oh yeah, the matchup isn't great. No team allows fewer yards per catch to wideouts than the Niners (9.7), and they're thrown at a ton by offenses that
have to play from behind. Only three receivers -- Puka Nacua (20 targets), Marquise Brown (10 targets) and Michael Wilson (two touchdowns) have had more than 15
PPR points against them. And only one other receiver -- Tutu Atwell (nine target) -- had more than 12 PPR points.
</item 50>

<item 51>
NFL Week 13 injuries: Browns to start Joe Flacco with Dorian Thompson-Robinson out, Saints' Olave questionable (CBSSports.com, 2023-12-01T22:52:36+00:00)
Washington could be without its starting center this weekend as it plays host to Miami. Larsen, who missed Wednesday's practice due to a knee injury, was
limited on Friday and is officially listed as questionable. Meanwhile, the Commanders will play Week 13 without Forbes Jr., who has been sidelined since
suffering an elbow injury against the Giants in Week 11. Head coach Ron Rivera called the injury "significant" earlier this week.

Broncos at Texans (-3)

Denver's lone player with an injury designation for Week 13 is Mathis, who did not practice Friday after being limited on Thursday. Outside of him, the Broncos
are healthy heading into this matchup with Houston. Wideouts Jerry Jeudy (groin) and Marvin Mims Jr. (hamstring) were both full participants on Friday after
being limited on Thursday.

Dell logged back-to-back days as a limited participant in practice to end the week and is officially questionable.
</item 51>

<item 52>
Fantasy Football RB Rankings Week 14: Who to start, best sleepers at running back (Sporting News, 2023-12-07T22:07:00+00:00)
The injury highlights the importance of handcuffs, as Zack Moss has put up multiple top-10 performances with and without Taylor this season.

Seattle's Kenneth Walker III (oblique) missed Week 12 and 13, and we're not confident in his ability to suit up against the hard-hitting 49ers coming off a big
win in Philly. Backup Zach Charbonnet exited early because of a knee injury against the Cowboys, putting his status in doubt, too. If both are out, DeeJay
Dallas would start and lead the backfield in touches.

Green Bay's Aaron Jones has a sprained MCL, and sources have revealed that the Packers continue to consider him week-to-week. He only logged four carries before
sustaining the injury in Week 11, then he sat for the Packers' Thanksgiving Day tilt with the Lions and their Sunday Night Football win over Kansas City. Why
rush him back if Green Bay is already winning games?
</item 52>

<item 53>
NFL Week 13 injuries: Browns to start Joe Flacco with Dorian Thompson-Robinson out, Saints' Olave questionable (CBSSports.com, 2023-12-01T22:52:36+00:00)
The wideout still has another hurdle to pass to be cleared to play on Sunday so his status is still very much up in the air.

Falcons (-2) at Jets

Atlanta is relatively healthy coming into this matchup. Hollins, who injured his ankle in Week 9, participated on a limited basis every day this week, which is
encouraging for his prospects to play on Sunday. As for Hughes, he practiced fully all week after suffering a hand injury last weekend against the Saints.
Linebacker Nate Landman (thigh) missed practice on Wednesday but will play vs. New York.

Rodgers was once again limited in practice but, as expected, will not be activated this week and will be sidelined as he continues his recovery from a torn
Achilles just a few months ago. Meanwhile, Cook popped up on the injury report due to a shoulder injury, but head coach Robert Saleh told reporters Friday that
the back should be fine.
</item 53>

<item 54>
Fantasy Football RB Rankings Week 12: Who to start, best sleepers at running back (Sporting News, 2023-11-21T21:33:00+00:00)
They weren't awful against Dallas last week on a per-touch basis, but neither finished with more than 12 touches. Even worse, no Carolina back has enjoyed a
60-yard rushing game since Oct. 15. Bad offensive line, negative game scripts, awful coaching — it's all bad, man!

What are the biggest RB injuries in Week 12?

Second-year Seahawks back Kenneth Walker III suffered an oblique injury in Week 11 that the ever-hip Pete Carroll referred to as "legit." KW3 has a better
chance of missing multiple weeks than he does at suiting up for Seattle's Thanksgiving Day game against Fred Warner and the 49ers. If Zach Charbonnet somehow
finds himself on your waiver wire, an early Merry Christmas to you if you have the FAAB or a high waiver priority.

The hits just keep on coming for the Packers' Aaron Jones, who suffered a knee injury but seems to have at least caught a break in the sense that he didn't tear
his ACL.
</item 54>

<item 55>
Fantasy Football Week 13 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-11-30T22:10:58+00:00)
Weekly Breakdown MATCHUP @ LAR LAR -3.5 O/U 40.5 OPP VS WR 3rd PROJ PTS 11.4 WR RNK 23rd YTD Stats REC 47 TAR 86 REYDS 765 TD 2 FPTS/G 12.3 It appears Cooper
will play through a rib injury he suffered last week, and it also appears he'll catch passes from new Browns quarterback Joe Flacco. This is actually a great
thing for Cooper; over his past two weeks his ADOT sat at 7.57 yards, which was roughly half of what it was in his first nine games at 15.42 yards. Clearly, the
Browns wanted Cooper to run shorter routes and make things easier on rookie QB Dorian Thompson-Robinson.
</item 55>

<item 56>
VAR apology list in Premier League: Updated list of PGMOL apologies for referee mistakes in English football (Sporting News, 2023-09-30T21:27:02+00:00)
The Professional Game Match Officials Limited (PGMOL), the body responsible for referees in English football, has apologised on multiple occasions for mistakes
made during matches. Here is an updated list of the incidents.

MORE: Kevin De Bruyne injury: How long Man City ace could be out

Every PGMOL apology for referee mistakes

Here are all the apologies carried out by the PGMOL for wrong decisions later accepted to have been wrong.

Tottenham 2-1 Liverpool: September 30, 2023

Liverpool had their hearts broken as Joel Matip's stoppage-time own-goal gave Tottenham a 2-1 victory. The Reds, down to nine men after a red card to Curtis
Jones and two bookings for Diogo Jota, were looking to hang on at the death, and nearly did so until the cruel finish.

However, the game could have turned out completely different had the VAR booth not committed a "serious human error" earlier in the match.
</item 56>

<item 57>
Fantasy Injury Updates Week 10: News on Justin Jefferson, Ja'Marr Chase, T.J. Hockenson & Curtis Samuel (Sporting News, 2023-11-11T21:17:00+00:00)
Chase put a scare into the fantasy community on Wednesday when he was listed as a DNP, but after logging back-to-back limited practices, there is a lot more
optimism heading into Sunday. With Tee Higgins (hamstring) already ruled out, Cincinnati will be hoping Chase can feel healthy enough to play when they host the
Texans at 1 p.m. ET on Sunday.

The Bengals, despite getting a win over the Bills last Sunday, are a beaten-down bunch. Their defense will also be without a prominent edge rusher, Sam Hubbard,
complicating matters for a team hoping to stack wins as they compete for an AFC North crown. On the bright side, Joe Burrow appears to be finally healthy and
should have no problems targeting Tyler Boyd, Trenton Irwin, and Andrei Iosivas in the passing game. If Chase does end up suiting up, fantasy owners should
enjoy an even larger workload than the receiver already had.
</item 57>

<item 58>
NFL Week 13 injuries: Browns to start Joe Flacco with Dorian Thompson-Robinson out, Saints' Olave questionable (CBSSports.com, 2023-12-01T22:52:36+00:00)
5. As for Cox, he is questionable despite not practicing Friday due to a groin injury.

Browns at Rams (-3.5)

Browns : QB Dorian Thompson-Robinson (concussion), WR Marquise Goodwin (concussion), CB Denzel Ward (shoulder) OUT; C Nick Harris (knee) QUESTIONABLE

: QB Dorian Thompson-Robinson (concussion), WR Marquise Goodwin (concussion), CB Denzel Ward (shoulder) OUT; C Nick Harris (knee) QUESTIONABLE Rams: S Quentin
Lake (hamstring) OUT; OLB Michael Hoecht (knee) QUESTIONABLE

The Browns have officially ruled out Thompson-Robinson due to a concussion he suffered last week against Denver, which will thrust veteran Joe Flacco in as the
starter for Cleveland in Los Angeles.

For the Rams, Hoecht didn't practice the last two days after being a limited participant on Wednesday.
</item 58>

<item 59>
Ole Miss vs. Georgia live score, updates, highlights from Week 11 college football game (Sporting News, 2023-11-11T22:23:02+00:00)
Unfortunately for Rebels, injury continues to plague them after Micah Pettus, the team's top lineman, was ruled out with a broken foot bone he sustained during
practice. Quarterback Jaxson Dart left the game at the end of the third quarter after a hit from two Georgia defenders. He held his collarbone as he entered the
injury tent. He was taken back to the locker room for further examination and did not return to the game, leaving it in the hands of Spencer Sanders. Before his
exit, Dart passed for 112 yards and one interception. Sanders finished it up with 61 yards on four completions. For the Dawgs, Carson Beck finished with 306
yards, two touchdowns and one interception. It's the fourth game this season with over 300 yards and at least two touchdowns. In its trip to the SEC
Championship for the third-straight year, Georgia will face Alabama, who it lost to in 2021.
</item 59>

<item 60>
NFL Fantasy Football Week 5 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-04T21:56:25+00:00)
FILM ON PITTS: There's no change in his explosiveness whatsoever. He's running at a slower-than-expected pace, but the Jaguars still paid him plenty of
attention defensively, creating a double-whammy that left him without many targets. His longest opportunity was a deep ball from Desmond Ridder that was
well-placed and perfectly leapt for by Pitts, but he couldn't secure it to the ground, nor was he separated from cornerback Tyson Campbell at all. Pitts was
best on shorter routes against zone coverage.

ARTHUR SMITH: Said this week that Pitts isn't at 100%, adding " there are certain things, you see him, and he's doing really well right now and there are
certain things that he can't [do well]. It's been a journey back. He's going to get there."
</item 60>

<item 61>
Pochettino looks past James mistake and stakes future on his ‘Chelsea DNA’ (The Guardian, 2023-10-20T21:30:28+00:00)
That would have been too simple. Rather for their next game at Burnley on 7 October when, according to Pochettino, James was back in contention for selection.

The 23-year-old was not called up by England for the internationals over this past window and the idea was that he would work to be ready for Saturday’s home
derby against Arsenal. And yet there have been stories of an injury setback, even that he may need surgery to get to the root of his hamstring issues.

Pochettino bustled his way through those questions on Friday and the best read on James’s status was that he would be back in the squad for Arsenal, most likely
as a substitute. Gusto is available again after his own suspension. “It [James’s hamstring] is much better,” Pochettino said. “He is progressing well, still not
at his best, 100%. But maybe to help the team at some point …. maybe, he is close to be there.”
</item 61>

<item 62>
Breaking down each VAR intervention from wild first half of Tottenham vs Chelsea officiated by Michael Oliver (Sporting News, 2023-11-06T22:06:19+00:00)
📸 - The official VAR image, offside! No 2-0! pic.twitter.com/Md6DIzmAVe — 𝐀𝐅𝐂 𝐀𝐉𝐀𝐗 💎 (@TheEuropeanLad) November 6, 2023

Destiny Udogie escapes red card for bad tackle

Tottenham were lucky to remain with 11 players in the 18th minute, as Destiny Udogie went in with a horrible two-footed tackle on Raheem Sterling out wide.

Udogie would see yellow initially from Michael Oliver, but there was a subsequent VAR check to review for a possible red card, and for good reason. Udogie's
tackle was extremely dangerous, as he lunged in with a leg-breaker.

Ultimately, the Tottenham defender would escape without further punishment for one reason: he missed.
</item 62>

<item 63>
Fantasy Football RB PPR Rankings Week 14: Who to start, best sleepers at running back (Sporting News, 2023-12-07T22:09:00+00:00)
Kansas City's front-seven has been tough against speedsters and more vulnerable to power backs, and the 5-11, 190-pound Cook 100 percent falls into the former
category. Temper your expectations.

We feel the same way about Zach Charbonnet (@ 49ers). It's great that the Week 13 injury he sustained on Thursday night was just a bruise, but it's not great
that he now faces a 49ers defense that already stifled him on Thanksgiving. You're probably starting him if Kenneth Walker III remains out with his oblique
injury, but consider other options if you have the depth. The floor and ceiling remain low against this uber-confident Niners D.

Tyler Allgeier (vs. Buccaneers) continues to see his role diminish for the NFC South-leading Falcons, with dum-dum coach Arthur Smith finally realizing that
Bijan Robinson gives the ATL the best chance at continued success.
</item 63>

<item 64>
Theatre producer and Everton chairman Kenwright dies (BBC News - Entertainment & Arts, 2023-10-24T16:32:09+00:00)
Last updated on .From the section Everton

Bill Kenwright became Everton chairman in 2004

Everton chairman Bill Kenwright, an acclaimed West End theatre and film producer, has died at the age of 78.

Kenwright had surgery to remove a cancerous tumour from his liver eight weeks ago.

He had been on the board at Everton since 1989, taking over as chairman at Goodison Park in 2004.

The Liverpudlian was also one of the UK's most successful theatre producers and played Gordon Clegg in Coronation Street between 1968 and 2012.

Kenwright was awarded a CBE for his services to film and theatre in the 2001 New Years Honours List.

Everton said it was "in mourning" following the death of the club's longest-serving chairman for more than a century.

"The club has lost a chairman, a leader, a friend, and an inspiration," Everton added.

On 12 October, the Premier League club said Kenwright had a cancerous tumour removed from his liver six weeks before.
</item 64>

<item 65>
Fantasy Football Week 16 Wide Receiver Preview: Waiver adds, DFS plays, and more (CBSSports.com, 2023-12-19T20:51:32+00:00)
Weekly Breakdown MATCHUP vs JAC TB -1 O/U 43 OPP VS WR 25th PROJ PTS 13.4 WR RNK 24th YTD Stats REC 68 TAR 108 REYDS 814 TD 2 FPTS/G 11.9 Calvin Ridley WR JAC
Jacksonville

Age: 29 • Experience: 5 yrs. Weekly Breakdown MATCHUP @ TB TB -1 O/U 43 OPP VS WR 16th PROJ PTS 11.6 WR RNK 26th YTD Stats REC 60 TAR 109 REYDS 781 TD 5 FPTS/G
12.4

WR Preview Waiver Wire Targets

Week 16 Adds (WR Preview) Curtis Samuel WR WAS Washington • #4

Age: 27 • Experience: 7 yrs.
</item 65>

<item 66>
Michigan conquers Ohio State again, but the story is far from over for 2023 Wolverines (Sporting News, 2023-11-25T23:20:56+00:00)
ANN ARBOR, Mich. – Michigan quarterback J.J. McCarthy heard Zak Zinter scream. McCarthy saw tackle Karsen Barnhart holding Zinter's foot in the air. This was a
gruesome scene at Michigan Stadium on Saturday.

No. 3 Michigan lost their senior All-American guard to a broken tibia and fibula with 2:19 left in the third quarter of a tie game with No. 2 Ohio State. Zinter
was carted off in what could have been a turning point in the other direction given his value on the interior line.

"It was a sight that I wouldn't wish upon anyone to see," McCarthy said. "At that moment, seeing the look in everyone's eyes, seeing them rally together, it was
spiritual, honestly. It was a different drive that hit on everybody."

On the next play, senior running back Blake Corum finally broke through the second level of the Buckeyes' defense for a 22-yard go-ahead touchdown.
</item 66>

<item 67>
Bills vs. Bengals live score, updates, highlights from NFL 'Sunday Night Football' game (Sporting News, 2023-11-05T23:47:00+00:00)
He's the lone analyst that picks the Bills to beat the Bengals.

Almost a Bengals sweep... 👀 pic.twitter.com/KDh3awFV2y — Sunday Night Football on NBC (@SNFonNBC) November 6, 2023

7:56 p.m.: Before the game, former punter Kevin Huber and former long-snapper Clark Harris are recognized on the jumbotron. The two are Cincinnati legends, with
Huber having played in the most games as a Bengal (216 games) and Harris ranking fourth (202 games). The two are the rulers of the jungle for the game.

7:45 p.m.: Rasul Douglas will be playing, but The Athletic's Joe Buscaglia is reporting Dane Jackson is getting more of the first-team snaps during warmups.

6:50 p.m.: As expected, Hamlin is officially listed as inactive for the game.

6:47 p.m.: Hamlin running out onto the field before the game.

6:29 p.m.
</item 67>

<item 68>
Breaking down each VAR intervention from wild first half of Tottenham vs Chelsea officiated by Michael Oliver (Sporting News, 2023-11-06T22:06:19+00:00)
Ultimately, the Tottenham defender would escape without further punishment for one reason: he missed. Just like Kai Havertz of Arsenal earlier this weekend, who
also remained on the pitch against Newcastle for a similar incident, Udogie's tackle would not reach the necessary threshold for a straight red as he did not
make full contact with Sterling.

NBC commentator Jon Champion, who has real-time access to the VAR communication, said that the review ultimately determined the tackle was an "upper-level
yellow card" that didn't quite meet the qualifications for a red.

Cristian Romero next to avoid straight red card despite review

Just four minutes later, another Tottenham defender would again get away with just a booking despite a VAR review for a possible red card.

Cristian Romero, often criticised for his reckless nature, had a kick-out at Levi Colwill in the 22nd minute behind the play while laying on the ground. He
kicked Colwill right in the calf, but no action was taken.

This passage of play was not reviewed by VAR.
</item 68>

<item 69>
2023 NFL Week 11 injury roundup: Rams WR Cooper Kupp, Seahawks RB Kenneth Walker III among stars to exit (CBSSports.com, 2023-11-19T23:51:49+00:00)
I agree to receive the "Pick Six Newsletter" and marketing communications, updates, special offers (including partner offers), and other information from CBS
Sports and the Paramount family of companies. By pressing sign up, I confirm that I have read and agree to the Terms of Use and acknowledge Paramount's Privacy
Policy See All Newsletters Please check the opt-in box to acknowledge that you would like to subscribe. Thanks for signing up! Keep an eye on your inbox. Sorry!
There was an error processing your subscription.

WR

Tyreek Hill (Dolphins) briefly left Miami's game against the Raiders while nursing a hand injury and was ruled questionable to return, but he took the field
again soon afterward, securing seven catches for 80 yards and a score before halftime. Hill has left multiple games this year, only to return in all of them.

Cooper Kupp (Rams) had just a single catch against the Seahawks before suffering an ankle injury while blocking.
</item 69>

<item 70>
McRae reveals hidden message he carried under collar of his grand final shirt (The Sydney Morning Herald, 2023-09-30T22:45:39+00:00)
Adams and McStay – who arrived from the Lions in the off-season as an unrestricted free agent – leaned on each other to cope with the difficult moment, and
needed some privacy to deal with their personal misfortune before they could share in the team’s raucous celebrations. McStay may need as many as eight weeks to
recover from the high-grade medial collateral ligament strain he suffered in the club’s preliminary final win. Collingwood’s injured recruit Dan McStay at
training before the grand final. Credit: Eddie Jim “We shed a few tears and a bit of emotion, then we brushed it off and said, ‘Let’s get on with it and go
celebrate with the boys’. It was a great night,” Adams said. John Noble was another sad story, playing every game in the home-and-away season, only to be
dropped for Collingwood’s qualifying final clash with Melbourne and never forcing his way back in.
</item 70>

<item 71>
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
</item 71>

<item 72>
Chelsea and England star Cole Palmer: ‘The most annoying thing about moving down south? The traffic’ (The Independent - Sports, 2023-11-15T22:32:05+00:00)
Sign up to Miguel Delaney’s Reading the Game newsletter sent straight to your inbox for free Sign up to Miguel’s Delaney’s free weekly newsletter Please enter a
valid email address Please enter a valid email address SIGN UP I would like to be emailed about offers, events and updates from The Independent. Read our
privacy notice Thanks for signing up to the

Football email {{ #verifyErrors }} {{ message }} {{ /verifyErrors }} {{ ^verifyErrors }} Something went wrong. Please try again later {{ /verifyErrors }}

Cole Palmer is a man on the move, from Manchester City to Chelsea, from perennial substitute to pivotal starter, from England Under-21 player to, perhaps, a
senior international. It has been a few months of swift change. He is trying to get used to new surroundings, having swapped south Manchester – in his native
Wythenshawe – for the South.

“The most annoying thing? The traffic, that’s the biggest thing for me. Everything else I am enjoying.
</item 72>

<item 73>
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
</item 73>

<item 74>
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
</item 74>

<item 75>
VAR apology list in Premier League: Updated list of PGMOL apologies for referee mistakes in English football (Sporting News, 2023-09-30T21:27:02+00:00)
Referees once again took the headlines on the opening matchday of the 2023/24 Premier League season following an incorrect VAR decision in Manchester United's
fixture against Wolves.

Gary O'Neil's side were denied a late penalty after a foul from Andre Onana on Sasa Kalajdzic. Despite a long VAR check, Wolves were not given the spot-kick in
injury time and lost the match 1-0.

Fans watching the game were in disbelief when the call was made, and similar incidents have happened since, including against United. Bruno Fernandes said he
would wait for an apology — one that never arrived — after Cristian Romero was not punished for handball in the Red Devils' 2-0 loss to Tottenham, and plenty of
onlookers debated the penalty won by Marcus Rashford in United's subsequent comeback win at home to Nottingham Forest.
</item 75>

<item 76>
Ole Miss vs. Georgia live score, updates, highlights from Week 11 college football game (Sporting News, 2023-11-11T22:23:02+00:00)
6 p.m. – The Rebels are ready.

5:30 p.m. – Will we see Brock Bowers return for his final home game? The tight end is a game-time decision but could return after a high ankle sprain sustained
a month ago.

No. 19 out for warmups pic.twitter.com/tXyTEzATk7 — Brooks Austin (@BrooksAustinBA) November 11, 2023

5:15 p.m. – The weather in Athens is anything but ideal for a top-10 matchup. It'll be in the 40s with a chance of rainfall throughout the evening. But is that
better for the Dawgs? Last time ESPN's College Gameday was in town for a crucial SEC matchup, Georgia beat Tennessee on a rainy afternoon. Kirby Smart and his
crew seem to only get better in rough weather.

What channel is Ole Miss vs. Georgia on today?
</item 76>

<item 77>
Manchester divided as ten Hag pressure mounts after United Hammered, Ange-ball pays off again as Spurs sink Everton (The Roar | Sports Writers Blog,
2023-12-23T21:19:19+00:00)
Klopp said Tsimikas’ collarbone was “definitely broken”, adding he will be “out for a long time.”

Andros Townsend headed the winner for Luton Town as they beat Newcastle United in their first match since captain Tom Lockyer suffered a cardiac arrest agasint
Bournemouth a week earlier.

Advertisement

PURE ANGE-BALL! Richarlison scores against his old club to give Tottenham the lead! ????



"What a superb goal from Spurs – they just passed their way through Everton!" ????



Stream #PLGoalRush LIVE, or on-demand after, here ????
</item 77>

<item 78>
2023 NFL Week 11 injury roundup: Rams WR Cooper Kupp, Seahawks RB Kenneth Walker III among stars to exit (CBSSports.com, 2023-11-19T23:51:49+00:00)
DB

Talanoa Hufanga (49ers), who emerged as a ballhawk for San Francisco's secondary in 2022, left early against the Buccaneers with a knee injury. Coach Kyle
Shanahan confirmed to reporters on Monday that the All-Pro suffered a season-ending ACL tear.
</item 78>

<item 79>
Man United vs Fulham: Red Devils determined to get back to winning ways in PL (Insidesport, 2023-11-03T21:58:48+00:00)
Fulham Team News

Fulham’s injury situation has not improved so far. Adama Traore is suffering from a thigh problem, Issa Diop is nursing a foot issue, Kenny Tete is out due to
an unspecified injury, and Tosin Adarabioyo is suffering from a groin injury.

However, Tete, Adarabioyo, and Traore are on their way to being fully fit. They could at least make the squad for this weekend’s high-stake match against Man
United. Yet, the coach’s biggest concern revolves around his No. 9 position. With his first goal in the EFL Cup match, Rodrigo Muniz proved that he can be a
good pick for the role on a regular basis.

The coach hinted that he could choose Muniz over Raul Jimenez who is yet to score since he joined Fulham this summer.
</item 79>

<item 80>
Salah seals emphatic Europa League win for Liverpool against Toulouse (The Guardian, 2023-10-26T21:00:42+00:00)
Aberdeen's hopes of navigating a route through the Conference League group stage suffered a major blow as they threw away a two-goal lead to lose 3-2 against
PAOK Salonika on a rollercoaster night at Pittodrie. Having weathered heavy pressure in a goalless first half, Barry Robson's men found themselves two goals to
the good by the hour mark thanks to Bojan Miovski and Dante Polvara. But a stunning comeback saw the Greek side take three points after goals from substitutes
Kiril Despodov and captain Vieirinha, and a stoppage-time penalty, awarded after a VAR check, from Stefan Schwab. Was this helpful? Thank you for your feedback.

It was soon three when Gravenberch surged through the Toulouse midfield and found Jones. The midfielder’s shot was blocked by Mikkel Desler but rebounded off
Jones into the path of Núñez, who lashed a thumping finish into the roof of Restes’ net.

The second half was memorable mainly for misses.
</item 80>

<item 81>
NFL Fantasy Football Week 6 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-12T21:23:16+00:00)
Consider Lockett more of a top-30 receiver who you should sit for Zay Flowers and Gabe Davis but start over Chris Godwin, Nico Collins, Garrett Wilson and Drake
London.

BENGALS: Consistently tailor their defensive coverage to the weaknesses of the offenses they face. So when a quarterback struggles against zone coverage,
they'll play more zone. This worked in their favor in all but one game this year (Ravens, Week 2). Cincinnati also has turned on the juice for their pass rush,
getting a pressure rate of over 40% in each of their past three games, all versus teams with suspect offensive lines.

BENGALS: Are expected to be without top cornerback Chidobe Awuzie, and they also are rotating one of their safety spots between Nick Scott and Jordan Battle.
D.J. Turner is expected to replace Awuzie.
</item 81>

<item 82>
Liverpool vs Everton begins epic Saturday with World Cup semi-final as well as Chelsea vs Arsenal – and it could end with a bang (TalkSport,
2023-10-20T21:34:28+00:00)
And they boast a brilliant record against the Blues of late, winning six out of their last seven meetings.

They have triumphed in their last three trips to Stamford Bridge, and haven't lost in a London derby since May 2022.

As for Chelsea, they go into their match against the Gunners hopeful of a result having won their last three games in all competitions.

Both teams face injury doubts, with Arsenal to potentially be without Bukayo Saka and William Saliba, while it remains to be seen if Reece James and Armando
Broja feature for the hosts.

9 Saka missed Arsenal's 1-0 victory over City at the Emirates last time out Credit: Getty

Sheffield United desperate for points against lacklustre Manchester United

With just one point on the board, Sheffield United's season could not be going much worse given they are rock bottom.

With a fixture against Manchester United on the cards, though, they may feel their hopes could be looking up given Erik ten Hag's men disappointing season.
</item 82>

<item 83>
NFL Week 13 injuries: Browns to start Joe Flacco with Dorian Thompson-Robinson out, Saints' Olave questionable (CBSSports.com, 2023-12-01T22:52:36+00:00)
Armstead has stated that he will suit up on Sunday.

Philadelphia saw the return of tight end Goedert at practice on Friday. He was limited for the session and officially listed as doubtful after practicing for
the first time since suffering a fractured forearm against the Cowboys on Nov. 5. As for Cox, he is questionable despite not practicing Friday due to a groin
injury.
</item 83>

<item 84>
NFL Week 13 injuries: Browns to start Joe Flacco with Dorian Thompson-Robinson out, Saints' Olave questionable (CBSSports.com, 2023-12-01T22:52:36+00:00)
Bengals at Jaguars (-8.5), Monday

Bengals : CB Cam Taylor-Britt (quad/ankle), D'Ante Smith (knee), Jay Tufele (illness) OUT

: CB Cam Taylor-Britt (quad/ankle), D'Ante Smith (knee), Jay Tufele (illness) OUT Jaguars: CB Tyson Campbell (hamstring), RB Travis Etienne (ribs), TE Brenton
Strange (foot), TE Luke Farrell (toe) QUESTIONABLE

The Bengals got good news on the injury front ahead of Monday's game against the Jaguars. Wide receiver Tee Higgins (hamstring/ankle) and linebacker Logan
Wilson (ankle/foot) are set to play, according to head coach Zac Taylor. Higgins and Wilson were full participants in practice. Cornerback Cam Taylor-Britt hurt
his foot in practice and is out.
</item 84>

<item 85>
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
</item 85>

<item 86>
Inter Miami vs FC Cincinnati live score, result, updates, highlights with Lionel Messi's MLS playoffs on line (Sporting News, 2023-10-07T21:30:00+00:00)
MORE: Details on Lionel Messi's injury as he battles "fatigue" and a "scar tissue issue".

La idea es que Leo Messi vuelva a sumar minutos mañana cuando Inter Miami juegue contra Cincinnati. pic.twitter.com/ePNWHb4vSs — Gastón Edul (@gastonedul)
October 6, 2023

Inter Miami vs FC Cincinnati lineups & team news

There was a report that Lionel Messi could finally make his return from injury, and while it felt unlikely that Messi would start three days after missing out
entirely, he appears on the bench today and could be available for a second half appearance if required.

Jordi Alba remains sidelined with a hamstring injury, missing since his injury just minutes before Messi, and without a publicly known timeline for a return.
Also missing is striker Leonardo Campana who was absent from the midweek defeat to the Chicago Fire with a muscle injury, although Gerardo Martino described his
issue as "minor."
</item 86>

<item 87>
Fantasy Injury Updates Week 10: News on Justin Jefferson, Ja'Marr Chase, T.J. Hockenson & Curtis Samuel (Sporting News, 2023-11-11T21:17:00+00:00)
ET, according to ESPN's Jeremy Fowler. The receiver was limited in practice thanks to the toe injury that has plagued him since leaving Week 8's game against
the Eagles, but barring a setback, he will return to action this weekend.

The receiver was working his way into flex territory with a couple of good performances before picking up the injury. However, we wouldn't consider him to be
worthy of a fantasy start this week unless you're in a bind. It's unclear whether Samuel will regain the same workload or if he'll be playing in limited action
as he works his way back. Jamison Crowder emerged in Samuel's absence and Jahan Dotson has seen his production drastically increase, as well. The wait-and-see
approach is the preferred one heading into Week 10.
</item 87>

<item 88>
Breaking down each VAR intervention from wild first half of Tottenham vs Chelsea officiated by Michael Oliver (Sporting News, 2023-11-06T22:06:19+00:00)
This passage of play was not reviewed by VAR. It's unclear exactly why, but an assumption can be made that the VAR official was likely not aware of the incident
until play had already restarted, at which point he cannot intervene. Those incidents have to be caught before play restarts, which is extremely difficult to do
as the game continues to flow.

🚨| Romero intentionally kicks Colwill but VAR have not checked it. pic.twitter.com/VCfMjmYEWR — CentreGoals. (@centregoals) November 6, 2023

Raheem Sterling goal disallowed

Chelsea were the next to see a goal chalked off, as Raheem Sterling's 21st minute equaliser was nullified on review. The decision came as Sterling's initial
effort, blocked by Pedro Porro, deflected straight back onto the arm of the Chelsea forward.
</item 88>

<item 89>
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
</item 89>

<item 90>
Liverpool vs Everton begins epic Saturday with World Cup semi-final as well as Chelsea vs Arsenal – and it could end with a bang (TalkSport,
2023-10-20T21:34:28+00:00)
talkSPORT will bring you updates as England take on South Africa in the Cricket World Cup in Mumbai from 9:30am

9 It's not looking good for England's cricketers

UFC 294 in Abu Dhabi

Islam Makhachev is hoping to to successfully defend his UFC lightweight belt against Alexander Volkanovski at UFC 294.

Makhachev retained his title in February this year as he defeated Volkanovski by a unanimous decision.

But despite the Russian's favourable 24-1-0 record, Volkanovski will be desperate to add the lightweight belt to his featherweight title.

Also on the cards, Kamaru Usman will fight Khamzat Chimaev in a middleweight title eliminator.

Expect fireworks.
</item 90>

<item 91>
Fantasy Football Week 13 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-11-30T22:10:58+00:00)
Can't say the same against the run as the Redbirds rank in the bottom five in pretty much every single run defense metric except yards after contact per rush
(where they're league average). A running back has scored 17 or more PPR points in four of their past five against the Cardinals, and eight in 12 games have hit
at least 15 PPR points. With Pickett nursing an ankle injury we could see the Steelers run the ball aplenty.
</item 91>

<item 92>
Fantasy Football RB Rankings Week 12: Who to start, best sleepers at running back (Sporting News, 2023-11-21T21:33:00+00:00)
Both Darrell Henderson and Royce Freeman (@ Cardinals) would be decent dart throws in a week filled with byes, but their floors in a zero-bye week seem too low
to trust them if Kyren Williams is back and at 100 percent. The matchup is juicy, but we have no idea what to expect from a touch-share perspective, so these
guys are no better than desperation, TD-dependent flexes.

Speaking of guys coming back from injury, we saw Khalil Herbert (@ Vikings) return to the field for Chicago against the Lions last week. However, his
performance left a lot to be desired. He finished with just 35 yards on 16 carries and wound up vastly overshadowed by Justin Field's 100-yard rushing effort.
Even worse, D'Onta Foreman scored the Chi's lone rushing TD on the afternoon.

I don't care what your team looks like — you shouldn't be starting Chuba Hubbard or Miles Sanders (@ Titans).
</item 92>

<item 93>
Fantasy Football WR PPR Rankings Week 13: Who to start, best sleepers at wide receiver (Sporting News, 2023-11-29T20:27:00+00:00)
Geno Smith's struggles complicate their fantasy prospects, as well.

What are the biggest WR injuries in Week 13?

Entering Week 12, Tee Higgins (hamstring), Noah Brown (knee), Treylon Burks (concussion), and Kadarius Toney (ankle, hip) all failed to suit up due to nagging
injuries. For now, we've placed Higgins and Brown in our Week 13 list and omitted Burks and Toney. We'll also be eyeing the Chargers' Josh Palmer (knee), who's
eligible to come off the IR this week but probably won't suit up against New England.

Last week, the most notable WRs to exit their respective games with injuries include Chris Olave (concussion), Amari Cooper (ribs), Demario Douglas
(concussion), and Rashid Shaheed (thigh). We've left all four WRs out of our initial Week 13 list but will closely monitor their teams' injury reports to gauge
their playing prospects.
</item 93>

<item 94>
Liverpool vs Everton begins epic Saturday with World Cup semi-final as well as Chelsea vs Arsenal – and it could end with a bang (TalkSport,
2023-10-20T21:34:28+00:00)
And Klopp has a new worry: the Reds are without Andy Robertson after the left-back suffered a shoulder injury while on international duty with Scotland.

His absence could open up a gap in their defence for Everton to exploit.

The Toffees were winless in their first five games, but have since picked up two much-needed wins.

But Everton are up against the odds, having won just one out of the last 25 previous Merseyside derbies.

Liverpool vs Everton is live on talkSPORT at 12:30pm on October 21

9 Sean Dyche's side go into the Merseyside derby as heavy underdogs Credit: Getty

Crystal Palace look to end Newcastle's building momentum

The Magpies have been on an impressive run of form across all competitions, knocking last year's treble winners Manchester City out the Carabao Cup while
defeating Paris Saint-Germain 4-1 in the Champions League.

But Crystal Palace proved to be an unbeatable obstacle for Newcastle last season.
</item 94>

<item 95>
Fantasy Football RB PPR Rankings Week 14: Who to start, best sleepers at running back (Sporting News, 2023-12-07T22:09:00+00:00)
Tampa Bay has a pretty good run defense, so Allgeier can't be trusted outside of a desperate flex play with his nine average touches over the past two weeks.

What are the biggest RB injuries in Week 14?

Derrick Henry (head/neck), Rhamondre Stevenson (ankle), Kenneth Walker III (oblique), Aaron Jones (knee), D'Onta Foreman (ankle), and D'Andre Swift
(undisclosed) are all "questionable," at best, to suit up this week. We know for sure Stevenson will be out, but we'll have to monitor the injury reports
throughout the week for the rest.

Jonathan Taylor (thumb) underwent surgery last week and will miss the next few weeks, opening the door for Zack Moss to resume his role as Indy's super-sub lead
back.

We'll be updating these RB PPR rankings all week, so check back for the latest player movement.
</item 95>

<item 96>
Fantasy Football Week 13 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-11-30T22:10:58+00:00)
The uncertainty of how the Panthers will operate moving forward with Brown basically calling the offense without any oversight keeps these players are risky
Fantasy propositions, though it helps a lot that the Buccaneers are dealing with injuries that hurt their run defense (Lavonte David and Devin White will miss
the game) and keep their pass defense from improving (Jamel Dean will miss the game).
</item 96>

<item 97>
Liverpool enjoy home comforts but Dyche’s Everton offer derby danger | Andy Hunter (The Guardian, 2023-10-20T21:30:28+00:00)
Crazy. We had to learn to control the game. It is all the things that happen with time – there is no shortcut to that. With the signs we showed so far I am
absolutely fine but I don’t know the final destination, I can’t even see that yet, but it is not a problem because other teams have similar problems. It is
about how quick we can make the steps and that is what we are working on every day but the problem now is how could we work since Brighton [where Liverpool drew
2-2]? We had three, with a goalie, first-team players in training until Wednesday. Tomorrow we have to play with what we did so far but in general you have to
gain stability bit by bit.”

It has arrived in attack, where Darwin Núñez’s influence has improved considerably since his match-winning cameo for the 10 men of Liverpool at Newcastle.
</item 97>

<item 98>
VAR apology list in Premier League: Updated list of PGMOL apologies for referee mistakes in English football (Sporting News, 2023-09-30T21:27:02+00:00)
Soon after Jones was sent off in the first half, Luis Diaz scored to put Liverpool 1-0 up despite their disadvantage, but the flag went up for offside. Replay
seemed to show that Diaz was onside, but VAR Darren England ended his check without even drawing lines, and the decision stood.

The PGMOL issued a statement after the match stating the goal should have stood.

Manchester United 1-0 Wolves: August 14, 2023

After Onana clattered into Kalajdzic in the box, O'Neil thought a penalty would certainly be awarded. Instead, the Wolves manager received a yellow card on the
sidelines for his protests.

PGMOL's Jonathan Moss did apologise for the mistake after the match, which O'Neil said he appreciated.

Tottenham 2-1 Brighton: April 8, 2023

Chief Refereeing Officer Howard Webb had to give to Brighton an apology after they were denied a penalty in their away trip to Tottenham.
</item 98>

<item 99>
NBL Round 5: Harvey hits Hail Mary for Hawks to topple Taipans at the buzzer, Cheatham injured as Sixers upset Breakers (The Roar | Sports Writers Blog,
2023-10-26T21:11:37+00:00)
Without a hand being laid on him, ex-NBA forward Cheatham pulled up sore changing direction early in the fourth quarter at Auckland’s Spark Arena on Thursday
night.

Cheatham was unable to put weight on his right leg as Breakers staff helped him from the floor.

He played no further part in the match, but there was no immediate update on his condition after full-time.

Advertisement

“It’s too early. We’ll pray for good news,” said Breakers coach Mody Maor.

Any long-term injury would be troublesome for last season’s runners-up, who have begun the campaign with only one win from four games.

Cheatham’s injury proved a turning point. The Breakers had maintained a slim lead for most of the night as Cheatham (17 points, eight rebounds) combined with
guard Will McDowell-White to strong effect.
</item 99>

<item 100>
NFL Fantasy Football Week 6 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-12T21:23:16+00:00)
Half of the six targets were uncatchable. In his previous three with Deshaun Watson he had at least seven targets in each, and at least 16 PPR points in the
past two.

SINCE THEN: Watson hasn't practiced with the team. P.J. Walker was named the starter despite being on the team's practice squad and took reps this week in
preparation for their game against the Niners.

WALKER: Was cut by the Bears after the preseason before linking up with Cleveland. Has five career passing touchdowns in 15 games (seven starts). He's had
multiple passing scores in one NFL game. Only twice has a receiver had 100 yards from Walker (both by D.J. Moore, once in 2020, once in 2022). Two other times
has a WR had at least 70 yards (Curtis Samuel in 2020, Terrace Marshall in 2022).
</item 100>
=== END ITEM id=d2c51c501f597941 ===

=== ITEM id=7ab3251fc58f971c ===
Gold sentences and facts:
Sentence 0: Here's your chance to claim $5,000+ in bonuses as a new user from Caesars Sportsbook, BetMGM, Bet365, FanDuel, DraftKings, and BetRivers.
  Fact 0.0: The reader has a chance to claim $5,000+ in bonuses as a new user.
  Fact 0.1: A new user can claim bonuses from Caesars Sportsbook.
  Fact 0.2: A new user can claim bonuses from BetMGM.
  Fact 0.3: A new user can claim bonuses from Bet365.
  Fact 0.4: A new user can claim bonuses from FanDuel.
  Fact 0.5: A new user can claim bonuses from DraftKings.
  Fact 0.6: A new user can claim bonuses from BetRivers.
Sentence 1: Collect your welcome bonus: As a new customer, you’ll receive a welcome bonus.
  Fact 1.0: The reader can collect a welcome bonus.
  Fact 1.1: A new customer receives a welcome bonus.

Retrieved items:
<item 1>
The Halloween Countdown: 31 days of horror to watch (Polygon, 2023-10-10T15:27:53+00:00)
Very loosely based on Argento’s own experience directing a failed production of Verdi’s Macbeth, Opera is a sumptuous and suspenseful slasher that holds the
viewer’s attention and absolutely refuses to let go. Like Betty, one feels compelled to watch this macabre display of ritualistic horror play out to its
conclusion in a desperate search of answers. Fortunately, unlike Betty, you won’t have to worry about losing your eyes if you find yourself overwhelmed by the
frights. —TE

Oct. 29: House of 1000 Corpses

Where to watch: AMC Plus, Pluto TV, Tubi

House of 1000 Corpses may be the true answer to the question “What is the best The Texas Chain Saw Massacre sequel?”

The movie follows a group of teens who are putting together a book on roadside attractions when they stumble upon a serial killer-themed rest stop run by a
weirdo. After their strange encounter, the kids set off on the road, only for their car to break down a few miles later.
</item 1>

<item 2>
The Halloween Countdown: 31 days of horror to watch (Polygon, 2023-10-10T15:27:53+00:00)
Your time is precious, and your options are endless. The good news: We watch it all so you don’t have to. Polygon’s What to Watch highlights the best, the
funniest, the scariest, and the most exciting in movies, TV, and everything in between. Stop scrolling, start watching!

If there’s one thing we love here at Polygon, it’s the Halloween season.

We cover horror year-round, whether it’s the best horror movies you can stream at home or the best horror movies on Netflix, and generally, we have our finger
on the undead pulse of the newest and ghastliest releases in horror. We even have a list of the best horror movies of the year (ranked by scariness, of course).

Even still, Halloween is an especially spooky time of year, and it warrants special attention and celebration.
</item 2>

<item 3>
5 things to know before watching the Five Nights at Freddy’s movie (Polygon, 2023-10-26T21:11:01+00:00)
The possessed-animatronics horror video game Five Nights at Freddy’s is less than a decade old, but a movie version has been in the works almost since the game
launched, which seems especially protracted for a development process. Warner Bros. bought the movie rights in 2015, and the project moved over to Blumhouse a
couple of years later. Game creator Scott Cawthon wrote and rewrote the script, while kid-horror filmmakers like Gil Kenan (Monster House) and Chris Columbus
(Gremlins, the first Harry Potter and Percy Jackson movies) signed on and jumped off the project.

Finally, under director Emma Tammi, the film is hitting theaters simultaneously with a Peacock streaming debut. The franchise’s most hardcore fans probably
consider themselves beyond ready to watch the Freddy’s film. But what about everyone else? If you’re not sure, have a look at this handy (and extremely
spoiler-light) Q&A.
</item 3>

<item 4>
The Halloween Countdown: 31 days of horror to watch (Polygon, 2023-10-10T15:27:53+00:00)
Enjoyment of Dracula 2000 may vary depending on a tolerance for Hot Topic goofs, but if you don’t catch up with it, you can’t enjoy Lussier’s sequel, Dracula
II: Ascension, starring Jason Scott Lee as a martial-arts-fighting priest who kicks Dracula’s ass! —Matt Patches

Oct. 7: Prince of Darkness

Where to watch: Peacock

There aren’t enough movies about the end of the world. Sure, there are plenty of movies about preventing catastrophe at that scale, but rarely do they have the
follow-through to truly get existentially terrifying. Thankfully, John Carpenter is no coward, and he made three movies about different kinds of apocalypses.
But while The Thing is the best known, and In the Mouth of Madness is the zaniest, Prince of Darkness might be the scariest of the trilogy.

Prince of Darkness follows a group of college researchers who are sent to study a strange canister found in the basement of a church.
</item 4>

<item 5>
The Halloween Countdown: 31 days of horror to watch (Polygon, 2023-10-10T15:27:53+00:00)
Even still, Halloween is an especially spooky time of year, and it warrants special attention and celebration.

For the past three years, Polygon has put together a Halloween countdown calendar, selecting 31 of our staff’s top horror-themed or Halloween-adjacent picks
across movies and TV throughout the month of October, all available to watch at home. We’ve loved doing it, so much so that we’re bringing it back again — this
time with a whole new batch of films and shows to choose from.

Every day for the month of October, we’ll add a new recommendation to this countdown and tell you where you can watch it. So curl up on the couch, dim the
lights, and grab some popcorn for a terrifying and entertaining marathon of horrific delights.

Oct. 1: Messiah of Evil

Where to watch: Prime Video, Shudder, Pluto TV, Plex

A undersung classic of ’70s cosmic horror, Willard Huyck and Gloria Katz’s Messiah of Evil is a surreal and nightmarish experience that more than deserves its
status as a cult horror classic.
</item 5>

<item 6>
The best horror movies you can watch right now (Polygon, 2023-09-28T15:26:41+00:00)
After inadvertently being resurrected by a drop of blood spilled by Larry on the floor of the house’s attic, Frank seduces Julia into luring new men to the
house so that he can drain their life force and fully regain his mortal form. Surrounding this core narrative is the the story of the Lament Configuration, a
puzzle box Frank acquired before his untimely death. When solved, it conjures hellish beings known as Cenobites to the mortal plane of existence, which indulge
in hellish exercises of sadomasochistic mutilation. Easily the best and most enduring of the Hellraiser movie series, Barker’s 1987 original is a must-watch for
horror fans. —TE

Hereditary

Director: Ari Aster

Cast: Toni Collette, Alex Wolff, Milly Shapiro

Where to watch: Max

Hereditary is a victim of its own success.
</item 6>

<item 7>
The Halloween Countdown: 31 days of horror to watch (Polygon, 2023-10-10T15:27:53+00:00)
The film centers on Arletty, a young woman who journeys to the remote coastal town of Point Dume, California, to track down her missing father, a reclusive
painter who is plagued by disturbing visions of an impending apocalypse. After crossing paths with an eccentric aristocrat and his groupie companions, the group
bears witness to the realization of the painter’s prophecy, as the townspeople are transformed into flesh-eating ghouls and the messiah of evil makes his return
to the mortal realm.

At times convoluted and bizarre, Messiah of Evil is a genuinely entertaining horror thriller filled with memorable scares and chilling set-pieces. From a
cross-eyed albino man gleefully devouring a field mouse whole to a unsuspecting woman watching a Western as the theater around her slowly fills with ravenous
undead zombies, it’s a dreamy and frightening experience that lands somewhere between the ineffable gothic horror of H.P. Lovecraft and the anti-consumerist
allegories of George A. Romero.
</item 7>

<item 8>
The Halloween Countdown: 31 days of horror to watch (Polygon, 2023-10-10T15:27:53+00:00)
The story of The Exorcist III’s production is a tangled mess of false starts, studio interference, and thwarted artistic ambitions. Though it was originally
conceived with Friedkin in mind to return as director, he eventually left the project due to creative differences with Blatty. John Carpenter was at one time
tapped to direct, before he too stepped aside to allow Blatty himself to take the reins. Morgan Creek Productions, which financed the film, demanded Blatty
reshoot the film’s final act, which originally ended without an exorcism. Even given these complications, The Exorcist III remains a fascinating and terrifying
film in its own right, with a hodgepodge of fascinating creative choices and literary dialogue that makes the film feel like a sister piece to Blatty’s The
Ninth Configuration. It may not have been the Exorcist sequel that audiences at the time wanted, but it’s still a damn fine picture. —TE
</item 8>

<item 9>
The Halloween Countdown: 31 days of horror to watch (Polygon, 2023-10-10T15:27:53+00:00)
—PV

Oct. 31: The Exorcist III

Where to watch: Prime Video, Criterion Channel

Nearly 50 years since it was first released in theaters, William Friedkin’s The Exorcist has withstood the test of time as one of the most terrifying and
influential horror films in American cinema. The sequel that came closest to capturing that same lightning in a bottle is The Exorcist III, written and directed
by William Peter Blatty, the original screenwriter of the first film and author of the novel on which it was based.

When a series of gruesome murders matching the modus operandi of an infamous serial killer begins to terrorize the city of Georgetown, Lt. William F. Kinderman
(George C. Scott) investigates the psych ward of a local hospital in search of leads. Confronted with a suspect whose appearance matches that of his late friend
Father Damien Karras, Kinderman finds himself thrust into a battle for his very soul as he confronts an evil beyond his comprehension.
</item 9>

<item 10>
The Halloween Countdown: 31 days of horror to watch (Polygon, 2023-10-10T15:27:53+00:00)
Lovecraft and the anti-consumerist allegories of George A. Romero. The horrors hold up over 50 years after it was first released, and with just the right
combination of inadvertent silliness and legitimate terror, Messiah of Evil is a satisfying watch and an excellent start to a month of horrors. —Toussaint Egan

Oct. 2: Ginger Snaps

Where to watch: Criterion Channel, Shudder, Peacock, Vudu, Tubi, Freevee, Plex

There’s a sizable hole in the monster movie canon: There just aren’t enough good werewolf movies! Ginger Snaps is here to fix that. And if you love Jennifer’s
Body, you’re in for a bloody treat.

Directed by future Orphan Black co-creator John Fawcett, Ginger Snaps is a delightfully gruesome story about two sisters. One of them has gotten her period for
the first time, and is turned into a werewolf shortly after. Things get messy, fast.
</item 10>
=== END ITEM id=7ab3251fc58f971c ===
