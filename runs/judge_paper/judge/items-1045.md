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

=== ITEM id=be73cec7382825f6 ===
Gold sentences and facts:
Sentence 0: The Buckeyes settled for a 52-yard field goal late in the first half, which Jayden Fielding missed wide left.
  Fact 0.0: The Buckeyes settled for a 52-yard field goal late in the first half.
  Fact 0.1: Jayden Fielding missed the Buckeyes' 52-yard field goal wide left.
Sentence 1: His 31-yard field goal is good, and the Buffs take the first lead of the game on their first drive.
  Fact 1.0: A 31-yard field goal is good.
  Fact 1.1: The Buffs take the first lead of the game on their first drive.

Retrieved items:
<item 1>
Patriots vs. Broncos live score, updates, highlights from NFL Sunday night football game (Sporting News, 2023-12-24T23:00:03+00:00)
The Christmas Eve showdown between the Patriots and the Broncos was expected to be a solid victory by Denver against a team that had won just three games
heading into the contest. It was anything but that.

In a game that it had it all between critical turnovers and scores that came down to the final minute, Patriots kicker Chad Ryland hit a 56-yard field goal to
send the Patriots to a 26-23 victory.

Coming into this game, New England had played the Chiefs tough but lost at home to fall to 3-11 and improve its chances of landing the No. 1 pick in next year's
draft. Denver aimed to get back to the right side of the win column after getting blown out by the Lions in an all-around abysmal performance.

MORE: Chad Ryland redeems himself with game-winning field goal

In this one, it took the Broncos a while once more to put points on the board, with 16 of them coming in the fourth quarter in a comeback effort that proved
just a hit too late.
</item 1>

<item 2>
Chicago Fire vs Inter Miami live score, result, updates, highlights from Lionel Messi in MLS match (Sporting News, 2023-10-04T23:33:00+00:00)
Without Lionel Messi, Inter Miami have looked exceedingly mortal. They put in a 90-minute performance littered with individual mistakes, and the Chicago Fire
made them pay, winning 4-1 via braces from Xherdan Shaqiri and substitute Maren Haile-Selassie.

The result boosts Chicago’s postseason hopes significantly, moving them into a playoff position in the Eastern Conference standings.

Inter Miami’s playoff hopes, conversely, are now on life support, needing to win out in their final three games and get outside help. The club is winless in
four games with Messi out injured, and it has seen their chances at a postseason berth nearly evaporate.

Shaqiri’s brace was the biggest talking point for the hosts, as the former Liverpool midfielder has not always been at his best since joining MLS, but in front
of a record Chicago Fire crowd at Soldier Field, he rose to the occasion.

The Sporting News followed the Chicago Fire vs. Inter Miami match live, providing score updates, commentary and highlights as they happened.
</item 2>

<item 3>
NFL Fantasy Football Week 5 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-04T21:56:25+00:00)
You just might not have a better option than Fields.

LAST WEEK: Posted his first-ever game with four passing touchdowns and over 300 yards against a Broncos defense that was devoid of a serious pass rush or pass
coverage skills through the first three quarters. By the fourth quarter the Broncos were generating heavy pressure and Fields finished 5 of 11 for 50 yards and
no touchdowns with two turnovers.

FILM: For three quarters Fields looked terrific firing downfield from clean pockets with plenty of great reads and accurate throws along with a play or two
where he manipulated defenders with the threat of him running only to throw over their heads, including on Kmet's short touchdown. His off-target rate was
literally 0.0%. But by the fourth only some plays were fine -- on others he either was pressured, threw off his back foot, threw with too much velocity or
wasn't on the same page as his target.
</item 3>

<item 4>
Colorado vs. Arizona State live score, updates, highlights from Week 6 college football game (Sporting News, 2023-10-07T21:21:03+00:00)
Colorado needed a bounce back win and got one. Meanwhile Arizona State kept on struggling, falling to 1-5.

MORE: Watch Colorado vs. Arizona State live with Fubo (free trial)

The Sporting News tracked updates and highlights from college football's Week 6 game between Colorado and Arizona State.

Colorado vs. Arizona State score

Q1 Q2 Q3 Q4 Final Colorado 7 7 0 13 27 Arizona State 7 10 0 7 24

Colorado vs. Arizona State results, highlights from Week 6 game

FINAL: Colorado 27, Arizona State 24

Colorado 27, Arizona State 24

10:17 p.m. FIELD GOAL – Absolutely clutch. Mata makes a 43-yard field goal look easy to give Colorado the lead with 12 seconds on the clock.

10:14 p.m. – Bullseye! Sanders finds Antonio on a 43-yard pass. Colorado takes over on the ASU 25.
</item 4>

<item 5>
Inter Miami vs FC Cincinnati live score, result, updates, highlights with Lionel Messi's MLS playoffs on line (Sporting News, 2023-10-07T21:30:00+00:00)
FC Cincinnati are otherwise light on availability concerns, with only a leg injury for Arquimides Ordonez and a long-term injury to young midfielder Stiven
Jimenez to worry about.

Talisman forward Luciano Acosta was partially rested midweek, coming off the bench to score the team's only goal against the Red Bulls in 32 minutes of play, so
he is ready to go as he chases the league's Golden Boot award. Also rested midweek was Alvaro Barreal who leads the team in minutes and also logged just 32 of
them on Wednesday.

FC Cincinnati starting lineup (5-3-2): Celentano (GK) — Arias, Hagglund, Miazga, Murphy, Barreal — Nwobodo, Acosta, Moreno — Vazquez, Santos.

FC Cincinnati subs (9): Kann (GK), Kubo, Mosquera, Angulo, Badji, Pinto, Halsey, Valenzuela, Boupendza.
</item 5>

<item 6>
Inter Miami vs FC Cincinnati live score, result, updates, highlights with Lionel Messi's MLS playoffs on line (Sporting News, 2023-10-07T21:30:00+00:00)
1 hour to kickoff: With lineups still yet to be released, let's take a look at one player who is, in fact, an MVP candidate for this year.

FC Cincinnati forward Luciano Acosta is one back of Denis Bouanga for the MLS goal scoring lead with two matches left, sitting on 16 for the year. With 13
assists alongside that, his 29 total goal contributions are best in the league, above the 27 of Thiago Almada and 25 of Hany Mukhtar. Acosta ranks as the
second-highest rated player by WhoScored this season across all of Major League Soccer.

You wonder if he's going to be rested tonight with the Supporters' Shield and No. 1 seed locked up. It would be devastating for him to be injured in a
meaningless game, regardless of the opponent.

"Luciano Acosta's the MVP... He's been an unbelievable player for us this year."
</item 6>

<item 7>
Browns survive Bears Hail Mary on wild, near-catch by Darnell Mooney: 'Damn near s— myself' (Sporting News, 2023-12-17T23:06:30+00:00)
"And what a nice play by T Scott to put us in position and, again, we were just too far back. We ended up throwing the Hail Mary at the end and we had a chance
there too, that's just about focus and concentration at the very end and making a really good play."

Chicago had plenty of other issues that contributed to coming up short to the Browns. The Bears' only touchdowns came when an interception spotted Chicago at
the Browns' 1-yard line and another on a pick-six. The offense went three-and-out eight times. And rather than try for a go-ahead, 54-yard field goal, the Bears
opted for a Hail Mary from the Browns' 37 with no time left in the first half, leading to Fields' first interception. The only two Chicago turnovers came on
Fields' Hail Mary tries.
</item 7>

<item 8>
NFL roundup: Pats hit new low as Lions reach 9-3 for first time since 1962 (The Guardian, 2023-12-03T22:11:26+00:00)
Atlanta entered tied atop the division with New Orleans, who lost to Detroit.

But this game appealed only to fans of sloppy play, punting and penalty flags. The Jets (4-8) got an early safety but couldn’t get anything going – again – on
offense in losing their fifth straight game. New York even pulled quarterback Tim Boyle and replaced him with Trevor Siemian in the fourth quarter to try to
provide a spark, but it was more of the same misery in what appeared a half-filled MetLife Stadium.

Cleveland Browns 19–36 Los Angeles Rams

Matthew Stafford threw three touchdown passes, Puka Nacua became the first Rams rookie wide receiver to reach 1,000 yards in a season, and Los Angeles extended
their winning streak to three games. The Rams have won three straight for the first time since winning the Super Bowl after the 2021 season. At 6-6, they are
back in the playoff race after dropping three straight before their bye week.
</item 8>

<item 9>
Jets vs. Raiders live score, updates, highlights from NFL 'Sunday Night Football' game (Sporting News, 2023-11-12T23:30:03+00:00)
With five seconds left, the Raiders call their second timeout.

11:22 p.m.: Wilson spikes the first pass with no one open, then connects with Conklin on a dump-off for nine yards. He goes back to Conklin with the clock
rolling, and the tight end makes an impressive leaping grab at the Raiders' 45. The Jets use their last timeout with 12 seconds left.

A big pass to Tyler Conklin gets the Jets into opposing territory



📺: #NYJvsLV on NBC

📱: Stream on #NFLPlus https://t.co/lfCP1NFAnO pic.twitter.com/oW0Nb8335s — NFL (@NFL) November 13, 2023

11:20 p.m.: Even after backing up five yards, Cole still boots the ball too far. He tries to angle it toward the sideline, and it bounces into the end zone for
a touchback.
</item 9>

<item 10>
Chiefs vs. Packers live score, updates, highlights from NFL 'Sunday Night Football' game (Sporting News, 2023-12-03T23:00:04+00:00)
Packers 27, Chiefs 19

11:12 p.m.: Love takes a sack 10 yards behind the line-of-scrimmage, setting up a rather difficult 48-yard attempt for Carlson on fourth down. The Packers
desperately need their rookie kicker to come through in the clutch.

11:11 p.m.: Watson goes down with yet another injury. This one looked like it could be a hamstring niggle. Really unfortunate end to the game for the
second-year wideout, who was enjoying one of the best games of his young career.

10:59 p.m.: INTERCEPTION! It looked like the Chiefs were on the brink of another lengthy drive. At least until Mahomes tossed a pass toward Packers corner
Keisean Nixon. The noted return man pulled down an over-the-shoulder grab to break Missourians' hearts. What a grab. What a moment!

10:53 p.m.: FIELD GOAL!
</item 10>

<item 11>
Colorado vs. Arizona State live score, updates, highlights from Week 6 college football game (Sporting News, 2023-10-07T21:21:03+00:00)
9:35 p.m. – That's a huge run for the Buffaloes! Weaver returns it 51 yards, only stopping after tripping himself up. Colorado is 25 yards away from another
touchdown.

Colorado 21, Arizona State 17

9:26 p.m. TOUCHDOWN – Colorado pulls ahead! Sanders finds Javon Antonio for a 9-yard touchdown, his first of the year. It's now a four-point lead for Colorado
with 14 minutes to go.

Buffs are up thanks to @7asavage



CU 21, ASU 17



📺 Pac-12 Network pic.twitter.com/3R8w2Qdm9e — Colorado Buffaloes Football (@CUBuffsFootball) October 8, 2023

End of third quarter: Colorado 14, Arizona State 17

9:13 p.m. PENALTY – Sun Devils Ed Woods is ejected for targeting.
</item 11>

<item 12>
Vikings vs. Broncos live score, updates, highlights from NFL 'Sunday Night Football' game (Sporting News, 2023-11-19T23:00:04+00:00)
Broncos 3, Vikings 0

8:30 p.m.: FIELD GOAL — Will Lutz buries a 31-yarder to get the Broncos on the board. The Vikings' defense held strong after the turnover. LB Danielle Hunter
came up with a timely sack to force a 3rd & 15 and the Broncos couldn't convert.

8:26 p.m.: FUMBLE! Vikings QB Joshua Dobbs attempted to pick up a first down on 3rd & 1 and he was absolutely popped by SS Kareem Jackson. LB Baron Browning
pried the ball loose as Dobbs was hit and the Broncos recovered the fumble, taking over at Minnesota's 30.

8:24 p.m.: The Vikings received to start the game and after a touchback on the opening kick, we are underway!

Pregame

8:21 p.m.: Broncos legend Peyton Manning shows some love to Vikings star WR Justin Jefferson ahead of the game.
</item 12>

<item 13>
Colorado vs. UCLA live score, updates, highlights from Week 9 college football game (Sporting News, 2023-10-28T22:08:03+00:00)
ET (4:30 p.m. PT) from the Rose Bowl Stadium in California.

What channel is Colorado vs. UCLA on today

TV channel : ABC

: ABC Live stream: ESPN, Fubo

This primetime Pac-12 game can be found on ABC with it's top-team of Chris Fowler and Kirk Herbstreit in the booth.

For cord-cutters, live streaming options are available on the ESPN app or on Fubo, which offers a free trial.
</item 13>

<item 14>
Vikings vs. Broncos live score, updates, highlights from NFL 'Sunday Night Football' game (Sporting News, 2023-11-19T23:00:04+00:00)
9:21 p.m.: Joshua Dobbs and center Garrett Bradbury have had a few bad exchanges when the newly acquired QB is under center. Dobbs just fumbled his second snap
of the game but has recovered both of them.

Vikings 10, Broncos 6

9:18 p.m.: FIELD GOAL — Will Lutz gets another 31-yarder to go for the Broncos. Russell Wilson found Jerry Jeudy for a five-yard gain on 3rd & 7, coming just
short of the line to gain. Head coach Sean Payton elected to kick the field goal instead of going for it on Minnesota's 12-yard line.

9:14 p.m.: Russell Wilson and Courtland Sutton connect on a 33-yard pitch and catch to give the Broncos their first big play of the game! Denver is in the red
zone for the first time tonight.

Vikings 10, Broncos 3

9:08 p.m.
</item 14>

<item 15>
Jaguars vs. Saints live score, updates, highlights from NFL 'Thursday Night Football' game (Sporting News, 2023-10-19T22:00:04+00:00)
It was Paulson Adebo that punched the ball out. New Orleans gets the ball right back at their own 41-yard line.

8:33 p.m. — Missed field goal! Grupe pushes his attempt to the right, and the Saints come up with nothing on their opening drive. The rookie has now missed
three of his last five kicks.

8:33 p.m. — Jacksonville blitzes on third-and-seven and the pressure forces Carr to throw the ball away. New Orleans will have to settle for a 51-yard field
goal attempt by rookie Blake Grupe.

8:29 p.m. — Carr uses his legs to pick up another first down and get across midfield. The Saints QB certainly isn't known for scrambling, but he gets the job
done by rolling out to his right and rushing for a gain of 10 yards.

8:28 p.m.
</item 15>

<item 16>
Dolphins vs. Eagles live score, updates, highlights from NFL 'Sunday Night Football' game (Sporting News, 2023-10-22T23:28:04+00:00)
It'll be a field goal try for Jake Elliott.

8:30 p.m. — Jalen Hurts is celebrating National Tight Ends Day by hitting Dallas Goedert on back-to-back passes for a total of 45 yards. The Eagles are into the
red zone.

Fitting that the first big play of the game goes to Dallas Goedert on #NationalTightEndsDay



📺: #MIAvsPHI on NBC

📱: Stream on #NFLPlus https://t.co/0rIJPFKW6L pic.twitter.com/DPfTTVQbnz — NFL (@NFL) October 23, 2023

8:28 p.m. — Swift has five touches already. No one else has seen the ball.

8:26 p.m. — The Eagles are riding D'Andre Swift early, as the running back is up to three carries through three snaps. He gets Philadelphia a first down.

8:23 p.m.
</item 16>

<item 17>
Chicago Fire vs Inter Miami live score, result, updates, highlights from Lionel Messi in MLS match (Sporting News, 2023-10-04T23:33:00+00:00)
Chicago are into a playoff position, while Inter Miami's playoff chances take a massive hit with the defeat. Lionel Messi's club no longer controls its own
destiny, needing outside help to make the postseason.

Via @IanHest, #InterMiamiCF would be on the brink of playoff elimination if this result holds. pic.twitter.com/xqyyY5ifzP — Kyle Bonn (@the_bonnfire) October 5,
2023

Chicago Fire vs. Inter Miami: Second Half

90th min: Facundo Farias is booked for a hip check on Miguel Navarro, which might have been a retaliation foul for Navarro's jawing at Sergio Busquets.

Three minutes of added time are shown on the board.

83rd min: Referee Joe Dickerson has to step between Sergio Busquets and Miguel Navarro as the two are jawing at each other. It wasn't hard to read Navarro's
lips as he walked away, and his words aren't repeatable here.
</item 17>

<item 18>
Browns survive Bears Hail Mary on wild, near-catch by Darnell Mooney: 'Damn near s— myself' (Sporting News, 2023-12-17T23:06:30+00:00)
The Bears' prayers were nearly answered on a game-ending Hail Mary. Instead, Cleveland fans exhaled as their team came away with an improbable win against
Chicago.

Justin Fields and the Bears had the ball at the Cleveland 45 trailing by three with five seconds left on the clock. There was no time for a play and a field
goal without any timeouts to try for a shorter kick.

Chicago's only hope of winning the game — and keeping its surging playoff trending up — would rest on a Hail Mary, a play so dubbed because it takes a miracle
to succeed.

The Bears nearly had their miracle. The ball had the distance and it was tipped in the air several times and began to fall into the arms of Bears receiver
Darnell Mooney, who was falling toward the endzone turf. He had his hands on it as it began to fall into his chest.
</item 18>

<item 19>
Chicago Fire vs Inter Miami live score, result, updates, highlights from Lionel Messi in MLS match (Sporting News, 2023-10-04T23:33:00+00:00)
That was nearly a big chance for the hosts.

Kamal Miller has been exceptional so far tonight.#InterMiamiCF — Alex Windley (@aaw_1998) October 5, 2023

23rd min: Chance, Inter Miami! Noah Allen nearly produces the vertical ball of the century, slicing up the Fire defense to find DeAndre Yedlin streaking at the
far post, and while the American international slides to keep the ball in play, which Chris Brady spills, Josef Martinez can't quite reach the loose ball before
it's cleared. They were so close to slamming that home!

21st min: The Fire nearly create another chance, with two players combining down the left to beat DeAndre, but the cross to the feet of a leaping Fabian Herbers
is put wide of the post. The German attempted a leaping karate kick but couldn't get the contact right.

15th min: Chance, Chicago Fire! HOW DID THE FIRE NOT SCORE?!?
</item 19>

<item 20>
Bears vs. Panthers live score, updates, highlights from NFL 'Thursday Night Football' game (Sporting News, 2023-11-09T23:04:33+00:00)
Former Bears kicker Eddy Pineiro drills the field goal, so Carolina still adds to its lead.

End of first quarter: Panthers 7, Bears 3

8:50 p.m. — Young throws downfield and hits Mike Strachan for a 44-yard gain. Mike Strachan? Mike Strachan. The Panthers are suddenly in business in Bears
territory.

Mike Strachan's first catch as a Panther goes for 45 yards!#CARvsCHI on Prime Video

Also available on #NFLPlus pic.twitter.com/15stE4IGAz — NFL (@NFL) November 10, 2023

Panthers 7, Bears 3

8:45 p.m. — Santos drills it anyway! A 54-yard kick. The Bears cut the deficit to 7-3.

8:44 p.m.
</item 20>

<item 21>
Inter Miami vs FC Cincinnati live score, result, updates, highlights with Lionel Messi's MLS playoffs on line (Sporting News, 2023-10-07T21:30:00+00:00)
So close again for the hosts!

22' | Josef takes a chance and it hits the post🤯#MIAvCIN | 0-0 pic.twitter.com/BbD0dWKnzP — Inter Miami CF (@InterMiamiCF) October 8, 2023

20th min: Josef Martinez's hold-up play creates a chance for a charging Benja Cremaschi, but he puts his driven effort wide.

14th min: Inter Miami have a very dangerous free-kick as Josef Martinez is bundled over from behind by Junior Moreno. The ball is shaded right of the middle
just inches outside the penalty area, and it'll be Facundo Farias to take with his fellow Argentine Lionel Messi on the bench. The floated delivery find the
head of a teammate, but the shot is weak and easily caught by Roman Celentano.

12th min: Chance, FC Cincinnati! WHAT A SAVE DRAKE CALLENDER!
</item 21>

<item 22>
NFL Fantasy Football Week 6 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-12T21:23:16+00:00)
But the biggest detriment is that he can't consistently overcome bad blocking, which has been a consistent problem for the Denver O-line all year and certainly
something that impacted Javonte Williams in his first handful of games.

WHEN TRAILING: When the Jets were up by eight or more points in the second half, McLauglhin played 4 of 15 snaps and had two carries. That's in line with a
trend we've seen from Denver all year: Of the 117 snaps they've played when down by eight or more points, McLaughlin has played 26 snaps (13 touches, seven at
Chicago), Williams has played 27 snaps (15 touches, almost all against Miami) and Samaje Perine has played 58 snaps (13 touches) -- more than both backs
combined. The only guy who's averaged a strong number on the ground in this specific situation? You guessed it -- McLaughlin (6.1 yards per carry).
</item 22>

<item 23>
Prisco's Week 12 NFL picks: Jaguars beat Texans in AFC South battle, Saints get key divisional win vs. Falcons (CBSSports.com, 2023-11-23T21:30:38+00:00)
Kyler Murray has played well in his two starts, which I think will continue here. The Cardinals will slow down Matt Stafford and gang to win it.

Pick: Cardinals 27, Rams 23

Sunday, 4:05 p.m. ET (Fox, fubo)

This is suddenly a big game for both teams. The Broncos have won four straight, while the Browns are 7-3 and would be in the playoffs right now. This will be
the first road start for Browns rookie quarterback Dorian Thompson-Robinson, which won't be easy. But the Cleveland defense is dominant. They will shut down the
Denver offense in a low-scoring game to win it.

Pick: Browns 20, Broncos 12

Sunday, 4:25 p.m. ET (CBS, Paramount+)

The Chiefs are coming off a tough Monday night loss and now must travel on a short week.
</item 23>

<item 24>
Colorado vs. Arizona State live score, updates, highlights from Week 6 college football game (Sporting News, 2023-10-07T21:21:03+00:00)
Colorado fans, you're back. For now.

A 43-yard pass from Shedeur Sanders on the Buffaloes' final drive helped set up Alejandro Mata for the game-winning 43-yard field goal with 12 seconds left,
delivering Colorado the 27-24 win over Arizona State for their first Pac-12 win of the season. It snaps their eight-game conference losing streak.

Sanders had a mediocre day by his lofty standards, but his 239 yards and a touchdown were enough in perhaps Colorado's most crucial game so far. It brought the
team to 4-2, but had it lost, it would have been much harder for the Buffs to finish the season bowl eligible. A win gives them a leg up in a competitive
conference, particularly facing four ranked teams down the stretch of the season.

If Colorado wants to solidify a bowl game, it'll have to fix many things, one of them being the offensive line.
</item 24>

<item 25>
NFL roundup: Pats hit new low as Lions reach 9-3 for first time since 1962 (The Guardian, 2023-12-03T22:11:26+00:00)
Wilson was intercepted a season-high three times, all of them after halftime. Derek Stingley Jr had the first two picks for the Texans. Rookie CJ Stroud threw
for 274 yards for Houston.

Indianapolis Colts 31-28 Tennessee Titans

Gardner Minshew threw a four-yard touchdown pass to Michael Pittman with 2:31 left in overtime and the Indianapolis Colts (7-5) beat the Tennessee Titans (4-8)
for their fourth straight victory. The Colts had to score a touchdown after Nick Folk made a 46-yard field goal with 4:19 left in overtime to put Tennessee up.
Tennessee lost their first true home game this season and have yet to win consecutive games. The Titans blew a 17-7 lead.
</item 25>

<item 26>
Inter Miami vs FC Cincinnati live score, result, updates, highlights with Lionel Messi's MLS playoffs on line (Sporting News, 2023-10-07T21:30:00+00:00)
Inter Miami fans chant "Mess, Messi" as they have a free kick! pic.twitter.com/VkyvULPUHp — Mundo Albiceleste ⭐🌟⭐🇦🇷 (@MundoAlbicelest) October 8, 2023

24th min: Chance, Inter Miami! THEY'VE HIT THE WOODWORK AGAIN! Benja Cremaschi is fed through on a one-two with Josef Martinez, and with the outside of his
right foot, he bends it around Matt Miazga but clatters the near post. The THIRD time Inter Miami have hit the bar in under a half-hour! How unlucky??

22nd min: Chance, Inter Miami! Inter Miami hit the woodwork again! Josef Martinez, who has been electric so far, rips a shot from a tight angle about 15 yards
out, and while it beats Roman Celentano, it also clatters the far post and deflects out!
</item 26>

<item 27>
Jets vs. Raiders live score, updates, highlights from NFL 'Sunday Night Football' game (Sporting News, 2023-11-12T23:30:03+00:00)
If you tuned into "Sunday Night Football" expecting to see some offense, you tuned in on the wrong weekend.

The Jets and Raiders were locked into an old-school, defense-heavy primetime clash that featured nothing but six field goals through the first three quarters of
the game. It wasn't until 50 seconds into the fourth quarter that Las Vegas finally picked up the first touchdown, with Michael Mayer making a leaping grab in
the end zone to put the Raiders up 16-9.

In such a low-scoring game, that touchdown felt, even as early as it was, that it could be a game-winner. And indeed, it was as the Raiders held the Jets to
just a field goal the rest of the game and picked off Zach Wilson late to stymy New York in a 16-12 win.

MORE: What to know about Raiders 'SNF' hero Robert Spillane

It wasn't exactly a quarterback showdown that many might have expected in the preseason when this game appeared to be Aaron Rodgers vs.
</item 27>

<item 28>
UTSA vs. Marshall odds, props, predictions: Roadrunners lay big spread in Frisco Bowl (Sporting News, 2023-12-18T22:29:26+00:00)
Chase Harrison, QB (transfer portal)

Brandon McElroy, DL (transfer portal)

Caleb Coombs, WR (transfer portal)

Zion Odoms, WR (transfer portal)

Jayven Hall, LB (transfer portal)

Cam Fancher, QB (transfer portal)

Thomas Lane, DL (transfer portal)

Chinazo Obobi, OT (transfer portal)

Maurice Jones, RB (transfer portal)

Sean Sallis, TE (transfer portal)

Myles Bell, DB (transfer portal)

Trent Holler, iOL (transfer portal)

Joshua McTier, CB (transfer portal)

Jabarrek Hopkins, DL (transfer portal)

CK Obobi, OL (transfer portal)

Sean Ellis, TE (transfer portal)

TyShawn Hurst, OT (transfer portal)

Elijah Russell, EDGE (transfer portal)

UTSA vs.
</item 28>

<item 29>
Colorado vs. Arizona State live score, updates, highlights from Week 6 college football game (Sporting News, 2023-10-07T21:21:03+00:00)
On 4th & goal Xavier Weaver takes it in for @CUBuffsFootball 🦬



This game is tied 💪 pic.twitter.com/bGwGZ7b0IF — FOX College Football (@CFBONFOX) October 7, 2023

7:06 p.m. – Anthony Hankerson gets two first downs for the Buffs on a four-yard run and a 12-yard run.

7:00 p.m. – It was a three-and-out for Colorado, and now it's a three-and-out for Arizona State.

6:54 p.m. – Intentional grounding is called on Shedeur Sanders on first down.

Colorado 0, Arizona State 7

6:47 p.m. TOUCHDOWN – The Sun Devils start quick, scoring a touchdown on its first drive behind Elijhah Badger's 48 yards. Its 7-0 just a few minutes in.

6:35 p.m.
</item 29>

<item 30>
Bears vs. Panthers live score, updates, highlights from NFL 'Thursday Night Football' game (Sporting News, 2023-11-09T23:04:33+00:00)
After Eddy Pineiro cut Carolina's deficit to three early in the fourth quarter, Frank Reich put his trust in Pineiro again with the game on the line, sending
him out for a 59-yard attempt on fourth-and-10 in the final minutes rather than trusting the offense to pick up the 10 yards. Pineiro's kick was short.

While the decision to kick from 59 yards out might be second-guessed, Carolina's offense didn't give Reich much reason to believe. Bryce Young struggled to
develop any kind of rhythm through the first three quarters, and even the Panthers' most successful drives were laborious, consistently requiring conversions on
third and fourth down.

While Young got little help from his receivers, who failed to get open all night, his struggles — including multiple passes that were nearly intercepted — are a
reminder that the Panthers' offense is still a long way from resembling anything like the unit owner David Tepper envisioned when the franchise traded for the
No. 1 pick in March.
</item 30>

<item 31>
Colorado vs. UCLA live score, updates, highlights from Week 9 college football game (Sporting News, 2023-10-28T22:08:03+00:00)
UCLA topped Colorado 28-16 at the Rose Bowl in Deion Sanders and Chip Kelly's only Pac-12 matchup.

The Bruins improved to 6-2, clinching bowl eligibility and keeping them alive in the Pac-12 race with a 3-2 league mark.

In his third start of the season, Bruins' QB Ethan Garbers passed for 269 yards and two touchdowns. He and Collin Schlee combined for two interceptions in
UCLA's first-half turnover-fest. It had four, including two fumbles by Carson Steele. Luckily for Kelly, his team fixed the issue and did not have any turnovers
in the second-half.

Colorado, on the other hand, could not figure out how to protect QB Shedeur Sanders, who tossed for 217 yards and one touchdown. He was sacked seven times,
tying a season-high.

It didn't help that the Colorado running game was absent in the second half.
</item 31>

<item 32>
NFL Fantasy Football Week 6 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-12T21:23:16+00:00)
LAST WEEK: Despite splitting first-half snaps almost evenly and losing out on second-half snaps considerably, McLaughlin made three splash plays of 21, 22 and
38 yards including a touchdown on a tipped-then-secured designed screen that pierced through the middle of the Jets defense. Those three touches were his best
-- his other nine touches each failed to exceed 3 yards. And as mentioned two sentences ago, he played just 10 snaps in the second half when the Broncos were
trailing.

FILM: McLaughlin has good vision and speed, both of which were essential in reading the Jets defense at the first and second levels and choosing the correct
paths to run on his big plays. His lateral agility was a differentiator on his touchdown catch. McLaughlin's worst asset is his power as he just won't finish
runs with physicality like his backfield mates do.
</item 32>

<item 33>
NFL Best Bets Week 9: Colts clobber Panthers, Dolphins drop Chiefs in Germany (Sporting News, 2023-11-03T20:56:00+00:00)
The past few weeks of the 2023 NFL season have reminded us that no matter how smooth you sail to start the voyage, choppy waters will surely come at some point.
We started the first six weeks with a best bets winning percentage of well over .700, but we have come crashing back down to Earth over the past two weeks. We
love our Week 9 best bets, though, and we're confident that we can all get back on track and make you some cash.

Every Friday afternoon, we unveil my favorite moneyline, spread, over/under, and player prop wagers for the upcoming weekend. We run the gauntlet and handpick
the best value opportunities on Caesars and SuperDraft, then tell you exactly how and why you should attack the books to make some money. And through eight
weeks, despite some bumps in the road, we have made plenty of money.
</item 33>

<item 34>
Jaguars vs. Saints live score, updates, highlights from NFL 'Thursday Night Football' game (Sporting News, 2023-10-19T22:00:04+00:00)
Foye Oluokun makes a house call off the deflection ‼️ #JAXvsNO on Prime Video

Also available on #NFLPlus https://t.co/89nUeCngbn pic.twitter.com/6dwZB25nxD — NFL (@NFL) October 20, 2023

10:12 p.m. — An interesting play call for the Jaguars on third down just now. Needing six yards at their own 29, the call from the sidelines is a draw, and
Etienne Jr. comes up less than a yard short of a first down. Pederson elects not to go for it, and the Jaguars will punt away. Jacksonville is playing a
dangerous game, letting New Orleans stick around.

Jaguars 17, Saints 9

10:06 p.m. — Field goal! Grupes improves to three of four on the night, booting a 42-yarder.
</item 34>

<item 35>
Inter Miami vs FC Cincinnati live score, result, updates, highlights with Lionel Messi's MLS playoffs on line (Sporting News, 2023-10-07T21:30:00+00:00)
Messi, who had missed the last four games due to injury, returned to play 35 minutes off the bench, but by the time he stepped on the pitch, Inter Miami’s early
dominance in the match had come and gone. Messi played a part, but he was unable to fully change the tide of the game.

Eventually, Alvaro Barreal’s 78th-minute goal ended any hope that Inter Miami had to complete a stunning playoff charge, and the defeat coupled with CF
Montreal’s big 4-1 victory over Portland means Inter Miami are officially out of the postseason running.

The Sporting News followed the Inter Miami vs. FC Cincinnati match live, providing score updates, commentary and highlights as they happened.

MORE: Inter Miami playoff scenarios before potential do-or-die match against FC Cincinnati

Inter Miami vs FC Cincinnati final score

Score Goal scorers Inter Miami 0 – FC Cincinnati 1 Barreal (78')

Kickoff: 7:39 p.m.
</item 35>

<item 36>
Patriots vs. Broncos live score, updates, highlights from NFL Sunday night football game (Sporting News, 2023-12-24T23:00:03+00:00)
: Elliott takes it up the middle to the Denver 28

Broncos 7, Patriots 3

09:35 p.m.: Lutz's 57-yard field goal attempt is no good for the Broncos as it comes up short. That ends a first half filled with missed opportunities for both
teams in what's been a low-scoring affair to this point.

09:30 p.m.: Ryland's field goal try from 47 yards is no good as it sails wide right.

09:25 p.m.: Broncos wide receiver Courtland Sutton was evaluated for a possible concussion. The team officially announces it as him being evaluated for a
"possible head injury."

09:12 p.m.: McLaughlin takes it to the Denver 37 for a gain of 7 yards as the Patriots look to get within range to score the second time of the evening.

8:54 p.m.: The Patriots put their first points on the board with a 33-yard field goal from Chad Ryland.
</item 36>

<item 37>
Inter Miami vs FC Cincinnati live score, result, updates, highlights with Lionel Messi's MLS playoffs on line (Sporting News, 2023-10-07T21:30:00+00:00)
HALFTIME: Inter Miami 0-0 FC Cincinnati

Needing victory in this match to ensure they remain in the MLS playoff picture, Inter Miami completely dominated the confirmed 2023 regular season champs but
somehow did not manage to capitalize on their overwhelming pressure. Inter Miami were so unlucky not to cash in on multiple occasions, hitting the woodwork an
insane three times, all in the first 30 minutes. Lionel Messi may yet have a job to do tonight, currently available off the bench.

FC Cincinnati had one really good look that they missed as well but otherwise were on their heels the entire 45 minutes. Head coach Pat Noonan tells the
broadcast he feels FC Cincinnati had "a lack of courage on the ball" that half, "as bad as its been all year."

Elsewhere, Inter Miami will be nervous about other results from around the league.
</item 37>

<item 38>
Inter Miami vs FC Cincinnati live score, result, updates, highlights with Lionel Messi's MLS playoffs on line (Sporting News, 2023-10-07T21:30:00+00:00)
WHAT A SAVE DRAKE CALLENDER! Cincinnati hit on the counter again, as Brandon Vazquez brilliantly settles a ball from deep and finds Sergio Santos. The
Cincinnati forward is tackled by Tomas Aviles as he gets his shot off, and that may have put him off just enough to allow Callender to make a diving,
full-stretch stop.

End-to-end stuff so far!

9th min: Chance, Inter Miami! Oh no, Tomas Aviles has missed with the goal gaping! On a corner, the initial header drops in Aviles's lap, and he produces a
finish of real quality, whipping his leg horizontally at the bouncing ball up near his hip, but he crashes the shot off the crossbar! It had serious pace and
should have bulged the net!

Chance, FC Cincinnati! Just seconds after Aviles hit the woodwork, FC Cincinnati counter with real venom, and Sergio Santos puts his chance wide!
</item 38>

<item 39>
Chicago Fire vs Inter Miami live score, result, updates, highlights from Lionel Messi in MLS match (Sporting News, 2023-10-04T23:33:00+00:00)
"I just don't have them."



🤔🧐🤨#InterMiamiCF #Messi𓃵 pic.twitter.com/U7doZYX491 — Franco Panizo (@FrancoPanizo) October 4, 2023

Chicago Fire vs. Inter Miami lineups & team news

Frank Klopas has rotated his two young attackers, Brian Gutierrez and Jairo Torres along with famed Swiss international Xherdan Shaqiri in the attack. He has
also done his best to keep his striker stop rolling along with Kei Kamara and Georgios Koutsias alternating up front and MLS veteran Kacper Przybylko coming off
the bench.

Injuries to defenders Mauricio Pineda and Miguel Navarro, plus striker Victor Bezerra, will keep them out for today's match, while Chris Mueller is done for the
year with a hip problem.
</item 39>

<item 40>
Colorado vs. Arizona State live score, updates, highlights from Week 6 college football game (Sporting News, 2023-10-07T21:21:03+00:00)
PENALTY – Sun Devils Ed Woods is ejected for targeting. He will miss the remainder of this game as well as the first half against Washington in two weeks.

9:10 p.m. – That's the fourth sack allowed by the Colorado O-line. The Buffaloes now lead FBS with most sacks allowed this season at 30.

8:52 p.m. – Skattebo is heating up under the sun. He's added 24 rushing yards from Arizona State's possession.

End of second quarter: Colorado 14, Arizona Sate 17

Colorado 14, Arizona State 17

8:20 p.m. FIELD GOAL – Dario Longhetto nails a 33-yard field goal to end the half. The Sun Devils have a narrow lead over the Buffs at the halfway mark.

Colorado 14, Arizona State 14

8:14 p.m. TOUCHDOWN – Sanders ties it up! The QB keeps it for 16 yards and a touchdown.
</item 40>

<item 41>
Inter Miami vs FC Cincinnati live score, result, updates, highlights with Lionel Messi's MLS playoffs on line (Sporting News, 2023-10-07T21:30:00+00:00)
Álvaro Barreal off the rebound to give @fccincinnati the lead late against Inter Miami. pic.twitter.com/DpZEQ2O01H — Major League Soccer (@MLS) October 8, 2023

76th min: Two more changes for Inter Miami, as Josef Martinez is curiously withdrawn, replaced by Nick Stefanelli. Robert Taylor also enters for DeAndre Yedlin.
They have to win to stay alive in the MLS playoff race.

67th min: Chance, FC Cincinnati! ANOTHER BONKERS SAVE BY DRAKE CALLENDER! HE'S UNBEATABLE TONIGHT! Aaron Boupendza crosses to Brandon Vazquez on the counter,
and while the latter is clean through on goal, Callender comes across his goal to make a sensational stop!

The flag goes up, meaning it won't count as a save, but the replay looked incredibly close, so Callender's stop means that VAR does not have a chance to
intervene.
</item 41>

<item 42>
Gio Reyna looks good in start for USMNT vs. Germany, but those involved in the finish are punished in 3-1 defeat (Sporting News, 2023-10-14T23:41:07+00:00)
“I think he played a good game, showed his quality. I think everyone in the first half played well. So it was good to see him, good to have him on the field.”

It was not good to not have him on the field, because the collapse from a 1-1 halftime tie not only followed Reyna’s departure, but probably was connected. The
U.S. was able to threaten the opposing goal enough with Reyna on the field to cause the opposition to concern itself with defending more often than it
preferred. Replacement Luca de la Torre dulled the attack and did nothing to dissuade Germany’s rampage toward the American net.

There was plenty of blame to be shared on the goals by Niclas Fullkrug in the 58th minute and Jamal Musiala three minutes later.
</item 42>

<item 43>
Vikings vs. Broncos live score, updates, highlights from NFL 'Sunday Night Football' game (Sporting News, 2023-11-19T23:00:04+00:00)
The Broncos completed a major comeback in the Mile High City.

Both Denver and Minnesota entered "Sunday Night Football" on red-hot win streaks but only one team could keep the momentum rolling. It looked like it was going
to be Joshua Dobbs and the Vikings for the majority of the contest, but the Broncos wouldn't let up.

Trailing by eight points going into the final frame, Denver's offense finally woke up in the game's biggest moment. The Broncos hadn't found the end zone all
night but Russell Wilson still found a way to lead an impressive game-winning drive in the final three minutes.

Courtland Sutton was as clutch as can be, coming up with a key catch on fourth down to keep the drive alive. When Denver brought the ball into the red zone
where it had stalled out several times throughout the night, Wilson went right back to his guy Sutton.

Sutton hauled in a monster touchdown reception to give the Broncos a one-point lead with a minute remaining, and Dobbs had run out of magic.
</item 43>

<item 44>
NFL Fantasy Football Week 6 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-12T21:23:16+00:00)
It's a far cry from Week 1.

MINSHEW: Clearly more of a developed downfield thrower than Anthony Richardson and has thrown at Pittman on 26.4% of his routes run including four red-zone
targets and an end-zone target. Even last week when Minshew played less than a half Pittman saw a target once every five routes he ran with Minshew.

JAGUARS: Rank fifth with a 77.7% zone coverage rate, and they've consistently played a lot of zone in every game. They were also busted up by outside receivers
(over 17.0 yards per catch) last week against Buffalo and two weeks ago against the Texans. Pittman tends to play outside more than in the slot. Finally, an
outside receiver has scored at least once against the Jags in every game this season, including Pittman back in Week 1.
</item 44>

<item 45>
Fantasy Football Week 13 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-11-30T22:10:58+00:00)
Mark Andrews had a great game against Seattle, but even he didn't score. I suspect the Seahawks will have a plan to try and cover Ferguson near the end zone
without having to sacrifice too much elsewhere. He's a low-upside play I'd sit for Juwan Johnson, Gerald Everett and Logan Thomas.

Start 'em 10.0-7.00 6.99 - 5.01 5.00 - 0 Sit 'em

Seven straight quarterbacks to face the Broncos have failed to get to 20 Fantasy points, including Josh Allen once and Patrick Mahomes twice! Denver's forced at
least one turnover from a quarterback in eight straight games and kept quarterbacks to under 7.0 yards per attempt in five straight. Denver also seems to be
playing a lot of heavy zone coverage to deter quarterbacks from throwing deep, mostly Cover-3, and they're also starting to blitz more over the last month than
they did before.
</item 45>

<item 46>
N.F.L. Executives on the Biggest Questions Facing Super Bowl Contenders (The New York Times, 2023-10-16T23:58:53+00:00)
Three of the seven teams passed, resulting in one first down, one sack and Hurts’ interception. Of these three teams, only the Eagles lost the game.

3. The 49ers lost Sunday when their rookie kicker missed from 41 yards. Is San Francisco wise trusting a championship roster to a rookie’s foot come playoff
time?

The 49ers raised eyebrows among some when they used their second-highest pick in the 2023 draft — a third-round choice, 99th overall — for a kicker. The
selection of Michigan’s Jake Moody gave the 49ers a cost-effective successor to veteran Robbie Gould. General manager John Lynch called Moody a potentially
“foundational” player whose stock in the draft was so high, other teams were trying to trade ahead of San Francisco for a shot at him.

Who do you want lining up the potential winning kick in a Super Bowl?
</item 46>

<item 47>
Bears vs. Panthers live score, updates, highlights from NFL 'Thursday Night Football' game (Sporting News, 2023-11-09T23:04:33+00:00)
The Bears will punt, giving the Panthers a chance to tie the game or more.

10:46 p.m. — Bagent finds Kmet for a first down and more. Another nice play by Kmet, who has become a trusted target for Bagent in these four starts for the
rookie.

Bears 16, Panthers 13

10:42 p.m. FIELD GOAL — Pineiro's kick is good, and the Panthers now trail by a field goal.

10:41 p.m. — Young's pass to Marshall is incomplete on third down. Marshall was well-defended, but he probably should've had that. It'll be a field goal try for
Carolina.

10:38 p.m. — You have to give some credit to Carolina's offense for converting, as Young finds Tremble for the first down on 4th & 6. The Panthers turn a 1st &
30 into a fresh set of downs.

10:37 p.m.
</item 47>

<item 48>
Inter Miami vs FC Cincinnati live score, result, updates, highlights with Lionel Messi's MLS playoffs on line (Sporting News, 2023-10-07T21:30:00+00:00)
@fccincinnati goalkeeper @roman_413 had nothing but praise for @LuchoAcosta94 and Pat Noonan 🟠🔵 pic.twitter.com/Xyp2ozr7n5 — CBS Sports Golazo ⚽️
(@CBSSportsGolazo) October 3, 2023

1 hour 15 mins to kick: Lionel Messi was nominated for not only MLS Newcomer of the Year for the 2023 season, but also 2023 MLS MVP.

Laughable doesn't even begin to describe that blatant PR move, considering that most of the damage Messi has done for Inter Miami has come in Cup competitions.
He has one goal and two assists in 247 minutes of play in actual MLS competition. Lionel Messi has been fun, but let's hope the league doesn't make a mockery of
itself by actually giving Messi either honor.
</item 48>

<item 49>
Chicago Fire vs Inter Miami live score, result, updates, highlights from Lionel Messi in MLS match (Sporting News, 2023-10-04T23:33:00+00:00)
Expect some attacking changes in the second half. #InterMiamiCF — Alex Windley (@aaw_1998) October 5, 2023

Around the rest of the league, the New York Red Bulls have picked up a massive three points off Eastern Conference leaders FC Cincinnati with a 2-1 win, while
CF Montreal grabbed a last-gasp equalizer against the Houston Dynamo, and Charlotte FC pummeled Toronto FC 3-0. Not exactly an ideal trio of results for others
chasing a playoff spot.

Chicago Fire vs. Inter Miami: First Half

42nd min: Sergio Busquets picks up the first yellow card of the match for what can only be described as a football tackle on Brian Gutierrez near the center
circle. Busquets has no argument there.

38th min: Chance, Chicago Fire! OFF THE INSIDE OF THE POST AND OUT! So close to an opener!
</item 49>

<item 50>
Colorado vs. UCLA live score, updates, highlights from Week 9 college football game (Sporting News, 2023-10-28T22:08:03+00:00)
Collin Schlee, who came in for Garbers, is interception by Hunter. Colorado takes over at its own 41-yard line with just over six minutes left in the half.

8:40 p.m. FUMBLE – The second turnover of the night goes to the Buffs again! Steele runs for 18-yards but lets go of the ball on the tackle. Carter Stoutmire
recovers it for the possession.

Shilo Sanders' HUGE hit forces the fumble 💪 pic.twitter.com/UaSXN4nl1q — Pac-12 Conference (@pac12) October 29, 2023

Colorado 6, UCLA 7

8:29 p.m. TOUCHDOWN – UCLA takes the lead! Garbers finds Carson Steele on a three-yard pass to put the Bruins on the board. They now lead 7-6 with 14:20 on the
clock in the second quarter.

GUTSY.
</item 50>

<item 51>
Browns survive Bears Hail Mary on wild, near-catch by Darnell Mooney: 'Damn near s— myself' (Sporting News, 2023-12-17T23:06:30+00:00)
"I damn near s— myself, dog," Njoku said, per The Athletic's Zac Jackson.

The Bears' sideline appeared convinced it had won. Fields said that as the ball began to fall into Mooney's hands, the Chicago sideline began to erupt before
realizing the ball would not be ending in Bears' hands.

Fields' counterpart, Joe Flacco, could only watch on the jumbotron as the play unfolded. He had just driven the Browns to the go-ahead score the drive before.
In his 16 years in the NFL, he's seen a lot, and he knew to hold off on any celebration until the game was official.

"When you watch and or are a part of all the games that go on every week, you know not to get excited until that clock hits zero," Flacco said.
</item 51>

<item 52>
Jets vs. Raiders live score, updates, highlights from NFL 'Sunday Night Football' game (Sporting News, 2023-11-12T23:30:03+00:00)
10:26 p.m.: For the second time today, O'Connell takes a bad sack out to the Jets 42 that takes the Raiders out of field-goal range, but Jacobs catches a short
pass and runs it up — ruled a carry since it was a lateral — for 12 yards to bring up a much more manageable third-and-6.

10:25 p.m.: Following a rush of no gain by Jacobs, Meyers makes a catch in front of a few defenders and makes them miss to pick up eight yards. O'Connell then
goes with a pop pass to Jacobs, who bursts to the right sideline for three yards and the first down to the Jets 34.

10:19 p.m.: Quick three-and-out by the Jets, as Wilson steps up and fires an incompletion and Hall is brought down for a loss of a yard by Butler.
</item 52>

<item 53>
Chiefs vs. Jets live score, updates, highlights from NFL 'Sunday Night Football' game (Sporting News, 2023-10-01T23:47:00+00:00)
Jets LB CJ Mosley picks off Patrick Mahomes on the first play of the drive. He was looking at TE Travis Kelce on a 35-yard pass but Mosley read it perfectly and
Mahomes underthrew his target. The Jets take over from their own 35 with two timeouts and 18 seconds on the clock.

9:46 p.m.: MISSED FIELD GOAL. Greg Zuerlein's 52-yard attempt hits off the upright to bring a disappointing end to the Jets' drive. Patrick Mahomes and the
Chiefs will have 26 seconds and two timeouts from their own 34 to try and make something happen before the end of the half.

9:43 p.m.: Zach Wilson is marching the troops! The Jets QB is putting together an impressive two-minute drill. He has led an eight-play, 41-yard drive to bring
the Jets to Kansas City's 34 yard line.

Chiefs 20, Jets 12

9:36 p.m.: FIELD GOAL.
</item 53>

<item 54>
NFL Best Bets Week 9: Colts clobber Panthers, Dolphins drop Chiefs in Germany (Sporting News, 2023-11-03T20:56:00+00:00)
MORE SUPERDRAFT DFS & FANTASY PROPS: How to play SuperDraft

This one should be a tightly-contested, high-scoring game — with heavy playoff implications, because we all know this could be an AFC Championship preview — so
expect Rice to be very busy through all four frames in Frankfurt. The Dolphins have allowed the fifth-most receptions and the 12th-most yards to wideouts this
season. If a bounce-back week is meant to be, Sir Rashee is our destiny.

Austin Ekeler, RB, Chargers @ Jets — OVER 29.5 receiving yards

Guess who's back? Back again...Ek is back....tell a friend. The Chargers finally got their best player back to full health, with Ekeler exploding for 123 yards
and a TD against the Bears last weekend. A whopping 94 of those 123 yards came through the air, as the PPR god gashed Chicago for chunk yardage a handful of
times.
</item 54>

<item 55>
Colorado vs. UCLA live score, updates, highlights from Week 9 college football game (Sporting News, 2023-10-28T22:08:03+00:00)
GUTSY.



After two QB scrambles, Ethan Garbers finds Carson Steele for the wide-open TD 🙌



UCLA 7, Colorado 6

14:19 – 2nd quarter



📺: ABC

📲: https://t.co/rTM1Y5TWC9#GoBruins pic.twitter.com/4ZzaQVuu1P — UCLA Football (@UCLAFootball) October 29, 2023

End of first quarter: Colorado 6, UCLA 0

8:22 p.m. – Garbers is sacked! Jordan Domineck takes down the Bruin to force 4th &11. But UCLA goes for it, and Garbers finds the first down himself on a
13-yard run.

Colorado 6, UCLA 0

8:05 p.m. – Colorado settles for a field goal again, this time for a 39-yarder.
</item 55>

<item 56>
Colorado vs. UCLA live score, updates, highlights from Week 9 college football game (Sporting News, 2023-10-28T22:08:03+00:00)
– Colorado settles for a field goal again, this time for a 39-yarder. It leads UCLA 6-0 with 7:44 remaining in the first quarter.

7:59 p.m. INTERCEPTION – Welcome to Hollywood, Travis Hunter! The two-way star picks off Ethan Garbers at the Bruins' 36-yard line. After review, the
interception stands, and it's Buffalo ball. Colorado defense with a statement play early on.

WHAT A PICK BY TRAVIS HUNTER 🔥@CUBuffsFootball pic.twitter.com/3i9twSG1uq — FOX College Football (@CFBONFOX) October 29, 2023

Colorado 3, UCLA 0

7:54 p.m. FIELD GOAL – Sanders can't find the end zone, but Alejandro Mata finds space between the uprights. His 31-yard field goal is good, and the Buffs take
the first lead of the game on their first drive.

7:52 p.m.
</item 56>

<item 57>
Colorado vs. Arizona State live score, updates, highlights from Week 6 college football game (Sporting News, 2023-10-07T21:21:03+00:00)
Colorado takes over on the ASU 25.

SHEDEUR TO ANTONIO FOR A HUGE GAIN 🦬@CUBuffsFootball pic.twitter.com/tvfWjRWWAE — FOX College Football (@CFBONFOX) October 8, 2023

Colorado 24, Arizona State 24

10:11 p.m. TOUCHDOWN – Don't count the Sun Devils out yet! Bourguet finds Troy Omeire on a 15-yard pass to tie it up. Only 50 seconds remain.

9:48 p.m. – It's blocked! Carter Stoutmire gets in front of the 52-yard field goal attempt from Longhetto to deny a Sun Devil score.

9:38 p.m. FIELD GOAL – Alejandro Mata nails a 42-yard field goal to extend Colorado's lead to seven. Ten minutes are left in the game.

9:35 p.m. – That's a huge run for the Buffaloes!
</item 57>

<item 58>
Chicago Fire vs Inter Miami live score, result, updates, highlights from Lionel Messi in MLS match (Sporting News, 2023-10-04T23:33:00+00:00)
🤔 pic.twitter.com/Svq79fvbBG — Major League Soccer (@MLS) October 4, 2023

1 hour to kickoff: With lineups not yet released, there is finally a report regarding Lionel Messi's availability. According to Michelle Kaufman of the Miami
Herald, Messi "did not arrive with the team" to Soldier Field. While that's not outright confirmation that he will not take part in the match, it's a very
strong indication that he will not be involved tonight.

1 hour 15 mins to kick: With lineups due imminently, and nothing to indicate if Lionel Messi is involved tonight or not, the Chicago Fire media team has jokes.
</item 58>

<item 59>
Vikings vs. Broncos live score, updates, highlights from NFL 'Sunday Night Football' game (Sporting News, 2023-11-19T23:00:04+00:00)
COURTLAND SUTTON INCREDIBLE 4TH DOWN GRAB ‼️



🎥: @NFL

pic.twitter.com/T0Mrch9Tf9 — The Sporting News (@sportingnews) November 20, 2023

Vikings 20, Broncos 15

11:02 p.m.: FIELD GOAL — Greg Joseph knocks it in from 30 yards. The Vikings' offense was moving, and it did its job by chewing a lot of clock on a 14-play,
seven-minute and 20-second drive. Minnesota stalled out on Denver's 12-yard line, settling for a field goal.

Can the Vikings' defense keep the Broncos out of the end zone as it has all night? The drive of the game is coming up.

10:57 p.m.: The spot on the field was overturned! Ty Chandler was ruled just short of the line to gain, setting up a 4th & inches.
</item 59>

<item 60>
Chicago Fire vs Inter Miami live score, result, updates, highlights from Lionel Messi in MLS match (Sporting News, 2023-10-04T23:33:00+00:00)
OFF THE INSIDE OF THE POST AND OUT! So close to an opener!

An exceptional run from Georgios Koutsias sees the Greek take a shot on goal which deflects off the foot of Serhiy Kryvtsov and slowly rolls towards the net.
The deflection wrong-footed Drake Callender, but the ball agonizingly hits the inside of the woodwork and comes out, and Inter Miami clear.

35th min: The hosts have gained a foothold on this match, and have earned a few consecutive corners. Chicago screams for a penalty as the ball hits Serhiy
Kryvtsov in the arm, but VAR does not intervene as his arm was down in a natural position. Four straight corners go begging.

29th min: Kamal Miller makes an exquisite block on Brian Gutierrez who appeared to be in, and had an eye for a shot to the far post. That was nearly a big
chance for the hosts.
</item 60>

<item 61>
Jaguars vs. Saints live score, updates, highlights from NFL 'Thursday Night Football' game (Sporting News, 2023-10-19T22:00:04+00:00)
The Saints get the ball to start the half, in desperate need of generating anything on offense.

End of first half: Jaguars 17, Saints 6

9:44 p.m. — Carr fumbles the ball, it's recovered by one of his lineman, and that's how the first half ends. A fitting end for the Saints, who could get nothing
going on offense, leaving the field to boos from the New Orleans fans.

Jaguars 17, Saints 6

9:42 p.m. — Field goal! McManus sinks his first attempt of the night, extending the Jaguars lead to 11 with just 32 seconds left in the half. Pederson's
aggression pays off, as the fake punt leads to three points.

9:41 p.m. — Call reversed! After review, the officials deem the ball hit the ground as Lattimore was trying to make the completion, ruling the play an
incomplete pass rather than an interception.
</item 61>

<item 62>
Colorado vs. Arizona State live score, updates, highlights from Week 6 college football game (Sporting News, 2023-10-07T21:21:03+00:00)
Arizona State

Date : Saturday, Oct. 7

: Saturday, Oct. 7 Time : 6:30 p.m. ET

: 6:30 p.m. ET Location : Mountain America Stadium in Tempe, Arizona

: Mountain America Stadium in Tempe, Arizona TV channel : Pac-12 Network

: Pac-12 Network Live stream: Pac-12 Network, Fubo

It'll be a sold out Mountain America Stadium in Tempe. Colorado-Arizona State kicks off at 6:30 p.m. ET on Saturday, Oct. 7. Viewing is available only on Pac-12
Network for those who have the cable extension. Streaming the game is available through Fubo, which offers a free trial.
</item 62>

<item 63>
Vikings vs. Broncos live score, updates, highlights from NFL 'Sunday Night Football' game (Sporting News, 2023-11-19T23:00:04+00:00)
10:42 p.m.: A 15-yard facemask penalty on Broncos WR Marvin Mims Jr. backed up the offense and the Vikings' defense held strong again as a result. Here comes
Will Lutz for his fifth field goal attempt of the game.

10:38 p.m.: INTERCEPTION! Joshua Dobbs was hit as he threw and Broncos CB Ja'Quan McMillan came up with the pick. It was LB Josey Jewell who brought the
pressure on Dobbs. Denver takes over from the Vikings' 9-yard line after McMillan picked up nine yards on the interception return.

Vikings 17, Broncos 12

10:34 p.m.: FIELD GOAL — Will Lutz buries his fourth field goal of the day, this time from 37 yards. Lutz has been automatic in this one, going a perfect
4-for-4 to account for all of Denver's points.

10:32 p.m.
</item 63>

<item 64>
N.F.L. Executives on the Biggest Questions Facing Super Bowl Contenders (The New York Times, 2023-10-16T23:58:53+00:00)
Who do you want lining up the potential winning kick in a Super Bowl? It’s an interesting question in a season that has seen Moody and Saints rookie Blake Grupe
miss game-winning tries, but the data could be on the 49ers’ side.

FGA Dist Rookies Veterans <33 Yds 97% [31-32] 94% [330-351] 34-44 Yds 75% [21-28] 82% [230-279] 45-50 Yds 78% [7-9] 73% [121-165] 51-55 Yds 100% [8-8] 59%
[33-56] 56+ Yds 0% [0-1] 55% [6-11] Totals 86% [67-78] 84% [720-862]

The table above shows field-goal make rates in the postseason since 2000 for rookie kickers vs. veterans.
</item 64>

<item 65>
Patriots vs. Broncos live score, updates, highlights from NFL Sunday night football game (Sporting News, 2023-12-24T23:00:03+00:00)
Broncos 7, Patriots 3

8:40 p.m.: The Broncos post the first score of the night with a 3-yard rush by Javonte Williams.

8:36 p.m.: Marvin Mims Jr. has been a huge playmaker this season and further proves that with a 52-yard return.

8:21 p.m.: The Broncos go for it on fourth down but come up short, so Zappe's fumble proves inconsequential.

8:19 p.m.: The two teams nearly traded turnovers as Russell Wilson appeared to be intercepted in the end zone but it is ruled an incomplete pass.

8:17 p.m.: Bailey Zappe pump fakes and fumbles the ball, immediately giving the Broncos offense a chance.

Pregame: Tyquan Thornton is aligned in the spot of Matthew Slater as a gunner on the punt coverage unit in warmups.
</item 65>

<item 66>
Colorado vs. UCLA live score, updates, highlights from Week 9 college football game (Sporting News, 2023-10-28T22:08:03+00:00)
UCLA records its sixth sack of the night with the latest.

Colorado 6, UCLA 14

9:40 p.m. TOUCHDOWN – Bruins bouncing back quick after halftime! Garbers finds Logan Loya and Moliki Matavao on two plays, and UCLA has another touchdown on the
board.

Halftime: Colorado 6, UCLA 7

9:15 p.m. FUMBLE – Turnover No. 4 for the Bruins. Steele fumbles on his run again, and it's recovered by Cam'Ron Silmon-Craig in Colorado territory. While the
Buffaloes have prevented more points UCLA, they've only scored three off turnovers. Twenty seconds remain in the half.

The ball pops out and @CUBuffsFootball has it! 🦬🔒 pic.twitter.com/tV7FH3moiu — FOX College Football (@CFBONFOX) October 29, 2023

9:07 p.m. – UCLA misses the field goal!
</item 66>

<item 67>
Chicago Fire vs Inter Miami live score, result, updates, highlights from Lionel Messi in MLS match (Sporting News, 2023-10-04T23:33:00+00:00)
15th min: Chance, Chicago Fire! HOW DID THE FIRE NOT SCORE?!? Drake Callender completely misreads the flight of a corner and undercuts the ball, but he recovers
at the near post to keep out a shot, and then Benjamin Cremaschi gets a massive block on the follow up to see the ball pop up over the crossbar.

The hosts had the ball right on the goal mouth and couldn't capitalize!

#InterMiamiCF is having such trouble clearing their lines lately… — Ian Hest (@IanHest) October 5, 2023

14th min: Inter Miami look a bit cleaner in the attack now, and they have begun to trouble the Chicago attacking third. Facundo Farias nearly has a chance at
the top of the penalty area, but he's dispossessed before he can pick out a pass or a shot. He had teammates streaking on goal.
</item 67>

<item 68>
Gio Reyna looks good in start for USMNT vs. Germany, but those involved in the finish are punished in 3-1 defeat (Sporting News, 2023-10-14T23:41:07+00:00)
So many national team fans sided with Reyna during the controversy that developed as the team competed at last winter's World Cup. They got what they argued for
in this game, but not what they wished.
</item 68>

<item 69>
Bears vs. Panthers live score, updates, highlights from NFL 'Thursday Night Football' game (Sporting News, 2023-11-09T23:04:33+00:00)
— And now this one is officially over. The Bears pick up their third win of the season, while Carolina drops to 1-8 and faces more tough questions about Bryce
Young and the offense.

11:08 p.m. — Bagent hits Mooney for the first down on 3rd & 7, and the Panthers now won't be able to stop the clock.

11:06 p.m. — Pineiro's attempt is SHORT from 59 yards out. The gamble fails for Frank Reich. The Panthers will hope to get the ball back.

11:04 p.m. — Young is very nearly intercepted on 3rd & 10, and the Panthers say they've seen enough of the offense. Eddy Pineiro is attempting a 59-yard field
goal to tie the game.

11:00 p.m. — Miles Sanders gets himself wide open on fourth down to extend the drive.
</item 69>

<item 70>
Vikings vs. Broncos live score, updates, highlights from NFL 'Sunday Night Football' game (Sporting News, 2023-11-19T23:00:04+00:00)
The Vikings decided to go for it and Joshua Dobbs picked up the first down on a QB sneak.

10:54 p.m.: Ty Chandler continues to rumble. The spell RB picked up a massive gain on 3rd & 20 and it appeared that he got the first down, but the spot on the
field is under review. This could be a crucial call.

10:48 p.m.: It looked like it was going to be a three-and-out for the Vikings offense but a fake punt has kept the drive alive! Ty Chandler picked up 31 yards
on a gutsy call from head coach Kevin O'Connell to bring the ball into Denver's territory.

Vikings 17, Broncos 15

10:43 p.m.: FIELD GOAL — Will Lutz knocks one in from 28 yards, his fifth made field goal of the day. Lutz has been the Broncos' lone source of offense,
bringing Denver within one score with 10:37 to play in the game.
</item 70>

<item 71>
Colorado vs. UCLA live score, updates, highlights from Week 9 college football game (Sporting News, 2023-10-28T22:08:03+00:00)
Can the Buffaloes finally connect for a touchdown before its too late?

UCLA adds on another 💪@UCLAFootball makes it 21-9 in the 4th quarter vs Colorado pic.twitter.com/6Eu9gn6kgn — FOX College Football (@CFBONFOX) October 29, 2023

End of third quarter: Colorado 9, UCLA 14

10:20 p.m. – What a catch for the Bruin! After review, Garbers pass to Loya is ruled complete.

Colorado 9, UCLA 14

10:06 p.m. FIELD GOAL – Mata nails a 34-yard field goal to lessen Colorado's deficit to five. Four minutes remain in the third quarter.

10:03 p.m. – Sacked! Sanders is taken down by Laiatu Latu, who has multiple tonight. UCLA records its sixth sack of the night with the latest.
</item 71>

<item 72>
Chicago Fire vs Inter Miami live score, result, updates, highlights from Lionel Messi in MLS match (Sporting News, 2023-10-04T23:33:00+00:00)
For Chicago, Kei Kamara is on the bench, with Georgios Koutsias slotted in up front just as he was on the weekend. Mexican playmaker Jairo Torres sits, with
Fabian Herbers deployed on the right wing.

Lionel Messi misses his fourth straight Inter Miami game due to injury 🤕 pic.twitter.com/W1DRc9NJNo — B/R Football (@brfootball) October 4, 2023

45 mins to kickoff: Lineups are now exceedingly late, but while we wait, a nod to Chicago Fire striker Kei Kamara, who could tie Landon Donovan on the all-time
Major League Soccer scoring list tonight with a goal. Kamara's 144 goals are just one behind the former USMNT great, while Chris Wondolowski's 171 goals leads
the league's career chart.

Will @keikamara tie Landon Donovan tonight for second on the all-time scoring list?
</item 72>

<item 73>
Colorado vs. Arizona State live score, updates, highlights from Week 6 college football game (Sporting News, 2023-10-07T21:21:03+00:00)
The play is being reviewed for a touchdown.

7:41 p.m. – Colorado is called offsides, giving Arizona State a first down at the Buffs' 48.

7:34 p.m. – After scoring a touchdown on its opening possession, Arizona State has three straight three-and-outs.

7:30 p.m. – The Sun Devils record their third sack of the day already, this time forcing fourth down for the Buffaloes. Safe to say their defense came prepared.

End of first quarter: Colorado 7, Arizona State 7

Colorado 7, Arizona State 7

7:17 p.m. TOUCHDOWN – The Buffs tie it up! Needing one yard on the fourth down, Sanders hands it off to Xavier Weaver who weaves his way into the end zone. Its
7-7 in Tempe with two minutes remaining in the first quarter.

On 4th & goal Xavier Weaver takes it in for @CUBuffsFootball 🦬
</item 73>

<item 74>
Colorado vs. UCLA live score, updates, highlights from Week 9 college football game (Sporting News, 2023-10-28T22:08:03+00:00)
Colorado vs. UCLA score

Q1 Q2 Q3 Q4 F Colorado 6 0 3 7 16 UCLA 0 7 7 14 28

Colorado vs. UCLA results, highlights from Week 9 game

Final: Colorado 16, UCLA 28

Colorado 16, UCLA 28

10:58 p.m. TOUCHDOWN – The Buffaloes aren't going down without a fight. Sanders gets his first touchdown of the night, finding Jimmy Horn Jr. in the end zone on
an 18-yard toss. Colorado attempts an onside kick with only two-and-a-half minutes left, but the Bruins recover it.

Colorado 9, UCLA 28

10:45 p.m. TOUCHDOWN – Collin Schlee finds the end zone! His seven-yard run helps add seven points for the Bruins, who lead 28-9 with six minutes left in the
game.
</item 74>

<item 75>
Browns survive Bears Hail Mary on wild, near-catch by Darnell Mooney: 'Damn near s— myself' (Sporting News, 2023-12-17T23:06:30+00:00)
"There's so many things that can happen, whether we've been a part of it or whether like I said you've seen it on Sundays just watching TV so believe me I was
over there and guys want to come up and congratulate and things like that and I'm just sitting there waiting for that clock to hit zero."

MORE: Updated NFL playoff picture in Week 15

Browns edge rusher Myles Garrett was on the field as one of the few players at the line of scrimmage, focusing more on pressuring Fields than picking him off.
And when Fields got the throw off, he, too, became a spectator.

"Shoot, too good of a view," Garrett said. "He kinda looked at me just made sure I was far enough and then tossed it up and I was like, 'Please, Lord, don't let
us be on one of these YouTube compilations.'"

It had been a wild ending to the game.
</item 75>

<item 76>
Bears vs. Panthers live score, updates, highlights from NFL 'Thursday Night Football' game (Sporting News, 2023-11-09T23:04:33+00:00)
End of first half: Panthers 10, Bears 9

9:40 p.m. FIELD GOAL — In a game that might be lucky to see an offensive touchdown, the Bears deserve some credit for that drive to work their way into field
goal range. They trail 10-9 at the half.

9:39 p.m. — Bagent narrowly avoids a sack and actually runs for positive yardage to move Santos' attempt closer. Chicago is down to three seconds, so here comes
Santos.

9:38 p.m. — A perfectly executed play to get the ball to Cole Kmet, who picks up the first down and then some but still gets out of bounds to stop the clock.
The Bears are in field goal range.
</item 76>

<item 77>
Jaguars vs. Saints live score, updates, highlights from NFL 'Thursday Night Football' game (Sporting News, 2023-10-19T22:00:04+00:00)
With it being fourth down, McManus comes out for a field goal.

9:38 p.m. — Interception! Marcus Lattimore comes up with a massive play for the Saints, picking off Lawrence down inside the 10-yard line with just 34 seconds
to go in the half. Lawrence was looking for Agnew, but Lattimore dove in front of the receiver for the pick. The officials are going to review to see if
Lattimore maintained possession, otherwise it may be an incomplete pass.

9:36 p.m. — Fake punt! Not so fast! Pederson dips into bag of tricks and calls for the fake punt. Logan Cooke connects with Tim Jones for a 17-yard gain, and
the Jaguars keep possession. That took some stones on Pederson's part, but it pays off, giving Lawrence and the offense a chance to find the end zone before the
end of the half.

9:35 p.m.
</item 77>

<item 78>
Chicago Fire vs Inter Miami live score, result, updates, highlights from Lionel Messi in MLS match (Sporting News, 2023-10-04T23:33:00+00:00)
The broadcast reports that Gutierrez was pulled due to a sore knee.

52nd min: PENALTY, INTER MIAMI! Handball on Jonathan Dean who blocks a shot by Benja Cremaschi! He is lucky to only get a yellow card here, as he was prone on
the ground and had his arms outstretched as he turned his back to the shot.

Josef Martinez steps up to the spot...GOAL! INTER MIAMI! The penalty is slammed into the back of the net, with Chris Brady going the wrong way as well! Chicago
had their lead for just three minutes.

50th min: GOAL! CHICAGO FIRE! DeAndre Yedlin completely loses his runner at the back post, and Xherdan Shaqiri makes him pay! The U.S. international, up the
field from the attack, had no clue that Shaqiri was making his run and dogged it back in transition.
</item 78>

<item 79>
Inter Miami vs FC Cincinnati live score, result, updates, highlights with Lionel Messi's MLS playoffs on line (Sporting News, 2023-10-07T21:30:00+00:00)
Officially the fourth time they've hit the woodwork, although this one wasn't quite as dramatic.

89th min: Lionel Messi draws a very contentious foul by Yuya Kubo whose tackle from behind looked clean but draws a whistle regardless. Messi's delivery from a
deep central position is floated in and headed out by Matt Miazga for a corner.

The ensuing set-piece is played short and eventually comes to Messi whose deep delivery is seeking the far corner but nervously tapped wide by Roman Celentano.
Eventually, a foul by Messi ends the chance and leaves Santiago Arias rolling around on the ground.

88th min: Two more changes for FC Cincinnati as Dominique Badji enters for Luciano Acosta, whose night is done a few minutes early despite trailing by a goal in
the Golden Boot race. Malik Pinto also comes on for Brandon Vazquez.
</item 79>

<item 80>
Colorado vs. Arizona State live score, updates, highlights from Week 6 college football game (Sporting News, 2023-10-07T21:21:03+00:00)
It allowed five sacks on Sanders, bringing the season total to a nation-leading 31. The defense picked up five sacks themselves to maintain pressure on Trenton
Bourguet.

For Arizona State, Bourguet put all he could into Saturday's narrow loss. He's one of two healthy quarterbacks on the roster, and passed for 335 yards and a
touchdown. The Sun Devils had a real shot at their first conference win. But as the game came to a close, Bourguet seemed roughed up and slow, though he did
drive the team down the field 94 yards in 13 plays for the game-tying touchdown with 50 seconds left.

Luckily for Arizona State, next week is a bye week. What's not so lucky is that it'll go on to play three ranked teams over the following three weeks.

We didn't see the worst of either team, though we didn't see the best either. Colorado needed a bounce back win and got one.
</item 80>

<item 81>
Chicago Fire vs Inter Miami live score, result, updates, highlights from Lionel Messi in MLS match (Sporting News, 2023-10-04T23:33:00+00:00)
The Chicago Fire haven't yet been able to take advantage.

Kickoff: A huge game in the MLS playoff race is under way from Soldier Field! The NFL venue in the Windy City is sold out.

"Drop a photo that will create FOMO."



Admin: pic.twitter.com/PDqEVFTteJ — Chicago Fire FC (@ChicagoFire) October 5, 2023

Chicago Fire vs. Inter Miami: Pre-match commentary, analysis, stats, and more

10 mins to kickoff: With both these clubs looking to reach the playoffs, they will be keeping a close eye on other results around the Eastern Conference
tonight. They're not going according to plan.

Charlotte FC, who sat 14th coming into tonight's match, lead Toronto FC 2-0, which for the moments jumps them above Inter Miami.
</item 81>

<item 82>
Browns survive Bears Hail Mary on wild, near-catch by Darnell Mooney: 'Damn near s— myself' (Sporting News, 2023-12-17T23:06:30+00:00)
It had been a wild ending to the game. Chicago had led 17-7 heading into the fourth quarter, but on successive drives, the Browns scored a field goal to open
the fourth quarter, scored a game-tying touchdown with 3:08 left in the contest. Cleveland forced a three-and-out, then drove down the field in eight plays to
kick a go-ahead field goal with 32 seconds left, burning all of Chicago's timeouts in the process.

The Bears got a bit of life when Tyler Scott caught a short pass from Fields and raced up the right sideline 30 yards to move up to Cleveland's 45. But after
back-to-back incompletions, the Bears were forced to try for the Hail Mary rather than kick the long field goal.

"At the end of the game, I loved the way our guys battled at the end," Bears coach Matt Eberflus said.
</item 82>

<item 83>
Vikings vs. Broncos live score, updates, highlights from NFL 'Sunday Night Football' game (Sporting News, 2023-11-19T23:00:04+00:00)
Vikings 10, Broncos 3

9:08 p.m.: FIELD GOAL — Greg Joseph drills a 34-yarder to give the Vikings a 10-3 lead with 8:09 to play in the half. The Vikings' offense stalled out in the
red zone as Joshua Dobbs was sacked for an 11-yard loss on 3rd & Goal.

9:06 p.m.: Joshua Dobbs has some magic in him. In a cloud of smoke in the pocket, Dobbs found TE TJ Hockenson on a 29-yard gain to pick up a first down on 3rd &
2. The big play brings the Vikings into the red zone.

9:00 p.m.: Another three-and-out for Russell Wilson and the Broncos. On 3rd & 3, Wilson was under heavy pressure and threw the ball away as he was nearly
brought down for a sack.
</item 83>

<item 84>
Inter Miami vs FC Cincinnati live score, result, updates, highlights with Lionel Messi's MLS playoffs on line (Sporting News, 2023-10-07T21:30:00+00:00)
65th min: Chance, FC Cincinnati! Inter Miami have hit the post three times, but now it's FC Cincinnati who have struck the woodwork! Brandon Vazquez gets his
head to a corner but it comes off the post.

64th min: Inter Miami have to remain wary of the vicious FC Cincinnati counter-attack, and here a lovely Luciano Acosta switch laterally across the top of the
penalty area finds Alvaro Barreal in space, but his hit is blocked.

Pat Noonan makes his second change of the match, bringing Yuya Kubo on for Junior Moreno in midfield.

58th min: Lionel Messi's influence has been immediate, and he earns a dangerous free-kick just outside the penalty area, well within his range. Messi appears to
be struggling to get good footing on this field, but it's the shirt pull from Miazga which brings him down.

Messi takes it but puts it way over the crossbar. A bit rusty?
</item 84>

<item 85>
N.F.L. Executives on the Biggest Questions Facing Super Bowl Contenders (The New York Times, 2023-10-16T23:58:53+00:00)
veterans. The numbers stack up pretty evenly across various distance cutoffs. Based on these numbers, there’s no statistical indication rookies are less
effective in the postseason.

“Playoff teams have a different aura, different feel in the locker room,” a veteran evaluator said. “Percentages are one thing, but who is the guy who trots out
there to try the game-winner? It’s a different pressure shooting free throws up by 20, down by 20, than in that type of situation.”

Advertisement

Fans might recall rookie kickers missing playoff kicks: Rodrigo Blankenship with the Colts in 2020, Harrison Butker with the Chiefs in 2017, Nate Kaeding with
the Chargers in 2005. But the NFL has seen plenty of rookies kick effectively in the postseason as well. Evan McPherson made all 14 postseason tries with the
2021 Super Bowl Bengals. The 2017 Eagles won a Super Bowl with rookie Jake Elliott performing well.
</item 85>

<item 86>
Inter Miami vs FC Cincinnati live score, result, updates, highlights with Lionel Messi's MLS playoffs on line (Sporting News, 2023-10-07T21:30:00+00:00)
Messi takes it but puts it way over the crossbar. A bit rusty?

57th min: Jean Mota has a shot from distance, and it looks to be a vicious knuckler with some dip, but it goes just wide.

55th min: HERE COMES LIONEL MESSI! He will enter with around 35 minutes to save Inter Miami's season, replacing Tomas Aviles in a very attacking change. Jean
Mota also comes on to replace David Ruiz, which is harsh as the young midfielder had been very good.

50th min: Chance, FC Cincinnati! DRAKE CALLENDER MAKES A MASSIVE DOUBLE SAVE! FC Cincinnati are RIGHT on the doorstep, but he comes up huge twice! Aaron
Boupendza is the most aggrieved, as his follow-up came from directly in front of the goal but he was stymied by Callender's outstretched right foot!
</item 86>

<item 87>
Colorado vs. UCLA live score, updates, highlights from Week 9 college football game (Sporting News, 2023-10-28T22:08:03+00:00)
– UCLA misses the field goal! RJ Lopez's 24-yard attempt hits the left upright and is no good. The Bruins keep a one-point lead with a 1:37 left in the second
quarter.

9:03 p.m. PENALTY – Shilo Sanders is ejected for targeting after a hit on Carsen Ryan. He will be out for the remainder of the game.

Shilo Sanders was disqualified for targeting after this hit pic.twitter.com/1EJ7IOdKKs — FOX College Football (@CFBONFOX) October 29, 2023

8:58 p.m. – Colorado goes three-and-out, unable to score on any turnovers. For UCLA, Garbers is back, and he takes a 17-yard run to the UCLA 37 for a first
down.

8:51 p.m. INTERCEPTION – Third times a charm! Collin Schlee, who came in for Garbers, is interception by Hunter.
</item 87>

<item 88>
Colorado vs. Arizona State live score, updates, highlights from Week 6 college football game (Sporting News, 2023-10-07T21:21:03+00:00)
The QB keeps it for 16 yards and a touchdown. Less than a minute is left until halftime.

8:09 p.m. – Sanders finds Jimmy Horn Jr. for 24 yards. Colorado is within striking distance, the ASU 39, with only two minutes left.

8:06 p.m. – ASU's head coach Kenny Dillingham could not be more unhappy with the referees, who ruled Bourguet's run just short of a first down. It's fourth down
for the Sun Devils.

Colorado 7, Arizona State 14

7:48 p.m. TOUCHDOWN – Cam Skattebo pushes through the huddle for the score! It's his fifth rushing touchdown of the season. ASU now leads 14-7 with 6:38 left in
the half.

7:45 p.m. – Andre Johnson reaches over the pylon with one hand, but the ball is knocked out. The play is being reviewed for a touchdown.
</item 88>

<item 89>
Colorado vs. Arizona State live score, updates, highlights from Week 6 college football game (Sporting News, 2023-10-07T21:21:03+00:00)
Its 7-0 just a few minutes in.

6:35 p.m. – The game will begin on the Pac-12 Network Arizona due to the Washington State vs. UCLA game running long.

6:10 p.m. – Starting safety Trevor Wood is out for Colorado. Good news is Shilo Sanders will start.

6:05 p.m. – The Sun Devils are looking to end their eight-gam losing streak against FBS teams. Ironically, their last FBS win was against Colorado when Trenton
Bourguet set a school record for most passing yards in a career start. He's expected to be Arizona State's starting QB tonight.

5:50 p.m. – Colorado to debut new helmets in Tempe.

5:20 p.m. – The Buffs have entered the chat.

MORE: Best Colorado player prop odds for Week 6 college football matchup against Arizona State

How to watch Colorado vs. Arizona State

Date : Saturday, Oct.
</item 89>

<item 90>
Bears vs. Vikings live score, updates, highlights from NFL 'Monday Night Football' game (Sporting News, 2023-11-27T23:32:04+00:00)
It looked like that score might be the difference, particularly after Fields fumbled away the ball again on the next possession, but the Bears' defense put the
ball back in his hands in the final minutes. Fields delivered.

Facing 3rd & 10 outside of field goal range, Fields hit an open D.J. Moore for a 36-yard gain. At that point, all the Bears needed to do was kneel and put Cairo
Santos in position to win the game. Santos drilled the 30-yard attempt with 10 seconds remaining, and that would seal it for Chicago.

MORE: Vikings playoff chances after loss to Bears

As he embarks on a crucial stretch that could determine his future with the Bears, Fields flashed the good and the bad. He was excellent on Chicago's opening
drive, albeit one that ended without points, but missed throws and costly turnovers allowed the Vikings to hang around.
</item 90>

<item 91>
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
</item 91>

<item 92>
Jets vs. Raiders live score, updates, highlights from NFL 'Sunday Night Football' game (Sporting News, 2023-11-12T23:30:03+00:00)
But O'Connell way overthrows Adams, and the Raiders have to settle for the 40-yard field goal to tie the game with 2:36 left in the third quarter. Adams was
furious there wasn't a penalty. All kicks so far tonight. Jets 9, Raiders 9

10:27 p.m.: O'Connell rifles a quick pass out to Hunter Renfrow, who races up and gets near the line of scrimmage but is just inches shy of the line. The
Raiders opt to go for it and hand the ball to Jacobs, who finds a hole on the right side and gets the necessary yardage to the Jets 23 for the first.

10:26 p.m.
</item 92>

<item 93>
Vikings vs. Broncos live score, updates, highlights from NFL 'Sunday Night Football' game (Sporting News, 2023-11-19T23:00:04+00:00)
10:01 p.m.: The Vikings are moving the chains. They've picked up two first downs in five plays, gaining 43 yards in the process. Alexander Mattison picked up 23
yards on the ground, and Joshua Dobbs and Jordan Addison connected on an 18-yard pass to pick up the bulk of the yardage.

9:56 p.m.: A holding penalty derailed the Broncos' drive as they came up way short on a 3rd & 20 attempt. Minnesota will take over from its own 31 after a
19-yard punt return from Brandon Powell.

9:54 p.m.: Denver receives to start the second half and we're back in action.

End of first half: Vikings 10, Broncos 9

9:38 p.m.: FIELD GOAL — Will Lutz from 52 yards... it's good! The Broncos kicker steps up to bury a long field goal as the first half comes to a close.
</item 93>

<item 94>
NFL Fantasy Football Week 5 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-04T21:56:25+00:00)
His off-target rate in the fourth was 9.1%, which was closer to what he had averaged in his first three games (13.6%).

BRONCOS: Have allowed season-best Fantasy numbers to each of the past three quarterbacks they've played this season. They're a terrific matchup for anybody
right now. Remember this when we talk about the Jets.

COMMANDERS: Sport a slightly below-average pass rush pressure rate thanks to only one game all year with a rate over 30% (52.7% at Denver, and Wilson still had
a great game even if you take away a Hail Mary touchdown catch). They rank seventh in sacks (13) and 13h in pressures (53), which again floats around league
average.

BEARS: Rank league-worst in pass rush pressure rate allowed at 47.4% of their snaps -- nearly one out of every two drop backs is pressured.
</item 94>

<item 95>
NFL Fantasy Football Week 5 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-04T21:56:25+00:00)
Bringing back guard Teven Jenkins might help a little, but they're still a struggling unit.

DEEP BALL: Washington ranks bottom-five in attempts, yards per catch and touchdowns allowed on pass plays of 15-plus Air Yards over the past three weeks. This
is an area Fields can exploit -- he's tied for the most touchdowns on throws of 15-plus Air Yards this season than Fields with four and is around league-average
in completion rate, attempts per game and Expected Points Added (EPA) per drop back.

LAST SEASON: Washington got a season-best 57.1% pass rush pressure rate on Fields and blitzed him on 40.5% of his drop backs. Fields had one passing touchdown
and one interception and still had nearly 20 Fantasy points because he ran for 88 yards.

RUSHING: Fields followed up a Week 3 game with seven designed runs with just two designed runs in Week 4 for two total yards.
</item 95>

<item 96>
Chicago Fire vs Inter Miami live score, result, updates, highlights from Lionel Messi in MLS match (Sporting News, 2023-10-04T23:33:00+00:00)
Robert Taylor's cheeky backheel gets the ball to Facundo Farias, but his shot is blocked. It's all sauce from the visitors right now, and Chicago are under
siege.

59th min: Chance, Inter Miami! BENJA CREMASCHI PUTS A GOLDEN OPPORTUNITY WIDE! Moments after Facundo Farias misses, Josef Martinez again unlocks the Chicago
Fire defense but with only the goalkeeper to beat, Cremaschi misses the target across the face of goal! He HAD to take the lead!

58th min: Chance, Inter Miami! Josef Martinez squares to Facundo Farias on the break, but the young Argentine misses wide left! That's a terrible miss with the
goal gaping! The chance was there, as he was free on goal!

55th min: Chicago coach Frank Klopas makes the first substitution, withdrawing Brian Gutierrez and bringing on Maren Haile-Selassie.
</item 96>

<item 97>
Colorado vs. UCLA live score, updates, highlights from Week 9 college football game (Sporting News, 2023-10-28T22:08:03+00:00)
Collin Schlee wasn't going to be kept out of the end zone 💪@UCLAFootball makes it 28-9 pic.twitter.com/MtTdXmvF6B — FOX College Football (@CFBONFOX) October 29,
2023

10:43 p.m. – Harden going the distance with 25-yards and a hurdle.

Colorado 9, UCLA 21

10:23 p.m. TOUCHDOWN – Too easy for TJ Harden. He caps off an 80-yard drive with a short run for a touchdown. UCLA extends its lead to 12 with 14 minutes on the
clock. Can the Buffaloes finally connect for a touchdown before its too late?
</item 97>

<item 98>
Bears vs. Panthers live score, updates, highlights from NFL 'Thursday Night Football' game (Sporting News, 2023-11-09T23:04:33+00:00)
9:28 p.m. — An illegal contact penalty, the second of the night on the Bears, gives Carolina a fresh set of downs.

9:25 p.m. — The Panthers will have 3rd & 3 out of the two-minute warning, just hoping to get in position for some points before halftime. They'll start the
second half with the ball as well.

Panthers 10, Bears 6

9:19 p.m. FIELD GOAL — Bagent is fortunate just to hold onto a bad snap, but his pass is incomplete and brings out Cairo Santos. It's 10-6, Panthers.

9:17 p.m. — A second big play by rookie Tyler Scott, who catches a 15-yard pass from Bagent to give the Bears a fresh set of downs. Chicago is into field goal
range.
</item 98>

<item 99>
Baker Mayfield stats today: Buccaneers QB keeps Tampa Bay atop NFC South with sparkling showing vs. Packers (Sporting News, 2023-12-17T23:05:13+00:00)
He joins Nick Foles (11/3/13 at Oakland) & Ken O'Brien (11/2/86 at Seattle). pic.twitter.com/HvNVWY7MY3 — NFL345 (@NFL345) December 17, 2023

Not bad for an undersized sidewinder who looked to be approaching the end of his rope as a starting QB one year prior.

You can't get better than perfect. And Mayfield was about as close to it as any QB has been in a road matchup.
</item 99>

<item 100>
European roundup: Bellingham helps Real Madrid go top while Bayern draw (The Guardian, 2023-09-30T21:17:47+00:00)
But Martínez opened the scoring in the 62nd minute when he collected Marcus Thuram’s low cross and chipped a finish over Guillermo Ochoa.

skip past newsletter promotion Sign up to Soccer with Jonathan Wilson Free weekly newsletter Jonathan Wilson brings expert analysis on the biggest stories from
European soccer Privacy Notice: Newsletters may contain info about charities, online ads, and content funded by outside parties. For more information see our
Newsletters may contain info about charities, online ads, and content funded by outside parties. For more information see our Privacy Policy . We use Google
reCaptcha to protect our website and the Google Privacy Policy and Terms of Service apply. after newsletter promotion

The substitute Lautaro Martínez converts a penalty against Salernitana. Photograph: Massimo Pica/EPA

Three minutes later Mateusz Legowski’s apparent equaliser was disallowed due to an offside.
</item 100>
=== END ITEM id=be73cec7382825f6 ===

=== ITEM id=b6eb4d14b65c8301 ===
Gold sentences and facts:
Sentence 0: New customers can get up to $1000 in bonus bets if they lose their first bet.
  Fact 0.0: New customers can get up to $1000 in bonus bets.
  Fact 0.1: New customers can get bonus bets if they lose their first bet.
Sentence 1: We expect Caesars to offer the same offer in VT is has elsewhere, with new sign-ups eligible to receive up to a $1,000 Bonus Bet if their first bet loses.
  Fact 1.0: Caesars is expected to offer the same offer in VT as it has elsewhere.
  Fact 1.1: New sign-ups are eligible to receive up to a $1,000 Bonus Bet.
  Fact 1.2: New sign-ups are eligible for the Bonus Bet if their first bet loses.

Retrieved items:
<item 1>
Uber sexual assault survivors call for in-car cameras, tech upgrades (TechCrunch, 2023-10-13T19:26:48+00:00)
The number of incidents increased in the categories of nonconsensual kissing, nonconsensual touching and nonconsensual penetration, or rape, between 2017 and
2019.

The total number of rape incidents also rose from 2017 to 2019. And while the number of reported rapes dropped in 2020, the rate has remained the same since
2017. In other words, there has not been an improvement in the rate of rapes on the platform.

Sexual assaults are an ongoing problem for Uber

Uber has been sued numerous times over the past few years by passengers who claim they were sexually assaulted during a ride. Lyft has faced similar lawsuits
and accusations.

Under Uber’s terms of use, class action lawsuits can’t be filed against the company in cases of sexual assault, so each case has to be heard individually. This
has prevented survivors from advocating for themselves collectively.

Judge Charles Breyer in the Northern District of California will preside over pretrial hearings.
</item 1>

<item 2>
Tesla ‘digs its own grave with the Cybertruck,’ Convoy collapses and Rivian scores a win at Rebelle (TechCrunch, 2023-10-22T19:15:29+00:00)
Tesla’s shares, which are priced more as a tech stock than as an automaker’s stock, were down after Q3 earnings. Investors were clearly feeling skittish after
Tesla reported a gross margin of 17.9%, down from 25.1% in the same period last year. That’s also down from Q2’s margins of 18.2%. As a result, profits fell 44%
to $1.85 billion in Q3 from the same year-ago period.

Investors see the falling margins and Tesla’s price cuts as proof that demand is lessening for the vehicles as other EVs take market share and rising interest
rates make it difficult for many buyers to afford big ticket purchases. The company also reported that solar deployments slipped 48% in Q3 from the same period
last year. But the company made up for it by pulling in a 90% spike in energy storage deployments.

Tesla also gave some updates about its long-delayed Cybertruck.
</item 2>

<item 3>
Uber’s Q3 numbers include impressive profitability gains, slower-than-expected growth (TechCrunch, 2023-11-07T17:03:41+00:00)
In business segment terms, Uber generated $17.9 billion in ride-hailing bookings (+31% year over year), and $16.1 billion worth of delivery bookings (+18%).
Those key business groups at the company generated $5.1 billion and $2.9 billion in revenue, respectively, during the September quarter.

There’s nuance to the revenue figures that we need to consider as the rule-changes in question do affect startups that operate in related categories. On the
ride-hailing front, Uber told its investors that its Q3 2023 revenue was “negatively impacted by business model changes in some countries that classified
certain sales and marketing costs as contra revenue by $161 million.” That was not the only legal change that led to Uber’s results changing shape. Under its
delivery business results, the company added that its revenue result was “negatively impacted by business model changes that classified certain sales and
marketing costs as contra revenue by $360 million.”

The combined impact of those two items was 8 percentage points of growth.
</item 3>

<item 4>
Uber’s Q3 numbers include impressive profitability gains, slower-than-expected growth (TechCrunch, 2023-11-07T17:03:41+00:00)
Uber reported third-quarter earnings on Tuesday that show a profitable ride-hail and delivery company that’s chugging along in spite of slowing growth in some
sectors.

The company reported revenue of $9.3 billion, an 11% increase year over year. Investors had expected Uber to report revenues of around $9.5 billion (FactSet,
Refinitiv), meaning that despite the company’s growth, it fell short of estimates. Turning to profitability, Uber reported net income of $221 million in the
third quarter, or 10 cents per share, compared with a net loss of $1.2 billion, or 61 cents per share, in the same quarter last year. Again, the company fell
short of expectations that it would generate 12 cents in per-share profit.

Looking ahead, Uber anticipates gross bookings of $36.5 billion to $37.5 billion, up just 6% at the top-end from its Q3 result.
</item 4>

<item 5>
Uber’s Q3 numbers include impressive profitability gains, slower-than-expected growth (TechCrunch, 2023-11-07T17:03:41+00:00)
Of course, adjusted EBITDA is to profit as hidden heels in men’s shoes are to height, so we’ll want some harder figures as well. In the third quarter, Uber
generated $394 million in operating income and $219 million worth of net income. Mix in $966 million worth of positive operating cash flow, and Uber looks very
healthy, given that all of the figures in this paragraph were improvements on its year-ago results.

But while food delivery and scooting humans about town were profitable for Uber in the quarter, the final major portion of its business had a more lackluster
quarter.

Freight falls

One area that continues to drag on Uber is its freight business.

While Uber’s ride-hailing and delivery business saw an uptick in gross bookings in the third quarter, Uber Freight experienced a 27% drop year-over-year.
Revenue, as a result, had a similar fall.

The business unit reported revenue of $1.3 billion in the third quarter, a 27% drop from the same period last year.
</item 5>

<item 6>
Uber sexual assault survivors call for in-car cameras, tech upgrades (TechCrunch, 2023-10-13T19:26:48+00:00)
Uber hasn’t released safety data for 2021 and 2022, and won’t say when it plans to do so. Older data backs up some of Abrams’ claims.

Uber has published two safety reports — one containing data from 2017 to 2018 and another with data from 2019 to 2020. The ride-hail company claims that the
rate of sexual assault reported on the app decreased 38% between its first and second reports.

While the rate decrease is positive, not all the data delivers the same message. Over that same time period, the total number of sexual assault reports across
five categories fell from 5,981 to 3,824. That drop could be explained by the decrease of trips from 1.4 billion in 2019 to 650 million in 2020.

And while the total number of incidents as a percentage of total number of rides does decrease through the years, the number of incidents in certain categories
actually increases.
</item 6>

<item 7>
Uber’s Q3 numbers include impressive profitability gains, slower-than-expected growth (TechCrunch, 2023-11-07T17:03:41+00:00)
After reporting, shares of Uber are up 1.6% in trading, after an up-and-down morning of shares exchange.

Against a backdrop of macroeconomic uncertainty, the former startup darling’s results can be viewed as an indication that its business model has matured and is
now in a stable, profitable and cash-generating position. On the other hand, under-performing demand results from a company as global as Uber could indicate
that consumer spend is coming in lighter than anticipated.

For startups in the transport and on-demand sectors, Uber’s earnings are a regular informational zeitgeist. So let’s work to understand where Uber’s revenue
came from in the third quarter, and how each of its top-line sources converted — or not — to bottom-line results.

Where did the money come from?

In the third quarter, Uber saw total bookings rise from $29.1 billion to $35.3 billion, a gain of around 21%.
</item 7>

<item 8>
Uber’s Q3 numbers include impressive profitability gains, slower-than-expected growth (TechCrunch, 2023-11-07T17:03:41+00:00)
The question is whether Uber believes in the long-term income potential of freight.

So what?

When discussing Uber’s results internally, our vibe was that it is a very profitable and healthy company today, albeit one that is not growing as fast as the
market had hoped. The fact that Uber’s share price is up as we write this indicates that investors are content thus far to excuse the small Q3 misses and keep
their eyes more focused on its year-over-year improvements and forward guidance. More when we get Lyft’s numbers.
</item 8>

<item 9>
Uber’s Q3 numbers include impressive profitability gains, slower-than-expected growth (TechCrunch, 2023-11-07T17:03:41+00:00)
On a quarter-over-quarter basis, Uber Freight had a 1% gain in revenue.

The results don’t get any better once we turn to net income. On an adjusted basis, Uber Freight lost $13 million in the third quarter compared to a $1 million
profit in the same quarter last year.

Uber said the dismal year-over-year revenue results for its freight business was driven by lower revenue per load and volume. Both are consequences of the
challenging freight market cycle.

Uber Freight isn’t alone. Other more recent entrants to the freight and logistics industry such as Flexport and Convoy have struggled this year. In Convoy’s
case, the business was forced to shut down with its assets gobbled up by Flexport.

Uber Freight continues to plug along, despite these economic headwinds. The question is whether Uber believes in the long-term income potential of freight.

So what?
</item 9>

<item 10>
Uber’s Q3 numbers include impressive profitability gains, slower-than-expected growth (TechCrunch, 2023-11-07T17:03:41+00:00)
The combined impact of those two items was 8 percentage points of growth.

The profit perspective

From bookings to revenues to profits: How did the main portions of Uber’s business generate black ink?

Turning to what Uber calls “segment adjusted EBITDA,” it’s not hard to see how the company managed to rack up more income in its most recent quarter.
Ride-hailing adjusted profit rose to $1.29 billion, up 43% from $898 million in the year-ago quarter, while delivery saw its own profitability skyrocket from
$181 million in Q3 2022 to $413 million in its most recent fiscal period.

Uber did spend more during Q3 2023 when compared to its Q3 2022 result, but the 5% gain in “Corporate G&A and Platform R&D” costs to $595 million was far less
than the gains we saw above. So, Uber’s adjusted EBITDA rose from $516 million in aggregate to $1.09 billion in the third quarter of this year.
</item 10>
=== END ITEM id=b6eb4d14b65c8301 ===

=== ITEM id=7b65f35b2fc2c09e ===
Gold sentences and facts:
Sentence 0: Research shows that even a small reduction in sleep has been shown to impact your immune system, however.
  Fact 0.0: Research shows that even a small reduction in sleep impacts the immune system.
Sentence 1: “This sex difference in insomnia emerges after puberty, suggesting that hormonal differences such as menstruation, pregnancy and menopause may be the cause of this increased risk in women,” Dr Browning says.
  Fact 1.0: Dr Browning says this sex difference in insomnia emerges after puberty.
  Fact 1.1: Dr Browning says the sex difference suggests hormonal differences may be the cause of the increased insomnia risk in women.
  Fact 1.2: Dr Browning names menstruation, pregnancy and menopause as examples of hormonal differences.
  Fact 1.3: Dr Browning says hormonal differences such as menstruation, pregnancy and menopause may cause this increased risk in women.

Retrieved items:
<item 1>
Rogue Trader’s role-playing embraces the brutality and freedom of Warhammer 40K (Polygon, 2023-12-06T13:00:00+00:00)
“In the grim darkness of the far future, there is only war.” That’s the tagline of Warhammer 40,000, one of the most over-the-top and brutal sci-fi settings
around. But even in a merciless, brutal dystopia that grinds its people into dust, some get to enjoy being on the top of the food chain. Warhammer 40,000: Rogue
Trader is about the perils and pleasures of being atop that particular hierarchy.

Rogue Traders in 40K are freelance explorers, scouting the far frontiers of the Imperium of Man. It’s a risky career, but it comes with luxurious rewards:
agency, freedom, power, and a giant flagship. As the player, I explore a system of the Imperium with a loyal crew of companions, making impactful choices and
determining the fate of those around me.

Rogue Trader is a computer role-playing game in the vein of Baldur’s Gate 3 or Pillars of Eternity.
</item 1>

<item 2>
The best video games of the year so far (Polygon, 2023-10-31T16:30:45+00:00)
—Mike Mahardy

How the Polygon top 50 list works

Over the past few weeks, the Polygon staff voted, championed, debated, and ultimately threw up its hands and marveled at the list of mammoths, curiosities,
puzzle boxes, and black holes that is our top 50 games of 2023. Any video games that were released in 2023, received substantial updates in 2023, or achieved
renewed cultural relevance in 2023 were eligible for this list. Last year, the cutoff for consideration was Nov. 30. (You’ll notice a certain Firaxis Games
joint fairly high up our list.) This year, the cutoff was the same. Should we be thoroughly enamored with Warhammer 40,000: Rogue Trader or Avatar: Frontiers of
Pandora, we’ll make sure they’re considered for next year’s top 50.

Top 50

50. Mr. Sun’s Hatbox

Developer: Kenny Sun

Where to play: Nintendo Switch and Windows PC

Mr.
</item 2>

<item 3>
The best tabletop RPGs we played in 2023 (Polygon, 2023-12-13T15:00:00+00:00)
But the fresh grand dame of my tabletop world is Triangle Agency, a gleaming, razor-sharp new TTRPG that feels like a cross between The X-Files, Annihilation,
and playing Control with the Robert Anton Wilson mod turned on.

In Triangle Agency, you are super-abled agents investigating and stopping ontological anomalies from destroying reality, with mechanics that feel like the
platonic ideal mix of analytic crunch and grand narrative weirdness. And my god, it is gorgeous, with jaw-dropping art and design that puts it on the same shelf
as Wanderhome and Mörk Borg. The full game ships early in 2024, but you can get the digital rules and start playing right now, for the low, low price of
“whatever you think is appropriate.”

Twilight: 2000

Charlie Hall, senior editor, tabletop

I firmly believe that tabletop role-play should always take into account the audience that sits down at the table to play. That’s part of the reason why I hate
best-of lists for this particular category of games in particular.
</item 3>

<item 4>
Rogue Trader’s role-playing embraces the brutality and freedom of Warhammer 40K (Polygon, 2023-12-06T13:00:00+00:00)
The victory tasted like ash in my mouth after all that frustration.

Voidship combat is another aspect that feels clunky and frustrating. Like the squad-based skirmishes, naval encounters are also turn-based, where positioning is
ultra important. Space naval battles should feel tense, but instead, I’m mostly annoyed at having to continually rotate my ship and set up my zones of attack. I
wish I could delegate these annoyances to my Senechal — to delegate the duties of character leveling and ship combat, the better to appreciate all of the things
Rogue Trader is doing so well.

Rogue Trader is a dense, vast game, and much of it has clearly been crafted with love for the expansive lore of the 40K canon. While there are small annoyances
and clunky features along the way, the political intrigue, cast of characters, and moral choices have me hooked.
</item 4>

<item 5>
Rogue Trader’s role-playing embraces the brutality and freedom of Warhammer 40K (Polygon, 2023-12-06T13:00:00+00:00)
Dogmatic choices exult the God-Emperor and loathe the mutant and the witch, the Heretical options pursue corruptive power, and the Ionoclast path is the closest
thing we have to modern-day “good guy” morality.

After the events of the tutorial, my beautiful voidship is run down and my staff is struggling. As the newly anointed Rogue Trader, I have to trek around the
various planets of the Koronus Expanse to get a new Navigator, fix my ship up, and avoid any major diplomatic incidents with the locals. My voidship is the size
of a modern city; I’m as much a governor as a captain, and I have to manage the ship, its cargo, and its many occupants.

In the process of getting back on my feet, I uncover a nefarious cult and a deep conspiracy. It’s not an easy job, but I’m blessed with a handful of companions
from the Imperium to help me out.
</item 5>

<item 6>
Rogue Trader’s role-playing embraces the brutality and freedom of Warhammer 40K (Polygon, 2023-12-06T13:00:00+00:00)
He’s cruel, depraved, and an absolute hoot. Yes, he may flay a few too many people for my tastes, but he’s the best companion to bring to a party.

In the grand scheme of things, this is one of the most complete and detailed explorations of the 40K universe you can find. The game is an homage to the
Warhammer 40K RPGs from Fantasy Flight Games, including Rogue Trader and Dark Heresy. I spent years as a teenager and young adult poring over these sourcebooks
(which Rogue Trader is inspired by), learning more about the tiny details of life in the Imperium. Developer Owlcat Games has paid the same attention to every
detail of the Koronus Expanse. I delve into ancient facilities staffed by tech-priests of the Machine God, the hostile xenos city of Commorragh, or massive
cities built to honor the God-Emperor of Mankind.
</item 6>

<item 7>
The best tabletop RPGs we played in 2023 (Polygon, 2023-12-13T15:00:00+00:00)
Players take on the role of soldiers and other hangers-on just trying to make it out alive. The heady mix of dice-driven gunplay and a playing-card-style
encounter creation makes it easy to run. But the fun begins even before that with character generation, which uses old school randomized tables to simulate an
entire military career. If you’re looking to get your video gaming buddies around the table for something that doesn’t require a headset or shouting down
pre-teens in a glitchy public lobby, the all-inclusive starter set comes highly recommended.

Project ECCO

It’s a near-impossible task to get a consistent group together, so solo TTRPGs have become my primary method of gaming. However, the barrier for me is finding a
game with a balance of narrative and mechanics that isn’t overwhelmingly crunchy, but also isn’t open-ended to the point of choice paralysis. For me, Project
ECCO found that balance.
</item 7>

<item 8>
Rogue Trader’s role-playing embraces the brutality and freedom of Warhammer 40K (Polygon, 2023-12-06T13:00:00+00:00)
For 40K fans, this is a rare treat — a game that digs past the heroic facade of bolters and battles and taps into the grimdark dystopia that makes this
particular sci-fi setting so damn compelling.

Warhammer 40,000: Rogue Trader will be released on Dec. 7 on Windows PC. The game was reviewed using a pre-release download code provided by Owlcat Games. Vox
Media has affiliate partnerships. These do not influence editorial content, though Vox Media may earn commissions for products purchased via affiliate links.
You can find additional information about Polygon’s ethics policy here .
</item 8>

<item 9>
Rogue Trader’s role-playing embraces the brutality and freedom of Warhammer 40K (Polygon, 2023-12-06T13:00:00+00:00)
I create my custom protagonist, determining stats and bonuses based on my backstory, home world, and so on. I’m then thrust into a high-pressure job interview
as a potential heir to a Rogue Trader aboard her flagship. Little do we know that there’s a coup in the works, and traitors aboard the ship. After a deadly
struggle against heretics and demons, I ascend to the position of Rogue Trader. I’m the captain now, and I get to decide how to run my ship.

This is a vast game, with tons of features you’d expect from a CRPG — companions (each with their own narrative paths and conversation trees), top-down
strategic combat, and branching choices that impact the world around you. I get to make choices all day — in fact, that’s part of the gig of a Rogue Trader.
Some are more important than others; whenever I get to specific points in the story, I can choose from one of three major paths.
</item 9>

<item 10>
Rogue Trader’s role-playing embraces the brutality and freedom of Warhammer 40K (Polygon, 2023-12-06T13:00:00+00:00)
I can call on them to unlock doors or perform other environmental checks, but they come most in handy in combat.

When I meet opposition, it’s deeply satisfying to control my troops in a turn-based battle. Each fight takes place on a grid; it’s very similar to Baldur’s Gate
3 or even XCOM 2. Some positions provide cover, while others are out in the open. Friendly fire is also a very real concern. An arc of auto-fire from a bolter,
or a Navigator’s third eye opening, can harm friend as well as foe. My Rogue Trader is a sniper, and she would be lost without her Senechal taking the front
lines. There’s a lot of firepower at my disposal, and it’s mostly quite satisfying to use — even if I occasionally shred my poor Senechal with a devastating
AOE.

I can chat with the companions between battles, learning more about their pasts.
</item 10>
=== END ITEM id=7b65f35b2fc2c09e ===
