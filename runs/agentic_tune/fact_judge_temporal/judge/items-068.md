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

=== ITEM id=82ccf3c0a5465716 ===
Gold sentences and facts:
Sentence 0: Despite the suspension, Jim Harbaugh held his usual weekly news conference Monday, though he mostly avoided addressing the only topic on anyone's mind.
  Fact 0.0: Jim Harbaugh was suspended.
  Fact 0.1: Despite the suspension, Jim Harbaugh held his usual weekly news conference on Monday.
  Fact 0.2: Jim Harbaugh mostly avoided addressing the only topic on anyone's mind.
Sentence 1: It would be Harbaugh's second suspension of the season, as the Wolverines self-imposed a three-game ban on Harbaugh to start the season amid an NCAA investigation that found Harbaugh lied to investigators.
  Fact 1.0: The suspension in question would be Harbaugh's second suspension of the season.
  Fact 1.1: The Wolverines self-imposed a three-game ban on Harbaugh to start the season.
  Fact 1.2: The three-game ban was imposed amid an NCAA investigation.
  Fact 1.3: The NCAA investigation found that Harbaugh lied to investigators.
Sentence 2: Harbaugh reportedly watched the game from the team hotel, and now the leading piece of drama returns.
  Fact 2.0: Harbaugh reportedly watched the game from the team hotel.
  Fact 2.1: The leading piece of drama returns now.

Retrieved items:
<item 1>
Michigan sign-stealing scandal news: Latest updates on possible Jim Harbaugh suspension from NCAA cheating investigation (Sporting News,
2023-11-06T23:58:04+00:00)
The Conference has confirmed that it is not aware of any information suggesting Coach Harbaugh's involvement in the allegations. The University continues to
cooperate fully with the NCAA's investigation.

From the Big Ten:

"The Big Ten Conference's commitment to student-athletes, sportsmanship and the commissioner's duty to protect the integrity of competition will never waver.
Today's decision by the University of Michigan to withdraw its legal challenge against the conference's November 10th notice of disciplinary action is
indicative of the high standards and values that the conference and the university seek to uphold, the University of Michigan is a valued member of the Big Ten
Conference and the conference will continue to work cooperatively with the university and the NCAA during this process."

Monday, Nov. 13

Harbaugh says he'll state his case at Friday hearing

Despite the suspension, Jim Harbaugh held his usual weekly news conference Monday, though he mostly avoided addressing the only topic on anyone's mind.

Graph facts:
- Cowboys vs. 49ers live score, updates, highlights from NFL 'Sunday Night Football' game -MENTIONS-> Jim Harbaugh
- Michigan sign-stealing scandal winners and losers, from Jim Harbaugh to Tony Petitti -MENTIONS-> Jim Harbaugh
- Michigan sign-stealing scandal winners and losers, from Jim Harbaugh to Tony Petitti -MENTIONS-> sign-stealing scandal
- Michigan conquers Ohio State again, but the story is far from over for 2023 Wolverines -MENTIONS-> Jim Harbaugh
- Michigan didn't need Jim Harbaugh against Penn State, but Ohio State is a different game -MENTIONS-> Jim Harbaugh

Entities: sign-stealing scandal, Jim Harbaugh, Big Ten Conference, University of Michigan, University of Michigan legal challenge withdrawal, University of
Michigan notice of disciplinary action, Jim Harbaugh news conference, Jim Harbaugh hearing, cheating investigation, Michigan sign-stealing scandal news: Latest
updates on possible Jim Harbaugh suspension from NCAA cheating investigation, Sporting News, NCAA investigation, NCAA
</item 1>

<item 2>
Michigan didn't need Jim Harbaugh against Penn State, but Ohio State is a different game (Sporting News, 2023-11-11T21:32:44+00:00)
Will Harbaugh be on the field before the end of the regular season? The Big Ten made a Friday ruling that will prohibit Harbaugh from being on the field for the
final three regular-season games as a result of the in-person scouting and sign-stealing scandal that led to the resignation of staffer Conor Stallions. ESPN
reported there will be a temporary restraining order hearing on Nov. 17 that will answer that question.

On the field, Michigan answered the question about its College Football Playoff. Sort of. The Wolverines proved they could beat Penn State (8-2, 5-2) in a
20-yard box. Neither team completed a pass of 20 yards or more. Michigan trailed 3-0 after the first quarter. The Wolverines had eight carries for 10 yards, and
Penn State's Abdul Carter set the tone with a jarring hit on Blake Corum that nearly resulted in a fumble.

Michigan took control with the running game.

Graph facts:
- USC basketball futures odds to win Pac-12, NCAA tourney ahead of Bronny James collegiate debut -MENTIONS-> Michigan
- After a Stunning Snub, College Football Is Facing an Identity Crisis -MENTIONS-> Michigan
- 2024 NFL Mock Draft: Bengals' Joe Burrow gets another playmaker; four QBs go in Round 1, but Pats opt for WR -MENTIONS-> Michigan
- Michigan sign-stealing scandal news: Latest updates on possible Jim Harbaugh suspension from NCAA cheating investigation -MENTIONS-> Michigan
- Michigan sign-stealing scandal winners and losers, from Jim Harbaugh to Tony Petitti -MENTIONS-> Michigan

Entities: Michigan, Penn State, temporary restraining order hearing, Big Ten, Blake Corum, ESPN, Jim Harbaugh, Ohio State, Conor Stallions, College Football
Playoff, Sporting News, in-person scouting and sign-stealing scandal, Abdul Carter, Michigan-Penn State game, Michigan didn't need Jim Harbaugh against Penn
State, but Ohio State is a different game
</item 2>

<item 3>
Graph query row:
article: Michigan sign-stealing scandal winners and losers, from Jim Harbaugh to Tony Petitti
date: 2023-11-11T22:50:13+00:00
report: Michigan sign-stealing scandal winners and losers, from Jim Harbaugh to Tony Petitti (Sporting News, 2023-11-11T22:50:13+00:00)
The Big Ten announced Michigan coach Jim Harbaugh will be prohibited from coaching on the sideline for the remainder of the regular season.

At least, in theory. The Friday ruling is the next chapter in an NCAA investigation into Michigan for in-person scouting and sign stealing in a scandal that led
to the resignation of staffer Connor Stallions on Nov. 3. Harbaugh will be allowed to coach during the week but will not be able to be on the sideline. Michigan
did not get a ruling on a temporary restraining order ahead of the showdown against No. 10 Penn State.

The Wolverines still won 24-15 under interim coach Sherrone Moore, but the fallout from the scandal continues to have a far-reaching impact on the Big Ten,
Michigan and college football beyond this year's College Football Playoff race.

A look at the winners and losers from the scandal to date:

Michigan scandal winners and losers

Winners

Tony Petitti

Petitti certainly did not hit Michigan with a "slap on the wrist."
</item 3>

<item 4>
Michigan sign-stealing scandal news: Latest updates on possible Jim Harbaugh suspension from NCAA cheating investigation (Sporting News,
2023-11-06T23:58:04+00:00)
OFFICIAL: Jim Harbaugh suspended for the remaining three regular season games.



"Michigan has been found in violation of the Big Ten Sportsmanship Policy for conducting an impermissible, in-person scouting operation over multiple years..."
https://t.co/htnUkshmnF — Chris Vannini (@ChrisVannini) November 10, 2023

Jim Harbaugh joins Michigan for trip to State College

According to the sleuths on the ground in Ann Arbor, Harbaugh is on the team bus and heading to the airport with his team as Michigan awaits word on a potential
suspension.

Jim Harbaugh boarding the bus pic.twitter.com/CWBJbwEEhi — Austin Meek (@byAustinMeek) November 10, 2023

Michigan still awaiting word from Big Ten

According to ESPN's Adam Rittenberg, Michigan still hasn't heard about an impending punishment from the Big Ten as its 1 p.m. ET flight looms.

Graph facts:
- Michigan sign-stealing scandal winners and losers, from Jim Harbaugh to Tony Petitti -MENTIONS-> Michigan
- After a Stunning Snub, College Football Is Facing an Identity Crisis -MENTIONS-> Michigan
- 2024 NFL Mock Draft: Bengals' Joe Burrow gets another playmaker; four QBs go in Round 1, but Pats opt for WR -MENTIONS-> Michigan
- USC basketball futures odds to win Pac-12, NCAA tourney ahead of Bronny James collegiate debut -MENTIONS-> Michigan
- Michigan conquers Ohio State again, but the story is far from over for 2023 Wolverines -MENTIONS-> Michigan

Entities: Michigan, Big Ten, ESPN, Jim Harbaugh suspension, Jim Harbaugh, Chris Vannini, Big Ten Sportsmanship Policy, impermissible, in-person scouting
operation, Austin Meek, Ann Arbor, State College, Adam Rittenberg, NCAA cheating investigation, Michigan sign-stealing scandal news: Latest updates on possible
Jim Harbaugh suspension from NCAA cheating investigation, Sporting News
</item 4>

<item 5>
Michigan sign-stealing scandal news: Latest updates on possible Jim Harbaugh suspension from NCAA cheating investigation (Sporting News,
2023-11-06T23:58:04+00:00)
Although the Big Ten has closed its investigation, we are continuing to cooperate with the NCAA as it moves forward with its ongoing investigation. Consistent
with our commitment to integrity, we will continue to take the appropriate actions, including disciplinary measures, based on information we obtain. Earlier
today, Michigan Athletics relieved Chris Partridge of his duties as a member of the Michigan Football staff. Due to employee privacy laws, we are unable to
comment further."

Because Michigan and Harbaugh on Thursday dropped their litigation against the Big Ten, he will not be on Michigan's sideline this week against Maryland and
next week against Ohio State to end the regular season.

Former Wolverines assistant and Harbaugh confidant Biff Poggi on Friday offered his former boss an invitation. Poggi, in his first year as Charlotte's head
coach, asked Harbaugh to join the 49ers for their games vs. Rice on Saturday and at South Florida on Nov. 25.

Graph facts:
- Michigan sign-stealing scandal winners and losers, from Jim Harbaugh to Tony Petitti -MENTIONS-> Michigan
- After a Stunning Snub, College Football Is Facing an Identity Crisis -MENTIONS-> Michigan
- 2024 NFL Mock Draft: Bengals' Joe Burrow gets another playmaker; four QBs go in Round 1, but Pats opt for WR -MENTIONS-> Michigan
- USC basketball futures odds to win Pac-12, NCAA tourney ahead of Bronny James collegiate debut -MENTIONS-> Michigan
- Michigan conquers Ohio State again, but the story is far from over for 2023 Wolverines -MENTIONS-> Michigan

Entities: Michigan, sign-stealing scandal, Maryland, Big Ten, Chris Partridge, Michigan Football, Biff Poggi, Charlotte, South Florida, Jim Harbaugh, NCAA
cheating investigation, Michigan sign-stealing scandal news: Latest updates on possible Jim Harbaugh suspension from NCAA cheating investigation, Michigan
Athletics, Ohio State, Sporting News
</item 5>

<item 6>
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
</item 6>

<item 7>
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
</item 7>

<item 8>
Michigan sign-stealing scandal news: Latest updates on possible Jim Harbaugh suspension from NCAA cheating investigation (Sporting News,
2023-11-06T23:58:04+00:00)
ET flight looms.

Michigan's administration has not heard from the Big Ten as of this hour, per sources. The team will begin its preparations to leave for Penn State around 1
p.m. ET. Really going down to the wire with any discipline, which, as several sources noted, isn't really fair to players. — Adam Rittenberg (@ESPNRittenberg)
November 10, 2023

Harbaugh still expected to fly to State College

Dinich reported earlier on Friday that punishment to Michigan could come around the time the team is expected to get on a plane and fly to Pennsylvania for its
game against Penn State. But regardless of when the punishment is announced, Harbaugh's attorney, Tom Mars, reportedly told ESPN the coach will be headed to
State College.

"I'm confident he's going to be on that plane one way or another," Mars reportedly told ESPN.

Graph facts:
- Michigan sign-stealing scandal winners and losers, from Jim Harbaugh to Tony Petitti -MENTIONS-> Michigan
- After a Stunning Snub, College Football Is Facing an Identity Crisis -MENTIONS-> Michigan
- 2024 NFL Mock Draft: Bengals' Joe Burrow gets another playmaker; four QBs go in Round 1, but Pats opt for WR -MENTIONS-> Michigan
- USC basketball futures odds to win Pac-12, NCAA tourney ahead of Bronny James collegiate debut -MENTIONS-> Michigan
- Michigan conquers Ohio State again, but the story is far from over for 2023 Wolverines -MENTIONS-> Michigan

Entities: Michigan, sign-stealing scandal, Penn State, Big Ten, suspension, Jim Harbaugh, State College, game against Penn State, Tom Mars, Dinich, Adam
Rittenberg, cheating investigation, Michigan sign-stealing scandal news: Latest updates on possible Jim Harbaugh suspension from NCAA cheating investigation,
ESPN, Sporting News
</item 8>

<item 9>
Michigan sign-stealing scandal news: Latest updates on possible Jim Harbaugh suspension from NCAA cheating investigation (Sporting News,
2023-11-06T23:58:04+00:00)
pic.twitter.com/Qm9iehYVTR — Pete Thamel (@PeteThamel) November 11, 2023

Harbaugh will not be on the sidelines vs. Penn State

Michigan filed a temporary restraining order against the Big Ten on Friday night, hoping to block Harbaugh's suspension. But there will not be a ruling before
the Wolverines' game vs. Penn State, with the hearing scheduled for Friday, Nov. 17.

No ruling on Michigan’s TRO filing today, so Jim Harbaugh will not coach vs. Penn State.



Statement from the school: “We look forward to presenting our case next week where we intend to demonstrate that the Big Ten has not acted legally or fairly.”

Graph facts:
- Michigan sign-stealing scandal winners and losers, from Jim Harbaugh to Tony Petitti -MENTIONS-> Michigan
- After a Stunning Snub, College Football Is Facing an Identity Crisis -MENTIONS-> Michigan
- 2024 NFL Mock Draft: Bengals' Joe Burrow gets another playmaker; four QBs go in Round 1, but Pats opt for WR -MENTIONS-> Michigan
- USC basketball futures odds to win Pac-12, NCAA tourney ahead of Bronny James collegiate debut -MENTIONS-> Michigan
- Michigan conquers Ohio State again, but the story is far from over for 2023 Wolverines -MENTIONS-> Michigan

Entities: Michigan, Penn State, Big Ten, Jim Harbaugh suspension, Jim Harbaugh, NCAA cheating investigation, Michigan game vs. Penn State, temporary restraining
order filing, Michigan sign-stealing scandal news: Latest updates on possible Jim Harbaugh suspension from NCAA cheating investigation, Pete Thamel, hearing,
Sporting News, NCAA
</item 9>

<item 10>
Michigan sign-stealing scandal news: Latest updates on possible Jim Harbaugh suspension from NCAA cheating investigation (Sporting News,
2023-11-06T23:58:04+00:00)
We have to wait for the news, but a suspension is certainly on the table. — Heather Dinich (@CFBHeather) November 10, 2023

If it’s more than 2 games or $10k, the Big Ten will need special permission from an executive board composed of league reps https://t.co/6Z7jb2zqH4 — Heather
Dinich (@CFBHeather) November 10, 2023

Punishment expected on Friday

The college football world will wait with bated breath on Friday as it awaits what Michigan and Jim Harbaugh will face from the Big Ten. ESPN's Heather Dinich
reported the punishment is expected to be announced early Friday afternoon. She noted the team is leaving for State College at 1 p.m. CT for the Penn State game
on Saturday.

sources indicate a Big Ten response is coming early this afternoon.

Graph facts:
- Michigan sign-stealing scandal winners and losers, from Jim Harbaugh to Tony Petitti -MENTIONS-> Michigan
- After a Stunning Snub, College Football Is Facing an Identity Crisis -MENTIONS-> Michigan
- 2024 NFL Mock Draft: Bengals' Joe Burrow gets another playmaker; four QBs go in Round 1, but Pats opt for WR -MENTIONS-> Michigan
- USC basketball futures odds to win Pac-12, NCAA tourney ahead of Bronny James collegiate debut -MENTIONS-> Michigan
- Michigan conquers Ohio State again, but the story is far from over for 2023 Wolverines -MENTIONS-> Michigan

Entities: Michigan, college football, Michigan sign-stealing scandal, Big Ten, ESPN, Jim Harbaugh, NCAA cheating investigation, State College, executive board,
Heather Dinich, punishment announcement, Penn State game, Michigan sign-stealing scandal news: Latest updates on possible Jim Harbaugh suspension from NCAA
cheating investigation, Sporting News, NCAA
</item 10>
=== END ITEM id=82ccf3c0a5465716 ===

=== ITEM id=e25ac877c72684c3 ===
Gold sentences and facts:
Sentence 0: As an added bonus, Harris is also the preferred goal-line back, scoring in three of those games.
  Fact 0.0: Harris is also the preferred goal-line back.
  Fact 0.1: Being the preferred goal-line back is an added bonus for Harris.
  Fact 0.2: Harris scored in three of those games.
Sentence 1: The Wildcats on their first drive put a touchdown on the board a seven play, 61 yard drive, thanks to a 25-yard dash before diving into the end zone by running back Jonah Coleman.
  Fact 1.0: The Wildcats scored a touchdown on their first drive.
  Fact 1.1: The first drive was a seven-play, 61-yard drive.
  Fact 1.2: Running back Jonah Coleman made a 25-yard dash on the drive before diving into the end zone.

Retrieved items:
<item 1>
Graph query row:
report: No. 17 Arizona dominates No. 22 Utah, 42-18
published_on: 2023-11-18T23:45:32+00:00
passage: No. 17 Arizona dominates No. 22 Utah, 42-18 (Rivals, 2023-11-18T23:45:32+00:00)
Saturday’s game between Arizona and Utah was many things for the Wildcats: Senior Day, Military Appreciation Day, and their last home game of the season. With
the UA needing a win to keep its own destiny alive for a Pac-12 Championship appearance, it handled the pressure once again against a ranked team.

The Wildcats not only got halfway through en route to winning out but did it in statement fashion Saturday at Arizona Stadium in front of a sold out stadium,
blowing out No. 22 Utah, 42-18, extending their winning streak to five games, as well as marking their first win over the Utes since 2015. It is also the first
time the UA took down four ranked teams in a season since 1989, with it being the only team in the country this season to do so.

"Great win for the program," Arizona head coach Jedd Fisch said following the win.
</item 1>

<item 2>
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
</item 2>

<item 3>
Patriots vs. Steelers FanDuel Picks: NFL DFS lineup advice for Week 14 TNF single-game tournaments (Sporting News, 2023-12-07T21:26:00+00:00)
Expect the RB to be fed a ton. Whether that results in a touchdown remains to be seen, but if anyone is scoring a TD on the Pats roster Thursday night, it's
Elliott.

FLEX RB Najee Harris, Steelers ($14,000). Much like the Patriots, this Steelers offense will want to establish the ground game, and that starts with Harris. It
wasn't long ago that the RB was the subject of major criticism for looking slow and ineffective, but that narrative has flipped in a big way. With an average of
15 carries per game in the past five, we can expect at least that and maybe more from a Pittsburgh team that is starting Mitch Trubisky on Thursday night. As an
added bonus, Harris is also the preferred goal-line back, scoring in three of those games. The Patriots' defense doesn't give up much, making Harris our best
bet for success in an old-fashioned defensive struggle between two of the worst offenses the NFL has to offer.
</item 3>

<item 4>
NFL Fantasy Football Week 5 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-04T21:56:25+00:00)
At least all of them went to his wideouts. All were in the red zone, none were on deep throws and none were on WR screens.

Sit Him (Lineup Decisions) Projections powered by Sportsline Najee Harris RB PIT Pittsburgh • #22

Age: 25 • Experience: 3 yrs. Weekly Breakdown MATCHUP vs BAL BAL -4 O/U 38 OPP VS RB 12th PROJ PTS 9.3 RB RNK 30th YTD Stats RUYDS 210 REC 4 REYDS 34 TD 0
FPTS/G 7.1 The Steelers' lack of scoring opportunities really crushes the upside for Harris, who admittedly has improved this season and could be on his way to
being a reliable RB2. The Ravens have been pretty good against the run but they can give up some numbers. Harris is a volume-based back who is at best a low-end
starting option given everything that's going on.

Graph facts:
- Brock Purdy -MENTIONS-> NFL
- Raiders vs. Lions live score, updates, highlights from NFL 'Monday Night Football' game -MENTIONS-> NFL
- Fans spot Travis Kelce wearing Taylor Swift-themed friendship bracelet before she attended his game -MENTIONS-> NFL
- Chiefs vs. Packers live score, updates, highlights from NFL 'Sunday Night Football' game -MENTIONS-> NFL
- Christmas Day preview: 49ers, Ravens square off in potential Super Bowl sneak peek -MENTIONS-> NFL

Entities: NFL, NFL Fantasy Football Week 5 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game, CBSSports.com, fantasy football, Ravens,
Najee Harris, Pittsburgh, SportsLine, Steelers
</item 4>

<item 5>
NFL Fantasy Football Week 5 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-04T21:56:25+00:00)
TOUCHDOWNS: Harris has zero. Somehow it gets worse: He has one carry inside the 15-yard line. The Steelers have run the second-fewest red-zone plays (18, one
more than New England) and second-fewest plays inside the 5-yard line (two, one more than New England). The Steelers have played four games! This is nuts.

RAVENS: Have yet to allow a rushing score to a running back this year. They've been otherwise solid against RBs, holding them to 4.0 yards per carry on the
season and roughly league-average in other metrics including missed tackles (15).
</item 5>

<item 6>
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

Graph facts:
- Raiders vs. Lions live score, updates, highlights from NFL 'Monday Night Football' game -MENTIONS-> Raiders
- Kelce -MENTIONS-> Raiders
- NFL Fantasy Football Week 5 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game -MENTIONS-> Raiders
- 2023 NFL Week 11 injury roundup: Rams WR Cooper Kupp, Seahawks RB Kenneth Walker III among stars to exit -MENTIONS-> Raiders
- Christmas Day preview: 49ers, Ravens square off in potential Super Bowl sneak peek -MENTIONS-> Raiders

Entities: Raiders, Jaguars, Vikings, Alexander Mattison, Bills, Titans, Bengals, Packers, Texans, Dolphins, Eagles, fantasy football, PPR, Ravens, running back
</item 6>

<item 7>
NFL Fantasy Football Week 5 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-04T21:56:25+00:00)
I'd rather start Breece Hall, Rhamondre Stevenson, Dameon Pierce and Jaleel McLaughlin (if Javonte Williams sits out).

LAST WEEK: Harris had his first game with over 100 total yards against the Texans and averaged 5.1 yards per rush, just shy of the season-high he technically
had on six carries against the 49ers in Week 1. A 32-yard reception helped him cross the triple-digit barrier.

HARRIS: This might shock you, but Harris is averaging 4.3 yards per run this year with a 12.2% explosive run rate. Would you believe the explosive run rate,
which reflects the frequency of his 12-plus-yard runs, is fifth-best among 44 qualifying running backs?! He's also 13th in avoided tackle rate and top-10 in
yards after contact per carry at 3.53.

TOUCHDOWNS: Harris has zero.

Graph facts:
- UTSA vs. Marshall odds, props, predictions: Roadrunners lay big spread in Frisco Bowl -MENTIONS-> Harris
- Biden’s AI EO hailed as broad, but not deep without legislation to match -MENTIONS-> Harris
- Bryce Harper 'Atta Boy' controversy, explained: Why Orlando Arcia's comments led to backlash, Alanna Rizzo rant -MENTIONS-> Harris
- Harris -PARTICIPATED_IN-> Week 17 win
- Harris -PARTICIPATED_IN-> Week 14 game

Entities: Harris, NFL Fantasy Football Week 5 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game, Rhamondre Stevenson, CBSSports.com,
Texans, Jaleel McLaughlin, Breece Hall, Javonte Williams, Dameon Pierce, 49ers
</item 7>

<item 8>
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

Graph facts:
- Raiders vs. Lions live score, updates, highlights from NFL 'Monday Night Football' game -MENTIONS-> Raiders
- Kelce -MENTIONS-> Raiders
- NFL Fantasy Football Week 5 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game -MENTIONS-> Raiders
- 2023 NFL Week 11 injury roundup: Rams WR Cooper Kupp, Seahawks RB Kenneth Walker III among stars to exit -MENTIONS-> Raiders
- Christmas Day preview: 49ers, Ravens square off in potential Super Bowl sneak peek -MENTIONS-> Raiders

Entities: Raiders, Jaguars, Vikings, Alexander Mattison, Saints, Bills, Titans, Bengals, Packers, Texans, Dolphins, fantasy football, Ravens, running back,
Kareem Hunt
</item 8>

<item 9>
Fantasy Football Week 13 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-11-30T22:10:58+00:00)
With Pickett nursing an ankle injury we could see the Steelers run the ball aplenty.

STARTS: Najee Harris, Kyler Murray, Trey McBride, Marquise Brown, Pat Freiermuth, Diontae Johnson (PPR flex), Jaylen Warren (PPR only), Steelers DST

Najee Harris, Kyler Murray, Trey McBride, Marquise Brown, Pat Freiermuth, Diontae Johnson (PPR flex), Jaylen Warren (PPR only), Steelers DST SITS: Kenny
Pickett, James Conner, Rondale Moore, Cardinals DST

Kenny Pickett, James Conner, Rondale Moore, Cardinals DST SLEEPERS: Greg Dortch (PPR only), George Pickens

Sit Him (Lineup Decisions) Projections powered by Sportsline James Conner RB ARI Arizona • #6

Age: 28 • Experience: 7 yrs.

Graph facts:
- Fantasy Football Week 13 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game -MENTIONS-> Brock Purdy
- Fantasy Football Week 13 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game -MENTIONS-> Mark Andrews
- Fantasy Football Week 13 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game -MENTIONS-> Alex Anzalone
- Fantasy Football Week 13 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game -MENTIONS-> NFL
- Fantasy Football Week 13 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game -MENTIONS-> Jordan Love

Entities: Fantasy Football Week 13 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game, Lineup Decisions, CBSSports.com, Rondale Moore,
Cardinals, fantasy football, Marquise Brown, Greg Dortch, ankle injury, Diontae Johnson, George Pickens, James Conner, Jaylen Warren, Najee Harris, Kenny
Pickett
</item 9>

<item 10>
Fantasy Football RB PPR Rankings Week 14: Who to start, best sleepers at running back (Sporting News, 2023-12-07T22:09:00+00:00)
We still like Jaylen Warren better overall, but you can confidently take Najee Harris (vs. Patriots) out of the doghouse for good. Harris has been running with
purpose, and his role in the passing game continues to make him a high-ceiling RB2 in a run-first offense that will be without Kenny Pickett (ankle) this week.
New England might be banged up, but it's still a tough D to get chunk yardage against through the air. Look for Harris to see lots of screens and dump-downs in
a rough-and-tumble Thursday night game in Pittsburgh.

We love Baltimore's Keaton Mitchell (vs. Rams) and think he has the potential to smash Los Angeles at home this week.

Graph facts:
- Fantasy Football Week 13 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game -MENTIONS-> fantasy football
- NFL Fantasy Football Week 5 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game -MENTIONS-> fantasy football
- Fantasy Football WR PPR Rankings Week 13: Who to start, best sleepers at wide receiver -MENTIONS-> fantasy football
- Fantasy Football Week 16 Wide Receiver Preview: Waiver adds, DFS plays, and more -MENTIONS-> fantasy football
- FanDuel Picks Week 15: Best NFL DFS lineup advice for daily fantasy football tournaments -MENTIONS-> fantasy football

Entities: fantasy football, running back, Sporting News, Rams, Fantasy Football RB PPR Rankings Week 14: Who to start, best sleepers at running back, Jaylen
Warren, Najee Harris, Keaton Mitchell, Kenny Pickett, Los Angeles, Pittsburgh, Patriots, Baltimore
</item 10>
=== END ITEM id=e25ac877c72684c3 ===

=== ITEM id=14dffbfa28c9ba38 ===
Gold sentences and facts:
Sentence 0: The process also loops in the European Commission to help facilitate dialogue, assess issues and bring pressure to bear on unfair practices.
  Fact 0.0: The process also involves the European Commission.
  Fact 0.1: The European Commission helps facilitate dialogue.
  Fact 0.2: The European Commission helps assess issues.
  Fact 0.3: The European Commission helps bring pressure to bear on unfair practices.
Sentence 1: We continue to work through the process with the European Commission and are focused on addressing its questions and any identified concerns at this stage.
  Fact 1.0: The speaker continues to work through the process with the European Commission.
  Fact 1.1: The speaker is focused on addressing the European Commission's questions.
  Fact 1.2: The speaker is focused on addressing any identified concerns at this stage.
Sentence 2: Its earlier actions were focused on concerns about the spread of illegal content and disinformation related to the Israel-Hamas war.
  Fact 2.0: The European Commission's earlier actions were focused on concerns about the spread of illegal content.
  Fact 2.1: The European Commission's earlier actions were focused on concerns about the spread of disinformation related to the Israel-Hamas war.

Retrieved items:
<item 1>
Amazon’s iRobot purchase sucks up formal competition concerns in EU (TechCrunch, 2023-11-27T23:29:10+00:00)
Today’s announcement by the EU, following several months of in-depth examination looking at how the transaction might impact competition, lays out formal
concerns for Amazon to respond to. The EU’s objections focus on foreclosure risks.

The EU says its concerns focus on whether the deal will give Amazon the ability and incentive to foreclose iRobot’s rivals by deploying strategies aimed at
preventing competitors from selling RVCs on Amazon’s marketplace and/or making it harder for them to do so — such as by delisting rival products; reducing their
visibility in both non-paid (i.e. organic) and paid results (i.e. ads) displayed on its marketplace; limiting their access to certain widgets it operates (such
as the “other products you may like” feature) or “certain commercially-attractive product labels” (e.g. ‘Amazon’s choice’ or ‘Works With Alexa’); and/or by
directly or indirectly raising the costs of iRobot’s rivals to advertise and sell their RVCs on its marketplace.

Graph facts:
- The 26 best Amazon Prime Day deals you can still shop for today -MENTIONS-> iRobot
- iRobot -PARTICIPATED_IN-> Amazon’s iRobot purchase
- iRobot -MENTIONS-> intense competition
- A comprehensive list of 2023 tech layoffs -MENTIONS-> iRobot
- iRobot -PARTICIPATED_IN-> lay off 7% of its workforce, roughly 85 employees

Entities: RVCs, Amazon’s marketplace, Amazon’s choice, Works With Alexa, other products you may like, foreclosure risks, iRobot, EU, Amazon’s iRobot purchase
sucks up formal competition concerns in EU, TechCrunch, Alexa, competition, Amazon
</item 1>

<item 2>
Amazon’s iRobot purchase sucks up formal competition concerns in EU (TechCrunch, 2023-11-27T23:29:10+00:00)
The EU has reached this pass of laying out formal concerns after conducting what it describes as a “wide-ranging investigation to understand the market and the
potential impact of the deal”, which it says included analysing internal documents provided by Amazon and iRobot and gathering views from market participants
such as suppliers of RVCs and other smart home devices, as well as from providers of online sales channels.

It also says it closely cooperated with other competition authorities during both the initial investigation and the in-depth investigation, adding that it will
continue to do so during the remainder of the in-depth investigation. Although the U.K.’s competition authority already cleared the deal back in June.

Amazon was contacted for a response to the EU’s statement of objections. A company spokesperson sent us this statement in which it claims iRobot faces “intense
competition” from other RVC suppliers:

We continue to work through the process with the European Commission and are focused on addressing its questions and any identified concerns at this stage.

Graph facts:
- iRobot -PARTICIPATED_IN-> lay off 7% of its workforce, roughly 85 employees
- A comprehensive list of 2023 tech layoffs -MENTIONS-> iRobot
- The 26 best Amazon Prime Day deals you can still shop for today -MENTIONS-> iRobot
- iRobot -PARTICIPATED_IN-> Amazon’s iRobot purchase
- After warning, the EU opens investigation into X over alleged disinformation -MENTIONS-> EU

Entities: iRobot, EU, U.K.’s competition authority, initial investigation, in-depth investigation, deal clearance, intense competition, Amazon’s iRobot purchase
sucks up formal competition concerns in EU, TechCrunch, European Commission, Amazon
</item 2>

<item 3>
Amazon’s iRobot purchase sucks up formal competition concerns in EU (TechCrunch, 2023-11-27T23:29:10+00:00)
European Union regulators have sent Amazon a formal statement of objections setting out competition concerns over its proposed acquisition of robot vacuum
maker, iRobot.

The step does not confirm the EU will seek to block the deal but it shows antitrust regulators do remain concerned. It could therefore push Amazon to offer
remedies to EU authorities to settle the investigation and avoid the risk of the acquisition being blocked.

Amazon has already lowered its acquisition price for iRobot — which was originally an all-cash offer of $1.7 billion back in August 2022 — by 15% owing to
increased debt taken on by its target while regulatory snags delayed closing the deal at the original purchase price.

The bloc has been closely scrutinizing the Amazon-iRobot deal since July when the EU announced its in-depth probe. The Commission said at the time it was
worried the transaction would allow Amazon to restrict competition in the market for robot vacuum cleaners (RVCs) and strengthen its position as online
marketplace provider in a number of ways.

Graph facts:
- The 26 best Amazon Prime Day deals you can still shop for today -MENTIONS-> iRobot
- iRobot -PARTICIPATED_IN-> Amazon’s iRobot purchase
- iRobot -MENTIONS-> intense competition
- A comprehensive list of 2023 tech layoffs -MENTIONS-> iRobot
- iRobot -PARTICIPATED_IN-> lay off 7% of its workforce, roughly 85 employees

Entities: Amazon-iRobot deal, in-depth probe, competition concerns, robot vacuum cleaners, iRobot, Amazon’s iRobot purchase sucks up formal competition concerns
in EU, TechCrunch, European Union, Commission, Amazon
</item 3>

<item 4>
Amazon’s iRobot purchase sucks up formal competition concerns in EU (TechCrunch, 2023-11-27T23:29:10+00:00)
At the time the EU’s then competition chief, Margrethe Vestager, trumpeted the outcome as setting what she dubbed “new rules for how Amazon operates its
business in Europe” — arguing the settlement meant the ecommerce giant could “no longer abuse its dual role” as marketplace operator and own brand retailer; and
further claiming the outcome would ensure “competing independent retailers and carriers as well as consumers will benefit from these changes opening up new
opportunities and choice.”

Given the commitments the EU already extracted from Amazon after those earlier investigations it’s interesting to see the bloc remains concerned over levers the
ecommerce behemoth still has its disposal to potentially influence competitive outcomes on its marketplace vis-à-vis iRobot vs rival robot vacs.

Since then Amazon has also been named a gatekeeper under the EU’s shiny new ex ante competition regulation, the Digital Markets Act — which lists its
marketplace as a core platform service. This designation means Amazon must comply with a series of up-front obligations, which include restrictions on self
preferencing.

Graph facts:
- The 26 best Amazon Prime Day deals you can still shop for today -MENTIONS-> iRobot
- iRobot -MENTIONS-> intense competition
- Commission -SAID-> competition concerns
- A comprehensive list of 2023 tech layoffs -MENTIONS-> iRobot
- iRobot -PARTICIPATED_IN-> lay off 7% of its workforce, roughly 85 employees

Entities: competition concerns, iRobot, EU, Amazon’s iRobot purchase sucks up formal competition concerns in EU, Margrethe Vestager, Amazon’s iRobot purchase,
Amazon marketplace, robot vacs, self preferencing, TechCrunch, settlement, Europe, Amazon, Digital Markets Act
</item 4>

<item 5>
After warning, the EU opens investigation into X over alleged disinformation (TechCrunch, 2023-10-13T00:38:56+00:00)
X owner Elon Musk last projected revenues of about $3 billion for this year. It’s anyone’s guess if that’s accurate now, but as an example, that would result in
a fine of up to $180 million.

Indirectly, an investigation will also be a key way for the public to have significantly more transparency around how X is being used (and abused): since Musk
took Twitter private, the company no longer produces quarterly earnings reports and is not held to account as closely as a result.

EU officials confirmed to TechCrunch that an investigation has not been taken at this point. Clearly, though, yesterday’s development (and the pace at which
regulators are moving) is a strong indication of the bloc’s direction of travel.

The Commission said it is looking at X’s compliance with the DSA across a number of areas — including with regard to its policies and practices regarding
notices on illegal content; complaint handling; risk assessment and measures to mitigate the risks identified.

Graph facts:
- After warning, the EU opens investigation into X over alleged disinformation -MENTIONS-> hate speech
- After warning, the EU opens investigation into X over alleged disinformation -MENTIONS-> Saturday’s attacks
- After warning, the EU opens investigation into X over alleged disinformation -MENTIONS-> terrorist and violent content
- Elon Musk’s X faces first DSA probe in EU over illegal content risks, moderation, transparency and deceptive UX design -MENTIONS-> risk assessment
- After warning, the EU opens investigation into X over alleged disinformation -MENTIONS-> Commission services

Entities: Investigation into X over alleged disinformation, complaint handling, risk assessment, After warning, the EU opens investigation into X over alleged
disinformation, disinformation, illegal content, EU, TechCrunch, X, Elon Musk, Commission, DSA, Twitter
</item 5>

<item 6>
Graph query row:
article: SBF’s trial starts soon, but how did he — and FTX — get here?
date: 2023-10-01T14:00:29+00:00
publisher: TechCrunch
connected_entity: Commodity Futures Trading Commission
page: SBF’s trial starts soon, but how did he — and FTX — get here? (TechCrunch, 2023-10-01T14:00:29+00:00)
passage: SBF’s trial starts soon, but how did he — and FTX — get here? (TechCrunch, 2023-10-01T14:00:29+00:00)
Within two years, more than 80 investors provided about $2 billion in capital to FTX, helping Bankman-Fried propel his vision into a reality. In January 2022,
the company raised $400 million in a Series C round, boosting its valuation to $32 billion. That was its last round of public funding.

The company gained somewhat mainstream recognition with branding deals and partnerships. For example, in 2021 it bought the naming rights for the Miami Heat’s
home arena. FTX also got its name branded on Major League Baseball umpires’ polos, and it partnered with celebrities like Tom Brady and his ex-wife, Gisele
Bündchen, as well as Steph Curry, Shaquille O’Neal and Naomi Osaka, among others. He also had close ties to U.S. regulators and government officials, many of
whom he donated to.
</item 6>

<item 7>
Elon Musk’s X faces first DSA probe in EU over illegal content risks, moderation, transparency and deceptive UX design (TechCrunch, 2023-12-18T13:04:56+00:00)
Per the Commission, the EU’s investigation of X focuses on the following areas and issues:

Compliance with DSA obligations related to countering the dissemination of illegal content in the EU — “ notably in relation to the risk assessment and
mitigation measures adopted by X to counter the dissemination of illegal content in the EU, as well as the functioning of the notice and action mechanism for
illegal content in the EU mandated by the DSA, including in light of X’s content moderation resources

— “ The effectiveness of measures taken by X to combat info manipulation — with a focus on the effectiveness of its “Community Notes” system of crowdsourcing
fact-checking to aggregated views of users, who may vote to label disputed tweets with additional context.

Graph facts:
- After warning, the EU opens investigation into X over alleged disinformation -MENTIONS-> illegal content
- Why Mozilla is betting on a decentralized social networking future -MENTIONS-> illegal content
- EU fires urgent warning at Elon Musk’s X over illegal content and disinformation following Hamas attacks -MENTIONS-> illegal content
- Meta says it’s prioritizing livestreaming checks during Israel-Hamas war -MENTIONS-> illegal content
- The UK just laid out new rules for the internet — it only gets harder from here -MENTIONS-> illegal content

Entities: illegal content, fact-checking, moderation, TechCrunch, EU investigation of X, deceptive UX design, info manipulation, Elon Musk’s X faces first DSA
probe in EU over illegal content risks, moderation, transparency and deceptive UX design, X, EU, Elon Musk, transparency, Community Notes, Commission, DSA
</item 7>

<item 8>
Elon Musk’s X faces first DSA probe in EU over illegal content risks, moderation, transparency and deceptive UX design (TechCrunch, 2023-12-18T13:04:56+00:00)
It’s official! Elon Musk’s X marks the spot of the first confirmed investigation opened by the European Union under its rebooted digital rulebook, the Digital
Services Act (DSA).

Announcing the opening of a “formal proceeding” today, the European Commission said the investigation will look at whether the social networking platform may
have breached rules linked to risk management, content moderation, dark patterns, advertising transparency and data access for researchers.

The opening of a formal DSA investigation on X comes hard on the heels of a complaint against X’s adtech by privacy rights group, noyb — although today’s formal
proceeding is unlikely to be directly linked as the Commission has been probing the platform for months via a flurry of requests for information. Its earlier
actions were focused on concerns about the spread of illegal content and disinformation related to the Israel-Hamas war.

Back in October the Commission sent Musk’s company an “urgent” formal request for information about how it was responding to information risks arising out of
the conflict.

Graph facts:
- The Israel-Hamas War Is Drowning X in Disinformation -MENTIONS-> disinformation
- EU fires urgent warning at Elon Musk’s X over illegal content and disinformation following Hamas attacks -MENTIONS-> disinformation
- How ego and fear fuelled the rise of artificial intelligence -MENTIONS-> disinformation
- One year later, ChatGPT is still alive and kicking -MENTIONS-> disinformation
- After warning, the EU opens investigation into X over alleged disinformation -MENTIONS-> disinformation

Entities: disinformation, illegal content, risk management, content moderation, TechCrunch, European Commission, formal DSA investigation on X, noyb, X’s
adtech, complaint against X’s adtech, dark patterns, advertising transparency, data access for researchers, formal request for information, Elon Musk’s X faces
first DSA probe in EU over illegal content risks, moderation, transparency and deceptive UX design
</item 8>

<item 9>
Amazon’s iRobot purchase sucks up formal competition concerns in EU (TechCrunch, 2023-11-27T23:29:10+00:00)
iRobot, which faces intense competition from other vacuum cleaner suppliers, offers practical and inventive products. We believe Amazon can offer a company like
iRobot the resources to accelerate innovation and invest in critical features while lowering prices for consumers.

Late last year the EU settled with Amazon over two earlier competition probes — one delving into antitrust concerns about Amazon’s use of merchant data to
bolster its own retail business; and a second one looking at how it operated the “Buy Box” and its Prime loyalty program.

In those cases Amazon agreed a series of commitments to end the probes — including saying it would stop using non-public data from its marketplace sellers and
boost transparency for merchants; and pledging to treat all sellers equally, regardless of whether they pay for its logistics services.

Graph facts:
- The 26 best Amazon Prime Day deals you can still shop for today -MENTIONS-> iRobot
- iRobot -MENTIONS-> intense competition
- Commission -SAID-> competition concerns
- A comprehensive list of 2023 tech layoffs -MENTIONS-> iRobot
- iRobot -PARTICIPATED_IN-> Amazon’s iRobot purchase

Entities: competition concerns, iRobot, EU, antitrust concerns, Buy Box, Amazon’s iRobot purchase sucks up formal competition concerns in EU, TechCrunch, Prime,
Amazon
</item 9>

<item 10>
Elon Musk’s X faces first DSA probe in EU over illegal content risks, moderation, transparency and deceptive UX design (TechCrunch, 2023-12-18T13:04:56+00:00)
EU regulators are also set to assess Musk & co’s claims that replacing a formal content moderation function with crowdsourced opinions is a valid approach to
tackling diverse information risks; and, consequently, to look into how much Musk’s gutting of content moderation teams and processes has cost X’s users in
terms of harm exposure.

To say the Commission’s investigation looks fascinating and timely is an understatement.

EU lawmakers can certainly bask in the glow of being prepared for the possibility of an erratic billionaire running a wrecking ball through a major social
platform. But of course they will need to follow through — and it’s clear the real work is just starting. So this probe looks set to be a test of how much
stomach the EU has to actually enforce the rules it loves to lay down on paper.

The investigation may also test Musk’s mettle for what could be an expensive head-on clash with EU regulators.

Graph facts:
- EU fires urgent warning at Elon Musk’s X over illegal content and disinformation following Hamas attacks -MENTIONS-> EU
- Amazon’s iRobot purchase sucks up formal competition concerns in EU -MENTIONS-> EU
- ChatGPT: Everything you need to know about the AI-powered chatbot -MENTIONS-> EU
- Biden’s AI EO hailed as broad, but not deep without legislation to match -MENTIONS-> EU
- After warning, the EU opens investigation into X over alleged disinformation -MENTIONS-> EU

Entities: EU, content moderation, TechCrunch, deceptive UX design, Elon Musk’s X faces first DSA probe in EU over illegal content risks, moderation,
transparency and deceptive UX design, first DSA probe in EU, illegal content risks, X, EU regulators, the Commission, crowdsourced opinions, diverse information
risks, harm exposure, Elon Musk, EU lawmakers
</item 10>
=== END ITEM id=14dffbfa28c9ba38 ===
