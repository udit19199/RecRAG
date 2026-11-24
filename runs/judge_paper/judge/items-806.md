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

This file has 4 items.

=== ITEM id=03d235a4e271fc0a ===
Gold sentences and facts:
Sentence 0: It was a Michigan quarterback that led the Buccaneers to the promised land last time so perhaps lightning strikes twice.
  Fact 0.0: A Michigan quarterback led the Buccaneers to the promised land last time.
  Fact 0.1: Lightning may strike twice.
Sentence 1: Source: The Big Ten is reviewing the lengthy written responses from both Michigan and Jim Harbaugh.
  Fact 1.0: The source for this is the Big Ten.
  Fact 1.1: The Big Ten is reviewing lengthy written responses.
  Fact 1.2: The written responses are from Michigan.
  Fact 1.3: The written responses are from Jim Harbaugh.

Retrieved items:
<item 1>
Michigan sign-stealing scandal winners and losers, from Jim Harbaugh to Tony Petitti (Sporting News, 2023-11-11T22:50:13+00:00)
That ruling showed Petitti is not afraid to take on one of the Big Two within the conference. In a letter to Michigan, Petitti tried to dispel those notions.
Via The Athletic's Nicole Auerbach:

“The Conference does not play favorites among its members, nor does it take actions towards its members based on prejudgment or bias. Failing to act under the
extraordinary circumstances here could lead other Conference members to reasonably conclude that the conference has chosen to favor the University over all
other members.”

Tony Petitti, in letter to Michigan, says that the school initially refused to consent to the Big Ten obtaining the spreadsheet and other documents from the
NCAA related to its investigation.



Also makes point to say he's not playing favorites. pic.twitter.com/OgTvTNY1QI — Nicole Auerbach (@NicoleAuerbach) November 10, 2023

Most speculation pinned a potential Harbaugh suspension at two games knowing that the regular-season finale against No.
</item 1>

<item 2>
Michigan sign-stealing scandal news: Latest updates on possible Jim Harbaugh suspension from NCAA cheating investigation (Sporting News,
2023-11-06T23:58:04+00:00)
— Pete Thamel (@PeteThamel) November 8, 2023

Tuesday, Nov. 7

Michigan says three other Big Ten schools shared decoded Wolverines signals

While Michigan tries to navigate the storm around its program, the Wolverines are also trying to make one thing clear: they're hardly alone in trying to
decipher opponents' signals.

Michigan has shared evidence with the Big Ten and NCAA showing staff members at Ohio State, Purdue and Rutgers shared intelligence on the Wolverines' signals
prior to last year's Michigan-Purdue Big Ten Championship Game, The Athletic reported.

Documents breaking down Michigan's signals for numerous plays were compiled by Purdue staffers ahead of the game based on information provided by Ohio State and
Rutgers football personnel, the report says. The person who supplied the documents told The Athletic he spent 10 to 12 hours per week trying to decode signals
based on game video and said the practice of sharing decoded signals with other schools is common.
</item 2>

<item 3>
Michigan sign-stealing scandal winners and losers, from Jim Harbaugh to Tony Petitti (Sporting News, 2023-11-11T22:50:13+00:00)
On the other, he could emerge as a martyr for Michigan fans given there is reportedly no proof yet he knew of Stalion's rule-breaking.

Harbaugh will have a temporary restraining order hearing on Nov. 17, according to ESPN.com. That will determine whether he returns to the field for the final
two games against Maryland and No. 1 Ohio State.

It's an inescapable fact that Harbaugh is 31-3 – and Michigan is 35-3 – since the program bottomed out at the end of the 2020 season. While it is almost
impossible to quantify how much the alleged in-person scouting and sign-stealing would have helped the Wolverines over the past two seasons, it will be a tough
stain to erase knowing those violations are now on the Big Ten record.

What will Harbaugh do now? That is almost impossible to predict given his combative personality. Would he sign a contract extension and continue to fight the
NCAA? Will Harbaugh finally take the parachute to the NFL?
</item 3>

<item 4>
Michigan sign-stealing scandal news: Latest updates on possible Jim Harbaugh suspension from NCAA cheating investigation (Sporting News,
2023-11-06T23:58:04+00:00)
However, the primary difference is Michigan is accused of using in-person scouting, which has been prohibited under NCAA rules for nearly 30 years, to acquire
its information.

Monday, Nov. 6

No connection between Buckeyes coach, Michigan investigation

Ohio State coach Ryan Day and his family have no connection to the Big Ten and NCAA investigations into Michigan, Yahoo Sports reported.

A report emerged over the weekend indicating Day may have had some involvement in the probe into the Buckeyes' arch-rivals, but the NCAA told the Big Ten there
are no ties that it knows of between the Ohio State coach and the investigation.

NCAA shares findings with Big Ten as Harbaugh suspension looms

Yahoo Sports reported Monday that the NCAA shared its investigation findings with the Big Ten, noting that the findings "do not connect the in-person scouting
and recording of opponents’ sidelines to Harbaugh."

Still, the conference is considering a suspension for Harbaugh and could hand down punishments before the end of the week.
</item 4>

<item 5>
Michigan sign-stealing scandal winners and losers, from Jim Harbaugh to Tony Petitti (Sporting News, 2023-11-11T22:50:13+00:00)
Will the Harbaugh-Petitti battle lead to an unsettling end for Michigan? President Santa Ono and athletic director Warde Manuel have voiced their support for
Harbaugh, but that has been more about due process than admitting to any wrongdoing.

It has thrust the Wolverines – a talented team led by J.J. McCarthy, Blake Corum and a defense that allows 7.5 points per game – into a villain role, complete
with visits from Ric Flair. That only works when you are winning. One loss, and all this could unravel, and the Terps and Buckeyes still remain on the schedule.

Will a national championship outweigh the stain on their reputation? The answer is almost always yes for the fans, but the program might feel a different
long-term result.
</item 5>

<item 6>
Michigan sign-stealing scandal news: Latest updates on possible Jim Harbaugh suspension from NCAA cheating investigation (Sporting News,
2023-11-06T23:58:04+00:00)
Public reprimands/fines are common penalties under B1G sportsmanship policy. — Adam Rittenberg (@ESPNRittenberg) November 9, 2023

Michigan's 10-page letter to Big Ten released

Yahoo Sports' Dan Wetzel posted Michigan's full 10-page letter to the Big Ten on X (formerly Twitter). The letter outlines Michigan's belief that the Big Ten
would be acting too hastily in suspending Harbaugh or levying any punishment toward Michigan before the completion of a full NCAA investigation.

In the letter, Michigan said the university has not had the chance to review all the evidence and respond, which oversteps the due process allowed for Michigan.
It said Pettiti should not have unilateral authority to punish violations without first having a completed investigation and then a response from the Compliance
and Reinstatement Committee, which would recommend penalties for rules violations.
</item 6>

<item 7>
Michigan sign-stealing scandal winners and losers, from Jim Harbaugh to Tony Petitti (Sporting News, 2023-11-11T22:50:13+00:00)
1 Ohio State is a huge television draw and Petitti has a background in TV with the MLB Network. Petitti went heavy-handed instead, maybe a little too heavy
considering the pending legal response from Michigan.

This no doubt is risky on several fronts. The Big Ten made it clear that they still have not linked Harbaugh to any wrongdoing and that this an institutional
penalty based on the findings. The NCAA also will continue its investigation into the Wolverines. The timing of Friday's announcement also was poor – less than
24 hours before the game against Penn State. Petitti upheld sportsmanship within the conference, but it could lead to a long-term rift with Michigan.

Regardless, Petitti did the right thing.

DECOURCY: Michigan got off easy with Harbaugh suspension

Ohio State-Michigan TV ratings

With or without Harbaugh, The Game will draw ridiculous television ratings.

According to Sports Media Watch, three college football games have drew a Nielsen rating of 5.0 or better this season.
</item 7>

<item 8>
Michigan sign-stealing scandal winners and losers, from Jim Harbaugh to Tony Petitti (Sporting News, 2023-11-11T22:50:13+00:00)
Helmet technology

CBSSports.com reported on Oct. 20 that the NCAA rules committee is giving teams the option of coach-to-player helmet communication to relay plays from the
sideline during bowl season this year. Tablets will inevitably follow, and here is the sliver of good that will come from this scandal.

It's about damn time. Death to sandwich-board playcalling, please. Bring on those playsheets where coaches simply have to cover their mouth. This is one
instance where emulating the NFL game is overdue.

Sherrone Moore

If Moore is going to be a head coach, then Saturday’s victory against the Nittany Lions was a fitting audition. The Wolverines rushed for 227 yards – including
32 straight run plays – in grinding out a top-10 victory at one of the toughest environments in the FBS at Beaver Stadium.

Moore was emotional - complete with NSFW language – in the aftermath.

Sherrone Moore filled in for Jim Harbaugh as Michigan's coach today.
</item 8>

<item 9>
Michigan sign-stealing scandal news: Latest updates on possible Jim Harbaugh suspension from NCAA cheating investigation (Sporting News,
2023-11-06T23:58:04+00:00)
Michigan noted that regarding evidence, there is nothing it is aware of to claim athletics personnel at Michigan committed the allegations, and said the rule
pertaining to electronic recordings of games is limited to the use of field equipment used by the institution for the game it is currently playing. It also said
Michigan has not seen much evidence, but rather has just heard a summary of second- and third-hand information.

The letter also said given the use of sign decoding by other universities, there is little reason to believe Stalions' efforts proved meaningful, and pointed
out the teams' average margin of victory since Stalions was suspended. It said the information obtained is "at most a reflection of information that teams can
and easily do obtain through public and permissible sources," and because of that it is "highly dubious" that Stalions' observations would have a "material
effect on the integrity of the competition."
</item 9>

<item 10>
Michigan sign-stealing scandal news: Latest updates on possible Jim Harbaugh suspension from NCAA cheating investigation (Sporting News,
2023-11-06T23:58:04+00:00)
Here is Michigan’s 10-page letter to the Big Ten responding to the sportsmanship investigation, reporting with @RossDellenger 1/3 pic.twitter.com/P0oYgIbS67 —
Dan Wetzel (@DanWetzel) November 9, 2023

No timetable for Big Ten decision

The Big Ten is reviewing the "lengthy" official response from Michigan and Jim Harbaugh and it's unclear when a potential ruling might come down, says ESPN's
Pete Thamel. His colleague Adam Rittenberg reported Wednesday that a decision was expected by Thursday.

Source: The Big Ten is reviewing the lengthy written responses from both Michigan and Jim Harbaugh. There’s no indication of any timetable on a decision on
potential Big Ten discipline in the case of Michigan’s alleged illegal signal stealing.
</item 10>

<item 11>
Michigan sign-stealing scandal news: Latest updates on possible Jim Harbaugh suspension from NCAA cheating investigation (Sporting News,
2023-11-06T23:58:04+00:00)
Within it, the school warned Big Ten commissioner Tony Pettiti against levying "premature" disciplinary action against Jim Harbaugh and alleged that in-person
scouting and sign-stealing are commonplace across the NCAA.

"The conference should act cautiously when setting precedent given the reality that in-person scouting, collusion among opponents, and other questionable
practices may well be far more prevalent than believed," the letter read.

Michigan also included with the letter "two attached documents of Michigan's offensive and defensive signs allegedly stolen and shared by other Big Ten teams,"
per Yahoo Sports.

Conference awaits Michigan response

The Big Ten expects a formal response from Michigan to the notice of discipline by the end of the day Wednesday, ESPN's Pete Thamel reports. His colleague Adam
Rittenberg adds that a ruling is expected to come down Thursday.

Sources: The Big Ten expects a response from Michigan on the notice of discipline by the end of day on Wednesday. Don’t expect any ruling by the Big Ten on
Wednesday, as they are expected to take time to absorb Michigan’s response.
</item 11>

<item 12>
Michigan didn't need Jim Harbaugh against Penn State, but Ohio State is a different game (Sporting News, 2023-11-11T21:32:44+00:00)
Will Harbaugh be on the field before the end of the regular season? The Big Ten made a Friday ruling that will prohibit Harbaugh from being on the field for the
final three regular-season games as a result of the in-person scouting and sign-stealing scandal that led to the resignation of staffer Conor Stallions. ESPN
reported there will be a temporary restraining order hearing on Nov. 17 that will answer that question.

On the field, Michigan answered the question about its College Football Playoff. Sort of. The Wolverines proved they could beat Penn State (8-2, 5-2) in a
20-yard box. Neither team completed a pass of 20 yards or more. Michigan trailed 3-0 after the first quarter. The Wolverines had eight carries for 10 yards, and
Penn State's Abdul Carter set the tone with a jarring hit on Blake Corum that nearly resulted in a fumble.

Michigan took control with the running game.
</item 12>

<item 13>
Michigan sign-stealing scandal winners and losers, from Jim Harbaugh to Tony Petitti (Sporting News, 2023-11-11T22:50:13+00:00)
Will Harbaugh finally take the parachute to the NFL? Those are questions bound to come up at the end of the 2023 season.

BENDER: Michigan will need Jim Harbaugh against Ohio State

Big Ten

This is the fourth football scandal in less than a calendar year.

Three former Iowa football players and a student manager were charged in an illegal sports wagering scandal. Several student-athletes face allegations of
placing bets on games in which they played in. Northwestern coach Pat Fitzgerald was fired on July 10 after a hazing investigation into the program. Fitzgerald
is suing the university for wrongful termination. Michigan State fired Mel Tucker on Sept. 27 after allegations of sexual misconduct from Brenda Tracy, a
speaker on sexual assault awareness.

Now, Michigan has a scandal on the field, and they countered with allegations that Ohio State and Rutgers leaked Michigan's signs to Purdue in last year's Big
Ten championship game.
</item 13>

<item 14>
Michigan sign-stealing scandal winners and losers, from Jim Harbaugh to Tony Petitti (Sporting News, 2023-11-11T22:50:13+00:00)
According to ESPN.com, Stalions allegedly purchased tickets to more than 35 games involving Big Ten and potential CFP opponents.

Photos also surfaced os a person who resembles Stalions on the Central Michigan sideline during the Chippewas' opener against Michigan State on Sept. 1. Central
Michigan has yet to confirm who that person was, another layer to a wild Stalions story that has supplied endless meme and conspiracy material on social media.

Stalions might be the fall guy who saved Michigan from further wrath from the Big Ten and NCAA or the reason the Wolverines' national championship season
unraveled. That fate is yet to be determined, but at minimum he will be one of those controversial names we'll never forget, like Harvey Updyke and Nevin
Shapiro.

Jim Harbaugh

On one hand, Harbaugh could serve his second suspension within one season and further NCAA punishment could be coming down the line.
</item 14>

<item 15>
Michigan sign-stealing scandal winners and losers, from Jim Harbaugh to Tony Petitti (Sporting News, 2023-11-11T22:50:13+00:00)
ESPN reported that Big Ten coaches and athletic directors encouraged Big Ten commissioner Petitti to act last week, but that resulted in action before the NCAA
investigation concluded.

Do you wonder what the SEC and commissioner Greg Sankey privately think about all this? For the Big Ten, it is an ugly run of bad behavior on and off the field
for a conference that is welcoming four new members next season in Oregon, UCLA, USC and Washington.

To be determined

Michigan

Michigan is 10-0 and on track to meet No. 1 Ohio State for another blockbuster matchup in college football’s best rivalry. On the field, the Wolverines remain
on track for a third straight Big Ten championship and shot at the College Football Playoff.

But this is about more than football.

Will more penalties come from the Big Ten and NCAA as a result of the in-person scouting scandal that has not left the college football consciousness this
month? Will the Harbaugh-Petitti battle lead to an unsettling end for Michigan?
</item 15>

<item 16>
Michigan sign-stealing scandal news: Latest updates on possible Jim Harbaugh suspension from NCAA cheating investigation (Sporting News,
2023-11-06T23:58:04+00:00)
“We went through and made sure we asked all the questions and got our compliance people involved. None of that came back at all. I can answer very strongly that
that did not happen.”

Michigan submits response to Big Ten

In the latest step toward a resolution, Michigan has submitted its response to the Big Ten's notice of discipline. ESPN reported earlier Wednesday that no
ruling from the conference is expected until Thursday, as the Big Ten is planning to "take time to absorb" what Michigan has to say.

Michigan has sent its response to the Big Ten, a school official tells ESPN. Commissioner Tony Petitti now must decide how to proceed with potential discipline.
— Adam Rittenberg (@ESPNRittenberg) November 8, 2023

Yahoo Sports obtained a copy of the 10-page letter sent by Michigan to the Big Ten.
</item 16>

<item 17>
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
</item 17>

<item 18>
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
</item 18>

<item 19>
Michigan didn't need Jim Harbaugh against Penn State, but Ohio State is a different game (Sporting News, 2023-11-11T21:32:44+00:00)
Who is the good guy? Who is the villain? Ohio State and Michigan locked in their answers a long time ago. More than 17 million people watched last year's
matchup – a 45-23 Michigan victory where Stalions' involvement continues to be scrutinized through a double-barrel Big Ten and NCAA investigation. Big Ten
commissioner Tony Petitti's ruling sent a shockwave through the conference Friday, mainly because the third game means Harbaugh will not be on the sideline for
The Game. Early forecasts say 20 million will tune in for The Game this year.

Will Harbaugh really not be there? That is for the courts to decide, but Moore’s comments suggest Harbaugh, president Santa Ono and athletic director Warde
Manuel are on a united front; consequences be damned in 2024 and beyond.

In the short term? The College Football Playoff committee could slide Michigan up to No. 2 now, and the debate with the Buckeyes and No.
</item 19>

<item 20>
Michigan didn't need Jim Harbaugh against Penn State, but Ohio State is a different game (Sporting News, 2023-11-11T21:32:44+00:00)
2 now, and the debate with the Buckeyes and No. 1 Georgia is going to perk up more than ever. They are the three best teams in the country, but only one has
uncertainty with the head coach.

Whether they admit it or not, Michigan still needs Harbaugh for those moments, especially in two weeks.

The reality is he might not be present for that one either.
</item 20>
=== END ITEM id=03d235a4e271fc0a ===

=== ITEM id=5eaf260d69d76236 ===
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
Dolphins vs. Eagles live score, updates, highlights from NFL 'Sunday Night Football' game (Sporting News, 2023-10-22T23:28:04+00:00)
A key 3rd & 8 here for the Eagles.

8:40 p.m. — The Eagles quickly force a punt after Tagovailoa's pass on 3rd & 15 falls short. So, neither of these explosive offenses start with a touchdown.

8:39 p.m. — Haason Reddick swallows up Raheem Mostert for a five-yard loss to quickly set the Dolphins back.

8:38 p.m. — Tyreek Hill is on the board with a 15-yard catch on a Tagovailoa pass.

Eagles 3, Dolphins 0

8:33 p.m. FIELD GOAL — Elliott connects from 24 yards out, and the Eagles strike first despite their promising drive fizzling out.

8:32 p.m. — Eagles fans are unhappy after Philadelphia opts for three consecutive runs in the red zone. It'll be a field goal try for Jake Elliott.

8:30 p.m.
</item 1>

<item 2>
Dolphins vs. Eagles live score, updates, highlights from NFL 'Sunday Night Football' game (Sporting News, 2023-10-22T23:28:04+00:00)
That's a pick six, and the Dolphins have tied the game just like that after the no-call.

10:22 p.m. — This looks more obvious than a certain penalty on James Bradberry from February.

Looks like the refs missed another facemask, this one on James Bradberry pic.twitter.com/fYYeWnEWaGhttps://t.co/54oa1jfeHf — Dov Kleiman (@NFL_DovKleiman)
October 23, 2023

10:20 p.m. — James Bradberry appears to get ahold of Cedrick Wilson's face mask on 4th & 3, but no flag flies. The Dolphins turn it over on downs.

10:19 p.m. — Tyreek Hill just can't hang onto what would've been a big-time catch. It's not a fumble, but it is a fourth down for the Dolphins. They'll go for
it.

10:16 p.m.
</item 2>

<item 3>
Fantasy Football Week 16 Wide Receiver Preview: Waiver adds, DFS plays, and more (CBSSports.com, 2023-12-19T20:51:32+00:00)
There will be many difficult decisions to be made in Week 16. It is the Fantasy Football semifinals in most leagues and there is a certain finality to every
decision. There's no shortage of things to stress over as a Fantasy manager. Whether or not you start Rashee Rice shouldn't be one of them. While Rice is just
WR29 on the season, that's mostly due to a slow start, and his hot start has pushed him into must-start, borderline top-12 territory.

The Chiefs' star rookie has four straight games with at least nine targets and he's scored at least 14 PPR Fantasy points in all of those games. Since Week 12,
only Deebo Samuel and CeeDee Lamb have scored more PPR Fantasy points that Rice.

On paper, the matchup doesn't look great, because the Raiders have been above average in terms of limiting the Fantasy production of wide receivers.
</item 3>

<item 4>
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
</item 4>

<item 5>
Dolphins vs. Eagles live score, updates, highlights from NFL 'Sunday Night Football' game (Sporting News, 2023-10-22T23:28:04+00:00)
to stretch the lead back to two possessions.

10:05 p.m. — Miami will punt from just past midfield, after a drive that started with some promise.

10:04 p.m. — The Dolphins face 3rd & 18, largely the result of a holding penalty. Tagovailoa has already converted one of these tonight.

10:01 p.m. — Raheem Mostert has struggled to get going, but he takes the Dolphins into Eagles territory with a solid 7-yard gain.

10:00 p.m. — Hill picks up where he left off and goes 17 yards to give this Dolphins drive an early spark.

End of first half: Eagles 17, Dolphins 10

9:45 p.m. — Any hope the Eagles had of doing something before halftime ends with a Jaelan Phillips sack of Hurts. Philadelphia will go to halftime up 17-10.

9:43 p.m. — Hill is up to 65 yards on eight catches.
</item 5>

<item 6>
Dolphins vs. Eagles live score, updates, highlights from NFL 'Sunday Night Football' game (Sporting News, 2023-10-22T23:28:04+00:00)
9:43 p.m. — Hill is up to 65 yards on eight catches. No other Dolphins receiver has more than two catches, and Hill now has 879 yards on the season.

Eagles 17, Dolphins 10

9:39 p.m. TOUCHDOWN — Tagovailoa hits Hill in the end zone for a touchdown! Hill was facing double coverage but simply beat both defenders. Miami has the
deficit back to one possession.

9:38 p.m. — Rookie Nolan Smith sacks Tagovailoa with 44 seconds remaining in the half.

9:37 p.m. — The Dolphins are down to the Eagles' 29-yard line after catches by Braxton Berrios and Tyreek Hill. They have a chance to score on back-to-back
possessions, as they get the ball out of halftime.

9:33 p.m. TWO-MINUTE WARNING — Woah!
</item 6>

<item 7>
N.F.L. Executives on the Biggest Questions Facing Super Bowl Contenders (The New York Times, 2023-10-16T23:58:53+00:00)
Receiving Yards Through Six NFL Games Rk Player Season Yds 1 Don Hutson 1942 819 2 Tyreek Hill 2023 812 3 Wes Welker 2011 785 4 Elroy Hirsch 1951 766 5 Santana
Moss 2005 743 6 DeAndre Hopkins 2015 726 7 Raymond Berry 1960 725 8 Jerry Rice 1986 723 9 Bob Boyd 1954 718 10 Bobby Mitchell 1962 714

• Jared Goff is enjoying the second-best start to a season for his career from a statistical standpoint, behind only his 2018 Super Bowl season with the Rams.
With the Lions at 5-1 and Goff heading toward a likely contract extension, could another team come after his injured backup, former University of Tennessee
quarterback Hendon Hooker?
</item 7>

<item 8>
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
</item 8>

<item 9>
MVPs for all 16 AFC teams at the midpoint (Yardbarker, 2023-11-09T21:51:02+00:00)
MIAMI DOLPHINS | WR Tyreek Hill: Arguably the best wide receiver in the NFL, Hill has somehow reached a new level of dominance in his age-29 season. The
seven-time Pro Bowler leads the NFL in receiving yards (1,076) and TD catches (eight). As long as the Dolphins maintain their success on offense down the
stretch, Hill should earn Offensive Player of the Year honors. He's on pace to break Calvin Johnson's season record for receiving yards (1,964).



NEW YORK JETS | LB Quincy Williams: While his brother, Quinnen, may be the better-known Jets defender in the family, Williams — the AFC Defensive Player of the
Month for October — is also making a name for himself. Aside from being the fifth-highest-rated linebacker in the NFL by Pro Football Focus, Williams is tied
for 17th in tackles (74) and tied for ninth in TFLs (nine) among all defenders.
</item 9>

<item 10>
NFL roundup: Pats hit new low as Lions reach 9-3 for first time since 1962 (The Guardian, 2023-12-03T22:11:26+00:00)
Keenan Allen came back from a thigh injury to catch five passes for 58 yards. But the only scoring the Chargers could manage was a pair of 38-yard field goals
in the second quarter. New England made a quarterback switch, benching Mac Jones in favor of Bailey Zappe. But the Patriots (2-10) still lost their fifth
straight game. For the first time in franchise history, the Patriots have been shut out at home twice in a season. Zappe was sacked five times – two by Khalil
Mack, giving him a career-high 15 for the season and 99 in his career. Zappe was 13 for 25 for 141 yards; he did avoid an interception for the first time in
three games, the previous two in relief of Jones.

Tyreek Hill had two touchdowns among his 157 yards receiving to help the Miami Dolphins rout the Washington Commanders 45-15.
</item 10>

<item 11>
Fantasy Football Week 16 Wide Receiver Preview: Waiver adds, DFS plays, and more (CBSSports.com, 2023-12-19T20:51:32+00:00)
Weekly Breakdown MATCHUP @ MIA MIA -1.5 O/U 51 OPP VS WR 15th PROJ PTS 19.4 WR RNK 2nd YTD Stats REC 103 TAR 141 REYDS 1306 TD 10 FPTS/G 21.6 The choice between
Lamb and Tyreek Hill will not be an easy one. The tiebreaker for me is that Lamb is $900 cheaper and Hill may still not be 100% by the time the Cowboys and
Dolphins kick off. Last week was the first time since Week 9 that Lamb failed to reach the end zone and the second time since Week 5 that he didn't score at
least 17 PPR Fantasy points.

Contrarian DFS Play (WR Preview) Projections powered by Sportsline Jaylen Waddle WR MIA Miami • #17

Age: 25 • Experience: 3 yrs.
</item 11>

<item 12>
NFL Best Bets Week 9: Colts clobber Panthers, Dolphins drop Chiefs in Germany (Sporting News, 2023-11-03T20:56:00+00:00)
Expect more of the same from QB Justin Herbert and his stud back against the Jets, who have allowed 46 receiving yards per game to RBs (sixth-most in the NFL).
They'll try to shut him down this Sunday, but it would feel so empty without AE!

Jahan Dotson, WR, Commanders @ Patriots — OVER 45.5 receiving yards

We've come full circle with Dotson, who made our preseason sleepers column then faded into early-season obscurity only to emerge again over the past two weeks.
Curtis Samuel was clearly not right last week, and the second-year wideout more than made up for it. Dotson caught 8-of-10 targets for 108 yards and a TD, a
stellar encore to his Week 7 line of five catches for 43 yards. The banged-up Patriots defense will be primarily focused on stopping stud Commanders receiver
Terry McLaurin, which should open up plenty of opportunities for Dotson.
</item 12>

<item 13>
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
</item 13>

<item 14>
Fantasy Football Week 16 Wide Receiver Preview: Waiver adds, DFS plays, and more (CBSSports.com, 2023-12-19T20:51:32+00:00)
The Texans offense looks like one to avoid as long as C.J. Stroud is out. Ja'Marr Chase WR CIN Cincinnati • #1

Age: 23 • Experience: 3 yrs. Tee Higgins is a No. 2 receiver and Tyler Boyd is a fine flex.

WR Preview Numbers to Know

69.9% -- Nearly 70% of Rashee Rice's receiving yards have come after the catch, that's tops amongst wide receivers with at least 50 targets.

-- Nearly 70% of Rashee Rice's receiving yards have come after the catch, that's tops amongst wide receivers with at least 50 targets. 11.4 -- Deebo Samuel is
on pace to become just the fourth player since 1992 to average more than 11 yards per target with at least 60 targets in multiple seasons. The other three are
Tyreek Hill, Tyler Lockett, and DeSean Jackson.
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
NFL Best Bets Week 9: Colts clobber Panthers, Dolphins drop Chiefs in Germany (Sporting News, 2023-11-03T20:56:00+00:00)
Chiefs (Germany game)

We get it — the Dolphins have yet to defeat a true contender this season and the Chiefs are the reigning champions of the NFL. However, Kansas City just got its
doors blown off by Russell Wilson and the lowly Broncos, a game in which Patrick Mahomes and company suffered five turnovers and three sacks.

We expect a statement win from Mike McDaniel and the Dolphins on neutral ground. The Dolphins have been in Frankfurt since early this week, having flown
overnight on Monday to give them multiple days to acclimate to the time difference and their surroundings. The Chiefs, meanwhile, waited until after practice on
Thursday to take the 11-hour voyage.

This also happens to be a revenge game for former Chiefs receiver Tyreek Hill, who's on pace for multiple single-season receiving records and poised for his
first Offensive Player of the Year award.
</item 16>

<item 17>
Fantasy Football Week 13 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-11-30T22:10:58+00:00)
The Commanders offensive line has had some ugly showings lately against the Cowboys (four sacks, 49% pass rush pressure rate allowed), the Giants (four sacks,
32.7% pressure rate allowed) and the Seahawks (three sacks, 46.8% pressure rate allowed). Somehow these results are better than what they were allowing through
much of the first two months of the year. Sam Howell getting the ball out faster has been a positive in this regard but a negative for his numbers in three of
his past four games (the one game he did really well in was sort of fluky). The matchup is a good test for the Dolphins pass rush post-Phillips, one that it
should pass without giving up much in the way of deep completions.

STARTS: Tyreek Hill, Jaylen Waddle, Raheem Mostert, Tua Tagovailoa, Terry McLaurin (low-end WR2),
</item 17>

<item 18>
N.F.L. Executives on the Biggest Questions Facing Super Bowl Contenders (The New York Times, 2023-10-16T23:58:53+00:00)
That figure was 7.8 in Mahomes’ first year as the starter and has climbed every season since. It’s a testament to the Chiefs’ adaptability, but also a
reflection of plummeting explosive play rates.

Exec comment: “Yeah, it looks harder for them on offense until Mahomes finds (Travis) Kelce on every major third down, every major two-minute play, every major
touchdown. Kansas City has the luxury of their best players being their toughest players. Chris Jones is a very tough player. Kelce is a very tough player.
Mahomes is a very tough player. You wonder why they win the games? Because the games are won by the toughest teams most of the time.”

• Philadelphia Eagles: Has Jalen Hurts regressed? Where are the explosive pass plays?

What I think: Explosive passing is down through most of the league, including for the Eagles.
</item 18>

<item 19>
N.F.L. Executives on the Biggest Questions Facing Super Bowl Contenders (The New York Times, 2023-10-16T23:58:53+00:00)
Nobody has really emerged in the AFC to unseat these teams at the top.”

• Miami Dolphins: Has Tua Tagovailoa proven he’s built to last?

What I think: It was tough watching Tagovailoa last season without worrying for his physical well-being. I’ve never felt that way watching him so far this
season, and will feel even better if he plays a full game against Philadelphia in Week 7. New England, Kansas City, the Jets (twice), Washington, Tennessee,
Dallas, Baltimore and Buffalo will all get their shots at Tagovailoa. I’m cautiously optimistic, but think there will be games when defense muddy the picture
sufficiently to create turnovers.

Exec comment: “Interesting team. They put up 70 on Denver and should have put up 70 on the Giants. No one can keep up with that kind of firepower. But they also
haven’t really beaten anyone. They were down 14 points to Carolina.
</item 19>

<item 20>
Fantasy Football Week 16 Wide Receiver Preview: Waiver adds, DFS plays, and more (CBSSports.com, 2023-12-19T20:51:32+00:00)
The other three are Tyreek Hill, Tyler Lockett, and DeSean Jackson.

-- Deebo Samuel is on pace to become just the fourth player since 1992 to average more than 11 yards per target with at least 60 targets in multiple seasons.
The other three are Tyreek Hill, Tyler Lockett, and DeSean Jackson. 18 -- Puka Nacua needs 18 more catches to set the rookie record for most catches in a
season.

-- Puka Nacua needs 18 more catches to set the rookie record for most catches in a season. 152.67 -- After missing last week's game, Tyreek Hill now needs to
average almost 153 yards per game in his final three to reach his stated goal of 2,000 receiving yards this season.

-- After missing last week's game, Tyreek Hill now needs to average almost 153 yards per game in his final three to reach his stated goal of 2,000 receiving
yards this season.
</item 20>

<item 21>
Fantasy Football Week 16 Wide Receiver Preview: Waiver adds, DFS plays, and more (CBSSports.com, 2023-12-19T20:51:32+00:00)
Weekly Breakdown MATCHUP vs DAL MIA -1.5 O/U 51 OPP VS WR 7th PROJ PTS 14.2 WR RNK 16th YTD Stats REC 71 TAR 99 REYDS 964 TD 4 FPTS/G 14.8 We expect Hill to
return this week, which should drive people off of Waddle. But there is at least a chance that Hill won't be 100% and Waddle will still be Tua Tagovailoa's top
target. Waddle has been the quintessential DFS tournament play as of late with three of seven games over 19 PPR Fantasy points, topping 26 twice.

WR Preview Heath's Projections

My full set of Week 16 Fantasy Football projections for every position are now available on SportsLine. Find out which of my favorite plays are projected to
score higher than consensus rankings and which don't live up to their draft hype, at least in Week 16.
</item 21>

<item 22>
Fantasy Football WR PPR Rankings Week 14: Who to start, best sleepers at wide receiver (Sporting News, 2023-12-07T22:11:00+00:00)
The league's leading receiver, Tyreek Hill (vs. Titans in Week 14), stands as the unquestioned WR1 for Week 14 after torching the Commanders to the tune of five
catches, 157 yards, and two TDs. CeeDee Lamb (vs. Eagles) and the Cowboys look to avenge their road loss to the Eagles in Week 9, as Dallas' WR1 routinely has
double-digit reception potential.

Amon-Ra St. Brown (@ Bears) was relatively quiet this past week, but he's likely to rebound against a Bears defense that he totaled nine catches for 95 yards
against in Week 11. A.J. Brown (@ Cowboys) reached the century mark for the first time in four games in Week 13, and another high-usage day looks to be in store
for a potential shootout.

Keenan Allen (vs. Broncos) caught just five-of-nine targets last week, but as L.A.
</item 22>

<item 23>
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
</item 23>

<item 24>
MVPs for all 16 AFC teams at the midpoint (Yardbarker, 2023-11-09T21:51:02+00:00)
He’s third in the NFL in passing yards (2,442), fourth in TD passes (17) and third in QBR (73). The 28-year-old might capture his third MVP, as oddschecker
notes he’s favored to win the award again.

LAS VEGAS RAIDERS | Defensive end Maxx Crosby: Despite a weak supporting cast, Crosby thrives. He has 9.5 of the Raiders’ 24 sacks and is tied for the league
lead in tackles for loss (13). If the two-time Pro Bowler keeps dominating, he will generate more Defensive Player of the Year buzz.

LOS ANGELES CHARGERS | WR Keenan Allen: The 31-year-old is the centerpiece of the offense and leads the team in yards receiving (720) and receptions (62).
Through Week 9, Pro Football Focus gave him a receiving grade of 84.6, eighth among WRs.
</item 24>

<item 25>
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
</item 25>

<item 26>
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
</item 26>

<item 27>
Fantasy Football WR PPR Rankings Week 13: Who to start, best sleepers at wide receiver (Sporting News, 2023-11-29T20:27:00+00:00)
Tyreek Hill (@ Commanders in Week 13) did his thing despite a tough matchup in Week 12, resulting in Miami's WR1 returning to the top spot against a putrid
Washington stop unit. CeeDee Lamb (vs. Seahawks) isn't far behind, as Dallas' WR1 aims to maintain his efficient play against Seattle.

Amon-Ra St. Brown (@ Saints) has been a target hog the past several weeks, seeing at least nine looks in each of his past six games. Don't expect his workload
to decrease in a bounce-back game for the Lions. Keenan Allen (@ Patriots) looks to continue his dominant ways, as Justin Herbert's go-to target routinely has
double-digit target potential.

Mike Evans (vs. Panthers) sits in our top five for Week 13, totaling four TDs over the past three weeks. A.J. Brown (vs.
</item 27>

<item 28>
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
</item 28>

<item 29>
N.F.L. Executives on the Biggest Questions Facing Super Bowl Contenders (The New York Times, 2023-10-16T23:58:53+00:00)
Six of the 17 won anyway. The Dolphins were the only one of those 223 teams to win and cover the spread, which was 13.5 for this game.

• Six catches for 163 yards and a touchdown against Carolina left the Dolphins’ Tyreek Hill with 812 yards through the first six games. That ranks second in NFL
history to Hall of Famer Don Hutson, who had 819 yards through six with Green Bay in 1942.
</item 29>

<item 30>
N.F.L. Executives on the Biggest Questions Facing Super Bowl Contenders (The New York Times, 2023-10-16T23:58:53+00:00)
Especially when you face a defensive coordinator who says, ‘You show me you can beat me throwing the ball outside the numbers — best of luck, and to do that,
you are going to have to take a five-step drop and I’m going to have Myles Garrett chasing you every play.'”

Advertisement

• Kansas City Chiefs: Is the explosive passing game gone for good?

What I think: The Chiefs have never scored less per game on offense through six games or a full season since Patrick Mahomes became their quarterback. They have
also never enjoyed larger average point differentials (9.8 per game), because their defense is so much better. I think the pass offense will become slightly
more explosive as Mahomes gets reps with his receivers, but I’m not expecting a huge shift.

The Chiefs are averaging 9.5 plays per touchdown drive when starting at least 70 yards from the opponent’s goal line, per TruMedia.
</item 30>
=== END ITEM id=5eaf260d69d76236 ===

=== ITEM id=473e3f0f40723ecd ===
Gold sentences and facts:
Sentence 0: UPDATE: NFL Network's Ian Rapoport reported Sunday morning that Hockenson is expected to play against New Orleans.
  Fact 0.0: NFL Network's Ian Rapoport reported on a Sunday morning.
  Fact 0.1: Ian Rapoport reported that Hockenson is expected to play against New Orleans.
Sentence 1: Head coach Dennis Allen told reporters Friday that despite Olave being a full participant in Friday's practice he is still in concussion protocol though has a chance to play.
  Fact 1.0: Head coach Dennis Allen told reporters on Friday.
  Fact 1.1: Dennis Allen said Olave was a full participant in Friday's practice.
  Fact 1.2: Dennis Allen said Olave is still in concussion protocol.
  Fact 1.3: Dennis Allen said Olave has a chance to play.

Retrieved items:
<item 1>
NFL Fantasy Football Week 6 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-12T21:23:16+00:00)
But they float around league average in catch rate allowed (77.8%, below league average) and yards per catch (8.9, better than league average). However, no team
has allowed more first-down receptions to tight ends (17). Keep in mind, this defense hasn't seen tight ends better than Evan Engram, Sam LaPorta or Dalton
Schultz.

COMMANDERS: Throw to their tight ends on 26.3% of their throws, fourth-most in football. But that hasn't always meant a huge target share for Thomas. In Weeks 1
and 5 he saw at least 22.9% of Sam Howell's targets. He left Week 2 after suffering a concussion and was absent in Week 3; then was not part of the game plan
much in Week 4 (7.7% target share against Philadelphia). Perhaps it's all health related and Thomas' volume from last week will stick.
</item 1>

<item 2>
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
</item 2>

<item 3>
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
</item 3>

<item 4>
NFL Week 13 injuries: Browns to start Joe Flacco with Dorian Thompson-Robinson out, Saints' Olave questionable (CBSSports.com, 2023-12-01T22:52:36+00:00)
Washington could be without its starting center this weekend as it plays host to Miami. Larsen, who missed Wednesday's practice due to a knee injury, was
limited on Friday and is officially listed as questionable. Meanwhile, the Commanders will play Week 13 without Forbes Jr., who has been sidelined since
suffering an elbow injury against the Giants in Week 11. Head coach Ron Rivera called the injury "significant" earlier this week.

Broncos at Texans (-3)

Denver's lone player with an injury designation for Week 13 is Mathis, who did not practice Friday after being limited on Thursday. Outside of him, the Broncos
are healthy heading into this matchup with Houston. Wideouts Jerry Jeudy (groin) and Marvin Mims Jr. (hamstring) were both full participants on Friday after
being limited on Thursday.

Dell logged back-to-back days as a limited participant in practice to end the week and is officially questionable.
</item 4>

<item 5>
Fantasy Football Week 13 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-11-30T22:10:58+00:00)
After the injuries to Shaheed and Olave last Sunday it was Alvin Kamara who predictably led the way in targets per routes run (30.8%). Kamara gets a boost both
in the pass game and run game with Lions linebacker Alex Anzalone missing action with a hand injury. For the Lions offense, expect the run game to get a bit
more focus with Saints linebacker Pete Werner sidelined and All-Pro defensive end Cameron Jordan questionable to play.

STARTS: Jared Goff, Amon-Ra St. Brown, Jahmyr Gibbs, Alvin Kamara, Chris Olave (if he passes concussion protocol), David Montgomery, Sam LaPorta, Taysom Hill

Jared Goff, Amon-Ra St.
</item 5>

<item 6>
NFL Fantasy Football Week 6 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-12T21:23:16+00:00)
Weekly Breakdown MATCHUP @ CHI MIN -2.5 O/U 44.5 OPP VS WR 20th PROJ PTS 10.7 WR RNK 33rd YTD Stats REC 13 TAR 26 REYDS 166 TD 2 FPTS/G 8.3 No doubt there's
some potential here, but some hunches keep me from falling for Osborn this week. One: I think the Vikings will try to get their run game going since they don't
have Justin Jefferson. Two: Jordan Addison is far more explosive and T.J. Hockenson has been a much more relied-upon weapon. They should have more targets than
Osborn. And the expected zone coverage Chicago will play further will limit Osborn unless he sees more targets than expected.
</item 6>

<item 7>
Fantasy Injury Updates Week 10: News on Justin Jefferson, Ja'Marr Chase, T.J. Hockenson & Curtis Samuel (Sporting News, 2023-11-11T21:17:00+00:00)
ET, according to ESPN's Jeremy Fowler. The receiver was limited in practice thanks to the toe injury that has plagued him since leaving Week 8's game against
the Eagles, but barring a setback, he will return to action this weekend.

The receiver was working his way into flex territory with a couple of good performances before picking up the injury. However, we wouldn't consider him to be
worthy of a fantasy start this week unless you're in a bind. It's unclear whether Samuel will regain the same workload or if he'll be playing in limited action
as he works his way back. Jamison Crowder emerged in Samuel's absence and Jahan Dotson has seen his production drastically increase, as well. The wait-and-see
approach is the preferred one heading into Week 10.
</item 7>

<item 8>
Fantasy Injury Updates Week 10: News on Justin Jefferson, Ja'Marr Chase, T.J. Hockenson & Curtis Samuel (Sporting News, 2023-11-11T21:17:00+00:00)
After turning back the clocks last week, it now gets darker earlier for the foreseeable future. That darkness allows us to see the stars on a clear night, so
it's only natural to wonder if the pass-catching stars will shine bright in Week 10 of the NFL season. We already know Tee Higgins and Nico Collins are OUT,
while Justin Jefferson, Ja'Marr Chase, and T.J. Hockenson are among the other star pass-catchers beaten up heading into the week. Can we expect them on the
field when the action kicks off Sunday?

Below, we'll give you the latest news on these ailing pass-catchers and break down how their injuries could potentially affect the Week 10 fantasy landscape.

WEEK 10 INJURY UPDATES: Early WRs | RBs

Is Justin Jefferson playing Week 10?

UPDATE: The Vikings will not activate Jefferson before Sunday's game and have officially ruled him OUT, according to NFL Network's Tom Pelissero.
</item 8>

<item 9>
NFL Week 13 injuries: Browns to start Joe Flacco with Dorian Thompson-Robinson out, Saints' Olave questionable (CBSSports.com, 2023-12-01T22:52:36+00:00)
Bengals at Jaguars (-8.5), Monday

Bengals : CB Cam Taylor-Britt (quad/ankle), D'Ante Smith (knee), Jay Tufele (illness) OUT

: CB Cam Taylor-Britt (quad/ankle), D'Ante Smith (knee), Jay Tufele (illness) OUT Jaguars: CB Tyson Campbell (hamstring), RB Travis Etienne (ribs), TE Brenton
Strange (foot), TE Luke Farrell (toe) QUESTIONABLE

The Bengals got good news on the injury front ahead of Monday's game against the Jaguars. Wide receiver Tee Higgins (hamstring/ankle) and linebacker Logan
Wilson (ankle/foot) are set to play, according to head coach Zac Taylor. Higgins and Wilson were full participants in practice. Cornerback Cam Taylor-Britt hurt
his foot in practice and is out.
</item 9>

<item 10>
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
</item 10>

<item 11>
Fantasy Injury Updates Week 10: News on Justin Jefferson, Ja'Marr Chase, T.J. Hockenson & Curtis Samuel (Sporting News, 2023-11-11T21:17:00+00:00)
If he's out, Boyd is a must-start, and you can try one of Cincinnati's secondary receivers in deep leagues.

WEEK 10 STANDARD RANKINGS:

QBs | RBs | WRs | TEs | D/STs | Kickers

Is T.J. Hockenson playing Week 10?

UPDATE: NFL Network's Ian Rapoport reported Sunday morning that Hockenson is expected to play against New Orleans.

UPDATE 2: Hockenson is officially ACTIVE.

Hockenson (ribs) is listed as "questionable" for Week 10, but he's already said he plans to play on Sunday. The tight end has dealt with his fair share of
injury woes this season, however, he's yet to miss a game.
</item 11>

<item 12>
Fantasy Football TE PPR Rankings Week 14: Who to start, best sleepers at tight end (Sporting News, 2023-12-07T22:14:00+00:00)
Jaguars), and Dalton Kincaid (@ Chiefs) aren't far behind. LaPorta and Njoku both have great matchups against defenses that entered last week in the top 10 in
fantasy points per game (FPPG) allowed to TEs, and we know what Kelce, Hockenson, and Kincaid do every week in PPR formats.

George Kittle (vs. Seahawks) doesn't have a great matchup -- especially when you remember that he had just three catches for 19 yards against Seattle in Week 12
-- but he remains a must-start given his high ceiling in any matchup.

WEEK 14 FANTASY ADVICE:

Sleepers | Busts | Projections | Start-sit

Who are the best fantasy TE sleepers, pickups, & streamers in Week 14?

It can be tough to trust Taysom Hill (vs.
</item 12>

<item 13>
Fantasy Football Week 13 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-11-30T22:10:58+00:00)
The uncertainty of how the Panthers will operate moving forward with Brown basically calling the offense without any oversight keeps these players are risky
Fantasy propositions, though it helps a lot that the Buccaneers are dealing with injuries that hurt their run defense (Lavonte David and Devin White will miss
the game) and keep their pass defense from improving (Jamel Dean will miss the game).
</item 13>

<item 14>
Fantasy Football Week 13 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-11-30T22:10:58+00:00)
They were even less kind to Geno Smith, and this week shouldn't be much different. Tack on the tough assignment in Stephon Gilmore in coverage (unless the
Cowboys use shadow coverage for the first time this year), and Lockett could be in for another lean night. I don't mind him as a WR3 over Drake London, DeAndre
Hopkins and Diontae Johnson, but I'd start Rashee Rice and Adam Thielen over Lockett.

Sit Him (Lineup Decisions) Projections powered by Sportsline Jake Ferguson TE DAL Dallas • #87

Age: 24 • Experience: 2 yrs.
</item 14>

<item 15>
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
</item 15>

<item 16>
NFL Fantasy Football Week 6 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-12T21:23:16+00:00)
Weekly Breakdown MATCHUP vs NO NO -1.5 O/U 42.5 OPP VS WR 14th PROJ PTS 12.8 WR RNK 23rd YTD Stats REC 25 TAR 36 REYDS 467 TD 3 FPTS/G 17.9 I'm not sure the
Saints will back off of their aggressive man-to-man coverage even if Stroud's stats suggest they should, and that could open the door for Collins to pick up
some targets. Unfortunately, neither of his matchups are particularly favorable. I could see Collins getting a lot of short targets and for that reason alone
I'd be OK starting him as a flex option in PPR ahead of Tyler Lockett and Chris Godwin, but behind Jakobi Meyers, Jaylen Waddle and Gabe Davis.

LAST WEEK: Even in a game where the Texans lost Tank Dell to a concussion before the half, Collins couldn't get much going.
</item 16>

<item 17>
NFL Week 13 injuries: Browns to start Joe Flacco with Dorian Thompson-Robinson out, Saints' Olave questionable (CBSSports.com, 2023-12-01T22:52:36+00:00)
Cornerback Cam Taylor-Britt hurt his foot in practice and is out.

For the Jaguars, Travis Etienne (ribs) is also expected to play, Doug Pederson told reporters, despite the running back being limited in practice all week.
Strange was a late addition to the injury report, not practicing Saturday.
</item 17>

<item 18>
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
</item 18>

<item 19>
NFL Fantasy Football Week 6 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-12T21:23:16+00:00)
DEEP PASSES: The Commanders have allowed the seventh-most receptions of throws that traveled 15-plus Air Yards this season with 19. Of those 19, only two were
caught by tight ends (Cole Kmet, Dallas Goedert) and neither went for more than 23 yards.

Start 'em 10.0-7.00 6.99 - 5.01 5.00 - 0 Sit 'em

Sit Him (Lineup Decisions) Projections powered by Sportsline K.J. Osborn WR MIN Minnesota • #17

Age: 26 • Experience: 4 yrs.
</item 19>

<item 20>
Fantasy Football Week 13 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-11-30T22:10:58+00:00)
0-7.00 6.99 - 5.01 5.00 - 0 Sit 'em

The Lions need an answer for their ugly pass defense -- since their Week 9 bye they've been mostly great against the run (3.1 yards per carry allowed, one RB to
get 15-plus PPR points) but they rank bottom-six in completion rate allowed (68.4%), yards per attempt allowed (8.0), passing ADOT against (8.84 yards) and
defensive Expected Points Added (EPA) per dropback (-0.40, which is league-worst). And of their second-most eight passing touchdowns given up in three games,
five have come on passes of 15-plus Air Yards, which is tied for most in the league. They should catch a break with field stretcher Rashid Shaheed (thigh)
missing the matchup for New Orleans; Chris Olave looks like he will play.
</item 20>

<item 21>
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
</item 21>

<item 22>
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
</item 22>

<item 23>
NFL Fantasy Football Week 6 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-12T21:23:16+00:00)
Perhaps it's all health related and Thomas' volume from last week will stick.

ZONE COVERAGE: Atlanta usually plays a lot of zone coverage early on and if it works they stick with it for the whole game (Week 5 was an example of that). In
Weeks 3 and 4 against the Lions and Jaguars they played much more man coverage, especially when they needed a fourth-quarter turnover. And because Howell is
much less explosive against zone looks, I expect the Falcons to stick with zone much more this week.

Start Him (Lineup Decisions) Projections powered by Sportsline Kyle Pitts TE ATL Atlanta • #8

Age: 23 • Experience: 3 yrs.
</item 23>

<item 24>
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
</item 24>

<item 25>
NFL Week 13 injuries: Browns to start Joe Flacco with Dorian Thompson-Robinson out, Saints' Olave questionable (CBSSports.com, 2023-12-01T22:52:36+00:00)
The wideout still has another hurdle to pass to be cleared to play on Sunday so his status is still very much up in the air.

Falcons (-2) at Jets

Atlanta is relatively healthy coming into this matchup. Hollins, who injured his ankle in Week 9, participated on a limited basis every day this week, which is
encouraging for his prospects to play on Sunday. As for Hughes, he practiced fully all week after suffering a hand injury last weekend against the Saints.
Linebacker Nate Landman (thigh) missed practice on Wednesday but will play vs. New York.

Rodgers was once again limited in practice but, as expected, will not be activated this week and will be sidelined as he continues his recovery from a torn
Achilles just a few months ago. Meanwhile, Cook popped up on the injury report due to a shoulder injury, but head coach Robert Saleh told reporters Friday that
the back should be fine.
</item 25>

<item 26>
Fantasy Injury Updates Week 10: News on Justin Jefferson, Ja'Marr Chase, T.J. Hockenson & Curtis Samuel (Sporting News, 2023-11-11T21:17:00+00:00)
Jefferson (hamstring) is officially listed as "questionable" for Sunday's game against the Saints. Kickoff is slated for 1 p.m. ET, but fantasy owners won't
have to wait long for news on the star receiver. In order to be eligible to play, Jefferson will need to be activated by 4 p.m. ET on Saturday. Earlier in the
week head coach Kevin O'Connell said that it would be "a little aggressive" for Jefferson to get back on the field this week. However, the receiver's three
straight limited practices were enough for the team to at least weigh the decision, but thankfully, an early answer awaits.

WEEK 10 FANTASY ADVICE:

Sleepers | Busts | Projections | Start 'em, sit 'em

Jefferson's fantasy outlook is always through the roof, but it will be interesting to see if that changes with Josh Dobbs under center instead of Kirk Cousins.
</item 26>

<item 27>
NFL Week 13 injuries: Browns to start Joe Flacco with Dorian Thompson-Robinson out, Saints' Olave questionable (CBSSports.com, 2023-12-01T22:52:36+00:00)
Armstead has stated that he will suit up on Sunday.

Philadelphia saw the return of tight end Goedert at practice on Friday. He was limited for the session and officially listed as doubtful after practicing for
the first time since suffering a fractured forearm against the Cowboys on Nov. 5. As for Cox, he is questionable despite not practicing Friday due to a groin
injury.
</item 27>

<item 28>
Fantasy Football WR PPR Rankings Week 13: Who to start, best sleepers at wide receiver (Sporting News, 2023-11-29T20:27:00+00:00)
Geno Smith's struggles complicate their fantasy prospects, as well.

What are the biggest WR injuries in Week 13?

Entering Week 12, Tee Higgins (hamstring), Noah Brown (knee), Treylon Burks (concussion), and Kadarius Toney (ankle, hip) all failed to suit up due to nagging
injuries. For now, we've placed Higgins and Brown in our Week 13 list and omitted Burks and Toney. We'll also be eyeing the Chargers' Josh Palmer (knee), who's
eligible to come off the IR this week but probably won't suit up against New England.

Last week, the most notable WRs to exit their respective games with injuries include Chris Olave (concussion), Amari Cooper (ribs), Demario Douglas
(concussion), and Rashid Shaheed (thigh). We've left all four WRs out of our initial Week 13 list but will closely monitor their teams' injury reports to gauge
their playing prospects.
</item 28>

<item 29>
NFL Fantasy Football Week 6 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-12T21:23:16+00:00)
He has more upside than Jerry Jeudy, Jahan Dotson and Tutu Atwell but I'd rather take a shot on Drake London, Terry McLaurin or Chris Godwin.

LAST WEEK: Over 19 snaps without Justin Jefferson last week, Osborn ran 18 routes and caught both of his two targets for 30 total yards. The receiving average
was delightful, but the small-sample target share (12.5%) was saddening. What's funny is that Osborn's target share was much higher when Jefferson was playing
(23.3%) but his receiving average tanked (6.3).

FILM: Osborn didn't beat coverage with his speed last week and his burst and agility were just OK. His 22-yard catch came on a Kirk Cousins scramble where he
waited in a spot and Cousins hit him, then Osborn ran inside instead of outside where he might have been able to score.
</item 29>

<item 30>
Fantasy Injury Updates Week 10: News on Justin Jefferson, Ja'Marr Chase, T.J. Hockenson & Curtis Samuel (Sporting News, 2023-11-11T21:17:00+00:00)
Chase put a scare into the fantasy community on Wednesday when he was listed as a DNP, but after logging back-to-back limited practices, there is a lot more
optimism heading into Sunday. With Tee Higgins (hamstring) already ruled out, Cincinnati will be hoping Chase can feel healthy enough to play when they host the
Texans at 1 p.m. ET on Sunday.

The Bengals, despite getting a win over the Bills last Sunday, are a beaten-down bunch. Their defense will also be without a prominent edge rusher, Sam Hubbard,
complicating matters for a team hoping to stack wins as they compete for an AFC North crown. On the bright side, Joe Burrow appears to be finally healthy and
should have no problems targeting Tyler Boyd, Trenton Irwin, and Andrei Iosivas in the passing game. If Chase does end up suiting up, fantasy owners should
enjoy an even larger workload than the receiver already had.
</item 30>
=== END ITEM id=473e3f0f40723ecd ===

=== ITEM id=d9ea9e05ae89c513 ===
Gold sentences and facts:
Sentence 0: India were playing against Australia in the second semi-final of the 2015 World Cup in Sydney on March 26.
  Fact 0.0: India played against Australia in the second semi-final of the 2015 World Cup.
  Fact 0.1: The India versus Australia match was held in Sydney.
  Fact 0.2: The India versus Australia match was played on March 26.
Sentence 1: It's crunch time for England in the World Cups as they face South Africa in the semi-final of the Rugby World Cup and in the ODI Cricket World Cup.
  Fact 1.0: It is crunch time for England in the World Cups.
  Fact 1.1: England faces South Africa in the semi-final of the Rugby World Cup.
  Fact 1.2: England faces South Africa in the semi-final of the ODI Cricket World Cup.

Retrieved items:
<item 1>
ICC World Cup 2023: India at Cricket World Cup semi-finals so far (Zee Business, 2023-11-11T19:12:00+00:00)
India seamer Javagal Srinath took three top-order wickets, including Sanath Jayasuriya and Romesh Kaluwitharana.

Fans at the Eden Garden were cheering loudly till Tendulkar was at the crease.

But once he fell for 65 with the team at 98, India suffered one of the worst collapses in the batting history of the World Cups.

The crowd turned angry and hostile towards Sri Lanka, creating a ruckus at the stadium, and the match was interrupted.

With no respite from onlookers' side, umpires decided to call off the match when India were reeling at 120 for eight in 34.1 overs.

The long-lasting memories of the crowd's bad behaviour and a weeping Vinod Kambli returning to the pavilion are still fresh in the memories of the cricket fans
of that generation.
</item 1>

<item 2>
ICC World Cup 2023: India at Cricket World Cup semi-finals so far (Zee Business, 2023-11-11T19:12:00+00:00)
They moved into the last-four stage on the next two occasions when the World Cup was held in India, in 1987 and 1996.

India were in semi-finals in 2003 in South Africa.

Eight years later, they were again in last four at the 2011 World Cup as hosts, and repeated their feat in 2015, 2019, and 2023.

It means that in their last six World Cups, India have been in the semi-finals on five occasions.

Also Read: ICC Cricket World Cup 2023 semi-final schedule: India to play New Zealand in Mumbai; South Africa to face Australia in Kolkata; know venue, squads

In this write-up, ZeeBiz will take you through the memory lane of India's semifinal journey in the biggest cricket extravaganza in the world.
</item 2>

<item 3>
ICC World Cup 2023: India at Cricket World Cup semi-finals so far (Zee Business, 2023-11-11T19:12:00+00:00)
ICC World Cup 2023: Rohit Sharma-led India are playing Netherlands in their ninth ICC World Cup 2023 round-robin match at the M Chinnaswamy Stadium in Bengaluru
on Sunday (November 12, 2023). India have already booked a place in semi-finals, thrashing Sri Lanka by 302 runs at the Wankhede Stadium in Mumbai on November
2, 2023. India will face New Zealand in the ongoing World Cup semi-final match at the Wankhede Stadium in Mumbai on Wednesday (November 15). This is the seventh
time India have entered into the last-four stage of the 50-over World Cup.

Also Read: ICC Cricket World Cup 2023 Prize Money: How much finalists, semi-finalists, and other teams will win at ODI World Cup

India first entered semifinals of the World Cup at the 1983 Prudential Cup in England, where they emerged as the tournament winners.
</item 3>

<item 4>
Where to watch England vs South Africa: Live stream, TV channel, lineups, odds for 2023 Rugby World Cup semifinal (Sporting News, 2023-10-20T22:00:05+00:00)
England and South Africa face off in the 2023 Rugby World Cup's second semi-final, played at the Stade de France in Saint-Denis, France. The game kicks off at 9
p.m. local time on Saturday, October 21, 2023.

Here are the kick-off times for this game around the world:

Location Date Time USA Sat, Oct. 21 3:00 p.m. ET Canada Sat, Oct. 21 3:00 p.m. ET Australia Sun, Oct. 22 6:00 a.m. AEDT UK Sat, Oct. 21 8:00 p.m. BST New
Zealand Sun, Oct. 22 8:00 a.m. NZDT India Sun, Oct. 22 12:30 a.m. IST Singapore Sun, Oct. 22 3:00 a.m. SGT Malaysia Sun, Oct.
</item 4>

<item 5>
ICC World Cup 2023: India at Cricket World Cup semi-finals so far (Zee Business, 2023-11-11T19:12:00+00:00)
In reply, Mohinder Amarnath played a useful innings of 46, while Yashpal Sharma scored a fine 61.

Sandeep Patil played a fiery innings of 51 not out in just 32 balls as India stormed into semifinals for the first time, beating England by six wickets with 5.2
overs to spare.

Also Read: ICC World Cup 2023: List of centuries; Most tons in World Cup so far

1987 Reliance Cup, against England

Four years later, the World Cup was being played outside England for the first time, and it was India which was hosting it.

In a co-incidence, India was against England in the semi-finals once again, at the Wankhede Stadium, Mumbai, on November 5.

This time, England's Man-of-the-Match skipper Graham Gooch smashed 115 runs in 136 balls to lay a solid foundation for his team's fighting total of 254 for 6
wickets.
</item 5>

<item 6>
Where to watch England vs South Africa: Live stream, TV channel, lineups, odds for 2023 Rugby World Cup semifinal (Sporting News, 2023-10-20T22:00:05+00:00)
22 3:00 a.m. SGT Malaysia Sun, Oct. 22 3:00 a.m. MYT Hong Kong Sun, Oct. 22 3:00 a.m. HKT

England vs South Africa live stream, TV channel

TV networks worldwide will be showing all 48 matches at the tournament live, including many on free-to-air television.

On social platforms alone, video content from the Rugby World Cup and organisers World Rugby drew more than 2.1 billion views in 2019 — almost six times the
previous figure.
</item 6>

<item 7>
Where to watch England vs South Africa: Live stream, TV channel, lineups, odds for 2023 Rugby World Cup semifinal (Sporting News, 2023-10-20T22:00:05+00:00)
England vs. South Africa betting odds, prediction

The last time these two sides met was a tumultuous period to say the least, particularly for English rugby. South Africa's dominant 27-13 victory at Twickenham
proved to be the last game in charge for Eddie Jones, the coach who had taken England to the 2019 World Cup final. In his place, Steve Borthwick was hired, and
he's just one game away from repeating that feat.

England's form at this tournament has been perfect — at face value. Argentina, Japan, Chile, Samoa and Fiji have all provided highlights — Los Pumas are also in
the semi-finals — and England have beaten all of them. However, only Japan and Chile were defeated with bonus points, and the narrow wins over both Samoa and
Fiji in their last two games showed that England are certainly gettable.
</item 7>

<item 8>
Where to watch England vs South Africa: Live stream, TV channel, lineups, odds for 2023 Rugby World Cup semifinal (Sporting News, 2023-10-20T22:00:05+00:00)
The last time Cheslin Kolbe played England at a Rugby World Cup... 🇿🇦 🏆#RWC2023 | #ENGvRSA pic.twitter.com/Riw2uaSaKa — Rugby World Cup (@rugbyworldcup) October
19, 2023

Both sides' journeys have seen twists, turns, and bumps in the road, but it's led them both to within 80 minutes of a Rugby World Cup final. England will no
doubt be looking to echo the spirit of 2019, when they beat the All Blacks 19-7 to reach the final, but in South Africa, they face a side who have taken their
game to new heights just to reach the semis.

MORE: 2023 Rugby World Cup final: Details, kick-off time and every team's route

What time does England vs South Africa kick off?
</item 8>

<item 9>
ICC World Cup 2023 Most Runs and Wickets: Full List of top 10 batsmen and bowlers in 2023 edition of cricket World Cup, Know where Virat Kohli, Rohit Sharma,
Shami, Bumrah, rank (Zee Business, 2023-11-18T15:51:59+00:00)
ICC Men's ODI World Cup 2023 Most Runs and Wickets: With just the final of the ICC Men's ODI World Cup 2023 won by Australia beating India by six wickets on
Sunday, November 19, at Narendra Modi Stadium in Ahmedabad, the whole world is ready to witness who will be crowned the World Champions in the 50-over format of
the game. India defeated New Zealand by 70 runs in the first semi-final on Wednesday, November 15, at the Wankhede Stadium in Mumbai, and Australia beat South
Africa by three wickets in the second semi-final on Thursday, November 16, at the Eden Gardens in Kolkata.

During all these matches played, we witnessed some great knocks by the batter as well as some brilliant spells by the bowlers, which annihilated the opposition
and kept the audience in awe.
</item 9>

<item 10>
ICC World Cup 2023: India at Cricket World Cup semi-finals so far (Zee Business, 2023-11-11T19:12:00+00:00)
2023 World Cup, against New Zealand

To be played
</item 10>

<item 11>
IND vs AUS Head-to-Head Record in ODI World Cup: Check India vs Australia match stats in World Cup Finals and ODI series (Zee Business,
2023-11-18T13:02:06+00:00)
Australia posted a target of 329 runs, losing seven wickets in the process. Steven Smith scored a brilliant 105 runs off 93 balls, with Aaron Finch supporting
the innings with an 81-run knock of 116 balls. Umesh Yadav picked up four wickets for India. India crumbled within 233 runs in 46.5 overs. MS Dhoni was the
highest scorer for the Indian side in that match, with 65 runs off 65 balls.

James Faulkner picked up three wickets for Australia, with Mitchell Johnson and Mitchell Stark taking two wickets each.

Australia went all the way to lift their fifth ODI World Cup trophy, defeating New Zealand in the final.

2019, Match 14, ICC Cricket World Cup, The Oval

India took their revenge in this World Cup as they annihilated the Australian bowlers by posting a target of 353 runs in 50 overs.
</item 11>

<item 12>
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
</item 12>

<item 13>
Where to watch England vs South Africa: Live stream, TV channel, lineups, odds for 2023 Rugby World Cup semifinal (Sporting News, 2023-10-20T22:00:05+00:00)
TV channel Streaming USA — Peacock Canada TSN TSN+ UK ITV ITV X Australia Channel 9 Stan Sport, 9Now New Zealand Sky GO Sky GO India — Fan Code Hong Kong BeIN
Sports BeIN Sports Malaysia BeIN Sports BeIN Sports Singapore BeIN Sports BeIN Sports

MORE: Every single team to have lifted the Rugby World Cup trophy

England vs South Africa lineups, team news

"The start is crucial, the end is vital..."



- Steve Borthwick explains changing 3️⃣ of his England starters for Saturday's Rugby World Cup semi-final, from Liam Heagney 👨‍💻 in Paris. #RWC2023 #ENGvRSA
#EnglandRugby
</item 13>

<item 14>
Where to watch England vs South Africa: Live stream, TV channel, lineups, odds for 2023 Rugby World Cup semifinal (Sporting News, 2023-10-20T22:00:05+00:00)
The Springboks have done it tough, too, having already faced two of the tournament favourites on their way to the semi-finals. Comfortable defeats of Scotland,
Romania and Tonga couldn't mask the disappointment of the 13-8 loss to Ireland, which consigned South Africa to second in Pool B. However, with Ireland sent
home by the All Blacks in the quarters, the Bokke showed their World Cup mettle to edge the hosts 29-28 at the Stade de France, sneaking past Les Bleus in one
of the best games of rugby you're ever likely to see.
</item 14>

<item 15>
IND vs AUS Head-to-Head Record in ODI World Cup: Check India vs Australia match stats in World Cup Finals and ODI series (Zee Business,
2023-11-18T13:02:06+00:00)
IND vs AUS Head-to-Head Record in ODI World Cup: It feels like a call back from the past, as two decades ago, these two goliaths—India and Australia—of the
cricketing world faced each other in the final of the ICC Men's ODI World Cup in Johannesburg, South Africa.

Also Read: India vs Australia World Cup Final Free Live Streaming: When and how to watch ICC World Cup 2023 Final IND vs AUS match live on Mobile apps, tv,
laptop, online

Once again, we are at the same stage as the two-time World Champions India will be locking horns against the five-time World Champions Australia for the finale
of the ICC Men's ODI World Cup 2023 on Sunday, November 19, at the Narendra Modi Stadium in Ahmedabad.
</item 15>

<item 16>
ICC World Cup 2023: India at Cricket World Cup semi-finals so far (Zee Business, 2023-11-11T19:12:00+00:00)
India were playing their sixth semi-finals at a place where they played their first World Cup semi-final in 1983, Manchester. However, with a different team-
New Zealand.

Indian bowlers raised hopes when they restricted New Zealand to 239 for 8 in 50 overs with Bhuvneshwar Kumar taking three wickets.

Ross Taylor 74 runs (90 balls) and skipper Kane Willamson 67 runs (95 balls) were the top scorers for New Zealand. But Kiwi Man-of-the-Match bowler Matt Henry
sent openers KL Rahul and Rohit Sharma back to the pavilion soon, followed by the dismissal of Dinesh Karthik in the 10th over of India's innings.

With the Indian top-order failing, Ravindra Jadeja (77 runs in 59 balls) and wicketkeeper MS Dhoni (50 runs in 72 balls) tried to steady the Indian ship.

But once they fell, it was all over for India as they lost the battle by 18 runs.
</item 16>

<item 17>
Where to watch England vs South Africa: Live stream, TV channel, lineups, odds for 2023 Rugby World Cup semifinal (Sporting News, 2023-10-20T22:00:05+00:00)
In short, this semi-final requires a huge step up in quality for Borthwick's side, and that's why we think that the Springboks will be victorious at the Stade
de France. South Africa have already shown they can match, and beat, the world's best, and we think that they'll do the latter in Saint-Denis.
</item 17>

<item 18>
ICC World Cup 2023: India at Cricket World Cup semi-finals so far (Zee Business, 2023-11-11T19:12:00+00:00)
India moved to finals and beat Sri Lanka to win their second World Cup title.



2015 World Cup, against Australia

India were playing against Australia in the second semi-final of the 2015 World Cup in Sydney on March 26. Australia outplayed India in almost all departments
of the game.

Australian Man-of-the-Match Steve Smith (105 runs in 93 balls) and Aaron Finch (81 runs in 116 balls) were the guiding forces in taking Australia to 328 runs
for 7 wickets in 50 overs.

In reply, India needed a few big innings, but apart from skipper MS Dhoni (65 runs in 65 balls), Shikhar Dhawan (45 runs in 34 balls), and Ajinkya Rahane (44
runs in 68 balls), the batting order collapsed miserably as India lost the match by 95 runs.



2019 World Cup, against New Zealand

The cricket World Cup returned to England for the fifth time.
</item 18>

<item 19>
ICC World Cup 2023: India at Cricket World Cup semi-finals so far (Zee Business, 2023-11-11T19:12:00+00:00)
India, one of the favourites to win the World Cup, were very much on the path to victory till Mohammad Azharuddin, who top-scored with 64 runs was at the
crease.

But with the fall of his wicket, the Indian lower order collapsed like a house of placards as India lost the match by 35 runs.

1996 Wills World Cup, against Sri Lanka

The 1996 Cricket World Cup is known for Sri Lanka's emergence as a cricketing power on the horizon of world cricket and the beginning of Sachin Tendulkar's
extraordinary dominance in batting history.

Playing in the first semi-final at the Eden Gardens in Kolkata on March 13, India were favourites again, and they raised their hopes when they managed to
restrict Sri Lanka to 251 for 8 in 50 overs despite Man-of-the-Match Aravinda de Silva's 66 and Roshan Mahanama's 58.
</item 19>

<item 20>
Where to watch England vs South Africa: Live stream, TV channel, lineups, odds for 2023 Rugby World Cup semifinal (Sporting News, 2023-10-20T22:00:05+00:00)
The 2019 Rugby World Cup final between England and South Africa was one of many firsts: the game itself capped off a sensational tournament played outside of
the sport's traditional strongholds for the first time; it was the first final in which South Africa scored a try, and by full-time, the 32-12 score in the
Springboks' favour sealed a record-equalling third title for the Rainbow Nation.

Fast forward four years, and South Africa once again prepare to take on England in an all-or-nothing match to decide which nation gets a shot at World Cup glory
— but this time around, the pair meet in the semi-finals of the 2023 tournament, rather than the decider.

That won't make an ounce of difference for either camp, though, as these two long-time rivals prepare for a battle that could re-define the definition of a
'physical encounter.'
</item 20>

<item 21>
ICC World Cup 2023: India at Cricket World Cup semi-finals so far (Zee Business, 2023-11-11T19:12:00+00:00)
2003 World Cup, against Kenya

India were playing underdogs Kenya in the last-four stage match in Durban on March 20.

With Man-of-the-Match India skipper Sourav Ganguly scoring an unbeaten 111 runs in 114 balls and Man-of-the-Tournament Tendulkar scoring an 111-ball 83, India
comfortably scored 270 for four wickets in 50 overs.

In reply, Kenya were no match for Indian dominance, and barring their skipper Steve Tikolo, who top-scored with 56 runs, no Kenyan batsman could score more than
29 runs.

With Zaheer Khan taking 3 wickets in 9.2 overs for just 14 runs, India could easily beat Kenya by 91 runs.

Though India breezed into semi-finals easily, the tournament is also remembered for Kenya's best run in the World Cup as the team from a country with limited
cricket facilities and no proper grassroots structure could reach as far as semi-finals.
</item 21>

<item 22>
Where to watch England vs South Africa: Live stream, TV channel, lineups, odds for 2023 Rugby World Cup semifinal (Sporting News, 2023-10-20T22:00:05+00:00)
Click below for more ⤵️ https://t.co/LeUuw4kPfP — RugbyPass (@RugbyPass) October 19, 2023

Marcus Smith hardly put a foot wrong at fullback against Fiji, but the world champions are an entirely different proposition for England to deal with at the
back. The high-ball security and hulking frame of Freddie Steward returns to the No. 15 jersey in one of three changes, while at inside centre, Manu Tuilagi
starts hoping to repeat his heroics of four years ago, when his early try in a dominant semi-final performance took England past the All Blacks.

England's other two changes come in the tight five, with Steve Borthwick making some big calls ahead of this game. Joe Marler starts ahead of vice-captain Ellis
Genge at loosehead prop, and Ollie Chessum is replaced in the second-row by George Martin.
</item 22>

<item 23>
Where to watch England vs South Africa: Live stream, TV channel, lineups, odds for 2023 Rugby World Cup semifinal (Sporting News, 2023-10-20T22:00:05+00:00)
The latter is winning only his seventh England cap but gets the starting nod alongside Maro Itoje, with Genge, Chessum and the rest of the replacements hoping
to make an impact sizeable enough to overcome the Springboks' infamous 'bomb squad'.

Their time is now 🙌



Here is the @EnglandRugby team to take on reigning champions South Africa in the semi-final in Paris#RWC2023 | #ENGvRSA pic.twitter.com/nkf6rXBPUU — Rugby World
Cup (@rugbyworldcup) October 19, 2023

England starting XV: Freddie Steward, Jonny May, Joe Marchant, Manu Tuilagi, Elliot Daly, Owen Farrell, Alex Mitchell; Joe Marler, Jamie George, Dan Cole, Maro
Itoje, George Martin, Courtney Lawes, Tom Curry, Ben Earl.
</item 23>

<item 24>
ICC World Cup 2023: India at Cricket World Cup semi-finals so far (Zee Business, 2023-11-11T19:12:00+00:00)
2011 World Cup, against Pakistan

Fans from India and Pakistan wouldn't have expected a better occasion than a World Cup for the meeting of two teams.

The World Cup returned to India after 1996, and India and Pakistan were facing each other in the second semi-final at the IS Bindra Stadium in Mohali on March
30.

Man-of-the-Match Tendulkar rose to the occasion, scoring 85 runs for India.

Virender Sehwag scored 38 while Suresh Raina remained unbeaten on 36 as India scored 260 for 9 in 50 overs.

Pakistan seamer Wahab Riaz took 5 for 46 in 10 overs. In reply, Pakistan batsmen Misbah-ul-Haq (56 runs) and Mohammad Hafeez (43 runs) showed some resistance
from Pakistan, but in lack of a big innings lower down the order hit Pakistan's prospects badly as they were all out for 231 runs in 49.5 overs.
</item 24>

<item 25>
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
</item 25>

<item 26>
ICC World Cup 2023: India at Cricket World Cup semi-finals so far (Zee Business, 2023-11-11T19:12:00+00:00)
Also Read: IND vs NZ FREE Live Streaming: When and How to watch India vs New Zealand Cricket World Cup 2023 Semi-Final Match Live on Web, TV, mobile apps online

Also Read: AUS vs SA FREE Live Streaming: When and How to watch Australia vs South Africa Cricket World Cup 2023 Semi-Final Match Live on Web, TV, mobile apps
online

1983 Prudential Cup, against England

Having lost in the first round of the 1975 and 1979 World Cups, India were underdogs in the third World Cup, which was being held in England for the third
consecutive time.

However, in the World Cup, India wronged many of their critics, reaching semi-finals and playing against mighty hosts England in Manchester on June 22.

Skipper Kapil Dev struck three wickets while Roger Binny and Man-of-the-Match Mohinder Amarnath took two each as India restricted England to 213 all out in 60
overs.
</item 26>

<item 27>
Where to watch England vs South Africa: Live stream, TV channel, lineups, odds for 2023 Rugby World Cup semifinal (Sporting News, 2023-10-20T22:00:05+00:00)
England replacements: Theo Dan, Ellis Genge, Kyle Sinckler, Ollie Chessum, Billy Vunipola, Danny Care, George Ford, Ollie Lawrence.

The Springboks are unchanged from that epic quarter-final with France, having all played their part as Jacques Nienaber's side hung on in the second half. The
contribution of backs Kurt-Lee Arendse, Damian de Allende and Cheslin Kolbe saw all three cross for first-half tries, with Kolbe's coming off the back of a
charged-down conversion from France's Thomas Ramos. Preventing those two points proved crucial in the end.

The key contribution, though, came from Eben Etzebeth, who bulldozed his way over to score the only five-pointer of the second half and wrestle the game back
into South African hands.
</item 27>

<item 28>
Where to watch England vs South Africa: Live stream, TV channel, lineups, odds for 2023 Rugby World Cup semifinal (Sporting News, 2023-10-20T22:00:05+00:00)
The #Springboks are gearing up for another tough challenge this week with a semi-final clash against England 🇿🇦🏴󠁧󠁢󠁥󠁮󠁧󠁿#StrongerTogether #RWC2023
pic.twitter.com/HLSnnjWggB — Springboks (@Springboks) October 18, 2023

England are the only side still competing who are yet to lose a game, and that will stand them in good stead coming into this semi-final. Dominant wins over
Argentina, Japan and Chile sealed passage to the last eight, and after being given an almighty scare by Samoa in their final pool match, England overcame a
Fijian fightback in Marseille to triumph 30-24. Despite being outscored by three tries to two, captain Owen Farrell kicked 20 points, sending his side into the
last four at the expense of 2023's breakout stars.
</item 28>

<item 29>
IND vs AUS Head-to-Head Record in ODI World Cup: Check India vs Australia match stats in World Cup Finals and ODI series (Zee Business,
2023-11-18T13:02:06+00:00)
Australia batted first and posted a target of 261 runs courtesy of a 104-run knock of 118 balls by their captain, Ricky Ponting, with the loss of six wickets in
the process.

In the second innings, Sachin Tendulkar, Yuvraj Singh, and Gautam Gambhir scored three half-centuries as India changed down the target within 47.4 overs, losing
5 wickets in the process.

India went all the way to lift their second ODI World Cup trophy, defeating Sri Lanka in the finals.

2015, 2nd Semi-Final, ICC Cricket World Cup, Sydney

Australia got their revenge for the 2011 quarterfinals in this World Cup as they defeated India by 95 runs. Australia posted a target of 329 runs, losing seven
wickets in the process.
</item 29>

<item 30>
Where to watch England vs South Africa: Live stream, TV channel, lineups, odds for 2023 Rugby World Cup semifinal (Sporting News, 2023-10-20T22:00:05+00:00)
He and Franco Mostert pack down together in the second row, while hooker Bongi Mbonambi, utility Deon Fourie, veteran Duane Vermeulen and the devilish Kwagga
Smith will all be looking to back up superb quarter-final performances to take the Springboks to a second consecutive World Cup final.

South Africa starting XV: Damian Willemse, Kurt-Lee Arendse, Jesse Kriel, Damian de Allende, Cheslin Kolbe, Manie Libbok, Cobus Reinach; Steven Kitshoff, Bongi
Mbonambi, Frans Malherbe, Eben Etzebeth, Franco Mostert, Siya Kolisi, Pieter-Steph du Toit, Duane Vermeulen.

South Africa replacements: Deon Fourie, Ox Nche, Vincent Koch, RG Snyman, Kwagga Smith, Faf de Klerk, Handre Pollard, Willie le Roux.
</item 30>
=== END ITEM id=d9ea9e05ae89c513 ===
