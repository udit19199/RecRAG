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

=== ITEM id=4397fee98ce3e4eb ===
Gold sentences and facts:
Sentence 0: It would be Harbaugh's second suspension of the season, as the Wolverines self-imposed a three-game ban on Harbaugh to start the season amid an NCAA investigation that found Harbaugh lied to investigators.
  Fact 0.0: The upcoming suspension would be Harbaugh's second suspension of the season.
  Fact 0.1: The Wolverines self-imposed a three-game ban on Harbaugh to start the season.
  Fact 0.2: The self-imposed ban was amid an NCAA investigation.
  Fact 0.3: The NCAA investigation found that Harbaugh lied to investigators.
Sentence 1: Harbaugh reportedly watched the game from the team hotel, and now the leading piece of drama returns.
  Fact 1.0: Harbaugh reportedly watched the game from the team hotel.
  Fact 1.1: The leading piece of drama is returning now.

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
</item 1>

<item 2>
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
</item 2>

<item 3>
Michigan sign-stealing scandal news: Latest updates on possible Jim Harbaugh suspension from NCAA cheating investigation (Sporting News,
2023-11-06T23:58:04+00:00)
The Big Ten is also asserting that they are not actually sanctioning Harbaugh, for whom they have no proof of wrongdoing, but that he "embodies the university
for purposes of its football program." pic.twitter.com/S660G6zQaa — Stewart Mandel (@slmandel) November 10, 2023

Jim Harbaugh suspended for remainder of regular season

The Big Ten has officially announced a suspension for Jim Harbaugh for the remaining three games of the regular season, including the Wolverines' rivalry game
with Ohio State.

In a statement, the Big Ten said Michigan conducted an "impermissible, in-person scouting operation over multiple years."

OFFICIAL: Jim Harbaugh suspended for the remaining three regular season games.
</item 3>

<item 4>
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
</item 4>

<item 5>
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
</item 6>

<item 7>
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
</item 7>

<item 8>
Michigan sign-stealing scandal news: Latest updates on possible Jim Harbaugh suspension from NCAA cheating investigation (Sporting News,
2023-11-06T23:58:04+00:00)
Judicial officer: Timothy P. Connors



Connors is a U-M lecturer, adjunct professor at Wayne State and Vermont. pic.twitter.com/WR4G4WTtfH — Chris Vannini (@ChrisVannini) November 11, 2023

Michigan players respond to Harbaugh suspension

J.J. McCarthy and a handful of other Michigan players tweeted "Bet" at the same time on Friday night, hours after Harbaugh was suspended for the remainder of
the regular season. McCarthy wore a shirt that said "Free Harbaugh" when the 59-year-old was serving a separate suspension at the start of the season.

Michigan players had a simple message on social media following Jim Harbaugh's suspension 👀
</item 8>

<item 9>
Michigan sign-stealing scandal news: Latest updates on possible Jim Harbaugh suspension from NCAA cheating investigation (Sporting News,
2023-11-06T23:58:04+00:00)
MORE: Why Jim Harbaugh won't find refuge in the NFL

It was reported earlier in the week that coaches were putting pressure on the Big Ten to hit Michigan with some form of punishment, with Purdue's Ryan Walters
publicly calling for action and claiming the allegations are no longer simply allegations.

Harbaugh was already suspended three games at the start of the season, in a ban imposed by the school itself in an attempt to ward off a separate NCAA
investigation.

Friday, Nov. 3

Suspended football analyst Connor Stalions resigns

Michigan football analyst Connor Stalions, the focus of the NCAA's investigation, resigned on Friday night, the university confirmed.

Statement from Michigan (confirming earlier reports that Connor Stalions is no longer with the program): "Connor Stalions resigned his position with Michigan
Athletics this afternoon. We are unable to comment further regarding this personnel matter."
</item 9>

<item 10>
Michigan sign-stealing scandal news: Latest updates on possible Jim Harbaugh suspension from NCAA cheating investigation (Sporting News,
2023-11-06T23:58:04+00:00)
pic.twitter.com/Qm9iehYVTR — Pete Thamel (@PeteThamel) November 11, 2023

Harbaugh will not be on the sidelines vs. Penn State

Michigan filed a temporary restraining order against the Big Ten on Friday night, hoping to block Harbaugh's suspension. But there will not be a ruling before
the Wolverines' game vs. Penn State, with the hearing scheduled for Friday, Nov. 17.

No ruling on Michigan’s TRO filing today, so Jim Harbaugh will not coach vs. Penn State.



Statement from the school: “We look forward to presenting our case next week where we intend to demonstrate that the Big Ten has not acted legally or fairly.”
</item 10>

<item 11>
Michigan sign-stealing scandal news: Latest updates on possible Jim Harbaugh suspension from NCAA cheating investigation (Sporting News,
2023-11-06T23:58:04+00:00)
Yahoo says a multi-game suspension for Harbaugh is on the table, though Michigan is fully expected to fight a ban if one is handed down. It would be Harbaugh's
second suspension of the season, as the Wolverines self-imposed a three-game ban on Harbaugh to start the season amid an NCAA investigation that found Harbaugh
lied to investigators.

Saturday, Nov. 4

Harbaugh 'likely' to be suspended

According to The Athletic's Pete Thamel, the Big Ten might be getting closer to handing down a suspension for Harbaugh.

"Over the past few days, Petitti has spoken with the athletic directors, he's spoken with the presidents, he's spoken with the coaches. And the next steps
appear that there will be some type of punishment for Michigan. The most likely scenario here is some type of suspension for Jim Harbaugh," Thamel said on
ESPN's "College GameDay."
</item 11>

<item 12>
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
</item 12>

<item 13>
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
</item 13>

<item 14>
Michigan sign-stealing scandal news: Latest updates on possible Jim Harbaugh suspension from NCAA cheating investigation (Sporting News,
2023-11-06T23:58:04+00:00)
Rice on Saturday and at South Florida on Nov. 25.

I am extending an open invitation to Jim Harbaugh to join me on the sidelines for the next two Charlotte football games. It would be my honor to stand next to
you again as we did for three years — Biff poggi (@BiffPoggi) November 17, 2023

Thursday, Nov. 16

Michigan accepts Harbaugh's three-game suspension

Michigan has dropped its litigation against the Big Ten over Harbaugh's three-game suspension, and it will accept the punishment levied against the head coach.

From Michigan:

"This morning, the University, Coach Harbaugh, and the Big Ten resolved their pending litigation. The Conference agreed to close its investigation, and the
University and Coach Harbaugh agreed to accept the three-game suspension. Coach Harbaugh, with the University's support, decided to accept this sanction to
return the focus to our student-athletes and their performance on the field.
</item 14>

<item 15>
Michigan sign-stealing scandal winners and losers, from Jim Harbaugh to Tony Petitti (Sporting News, 2023-11-11T22:50:13+00:00)
On the other, he could emerge as a martyr for Michigan fans given there is reportedly no proof yet he knew of Stalion's rule-breaking.

Harbaugh will have a temporary restraining order hearing on Nov. 17, according to ESPN.com. That will determine whether he returns to the field for the final
two games against Maryland and No. 1 Ohio State.

It's an inescapable fact that Harbaugh is 31-3 – and Michigan is 35-3 – since the program bottomed out at the end of the 2020 season. While it is almost
impossible to quantify how much the alleged in-person scouting and sign-stealing would have helped the Wolverines over the past two seasons, it will be a tough
stain to erase knowing those violations are now on the Big Ten record.

What will Harbaugh do now? That is almost impossible to predict given his combative personality. Would he sign a contract extension and continue to fight the
NCAA? Will Harbaugh finally take the parachute to the NFL?
</item 15>

<item 16>
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
</item 16>

<item 17>
Michigan sign-stealing scandal news: Latest updates on possible Jim Harbaugh suspension from NCAA cheating investigation (Sporting News,
2023-11-06T23:58:04+00:00)
He initially told reporters he would speak during a scheduled hearing Friday, but then backtracked a bit and sounded as if he wasn't committed to that course,
pending legal advice.

Harbaugh said he will speak at the hearing on Friday. He said he’s looking for “due process.” Mentioned a civics class he took 40 years ago. “You’re innocent
until proven guilty. I’d like that opportunity.” — Nicole Auerbach (@NicoleAuerbach) November 13, 2023

Harbaugh clarifies that he’s not sure whether he’ll testify Friday. “That’s not my dance floor,” he says. “I’ve watched a lot of shows. I’ve watched Judge Judy
a lot.” — Austin Meek (@byAustinMeek) November 13, 2023

Sunday, Nov.
</item 17>

<item 18>
Michigan sign-stealing scandal news: Latest updates on possible Jim Harbaugh suspension from NCAA cheating investigation (Sporting News,
2023-11-06T23:58:04+00:00)
Here is Michigan’s 10-page letter to the Big Ten responding to the sportsmanship investigation, reporting with @RossDellenger 1/3 pic.twitter.com/P0oYgIbS67 —
Dan Wetzel (@DanWetzel) November 9, 2023

No timetable for Big Ten decision

The Big Ten is reviewing the "lengthy" official response from Michigan and Jim Harbaugh and it's unclear when a potential ruling might come down, says ESPN's
Pete Thamel. His colleague Adam Rittenberg reported Wednesday that a decision was expected by Thursday.

Source: The Big Ten is reviewing the lengthy written responses from both Michigan and Jim Harbaugh. There’s no indication of any timetable on a decision on
potential Big Ten discipline in the case of Michigan’s alleged illegal signal stealing.
</item 18>

<item 19>
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
</item 19>

<item 20>
Michigan sign-stealing scandal news: Latest updates on possible Jim Harbaugh suspension from NCAA cheating investigation (Sporting News,
2023-11-06T23:58:04+00:00)
🗣️ "Bet" pic.twitter.com/ENB7O8MyPv — The Sporting News (@sportingnews) November 10, 2023

Michigan announces intention to dispute suspension

Michigan has released a statement expressing frustration with the Big Ten's process and says it plans to seek a court order to block Harbaugh's suspension from
taking effect.

NEWS: Michigan says it intends to seek a court order with Jim Harbaugh preventing disciplinary action from taking effect.



Full statement: pic.twitter.com/p92yUCLsTX — Matt Fortuna (@Matt_Fortuna) November 10, 2023

Big Ten claims no evidence Harbaugh was aware of scheme

While the Big Ten says it's "proven" that an illegal sign-stealing scheme existed, the conference could not prove that Harbaugh knew about it. So, why was
Harbaugh suspended? The Big Ten asserts that, because he is the head coach, he has a responsibility for preventing this kind of scheme from taking place.
</item 20>
=== END ITEM id=4397fee98ce3e4eb ===

=== ITEM id=5da52babfb7d239a ===
Gold sentences and facts:
Sentence 0: “It was the millionaires of the Premier League against the youngsters of the Championship,” reflected the former Denmark forward.
  Fact 0.0: The speaker is a former Denmark forward.
  Fact 0.1: The speaker reflected that it was the millionaires of the Premier League against the youngsters of the Championship.
Sentence 1: The Commanders are coming off a dud last week against the Giants, while Dallas is flying high.
  Fact 1.0: The Commanders are coming off a dud last week against the Giants.
  Fact 1.1: Dallas is flying high.

Retrieved items:
<item 1>
Twitch is the best way to watch sports, when you can actually watch sports on Twitch (Polygon, 2023-10-20T17:30:00+00:00)
Prime Video subscribers who streamed on Twitch.

Twitch creator EsfandTV, who requested to go by Esfand for this story, is a streamer with 1.2 million followers who follow not only for gaming content, but for
his sports streaming as well. He says that broadcast rights are one of the aspects hindering Twitch from becoming a full-blown sports streaming platform.
Leagues still own the rights when it comes to where sports are aired, and Esfand says they need to adapt if they want to reach young and international sports
fans.

“The NBA is king,” Esfand says. “When it comes to hockey, the NHL is king. But what’s happening is these different leagues, they are the boss. They’re the
governing body. They have all the rights to everything and when it comes to content, they can be in control.”
</item 1>

<item 2>
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
</item 2>

<item 3>
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
</item 3>

<item 4>
NASCAR Might Have Just Saved Themselves With the $7.7 Billion Media Rights Deal (Essentially Sports, 2023-12-01T20:55:57+00:00)
This is one of the reasons why NASCAR did not want to gamble solely on cable users. With Amazon Prime, they had the leisure to explore the streaming arena and
possibly get a glimpse of how it could look in the future. However, along with that, there was a major reason behind the deal.

As Stern said, “So partially, they [NASCAR] did it to spread their eggs in multiple baskets but also part out of it was to get more money and to make sure they
get more money in the industry at the time when their teams are saying we need more money, we are not profitable currently.”

WATCH THIS STORY: NASCAR’s Charter Conundrum: A Twist in the Tale?

The SBJ journalist then explained that NASCAR was more or less left with no choice when it came to this deal.
</item 4>

<item 5>
Is the Coverage on Taylor Swift and Travis Kelce’s Romance Nothing but Hype or a Pivotal Shift in NFL’s Century-Old Game Plan Towards Global Recognition?
(Essentially Sports, 2023-10-25T21:30:41+00:00)
$28 billion. And just like that, their target will be met.

USA Today via Reuters NFL, American Football – Taylor Swift reacts while sitting next to Donna Kelce watching the Kansas City Chiefs vs Chicago Bears game
during the first half at GEHA Field at Arrowhead Stadium, Kansas City, Missouri, USA | Image Credits: Denny Medley-USA TODAY Sports/Reuters

And while analysts like Stephen A. Smith may go around telling people they should be embarrassed of themselves for getting so invested in someone else’s
relationship, it is good business for the NFL. TMZ reported that #87’s jersey sales skyrocketed 400% and his Instagram followers by 276%. Yahoo Finance reported
a 53% rise in viewership among teen girls. Who would have thought in their wildest dreams that the NFL would manage to rope in this part of the population? The
October 1 matchup between the Chiefs and the Jets drew 27 million viewers.
</item 5>

<item 6>
Totals Betting - What is an Over/Under Bet? (Sporting News, 2023-11-01T22:08:31+00:00)
A bettor believes it will be a high-scoring game and places an "over" bet.

The game concludes with the Cowboys scoring 28 points and the Eagles scoring 24 points, leading to a combined score of 52 points. Since 52 exceeds the set total
of 47.5, the bettor who wagered on the "over" wins their bet.

Over/under odds in basketball

Imagine NBA over/under odds between the Charlotte Hornets and the Houston Rockets. The bookmaker sets the over/under total at 215.5 points for the combined
score of both teams. A bettor, expecting a defensive showdown, places an "under" bet.

The game ends with the Hornets scoring 105 points and the Rockets scoring 108 points, yielding a combined score of 213 points. Since 213 is below the set total
of 215.5, the bettor who wagered on the "under" successfully wins their bet.
</item 6>

<item 7>
N.F.L. Executives on the Biggest Questions Facing Super Bowl Contenders (The New York Times, 2023-10-16T23:58:53+00:00)
This team is relatively young. They are scrappy. The fact that they have been there before gives them a chance. I think the quarterback is growing.”

• Cincinnati Bengals: Can Joe Burrow hold up for a full season?

What I think: Burrow is looking healthier over the past two weeks. The bye week comes at a great time for Burrow to get even healthier, but check out the
schedule on the other side: San Francisco, Buffalo, Houston, Baltimore, Pittsburgh and Jacksonville. Cincy still has a second game with the Steelers, plus games
against Kansas City and Cleveland. I’m worried Burrow will wear down against these defenses.

Exec comment: “You cannot always expect Joe Burrow to create magic with a bottom-half offensive line. The line is my concern all the way.”

Advertisement

2. Are teams sometimes going too far playing to win? The Eagles come to mind after falling 20-14 to the Jets.
</item 7>

<item 8>
N.F.L. Executives on the Biggest Questions Facing Super Bowl Contenders (The New York Times, 2023-10-16T23:58:53+00:00)
They are gaining more than 15 yards on just 11 percent of pass plays, down from 21 percent through six games last season and 19 percent for 2022 overall. Jalen
Hurts, A.J. Brown, DeVonta Smith and Dallas Goedert are still on the team, so I’m inclined to trust the production will return. But there’s also a lingering
fear defenses have caught up to aspects of the Eagles’ scheme. Does Philly have a counterpunch?

Advertisement

Exec comment: “People are going to crush Hurts and the offense. They are just churning the clock so much. It just feels like they are imposing their will on
teams through the run game and through their short passing and it just hasn’t taken off from an explosive passing standpoint, so when they play a team that can
move the ball on them through the air, it presents some challenges.”
</item 8>

<item 9>
Is the Coverage on Taylor Swift and Travis Kelce’s Romance Nothing but Hype or a Pivotal Shift in NFL’s Century-Old Game Plan Towards Global Recognition?
(Essentially Sports, 2023-10-25T21:30:41+00:00)
ADVERTISEMENT Article continues below this ad

How can you not love Taylor Swift? Most of the 12 Grammy Awardee’s songs are about that warm fuzzy feeling and chemical reaction that we call love. And her
songs have earned the support of 103.7992 million Swifties in the world. Yeah, we don’t have a round figure for that. We have the exact figure up to four
decimals. That’s how much data we have on her. Football has 184 million fans. The league is able to generate a revenue of $18 billion from 184 million. If it
adds 104 (after rounding 103.7992) million to that number, guess how much revenue it will generate? $28 billion. And just like that, their target will be met.
</item 9>

<item 10>
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
</item 10>

<item 11>
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
</item 11>

<item 12>
Is the Coverage on Taylor Swift and Travis Kelce’s Romance Nothing but Hype or a Pivotal Shift in NFL’s Century-Old Game Plan Towards Global Recognition?
(Essentially Sports, 2023-10-25T21:30:41+00:00)
Billionaire entrepreneur Mark Cuban, who paid $1.8 million (approx.) in penalties on behalf of his team, the Dallas Mavericks, fondly remembered the former NBA
commissioner David Stern, on his death in January 2020. Stern came and instituted an anti-drug agreement and even banned All-Star player Michal Richardson for
life for drug abuse. The revitalized image of the NBA helped rake in the moolah from sponsors and made it more appealing to fans. He even went against the tide
to open up the Olympic basketball competition to NBA players resulting in the famous “Dream Team” of 1992. America has never failed to win a medal in the 19
Olympic basketball games that it has played since then.

ADVERTISEMENT Article continues below this ad

The NFL has had its fair share of, um, let’s say, “bumps” down the road as well.
</item 12>

<item 13>
Is the Coverage on Taylor Swift and Travis Kelce’s Romance Nothing but Hype or a Pivotal Shift in NFL’s Century-Old Game Plan Towards Global Recognition?
(Essentially Sports, 2023-10-25T21:30:41+00:00)
The October 1 matchup between the Chiefs and the Jets drew 27 million viewers. And became the most-watched Sunday TV show since the last Super Bowl! And the
last Super Bowl had an actual concert by Rihanna.

Revealed: The Dark Side Of Pat McAfee & Why Rob Gronkowski Is In The Firing Line

Kaepernick, who? Deflategate, what? In the media frenzy surrounding the two lovebirds, other issues get thrown out of the window. How can fans forget that the
NFL is also organizing games outside the American borders with a view to garnering international appeal? The NFL has scheduled 5 games abroad this season. It is
also addressing the concussion controversies. It recently added both new concussion rules and a new $100 million initiative called “Play Smart, Play Safe,” to
make the sport safer for players. And all the effort is showing.
</item 13>

<item 14>
Travis Kelce admits the NFL is overdoing it with their Taylor Swift coverage (The Independent - Life and Style, 2023-10-04T19:58:00+00:00)
“The Taylor Swift and Travis Kelce news has been a pop cultural moment we’ve leaned into in real time, as it’s an intersection of sport and entertainment, and
we’ve seen an incredible amount of positivity around the sport.”

“The vast majority of our content has remained focused on the game, our players and variety of other initiatives, including our Toy Story Funday Football
alt-cast, the international games and more,” they continued.
</item 14>

<item 15>
N.F.L. Executives on the Biggest Questions Facing Super Bowl Contenders (The New York Times, 2023-10-16T23:58:53+00:00)
He is regulated to pass from the pocket, but you are taking away the biggest weapon the offense had when you do that. Odell (Beckham) is not that guy anymore.
Who are you truly afraid of on that offense? Nobody but Lamar, and if you are not using Lamar in a certain capacity, he becomes just like the rest of these
quarterbacks around the league — hit or miss.”

• Jacksonville Jaguars: Are they ready for prime time?

What I think: Trevor Lawrence’s knee injury was the big concern coming out of Sunday. If he’s out, expectations shift dramatically. More than the other AFC
contenders, the Jaguars seem early in their life cycle, still finding their way. That makes them a little harder to trust in big spots but also leaves greater
space for growth. Let’s hope those test results on Lawrence’s knee come back clear.

Exec comment: “I like Jacksonville. Doug (Pederson) has a championship pedigree. This team is relatively young. They are scrappy.
</item 15>

<item 16>
Is the Coverage on Taylor Swift and Travis Kelce’s Romance Nothing but Hype or a Pivotal Shift in NFL’s Century-Old Game Plan Towards Global Recognition?
(Essentially Sports, 2023-10-25T21:30:41+00:00)
Revisiting the NBA caricature back in time, the New York Knicks became the first all-Black team in 1979. But their fans made their disapproval very apparent.
“It wasn’t so much New York attitudes, but fans around the league let you know how they felt about you as an opponent and as a Black man as well,” Knicks star
Earl Monroe told Newsday.

USA Today via Reuters May 15, 2021; San Antonio, Texas, USA; Phoenix Suns forward Torrey Craig (12) and San Antonio Spurs forward Keldon Johnson (3) fight for
loose ball as Suns guards Devin Booker (1) and Chris Paul (3) look on in the third quarterat AT&T Center. Mandatory Credit: Scott Wachter-USA TODAY Sports

We could list out all the scandals, fights, and controversies here, but you get the idea.
</item 16>

<item 17>
Are NBA Christmas Day Games a Better Tradition Than NFL Thanksgiving Day Games? (Essentially Sports, 2023-12-25T21:27:33+00:00)
However, when comparing the two festive traditions, which one holds more significance, or which tradition has garnered greater popularity?

How do NBA’s Christmas Day games compare to NFL’s Thanksgiving Day games?

ADVERTISEMENT Article continues below this ad

Both traditions hold a special place in the hearts of fans, creating an annual spectacle that goes beyond the typical sporting event. The NBA’s Christmas Day
games is a tradition which is even older than the NFL’s Super Bowl. The first Christmas Day game dates back to 1947, when the league featured three games as a
festive treat for basketball fans. Since then, the NBA Christmas Day has only grown in stature and significance. Last season, Christmas Day games averaged 4.27
viewers across all platforms. The significance of this tradition can be evaluated by the fact that these games attract about 200% more viewers than any other
regular season game.

USA Today via Reuters Credit: Kevin Jairaj-USA TODAY Sports

However, despite these impressive viewership numbers, the NFL’s Thanksgiving Day games eclipse them by a considerable margin.
</item 17>

<item 18>
Are NBA Christmas Day Games a Better Tradition Than NFL Thanksgiving Day Games? (Essentially Sports, 2023-12-25T21:27:33+00:00)
This season, NFL’s Thanksgiving Day games created history. Across all the three games, the NFL recorded an average viewership of 34.1 million. These figures
surpassed last year’s record viewership of 33.6 million. Furthermore, the combined unduplicated audience for all three games reached 133 million, marking the
second-highest total audience on Thanksgiving, surpassed only by last year’s numbers.

These figures paint a pretty one-sided picture, but the choice is much more nuanced. Comparing the NBA Christmas Day Games and NFL Thanksgiving Day Games comes
down to personal preference. Each tradition holds its own significance, adding to the diverse fabric of American sports culture. Whether one prefers the slam
dunks and three-pointers of the NBA or the hard-hitting action on the football field, both traditions offer a festive and entertaining way to celebrate holidays
with family and friends.

Are the NFL’s new decisions threatening the NBA’s Christmas Day dominance?
</item 18>

<item 19>
Is the Coverage on Taylor Swift and Travis Kelce’s Romance Nothing but Hype or a Pivotal Shift in NFL’s Century-Old Game Plan Towards Global Recognition?
(Essentially Sports, 2023-10-25T21:30:41+00:00)
From Colin Kaepernick taking a knee during the national anthem to protest against racism, to domestic violence by players, to the Deflategate and Bountygate
scandals. But there have been bigger crises as well. The Union of Concerned Scientists accused the NFL of trying to intimidate scientists studying the link
between CTE and pro football. The NFL is already way past other leagues in terms of revenue, and now with the pivotal shift in its strategy, “America’s favorite
pastime” is on its way to world domination.

ADVERTISEMENT Article continues below this ad

Editorial Disclaimer: The views expressed are those of the author and do not necessarily reflect the views of EssentiallySports.

Watch This Story: Will Patrick Mahomes Eclipse Tom Brady as the NFL Goat? Travis Kelce Responds
</item 19>

<item 20>
Moneyline Betting - What is the Moneyline and How Does it Work? (Sporting News, 2023-11-01T21:25:48+00:00)
Moneyline odds in football

NFL moneyline odds for a game between the Dallas Cowboys and the New York Giants might look something like this:

Dallas Cowboys: -130

-130 New York Giants: +110

Here, the Cowboys are the favorites, indicated by the negative odds. If you bet $130 on the Cowboys and they win, you would earn a profit of $100.

The New York Giants are the underdogs, as shown by the positive odds. If you wager $100 on the Giants and they pull off an upset victory, you would earn a
profit of $110.

Moneyline odds in basketball

NBA moneyline odds for a game between the Los Angeles Lakers and the Miami Heat might be presented as:

Los Angeles Lakers: -150

-150 Miami Heat: +130

In this scenario, the Lakers are favored to win, as indicated by the negative odds. If you back the Lakers with a bet of $150 and they win, you would earn a
profit of $100.

The Heat are the underdogs.
</item 20>

<item 21>
Which M.L.B. Playoff Team Has the Most Daunting Home Field? We Ranked Them. (The New York Times, 2023-10-08T21:56:00+00:00)
By Chad Jennings, C. Trent Rosecrans and Stephen J. Nesbitt

In one American League Wild Card Series, the whole thing turned on a play designed around crowd noise. It was too loud at Target Field, Twins shortstop Carlos
Correa realized, for the third-base coach to warn Blue Jays baserunners about a developing pickoff play. The Twins used the noise to their advantage, picked off
Vladimir Guerrero Jr. and secured a two-game sweep.

Advertisement

In the other AL Wild Card Series, empty seats were everywhere. The Rays are an excellent team that thrives on their underdog status — low payroll, injured
players, they always find a way — but playing in front of two of the smallest postseason crowds in the past 100 years, the Rays were swept by a Rangers team
that had nine fewer wins in the regular season. The indifference was deafening.

Home-field advantage, it turns out, just isn’t the same from one ballpark to the next.
</item 21>

<item 22>
Which M.L.B. Playoff Team Has the Most Daunting Home Field? We Ranked Them. (The New York Times, 2023-10-08T21:56:00+00:00)
It’s not that we don’t think Arizona can beat L.A. at home — the Diamondbacks split six games against the Dodgers at Chase Field this season — but it’s less
than a seven-hour drive from Dodger Stadium to downtown Phoenix, and we’re concerned that all the quirks of that ballpark could be rendered meaningless if
there’s a bunch of Dodgers blue in the crowd.

Advertisement

There’s a lot to like. The pool. The retractable roof. We love it. We hope you prove us wrong. Show up, go nuts, and leave a comment — click to subscribe! —
telling us we don’t know a rattlesnake from a water moccasin. We deserve it.

But if we had to go on the road for a playoff game, at least in this round, we’d choose your ballpark. Consider it a compliment! Or maybe we just like the dry
heat.

No. 7: Texas Rangers Team Venue Opened Capacity Att.
</item 22>

<item 23>
European roundup: Bellingham helps Real Madrid go top while Bayern draw (The Guardian, 2023-09-30T21:17:47+00:00)
If you already have the Guardian app, make sure you’re on the most recent version.

In the Guardian app, tap the Menu button at the bottom right, then go to Settings (the gear icon), then Notifications.

Turn on sport notifications. Was this helpful? Thank you for your feedback.

“After the break, we were more compact and certainly more energetic, hungrier and more determined,” Milan’s coach, Stefano Pioli, told DAZN. “That is a great
sign when playing every three days and it was a fully deserved victory.”

Milan suffered a potential injury blow as the midfielder Ruben Loftus-Cheek limped off in the first half. “It would be a shame if we lost Ruben, he felt a
twinge near the groin area,” Pioli said. “Unfortunately, injuries tend to come when players are in good shape.”
</item 23>

<item 24>
Are NBA Christmas Day Games a Better Tradition Than NFL Thanksgiving Day Games? (Essentially Sports, 2023-12-25T21:27:33+00:00)
Are the NFL’s new decisions threatening the NBA’s Christmas Day dominance?

The enduring success and importance of Christmas Day games suggest that this 77-year-old tradition will continue for the foreseeable future. However, the NFL’s
recent decisions could threaten the NBA’s stronghold on Christmas. Traditionally, the NFL has avoided Christmas Day games that fall outside of their
broadcasting window. However, last year the NFL averaged 22.1 million viewers on Christmas Day games, and 2023 could be even bigger.

via Reuters REUTERS/Kai Pfaffenbach |

The NFL has lined up a mouth-watering slate for Christmas. Superstars like Patrick Mahomes, Brock Purdy and Jalen Hurts are all set to take the field on
Christmas. According to some estimations, the NFL’s total viewership during the festive period could outperform that of the NBA by almost 75 million viewers.
This creates a huge dilemma for the NBA, as traditionally they have maintained a firm grip on Christmas.
</item 24>

<item 25>
Travis Kelce admits the NFL is overdoing it with their Taylor Swift coverage (The Independent - Life and Style, 2023-10-04T19:58:00+00:00)
“I appreciate you always trying to make it to both of our games.”

“She also told me thank you because she was in an alternate universe, she was sitting there with Deadpooland Wolverine,” Kelce added, referencing Reynolds’ and
Jackman’s respective superhero movie characters. “It was a hot ticket, man, you know what I’m saying? That suite was rockin’.

“They were enjoying themselves. Somebody told me that everybody was enjoying themselves in that suite,” Kelce said.

Not only has NFL viewership increased since Swift began attending her rumoured boyfriend’s games, but jersey sales for the Chiefs tight end have reportedly
spiked by 400 per cent, according to a report from the NFL’s e-commerce partner, Fanatics.

Following critcism the league is focusing too much on Swift, the NFL responded with an official statement handed to People on 4 October.

“We frequently change our bios and profile imagery based on what’s happening in and around our games, as well as culturally,” the message read.
</item 25>

<item 26>
World Series Game 2: Arizona Diamondbacks v Texas Rangers – live (The Guardian, 2023-10-28T23:00:28+00:00)
29 Oct 2023 03.21 GMT Final thoughts Okay, we officially have a series! Assuming one has no rooting interest in either of the remaining teams, there’s no
greater start to a World Series than a split in the first two games. A 1-1 split is an ideal situation for the casual fan as it usually portends a longer
series. It is, of course, a less than ideal situation for the Rangers. Today, their weaknesses were exposed and they know their two-game home stand could have
gone worse: they needed plenty of late-inning drama just to pull off a come-from-behind win in Game 1. What happens next is anybody’s guess. The World Series
heads to Arizona for Monday’s Game 3 where Rangers’ Max Scherzer and the Diamondbacks’ Brandon Pfaadt are currently penciled in as the probable starters. Stay
tuned to the Guardian for further coverage of the 2023 World Series as it develops but this will wrap up today’s Game 2 liveblog.
</item 26>

<item 27>
After a Stunning Snub, College Football Is Facing an Identity Crisis (The New York Times, 2023-12-03T23:04:37+00:00)
This sport’s identity crisis has existed since the inception of the College Football Playoff a decade ago, but never before have we had to come to terms with it
the way we did on Sunday.

That identity crisis? Whether the teams that play for the national championship should be the best or the most deserving. In the nine years before this season,
the best and most deserving seamlessly became one, resulting in cut-and-dried decisions on which teams would make the field.

Advertisement

In the final season of the four-team era — before it expands to 12 in 2024 — the CFP committee was charged with a very difficult decision that was guaranteed to
result in a worthy team feeling cheated. The committee, for the first time, was actually going to have to choose what it values more — the teams that earned it
or the teams that look the best on TV.

Florida State, the most deserving, became the first unbeaten Power 5 conference champion left out of the field. The Seminoles were the ones left feeling
cheated.
</item 27>

<item 28>
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
</item 28>

<item 29>
NASCAR Might Have Just Saved Themselves With the $7.7 Billion Media Rights Deal (Essentially Sports, 2023-12-01T20:55:57+00:00)
Is the new media rights deal a game-changer for NASCAR?

ADVERTISEMENT Article continues below this ad

Just a couple of days back, NASCAR finalized a $7.7 billion deal with Fox Sports, NBC, Amazon, and Warner Bros. Discovery. With such a lucrative deal securing
seven years of revenue, surely it seems like a lot of money. But is that really the case?

NASCAR secured almost a 40% increase over their current media rights deal yet there’s something that doesn’t fit perfectly. Sports Business Journal journalist
Adam Stern recently cleared a few things out when he attended the Dirty Mo Live podcast. With the Cup Series already facing a 5% drop in viewership this year,
Stern discussed how the new deal could impact the ratings.

At first, the renowned journalist explained how the sanctioning body would have loved to stay with Fox Sports and NBC. However, given that it would have most
likely resulted in less money, NASCAR opted to explore other options.
</item 29>

<item 30>
N.F.L. Executives on the Biggest Questions Facing Super Bowl Contenders (The New York Times, 2023-10-16T23:58:53+00:00)
The Jets would have taken possession with barely more than a minute remaining, no timeouts and Zach Wilson behind center facing a ferocious Eagles pass rush.
Was having Hurts pass in a situation when gaining a first down is unlikely the smart play under the circumstances?

“Are coaches not willing to go against the slight statistical benefit that WPA (win probability added) provides?” the exec asked. “Is it no longer OK to let
your defense win the game when it clearly has a huge talent/performance advantage? This Eagles game was 14-12, not 37-35.”

Critics attuned to the trend toward aggressiveness pounce when teams pass up chances to go for the kill. Coaches know team owners tend to be fans, not football
strategists. They know owners listen to these criticisms. That can provide an incentive to make the aggressive play, to go down swinging.

“Everybody thinks they are playing Brady, Manning or Rodgers in two-minute,” a veteran coach said.
</item 30>

<item 31>
Which M.L.B. Playoff Team Has the Most Daunting Home Field? We Ranked Them. (The New York Times, 2023-10-08T21:56:00+00:00)
The “intensity” of baseball in L.A. isn’t the same as it is in, say, St. Louis or Boston or any other city where athletes are the biggest celebrities in town.

Advertisement

But by the middle innings, Dodger Stadium is packed, and by the late innings of playoff games, the place has the over-the-top intensity of a Michael Bay car
chase (Mookie Betts as Bumblebee). More than one opposing pitcher this season had to tweak his PitchCom device to deal with the late-inning noise that came from
such a massive ballpark, one that once again held the largest total attendance in baseball this year.

“You’ve got a four-deck stadium,” Freddie Freeman said this summer. “It’s the only one in baseball. You’ve got 50,000 people, every single night here. For them
to go out of their way to make you feel good when you’re doing your job, it means a lot. I do appreciate it.
</item 31>

<item 32>
Travis Kelce admits the NFL is overdoing it with their Taylor Swift coverage (The Independent - Life and Style, 2023-10-04T19:58:00+00:00)
In conversation with his brother Jason, 35, the two addressed concerns from longtime football enthusiasts who were frustrated by the constant Swift references
throughout Sunday’s game at MetLife Stadium.

Not only did the football league promote last Sunday’s game by speculating whether the Grammy winner will attend the Chiefs-Jets match, but cameras were
constantly panning to the star-studded suite. During the game, Swift was joined an A-list group of guests, including her best friends Blake Lively and Ryan
Reynolds, Game of Thrones star Sophie Turner, and actor Hugh Jackman.

The NFL also introduced a special segment with Carson Daly and The Voice, where the famed host attempted to explain the specifics of football to “Swifties.”

In the New Heights podcase episode, titled “The NFL Needs to Calm Down”, Jason asked his brother plainly whether he thought the football was “overdoing it” with
the coverage of Swift during the game.

“I think everybody is just, like, overwhelmed,” the Chiefs tight end replied.
</item 32>

<item 33>
NASCAR Might Have Just Saved Themselves With the $7.7 Billion Media Rights Deal (Essentially Sports, 2023-12-01T20:55:57+00:00)
Oct 8, 2023; Concord, North Carolina, USA; Cars crowd into turn four after a restart during the Bank of America Roval 400 at Charlotte Motor Speedway Road
Course. Mandatory Credit: Jim Dedmon-USA TODAY Sports

There is no denying how the new media rights deal has stunned the NASCAR community. With many still trying to digest it, the deal has undoubtedly given rise to
divided opinions. While the figure of $7.7 billion sounds lucrative, there seems to be more to it than what meets the eye.

America’s Favorite Video Today

It is public knowledge that NASCAR officials invested plenty of time in negotiating the media rights deal. To secure an average annual revenue of $1.1 billion
is something, especially when the sport is supposedly on the wane. However, it looks like the sanctioning body has made it count when it mattered the most.

Is the new media rights deal a game-changer for NASCAR?
</item 33>

<item 34>
NFL power rankings: Lions, 49ers level up; Texans jump Bengals, Bills; Jets, Patriots sink to bottom for Week 11 (Sporting News, 2023-11-13T21:54:00+00:00)
Week 10 in the NFL had some of the league's best teams not playing. But that didn't stop the excitement, with five teams winning on walk-off field goals and
more games coming right down to the wire.

With that, there were some surprising results and expected domination. Some favorites dug deep to win. Some underdogs turned in shocking upsets.

Putting the latest wins and losses into a blender, here's Sporting News' updated look at how all teams stack up against each other, 1-32 going into Week 11:

NFL WEEK 11 PICKS: Straight up | Against the spread

NFL power rankings Week 11

1. Philadelphia Eagles 8-1 (previous week: 1)

The Eagles enjoyed a bye trying to get a little healthier for the more difficult stretch run in a top-heavy NFC as they have tough games at the Chiefs (the
Super Bowl 57 rematch), vs. the Bills, vs.
</item 34>

<item 35>
NASCAR Might Have Just Saved Themselves With the $7.7 Billion Media Rights Deal (Essentially Sports, 2023-12-01T20:55:57+00:00)
He said, “NASCAR is in a tough situation because they had to deliver more money as well and clearly if they went this route, that’s a good indication that’s
because they had to go this route to deliver more money for the industry.”

While NASCAR secured an impressive deal, there could be downsides to it and Stern explained it beautifully.

Adam Stern points out a key potential drawback with NASCAR’s new media rights deal

Safe to say, NASCAR has been struggling to keep their teams happy in terms of profitability. Since the introduction of the Next Gen cars, it has only become
more difficult for the Cup Series teams and teams in other competitions. While they have taken a commercial hit, they have piled up pressure on NASCAR to help
out in this case.

ADVERTISEMENT Article continues below this ad

In addition to that, Sports Business Journalist Adam Stern detailed one important issue with the new media rights deal. Speaking on the same podcast, he said,
“When you look at the deal with Warner Bros.
</item 35>

<item 36>
Is Google Search better than the rest? And is that fair? (The Age, 2023-10-22T18:00:00+00:00)
Some critics, including Megan Grey, contend that Google uses semantic keyword matching to make searches less precise on purpose, which widens the pool of
content served and increases auctions for its ads. The quality tests, then, would ensure that the systems were still returning results that were justifiably
related to the input.
</item 36>

<item 37>
Which M.L.B. Playoff Team Has the Most Daunting Home Field? We Ranked Them. (The New York Times, 2023-10-08T21:56:00+00:00)
No. 7: Texas Rangers Team Venue Opened Capacity Att. (2023) Rangers Globe Life Field 2020 40,300 31,272

Memorable postseason moment: There have been 16 postseason games in Globe Life Field history, but the Rangers have never appeared in a playoff game at their
new-ish home. How did that happen? Well, 2020. It was a weird time. The Dodgers played in all 16 of those during the 2020 expanded playoffs, sweeping the Padres
in the NLDS, going seven games with the Braves in the NLCS and then winning the World Series.

Our take: This is the great unknown. Globe Life Field opened in a season that had no fans, and it remained open for losing seasons in 2021 and 2022. Saturday
will be our first look at the place when the home team is actually playing meaningful baseball in October. Hard to rank it any higher until we know what we’re
dealing with.
</item 37>

<item 38>
Which M.L.B. Playoff Team Has the Most Daunting Home Field? We Ranked Them. (The New York Times, 2023-10-08T21:56:00+00:00)
Hard to rank it any higher until we know what we’re dealing with.

Largest crowd ever for a game at Globe Life Field. #StraightUpTX pic.twitter.com/4I7NIB5eF8 — Chris Halicke (@ChrisHalicke) May 1, 2022

The old Rangers ballpark could drum up a Texas-sized atmosphere worthy of Stevie Ray Vaughan and Pat Green, but the new place — at least in the regular season —
feels very much like a multi-events space that happens to be hosting a ballgame. Maybe that changes in the playoffs? The Rangers haven’t been home since Sept.
24 when they won their last five games in Arlington, the last three of which went a long way toward keeping the Mariners out of the postseason.

With a closed roof containing all the noise, Globe Life is sure to get awfully loud, and nothing turns a glorified convention center into a real ballpark quite
like a late-inning rally in October.

No.
</item 38>

<item 39>
NASCAR Might Have Just Saved Themselves With the $7.7 Billion Media Rights Deal (Essentially Sports, 2023-12-01T20:55:57+00:00)
Stern then pointed out how the media rights revenue plays a key role in sports as he used the MLS as an example.

He said, “You look at MLS. MLS takes a big deal with Apple, almost all of their games behind the Apple TV’s paywall. That was the money they could get and they
had to deliver for their industry. Their industry is expected to be a growing business and one of the most sporting ways to be a growing business in sports is
to have growing media rights revenue.”

While he explained that it was a similar case with NASCAR, he added one more angle because of which this deal possibly materialized. It is no secret that the
cable universe has been declining. As Stern discussed, every quarter, there is a drop in a couple of percentages of subscribers, which is very concerning.

This is one of the reasons why NASCAR did not want to gamble solely on cable users.
</item 39>

<item 40>
Point Spread Betting - What is a Point Spread Bet? (Sporting News, 2023-11-01T22:50:52+00:00)
What is a push in a spread bet?

A "push" in betting refers to a situation where the final result of an event matches the set point spread, resulting in neither a win nor a loss for bettors.
When a push occurs, the wagers are typically refunded.

For instance, let's consider an NFL game where the Dallas Cowboys are favored by 7 points against the Philadelphia Eagles.

If the game ends with the Cowboys winning by exactly 7 points, those who bet on either team experience a push. Their wagers would be refunded as the Cowboys
neither exceeded nor fell short of the 7-point spread.

Sportsbooks often employ half-points in their lines to avoid this scenario. (They don’t want to return bets.) For example, if our point spread was Cowboys -7.5,
the game could not possibly end in a push.

Point spread betting by sport

Spread betting is popular across sports, but there can be slight differences. Let’s break it down.
</item 40>

<item 41>
Is the Coverage on Taylor Swift and Travis Kelce’s Romance Nothing but Hype or a Pivotal Shift in NFL’s Century-Old Game Plan Towards Global Recognition?
(Essentially Sports, 2023-10-25T21:30:41+00:00)
However, the NFL does not just want to stop at $18 billion. It wants to become a $27 billion league in 4 years. But it must address a few skeletons in its
closet before reaching that goal.

If you were around in the 1970s (that would make you at least 50 years old today), you would know that the NBA did not have the mass appeal that it enjoys now.
With average game attendance hovering around 8,000 a game, it was one of the darkest periods of the basketball league. Drug abuse, brawls on the hardwood court
that sometimes spilled over onto coaches and players – it wasn’t a pretty sight. As reported in the Washington Post, even though there are no reliable figures
on cocaine use by players, “estimates by people in the game range from 40 percent to 75 percent.”
</item 41>

<item 42>
After a Stunning Snub, College Football Is Facing an Identity Crisis (The New York Times, 2023-12-03T23:04:37+00:00)
It’s very easy to fathom why the committee couldn’t choose between Alabama and Texas. Alabama is a one-loss SEC champion that beat Georgia on Saturday, ending
the Bulldogs’ 29-game winning streak. Texas, a one-loss Big 12 champion, beat Alabama by 10 points in Tuscaloosa in September.

Many fans were hoping that the SEC would be left out entirely for the first time, but the committee — charged with picking the “best teams” — couldn’t ignore
what the ultra-talented Crimson Tide accomplished. But if Alabama goes, how could the committee leave the team out that beat it?

It couldn’t.

This is probably the path of least resistance. Outside of Florida State fans, the general population will move on convinced that the Playoff semifinals will be
more entertaining with more high-level teams. The best teams, as they say, won out.

The problem with choosing the best is that it’s entirely subjective and ultimately misleading given this is a sport that routinely features unpredictable
results and unforeseen runs.
</item 42>

<item 43>
Travis Kelce admits the NFL is overdoing it with their Taylor Swift coverage (The Independent - Life and Style, 2023-10-04T19:58:00+00:00)
Per a Deadline report, NBC accumulated an average of 27m viewers, with a total of 29m tuning in during the second quarter.

“That’s gotta be an NFL record right there,” Jason told his younger brother, as he noted that about two million viewers were estimated to be female. “I think a
few of them might have been there for Deadpool and, uh, Taylor.”

In the YouTube comments under the New Heights video podcast, fellow fans shared their opinions on the NFL’s coverage. “I agree the NFL needs to calm down.
Pissing off football fans is not how you repay your boi [sic] for all he’s done for the sport,” one person wrote.

Another remarked: “Pity Taylor. All she ever wanted was to watch the football game. Now she’s being blamed and hated left and right.”

“The NFL needs to seriously tone it down and let Trav and Tay explore this beautiful thing they have going on in peace,” someone else said.
</item 43>

<item 44>
Travis Kelce admits the NFL is overdoing it with their Taylor Swift coverage (The Independent - Life and Style, 2023-10-04T19:58:00+00:00)
“I think everybody is just, like, overwhelmed,” the Chiefs tight end replied.

“Take away your feelings for Taylor,” Jason added. “What is your honest opinion on how the NFL is treating celebrities at the games?”

Seemingly beating around the bush, Kelce remarked: “I think it brings a little more to the atmosphere, it brings a little bit more to what you’re watching. But
at the same time, I think...”

Before his younger brother could continue his thought, the Philadelphia Eagles player chimed in: “They’re overdoing it.”

“They’re overdoing it a little bit, for sure,” Kelce agreed. “Especially my situation. I think they’re just trying to have fun with it.”

While sports lovers may not be too pleased with the NFL’s focus on Swift, the football league’s supposed obsession with ongoing Swift and Kelce dating rumours
has paid off in terms of ratings.
</item 44>

<item 45>
Totals Betting - What is an Over/Under Bet? (Sporting News, 2023-11-01T22:08:31+00:00)
Push

A push occurs when the combined score of a game matches the set total established by the bookmaker, meaning that bettors neither win nor lose, and their
original wager is refunded. However, to avoid this outcome, most sportsbooks employ half-points in their totals.

By using half points, such as setting a total at 200.5 instead of 200, the sports betting sites ensure that a game's combined score cannot possibly match the
set total exactly.

Totals betting by sport

Over/under betting is a major draw across many popular sports. Let’s break down how it can look across sports.

Over/under odds in football

Let's consider some NFL over/under odds between the Dallas Cowboys and the Philadelphia Eagles.

The bookmaker sets the over/under total at 47.5 points for the combined score of both teams. A bettor believes it will be a high-scoring game and places an
"over" bet.
</item 45>

<item 46>
N.F.L. Executives on the Biggest Questions Facing Super Bowl Contenders (The New York Times, 2023-10-16T23:58:53+00:00)
GO DEEPER Behind the AI magic that lets Amazon's Prime Vision show the NFL like never before

• Buffalo Bills: Are the defensive injuries too much to overcome?

What I think: The rest of the conference is the key variable. The Bills are diminished. They have imperfections. But Buffalo already defeated the Dolphins. The
other AFC teams aren’t exactly peaking. Kansas City and Cincinnati appear easier to defend this season than in the recent past. We’ll see how well the Dolphins
hold up. This can still be Buffalo’s season.

Exec comment: “Matt Milano and Tre’Davious White, those are huge injuries for that football team. The Bills are not as dominant as they felt going into it with
so many injuries on defense. But the division they are in, they will still be able to slug it out and get to where they need to go. Nobody has really emerged in
the AFC to unseat these teams at the top.”
</item 46>

<item 47>
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
</item 47>

<item 48>
N.F.L. Executives on the Biggest Questions Facing Super Bowl Contenders (The New York Times, 2023-10-16T23:58:53+00:00)
The Eagles come to mind after falling 20-14 to the Jets.

Most would agree teams have gotten smarter in their handling of fourth-down situations, going for it when it makes sense, instead of punting almost out of
habit. Making the aggressive play to win can absolutely be the smart play, regardless of results. It’s progress when teams seriously wrestle with these
questions, not just on fourth down, but in all aspects of their strategy.

But when the Eagles tossed a killer interception on third-and-9 from their own 46-yard line while protecting a 14-12 lead against the Jets with two minutes left
in the fourth quarter, a longtime NFL exec reached out with a question: “Have we come too far with coaches wanting to ‘win’ the game instead of sometimes being
more traditional?”

His point: Philadelphia could have run the ball on third-and-9, then punted from around midfield.
</item 48>

<item 49>
N.F.L. Executives on the Biggest Questions Facing Super Bowl Contenders (The New York Times, 2023-10-16T23:58:53+00:00)
• Detroit Lions: Have they fixed their defense?

What I think: The Lions are 11.3 EPA per game better on defense from last season to this season, the second-largest gain for any team since last season. The
schedule has something to do with that, but the 2022 defense was so bad, it couldn’t stop Carolina. Detroit will win the NFC North and return to the playoffs,
which will make this season successful no matter what happens from there. But we won’t find out until the postseason just how much better the defense might be,
because the remaining schedule features so many weak offenses. That includes Chicago twice, Green Bay, Denver and Las Vegas.

Exec comment: “Yes, I do buy them. This is a league full of teams who do not really know who they are week to week. Detroit knows who they are, and they play to
their identity. Anytime you know who you are and play to that style, you have a chance to be dangerous.”
</item 49>

<item 50>
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
</item 50>
=== END ITEM id=5da52babfb7d239a ===

=== ITEM id=28dddb0c80450023 ===
Gold sentences and facts:
Sentence 0: But “all of that, all of it, was built on lies,” Rehn declared to the jury.
  Fact 0.0: Rehn declared to the jury that 'all of that, all of it, was built on lies.'
Sentence 1: And after Bankman-Fried left Jane Street, the high-frequency trading firm where the former billionaire got his start in finance after MIT, he persuaded Yedidia to join him as a trader at Alameda and then as a developer at FTX.
  Fact 1.0: Bankman-Fried left Jane Street.
  Fact 1.1: Jane Street is a high-frequency trading firm.
  Fact 1.2: Bankman-Fried was a former billionaire.
  Fact 1.3: Bankman-Fried got his start in finance at Jane Street.
  Fact 1.4: Bankman-Fried got his start in finance after MIT.
  Fact 1.5: After leaving Jane Street, Bankman-Fried persuaded Yedidia to join him as a trader at Alameda.
  Fact 1.6: Bankman-Fried later persuaded Yedidia to join him as a developer at FTX.
Sentence 2: The prosecution painted Bankman-Fried as someone who knowingly committed fraud to achieve great wealth, power and influence, while the defense countered that the FTX founder acted in good faith, never meant to commit fraud or steal and basically got in over his head.
  Fact 2.0: The prosecution portrayed Bankman-Fried as someone who knowingly committed fraud to achieve great wealth, power and influence.
  Fact 2.1: The defense countered that the FTX founder, Bankman-Fried, acted in good faith.
  Fact 2.2: The defense said Bankman-Fried never meant to commit fraud or steal.
  Fact 2.3: The defense said Bankman-Fried basically got in over his head.

Retrieved items:
<item 1>
The FTX trial is bigger than Sam Bankman-Fried (The Verge, 2023-09-28T12:00:00+00:00)
First and foremost, they’re trying to prove their client is not guilty. But just in case they don’t get the outcome they want, they’re also laying the
groundwork for appeals and sentencing arguments. Any piece of evidence they want to introduce for those two purposes has to come into play in the trial.

Already, Bankman-Fried’s lawyers are marking arguments that his constitutional rights are being violated by his pretrial incarceration because, among other
things, his internet connection wasn’t good enough for him to plan his defense, notes LaVigne. That could be grounds for an appeal.

“Is he going to throw the entire industry under the bus?”

Bankman-Fried’s behavior after the fall of FTX suggests he’s something of a wild card. He may suggest he was acting on the advice of his lawyers. But he may
also introduce other evidence that could be troublesome — implying, for instance, that he was engaged in standard industry behavior or that everything that
happened was Binance’s fault.
</item 1>

<item 2>
Is Sam Bankman-Fried’s defense even trying to win? (The Verge, 2023-10-04T23:02:14+00:00)
He noted that Julliard was a licensed commodities broker, who was trading in crypto because he didn’t have to disclose it; that Julliard knew that crypto was
new and risky, and that Julliard didn’t review the terms of service agreement he’d assented to when making his FTX account.

Well, sure, but so what?

The next witness called was Bankman-Fried’s former college (and FTX) roommate, Adam Yedidia, about whom I expect I will have much more to say tomorrow.
</item 2>

<item 3>
Sex, Signal messages, and sabotaging FTX's code: SBF execs and Bahamas roommates tell all in court (Cnbc | World Business News Leader,
2023-10-06T20:17:00+00:00)
Bankman-Fried faces seven federal charges, including wire fraud, securities fraud and money laundering, that could put him in prison for the rest of his life.
Thus far, Bankman-Fried, 31, has remained mostly quiet in court intently listening to witnesses and at times writing notes to his attorneys. But as Wang
testified against him, Bankman-Fried looked visibly upset, shifting his gaze from his former friend to the ground, and at one point putting his head in his
hands.

Sam Bankman-Fried listens as Assistant U.S. Attorney Nicolas Roos questions Gary Wang during Bankman-Fried's fraud trial over the collapse of FTX, the bankrupt
cryptocurrency exchange, at Federal Court in New York City, U.S., October 6, 2023, in this courtroom sketch. Jane Rosenberg | Reuters

Wang, 30, was technology chief for FTX, which spiraled into bankruptcy in November.
</item 3>

<item 4>
Is Sam Bankman-Fried’s defense even trying to win? (The Verge, 2023-10-04T23:02:14+00:00)
“All of that was built on lies,” Rehn said.

In his opening statement, Rehn dodged explaining cryptocurrency to the jury. Instead, he punched hard on Bankman-Fried lying and stealing.

Bankman-Fried sat almost motionless, occasionally glancing at Rehn, as the prosecutor told the jury that Bankman-Fried sold stock in FTX and borrowed millions
from lenders by lying.

The story Rehn told is familiar to anyone following the news. In May and June of 2022, Alameda Research — the crypto trading company ostensibly helmed by
Caroline Ellison — didn’t have enough to pay its bills, so it pulled customer money to repay loans. By September, the hole in the FTX balance sheet was so big
that customers could never be repaid.

FTX “didn’t have a chief risk officer, which became an issue when the storm hit.”

When CoinDesk published its article in November 2022, people realized FTX was a house of cards, Rehn said. Meanwhile, Bankman-Fried tweeted.
</item 4>

<item 5>
SBF’s trial starts soon, but how did he — and FTX — get here? (TechCrunch, 2023-10-01T14:00:29+00:00)
U.S. Attorney Damian Williams called Bankman-Fried’s alleged crimes “one of the biggest financial frauds in American history,” in a press conference.

The aftermath

FTX co-founder and former CTO Gary Wang, and Alameda Research’s former CEO, Caroline Ellison, both pleaded guilty in December 2022 to federal criminal charges
in relation to the FTX collapse. They’re also facing civil penalties from the U.S. Securities and Exchange Commission (SEC) and Commodity Futures Trading
Commission (CFTC) alongside the criminal charges. Wang and Ellison plan to cooperate with prosecutors and will be major witnesses in the trial, given their
close ties to Bankman-Fried, FTX and Alameda.

In January, Bankman-Fried pleaded not guilty to all counts, which include wire fraud, conspiracy to commit money laundering and conspiracy to misuse customer
funds. He could face up to 115 years in jail if convicted on all charges.
</item 5>

<item 6>
In the end, the FTX trial was about the friends screwed along the way (The Verge, 2023-10-26T12:00:00+00:00)
Sure, Sam Bankman-Fried’s trial has revealed that FTX was basically a seething mass of scams. But it has also revealed how Bankman-Fried blew up the lives of
the people who are closest to him.

Of course everyone who had money trapped on the FTX exchange when it collapsed — as the result of Bankman-Fried’s trading firm Alameda Research purloining its
customers’ funds — had their lives altered. Even if they are made whole by the bankruptcy proceedings, that’s still money that they should have had access to at
any time they wanted it. That may mean houses that weren’t purchased, medical procedures that were left undone, and life savings lost.

There is a simultaneous drama of ruined lives playing out in the courtroom

Bankman-Fried is charged with seven counts: two counts of wire fraud and five conspiracy charges. As we near the end of the prosecution’s case, I have been
struck by how bizarre it all is.
</item 6>

<item 7>
Is Sam Bankman-Fried’s defense even trying to win? (The Verge, 2023-10-04T23:02:14+00:00)
When he was evaluating the exchange, he thought the sheer volume of users was important, too — at the time, FTX was among the top three biggest exchanges. Plus,
major VC firms had invested, and “they don’t commit hundreds of millions without doing due diligence, checking the books, the accountancy of the firm, going
through several compliance process[es], so that was a vote of confidence for me,” Juilliard said. (Evidently he had not paid attention to the Elizabeth Holmes
trial.)

He also noted FTX’s glossy ads — featuring Gisele Bündchen, for instance — suggested a very high budget. It wouldn’t make sense to spend that much money unless
FTX had very strong financials, Juilliard figured. He opened an account, transferred in both regular money and cryptocurrency, and used the exchange to execute
his plan: buying Bitcoin to sell back in five to ten years at higher prices.
</item 7>

<item 8>
Is Sam Bankman-Fried a bad ‘man’ or a good ‘boy’? Lawyers swap opening statements before first witnesses take the stand (Fortune, 2023-10-04T23:32:53+00:00)
How much? Almost $100,000, he said. And was he ever able to? “Never,” he told prosecutors.

Shortly afterward, as the trial neared late afternoon, the government called Adam Yedidia to the stand. A fast-talking graduate of MIT, he and Bankman-Fried
were close friends in college, he said. And after Bankman-Fried left Jane Street, the high-frequency trading firm where the former billionaire got his start in
finance after MIT, he persuaded Yedidia to join him as a trader at Alameda and then as a developer at FTX.

When Yedidia first took the stand, Danielle Sassoon, one of the lead prosecutors, said that the college friend of Bankman-Fried had legal immunity during his
testimony. Why did he make such a deal with the government, she asked.

“I was concerned that I had unwittingly written code that contributed to a crime,” he said.
</item 8>

<item 9>
The jury finally hears from Sam Bankman-Fried (The Verge, 2023-10-28T00:12:41+00:00)
Bankman-Fried said he worked 12 to 22 hours a day, and took one day off every couple of months. Because FTX had grown so much, he could no longer run both
companies, he said. Bankman-Fried handed the company off to Caroline Ellison and Sam Trabucco, who immediately after being named co-CEO promptly drifted away to
early retirement. (Quiet quitting king!) Bankman-Fried did remain involved in hedging and risk at Alameda, though.

About that $1 billion of revenue in 2021: Bankman-Fried definitely did not know that Singh, his employee, had backdated interest payments to get FTX “over the
line” to $1 billion. See, he’d just asked his employees to check and see if there was any source of funds that was missing to get to $1 billion. This testimony
was especially rambling.

Oh, also that MobileCoin loss? The one Wang said Alameda took to keep off FTX’s balance sheet?
</item 9>

<item 10>
How is it still getting worse for Sam Bankman-Fried? (The Verge, 2023-10-12T23:08:14+00:00)
I have been waiting for a juicy cross-examination, as I live for chaos and drama. I am beginning to think I am not going to get one.

Ellison had given, in her direct testimony, fairly damning evidence tying FTX CEO Sam Bankman-Fried to the conspiracy to take FTX customer funds. There were
fake balance sheets, one of which was sent to crypto lender Genesis. After a Genesis representative received the balance sheet, he texted Ellison to tell her
he’d spoken to Bankman-Fried — strongly suggesting that Bankman-Fried was aware of the contents of the fake balance sheet. Not great!

Sam Bankman-Fried, the founder of failed cryptocurrency exchange FTX, is on trial for seven counts of wire fraud and conspiracy. FTX was a fraud “from the
start,” the Securities and Exchange Commission alleges — with a “multi-billion-dollar deficiency caused by his own misappropriation of customer funds.” Follow
along for all the latest news and regular updates from the trial.
</item 10>

<item 11>
SBF’s trial starts soon, but how did he — and FTX — get here? (TechCrunch, 2023-10-01T14:00:29+00:00)
He could face up to 115 years in jail if convicted on all charges.

The crypto industry as a whole suffered from FTX’s collapse, which was the first of many. BlockFi filed for Chapter 11 in November 2022, as did Genesis Global
Trading in January.

Where we are today

Bankman-Fried will be represented by Cohen & Gresser, and Mark Cohen, a high-profile defense attorney and former federal prosecutor, will be the lead attorney.
If that name sounds familiar, it might be because he also represented Ghislaine Maxwell in her sex trafficking trial related to Jeffrey Epstein. He requested an
early release for Bankman-Fried but was denied.

With the trial beginning, we’re witnessing how FTX’s story ends. But what’s on our mind is what happens to the investors and creditors affected by the collapse?
And what happens to the billions in crypto assets tied up in legal proceedings?
</item 11>

<item 12>
Sam Bankman-Fried was a terrible boyfriend (The Verge, 2023-10-10T23:50:21+00:00)
FTX was a fraud “from the start,” the Securities and Exchange Commission alleges — with a “multi-billion-dollar deficiency caused by his own misappropriation of
customer funds.” Follow along for all the latest news and regular updates from the trial.

The day started off promisingly for the defense as it cross-examined Gary Wang, the chief technology officer of FTX and co-owner of both FTX and Alameda.
Christian Everdell, one of Bankman-Fried’s defense attorneys, couldn’t undo the damage of last week’s code review. But he managed to shake the rust off long
enough to make Wang sound less reliable, drowning the jury in confusing technicalities.

Last week, Wang testified that Alameda got access to a special credit line and an option to take its balance into the negative without triggering liquidation —
something he alleged other accounts at FTX didn’t get. Everdell tried to undermine this claim by talking about the spot margin program, which let users lend
each other assets for margin trading.
</item 12>

<item 13>
Sam Bankman-Fried didn’t ask where the $8 billion went (The Verge, 2023-10-31T19:52:07+00:00)
Oh, but he was testifying to Congress about keeping customer funds safe.

Sam Bankman-Fried, the founder of failed cryptocurrency exchange FTX, is on trial for seven counts of wire fraud and conspiracy. FTX was a fraud “from the
start,” the Securities and Exchange Commission alleges — with a “multi-billion-dollar deficiency caused by his own misappropriation of customer funds.” Follow
along for all the latest news and regular updates from the trial.

Who was making decisions to spend $8 billion of customer funds? Bankman-Fried couldn’t recall knowing anything about it. Were there rules or requirements for
how money borrowed from FTX would be returned? Were there rules for risk management? “I was concerned with overall risk management,” Bankman-Fried said.

But it was the testimony about June 2022 that resonated the most to me. Didn’t Bankman-Fried ask what “fiat@ftx” was? He did.
</item 13>

<item 14>
The FTX trial is bigger than Sam Bankman-Fried (The Verge, 2023-09-28T12:00:00+00:00)
Here are the hits:

Bankman-Fried gave interviews freely — and quickly rose to public prominence in the industry. Though FTX hadn’t been in the business as long as competing
exchanges such as Coinbase, Kraken, or Gemini, Bankman-Fried positioned himself as an important, boyish face for crypto. (At one point, Bankman-Fried told a
colleague at FTX that “I honestly think it’s negative EV [this may mean “expected value,” as in poker] for me to cut my hair. I think it’s important for people
to think I look crazy.”)

Because he was so successful at this kind of public relations, his fall from grace was another mark against an industry that was already roiled by bankruptcies
and scandals. Some additional trouble for the crypto industry is likely to come from one crucial element of the fraud trial — the part where the government must
prove intent.

The first part of proving the government’s case is pretty simple and a little boring: prosecutors must show that certain transactions took place.
</item 14>

<item 15>
The jury finally hears from Sam Bankman-Fried (The Verge, 2023-10-28T00:12:41+00:00)
It is honestly kind of incredible to watch a man torpedo his own credibility on direct testimony. We’re not even at the cross yet, and the judge has already
instructed him to answer the question he’s being asked by his own lawyer.

The jury is watching all of this intently.

The main thing that’s been clear so far from Bankman-Fried’s testimony is that the man really loves the sound of his own voice. So far, the count of “Objection,
narrative!” to Bankman-Fried’s answers, followed by “Sustained” is at three.

Also, sometimes when Bankman-Fried says “we,” he only means himself.

Sam Bankman-Fried, the founder of failed cryptocurrency exchange FTX, has been found guilty on seven counts including charges of wire fraud. FTX was a fraud
“from the start,” the Securities and Exchange Commission alleged — with a “multi-billion-dollar deficiency caused by his own misappropriation of customer
funds.” Follow along for all the latest news and regular updates from the trial.
</item 15>

<item 16>
How is it still getting worse for Sam Bankman-Fried? (The Verge, 2023-10-12T23:08:14+00:00)
The recording was later sent to Drappi, who sent it to the government. Joseph Bankman, the defendant’s father and a senior advisor to FTX’s philanthropic arm,
wasn’t in the room for the recording. Barbara Fried looked unhappy, rubbing her left temple as though she had a tremendous headache.

And in a small miracle of pacing, the government got Zac Prince, the founder and former CEO of crypto lender BlockFi, on the stand just long enough to blame
FTX’s bankruptcy for BlockFi’s subsequent bankruptcy before we broke for the day.

I’ve been asked by some people why the reporting coming out of the trial seems so skewed toward the prosecution. There’s an answer for this: the prosecution has
put together a strong, comprehensive indictment of Bankman-Fried’s behavior at FTX and Alameda. The defense, so far, has managed to do almost nothing in
response.
</item 16>

<item 17>
Sex, Signal messages, and sabotaging FTX's code: SBF execs and Bahamas roommates tell all in court (Cnbc | World Business News Leader,
2023-10-06T20:17:00+00:00)
Sam Bankman-Fried, the founder of bankrupt cryptocurrency exchange FTX, is seen during a hearing as a U.S judge revoked his bail, at a courthouse in New York,
U.S., August 11, 2023 in this courtroom sketch. Jane Rosenberg | Reuters

$65 billion line of credit

For months, Bankman-Fried has known that Wang and Ellison, who were integral members of his personal and professional inner circles, had turned on him. Both
pleaded guilty in December and have since been cooperating with the U.S. attorney's office in Manhattan. Wang's testimony, which stretched into Friday, was
given under a cooperation agreement with the government. Ellison is expected to take the stand under a similar arrangement.

U.S.
</item 17>

<item 18>
The FTX trial is bigger than Sam Bankman-Fried (The Verge, 2023-09-28T12:00:00+00:00)
In the indictment, the prosecutors allege that Bankman-Fried lied to FTX investors. Some of those investors may be called on to testify, which is not the kind
of thing VC big shots generally enjoy — and which might create collateral damage for the industry.

If, for instance, Sequoia Capital did due diligence around its investment in FTX, whatever Bankman-Fried told its partners could be important. Was FTX already
sending Alameda customer funds at that point? Did Bankman-Fried know about it? And did he tell VCs at the time? (If he told them it wasn’t happening, that would
establish intent.) Binance was also an early investor in FTX, and former executives may be called on to testify as well, says Hermine Wong, the former head of
policy at Coinbase and a former SEC regulator.

Bankman-Fried’s defense can also introduce risks for people who dealt with him. Defense lawyers have several simultaneous objectives. First and foremost,
they’re trying to prove their client is not guilty.
</item 18>

<item 19>
SBF’s trial starts soon, but how did he — and FTX — get here? (TechCrunch, 2023-10-01T14:00:29+00:00)
In a published DM exchange with a Vox reporter, he said he regretted filing for bankruptcy and thought that “regulators make everything worse.”

SBF arrested

Bankman-Fried was arrested in December 2022 in the Bahamas, where FTX was based. He was then extradited to the U.S. to face a number of criminal charges. He was
released on a $250 million bail bond, and he remained under house arrest at his parents’ home in Palo Alto. This was revoked in August after he was accused of
intimidating Alameda’s former CEO, Caroline Ellison, by leaking her private diary.

Ray represented the company during a House Financial Services Committee hearing regarding FTX. When asked whether the firm had significant risk management
systems, Ray said at the time that “there were virtually no internal controls and no separateness whatsoever” and added that he did not “trust a single piece of
paper” in the exchange’s organization. U.S.
</item 19>

<item 20>
Is Sam Bankman-Fried a bad ‘man’ or a good ‘boy’? Lawyers swap opening statements before first witnesses take the stand (Fortune, 2023-10-04T23:32:53+00:00)
“One year ago, it looked like the defendant was on the top of the world,” began Thane Rehn, a prosecutor for the government, in his opening statement. The
former CEO of FTX oversaw a supposedly thriving crypto exchange, jetted between international locales, and hobnobbed with celebrities like Tom Brady and Larry
David. He repeatedly emphasized to customers that their money was safe and secure.

But “all of that, all of it, was built on lies,” Rehn declared to the jury. “Behind the curtains, he was not what he appeared to be.” What followed was a
roughly 30-minute story that repeatedly emphasized how Bankman-Fried allegedly stole customer funds to facilitate his jet-setting lifestyle, donate millions to
political candidates, and finance risky bets.

The key to his alleged scheme? Alameda Research, a crypto hedge fund he also owned, argued Rehn.
</item 20>

<item 21>
The jury finally hears from Sam Bankman-Fried (The Verge, 2023-10-28T00:12:41+00:00)
There was a minimum of word salad today. I don’t know if he was more relaxed, or he’d just been more rehearsed, but I will certainly be watching to see if he
suddenly becomes much less coherent when Sassoon gets him for the cross.

Here is the story of FTX, from his point of view.

Bankman-Fried, who informed us he’s “somewhat introverted, naturally,” gave us a rather prolonged tour of his pre-Alameda Research life, which I will skip. In
2017, during a crypto bull run, he started his cryptocurrency trading firm. He knew “basically nothing” about cryptocurrency at the time, he explained, but he
wanted to do arbitrage on it anyway.

Alameda Research was named for Alameda County in California, which was where its first office was set up. As for its name, here’s what Bankman-Fried said on the
stand:

Effectively, we wanted to be under the radar at that point in time.
</item 21>

<item 22>
Is Sam Bankman-Fried a bad ‘man’ or a good ‘boy’? Lawyers swap opening statements before first witnesses take the stand (Fortune, 2023-10-04T23:32:53+00:00)
FTX was no Ponzi scheme. It was a “very innovative, successful company.” And the business practices between the two were reasonable, he argued, claiming that
Alameda acted legally as an FTX customer, payment processor, and market maker, or financial entity that acts as a trading partner for customers looking to buy
and sell cryptocurrencies.

In an analogy he employed throughout his opening statement, he said that “working at a startup is like building a plane as you’re flying it” and that businesses
sometimes fail. In fact, he specifically pointed the finger at Ellison, the former CEO of Alameda, who, he said, did not adequately protect her hedge fund from
the inherent risk of the crypto markets.

When the walls came closing in and the aforementioned plane approached the “eye of the storm,” Bankman-Fried did not act like someone who was guilty. Rather, he
was willing to give up his personal wealth to make customers whole, Cohen argued.
</item 22>

<item 23>
How is it still getting worse for Sam Bankman-Fried? (The Verge, 2023-10-12T23:08:14+00:00)
I was ready to hear this pursued further — but then it, too, was dropped.

I think the defense was also trying to suggest that the government had coerced Ellison’s testimony, by suggesting that she had pleaded guilty to a charge of
defrauding investors that she couldn’t have been involved in. After all, she didn’t prepare materials for them. Unfortunately, she did say that she had
conversations with investors as part of their due diligence — and, of course, Alameda was taking on losses from FTX to keep FTX’s balance sheet pristine. This
line of questioning felt like a waste of time.

There were rather a lot of sidebars during the cross-examination, to the point that when one occurred, several jurors looked entertained. There were a few
yesterday, too, including one in which the prosecution complained that Bankman-Fried was visibly scoffing at Ellison’s answers, according to the court
transcript.
</item 23>

<item 24>
Is Sam Bankman-Fried a bad ‘man’ or a good ‘boy’? Lawyers swap opening statements before first witnesses take the stand (Fortune, 2023-10-04T23:32:53+00:00)
Rather, he was willing to give up his personal wealth to make customers whole, Cohen argued.

“In the end, Sam started and built two billion-dollar businesses,” he concluded. “He didn’t steal any money.”

A Frenchman who lives in London testifies in New York

After lawyers from both sides depicted two very different Bankman-Frieds, the prosecution called its first two witnesses to the stand—and they weren’t
blockbuster names or former lieutenants-turned-government-cooperators, like Ellison.

The first was a victim: Marc-Antoine Julliard, a Paris-born cocoa trader who lives in London. In 2021, Julliard, who had coiffed hair and spoke with a strong
French accent, decided to invest in crypto and landed on FTX as his exchange of choice, where he traded cryptocurrencies like Bitcoin and Dogecoin.

On Nov. 8, in the crypto exchange’s final few days, he tried to pull out his cash and crypto.
</item 24>

<item 25>
Is Sam Bankman-Fried’s defense even trying to win? (The Verge, 2023-10-04T23:02:14+00:00)
“Sam acted in good faith and took reasonable business measures” is a pretty hard pill to swallow with that in mind.

Man, it’s no good when your defense lawyer has just made you sound worse than the prosecution already did. And while Cohen tried to make the common white-collar
defense argument that Bankman-Fried, as CEO, was simply too busy to oversee what everyone did every day, he just made me more suspicious. That’s why you hire a
risk officer and delegate! That’s the whole point! I could barely even hear Cohen blaming Caroline Ellison and Changpeng “CZ” Zhao for the debacle over the “no
risk officer” ringing in my ears.

Following the defense’s opening statements, things got still worse for Bankman-Fried. The prosecution called its first witness, Marc-Antoine Julliard, whose
money got stuck on FTX. Juilliard, who was born in Paris and lives in London, testified that he trusted FTX because Bankman-Fried came across as a leading
figure of the industry.
</item 25>

<item 26>
Sex, Signal messages, and sabotaging FTX's code: SBF execs and Bahamas roommates tell all in court (Cnbc | World Business News Leader,
2023-10-06T20:17:00+00:00)
Witness Adam Yedidia answers questions during Sam Bankman-Fried fraud trial over the collapse of FTX, the bankrupt cryptocurrency exchange, at Federal Court in
New York City, October 5, 2023, in this courtroom sketch. Jane Rosenberg | Reuters

Two of Sam Bankman-Fried's former friends from MIT, who also worked at crypto exchange FTX while living with the company's founder in the Bahamas, took the
stand in a Manhattan courtroom this week to testify against their former classmate, confidant, and boss — a man who allegedly ran a crypto empire that defrauded
thousands of customers out of billions of dollars. Gary Wang, the lesser-known co-founder of FTX, was asked by Assistant U.S. Attorney Nicolas Roos on Thursday,
"Did you commit financial crimes while working at FTX?" "Yes," responded Wang.
</item 26>

<item 27>
The FTX trial is bigger than Sam Bankman-Fried (The Verge, 2023-09-28T12:00:00+00:00)
If it seems like his peer group supported him, or worse, fawned over him, that’s a real problem for the industry’s reputation.”

After FTX declared bankruptcy, Bankman-Fried went on an extensive media tour, doing interviews with The New York Times’ Andrew Ross Sorkin and Good Morning
America. Even under house arrest, he did interviews, including with The New Yorker. Bankman-Fried seemed to think that if he just explained things, everyone
would understand that what he did was a mistake, not a crime.
</item 27>

<item 28>
SBF Trial: The latest updates from the FTX collapse’s courtroom drama (TechCrunch, 2023-10-06T17:22:02+00:00)
SBF trial: Everything to know from the FTX courtroom ahead of his testimony Sam Bankman-Fried is in court for alleged fraud and money laundering while leading
crypto exchange FTX and sister company Alameda. Here’s our latest coverage as the case nears its end.

The trial of Bankman-Fried, the former CEO and co-founder of collapsed crypto exchange FTX, is entering its next chapter as proceedings resume Thursday. As
confirmed during a teleconference Wednesday, SBF will be taking the stand, along with a handful of other witnesses as the onus shifts toward the defense after
the prosecution brought its case to its conclusion.

As former federal prosecutor Josh Naftalis told TechCrunch, SBF taking to the stand in his own trial is a “Hail Mary” ahead of a verdict and potential
sentencing for seven counts tied to fraud and money laundering.

“Once it goes to cross[-examination], he doesn’t get to say, ‘I’m done,’” Naftalis said.
</item 28>

<item 29>
SBF Trial: The latest updates from the FTX collapse’s courtroom drama (TechCrunch, 2023-10-06T17:22:02+00:00)
The prosecution painted a picture of an empire “built on lies” and made a point to show actual pictures of SBF alongside rich and powerful figures to illustrate
the peak of his power and influence prior to FTX’s implosion.

“This man stole billions of dollars from thousands of people,” prosecutor Thane Rehn told the court. “He defrauded sophisticated investors and lenders, and he
emptied the accounts of ordinary customers, too. He bought himself wealth, power and influence.”

The defense, on the other hand, depicted SBF and his colleagues as in over their heads, without ill-will as FTX collapsed around them.

“Sam and his colleagues were building the plane as they were flying it,” defense attorney Mark Cohen said. “They had to figure out how to navigate a world where
they were running FTX, building out its systems, dealing with hacking threats, managing the credit risk of their customers, managing hundreds of employees, all
while building up their actual exchange.”

Read a recap of the prosecution and defense’s full arguments here.
</item 29>

<item 30>
Is Sam Bankman-Fried a bad ‘man’ or a good ‘boy’? Lawyers swap opening statements before first witnesses take the stand (Fortune, 2023-10-04T23:32:53+00:00)
“I was concerned that I had unwittingly written code that contributed to a crime,” he said.

Soon, however, the clock neared 4:30 p.m., and court adjourned for the day. Yedidia will continue his testimony on Thursday, followed by Matt Huang, a former
partner at the high-powered venture capital firm Sequoia Capital, and then Gary Wang, a key Bankman-Fried lieutenant and one of the government’s star witnesses.
</item 30>

<item 31>
In the end, the FTX trial was about the friends screwed along the way (The Verge, 2023-10-26T12:00:00+00:00)
The general sense I got, from the testimony of Bankman-Fried’s former friends, was that things just kept escalating

During Yedidia’s testimony, we were shown a photograph in court: former President Bill Clinton and former Prime Minister Tony Blair onstage in suits next to
Bankman-Fried, in a rumpled T-shirt and cargo shorts. Sure, dressing down is a staple of Silicon Valley startup culture — and a sign of its swagger. The defense
counsel tried to elicit testimony from Yedidia that the cargo shorts were evidence of Bankman-Fried’s humility, but the photograph seemed to signify just the
opposite: that the man in cargo shorts is so important, he doesn’t need to dress up, even to meet former world leaders.

Now, in court, Bankman-Fried is wearing a suit.

In 2019, just months after FTX was founded, Wang oversaw code called “allow_negative” that let Alameda Research avoid being liquidated by FTX’s risk management.
It was turned on in July.
</item 31>

<item 32>
The FTX trial is bigger than Sam Bankman-Fried (The Verge, 2023-09-28T12:00:00+00:00)
The trial of Sam Bankman-Fried is likely to be more consequential than just whether the man himself is found guilty. Depending on what evidence is introduced
during the trial, it could be rough for the entire crypto industry.

“How much damage can this trial do to the already beaten-down reputation of the industry at this point?” asks Yesha Yadav, a law professor at Vanderbilt
University. “This trial is going to be an excruciating moment for the industry because no one knows what kind of evidence might come out.”

Bankman-Fried, the founder of FTX and Alameda Research, is facing seven counts of criminal charges: two counts of wire fraud, and five counts of conspiracy
charges. FTX was a failed cryptocurrency exchange founded in 2019. According to a now-deleted profile from FTX investors Sequoia Capital, FTX was founded
because of Bankman-Fried’s frustration with other exchanges when he was running Alameda Research, his crypto trading firm.
</item 32>

<item 33>
The FTX trial is bigger than Sam Bankman-Fried (The Verge, 2023-09-28T12:00:00+00:00)
That may be risky, but we already know that Bankman-Fried loves risk.

“Is he going to throw the entire industry under the bus?” Wong asks. “An idea like, ‘Everyone was doing this, it’s not fair I’m the only one who was charged?’”
That may not fly in a court of law, but it could absolutely damage public perception of crypto at large.

For instance, just before he was arrested, Bankman-Fried brought forward messages from a crypto group chat in testimony he planned to give before Congress. In
that testimony, he primarily blames Binance and his lawyers at the firm Sullivan & Cromwell.

It’s possible other messages from group chats may be brought into evidence during the trial. That can potentially be embarrassing for the whole industry, Wong
says.

“You can imagine some of these founders, CEOs, people of that echelon talk to each other somewhat informally about what’s going on,” she says.
</item 33>

<item 34>
Is Sam Bankman-Fried a bad ‘man’ or a good ‘boy’? Lawyers swap opening statements before first witnesses take the stand (Fortune, 2023-10-04T23:32:53+00:00)
Who is Sam Bankman-Fried, the former CEO of crypto exchange FTX? Is he a liar and a fraud? Or did he act in good faith, and like many a startup founder, fly too
close to the sun?

That answer ultimately lies with the jury, which was selected on Wednesday morning before lawyers for the government and then Bankman-Fried swapped two very
different stories of the former crypto mogul’s sudden rise and almost instantaneous fall.

Here’s what happened on the second day of the trial, which featured pointed allegations, a friend from MIT, and an audience replete with big names, including
Bankman-Fried’s professorial parents and Damian Williams, U.S. attorney for the Southern District of New York.

A conscious criminal…

The prosecution’s account of the alleged crimes by Bankman-Fried, who spent most of his day in court staring into a laptop while seated between his two
attorneys, featured a study in contrasts.
</item 34>

<item 35>
The FTX trial is bigger than Sam Bankman-Fried (The Verge, 2023-09-28T12:00:00+00:00)
According to the SEC, FTX was a fraud “from the start,” diverting customers’ funds to Alameda.

Sam Bankman-Fried, the founder of failed cryptocurrency exchange FTX, is on trial for seven counts of wire fraud and conspiracy. FTX was a fraud “from the
start,” the Securities and Exchange Commission alleges — with a “multi-billion-dollar deficiency caused by his own misappropriation of customer funds.” Follow
along for all the latest news and regular updates from the trial.

“Really just old-fashioned embezzlement.”

What’s left of FTX is now being led by John J. Ray III — you may remember him as the guy who cleaned up Enron and who also said FTX is worse than Enron. Just
before the trial began, the FTX lawyers filed suit against Bankman-Fried’s parents, saying they should give back millions of dollars they got from their kid.
Ray has also referred to Bankman-Fried’s conduct as “really just old-fashioned embezzlement.”
</item 35>

<item 36>
Is Sam Bankman-Fried’s defense even trying to win? (The Verge, 2023-10-04T23:02:14+00:00)
Meanwhile, Bankman-Fried tweeted. “FTX is fine. Assets are fine” and “We don’t invest customer assets even in treasuries.”

Pointing at Bankman-Fried, Rehn said, “This man stole billions of dollars from thousands of people.”

So how was the defense going to follow it up? I was very curious, having learned yesterday that Bankman-Fried had never been offered a plea deal since he and
his attorneys had told the government they wouldn’t negotiate. Surely there would be some manner of evidence, some something, that would have made him so
confident.

There was, instead, a metaphor.

Defense attorney Mark Cohen, with the energy of a patient father telling his obnoxious children a bedtime story, assured us that working at a startup was like
building a plane while flying it, and that FTX the plane had flown right into the perfect storm: the crypto crash.
</item 36>

<item 37>
How is it still getting worse for Sam Bankman-Fried? (The Verge, 2023-10-12T23:08:14+00:00)
The testimony set up an argument for the prosecution to bring in those tapes. The judge ruled in favor of the jury hearing the recordings and we briefly
recessed.

That was when Bankman-Fried’s mother approached the defense lawyers.

The late afternoon was short and snappy. Christian Drappi, a former Alameda software engineer who looked like a handsome funeral director in a black suit and
tie, testified briefly to set up the tape. When Changpeng Zhao, the CEO of Binance, announced on Twitter that he intended to acquire FTX, Ellison confessed the
theft of customer funds to him and a few other employees, Drappi said. The all-hands took place the following day, and was secretly recorded by an employee
who’d joined Alameda three days before.

In the recording, Ellison did indeed confess to stealing customer funds with Bankman-Fried’s approval. Drappi said he resigned less than 24 hours after the
meeting. The recording was later sent to Drappi, who sent it to the government.
</item 37>

<item 38>
Is Sam Bankman-Fried’s defense even trying to win? (The Verge, 2023-10-04T23:02:14+00:00)
I have never seen Sam Bankman-Fried so still as he was during the prosecution’s opening statement. The characteristic leg-jiggling was absent. He barely moved
as the prosecutor listed the evidence against him: internal company files, what customers were told, the testimony of his co-conspirators and his own words.

His hair was shorn, the result of a haircut from a fellow prisoner, the Wall Street Journal reported. He wore a suit bought at a discount at Macy’s, per the
Journal; it hung on him. He appeared to have lost some weight.

“All of that was built on lies.”

Bankman-Fried, at this time last year, had a luxury lifestyle as the CEO of crypto exchange FTX, said the assistant US attorney, Thane Rehn, in the cadence of a
high schooler delivering his lines in a student play. Bankman-Fried hung out with Tom Brady. He was on magazine covers, lived in a $30 million penthouse, and
spent time with world politicians.
</item 38>

<item 39>
The FTX trial is bigger than Sam Bankman-Fried (The Verge, 2023-09-28T12:00:00+00:00)
“We had him going out directly to the internet,” LaVigne says. If he wrote things on Twitter or said things in interviews that weren’t true, that’s more fodder
for the government’s case. “They can point to that and say, ‘This is what he said, this is what actually happened.’”

Other evidence may include Signal messages and testimony from co-conspirators who plead guilty to their own charges. Alameda Research CEO Caroline Ellison, who
was also sometimes Bankman-Fried’s girlfriend, may play an important role — Bankman-Fried leaked her diaries to The New York Times and was consequently jailed
for witness tampering. The government has indicated FTX co-founder Gary Wang and engineering head Nishad Singh will also be among the witnesses called to give
testimony about Bankman-Fried.

Bankman-Fried’s defense can also introduce risks for people who dealt with him

There may be testimony from lenders, venture capitalists, and customers in order to establish the basis for some charges.
</item 39>

<item 40>
Is Sam Bankman-Fried a bad ‘man’ or a good ‘boy’? Lawyers swap opening statements before first witnesses take the stand (Fortune, 2023-10-04T23:32:53+00:00)
“You will see the full picture.”

…or a well-meaning founder?

But Bankman-Fried, whose cheekbones were more prominent after spending about seven weeks in a Brooklyn prison, was no liar, according to Mark Cohen, one of his
lawyers. “Sam didn’t defraud anyone,” he said early on in his opening statement.

What the jury will see is a nerdy startup founder who acted in “good faith,” not the prosecution’s “cartoon of a villain.” (Cohen repeatedly harped on
Bankman-Fried’s allegedly good-faith actions throughout his address to the jury.)

Alameda was not subterranean or shady. It was a successful hedge fund, he said. FTX was no Ponzi scheme. It was a “very innovative, successful company.”
</item 40>

<item 41>
The FTX trial is bigger than Sam Bankman-Fried (The Verge, 2023-09-28T12:00:00+00:00)
Bankman-Fried is accused of using “billions of dollars in misappropriated FTX customer deposits” to help buy more than $200 million of real estate for himself,
make billions of dollars of investments for his own interest, and repay Alameda’s lenders, according to the indictment, Additionally, Bankman-Fried allegedly
used more than $100 million of customer funds to make political contributions; prosecutors can show evidence of those contributions in this trial, even though
they aren’t part of the charges brought. A second trial is scheduled for March 2024, with additional charges.

Before his fall, Bankman-Fried made himself out to be the Good Boy of crypto — the trustworthy face of a sometimes-shady industry. He was also very interested
in publicity, sitting for many interviews both before and after the fall of FTX. The quick rise of FTX as an industry force was at least in part due to
Bankman-Fried’s appetite for attention.
</item 41>

<item 42>
In the end, the FTX trial was about the friends screwed along the way (The Verge, 2023-10-26T12:00:00+00:00)
All the co-conspirators seem oddly younger than their years, as though they got stuck in the “math camp forever” environment Bankman-Fried created for them and
simply failed to grow up.

The prosecution has made a strong case that Bankman-Fried presided over the various forms of fraud that played out at Alameda Research and FTX, orchestrating
some of it and engaging in some of it directly. In the process of making this case, there is a simultaneous drama of ruined lives playing out in the courtroom.

Everyone who loved Bankman-Fried most in the world is now caught up in the case. We have:

Adam Yedidia, Bankman-Fried’s college roommate, who testified with immunity because he was concerned he may have unknowingly worked on code that contributed to
the fraud. “I love you Sam,” he texted at one point. “I’m not going anywhere.” That was before he realized FTX had stolen customer funds — at which point, he
resigned.
</item 42>

<item 43>
Sex, Signal messages, and sabotaging FTX's code: SBF execs and Bahamas roommates tell all in court (Cnbc | World Business News Leader,
2023-10-06T20:17:00+00:00)
"FTX defrauded all its customers," he said.

Matt Huang, co-founder of Paradigm Operations LP, right, arrives at court in New York, US, on Thursday, Oct. 5, 2023. Former FTX Co-Founder Sam Bankman-Fried is
charged with seven counts of fraud and money laundering following the collapse of his cryptocurrency empire last year. Photographer: Yuki Iwamura/Bloomberg via
Getty Images Yuki Iwamura | Bloomberg | Getty Images

Investment to zero

The third witness to take the stand was Matt Huang, co-founder and managing partner of Paradigm, a crypto venture capital firm that invested over $275 million
in FTX. That stake was wiped out. Huang testified about his firm's due diligence on FTX, and he told the court that Bankman-Fried assured him that funds would
be used for FTX and not Alameda.
</item 43>

<item 44>
SBF’s trial starts soon, but how did he — and FTX — get here? (TechCrunch, 2023-10-01T14:00:29+00:00)
SBF’s trial has started, this is how he and FTX got here

The highly anticipated criminal trial for Sam Bankman-Fried, former CEO of bankrupt crypto exchange FTX, started Tuesday to determine whether he’s guilty of
seven counts of fraud and conspiracy. And as one former federal prosecutor put it: “The odds seem to be stacked against him at this point.”

The 31-year-old co-founded FTX in 2019; within a few years the once third-largest crypto exchange’s valuation hit $32 billion at its peak. It’s now trying to
claw back any funds to distribute to creditors.

But how did the once third-largest crypto exchange get here?

Before FTX, Bankman-Fried co-founded crypto-trading firm Alameda Research in 2017. He co-founded FTX in 2019 as a complement to Alameda, to help bring in
revenue and liquidity for the trading arm.
</item 44>

<item 45>
SBF, riding high on FTX, reportedly offered $5B to Trump to cancel President re-run (TechCrunch, 2023-10-02T13:11:07+00:00)
Come Tuesday, our eyes will be trained on the Manhattan Federal Court for the trial of Sam Bankman-Fried, the disgraced cryptocurrency entrepreneur accused of
orchestrating “one of the biggest financial frauds in American history”: the multibillion-dollar collapse of FTX, the crypto exchange he founded and led.

SBF, as he’s known, has pleaded not-guilty to some seven charges of fraud and conspiracy. But, as we wait to hear both his version and the government’s version
of what happened, there will be another tale recounting how things played out: “Going Infinite,” a book by Michael Lewis chronicling the rise and fall of SBF,
is set to debut on the first day of the trial.

The episode has fired up a lot of people, indignant at what they feel was too-easy treatment for someone who many believe brought down the whole crypto house.

60 Minutes got rightly community noted here.
</item 45>

<item 46>
The jury finally hears from Sam Bankman-Fried (The Verge, 2023-10-28T00:12:41+00:00)
Perhaps you are wondering, was the jury also there? Reader, it was.

Bankman-Fried denied he knew about the effectively infinite line of credit Alameda Research received from FTX. This argument was peculiar; essentially my
take-away was that the CEO of a financial company simply didn’t pay attention to finances.

FTX couldn’t get bank accounts right away. Bankman-Fried anticipated it would take a year or two. Rather than wait, he decided to use Alameda as the “payment
provider” for bank transfers. “My understanding at the time was that there were teams managing the process,” he said. “At the time, I wasn’t entirely sure what
was happening.”

Well, sure, understandable! He’s an introvert!

Bankman-Fried definitely did not know that Singh, his employee, had backdated interest payments to get FTX “over the line” to $1 billion

In 2021, FTX was growing to millions of users, with $1 billion of revenue.
</item 46>

<item 47>
Is Sam Bankman-Fried a bad ‘man’ or a good ‘boy’? Lawyers swap opening statements before first witnesses take the stand (Fortune, 2023-10-04T23:32:53+00:00)
Alameda Research, a crypto hedge fund he also owned, argued Rehn. Using Caroline Ellison, his on-again off-again girlfriend and the CEO of Alameda, as a front,
Bankman-Fried had “secret access” to customer money—both cash and crypto—the government claimed.

Moreover, Bankman-Fried allegedly directed employees to conceal the flow of money into FTX’s coffers and forged financial documents distributed to lenders and
investors. “The defendant lied to the world,” Rehn alleged.

And who was this defendant? Not a crypto “boy” genius, as so many in the media (Fortune included) have written, but a “man” who “stole billions of dollars from
thousands of victims,” Rehn said. “You will see the full picture.”

…or a well-meaning founder?
</item 47>

<item 48>
Is Sam Bankman-Fried’s defense even trying to win? (The Verge, 2023-10-04T23:02:14+00:00)
This is to say nothing of the regulatory risks around crypto.

As Cohen droned on about airplanes, I couldn’t stop thinking about the missing risk officer. Bringing it up, I thought, was a tremendous mistake. The
prosecution hadn’t mentioned it. Either Bankman-Fried is stupid — unlikely — or he deliberately didn’t hire a risk officer. Was he worried about what one might
find?

Sure, as Cohen put it, Bankman-Fried was a math nerd who didn’t party. That paints a picture of someone who’s pretty deliberate, particularly since he
immediately left MIT and went to work on Wall Street. If he had been a party-hardy trainwreck, I could see overlooking a risk officer in order to do another
line, or a supermodel, or something else important. Why was the defense bringing this up?

But as Cohen tried to tell me that FTX’s and Alameda’s business relationships were “reasonable under the circumstances,” the lack of risk officer kept elbowing
me in the ribs.
</item 48>

<item 49>
The FTX trial is bigger than Sam Bankman-Fried (The Verge, 2023-09-28T12:00:00+00:00)
Whatever records the Southern District of New York has for the transactions will be shown.

“What conversations happened between him and his co-conspirators that are now cooperating against him?”

The second part is where all the drama is likely to come, says Christopher LaVigne, a litigation partner and co-chair of the cryptocurrency practice at the law
firm Withers. Prosecutors have to connect those transactions to Bankman-Fried, show that he knew what he was doing was wrong, and prove that he lied about it
anyway.

“What was he saying to his parents and his other advisors about this?” LaVigne says. “What conversations happened between him and his co-conspirators that are
now cooperating against him?”

To further establish intent, the government can use Bankman-Fried’s own words. The indictment calls Bankman-Fried’s tweets in November 2022 “false and
misleading.”

“We had him going out directly to the internet,” LaVigne says.
</item 49>

<item 50>
The FTX trial is bigger than Sam Bankman-Fried (The Verge, 2023-09-28T12:00:00+00:00)
We have some inkling of what the government will be arguing thanks to the filings prosecutors are making in this case. They argue that Bankman-Fried lied about
consumer protection and that Bankman-Fried’s statements that FTX was “avoiding or managing conflicts of interest” and that “as a general principle FTX
segregates customer assets from its own assets across our platforms” were lies, in a superseding indictment filed August 14th.

The indictment also says that there were special features in FTX’s code that “permitted Alameda to spend and withdraw unlimited amounts of money from FTX,”
which were created at Bankman-Fried’s direction. That effectively exempted Alameda from the kinds of risk management other customers faced. At the same time,
the government alleges, “Bankman-Fried publicly and repeatedly asserted that Alameda did not have privileged access to FTX.”
</item 50>
=== END ITEM id=28dddb0c80450023 ===
