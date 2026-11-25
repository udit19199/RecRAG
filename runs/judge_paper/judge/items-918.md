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

=== ITEM id=fc668495ce6b66f5 ===
Gold sentences and facts:
Sentence 0: Harbaugh reportedly watched the game from the team hotel, and now the leading piece of drama returns.
  Fact 0.0: Harbaugh reportedly watched the game from the team hotel.
  Fact 0.1: The leading piece of drama returns.
Sentence 1: The Big Ten suspended coach Jim Harbaugh for three games on Friday with the team on a plane en route to Penn State, and although Michigan had hoped to block the suspension, the Wolverines will be without their coach on the sidelines in Happy Valley.
  Fact 1.0: The Big Ten suspended coach Jim Harbaugh for three games.
  Fact 1.1: The Big Ten suspended Jim Harbaugh on Friday.
  Fact 1.2: The team was on a plane en route to Penn State when the suspension happened.
  Fact 1.3: Michigan had hoped to block the suspension.
  Fact 1.4: The Wolverines (Michigan) will be without their coach on the sidelines in Happy Valley.

Retrieved items:
<item 1>
Michigan didn't need Jim Harbaugh against Penn State, but Ohio State is a different game (Sporting News, 2023-11-11T21:32:44+00:00)
Did Michigan really need Jim Harbaugh on Saturday?

Perhaps the result provides the short-term answer. No. 3 Michigan beat No. 10 Penn State 24-15 at Beaver Stadium on Saturday. The Wolverines threw one pass in
the second half – and that didn't even technically count – in a risk-averse game plan under acting coach Sherrone Moore. The defense smothered the Nittany
Lions, and that will lead to more buildup ahead of the showdown with No. 1 Ohio State on Nov. 25.

Then Moore let it all there in the Fox postgame interview with Jenny Taft. Even if the answer is no, Harbaugh was ever-present.

"I want to thank the Lord and I want to thank Coach Harbaugh," Moore said. "I f------ love you man. I love the s--- out of you, man. I did this for you, for
this university, the president, our AD.
</item 1>

<item 2>
Michigan sign-stealing scandal news: Latest updates on possible Jim Harbaugh suspension from NCAA cheating investigation (Sporting News,
2023-11-06T23:58:04+00:00)
12

U-M president praises Wolverines for handling 'challenges and adversity'

The day after the Penn State win, University of Michigan President Santa Ono posted on social media, praising the team's ability to "respond to any challenge
head on with a conviction to do better and to emerge even stronger."

Countless members of the University of Michigan family have reached out to me over the weekend and I wanted to express my appreciation. Like any community, we
face our share of challenges and adversity. There have been many such moments in our history. But as our team showed so… — Santa Ono (@SantaJOno) November 12,
2023

The post drew immediate and extensive blowback on social media and beyond, as observers questioned Ono's tone with Michigan in the midst of a self-inflicted
controversy.

Saturday, Nov. 11

Warde Manuel issues strong statement against Big Ten

Michigan athletic director Warde Manuel did not mince words in a statement on the Big Ten's suspension of Harbaugh.
</item 2>

<item 3>
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
</item 3>

<item 4>
Is the Coverage on Taylor Swift and Travis Kelce’s Romance Nothing but Hype or a Pivotal Shift in NFL’s Century-Old Game Plan Towards Global Recognition?
(Essentially Sports, 2023-10-25T21:30:41+00:00)
Billionaire entrepreneur Mark Cuban, who paid $1.8 million (approx.) in penalties on behalf of his team, the Dallas Mavericks, fondly remembered the former NBA
commissioner David Stern, on his death in January 2020. Stern came and instituted an anti-drug agreement and even banned All-Star player Michal Richardson for
life for drug abuse. The revitalized image of the NBA helped rake in the moolah from sponsors and made it more appealing to fans. He even went against the tide
to open up the Olympic basketball competition to NBA players resulting in the famous “Dream Team” of 1992. America has never failed to win a medal in the 19
Olympic basketball games that it has played since then.

ADVERTISEMENT Article continues below this ad

The NFL has had its fair share of, um, let’s say, “bumps” down the road as well.
</item 4>

<item 5>
Michigan sign-stealing scandal winners and losers, from Jim Harbaugh to Tony Petitti (Sporting News, 2023-11-11T22:50:13+00:00)
Penn State-Ohio State (5.3), Colorado-Oregon (5.2) and Ohio State-Notre Dame (5.1) make up that list. Michigan-Penn State should join that list on Saturday, but
the intrigue around The Game this year could double all of those games.

Last year's matchup drew an 8.05 with 17.1 million viewers. Michigan won the last two meetings and has a chance to win three in a row in the rivalry for the
first time since 1995-97. Ohio State is No. 1 in the College Football Playoff rankings. Harbaugh (2-5) and Day (1-2) both have losing records in this rivalry
matchup, and the Big Ten East, CFP and perhaps national championship are on the line.

Will this game draw a 10 Nielsen rating and 20 million viewers? It's not out of the question, and there will be enough WWE-like promos leading up to this game
to get both fan-bases frothing.
</item 5>

<item 6>
Michigan sign-stealing scandal news: Latest updates on possible Jim Harbaugh suspension from NCAA cheating investigation (Sporting News,
2023-11-06T23:58:04+00:00)
Jim Harbaugh’s attorney, Tom Mars, tells ESPN he fully expects Jim Harbaugh to fly with the team to Penn State. “I’m confident he’s going to be on that plane
one way or another," Mars said. — Pete Thamel (@PeteThamel) November 10, 2023

Punishment options for Michigan

Dinich has reported the punishment could be anything, ranging from "indefinite suspension to public reprimand or lofty fine." However, she also reported that
there is potentially a limit to what the conference can do without permission from an executive board. Anything more than two games or a fine of $10,000 would
require special permission.

Please remember the Big Ten has considered a WIDE range of punishments - everything from indefinite suspension to public reprimand or lofty fine, sources tell
me n @ESPNRittenberg … no assumptions. We have to wait for the news, but a suspension is certainly on the table.
</item 6>

<item 7>
Michigan conquers Ohio State again, but the story is far from over for 2023 Wolverines (Sporting News, 2023-11-25T23:20:56+00:00)
That set the Wolverines up at the 7-yard line. Corum slammed into the line of scrimmage three times before a fourth-and-1.

Corum heard Moore scream, "Smash." Tight end A.J. Barner did his best Triple H impersonation in the huddle: "It's time to play the game."

Michigan goes for it on 4th and goal and Blake Corum punches it in for The Game's first touchdown.



That's Corum's 21st touchdown run of the season 😲pic.twitter.com/TrF5pIpni3 — 247Sports (@247Sports) November 25, 2023

"I gotta score," Corum said. "Run through the hole and I'm about to run somebody over. That makes the hairs on my arm stick up."

Corum scored on a one-yard touchdown for a 7-0 lead. Michigan never played from behind against the Buckeyes. What was Moore thinking in that situation?
</item 7>

<item 8>
Michigan’s Rivals Fume About Sign-Stealing: ‘This Was Damn Near Espionage’ (The New York Times, 2023-11-03T22:07:31+00:00)
“They’re reading the tea leaves and wondering why the Big Ten hasn’t done anything yet. Every week and every day that goes by, people are like, ‘Something’s
gotta give.’ It’s getting a little bit out of hand when you see him (allegedly) on the Central Michigan sideline. The playing field is not level right now. How
can you have a team that you know has a competitive advantage over you still being allowed to play? That’s what the coaches are grappling with.”

Advertisement

“It feels like (former commissioner) Kevin (Warren) taking over and then COVID,” the source continued. “Tony’s walking into this situation, and people are
calling for the league to make a statement before they have all the facts.”

Despite frustration from all corners of the conference, sources at four different Big Ten schools said they do not expect the conference to levy any sort of
punishment against Michigan before the season ends.
</item 8>

<item 9>
After a Stunning Snub, College Football Is Facing an Identity Crisis (The New York Times, 2023-12-03T23:04:37+00:00)
In the CFP, Florida State would have gotten second-string quarterback Tate Rodemaker back with a month to prepare for a semifinal game.

Florida State was robbed.

And its head coach didn’t hide his disappointment.

“I am disgusted and infuriated with the committee’s decision today to have what was earned on the field taken away because a small group of people decided they
knew better than the results of the games,” Mike Norvell said in a statement. “What is the point of playing games?”

As difficult as it would have been, the right thing for the committee to do would have been to leave Alabama out. Most of us know in our gut that the Crimson
Tide — the most talented team, on paper, in the sport — are one of the four best teams. Alabama is certainly equipped to win the whole thing.

But Alabama — like peers Georgia and Ohio State, teams with a wealth of raw talent on their rosters — lost a game (at home). Better teams have been left out in
the past than this Alabama team because losses had consequences.
</item 9>

<item 10>
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
</item 10>

<item 11>
Michigan didn't need Jim Harbaugh against Penn State, but Ohio State is a different game (Sporting News, 2023-11-11T21:32:44+00:00)
Will Harbaugh be on the field before the end of the regular season? The Big Ten made a Friday ruling that will prohibit Harbaugh from being on the field for the
final three regular-season games as a result of the in-person scouting and sign-stealing scandal that led to the resignation of staffer Conor Stallions. ESPN
reported there will be a temporary restraining order hearing on Nov. 17 that will answer that question.

On the field, Michigan answered the question about its College Football Playoff. Sort of. The Wolverines proved they could beat Penn State (8-2, 5-2) in a
20-yard box. Neither team completed a pass of 20 yards or more. Michigan trailed 3-0 after the first quarter. The Wolverines had eight carries for 10 yards, and
Penn State's Abdul Carter set the tone with a jarring hit on Blake Corum that nearly resulted in a fumble.

Michigan took control with the running game.
</item 11>

<item 12>
Is the Coverage on Taylor Swift and Travis Kelce’s Romance Nothing but Hype or a Pivotal Shift in NFL’s Century-Old Game Plan Towards Global Recognition?
(Essentially Sports, 2023-10-25T21:30:41+00:00)
Revisiting the NBA caricature back in time, the New York Knicks became the first all-Black team in 1979. But their fans made their disapproval very apparent.
“It wasn’t so much New York attitudes, but fans around the league let you know how they felt about you as an opponent and as a Black man as well,” Knicks star
Earl Monroe told Newsday.

USA Today via Reuters May 15, 2021; San Antonio, Texas, USA; Phoenix Suns forward Torrey Craig (12) and San Antonio Spurs forward Keldon Johnson (3) fight for
loose ball as Suns guards Devin Booker (1) and Chris Paul (3) look on in the third quarterat AT&T Center. Mandatory Credit: Scott Wachter-USA TODAY Sports

We could list out all the scandals, fights, and controversies here, but you get the idea.
</item 12>

<item 13>
Michigan State hires Jonathan Smith: Oregon State coach leaves alma mater to take over Spartans job (CBSSports.com, 2023-11-25T23:12:10+00:00)
Four Pac-12 members -- USC, UCLA, Washington and Oregon -- will join Smith in the Big Ten next fall.

Can Smith turn the Michigan State program around?

The task now for Smith is revitalizing a Michigan State program that has struggled to field consistent success in recent seasons. After winning 10 or more games
six times in a span of eight seasons under former coach Mark Dantonio from 2010-17, the Spartans have only won more than seven games in a season once since
then, finishing 11-2 under Tucker's second year at the helm in 2021.

What seemed to be a breakthrough season for Tucker in 2021 earned him a 10-year contract extension worth $95 million, but Tucker's remaining tenure in East
Lansing was all downhill afterwards. Michigan State regressed to 5-7 in 2022 before allegations of sexual harassment surrounding Tucker against rape survivor
and activist Brenda Tracy emerged just two games into the 2023 season.
</item 13>

<item 14>
Michigan’s Rivals Fume About Sign-Stealing: ‘This Was Damn Near Espionage’ (The New York Times, 2023-11-03T22:07:31+00:00)
… No doubt this all has had a profound effect.

“This guy’s being investigated for three different things now between the (alleged) illegal signal stealing, the (alleged) illegal recruiting during COVID and
that investigation into the offensive coordinator and alleged computer hacking. There are guys (on that call) who could lose jobs, and then there’s this guy
over here (Harbaugh) who is gonna get a new, bigger contract now, and they won’t do anything about him.”

Asked to describe the tone of the coaches’ sentiment expressed to Petitti, another Big Ten coach called it “angry” — particularly at the Big Ten’s lack of
action, or even apparent interest in taking any.

“Everybody’s upset,” that coach told The Athletic. “Why is nothing being done? We want to know, what else do you need to know to take action? We (the Big Ten
head coaches) want something done now.
</item 14>

<item 15>
After a Stunning Snub, College Football Is Facing an Identity Crisis (The New York Times, 2023-12-03T23:04:37+00:00)
Better teams have been left out in the past than this Alabama team because losses had consequences.

Alabama’s loss to Texas didn’t have a consequence because we’re enamored with the SEC and what it means to beat Georgia. It didn’t matter that Alabama — though
perceived to be an entirely different team in September — lost to the Longhorns. That game could have been a Playoff game in September. It turns out it was an
exhibition.

There are plenty of people who are against the expansion of the field to 12 because of the sanctity of the regular season. But if the games in the regular
season aren’t going to matter when it comes to picking the final four, then there are no consequences for expanding to 12.

Advertisement

The regular season didn’t decide who made it this year. Thirteen people did.

The games mattered. The results didn’t.

(Photo of Mike Norvell: Isaiah Vazquez / Getty Images)
</item 15>

<item 16>
Michigan conquers Ohio State again, but the story is far from over for 2023 Wolverines (Sporting News, 2023-11-25T23:20:56+00:00)
Marvin Harrison Jr. had five catches for 118 yards and a TD, but Michigan ran seven minutes off the clock after that with a field goal. That will be the other
point of criticism for Day and defensive coordinator Jim Knowles.

MORE: Will Texas A&M pursue Ryan Day?

Ohio State had one minute to score in the end, but Michigan safety Rod Moore, a Clayton, Ohio, product, came up with the game-sealing interception in yet
another recurring theme from the 1990s.

The Wolverines aren't just winning this game now. They expect to win this game. Just like last year, McCarthy and Corum took turns repeating the famous Kobe
Bryant mantra in the post-game press conference.

"The job is not finished," McCarthy said. "We've been here before."

"We've been going through things all year, so we know how to stick together and play for each other and always have our brother's back," Corum said.
</item 16>

<item 17>
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
</item 17>

<item 18>
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
</item 18>

<item 19>
Michigan State hires Jonathan Smith: Oregon State coach leaves alma mater to take over Spartans job (CBSSports.com, 2023-11-25T23:12:10+00:00)
"He's been a part of championship staffs, coached in the College Football Playoff, and understands what's required to be successful at the highest level,
learning from some of the game's most successful coaches.

"On the field, his teams are tough and physical, yet innovative. This year, Oregon State has controlled the line of scrimmage on both sides of the ball, ranking
among the nation's leaders in both rushing offense and rushing defense. He's shown not only the ability to recruit talented student-athletes who fit his system,
but also to develop and maximize players once they're in the program."

Smith, a former quarterback for the Beavers from 1998-2001, exits Oregon State with a 34-35 overall mark as coach, including a 1-1 record in bowl games, as he
now joins the Big Ten coaching ranks.
</item 19>

<item 20>
Is the Coverage on Taylor Swift and Travis Kelce’s Romance Nothing but Hype or a Pivotal Shift in NFL’s Century-Old Game Plan Towards Global Recognition?
(Essentially Sports, 2023-10-25T21:30:41+00:00)
The October 1 matchup between the Chiefs and the Jets drew 27 million viewers. And became the most-watched Sunday TV show since the last Super Bowl! And the
last Super Bowl had an actual concert by Rihanna.

Revealed: The Dark Side Of Pat McAfee & Why Rob Gronkowski Is In The Firing Line

Kaepernick, who? Deflategate, what? In the media frenzy surrounding the two lovebirds, other issues get thrown out of the window. How can fans forget that the
NFL is also organizing games outside the American borders with a view to garnering international appeal? The NFL has scheduled 5 games abroad this season. It is
also addressing the concussion controversies. It recently added both new concussion rules and a new $100 million initiative called “Play Smart, Play Safe,” to
make the sport safer for players. And all the effort is showing.
</item 20>

<item 21>
Michigan’s Rivals Fume About Sign-Stealing: ‘This Was Damn Near Espionage’ (The New York Times, 2023-11-03T22:07:31+00:00)
We (the Big Ten head coaches) want something done now. I don’t think people understand the advantage that what they’re (allegedly) doing gives you. People
think, ‘OK, now that everybody knows, we all can just move on.’ Like, ‘now, it’s fair.’ Well, no, it isn’t. Not at all. This changes the way you operate. A lot
of teams have been doing things a certain way for years. Now, it’s forcing you to teach your players a whole new way to communicate just for them. People think
that this is just advanced scouting. This was damn near espionage.”

A third Big Ten head coach told The Athletic that this is “one of the most egregious breaches in the spirit of the game” he’s ever heard of.

“They (Michigan) have been manipulating the game and cheating the game for two-and-a-half years.
</item 21>

<item 22>
Michigan’s Rivals Fume About Sign-Stealing: ‘This Was Damn Near Espionage’ (The New York Times, 2023-11-03T22:07:31+00:00)
To know exactly what the other team is doing, Michigan might as well have been playing with 15 guys on the field,” he said. “What’s the message the Big Ten is
sending now by doing nothing? Win now, pay later? We might as well just send people to (scout) their practices and their games. It doesn’t encourage anybody to
follow the rules. It’s just telling them to do the opposite and say, f— it.”

GO DEEPER Michigan should be punished, say 94% of CFB coaches in our poll. What else did it reveal?

The NCAA is investigating Michigan’s football program amid allegations that the Wolverines used illegal in-person scouting and the recording of signals to steal
signs this season. Ahead of the Michigan-Michigan State game on Oct. 21, the Big Ten approached MSU and said it was made aware of “credible evidence” regarding
the sign-stealing allegations. The Big Ten said it would monitor the NCAA’s investigation into Michigan.
</item 22>

<item 23>
N.F.L. Executives on the Biggest Questions Facing Super Bowl Contenders (The New York Times, 2023-10-16T23:58:53+00:00)
The Eagles come to mind after falling 20-14 to the Jets.

Most would agree teams have gotten smarter in their handling of fourth-down situations, going for it when it makes sense, instead of punting almost out of
habit. Making the aggressive play to win can absolutely be the smart play, regardless of results. It’s progress when teams seriously wrestle with these
questions, not just on fourth down, but in all aspects of their strategy.

But when the Eagles tossed a killer interception on third-and-9 from their own 46-yard line while protecting a 14-12 lead against the Jets with two minutes left
in the fourth quarter, a longtime NFL exec reached out with a question: “Have we come too far with coaches wanting to ‘win’ the game instead of sometimes being
more traditional?”

His point: Philadelphia could have run the ball on third-and-9, then punted from around midfield.
</item 23>

<item 24>
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
</item 24>

<item 25>
Michigan didn't need Jim Harbaugh against Penn State, but Ohio State is a different game (Sporting News, 2023-11-11T21:32:44+00:00)
McCarthy is a leader. Corum (26–145, 2 TDs) and Edwards (10-52, TD) can grind on anybody. The defense now has allowed just 7.5 points per game.

But the Wolverines will not be able to beat Ohio State the same way they beat Penn State. That 20-yard box will stretch across the entire country, and it
requires a coach ready to make those legacy-defining decisions. The Buckeyes have Heisman Trophy candidate Marvin Harrison Jr., and he was the separator when
Ohio State beat Penn State 20-12 on Oct. 21. The Buckeyes have a top-five defense, too, and all of the back-and-forth rage between those rivals will reach a
1970s pitch – 1973 for those who know – as they barrel toward another 11-0 vs. 11-0 showdown at the Big House.

Who is the good guy? Who is the villain?
</item 25>

<item 26>
Michigan sign-stealing scandal winners and losers, from Jim Harbaugh to Tony Petitti (Sporting News, 2023-11-11T22:50:13+00:00)
Sherrone Moore filled in for Jim Harbaugh as Michigan's coach today.



He was emotional after defeating Penn State.



"I want to thank Coach Harbaugh. ... I love the s— out of you man."



🎥: @CFBONFOX pic.twitter.com/NmXZWrGxCM — The Sporting News (@sportingnews) November 11, 2023

Moore, however, will emerge as a leading candidate for some jobs as a result, and that is a huge victory for one of the nation’s top assistants.

Losers

Connor Stalions

Stalions made a name for himself. He will be remembered as the Michigan staffer as the center of this sign-stealing scandal. Stalions resigned on Nov. 3. The
NCAA investigation whether Michigan was involved with in-person scouting, which is prohibited by the NCAA, put Stalions into the college football consciousness.
</item 26>

<item 27>
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
</item 27>

<item 28>
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
</item 28>

<item 29>
Michigan’s Rivals Fume About Sign-Stealing: ‘This Was Damn Near Espionage’ (The New York Times, 2023-11-03T22:07:31+00:00)
He has denied knowledge of the alleged scouting.

GO DEEPER Ten thoughts on the Big Ten's 2024 schedule

Required reading

(Photo: David Berding / Getty Images)
</item 29>

<item 30>
Michigan sign-stealing scandal winners and losers, from Jim Harbaugh to Tony Petitti (Sporting News, 2023-11-11T22:50:13+00:00)
According to ESPN.com, Stalions allegedly purchased tickets to more than 35 games involving Big Ten and potential CFP opponents.

Photos also surfaced os a person who resembles Stalions on the Central Michigan sideline during the Chippewas' opener against Michigan State on Sept. 1. Central
Michigan has yet to confirm who that person was, another layer to a wild Stalions story that has supplied endless meme and conspiracy material on social media.

Stalions might be the fall guy who saved Michigan from further wrath from the Big Ten and NCAA or the reason the Wolverines' national championship season
unraveled. That fate is yet to be determined, but at minimum he will be one of those controversial names we'll never forget, like Harvey Updyke and Nevin
Shapiro.

Jim Harbaugh

On one hand, Harbaugh could serve his second suspension within one season and further NCAA punishment could be coming down the line.
</item 30>

<item 31>
Michigan conquers Ohio State again, but the story is far from over for 2023 Wolverines (Sporting News, 2023-11-25T23:20:56+00:00)
Michigan never played from behind against the Buckeyes. What was Moore thinking in that situation?

"Let's go," Moore said. "That's it. The players just practiced so well this week, and when you have confidence they are going to execute at a high level, it's
just like first down."

Moore favored a conservative game plan the previous two weeks against No. 11 Penn State and Maryland; one that amounted to 91 running plays and 31 pass
attempts. Yet he was the aggressor in The Game. The Wolverines converted three fourth-down conversions in the first half. McCarthy – who finished 16 of 20 for
148 yards and a TD – threw a 22-yard touchdown to Roman Wilson between two defenders for a 14-3 lead.

Interim coach Sherrone Moore leads Michigan to win

​​"Coach Moore said from the get-go that he was going to the most-aggressive game he's ever called," McCarthy said.
</item 31>

<item 32>
Michigan didn't need Jim Harbaugh against Penn State, but Ohio State is a different game (Sporting News, 2023-11-11T21:32:44+00:00)
Michigan took control with the running game. Donovan Edwards ripped off a 22-yard run, and Corum added a 44-yard run to set up a 3-yard TD run. Edwards added a
22-yard TD run on the next drive. The same “Lightning and Lightning” tandem that burned the Nittany Lions last year staked Michigan to a 14-9 halftime lead.

MORE: Michigan got off easy with Harbaugh suspension

Then, the Wolverines took the ball out of quarterback J.J. McCarthy's hand. The Wolverines had 30 rushing attempts and zero passing attempts save a pass to
tight end Colston Loveland that drew a pass interference call. McCarthy finished 7 of 8 passing for 60 yards in what has to be the most conservative
play-calling victory in a top-10 showdown in recent memory. Was this old-school Big Ten football or Big Ten West football in disguise?
</item 32>

<item 33>
Michigan conquers Ohio State again, but the story is far from over for 2023 Wolverines (Sporting News, 2023-11-25T23:20:56+00:00)
Corum flashed Zinter's number "65" with his hands. From there, the Wolverines handled business for a third straight year in The Game in a 30-24 victory.

MORE: Corum gives tribute to teammate after key touchdown

Michigan tops Ohio State again

Michigan (12-0) beat Ohio State (11-1) for the third straight season. Corum, who missed last year's 45-23 victory against the Buckeyes because of a knee injury,
could feel Zinter's pain.

"Zak's my guy," Corum said. "He's another guy that came back for unfinished business. That's why he came back. Seeing him go down and hurt, it was very
emotional seeing what he was going through. We came together, and we knew we had to do it for him. The very next play, 'Boom.'"

Boom, indeed. This Michigan story is far from over.

It will continue at least into the Big Ten championship game against No.
</item 33>

<item 34>
Bryce Harper 'Atta Boy' controversy, explained: Why Orlando Arcia's comments led to backlash, Alanna Rizzo rant (Sporting News, 2023-10-13T21:58:11+00:00)
"Thanks to all the reporters, journos, baseball twitter weirdos, acquaintances and friends who had my back over the last 48 hours," Mintz wrote. "And also, of
course, the BBWAA. Community feels good, humbled to be a part of this one. Now, let's enjoy some ball."
</item 34>

<item 35>
Michigan sign-stealing scandal news: Latest updates on possible Jim Harbaugh suspension from NCAA cheating investigation (Sporting News,
2023-11-06T23:58:04+00:00)
Judicial officer: Timothy P. Connors



Connors is a U-M lecturer, adjunct professor at Wayne State and Vermont. pic.twitter.com/WR4G4WTtfH — Chris Vannini (@ChrisVannini) November 11, 2023

Michigan players respond to Harbaugh suspension

J.J. McCarthy and a handful of other Michigan players tweeted "Bet" at the same time on Friday night, hours after Harbaugh was suspended for the remainder of
the regular season. McCarthy wore a shirt that said "Free Harbaugh" when the 59-year-old was serving a separate suspension at the start of the season.

Michigan players had a simple message on social media following Jim Harbaugh's suspension 👀
</item 35>

<item 36>
Michigan sign-stealing scandal news: Latest updates on possible Jim Harbaugh suspension from NCAA cheating investigation (Sporting News,
2023-11-06T23:58:04+00:00)
The Big Ten is also asserting that they are not actually sanctioning Harbaugh, for whom they have no proof of wrongdoing, but that he "embodies the university
for purposes of its football program." pic.twitter.com/S660G6zQaa — Stewart Mandel (@slmandel) November 10, 2023

Jim Harbaugh suspended for remainder of regular season

The Big Ten has officially announced a suspension for Jim Harbaugh for the remaining three games of the regular season, including the Wolverines' rivalry game
with Ohio State.

In a statement, the Big Ten said Michigan conducted an "impermissible, in-person scouting operation over multiple years."

OFFICIAL: Jim Harbaugh suspended for the remaining three regular season games.
</item 36>

<item 37>
Michigan sign-stealing scandal news: Latest updates on possible Jim Harbaugh suspension from NCAA cheating investigation (Sporting News,
2023-11-06T23:58:04+00:00)
17

Michigan fires coach connected to destroying evidence in investigation

Yahoo Sports' Ross Dellenger and Dan Wetzel reported on Friday that the NCAA's investigation into the sign-stealing and advanced scouting operation has found
that Stalions was at least partially funded by a Michigan booster, named as "Uncle T," and that linebackers coach Chris Partridge, who was fired by Michigan on
Friday, is alleged to have destroyed evidence on a computer pertaining to the sign-stealing allegations.

The report states Partridge is not believed to have had prior knowledge of the sign-stealing operation, but then later worked to cover it up. In a statement on
his firing, Michigan appeared to indicate a connection between Partridge and the NCAA's investigation, which Dellenger and Wetzel reported is still in the early
stage:

From the outset, our focus has been on seeking due process and allowing the NCAA to conduct a fair and deliberate investigation.
</item 37>

<item 38>
Michigan sign-stealing scandal news: Latest updates on possible Jim Harbaugh suspension from NCAA cheating investigation (Sporting News,
2023-11-06T23:58:04+00:00)
sources indicate a Big Ten response is coming early this afternoon. Michigan leaves at 1 and flies midafternoon out of Detroit after getting through security
checks, per usual routine. — Heather Dinich (@CFBHeather) November 10, 2023

Thursday, Nov. 9

Big Ten considering 'range of possibilities' for Harbaugh

ESPN's Adam Rittenberg reported there are a "range of possibilities" the Big Ten is considering as punishment for Harbaugh. The possibility of a suspension has
often garnered most of the headlines, but Rittenberg reported the conference is also considering a significant fine.

While we wait ... although Big Ten has considered a suspension for Michigan coach Jim Harbaugh, it is considering a range of possibilities, including a
significant fine, sources told @CFBHeather and me. Public reprimands/fines are common penalties under B1G sportsmanship policy.
</item 38>

<item 39>
Michigan’s Rivals Fume About Sign-Stealing: ‘This Was Damn Near Espionage’ (The New York Times, 2023-11-03T22:07:31+00:00)
The Big Ten said it would monitor the NCAA’s investigation into Michigan.

Advertisement

“The Big Ten is so much more powerful than the NCAA,” that third Big Ten coach said. “Why are you just sitting back and doing nothing about this? The Big Ten
can’t pound its chest for the last 30 years about how it does the right thing ethically (when other conferences like the SEC won’t) and then have this go on. If
this were a team in the bottom half of the Big Ten, would this be handled in the same way?

“When a running back gets hurt against Michigan because they knew exactly what play was coming, will that kid and his family have the ability to sue the Big
Ten?”

The NCAA investigation is ongoing, a process that typically moves slowly, which makes it difficult to imagine it will reach a resolution by the time the
postseason begins.
</item 39>

<item 40>
Bryce Harper 'Atta Boy' controversy, explained: Why Orlando Arcia's comments led to backlash, Alanna Rizzo rant (Sporting News, 2023-10-13T21:58:11+00:00)
The only difference was that Janes did not name the player.

The reporting of Arcia's comments proved to be divisive across the baseball community. Among the most prominent to call out Mintz was Rizzo. The "High Heat"
co-host said she was irritated by "bloggers or podcasters" going into the clubhouse and "ruining it for the rest of us" when "some jackoff comes in at the end
of the season that gets a credential, God only knows why." She called the clubhouse "a sacred place" and defended the players who were upset by the reporting of
Arcia's comments, before she called Mintz out by name.

MORE: Every time Nick Castellanos has interrupted a somber moment with a home run

"First of all, this guy, Jake Mintz, that's not even a reporter," Rizzo said. "That's taking away from true reporters and true journalists.
</item 40>

<item 41>
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
</item 41>

<item 42>
Michigan sign-stealing scandal winners and losers, from Jim Harbaugh to Tony Petitti (Sporting News, 2023-11-11T22:50:13+00:00)
On the other, he could emerge as a martyr for Michigan fans given there is reportedly no proof yet he knew of Stalion's rule-breaking.

Harbaugh will have a temporary restraining order hearing on Nov. 17, according to ESPN.com. That will determine whether he returns to the field for the final
two games against Maryland and No. 1 Ohio State.

It's an inescapable fact that Harbaugh is 31-3 – and Michigan is 35-3 – since the program bottomed out at the end of the 2020 season. While it is almost
impossible to quantify how much the alleged in-person scouting and sign-stealing would have helped the Wolverines over the past two seasons, it will be a tough
stain to erase knowing those violations are now on the Big Ten record.

What will Harbaugh do now? That is almost impossible to predict given his combative personality. Would he sign a contract extension and continue to fight the
NCAA? Will Harbaugh finally take the parachute to the NFL?
</item 42>

<item 43>
Will the Panthers fire Frank Reich? Impatient David Tepper could turn up Carolina's hot seat with 'Thursday Night Football' loss (Sporting News,
2023-11-06T21:17:51+00:00)
And I don't think that people are thinking that there's trouble right now with those people. But would anybody be surprised with anything that happens in
Carolina this year? No."

"One thing about David Tepper is he's not very patient and he does whatever he wants..



I don't think anybody would be surprised if anything happens in Carolina with the season that they're having" ~ @AdamSchefter #PMSLive
pic.twitter.com/PV7bfVUtHp — Pat McAfee (@PatMcAfeeShow) November 6, 2023

Schefter pointed to Carolina's firing of Rhule as an example of Tepper's impatience. There had been rumors that the owner was going to wait things out with
Rhule in the weeks leading up to his in-season departure. Instead, he decided to pull the trigger and part with him.
</item 43>

<item 44>
N.F.L. Executives on the Biggest Questions Facing Super Bowl Contenders (The New York Times, 2023-10-16T23:58:53+00:00)
The Jets would have taken possession with barely more than a minute remaining, no timeouts and Zach Wilson behind center facing a ferocious Eagles pass rush.
Was having Hurts pass in a situation when gaining a first down is unlikely the smart play under the circumstances?

“Are coaches not willing to go against the slight statistical benefit that WPA (win probability added) provides?” the exec asked. “Is it no longer OK to let
your defense win the game when it clearly has a huge talent/performance advantage? This Eagles game was 14-12, not 37-35.”

Critics attuned to the trend toward aggressiveness pounce when teams pass up chances to go for the kill. Coaches know team owners tend to be fans, not football
strategists. They know owners listen to these criticisms. That can provide an incentive to make the aggressive play, to go down swinging.

“Everybody thinks they are playing Brady, Manning or Rodgers in two-minute,” a veteran coach said.
</item 44>

<item 45>
Bryce Harper 'Atta Boy' controversy, explained: Why Orlando Arcia's comments led to backlash, Alanna Rizzo rant (Sporting News, 2023-10-13T21:58:11+00:00)
Here's what you need to know about the controversy.

Bryce Harper Atta Boy controversy, explained

It all begins with the play. In Game 2, Harper led off the top of the ninth in a one-run game with a walk. J.T. Realmuto flew out to center for the first out,
but on the next play, Nick Castellanos lifted a deep fly into right-center. It appeared it would be extra bases, and if Harper took off, he'd be able to score
the tying run on the play.

The problem for the Phillies was that Harris made a leaping grab up against the wall to rob Castellanos of the hit. Harper had well since rounded second on his
way to third, but the throw from Harris missed both cut-off men by second. Austin Riley, however, ran in and scooped the ball and made an impressive throw to
Matt Olson to nab Harper and end the game.
</item 45>

<item 46>
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
</item 46>

<item 47>
After a Stunning Snub, College Football Is Facing an Identity Crisis (The New York Times, 2023-12-03T23:04:37+00:00)
The Seminoles were the ones left feeling cheated.

And with that one decision, the committee didn’t just choose teams in a given year. It revealed to the world the ugly truth about college football — this sport
is a beauty contest where decisions on which teams can win the national title are sometimes made as much in a cozy hotel boardroom in Grapevine, Texas, as they
are on the actual field.

The Playoff field is as follows: 1. Michigan, 2. Washington, 3. Texas, 4. Alabama.

Michigan and Washington made it through the season unscathed. Texas lost a nail-biter in its rivalry game to Oklahoma, and the Longhorns, wait for it, beat
Alabama.

We wouldn’t be having this discussion if Florida State’s star quarterback, Jordan Travis, didn’t break his leg two weeks ago. But the Seminoles team that just
beat Louisville 16-6 in the ACC Championship Game was relying on a third-string quarterback. The win was far from impressive.
</item 47>

<item 48>
Michigan State hires Jonathan Smith: Oregon State coach leaves alma mater to take over Spartans job (CBSSports.com, 2023-11-25T23:12:10+00:00)
"All of Beaver Nation should thank him for the lasting legacy he has left to his alma mater and wish him and his family nothing but the best moving forward."

Smith held multiple assistant jobs from 2002-17 before returning to his alma mater as coach in 2018. Smith spent six seasons as an assistant under former
college football coach Chris Petersen, first as Boise State's quarterbacks coach from 2012-13 and then as Washington's quarterbacks coach and offensive
coordinator from 2014-17. His time in Seattle included a Pac-12 title and College Football Playoff berth for Washington during the 2016 season.

Smith's exit from Oregon State comes at a point where the Beavers face much uncertainty as the Pac-12 prepares to lose 10 of its 12 members ahead of the 2024
season. Oregon State and Washington State are the only schools in the conference that have not been picked up by other power leagues.
</item 48>

<item 49>
Michigan State hires Jonathan Smith: Oregon State coach leaves alma mater to take over Spartans job (CBSSports.com, 2023-11-25T23:12:10+00:00)
Tucker was promptly suspended before the school fired him with cause several weeks later. The Spartans went on to finish 4-8 with assistant Harlon Barnett
serving as interim coach the rest of the way, ending the season with a 42-0 loss to Penn State on Friday.

While Michigan State has found itself playing in a stacked Big Ten East division for roughly the last decade -- Michigan, Ohio State and Penn State have all
been annual opponents -- that will change in Smith's debut season as the Big Ten expands and does away with divisions. Smith will also get the chance to face
some familiar opponents from his time at Oregon State given that four Pac-12 schools are joining the conference. As far as the 2024 season is concerned on that
front, Michigan State still faces Ohio State and Michigan (the Wolverines are a protected rival) but avoids Penn State. Oregon is the only newcomer it will face
next fall, with the Spartans set to travel to Eugene, Oregon, in early October.
</item 49>

<item 50>
Michigan sign-stealing scandal winners and losers, from Jim Harbaugh to Tony Petitti (Sporting News, 2023-11-11T22:50:13+00:00)
Will the Harbaugh-Petitti battle lead to an unsettling end for Michigan? President Santa Ono and athletic director Warde Manuel have voiced their support for
Harbaugh, but that has been more about due process than admitting to any wrongdoing.

It has thrust the Wolverines – a talented team led by J.J. McCarthy, Blake Corum and a defense that allows 7.5 points per game – into a villain role, complete
with visits from Ric Flair. That only works when you are winning. One loss, and all this could unravel, and the Terps and Buckeyes still remain on the schedule.

Will a national championship outweigh the stain on their reputation? The answer is almost always yes for the fans, but the program might feel a different
long-term result.
</item 50>

<item 51>
Is the Coverage on Taylor Swift and Travis Kelce’s Romance Nothing but Hype or a Pivotal Shift in NFL’s Century-Old Game Plan Towards Global Recognition?
(Essentially Sports, 2023-10-25T21:30:41+00:00)
From Colin Kaepernick taking a knee during the national anthem to protest against racism, to domestic violence by players, to the Deflategate and Bountygate
scandals. But there have been bigger crises as well. The Union of Concerned Scientists accused the NFL of trying to intimidate scientists studying the link
between CTE and pro football. The NFL is already way past other leagues in terms of revenue, and now with the pivotal shift in its strategy, “America’s favorite
pastime” is on its way to world domination.

ADVERTISEMENT Article continues below this ad

Editorial Disclaimer: The views expressed are those of the author and do not necessarily reflect the views of EssentiallySports.

Watch This Story: Will Patrick Mahomes Eclipse Tom Brady as the NFL Goat? Travis Kelce Responds
</item 51>

<item 52>
Michigan didn't need Jim Harbaugh against Penn State, but Ohio State is a different game (Sporting News, 2023-11-11T21:32:44+00:00)
Was this old-school Big Ten football or Big Ten West football in disguise?

Michigan's defense clamped on Penn State quarterback Drew Allar (10 of 22, 70 yards, TD), and the Nittany Lions had one play of more than 20 yards. Corum closed
the drama with a 30-yard TD.

The same-old questions about Nittany Lions coach James Franklin in big games resurfaced, but those are back-page headlines in the Big Ten East race. The quirky
tiebreaker involving Big Ten West opponents? That's out the door. The winner-take-all showdown between Ohio State and Michigan? There are a full two weeks of
anticipation for that now.

Harbaugh reportedly watched the game from the team hotel, and now the leading piece of drama returns. Moore's postgame emotion was a headline, but there is a
reality that set in through a brutal watch of a second half.

McCarthy is a leader.
</item 52>

<item 53>
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
</item 53>

<item 54>
Michigan sign-stealing scandal news: Latest updates on possible Jim Harbaugh suspension from NCAA cheating investigation (Sporting News,
2023-11-06T23:58:04+00:00)
— Pete Thamel (@PeteThamel) November 9, 2023

Big Ten commissioner to miss College Football Playoff meeting Thursday

Tony Pettiti is set to be absent from the College Football Playoff meeting set to take place in Dallas on Thursday. That leads to further speculation that
punishment could be coming soon in Michigan's sign-stealing scandal.

Big Ten commish Tony Petitti will NOT be in Dallas for today’s @CFBPlayoff meeting. — Heather Dinich (@CFBHeather) November 9, 2023

Wednesday, Nov. 8

Ohio State denies having shared Michigan's play signals with Purdue

Ohio State coach Ryan Day was asked about the allegations that Ohio State had given information about Michigan's signs to Purdue before the Big Ten championship
game in 2022. He denied them forthright.

“I can tell you this right now that nobody here did any of that,” Day said, per the Columbus Dispatch.
</item 54>

<item 55>
Michigan sign-stealing scandal news: Latest updates on possible Jim Harbaugh suspension from NCAA cheating investigation (Sporting News,
2023-11-06T23:58:04+00:00)
Public reprimands/fines are common penalties under B1G sportsmanship policy. — Adam Rittenberg (@ESPNRittenberg) November 9, 2023

Michigan's 10-page letter to Big Ten released

Yahoo Sports' Dan Wetzel posted Michigan's full 10-page letter to the Big Ten on X (formerly Twitter). The letter outlines Michigan's belief that the Big Ten
would be acting too hastily in suspending Harbaugh or levying any punishment toward Michigan before the completion of a full NCAA investigation.

In the letter, Michigan said the university has not had the chance to review all the evidence and respond, which oversteps the due process allowed for Michigan.
It said Pettiti should not have unilateral authority to punish violations without first having a completed investigation and then a response from the Compliance
and Reinstatement Committee, which would recommend penalties for rules violations.
</item 55>

<item 56>
Michigan sign-stealing scandal news: Latest updates on possible Jim Harbaugh suspension from NCAA cheating investigation (Sporting News,
2023-11-06T23:58:04+00:00)
He said while the program has been disappointed to hear about the allegations, it also feels that the conference skipped due process of a complete investigation
in suspending Harbaugh on Friday.

Manuel described the suspension as "completely unethical" and "an assault on the rights of everyone (especially in the Big Ten) to be judged by a fair and
complete investigation."

"All of the Head Coaches in the Big Ten (some who have been accused of actively participating in the trading of signals of opponents) and my Big Ten AD
colleagues can rejoice today that someone was 'held accountable,' but they should be worried about the new standard of judgment (without complete investigation)
that has been unleashed in this conference," Manuel said.

Statement from Michigan AD Warde Manuel. “I find that completely unethical, insulting to a well-established process within the NCAA, and an assault on the
rights of everyone (especially in the Big Ten) to be judged by a fair and complete investigation.”
</item 56>

<item 57>
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
</item 57>

<item 58>
Michigan’s Rivals Fume About Sign-Stealing: ‘This Was Damn Near Espionage’ (The New York Times, 2023-11-03T22:07:31+00:00)
The Big Ten does have the ability to act under its sportsmanship policy, but that doesn’t mean that it would want to act quickly or decisively before the NCAA
completes its entire investigation and allows Michigan a chance to respond to its findings.

This is an unprecedented situation; whatever Petitti decides to do (or not do) will set a precedent. The Big Ten itself doesn’t have investigators, so it needs
to rely upon the NCAA to do that part — and to determine who else was involved in the alleged scouting scheme. It’s not clear exactly what the coaches would
want the league to do to punish Michigan; banning the team from competing in the Big Ten championship, for example, would harm players who had nothing to do
with the sign-stealing apparatus.

A source briefed on the coaches’ call said Big Ten coaches are concerned about whether Michigan “should represent the Big Ten.”

“No matter what happens, if Michigan continues to move forward, the clouds will follow,” the source said.
</item 58>

<item 59>
How Did the Houston Astros Cheat in 2017? Everything to Know About a Dark Moment in MLB History (Essentially Sports, 2023-10-22T21:30:14+00:00)
Indeed, who would refuse such an offer?

Of course, not everyone had been comfortable with what the Astros were doing behind the scenes, like Mike Fiers. Of that original team, four are still in
Houston – Jose Altuve, Alex Bregman, Lance McCullers Jr., and Justin Verlander. What makes fans ponder their extreme methods is how genuinely skilled each is.
Yet they participated, willingly or unwillingly, in the biggest baseball scandal in recent years.

Is the Inquiry Still On?

After the Jan. 13, 2020 declaration, not much came from Manfred. Other than suspending the front office, MLB fined the Astros $5 million, the maximum amount,
and made them forfeit first and second-round draft picks. For better or for worse, the inquiry is over.

How Did the MLB World See This Entire Case?

Needless to say, cheating is never taken in a positive light.
</item 59>

<item 60>
Michigan sign-stealing scandal news: Latest updates on possible Jim Harbaugh suspension from NCAA cheating investigation (Sporting News,
2023-11-06T23:58:04+00:00)
Michigan noted that regarding evidence, there is nothing it is aware of to claim athletics personnel at Michigan committed the allegations, and said the rule
pertaining to electronic recordings of games is limited to the use of field equipment used by the institution for the game it is currently playing. It also said
Michigan has not seen much evidence, but rather has just heard a summary of second- and third-hand information.

The letter also said given the use of sign decoding by other universities, there is little reason to believe Stalions' efforts proved meaningful, and pointed
out the teams' average margin of victory since Stalions was suspended. It said the information obtained is "at most a reflection of information that teams can
and easily do obtain through public and permissible sources," and because of that it is "highly dubious" that Stalions' observations would have a "material
effect on the integrity of the competition."
</item 60>

<item 61>
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
</item 61>

<item 62>
How Did the Houston Astros Cheat in 2017? Everything to Know About a Dark Moment in MLB History (Essentially Sports, 2023-10-22T21:30:14+00:00)
The phrase “Take the trash out” took a life of its own.

Facing Cheating Allegations, Tom Brady’s Successor Mockingly Asserts that He’s not a Houston Astros Fan

ADVERTISEMENT Article continues below this ad

But it wasn’t just baseball fans who were furious, other sports also considered it a tragedy. Fair play is taken seriously; no one likes being cheated out of an
award. Even NBA star LeBron James spoke out in criticism against Houston.

Has MLB Taken Any Action to Stop It From Happening Again?

Before the details of the scandal came out, MLB had handed out warnings here and there against sign stealing. After this fiasco, new restrictions in 2020
prevented the video replay from including the catcher’s signs. In 2022, PitchCom devices also debuted. They allow a pitcher to communicate with the catcher
without signs and a few other players on the field can also wear it. But will these tactics help?
</item 62>

<item 63>
Bryce Harper 'Atta Boy' controversy, explained: Why Orlando Arcia's comments led to backlash, Alanna Rizzo rant (Sporting News, 2023-10-13T21:58:11+00:00)
She said that reporters abstaining from writing on his comments would be akin to "protecting players from themselves."

She also went on to say that what Arcia said wasn't a slur or hateful, and added the only reason anyone was concerned about it was because "they know it would
motivate Harper to beat [the Braves]. Which implies the Braves do not feel they can handle a motivated Bryce Harper and are willing to suggest that is why they
lost the other day."

"I didn't say something sooner because I thought this was overblown," Janes continued. "And frankly, I thought it wouldn't help. But I am genuinely disgusted by
the way some media members are acting like a reporter was in the wrong for this. He wasn't.

"And if the Braves let this be the story of their series, that's on them. Win and it goes away. Period."
</item 63>

<item 64>
Michigan sign-stealing scandal news: Latest updates on possible Jim Harbaugh suspension from NCAA cheating investigation (Sporting News,
2023-11-06T23:58:04+00:00)
He initially told reporters he would speak during a scheduled hearing Friday, but then backtracked a bit and sounded as if he wasn't committed to that course,
pending legal advice.

Harbaugh said he will speak at the hearing on Friday. He said he’s looking for “due process.” Mentioned a civics class he took 40 years ago. “You’re innocent
until proven guilty. I’d like that opportunity.” — Nicole Auerbach (@NicoleAuerbach) November 13, 2023

Harbaugh clarifies that he’s not sure whether he’ll testify Friday. “That’s not my dance floor,” he says. “I’ve watched a lot of shows. I’ve watched Judge Judy
a lot.” — Austin Meek (@byAustinMeek) November 13, 2023

Sunday, Nov.
</item 64>

<item 65>
Bryce Harper 'Atta Boy' controversy, explained: Why Orlando Arcia's comments led to backlash, Alanna Rizzo rant (Sporting News, 2023-10-13T21:58:11+00:00)
It had been reported Monday night when the Braves beat the Phillies in Game 2 of the NLDS, ending on a critical base-running play involving Bryce Harper, that
Atlanta shortstop Orlando Arcia taunted about the Phillies' star in the clubhouse after the game.

That report wound up generating plenty of controversy.

Fox Sports' Jake Mintz, one of the two brothers that runs the popular Cespedes Family BBQ account, reported that Arcia had been celebrating after Atlanta's 5-4
win by saying "Atta boy Harper" in the Braves' clubhouse, jeering at the fact that Harper was doubled off on the basepaths after an outstanding catch by Michael
Harris in center field and an impressive relay to get Harper at first to end the game.

The comments wound up coming back to bite the Braves two days later, with Harper homering twice — staring down Arcia after each — as he powered the Phillies to
the 10-2 Game 3 win.
</item 65>

<item 66>
N.F.L. Executives on the Biggest Questions Facing Super Bowl Contenders (The New York Times, 2023-10-16T23:58:53+00:00)
• Detroit Lions: Have they fixed their defense?

What I think: The Lions are 11.3 EPA per game better on defense from last season to this season, the second-largest gain for any team since last season. The
schedule has something to do with that, but the 2022 defense was so bad, it couldn’t stop Carolina. Detroit will win the NFC North and return to the playoffs,
which will make this season successful no matter what happens from there. But we won’t find out until the postseason just how much better the defense might be,
because the remaining schedule features so many weak offenses. That includes Chicago twice, Green Bay, Denver and Las Vegas.

Exec comment: “Yes, I do buy them. This is a league full of teams who do not really know who they are week to week. Detroit knows who they are, and they play to
their identity. Anytime you know who you are and play to that style, you have a chance to be dangerous.”
</item 66>

<item 67>
Michigan conquers Ohio State again, but the story is far from over for 2023 Wolverines (Sporting News, 2023-11-25T23:20:56+00:00)
Say what you will about the adults who were involved in the in-person scouting scandal. Staffer Conor Stallions and assistant coach Chris Partridge are gone.
Harbaugh was suspended. He addressed the team Friday, and McCarthy listened closely.

"The one thing he said that stuck out, the whole mantra, 'The Team, The Team, The Team,'" McCarthy said. "He said we are that team."

That is true. This is the most mature player-led Michigan football team on the field since 1997 – when the program beat Ohio State 20-14 in a top-five showdown
and split the national championship with Nebraska.

Since then, the Buckeyes found a way in these high-magnitude games, with 2006 and 2016 the best examples. On a stage that was somewhere in between those
classics, the Wolverines pointed the pressure in the other direction. Marvin Harrison Jr.
</item 67>

<item 68>
Michigan conquers Ohio State again, but the story is far from over for 2023 Wolverines (Sporting News, 2023-11-25T23:20:56+00:00)
They have, and the last two years ended with victories in the Big Ten championship game before losses to Georgia and TCU in the CFP semifinals. The NCAA
investigation will continue, but if Saturday proved anything, the Wolverines will play out the rest of the season on their terms.

This Michigan story is far from over, and Harbaugh returns Monday.

Boom, indeed.
</item 68>

<item 69>
Michigan’s Rivals Fume About Sign-Stealing: ‘This Was Damn Near Espionage’ (The New York Times, 2023-11-03T22:07:31+00:00)
For the past two weeks, coaches across college football have been riveted by the alleged Michigan sign-stealing scheme, but inside the Big Ten, the topic has
been more than just a curiosity. On Wednesday’s Big Ten coaches video call with commissioner Tony Petitti, and after Jim Harbaugh left the call, that
frustration was voiced loud and clear, according to conference coaches, who said they don’t feel like the new Big Ten commissioner is “motivated” to do anything
about the Wolverines.

Advertisement

“There is just a ton of frustration,” a Big Ten coach told The Athletic on Thursday morning. “Look at Jim Harbaugh’s record before this started. The guy was on
the hot seat before 2021, and now he’s like the king of college football. … No doubt this all has had a profound effect.
</item 69>

<item 70>
Bryce Harper 'Atta Boy' controversy, explained: Why Orlando Arcia's comments led to backlash, Alanna Rizzo rant (Sporting News, 2023-10-13T21:58:11+00:00)
Win and it goes away. Period."

The incident elicited a response from the Baseball Writers Association of American (BBWAA), which defended Mintz, a BBWAA member.

"For the record, Jake is a BBWAA member in good standing and has earned the right to be credentialed through his hard work and quality baseball coverage," the
BBWAA wrote in the statement. "To assert otherwise, in vulgar terms, is both unprofessional and unacceptable. The BBWAA is deeply troubled that the league's own
network would permit the disparaging of one of our members in this fashion. Scrutinizing our work is part of the territory but comments such as these should
have no place on MLBN."

Following the backlash to her comments, Rizzo apologized on "High Heat" to Mintz and the BBWAA. She said her comments were "completely inappropriate, it was
inaccurate and it was beyond unprofessional.
</item 70>

<item 71>
How Did the Houston Astros Cheat in 2017? Everything to Know About a Dark Moment in MLB History (Essentially Sports, 2023-10-22T21:30:14+00:00)
Luhnow’s phone showed signs of deleted information. Jan. 13, 2020, was when MLB suspended the disgraced GM and the skipper for the 2020 season. Only for Jim
Crane to fire both a few hours later. But for many, the solution was severely lacking.

Why Were the Astros Allowed to Keep Their World Series Trophy?

In the words of MLB Commissioner Robert Manfred, “Once you go down that road of changing what happens on the field, I just don’t know how you decide where you
stop.” While many thought taking away the World Series win, essentially nullifying 2017, was the best way to move forward, MLB thought otherwise. The
Coomissioner’s Trophy was still legitimate in the eyes of, well, the Commissioner.
</item 71>

<item 72>
Michigan’s Rivals Fume About Sign-Stealing: ‘This Was Damn Near Espionage’ (The New York Times, 2023-11-03T22:07:31+00:00)
Stalions bought tickets to games in at least seven Big Ten stadiums before those teams played the Wolverines over the past three seasons, including the 2023
season, sources told The Athletic last month. Purchasing the tickets is not a violation of NCAA rules, but using them to scout and record other teams would
violate the rules, prohibiting in-person, on-campus scouting and the audio or video recording of signals.

“They aren’t allegations. It happened,” Purdue coach Ryan Walters said Thursday night on his radio show ahead of Saturday’s game against the Wolverines.
“There’s video evidence. There’s ticket purchases and sales you can track back. We know for a fact they were at a number of our games. We’ve had to teach our
guys a new language.”

On Monday, coach Jim Harbaugh met with reporters and said “the people that know us the most think the most of us” as Michigan faces the NCAA investigation. He
has denied knowledge of the alleged scouting.
</item 72>

<item 73>
Michigan sign-stealing scandal news: Latest updates on possible Jim Harbaugh suspension from NCAA cheating investigation (Sporting News,
2023-11-06T23:58:04+00:00)
Michigan players and coaches arrive. No Jim Harbaugh on the bus. pic.twitter.com/wnGzqAxwcp — Adam Rittenberg (@ESPNRittenberg) November 11, 2023

Michigan offensive coordinator Sherrone Moore will serve as acting head coach if Harbaugh is not available.

Friday, Nov. 10

Michigan files temporary restraining order against the Big Ten

In an effort to delay Harbaugh's suspension, the university filed a temporary restraining order in Washtenaw County Trial Court on Friday night. If granted, the
suspension would be delayed, but the clock is ticking before the Wolverines' game vs. Penn State on Saturday.

Michigan's temporary restraining order request has been filed.



Plaintiffs: Jim Harbaugh, U-M regents

Defendants: Big Ten, Tony Petitti



Judicial officer: Timothy P. Connors
</item 73>

<item 74>
Bryce Harper 'Atta Boy' controversy, explained: Why Orlando Arcia's comments led to backlash, Alanna Rizzo rant (Sporting News, 2023-10-13T21:58:11+00:00)
When the Phillies beat the Braves in Game 4 to eliminate Atlanta and advance to the NLCS, players in the Philadelphia clubhouse celebrated by repeating, "Atta
boy Harper."

While the comments became a spark for the Phillies, it also became a subject of controversy. Players, including the Blue Jays' Kevin Gausman, decried the
situation as a mistake by Mintz to report the comments, which were said out loud in the locker room rather than said in an interview setting. Others defended
the situation, saying that anything in the clubhouse is open to be reported on.

MORE: Why Arcia's comments motivated Harper after NLDS Game 2

But perhaps the biggest backlash came from MLB Network's Alanna Rizzo, who blasted Mintz on "High Heat" for reporting on the incident. Following a BBWAA
statement, Rizzo both publicly and privately apologized to Mintz for her reaction.

Here's what you need to know about the controversy.
</item 74>

<item 75>
Michigan didn't need Jim Harbaugh against Penn State, but Ohio State is a different game (Sporting News, 2023-11-11T21:32:44+00:00)
2 now, and the debate with the Buckeyes and No. 1 Georgia is going to perk up more than ever. They are the three best teams in the country, but only one has
uncertainty with the head coach.

Whether they admit it or not, Michigan still needs Harbaugh for those moments, especially in two weeks.

The reality is he might not be present for that one either.
</item 75>

<item 76>
Bryce Harper 'Atta Boy' controversy, explained: Why Orlando Arcia's comments led to backlash, Alanna Rizzo rant (Sporting News, 2023-10-13T21:58:11+00:00)
After the game, Mintz wrote a well-reported story on how the Braves pulled off the improbable comeback, coming back from down four runs at one point, to win the
game, and deep in the article, wrote the following:

Kevin Pillar's young son, sporting a custom "Money Mike" headband, waddled over to his favorite center fielder for a fist bump. Third-string catcher Chadwick
Tromp strutted by Harris' locker offering a congratulatory "Mike is him!" All while Braves shortstop Orlando Arcia cackled emphatically about Harper's
misfortune, bellowing "ha-ha, atta-boy, Harper!" repeatedly as reporters circled the room.

As Mintz pointed out, he was not the only one to allude to the celebration from Arcia. The Washington Post's Chelsea Janes also wrote about it, saying that one
of Harris' teammates was yelling, "Atta boy Harper."
</item 76>

<item 77>
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
</item 77>

<item 78>
Michigan conquers Ohio State again, but the story is far from over for 2023 Wolverines (Sporting News, 2023-11-25T23:20:56+00:00)
It will continue at least into the Big Ten championship game against No. 17 Iowa. You know that off-the-field story by now. That is the one where this team
continues to overcome obstacles – self-inflicted and otherwise. The on-field story is a withering march through the Big Ten toward a third straight College
Football Playoff berth.

It could have ended with Ohio State. Jim Harbaugh's three-game suspension from the Big Ten as a result of the illegal in-person scouting and sign-stealing
operation dominated headlines more than the Wolverines dominated opponents in the second half of the season. Ohio State had not lost three in a row to Michigan
since 1995-97.

The Buckeyes were due, right?

Michigan acting coach Sherrone Moore had other plans. He beat Ohio State coach Ryan Day. Cornerback Will Johnson set the tone by intercepting Ohio State
quarterback Kyle McCord with 7:32 left in the first quarter.
</item 78>

<item 79>
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
</item 79>

<item 80>
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
</item 80>

<item 81>
How Did the Houston Astros Cheat in 2017? Everything to Know About a Dark Moment in MLB History (Essentially Sports, 2023-10-22T21:30:14+00:00)
But the hornet’s nest wasn’t broken open until next year, when The Athletic‘s Ken Rosenthal and Evan Drellich reported on it. After that, all hell broke loose.

What Is Sign Stealing and How Did the Astros Manage to Do It?

Powers of observation are as strong a skill as pitching or hitting and they’re perfectly legal in baseball. A batter can sometimes figure out from a pitcher’s
unconscious movements, or ‘tells’, what kind of ball is coming their way. At any given moment, non-verbal signs are flying around between a dugout and the field
as a way for the players to communicate with a manager or coach. But as long as it’s done without extra help, this sign stealing is, in fact, allowed. What
Houston did was not.

The Stros installed cameras in center field at Minute Maid Park to record the opposing team’s catcher and pitcher’s signs. Cameras in a ballpark are ideally for
challenging calls, but what was one more purpose?
</item 81>

<item 82>
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
</item 82>

<item 83>
Michigan sign-stealing scandal news: Latest updates on possible Jim Harbaugh suspension from NCAA cheating investigation (Sporting News,
2023-11-06T23:58:04+00:00)
Here is Michigan’s 10-page letter to the Big Ten responding to the sportsmanship investigation, reporting with @RossDellenger 1/3 pic.twitter.com/P0oYgIbS67 —
Dan Wetzel (@DanWetzel) November 9, 2023

No timetable for Big Ten decision

The Big Ten is reviewing the "lengthy" official response from Michigan and Jim Harbaugh and it's unclear when a potential ruling might come down, says ESPN's
Pete Thamel. His colleague Adam Rittenberg reported Wednesday that a decision was expected by Thursday.

Source: The Big Ten is reviewing the lengthy written responses from both Michigan and Jim Harbaugh. There’s no indication of any timetable on a decision on
potential Big Ten discipline in the case of Michigan’s alleged illegal signal stealing.
</item 83>

<item 84>
Michigan didn't need Jim Harbaugh against Penn State, but Ohio State is a different game (Sporting News, 2023-11-11T21:32:44+00:00)
I did this for you, for this university, the president, our AD. We've got the best players, best university, and the best alumni in the country. I love these
guys. These f------ guys right here. These guys right here. These guys did it."

MORE: Sherrone Moore gives tearful interview after Penn State win

Whoa, let's add the FCC standards to the Wolverines' list of potential violations. If Michigan (10-0, 7-0 Big Ten) was not the college football heel heading
into the final two weeks of the regular season, then they are public enemy No. 1 now. It might be “Michigan vs. Everybody,” and everybody has the same question.

Will Harbaugh be on the field before the end of the regular season?
</item 84>

<item 85>
Michigan sign-stealing scandal winners and losers, from Jim Harbaugh to Tony Petitti (Sporting News, 2023-11-11T22:50:13+00:00)
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
</item 85>

<item 86>
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
</item 86>

<item 87>
How Did the Houston Astros Cheat in 2017? Everything to Know About a Dark Moment in MLB History (Essentially Sports, 2023-10-22T21:30:14+00:00)
And there began the story of the rise and decline of one of baseball’s biggest controversies.

The Story Behind the Cheating Scandal That Rocked the 2017 MLB Season

ADVERTISEMENT Article continues below this ad

Jeff Luhnow had always been more about the numbers than people. Wanting to take out the humanity factor from baseball, he was almost an outcast in the sport
when Jim Crane hired him. But his radical approach had worked in St. Louis, giving them two championships in six years. Why not the Bayou City next?

The Houston Astros had not seen a World Series title yet and hadn’t even made the playoffs since 2005. Apparently desperate for any measures, they resorted to
cheating. Perhaps the biggest surprise was that Luhnow didn’t consider his methods to be dishonest.

As it turned out, the Astros had stolen signs throughout the 2017 season and the beginning of 2018.
</item 87>

<item 88>
Bryce Harper 'Atta Boy' controversy, explained: Why Orlando Arcia's comments led to backlash, Alanna Rizzo rant (Sporting News, 2023-10-13T21:58:11+00:00)
I got caught up in the heat of the moment."

"What I said was completely off base and I've had an interesting 24 hours and I've learned a lot more and I apologize to Jake. I talked to him already, I
apologized to him privately and I also apologize to him publicly," Rizzo said. "They do a good job in the clubhouse and I was completely inaccurate in my
assessment of what went down there."

Alanna Rizzo gave "a very heartfelt apology to Jake Mintz and the entire BBWAA" pic.twitter.com/rT9tjSWLY3 — Awful Announcing (@awfulannouncing) October 13,
2023

Mintz, though calling her initial comments "incredibly hurtful/uncalled for," wrote on X saying that he was "glad she apologized," and asked people not to toss
hate to her.
</item 88>

<item 89>
Michigan sign-stealing scandal news: Latest updates on possible Jim Harbaugh suspension from NCAA cheating investigation (Sporting News,
2023-11-06T23:58:04+00:00)
We are unable to comment further regarding this personnel matter." — Jesse Dougherty (@dougherty_jesse) November 4, 2023

In a statement, Stalions' lawyer said that his client hoped to avoid creating more distractions for Michigan and denied that any Wolverines staffer "told anyone
to break any rules."

MORE: Who is Connor Stalions?

Stalions is accused of buying numerous tickets to scout future opponents in-person, which is against NCAA rules. The NCAA is also investigating the possibility
that Stalions appeared on Central Michigan's sideline in Chippewas gear for a game against Michigan State.

Stalions was suspended by Michigan when the news of his alleged involvement first broke.
</item 89>

<item 90>
Michigan conquers Ohio State again, but the story is far from over for 2023 Wolverines (Sporting News, 2023-11-25T23:20:56+00:00)
ANN ARBOR, Mich. – Michigan quarterback J.J. McCarthy heard Zak Zinter scream. McCarthy saw tackle Karsen Barnhart holding Zinter's foot in the air. This was a
gruesome scene at Michigan Stadium on Saturday.

No. 3 Michigan lost their senior All-American guard to a broken tibia and fibula with 2:19 left in the third quarter of a tie game with No. 2 Ohio State. Zinter
was carted off in what could have been a turning point in the other direction given his value on the interior line.

"It was a sight that I wouldn't wish upon anyone to see," McCarthy said. "At that moment, seeing the look in everyone's eyes, seeing them rally together, it was
spiritual, honestly. It was a different drive that hit on everybody."

On the next play, senior running back Blake Corum finally broke through the second level of the Buckeyes' defense for a 22-yard go-ahead touchdown.
</item 90>

<item 91>
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
</item 91>

<item 92>
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
</item 92>

<item 93>
Michigan sign-stealing scandal news: Latest updates on possible Jim Harbaugh suspension from NCAA cheating investigation (Sporting News,
2023-11-06T23:58:04+00:00)
pic.twitter.com/Qm9iehYVTR — Pete Thamel (@PeteThamel) November 11, 2023

Harbaugh will not be on the sidelines vs. Penn State

Michigan filed a temporary restraining order against the Big Ten on Friday night, hoping to block Harbaugh's suspension. But there will not be a ruling before
the Wolverines' game vs. Penn State, with the hearing scheduled for Friday, Nov. 17.

No ruling on Michigan’s TRO filing today, so Jim Harbaugh will not coach vs. Penn State.



Statement from the school: “We look forward to presenting our case next week where we intend to demonstrate that the Big Ten has not acted legally or fairly.”
</item 93>

<item 94>
How Did the Houston Astros Cheat in 2017? Everything to Know About a Dark Moment in MLB History (Essentially Sports, 2023-10-22T21:30:14+00:00)
The Coomissioner’s Trophy was still legitimate in the eyes of, well, the Commissioner.

“Cheaters” Houston Astros Suddenly Gain Momentum as College Baseball Team Stirs Up Controversial Trash Can Past Along With Thousands

By the time the controversy was uncovered, the Astros were well into establishing a streak of successful years. Manfred called the process of stripping the Fall
Classic title “difficult and impractical.” Already unpopular among the masses, he made things worse for himself when he called the trophy a “piece of metal.”
Although he apologized later, the team kept the championship victory.

Why Were the Players Not Penalized?

It was an arduous process, investigating an entire organization. While the players were undoubtedly the ones who had put the tactics into play, the Astros staff
suffered the heaviest consequences. The ballplayers were instead granted full immunity in exchange for cooperating with the investigation. Indeed, who would
refuse such an offer?
</item 94>

<item 95>
Bryce Harper 'Atta Boy' controversy, explained: Why Orlando Arcia's comments led to backlash, Alanna Rizzo rant (Sporting News, 2023-10-13T21:58:11+00:00)
"That's taking away from true reporters and true journalists. Where were you on the random Tuesday in April in Cincinnati when this team was playing in
Cincinnati? Where were you on a rain delay, and you have to sit in the press box for all of this time? It's ridiculous."

"And then some jackoff comes in at the end of the season that gets a credential, God only knows why. And the clubhouse is a sacred space." Alanna Rizzo went off
on Jake Mintz's clubhouse reporting on MLBN's "High Heat" Thursday. pic.twitter.com/6dooqeLPnh — Awful Announcing (@awfulannouncing) October 12, 2023

Rizzo's comments immediately drew heat from the baseball journalism world, including from Janes, who posted a thread defending Mintz. Janes reported that
cameras and recorders were recording, and that Arcia yelled the comments while they were recording and with more than a dozen reporters around.
</item 95>

<item 96>
Michigan didn't need Jim Harbaugh against Penn State, but Ohio State is a different game (Sporting News, 2023-11-11T21:32:44+00:00)
Who is the good guy? Who is the villain? Ohio State and Michigan locked in their answers a long time ago. More than 17 million people watched last year's
matchup – a 45-23 Michigan victory where Stalions' involvement continues to be scrutinized through a double-barrel Big Ten and NCAA investigation. Big Ten
commissioner Tony Petitti's ruling sent a shockwave through the conference Friday, mainly because the third game means Harbaugh will not be on the sideline for
The Game. Early forecasts say 20 million will tune in for The Game this year.

Will Harbaugh really not be there? That is for the courts to decide, but Moore’s comments suggest Harbaugh, president Santa Ono and athletic director Warde
Manuel are on a united front; consequences be damned in 2024 and beyond.

In the short term? The College Football Playoff committee could slide Michigan up to No. 2 now, and the debate with the Buckeyes and No.
</item 96>

<item 97>
Michigan’s Rivals Fume About Sign-Stealing: ‘This Was Damn Near Espionage’ (The New York Times, 2023-11-03T22:07:31+00:00)
Earlier this week, Central Michigan said it is investigating whether suspended Michigan staffer Connor Stalions was on the CMU sideline during the Chippewas’
Sept. 1 game at Michigan State. Screenshots of a person who looks similar to Stalions began circulating online Monday night, and The Athletic obtained more
photos of the person on the sideline Tuesday.

Central Michigan is investigating if the person on the right here is Connor Stalions on the CMU sideline at Michigan State earlier this season. AD Amy Folan:
“We became aware of these photos late yesterday and we are in the process of determining the facts surrounding them." pic.twitter.com/ncazAghBbS — Chris Vannini
(@ChrisVannini) October 31, 2023

Stalions, who was suspended with pay by Michigan on Oct. 20, is at the center of the NCAA’s investigation into the alleged scouting and sign-stealing scheme.
</item 97>

<item 98>
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
</item 98>

<item 99>
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
</item 99>

<item 100>
Michigan conquers Ohio State again, but the story is far from over for 2023 Wolverines (Sporting News, 2023-11-25T23:20:56+00:00)
"For the big boys, for Blake, for myself, that's music to our ears that he has the confidence in us to go get the extra yard or two yards. That's who we are."

Michigan played to win the game. Ohio State, not so much. The Buckeyes settled for a 52-yard field goal late in the first half, which Jayden Fielding missed
wide left.

Even when Ohio State tied the score at 17-17 with 6:08 left in the third quarter on an up-tempo drive that featured eight straight run plays and a TreVeyon
Henderson touchdown, Michigan answered. The Wolverines scored on every drive in the second half – three field goals and a touchdown.

"It goes back to how they prepare, how they attack things and how they work," Moore said. "I would be doing them a disservice if I went out there and was
conservative."

Say what you will about the adults who were involved in the in-person scouting scandal.
</item 100>
=== END ITEM id=fc668495ce6b66f5 ===
