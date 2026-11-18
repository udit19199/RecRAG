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

=== ITEM id=ec23ee93fb3f5c1a ===
Gold sentences and facts:
Sentence 0: For the past two weeks, coaches across college football have been riveted by the alleged Michigan sign-stealing scheme, but inside the Big Ten, the topic has been more than just a curiosity.
  Fact 0.0: Coaches across college football have been riveted by the alleged Michigan sign-stealing scheme for the past two weeks.
  Fact 0.1: Inside the Big Ten, the alleged Michigan sign-stealing scheme has been more than just a curiosity.
Sentence 1: While it is almost impossible to quantify how much the alleged in-person scouting and sign-stealing would have helped the Wolverines over the past two seasons, it will be a tough stain to erase knowing those violations are now on the Big Ten record.
  Fact 1.0: It is almost impossible to quantify how much the alleged in-person scouting and sign-stealing would have helped the Wolverines over the past two seasons.
  Fact 1.1: The alleged in-person scouting and sign-stealing violations are now on the Big Ten record.
  Fact 1.2: Having the alleged violations on the Big Ten record will be a tough stain to erase.

Retrieved items:
<item 1>
Michigan’s Rivals Fume About Sign-Stealing: ‘This Was Damn Near Espionage’ (The New York Times, 2023-11-03T22:07:31+00:00)
To know exactly what the other team is doing, Michigan might as well have been playing with 15 guys on the field,” he said. “What’s the message the Big Ten is
sending now by doing nothing? Win now, pay later? We might as well just send people to (scout) their practices and their games. It doesn’t encourage anybody to
follow the rules. It’s just telling them to do the opposite and say, f— it.”

GO DEEPER Michigan should be punished, say 94% of CFB coaches in our poll. What else did it reveal?

The NCAA is investigating Michigan’s football program amid allegations that the Wolverines used illegal in-person scouting and the recording of signals to steal
signs this season. Ahead of the Michigan-Michigan State game on Oct. 21, the Big Ten approached MSU and said it was made aware of “credible evidence” regarding
the sign-stealing allegations. The Big Ten said it would monitor the NCAA’s investigation into Michigan.

Graph facts:
- He’s Hockey’s Brightest Young Star. This Is What Makes His Shot So Special. -PUBLISHED_BY-> The New York Times
- How U.S. Women’s Soccer Team Lured Emma Hayes Away From Chelsea -PUBLISHED_BY-> The New York Times
- After a Stunning Snub, College Football Is Facing an Identity Crisis -PUBLISHED_BY-> The New York Times
- N.F.L. Executives on the Biggest Questions Facing Super Bowl Contenders -PUBLISHED_BY-> The New York Times
- Which M.L.B. Playoff Team Has the Most Daunting Home Field? We Ranked Them. -PUBLISHED_BY-> The New York Times

Entities: The New York Times, Michigan, Big Ten, Michigan State, Michigan-Michigan State game, sign-stealing allegations, illegal in-person scouting, recording
of signals, NCAA, Michigan’s Rivals Fume About Sign-Stealing: ‘This Was Damn Near Espionage’
</item 1>

<item 2>
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

Graph facts:
- He’s Hockey’s Brightest Young Star. This Is What Makes His Shot So Special. -PUBLISHED_BY-> The New York Times
- How U.S. Women’s Soccer Team Lured Emma Hayes Away From Chelsea -PUBLISHED_BY-> The New York Times
- After a Stunning Snub, College Football Is Facing an Identity Crisis -PUBLISHED_BY-> The New York Times
- N.F.L. Executives on the Biggest Questions Facing Super Bowl Contenders -PUBLISHED_BY-> The New York Times
- Which M.L.B. Playoff Team Has the Most Daunting Home Field? We Ranked Them. -PUBLISHED_BY-> The New York Times

Entities: The New York Times, Michigan, Big Ten, SEC, sign-stealing, NCAA investigation, NCAA, Michigan’s Rivals Fume About Sign-Stealing: ‘This Was Damn Near
Espionage’
</item 2>

<item 3>
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

Graph facts:
- USC basketball futures odds to win Pac-12, NCAA tourney ahead of Bronny James collegiate debut -MENTIONS-> Michigan
- After a Stunning Snub, College Football Is Facing an Identity Crisis -MENTIONS-> Michigan
- 2024 NFL Mock Draft: Bengals' Joe Burrow gets another playmaker; four QBs go in Round 1, but Pats opt for WR -MENTIONS-> Michigan
- Michigan sign-stealing scandal news: Latest updates on possible Jim Harbaugh suspension from NCAA cheating investigation -MENTIONS-> Michigan
- Michigan conquers Ohio State again, but the story is far from over for 2023 Wolverines -MENTIONS-> Michigan

Entities: Michigan, college football, College Football Playoff, Michigan sign-stealing scandal winners and losers, from Jim Harbaugh to Tony Petitti, Greg
Sankey, Big Ten, in-person scouting scandal, ESPN, USC, Jim Harbaugh, Ohio State, UCLA, Sporting News, Washington, SEC
</item 3>

<item 4>
Michigan’s Rivals Fume About Sign-Stealing: ‘This Was Damn Near Espionage’ (The New York Times, 2023-11-03T22:07:31+00:00)
… No doubt this all has had a profound effect.

“This guy’s being investigated for three different things now between the (alleged) illegal signal stealing, the (alleged) illegal recruiting during COVID and
that investigation into the offensive coordinator and alleged computer hacking. There are guys (on that call) who could lose jobs, and then there’s this guy
over here (Harbaugh) who is gonna get a new, bigger contract now, and they won’t do anything about him.”

Asked to describe the tone of the coaches’ sentiment expressed to Petitti, another Big Ten coach called it “angry” — particularly at the Big Ten’s lack of
action, or even apparent interest in taking any.

“Everybody’s upset,” that coach told The Athletic. “Why is nothing being done? We want to know, what else do you need to know to take action? We (the Big Ten
head coaches) want something done now.

Graph facts:
- Which M.L.B. Playoff Team Has the Most Daunting Home Field? We Ranked Them. -MENTIONS-> The Athletic
- Michigan sign-stealing scandal winners and losers, from Jim Harbaugh to Tony Petitti -MENTIONS-> The Athletic
- Browns survive Bears Hail Mary on wild, near-catch by Darnell Mooney: 'Damn near s— myself' -MENTIONS-> The Athletic
- He’s Hockey’s Brightest Young Star. This Is What Makes His Shot So Special. -MENTIONS-> The Athletic
- How Did the Houston Astros Cheat in 2017? Everything to Know About a Dark Moment in MLB History -MENTIONS-> The Athletic

Entities: The Athletic, The New York Times, Michigan, Big Ten, Petitti, Harbaugh, alleged illegal signal stealing, alleged illegal recruiting during COVID,
alleged computer hacking, investigation into the offensive coordinator, Michigan’s Rivals Fume About Sign-Stealing: ‘This Was Damn Near Espionage’
</item 4>

<item 5>
Michigan’s Rivals Fume About Sign-Stealing: ‘This Was Damn Near Espionage’ (The New York Times, 2023-11-03T22:07:31+00:00)
We (the Big Ten head coaches) want something done now. I don’t think people understand the advantage that what they’re (allegedly) doing gives you. People
think, ‘OK, now that everybody knows, we all can just move on.’ Like, ‘now, it’s fair.’ Well, no, it isn’t. Not at all. This changes the way you operate. A lot
of teams have been doing things a certain way for years. Now, it’s forcing you to teach your players a whole new way to communicate just for them. People think
that this is just advanced scouting. This was damn near espionage.”

A third Big Ten head coach told The Athletic that this is “one of the most egregious breaches in the spirit of the game” he’s ever heard of.

“They (Michigan) have been manipulating the game and cheating the game for two-and-a-half years.

Graph facts:
- Which M.L.B. Playoff Team Has the Most Daunting Home Field? We Ranked Them. -MENTIONS-> The Athletic
- Michigan sign-stealing scandal winners and losers, from Jim Harbaugh to Tony Petitti -MENTIONS-> The Athletic
- Browns survive Bears Hail Mary on wild, near-catch by Darnell Mooney: 'Damn near s— myself' -MENTIONS-> The Athletic
- He’s Hockey’s Brightest Young Star. This Is What Makes His Shot So Special. -MENTIONS-> The Athletic
- How Did the Houston Astros Cheat in 2017? Everything to Know About a Dark Moment in MLB History -MENTIONS-> The Athletic

Entities: The Athletic, The New York Times, Michigan, Big Ten head coaches, A third Big Ten head coach, sign-stealing, cheating, espionage, advanced scouting,
Michigan’s Rivals Fume About Sign-Stealing: ‘This Was Damn Near Espionage’
</item 5>

<item 6>
Graph query row:
article: Michigan sign-stealing scandal winners and losers, from Jim Harbaugh to Tony Petitti
publication_date: 2023-11-11T22:50:13+00:00
connections: [{'name': 'Sporting News', 'relationship': 'PUBLISHED_BY'}, {'name': 'NFL', 'relationship': 'MENTIONS'}, {'name': 'The Athletic', 'relationship':
'MENTIONS'}, {'name': 'Michigan', 'relationship': 'MENTIONS'}, {'name': 'Connor Stallions', 'relationship': 'MENTIONS'}, {'name': 'in-person scouting',
'relationship': 'MENTIONS'}, {'name': 'sign stealing', 'relationship': 'MENTIONS'}, {'name': 'scandal', 'relationship': 'MENTIONS'}, {'name': 'showdown against
Penn State', 'relationship': 'MENTIONS'}, {'name': 'Nicole Auerbach', 'relationship': 'MENTIONS'}, {'name': 'sign-stealing scandal', 'relationship':
'MENTIONS'}, {'name': 'Big Two', 'relationship': 'MENTIONS'}, {'name': 'DeCourcy', 'relationship': 'MENTIONS'}, {'name': 'Sports Media Watch', 'relationship':
'MENTIONS'}, {'name': 'Michigan sign-stealing scandal', 'relationship': 'MENTIONS'}, {'name': 'Harbaugh suspension', 'relationship': 'MENTIONS'}, {'name':
"Friday's announcement", 'relationship': 'MENTIONS'}, {'name': 'sportsmanship', 'relationship': 'MENTIONS'}, {'name': 'television ratings', 'relationship':
'MENTIONS'}, {'name': 'college football', 'relationship': 'MENTIONS'}, {'name': 'wrongdoing', 'relationship': 'MENTIONS'}, {'name': 'legal response',
'relationship': 'MENTIONS'}, {'name': 'Penn State', 'relationship': 'MENTIONS'}, {'name': 'Day', 'relationship': 'MENTIONS'}, {'name': 'College Football
Playoff', 'relationship': 'MENTIONS'}, {'name': 'national championship', 'relationship': 'MENTIONS'}, {'name': 'WWE', 'relationship': 'MENTIONS'}, {'name':
'CBSSports.com', 'relationship': 'MENTIONS'}, {'name': 'NCAA rules committee', 'relationship': 'MENTIONS'}, {'name': 'coach-to-player helmet communication',
'relationship': 'MENTIONS'}, {'name': 'bowl season', 'relationship': 'MENTIONS'}, {'name': 'Nittany Lions', 'relationship': 'MENTIONS'}, {'name': 'Saturday’s
victory against the Nittany Lions', 'relationship': 'MENTIONS'}, {'name': 'Beaver Stadium', 'relationship': 'MENTIONS'}, {'name': 'Sherrone Moore',
'relationship': 'MENTIONS'}, {'name': 'defeating Penn State', 'relationship': 'MENTIONS'}, {'name': 'NCAA investigation into whether Michigan was involved with
in-person scouting', 'relationship': 'MENTIONS'}, {'name': 'CFP', 'relationship': 'MENTIONS'}, {'name': 'Harvey Updyke', 'relationship': 'MENTIONS'}, {'name':
'Nevin Shapiro', 'relationship': 'MENTIONS'}, {'name': 'ESPN.com', 'relationship': 'MENTIONS'}, {'name': "Chippewas' opener against Michigan State",
'relationship': 'MENTIONS'}, {'name': 'Stalion', 'relationship': 'MENTIONS'}, {'name': 'Michigan fans', 'relationship': 'MENTIONS'}, {'name': 'Maryland',
'relationship': 'MENTIONS'}, {'name': 'temporary restraining order hearing', 'relationship': 'MENTIONS'}, {'name': 'rule-breaking', 'relationship': 'MENTIONS'},
{'name': '2020 season', 'relationship': 'MENTIONS'}, {'name': 'Pat Fitzgerald', 'relationship': 'MENTIONS'}, {'name': 'Northwestern', 'relationship':
'MENTIONS'}, {'name': 'Brenda Tracy', 'relationship': 'MENTIONS'}, {'name': 'Mel Tucker', 'relationship': 'MENTIONS'}, {'name': 'Rutgers', 'relationship':
'MENTIONS'}, {'name': 'illegal sports wagering scandal', 'relationship': 'MENTIONS'}, {'name': 'hazing investigation', 'relationship': 'MENTIONS'}, {'name':
'wrongful termination', 'relationship': 'MENTIONS'}, {'name': 'sexual assault awareness', 'relationship': 'MENTIONS'}, {'name': "leaking Michigan's signs",
'relationship': 'MENTIONS'}, {'name': 'Greg Sankey', 'relationship': 'MENTIONS'}, {'name': 'Big Ten', 'relationship': 'MENTIONS'}, {'name': 'in-person scouting
scandal', 'relationship': 'MENTIONS'}, {'name': 'Santa Ono', 'relationship': 'MENTIONS'}, {'name': 'J.J. McCarthy', 'relationship': 'MENTIONS'}, {'name': 'Blake
Corum', 'relationship': 'MENTIONS'}, {'name': 'Ric Flair', 'relationship': 'MENTIONS'}, {'name': 'Terps', 'relationship': 'MENTIONS'}, {'name': 'Buckeyes',
'relationship': 'MENTIONS'}, {'name': 'ESPN', 'relationship': 'MENTIONS'}, {'name': 'USC', 'relationship': 'MENTIONS'}, {'name': 'MLB Network', 'relationship':
'MENTIONS'}, {'name': 'Stalions', 'relationship': 'MENTIONS'}, {'name': 'Jim Harbaugh', 'relationship': 'MENTIONS'}, {'name': 'Warde Manuel', 'relationship':
'MENTIONS'}, {'name': 'game against Penn State', 'relationship': 'MENTIONS'}, {'name': 'Connor Stalions', 'relationship': 'MENTIONS'}, {'name': 'the game',
'relationship': 'MENTIONS'}, {'name': 'Big Ten championship game', 'relationship': 'MENTIONS'}, {'name': 'Ohio State', 'relationship': 'MENTIONS'}, {'name':
'UCLA', 'relationship': 'MENTIONS'}, {'name': 'Colorado', 'relationship': 'MENTIONS'}, {'name': 'Iowa', 'relationship': 'MENTIONS'}, {'name': 'Washington',
'relationship': 'MENTIONS'}, {'name': 'Michigan-Penn State game', 'relationship': 'MENTIONS'}, {'name': 'SEC', 'relationship': 'MENTIONS'}, {'name': 'Notre
Dame', 'relationship': 'MENTIONS'}, {'name': 'Oregon', 'relationship': 'MENTIONS'}, {'name': 'Michigan State', 'relationship': 'MENTIONS'}, {'name': 'Big Ten
East', 'relationship': 'MENTIONS'}, {'name': 'Nielsen', 'relationship': 'MENTIONS'}, {'name': 'Tony Petitti', 'relationship': 'MENTIONS'}, {'name':
'sign-stealing', 'relationship': 'MENTIONS'}, {'name': 'NCAA investigation', 'relationship': 'MENTIONS'}, {'name': 'Central Michigan', 'relationship':
'MENTIONS'}, {'name': 'NCAA', 'relationship': 'MENTIONS'}, {'name': 'Purdue', 'relationship': 'MENTIONS'}, {'name': 'sexual misconduct', 'relationship':
'MENTIONS'}]
claims_and_passages: ['Michigan sign-stealing scandal winners and losers, from Jim Harbaugh to Tony Petitti (Sporting News, 2023-11-11T22:50:13+00:00)\nThe Big
Ten announced Michigan coach Jim Harbaugh will be prohibited from coaching on the sideline for the remainder of the regular season.\n\nAt least, in theory. The
Friday ruling is the next chapter in an NCAA investigation into Michigan for in-person scouting and sign stealing in a scandal that led to the resignation of
staffer Connor Stallions on Nov. 3. Harbaugh will be allowed to coach during the week but will not be able to be on the sideline. Michigan did not get a ruling
on a temporary restraining order ahead of the showdown against No. 10 Penn State.\n\nThe Wolverines still won 24-15 under interim coach Sherrone Moore, but the
fallout from the scandal continues to have a far-reaching impact on the Big Ten, Michigan and college football beyond this year\'s College Football Playoff
race.\n\nA look at the winners and losers from the scandal to date:\n\nMichigan scandal winners and losers\n\nWinners\n\nTony Petitti\n\nPetitti certainly did
not hit Michigan with a "slap on the wrist."', "Michigan sign-stealing scandal winners and losers, from Jim Harbaugh to Tony Petitti (Sporting News,
2023-11-11T22:50:13+00:00)\nThat ruling showed Petitti is not afraid to take on one of the Big Two within the conference. In a letter to Michigan, Petitti tried
to dispel those notions. Via The Athletic's Nicole Auerbach:\n\n“The Conference does not play favorites among its members, nor does it take actions towards its
members based on prejudgment or bias. Failing to act under the extraordinary circumstances here could lead other Conference members to reasonably conclude that
the conference has chosen to favor the University over all other members.”\n\nTony Petitti, in letter to Michigan, says that the school initially refused to
consent to the Big Ten obtaining the spreadsheet and other documents from the NCAA related to its investigation.\n\n\n\nAlso makes point to say he's not playing
favorites. pic.twitter.com/OgTvTNY1QI — Nicole Auerbach (@NicoleAuerbach) November 10, 2023\n\nMost speculation pinned a potential Harbaugh suspension at two
games knowing that the regular-season finale against No.", "Michigan sign-stealing scandal winners and losers, from Jim Harbaugh to Tony Petitti (Sporting News,
2023-11-11T22:50:13+00:00)\n1 Ohio State is a huge television draw and Petitti has a background in TV with the MLB Network. Petitti went heavy-handed instead,
maybe a little too heavy considering the pending legal response from Michigan.\n\nThis no doubt is risky on several fronts. The Big Ten made it clear that they
still have not linked Harbaugh to any wrongdoing and that this an institutional penalty based on the findings. The NCAA also will continue its investigation
into the Wolverines. The timing of Friday's announcement also was poor – less than 24 hours before the game against Penn State. Petitti upheld sportsmanship
within the conference, but it could lead to a long-term rift with Michigan.\n\nRegardless, Petitti did the right thing.\n\nDECOURCY: Michigan got off easy with
Harbaugh suspension\n\nOhio State-Michigan TV ratings\n\nWith or without Harbaugh, The Game will draw ridiculous television ratings.\n\nAccording to Sports
Media Watch, three college football games have drew a Nielsen rating of 5.0 or better this season.", "Michigan sign-stealing scandal winners and losers, from
Jim Harbaugh to Tony Petitti (Sporting News, 2023-11-11T22:50:13+00:00)\nPenn State-Ohio State (5.3), Colorado-Oregon (5.2) and Ohio State-Notre Dame (5.1) make
up that list. Michigan-Penn State should join that list on Saturday, but the intrigue around The Game this year could double all of those games.\n\nLast year's
matchup drew an 8.05 with 17.1 million viewers. Michigan won the last two meetings and has a chance to win three in a row in the rivalry for the first time
since 1995-97. Ohio State is No. 1 in the College Football Playoff rankings. Harbaugh (2-5) and Day (1-2) both have losing records in this rivalry matchup, and
the Big Ten East, CFP and perhaps national championship are on the line.\n\nWill this game draw a 10 Nielsen rating and 20 million viewers? It's not out of the
question, and there will be enough WWE-like promos leading up to this game to get both fan-bases frothing.", "Michigan sign-stealing scandal winners and losers,
from Jim Harbaugh to Tony Petitti (Sporting News, 2023-11-11T22:50:13+00:00)\nHelmet technology\n\nCBSSports.com reported on Oct. 20 that the NCAA rules
committee is giving teams the option of coach-to-player helmet communication to relay plays from the sideline during bowl season this year. Tablets will
inevitably follow, and here is the sliver of good that will come from this scandal.\n\nIt's about damn time. Death to sandwich-board playcalling, please. Bring
on those playsheets where coaches simply have to cover their mouth. This is one instance where emulating the NFL game is overdue.\n\nSherrone Moore\n\nIf Moore
is going to be a head coach, then Saturday’s victory against the Nittany Lions was a fitting audition. The Wolverines rushed for 227 yards – including 32
straight run plays – in grinding out a top-10 victory at one of the toughest environments in the FBS at Beaver Stadium.\n\nMoore was emotional - complete with
NSFW language – in the aftermath.\n\nSherrone Moore filled in for Jim Harbaugh as Michigan's coach today.", 'Michigan sign-stealing scandal winners and losers,
from Jim Harbaugh to Tony Petitti (Sporting News, 2023-11-11T22:50:13+00:00)\nSherrone Moore filled in for Jim Harbaugh as Michigan\'s coach today.\n\n\n\nHe
was emotional after defeating Penn State.\n\n\n\n"I want to thank Coach Harbaugh. ... I love the s— out of you man."\n\n\n\n🎥: @CFBONFOX
pic.twitter.com/NmXZWrGxCM — The Sporting News (@sportingnews) November 11, 2023\n\nMoore, however, will emerge as a leading candidate for some jobs as a
result, and that is a huge victory for one of the nation’s top assistants.\n\nLosers\n\nConnor Stalions\n\nStalions made a name for himself. He will be
remembered as the Michigan staffer as the center of this sign-stealing scandal. Stalions resigned on Nov. 3. The NCAA investigation whether Michigan was
involved with in-person scouting, which is prohibited by the NCAA, put Stalions into the college football consciousness.', "Michigan sign-stealing scandal
winners and losers, from Jim Harbaugh to Tony Petitti (Sporting News, 2023-11-11T22:50:13+00:00)\nAccording to ESPN.com, Stalions allegedly purchased tickets to
more than 35 games involving Big Ten and potential CFP opponents.\n\nPhotos also surfaced os a person who resembles Stalions on the Central Michigan sideline
during the Chippewas' opener against Michigan State on Sept. 1. Central Michigan has yet to confirm who that person was, another layer to a wild Stalions story
that has supplied endless meme and conspiracy material on social media.\n\nStalions might be the fall guy who saved Michigan from further wrath from the Big Ten
and NCAA or the reason the Wolverines' national championship season unraveled. That fate is yet to be determined, but at minimum he will be one of those
controversial names we'll never forget, like Harvey Updyke and Nevin Shapiro.\n\nJim Harbaugh\n\nOn one hand, Harbaugh could serve his second suspension within
one season and further NCAA punishment could be coming down the line.", "Michigan sign-stealing scandal winners and losers, from Jim Harbaugh to Tony Petitti
(Sporting News, 2023-11-11T22:50:13+00:00)\nOn the other, he could emerge as a martyr for Michigan fans given there is reportedly no proof yet he knew of
Stalion's rule-breaking.\n\nHarbaugh will have a temporary restraining order hearing on Nov. 17, according to ESPN.com. That will determine whether he returns
to the field for the final two games against Maryland and No. 1 Ohio State.\n\nIt's an inescapable fact that Harbaugh is 31-3 – and Michigan is 35-3 – since the
program bottomed out at the end of the 2020 season. While it is almost impossible to quantify how much the alleged in-person scouting and sign-stealing would
have helped the Wolverines over the past two seasons, it will be a tough stain to erase knowing those violations are now on the Big Ten record.\n\nWhat will
Harbaugh do now? That is almost impossible to predict given his combative personality. Would he sign a contract extension and continue to fight the NCAA? Will
Harbaugh finally take the parachute to the NFL?", "Michigan sign-stealing scandal winners and losers, from Jim Harbaugh to Tony Petitti (Sporting News,
2023-11-11T22:50:13+00:00)\nWill Harbaugh finally take the parachute to the NFL? Those are questions bound to come up at the end of the 2023 season.\n\nBENDER:
Michigan will need Jim Harbaugh against Ohio State\n\nBig Ten\n\nThis is the fourth football scandal in less than a calendar year.\n\nThree former Iowa football
players and a student manager were charged in an illegal sports wagering scandal. Several student-athletes face allegations of placing bets on games in which
they played in. Northwestern coach Pat Fitzgerald was fired on July 10 after a hazing investigation into the program. Fitzgerald is suing the university for
wrongful termination. Michigan State fired Mel Tucker on Sept. 27 after allegations of sexual misconduct from Brenda Tracy, a speaker on sexual assault
awareness.\n\nNow, Michigan has a scandal on the field, and they countered with allegations that Ohio State and Rutgers leaked Michigan's signs to Purdue in
last year's Big Ten championship game.", 'Michigan sign-stealing scandal winners and losers, from Jim Harbaugh to Tony Petitti (Sporting News,
2023-11-11T22:50:13+00:00)\nESPN reported that Big Ten coaches and athletic directors encouraged Big Ten commissioner Petitti to act last week, but that
resulted in action before the NCAA investigation concluded.\n\nDo you wonder what the SEC and commissioner Greg Sankey privately think about all this? For the
Big Ten, it is an ugly run of bad behavior on and off the field for a conference that is welcoming four new members next season in Oregon, UCLA, USC and
Washington.\n\nTo be determined\n\nMichigan\n\nMichigan is 10-0 and on track to meet No. 1 Ohio State for another blockbuster matchup in college football’s best
rivalry. On the field, the Wolverines remain on track for a third straight Big Ten championship and shot at the College Football Playoff.\n\nBut this is about
more than football.\n\nWill more penalties come from the Big Ten and NCAA as a result of the in-person scouting scandal that has not left the college football
consciousness this month? Will the Harbaugh-Petitti battle lead to an unsettling end for Michigan?', 'Michigan sign-stealing scandal winners and losers, from
Jim Harbaugh to Tony Petitti (Sporting News, 2023-11-11T22:50:13+00:00)\nWill the Harbaugh-Petitti battle lead to an unsettling end for Michigan? President
Santa Ono and athletic director Warde Manuel have voiced their support for Harbaugh, but that has been more about due process than admitting to any
wrongdoing.\n\nIt has thrust the Wolverines – a talented team led by J.J. McCarthy, Blake Corum and a defense that allows 7.5 points per game – into a villain
role, complete with visits from Ric Flair. That only works when you are winning. One loss, and all this could unravel, and the Terps and Buckeyes still remain
on the schedule.\n\nWill a national championship outweigh the stain on their reputation? The answer is almost always yes for the fans, but the program might
feel a different long-term result.']
</item 6>

<item 7>
Michigan’s Rivals Fume About Sign-Stealing: ‘This Was Damn Near Espionage’ (The New York Times, 2023-11-03T22:07:31+00:00)
“They’re reading the tea leaves and wondering why the Big Ten hasn’t done anything yet. Every week and every day that goes by, people are like, ‘Something’s
gotta give.’ It’s getting a little bit out of hand when you see him (allegedly) on the Central Michigan sideline. The playing field is not level right now. How
can you have a team that you know has a competitive advantage over you still being allowed to play? That’s what the coaches are grappling with.”

Advertisement

“It feels like (former commissioner) Kevin (Warren) taking over and then COVID,” the source continued. “Tony’s walking into this situation, and people are
calling for the league to make a statement before they have all the facts.”

Despite frustration from all corners of the conference, sources at four different Big Ten schools said they do not expect the conference to levy any sort of
punishment against Michigan before the season ends.

Graph facts:
- He’s Hockey’s Brightest Young Star. This Is What Makes His Shot So Special. -PUBLISHED_BY-> The New York Times
- How U.S. Women’s Soccer Team Lured Emma Hayes Away From Chelsea -PUBLISHED_BY-> The New York Times
- After a Stunning Snub, College Football Is Facing an Identity Crisis -PUBLISHED_BY-> The New York Times
- N.F.L. Executives on the Biggest Questions Facing Super Bowl Contenders -PUBLISHED_BY-> The New York Times
- Which M.L.B. Playoff Team Has the Most Daunting Home Field? We Ranked Them. -PUBLISHED_BY-> The New York Times

Entities: The New York Times, Michigan, Big Ten, Covid, sign-stealing, Kevin Warren, Tony, Central Michigan, Michigan’s Rivals Fume About Sign-Stealing: ‘This
Was Damn Near Espionage’
</item 7>

<item 8>
Michigan’s Rivals Fume About Sign-Stealing: ‘This Was Damn Near Espionage’ (The New York Times, 2023-11-03T22:07:31+00:00)
For the past two weeks, coaches across college football have been riveted by the alleged Michigan sign-stealing scheme, but inside the Big Ten, the topic has
been more than just a curiosity. On Wednesday’s Big Ten coaches video call with commissioner Tony Petitti, and after Jim Harbaugh left the call, that
frustration was voiced loud and clear, according to conference coaches, who said they don’t feel like the new Big Ten commissioner is “motivated” to do anything
about the Wolverines.

Advertisement

“There is just a ton of frustration,” a Big Ten coach told The Athletic on Thursday morning. “Look at Jim Harbaugh’s record before this started. The guy was on
the hot seat before 2021, and now he’s like the king of college football. … No doubt this all has had a profound effect.

Graph facts:
- Which M.L.B. Playoff Team Has the Most Daunting Home Field? We Ranked Them. -MENTIONS-> The Athletic
- Michigan sign-stealing scandal winners and losers, from Jim Harbaugh to Tony Petitti -MENTIONS-> The Athletic
- Browns survive Bears Hail Mary on wild, near-catch by Darnell Mooney: 'Damn near s— myself' -MENTIONS-> The Athletic
- He’s Hockey’s Brightest Young Star. This Is What Makes His Shot So Special. -MENTIONS-> The Athletic
- How Did the Houston Astros Cheat in 2017? Everything to Know About a Dark Moment in MLB History -MENTIONS-> The Athletic

Entities: The Athletic, The New York Times, Michigan, college football, Big Ten, Jim Harbaugh, Tony Petitti, sign-stealing scheme, Big Ten coaches video call,
Michigan’s Rivals Fume About Sign-Stealing: ‘This Was Damn Near Espionage’
</item 8>

<item 9>
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

Graph facts:
- Which M.L.B. Playoff Team Has the Most Daunting Home Field? We Ranked Them. -MENTIONS-> The Athletic
- Michigan sign-stealing scandal winners and losers, from Jim Harbaugh to Tony Petitti -MENTIONS-> The Athletic
- Browns survive Bears Hail Mary on wild, near-catch by Darnell Mooney: 'Damn near s— myself' -MENTIONS-> The Athletic
- He’s Hockey’s Brightest Young Star. This Is What Makes His Shot So Special. -MENTIONS-> The Athletic
- How Did the Houston Astros Cheat in 2017? Everything to Know About a Dark Moment in MLB History -MENTIONS-> The Athletic

Entities: The Athletic, Michigan, Rutgers, Big Ten, Jim Harbaugh, NCAA cheating investigation, Michigan sign-stealing scandal news: Latest updates on possible
Jim Harbaugh suspension from NCAA cheating investigation, Michigan-Purdue Big Ten Championship Game, decoded signals, Pete Thamel, Ohio State, Sporting News,
NCAA, Purdue
</item 9>

<item 10>
Michigan’s Rivals Fume About Sign-Stealing: ‘This Was Damn Near Espionage’ (The New York Times, 2023-11-03T22:07:31+00:00)
The Big Ten does have the ability to act under its sportsmanship policy, but that doesn’t mean that it would want to act quickly or decisively before the NCAA
completes its entire investigation and allows Michigan a chance to respond to its findings.

This is an unprecedented situation; whatever Petitti decides to do (or not do) will set a precedent. The Big Ten itself doesn’t have investigators, so it needs
to rely upon the NCAA to do that part — and to determine who else was involved in the alleged scouting scheme. It’s not clear exactly what the coaches would
want the league to do to punish Michigan; banning the team from competing in the Big Ten championship, for example, would harm players who had nothing to do
with the sign-stealing apparatus.

A source briefed on the coaches’ call said Big Ten coaches are concerned about whether Michigan “should represent the Big Ten.”

“No matter what happens, if Michigan continues to move forward, the clouds will follow,” the source said.
</item 10>
=== END ITEM id=ec23ee93fb3f5c1a ===

=== ITEM id=9270d9d019b0128d ===
Gold sentences and facts:
Sentence 0: Epic will also suggest that Google hid items from discovery by allowing execs and employees to have “off the record” chats, but Google will counter this as well, saying it has provided the court with thousands of chat logs and millions of documents.
  Fact 0.0: Epic will suggest that Google hid items from discovery by allowing execs and employees to have "off the record" chats.
  Fact 0.1: Google will also counter Epic's suggestion that Google hid items from discovery.
  Fact 0.2: Google has provided the court with thousands of chat logs and millions of documents.
Sentence 1: She admits that she may have misinterpreted the evidence, but maintains that Google manipulates Search to maximise ad revenue.
  Fact 1.0: She admits that she may have misinterpreted the evidence.
  Fact 1.1: She maintains that Google manipulates Search to maximise ad revenue.
Sentence 2: The case, filed by Arkansas-based publisher Helena World Chronicle, argues that Google “siphons off” news publishers’ content, their readers and ad revenue through anticompetitive means.
  Fact 2.0: The case was filed by Arkansas-based publisher Helena World Chronicle.
  Fact 2.1: The case argues that Google "siphons off" news publishers' content, their readers and ad revenue through anticompetitive means.

Retrieved items:
<item 1>
News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line (TechCrunch, 2023-12-15T17:56:02+00:00)
When online magazine The Atlantic modeled what would happen if Google integrated AI into search, it found that 75% of the time the AI would answer the user’s
query without requiring a click-through to its website, losing it traffic. This could have a major impact on publishers’ traffic going forward, as Google today
drives nearly 40% of their traffic, according to data from Similarweb.

Some publishers are now trying to get ahead of the problem. For example, Axel Springer just this week inked a deal with OpenAI to license its news for AI model
training. But overall, publishers believe they’ll lose somewhere between 20-40% of their website traffic when Google’s AI products fully roll out, The WSJ’s
report noted.

The lawsuit reiterates this concern, claiming that Google’s recent advances in AI-based search were implemented with “the goal of discouraging end-users from
visiting the websites of Class members who are part of the digital news and publishing line of commerce.”

Graph facts:
- A comprehensive list of 2023 tech layoffs -MENTIONS-> The Wall Street Journal
- Is Sam Bankman-Fried’s defense even trying to win? -MENTIONS-> The Wall Street Journal
- report on Thursday -PUBLISHED_BY-> The Wall Street Journal
- Publishers -PARTICIPATED_IN-> class action antitrust suit
- The inside story of Dave Clark's tumultuous last days at Flexport -MENTIONS-> The Wall Street Journal

Entities: class action antitrust suit, The Wall Street Journal, deal to license news for AI model training, News publisher files class action antitrust suit
against Google, citing AI’s harms to their bottom line, antitrust, The Atlantic, Axel Springer, TechCrunch, AI, Similarweb, Google, OpenAI
</item 1>

<item 2>
News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line (TechCrunch, 2023-12-15T17:56:02+00:00)
The case also arrives alongside the filing of the U.S. Justice Department’s lawsuit against Google for monopolizing digital ad technologies, and references the
2020 Justice Department’s civil antitrust suit over search and search advertising (which are different markets from digital ad technologies in the more recent
suit).

“The anticompetitive effects of Google’s scheme cause profound harm to competition, to consumers, to labor, and to a democratic free press,” reads an
announcement posted to the website of the law firm handling the case, Hausfeld.

“Plaintiff Helena World Chronicle, LLC invokes the Sherman Act and Clayton Act to seek class-wide monetary and injunctive relief to restore and ensure
competition for digital news and reference publishing and set up guardrails to preserve a free marketplace of ideas in the new era of artificial intelligence,”
it states.

A Google spokesperson offered a statement on the lawsuit, saying “This lawsuit is meritless.

Graph facts:
- News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line -MENTIONS-> Helena World Chronicle
- Alphabet -ACCUSED_OF-> Sherman Act
- What to know about Fortnite maker Epic Games’ antitrust battle with Google, starting today -MENTIONS-> Sherman Act
- News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line -MENTIONS-> Alphabet
- News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line -MENTIONS-> U.S. District Court in D.C.

Entities: Sherman Act, News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line, U.S. Justice Department,
Hausfeld, Helena World Chronicle, LLC, class action antitrust suit against Google, U.S. Justice Department lawsuit against Google, 2020 Justice Department civil
antitrust suit over search and search advertising, digital ad technologies, Clayton Act, search and search advertising, digital news and reference publishing,
class-wide monetary and injunctive relief, free marketplace of ideas, free press
</item 2>

<item 3>
Is Google Search better than the rest? And is that fair? (The Age, 2023-10-22T18:00:00+00:00)
The example given was replacing a search for “children’s clothing” with “NIKOLAI-brand kidswear”. Loading Google rejected this in very strong terms, saying the
piece was misleading and inaccurate while denying ever altering search terms. Wired removed the article for not meeting its standards, but the degree to which
it was shared on social media and boosted in write-ups at other outlets shows just how willing people are to accept foul play on Google’s part. A suspicion of
privacy invasion and inappropriate data handling follows the company at all times.

Grey herself is a former vice-president at DuckDuckGo, a privacy-focused company founded explicitly to counter giants such as Google. She admits that she may
have misinterpreted the evidence, but maintains that Google manipulates Search to maximise ad revenue. So what does Google say? In a post on X, Google’s
official search liaison, Danny Sullivan, said ad systems do not affect the organic results, i.e. the list of blue links in search results that are not
sponsored.

Graph facts:
- Apple defends Google Search deal in court: ‘There wasn’t a valid alternative’ -MENTIONS-> Google Search
- The people who ruined the internet -MENTIONS-> Google Search
- Early impressions of Google’s Gemini aren’t great -MENTIONS-> Google Search
- The restaurant nearest Google -MENTIONS-> Google Search
- Deepfake election risks trigger EU call for more generative AI safeguards -MENTIONS-> Google Search

Entities: Google Search, Is Google Search better than the rest? And is that fair?, Grey, DuckDuckGo, privacy invasion, inappropriate data handling, organic
search results, ad systems affecting organic results, Danny Sullivan, X, The Age, Google, WIRED, ad revenue
</item 3>

<item 4>
News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line (TechCrunch, 2023-12-15T17:56:02+00:00)
A Google spokesperson offered a statement on the lawsuit, saying “This lawsuit is meritless. People have many ways to access information and news content today
– through publishers’ websites, dedicated apps, social media platforms, print papers and more. Google links people to publishers’ websites more than 24 billion
times each month – at no cost to them.”

The complaint is available below.

Helena World Chronicle, LLC v. Google LLC and Alphabet Inc by TechCrunch on Scribd

Editor’s note: This post was updated after publication with a statement provided by Google on Friday evening.
</item 4>

<item 5>
News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line (TechCrunch, 2023-12-15T17:56:02+00:00)
A new class action lawsuit filed this week in the U.S. District Court in D.C. accuses Google and parent company Alphabet of anticompetitive behavior in
violation of U.S. antitrust law, the Sherman Act, and others, on behalf of news publishers. The case, filed by Arkansas-based publisher Helena World Chronicle,
argues that Google “siphons off” news publishers’ content, their readers and ad revenue through anticompetitive means. It also specifically cites new AI
technologies like Google’s Search Generative Experience (SGE) and Bard AI chatbot as worsening the problem.

In the complaint, Helena World Chronicle, which owns and publishes two weekly newspapers in Arkansas, argues that Google is “starving the free press” by sharing
publishers’ content on Google, losing them “billions of dollars.”

In addition to new AI technologies, the suit points to Google’s older question-and-answer technologies, like the “Knowledge Graph” launched in May 2012, as part
of the problem.

Graph facts:
- A comprehensive list of 2023 tech layoffs -MENTIONS-> Alphabet
- Earnings week ahead: Amazon, Microsoft, Alphabet, Meta, Ford, GM and more -MENTIONS-> Alphabet
- ASX set to rise ahead of inflation report after big tech lifts Wall Street -MENTIONS-> Alphabet
- Amazon’s Tye Brady discusses generative AI, humanoid robots and mobile manipulation -MENTIONS-> Alphabet
- Palestine’s growing tech industry has been literally blown apart by Israel’s war on Hamas -MENTIONS-> Alphabet

Entities: Alphabet, Helena World Chronicle, U.S. District Court in D.C., Search Generative Experience, Bard AI chatbot, Knowledge Graph, anticompetitive
behavior, U.S. antitrust law, Sherman Act, Arkansas, question-and-answer technologies, News publisher files class action antitrust suit against Google, citing
AI’s harms to their bottom line, class action lawsuit, TechCrunch, AI technologies
</item 5>

<item 6>
Graph query row:
article: Is Google Search better than the rest? And is that fair?
publisher: The Age
Google: Google
allegation: US v Google trial
</item 6>

<item 7>
Is Google Search better than the rest? And is that fair? (The Age, 2023-10-22T18:00:00+00:00)
The question remains, why do users prefer Google? And if the answer, as Walker says, is that Google is simply better, the question becomes whether that’s down
to its scale, its longevity, its ubiquity and all the user data it sucks up, which no other company could possibly match.

For the US, which has to prove that Google somehow broke the law to achieve the status quo, this legal case could be an uphill battle. But for the rest of us,
it offers a rare opportunity for some insight into what the tech giant does with its enormous market share. Is it still focused on being the best? Or, as has
been alleged, does it use its ubiquity to squeeze us for money even at the expense of product quality? How ‘organic’ are Google’s results? This month, Wired
posted an opinion article by lawyer and privacy advocate Megan Gray, which alleged that Google had accidentally revealed during the trial that it manipulates
people’s search queries for maximum ad revenue.

Graph facts:
- Apple defends Google Search deal in court: ‘There wasn’t a valid alternative’ -MENTIONS-> Google Search
- The people who ruined the internet -MENTIONS-> Google Search
- Early impressions of Google’s Gemini aren’t great -MENTIONS-> Google Search
- The restaurant nearest Google -MENTIONS-> Google Search
- Deepfake election risks trigger EU call for more generative AI safeguards -MENTIONS-> Google Search

Entities: Google Search, Wired opinion article, Megan Gray, Is Google Search better than the rest? And is that fair?, Walker, The Age, US, Google, trial, WIRED
</item 7>

<item 8>
News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line (TechCrunch, 2023-12-15T17:56:02+00:00)
In addition to damages, the suit is asking for an injunction that would require Google to obtain consent from publishers to use their website data to train its
general artificial intelligence products including Google’s own and those of rivals. It also asks Google to allow publishers who opt out of SGE to still show up
in Google search results, among other things.

The U.S. lawsuit follows an agreement Google reached last month with the Canadian government which would see the search giant paying Canadian media for use of
their content. Under the terms of the deal, Google will provide $73.5 million (100 million Canadian dollars) every year to news organizations in the country,
with funds distributed based on the news outlets’ headcount. Negotiations with Meta are still unresolved, though Meta began blocking news in Canada in August,
in light of the pressure to pay for the content under the new Canadian law.

The case also arrives alongside the filing of the U.S.
</item 8>

<item 9>
News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line (TechCrunch, 2023-12-15T17:56:02+00:00)
“When a user searches for information on a topic, Google displays a ‘Knowledge Panel’ to the right of the search results. This panel contains a summary of
content drawn from the Knowledge Graph database,” the complaint states. “Google compiled this massive database by extracting information from Publishers’
websites — what Google calls ‘materials shared across the web’ —and from ‘open source and licensed databases,'” it says.

By 2020, the Knowledge Graph had grown to 500 billion facts about 5 billion entities. But much of the “collective intelligence” that Google tapped into was
content “misappropriated from Publishers,” the complaint alleges.

Other Google technologies, like “Featured Snippets” where Google algorithmically extracts answers from webpages, were also cited as shifting traffic away from
publishers’ websites.

More importantly, perhaps, is the suit’s tackling of how AI will impact publishers’ businesses. The problem was recently detailed in a report on Thursday by The
Wall Street Journal, which led with a shocking statistic.

Graph facts:
- The inside story of Dave Clark's tumultuous last days at Flexport -MENTIONS-> The Wall Street Journal
- Europe’s artists and labels could lose $137m annually to US recorded music rightsholders if EU law isn’t changed, trade groups warn -MENTIONS-> Publishers
- A comprehensive list of 2023 tech layoffs -MENTIONS-> The Wall Street Journal
- Is Sam Bankman-Fried’s defense even trying to win? -MENTIONS-> The Wall Street Journal
- News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line -MENTIONS-> Alphabet

Entities: Knowledge Graph, Publishers, Knowledge Panel, Featured Snippets, class action antitrust suit, misappropriation of Publishers’ content, The Wall Street
Journal, report on Thursday, News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line, antitrust, TechCrunch, AI,
Google
</item 9>

<item 10>
News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line (TechCrunch, 2023-12-15T17:56:02+00:00)
SGE, it argues, offers web searchers a way to seek information in a conversational mode, but ultimately keeps users in Google’s “walled garden” as it
“plagiarizes” their content. Publishers also can’t block SGE because it uses the same web crawler as Google’s general search service, GoogleBot.

Plus, it says Google’s Bard AI was trained on a dataset that included “news, magazine and digital publications,” citing both a 2023 report from the News Media
Alliance and a Washington Post article about AI training data for reference. (The Post, which worked with researchers at the Allen Institute for AI, had found
that News and Media sites were the third largest category of AI training data.)

The case points to other concerns, too, like changing AdSense rates and evidence of improper spoliation of evidence on Google’s part, by its destruction of chat
messages — an issue raised in the recent Epic Games lawsuit against Google over app store antitrust issues, which Epic won.

Graph facts:
- News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line -MENTIONS-> Helena World Chronicle
- News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line -MENTIONS-> U.S. District Court in D.C.
- Is the Coverage on Taylor Swift and Travis Kelce’s Romance Nothing but Hype or a Pivotal Shift in NFL’s Century-Old Game Plan Towards Global Recognition?
-MENTIONS-> Washington Post
- News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line -MENTIONS-> Alphabet
- News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line -MENTIONS-> Search Generative Experience

Entities: SGE, Google Bard AI, GoogleBot, Google general search service, AdSense, News Media Alliance, Washington Post, Epic Games lawsuit against Google over
app store antitrust issues, app store antitrust issues, News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line,
class action antitrust suit against Google, Epic Games, Allen Institute for AI, TechCrunch, Google
</item 10>
=== END ITEM id=9270d9d019b0128d ===
