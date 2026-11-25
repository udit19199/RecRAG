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

=== ITEM id=ba82e32bb96da048 ===
Gold sentences and facts:
Sentence 0: Despite the suspension, Jim Harbaugh held his usual weekly news conference Monday, though he mostly avoided addressing the only topic on anyone's mind.
  Fact 0.0: Jim Harbaugh was suspended.
  Fact 0.1: Jim Harbaugh held his usual weekly news conference on Monday despite the suspension.
  Fact 0.2: Jim Harbaugh mostly avoided addressing the only topic on anyone's mind at the news conference.
Sentence 1: It would be Harbaugh's second suspension of the season, as the Wolverines self-imposed a three-game ban on Harbaugh to start the season amid an NCAA investigation that found Harbaugh lied to investigators.
  Fact 1.0: The suspension would be Harbaugh's second suspension of the season.
  Fact 1.1: The Wolverines self-imposed a three-game ban on Harbaugh to start the season.
  Fact 1.2: The self-imposed ban occurred amid an NCAA investigation.
  Fact 1.3: The NCAA investigation found that Harbaugh lied to investigators.
Sentence 2: Harbaugh reportedly watched the game from the team hotel, and now the leading piece of drama returns.
  Fact 2.0: Harbaugh reportedly watched the game from the team hotel.
  Fact 2.1: The leading piece of drama now returns.

Retrieved items:
<item 1>
Is the Coverage on Taylor Swift and Travis Kelce’s Romance Nothing but Hype or a Pivotal Shift in NFL’s Century-Old Game Plan Towards Global Recognition?
(Essentially Sports, 2023-10-25T21:30:41+00:00)
From Colin Kaepernick taking a knee during the national anthem to protest against racism, to domestic violence by players, to the Deflategate and Bountygate
scandals. But there have been bigger crises as well. The Union of Concerned Scientists accused the NFL of trying to intimidate scientists studying the link
between CTE and pro football. The NFL is already way past other leagues in terms of revenue, and now with the pivotal shift in its strategy, “America’s favorite
pastime” is on its way to world domination.

ADVERTISEMENT Article continues below this ad

Editorial Disclaimer: The views expressed are those of the author and do not necessarily reflect the views of EssentiallySports.

Watch This Story: Will Patrick Mahomes Eclipse Tom Brady as the NFL Goat? Travis Kelce Responds
</item 1>

<item 2>
Will the Panthers fire Frank Reich? Impatient David Tepper could turn up Carolina's hot seat with 'Thursday Night Football' loss (Sporting News,
2023-11-06T21:17:51+00:00)
Adam Thielen has looked rejuvenated in his new home, but running back Miles Sanders has been a bust acquisition. And the team's offensive line has allowed 74
pressures on the season, good for the 10th-most in the NFL.

Indeed, Carolina's offense has been less-than-impressive in 2023. That's the main reason that coach Frank Reich finds himself under fire for the team's
performance.

Reich already gave up play-calling to offensive coordinator Thomas Brown after the team's early-season struggles. Now, there are questions about whether the
team's lackluster offense — which averages the fifth-fewest yards (283.4) and sixth-fewest points (17.5) per game this season — could ultimately cause owner
David Tepper to move on from Reich after just one season.

Will the Panthers fire Frank Reich?

The Panthers aren't guaranteed to fire Reich even amid their poor season.
</item 2>

<item 3>
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
</item 3>

<item 4>
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
</item 4>

<item 5>
Michigan’s Rivals Fume About Sign-Stealing: ‘This Was Damn Near Espionage’ (The New York Times, 2023-11-03T22:07:31+00:00)
To know exactly what the other team is doing, Michigan might as well have been playing with 15 guys on the field,” he said. “What’s the message the Big Ten is
sending now by doing nothing? Win now, pay later? We might as well just send people to (scout) their practices and their games. It doesn’t encourage anybody to
follow the rules. It’s just telling them to do the opposite and say, f— it.”

GO DEEPER Michigan should be punished, say 94% of CFB coaches in our poll. What else did it reveal?

The NCAA is investigating Michigan’s football program amid allegations that the Wolverines used illegal in-person scouting and the recording of signals to steal
signs this season. Ahead of the Michigan-Michigan State game on Oct. 21, the Big Ten approached MSU and said it was made aware of “credible evidence” regarding
the sign-stealing allegations. The Big Ten said it would monitor the NCAA’s investigation into Michigan.
</item 5>

<item 6>
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
</item 6>

<item 7>
Shohei Ohtani rumors tracker: Latest news, updates on free agent's MLB future (Sporting News, 2023-11-27T21:20:11+00:00)
We have been promised very little of that, though. ESPN’s Jeff Passan reported this: “If visits between Ohtani and a team are reported publicly, it will be held
against the team.”

So, yeah. Probably won’t hear a ton, but that won’t stop the speculation. And it won’t stop us from passing along what precious few nuggets that are scattered
about.

MORE: Way-too-early MLB Power Rankings for 2024

Shohei Ohtani free agent rumors and updates

Ohtani has an 'affinity' for Boston, it seems

Date: Nov. 27

Source: Jeff Passan, ESPN

Hope everyone enjoyed the Thanksgiving holiday. Us baseball writers are pleased that Ohtani didn't decide to sign while the turkey was still warm. This isn't
exactly "breaking news" because rumors have been swirling, but it's an aside from an in-the-know reporter that just might give one fan base a little too much
hope and excitement.
</item 7>

<item 8>
Michigan sign-stealing scandal news: Latest updates on possible Jim Harbaugh suspension from NCAA cheating investigation (Sporting News,
2023-11-06T23:58:04+00:00)
pic.twitter.com/Qm9iehYVTR — Pete Thamel (@PeteThamel) November 11, 2023

Harbaugh will not be on the sidelines vs. Penn State

Michigan filed a temporary restraining order against the Big Ten on Friday night, hoping to block Harbaugh's suspension. But there will not be a ruling before
the Wolverines' game vs. Penn State, with the hearing scheduled for Friday, Nov. 17.

No ruling on Michigan’s TRO filing today, so Jim Harbaugh will not coach vs. Penn State.



Statement from the school: “We look forward to presenting our case next week where we intend to demonstrate that the Big Ten has not acted legally or fairly.”
</item 8>

<item 9>
Michigan State hires Jonathan Smith: Oregon State coach leaves alma mater to take over Spartans job (CBSSports.com, 2023-11-25T23:12:10+00:00)
Michigan State has hired Jonathan Smith as its next football coach, the school announced on Saturday. Smith, 44, has spent the past six seasons as coach at
Oregon State where he's guided the Beavers to 18 wins over the past two seasons, including a 10-3 mark and No. 17 finish in the AP Top 25 during a 2022 campaign
that earned him Pac-12 Coach of the Year honors.

Smith was among the initial pool of candidates linked to the Michigan State opening after the Spartans fired fourth-year coach Mel Tucker with cause midseason
amid sexual harassment allegations. Smith later emerged as the leading candidate to replace Tucker during the final week of the regular season, which saw Oregon
State suffer a 31-7 loss at rival Oregon.

"Jonathan has a proven track record of success, building the Oregon State program from the ground up by implementing a plan resulting in sustained historic
success for the Beavers," said Michigan State athletic director Alan Haller in announcing the hire.
</item 9>

<item 10>
NASCAR offseason news roundup: Short track testing in Phoenix, new spotters for stars, more moves for 2024 (CBSSports.com, 2023-12-07T20:15:30+00:00)
The matter of Almirola's replacement in the SHR No. 10 has been a rather poorly-kept secret: It is widely expected the team will hire Noah Gragson as its new
driver as Gragson looks to reclaim his standing as a bright young star following a disastrous rookie year in Cup with Legacy Motor Club. Gragson was recently
spotted at Stewart-Haas Racing's shop, and he also drove a Ford in this past weekend's Snowball Derby with Rette Jones Racing.

Gragson did not let on to any 2024 plans in a recent conversation with Matt Weaver of Sportsnaut, instead choosing to discuss his regrets and personal growth
from the end of his rookie season. Gragson resigned as the driver of the No. 42 in August while serving a month-long suspension after he was spotted having
liked a racially insensitive meme on his Instagram account.
</item 10>

<item 11>
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
</item 11>

<item 12>
N.F.L. Executives on the Biggest Questions Facing Super Bowl Contenders (The New York Times, 2023-10-16T23:58:53+00:00)
• San Francisco 49ers: What do they have in Purdy?

What I think: Let’s start with what the 49ers do not have in Purdy. They do not have an AFC North-certified, weatherproof Ben Roethlisberger clone able to
overpower the elements, break away from defensive linemen and thrive in the conditions Purdy faced against Cleveland. What they do have is a good, young
quarterback well-suited to run Kyle Shanahan’s offense at a high level by making quick decisions when given time to find open receivers.

Exec comment: “The 49ers knew they needed a good quarterback to win the Super Bowl, and they knew they could help Purdy be that. Give him a great receiving
tight end, two good receivers and a great halfback who fits the system, and they’ll get someone open on every pass play. But then Deebo Samuel gets hurt,
Christian McCaffrey gets hurt, and now we don’t have as many guys open.
</item 12>

<item 13>
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
</item 13>

<item 14>
Michigan conquers Ohio State again, but the story is far from over for 2023 Wolverines (Sporting News, 2023-11-25T23:20:56+00:00)
They have, and the last two years ended with victories in the Big Ten championship game before losses to Georgia and TCU in the CFP semifinals. The NCAA
investigation will continue, but if Saturday proved anything, the Wolverines will play out the rest of the season on their terms.

This Michigan story is far from over, and Harbaugh returns Monday.

Boom, indeed.
</item 14>

<item 15>
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
</item 15>

<item 16>
Michigan sign-stealing scandal news: Latest updates on possible Jim Harbaugh suspension from NCAA cheating investigation (Sporting News,
2023-11-06T23:58:04+00:00)
He initially told reporters he would speak during a scheduled hearing Friday, but then backtracked a bit and sounded as if he wasn't committed to that course,
pending legal advice.

Harbaugh said he will speak at the hearing on Friday. He said he’s looking for “due process.” Mentioned a civics class he took 40 years ago. “You’re innocent
until proven guilty. I’d like that opportunity.” — Nicole Auerbach (@NicoleAuerbach) November 13, 2023

Harbaugh clarifies that he’s not sure whether he’ll testify Friday. “That’s not my dance floor,” he says. “I’ve watched a lot of shows. I’ve watched Judge Judy
a lot.” — Austin Meek (@byAustinMeek) November 13, 2023

Sunday, Nov.
</item 16>

<item 17>
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
</item 17>

<item 18>
Michigan sign-stealing scandal news: Latest updates on possible Jim Harbaugh suspension from NCAA cheating investigation (Sporting News,
2023-11-06T23:58:04+00:00)
Judicial officer: Timothy P. Connors



Connors is a U-M lecturer, adjunct professor at Wayne State and Vermont. pic.twitter.com/WR4G4WTtfH — Chris Vannini (@ChrisVannini) November 11, 2023

Michigan players respond to Harbaugh suspension

J.J. McCarthy and a handful of other Michigan players tweeted "Bet" at the same time on Friday night, hours after Harbaugh was suspended for the remainder of
the regular season. McCarthy wore a shirt that said "Free Harbaugh" when the 59-year-old was serving a separate suspension at the start of the season.

Michigan players had a simple message on social media following Jim Harbaugh's suspension 👀
</item 18>

<item 19>
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
</item 19>

<item 20>
Michigan’s Rivals Fume About Sign-Stealing: ‘This Was Damn Near Espionage’ (The New York Times, 2023-11-03T22:07:31+00:00)
He has denied knowledge of the alleged scouting.

GO DEEPER Ten thoughts on the Big Ten's 2024 schedule

Required reading

(Photo: David Berding / Getty Images)
</item 20>

<item 21>
Michigan didn't need Jim Harbaugh against Penn State, but Ohio State is a different game (Sporting News, 2023-11-11T21:32:44+00:00)
McCarthy is a leader. Corum (26–145, 2 TDs) and Edwards (10-52, TD) can grind on anybody. The defense now has allowed just 7.5 points per game.

But the Wolverines will not be able to beat Ohio State the same way they beat Penn State. That 20-yard box will stretch across the entire country, and it
requires a coach ready to make those legacy-defining decisions. The Buckeyes have Heisman Trophy candidate Marvin Harrison Jr., and he was the separator when
Ohio State beat Penn State 20-12 on Oct. 21. The Buckeyes have a top-five defense, too, and all of the back-and-forth rage between those rivals will reach a
1970s pitch – 1973 for those who know – as they barrel toward another 11-0 vs. 11-0 showdown at the Big House.

Who is the good guy? Who is the villain?
</item 21>

<item 22>
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
</item 22>

<item 23>
Michigan’s Rivals Fume About Sign-Stealing: ‘This Was Damn Near Espionage’ (The New York Times, 2023-11-03T22:07:31+00:00)
… No doubt this all has had a profound effect.

“This guy’s being investigated for three different things now between the (alleged) illegal signal stealing, the (alleged) illegal recruiting during COVID and
that investigation into the offensive coordinator and alleged computer hacking. There are guys (on that call) who could lose jobs, and then there’s this guy
over here (Harbaugh) who is gonna get a new, bigger contract now, and they won’t do anything about him.”

Asked to describe the tone of the coaches’ sentiment expressed to Petitti, another Big Ten coach called it “angry” — particularly at the Big Ten’s lack of
action, or even apparent interest in taking any.

“Everybody’s upset,” that coach told The Athletic. “Why is nothing being done? We want to know, what else do you need to know to take action? We (the Big Ten
head coaches) want something done now.
</item 23>

<item 24>
Michigan State hires Jonathan Smith: Oregon State coach leaves alma mater to take over Spartans job (CBSSports.com, 2023-11-25T23:12:10+00:00)
The 2022 season, which marked the Beavers' first 10-win season since 2006 and highest AP Top 25 finish since 2000, resulted in a contract extension for Smith
through February 2029 that bumped his annual salary to $4.85 million. Oregon State then started the 2023 season 8-2 before dropping back-to-back games against
Washington and Oregon.

Earlier in the week, Oregon State athletic director Scott Barnes called it the "No. 1 priority" to reach a contract extension with Smith as speculation of this
move began to grow. In a statement Saturday after Smith's departure was finalized, Barnes thanked him for his efforts at Oregon State as the Beavers now embark
on the search for Smith's replacement.

"Coach Smith and his staff worked tirelessly to rebuild the culture at Oregon State, both on the field and in the classroom," Barnes said.
</item 24>

<item 25>
Michigan conquers Ohio State again, but the story is far from over for 2023 Wolverines (Sporting News, 2023-11-25T23:20:56+00:00)
Say what you will about the adults who were involved in the in-person scouting scandal. Staffer Conor Stallions and assistant coach Chris Partridge are gone.
Harbaugh was suspended. He addressed the team Friday, and McCarthy listened closely.

"The one thing he said that stuck out, the whole mantra, 'The Team, The Team, The Team,'" McCarthy said. "He said we are that team."

That is true. This is the most mature player-led Michigan football team on the field since 1997 – when the program beat Ohio State 20-14 in a top-five showdown
and split the national championship with Nebraska.

Since then, the Buckeyes found a way in these high-magnitude games, with 2006 and 2016 the best examples. On a stage that was somewhere in between those
classics, the Wolverines pointed the pressure in the other direction. Marvin Harrison Jr.
</item 25>

<item 26>
Michigan’s Rivals Fume About Sign-Stealing: ‘This Was Damn Near Espionage’ (The New York Times, 2023-11-03T22:07:31+00:00)
“They’re reading the tea leaves and wondering why the Big Ten hasn’t done anything yet. Every week and every day that goes by, people are like, ‘Something’s
gotta give.’ It’s getting a little bit out of hand when you see him (allegedly) on the Central Michigan sideline. The playing field is not level right now. How
can you have a team that you know has a competitive advantage over you still being allowed to play? That’s what the coaches are grappling with.”

Advertisement

“It feels like (former commissioner) Kevin (Warren) taking over and then COVID,” the source continued. “Tony’s walking into this situation, and people are
calling for the league to make a statement before they have all the facts.”

Despite frustration from all corners of the conference, sources at four different Big Ten schools said they do not expect the conference to levy any sort of
punishment against Michigan before the season ends.
</item 26>

<item 27>
Michigan State hires Jonathan Smith: Oregon State coach leaves alma mater to take over Spartans job (CBSSports.com, 2023-11-25T23:12:10+00:00)
Tucker was promptly suspended before the school fired him with cause several weeks later. The Spartans went on to finish 4-8 with assistant Harlon Barnett
serving as interim coach the rest of the way, ending the season with a 42-0 loss to Penn State on Friday.

While Michigan State has found itself playing in a stacked Big Ten East division for roughly the last decade -- Michigan, Ohio State and Penn State have all
been annual opponents -- that will change in Smith's debut season as the Big Ten expands and does away with divisions. Smith will also get the chance to face
some familiar opponents from his time at Oregon State given that four Pac-12 schools are joining the conference. As far as the 2024 season is concerned on that
front, Michigan State still faces Ohio State and Michigan (the Wolverines are a protected rival) but avoids Penn State. Oregon is the only newcomer it will face
next fall, with the Spartans set to travel to Eugene, Oregon, in early October.
</item 27>

<item 28>
Michigan conquers Ohio State again, but the story is far from over for 2023 Wolverines (Sporting News, 2023-11-25T23:20:56+00:00)
That set the Wolverines up at the 7-yard line. Corum slammed into the line of scrimmage three times before a fourth-and-1.

Corum heard Moore scream, "Smash." Tight end A.J. Barner did his best Triple H impersonation in the huddle: "It's time to play the game."

Michigan goes for it on 4th and goal and Blake Corum punches it in for The Game's first touchdown.



That's Corum's 21st touchdown run of the season 😲pic.twitter.com/TrF5pIpni3 — 247Sports (@247Sports) November 25, 2023

"I gotta score," Corum said. "Run through the hole and I'm about to run somebody over. That makes the hairs on my arm stick up."

Corum scored on a one-yard touchdown for a 7-0 lead. Michigan never played from behind against the Buckeyes. What was Moore thinking in that situation?
</item 28>

<item 29>
Michigan didn't need Jim Harbaugh against Penn State, but Ohio State is a different game (Sporting News, 2023-11-11T21:32:44+00:00)
I did this for you, for this university, the president, our AD. We've got the best players, best university, and the best alumni in the country. I love these
guys. These f------ guys right here. These guys right here. These guys did it."

MORE: Sherrone Moore gives tearful interview after Penn State win

Whoa, let's add the FCC standards to the Wolverines' list of potential violations. If Michigan (10-0, 7-0 Big Ten) was not the college football heel heading
into the final two weeks of the regular season, then they are public enemy No. 1 now. It might be “Michigan vs. Everybody,” and everybody has the same question.

Will Harbaugh be on the field before the end of the regular season?
</item 29>

<item 30>
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
</item 30>

<item 31>
USC vs. Notre Dame live score, updates, highlights from Week 7 college football game (Sporting News, 2023-10-14T22:00:05+00:00)
9:42 p.m.: Trading PI calls, as now Notre Dame is penalized for pass interference after Williams tried to find Rice on a deep shot down the middle. Morrison is
called for the penalty, his third of the night. USC moves back to its own 43.

9:40 p.m.: USC is called for its second offensive pass interference penalty, with Mario Williams being flagged after a 10-yard catch by Washington. The Trojans
will now have a first-and-25.

9:37 p.m.: Another short pass by Hartman, and Calen Bullock makes a great open-field play to tackle Ford after just a yard. Notre Dame punts, and Branch returns
it 21 yards up to his team's 43.

9:36 p.m.: Faison manages to get a yard on a short pass, then Love picks up four on the ground to set up a third-and-5 at the Notre Dame 30.

9:32 p.m.
</item 31>

<item 32>
NFL power rankings: Lions, 49ers level up; Texans jump Bengals, Bills; Jets, Patriots sink to bottom for Week 11 (Sporting News, 2023-11-13T21:54:00+00:00)
They can win with offense or defense. Dan Campbell is coach of the year.

4. San Francisco 49ers 6-3 (8)

The 49ers had a surprise offensive get-well game in Jacksonville with Brock Purdy leading the way and all his key supporting weapons doing damage, even without
Christian McCaffrey scoring. The biggest development was the defense going back to dominating a good offense.

5. Baltimore Ravens 7-3 (2)

The Ravens' offense is just fine with the running and passing of Lamar Jackson, but he still can't afford to make big mistakes, which he did vs. the Browns. The
defense needs to pick up the pieces, too, stat, ahead of facing the Bengals on a short week.

6. Miami Dolphins 6-3 (6)

The Dolphins got a much-needed bye to figure out how to beat the better teams in the league.
</item 32>

<item 33>
Raiders vs. Lions live score, updates, highlights from NFL 'Monday Night Football' game (Sporting News, 2023-10-30T22:20:03+00:00)
@jackharlow will be the performer for our Thanksgiving Halftime Show powered by @Verizon!



📰 https://t.co/oUNhlBkWe3 pic.twitter.com/vMCAMpCFfk — Detroit Lions (@Lions) October 30, 2023

7:20 p.m. — Jack Harlow is in the house supporting the Lions.

.@JackHarlow repping the Honolulu blue 🦁



Catch his performance at the @Lions Thanksgiving Halftime Show powered by @Verizon pic.twitter.com/haMpQ95y5U — NFL (@NFL) October 30, 2023

6:49 p.m. — Brian Hoyer is inactive tonight, so Aidan O'Connell is the Raiders' backup quarterback behind Jimmy Garoppolo

6:45 p.m. — Lions WR Amon-Ra St. Brown is officially active after being listed as questionable with an illness.

6:25 p.m.
</item 33>

<item 34>
Will the Panthers fire Frank Reich? Impatient David Tepper could turn up Carolina's hot seat with 'Thursday Night Football' loss (Sporting News,
2023-11-06T21:17:51+00:00)
The Panthers entered the 2023 NFL season shrouded in optimism. The team had moved on from Matt Rhule, who proved to be an ineffective NFL coach, and added a
potential franchise quarterback to the fold in Bryce Young.

The hope was that those moves would turn Carolina into a contender in the wide-open NFC South.

Instead, they find themselves not only at the bottom of the division but also sporting one of the worst records in the NFL at 1-7.

Young hasn't yet looked comfortable in the Panthers' offense, and he is coming off one of his worst games as a starter against the Colts. He threw three
interceptions in that contest, including two pick-sixes to Kenny Moore in a 27-13 defeat.

MORE: Full list of rookie quarterbacks to start an NFL game in 2023

The rest of the Panthers' offense hasn't been much better.
</item 34>

<item 35>
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
</item 35>

<item 36>
NFL power rankings: Lions, 49ers level up; Texans jump Bengals, Bills; Jets, Patriots sink to bottom for Week 11 (Sporting News, 2023-11-13T21:54:00+00:00)
But the reeling defense and pass protection continue to be un-winnable assets.

24. Atlanta Falcons 4-5 (20)

The Falcons were forced into another QB change because of injury. They found a few more answers offensively but that position still made it an overall mess. The
defense also doesn't seem to be well prepared to different styles of offenses.

25. Green Bay Packers 3-6 (25)

The Packers saw some improved work from Jordan Love and his young weapons as they spread the ball well in the passing game, but the run defense woes and key
holes in coverage have all but doomed their playoff hopes.

26. Los Angeles Rams 3-6 (26)

The Rams are hoping to find a few more answers during a bye, as Sean McVay has seen his good coaching job earlier in the season go by the wayside with key
offensive injuries.
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
</item 38>

<item 39>
Michigan’s Rivals Fume About Sign-Stealing: ‘This Was Damn Near Espionage’ (The New York Times, 2023-11-03T22:07:31+00:00)
For the past two weeks, coaches across college football have been riveted by the alleged Michigan sign-stealing scheme, but inside the Big Ten, the topic has
been more than just a curiosity. On Wednesday’s Big Ten coaches video call with commissioner Tony Petitti, and after Jim Harbaugh left the call, that
frustration was voiced loud and clear, according to conference coaches, who said they don’t feel like the new Big Ten commissioner is “motivated” to do anything
about the Wolverines.

Advertisement

“There is just a ton of frustration,” a Big Ten coach told The Athletic on Thursday morning. “Look at Jim Harbaugh’s record before this started. The guy was on
the hot seat before 2021, and now he’s like the king of college football. … No doubt this all has had a profound effect.
</item 39>

<item 40>
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
</item 40>

<item 41>
USC vs. Notre Dame live score, updates, highlights from Week 7 college football game (Sporting News, 2023-10-14T22:00:05+00:00)
He then rushes up the middle for three to set up a first-and-goal at the 7.

10:13 p.m.: An explosive punt return from Branch, who dances around the field and burns down the left sideline, being stopped only by the punter at the Notre
Dame 18 to prevent a touchdown. USC ball with 11:13 left to play, with Notre Dame leading by 18.

10:12 p.m.: Notre Dame picks up just a yard, then Jadarian Price is tackled for no gain. Hartman fires an incomplete pass, and the Fighting Irish will have to
punt.

10:10 p.m.: Notre Dame converts the third down with Hartman firing a quick strike to Staes for seven yards up to the Fighting Irish 33.

10:09 p.m.: The start of the fourth quarter looks like a lot like most of the second half so far: back-to-back runs by Estime.
</item 41>

<item 42>
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
</item 42>

<item 43>
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
</item 43>

<item 44>
NFL roundup: Pats hit new low as Lions reach 9-3 for first time since 1962 (The Guardian, 2023-12-03T22:11:26+00:00)
The 49ers spent an offseason chirping at the Birds – Samuel called one Eagles cornerback “trash” – and then entered the Linc as a rare betting favorite against
a 10-win team at home.

The Eagles (10-2) suffered a scare when Hurts jogged off the field in the fourth and went straight to the locker room. He was cleared to play and returned to
action with about 10 minutes left. It was not immediately clear when Hurts was injured.

By the time last season’s NFL MVP runner-up returned, the Eagles trailed 35-13 and were on their way toward their first home loss of the season. Hurts did hit
DeVonta Smith for a two-yard TD on his return drive. He finished 26-of-45 passing for 298 yards. Purdy put the finishing touches on the win with a 46-yard TD to
Samuel.

Perhaps all that braggadocio paid off for the 49ers.
</item 44>

<item 45>
Michigan sign-stealing scandal news: Latest updates on possible Jim Harbaugh suspension from NCAA cheating investigation (Sporting News,
2023-11-06T23:58:04+00:00)
Here is Michigan’s 10-page letter to the Big Ten responding to the sportsmanship investigation, reporting with @RossDellenger 1/3 pic.twitter.com/P0oYgIbS67 —
Dan Wetzel (@DanWetzel) November 9, 2023

No timetable for Big Ten decision

The Big Ten is reviewing the "lengthy" official response from Michigan and Jim Harbaugh and it's unclear when a potential ruling might come down, says ESPN's
Pete Thamel. His colleague Adam Rittenberg reported Wednesday that a decision was expected by Thursday.

Source: The Big Ten is reviewing the lengthy written responses from both Michigan and Jim Harbaugh. There’s no indication of any timetable on a decision on
potential Big Ten discipline in the case of Michigan’s alleged illegal signal stealing.
</item 45>

<item 46>
NASCAR offseason news roundup: Short track testing in Phoenix, new spotters for stars, more moves for 2024 (CBSSports.com, 2023-12-07T20:15:30+00:00)
"I think the cars can be quieter just to help the fan experience."

Notable crew changes

Not one, but two of the biggest stars in NASCAR will have new eyes in the sky in 2024, as the spotters for both Chase Elliott and Ryan Blaney announced they
will be stepping down. Eddie D'Hondt announced Saturday he would not return as Elliott's spotter next season, followed Monday by Josh Williams announcing he
would step down as Blaney's spotter.

According to Bob Pockrass of Fox Sports, former NASCAR Xfinity Series driver Tim Fedewa, the longtime spotter for Kevin Harvick, will take over as the spotter
for Blaney next season. A possible landing spot for Williams exists at Spire Motorsports, as Williams has previously spotted for Zane Smith in the Craftsman
Truck Series.

Spire Motorsports announced a number of personnel moves this week, hiring Stephen Doran as the crew chief for Smith's No.
</item 46>

<item 47>
Michigan conquers Ohio State again, but the story is far from over for 2023 Wolverines (Sporting News, 2023-11-25T23:20:56+00:00)
"For the big boys, for Blake, for myself, that's music to our ears that he has the confidence in us to go get the extra yard or two yards. That's who we are."

Michigan played to win the game. Ohio State, not so much. The Buckeyes settled for a 52-yard field goal late in the first half, which Jayden Fielding missed
wide left.

Even when Ohio State tied the score at 17-17 with 6:08 left in the third quarter on an up-tempo drive that featured eight straight run plays and a TreVeyon
Henderson touchdown, Michigan answered. The Wolverines scored on every drive in the second half – three field goals and a touchdown.

"It goes back to how they prepare, how they attack things and how they work," Moore said. "I would be doing them a disservice if I went out there and was
conservative."

Say what you will about the adults who were involved in the in-person scouting scandal.
</item 47>

<item 48>
Michigan conquers Ohio State again, but the story is far from over for 2023 Wolverines (Sporting News, 2023-11-25T23:20:56+00:00)
ANN ARBOR, Mich. – Michigan quarterback J.J. McCarthy heard Zak Zinter scream. McCarthy saw tackle Karsen Barnhart holding Zinter's foot in the air. This was a
gruesome scene at Michigan Stadium on Saturday.

No. 3 Michigan lost their senior All-American guard to a broken tibia and fibula with 2:19 left in the third quarter of a tie game with No. 2 Ohio State. Zinter
was carted off in what could have been a turning point in the other direction given his value on the interior line.

"It was a sight that I wouldn't wish upon anyone to see," McCarthy said. "At that moment, seeing the look in everyone's eyes, seeing them rally together, it was
spiritual, honestly. It was a different drive that hit on everybody."

On the next play, senior running back Blake Corum finally broke through the second level of the Buckeyes' defense for a 22-yard go-ahead touchdown.
</item 48>

<item 49>
Michigan didn't need Jim Harbaugh against Penn State, but Ohio State is a different game (Sporting News, 2023-11-11T21:32:44+00:00)
Who is the good guy? Who is the villain? Ohio State and Michigan locked in their answers a long time ago. More than 17 million people watched last year's
matchup – a 45-23 Michigan victory where Stalions' involvement continues to be scrutinized through a double-barrel Big Ten and NCAA investigation. Big Ten
commissioner Tony Petitti's ruling sent a shockwave through the conference Friday, mainly because the third game means Harbaugh will not be on the sideline for
The Game. Early forecasts say 20 million will tune in for The Game this year.

Will Harbaugh really not be there? That is for the courts to decide, but Moore’s comments suggest Harbaugh, president Santa Ono and athletic director Warde
Manuel are on a united front; consequences be damned in 2024 and beyond.

In the short term? The College Football Playoff committee could slide Michigan up to No. 2 now, and the debate with the Buckeyes and No.
</item 49>

<item 50>
Michigan didn't need Jim Harbaugh against Penn State, but Ohio State is a different game (Sporting News, 2023-11-11T21:32:44+00:00)
Did Michigan really need Jim Harbaugh on Saturday?

Perhaps the result provides the short-term answer. No. 3 Michigan beat No. 10 Penn State 24-15 at Beaver Stadium on Saturday. The Wolverines threw one pass in
the second half – and that didn't even technically count – in a risk-averse game plan under acting coach Sherrone Moore. The defense smothered the Nittany
Lions, and that will lead to more buildup ahead of the showdown with No. 1 Ohio State on Nov. 25.

Then Moore let it all there in the Fox postgame interview with Jenny Taft. Even if the answer is no, Harbaugh was ever-present.

"I want to thank the Lord and I want to thank Coach Harbaugh," Moore said. "I f------ love you man. I love the s--- out of you, man. I did this for you, for
this university, the president, our AD.
</item 50>

<item 51>
Bryce Harper 'Atta Boy' controversy, explained: Why Orlando Arcia's comments led to backlash, Alanna Rizzo rant (Sporting News, 2023-10-13T21:58:11+00:00)
She said that reporters abstaining from writing on his comments would be akin to "protecting players from themselves."

She also went on to say that what Arcia said wasn't a slur or hateful, and added the only reason anyone was concerned about it was because "they know it would
motivate Harper to beat [the Braves]. Which implies the Braves do not feel they can handle a motivated Bryce Harper and are willing to suggest that is why they
lost the other day."

"I didn't say something sooner because I thought this was overblown," Janes continued. "And frankly, I thought it wouldn't help. But I am genuinely disgusted by
the way some media members are acting like a reporter was in the wrong for this. He wasn't.

"And if the Braves let this be the story of their series, that's on them. Win and it goes away. Period."
</item 51>

<item 52>
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
</item 52>

<item 53>
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
</item 53>

<item 54>
Michigan State hires Jonathan Smith: Oregon State coach leaves alma mater to take over Spartans job (CBSSports.com, 2023-11-25T23:12:10+00:00)
"All of Beaver Nation should thank him for the lasting legacy he has left to his alma mater and wish him and his family nothing but the best moving forward."

Smith held multiple assistant jobs from 2002-17 before returning to his alma mater as coach in 2018. Smith spent six seasons as an assistant under former
college football coach Chris Petersen, first as Boise State's quarterbacks coach from 2012-13 and then as Washington's quarterbacks coach and offensive
coordinator from 2014-17. His time in Seattle included a Pac-12 title and College Football Playoff berth for Washington during the 2016 season.

Smith's exit from Oregon State comes at a point where the Beavers face much uncertainty as the Pac-12 prepares to lose 10 of its 12 members ahead of the 2024
season. Oregon State and Washington State are the only schools in the conference that have not been picked up by other power leagues.
</item 54>

<item 55>
Michigan’s Rivals Fume About Sign-Stealing: ‘This Was Damn Near Espionage’ (The New York Times, 2023-11-03T22:07:31+00:00)
Earlier this week, Central Michigan said it is investigating whether suspended Michigan staffer Connor Stalions was on the CMU sideline during the Chippewas’
Sept. 1 game at Michigan State. Screenshots of a person who looks similar to Stalions began circulating online Monday night, and The Athletic obtained more
photos of the person on the sideline Tuesday.

Central Michigan is investigating if the person on the right here is Connor Stalions on the CMU sideline at Michigan State earlier this season. AD Amy Folan:
“We became aware of these photos late yesterday and we are in the process of determining the facts surrounding them." pic.twitter.com/ncazAghBbS — Chris Vannini
(@ChrisVannini) October 31, 2023

Stalions, who was suspended with pay by Michigan on Oct. 20, is at the center of the NCAA’s investigation into the alleged scouting and sign-stealing scheme.
</item 55>

<item 56>
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
</item 56>

<item 57>
Shohei Ohtani rumors tracker: Latest news, updates on free agent's MLB future (Sporting News, 2023-11-27T21:20:11+00:00)
20

Source: Ken Rosenthal, The Athletic

Buried at the bottom of Rosenthal’s piece on what an Ohtani contract might look like — incentives, opt-outs and all sorts of stuff — is this little bit of
reporting:

An agreement at the winter meetings in Nashville the first week of December would be the ideal outcome for MLB, but not necessarily for Ohtani. He perhaps will
be better off waiting for the signing of Japanese righty Yoshinobu Yamamoto, who is expected to be posted for free agency as soon as Monday, triggering a 45-day
window for teams to negotiate with him. Losing out on the 25-year-old Yamamoto, who could command more than $200 million, might put certain teams on tilt,
improving the landscape for Ohtani.

Interesting, certainly.

Mariners out of the Ohtani chase?

Date: Nov.
</item 57>

<item 58>
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
</item 58>

<item 59>
Michigan’s Rivals Fume About Sign-Stealing: ‘This Was Damn Near Espionage’ (The New York Times, 2023-11-03T22:07:31+00:00)
Stalions bought tickets to games in at least seven Big Ten stadiums before those teams played the Wolverines over the past three seasons, including the 2023
season, sources told The Athletic last month. Purchasing the tickets is not a violation of NCAA rules, but using them to scout and record other teams would
violate the rules, prohibiting in-person, on-campus scouting and the audio or video recording of signals.

“They aren’t allegations. It happened,” Purdue coach Ryan Walters said Thursday night on his radio show ahead of Saturday’s game against the Wolverines.
“There’s video evidence. There’s ticket purchases and sales you can track back. We know for a fact they were at a number of our games. We’ve had to teach our
guys a new language.”

On Monday, coach Jim Harbaugh met with reporters and said “the people that know us the most think the most of us” as Michigan faces the NCAA investigation. He
has denied knowledge of the alleged scouting.
</item 59>

<item 60>
49ers cap space 2024: Why Chase Young trade may not be a pure rental for San Francisco (Sporting News, 2023-10-31T22:22:43+00:00)
The 49ers made the biggest move ahead of the 2022 NFL trade deadline when they acquired running back Christian McCaffrey from the Panthers. They once again made
a splash ahead of the 2023 trade deadline when they traded for Commanders pass rusher Chase Young.

Young was among the most notable players dealt ahead of the 2023 trade deadline. The No. 2 pick in the 2020 NFL Draft is in the midst of a breakout season,
totaling five sacks in eight games and generating a solid 18 pressures while getting into the backfield often for the Commanders.

But as Washington jettisoned its top two defensive ends, Young and Montez Sweat, the 49ers found themselves positioned to pounce. They traded a 2024 third-round
pick to get the talented defender and reunite him with his college teammate and fellow No. 2 pick, Nick Bosa.
</item 60>

<item 61>
Michigan sign-stealing scandal news: Latest updates on possible Jim Harbaugh suspension from NCAA cheating investigation (Sporting News,
2023-11-06T23:58:04+00:00)
We are unable to comment further regarding this personnel matter." — Jesse Dougherty (@dougherty_jesse) November 4, 2023

In a statement, Stalions' lawyer said that his client hoped to avoid creating more distractions for Michigan and denied that any Wolverines staffer "told anyone
to break any rules."

MORE: Who is Connor Stalions?

Stalions is accused of buying numerous tickets to scout future opponents in-person, which is against NCAA rules. The NCAA is also investigating the possibility
that Stalions appeared on Central Michigan's sideline in Chippewas gear for a game against Michigan State.

Stalions was suspended by Michigan when the news of his alleged involvement first broke.
</item 61>

<item 62>
NFL power rankings: Lions, 49ers level up; Texans jump Bengals, Bills; Jets, Patriots sink to bottom for Week 11 (Sporting News, 2023-11-13T21:54:00+00:00)
Their defense should be getting a little better to support their often explosive offense. They should feel good about a Week 11 rebound vs. the Raiders.

7. Pittsburgh Steelers 6-3 (9)

The Steelers are playing the way Mike Tomlin wants, with the running game rolling to big totals with Jaylen Warren and Najee Harris and the defense doing what
it needs, more so vs. pass than run, in key moments. You can't argue with the overachieving results.

8. Cleveland Browns 6-3 (10)

The Browns proved they can win games with defense in another way, with big plays. They also have more offensive pop when Deshaun Watson plays well and the
running game doesn't slump. Their playoff hopes look much better after their stunning comeback vs. the Ravens.

9.
</item 62>

<item 63>
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
</item 63>

<item 64>
Michigan sign-stealing scandal winners and losers, from Jim Harbaugh to Tony Petitti (Sporting News, 2023-11-11T22:50:13+00:00)
Will the Harbaugh-Petitti battle lead to an unsettling end for Michigan? President Santa Ono and athletic director Warde Manuel have voiced their support for
Harbaugh, but that has been more about due process than admitting to any wrongdoing.

It has thrust the Wolverines – a talented team led by J.J. McCarthy, Blake Corum and a defense that allows 7.5 points per game – into a villain role, complete
with visits from Ric Flair. That only works when you are winning. One loss, and all this could unravel, and the Terps and Buckeyes still remain on the schedule.

Will a national championship outweigh the stain on their reputation? The answer is almost always yes for the fans, but the program might feel a different
long-term result.
</item 64>

<item 65>
Michigan conquers Ohio State again, but the story is far from over for 2023 Wolverines (Sporting News, 2023-11-25T23:20:56+00:00)
Michigan never played from behind against the Buckeyes. What was Moore thinking in that situation?

"Let's go," Moore said. "That's it. The players just practiced so well this week, and when you have confidence they are going to execute at a high level, it's
just like first down."

Moore favored a conservative game plan the previous two weeks against No. 11 Penn State and Maryland; one that amounted to 91 running plays and 31 pass
attempts. Yet he was the aggressor in The Game. The Wolverines converted three fourth-down conversions in the first half. McCarthy – who finished 16 of 20 for
148 yards and a TD – threw a 22-yard touchdown to Roman Wilson between two defenders for a 14-3 lead.

Interim coach Sherrone Moore leads Michigan to win

​​"Coach Moore said from the get-go that he was going to the most-aggressive game he's ever called," McCarthy said.
</item 65>

<item 66>
Michigan’s Rivals Fume About Sign-Stealing: ‘This Was Damn Near Espionage’ (The New York Times, 2023-11-03T22:07:31+00:00)
We (the Big Ten head coaches) want something done now. I don’t think people understand the advantage that what they’re (allegedly) doing gives you. People
think, ‘OK, now that everybody knows, we all can just move on.’ Like, ‘now, it’s fair.’ Well, no, it isn’t. Not at all. This changes the way you operate. A lot
of teams have been doing things a certain way for years. Now, it’s forcing you to teach your players a whole new way to communicate just for them. People think
that this is just advanced scouting. This was damn near espionage.”

A third Big Ten head coach told The Athletic that this is “one of the most egregious breaches in the spirit of the game” he’s ever heard of.

“They (Michigan) have been manipulating the game and cheating the game for two-and-a-half years.
</item 66>

<item 67>
Bryce Harper 'Atta Boy' controversy, explained: Why Orlando Arcia's comments led to backlash, Alanna Rizzo rant (Sporting News, 2023-10-13T21:58:11+00:00)
"That's taking away from true reporters and true journalists. Where were you on the random Tuesday in April in Cincinnati when this team was playing in
Cincinnati? Where were you on a rain delay, and you have to sit in the press box for all of this time? It's ridiculous."

"And then some jackoff comes in at the end of the season that gets a credential, God only knows why. And the clubhouse is a sacred space." Alanna Rizzo went off
on Jake Mintz's clubhouse reporting on MLBN's "High Heat" Thursday. pic.twitter.com/6dooqeLPnh — Awful Announcing (@awfulannouncing) October 12, 2023

Rizzo's comments immediately drew heat from the baseball journalism world, including from Janes, who posted a thread defending Mintz. Janes reported that
cameras and recorders were recording, and that Arcia yelled the comments while they were recording and with more than a dozen reporters around.
</item 67>

<item 68>
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
</item 68>

<item 69>
Michigan sign-stealing scandal winners and losers, from Jim Harbaugh to Tony Petitti (Sporting News, 2023-11-11T22:50:13+00:00)
Penn State-Ohio State (5.3), Colorado-Oregon (5.2) and Ohio State-Notre Dame (5.1) make up that list. Michigan-Penn State should join that list on Saturday, but
the intrigue around The Game this year could double all of those games.

Last year's matchup drew an 8.05 with 17.1 million viewers. Michigan won the last two meetings and has a chance to win three in a row in the rivalry for the
first time since 1995-97. Ohio State is No. 1 in the College Football Playoff rankings. Harbaugh (2-5) and Day (1-2) both have losing records in this rivalry
matchup, and the Big Ten East, CFP and perhaps national championship are on the line.

Will this game draw a 10 Nielsen rating and 20 million viewers? It's not out of the question, and there will be enough WWE-like promos leading up to this game
to get both fan-bases frothing.
</item 69>

<item 70>
Michigan didn't need Jim Harbaugh against Penn State, but Ohio State is a different game (Sporting News, 2023-11-11T21:32:44+00:00)
2 now, and the debate with the Buckeyes and No. 1 Georgia is going to perk up more than ever. They are the three best teams in the country, but only one has
uncertainty with the head coach.

Whether they admit it or not, Michigan still needs Harbaugh for those moments, especially in two weeks.

The reality is he might not be present for that one either.
</item 70>

<item 71>
Michigan sign-stealing scandal news: Latest updates on possible Jim Harbaugh suspension from NCAA cheating investigation (Sporting News,
2023-11-06T23:58:04+00:00)
Public reprimands/fines are common penalties under B1G sportsmanship policy. — Adam Rittenberg (@ESPNRittenberg) November 9, 2023

Michigan's 10-page letter to Big Ten released

Yahoo Sports' Dan Wetzel posted Michigan's full 10-page letter to the Big Ten on X (formerly Twitter). The letter outlines Michigan's belief that the Big Ten
would be acting too hastily in suspending Harbaugh or levying any punishment toward Michigan before the completion of a full NCAA investigation.

In the letter, Michigan said the university has not had the chance to review all the evidence and respond, which oversteps the due process allowed for Michigan.
It said Pettiti should not have unilateral authority to punish violations without first having a completed investigation and then a response from the Compliance
and Reinstatement Committee, which would recommend penalties for rules violations.
</item 71>

<item 72>
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
</item 72>

<item 73>
Michigan’s Rivals Fume About Sign-Stealing: ‘This Was Damn Near Espionage’ (The New York Times, 2023-11-03T22:07:31+00:00)
The Big Ten does have the ability to act under its sportsmanship policy, but that doesn’t mean that it would want to act quickly or decisively before the NCAA
completes its entire investigation and allows Michigan a chance to respond to its findings.

This is an unprecedented situation; whatever Petitti decides to do (or not do) will set a precedent. The Big Ten itself doesn’t have investigators, so it needs
to rely upon the NCAA to do that part — and to determine who else was involved in the alleged scouting scheme. It’s not clear exactly what the coaches would
want the league to do to punish Michigan; banning the team from competing in the Big Ten championship, for example, would harm players who had nothing to do
with the sign-stealing apparatus.

A source briefed on the coaches’ call said Big Ten coaches are concerned about whether Michigan “should represent the Big Ten.”

“No matter what happens, if Michigan continues to move forward, the clouds will follow,” the source said.
</item 73>

<item 74>
NFL roundup: Pats hit new low as Lions reach 9-3 for first time since 1962 (The Guardian, 2023-12-03T22:11:26+00:00)
Brock Purdy threw for 314 yards and four touchdowns, Deebo Samuel scored three TDs and Eagles quarterback Jalen Hurts was checked for a concussion in the San
Francisco 49ers’ 42-19 win over Philadelphia on Sunday.

The fiery NFC championship game rematch also featured 49ers’ touchdowns on six straight possessions and a pair of ejections that showed this year’s title game
may not run again through Philly, after all.

49ers' Dre Greenlaw has been disqualified from tonight's game after this incident on the Eagles sideline. pic.twitter.com/2N4W17N8u9 — FOX Sports: NFL
(@NFLonFOX) December 3, 2023

The 49ers (9-3) left Philly 10 months ago battered in the NFC championship game behind injuries to Purdy and his backup, and they never stood a chance in a 31-7
loss.
</item 74>

<item 75>
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
</item 75>

<item 76>
Will the Panthers fire Frank Reich? Impatient David Tepper could turn up Carolina's hot seat with 'Thursday Night Football' loss (Sporting News,
2023-11-06T21:17:51+00:00)
Below is a year-by-year breakdown of Reich's record with the Colts and Panthers.

Year Team Record 2018 Colts 10-6 2019 Colts 7-9 2020 Colts 11-5 2021 Colts 9-8 2022 Colts* 3-5-1 2023 Panthers 1-7 Total Colts/Panthers 41-40-1

* Fired after Week 9 of the season.

Reich made it to the postseason in two of his first three seasons with the Colts. The team posted a record of 1-2 in three contests across those two
appearances.

It's also worth noting that Reich has had a different Week 1 starter at quarterback in each of his first six seasons as a head coach. He was hoping to break
that streak with Young in Carolina, but he will need to first make it to a seventh season with the club.
</item 76>

<item 77>
Michigan State hires Jonathan Smith: Oregon State coach leaves alma mater to take over Spartans job (CBSSports.com, 2023-11-25T23:12:10+00:00)
"He's been a part of championship staffs, coached in the College Football Playoff, and understands what's required to be successful at the highest level,
learning from some of the game's most successful coaches.

"On the field, his teams are tough and physical, yet innovative. This year, Oregon State has controlled the line of scrimmage on both sides of the ball, ranking
among the nation's leaders in both rushing offense and rushing defense. He's shown not only the ability to recruit talented student-athletes who fit his system,
but also to develop and maximize players once they're in the program."

Smith, a former quarterback for the Beavers from 1998-2001, exits Oregon State with a 34-35 overall mark as coach, including a 1-1 record in bowl games, as he
now joins the Big Ten coaching ranks.
</item 77>

<item 78>
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
</item 78>

<item 79>
Michigan didn't need Jim Harbaugh against Penn State, but Ohio State is a different game (Sporting News, 2023-11-11T21:32:44+00:00)
Will Harbaugh be on the field before the end of the regular season? The Big Ten made a Friday ruling that will prohibit Harbaugh from being on the field for the
final three regular-season games as a result of the in-person scouting and sign-stealing scandal that led to the resignation of staffer Conor Stallions. ESPN
reported there will be a temporary restraining order hearing on Nov. 17 that will answer that question.

On the field, Michigan answered the question about its College Football Playoff. Sort of. The Wolverines proved they could beat Penn State (8-2, 5-2) in a
20-yard box. Neither team completed a pass of 20 yards or more. Michigan trailed 3-0 after the first quarter. The Wolverines had eight carries for 10 yards, and
Penn State's Abdul Carter set the tone with a jarring hit on Blake Corum that nearly resulted in a fumble.

Michigan took control with the running game.
</item 79>

<item 80>
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
</item 80>

<item 81>
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
</item 81>

<item 82>
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
</item 82>

<item 83>
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
</item 83>

<item 84>
Michigan State hires Jonathan Smith: Oregon State coach leaves alma mater to take over Spartans job (CBSSports.com, 2023-11-25T23:12:10+00:00)
Four Pac-12 members -- USC, UCLA, Washington and Oregon -- will join Smith in the Big Ten next fall.

Can Smith turn the Michigan State program around?

The task now for Smith is revitalizing a Michigan State program that has struggled to field consistent success in recent seasons. After winning 10 or more games
six times in a span of eight seasons under former coach Mark Dantonio from 2010-17, the Spartans have only won more than seven games in a season once since
then, finishing 11-2 under Tucker's second year at the helm in 2021.

What seemed to be a breakthrough season for Tucker in 2021 earned him a 10-year contract extension worth $95 million, but Tucker's remaining tenure in East
Lansing was all downhill afterwards. Michigan State regressed to 5-7 in 2022 before allegations of sexual harassment surrounding Tucker against rape survivor
and activist Brenda Tracy emerged just two games into the 2023 season.
</item 84>

<item 85>
Will the Panthers fire Frank Reich? Impatient David Tepper could turn up Carolina's hot seat with 'Thursday Night Football' loss (Sporting News,
2023-11-06T21:17:51+00:00)
And I don't think that people are thinking that there's trouble right now with those people. But would anybody be surprised with anything that happens in
Carolina this year? No."

"One thing about David Tepper is he's not very patient and he does whatever he wants..



I don't think anybody would be surprised if anything happens in Carolina with the season that they're having" ~ @AdamSchefter #PMSLive
pic.twitter.com/PV7bfVUtHp — Pat McAfee (@PatMcAfeeShow) November 6, 2023

Schefter pointed to Carolina's firing of Rhule as an example of Tepper's impatience. There had been rumors that the owner was going to wait things out with
Rhule in the weeks leading up to his in-season departure. Instead, he decided to pull the trigger and part with him.
</item 85>

<item 86>
Michigan sign-stealing scandal winners and losers, from Jim Harbaugh to Tony Petitti (Sporting News, 2023-11-11T22:50:13+00:00)
On the other, he could emerge as a martyr for Michigan fans given there is reportedly no proof yet he knew of Stalion's rule-breaking.

Harbaugh will have a temporary restraining order hearing on Nov. 17, according to ESPN.com. That will determine whether he returns to the field for the final
two games against Maryland and No. 1 Ohio State.

It's an inescapable fact that Harbaugh is 31-3 – and Michigan is 35-3 – since the program bottomed out at the end of the 2020 season. While it is almost
impossible to quantify how much the alleged in-person scouting and sign-stealing would have helped the Wolverines over the past two seasons, it will be a tough
stain to erase knowing those violations are now on the Big Ten record.

What will Harbaugh do now? That is almost impossible to predict given his combative personality. Would he sign a contract extension and continue to fight the
NCAA? Will Harbaugh finally take the parachute to the NFL?
</item 86>

<item 87>
NFL power rankings: Lions, 49ers level up; Texans jump Bengals, Bills; Jets, Patriots sink to bottom for Week 11 (Sporting News, 2023-11-13T21:54:00+00:00)
32. Carolina Panthers 1-8 (31)

The Panthers found better defense and special teams to keep them in the game in Chicago, but Bryce Young still can't unlock a limited passing game and the
offensive line and running game hasn't help nearly enough.
</item 87>

<item 88>
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
</item 88>

<item 89>
Michigan didn't need Jim Harbaugh against Penn State, but Ohio State is a different game (Sporting News, 2023-11-11T21:32:44+00:00)
Michigan took control with the running game. Donovan Edwards ripped off a 22-yard run, and Corum added a 44-yard run to set up a 3-yard TD run. Edwards added a
22-yard TD run on the next drive. The same “Lightning and Lightning” tandem that burned the Nittany Lions last year staked Michigan to a 14-9 halftime lead.

MORE: Michigan got off easy with Harbaugh suspension

Then, the Wolverines took the ball out of quarterback J.J. McCarthy's hand. The Wolverines had 30 rushing attempts and zero passing attempts save a pass to
tight end Colston Loveland that drew a pass interference call. McCarthy finished 7 of 8 passing for 60 yards in what has to be the most conservative
play-calling victory in a top-10 showdown in recent memory. Was this old-school Big Ten football or Big Ten West football in disguise?
</item 89>

<item 90>
NFL roundup: Pats hit new low as Lions reach 9-3 for first time since 1962 (The Guardian, 2023-12-03T22:11:26+00:00)
Joe Flacco was 23 of 44 for 254 yards with two touchdowns and an interception in his Browns debut. Cleveland fell to 7-5 and occupy the final playoff spot in
the AFC.

Arizona Cardinals 24–10 Pittsburgh Steelers

James Conner ran for 105 yards and a pair of touchdowns against his old team as the Arizona Cardinals beat the Pittsburgh Steelers. It was the Cardinals’ first
win at Pittsburgh since 1969. Arizona took control late in the first half with a 99-yard touchdown drive that ended with a five-yard strike from Kyler Murray to
Trey McBride. The Cardinals led the rest of the way, enduring a pair of lengthy delays due to severe weather. The Steelers lost starting quarterback Kenny
Pickett to an ankle injury late in the first half and scored their only touchdown in garbage time.
</item 90>

<item 91>
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
</item 91>

<item 92>
Michigan sign-stealing scandal news: Latest updates on possible Jim Harbaugh suspension from NCAA cheating investigation (Sporting News,
2023-11-06T23:58:04+00:00)
Michigan noted that regarding evidence, there is nothing it is aware of to claim athletics personnel at Michigan committed the allegations, and said the rule
pertaining to electronic recordings of games is limited to the use of field equipment used by the institution for the game it is currently playing. It also said
Michigan has not seen much evidence, but rather has just heard a summary of second- and third-hand information.

The letter also said given the use of sign decoding by other universities, there is little reason to believe Stalions' efforts proved meaningful, and pointed
out the teams' average margin of victory since Stalions was suspended. It said the information obtained is "at most a reflection of information that teams can
and easily do obtain through public and permissible sources," and because of that it is "highly dubious" that Stalions' observations would have a "material
effect on the integrity of the competition."
</item 92>

<item 93>
Michigan State hires Jonathan Smith: Oregon State coach leaves alma mater to take over Spartans job (CBSSports.com, 2023-11-25T23:12:10+00:00)
"As I start this next chapter of my coaching career, I want to first thank all the players, coaches and colleagues I have worked with the last six years," Smith
said in a statement posted to social media. "I will cherish the memories and friendships. I also want to express my deepest gratitude to the people of
OSU/Corvallis for your support. When I first arrived as a freshman college student, I had no idea the opportunities and relationships this community would
provide for me and my family. The collective impact you have made makes it impossible to thank everyone individually, but I am forever grateful."

Oregon State left in dust amid uncertain future

Although Smith narrowly finished below. 500 in six seasons at the helm in Corvallis, Oregon, the Beavers made clear progress during the back half of his tenure.
Smith guided Oregon State to seven victories in 2021 after the program went 9-22 during his first three years.
</item 93>

<item 94>
Michigan sign-stealing scandal news: Latest updates on possible Jim Harbaugh suspension from NCAA cheating investigation (Sporting News,
2023-11-06T23:58:04+00:00)
The Big Ten is also asserting that they are not actually sanctioning Harbaugh, for whom they have no proof of wrongdoing, but that he "embodies the university
for purposes of its football program." pic.twitter.com/S660G6zQaa — Stewart Mandel (@slmandel) November 10, 2023

Jim Harbaugh suspended for remainder of regular season

The Big Ten has officially announced a suspension for Jim Harbaugh for the remaining three games of the regular season, including the Wolverines' rivalry game
with Ohio State.

In a statement, the Big Ten said Michigan conducted an "impermissible, in-person scouting operation over multiple years."

OFFICIAL: Jim Harbaugh suspended for the remaining three regular season games.
</item 94>

<item 95>
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
</item 95>

<item 96>
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
</item 96>

<item 97>
Michigan sign-stealing scandal winners and losers, from Jim Harbaugh to Tony Petitti (Sporting News, 2023-11-11T22:50:13+00:00)
According to ESPN.com, Stalions allegedly purchased tickets to more than 35 games involving Big Ten and potential CFP opponents.

Photos also surfaced os a person who resembles Stalions on the Central Michigan sideline during the Chippewas' opener against Michigan State on Sept. 1. Central
Michigan has yet to confirm who that person was, another layer to a wild Stalions story that has supplied endless meme and conspiracy material on social media.

Stalions might be the fall guy who saved Michigan from further wrath from the Big Ten and NCAA or the reason the Wolverines' national championship season
unraveled. That fate is yet to be determined, but at minimum he will be one of those controversial names we'll never forget, like Harvey Updyke and Nevin
Shapiro.

Jim Harbaugh

On one hand, Harbaugh could serve his second suspension within one season and further NCAA punishment could be coming down the line.
</item 97>

<item 98>
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
</item 98>

<item 99>
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
</item 99>

<item 100>
49ers cap space 2024: Why Chase Young trade may not be a pure rental for San Francisco (Sporting News, 2023-10-31T22:22:43+00:00)
That will make it easier for them to focus on keeping Young if they so desire.

As such, fans should view his acquisition as more of an audition to remain with the 49ers than a pure rental.
</item 100>
=== END ITEM id=ba82e32bb96da048 ===
