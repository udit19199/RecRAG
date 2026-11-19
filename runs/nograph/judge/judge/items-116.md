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

This file has 6 items.

=== ITEM id=6dffd169043178d5 ===
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
Michigan sign-stealing scandal news: Latest updates on possible Jim Harbaugh suspension from NCAA cheating investigation (Sporting News,
2023-11-06T23:58:04+00:00)
He initially told reporters he would speak during a scheduled hearing Friday, but then backtracked a bit and sounded as if he wasn't committed to that course,
pending legal advice.

Harbaugh said he will speak at the hearing on Friday. He said he’s looking for “due process.” Mentioned a civics class he took 40 years ago. “You’re innocent
until proven guilty. I’d like that opportunity.” — Nicole Auerbach (@NicoleAuerbach) November 13, 2023

Harbaugh clarifies that he’s not sure whether he’ll testify Friday. “That’s not my dance floor,” he says. “I’ve watched a lot of shows. I’ve watched Judge Judy
a lot.” — Austin Meek (@byAustinMeek) November 13, 2023

Sunday, Nov.
</item 1>

<item 2>
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
</item 2>

<item 3>
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
</item 3>

<item 4>
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
</item 4>

<item 5>
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
</item 5>

<item 6>
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
Michigan didn't need Jim Harbaugh against Penn State, but Ohio State is a different game (Sporting News, 2023-11-11T21:32:44+00:00)
Will Harbaugh be on the field before the end of the regular season? The Big Ten made a Friday ruling that will prohibit Harbaugh from being on the field for the
final three regular-season games as a result of the in-person scouting and sign-stealing scandal that led to the resignation of staffer Conor Stallions. ESPN
reported there will be a temporary restraining order hearing on Nov. 17 that will answer that question.

On the field, Michigan answered the question about its College Football Playoff. Sort of. The Wolverines proved they could beat Penn State (8-2, 5-2) in a
20-yard box. Neither team completed a pass of 20 yards or more. Michigan trailed 3-0 after the first quarter. The Wolverines had eight carries for 10 yards, and
Penn State's Abdul Carter set the tone with a jarring hit on Blake Corum that nearly resulted in a fumble.

Michigan took control with the running game.
</item 9>

<item 10>
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
</item 10>
=== END ITEM id=6dffd169043178d5 ===

=== ITEM id=b1ccd9b3ee45706a ===
Gold sentences and facts:
Sentence 0: As an added bonus, Harris is also the preferred goal-line back, scoring in three of those games.
  Fact 0.0: Harris is also the preferred goal-line back.
  Fact 0.1: Harris scored in three of those games.
Sentence 1: The Wildcats on their first drive put a touchdown on the board a seven play, 61 yard drive, thanks to a 25-yard dash before diving into the end zone by running back Jonah Coleman.
  Fact 1.0: The Wildcats scored a touchdown on their first drive.
  Fact 1.1: The Wildcats' first drive was a seven play drive.
  Fact 1.2: The Wildcats' first drive covered 61 yards.
  Fact 1.3: The Wildcats' first-drive touchdown came thanks to a 25-yard dash.
  Fact 1.4: The 25-yard dash was followed by a dive into the end zone.
  Fact 1.5: Running back Jonah Coleman dove into the end zone.

Retrieved items:
<item 1>
Fantasy Football RB PPR Rankings Week 14: Who to start, best sleepers at running back (Sporting News, 2023-12-07T22:09:00+00:00)
We still like Jaylen Warren better overall, but you can confidently take Najee Harris (vs. Patriots) out of the doghouse for good. Harris has been running with
purpose, and his role in the passing game continues to make him a high-ceiling RB2 in a run-first offense that will be without Kenny Pickett (ankle) this week.
New England might be banged up, but it's still a tough D to get chunk yardage against through the air. Look for Harris to see lots of screens and dump-downs in
a rough-and-tumble Thursday night game in Pittsburgh.

We love Baltimore's Keaton Mitchell (vs. Rams) and think he has the potential to smash Los Angeles at home this week.
</item 1>

<item 2>
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
</item 2>

<item 3>
No. 17 Arizona dominates No. 22 Utah, 42-18 (Rivals, 2023-11-18T23:45:32+00:00)
"Great win for the program," Arizona head coach Jedd Fisch said following the win. "Great win for our team, great win for our seniors, great win for Tucson,
appreciate Tucson coming out in full force, selling out the stadium. ... I think you might as well put the season tickets on sale for next year on Monday and
see if we can get six sell outs."

Leading the Wildcats was junior safety Dalton Johnson, logging 11 tackles as well as two tackles for loss in the UA’s routing.

"Dalton is a fantastic tackler," Fisch said. "When he's around the ball, he gets them down."

After failing continuously on trick plays involving receiver Tetairoa McMillan passing behind the scrimmage this season, Arizona head coach Jedd Fisch designed
it again — this time to perfection — with McMillan finding a wide-open fifth-year senior Michael Wiley to score a touchdown in his last game in Tucson.
</item 3>

<item 4>
NFL Fantasy Football Week 5 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-04T21:56:25+00:00)
At least all of them went to his wideouts. All were in the red zone, none were on deep throws and none were on WR screens.

Sit Him (Lineup Decisions) Projections powered by Sportsline Najee Harris RB PIT Pittsburgh • #22

Age: 25 • Experience: 3 yrs. Weekly Breakdown MATCHUP vs BAL BAL -4 O/U 38 OPP VS RB 12th PROJ PTS 9.3 RB RNK 30th YTD Stats RUYDS 210 REC 4 REYDS 34 TD 0
FPTS/G 7.1 The Steelers' lack of scoring opportunities really crushes the upside for Harris, who admittedly has improved this season and could be on his way to
being a reliable RB2. The Ravens have been pretty good against the run but they can give up some numbers. Harris is a volume-based back who is at best a low-end
starting option given everything that's going on.
</item 4>

<item 5>
No. 17 Arizona dominates No. 22 Utah, 42-18 (Rivals, 2023-11-18T23:45:32+00:00)
The UA then ran away from the Utes, outscoring them 14-8, only allowing a garbage time touchdown in the last minute to secure its 8th win of the season.

The last touchdown for Arizona came with backup quarterback Jayden de Laura hitting McMillan for a 51-yard touchdown.

"I wanted Jayden to go out with a touchdown pass this season," Fisch said. "I think that he deserves that."

When asked about whether the win against Utah, who has beat the Wildcats the last six meetings, signified anything further about the program, Fisch said that
the win signified that they are having a great year.

"[It signifies] that we're not gonna back down from anybody, and we feel really good about trying to continue to improve, get more physical, get tougher," Fisch
said.
</item 5>

<item 6>
No. 17 Arizona dominates No. 22 Utah, 42-18 (Rivals, 2023-11-18T23:45:32+00:00)
Saturday’s game between Arizona and Utah was many things for the Wildcats: Senior Day, Military Appreciation Day, and their last home game of the season. With
the UA needing a win to keep its own destiny alive for a Pac-12 Championship appearance, it handled the pressure once again against a ranked team.

The Wildcats not only got halfway through en route to winning out but did it in statement fashion Saturday at Arizona Stadium in front of a sold out stadium,
blowing out No. 22 Utah, 42-18, extending their winning streak to five games, as well as marking their first win over the Utes since 2015. It is also the first
time the UA took down four ranked teams in a season since 1989, with it being the only team in the country this season to do so.

"Great win for the program," Arizona head coach Jedd Fisch said following the win.
</item 6>

<item 7>
No. 17 Arizona dominates No. 22 Utah, 42-18 (Rivals, 2023-11-18T23:45:32+00:00)
Following the trickery, another huge roar of the Wildcat faithful ensued following a blocked punt and score by Anthony Ward, the first Arizona blocked punt and
score since Nov. 13, 2021.

The Wildcats’ momentum was rolling like a well-oiled machine on both ends of the ball in the first half, scoring offensive touchdowns on three of four drives,
led by Michael Wiley tallying two of them, and outgaining the Utes by 98 yards on the ground, as well as forcing the Utes to two three-and-outs, three punts and
a red zone interception, headlined by captain linebacker Jacob Manu, the first Arizona linebacker interception since Anthony Pandy against USC in 2021. With
rain starting to pour late in the first half in a rare Arizona “rain game”, the overcast didn’t hinder the Zona Zoo from leaving the stadium, nor the Wildcats
from letting up their progression, resulting in the UA taking staggering 28-7 lead at halftime.
</item 7>

<item 8>
Fantasy Football RB Rankings Week 14: Who to start, best sleepers at running back (Sporting News, 2023-12-07T22:07:00+00:00)
We'll talk more about Derrick Henry (concussion) in the injury section, but we can say with confidence here and now that Tyjae Spears (@ Dolphins) should be the
hottest waiver-wire pickup of Week 14. Tennessee is a run-first offense and King Henry might be sidelined with a head injury, so Spears could very well pick up
where he left off against Indy last weekend. The young back averaged 4.7 yards per carry (just shy of his 5.1 YPC entering the week) en route to 88 yards.

Another backup getting a boost this week is the Patriots' Ezekiel Elliott (@ Steelers), who will assume lead-back duties after Rhamondre Stevenson went down
with a pretty bad ankle injury in Week 13. Zeke might put in a vintage performance considering James Conner just went back to Pittsburgh and hung 105 yards and
two TDs on his old squad.

Could this be the weekend that Keaton Mitchell (vs.
</item 8>

<item 9>
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
</item 9>

<item 10>
No. 17 Arizona dominates No. 22 Utah, 42-18 (Rivals, 2023-11-18T23:45:32+00:00)
"I don't think the rain effected us at all," Fisch said.

With the rain starting to clear up to start the second half, overtime it was clear that the Wildcats’ defense was holding a cloud over the Utes’ heads. Arizona
got off to a slow start but picked up much of its thunderous momentum after junior safety Gunner Maldonado broke up a pass on fourth and three. Since the
fourth-down stop, the UA logged an interception and another turnover on downs.

Although the defense was surging, the Wildcats not scoring since the first drive of the second quarter — their offense seemed to have lost its touch.

But then “Swag Surfin” by Fast Life Yungstaz came on, signaling the fourth quarter, the quarter in which going into the game, Arizona was outscoring opponents
90-37. The Wildcats on their first drive put a touchdown on the board a seven play, 61 yard drive, thanks to a 25-yard dash before diving into the end zone by
running back Jonah Coleman.
</item 10>
=== END ITEM id=b1ccd9b3ee45706a ===

=== ITEM id=18f9c37c336a49e9 ===
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
  Fact 2.0: The earlier actions were focused on concerns about the spread of illegal content.
  Fact 2.1: The earlier actions were focused on concerns about disinformation.
  Fact 2.2: The illegal content and disinformation are related to the Israel-Hamas war.

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
</item 1>

<item 2>
Elon Musk’s X faces first DSA probe in EU over illegal content risks, moderation, transparency and deceptive UX design (TechCrunch, 2023-12-18T13:04:56+00:00)
EU regulators are also set to assess Musk & co’s claims that replacing a formal content moderation function with crowdsourced opinions is a valid approach to
tackling diverse information risks; and, consequently, to look into how much Musk’s gutting of content moderation teams and processes has cost X’s users in
terms of harm exposure.

To say the Commission’s investigation looks fascinating and timely is an understatement.

EU lawmakers can certainly bask in the glow of being prepared for the possibility of an erratic billionaire running a wrecking ball through a major social
platform. But of course they will need to follow through — and it’s clear the real work is just starting. So this probe looks set to be a test of how much
stomach the EU has to actually enforce the rules it loves to lay down on paper.

The investigation may also test Musk’s mettle for what could be an expensive head-on clash with EU regulators.
</item 2>

<item 3>
Meta to offer ad-free subscription in Europe in bid to keep tracking other users (TechCrunch, 2023-10-30T13:52:00+00:00)
These included findings made by the Court of Justice of the European Union when delivering judgment on July 4, 2023 in a case in which the Court examined the
legal bases on which Meta’s processing of users’ data for behavioural advertising purposes is grounded.

“Acting in consultation with its fellow European supervisory authorities, the DPC has been engaged in a detailed regulatory assessment of the consent-based
model since it was first proposed by Meta in July. That exercise is being led by the DPC, reflecting its position as Lead Supervisory Authority for Facebook and
Instagram in Europe. The exercise has not yet concluded, and no findings have been made to date. It is due to be completed shortly, at which point the DPC will
notify Meta if it considers that the manner in which its new user offerings are to be implemented is compatible with Meta’s obligations under GDPR.”

So Meta’s move to offer users a subscription or tracking hasn’t been signed off by data protection authorities as yet — and it’s clear that further regulatory
interventions could follow.
</item 3>

<item 4>
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
</item 4>

<item 5>
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
</item 5>

<item 6>
Meta to offer ad-free subscription in Europe in bid to keep tracking other users (TechCrunch, 2023-10-30T13:52:00+00:00)
The Commission is the sole enforcer of the DMA.

Meta is already in the Commission’s sights over its approach to the DSA — with the EU’s executive, in recent days, requesting more information from the tech
giant vis-à-vis its approach to content threats flowing from the Israel-Hamas war and about election security issues. But it remains to be seen whether the EU
will apply the same close scrutiny to Meta’s ad tracking offer.

In its blog post Meta claims the choice to offer people to pay it for their privacy or else agree to being tracked “balances the requirements of European
regulators while giving users choice and allowing Meta to continue serving all people in the EU, EEA and Switzerland”. But, well, it would say that.

The subscription is also only being offered to people aged 18 and up — which raises questions over how it will comply with requirements in the DSA and DMA not
to processing children’s data for ad targeting.
</item 6>

<item 7>
Amazon’s iRobot purchase sucks up formal competition concerns in EU (TechCrunch, 2023-11-27T23:29:10+00:00)
iRobot, which faces intense competition from other vacuum cleaner suppliers, offers practical and inventive products. We believe Amazon can offer a company like
iRobot the resources to accelerate innovation and invest in critical features while lowering prices for consumers.

Late last year the EU settled with Amazon over two earlier competition probes — one delving into antitrust concerns about Amazon’s use of merchant data to
bolster its own retail business; and a second one looking at how it operated the “Buy Box” and its Prime loyalty program.

In those cases Amazon agreed a series of commitments to end the probes — including saying it would stop using non-public data from its marketplace sellers and
boost transparency for merchants; and pledging to treat all sellers equally, regardless of whether they pay for its logistics services.
</item 7>

<item 8>
After warning, the EU opens investigation into X over alleged disinformation (TechCrunch, 2023-10-13T00:38:56+00:00)
X owner Elon Musk last projected revenues of about $3 billion for this year. It’s anyone’s guess if that’s accurate now, but as an example, that would result in
a fine of up to $180 million.

Indirectly, an investigation will also be a key way for the public to have significantly more transparency around how X is being used (and abused): since Musk
took Twitter private, the company no longer produces quarterly earnings reports and is not held to account as closely as a result.

EU officials confirmed to TechCrunch that an investigation has not been taken at this point. Clearly, though, yesterday’s development (and the pace at which
regulators are moving) is a strong indication of the bloc’s direction of travel.

The Commission said it is looking at X’s compliance with the DSA across a number of areas — including with regard to its policies and practices regarding
notices on illegal content; complaint handling; risk assessment and measures to mitigate the risks identified.
</item 8>

<item 9>
Elon Musk’s X faces first DSA probe in EU over illegal content risks, moderation, transparency and deceptive UX design (TechCrunch, 2023-12-18T13:04:56+00:00)
Per the Commission, the EU’s investigation of X focuses on the following areas and issues:

Compliance with DSA obligations related to countering the dissemination of illegal content in the EU — “ notably in relation to the risk assessment and
mitigation measures adopted by X to counter the dissemination of illegal content in the EU, as well as the functioning of the notice and action mechanism for
illegal content in the EU mandated by the DSA, including in light of X’s content moderation resources

— “ The effectiveness of measures taken by X to combat info manipulation — with a focus on the effectiveness of its “Community Notes” system of crowdsourcing
fact-checking to aggregated views of users, who may vote to label disputed tweets with additional context.
</item 9>

<item 10>
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
</item 10>
=== END ITEM id=18f9c37c336a49e9 ===

=== ITEM id=24f9e37a1cf6f03a ===
Gold sentences and facts:
Sentence 0: Wang testified that he heard Bankman-Fried tell another trader that it was okay to continue withdrawing money from the account as long as the amount was less than FTX’s total trading revenue.
  Fact 0.0: Wang testified that he heard Bankman-Fried tell another trader that it was okay to continue withdrawing money from the account.
  Fact 0.1: Bankman-Fried told another trader that it was okay to continue withdrawing money from the account as long as the amount was less than FTX's total trading revenue.
Sentence 1: During cross-examination, Huang said Paradigm pressed Bankman-Fried on the board issue and was told he didn't want investors as directors but he did plan on having a board with experts.
  Fact 1.0: Huang testified during cross-examination.
  Fact 1.1: Huang said Paradigm pressed Bankman-Fried on the board issue.
  Fact 1.2: Paradigm was told that Bankman-Fried did not want investors as directors.
  Fact 1.3: Bankman-Fried planned on having a board with experts.
Sentence 2: The prosecution painted Bankman-Fried as someone who knowingly committed fraud to achieve great wealth, power and influence, while the defense countered that the FTX founder acted in good faith, never meant to commit fraud or steal and basically got in over his head.
  Fact 2.0: The prosecution portrayed Bankman-Fried as someone who knowingly committed fraud to achieve great wealth, power and influence.
  Fact 2.1: The defense argued that the FTX founder acted in good faith.
  Fact 2.2: The defense argued that the FTX founder never meant to commit fraud or steal.
  Fact 2.3: The defense argued that the FTX founder basically got in over his head.

Retrieved items:
<item 1>
In the end, the FTX trial was about the friends screwed along the way (The Verge, 2023-10-26T12:00:00+00:00)
That was before he realized FTX had stolen customer funds — at which point, he resigned.

Gary Wang, the co-founder of FTX and Alameda Research and another college roommate. He took the stand after Yedidia and sang like a canary about the secret
privileges he’d created for Alameda in the FTX code.

Nishad Singh, a longtime family friend, who copped to what sounded an awful lot like classic embezzlement. He said he’d backdated transactions on FTX to help
create doctored balance sheets to show to investors. He pleaded guilty to six charges, including wire fraud, money laundering, and conspiracy counts around his
political donations.

Caroline Ellison, Bankman-Fried’s ex-girlfriend. She told us about the seven phony balance sheets she’d created. She was clear: Alameda took customer funds;
Bankman-Fried knew about it and even invested them against her advice.

Joe Bankman, Bankman-Fried’s father.
</item 1>

<item 2>
Is Sam Bankman-Fried a bad ‘man’ or a good ‘boy’? Lawyers swap opening statements before first witnesses take the stand (Fortune, 2023-10-04T23:32:53+00:00)
FTX was no Ponzi scheme. It was a “very innovative, successful company.” And the business practices between the two were reasonable, he argued, claiming that
Alameda acted legally as an FTX customer, payment processor, and market maker, or financial entity that acts as a trading partner for customers looking to buy
and sell cryptocurrencies.

In an analogy he employed throughout his opening statement, he said that “working at a startup is like building a plane as you’re flying it” and that businesses
sometimes fail. In fact, he specifically pointed the finger at Ellison, the former CEO of Alameda, who, he said, did not adequately protect her hedge fund from
the inherent risk of the crypto markets.

When the walls came closing in and the aforementioned plane approached the “eye of the storm,” Bankman-Fried did not act like someone who was guilty. Rather, he
was willing to give up his personal wealth to make customers whole, Cohen argued.
</item 2>

<item 3>
SBF’s trial starts soon, but how did he — and FTX — get here? (TechCrunch, 2023-10-01T14:00:29+00:00)
regulators and government officials, many of whom he donated to.

Bankman-Fried was even compared to Warren Buffet and many called him the white horse of crypto (TechCrunch never did, for what it’s worth).

But in early November 2022, that all changed.

FTX’s collapse

Concerns surrounding FTX’s liquidity grew after CoinDesk published a copy of Alameda’s balance sheet, showing the firm held $14.6 billion in assets and $8
billion in liabilities as of June 30, 2022.

But there was a problem: The report showed Alameda’s largest asset was $3.66 billion of “unlocked FTT” and $2.16 billion of “FTT collateral.” FTT was the token
behind FTX.

The balance sheet showed that the $5.82 billion in FTT tokens that Alameda owned was 193% higher than the total FTT market cap, which was about $3 billion at
the time.
</item 3>

<item 4>
The FTX trial is bigger than Sam Bankman-Fried (The Verge, 2023-09-28T12:00:00+00:00)
Bankman-Fried is accused of using “billions of dollars in misappropriated FTX customer deposits” to help buy more than $200 million of real estate for himself,
make billions of dollars of investments for his own interest, and repay Alameda’s lenders, according to the indictment, Additionally, Bankman-Fried allegedly
used more than $100 million of customer funds to make political contributions; prosecutors can show evidence of those contributions in this trial, even though
they aren’t part of the charges brought. A second trial is scheduled for March 2024, with additional charges.

Before his fall, Bankman-Fried made himself out to be the Good Boy of crypto — the trustworthy face of a sometimes-shady industry. He was also very interested
in publicity, sitting for many interviews both before and after the fall of FTX. The quick rise of FTX as an industry force was at least in part due to
Bankman-Fried’s appetite for attention.
</item 4>

<item 5>
SBF’s trial starts soon, but how did he — and FTX — get here? (TechCrunch, 2023-10-01T14:00:29+00:00)
SBF’s trial has started, this is how he and FTX got here

The highly anticipated criminal trial for Sam Bankman-Fried, former CEO of bankrupt crypto exchange FTX, started Tuesday to determine whether he’s guilty of
seven counts of fraud and conspiracy. And as one former federal prosecutor put it: “The odds seem to be stacked against him at this point.”

The 31-year-old co-founded FTX in 2019; within a few years the once third-largest crypto exchange’s valuation hit $32 billion at its peak. It’s now trying to
claw back any funds to distribute to creditors.

But how did the once third-largest crypto exchange get here?

Before FTX, Bankman-Fried co-founded crypto-trading firm Alameda Research in 2017. He co-founded FTX in 2019 as a complement to Alameda, to help bring in
revenue and liquidity for the trading arm.
</item 5>

<item 6>
The FTX trial is bigger than Sam Bankman-Fried (The Verge, 2023-09-28T12:00:00+00:00)
We have some inkling of what the government will be arguing thanks to the filings prosecutors are making in this case. They argue that Bankman-Fried lied about
consumer protection and that Bankman-Fried’s statements that FTX was “avoiding or managing conflicts of interest” and that “as a general principle FTX
segregates customer assets from its own assets across our platforms” were lies, in a superseding indictment filed August 14th.

The indictment also says that there were special features in FTX’s code that “permitted Alameda to spend and withdraw unlimited amounts of money from FTX,”
which were created at Bankman-Fried’s direction. That effectively exempted Alameda from the kinds of risk management other customers faced. At the same time,
the government alleges, “Bankman-Fried publicly and repeatedly asserted that Alameda did not have privileged access to FTX.”
</item 6>

<item 7>
SBF’s trial starts soon, but how did he — and FTX — get here? (TechCrunch, 2023-10-01T14:00:29+00:00)
U.S. Attorney Damian Williams called Bankman-Fried’s alleged crimes “one of the biggest financial frauds in American history,” in a press conference.

The aftermath

FTX co-founder and former CTO Gary Wang, and Alameda Research’s former CEO, Caroline Ellison, both pleaded guilty in December 2022 to federal criminal charges
in relation to the FTX collapse. They’re also facing civil penalties from the U.S. Securities and Exchange Commission (SEC) and Commodity Futures Trading
Commission (CFTC) alongside the criminal charges. Wang and Ellison plan to cooperate with prosecutors and will be major witnesses in the trial, given their
close ties to Bankman-Fried, FTX and Alameda.

In January, Bankman-Fried pleaded not guilty to all counts, which include wire fraud, conspiracy to commit money laundering and conspiracy to misuse customer
funds. He could face up to 115 years in jail if convicted on all charges.
</item 7>

<item 8>
Sam Bankman-Fried was a terrible boyfriend (The Verge, 2023-10-10T23:50:21+00:00)
FTX was a fraud “from the start,” the Securities and Exchange Commission alleges — with a “multi-billion-dollar deficiency caused by his own misappropriation of
customer funds.” Follow along for all the latest news and regular updates from the trial.

The day started off promisingly for the defense as it cross-examined Gary Wang, the chief technology officer of FTX and co-owner of both FTX and Alameda.
Christian Everdell, one of Bankman-Fried’s defense attorneys, couldn’t undo the damage of last week’s code review. But he managed to shake the rust off long
enough to make Wang sound less reliable, drowning the jury in confusing technicalities.

Last week, Wang testified that Alameda got access to a special credit line and an option to take its balance into the negative without triggering liquidation —
something he alleged other accounts at FTX didn’t get. Everdell tried to undermine this claim by talking about the spot margin program, which let users lend
each other assets for margin trading.
</item 8>

<item 9>
Sex, Signal messages, and sabotaging FTX's code: SBF execs and Bahamas roommates tell all in court (Cnbc | World Business News Leader,
2023-10-06T20:17:00+00:00)
Additionally, he was promised that Alameda had no preferential treatment on the FTX platform, even though the hedge fund was one of its top traders. Huang said
he was concerned about FTX's lack of a board of directors, but he eventually invested anyway. During cross-examination, Huang said Paradigm pressed
Bankman-Fried on the board issue and was told he didn't want investors as directors but he did plan on having a board with experts. — CNBC's Dawn Giel
contributed to this report.
</item 9>

<item 10>
The FTX trial is bigger than Sam Bankman-Fried (The Verge, 2023-09-28T12:00:00+00:00)
According to the SEC, FTX was a fraud “from the start,” diverting customers’ funds to Alameda.

Sam Bankman-Fried, the founder of failed cryptocurrency exchange FTX, is on trial for seven counts of wire fraud and conspiracy. FTX was a fraud “from the
start,” the Securities and Exchange Commission alleges — with a “multi-billion-dollar deficiency caused by his own misappropriation of customer funds.” Follow
along for all the latest news and regular updates from the trial.

“Really just old-fashioned embezzlement.”

What’s left of FTX is now being led by John J. Ray III — you may remember him as the guy who cleaned up Enron and who also said FTX is worse than Enron. Just
before the trial began, the FTX lawyers filed suit against Bankman-Fried’s parents, saying they should give back millions of dollars they got from their kid.
Ray has also referred to Bankman-Fried’s conduct as “really just old-fashioned embezzlement.”
</item 10>
=== END ITEM id=24f9e37a1cf6f03a ===

=== ITEM id=1c127c0a9033e83a ===
Gold sentences and facts:
Sentence 0: When Google’s search head Prabhakar Raghavan testified in court on October 28, he revealed that the tech giant had paid $26.3 billion in 2021 to multiple browsers, phones and platforms, from companies including Apple, Samsung and Mozilla, The Verge reports.
  Fact 0.0: Google's search head Prabhakar Raghavan testified in court on October 28.
  Fact 0.1: Prabhakar Raghavan revealed in court testimony that Google paid $26.3 billion in 2021.
  Fact 0.2: The $26.3 billion was paid to multiple browsers, phones and platforms.
  Fact 0.3: The payments were made to companies including Apple, Samsung and Mozilla.
  Fact 0.4: The Verge reports this.
Sentence 1: It will attempt to argue that Google restrains competition within two separate markets, including the distribution of apps to Android users and the market for payment processing solutions for content inside Android apps.
  Fact 1.0: The case (the sentence's subject, referred to as 'It') will attempt to argue that Google restrains competition within two separate markets.
  Fact 1.1: One of the two markets is the distribution of apps to Android users.
  Fact 1.2: The other market is the market for payment processing solutions for content inside Android apps.
Sentence 2: The case, filed by Arkansas-based publisher Helena World Chronicle, argues that Google “siphons off” news publishers’ content, their readers and ad revenue through anticompetitive means.
  Fact 2.0: The case was filed by Arkansas-based publisher Helena World Chronicle.
  Fact 2.1: The case argues that Google 'siphons off' news publishers' content through anticompetitive means.
  Fact 2.2: The case argues that Google 'siphons off' news publishers' readers through anticompetitive means.
  Fact 2.3: The case argues that Google 'siphons off' news publishers' ad revenue through anticompetitive means.

Retrieved items:
<item 1>
5 things we learned so far about the Google antitrust case (TechCrunch, 2023-10-31T02:30:32+00:00)
Google CEO Sundar Pichai testified Monday in the U.S. government’s antitrust trial against the company. The executive defended Google’s business tactics,
including its deal with Apple and other partners to make Google the default search engine.

The lawsuit stems from a 2020 antitrust claim over Google’s dominance in the online search market. The claim is separate from one filed in January by the
Department of Justice, along with eight states including New York, California and Colorado, that aims to “halt Google’s anticompetitive scheme, unwind Google’s
monopolistic grip on the market and restore competition to digital advertising.”

The government has argued that Google uses its platforms and deals with partners to block out any competition in search or advertising, thus hindering
competitors from accessing the data they’d need to improve their products.

Google argues that it’s just doing business. Everybody wants Google as the default engine because it’s the best. That doesn’t make its actions illegal, the
company says.
</item 1>

<item 2>
News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line (TechCrunch, 2023-12-15T17:56:02+00:00)
SGE, it argues, offers web searchers a way to seek information in a conversational mode, but ultimately keeps users in Google’s “walled garden” as it
“plagiarizes” their content. Publishers also can’t block SGE because it uses the same web crawler as Google’s general search service, GoogleBot.

Plus, it says Google’s Bard AI was trained on a dataset that included “news, magazine and digital publications,” citing both a 2023 report from the News Media
Alliance and a Washington Post article about AI training data for reference. (The Post, which worked with researchers at the Allen Institute for AI, had found
that News and Media sites were the third largest category of AI training data.)

The case points to other concerns, too, like changing AdSense rates and evidence of improper spoliation of evidence on Google’s part, by its destruction of chat
messages — an issue raised in the recent Epic Games lawsuit against Google over app store antitrust issues, which Epic won.
</item 2>

<item 3>
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
</item 3>

<item 4>
More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo (TechCrunch, 2023-11-13T14:37:44+00:00)
Before we go into some of these tidbits…

Why the Google vs. U.S. antitrust case matters

The government has argued that Google uses its platforms and deals with partners to block out any competition in search or advertising, thus hindering
competitors from accessing the data they’d need to improve their products. If Judge Amit Mehta rules against Google, the search giant may have to change its
behavior and share its APIs with third-party developers. It may also be banned from making anticompetitive and exclusive deals with smartphone and computer
manufacturers and wireless carriers. Google might end up having to turn over all or most of the data it has collected to other search engines so they can
improve their products and attract more users. The DOJ has said that Google gets 16 times more data than Bing does everyday. Enforcers want to show that
antitrust law remains relevant and that even though Google is basically the God of the internet, it’s still no match for the U.S. law.
</item 4>

<item 5>
5 things we learned so far about the Google antitrust case (TechCrunch, 2023-10-31T02:30:32+00:00)
These are exactly the arguments that other search engine companies, like DuckDuckGo, Brave or Microsoft’s Bing, make when they accuse Google of being
anticompetitive by making deals with Apple and others. The DOJ doubled down on this, saying Google has become the monopoly it denounced years ago.

What does it all mean?

The case is expected to continue for several weeks, bringing to a head one of the biggest fights in tech antitrust since the U.S. took Microsoft to trial in the
1990s.

If the judge rules against Google, the outcome could look a lot like the Microsoft deal, in which the computer company was required to change its behavior and
share its APIs with third-party developers. Microsoft was also banned from making anticompetitive and exclusive deals with computer manufacturers.

Google might end up having to turn over all or most of the data it has collected to other search engines so they can improve their products and attract more
users. The DOJ has said that Google gets 16 times more data than Bing does everyday.
</item 5>

<item 6>
News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line (TechCrunch, 2023-12-15T17:56:02+00:00)
A Google spokesperson offered a statement on the lawsuit, saying “This lawsuit is meritless. People have many ways to access information and news content today
– through publishers’ websites, dedicated apps, social media platforms, print papers and more. Google links people to publishers’ websites more than 24 billion
times each month – at no cost to them.”

The complaint is available below.

Helena World Chronicle, LLC v. Google LLC and Alphabet Inc by TechCrunch on Scribd

Editor’s note: This post was updated after publication with a statement provided by Google on Friday evening.
</item 6>

<item 7>
More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo (TechCrunch, 2023-11-13T14:37:44+00:00)
law. The Google outcome could also have a ripple effect on other Big Tech cases. The FTC sued Amazon in September for using anticompetitive and unfair
strategies to illegally maintain its monopoly power. The DOJ has been investigating Apple for years over the company’s policy for third-party apps on its
devices and whether it unfairly favors its own products. There’s an ongoing case between the FTC and Facebook, wherein the agency calls on Facebook to sell
Instagram and WhatsApp.

This isn’t Google’s only antitrust case in trial right now. The search engine giant last week settled a separate antitrust lawsuit with dating site Match Group.
On November 6, Google went to trial with Fortnite maker Epic Games. The latter hopes to prove that Google engages in anticompetitive behavior with regard to its
Android app store, Google Play, and its commission structure.

Now, onto the roundup!
</item 7>

<item 8>
5 things we learned so far about the Google antitrust case (TechCrunch, 2023-10-31T02:30:32+00:00)
The DOJ has said that Google gets 16 times more data than Bing does everyday.

The Google outcome could also have a ripple effect on other Big Tech cases. The FTC sued Amazon in September for using anticompetitive and unfair strategies to
illegally maintain its monopoly power. The DOJ has been investigating Apple for years over the company’s policy for third-party apps on its devices and whether
it unfairly favors its own products. There’s an ongoing case between the FTC and Facebook, wherein the agency calls on Facebook to sell Instagram and WhatsApp.

Enforcers will want to show that antitrust law is still relevant and can successfully take on the biggest, most powerful companies in the world.
</item 8>

<item 9>
News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line (TechCrunch, 2023-12-15T17:56:02+00:00)
In addition to damages, the suit is asking for an injunction that would require Google to obtain consent from publishers to use their website data to train its
general artificial intelligence products including Google’s own and those of rivals. It also asks Google to allow publishers who opt out of SGE to still show up
in Google search results, among other things.

The U.S. lawsuit follows an agreement Google reached last month with the Canadian government which would see the search giant paying Canadian media for use of
their content. Under the terms of the deal, Google will provide $73.5 million (100 million Canadian dollars) every year to news organizations in the country,
with funds distributed based on the news outlets’ headcount. Negotiations with Meta are still unresolved, though Meta began blocking news in Canada in August,
in light of the pressure to pay for the content under the new Canadian law.

The case also arrives alongside the filing of the U.S.
</item 9>

<item 10>
News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line (TechCrunch, 2023-12-15T17:56:02+00:00)
A new class action lawsuit filed this week in the U.S. District Court in D.C. accuses Google and parent company Alphabet of anticompetitive behavior in
violation of U.S. antitrust law, the Sherman Act, and others, on behalf of news publishers. The case, filed by Arkansas-based publisher Helena World Chronicle,
argues that Google “siphons off” news publishers’ content, their readers and ad revenue through anticompetitive means. It also specifically cites new AI
technologies like Google’s Search Generative Experience (SGE) and Bard AI chatbot as worsening the problem.

In the complaint, Helena World Chronicle, which owns and publishes two weekly newspapers in Arkansas, argues that Google is “starving the free press” by sharing
publishers’ content on Google, losing them “billions of dollars.”

In addition to new AI technologies, the suit points to Google’s older question-and-answer technologies, like the “Knowledge Graph” launched in May 2012, as part
of the problem.
</item 10>
=== END ITEM id=1c127c0a9033e83a ===

=== ITEM id=39d894817df0095a ===
Gold sentences and facts:
Sentence 0: Of Miami's five remaining opponents, four have pass defenses ranked in the league's top eight.
  Fact 0.0: Miami has five remaining opponents.
  Fact 0.1: Four of Miami's five remaining opponents have pass defenses ranked in the league's top eight.
Sentence 1: 152.67 -- After missing last week's game, Tyreek Hill now needs to average almost 153 yards per game in his final three to reach his stated goal of 2,000 receiving yards this season.
  Fact 1.0: Tyreek Hill missed last week's game.
  Fact 1.1: Tyreek Hill now needs to average almost 153 yards per game in his final three games.
  Fact 1.2: The figure cited is 152.67 yards per game.
  Fact 1.3: Tyreek Hill's stated goal is 2,000 receiving yards this season.

Retrieved items:
<item 1>
N.F.L. Executives on the Biggest Questions Facing Super Bowl Contenders (The New York Times, 2023-10-16T23:58:53+00:00)
Six of the 17 won anyway. The Dolphins were the only one of those 223 teams to win and cover the spread, which was 13.5 for this game.

• Six catches for 163 yards and a touchdown against Carolina left the Dolphins’ Tyreek Hill with 812 yards through the first six games. That ranks second in NFL
history to Hall of Famer Don Hutson, who had 819 yards through six with Green Bay in 1942.
</item 1>

<item 2>
Fantasy Football Week 16 Wide Receiver Preview: Waiver adds, DFS plays, and more (CBSSports.com, 2023-12-19T20:51:32+00:00)
Weekly Breakdown MATCHUP vs DAL MIA -1.5 O/U 51 OPP VS WR 7th PROJ PTS 14.2 WR RNK 16th YTD Stats REC 71 TAR 99 REYDS 964 TD 4 FPTS/G 14.8 We expect Hill to
return this week, which should drive people off of Waddle. But there is at least a chance that Hill won't be 100% and Waddle will still be Tua Tagovailoa's top
target. Waddle has been the quintessential DFS tournament play as of late with three of seven games over 19 PPR Fantasy points, topping 26 twice.

WR Preview Heath's Projections

My full set of Week 16 Fantasy Football projections for every position are now available on SportsLine. Find out which of my favorite plays are projected to
score higher than consensus rankings and which don't live up to their draft hype, at least in Week 16.
</item 2>

<item 3>
N.F.L. Executives on the Biggest Questions Facing Super Bowl Contenders (The New York Times, 2023-10-16T23:58:53+00:00)
Receiving Yards Through Six NFL Games Rk Player Season Yds 1 Don Hutson 1942 819 2 Tyreek Hill 2023 812 3 Wes Welker 2011 785 4 Elroy Hirsch 1951 766 5 Santana
Moss 2005 743 6 DeAndre Hopkins 2015 726 7 Raymond Berry 1960 725 8 Jerry Rice 1986 723 9 Bob Boyd 1954 718 10 Bobby Mitchell 1962 714

• Jared Goff is enjoying the second-best start to a season for his career from a statistical standpoint, behind only his 2018 Super Bowl season with the Rams.
With the Lions at 5-1 and Goff heading toward a likely contract extension, could another team come after his injured backup, former University of Tennessee
quarterback Hendon Hooker?
</item 3>

<item 4>
MVPs for all 16 AFC teams at the midpoint (Yardbarker, 2023-11-09T21:51:02+00:00)
MIAMI DOLPHINS | WR Tyreek Hill: Arguably the best wide receiver in the NFL, Hill has somehow reached a new level of dominance in his age-29 season. The
seven-time Pro Bowler leads the NFL in receiving yards (1,076) and TD catches (eight). As long as the Dolphins maintain their success on offense down the
stretch, Hill should earn Offensive Player of the Year honors. He's on pace to break Calvin Johnson's season record for receiving yards (1,964).



NEW YORK JETS | LB Quincy Williams: While his brother, Quinnen, may be the better-known Jets defender in the family, Williams — the AFC Defensive Player of the
Month for October — is also making a name for himself. Aside from being the fifth-highest-rated linebacker in the NFL by Pro Football Focus, Williams is tied
for 17th in tackles (74) and tied for ninth in TFLs (nine) among all defenders.
</item 4>

<item 5>
Fantasy Football WR PPR Rankings Week 14: Who to start, best sleepers at wide receiver (Sporting News, 2023-12-07T22:11:00+00:00)
The league's leading receiver, Tyreek Hill (vs. Titans in Week 14), stands as the unquestioned WR1 for Week 14 after torching the Commanders to the tune of five
catches, 157 yards, and two TDs. CeeDee Lamb (vs. Eagles) and the Cowboys look to avenge their road loss to the Eagles in Week 9, as Dallas' WR1 routinely has
double-digit reception potential.

Amon-Ra St. Brown (@ Bears) was relatively quiet this past week, but he's likely to rebound against a Bears defense that he totaled nine catches for 95 yards
against in Week 11. A.J. Brown (@ Cowboys) reached the century mark for the first time in four games in Week 13, and another high-usage day looks to be in store
for a potential shootout.

Keenan Allen (vs. Broncos) caught just five-of-nine targets last week, but as L.A.
</item 5>

<item 6>
Fantasy Football WR PPR Rankings Week 13: Who to start, best sleepers at wide receiver (Sporting News, 2023-11-29T20:27:00+00:00)
Tyreek Hill (@ Commanders in Week 13) did his thing despite a tough matchup in Week 12, resulting in Miami's WR1 returning to the top spot against a putrid
Washington stop unit. CeeDee Lamb (vs. Seahawks) isn't far behind, as Dallas' WR1 aims to maintain his efficient play against Seattle.

Amon-Ra St. Brown (@ Saints) has been a target hog the past several weeks, seeing at least nine looks in each of his past six games. Don't expect his workload
to decrease in a bounce-back game for the Lions. Keenan Allen (@ Patriots) looks to continue his dominant ways, as Justin Herbert's go-to target routinely has
double-digit target potential.

Mike Evans (vs. Panthers) sits in our top five for Week 13, totaling four TDs over the past three weeks. A.J. Brown (vs.
</item 6>

<item 7>
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
</item 7>

<item 8>
Fantasy Football Week 16 Wide Receiver Preview: Waiver adds, DFS plays, and more (CBSSports.com, 2023-12-19T20:51:32+00:00)
Weekly Breakdown MATCHUP @ MIA MIA -1.5 O/U 51 OPP VS WR 15th PROJ PTS 19.4 WR RNK 2nd YTD Stats REC 103 TAR 141 REYDS 1306 TD 10 FPTS/G 21.6 The choice between
Lamb and Tyreek Hill will not be an easy one. The tiebreaker for me is that Lamb is $900 cheaper and Hill may still not be 100% by the time the Cowboys and
Dolphins kick off. Last week was the first time since Week 9 that Lamb failed to reach the end zone and the second time since Week 5 that he didn't score at
least 17 PPR Fantasy points.

Contrarian DFS Play (WR Preview) Projections powered by Sportsline Jaylen Waddle WR MIA Miami • #17

Age: 25 • Experience: 3 yrs.
</item 8>

<item 9>
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
</item 9>

<item 10>
Fantasy Football Week 16 Wide Receiver Preview: Waiver adds, DFS plays, and more (CBSSports.com, 2023-12-19T20:51:32+00:00)
The other three are Tyreek Hill, Tyler Lockett, and DeSean Jackson.

-- Deebo Samuel is on pace to become just the fourth player since 1992 to average more than 11 yards per target with at least 60 targets in multiple seasons.
The other three are Tyreek Hill, Tyler Lockett, and DeSean Jackson. 18 -- Puka Nacua needs 18 more catches to set the rookie record for most catches in a
season.

-- Puka Nacua needs 18 more catches to set the rookie record for most catches in a season. 152.67 -- After missing last week's game, Tyreek Hill now needs to
average almost 153 yards per game in his final three to reach his stated goal of 2,000 receiving yards this season.

-- After missing last week's game, Tyreek Hill now needs to average almost 153 yards per game in his final three to reach his stated goal of 2,000 receiving
yards this season.
</item 10>
=== END ITEM id=39d894817df0095a ===
