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

This file has 20 items.

=== ITEM id=a019-5d4a34a15ebd4a3f ===
Gold sentences and facts:
Sentence 0: For the masses of NFL fans across the U.S., Thursday's clash between the Lions and Packers will be aired on Amazon Prime Video.
  Fact 0.0: Thursday's game between the Lions and Packers will be aired on Amazon Prime Video.

Retrieved items:
<item 1>
Bears vs. Panthers live score, updates, highlights from NFL 'Thursday Night Football' game (Sporting News, 2023-11-09T23:04:33+00:00)
The Bears and Panthers entered Thursday night looking for progress, not playoff contention. If any team achieved that goal, it was Chicago.

It wasn't an aesthetically pleasing game, but that was never the expectation. The Bears grinded out a 16-13 victory behind a terrific defensive performance, a
solid running game, and a few timely throws by Tyson Bagent.

The Panthers jumped out to an early lead on a punt returned for a touchdown by Ihmir Smith-Marsette, but they never found the end zone again. The only offensive
touchdown of the night came in the third quarter, when D'Onta Foreman rumbled his way into the end zone to give Chicago a 16-10 lead.

NFL WEEK 10 PICKS: Straight up | Against the spread

As it turned out, 16 points was all the Bears needed.
</item 1>

<item 2>
Bears vs. Panthers live score, updates, highlights from NFL 'Thursday Night Football' game (Sporting News, 2023-11-09T23:04:33+00:00)
9:28 p.m. — An illegal contact penalty, the second of the night on the Bears, gives Carolina a fresh set of downs.

9:25 p.m. — The Panthers will have 3rd & 3 out of the two-minute warning, just hoping to get in position for some points before halftime. They'll start the
second half with the ball as well.

Panthers 10, Bears 6

9:19 p.m. FIELD GOAL — Bagent is fortunate just to hold onto a bad snap, but his pass is incomplete and brings out Cairo Santos. It's 10-6, Panthers.

9:17 p.m. — A second big play by rookie Tyler Scott, who catches a 15-yard pass from Bagent to give the Bears a fresh set of downs. Chicago is into field goal
range.
</item 2>

<item 3>
Lions vs. Packers live score, updates, highlights from NFL 'Thursday Night Football' game (Sporting News, 2023-09-28T23:54:07+00:00)
: Jordan Love and the Packers heading onto the field ahed of tonight's game! We're about 45 minutes from kickoff at the Pantheon of football excellence, Lambeau
Field.

Lions vs. Packers start time

Date: Thursday, Sept. 28

Thursday, Sept. 28 Start time: 8:15 p.m. ET

Thursday night's game between the Lions and Packers will feature in the primetime slot of 8:15 p.m. ET.

What channel is Lions vs. Packers on today?
</item 3>

<item 4>
Bears vs. Panthers live score, updates, highlights from NFL 'Thursday Night Football' game (Sporting News, 2023-11-09T23:04:33+00:00)
The broadcast will be spearheaded by play-by-play announcer Al Michaels and color commentator Kirk Herbstreit. Sideline reporter Kaylee Hartung will offer
sideline updates throughout the game.

Fans in the Chicago and Charlotte markets can watch the Amazon broadcast over the air on local channels.

Those in Canada can stream it on DAZN, which carries every NFL game all season.
</item 4>

<item 5>
Bears vs. Panthers live score, updates, highlights from NFL 'Thursday Night Football' game (Sporting News, 2023-11-09T23:04:33+00:00)
8:09 p.m. — Perhaps the most interesting storyline tonight will be whether Al Michaels can keep a brave face throughout what might not be the most appealing
game.

8:00 p.m. — We're going to keep spirits high tonight despite a matchup of teams with a combined three wins. All football is good football! Let's see how long
this mentality lasts...

7:46 p.m. — It's 52 degrees and clear in Chicago right now, so certainly not as cold as it could be in the Windy City this time of year. Still, it's a rare
colder-weather game for Bryce Young.

7:30 p.m. — The Bears will be donning orange tonight, including an alternate orange helmet.

7:10 p.m. — D.J. Moore is back facing his former team tonight, eight months after he was traded to the Bears in the deal that sent the No. 1 pick to Carolina.

6:50 p.m.
</item 5>

<item 6>
Bears vs. Panthers live score, updates, highlights from NFL 'Thursday Night Football' game (Sporting News, 2023-11-09T23:04:33+00:00)
1 pick in March.

MORE: Why Frank Reich's job might already be at risk

The Bears (3-7) are hoping to get Justin Fields back in time for their Week 11 game in Detroit, but they've gotten a shot in the arm from Bagent, who improved
to 2-2 as a starter with the win. The victory on Thursday was a win-win for Chicago, as the franchise owns the first-round pick of the Panthers (1-8).

The Sporting News tracked live scoring updates and highlights from Bears vs. Panthers on "Thursday Night Football." Check out all of the key moments you might
have missed.

Bears vs. Panthers final score

1 2 3 4 F Panthers 7 3 0 3 13 Bears 3 6 7 0 16

Bears vs. Panthers results, highlights from Thursday Night Football

(All times Eastern)

Final: Bears 16, Panthers 13

11:11 p.m.
</item 6>

<item 7>
Bears vs. Panthers live score, updates, highlights from NFL 'Thursday Night Football' game (Sporting News, 2023-11-09T23:04:33+00:00)
— After an offensive pass interference call backs up the Bears, Chicago shows no interest in taking a shot and hands it off on 3rd & 19. The Panthers will have
solid field position.

10:26 p.m. — The Bears come out with three Foreman runs in four plays. After turnovers doomed them last week, they're playing it conservatively.

10:21 p.m. — Young throws behind the line of scrimmage on 3rd & 15, which doesn't tend to convert. It'll be a Panthers punt, giving the Bears poor field
position in the process.

10:19 p.m. — Now Young hits Adam Thielen for another first down. How about that? Carolina is just about to midfield.

10:18 p.m. — A rare first down for the Panthers as Young throws to Marshall for an 11-yard gain.

Bears 16, Panthers 10

10:13 p.m.
</item 7>

<item 8>
Bears vs. Panthers live score, updates, highlights from NFL 'Thursday Night Football' game (Sporting News, 2023-11-09T23:04:33+00:00)
Bears vs. Panthers start time

Date: Thursday, Nov. 9

Thursday, Nov. 9 Start time: 8:15 p.m. ET

Thursday night's game between the Bears and Panthers is set to kick off at 8:15 p.m. ET (7:15 p.m. local time) from Soldier Field in Chicago.

What channel is Bears vs. Panthers on today?

Game: Chicago Bears vs. Carolina Panthers

Chicago Bears vs. Carolina Panthers Date: Thursday, Nov. 9

Thursday, Nov. 9 TV channel (Chicago) : WFLD (Channel 32)

: WFLD (Channel 32) TV channel (Charlotte) : WSOC (Channel 9)

: WSOC (Channel 9) Live stream: Amazon Prime Video | DAZN (in Canada)

For the masses of NFL fans across the U.S., Thursday's clash between the Bears and Panthers will be aired on Amazon Prime Video.
</item 8>

<item 9>
Lions vs. Packers live score, updates, highlights from NFL 'Thursday Night Football' game (Sporting News, 2023-09-28T23:54:07+00:00)
Sideline reporter Kaylee Hartung will offer updates throughout the affair.

Fans in the Detroit and Green Bay markets can watch the Amazon broadcast over the air on local channels.

Those in Canada can stream it on DAZN, which carries every NFL game all season.
</item 9>

<item 10>
Lions vs. Packers live score, updates, highlights from NFL 'Thursday Night Football' game (Sporting News, 2023-09-28T23:54:07+00:00)
ET.

What channel is Lions vs. Packers on today?

Game: Detroit Lions at Green Bay Packers

Detroit Lions at Green Bay Packers Date: Thursday, September 28

Thursday, September 28 TV channel (Detroit) : WJBK (Channel 2)

: WJBK (Channel 2) TV channel (Green Bay) : WITI (Channel 6) or WGBA (Channel 26)

: WITI (Channel 6) or WGBA (Channel 26) Live stream: Amazon Prime Video | DAZN (in Canada)

For the masses of NFL fans across the U.S., Thursday's clash between the Lions and Packers will be aired on Amazon Prime Video. The broadcast will be
spearheaded by legendary play-by-play announcer Al Michaels. He'll be joined by longtime ESPN college football commentator Kirk Herbstreit, who will offer
occasional musings as an analyst. Sideline reporter Kaylee Hartung will offer updates throughout the affair.
</item 10>
=== END ITEM id=a019-5d4a34a15ebd4a3f ===

=== ITEM id=a036-76782bcf6670e1ed ===
Gold sentences and facts:
Sentence 0: The former All-Pro, who battled a lingering hamstring injury earlier this year, received attention on the sidelines after limping off, then left the game for further evaluation in the locker room.
  Fact 1.3: The former All-Pro then left the game for further evaluation in the locker room.

Retrieved items:
<item 1>
2023 NFL Week 11 injury roundup: Rams WR Cooper Kupp, Seahawks RB Kenneth Walker III among stars to exit (CBSSports.com, 2023-11-19T23:51:49+00:00)
I agree to receive the "Pick Six Newsletter" and marketing communications, updates, special offers (including partner offers), and other information from CBS
Sports and the Paramount family of companies. By pressing sign up, I confirm that I have read and agree to the Terms of Use and acknowledge Paramount's Privacy
Policy See All Newsletters Please check the opt-in box to acknowledge that you would like to subscribe. Thanks for signing up! Keep an eye on your inbox. Sorry!
There was an error processing your subscription.

WR

Tyreek Hill (Dolphins) briefly left Miami's game against the Raiders while nursing a hand injury and was ruled questionable to return, but he took the field
again soon afterward, securing seven catches for 80 yards and a score before halftime. Hill has left multiple games this year, only to return in all of them.

Cooper Kupp (Rams) had just a single catch against the Seahawks before suffering an ankle injury while blocking.
</item 1>

<item 2>
2023 NFL Week 11 injury roundup: Rams WR Cooper Kupp, Seahawks RB Kenneth Walker III among stars to exit (CBSSports.com, 2023-11-19T23:51:49+00:00)
The former All-Pro, who battled a lingering hamstring injury earlier this year, received attention on the sidelines after limping off, then left the game for
further evaluation in the locker room. He was ruled questionable to return. NFL Media reported on Monday that Kupp suffered a low ankle sprain and will be
evaluated going forward.

Darius Slayton (Giants) had a big impact against the Commanders, hauling in a deep score from Tommy DeVito, before exiting with an arm injury. He was ruled out
for the remainder of the contest halfway through the third quarter.

EDGE

Joey Bosa (Chargers) appeared to injure his right leg or foot while held in a block during a first-quarter pass-rushing attempt against the Packers. He limped
to the sidelines and was carted to the locker room, then ruled questionable to return.
</item 2>

<item 3>
2023 NFL Week 11 injury roundup: Rams WR Cooper Kupp, Seahawks RB Kenneth Walker III among stars to exit (CBSSports.com, 2023-11-19T23:51:49+00:00)
DB

Talanoa Hufanga (49ers), who emerged as a ballhawk for San Francisco's secondary in 2022, left early against the Buccaneers with a knee injury. Coach Kyle
Shanahan confirmed to reporters on Monday that the All-Pro suffered a season-ending ACL tear.
</item 3>

<item 4>
2023 NFL Week 11 injury roundup: Rams WR Cooper Kupp, Seahawks RB Kenneth Walker III among stars to exit (CBSSports.com, 2023-11-19T23:51:49+00:00)
He was seen walking on the sidelines shortly afterward but ruled doubtful to return. Rookie Zach Charbonnet, who entered Week 11 with just over 200 rushing
yards on the year, replaced Walker as Seattle's top ball carrier. Following the game, Seahawks head coach Pete Carroll told reporters Walker suffered an
"oblique strain that was legit" and could miss time.

D'Onta Foreman (Bears) remained a headliner of Chicago's backfield against the Lions, even with Khalil Herbert back from injured reserve, but managed just 14
yards on the ground before suffering an ankle injury. He walked off with a limp and attempted to run on the sidelines but did not return to the field after the
Bears ruled him questionable.

Pick Six Newsletter Crafted By The Best NFL Experts Get the day's big stories + fun stuff you love like mock drafts, picks and power rankings.
</item 4>

<item 5>
Fantasy Football Week 13 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-11-30T22:10:58+00:00)
If they're ailing, the Rams would be smart to try and use the run game early and often. Expect a low-scoring game.

STARTS: Kyren Williams, Amari Cooper, Puka Nacua, David Njoku, Jerome Ford,

Kyren Williams, Amari Cooper, Puka Nacua, David Njoku, Jerome Ford, SITS: Cooper Kupp (flex at best), Kareem Hunt, Tutu Atwell, Tyler Higbee, Royce Freeman,
both QBs, both DSTs (low-end options)

Cooper Kupp (flex at best), Kareem Hunt, Tutu Atwell, Tyler Higbee, Royce Freeman, both QBs, both DSTs (low-end options) SLEEPERS: Elijah Moore (PPR flex)

Start Him (Lineup Decisions) Projections powered by Sportsline Amari Cooper WR CLE Cleveland • #2

Age: 29 • Experience: 9 yrs.
</item 5>

<item 6>
Fantasy Football WR PPR Rankings Week 14: Who to start, best sleepers at wide receiver (Sporting News, 2023-12-07T22:11:00+00:00)
What are the biggest WR injuries in Week 14?

In Week 13, Demario Douglas (concussion) and Rashid Shaheed (thigh) missed game action, and Tank Dell (leg), Amari Cooper (concussion), Christian Watson
(hamstring), and Marquise Brown (heel) exited early from their respective games.

UPDATE: Christian Kirk suffered a groin injury on Monday night, and he figures to be more doubtful than questionable for this week's game against Cleveland. If
he's out, Zay Jones and Tim Jones will see more targets, but with Trevor Lawrence (ankle) also hurt, it's tough to count on either.

UPDATE: Douglas (concussion) has been ruled OUT for Week 14.

As usual, we'll update these WR PPR rankings throughout the week, so check back for the latest player movement and analysis.
</item 6>

<item 7>
2023 NFL Week 11 injury roundup: Rams WR Cooper Kupp, Seahawks RB Kenneth Walker III among stars to exit (CBSSports.com, 2023-11-19T23:51:49+00:00)
Smith did return, however, for the final drive of the game in the losing effort. Head coach Pete Carroll said that Smith will "make it back." When asked if that
means for Thursday's matchup against San Francisco, the coach said, "I would think so."

Sam Howell (Commanders) received medical attention on the sidelines after enduring several hard hits to the upper body on a touchdown run against the Giants. He
was quickly cleared of a potential concussion, however, and remained in the game. Veteran journeyman Jacoby Brissett is Washington's backup.

RB

Aaron Jones (Packers) suited up against the Chargers despite a lingering hamstring issue, only to manage just five carries before suffering a knee injury. The
former Pro Bowler needed help to leave the field and was then carted to the locker room, replaced by A.J. Dillon.
</item 7>

<item 8>
2023 NFL Week 11 injury roundup: Rams WR Cooper Kupp, Seahawks RB Kenneth Walker III among stars to exit (CBSSports.com, 2023-11-19T23:51:49+00:00)
Dillon. Green Bay ruled him questionable to return, then downgraded him to out, but coach Matt LaFleur said after the Packers' win over Los Angeles that he
doesn't expect the injury to be a long-term issue.

De'Von Achane (Dolphins) returned from injured reserve -- and a four-game absence -- but limped to the sidelines after his second touch of the first quarter
against the Raiders. Miami ruled him questionable to return with a knee injury, and while he later returned to the sidelines from a locker-room visit, he was
replaced on the field indefinitely by Raheem Mostert. Head coach Mike McDaniel told reporters on Monday that Achane is day-to-day. While he hasn't been ruled
out of Friday's matchup with the Jets, he's not been given the green light either, so his status is worth monitoring.

Kenneth Walker III (Seahawks) took four carries against the Rams before leaving with an oblique injury.
</item 8>

<item 9>
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
</item 9>

<item 10>
Fantasy Football WR PPR Rankings Week 13: Who to start, best sleepers at wide receiver (Sporting News, 2023-11-29T20:27:00+00:00)
Falcons

13 Calvin Ridley, Jaguars vs. Bengals

14 Cooper Kupp, Rams vs. Browns

15 DeVonta Smith, Eagles vs. 49ers

16 DK Metcalf, Seahawks @ Cowboys

17 Chris Olave, Saints vs. Lions

18 Deebo Samuel, 49ers @ Eagles

19 DeAndre Hopkins, Titans vs. Colts

20 Courtland Sutton, Broncos @ Texans

21 Christian Kirk, Jaguars vs. Bengals

22 Terry McLaurin, Commanders vs. Dolphins

23 Nico Collins, Texans vs. Broncos

24 Rashee Rice, Chiefs @ Packers

25 Puka Nacua, Rams vs. Browns

26 Chris Godwin, Buccaneers vs. Panthers

27 Marquise Brown, Cardinals @ Steelers

28 Christian Watson, Packers vs. Chiefs

29 Adam Thielen, Panthers @ Buccaneers

30 Romeo Doubs, Packers vs.
</item 10>
=== END ITEM id=a036-76782bcf6670e1ed ===

=== ITEM id=a098-c8f90e2616bebb63 ===
Gold sentences and facts:
Sentence 0:  Ельцина, "Uralʹskiĭ federalʹnyĭ universitet imeni pervogo Prezidenta Rossii B.N. Yelʹtsina", often shortened to UrFU, УрФУ ) (formed by a merger of the Ural State Technical University and Ural State University) is one of the leading educational institutions in the Ural region.
  Fact 2.1: The full name of the university is 'Uralʹskiĭ federalʹnyĭ universitet imeni pervogo Prezidenta Rossii B.N. Yelʹtsina'.

Retrieved items:
<item 1>
Cal Poly Pomona College of Environmental Design
The Cal Poly Pomona College of Environmental Design (CENV) is a college part of the California State Polytechnic University, Pomona (Cal Poly Pomona).

 The college houses over 1,600 students; making it one of largest environmental design programs in the United States.

 The college offers bachelor's degrees in five departments, as well as three master's degree programs.

 It is the only academic unit within the California State University system to be associated with a Pritzker Prize laureate (often referred to as "The Nobel
Prize in Architecture").
</item 1>

<item 2>
Julian A. McPhee
Julian Aeneas McPhee (February 7, 1896 – November 10, 1967) was the sixth university president of California Polytechnic State University, San Luis Obispo (Cal
Poly SLO) from 1933 to 1966 and the first president of California State Polytechnic University, Pomona (Cal Poly Pomona) from 1938 to 1966.
</item 2>

<item 3>
Cal Poly Pomona University Library
The Cal Poly Pomona University Library is the main library on the campus of California State Polytechnic University, Pomona (Cal Poly Pomona).

 It has in its collection 670,580 books; 6,883 serial subscriptions and 10,417 audiovisual materials.
</item 3>

<item 4>
1981 Cal Poly Pomona Broncos football team
The 1981 Cal Poly Pomona Broncos football team represented California State Polytechnic University, Pomona during the 1981 NCAA Division II football season.

 Cal Poly Pomona competed in the California Collegiate Athletic Association (CCAA).
</item 4>

<item 5>
Cal Poly Pomona Broncos women's basketball
The Cal Poly Pomona Broncos women's basketball team is the women's basketball team that represents California State Polytechnic University, Pomona, in Pomona,
California.

 The school's team currently competes in the California Collegiate Athletic Association.
</item 5>

<item 6>
International Polytechnic High School
International Polytechnic High School, commonly abbreviated merely as iPoly, is a public college preparatory demonstration high school (9-12) located on the
California State Polytechnic University, Pomona (Cal Poly Pomona) campus and operated by the Los Angeles County Office of Education (LACOE) in conjunction with
the College of Education and Integrative Studies at the university.

 iPoly's curriculum is fully accredited by the Western Association of Schools and Colleges and is approved by the University of California and California State
University.

 It maintains a unique affiliation with the Los Angeles County High School for the Arts (LACHSA), which is also run by LACOE.

 Since iPoly does not lie in a fixed school district, it draws students from throughout Los Angeles, San Bernardino, Orange and Riverside counties.

 The majority of students come from the Pomona and San Gabriel valleys.

 In 2009 and 2013, iPoly was honored as a California Distinguished School by the California Department of Education.
</item 6>

<item 7>
Cal Poly Pomona College of Engineering
The Cal Poly Pomona College of Engineering is the engineering college at California State Polytechnic University, Pomona (Cal Poly Pomona or Cal Poly) located
in Pomona, California, United States.

 Well known for its "learn by doing" philosophy, the college's motto is: ""making imagination real"".

 Cal Poly has one of the top engineering college among public schools in California and, with over 5,600 students (as of fall 2015), it is also the largest
engineering college in Southern California, the second largest college of engineering in the California State University system, and the seventeenth largest
engineering college in the United States.

 In the 2017 "U.S. News & World Report" the College of Engineering is ranked the 5th best undergraduate program among public universities (11th overall) in the
West for Master's-granting universities, and "has one of the top ranked engineering programs, and graduates roughly 1 of every 14 engineers in the state of
California."
</item 7>

<item 8>
California State Polytechnic University, Pomona
California State Polytechnic University, Pomona (CPP, Cal Poly Pomona, or Cal Poly) is a public polytechnic university located in Pomona, California, United
States.

 It is one of two polytechnics in the California State University system.
</item 8>

<item 9>
Cal Poly Pomona Broncos
The Cal Poly Pomona Broncos or Cal Poly Broncos are the athletic sports teams for the California State Polytechnic University, Pomona (Cal Poly Pomona).

 Cal Poly Pomona has 10 varsity sports teams and offers student participation in a wide range of sports including soccer, volleyball, track and field,
basketball, softball, and baseball.

 Cal Poly Pomona participates at the NCAA's Division II (DII) level in the California Collegiate Athletic Association (CCAA).

 The Broncos have 65 CCAA championships and 14 NCAA National Championships.

 Current and former Cal Poly athletes have won 7 Olympic medals (3 gold, 1 silver, and 3 bronze).
</item 9>

<item 10>
Lyman: first U.S. boys’ reform school (Massachusetts, c.1886–1971), now historic. Massachusetts International Academy: Chinese-student U.S.-college prep
boarding school. Polytech’Paris-UPMC: French research-linked engineering school in an 11-school network. Cal Poly Pomona: public polytechnic; library holds
670,580 books; engineering enrolls 5,600+; environmental design 1,600+; women’s basketball compet
</item 10>
=== END ITEM id=a098-c8f90e2616bebb63 ===

=== ITEM id=a065-481dce4174203b32 ===
Gold sentences and facts:
Sentence 0: On his podcast, he later invited the “Anti-Hero” singer to one of his games at Arrowhead Stadium - an invitation she famously accepted on 24 September.
  Fact 1.3: The 'Anti-Hero' singer famously accepted the invitation on 24 September.

Retrieved items:
<item 1>
Taylor Swift reveals secret start to her relationship with Travis Kelce (The Independent - Life and Style, 2023-12-06T14:23:01+00:00)
Elsewhere in the conversation with Time, the “Anti-Hero” singer said that she tries not to let the paparazzi get to her, even though so many of her outings with
pals go on to make headlines. “Over the years, I’ve learned I don’t have the time or bandwidth to get pressed about things that don’t matter. Yes, if I go out
to dinner, there’s going to be a whole chaotic situation outside the restaurant. But I still want to go to dinner with my friends,” she said.

She emphasised how her perspective about the media has changed over the years, adding: “Life is short. Have adventures. Me locking myself away in my house for a
lot of years - I’ll never get that time back. I’m more trusting now than I was six years ago.”

With his love life in the public eye, Kelce has also opened up about his relationship with Swift.
</item 1>

<item 2>
Fans spot Travis Kelce wearing Taylor Swift-themed friendship bracelet before she attended his game (The Independent - Life and Style,
2023-09-26T22:05:34+00:00)
Stay ahead of the trend in fashion and beyond with our free weekly Lifestyle Edit newsletter Stay ahead of the trend in fashion and beyond with our free weekly
Lifestyle Edit newsletter Please enter a valid email address Please enter a valid email address SIGN UP I would like to be emailed about offers, events and
updates from The Independent. Read our privacy notice Thanks for signing up to the

Lifestyle Edit email {{ #verifyErrors }} {{ message }} {{ /verifyErrors }} {{ ^verifyErrors }} Something went wrong. Please try again later {{ /verifyErrors }}

Fans have spotted Travis Kelce wearing a friendship bracelet with Taylor Swift lyrics on it, just days before she attended his Kansas City Chiefs game.

In a video posted to TikTok on 25 September, a fan shared a photo of Kelce posing with the piece of jewellery on his wrist. The post came after Swift was seen
enthusiastically cheering him on in the box seats at Arrowhead Stadium, fuelling speculation that she and the athlete are dating.
</item 2>

<item 3>
Taylor Swift is Time’s Person of the Year (The Independent - Life and Style, 2023-12-06T13:55:17+00:00)
Since then, Swift has released four “Taylor’s Version” albums: Fearless (Taylor’s Version) in April 2021, Red (Taylor’s Version) in November 2021, Speak Now
(Taylor’s Version) in July 2023, and 1989 (Taylor’s Version) in October 2023.

“It’s not lost on me that the two great catalysts for this happening were two horrendous things that happened to me,” Swift said, when asked about the success
of her tour. “The first was getting cancelled within an inch of my life and sanity. The second was having my life’s work taken away from me by someone who hates
me.”

While Swift is arguably more famous now than ever before, the Grammy winner revealed that the constant attention and swarms of paparazzi won’t stop her from
living her life. There’s also constant rumours surrounding her personal life, specifically rumours about her relationship with Alwyn that her publicist, Tree
Paine, was forced to publicly deny.
</item 3>

<item 4>
Taylor Swift reveals secret start to her relationship with Travis Kelce (The Independent - Life and Style, 2023-12-06T14:23:01+00:00)
Stay ahead of the trend in fashion and beyond with our free weekly Lifestyle Edit newsletter Stay ahead of the trend in fashion and beyond with our free weekly
Lifestyle Edit newsletter Please enter a valid email address Please enter a valid email address SIGN UP I would like to be emailed about offers, events and
updates from The Independent. Read our privacy notice Thanks for signing up to the

Lifestyle Edit email {{ #verifyErrors }} {{ message }} {{ /verifyErrors }} {{ ^verifyErrors }} Something went wrong. Please try again later {{ /verifyErrors }}

Taylor Swift has revealed when her relationship with Travis Kelce really started.

The 33-year-old singer spoke candidly about the romance during a recent interview with Time after she was named the magazine’s 2023 Person of the Year. Speaking
to the publication, she said she connected with Kelce in July, after he confessed on his podcast that he tried and failed to give her a friendship bracelet with
his phone number on it during one of her Eras Tour shows.
</item 4>

<item 5>
Taylor Swift is Time’s Person of the Year (The Independent - Life and Style, 2023-12-06T13:55:17+00:00)
Not to mention her rumoured relationship with Kansas City Chiefs tight end Travis Kelce caused a 400 per cent spike in NFL jersey sales.

Since her first NFL appearance in September, Swift’s occasional attendance at Kelce’s games has received pushback from NFL fans who believe her presence is
distracting from the game itself. “I’m just there to support Travis,” Swift admitted. “I have no awareness of if I’m being shown too much and pissing off a few
dads, Brads, and Chads.”

She playfully added: “Football is awesome, it turns out. I’ve been missing out my whole life.”

The “Bad Blood” singer revealed that their romance began when Kelce gave her shoutout on his podcast, admitting he attempted to give Swift his phone number
after one of her Eras Tour shows. The pair immediately began “hanging out” and by the time she attended her first Chiefs game in September, they were already a
couple. “I think some people think that they saw our first date at that game?
</item 5>

<item 6>
Fans spot Travis Kelce wearing Taylor Swift-themed friendship bracelet before she attended his game (The Independent - Life and Style,
2023-09-26T22:05:34+00:00)
After the game, which saw the Chiefs beat the Chicago Bears 41-10, Kelce was seen leaving the stadium in a blue and white splatter-paint denim suit. Eagle-eyed
fans were quick to believe that Kelce’s outfit paid homage to Swift’s forthcoming album, 1989 (Taylor’s Version), which is due for re-release on 27 October.

After the two escaped the crowd at Arrowhead Stadium in Kelce’s convertible, according to photos and videos posted online, the football star reportedly rented
out Prime Social Rooftop in Kansas City for a post-game celebration with his family, his teammates, and Swift.

“Travis bought out the restaurant for his family and team. Taylor arrived, wearing a denim dress, and was seen snacking, having some cocktails and dancing
alongside Travis,” the witness told Entertainment Tonight. “The two were very affectionate with one another but kept things fun and lighthearted.
</item 6>

<item 7>
Taylor Swift is Time’s Person of the Year (The Independent - Life and Style, 2023-12-06T13:55:17+00:00)
“Over the years, I’ve learned I don’t have the time or bandwidth to get pressed about things that don’t matter,” Swift told Time. “Yes, if I go out to dinner,
there’s going to be a whole chaotic situation outside the restaurant. But I still want to go to dinner with my friends.

“Life is short. Have adventures. Me locking myself away in my house for a lot of years - I’ll never get that time back,” she added. “I’m more trusting now than
I was six years ago.”
</item 7>

<item 8>
Taylor Swift is Time’s Person of the Year (The Independent - Life and Style, 2023-12-06T13:55:17+00:00)
“I think some people think that they saw our first date at that game? We would never be psychotic enough to hard launch a first date,” Swift joked.

Back in April, it was revealed that Swift and British actor Joe Alwyn had broken up after six years of dating. The former couple were notoriously private, as
the Conversations with Friends star previously expressed he didn’t want to discuss their relationship because it “feeds into a weird part of the culture”.

Now, Swift has shared that she has nothing to hide in her new relationship with Kelce. “When you say a relationship is public, that means I’m going to see him
do what he loves, we’re showing up for each other, other people are there and we don’t care,” she said. “The opposite of that is you have to go to an extreme
amount of effort to make sure no one knows that you’re seeing someone. And we’re just proud of each other.”
</item 8>

<item 9>
Taylor Swift-Travis Kelce timeline: Everything to know about rumored romance between pop star, Chiefs TE (CBSSports.com, 2023-10-12T23:21:52+00:00)
21: Kelce sends Swift an invite

"So inviting, I almost jump in." - Gold Rush

While on the Pat McAfee show, Kelce discussed the rumors and sounded somehow surprised people are talking about it.

"It is hilarious how much traction this has actually got," he said of all the talk. "It's like that old game in school called telephone where everybody is just
whispering in each others ear, just hearing some random stuff. Especially when you got Jason Kelce on live television telling people just both sides."

He then admitted that he invited Swift to one of his games, joking that they would compete to see who put on the bigger show at Arrowhead Stadium, home of the
Chiefs.

"I threw it out there, I threw the ball in her court and I told her you know, I've seen you rock the stage in Arrowhead you might have to come see me rock the
stage in Arrowhead and see which one is a little more lit. We will see what happens in the near future."
</item 9>

<item 10>
Fans spot Travis Kelce wearing Taylor Swift-themed friendship bracelet before she attended his game (The Independent - Life and Style,
2023-09-26T22:05:34+00:00)
Travis’ teammates also showed up to the after-party, as well as his mom, Donna Kelce. The party lasted until 2 am.”

The late-night outing came days after the NFL star officially addressed ongoing dating rumours about him and the “Anti-Hero” singer. During an episode of The
Pat McAfee Show on 21 September, he mocked his brother Jason Kelce for joking that he “100 per cent” believed his younger sibling was dating Swift.

“It’s hilarious how much traction this has actually gotten, right now it’s like a game of telephone,” Kelce said. “Where everybody’s whispering in each other’s
ears and hearing random stuff. No one actually knows what’s going on, especially when you got Jason Kelce on live television telling people just both sides - he
doesn’t know, it’s true, this and that.”

He also noted that he was still open to pursuing a relationship with Swift, revealing that he subtly asked if she’d want to attend one of his football games.
</item 10>
=== END ITEM id=a065-481dce4174203b32 ===

=== ITEM id=a086-f36f532e274e90ae ===
Gold sentences and facts:
Sentence 0: Lillian Florence Hellman (June 20, 1905 – June 30, 1984) was an American dramatist and screenwriter known for her success as a playwright on Broadway, as well as her left-wing sympathies and political activism.
  Fact 1.2: Lillian Florence Hellman was an American dramatist.

Retrieved items:
<item 1>
Lucy Maud Montgomery
Lucy Maud Montgomery {'1': ", '2': ", '3': ", '4': "} (November 30, 1874 – April 24, 1942) published as L.M. Montgomery, was a Canadian author best known for a
series of novels beginning in 1908 with "Anne of Green Gables".

 The book was an immediate success.

 The central character, Anne Shirley, an orphaned girl, made Montgomery famous in her lifetime and gave her an international following.
</item 1>

<item 2>
Alice Munro
Alice Ann Munro ( , "née" Laidlaw ; born 10 July 1931) is a Canadian short story writer and Nobel Prize winner.

 Munro's work has been described as having revolutionized the architecture of short stories, especially in its tendency to move forward and backward in time.

 Her stories have been said to "embed more than announce, reveal more than parade."
</item 2>

<item 3>
Lillian Hellman: An Imperious Life
Lillian Hellman: An Imperious Life is a 2014 book by Dorothy Gallagher.

 It is a critical biography of the American playwright and writer Lillian Hellman.
</item 3>

<item 4>
The Autumn Garden
The Autumn Garden is a 1951 play by Lillian Hellman.

 The play is set in September, 1949 in a summer home in a resort on the Gulf of Mexico, about 100 miles from New Orleans.

 The play is a study of the defeats, disappointments and diminished expectations of people reaching middle age.

 For inspiration, Hellman drew on her memories of her time in her aunts' boardinghouse.

 Dashiell Hammett, who had been Hellman's lover for 20 years, helped her write the play and received 15 percent of the royalties.

 Of all Hellman's plays it was her favorite.
</item 4>

<item 5>
Dash and Lilly
Dash and Lilly is a 1999 Emmy and Golden Globe Award-nominated biographical television film about writers Dashiell Hammett and Lillian Hellman.

 The film was directed by actress Kathy Bates and written by Jerrold L. Ludwig.

 It stars Sam Shepard and Judy Davis.
</item 5>

<item 6>
The entries profile writers, scholars, artists, publishers, and advocates. Highlights include computer scientist Nuria Oliver’s data-science leadership and
human-behavior research; Helen Keller’s childhood story “The Frost King”; Booker Prize winner Keri Hulme; Alice Walker’s Pulitzer-winning *The Color Purple* and
anti-FGM work; and Lucille Watahomigie’s Hualapai language revitalization. Others include novelists Sarra Manning and Kim Edwards, poet Tara Betts
</item 6>

<item 7>
The entries profile women in literature, theatre, music, visual art, business, and activism. Highlights include Pearson CEO Marjorie Scardino, the first woman
to lead a FTSE 100 company; Nobel-winning short-story writer Alice Munro; actress Elizabeth Taylor; pioneering Restoration writer Aphra Behn; and Aboriginal
activist Mum Shirl. They also cover Lucy Maud Montgomery’s Anne novels, Jean Webster’s *When Patty Went to College*, Margaret Landon’s *Anna and the King
</item 7>

<item 8>
Mollie Gillen
Mollie Gillen (née Woolnough; 1908–2009) was an Australian historian, researcher, writer and novelist.

 Her work on the First Fleet, in "The search for John Small," "First Fleeter" and The Founders of Australia: A Biographical Dictionary of the First Fleet,
explored the idea that many of the founding families of Australia were descended from the convict population, rather than those sent to guard them.

 Gillen's article Maud Montgomery: The Girl Who Wrote Green Gables instigated a new era in scholarship on Lucy Maud Montgomery.
</item 8>

<item 9>
The Little Foxes (film)
The Little Foxes (1941) is an American drama film directed by William Wyler.

 The screenplay by Lillian Hellman is based on her 1939 play of the same name.

 Hellman's ex-husband Arthur Kober, Dorothy Parker and her husband Alan Campbell contributed additional scenes and dialogue.
</item 9>

<item 10>
Lillian Hellman
Lillian Florence Hellman (June 20, 1905 – June 30, 1984) was an American dramatist and screenwriter known for her success as a playwright on Broadway, as well
as her left-wing sympathies and political activism.

 She was blacklisted after her appearance before the House Committee on Un-American Activities (HUAC) at the height of the anti-communist campaigns of 1947–52.

 Although she continued to work on Broadway in the 1950s, her blacklisting by the American film industry caused a drop in her income.

 Many praised Hellman for refusing to answer questions by HUAC, but others believed, despite her denial, that she had belonged to the Communist Party.
</item 10>
=== END ITEM id=a086-f36f532e274e90ae ===

=== ITEM id=a031-e64be96ed2e45f0d ===
Gold sentences and facts:
Sentence 0: And after Bankman-Fried left Jane Street, the high-frequency trading firm where the former billionaire got his start in finance after MIT, he persuaded Yedidia to join him as a trader at Alameda and then as a developer at FTX.
  Fact 1.1: Jane Street is a high-frequency trading firm.

Retrieved items:
<item 1>
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
</item 1>

<item 2>
The FTX trial is bigger than Sam Bankman-Fried (The Verge, 2023-09-28T12:00:00+00:00)
The trial of Sam Bankman-Fried is likely to be more consequential than just whether the man himself is found guilty. Depending on what evidence is introduced
during the trial, it could be rough for the entire crypto industry.

“How much damage can this trial do to the already beaten-down reputation of the industry at this point?” asks Yesha Yadav, a law professor at Vanderbilt
University. “This trial is going to be an excruciating moment for the industry because no one knows what kind of evidence might come out.”

Bankman-Fried, the founder of FTX and Alameda Research, is facing seven counts of criminal charges: two counts of wire fraud, and five counts of conspiracy
charges. FTX was a failed cryptocurrency exchange founded in 2019. According to a now-deleted profile from FTX investors Sequoia Capital, FTX was founded
because of Bankman-Fried’s frustration with other exchanges when he was running Alameda Research, his crypto trading firm.
</item 2>

<item 3>
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
</item 3>

<item 4>
How is it still getting worse for Sam Bankman-Fried? (The Verge, 2023-10-12T23:08:14+00:00)
The recording was later sent to Drappi, who sent it to the government. Joseph Bankman, the defendant’s father and a senior advisor to FTX’s philanthropic arm,
wasn’t in the room for the recording. Barbara Fried looked unhappy, rubbing her left temple as though she had a tremendous headache.

And in a small miracle of pacing, the government got Zac Prince, the founder and former CEO of crypto lender BlockFi, on the stand just long enough to blame
FTX’s bankruptcy for BlockFi’s subsequent bankruptcy before we broke for the day.

I’ve been asked by some people why the reporting coming out of the trial seems so skewed toward the prosecution. There’s an answer for this: the prosecution has
put together a strong, comprehensive indictment of Bankman-Fried’s behavior at FTX and Alameda. The defense, so far, has managed to do almost nothing in
response.
</item 4>

<item 5>
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
</item 5>

<item 6>
Is Sam Bankman-Fried’s defense even trying to win? (The Verge, 2023-10-04T23:02:14+00:00)
I have never seen Sam Bankman-Fried so still as he was during the prosecution’s opening statement. The characteristic leg-jiggling was absent. He barely moved
as the prosecutor listed the evidence against him: internal company files, what customers were told, the testimony of his co-conspirators and his own words.

His hair was shorn, the result of a haircut from a fellow prisoner, the Wall Street Journal reported. He wore a suit bought at a discount at Macy’s, per the
Journal; it hung on him. He appeared to have lost some weight.

“All of that was built on lies.”

Bankman-Fried, at this time last year, had a luxury lifestyle as the CEO of crypto exchange FTX, said the assistant US attorney, Thane Rehn, in the cadence of a
high schooler delivering his lines in a student play. Bankman-Fried hung out with Tom Brady. He was on magazine covers, lived in a $30 million penthouse, and
spent time with world politicians.
</item 6>

<item 7>
The FTX trial is bigger than Sam Bankman-Fried (The Verge, 2023-09-28T12:00:00+00:00)
In the indictment, the prosecutors allege that Bankman-Fried lied to FTX investors. Some of those investors may be called on to testify, which is not the kind
of thing VC big shots generally enjoy — and which might create collateral damage for the industry.

If, for instance, Sequoia Capital did due diligence around its investment in FTX, whatever Bankman-Fried told its partners could be important. Was FTX already
sending Alameda customer funds at that point? Did Bankman-Fried know about it? And did he tell VCs at the time? (If he told them it wasn’t happening, that would
establish intent.) Binance was also an early investor in FTX, and former executives may be called on to testify as well, says Hermine Wong, the former head of
policy at Coinbase and a former SEC regulator.

Bankman-Fried’s defense can also introduce risks for people who dealt with him. Defense lawyers have several simultaneous objectives. First and foremost,
they’re trying to prove their client is not guilty.
</item 7>

<item 8>
The FTX trial is bigger than Sam Bankman-Fried (The Verge, 2023-09-28T12:00:00+00:00)
If it seems like his peer group supported him, or worse, fawned over him, that’s a real problem for the industry’s reputation.”

After FTX declared bankruptcy, Bankman-Fried went on an extensive media tour, doing interviews with The New York Times’ Andrew Ross Sorkin and Good Morning
America. Even under house arrest, he did interviews, including with The New Yorker. Bankman-Fried seemed to think that if he just explained things, everyone
would understand that what he did was a mistake, not a crime.
</item 8>

<item 9>
The FTX trial is bigger than Sam Bankman-Fried (The Verge, 2023-09-28T12:00:00+00:00)
Bankman-Fried is accused of using “billions of dollars in misappropriated FTX customer deposits” to help buy more than $200 million of real estate for himself,
make billions of dollars of investments for his own interest, and repay Alameda’s lenders, according to the indictment, Additionally, Bankman-Fried allegedly
used more than $100 million of customer funds to make political contributions; prosecutors can show evidence of those contributions in this trial, even though
they aren’t part of the charges brought. A second trial is scheduled for March 2024, with additional charges.

Before his fall, Bankman-Fried made himself out to be the Good Boy of crypto — the trustworthy face of a sometimes-shady industry. He was also very interested
in publicity, sitting for many interviews both before and after the fall of FTX. The quick rise of FTX as an industry force was at least in part due to
Bankman-Fried’s appetite for attention.
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
=== END ITEM id=a031-e64be96ed2e45f0d ===

=== ITEM id=a132-68fe994af99e4f66 ===
Gold sentences and facts:
Sentence 0: Carlos Santana    (born July 20, 1947) is a Mexican and American musician who first became famous in the late 1960s and early 1970s with his band, Santana, which pioneered a fusion of rock and Latin American music.
  Fact 1.4: Carlos Santana first became famous with his band Santana.

Retrieved items:
<item 1>
Graph query row:
entities: ['Santana', 'Inner Secrets']
relationships: ['CREATED_BY']
</item 1>

<item 2>
The Secret (Vib Gyor EP)
The Secret EP is the EP from the Leeds band Vib Gyor.

 It was released on 5 March 2007, and is the follow-up to their well-received debut single, "Fallen".

 It was recorded in Bath and produced by Paul Corkett.

 It also has a hidden fifth instrumental track - which the band recently revealed was called <:.

~.

:>
</item 2>

<item 3>
Information (Berlin album)
Information is the first album by Berlin, released in 1980 by Vinyl Records.

 It was recorded during the period when Terri Nunn had temporarily left the band to pursue an acting career and Virginia Macolino performed the lead vocals.

 Several songs were written by previous lead singer Toni Childs who went solo prior to the album's release.
</item 3>

<item 4>
Inner Secrets
Inner Secrets is the ninth studio album by Santana.

 It was released in 1978 and marks the start of the phase of Santana's career where he moved away from the fusion of Latin, jazz, rock and blues that marked his
previous records and began to move towards an album-oriented rock direction.

 As such, the album's quality is widely disputed among fans. "

Stormy" and "One Chain (Don't Make No Prison)" were both hit singles.

 In The Netherlands "Well All Right" was released as a single and reached #22 in the top 40.
</item 4>

<item 5>
Hot Streets
Hot Streets is the tenth studio album (twelfth overall) by the American band Chicago, released in 1978.

 In many ways, "Hot Streets" marked the beginning of a new era for the band, turning to disco music, a move which would be derided in retrospect.

 It was also the band's first album with all-new material released since their second that didn't have a numbered title.

 It was also the first album not to feature original guitarist/vocalist Terry Kath, who died from an accidental self-inflicted gunshot wound in January 1978.

 He was replaced by Donnie Dacus on this album.
</item 5>

<item 6>
The V.I.P.s (band)
That album was published in 1967 on Liberty Records and contained only five songs from two minutes to more than 15 minutes of psychedelic and almost meditative
state kind of music.
</item 6>

<item 7>
Moment of Truth (Terri Nunn album)
Moment of Truth is the début and sole solo album from American singer and actress Terri Nunn, best known as lead singer of the American New Wave/Synthpop band
Berlin.
</item 7>

<item 8>
Suicide (1977 album)
Suicide is the debut album from the American rock band Suicide.

 It was released in 1977 on Red Star Records and produced by Craig Leon and Marty Thau.

 The album was recorded in four days at Ultima Sound Studios in New York and featured Martin Rev's minimalist electronics and harsh, repetitive rhythms paired
with Alan Vega's rock and roll-inspired vocals and depictions of urban life.
</item 8>

<item 9>
Chicago 13
Chicago 13 is the eleventh studio album by the American band Chicago, released in 1979.

 The follow-up to "Hot Streets", "Chicago 13" is often critically disfavored.

 This would be the band's final release that features lead guitarist Donnie Dacus, who had followed late, founding lead guitarist, Terry Kath.

 All band members would contribute to the songwriting (one of only two albums where this is the case, with the other being "Chicago VII").
</item 9>

<item 10>
The V.I.P.s (band)
The V.I.P.s were a British R&B musical ensemble formed in Carlisle, Cumberland, (North West England) in late 1963, out of an earlier outfit known as The
Ramrods, who had formed in Carlisle in 1960.

 After a change of personnel in April 1967, the band changed their name to "Art", and released the album "Supernatural Fairy Tales".

 They also participated to a psychedelic bizarre album called "Featuring The Human Host And The Heavy Metal Kids" by a collective known as Hapshash and the
Coloured Coat, formed by Guy Stevens and an influential British graphic design and avant-garde musical partnership between Michael English and Nigel Waymouth.

 The musicians involved in that project were Mike Harrison on keys and vocals, Luther Grosvenor on guitars, Greg Ridley on bass and Mike Kellie on drums, as
well as performances by Stevens, English and Waymouth.

 It was the first time that the term "heavy metal" was ever used in music, even though that album had nothing to do with heavy metal music, being closer to
psychedelic music.
</item 10>
=== END ITEM id=a132-68fe994af99e4f66 ===

=== ITEM id=a046-e9d2057fdf19cf79 ===
Gold sentences and facts:
Sentence 0: Wired removed the article for not meeting its standards, but the degree to which it was shared on social media and boosted in write-ups at other outlets shows just how willing people are to accept foul play on Google’s part.
  Fact 0.0: Wired removed the article for not meeting its standards.

Retrieved items:
<item 1>
News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line (TechCrunch, 2023-12-15T17:56:02+00:00)
A new class action lawsuit filed this week in the U.S. District Court in D.C. accuses Google and parent company Alphabet of anticompetitive behavior in
violation of U.S. antitrust law, the Sherman Act, and others, on behalf of news publishers. The case, filed by Arkansas-based publisher Helena World Chronicle,
argues that Google “siphons off” news publishers’ content, their readers and ad revenue through anticompetitive means. It also specifically cites new AI
technologies like Google’s Search Generative Experience (SGE) and Bard AI chatbot as worsening the problem.

In the complaint, Helena World Chronicle, which owns and publishes two weekly newspapers in Arkansas, argues that Google is “starving the free press” by sharing
publishers’ content on Google, losing them “billions of dollars.”

In addition to new AI technologies, the suit points to Google’s older question-and-answer technologies, like the “Knowledge Graph” launched in May 2012, as part
of the problem.
</item 1>

<item 2>
Early impressions of Google’s Gemini aren’t great (TechCrunch, 2023-12-07T16:12:30+00:00)
prompts that get around the safety filters in place to attempt to prevent it from discussing controversial topics.

Using an automated method to algorithmically change the context of prompts until Gemini Pro’s guardrails failed, AI security researchers at Robust Intelligence,
a startup selling model-auditing tools, managed to get Gemini Pro to suggest ways to steal from a charity and assassinate a high-profile individual (albeit with
“nanobots” — admittedly not the most realistic weapon of choice).

Now, Gemini Pro isn’t the most capable version of Gemini — that model, Gemini Ultra, is set to launch sometime next year in Bard and other products. Google
compared the performance of Gemini Pro to GPT-4’s predecessor, GPT-3.5, a model that’s around a year old.

But Google nevertheless promised improvements in reasoning, planning and understanding with Gemini Pro over the previous model powering Bard, claiming Gemini
Pro was better at summarizing content, brainstorming and writing. Clearly, it has some work to do in those departments.
</item 2>

<item 3>
News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line (TechCrunch, 2023-12-15T17:56:02+00:00)
When online magazine The Atlantic modeled what would happen if Google integrated AI into search, it found that 75% of the time the AI would answer the user’s
query without requiring a click-through to its website, losing it traffic. This could have a major impact on publishers’ traffic going forward, as Google today
drives nearly 40% of their traffic, according to data from Similarweb.

Some publishers are now trying to get ahead of the problem. For example, Axel Springer just this week inked a deal with OpenAI to license its news for AI model
training. But overall, publishers believe they’ll lose somewhere between 20-40% of their website traffic when Google’s AI products fully roll out, The WSJ’s
report noted.

The lawsuit reiterates this concern, claiming that Google’s recent advances in AI-based search were implemented with “the goal of discouraging end-users from
visiting the websites of Class members who are part of the digital news and publishing line of commerce.”
</item 3>

<item 4>
Google fakes an AI demo, Grand Theft Auto VI goes viral and Spotify cuts jobs (TechCrunch, 2023-12-09T21:16:17+00:00)
Most read

AI, faked: Google unveiled a new flagship AI model this week called Gemini. But it didn’t release the full model, Gemini Ultra — only a “lite” version called
Gemini Pro. In a press briefing and blog posts, Google touted Gemini’s coding capabilities and multimodal prowess, claiming that the model can understand
images, audio and videos just as well as text. But Gemini Pro — which is strictly text-in, text-out — has proven to be mistake-prone. And in a worse look for
Google, the company was caught faking a Gemini demo by tuning text prompts with still images off camera.

Offensive notebooks: In another Google PR blunder, people who attended the company’s K&I Black Summit in August were given third-party notebooks containing
highly insensitive language. My colleague Dominic-Madori writes that the inside of the notebooks were printed with the phrase “I was just cotton the moment, but
I came back to take your notes” (emphasis ours).
</item 4>

<item 5>
News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line (TechCrunch, 2023-12-15T17:56:02+00:00)
A Google spokesperson offered a statement on the lawsuit, saying “This lawsuit is meritless. People have many ways to access information and news content today
– through publishers’ websites, dedicated apps, social media platforms, print papers and more. Google links people to publishers’ websites more than 24 billion
times each month – at no cost to them.”

The complaint is available below.

Helena World Chronicle, LLC v. Google LLC and Alphabet Inc by TechCrunch on Scribd

Editor’s note: This post was updated after publication with a statement provided by Google on Friday evening.
</item 5>

<item 6>
News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line (TechCrunch, 2023-12-15T17:56:02+00:00)
SGE, it argues, offers web searchers a way to seek information in a conversational mode, but ultimately keeps users in Google’s “walled garden” as it
“plagiarizes” their content. Publishers also can’t block SGE because it uses the same web crawler as Google’s general search service, GoogleBot.

Plus, it says Google’s Bard AI was trained on a dataset that included “news, magazine and digital publications,” citing both a 2023 report from the News Media
Alliance and a Washington Post article about AI training data for reference. (The Post, which worked with researchers at the Allen Institute for AI, had found
that News and Media sites were the third largest category of AI training data.)

The case points to other concerns, too, like changing AdSense rates and evidence of improper spoliation of evidence on Google’s part, by its destruction of chat
messages — an issue raised in the recent Epic Games lawsuit against Google over app store antitrust issues, which Epic won.
</item 6>

<item 7>
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
ChatGPT: Everything you need to know about the AI-powered chatbot (TechCrunch, 2023-09-28T20:03:39+00:00)
In its privacy policy, the ChatGPT maker makes a passing acknowledgement of the objection requirements attached to relying on “legitimate interest” (LI),
pointing users towards more information about requesting an opt out — when it writes: “See here for instructions on how you can opt out of our use of your
information to train our models.”

What controversies have surrounded ChatGPT?

Recently, Discord announced that it had integrated OpenAI’s technology into its bot named Clyde where two users tricked Clyde into providing them with
instructions for making the illegal drug methamphetamine (meth) and the incendiary mixture napalm.

An Australian mayor has publicly announced he may sue OpenAI for defamation due to ChatGPT’s false claims that he had served time in prison for bribery. This
would be the first defamation lawsuit against the text-generating service.

CNET found itself in the midst of controversy after Futurism reported the publication was publishing articles under a mysterious byline completely generated by
AI.
</item 9>

<item 10>
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
</item 10>
=== END ITEM id=a046-e9d2057fdf19cf79 ===

=== ITEM id=a058-85b59bc880b69feb ===
Gold sentences and facts:
Sentence 0: He immediately left the game and proceeded to the locker room following the hit, and it was confirmed that he would not return with a head injury.
  Fact 0.1: He went to the locker room after the hit.

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
Fantasy Football RB Rankings Week 14: Who to start, best sleepers at running back (Sporting News, 2023-12-07T22:07:00+00:00)
We'll talk more about Derrick Henry (concussion) in the injury section, but we can say with confidence here and now that Tyjae Spears (@ Dolphins) should be the
hottest waiver-wire pickup of Week 14. Tennessee is a run-first offense and King Henry might be sidelined with a head injury, so Spears could very well pick up
where he left off against Indy last weekend. The young back averaged 4.7 yards per carry (just shy of his 5.1 YPC entering the week) en route to 88 yards.

Another backup getting a boost this week is the Patriots' Ezekiel Elliott (@ Steelers), who will assume lead-back duties after Rhamondre Stevenson went down
with a pretty bad ankle injury in Week 13. Zeke might put in a vintage performance considering James Conner just went back to Pittsburgh and hung 105 yards and
two TDs on his old squad.

Could this be the weekend that Keaton Mitchell (vs.
</item 2>

<item 3>
2023 NFL Week 11 injury roundup: Rams WR Cooper Kupp, Seahawks RB Kenneth Walker III among stars to exit (CBSSports.com, 2023-11-19T23:51:49+00:00)
The former All-Pro, who battled a lingering hamstring injury earlier this year, received attention on the sidelines after limping off, then left the game for
further evaluation in the locker room. He was ruled questionable to return. NFL Media reported on Monday that Kupp suffered a low ankle sprain and will be
evaluated going forward.

Darius Slayton (Giants) had a big impact against the Commanders, hauling in a deep score from Tommy DeVito, before exiting with an arm injury. He was ruled out
for the remainder of the contest halfway through the third quarter.

EDGE

Joey Bosa (Chargers) appeared to injure his right leg or foot while held in a block during a first-quarter pass-rushing attempt against the Packers. He limped
to the sidelines and was carted to the locker room, then ruled questionable to return.
</item 3>

<item 4>
2023 NFL Week 11 injury roundup: Rams WR Cooper Kupp, Seahawks RB Kenneth Walker III among stars to exit (CBSSports.com, 2023-11-19T23:51:49+00:00)
Dillon. Green Bay ruled him questionable to return, then downgraded him to out, but coach Matt LaFleur said after the Packers' win over Los Angeles that he
doesn't expect the injury to be a long-term issue.

De'Von Achane (Dolphins) returned from injured reserve -- and a four-game absence -- but limped to the sidelines after his second touch of the first quarter
against the Raiders. Miami ruled him questionable to return with a knee injury, and while he later returned to the sidelines from a locker-room visit, he was
replaced on the field indefinitely by Raheem Mostert. Head coach Mike McDaniel told reporters on Monday that Achane is day-to-day. While he hasn't been ruled
out of Friday's matchup with the Jets, he's not been given the green light either, so his status is worth monitoring.

Kenneth Walker III (Seahawks) took four carries against the Rams before leaving with an oblique injury.
</item 4>

<item 5>
Fantasy Football RB Rankings Week 14: Who to start, best sleepers at running back (Sporting News, 2023-12-07T22:07:00+00:00)
We also don't love Jerome Ford or Kareem Hunt (vs. Jaguars), as Jacksonville entered Week 12 allowing just 12.3 standard fantasy points per game to RBs (third
lowest in the NFL).

We say it almost every week, but the song remains the same: don't start Carolina backs Chuba Hubbard or Miles Sanders (@ Saints). It's even worse than usual
this week, as New Orleans has a pretty hard-hitting front-seven and the Bayou is no place for below-average offenses.

What are the biggest RB injuries in Week 14?

Two-time rushing champ Derrick Henry (head) suffered a huge blow to the head and neck area in Tennessee's Week 13 loss to the Colts, so don't be surprised if
the king gives up his throne to Tyjae Spears this week. He immediately left the game and proceeded to the locker room following the hit, and it was confirmed
that he would not return with a head injury.
</item 5>

<item 6>
2023 NFL Week 11 injury roundup: Rams WR Cooper Kupp, Seahawks RB Kenneth Walker III among stars to exit (CBSSports.com, 2023-11-19T23:51:49+00:00)
Smith did return, however, for the final drive of the game in the losing effort. Head coach Pete Carroll said that Smith will "make it back." When asked if that
means for Thursday's matchup against San Francisco, the coach said, "I would think so."

Sam Howell (Commanders) received medical attention on the sidelines after enduring several hard hits to the upper body on a touchdown run against the Giants. He
was quickly cleared of a potential concussion, however, and remained in the game. Veteran journeyman Jacoby Brissett is Washington's backup.

RB

Aaron Jones (Packers) suited up against the Chargers despite a lingering hamstring issue, only to manage just five carries before suffering a knee injury. The
former Pro Bowler needed help to leave the field and was then carted to the locker room, replaced by A.J. Dillon.
</item 6>

<item 7>
2023 NFL Week 11 injury roundup: Rams WR Cooper Kupp, Seahawks RB Kenneth Walker III among stars to exit (CBSSports.com, 2023-11-19T23:51:49+00:00)
I agree to receive the "Pick Six Newsletter" and marketing communications, updates, special offers (including partner offers), and other information from CBS
Sports and the Paramount family of companies. By pressing sign up, I confirm that I have read and agree to the Terms of Use and acknowledge Paramount's Privacy
Policy See All Newsletters Please check the opt-in box to acknowledge that you would like to subscribe. Thanks for signing up! Keep an eye on your inbox. Sorry!
There was an error processing your subscription.

WR

Tyreek Hill (Dolphins) briefly left Miami's game against the Raiders while nursing a hand injury and was ruled questionable to return, but he took the field
again soon afterward, securing seven catches for 80 yards and a score before halftime. Hill has left multiple games this year, only to return in all of them.

Cooper Kupp (Rams) had just a single catch against the Seahawks before suffering an ankle injury while blocking.
</item 7>

<item 8>
Fantasy Football Week 13 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-11-30T22:10:58+00:00)
Sit Him (Lineup Decisions) Projections powered by Sportsline Cooper Kupp WR LAR L.A. Rams • #10

Age: 30 • Experience: 7 yrs. Weekly Breakdown MATCHUP vs CLE LAR -3.5 O/U 40 OPP VS WR 14th PROJ PTS 13.8 WR RNK 25th YTD Stats REC 27 TAR 51 REYDS 393 TD 1
FPTS/G 10.6 You're never supposed to sit your studs, but what if your stud is playing through something that made him look like a dud last week? That was Kupp,
who in Week 12 ran a ton of short routes and couldn't separate and didn't have much sudden movement. If he's going to come through this week it'll likely come
on a ton of short targets and/or some targets inside the 10-yard line.
</item 8>

<item 9>
2023 NFL Week 11 injury roundup: Rams WR Cooper Kupp, Seahawks RB Kenneth Walker III among stars to exit (CBSSports.com, 2023-11-19T23:51:49+00:00)
DB

Talanoa Hufanga (49ers), who emerged as a ballhawk for San Francisco's secondary in 2022, left early against the Buccaneers with a knee injury. Coach Kyle
Shanahan confirmed to reporters on Monday that the All-Pro suffered a season-ending ACL tear.
</item 9>

<item 10>
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
</item 10>
=== END ITEM id=a058-85b59bc880b69feb ===

=== ITEM id=a059-5f6dd309cd6d764c ===
Gold sentences and facts:
Sentence 0: During character creation, you must pick the otherworldy patron that you’ve decided to serve.
  Fact 1.1: The patron you must pick during character creation is the one you have decided to serve.

Retrieved items:
<item 1>
The best Warlock subclass, feats, and build in Baldur’s Gate 3 (Polygon, 2023-12-01T22:53:00+00:00)
The Warlock is one of twelve classes in Baldur’s Gate 3. Warlocks are extreme knowledge-seekers who are willing to form a pact with otherwordly beings in order
to gain ancient secrets that’ll bolster their magical arsenal.

To deal with such powerful creatures, Warlocks need to have strong mental power. They need to be able to make a deal with the devil and survive. As such, their
primary ability score is Charisma.

There’s no single best class in Baldur’s Gate 3, and the game gives you as many opportunities for creative problem solving as it does for punching. Finding the
best class really just boils down to how you want to play. Even when playing solo, you’ll still manage the rest of your party, so you’ll get to experiment with
multiple classes at once.

Read on to find out everything you need to know about the Warlock class in Baldur’s Gate 3, and learn all about the best Warlock subclass, feats, and build.
</item 1>

<item 2>
How to choose the best class for you in Baldur’s Gate 3 (Polygon, 2023-12-18T17:14:20+00:00)
They also get a way to heal themselves and way to get a second attack once per short rest.

At level 3, you get to choose a subclass from Battle Master, Eldritch Knight, and Champion. Battlemasters get a set of Superiority Dice that let you perform
Maneuvers to control the flow of battle and the battlefield. Eldritch Knights add magic into the Fighter’s arsenal. Champions keep it simple with a better
change to deal a Critical Hit (for double damage).

Champion is the best subclass for a Fighter. Consult our Fighter build guide for more details on why, and to see which skills and feats to pick.

Best ranged class for beginners: Ranger

Rangers are agile hunters that focus on Dexterity as their primary Ability. Rangers are an interesting mix of melee, stealth, skills, and magic. That also makes
it harder to get the most out of the class. Rangers get Light and Medium Armors and some of the more basic weapons.
</item 2>

<item 3>
The best Warlock subclass, feats, and build in Baldur’s Gate 3 (Polygon, 2023-12-01T22:53:00+00:00)
Spell Sniper. You learn a cantrip, and the number you need to roll a Critical Hit while attacking is reduced by 1. This effect can stack.

For more information on the classes in Baldur’s Gate 3, consult our overviews on the Barbarian, Bard, Cleric, Druid, Fighter, Monk, Paladin, Ranger, Rogue,
Sorcerer, and Wizard classes. If you’re stumped by the choice, consult our guide on what class to start with.
</item 3>

<item 4>
The best Warlock subclass, feats, and build in Baldur’s Gate 3 (Polygon, 2023-12-01T22:53:00+00:00)
Your Charisma Increases by 1, to a maximum of 20. Your Proficiency Bonus is also doubled for Deception and Performance Checks.

. Your Charisma Increases by 1, to a maximum of 20. Your Proficiency Bonus is also doubled for Deception and Performance Checks. Elemental Adept . Your spells
ignore Resistance to a damage type of your choice. When you cast spells of that type, you cannot roll a 1.

. Your spells ignore Resistance to a damage type of your choice. When you cast spells of that type, you cannot roll a 1. Moderately Armored . You gain Armor
Proficiency with Medium Armor and shields, and your Strength or Dexterity increases by 1, to a maximum of 20.

. You gain Armor Proficiency with Medium Armor and shields, and your Strength or Dexterity increases by 1, to a maximum of 20. Spell Sniper.
</item 4>

<item 5>
How to choose the best class for you in Baldur’s Gate 3 (Polygon, 2023-12-18T17:14:20+00:00)
For more information on the classes in Baldur’s Gate 3, consult our overviews on the Barbarian, Bard, Cleric, Druid, Fighter, Monk, Paladin, Ranger, Rogue,
Sorcerer, Warlock, and Wizard classes.
</item 5>

<item 6>
The best Warlock subclass, feats, and build in Baldur’s Gate 3 (Polygon, 2023-12-01T22:53:00+00:00)
Warlock class features

Hit Points

Hit Dice — 1d8 per warlock level

1d8 per warlock level Hit Points at 1st Level — 8 + your Constitution modifier

— 8 + your Constitution modifier Hit Points at higher levels — 5 + your Constitution modifier per warlock level after 1st

Cantrips

The Warlock starts with two cantrips from the Warlock cantrip list.

Baldur’s Gate 3 Warlock Cantrips Cantrip Description Cantrip Description Blade Ward Take only half the damage from Bludgeoning, Piercing, and Slashing attacks
Bone Chill Prevent the target from healing until your next turn. An undead target receives Disadvantage on Attack Rolls. Eldritch Blast Conjure 1 beam(s) of
crackling energy. Friends Gain Advantage on Charisma Checks against a non-hostile creature. Mage Hand Create a spectral hand that can manipulate and interact
with objects. Minor Illusion Create an illusion that compels nearby creatures to investigate.
</item 6>

<item 7>
How to choose the best class for you in Baldur’s Gate 3 (Polygon, 2023-12-18T17:14:20+00:00)
For weapons, Barbarians get a wide range of heavy hitters like battleaxes and warhammers. Every Barbarian gets the Rage ability that lets them hit even harder
and reduce damage during combat (and only during combat).

When you hit level 3, you get to choose from the Wildheart, Berserker, or Wild Magic subclasses. Wildheart lets you modify your Rage ability with animal-themed
tweaks like healing, jumping, or a stampede. Berserker is a lot more straightforward and focuses on hitting (extra) things and throwing. And Wild Magic adds a
magical flair to your Rage with a random magical effect and improved saving throws against magic for you and your allies.

Berserker is the best subclass for a Barbarian. Consult our Barbarian build guide for more details on why, and to see which skills and feats to pick.

Best class for magic: Druid

Druids are the nature-based casters of Baldur’s Gate 3.
</item 7>

<item 8>
How to choose the best class for you in Baldur’s Gate 3 (Polygon, 2023-12-18T17:14:20+00:00)
Rangers get Light and Medium Armors and some of the more basic weapons.

Rangers also make a lot of choices during character creation (and leveling up) that determine how they play. At level 1, you’ll choose a Favored Enemy — a
preferred quarry — and a preferred environment that grants you some magical spells.

At level 3, you get to choose a Ranger subclass from Beast Master, Gloom Stalker, or Hunter. Beast Masters get a Companion animal that fights along side you,
Gloom Stalkers focus on stealth, and Hunters get to specialize their melee abilities.

Hunter is the best subclass for a Ranger. Consult our Ranger build guide for more details on why, and to see which skills and feats to pick.

Best class for damage: Barbarian

Barbarians are all about combat. They get light and medium armor (and shields), but they’ve also got the option to keep their AC up even without wearing armor.
</item 8>

<item 9>
The best Warlock subclass, feats, and build in Baldur’s Gate 3 (Polygon, 2023-12-01T22:53:00+00:00)
If you’re bound to The Archfey, you’ll gain an action named “Fey Presence,” which will charm or frighten nearby foes.

The Archfey also has two exclusive spells — Faerie Fire and Sleep.

Best Warlock subclass and build in BG3

Baldur’s Gate 3 races don’t really have any direct impact on your class, but certain races and subraces come with features that play well with certain classes
like extra movement or proficiency with certain weapons.
</item 9>

<item 10>
The best Warlock subclass, feats, and build in Baldur’s Gate 3 (Polygon, 2023-12-01T22:53:00+00:00)
For Warlocks, it’s best to focus on weapon and armor proficiencies:

Drow ’s Drow Weapon Training adds weapons like the rapiers and hand crossbows

’s Drow Weapon Training adds weapons like the rapiers and hand crossbows Humans ’ and Half-Elves ’ Civil Militia add spears, pikes, halberds, and glaives along
with shields

’ and ’ Civil Militia add spears, pikes, halberds, and glaives along with shields Githyanki Martial Prodigy means you get medium armor along with longswords and
greatswords

Martial Prodigy means you get medium armor along with longswords and greatswords Dwarven Combat Training gets you battleaxes, handaxes, light hammers, and
warhammers. The Shield Dwarf subrace also gets you Medium Armor

When you’re creating and leveling up your Warlock, Charisma (for spellcasting) is your most important stat.
</item 10>
=== END ITEM id=a059-5f6dd309cd6d764c ===

=== ITEM id=a040-3461843f84b2ecae ===
Gold sentences and facts:
Sentence 0: Using Caroline Ellison, his on-again off-again girlfriend and the CEO of Alameda, as a front, Bankman-Fried had “secret access” to customer money—both cash and crypto—the government claimed.
  Fact 0.2: Caroline Ellison was the CEO of Alameda.

Retrieved items:
<item 1>
In the end, the FTX trial was about the friends screwed along the way (The Verge, 2023-10-26T12:00:00+00:00)
A substantial chunk had been spent on various investments, including in the celebrity-connected venture firm K5, trading firm Modulo (co-founded by one of
Bankman-Fried’s ex-girlfriends), and AI company Anthropic.

Another $413 million had gone directly to Bankman-Fried himself, through his wholly-owned company Paper Bird. “Outflows to insiders,” which suggested Singh’s
not-quite-loans were included, was $192 million; real estate was another $228 million; and the donations to Gabe Bankman-Fried’s Guarding Against Pandemics,
combined with the philanthropy from the FTX Foundation, totaled less than $40 million.

It is easy to lie with words and harder to lie with money.
</item 1>

<item 2>
Sam Bankman-Fried was a terrible boyfriend (The Verge, 2023-10-10T23:50:21+00:00)
It was that Ellison’s calculations assumed Alameda could borrow $1.8 billion in normal dollars and $1.5 billion in crypto from FTX. The spreadsheet makes this
clear with a row labeled “FTX borrows,” which Ellison said were customer funds.

Meanwhile, echoing Bankman-Fried and Ellison’s romantic relationship, FTX was keeping cozy private ties with Alameda yet publicly holding it at arm’s length. On
January 14th, 2022, Bankman-Fried tweeted, “We’re launching a $2b venture fund: FTX Ventures!” Those funds came from Alameda, Ellison testified. But
Bankman-Fried didn’t want to go public with the source of the funds. He said he thought Alameda’s brand was less good, and he didn’t want his name associated
with it. Alameda also bought Robinhood shares for Bankman-Fried, who moved them to a vehicle called “Emergent Fidelity Technologies” to avoid association with
Alameda.
</item 2>

<item 3>
SBF Trial: The latest updates from the FTX collapse’s courtroom drama (TechCrunch, 2023-10-06T17:22:02+00:00)
Much of said funding came from customers, and Easton testified that he found that user deposits were used by both FTX and Alameda to purchase real estate, make
investments or direct funding toward political causes and charities.

SBF trial: What we learned in week two

The second week of the trial’s standout testimony came from Alameda Research’s former CEO Caroline Ellison, who claimed she took $14 billion from customers to
repay debts to lenders, using them as a line of credit under the instruction of SBF. Ellison also went into detail about how a $2 billion venture fund, FTX
Ventures, was funded by money that had previously been allocated to Alameda from third-party lenders.

And though there are no charges of bribery in the case, Ellison also testified to paying Chinese officials $150 million to restore access to $1 billion in
frozen trading accounts.

SBF trial: What we learned in the week one

As expected, the opening statements in the SBF trial were spicy.
</item 3>

<item 4>
Sam Bankman-Fried was a terrible boyfriend (The Verge, 2023-10-10T23:50:21+00:00)
Using the effectively unlimited funds “allowed us to make profitable trades we couldn’t have made otherwise”

Bankman-Fried didn’t grant Ellison equity, even though she asked; he told her it would be too complicated. Instead, she got a $200,000 salary, even as CEO, and
bonuses twice a year, which ranged from $100,000 to $20 million.

Initially, Alameda and FTX were “very integrated,” Ellison said. They were run by the same team, from the same office. And when Alameda was scrounging for
funds, Bankman-Fried told Ellison that FTX would be a good source of capital. The $65 billion line of credit Alameda Research had meant that it did not have to
post collateral. There was no contract and no written terms, she testified. It also wasn’t visible to FTX’s auditors — she’d raised the question with
Bankman-Fried, and he told her not to worry about it.
</item 4>

<item 5>
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
</item 5>

<item 6>
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
</item 6>

<item 7>
The jury finally hears from Sam Bankman-Fried (The Verge, 2023-10-28T00:12:41+00:00)
Then, Nishad Singh, about a month after founding Alameda Research. Finally, Caroline Ellison.

Not having a risk team, when you are any kind of financial anything, is certainly a choice

Though Bankman-Fried was the CEO, and also the majority owner, he wanted to be clear: he did not supervise Wang’s direct work. Anyway, after a bunch of wildly
successful arbitrage — 50 percent to 100 percent annualized returns, per his testimony — he decided to found a cryptocurrency exchange, FTX. He figured he’d
fail; that there was only a 20 percent chance of success. Bankman-Fried did not define what he expected the time period to be on this estimate, but arguably 20
percent was a much higher chance of success than FTX would enjoy once Alameda dipped into the customer deposits.

By the way, because he was such a good guy, Bankman-Fried made a point of “periodically” handling support tickets himself.
</item 7>

<item 8>
The jury finally hears from Sam Bankman-Fried (The Verge, 2023-10-28T00:12:41+00:00)
Bankman-Fried said he worked 12 to 22 hours a day, and took one day off every couple of months. Because FTX had grown so much, he could no longer run both
companies, he said. Bankman-Fried handed the company off to Caroline Ellison and Sam Trabucco, who immediately after being named co-CEO promptly drifted away to
early retirement. (Quiet quitting king!) Bankman-Fried did remain involved in hedging and risk at Alameda, though.

About that $1 billion of revenue in 2021: Bankman-Fried definitely did not know that Singh, his employee, had backdated interest payments to get FTX “over the
line” to $1 billion. See, he’d just asked his employees to check and see if there was any source of funds that was missing to get to $1 billion. This testimony
was especially rambling.

Oh, also that MobileCoin loss? The one Wang said Alameda took to keep off FTX’s balance sheet?
</item 8>

<item 9>
In the end, the FTX trial was about the friends screwed along the way (The Verge, 2023-10-26T12:00:00+00:00)
Then a bigger one. Then an effectively limitless one.

Ellison testified to something similar, choking back tears: “When I started working at Alameda, I don’t think I would have believed you if you told me I would
be sending false balance sheets to our lenders or taking customer money, but over time, it was something I became more comfortable with.”

Ellison’s tenure as CEO of Alameda Research was an exercise in optics, she testified. Alameda Research was, by then, FTX’s sin-eater. When FTX took a loss in
the hundreds of million dollars because a trader had exploited its system, Wang moved that loss onto Alameda’s balance sheet. FTX’s “insurance fund,” meant to
keep traders from getting stuck with other traders’ losses, was totally made up — Alameda simply picked up the tab.

Even that wasn’t good enough.
</item 9>

<item 10>
In the end, the FTX trial was about the friends screwed along the way (The Verge, 2023-10-26T12:00:00+00:00)
Wang trusted him.

Wang also set Alameda Research up with an enormous line of credit, unmatched by anyone else on FTX. Ultimately, the number was $65 billion — the result of
repeated increases as Alameda Research repeatedly ran up against its credit limit. This line of capital wasn’t visible to auditors, Ellison testified. That
credit line was used primarily for trading because it “allowed us to make profitable trades we couldn’t have made otherwise,” Ellison testified.

“Over time, it was something I became more comfortable with.”

Wang’s testimony revealed how the fraud got bigger, little by little. First, it was okay to dip into FTX’s profits. Then, it was okay to take more. Then, it was
okay to take even more. First, it was a little line of credit just to provide liquidity. Then a bigger one. Then an effectively limitless one.
</item 10>
=== END ITEM id=a040-3461843f84b2ecae ===

=== ITEM id=a113-98cc0b9220a49e55 ===
Gold sentences and facts:
Sentence 0: Colonel Charles Hastings Judd (September 8, 1835 – April 18, 1890) was an American businessman, rancher, courtier and politician of the Kingdom of Hawaii.
  Fact 0.7: Charles Hastings Judd was a businessman, rancher, courtier and politician of the Kingdom of Hawaii.

Retrieved items:
<item 1>
Corps of Military Staff Clerks
The Corps of Military Staff Clerks was an administrative corps of the Canadian Army.

 The Headquarters Corps of Military Staff Clerks was established in 1905.
</item 1>

<item 2>
Helmuth von Moltke the Younger
Helmuth Johann Ludwig von Moltke (] ; 23 May 1848 – 18 June 1916), also known as Moltke the Younger, was a nephew of "Generalfeldmarschall" (Field Marshal)
Helmuth Karl Bernhard von Moltke and served as the Chief of the German General Staff from 1906 to 1914.

 The two are often differentiated as "Moltke the Elder" and "Moltke the Younger".
</item 2>

<item 3>
Algernon Willis
Admiral of the Fleet Sir Algernon Usborne Willis (17 May 1889 – 12 April 1976) was a Royal Navy officer.

 He served in the First World War and saw action at the Battle of Jutland in May 1916.

 He also served in the Second World War as Commander-in-Chief, South Atlantic in which capacity he led actions against German and Japanese raiding ships.

 He continued his war service as Flag Officer commanding 3rd Battle Squadron and Second in command of the Eastern Fleet and then as Flag Officer commanding
Force H, the force which covered North African Operations, the Allied invasion of Sicily in July 1943 and then the Allied invasion of Italy in September 1943.

 He spent the final years of the war as Commander-in-Chief, Levant, in which capacity he conducted naval operations in support of the Dodecanese Campaign, and
then as Second Sea Lord, in which capacity he arranged the manpower for the campaign in the Pacific Ocean against the Imperial Japanese Navy.

 After the war he served as Commander-in-Chief, Mediterranean Fleet, in which role he was faced with unrest in Mandatory Palestine, before he became
Commander-in-Chief, Portsmouth.
</item 3>

<item 4>
Franz Halder
Franz Halder (30 June 1884 – 2 April 1972) was a German general and the chief of the "Oberkommando des Heeres" staff (OKH, Army High Command) from 1938 until
September 1942, when he was dismissed after frequent disagreements with Adolf Hitler.

 Until December 1941 Halder's military position corresponded to the old Chief of the General Staff position, which during World War I had been the highest
military office in the German Imperial Army.

 Halder's diary during his time as chief of OKH General Staff has been a source for authors that have written about such subjects as Hitler, World War II and
the Nazi Party.

 In William Shirer's "The Rise and Fall of the Third Reich", Halder's diary is cited hundreds of times.
</item 4>

<item 5>
Arthur Tedder, 1st Baron Tedder
Marshal of the Royal Air Force Arthur William Tedder, 1st Baron Tedder, {'1': ", '2': ", '3': ", '4': "} (11 July 1890 – 3 June 1967) was a senior Royal Air
Force commander.

 He was a pilot and squadron commander in the Royal Flying Corps in the First World War and he went on to serve as a senior officer in the Royal Air Force
during the inter-war years when he served in Turkey, Great Britain and the Far East.

 During the Second World War, as Air Officer Commanding RAF Middle East Command, Tedder directed air operations in the Mediterranean and North Africa, including
the evacuation of Crete and "Operation Crusader" in North Africa.

 His bombing tactics became known as the "Tedder Carpet".

 Later in the war Tedder took command of Mediterranean Air Command and in that role was closely involved in the planning of the Allied invasion of Sicily and
then the Allied invasion of Italy.

 When Operation Overlord—the invasion of France—came to be planned, Tedder was appointed Deputy Supreme Commander at Supreme Headquarters Allied Expeditionary
Force under General Eisenhower.
</item 5>

<item 6>
German General Staff
The German General Staff, originally the Prussian General Staff and officially Great General Staff ("Großer Generalstab"), was a full-time body at the head of
the Prussian Army and later, the German Army, responsible for the continuous study of all aspects of war, and for drawing up and reviewing plans for
mobilization or campaign.

 It existed unofficially from 1806, and was formally established by law in 1814, the first general staff in existence.

 It was distinguished by the formal selection of its officers by intelligence and proven merit rather than patronage or wealth, and by the exhaustive and
rigorously structured training which its staff officers undertook.

 Its rise and development gave the German armed forces a decisive strategic advantage over their adversaries for nearly a century and a half.
</item 6>

<item 7>
Herbert Büchs
Herbert Büchs (20 November 1913 – 19 May 1996) was a Lieutenant General of the German Air Force and a former Luftwaffe staff officer in Nazi Germany's
Oberkommando der Wehrmacht (OKW) during World War II.

 As second adjutant to General Alfred Jodl with the rank of Major he is notable for being present in the conference room when the 20 July plot bomb exploded in
1944.
</item 7>

<item 8>
Kenneth Anderson (British Army officer)
General Sir Kenneth Arthur Noel Anderson, (25 December 1891 – 29 April 1959) was a senior British Army officer who saw service in both World wars.

 He is mainly remembered as the commander of the British First Army during Operation Torch, the Allied invasion of North Africa.

 He had an outwardly reserved character and did not court popularity either with his superiors or with the public.

 His American superior, General Dwight D. Eisenhower, wrote that he was "blunt, at times to the point of rudeness".

 In consequence he is less well known than many of his contemporaries.

 Richard Mead wrote that "He handled a difficult campaign more competently than his critics suggest, but competence without flair was not good enough for a top
commander in 1944".
</item 8>

<item 9>
HMS Queen Elizabeth March
"HMS Queen Elizabeth March", composed by WO2 bandmaster John Morrish RM, was the winning composition for the 2012 Royal Marines Band Service March Competition,
sponsored by the Aircraft Carrier Alliance.

 The march was written for the first of the new generation, of Royal Navy Queen Elizabeth-class aircraft carriers.
</item 9>

<item 10>
Marisha Chamberlin
Marisha Anne Chamberlain (born January 6, 1952) is an American writer.

 Her most recent work, the libretto for "Mortals & Angels", a collaboration with American composer Carol Barnett, premiered at Carnegie Hall in 2016, a
companion piece to their widely produced collaborative piece, "The World Beloved: A Bluegrass Mass".

 Her stage plays, both original works and adaptations are widely produced.

 Her debut novel, "The Rose Variations", was published by Soho Press in 2009.

 Her play, "Scheherazade", won the Dramatists Guild/CBS National Award, and in her screenplay version, played on public television across the country, and was
screened at the British Film Institute Festival in the category of Best of American Public Television.
</item 10>
=== END ITEM id=a113-98cc0b9220a49e55 ===

=== ITEM id=a068-46c70067a510142e ===
Gold sentences and facts:
Sentence 0: The 'Burbs is a 1989 American comedy thriller film directed by Joe Dante starring Tom Hanks, Bruce Dern, Carrie Fisher, Rick Ducommun, Corey Feldman, Wendy Schaal and Henry Gibson.
  Fact 0.6: The 'Burbs stars Rick Ducommun.

Retrieved items:
<item 1>
Pizza 73
Pizza 73 is a Canadian restaurant chain that offers a number of different styles of pizza, along with chicken wings.

 It has been operated by Pizza Pizza since 2007.

 Toronto-based Pizza Pizza had acquired the restaurant for a total of $CAN70.2 million.

 There are 89 locations throughout Western Canada, which include the provinces of British Columbia, Alberta, and Saskatchewan.

 The restaurant's name originates from its original phone number: 473–7373.

 Founded by David Tougas and Guy Goodwin in 1985, Pizza 73 is headquartered in Edmonton, Alberta, Canada.
</item 1>

<item 2>
Night Life (film)
Night Life is a 1989 American zombie comedy film directed by David Acomba and starring Scott Grimes, John Astin, and Cheryl Pollak.

 It centers around a high school nerd named Archie Melville who is harassed from beyond the grave by the zombie corpses of his former high school bullies
(recently deceased) after they are brought back to life by a freak lightning storm.

 It is also known as Grave Misdemeanors.
</item 2>

<item 3>
Jebb Fink
Jebb Fink is an American-Canadian stand-up comedian and television personality.

 Best known as a reporter and host for "the Big Breakfast" on A-Channel, Calgary.

 He won a Gemini as co-creator of the CBC comedy "An American in Canada" and was a weather presenter on "Your City" for City TV Calgary, he also hosted the
short-lived late night talk show "Global Late Night" for the Global Television Network in 2004.
</item 3>

<item 4>
Dunston Checks In
Dunston Checks In is a 1996 American family comedy film starring Eric Lloyd, Graham Sack, Jason Alexander, Faye Dunaway, Rupert Everett, Paul Reubens, Glenn
Shadix, and introducing Sam the Orangutan as Dunston.

 It was written by John Hopkins and Bruce Graham and directed by Ken Kwapis.
</item 4>

<item 5>
Araby Lockhart
Araby Lockhart (born December 4, 1926) is a Canadian stage and television actress, best known for her performances in the films "Capote" and "Police Academy"
and her stage performances as a member of Hart House Theatre and the Straw Hat Players.

 Lockhart has also served as President of the Actors' Fund of Canada.
</item 5>

<item 6>
The 'Burbs
The 'Burbs is a 1989 American comedy thriller film directed by Joe Dante starring Tom Hanks, Bruce Dern, Carrie Fisher, Rick Ducommun, Corey Feldman, Wendy
Schaal and Henry Gibson.

 The film was written by Dana Olsen, who also has a cameo in the movie.

 The film pokes fun at suburban environments and their eccentric dwellers.
</item 6>

<item 7>
Marty Belafsky
Marty Belafsky (born September 19, 1975) is an American actor/comedian born in Los Angeles, California.

 He began acting professionally at age 13 and was soon cast as Louis Plumb on the short-lived NBC series "Hull High".

 Shortly thereafter, Belafsky landed the role of Crutchy in the Disney musical film, "Newsies".

 He continued acting through his teens, making appearances in such television shows as "The Wonder Years", "Great Scott" and "Step By Step" and the film
"Wrestling Ernest Hemingway".

 Belafsky also voiced Kent Swanson in the video game series "Dead Rising" released in 2006.
</item 7>

<item 8>
Freebie and the Bean
Freebie and the Bean is a 1974 American action-comedy film about two off-beat police detectives who wreak havoc in San Francisco attempting to bring down a
local organized crime boss.

 The picture, a precursor to the buddy cop film genre popularized a decade later, stars James Caan, Alan Arkin, Loretta Swit and Valerie Harper.

 Harper was nominated for the Golden Globe for New Star of the Year for playing the Hispanic wife of Alan Arkin.

 The film was directed by Richard Rush.

 An article in Rolling Stone magazine alleged that Stanley Kubrick called "Freebie and the Bean" the best film of 1974.

 Arkin and Caan would not appear in another movie together until the 2008 film adaptation of "Get Smart".
</item 8>

<item 9>
Neon City
Neon City (also known as Anno 2053 in Italy, Neonski Grad in Serbia) is a 1991 Canadian post-apocalyptic science fiction film directed by Monte Markham and
written by Jeff Begun and Ann Lewis Hamilton, under the pseudonym Buck Finch.

 The cast includes Michael Ironside, Vanity, Lyle Alzado, Valerie Wildman, Nick Klar, Juliet Landau, Richard Sanders and Markham.
</item 9>

<item 10>
Vegas Vacation
Vegas Vacation is a 1997 American comedy film directed by Stephen Kessler.

 It is the fourth installment in "National Lampoon"’s "Vacation" film series, and was written by Elisa Bell, based on a story by Bell and Bob Ducsay.

 The film stars Chevy Chase, Beverly D'Angelo and Randy Quaid, with Ethan Embry and Marisol Nichols as Griswold children Rusty and Audrey.

 The film opened at #4 at the box office and grossed over $36.4 million domestically.

 "Vegas Vacation" became the first theatrical "Vacation" film not to carry the National Lampoon label or a screenwriting credit from John Hughes.
</item 10>
=== END ITEM id=a068-46c70067a510142e ===

=== ITEM id=a063-4a47056b00c41fe3 ===
Gold sentences and facts:
Sentence 0: Using Caroline Ellison, his on-again off-again girlfriend and the CEO of Alameda, as a front, Bankman-Fried had “secret access” to customer money—both cash and crypto—the government claimed.
  Fact 0.2: Bankman-Fried used Caroline Ellison as a front.

Retrieved items:
<item 1>
Sam Bankman-Fried was a terrible boyfriend (The Verge, 2023-10-10T23:50:21+00:00)
Alameda’s credit line — which was taken in increments of $100,000 to $10 million at a time — was used for trading. Using the effectively unlimited funds
“allowed us to make profitable trades we couldn’t have made otherwise,” Ellison testified.

Customer funds were also used when Bankman-Fried bought back FTX shares from Binance, an early investor, in the summer of 2021. Bankman-Fried told Ellison it
was “really important,” otherwise “Binance would do things to mess with FTX.” Ellison says she told him Alameda didn’t have the money. So Bankman-Fried took $1
billion of FTX customer funds to buy out Binance, the first time Ellison recalled an amount that large. It was Bankman-Fried’s decision, she said, as he was the
CEO of FTX.

FTT was one of several “Sam coins”

There was also the FTT token, which was created by Bankman-Fried and Wang.
</item 1>

<item 2>
Sex, Signal messages, and sabotaging FTX's code: SBF execs and Bahamas roommates tell all in court (Cnbc | World Business News Leader,
2023-10-06T20:17:00+00:00)
Those privileges ultimately resulted in Alameda owing FTX $8 billion worth of customer deposits. "We gave special privileges on FTX that gave unlimited
withdrawals on the platform to Alameda," Wang said. Alameda was allowed to withdraw and transfer those funds and had a $65 billion line of credit. "When
customers deposited USD, it went to Alameda," he said. "It existed in the computer code. Alameda could have negative balances and unlimited withdrawals." That
"bug" in the code was written by Nishad Singh, who was FTX's director of engineering, and reviewed by Wang. Bankman-Fried was calling the shots, Wang said. Wang
also told the court about a $1 million personal loan he received and a $200 million to $300 million loan in his name from Alameda that was never deposited into
his account, but rather was used to make investments into other companies on behalf of FTX.
</item 2>

<item 3>
Today the FTX jury suffered through a code review (The Verge, 2023-10-06T22:21:46+00:00)
In late 2019, Alameda had a negative balance on FTX. Because they sat in an open-plan office, Wang heard a trader ask Bankman-Fried if it was okay to keep
withdrawing money from that account. Bankman-Fried said it was okay as long as it was lower than FTX’s total trading revenue. He was the CEO of both companies
at the time.

Alameda also allegedly kept FTX from looking bad by assuming some of its losses.

The amount that Alameda was allowed to go negative began to creep up. Around 2019 or 2020, Wang checked the database and discovered that Alameda was negative by
about $200 million, which was more than the $150 million FTX made in revenue. That had to mean Alameda was taking customer money. It surprised him and he says
he discussed it with Bankman-Fried, who told him that he just needed to take into account the value of FTT. Wang said he trusted Bankman-Fried’s judgment and
didn’t pursue the issue further.
</item 3>

<item 4>
Sex, Signal messages, and sabotaging FTX's code: SBF execs and Bahamas roommates tell all in court (Cnbc | World Business News Leader,
2023-10-06T20:17:00+00:00)
In June 2022, he submitted a report to Bankman-Fried on Signal that showed $8 billion in customer money held in an internal database tracking the cash wired to
an Alameda account called "fiat at ftx.com" was missing. Yedidia said he and Bankman-Fried spoke about it at the pickleball court at the resort in Nassau,
Bahamas. He asked his boss if things were OK. He was concerned because it "seemed like a lot of money" from FTX customers was at risk. "Sam said, we were
bulletproof last year. We aren't bulletproof this year," Yedidia testified. Yedidia said he asked when they would be bulletproof again. Bankman-Fried said he
wasn't sure, but it may be six months to three years. Yedidia said Bankman-Fried appeared "worried or nervous," which he said was atypical.
</item 4>

<item 5>
Sam Bankman-Fried didn’t ask where the $8 billion went (The Verge, 2023-10-31T19:52:07+00:00)
Instead, Bankman-Fried directed alleged co-conspirator Caroline Ellison to repay third-party loans and went on making investments.

Bankman-Fried said that he found out about the fiat@ftx account — that’s the one that tracked how much Alameda owed FTX — in June 2022 while his senior staff
was focused on fixing the software bug. He did not, however, discover what the account was for until October. I know this sounds unbelievable, but this is his
actual testimony.

His employees told him “they were busy and I should stop asking questions because it was distracting.”

Simple questions nailed Bankman-Fried to the wall. Had he given his employees at Alameda the direction not to spend FTX customer deposits? Had he put in any
policies to prevent Alameda employees from spending FTX customer money? Did he put in place any measures at Alameda to protect the FTX customer money? No, no,
and no. Oh, but he was testifying to Congress about keeping customer funds safe.
</item 5>

<item 6>
Graph query row:
article_title: Is Sam Bankman-Fried a bad ‘man’ or a good ‘boy’? Lawyers swap opening statements before first witnesses take the stand (Fortune,
2023-10-04T23:32:53+00:00)
publisher: Fortune
connected_entities: ['Matt Huang', 'Damian Williams', 'Gary Wang', 'Caroline Ellison', 'crypto', 'alleged crimes', 'Thane Rehn', 'Larry David', 'stealing
customer funds', 'theft', 'Mark Cohen', 'Ponzi scheme', 'crypto markets', 'Bitcoin', 'Dogecoin', 'attempted withdrawal of cash and crypto', 'Adam Yedidia',
'Jane Street', 'Sequoia Capital', 'Brooklyn', '(+22 more)']
</item 6>

<item 7>
The FTX trial is bigger than Sam Bankman-Fried (The Verge, 2023-09-28T12:00:00+00:00)
We have some inkling of what the government will be arguing thanks to the filings prosecutors are making in this case. They argue that Bankman-Fried lied about
consumer protection and that Bankman-Fried’s statements that FTX was “avoiding or managing conflicts of interest” and that “as a general principle FTX
segregates customer assets from its own assets across our platforms” were lies, in a superseding indictment filed August 14th.

The indictment also says that there were special features in FTX’s code that “permitted Alameda to spend and withdraw unlimited amounts of money from FTX,”
which were created at Bankman-Fried’s direction. That effectively exempted Alameda from the kinds of risk management other customers faced. At the same time,
the government alleges, “Bankman-Fried publicly and repeatedly asserted that Alameda did not have privileged access to FTX.”
</item 7>

<item 8>
Today the FTX jury suffered through a code review (The Verge, 2023-10-06T22:21:46+00:00)
Other customers didn’t have the same privileges.

Alameda also allegedly kept FTX from looking bad by assuming some of its losses. In 2021, FTX was facing a “several hundred million dollar” loss from an exploit
in MobileCoin, Wang said. (This seems to have happened in April 2021, though Wang didn’t give a month.) Bankman-Fried directed him to have Alameda take it on
instead, because FTX’s balance sheets were visible to investors and Alameda’s weren’t.

Yesterday, Paradigm Capital’s Matt Huang showed us the balance sheets he received from Bankman-Fried. They didn’t include the loss, which should have been
included in the second quarter numbers he’d received — they showed $63 million in trading expenses and $63 million in “other expenses” in that quarter, and an
estimated profit of $732 million.

“FTX was not fine and assets were not fine.”

By 2022, Alameda’s negative balance on FTX was a cause for concern for Caroline Ellison, Singh, and Wang.
</item 8>

<item 9>
Sex, Signal messages, and sabotaging FTX's code: SBF execs and Bahamas roommates tell all in court (Cnbc | World Business News Leader,
2023-10-06T20:17:00+00:00)
That was all done by Bankman-Fried, he testified. In early 2020, Wang said he discovered for the first time Alameda's negative balance exceeded FTX's revenue,
an indication that Alameda was taking customer funds. Wang said he brought this to Bankman-Fried's attention several times. In late 2021, Wang discovered
Alameda had withdrawn $3 billion from its $65 billion line of credit. Wang's compensation was a base salary of $200,000 per year plus stock. He owned roughly
17% of FTX. Even though they were co-founders, "ultimately it was Sam's decision to make" when there were disagreements, he said.
</item 9>

<item 10>
Today the FTX jury suffered through a code review (The Verge, 2023-10-06T22:21:46+00:00)
Wang said he trusted Bankman-Fried’s judgment and didn’t pursue the issue further. Later, though, Alameda’s balance was more negative than FTT and the trading
revenue, and Wang had more conversations with Bankman-Fried about it. Wang said he knew at the time that Alameda was using FTX’s customers’ funds and that he
knew it was wrong.

We heard yesterday about the line of credit Alameda had on FTX, a staggering $65 billion. Today we saw it in the database. The line of credit didn’t start that
big — Wang testified that Bankman-Fried had asked him to increase it a few times because Alameda kept running into its limits. First it was “a few hundred
million,” then a billion, then even that wasn’t enough. It wasn’t clear how Wang arrived at $65 billion specifically, but he said he’d discussed the number with
Bankman-Fried. Other customers didn’t have the same privileges.
</item 10>
=== END ITEM id=a063-4a47056b00c41fe3 ===

=== ITEM id=a026-5dedff68e76c068c ===
Gold sentences and facts:
Sentence 0: Australian Man-of-the-Match Steve Smith (105 runs in 93 balls) and Aaron Finch (81 runs in 116 balls) were the guiding forces in taking Australia to 328 runs for 7 wickets in 50 overs.
  Fact 2.2: Aaron Finch scored 81 runs in 116 balls.

Retrieved items:
<item 1>
IND vs AUS Head-to-Head Record in ODI World Cup: Check India vs Australia match stats in World Cup Finals and ODI series (Zee Business,
2023-11-18T13:02:06+00:00)
Australia posted a target of 329 runs, losing seven wickets in the process. Steven Smith scored a brilliant 105 runs off 93 balls, with Aaron Finch supporting
the innings with an 81-run knock of 116 balls. Umesh Yadav picked up four wickets for India. India crumbled within 233 runs in 46.5 overs. MS Dhoni was the
highest scorer for the Indian side in that match, with 65 runs off 65 balls.

James Faulkner picked up three wickets for Australia, with Mitchell Johnson and Mitchell Stark taking two wickets each.

Australia went all the way to lift their fifth ODI World Cup trophy, defeating New Zealand in the final.

2019, Match 14, ICC Cricket World Cup, The Oval

India took their revenge in this World Cup as they annihilated the Australian bowlers by posting a target of 353 runs in 50 overs.
</item 1>

<item 2>
IND vs AUS Head-to-Head Record in ODI World Cup: Check India vs Australia match stats in World Cup Finals and ODI series (Zee Business,
2023-11-18T13:02:06+00:00)
India set a target of 248 runs after being bowled out in 55.5 overs. Yashpal Sharma scored 40 off 40 for India in that innings. Jeff Thompson and Rodney Hogg
picked up three wickets each for Australia.

Australia crumbled within 129 runs, with Roger Binny and Madan Lal picking up four wickets for India. India went on to win their first-ever ODI World Cup,
defeating the West Indies in the final.

1987, Match 3, Reliance World Cup, Chennai

This was the first time the World Cup was hosted outside of England. This was one of the closest matches that have been played between these two sides, as
Australia won the match by just one run. After batting first, Australia posted a target of 271 runs courtesy of a 110-run off-141-ball knock by Geoff Marsh.
</item 2>

<item 3>
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
</item 3>

<item 4>
ICC World Cup 2023: India at Cricket World Cup semi-finals so far (Zee Business, 2023-11-11T19:12:00+00:00)
India, one of the favourites to win the World Cup, were very much on the path to victory till Mohammad Azharuddin, who top-scored with 64 runs was at the
crease.

But with the fall of his wicket, the Indian lower order collapsed like a house of placards as India lost the match by 35 runs.

1996 Wills World Cup, against Sri Lanka

The 1996 Cricket World Cup is known for Sri Lanka's emergence as a cricketing power on the horizon of world cricket and the beginning of Sachin Tendulkar's
extraordinary dominance in batting history.

Playing in the first semi-final at the Eden Gardens in Kolkata on March 13, India were favourites again, and they raised their hopes when they managed to
restrict Sri Lanka to 251 for 8 in 50 overs despite Man-of-the-Match Aravinda de Silva's 66 and Roshan Mahanama's 58.
</item 4>

<item 5>
ICC World Cup 2023: India at Cricket World Cup semi-finals so far (Zee Business, 2023-11-11T19:12:00+00:00)
India were playing their sixth semi-finals at a place where they played their first World Cup semi-final in 1983, Manchester. However, with a different team-
New Zealand.

Indian bowlers raised hopes when they restricted New Zealand to 239 for 8 in 50 overs with Bhuvneshwar Kumar taking three wickets.

Ross Taylor 74 runs (90 balls) and skipper Kane Willamson 67 runs (95 balls) were the top scorers for New Zealand. But Kiwi Man-of-the-Match bowler Matt Henry
sent openers KL Rahul and Rohit Sharma back to the pavilion soon, followed by the dismissal of Dinesh Karthik in the 10th over of India's innings.

With the Indian top-order failing, Ravindra Jadeja (77 runs in 59 balls) and wicketkeeper MS Dhoni (50 runs in 72 balls) tried to steady the Indian ship.

But once they fell, it was all over for India as they lost the battle by 18 runs.
</item 5>

<item 6>
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
</item 6>

<item 7>
IND vs AUS Head-to-Head Record in ODI World Cup: Check India vs Australia match stats in World Cup Finals and ODI series (Zee Business,
2023-11-18T13:02:06+00:00)
Damien Fleming took five wickets for Australia as Australia won the match by 16 runs.

1999, 1st Super, ICC World Cup, The Oval

The Australian dominance continued in this World Cup as well, as they defeated India by 77 runs. Australia posted a target of 283 runs in 50 overs courtesy of a
brilliant 83 of 99 balls by Mark Waugh.

India had an early batting collapse but was stabilised by a sensible 100 off 138 balls by Ajay Jadeja. Robin Singh also scored 75 runs off 94 balls. However,
India was bowled out within 205 runs in 48.2 overs. Glenn McGrath picked up three wickets for Australia, with Damien Fleming and Steve Waugh picking up two
each.

2003, Match 11, ICC World Cup, Centurion

This was India's biggest batting collapse in the history of the ODI World Cup.
</item 7>

<item 8>
IND vs AUS Head-to-Head Record in ODI World Cup: Check India vs Australia match stats in World Cup Finals and ODI series (Zee Business,
2023-11-18T13:02:06+00:00)
This innings, by Ponting, had famously created the conspiracy that he had Spring in his bat.

India gave up a tough fight, with Virender Sehwag making 82 runs off 81 balls and Rahul Dravid also scoring 47 runs off 57 balls. However, the lack of support
from the other batters led to India being bowled out within 234 runs. Brett Lee and Andrew Symonds took two wickets each as Australia lifted the World Cup for
the third time.

Also Read: India vs Australia, World Cup final in 2003 vs 2023: Can India change the course of history in the final?

2011, 2nd Quarter-Final, ICC Cricket World Cup, Ahmedabad

After so much dominance by the Australian side in the World Cups against India, India finally got their revenge in this World Cup as they defeated them by five
wickets.
</item 8>

<item 9>
IND vs AUS Head-to-Head Record in ODI World Cup: Check India vs Australia match stats in World Cup Finals and ODI series (Zee Business,
2023-11-18T13:02:06+00:00)
India fought well, courtesy of two well-made 70s by Krishnamachari Srikkanth and Navjot Singh Sidhu. Unfortunately, India was bowled out in 49.5 overs with 269
runs on the board. Craig McDermott took four wickets for Australia.

1987, Match 15, Reliance World Cup, Delhi

This is the second time India has met Australia in this World Cup, with India winning this one by 56 runs. India batting first set a target of 290 runs courtesy
of four half-centuries by Sunil Gavaskar, Navjot Singh Sidhu, Dilip Vengsarkar, and Mohammad Azharuddin. Australia was bowled out within 233 runs, with David
Boon scoring 62 off 59 balls. Mohammad Azharuddin and Maninder Singh picked up three wickets for India.
</item 9>

<item 10>
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
</item 10>
=== END ITEM id=a026-5dedff68e76c068c ===

=== ITEM id=a104-5fc2db26ecc09089 ===
Gold sentences and facts:
Sentence 0: Briana Corrigan (born 30 May 1965) is a Northern Irish singer.
  Fact 1.0: Briana Corrigan was born on 30 May 1965.

Retrieved items:
<item 1>
Dave Hemingway
Dave Hemingway (born David Robert Hemingway, 20 September 1960) is an English musician and songwriter, and was a vocalist for the Hull-based band The Beautiful
South until they disbanded in 2007.

 Previously he had been a member of The Housemartins.
</item 1>

<item 2>
Come from the Heart
"Come from the Heart" is a country music song written by Richard Leigh and Susanna Clark and published in 1987.

 It is most known through the 1989 single by Kathy Mattea, released in conjunction with her album "Willow in the Wind", though the song was first recorded and
released on the 1987 Don Williams album "Traces" and also released in 1988 by Clark's husband on his album "Old Friends".
</item 2>

<item 3>
Takes a Little Time (Amy Grant song)
"Takes a Little Time" was a maxi-single released in 1997 (see 1997 in music) to promote Amy Grant's album "Behind the Eyes", which was also released that year.

 "Takes a Little Time" included two songs from "Behind the Eyes", as well as a new version of Grant's 1982 Christian radio hit, "El Shaddai".

 The maxi-single was also an enhanced CD (ECD), meaning it had a video that consumers could watch when they put it in their home computer.

 The ECD portion of the CD contained a live acoustic version of the song After the Fire, which Grant later released on her 2003 album "Simple Things"; Grant
stated on Oprah that she had written the song for her mother.
</item 3>

<item 4>
The Beautiful South
The Beautiful South was an English pop/rock group formed in 1988 by two former members of the Hull group the Housemartins—Paul Heaton and Dave Hemingway—both of
whom performed lead and backing vocals.

 Other members throughout the band's tenure were former Housemartins roadie Sean Welch (bass), Dave Stead (drums) and Dave Rotheray (guitar).

 After the band's first album (recorded as a quintet), they were joined by a succession of female vocalists, all of whom performed lead and backing vocals
alongside Heaton and Hemingway – Briana Corrigan for albums two and three after appearing as a guest vocalist on one, followed by Jacqui Abbott for the fourth
through seventh albums, and finally Alison Wheeler for the final three Beautiful South albums.
</item 4>

<item 5>
Burn Your Playhouse Down – The Unreleased Duets
Burn Your Playhouse Down – The Unreleased Duets is the 60th and final studio album by American country music singer George Jones released on August 19, 2008 on
the Bandit Records label.

 It features duets never before released, including some that were cut from his 1994 duets album "The Bradley Barn Sessions".

 The only new recording in the collection is "You And Me And Time", a song Jones recorded with his daughter by Tammy Wynette, Georgette.

 A music video accompanied the song.

 The album features several duets with artist from outside the country music pantheon, including Mark Knopfler, Leon Russell, and Keith Richards.

 The album peaked at number 15 on the Billboard country albums chart.

 Of the album's title track, Andrew Meuller of "Uncut" opined in July 2013, "The segue from Richards trying to sing like Jones to Jones actually singing like
Jones is hilarious."
</item 5>

<item 6>
Briana Corrigan
Briana Corrigan (born 30 May 1965) is a Northern Irish singer.

 She was the first female singer for The Beautiful South, from 1988 to 1992.
</item 6>

<item 7>
Just a Little Time
Just a Little Time is an album by Lynsey de Paul.

 It was originally released in 30 September 1994 on the Music Deluxe (catalogue number MSCD9) label but has since been released on the Tring International PLC
label (catalogue number JHD128) and on Arc Records/The Magic Collection (Catalogue number MEC 949080).

 All of the songs on the album are written or co-written by de Paul.

 The songs "Sugar Me", Getting a Drag", "Storm in a Teacup and "Dancing on a Saturday Night" are radical re-recordings of her earlier hits performed in an
updated and uptempo fashion.

 "Won't Somebody Dance with Me" is still performed as a ballad, but has been updated and extended with an extra refrain.

 Never before released songs on the album include the ballads "Words Don't Mean A Thing" and title track "Just a Little Time" as well as the more uptempo
"Instant Love" and "Now & Then".
</item 7>

<item 8>
Hugh Whitaker
Hugh Whitaker (born 18 May 1961) is an English musician and the former drummer for the British indie rock band The Housemartins.

 He replaced original drummer Chris Lang and drummed for the band's first album, "London 0 Hull 4", and its attendant single releases.

 He left the band before the recording of their second album, "The People Who Grinned Themselves to Death".

 Whitaker left the band on amicable terms and even participated in the promotional video for the band's first single without him, "Five Get Over Excited",
wherein he was kidnapped by his replacement, Dave Hemingway, and locked in a hessian sack.
</item 8>

<item 9>
Takes a Little Time (Total Contrast song)
"Takes a Little Time" is a 1985 dance hit by British, techno-soul, duo, Total Contrast.

 Their debut American release went to number one on the U.S. dance chart for one week.

 "Takes a Little Time" did not make the Hot 100 but, peaked at number eighty on the R&B singles chart.
</item 9>

<item 10>
A Little Time
"A Little Time" is a song by The Beautiful South, and is the band's only single to reach number 1 in the UK Singles Chart.

 It consists of a duet featuring vocalists Dave Hemingway and Briana Corrigan.
</item 10>
=== END ITEM id=a104-5fc2db26ecc09089 ===

=== ITEM id=a110-0ca93d193e3ff499 ===
Gold sentences and facts:
Sentence 0: Management consulting is the practice of helping organizations to improve their performance, operating primarily through the analysis of existing organizational problems and the development of plans for improvement.
  Fact 1.0: Management consulting is the practice of helping organizations improve their performance.

Retrieved items:
<item 1>
Abdal
Abdāl ("lit": substitutes, but which can also mean "generous" ["karīm"] and "noble" ["sharīf"]) is a term used in Islamic metaphysics and Islamic mysticism,
both Sunni and Shiite, to refer to a particularly important group of God's saints.

 In the tradition of Sunni Islam in particular, the concept attained an especially important position in the writings of the Sunni mystics and theologians,
whence it appears in the works of Sunni authorities as diverse as Abu Talib al-Makki (d. 956), Ali Hujwiri (d. 1072), Ibn Asakir (d. 1076), Khwaja Abdullah
Ansari (d. 1088), and Ibn Khaldun (d. 1406).
</item 1>

<item 2>
Eenasul Fateh
Eenasul Fateh (Bengali: ঈনাসুল ফাতেহ ; born 3 April 1959), also known by his stage name Aladin, is a Bangladeshi-British cultural practitioner, magician, live
artist and former international management consultant.
</item 2>

<item 3>
Arabian Nights and Days
Arabian Nights and Days (1979) is a novel by Egyptian writer Naguib Mahfouz, winner of the Nobel Prize for Literature.

 The novel serves as a sequel and companion piece for "One Thousand and One Nights" and includes many of the same characters that appeared in the original work
such as Shahryar, Scheherazade, and Aladdin.
</item 3>

<item 4>
Da'ud Abu al-Fadl
Da'ud Abu al-Fadl (1161–1242) was a Karaite Jewish physician who lived in the Ayyubid Egypt in the twelfth century CE.

 He born in Cairo in 1161 and died there about 1242.

 Having studied medicine under the Jewish physician Hibat Allah ibn Jami, and under Abu al-Fafa'il ibn Naqid, he became the court physician of the sultan
al-Malik al-'Adil Abu Bakr ibn Ayyub, the brother and successor of Saladin.

 He was also chief professor at the al-Nasiri Hospital at Cairo, where he had a great many pupils, among them being the historian ibn Abi Usaibiyyah.

 The latter declared that Abu al-Fadl was the most skillful physician of the time and that his success in curing the sick was miraculous.

 Abu al-Fadl was the author of an Arabic pharmacopoeia in twelve chapters, entitled "Aḳrabadhin", and treating chiefly of antidotes.
</item 4>

<item 5>
Alan Kamil
Alan C. "Al" Kamil is an American experimental psychologist.

 He is the Director, School of Biological Sciences and George Holmes Professor of Biological Sciences and Psychology at the University of Nebraska, Lincoln.

 Kamil's work focusses on the evolution of memory and adaptive specializations of learning in many animal species, especially the Clark's nutcracker and other
birds.

 Kamil has published peer reviewed articles on both theoretical aspects of comparative psychology and animal cognition, and on empirical studies of animal
learning and memory.

 In 2013 Kamil was honoured by the Comparative Cognition Society for his contributions to the study of animal cognition.
</item 5>

<item 6>
Lawrence A. Gordon
Lawrence A. Gordon is the EY Alumni Professor of Managerial Accounting and Information Assurance at the University of Maryland’s Robert H. Smith School of
Business.

 He is also an Affiliate Professor in the University of Maryland Institute for Advanced Computer Studies.

 Dr. Gordon earned his Ph.D. in Managerial Economics from Rensselaer Polytechnic Institute.

 An internationally known scholar in the areas of managerial accounting (often called management accounting) and cybersecurity economics, Dr. Gordon's research
focuses on such issues as economic aspects of information security (including cybersecurity or computer security), corporate performance measures, cost
management systems, and capital investments.

 He is the author of approximately 100 articles, published in such journals as The Accounting Review, ACM Transactions on Information and System Security,
Communications of the ACM, Journal of Financial and Quantitative Analysis, Journal of Computer Security, MIS Quarterly, Accounting, Organizations and Society,
and the Journal of Accounting and Public Policy.

 Dr. Gordon's current research emphasizes the importance of applying concepts from economics and managerial accounting to an information-based economy.

 Dr.
</item 6>

<item 7>
Alex Algard
Alex Algard is an Internet entrepreneur.

 He is the founder and former CEO of CarDomain, the founder and former CEO of Whitepages.com, and the founder and current CEO of Hiya.
</item 7>

<item 8>
Allen Klein
Allen Klein (December 18, 1931 July 4, 2009) was an American businessman, music publisher, writers' representative, filmmaker and record label executive, most
noted for his tough persona and aggressive, innovative negotiation tactics, many of which established higher industry standards for compensating recording
artists.

 He founded ABKCO Music & Records Incorporated.

 Klein revolutionized the income potential of recording artists, who previously had been routinely victimized by onerous record company contracts.

 He first scored massive monetary and contractual windfalls for Buddy Knox and Jimmy Bowen, one-hit rockabillies of the late 1950s, then parlayed his early
successes into a position managing Sam Cooke, and eventually managed the Beatles and the Rolling Stones simultaneously, along with many other artists, becoming
one of the most powerful individuals in the music industry during his era.
</item 8>

<item 9>
Aladdin's Eatery
Aladdin's Eatery is a chain of franchised restaurants in the American Midwest and South-East, specializing in Lebanese cuisine.

 Adapted to American tastes, the sites are fast casual restaurants that also offer take out.
</item 9>

<item 10>
Disney Renaissance
In the history of The Walt Disney Company, the Disney Renaissance refers to the era from 1989 to 1999 during which Walt Disney Feature Animation (renamed Walt
Disney Animation Studios in 2006) experienced a creative resurgence in producing successful animated films based on well-known stories, which restored public
and critical interest in The Walt Disney Company as a whole.

 During this era, the studio produced and released ten animated films: "The Little Mermaid" (1989), "The Rescuers Down Under" (1990), "Beauty and the Beast"
(1991), "Aladdin" (1992), "The Lion King" (1994), "Pocahontas" (1995), "The Hunchback of Notre Dame" (1996), "Hercules" (1997), "Mulan" (1998) and "Tarzan"
(1999).
</item 10>
=== END ITEM id=a110-0ca93d193e3ff499 ===

=== ITEM id=a117-11fbe07c18b5b2ca ===
Gold sentences and facts:
Sentence 0:  Raised in Fayetteville, North Carolina, Cole initially gained recognition as a rapper following the release of his debut mixtape, "The Come Up", in early 2007.
  Fact 1.4: Cole's debut mixtape The Come Up was released in early 2007.

Retrieved items:
<item 1>
Doug Grean
Doug Grean is an American record producer, audio engineer, guitarist, based in Los Angeles.

 He has worked with artists Sheryl Crow, Ricki Lee Jones, Glen Campbell, Scott Weiland, Slash, Velvet Revolver, Stone Temple Pilots, The Crystal Method, Tim
McGraw, Cyndi Lauper, Annabella Lwin (Bow Wow Wow), Corey Glover (Living Color), Camp Freddy, DJ Hurricane and John Taylor (Duran Duran).

 Grean was formerly the lead guitarist and touring musical director for Scott Weiland and the Wildabouts.

 Some of Grean's television appearances with Weiland include: VH1 Legends, Last Call with Carson Daly, The Tonight Show with Jay Leno, and the Grammy awards
with Velvet Revolver as keyboardist, performing "All Across the Universe" with Stevie Wonder, Norah Jones, Bono, Steven Tyler, Brian Wilson, and Alison Krauss.
</item 1>

<item 2>
Danny Shirley
Danny Shirley (born August 12, 1956) is an American country music singer.

 He is best known as the lead singer of the country rock band Confederate Railroad, a role he has held since its formation in 1987.
</item 2>

<item 3>
Diamond Rio
Diamond Rio is an American country and Christian country music band.

 The band was founded in 1982 as an attraction for the Opryland USA theme park in Nashville, Tennessee, and was originally known as the Grizzly River Boys, then
the Tennessee River Boys.

 It was founded by Matt Davenport, Danny Gregg, and Ty Herndon, the last of whom became a solo artist in the mid-1990s.

 After undergoing several membership changes in its initial years, the band has consisted of the same six members since 1989: Marty Roe (lead vocals, rhythm
guitar), Gene Johnson (mandolin, guitar, fiddle, tenor vocals), Jimmy Olander (lead guitar, Dobro, banjo), Brian Prout (drums), Dan Truman (keyboards, organ,
synthesizer), and Dana Williams (bass guitar, baritone vocals).
</item 3>

<item 4>
Lil Ru
Sylvester Samuels better known by his stage name Lil Ru, is an American rapper from Ridgeway, South Carolina currently signed to Def Jam Recordings.

 His debut album, 21 & Up was released on August 25, 2009.

 The Ridgeway native was 16 when he made his professional foray into the music business.

 Inspired by New Orleans’ innovative Cash Money Crew, Ru began making a name for himself on his local music scene, doing live shows and pressing up his own CDs.

 His hard-hitting lyrics and entrepreneurial spirit caught the attention of fellow South Carolinian Angie Stone.

 Shortly after the neo-soul songstress helped him secure a deal with Elektra Records, Ru found himself unsigned again, among the artists lost in the shuffle
after the label merged with Atlantic Records.

 Music fans first heard him on his 2001 debut single Will Destroy.

 He then released his 2002 follow up, Shawty What You Doin’.

 Both songs reached the Billboard R&B/Hip-Hop charts and helped land him at his next label home, Capitol Records.
</item 4>

<item 5>
Daron Jones
Daron Tavaris Jones (born December 27, 1976) is an American singer-songwriter, record producer.

 He is a member in Bad Boy Records group 112.

 His debut solo album titled "Uncensored" was released December 5, 2010.

 The first single from the album is the digitally-released "Money."

 Daron digitally released "Christmas in Atlanta" in December 2014, which reached the top 30 on the iTunes R&B chart.

 Daron is currently in the studio working on new music.
</item 5>

<item 6>
Danja (record producer)
Floyd Nathaniel "Nate" Hills (born February 22, 1982), professionally known as Danja , is an American record producer and songwriter from Virginia Beach,
Virginia.

 He has produced songs for prominent artists such as Justin Bieber, Monica, Britney Spears, Usher, Keri Hilson, T.I., Nelly Furtado, Kevin Cossom, Ciara, Mariah
Carey, Timbaland, Madonna, J. Cole, Kanye West, Whitney Houston, Missy Elliott, M.I.A., Justin Timberlake, JoJo, Joe Jonas, Simple Plan, The Clutch, Pink,
T-Pain, Diddy, Meek Mill, Björk and Duran Duran.
</item 6>

<item 7>
Anna Graceman
Anna Graceman is a young American singer-songwriter and pianist whose self-penned songs and music have gained attention at home and abroad.

 Her style of music incorporates elements of folk, soul and rock.

 As of 2016 she had written and performed over 70 of her own published songs; writing as well as producing her releases.

 Anna continues to divide her time between her solo projects and her new project - the ‘Graceman’ Band - a trio that includes Allie Graceman (Rhythm/Lead
Guitar) and Landon Graceman (Drums).
</item 7>

<item 8>
Raleigh Moncrief
Raleigh Moncrief is an American electronic musician, record producer and recording engineer who has worked with Zach Hill, Dirty Projectors and !!!

.
</item 8>

<item 9>
Daniel Boone Native Gardens
Daniel Boone Native Gardens, located in Boone, North Carolina, United States, has a collection of North Carolina native plants in an informal landscaped design.

 The gardens are open daily from May to October.
</item 9>

<item 10>
Kaydean
Kaydean is a British-American record and TV producer, composer, DJ, record label executive and artist manager.

 Kaydean produces, composes and arranges music in the Dance, Electronica, Latin Pop, Contemporary R&B, and Electronica genres.

 Born in London, he has worked with many artists including Janet Jackson, Maxi Priest, Robyn, Man Parrish, Angel "Cucco" Peña, Baron Lopez, Raekwon, Ky-Mani
Marley, Frankie Cutlass and Nocera amongst others.

 He is also known for discovering Latin Pop Star Janid and managing her career.

 He produced her Spanish album "La Magia" under the Sony Music Latin imprint label, Handshake and her Hit Single "Penicilina".

 In 2015 he debuted as a TV executive producer with Janid's reality show on Mega TV, The first episode aired on September 6, 2015.
</item 10>
=== END ITEM id=a117-11fbe07c18b5b2ca ===

=== ITEM id=a114-98c1b98969743360 ===
Gold sentences and facts:
Sentence 0:  Prior to the presidency, he was the Governor of Arkansas during two separate terms, from 1979 to 1981, and again from 1983 to 1992.
  Fact 2.2: William Jefferson Clinton's first term as Governor of Arkansas was from 1979 to 1981.

Retrieved items:
<item 1>
National Intelligence Estimate
National Intelligence Estimates (NIEs) are United States federal government documents that are the authoritative assessment of the Director of National
Intelligence (DNI) on intelligence related to a particular national security issue.

 NIEs are produced by the National Intelligence Council and express the coordinated judgments of the United States Intelligence Community, the group of 16 U.S.
intelligence agencies.

 NIEs are classified documents prepared for policymakers.
</item 1>

<item 2>
William Flynn Martin
William Flynn Martin (born October 4, 1950) is an American energy economist, educator and international diplomat.

 Martin served as Special Assistant to President Reagan for National Security Affairs, Executive Secretary of the National Security Council in the West Wing of
the White House and Deputy Secretary of the Department of Energy during the Ronald Reagan administration.

 He was President of the Council of the University for Peace, appointed to the Council by Secretary General of the United Nations Kofi Annan and served as the
Executive Director of the Republican Platform Committee during the re-election bid of George H.W. Bush.

 He has held senior appointments and advisory positions under several Presidents including: Ronald Reagan, George H.W. Bush and George W. Bush.
</item 2>

<item 3>
Steven K. Galson
Steven Kenneth Galson (born 1956) is an American public health physician.

 He is currently Senior Vice President for Global Regulatory Affairs at Amgen, the S. California-based innovative global biopharmaceutical company.

 He is also Professor-at-Large at the Keck Graduate Institute for Applied Life Sciences in Clarmont, California.

 He is a retired rear admiral in the United States Public Health Service Commissioned Corps and public health administrator who served as the acting Surgeon
General of the United States from October 1, 2007 – October 1, 2009.

 He served concurrently as acting Assistant Secretary for Health from January 22, 2009 to June 25, 2009, and as the Deputy Director and Director of the Center
for Drug Evaluation and Research (CDER) at the Food and Drug Administration from 2001 to 2007.

 As the Acting Surgeon General, he was the commander of the United States Public Health Service Commissioned Corps and, while serving as the Assistant Secretary
for Health, was the operational head of the Public Health Service.
</item 3>

<item 4>
National Intelligence Cross
The National Intelligence Cross is a decoration of the United States Intelligence Community (IC) awarded under the National Intelligence Awards (NIA) Program by
the Office of the Director of National Intelligence (ODNI).

 It is the highest award presented by the Office of the Director of National Intelligence.

 It is equivalent of the Central Intelligence Agency's Distinguished Intelligence Cross.
</item 4>

<item 5>
Robert Suettinger
Robert Suettinger was United States President Bill Clinton's national intelligence officer for East Asia at the National Intelligence Council (NIC) from
1997-1998.

 While there, Suettinger oversaw the preparation of national intelligence estimates for the director of the Central Intelligence Agency.

 His areas of specialty included the People's Republic of China and the North Korean nuclear weapons program.

 After working in the Clinton administration, Suettinger has been a senior analyst for the Brookings Institution where he wrote the book "Beyond Tiananmen – The
Politics of U.S.-China Relations, 1989-2000".

 He also is a senior advisor at The Stimson Center.
</item 5>

<item 6>
National Intelligence Board
The National Intelligence Board (formerly the National Foreign Intelligence Board) is a body of senior U.S. intelligence community leaders led by the Director
of National Intelligence (DNI).

 The Board is tasked with reviewing and approving National Intelligence Estimates (NIEs).
</item 6>

<item 7>
1997 Alabama Crimson Tide football team
The 1997 Alabama Crimson Tide football team represented the University of Alabama in the college football season of 1997–98.

 The team's head coach was Mike DuBose, who was entering his first year at Alabama.

 They played their home games at both Bryant–Denny Stadium in Tuscaloosa and Legion Field in Birmingham, Alabama and competed in the West Division of the
Southeastern Conference.

 Alabama would finish with a record of 4–7 record in suffering the program's first losing season since the 1984 season.

 The loss against Kentucky marked Alabama's first ever overtime game, as overtime rules for college football had gone into effect the previous season.
</item 7>

<item 8>
Lawrence Eagleburger
Lawrence Sidney Eagleburger (August 1, 1930 – June 4, 2011) was an American statesman and career diplomat, who served briefly as the Secretary of State under
President George H. W. Bush.

 Previously, he had served in lesser capacities under Presidents Richard Nixon, Jimmy Carter, and Ronald Reagan, and as Deputy Secretary of State under George
H. W. Bush.
</item 8>

<item 9>
Yannis Pappas
Yannis Pappas is a Greek-American comedian born and raised in Park Slope Brooklyn, New York.

 Both of his parents, Anna and Chris, practiced law.

 He is the youngest of three children.

 He has two older brothers, Peter and Nikos Pappas.

 His brother Peter is also a lawyer who worked for the Clinton and Obama administrations.

 Yannis attended college at the American University in Washington, D.C. where he studied American Studies and History.

 Pappas previously also worked in the social work field where he participated in 9/11 disaster relief for two and a half years and worked with the mentally ill
and people dealing with homelessness for three years with Lutheran Social Services.
</item 9>

<item 10>
Director of the Defense Intelligence Agency
The Director of the Defense Intelligence Agency is an intelligence officer who, upon nomination by the President and confirmation by the Senate, serves as the
nation's highest-ranking military intelligence officer.

 As the chief of the Defense Intelligence Agency, the Director is the primary intelligence adviser to the Secretary of Defense and the Chairman of the Joint
Chiefs of Staff and also answers to the Director of National Intelligence through the civilian Under Secretary of Defense for Intelligence.

 The Director is also the Commander of the Joint Functional Component Command for Intelligence, Surveillance and Reconnaissance, a subordinate command of United
States Strategic Command.

 Additionally, he chairs the Military Intelligence Board, which coordinates activities of the entire defense intelligence community.
</item 10>
=== END ITEM id=a114-98c1b98969743360 ===

=== ITEM id=a130-766d6c78a89f3074 ===
Gold sentences and facts:
Sentence 0:  The album was Prince's third to be released that year (along with "Chaos and Disorder" and the soundtrack album of the Spike Lee movie "Girl 6"), which made 1996 one of the most prolific years for material released by Prince.
  Fact 0.3: 1996 was one of the most prolific years for material released by Prince.

Retrieved items:
<item 1>
Paul Schrader
Paul Joseph Schrader (born July 22, 1946) is an American screenwriter, film director, and film critic.

 Schrader wrote or co-wrote screenplays for four Martin Scorsese films: "Taxi Driver" (1976), "Raging Bull" (1980), "The Last Temptation of Christ" (1988), and
"Bringing Out the Dead" (1999).

 Schrader has also directed 18 feature films, including his directing debut crime drama, "Blue Collar" (co-written with his brother, Leonard Schrader), the
crime drama "Hardcore" (a loosely autobiographical film also written by Schrader), his 1982 remake of the horror classic "Cat People", the crime drama "American
Gigolo" (1980), the biographical drama "" (1985), the cult film "Light Sleeper" (1992), the drama "Affliction" (1997), the biographical film "Auto Focus"
(2002), and the erotic dramatic thriller "The Canyons" (2013).
</item 1>

<item 2>
Still Kicking (film)
Still Kicking: Six Artistic Women of Project Arts & Longevity is a 2006 32-minute documentary film by Pacific Grove filmmaker Greg Young, featuring six Bay Area
women role models over 85 years old who remained artistically active.

 The catalyst for Young's film was Amy Gorman and Frances Kandl's Project Arts & Longevity through which they were exploring the link between longevity and
artistic vitality.

 Along with the film the joint project resulted in a book entitled Aging Artfully.
</item 2>

<item 3>
Mark Lee Ping-bing
Mark Lee Ping-bing (李屏賓; born August 8, 1954) is a Taiwanese cinematographer, photographer and author with over 70 films and 21 international awards to his
credit including 2 Glory Of The Country Awards from the Government Information Office of Taiwan and the president of Taiwan's Light Of The Cinema Award.

 Lee began his film career in 1977 and in 1985 he started his prolific collaboration with Taiwanese filmmaker Hou Hsiao-hsien.

 Known best for his use of natural lighting utilizing real film and graceful camera movement, Lee received the Grand Technical Prize at the Cannes Film Festival
in 2000 for "In the Mood for Love".

 A member of the Academy of Motion Picture Arts and Sciences, Lee was honored with nominations by the American Society of Cinematographers for its 2014 First
Annual Spotlight Award for Best Cinematography for his work on the 2012 film "Renoir" and by the French Academy of Cinema Arts for a Cesar Award for Best
Cinematography in 2014 also for the film "Renoir".
</item 3>

<item 4>
Spike Lee
Shelton Jackson "Spike" Lee (born March 20, 1957) is an American film director, producer, writer, and actor.

 His production company, 40 Acres and a Mule Filmworks, has produced over 35 films since 1983.
</item 4>

<item 5>
Chris Gorak
Chris Gorak is an American film director who started as an art director and production designer.

 He directed the 2006 film "Right at Your Door" and the 2011 film "The Darkest Hour".

 Gorak began working in the film industry in the 1990s as an art director.

 In the 2000s, he worked on several films as production designer before becoming a film director.
</item 5>

<item 6>
Cannibal boom
Only four other cannibal films were made after 1981 until the fad's conclusion in 1988 with Antonio Climati's "Natura Contro".
</item 6>

<item 7>
Mel Gibson filmography
Mel Gibson, AO, is an American actor, director, producer and screenwriter who made his acting debut on the Australian television drama series "The Sullivans"
(1976– 1983).

 While a student at the National Institute of Dramatic Art in Sydney, he was given an uncredited role in "I Never Promised You a Rose Garden" and subsequently
appeared as a leading actor in the micro budget surf drama "Summer City" (both in 1977).

 Gibson rose to prominence during the Australian New Wave cinema movement in the early 1980s, having appeared in his breakthrough role in George Miller's
dystopian action film "Mad Max" (1979), portraying the eponymous hero.

 He reprised the role in its sequels, "Mad Max 2" (1981) and "Mad Max Beyond Thunderdome" (1985).

 He appeared in Peter Weir's war drama "Gallipoli" (1981) and the romantic drama "The Year of Living Dangerously" (1982).
</item 7>

<item 8>
The Human Centipede 2 (Full Sequence)
The Human Centipede 2 (Full Sequence) is a 2011 Dutch black and white exploitation-horror film written, directed, and co-produced by Dutch filmmaker Tom Six.

 The sequel to Six's 2009 film "The Human Centipede (First Sequence)", the film stars Laurence R. Harvey as a mentally impaired English man who watches and
becomes obsessed with the first "Human Centipede" film, and decides to make his own "centipede" consisting of twelve people, including Ashlynn Yennie, an
actress from the first film.
</item 8>

<item 9>
Girl 6
Girl 6 is a 1996 American comedy-drama film by director Spike Lee about a phone sex operator.

 Theresa Randle played the title character, and playwright Suzan-Lori Parks wrote the screenplay.

 The soundtrack is composed entirely of songs written by Prince.

 The film was screened in the Un Certain Regard section at the 1996 Cannes Film Festival.

 Directors Quentin Tarantino and Ron Silver make cameo appearances as film directors at a pair of interesting auditions.

 It is the first film directed by Lee in which he did not write the screenplay.
</item 9>

<item 10>
Dan Ouellette
Giger and Beksinski, and he has been featured in magazines internationally.

 David Bowie commented while looking at Dan's art that he has noticed a strong influence of science fiction on contemporary art.

 Giger saw a different aspect, saying simply "Very bony."

 He grew to adore the cinema of Fellini and later to marvel at Lynch's "Eraserhead".

 Rather than pursue a formal education in the fine arts he chose to study the craft of film making and after college he became a production designer working on
feature films in New York City starting with his work as a production designer for Hal Hartley in 1990 with Trust and then, in 1992, with Simple Men.
</item 10>
=== END ITEM id=a130-766d6c78a89f3074 ===
