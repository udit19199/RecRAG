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

=== ITEM id=3af467c5a7567a37 ===
Gold sentences and facts:
Sentence 0: For the Swans, however, the victory has them challenging for a top-eight spot alongside the likes of Gold Coast, Geelong, Collingwood, St Kilda and Carlton.
  Fact 0.0: The victory has the Swans challenging for a top-eight spot.
  Fact 0.1: The Swans are challenging for a top-eight spot alongside teams such as Gold Coast.
  Fact 0.2: The Swans are challenging for a top-eight spot alongside teams such as Geelong.
  Fact 0.3: The Swans are challenging for a top-eight spot alongside teams such as Collingwood.
  Fact 0.4: The Swans are challenging for a top-eight spot alongside teams such as St Kilda.
  Fact 0.5: The Swans are challenging for a top-eight spot alongside teams such as Carlton.
Sentence 1: With only 0.4 of a percentage point separating Sydney in ninth and St Kilda in eighth, the win assured the Swans of retaking eighth spot.
  Fact 1.0: Only 0.4 of a percentage point separates Sydney in ninth place and St Kilda in eighth place.
  Fact 1.1: Sydney is in ninth place.
  Fact 1.2: St Kilda is in eighth place.
  Fact 1.3: The win assured the Swans of retaking eighth spot.
Sentence 2: “Our team has put a lot of work into sourcing some wonderful food offerings, entertainment and activities for kids, while we’ve had a local focus in promoting our games,” she said.
  Fact 2.0: A speaker, referred to as 'she', said the following quote.
  Fact 2.1: The speaker's team has put a lot of work into sourcing some wonderful food offerings.
  Fact 2.2: The speaker's team has put a lot of work into sourcing entertainment for kids.
  Fact 2.3: The speaker's team has put a lot of work into sourcing activities for kids.
  Fact 2.4: The speaker's team has had a local focus in promoting its games.

Retrieved items:
<item 1>
Ground-breaking Gold Coast: Suns make AFLW finals for a second time (The Age, 2023-11-03T23:45:04+00:00)
“But I go out with on a happy note. I love these girls and I’ll be over at the pancake pocket [Pancake Parlour at Whitten Oval] watching them from now on and
have a few bevs on the sideline.” Roos’ lock in top-four spot; Dogs have issues to work through in off-season North Melbourne secured third spot on the ladder
and the double chance in finals after downing the Western Bulldogs by a whopping 46 points. They will head into finals with momentum but a glaring issue to work
on: kicking accuracy. Despite the comfortable win, they could have done a lot more damage after booting double the number of behinds to goals.

Star midfielder Ash Riddell, who had 36 touches and seven clearances, said she was excited to go up against Melbourne or Adelaide in the first weekend of finals
depending on other results this round, as they were the only two sides who beat them this year.
</item 1>

<item 2>
NBL Round 5: Harvey hits Hail Mary for Hawks to topple Taipans at the buzzer, Cheatham injured as Sixers upset Breakers (The Roar | Sports Writers Blog,
2023-10-26T21:11:37+00:00)
“I don’t want to discredit Illawarra (but) during the game, I felt like we were better-equipped to close it out, but we didn’t and Illawarra did. It’s the one
that got away.”

Nine first-quarter points from Froling helped the Hawks to a nine-point lead at the first change.

Froling (16 points, 13 rebounds) had his double-double with seven minutes remaining in the third quarter.

But Cairns did a better job at targeting him later on, keeping the rangy centre scoreless in the second half.

“He had a great game,” Jackomas said.

Advertisement

Import Justin Robinson, who had struggled from the field to begin the season, hit a three that forced a timeout and restored a game-high 13-point lead in the
third quarter.

But lapses in concentration threatened to cost the Hawks.
</item 2>

<item 3>
NBL Round 4: Creek pours in points as Phoenix swamp Sixers, CJ claims Kings didn’t look after DJ (The Roar | Sports Writers Blog, 2023-10-19T21:03:08+00:00)
Williams produced 21 points and six rebounds for South East Melbourne but Brisbane will still be without Baynes for next weekend’s double-header away to Perth
and at home to Tasmania, and he needs to find some answers.

“We’ve got to have solutions right now for two more games without Banger (Baynes) and that’s the thing that’s sitting right in the front of my head, but scoring
73 points isn’t enough to get it done in this league,” Schueller said.

“Then giving them 96 is way too many as well so we have a lot of stuff that we still have to work on.”

JLA return lifts United to win in Wollongong

Advertisement

Melbourne United have celebrated the long-awaited return of Jo Lual-Acuil but it was fellow centre Ariel Hukporti who starred in a 101-91 defeat of the
lllawarra Hawks.
</item 3>

<item 4>
Who’s the next Gawn, Grundy or Dusty? AFL’s talent guru names top 40 draft prospects and compares them to current stars (The Roar | Sports Writers Blog,
2023-11-15T23:48:27+00:00)
Weapon: Versatility and contested ball-winning ability

AFL player comparison: Callum Mills (Sydney Swans)

"All through my junior football, my dad placed a big emphasis on playing every position. I think that came to the fore this year a little bit."



Tassie’s 'Mr Fix-It' James Leake is shaping as a first-round pick following an impressive season.https://t.co/fwnC96Uzdw — AFL (@AFL) November 3, 2023

Kane McAuliffe

State: South Australia

State League Club: North Adelaide

Community Club: Central Augusta

Date of Birth: 01/03/05

Height: 187cm

A powerful midfielder who wins his own ball and is a penetrating left-foot kick. Is a strong and aggressive tackler as well as being an accomplished extractor
of the ball from stoppages using his clean hands and vision to set up the play.
</item 4>

<item 5>
Ground-breaking Gold Coast: Suns make AFLW finals for a second time (The Age, 2023-11-03T23:45:04+00:00)
“I’m excited like I think the group’s coming together really nicely and where really well-prepared compared to previous years,” Riddell told Fox Footy. “So, I
can’t wait I think it’s an exciting time for us.” The game was one-way traffic against the Dogs with the game played almost entirely in the Roos’ attacking
half. Riddell credited her side’s “team performance” after two losses to the Dees and Crows. North Melbourne have secured a top-four spot for finals. Credit:
Getty For the Western Bulldogs, it was another tough day out as they notched their ninth loss of the season. They won just one game this year, an eight-point
victory last weekend over 17th-placed West Coast.

Britney Gutknecht kicked the Dogs’ only goal against the Kangaroos, which came early in the final term to avoid a record for the lowest AFLW score in the
history of the competition.
</item 5>

<item 6>
Comeback kings Richmond take the AFLW points in Cairns (The Age, 2023-10-20T23:51:34+00:00)
Michael Prior guided West Coast to a gutsy win. Credit: Getty

Aisling McCarthy and Amy Franklin drew first blood for West Coast within the opening five minutes, while the hosts were kept goalless. The Bombers only added
three behinds in the second term and the Eagles failed to score as the blustery conditions took their toll on the game. West Coast’s Kellie Gibson broke the
goal drought with the only major of the third quarter, before the home side came to life with the wind at their backs in the final term, with skipper Bonnie
Toogood and Amber Clarke reducing the margin to four points. Toogood continued to pepper the scoreboard, but the Eagles held on to secure a rare victory. “We’ve
had a horror week back at home, and it’s been really bloody tough on all of us, and Mick as well. I’m just really proud of the girls,” West Coast captain Emma
Swanson said post-game on the broadcast. “I can name the whole team.
</item 6>

<item 7>
Ground-breaking Gold Coast: Suns make AFLW finals for a second time (The Age, 2023-11-03T23:45:04+00:00)
But two goals from Amelia Radford and Georgia Gee kept the hosts close, while defender Ellyse Gamble (nine intercept possessions) proved a roadblock for the
Suns’ attackers. Single lit up the park with a clinical slice through the corridor, dodging two tackles to put the ball in the hands of Georgia Clayden, who
capped the centre clearance with the only major of the third term. With both sides goalless in the final quarter, the Suns were able to cruise home. With AAP
</item 7>

<item 8>
The story behind anguished Wallabies star's photo (Wide World Of Sports, 2023-10-07T22:10:40+00:00)
"Last week 45,000 fans (versus Portugal), last night (60,000 in Lyon). Rugby's such a small aspect in terms of Australians but our fans have turned up here,
we've seen them and spoken to them and their unconditional love for us, the fact that they came 20,000km to get here and support us, has been awesome.

"My message is just to understand that we're working extremely hard, there's no stone unturned, especially the youth in this team, to leave the jersey in a
better place and just to stick with us.

"I know the results haven't been coming our way the last couple of years but when it does turn, that's what we're building towards.﻿"
</item 8>

<item 9>
NBL Round 5: Harvey hits Hail Mary for Hawks to topple Taipans at the buzzer, Cheatham injured as Sixers upset Breakers (The Roar | Sports Writers Blog,
2023-10-26T21:11:37+00:00)
Melbourne led by as many as 16 points midway through the fourth quarter on the back of a three ball from Luke Travers (14 points, 13 rebounds), having outscored
the JackJumpers by 78 points to 49 after the first five minutes.

The JackJumpers pulled off a remarkable comeback on Melbourne back in round two and got as close as four points in the final minute, but United held firm for a
fifth straight win to stay on top at 7-1.

Dean Vickerman was celebrating his 300th game coaching in the NBL, and the Melbourne boss was full of praise for his team.

“Tonight was a little bit like my career,” he said.

“We didn’t start great but once we found our rhythm and especially the way we came out of the half and had a great third quarter, and held on down a few players
at the end there. But we held on.”
</item 9>

<item 10>
Comeback kings Richmond take the AFLW points in Cairns (The Age, 2023-10-20T23:51:34+00:00)
Comeback kings Richmond take the points in Cairns A late goal by Emelia Yassir has helped Richmond to their first win in four games, putting their AFLW finals
campaign back on track with an 11-point victory over Hawthorn. Tight tussle: Richmond’s Emelia Yassir is tackled. Credit: Getty Images Separated by one point at
half-time and down by nine points entering the last quarter, the Tigers pulled off a 6.5 (41) to 4.6 (30) victory to keep their finals hopes alive at Cazaly’s
Stadium in Cairns on Friday night.

Hawthorn were limited to just one point in the decisive final quarter. Midfield duo Monique Conti (35 disposals, seven clearances) and Grace Egan (28, 5) were
typically influential for Richmond, while Hawthorn were led by Emily Bates (18, four) and Jasmine Fleming (18, one). “Richmond were just absolute beasts,
especially their midfielders.
</item 10>

<item 11>
Comeback kings Richmond take the AFLW points in Cairns (The Age, 2023-10-20T23:51:34+00:00)
J’Noemi Anderson is tackled by a group of Dockers. Credit: AFL Photos The Dockers dominated the Saints from the start to win 6.9 (45) to 3.3 (21) at Fremantle
Oval. Having been on the verge of making their first finals appearance, St Kilda (4-4) could now miss the top eight even if they beat the fourth-placed Brisbane
Lions and the ninth-placed Carlton.

Dana East was everywhere for Fremantle, leading the game in disposals (19) and contested possessions (11), alongside Orlagh Lally (19 disposals, five
clearances). Jamiee Lambert was the shining light for the otherwise subdued Saints, collecting a game-high seven clearances and 13 contested possessions. Docker
Hayley Miller hunts a loose ball against St Kilda. Credit: Getty Fremantle flew out of the gates from the first siren, producing their highest-scoring quarter
of the season with 26 points in the opening term.
</item 11>

<item 12>
NBL Round 5: Harvey hits Hail Mary for Hawks to topple Taipans at the buzzer, Cheatham injured as Sixers upset Breakers (The Roar | Sports Writers Blog,
2023-10-26T21:11:37+00:00)
The guys are going in the right direction. “I’m not disappointed, I’m stoked.”

Bullets shoot down weakened Wildcats

Advertisement

The Brisbane Bullets have shrugged off the absence of two important players to outwork the Wildcats at their home RAC Arena in Perth and produce a performance
full of heart for their 84-79 NBL win.

Both teams were 2-4 coming into Friday’s match and on losing slides, but it was the Wildcats who had the pressure building, even with the Bullets being without
key pair, the suspended Aron Baynes and injured Shannon Scott.

Brisbane scored the first seven points before opening up a 20-point lead in the third quarter and going on to win by five after holding out against a furious
late charge from the home team.

The team effort of the Bullets was outstanding as they got another 17 points from captain Nathan Sobey.
</item 12>

<item 13>
Ground-breaking Gold Coast: Suns make AFLW finals for a second time (The Age, 2023-11-03T23:45:04+00:00)
Chloe Scheer was once again the star in front of goals with three majors, while the Cats finished with seven individual goalkickers. Georgie Prespakis (35
touches, eight clearances) and Amy McDonald (23 touches, eight clearances) were a formidable midfield duo, while the retiring Renee Garing had a whopping 11
tackles. Overall, it was a strong team effort from a side that will be hungry for a finals win. Last year, the Cats had an impressive home-and-away run but
bowed out narrowly in an elimination final to North Melbourne. Geelong young defender Rebecca Webster told this masthead earlier in the week how important the
home crowd is for them come knockout finals. “Playing at GMHBA is like a massive advantage for us,” said Webster. “We love going in those four walls. We train
there three, four times a week.
</item 13>

<item 14>
NBL Round 5: Harvey hits Hail Mary for Hawks to topple Taipans at the buzzer, Cheatham injured as Sixers upset Breakers (The Roar | Sports Writers Blog,
2023-10-26T21:11:37+00:00)
But Goulding caught fire to lead United to a 52-51 at the main break.

Defence was key in the second half as both sides tightened up, with Melbourne holding the visitors to just 15 points in the final period.

Travers returned after half-time, finishing with six points and a game-high 11 rebounds.

Advertisement

Ariel Hukporti retained his spot in Melbourne’s starting five ahead of Jo Lual-Acuil Jr and impressed with seven points and 10 rebounds, including four on the
offensive end.

Bullets ride late luck to win over Tasmania

Nathan Sobey has peeled off a game-high 28 points as Brisbane survived a late Tasmania fightback to continue their NBL resurgence with a tense 90-87 win.
</item 14>

<item 15>
Who’s the next Gawn, Grundy or Dusty? AFL’s talent guru names top 40 draft prospects and compares them to current stars (The Roar | Sports Writers Blog,
2023-11-15T23:48:27+00:00)
Weapon: Aerial skill and goal sense

AFL player comparison: Isaac Heeney (Sydney Swans)

Charlie Edwards

State: Victoria Country

State League Club: Sandringham Dragons

Community Club: Sandhurst

Date of Birth: 18/05/05

Height: 191cm

Medium defender and midfielder who emerged as the year progressed and is now considered a possible Draft bolter given his rapid progression for the Sandringham
Dragons in their road to the Coates Talent League premiership. Showed real class with his athleticism and ball use once moved into the midfield with his
29-disposal performance against the Dandenong Stingrays in Round 13 an eye-catcher. Is also a talented cricketer and has spent only one year in the AFL Talent
Pathway system after playing underaged football at Sandhurst before moving to Melbourne Grammar.
</item 15>

<item 16>
NBL Round 4: Creek pours in points as Phoenix swamp Sixers, CJ claims Kings didn’t look after DJ (The Roar | Sports Writers Blog, 2023-10-19T21:03:08+00:00)
A third straight defeat for the Wildcats has them struggling at 2-4 this season.

Coach John Rillie is focused know they aren’t playing well and the improvement needs to start offensively.

“We’re not playing good basketball at this point in time,” Rillie said.

“An area we need to improve on is our offence. That’s where we have to find some type of rhythm and we’ve yet to do that in these last couple of games.

Advertisement

“The game’s played at both ends and you need to find a rhythm to generate some natural enthusiasm, and we haven’t been able to do that for good stretches.”

Brisbane coach Justin Schueller always knew losing his superstar centre Baynes with a five-game suspension was going to be challenging, but that was magnified
on Saturday night.
</item 16>

<item 17>
‘Right people in right roles’: QRU breaks silence on RA’s reform hopes, backs review into Wallabies’ RWC flop (The Roar | Sports Writers Blog,
2023-10-10T23:06:29+00:00)
We need to maximize the development of the players. At the end of the day, that’s the key to success and alignment of your resources, for a small country.

“Because Australia, if you look at the history of rugby, when we were amateur, we were probably ahead of the game because we had the influence of rugby league.
We had the influence of AFL and our players were more professional than the rest of the world.”

He added: “I know what’s going on and we need to get a better system of developing players. We need to develop better players and we need to develop a larger
batch of better players and that only comes through changing the whole system.”

While all five Australian Super Rugby sides fundamentally are on board with regards to moving to a centralised high-performance program, The Roar added that
some states, including the ACT Brumbies as well as the QRU, don’t want to give up their commercial assets.
</item 17>

<item 18>
Who’s the next Gawn, Grundy or Dusty? AFL’s talent guru names top 40 draft prospects and compares them to current stars (The Roar | Sports Writers Blog,
2023-11-15T23:48:27+00:00)
Announced himself as an AFL prospect when he kicked six goals from 26 disposals in the GWV Rebels’ loss to the Dandenong Stingrays in Round 4. He holds his
width when playing on the wing which provides his team with an outlet option with its ball movement. Had a consistent year at club level averaging 19.6
disposals in 14 matches for the Rebels and finished third in the best and fairest while playing in all four matches for Victoria Country. A strong all-round
athlete, he ran a time of 2.98 seconds in the 20m sprint during pre-season testing and then completed the 2km time trial in 6:13 at the national Draft Combine.
</item 18>

<item 19>
Who’s the next Gawn, Grundy or Dusty? AFL’s talent guru names top 40 draft prospects and compares them to current stars (The Roar | Sports Writers Blog,
2023-11-15T23:48:27+00:00)
Kicked 39 goals across all matches in the talent pathway this year, which included games for Victoria Metro and the Australia under-18 team. Tested extremely
well at the national Draft Combine with results of 3.02 seconds on the 20m sprint along with 6:35 for the 2km time trial indications of his athletic prowess. Is
the nephew of two-time Richmond premiership star Josh Caddy.

Weapon: Contested marking and goalkicking

AFL player comparison: Taylor Walker (Adelaide Crows)

Caiden Cleary

State: NSW-ACT

State League Club: Sydney Swans Academy

Community Club: Sydney University

Date of Birth: 05/03/05

Height: 182cm

An in-and-under player who extracts the ball well from stoppages.
</item 19>

<item 20>
Ground-breaking Gold Coast: Suns make AFLW finals for a second time (The Age, 2023-11-03T23:45:04+00:00)
Melbourne’s Lauren Peace fights for the ball againt the Lions. Credit: Getty Images The Lions were at risk of slipping from the top four after last week’s
21-point loss to St Kilda, but will now finish the home-and-away season fourth, avoiding a sudden-death elimination final and marking coach Craig Starcevich’s
75th game in style.

Meanwhile, Melbourne ceded the minor premiership to Adelaide and dropped to second spot. The Lions put on a show across the park, with powerhouse duo Ally
Anderson (22 disposals) and Jade Ellenger (22) going to work in the midfield while Dakota Davidson (two goals) led the charge in the forward line. Captain Bre
Koenen was superb in defence as she played on Melbourne’s leader Kate Hore, collecting a game-high 12 intercept possessions and limiting the prolific goalkicker
to just one goal. Lions duo Charlotte Mullins and Ruby Svarc celebrate.
</item 20>

<item 21>
Who’s the next Gawn, Grundy or Dusty? AFL’s talent guru names top 40 draft prospects and compares them to current stars (The Roar | Sports Writers Blog,
2023-11-15T23:48:27+00:00)
Weapon: Intercept marking

AFL player comparison: Will Day (Hawthorn)

Advertisement

James Leake

State: Tasmania

State League Club: Tasmania Devils

Community Club: Launceston

Date of Birth: 02/10/05

Height: 187cm

Medium defender and forward who won All Australian honours as a defender for the Allies at the National Championships in the middle of the season then played
some eye-catching matches as a forward for the Tasmania Devils in the Coates Talent League during a breakout season. With mercurial ability in the air and
uncanny ability around goals he looks to have unique versatility which was on display during 11 matches for the Devils where averaged 16.1 disposals and kicked
17 goals, which included a five-goal haul against the Dandenong Stingrays in Round 17.
</item 21>

<item 22>
‘So used to seeing us winning’: Healy denies Aussie era ending after India loss, calls for more Tests (The Roar | Sports Writers Blog,
2023-12-24T22:28:25+00:00)
The current tour of India is not a multi-format series, but the defeat in the one-off Test marks Australia’s first loss in a four-day game since January 2014.

But quizzed over whether the defeat suggested Australia’s unprecedented period of dominance was ending, Healy was defiant.

“It’s probably not even a blip,” Healy said.

Advertisement

“It (Test cricket) is a format where we’re not overly familiar with, in conditions that we’re not familiar with.

“I see more positives out of this and than any sort of negativity.

“There is always a lot of expectation and external noise about our team and how we’re performing.

“They are so used to seeing us winning, that I think, at times, they forget that we’ve actually seen quite a lot of change over the last 12 to 18 months.

“That kind of goes unnoticed external to our group.”

Australia have spoken about shifting their mindset in white-ball cricket after the losses in England, pursuing a more attacking approach.
</item 22>

<item 23>
‘So used to seeing us winning’: Healy denies Aussie era ending after India loss, calls for more Tests (The Roar | Sports Writers Blog,
2023-12-24T22:28:25+00:00)
Alyssa Healy has rejected any suggestion Australia’s stranglehold on international women’s cricket is slipping, saying their Test loss to India is barely a blip
on the team’s record.

Australia stumbled to their first Test defeat to India by eight wickets in Mumbai on Sunday, after being outplayed by the hosts in foreign spinning conditions.

The loss comes after one of the more difficult periods for the Australian side in the past six years, with the retirements of Meg Lanning and Rachael Haynes.

Facebook Twitter Whatsapp Reddit Email Share

After winning their fourth straight global ICC event earlier this year in the 50-over World Cup, Australia retained the Ashes via a drawn multi-format series in
England in July.

That tour included losing four of six white-ball matches, with Australia’s win in the Test enough to have them level on points with England and hold onto the
trophy.
</item 23>

<item 24>
Craft beer and cheese boxes: Swans tap into something ‘uniquely Sydney’ to draw AFLW crowds | Jack Snape (The Guardian, 2023-10-26T23:42:29+00:00)
Fans of Australian rules football in the Harbour City have a reputation for being fickle. After a winless campaign in 2022, the Swans – led by ex-Collingwood
player Chloe Molloy – have been competitive this season. The team won their first AFLW match in round one and are within a game of the top eight with two rounds
to play.

But general manager of women’s football at the AFL, Nicole Livingstone, said the Swans’ off-field offering had tapped into something “uniquely Sydney” and
helped make matches more attractive.

“To go to North Sydney Oval and to have quince paste and hand-picked grapes, cheddar cheese and biscuits in a little box that’s got this beautiful wrapping on
it – they’ve really thought about what they’re doing and made it an experience,” she said.
</item 24>

<item 25>
Ground-breaking Gold Coast: Suns make AFLW finals for a second time (The Age, 2023-11-03T23:45:04+00:00)
Lions duo Charlotte Mullins and Ruby Svarc celebrate. Credit: Getty Images The Lions got the game on their terms early with five repeat entries into the
attacking 50m in the opening 10 minutes of the game. Davidson and Ellie Hampson drew first blood before Casey Sherriff kicked Melbourne’s opening goal.

Tayla Harris, Tyla Hanks and Hore slotted a goal each in the second term to put their side ahead. Crumbling under the Brisbane attack, the visitors conceded
four unanswered goals from Taylor Smith, Sophie Conway (two goals) and Ruby Svarc while remaining goalless in the third term. Midfield stars Olivia Purcell (14
contested possessions) and Tyla Hanks (11) showed their mettle as they attempted to drag their side over the line, but the Demons were outclassed and sorely
missing Paxy Paxman (hamstring). Melbourne could be without Maeve Chaplin in their finals campaign, after the defender sustained a concussion injury in the
third term.
</item 25>

<item 26>
Ground-breaking Gold Coast: Suns make AFLW finals for a second time (The Age, 2023-11-03T23:45:04+00:00)
AAP Cats secure elimination final home-ground advantage

Geelong have not only secured their spot in the top eight with a huge 40-point win over Hawthorn on Saturday at GMHBA Stadium but locked in home-ground
advantage for the first elimination final. The Cats finished sixth, with the 9.6 (60) to 3.2 (20) win over the Hawks boosting their percentage to a healthy
141.2, almost 40 ahead of Essendon, who sit one rung below them on equal points. Momentum is also in Geelong’s favour after two big wins in a row, having
demolished Richmond by 38 points last weekend. Chloe Scheer (left) and Julia Crockett-Grills of the Cats celebrate. Credit: Getty Against the Hawks, they
controlled ball movement from the get-go, kicking four goals straight to open the match and keeping their opposition goalless to half-time.
</item 26>

<item 27>
NBL Round 4: Creek pours in points as Phoenix swamp Sixers, CJ claims Kings didn’t look after DJ (The Roar | Sports Writers Blog, 2023-10-19T21:03:08+00:00)
“When you’re missing a ton of shots you’ve got to win the offensive rebounding battle and they beat us on that.”

Tasmania burst out to a double-figure lead when they piled on 36 points during the opening period, and the margin blew out to 18 before halftime.

Doyle (12 points), Deng and Jack McVeigh (both 10) all did plenty of damage on the scoreboard in the first half for the visitors, who never trailed. Galloway’s
huge dunk sparked Sydney early in the third period and they cut the margin to three.

Advertisement

But he and Bolden soon exited and Crawford’s scintillating run gave the JackJumpers the upper hand. The Kings are hopeful Galloway will be fit for Saturday’s
trip to take on the Cairns Taipans.

“The doctors may have other plans, but he’s doing alright,” Abdelfattah said.
</item 27>

<item 28>
Who’s the next Gawn, Grundy or Dusty? AFL’s talent guru names top 40 draft prospects and compares them to current stars (The Roar | Sports Writers Blog,
2023-11-15T23:48:27+00:00)
CLICK HERE for a seven-day free trial to watch the AFL on KAYO

KEVIN SHEEHAN’S TOP 40 PROSPECTS: VIA THE AFL

Kynan Brown

State: Victoria Metro

State League Club: Oakleigh Chargers

Community Club: Blackburn

Date of Birth: 13/01/05

Height: 181cm

Left-footed inside midfielder who has quick hands, sharp vision and creates play through the middle of the ground with his creativity. Had a remarkably
consistent year which included starring for Victoria Metro in the National Championships averaging 22.8 disposals at 80 percent efficiency as well as 4.5
clearances and 4.8 tackles to win their best and fairest award. Had similar numbers with the Oakleigh Chargers in the Coates Talent League including kicking
five goals and having 25 disposals in a best on ground performance in the Wildcard Round win against the Northern Knights.
</item 28>

<item 29>
Ground-breaking Gold Coast: Suns make AFLW finals for a second time (The Age, 2023-11-03T23:45:04+00:00)
“It was an awesome game and four quarters of Port Adelaide footy … It was awesome to finish up like that,” Phillips told Fox Footy after the 13.10 (88) to 4.6
(30) win over the Giants. Erin Phillips celebrates a goal during her final AFLW match. Credit: AFL Photos Earlier, she told the ABC that she wanted her legacy
to include doing everything she could to grow the game.

“Hopefully I can be remembered as someone who took their opportunities, loved the game and was willing to do whatever it took to ensure the next generation was
in a better position,” she said. Richelle Cranston of the Bulldogs is chaired off the ground by teammates Isabelle Pritchard and Ellie Blackburn after
announcing her retirement. Credit: Getty The 38-year-old kicked a momentous goal from a set shot in the final quarter, and she was smothered by her jubilant
teammates afterwards.
</item 29>

<item 30>
Ground-breaking Gold Coast: Suns make AFLW finals for a second time (The Age, 2023-11-03T23:45:04+00:00)
The Dogs have plenty to work on for next season as they look to regain a competitive edge, which was lost after a finals berth last year. Along with being in a
rebuild, they were badly injury-hit this season and used 32 players, more than any other club. Suns secure finals spot AAP Unsung hero Lucy Single has helped
Gold Coast book their second AFLW finals appearance with a 14-point win over Essendon. Losing midfield powerhouse Charlie Rowbottom to an ankle injury proved no
obstacle in the Suns’ maiden clash with the Bombers, as they finished the regular season in fifth spot after a 6.3 (39) to 3.7 (25) victory to kick off pride
round at Great Barrier Reef Arena on Friday.

While their men’s side have yet to make a finals appearance, this is the second time Gold Coast’s women have clinched a play-off spot, having also done so in
the cancelled 2020 season.
</item 30>

<item 31>
NBL Round 5: Harvey hits Hail Mary for Hawks to topple Taipans at the buzzer, Cheatham injured as Sixers upset Breakers (The Roar | Sports Writers Blog,
2023-10-26T21:11:37+00:00)
Without Miller running the point, the home side’s offence suddenly looked shaky in the third term as Sydney briefly took the lead on a Hogg triple.

The undermanned Taipans responded with a 15-0 burst either side of three-quarter time, highlighted by Swedish young gun Bobi Klintman’s thunderous dunk down the
middle over rival Next Star Alex Toohey.

Down by 14, Sydney closed the gap to four points before unsportsmanlike fouls against Denzel Valentine and Makuach Maluach ended their revival.

Lamenting “the defensive mistakes that we made”, Kings coach Mahmoud Abdelfattah added: “They made more plays than we did, they got to the 50/50 balls and they
stuck together.

“Kudos to them, they deserved it.”
</item 31>

<item 32>
Who’s the next Gawn, Grundy or Dusty? AFL’s talent guru names top 40 draft prospects and compares them to current stars (The Roar | Sports Writers Blog,
2023-11-15T23:48:27+00:00)
Continued that form into this season and was outstanding for Western Australia at the National Championships winning his team’s MVP award and All-Australian
honours after averaging 23.5 disposals and 6.5 marks playing as a defender and midfielder. Played a handful of senior games for Claremont and ended the season
being one of his team’s best in a premiership win at under-18 level. His agility, speed for his size and precise kicking are all features of his game.
</item 32>

<item 33>
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
</item 33>

<item 34>
Who’s the next Gawn, Grundy or Dusty? AFL’s talent guru names top 40 draft prospects and compares them to current stars (The Roar | Sports Writers Blog,
2023-11-15T23:48:27+00:00)
Was a threatening target up forward for the Allies during this year’s National Championships averaging 13 disposals and seven marks as well as kicking 11 goals
in four matches to be a vital cog in the title-winning team. Overall, he kicked 29 goals in eight matches this season to round off a splendid underage career.
Will receive a bid from a rival club very early on Draft night which the Suns will gladly match.

Weapon: Aerial ability and second efforts at ground level

AFL player comparison: Charlie Curnow (Carlton)

Nick Watson

State: Victoria Metro

State League Club: Eastern Ranges

Community Club: Montrose

Date of Birth: 24/02/05

Height: 170cm

A quick, agile and crafty forward who has a real appetite for the contest. Gave a glimpse of his potential in 2022 when he booted 23 goals in 10 matches in the
talent pathway.
</item 34>

<item 35>
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
</item 35>

<item 36>
Who’s the next Gawn, Grundy or Dusty? AFL’s talent guru names top 40 draft prospects and compares them to current stars (The Roar | Sports Writers Blog,
2023-11-15T23:48:27+00:00)
Also featured among Victoria Country’s best at the National Championships and produced a strong performance when representing Australia as part of the AFL
Academy in the clash against Port Adelaide’s SANFL team, racking up 19 disposals and taking five marks. His running capacity was confirmed when he finished
second in the 2km time trial at the national Draft Combine with a time of 5:52. He also demonstrated his power by producing the best running vertical jump with
his 98cm effort.

Weapon: Elite running and ball-winning ability

AFL player comparison: Sam Walsh (Carlton)

Caleb Windsor

State: Victoria Metro

State League Club: Eastern Ranges

Community Club: Lysterfield

Date of Birth: 09/06/05

Height: 184cm

A wingman with pace to burn. His preparedness to take the game on is a real weapon along with his decision-making and ball use.
</item 36>

<item 37>
NBL Round 5: Harvey hits Hail Mary for Hawks to topple Taipans at the buzzer, Cheatham injured as Sixers upset Breakers (The Roar | Sports Writers Blog,
2023-10-26T21:11:37+00:00)
“My words during the week were that we’re just not playing good, that showed again tonight,” Perth coach John Rillie said.

“Our offence isn’t functioning very good right now. We’re doing a great job holding teams to scores where we could put enough points on the board to win, in
saying that we’re not doing that.

“That’s certainly something we have to address and resolve here pretty quickly.”

Advertisement

Brisbane, who were on a four-game losing streak, started with the game’s first seven points and were still leading 20-9 by the end of the first quarter.

While the Wildcats had their moments in the second quarter, Brisbane did well to lead 42-36 at the long break.

It was 17-year-old Rocco Zikarsky who stood tall in the paint in the third quarter with Harrison in foul trouble.
</item 37>

<item 38>
NBL Round 4: Creek pours in points as Phoenix swamp Sixers, CJ claims Kings didn’t look after DJ (The Roar | Sports Writers Blog, 2023-10-19T21:03:08+00:00)
“Lack of focus, missing key players, making mistakes on covers.

“I don’t care where we flew from, what our schedule is, what happened the day before – I care zero. This isn’t the level of execution that we expect from
ourselves.”

Shea Ili (16 points, four assists) was important for Melbourne and Luke Travers finished with 16 points, including 15 in the first half.

Sixers, Phoenix continue to rise

Advertisement

The Adelaide 36ers have told each other some hard truths before responding in style to heap more pressure on the Perth Wildcats with their 89-78 victory.

In the other NBL match on Saturday, South East Melbourne Phoenix made it three straight wins, beating the Bullets 96-73 as Alan Williams led the way against a
side looking lost without the suspended Aron Baynes.
</item 38>

<item 39>
Comeback kings Richmond take the AFLW points in Cairns (The Age, 2023-10-20T23:51:34+00:00)
“I can name the whole team. Everyone had their second in the last couple minutes.

“That was such a team effort, everyone had a moment there when they had to put their head over the footy and lay a tackle and not let it slip.” Swanson (25
disposals, eight clearances) stood out for the Eagles, while Bombers workhorse Maddy Prespakis (30 disposals, nine clearances) was typically prominent in the
midfield alongside Georgia Nanscawen (21, six). With two rounds left, West Coast have a chance to build their winning streak when they meet the bottom of the
table Western Bulldogs next round. Slipping one place to sixth, Essendon have must-win matches against Carlton and Gold Coast on the road to the finals. The
Bombers will be without Steph Cain, who was sidelined with a concussion after an elbow to the head from Swanson.
</item 39>

<item 40>
Comeback kings Richmond take the AFLW points in Cairns (The Age, 2023-10-20T23:51:34+00:00)
Magpies ruck Sabrina Frederick, who starred with 39 hit-outs and used her height to bring the ball to ground at the advantage of her midfielders, said her side
had been practising for games like this.

“Even in the chaos, it’s quite comfortable because we train it. We train what scenarios to do in what moments, and even though it’s a bit crazy, it felt so
calm. We put players in place and we know what we’re doing,” said Frederick. “But close games are good ones, for everyone. We knew they were going to come out
like that [in the last quarter]. When they’re down, they’re just going to give us everything.” Collingwood’s Brianna Davey kicks her team forward against
Geelong. Credit: Getty Frederick admitted at times she “couldn’t see” and felt like she could be “blown away”.
</item 40>

<item 41>
Ground-breaking Gold Coast: Suns make AFLW finals for a second time (The Age, 2023-11-03T23:45:04+00:00)
“We love going in those four walls. We train there three, four times a week. So having that familiarity of the ground is awesome ... So I would be super-excited
to make finals for one, but then play at home would be just that extra and having all our support behind us really drives us on the field.” Tears abound in
Phillips’ emotional farewell

There were emotional scenes at Alberton Oval as the siren sounded on AFLW great Erin Phillips’ last game, with Port Adelaide saluting her with a huge 58-point
win over Greater Western Sydney on Saturday. The crowd, which included Phillips’ father Greg and mother Julie (both in tears after the game), more family and
friends, and Adelaide captain Chelsea Randall, erupted as the curtain finally came down for the three-time premiership player after eight seasons and 67 games.
</item 41>

<item 42>
NBL Round 5: Harvey hits Hail Mary for Hawks to topple Taipans at the buzzer, Cheatham injured as Sixers upset Breakers (The Roar | Sports Writers Blog,
2023-10-26T21:11:37+00:00)
The JackJumpers opened up on fire, racing to a 17-4 lead inside the opening five minutes including three balls to Crawford and McVeigh, but United soon steadied
and it was the introduction of Lual-Acuil that inspired it.

Melbourne would close the quarter outscoring Tasmania 17 points to four before three-pointers midway through the second stanza from Bowen and Clark put them up
five.

Advertisement

United were still leading 43-42 at halftime, on the back of 11 extra rebounds and 10 second-chance points, before captain Chris Goulding opened his account at
the start of the second half from downtown.

Clark hit his second long ball of the game soon after to push Melbourne’s lead out to nine, his third stretched it to double-figures and Bowen landed two
surprise ones of his own.
</item 42>

<item 43>
Craft beer and cheese boxes: Swans tap into something ‘uniquely Sydney’ to draw AFLW crowds | Jack Snape (The Guardian, 2023-10-26T23:42:29+00:00)
But more modest attendances in subsequent games, including two sub-2,000 crowds in Perth, suggest the popularity of the Matildas’ run to the semi-finals has
only had a limited impact on support for domestic women’s sport.

skip past newsletter promotion Sign up to Australia Sport Free newsletter Get a daily roundup of the latest sports news, features and comment from our
Australian sports desk Privacy Notice: Newsletters may contain info about charities, online ads, and content funded by outside parties. For more information see
our Newsletters may contain info about charities, online ads, and content funded by outside parties. For more information see our Privacy Policy . We use Google
reCaptcha to protect our website and the Google Privacy Policy and Terms of Service apply. after newsletter promotion

Livingstone said the “green and gold fever” associated with the Women’s World Cup was a one-off and the AFL has plans in place for long-term growth.
</item 43>

<item 44>
Ground-breaking Gold Coast: Suns make AFLW finals for a second time (The Age, 2023-11-03T23:45:04+00:00)
Single filled the massive Rowbottom-size gap, collecting 25 disposals and six clearances in her prolific performance across the field while also playing a
crucial tagging role on Maddy Prespakis. Coast with the most: Georgia Clayden of the Suns celebrates after kicking a goal. Credit: AFL Photos / Getty Images The
former rugby sevens player was well-supported by Claudia Whitford (25 possessions) and Alison Drennan (22 touches). Prespakis was held to eight contested
possessions, well below a season average of 14 per game, while Georgia Nanscawen proved a potent threat with 13 contested possessions and three score
involvements.

The Bombers broke through first with a goal from Amber Clarke but young talent Darcie Davies (two goals) was able to respond after being gifted 50 metres from a
protected area infringement by Mia Bush. Daisy D’Arcy, Jacqueline Dupuy, Jamie Stanton and Davies attempted to break away from the low-scoring tug-of-war in the
second with goals of their own.
</item 44>

<item 45>
Who’s the next Gawn, Grundy or Dusty? AFL’s talent guru names top 40 draft prospects and compares them to current stars (The Roar | Sports Writers Blog,
2023-11-15T23:48:27+00:00)
Has a sharp competitive edge which has seen him already play 10 senior matches for Subiaco in the WAFL. Tested impressively at the national Draft Combine to
record the second quickest time on the agility test (8.19) and a top ten finish in the 2km time trial (6:17).

Weapon: Power and clean hands

AFL player comparison: Cameron Rayner (Brisbane Lions)

Jed Walter

State: Queensland

State League Club: Gold Coast Suns Academy

Community Club: Palm Beach Currumbin

Date of Birth: 08/06/05

Height: 194cm

A strong and powerful tall forward who has taken all before him across the past two years for the Gold Coast Suns Academy and Allies, winning All Australian
honours at under 18 level in both seasons. Has a real physical presence up forward with his appetite for the contest, along with his second efforts and tackling
pressure.
</item 45>

<item 46>
Craft beer and cheese boxes: Swans tap into something ‘uniquely Sydney’ to draw AFLW crowds | Jack Snape (The Guardian, 2023-10-26T23:42:29+00:00)
“We’ll continue to keep building the base and keep building the foundations and have it not just strong at year seven, season eight, but strong for generations
to come.”

In a bid to improve the spectacle of the AFLW, the league introduced longer quarters this year and an interchange cap to help increase scoring. The effect,
according to Livingstone, has been positive.

“We’ve increased game time by eight minutes per match, on average, and we’ve had record scores across the rounds,” she said. “Whereas leading into the season we
hadn’t scored more than 100 goals in a round, we’ve regularly been scoring over 100 goals in a round.”

The Swans recorded their highest total last round, in their 14-12-96 to 6-3-39 defeat of the Western Bulldogs, the league’s winless bottom side. Molloy recorded
17 score involvements to set a new AFLW mark.

But Livingstone said the AFL was happy with the level of competition.
</item 46>

<item 47>
Who’s the next Gawn, Grundy or Dusty? AFL’s talent guru names top 40 draft prospects and compares them to current stars (The Roar | Sports Writers Blog,
2023-11-15T23:48:27+00:00)
Played all four games for Victoria Metro at the National Championships and was his team’s best player in the loss to the Allies, while also shining with 22
disposals in the win over Western Australia. Across his 16 matches with the Eastern Ranges in the Coates Talent League he averaged 19.4 disposals to be one of
their most consistent players during their run to the grand final. Recorded a slick time of 2.92 seconds on the 20m sprint during the national Draft Combine
highlighting his dazzling acceleration and backed it up by recording 6:28 for the 2km time trial which demonstrated a combination of elite speed and endurance
which are essential to play on the wing.

Weapon: Elite speed and ball use

AFL player comparison: Nick Martin (Essendon)
</item 47>

<item 48>
Who’s the next Gawn, Grundy or Dusty? AFL’s talent guru names top 40 draft prospects and compares them to current stars (The Roar | Sports Writers Blog,
2023-11-15T23:48:27+00:00)
An excellent decision-maker and ball user by hand or foot, he averaged 19.5 disposals, 12.5 contested possessions and 6.5 clearances to finish third in the
Larke Medal voting and took out Victoria Country’s MVP. Further added to his very impressive record by earning selection in the All Australian team for the
second year in a row. Handled the step up to playing senior football with three impressive games in the VFL and was the best player for the Australia under-18
team in its match against Port Adelaide’s SANFL team.
</item 48>

<item 49>
Comeback kings Richmond take the AFLW points in Cairns (The Age, 2023-10-20T23:51:34+00:00)
Molloy, the Swans’ star off-season recruit, turned on one of the best performances of her career to demolish their injury-hit hosts.

The former Collingwood ace was a class above as she kicked 3.4 from her 28 possessions and played a crucial role in setting up numerous other Sydney goals.
Loading “She [Molloy] was itching to get in there [in the midfield],” Swans coach Scott Gowans said. “We kept her forward deliberately. It was important Chloe
played a good block of time forward so we could have a look at it.” The Blues lost to Greater Western Sydney by two points on Saturday afternoon, denying them
the chance to leapfrog into the top eight, 8.1 (49) to 7.5 (47).

Narrm reclaim top spot Reigning premiers Narrm have reclaimed top spot on the ladder and their position as flag favourites after defeating third-placed North
Melbourne by 23 points at Ikon Park on Saturday night.
</item 49>

<item 50>
Comeback kings Richmond take the AFLW points in Cairns (The Age, 2023-10-20T23:51:34+00:00)
If the Lions had lost, they were at risk of falling out of the top four with second-season side Essendon chasing their spot. Last season’s best and fairest Ally
Anderson said her side loved playing at its training ground in Springfield, which it has made a fortress, and work well in hot conditions . “It was really hot.
It was a grind… But yeah, it was a good finish,” Anderson told Fox Footy post-game. “Every week that we play here, every time that we play here, we get better
and better. So, it was just a matter of time before we finally got the first win and then it hopefully [will] open the floodgates a bit.”

The Lions next play St Kilda before taking on the reigning premiers Melbourne in the final round. After this weekend, they have staked their claim as being
premiership contenders again. Adelaide will play top-four side North Melbourne next weekend before 18th-placed Western Bulldogs.
</item 50>

<item 51>
Craft beer and cheese boxes: Swans tap into something ‘uniquely Sydney’ to draw AFLW crowds | Jack Snape (The Guardian, 2023-10-26T23:42:29+00:00)
But Livingstone said the AFL was happy with the level of competition. “Our ladder, from position five to 13 – there’s only a game separating it. It just shows
that this is a really competitive competition.”
</item 51>

<item 52>
NBL Round 5: Harvey hits Hail Mary for Hawks to topple Taipans at the buzzer, Cheatham injured as Sixers upset Breakers (The Roar | Sports Writers Blog,
2023-10-26T21:11:37+00:00)
“Kudos to them, they deserved it.”

Advertisement

Phoenix hold off fast-finishing Breakers

South East Melbourne Phoenix have survived a massive late scare to clinch a tense 103-100 victory over the fast-finishing New Zealand Breakers at John Cain
Arena.

The Phoenix were eyeing off a percentage pile-on when they led by a whopping 26 points on Saturday before the bottom-placed, injury-hit Breakers produced a
brave rearguard to almost pinch a miracle.

The Phoenix’s big four – Alan Williams (21 points, 13 rebounds), Mitch Creek (20 points), Gary Browne (18 points, eight assists) and Will Cummings (16 points,
all in the first half) – led the way but the final stages got too close for comfort.
</item 52>

<item 53>
Who’s the next Gawn, Grundy or Dusty? AFL’s talent guru names top 40 draft prospects and compares them to current stars (The Roar | Sports Writers Blog,
2023-11-15T23:48:27+00:00)
He has been part of the Sydney Swans’ Academy since the under-10s.

Weapon: Contested ball-winning ability

AFL player comparison: Tim Taranto (Richmond)

Lance Collard

State: Western Australia

State League Club: Subiaco

Community Club: North Beach

Date of Birth: 26/03/05

Height: 180cm

Small forward who is extremely dangerous around goals. Booted 39 goals in 16 matches in Western Australia’s pathway this year where he played predominantly at
Colts level. Also represented his state in three games at the National Championships which gave him a taste of playing against the nation’s best. He then
highlighted his potential by kicking five-goal bags in three of the final five games of the WAFL Colts season.
</item 53>

<item 54>
Comeback kings Richmond take the AFLW points in Cairns (The Age, 2023-10-20T23:51:34+00:00)
The Kangaroos’ Alice O’Loughlin and Melbourne’s Maeve Chaplin battle for the ball. Credit: AFL Photos It was a hard-fought, low-scoring battle between the two
powerhouse sides that play a similar game style. Each set up well, are fast and like a good dose of contested marking. But Narrm controlled more of the game by
tightening around the contest and being cleaner going forward. They also defended extremely well by suffocating the opportunities North created and prevailed
5.7 (37) to 2.2 (14). While the Demons scored just one goal a quarter in the second half – Tyla Hanks in the third and Eden Zanker in the fourth – the Kangaroos
were goalless for three terms, having led by five points at the first change. Their opening goals were thanks to Niamh Martin and Tess Craven.

Then a crafty goal from Demons skipper Kate Hore to open the second term put her side up, and Melbourne never gave up the lead.
</item 54>

<item 55>
Who’s the next Gawn, Grundy or Dusty? AFL’s talent guru names top 40 draft prospects and compares them to current stars (The Roar | Sports Writers Blog,
2023-11-15T23:48:27+00:00)
He capped off a great year playing in the Sandringham Dragons premiership team in the Coates Talent League.

Weapon: Height and intercept marking

AFL player comparison: Jake Lever (Melbourne)

Connor O’Sullivan

State: NSW/ACT

State League Club: Murray Bushrangers

Community Club: Albury

Date of Birth: 19/05/05

Height: 198cm

Advertisement

Athletic tall defender who was outstanding for the title-winning Allies at the National Championships winning their MVP award and All Australian honours. He
averaged 17.8 disposals at 86 percent efficiency as well as averaging 8.2 marks and providing great rebound to lead the Allies defence. He has an exciting
natural leap, good speed and elite endurance highlighted with a top 10 placing in the 2km time trial at the national AFL Draft Combine with a time of 6:11.
</item 55>

<item 56>
Comeback kings Richmond take the AFLW points in Cairns (The Age, 2023-10-20T23:51:34+00:00)
It was the Western Bulldogs’ eighth loss in a row, and their biggest-ever defeat, and they have just two rounds left to clinch a win this season. They next play
17th-placed West Coast and then top-four side North Melbourne. For the Swans, however, the victory has them challenging for a top-eight spot alongside the likes
of Gold Coast, Geelong, Collingwood, St Kilda and Carlton.

The Swans, who went winless in their debut season last year, were propelled to victory by a Chloe Molloy masterclass. After a five-goal second term, which
included a four-minute burst of three majors, Sydney had already brought up their largest score in an AFLW game by halftime. Swan Chloe Molloy celebrates a
goal. Credit: AFL Photos / Getty Images The Bulldogs held their own in the last quarter, but the damage had already been done. The total was also the biggest
the Dogs have conceded in their history, topping the 83 Melbourne kicked against them in round three.
</item 56>

<item 57>
Who’s the next Gawn, Grundy or Dusty? AFL’s talent guru names top 40 draft prospects and compares them to current stars (The Roar | Sports Writers Blog,
2023-11-15T23:48:27+00:00)
Was impossible to contain during this year’s National Championships booting 14 majors in four matches for Victoria Metro to top the goal kicking. He also
averaged 12.2 disposals and was rewarded with All Australian selection. He further expanded his game by spending more in the midfield and across half back for
the Eastern Ranges and averaged 20 disposals while kicking 11 goals in nine matches in the Coates Talent League.

Weapon: Speed, agility and leap

AFL player comparison: Shai Bolton (Richmond)

Advertisement

This is just what Nick Watson does ????
</item 57>

<item 58>
Comeback kings Richmond take the AFLW points in Cairns (The Age, 2023-10-20T23:51:34+00:00)
Dogs forward reveals battle with kidney disease Western Bulldogs forward Richelle Cranston has revealed she’s been living, and playing, with stage five chronic
kidney disease. “I still manage to do the things I love,” the inaugural AFLW player posted to Instagram. “It’s just really hard sometimes.” Richelle Cranston of
the Bulldogs: “It’s just really hard sometimes.” Credit: Getty Images The 34-year-old posted the caption along with a video detailing her life with the disease
including weekly blood tests and daily blood pressure checks, tracking water and food intake, taking medications including resonium (to lower potassium levels),
all while training in the high-performance environment.

The Dogs are Cranston’s third club after starting at Melbourne and having a stint with Geelong before heading to the Bulldogs last year. Cranston lined up for
the Bulldogs in her side’s 57-point loss to Sydney on Saturday at Whitten Oval, 14.12 (96) to 6.3 (39).
</item 58>

<item 59>
England choose survival over oblivion and a chance of summer trip to France | Jonathan Liew (The Guardian, 2023-12-01T22:57:40+00:00)
These are issues not so much of tactics but cohesion and energy: the timing of runs, the sharpness required not just to anticipate where the ball is going to
be, but actually get there. You can still chase when you’re running on fumes, but to create: this needs an idea and a vision, the clarity of mind to see the
picture before it has materialised.

skip past newsletter promotion Sign up to Moving the Goalposts Free weekly newsletter No topic is too small or too big for us to cover as we deliver a weekly
roundup of the wonderful world of women’s football Privacy Notice: Newsletters may contain info about charities, online ads, and content funded by outside
parties. For more information see our Newsletters may contain info about charities, online ads, and content funded by outside parties. For more information see
our Privacy Policy . We use Google reCaptcha to protect our website and the Google Privacy Policy and Terms of Service apply.
</item 59>

<item 60>
Who’s the next Gawn, Grundy or Dusty? AFL’s talent guru names top 40 draft prospects and compares them to current stars (The Roar | Sports Writers Blog,
2023-11-15T23:48:27+00:00)
Was also named captain of the All Australian team after leading the Allies to their first title. Caught the eye with a strong performance for the Australia
under-18 team in its representative match against Port Adelaide’s SANFL side and backed it up in the second match against Carlton’s VFL side where he collected
24 disposals to be one of Australia’s best in victory. Finished the year as a back-to-back Coates Talent League premiership player with the Sandringham Dragons
where he averaged 31 disposals across nine matches.
</item 60>

<item 61>
NBL Round 5: Harvey hits Hail Mary for Hawks to topple Taipans at the buzzer, Cheatham injured as Sixers upset Breakers (The Roar | Sports Writers Blog,
2023-10-26T21:11:37+00:00)
Shea Ili (17 points) and Tanner Krebs (14) also reached double figures for Melbourne.

The result extended United’s winning streak to six games, giving them an 8-1 record on top of the ladder.

Import forward Gary Clark (24 points) and Justin Robinson (20) tried to give Illawarra a lift but Tyler Harvey managed just 10 points, three days after his
clutch three-pointer sealed a thrilling win over Cairns.

Advertisement

In a sloppy start, United committed five turnovers in the first period and trailed at quarter-time for the first time this season.

Clark’s eight points on perfect shooting led Illawarra to a 29-24 advantage at that point.

His tally swelled to 16 by the main break and the Hawks’ lead reached double figures soon after United temporarily lost Travers to injury.

But Goulding caught fire to lead United to a 52-51 at the main break.
</item 61>

<item 62>
Who’s the next Gawn, Grundy or Dusty? AFL’s talent guru names top 40 draft prospects and compares them to current stars (The Roar | Sports Writers Blog,
2023-11-15T23:48:27+00:00)
He also impressed for the Australia under-18 team in its second match against Carlton’s VFL side at Marvel Stadium, gathering 26 disposals and six clearances to
claim the medal as his team’s best player.

Weapon: One-touch hands and creativity

AFL player comparison: Touk Miller (Gold Coast Suns)

Advertisement

Ryley Sanders

State: Tasmania

State League Club: Sandringham Dragons

Community Club: North Launceston

Date of Birth: 21/01/05

Height: 185cm

Strong-bodied midfielder who is powerful and courageous in his attack on the ball and the pressure he exerts on the opposition. Strong two-way running is
another feature of his game and was on display in dominating for the Allies at the National Championships, averaging 35 disposals at 80 percent efficiency, 14.5
contested possessions, five clearances and four tackles to win the Larke Medal as the best and fairest player.
</item 62>

<item 63>
Ground-breaking Gold Coast: Suns make AFLW finals for a second time (The Age, 2023-11-03T23:45:04+00:00)
WEEK ONE FINALS FIXTURE SATURDAY NOVEMBER 11 First Qualifying Final: Adelaide v Brisbane Lions, 2.35pm local (3.05pm AEDT), Norwood Oval First Elimination
Final: Gold Coast v Sydney, 6.15pm local (7.15pm AEDT), Heritage Bank Stadium SUNDAY NOVEMBER 12 Second Elimination Final: Geelong v Essendon, 1.05pm AEDT,
GMHBA Stadium Second Qualifying Final: Melbourne v North Melbourne, 3.05pm AEDT, Ikon Park The eight was not decided until the final game of the regular season,
when Sydney downed Fremantle by 22 points to leapfrog St Kilda into the final spot, after the Saints had kept themselves in the race with a 20-point win over
Carlton earlier in the day, 7.4 (46) to 3.8 (26).
</item 63>

<item 64>
Comeback kings Richmond take the AFLW points in Cairns (The Age, 2023-10-20T23:51:34+00:00)
Conversely, the score was the most the Saints have conceded in any quarter this season.

Aine Tighe set the tone for the Dockers with a brilliant goal less than 30 seconds into the first quarter, followed by two majors from Gabby O’Sullivan and
another by Angelique Stannett. Ella Friend slotted a much-needed goal for the visitors to make it 4.2 (26) to 1.3 (9) at quarter-time. With less than a minute
till the long break, captain Hayley Miller (two goals) kicked the only major of the second quarter to give Fremantle a comfortable 26-point lead at half-time.
Despite two goals from Tyanna Smith and J’Noemi Anderson in the second half, the Saints struggled to eat into the deficit. Fremantle (4-4) face reigning
premiers Melbourne next, before finishing the season against 11th-placed Sydney.
</item 64>

<item 65>
Comeback kings Richmond take the AFLW points in Cairns (The Age, 2023-10-20T23:51:34+00:00)
“Richmond were just absolute beasts, especially their midfielders. They really worked hard all night,” Hawthorn coach Bec Goddard said after the game. “Our work
rate really dropped off and we just watched them play some really good footy. “If you’re a Hawks fan, you would’ve seen some really exciting moments of footy
tonight.

“Keep getting excited. We are coming, we’re growing, we’re getting better.” Hawthorn’s Irish ace Aine McDonagh kicked the opening goal before Tigers skipper
Katie Brennan slotted two unanswered majors to celebrate her 50th game in style. Fellow 50-gamer Greta Bodey put boot to ball to snatch her own goal before
quarter-time. A skirmish erupted in the second term after Brennan took exception to a tackle by Mattea Breed on Laura McClelland. McClelland was gifted a
50-metre penalty following a bump against Sarah Hosking by Sophie Locke and converted from inside the goal square while players were occupied by the tussle.
</item 65>

<item 66>
Ground-breaking Gold Coast: Suns make AFLW finals for a second time (The Age, 2023-11-03T23:45:04+00:00)
Down by four points at half-time in the low-scoring game, a third-term goal from Rebecca Privitelli sparked the Swans to a 5.10 (40) to 2.6 (18) win on Sunday
at Fremantle Oval.

The second-year expansion side stormed home with three unanswered goals to secure eighth spot. It’s a dramatic improvement, with the Swans winless in their
first season. “Any club that’s able to go from winless one year to a finals berth, it’s incredible,” Sydney co-captain Chloe Molloy told Fox Footy after the
match. “You can see the emotion on the girls’ faces. Coming into the season, the odds were against us. Just to get into the eight, we’ve managed to do that.
We’re stoked. It’s pretty special for the club.”
</item 66>

<item 67>
NBL Round 4: Creek pours in points as Phoenix swamp Sixers, CJ claims Kings didn’t look after DJ (The Roar | Sports Writers Blog, 2023-10-19T21:03:08+00:00)
In Adelaide, the 36ers responded to a disappointing loss on Thursday night, also at home to the Phoenix, and handed the Wildcats a third straight defeat with
their 11-point win.

Sitting in bottom spot with one win from five games coming into the macth, the players got together to draw a line in the sand.

It certainly worked and their effort, energy and desperation was dramatically improved against Perth, but so was the execution and a focus on getting Isaac
Humphries more involved inside.

He put up 26 points and eight rebounds in a match-winning performance but the shortening of the rotation to use just eight players from coach CJ Bruton worked
too.

Advertisement

“We met as a playing group yesterday and talked through a lot of stuff,” Humphries said.

“We just wanted to come out today and put into action everything we talked about.

“We executed what the coaches needed from us tonight and we executed what we, as a playing group, wanted to execute as well.”
</item 67>

<item 68>
NBL Round 5: Harvey hits Hail Mary for Hawks to topple Taipans at the buzzer, Cheatham injured as Sixers upset Breakers (The Roar | Sports Writers Blog,
2023-10-26T21:11:37+00:00)
Despite being without star big Aron Baynes (suspended) and import guard Shannon Scott (hamstring), the Bullets soared to a second straight victory on Sunday and
snapped a quirky streak in the process.

It was their first win after seven straight losses in Sunday fixtures, in a run that dates back to last season.

Far more importantly, the result levelled Brisbane’s current season record at 4-4 after five rounds.

But it didn’t come easily as fourth-quarter specialists Tasmania trimmed a seven-point deficit back to a single point when Majok Deng drove to the rack with 34
seconds remaining.

Advertisement

The two sides traded blows in the dying stages and the JackJumpers almost sent the game to overtime when they pinched Sam McDaniel’s inbounds pass on the final
play.

Jack McVeigh got a hand in to force the turnover and Tasmania found Jordon Crawford, but his decent look at an attempted game-leveller rimmed out.
</item 68>

<item 69>
Comeback kings Richmond take the AFLW points in Cairns (The Age, 2023-10-20T23:51:34+00:00)
The Crows elected not to challenge the decision. ‘Horror week’: Bombers stunned by struggling, controversy-hit Eagles West Coast withstood a late-term surge
from Essendon to narrowly claim their second win of the AFLW season.

Fresh off a controversy caused by comments their coach Michael Prior made following last week’s 70-point loss to Melbourne, the Eagles prevailed 3.5 (23) to 2.7
(19) in a slugfest at the appropriately named Windy Hill on Sunday. West Coast’s Kate Bartlett (left) and Isabella Lewis celebrate Sunday’s surprise win over
Essendon flanked by their teammates. Credit: Getty Prior had blasted the decision to pit strugglers West Coast against defending premiers Melbourne, saying:
“How we play that team is beyond me. That is what you get when you get fixturing like that.” He later backtracked on the criticism, but his comments did put
added pressure and focus on his team this week. Michael Prior guided West Coast to a gutsy win.
</item 69>

<item 70>
Who’s the next Gawn, Grundy or Dusty? AFL’s talent guru names top 40 draft prospects and compares them to current stars (The Roar | Sports Writers Blog,
2023-11-15T23:48:27+00:00)
Was also impressive for the Murray Bushrangers in the Coates Talent League, averaging 16 disposals and 3.7 tackles across 10 matches and booted nine goals. His
high natural leap and pace were both prominent at the national Draft Combine with his standing vertical leap of 78cm ranked second overall and his time of 2.96
seconds on the 20m sprint ranked in the top ten.

Weapon: Elusiveness and creativity

AFL player comparison: Tom Papley (Sydney Swans)

Will Graham

State: Queensland

State League Club: Gold Coast Suns Academy

Community Club: Palm Beach Currumbin

Date of Birth: 27/07/05

Height: 186cm

Adapted seamlessly to playing in defence for the Allies after predominantly playing midfield for the Gold Coast Suns Academy.
</item 70>

<item 71>
Ground-breaking Gold Coast: Suns make AFLW finals for a second time (The Age, 2023-11-03T23:45:04+00:00)
“I don’t think I’ve been as nervous to kick a goal in my life, to be honest,” she joked. “But nah, it was awesome.” On Saturday, the competition also said
goodbye to a range of players including Western Bulldogs forward Richelle Cranston, who has battled stage five chronic kidney disease while playing this season,
and Hawthorn’s Akec Makur Chuot, who played 40 games for three clubs.

Cranston, who was an inaugural AFLW player, will start kidney dialysis in January so had no option but to hang up the boots and focus on her treatment and
health. “[I’m feeling] sad. Emotions hit me when the siren went. It’s been a really emotional week, the girls have really gotten around me. But yeah, my health
has sort of deteriorated, it’s been a really hard year,” Cranston told Fox Footy after the Bulldogs lost to North Melbourne 7.13 (55) to 1.3 (9).
</item 71>

<item 72>
Who’s the next Gawn, Grundy or Dusty? AFL’s talent guru names top 40 draft prospects and compares them to current stars (The Roar | Sports Writers Blog,
2023-11-15T23:48:27+00:00)
Was an outstanding player for Victoria Country at the National Championships to earn All Australian honours, which included a match-winning performance against
Victoria Metro in the final game where he amassed 22 disposals, 10 marks and four goals. Was also dominant for the Gippsland Power to win their best and fairest
and earn selection in the Coates Talent League Team of the Year. Finished the year with 42 goals across 20 matches in all competitions. His natural leap was
highlighted at the national Draft Combine with his running vertical jump of 97cm ranked second overall. Hails from a talented football family, with his brother
Xavier now at Essendon and sister Yasmin playing for Port Adelaide.
</item 72>

<item 73>
Comeback kings Richmond take the AFLW points in Cairns (The Age, 2023-10-20T23:51:34+00:00)
The Lions’ Orla O’Dwyer speeds away against the Crows. Credit: Getty Images Adelaide started fast but the game evened up as the Lions had the better of the
second quarter with three goals to one. The Crows then took back control in the third with three goals to none, which set the stage for a tense final term. The
fourth quarter was a heated arm wrestle, and scores were tied with four minutes left on the clock. The Lions’ Bella Dawes kicked the only goal of the last term,
but it was a series of behinds in the final minutes that won the game for them, including one from Lily Postlethwaite that broke the stalemate.

Adelaide’s Niamh Kelly navigates her way through traffic against brisbane. Credit: Getty The two sides have had intense battles in the competition’s history,
including two grand finals, and this was no different.
</item 73>

<item 74>
Who’s the next Gawn, Grundy or Dusty? AFL’s talent guru names top 40 draft prospects and compares them to current stars (The Roar | Sports Writers Blog,
2023-11-15T23:48:27+00:00)
His natural leap was on display at the national Draft Combine where he featured in the top 10 performers in both the standing vertical jump (75cm) and running
vertical jump (91cm).

Weapon: Aerial ability and left-foot kicking

AFL player comparison: Hayden Young (Fremantle)

Angus Hastie

State: Victoria Country

State League Club: Geelong Falcons

Community Club: Geelong West Giants

Date of Birth: 19/09/05

Height: 190cm

An athletic medium defender who sets the play up from defence with his run and ball use. Has shown a preparedness to back himself and take the game on to create
opportunities for his team. His intercepting ability is also a strength which saw him represent Victoria Country in all four matches at the National
Championships. Played 13 matches with the Geelong Falcons in the Coates Talent League this year averaging 20 disposals and 4.6 marks.
</item 74>

<item 75>
The story behind anguished Wallabies star's photo (Wide World Of Sports, 2023-10-07T22:10:40+00:00)
Oct 8th, 2023

Oct 8th, 2023

S﻿AINT-ETIENNE: The Wallabies will have to endure a more tortuous Sunday night watch than even the worst rom com as their Rugby World Cup fate is decided by
Fiji and Portugal in Toulouse.

Australia will gather together in their plush Saint-Galmier hotel at 9pm local time (6am Monday AEDT) to view what will almost certainly be their World Cup
execution.

The Wallabies are currently sitting second in Pool C but a single bonus point would be enough for Fiji to leapfrog them and advance to the quarter-finals in
Marseille.

Watch all the action from Rugby World Cup 2023 on the home of rugby, Stan Sport.
</item 75>

<item 76>
Comeback kings Richmond take the AFLW points in Cairns (The Age, 2023-10-20T23:51:34+00:00)
Geelong coach Daniel Lowther said the strong winds made it hard for his side to clear the ball out of their defensive 50-metre zone and to move the ball with
any certainty around the ground.

Loading “At one point the rain hit our box, it was flowing,” Lowther said. “Both teams had to fight in the conditions, but it didn’t help the game. “Our marking
contest wasn’t too flash, our ground ball representation was a bit hit and miss, so our defence were under a bit more pressure than they should have been, but
they fought back really well,” Lowther said. Collingwood next play a Sydney line-up including former Magpies star Chloe Molloy, while Geelong will take on
Richmond at Ikon Park.

Dockers spoil Saints’ finals hopes Fremantle recorded a comfortable 24-point win yesterday to bring an abrupt end to St Kilda’s four-game streak and dash their
AFLW finals hopes. J’Noemi Anderson is tackled by a group of Dockers.
</item 76>

<item 77>
Comeback kings Richmond take the AFLW points in Cairns (The Age, 2023-10-20T23:51:34+00:00)
Bridie Hipwell kicked the only goal of the third quarter to give Hawthorn a game-high nine-point lead. Loading Richmond rebounded in the last term, with
Courtney Jones ending Hawthorn’s three-goal run with her first touch of the night, cutting the margin to three points. Dogged by inaccuracy in front of goal,
Yassir reduced the deficit to a single point before finally breaking through on her third attempt. That put the Tigers ahead by six points with 11 minutes left,
before Caitlin Greiser sealed the win with a massive goal from 45 metres.
</item 77>

<item 78>
Who’s the next Gawn, Grundy or Dusty? AFL’s talent guru names top 40 draft prospects and compares them to current stars (The Roar | Sports Writers Blog,
2023-11-15T23:48:27+00:00)
Weapon: Intercept ability and long kick

AFL player comparison: Callum Wilkie (St Kilda)

Cooper Simpson

State: Victoria Country

State League Club: Dandenong Stingrays

Community Club: Mt Martha

Date of Birth: 13/02/05

Height: 181cm

An ultra professional midfielder with pace, composure and great finishing skills. Also has a neat step through traffic and runs hard both offensively and
defensively. Has shown the ability to perform when challenged and in the big moments when the game is on the line. Started the season in good touch with the
Dandenong Stingrays in the Coates Talent League but suffered an ankle injury in Victoria Country’s opening game of the National Championships and was unable to
play any further matches at that level. Made an impressive return for the final four matches with the Stingrays including having 28 disposals against the
Tasmania Devils which reminded scouts of his class and AFL potential.
</item 78>

<item 79>
McRae reveals hidden message he carried under collar of his grand final shirt (The Sydney Morning Herald, 2023-09-30T22:45:39+00:00)
Adams and McStay – who arrived from the Lions in the off-season as an unrestricted free agent – leaned on each other to cope with the difficult moment, and
needed some privacy to deal with their personal misfortune before they could share in the team’s raucous celebrations. McStay may need as many as eight weeks to
recover from the high-grade medial collateral ligament strain he suffered in the club’s preliminary final win. Collingwood’s injured recruit Dan McStay at
training before the grand final. Credit: Eddie Jim “We shed a few tears and a bit of emotion, then we brushed it off and said, ‘Let’s get on with it and go
celebrate with the boys’. It was a great night,” Adams said. John Noble was another sad story, playing every game in the home-and-away season, only to be
dropped for Collingwood’s qualifying final clash with Melbourne and never forcing his way back in.
</item 79>

<item 80>
Who’s the next Gawn, Grundy or Dusty? AFL’s talent guru names top 40 draft prospects and compares them to current stars (The Roar | Sports Writers Blog,
2023-11-15T23:48:27+00:00)
Uses his dazzling speed to break the lines and kicks the ball with pinpoint accuracy on his preferred left foot. Averaged 32.8 disposals and 5.8 clearances
across four matches for the Allies at the National Championships to earn All Australian honours. Was equally dominant for the Tasmania Devils in the Coates
Talent League having more than 30 disposals in seven of his nine matches and also kicked 12 goals. Showed he could match it against more experienced opponents
by having 20 disposals and three clearances in Australia’s win over Carlton’s VFL team.
</item 80>

<item 81>
McRae reveals hidden message he carried under collar of his grand final shirt (The Sydney Morning Herald, 2023-09-30T22:45:39+00:00)
Loading “We just want to win. We come into the club every day and get better, and we just set ourselves up, so when we come to game day, we know we’ve done
everything, to get the result we want.” Maynard had a tumultuous start to September after his infamous collision with Melbourne’s Angus Brayshaw left the Demon
concussed and him defending himself at the tribunal. With a three-match ban hanging over his head, Maynard feared his dream to achieve premiership glory could
be lost before he beat the charge.

Collingwood’s triumph helped banish the pain of their grand final defeat to West Coast in 2018, when Maynard was unfortunate not to receive a free kick for a
block against Willie Rioli in the same passage that led to Dom Sheed’s match-winning goal. “Everyone knows that was a bizarre and crazy sort of week, mentally
[after the Brayshaw incident],” Maynard said. “I wasn’t going well because of the what-ifs, if I missed this.
</item 81>

<item 82>
Craft beer and cheese boxes: Swans tap into something ‘uniquely Sydney’ to draw AFLW crowds | Jack Snape (The Guardian, 2023-10-26T23:42:29+00:00)
“Even at Henson Park [the Swans’ other home ground in Sydney’s gentrified inner west], they’ve gone to microbreweries in the local area and invited them in, but
also promoted through their clientele to come to the football.”

The Swans forward Bella Smith said the players were proud of the club’s attendance record, and were grateful for the support. “We get there two hours before the
start of the game and the hill [at Henson Park] is bare. When you come out for the first team warm-up, the hill is slowly filling and then by the time the first
bounce gets around, it is full,” she said. “You can hear it, the atmosphere is amazing.”

The average attendance for the AFLW is a little over 2,500 a match this year, tracking approximately on par with previous seasons and with a finals uptick
likely to come.
</item 82>

<item 83>
NBL Round 5: Harvey hits Hail Mary for Hawks to topple Taipans at the buzzer, Cheatham injured as Sixers upset Breakers (The Roar | Sports Writers Blog,
2023-10-26T21:11:37+00:00)
But we held on.”

NBA and NBL champion Ian Clark top-scored with 19 points, with rookie Kyle Bowen adding a career-best 17 as did Jo Lual-Acuil along with 10 rebounds.

Advertisement

Jack McVeigh led the way for Tasmania with 20 points and seven rebounds, with Jordon Crawford contributing 16 points and four assists, and Milton Doyle 12
points and five assists.

Tasmania coach Scott Roth was philosophical afterwards, having given up 18 offensive rebounds and lost the battle on the glass by 16.

“We got a little uncharacteristic with our shot selection, which was a credit to them,” he said.

“Then they beat us up on the glass. It’s just little things when you get in those kind of games that are going to make the separation points.”
</item 83>

<item 84>
Ground-breaking Gold Coast: Suns make AFLW finals for a second time (The Age, 2023-11-03T23:45:04+00:00)
With only 0.4 of a percentage point separating Sydney in ninth and St Kilda in eighth, the win assured the Swans of retaking eighth spot. If the Saints had
qualified, it would have been their maiden finals appearance. Saints goalkicker Jesse Wardlaw had a heavy collision with Carlton’s Keeley Sherar late in the
final quarter of their clash.

Georgia Patrikios showed her class in the do-or-die match, collecting 25 disposals and eight contested possessions in a powerhouse display, while Hannah Priest
was a brick wall in defence, with nine intercept possessions. Abbie McKay worked overtime to counter Patrikios in the midfield, but her side was outclassed
across the park. McKay racked up 28 disposals and 12 clearances alongside Mimi Hill (30 disposals), while defender Sherar (17) toiled against a potent Saints
attack. Meanwhile, Sydney prevailed in a fierce tug-of-war with the Dockers to seal their first AFLW finals appearance.
</item 84>

<item 85>
NBL Round 5: Harvey hits Hail Mary for Hawks to topple Taipans at the buzzer, Cheatham injured as Sixers upset Breakers (The Roar | Sports Writers Blog,
2023-10-26T21:11:37+00:00)
Sobey and Smith then knocked down shots as the Bullets lead ballooned to 20.

The Wildcats did make a charge with a Pinder-inspired 11-0 run in the fourth to cut the deficit to three, but the Bullets answered with strong inside finishes
from Harrison and Bannan to secure the win.

United flex muscles in road win over JackJumpers

Melbourne United have had to withstand a hot start and finish from the Tasmania JackJumpers but in between were outstanding on the way to a 90-82 victory in the
NBL.

Advertisement

The league’s top two teams locked horns at a sold out MyState Bank Arena on Friday night and it was the JackJumpers who raced out of the blocks with a 17-4
start, but United took control from there.
</item 85>

<item 86>
NBL Round 5: Harvey hits Hail Mary for Hawks to topple Taipans at the buzzer, Cheatham injured as Sixers upset Breakers (The Roar | Sports Writers Blog,
2023-10-26T21:11:37+00:00)
Stepping up were acting skipper Bul Kuol (24, played all 40 minutes) and big Sam Waardenburg (22) for the Snakes, who shrugged off last Thursday’s
buzzer-beating heartbreak in Illawarra by taking the NBL’s most prized scalp.

Advertisement

It shot them from fifth to third on the ladder, while sending the Kings slumping from second to fourth.

“We kept moving down the bench to bring guys in,” Cairns coach Adam Forde said. “Everyone responded.

“I’m super happy for the group that they could fill some short-term reward with a win at home.”
</item 86>

<item 87>
NBL Round 5: Harvey hits Hail Mary for Hawks to topple Taipans at the buzzer, Cheatham injured as Sixers upset Breakers (The Roar | Sports Writers Blog,
2023-10-26T21:11:37+00:00)
The JackJumpers, who trailed by seven at three-quarter time, kept alive their record of winning every fourth term this season, but it wasn’t enough.

Brisbane captain Sobey led from the front as defensive specialist Mitch Norton kept Tasmania’s import Crawford (16 points) relatively quiet until the final
period.

Tyrrell Harrison and Josh Bannan (10 points each) were important for the Bullets, while Chris Smith (11) and Isaac White (10) gave solid contributions off the
bench.

The gut-wrenching loss ended a horror round five for the JackJumpers, who lost at home to Melbourne United before going down to the Bullets.

Advertisement

It left them with a 4-4 record after a strong 4-2 start to the season.
</item 87>

<item 88>
Comeback kings Richmond take the AFLW points in Cairns (The Age, 2023-10-20T23:51:34+00:00)
Hanks was pivotal for the Demons and starred with 26 disposals, seven tackles and one goal, while her partner in the midfield Olivia Purcell had 28 touches,
five clearances and five tackles. For the Roos, Ash Riddell led the disposal count with 31, while skipper Emma Kearney was crucial in defence. Melbourne’s
Alyssa Bannan is tackled by Lulu Pullar of the Kangaroos. Credit: Getty Hanks said with the state of competition, they just had to take it week by week, but it
was good to get the win. “We knew we were up against the best tonight,” said Hanks. “It’s a nice feeling tonight.”

Lions roar in big battle Meanwhile, on Saturday afternoon, the Brisbane Lions put an end to Adelaide’s unbeaten run with a three-point victory at Brighton Homes
Arena in a top-four thriller, 5.12 (42) to 6.3 (39), in a spectacle worthy of being a final.
</item 88>

<item 89>
Who’s the next Gawn, Grundy or Dusty? AFL’s talent guru names top 40 draft prospects and compares them to current stars (The Roar | Sports Writers Blog,
2023-11-15T23:48:27+00:00)
Weapon: Height and speed

AFL player comparison: Max Gawn (Melbourne)

"He's got the most upside of any player in our state."



You're going to want to learn more about South Australian Taylor Goad, a 206cm ruckman that is seriously quick.https://t.co/KztP0FJ4wL — AFL (@AFL) November 6,
2023

Advertisement

Phoenix Gothard

State: NSW/ACT

State League Club: Murray Bushrangers

Community Club: Albury

Date of Birth 07/09/05

Height: 178cm

A small forward who came to prominence with his creative forward play for the Allies in their unbeaten National Championship-winning run, where he was a
livewire at ground level and averaged 10.3 disposals, three tackles and a goal a game.
</item 89>

<item 90>
Craft beer and cheese boxes: Swans tap into something ‘uniquely Sydney’ to draw AFLW crowds | Jack Snape (The Guardian, 2023-10-26T23:42:29+00:00)
But crowds are short of targets that could accelerate the extension of the regular season, which grows from 10 games this year to 12 games by 2025.

5,474 fans were at North Sydney Oval to watch the Swans’ first AFLW victory earlier in the season. Photograph: Jack Snape/The Guardian

The league enjoyed strong attendances in round one, which were partly attributed to the so-called Matildas effect, but some clubs have struggled to attract fans
in large numbers this season. Crowds of less than 1,500 have been recorded in both traditional markets such as Melbourne and Perth, as well as areas targeted
for growth like Blacktown and the Gold Coast.

The A-League Women launched this month with a first round highlighted by a crowd of more than 11,000 in Sydney.
</item 90>

<item 91>
Ground-breaking Gold Coast: Suns make AFLW finals for a second time (The Age, 2023-11-03T23:45:04+00:00)
We’re stoked. It’s pretty special for the club.” Missing star ruck Ally Morphett to a season-ending wrist injury, Laura Gardiner stood the tallest in the
midfield, while Molloy (two goals), Montana Ham, Privitelli and Brooke Lochland hit the scoreboard. An off-season recruit from Geelong, 20-year-old Gardiner had
the ball on a string with a game-high 33 disposals and 18 contested possessions.

Demons denied the minor premiership The Brisbane Lions have toppled AFLW heavyweights Melbourne by 25 points to clinch a double chance in the finals and deny
the Demons the minor premiership. A year after Melbourne walked away with the 2022 premiership, the Lions avenged their upset grand final loss by defeating the
defending premiers 8.5 (53) to 4.4 (28) at Brighton Homes Arena on Saturday. Melbourne’s Lauren Peace fights for the ball againt the Lions.
</item 91>

<item 92>
Craft beer and cheese boxes: Swans tap into something ‘uniquely Sydney’ to draw AFLW crowds | Jack Snape (The Guardian, 2023-10-26T23:42:29+00:00)
Quince paste and hand-picked grapes have played a part in helping the Sydney Swans launch themselves to the top of the AFLW attendance charts, less than two
years after their first match in the women’s league.

The Swans play their final home match of the regular season this weekend with an average attendance of 4,365 from four games so far – 124 more than the next
best club Adelaide – and a marquee pride match against Collingwood at Henson Park to come on Sunday where a crowd in excess of 4,000 is expected.

The club’s focus has been to target local communities and make match days compelling for fans on and off the field, according to executive GM of marketing at
the Swans, Loretta Johns.

“Our team has put a lot of work into sourcing some wonderful food offerings, entertainment and activities for kids, while we’ve had a local focus in promoting
our games,” she said.

Fans of Australian rules football in the Harbour City have a reputation for being fickle.
</item 92>

<item 93>
Who’s the next Gawn, Grundy or Dusty? AFL’s talent guru names top 40 draft prospects and compares them to current stars (The Roar | Sports Writers Blog,
2023-11-15T23:48:27+00:00)
Weapon: Marking on the lead and goalkicking accuracy

AFL player comparison: Bayley Fritsch (Melbourne)

Ollie Murphy

State: Victoria Metro

State League Club: Sandringham Dragons

Community Club: East Brighton Vampires

Date of Birth: 16/09/05

Height: 200cm

A lightly built tall defender who sets up the play with his neat left-foot kicking. Although new to playing in defence this season he adapted remarkably well.
Shows real composure with the ball in hand and is strong overhead. Is also a capable decision maker on when to mark or spoil in one-on-one contests. Was
outstanding for Victoria Metro in the National Championships and won his team’s MVP award along with All Australian honours after averaging 11.2 disposals and
4.5 marks per match.
</item 93>

<item 94>
Comeback kings Richmond take the AFLW points in Cairns (The Age, 2023-10-20T23:51:34+00:00)
“Tackled above waist and below shoulders. No sling or double motion. Sheppard lands on her back. ′ ‘Girls’ play hard too,” Harford said on X, the social media
site formerly known as Twitter. This echoed Collingwood defender and Fox Footy commentator Ruby Schleicher’s thoughts at the time, who called it a “monster
tackle” but believed it was “fair”. “I don’t see any dirtiness in that,” Schleicher said on Fox Footy. “It was pretty heavy on her head, wasn’t it? I look at
that – I still see a fair tackle. There’s no doubt the match review [officer] will look at it because they have to I guess. The duty of care for the player.”

The decision a week comes after Adelaide premiership defender Najwa Allen was handed a three-ban suspension, the longest ban in AFLW history, for a high bump on
Western Bulldog Kirsten McLeod. The Crows elected not to challenge the decision.
</item 94>

<item 95>
McRae reveals hidden message he carried under collar of his grand final shirt (The Sydney Morning Herald, 2023-09-30T22:45:39+00:00)
Brisbane Lions (2001-03), Geelong (2007, 2009, 2011), Hawthorn (2013-15) and Richmond (2017, 2019-20) all won three flags in short succession this century, with
the Magpies’ challenge now to back up their memorable season. Brayden Maynard enjoys the Pies’ win. Credit: Eddie Jim Maynard initially baulked when asked
whether Collingwood could win back-to-back flags, but said their bulletproof culture instilled belief they could “do anything”. “We’ve got to lap [this
premiership] up, and we’re going to have some serious fun, and we might burn down Melbourne,” a still-disbelieving Maynard said.

“I’m looking forward to celebrating with the team because it’s taken two years of getting after it and getting better every day – we’ve got better for this
moment. Loading “We just want to win.
</item 95>

<item 96>
Who’s the next Gawn, Grundy or Dusty? AFL’s talent guru names top 40 draft prospects and compares them to current stars (The Roar | Sports Writers Blog,
2023-11-15T23:48:27+00:00)
His ferocious attack on the ball and his opponent helped set the tone for the Allies to dominate the midfield battle throughout the National Championships and
he was rewarded with All Australian honours. Averaged 29.2 disposals, 11.2 contested possessions and a team-high 6.8 tackles across four matches for the Allies
and showed excellent repeat speed and courage as well as clean hands playing both on the inside and outside. His 6:06 effort in the 2km time trial at the
national Draft Combine showcased his elite endurance. He has been part of the Sydney Swans’ Academy since the under-10s.
</item 96>

<item 97>
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
</item 97>

<item 98>
Who’s the next Gawn, Grundy or Dusty? AFL’s talent guru names top 40 draft prospects and compares them to current stars (The Roar | Sports Writers Blog,
2023-11-15T23:48:27+00:00)
Weapon: Consistent high-level performances

AFL player comparison: Andrew Brayshaw (Fremantle)

Ari Schoenmaker

Pronunciation: Show-en-maker

State: Tasmania

State League Club: Tasmania Devils

Community Club: Launceston

Date of Birth: 07/01/05

Height: 194cm

Left-footed tall defender who reads the game exceptionally well, is a good interceptor and prodigious kick of the ball. Had a very impressive back half of the
year, which included being selected for the Allies in their final two matches of the National Championships. He also showed the ability to play on the wing and
averaged 27 disposals at 75 percent efficiency as well as 5.9 marks and 9.6 rebound 50s in his eight matches with the Tasmania Devils in the Coates Talent
League to be one of their most impressive performers.
</item 98>

<item 99>
Comeback kings Richmond take the AFLW points in Cairns (The Age, 2023-10-20T23:51:34+00:00)
AAP Pies hold on against comeback Cats

Sunday’s clash between Collingwood and Geelong at Victoria Park was less about who was the overall better side than it was decided by who could adapt best to
the blustery wind and rain. The answer: the Magpies. Collingwood survived a fourth quarter comeback from Geelong to finish with a six-point victory and stay
firmly in finals contention, whereas the Cats now face an uphill battle to finish in the eight in two weeks’ time. Jordyn Allen of the Magpies is tackled by
Geelong’s Claudia Gunjaca in the round-eight match at Victoria Park. Credit: Getty Geelong scored just one point in the first quarter and had just 1.3 on the
board at three-quarter-time, but kicked two goals to none in the final term to close in on Collingwood.
</item 99>

<item 100>
‘So used to seeing us winning’: Healy denies Aussie era ending after India loss, calls for more Tests (The Roar | Sports Writers Blog,
2023-12-24T22:28:25+00:00)
Three looming ODIs against India in the next week and three ensuing T20s will be the first showing of that, with the hosts also a rising force in the white-ball
game.

Advertisement

“Within our group we’re really comfortable with where we’re at, and the progression we’re on,” Healy said.

“Ultimately there’s a big T20 World Cup in Bangladesh towards the end of next year, that is a real carrot for us.

“These are the sort of big trophies that we’re wanting to win.

“And yes, every series in between now and then is important, but at the same time we’re building towards that World Cup.

“Maybe you can ask me that (supremacy) question at the end of the six white-ball games to see how we’re going and we can answer that.

“These next six games will probably give us a better indication about how we’re tracking but also what we might need to improve on.”
</item 100>
=== END ITEM id=3af467c5a7567a37 ===
