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

=== ITEM id=d9ae41fd341a06e5 ===
Gold sentences and facts:
Sentence 0: For the Swans, however, the victory has them challenging for a top-eight spot alongside the likes of Gold Coast, Geelong, Collingwood, St Kilda and Carlton.
  Fact 0.0: The victory has the Swans challenging for a top-eight spot.
  Fact 0.1: The Swans are challenging for a top-eight spot alongside the likes of Gold Coast.
  Fact 0.2: The Swans are challenging for a top-eight spot alongside the likes of Geelong.
  Fact 0.3: The Swans are challenging for a top-eight spot alongside the likes of Collingwood.
  Fact 0.4: The Swans are challenging for a top-eight spot alongside the likes of St Kilda.
  Fact 0.5: The Swans are challenging for a top-eight spot alongside the likes of Carlton.
Sentence 1: With only 0.4 of a percentage point separating Sydney in ninth and St Kilda in eighth, the win assured the Swans of retaking eighth spot.
  Fact 1.0: Sydney is in ninth place.
  Fact 1.1: St Kilda is in eighth place.
  Fact 1.2: Only 0.4 of a percentage point separates Sydney in ninth and St Kilda in eighth.
  Fact 1.3: The win assured the Swans of retaking eighth spot.
Sentence 2: “Our team has put a lot of work into sourcing some wonderful food offerings, entertainment and activities for kids, while we’ve had a local focus in promoting our games,” she said.
  Fact 2.0: The speaker said that her team has put a lot of work into sourcing some wonderful food offerings.
  Fact 2.1: The speaker said that her team has put a lot of work into sourcing entertainment for kids.
  Fact 2.2: The speaker said that her team has put a lot of work into sourcing activities for kids.
  Fact 2.3: The speaker said that her team has had a local focus in promoting its games.

Retrieved items:
<item 1>
Comeback kings Richmond take the AFLW points in Cairns (The Age, 2023-10-20T23:51:34+00:00)
Hanks was pivotal for the Demons and starred with 26 disposals, seven tackles and one goal, while her partner in the midfield Olivia Purcell had 28 touches,
five clearances and five tackles. For the Roos, Ash Riddell led the disposal count with 31, while skipper Emma Kearney was crucial in defence. Melbourne’s
Alyssa Bannan is tackled by Lulu Pullar of the Kangaroos. Credit: Getty Hanks said with the state of competition, they just had to take it week by week, but it
was good to get the win. “We knew we were up against the best tonight,” said Hanks. “It’s a nice feeling tonight.”

Lions roar in big battle Meanwhile, on Saturday afternoon, the Brisbane Lions put an end to Adelaide’s unbeaten run with a three-point victory at Brighton Homes
Arena in a top-four thriller, 5.12 (42) to 6.3 (39), in a spectacle worthy of being a final.
</item 1>

<item 2>
Craft beer and cheese boxes: Swans tap into something ‘uniquely Sydney’ to draw AFLW crowds | Jack Snape (The Guardian, 2023-10-26T23:42:29+00:00)
“Even at Henson Park [the Swans’ other home ground in Sydney’s gentrified inner west], they’ve gone to microbreweries in the local area and invited them in, but
also promoted through their clientele to come to the football.”

The Swans forward Bella Smith said the players were proud of the club’s attendance record, and were grateful for the support. “We get there two hours before the
start of the game and the hill [at Henson Park] is bare. When you come out for the first team warm-up, the hill is slowly filling and then by the time the first
bounce gets around, it is full,” she said. “You can hear it, the atmosphere is amazing.”

The average attendance for the AFLW is a little over 2,500 a match this year, tracking approximately on par with previous seasons and with a finals uptick
likely to come.
</item 2>

<item 3>
Comeback kings Richmond take the AFLW points in Cairns (The Age, 2023-10-20T23:51:34+00:00)
“I can name the whole team. Everyone had their second in the last couple minutes.

“That was such a team effort, everyone had a moment there when they had to put their head over the footy and lay a tackle and not let it slip.” Swanson (25
disposals, eight clearances) stood out for the Eagles, while Bombers workhorse Maddy Prespakis (30 disposals, nine clearances) was typically prominent in the
midfield alongside Georgia Nanscawen (21, six). With two rounds left, West Coast have a chance to build their winning streak when they meet the bottom of the
table Western Bulldogs next round. Slipping one place to sixth, Essendon have must-win matches against Carlton and Gold Coast on the road to the finals. The
Bombers will be without Steph Cain, who was sidelined with a concussion after an elbow to the head from Swanson.
</item 3>

<item 4>
Ground-breaking Gold Coast: Suns make AFLW finals for a second time (The Age, 2023-11-03T23:45:04+00:00)
“I’m excited like I think the group’s coming together really nicely and where really well-prepared compared to previous years,” Riddell told Fox Footy. “So, I
can’t wait I think it’s an exciting time for us.” The game was one-way traffic against the Dogs with the game played almost entirely in the Roos’ attacking
half. Riddell credited her side’s “team performance” after two losses to the Dees and Crows. North Melbourne have secured a top-four spot for finals. Credit:
Getty For the Western Bulldogs, it was another tough day out as they notched their ninth loss of the season. They won just one game this year, an eight-point
victory last weekend over 17th-placed West Coast.

Britney Gutknecht kicked the Dogs’ only goal against the Kangaroos, which came early in the final term to avoid a record for the lowest AFLW score in the
history of the competition.
</item 4>

<item 5>
Ground-breaking Gold Coast: Suns make AFLW finals for a second time (The Age, 2023-11-03T23:45:04+00:00)
Single filled the massive Rowbottom-size gap, collecting 25 disposals and six clearances in her prolific performance across the field while also playing a
crucial tagging role on Maddy Prespakis. Coast with the most: Georgia Clayden of the Suns celebrates after kicking a goal. Credit: AFL Photos / Getty Images The
former rugby sevens player was well-supported by Claudia Whitford (25 possessions) and Alison Drennan (22 touches). Prespakis was held to eight contested
possessions, well below a season average of 14 per game, while Georgia Nanscawen proved a potent threat with 13 contested possessions and three score
involvements.

The Bombers broke through first with a goal from Amber Clarke but young talent Darcie Davies (two goals) was able to respond after being gifted 50 metres from a
protected area infringement by Mia Bush. Daisy D’Arcy, Jacqueline Dupuy, Jamie Stanton and Davies attempted to break away from the low-scoring tug-of-war in the
second with goals of their own.
</item 5>

<item 6>
Ground-breaking Gold Coast: Suns make AFLW finals for a second time (The Age, 2023-11-03T23:45:04+00:00)
Lions duo Charlotte Mullins and Ruby Svarc celebrate. Credit: Getty Images The Lions got the game on their terms early with five repeat entries into the
attacking 50m in the opening 10 minutes of the game. Davidson and Ellie Hampson drew first blood before Casey Sherriff kicked Melbourne’s opening goal.

Tayla Harris, Tyla Hanks and Hore slotted a goal each in the second term to put their side ahead. Crumbling under the Brisbane attack, the visitors conceded
four unanswered goals from Taylor Smith, Sophie Conway (two goals) and Ruby Svarc while remaining goalless in the third term. Midfield stars Olivia Purcell (14
contested possessions) and Tyla Hanks (11) showed their mettle as they attempted to drag their side over the line, but the Demons were outclassed and sorely
missing Paxy Paxman (hamstring). Melbourne could be without Maeve Chaplin in their finals campaign, after the defender sustained a concussion injury in the
third term.
</item 6>

<item 7>
Ground-breaking Gold Coast: Suns make AFLW finals for a second time (The Age, 2023-11-03T23:45:04+00:00)
We’re stoked. It’s pretty special for the club.” Missing star ruck Ally Morphett to a season-ending wrist injury, Laura Gardiner stood the tallest in the
midfield, while Molloy (two goals), Montana Ham, Privitelli and Brooke Lochland hit the scoreboard. An off-season recruit from Geelong, 20-year-old Gardiner had
the ball on a string with a game-high 33 disposals and 18 contested possessions.

Demons denied the minor premiership The Brisbane Lions have toppled AFLW heavyweights Melbourne by 25 points to clinch a double chance in the finals and deny
the Demons the minor premiership. A year after Melbourne walked away with the 2022 premiership, the Lions avenged their upset grand final loss by defeating the
defending premiers 8.5 (53) to 4.4 (28) at Brighton Homes Arena on Saturday. Melbourne’s Lauren Peace fights for the ball againt the Lions.
</item 7>

<item 8>
Comeback kings Richmond take the AFLW points in Cairns (The Age, 2023-10-20T23:51:34+00:00)
It was the Western Bulldogs’ eighth loss in a row, and their biggest-ever defeat, and they have just two rounds left to clinch a win this season. They next play
17th-placed West Coast and then top-four side North Melbourne. For the Swans, however, the victory has them challenging for a top-eight spot alongside the likes
of Gold Coast, Geelong, Collingwood, St Kilda and Carlton.

The Swans, who went winless in their debut season last year, were propelled to victory by a Chloe Molloy masterclass. After a five-goal second term, which
included a four-minute burst of three majors, Sydney had already brought up their largest score in an AFLW game by halftime. Swan Chloe Molloy celebrates a
goal. Credit: AFL Photos / Getty Images The Bulldogs held their own in the last quarter, but the damage had already been done. The total was also the biggest
the Dogs have conceded in their history, topping the 83 Melbourne kicked against them in round three.
</item 8>

<item 9>
Comeback kings Richmond take the AFLW points in Cairns (The Age, 2023-10-20T23:51:34+00:00)
Conversely, the score was the most the Saints have conceded in any quarter this season.

Aine Tighe set the tone for the Dockers with a brilliant goal less than 30 seconds into the first quarter, followed by two majors from Gabby O’Sullivan and
another by Angelique Stannett. Ella Friend slotted a much-needed goal for the visitors to make it 4.2 (26) to 1.3 (9) at quarter-time. With less than a minute
till the long break, captain Hayley Miller (two goals) kicked the only major of the second quarter to give Fremantle a comfortable 26-point lead at half-time.
Despite two goals from Tyanna Smith and J’Noemi Anderson in the second half, the Saints struggled to eat into the deficit. Fremantle (4-4) face reigning
premiers Melbourne next, before finishing the season against 11th-placed Sydney.
</item 9>

<item 10>
Ground-breaking Gold Coast: Suns make AFLW finals for a second time (The Age, 2023-11-03T23:45:04+00:00)
But two goals from Amelia Radford and Georgia Gee kept the hosts close, while defender Ellyse Gamble (nine intercept possessions) proved a roadblock for the
Suns’ attackers. Single lit up the park with a clinical slice through the corridor, dodging two tackles to put the ball in the hands of Georgia Clayden, who
capped the centre clearance with the only major of the third term. With both sides goalless in the final quarter, the Suns were able to cruise home. With AAP
</item 10>

<item 11>
Comeback kings Richmond take the AFLW points in Cairns (The Age, 2023-10-20T23:51:34+00:00)
Geelong coach Daniel Lowther said the strong winds made it hard for his side to clear the ball out of their defensive 50-metre zone and to move the ball with
any certainty around the ground.

Loading “At one point the rain hit our box, it was flowing,” Lowther said. “Both teams had to fight in the conditions, but it didn’t help the game. “Our marking
contest wasn’t too flash, our ground ball representation was a bit hit and miss, so our defence were under a bit more pressure than they should have been, but
they fought back really well,” Lowther said. Collingwood next play a Sydney line-up including former Magpies star Chloe Molloy, while Geelong will take on
Richmond at Ikon Park.

Dockers spoil Saints’ finals hopes Fremantle recorded a comfortable 24-point win yesterday to bring an abrupt end to St Kilda’s four-game streak and dash their
AFLW finals hopes. J’Noemi Anderson is tackled by a group of Dockers.
</item 11>

<item 12>
Ground-breaking Gold Coast: Suns make AFLW finals for a second time (The Age, 2023-11-03T23:45:04+00:00)
AAP Cats secure elimination final home-ground advantage

Geelong have not only secured their spot in the top eight with a huge 40-point win over Hawthorn on Saturday at GMHBA Stadium but locked in home-ground
advantage for the first elimination final. The Cats finished sixth, with the 9.6 (60) to 3.2 (20) win over the Hawks boosting their percentage to a healthy
141.2, almost 40 ahead of Essendon, who sit one rung below them on equal points. Momentum is also in Geelong’s favour after two big wins in a row, having
demolished Richmond by 38 points last weekend. Chloe Scheer (left) and Julia Crockett-Grills of the Cats celebrate. Credit: Getty Against the Hawks, they
controlled ball movement from the get-go, kicking four goals straight to open the match and keeping their opposition goalless to half-time.
</item 12>

<item 13>
Comeback kings Richmond take the AFLW points in Cairns (The Age, 2023-10-20T23:51:34+00:00)
Bridie Hipwell kicked the only goal of the third quarter to give Hawthorn a game-high nine-point lead. Loading Richmond rebounded in the last term, with
Courtney Jones ending Hawthorn’s three-goal run with her first touch of the night, cutting the margin to three points. Dogged by inaccuracy in front of goal,
Yassir reduced the deficit to a single point before finally breaking through on her third attempt. That put the Tigers ahead by six points with 11 minutes left,
before Caitlin Greiser sealed the win with a massive goal from 45 metres.
</item 13>

<item 14>
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
</item 14>

<item 15>
Craft beer and cheese boxes: Swans tap into something ‘uniquely Sydney’ to draw AFLW crowds | Jack Snape (The Guardian, 2023-10-26T23:42:29+00:00)
But Livingstone said the AFL was happy with the level of competition. “Our ladder, from position five to 13 – there’s only a game separating it. It just shows
that this is a really competitive competition.”
</item 15>

<item 16>
Craft beer and cheese boxes: Swans tap into something ‘uniquely Sydney’ to draw AFLW crowds | Jack Snape (The Guardian, 2023-10-26T23:42:29+00:00)
But crowds are short of targets that could accelerate the extension of the regular season, which grows from 10 games this year to 12 games by 2025.

5,474 fans were at North Sydney Oval to watch the Swans’ first AFLW victory earlier in the season. Photograph: Jack Snape/The Guardian

The league enjoyed strong attendances in round one, which were partly attributed to the so-called Matildas effect, but some clubs have struggled to attract fans
in large numbers this season. Crowds of less than 1,500 have been recorded in both traditional markets such as Melbourne and Perth, as well as areas targeted
for growth like Blacktown and the Gold Coast.

The A-League Women launched this month with a first round highlighted by a crowd of more than 11,000 in Sydney.
</item 16>

<item 17>
Craft beer and cheese boxes: Swans tap into something ‘uniquely Sydney’ to draw AFLW crowds | Jack Snape (The Guardian, 2023-10-26T23:42:29+00:00)
Fans of Australian rules football in the Harbour City have a reputation for being fickle. After a winless campaign in 2022, the Swans – led by ex-Collingwood
player Chloe Molloy – have been competitive this season. The team won their first AFLW match in round one and are within a game of the top eight with two rounds
to play.

But general manager of women’s football at the AFL, Nicole Livingstone, said the Swans’ off-field offering had tapped into something “uniquely Sydney” and
helped make matches more attractive.

“To go to North Sydney Oval and to have quince paste and hand-picked grapes, cheddar cheese and biscuits in a little box that’s got this beautiful wrapping on
it – they’ve really thought about what they’re doing and made it an experience,” she said.
</item 17>

<item 18>
Ground-breaking Gold Coast: Suns make AFLW finals for a second time (The Age, 2023-11-03T23:45:04+00:00)
Chloe Scheer was once again the star in front of goals with three majors, while the Cats finished with seven individual goalkickers. Georgie Prespakis (35
touches, eight clearances) and Amy McDonald (23 touches, eight clearances) were a formidable midfield duo, while the retiring Renee Garing had a whopping 11
tackles. Overall, it was a strong team effort from a side that will be hungry for a finals win. Last year, the Cats had an impressive home-and-away run but
bowed out narrowly in an elimination final to North Melbourne. Geelong young defender Rebecca Webster told this masthead earlier in the week how important the
home crowd is for them come knockout finals. “Playing at GMHBA is like a massive advantage for us,” said Webster. “We love going in those four walls. We train
there three, four times a week.
</item 18>

<item 19>
Ground-breaking Gold Coast: Suns make AFLW finals for a second time (The Age, 2023-11-03T23:45:04+00:00)
“But I go out with on a happy note. I love these girls and I’ll be over at the pancake pocket [Pancake Parlour at Whitten Oval] watching them from now on and
have a few bevs on the sideline.” Roos’ lock in top-four spot; Dogs have issues to work through in off-season North Melbourne secured third spot on the ladder
and the double chance in finals after downing the Western Bulldogs by a whopping 46 points. They will head into finals with momentum but a glaring issue to work
on: kicking accuracy. Despite the comfortable win, they could have done a lot more damage after booting double the number of behinds to goals.

Star midfielder Ash Riddell, who had 36 touches and seven clearances, said she was excited to go up against Melbourne or Adelaide in the first weekend of finals
depending on other results this round, as they were the only two sides who beat them this year.
</item 19>

<item 20>
Comeback kings Richmond take the AFLW points in Cairns (The Age, 2023-10-20T23:51:34+00:00)
Molloy, the Swans’ star off-season recruit, turned on one of the best performances of her career to demolish their injury-hit hosts.

The former Collingwood ace was a class above as she kicked 3.4 from her 28 possessions and played a crucial role in setting up numerous other Sydney goals.
Loading “She [Molloy] was itching to get in there [in the midfield],” Swans coach Scott Gowans said. “We kept her forward deliberately. It was important Chloe
played a good block of time forward so we could have a look at it.” The Blues lost to Greater Western Sydney by two points on Saturday afternoon, denying them
the chance to leapfrog into the top eight, 8.1 (49) to 7.5 (47).

Narrm reclaim top spot Reigning premiers Narrm have reclaimed top spot on the ladder and their position as flag favourites after defeating third-placed North
Melbourne by 23 points at Ikon Park on Saturday night.
</item 20>

<item 21>
Ground-breaking Gold Coast: Suns make AFLW finals for a second time (The Age, 2023-11-03T23:45:04+00:00)
WEEK ONE FINALS FIXTURE SATURDAY NOVEMBER 11 First Qualifying Final: Adelaide v Brisbane Lions, 2.35pm local (3.05pm AEDT), Norwood Oval First Elimination
Final: Gold Coast v Sydney, 6.15pm local (7.15pm AEDT), Heritage Bank Stadium SUNDAY NOVEMBER 12 Second Elimination Final: Geelong v Essendon, 1.05pm AEDT,
GMHBA Stadium Second Qualifying Final: Melbourne v North Melbourne, 3.05pm AEDT, Ikon Park The eight was not decided until the final game of the regular season,
when Sydney downed Fremantle by 22 points to leapfrog St Kilda into the final spot, after the Saints had kept themselves in the race with a 20-point win over
Carlton earlier in the day, 7.4 (46) to 3.8 (26).
</item 21>

<item 22>
Ground-breaking Gold Coast: Suns make AFLW finals for a second time (The Age, 2023-11-03T23:45:04+00:00)
Down by four points at half-time in the low-scoring game, a third-term goal from Rebecca Privitelli sparked the Swans to a 5.10 (40) to 2.6 (18) win on Sunday
at Fremantle Oval.

The second-year expansion side stormed home with three unanswered goals to secure eighth spot. It’s a dramatic improvement, with the Swans winless in their
first season. “Any club that’s able to go from winless one year to a finals berth, it’s incredible,” Sydney co-captain Chloe Molloy told Fox Footy after the
match. “You can see the emotion on the girls’ faces. Coming into the season, the odds were against us. Just to get into the eight, we’ve managed to do that.
We’re stoked. It’s pretty special for the club.”
</item 22>

<item 23>
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
</item 23>

<item 24>
Comeback kings Richmond take the AFLW points in Cairns (The Age, 2023-10-20T23:51:34+00:00)
J’Noemi Anderson is tackled by a group of Dockers. Credit: AFL Photos The Dockers dominated the Saints from the start to win 6.9 (45) to 3.3 (21) at Fremantle
Oval. Having been on the verge of making their first finals appearance, St Kilda (4-4) could now miss the top eight even if they beat the fourth-placed Brisbane
Lions and the ninth-placed Carlton.

Dana East was everywhere for Fremantle, leading the game in disposals (19) and contested possessions (11), alongside Orlagh Lally (19 disposals, five
clearances). Jamiee Lambert was the shining light for the otherwise subdued Saints, collecting a game-high seven clearances and 13 contested possessions. Docker
Hayley Miller hunts a loose ball against St Kilda. Credit: Getty Fremantle flew out of the gates from the first siren, producing their highest-scoring quarter
of the season with 26 points in the opening term.
</item 24>

<item 25>
Ground-breaking Gold Coast: Suns make AFLW finals for a second time (The Age, 2023-11-03T23:45:04+00:00)
The Dogs have plenty to work on for next season as they look to regain a competitive edge, which was lost after a finals berth last year. Along with being in a
rebuild, they were badly injury-hit this season and used 32 players, more than any other club. Suns secure finals spot AAP Unsung hero Lucy Single has helped
Gold Coast book their second AFLW finals appearance with a 14-point win over Essendon. Losing midfield powerhouse Charlie Rowbottom to an ankle injury proved no
obstacle in the Suns’ maiden clash with the Bombers, as they finished the regular season in fifth spot after a 6.3 (39) to 3.7 (25) victory to kick off pride
round at Great Barrier Reef Arena on Friday.

While their men’s side have yet to make a finals appearance, this is the second time Gold Coast’s women have clinched a play-off spot, having also done so in
the cancelled 2020 season.
</item 25>

<item 26>
Craft beer and cheese boxes: Swans tap into something ‘uniquely Sydney’ to draw AFLW crowds | Jack Snape (The Guardian, 2023-10-26T23:42:29+00:00)
But more modest attendances in subsequent games, including two sub-2,000 crowds in Perth, suggest the popularity of the Matildas’ run to the semi-finals has
only had a limited impact on support for domestic women’s sport.

skip past newsletter promotion Sign up to Australia Sport Free newsletter Get a daily roundup of the latest sports news, features and comment from our
Australian sports desk Privacy Notice: Newsletters may contain info about charities, online ads, and content funded by outside parties. For more information see
our Newsletters may contain info about charities, online ads, and content funded by outside parties. For more information see our Privacy Policy . We use Google
reCaptcha to protect our website and the Google Privacy Policy and Terms of Service apply. after newsletter promotion

Livingstone said the “green and gold fever” associated with the Women’s World Cup was a one-off and the AFL has plans in place for long-term growth.
</item 26>

<item 27>
Comeback kings Richmond take the AFLW points in Cairns (The Age, 2023-10-20T23:51:34+00:00)
If the Lions had lost, they were at risk of falling out of the top four with second-season side Essendon chasing their spot. Last season’s best and fairest Ally
Anderson said her side loved playing at its training ground in Springfield, which it has made a fortress, and work well in hot conditions . “It was really hot.
It was a grind… But yeah, it was a good finish,” Anderson told Fox Footy post-game. “Every week that we play here, every time that we play here, we get better
and better. So, it was just a matter of time before we finally got the first win and then it hopefully [will] open the floodgates a bit.”

The Lions next play St Kilda before taking on the reigning premiers Melbourne in the final round. After this weekend, they have staked their claim as being
premiership contenders again. Adelaide will play top-four side North Melbourne next weekend before 18th-placed Western Bulldogs.
</item 27>

<item 28>
Ground-breaking Gold Coast: Suns make AFLW finals for a second time (The Age, 2023-11-03T23:45:04+00:00)
With only 0.4 of a percentage point separating Sydney in ninth and St Kilda in eighth, the win assured the Swans of retaking eighth spot. If the Saints had
qualified, it would have been their maiden finals appearance. Saints goalkicker Jesse Wardlaw had a heavy collision with Carlton’s Keeley Sherar late in the
final quarter of their clash.

Georgia Patrikios showed her class in the do-or-die match, collecting 25 disposals and eight contested possessions in a powerhouse display, while Hannah Priest
was a brick wall in defence, with nine intercept possessions. Abbie McKay worked overtime to counter Patrikios in the midfield, but her side was outclassed
across the park. McKay racked up 28 disposals and 12 clearances alongside Mimi Hill (30 disposals), while defender Sherar (17) toiled against a potent Saints
attack. Meanwhile, Sydney prevailed in a fierce tug-of-war with the Dockers to seal their first AFLW finals appearance.
</item 28>

<item 29>
Ground-breaking Gold Coast: Suns make AFLW finals for a second time (The Age, 2023-11-03T23:45:04+00:00)
Melbourne’s Lauren Peace fights for the ball againt the Lions. Credit: Getty Images The Lions were at risk of slipping from the top four after last week’s
21-point loss to St Kilda, but will now finish the home-and-away season fourth, avoiding a sudden-death elimination final and marking coach Craig Starcevich’s
75th game in style.

Meanwhile, Melbourne ceded the minor premiership to Adelaide and dropped to second spot. The Lions put on a show across the park, with powerhouse duo Ally
Anderson (22 disposals) and Jade Ellenger (22) going to work in the midfield while Dakota Davidson (two goals) led the charge in the forward line. Captain Bre
Koenen was superb in defence as she played on Melbourne’s leader Kate Hore, collecting a game-high 12 intercept possessions and limiting the prolific goalkicker
to just one goal. Lions duo Charlotte Mullins and Ruby Svarc celebrate.
</item 29>

<item 30>
Comeback kings Richmond take the AFLW points in Cairns (The Age, 2023-10-20T23:51:34+00:00)
The Lions’ Orla O’Dwyer speeds away against the Crows. Credit: Getty Images Adelaide started fast but the game evened up as the Lions had the better of the
second quarter with three goals to one. The Crows then took back control in the third with three goals to none, which set the stage for a tense final term. The
fourth quarter was a heated arm wrestle, and scores were tied with four minutes left on the clock. The Lions’ Bella Dawes kicked the only goal of the last term,
but it was a series of behinds in the final minutes that won the game for them, including one from Lily Postlethwaite that broke the stalemate.

Adelaide’s Niamh Kelly navigates her way through traffic against brisbane. Credit: Getty The two sides have had intense battles in the competition’s history,
including two grand finals, and this was no different.
</item 30>
=== END ITEM id=d9ae41fd341a06e5 ===

=== ITEM id=73127a0972b58bfd ===
Gold sentences and facts:
Sentence 0: Johnny Cardoso had to pull out of the September camp due to an ankle injury, but he has returned to action with Brazilian club Internacional, indicating a return to fitness, and has been included on the squad.
  Fact 0.0: Johnny Cardoso had to pull out of the September camp due to an ankle injury.
  Fact 0.1: Johnny Cardoso has returned to action with Brazilian club Internacional.
  Fact 0.2: Johnny Cardoso's return to action indicates a return to fitness.
  Fact 0.3: Johnny Cardoso has been included on the squad.
Sentence 1: He and his fresh legs are back just in time as rookie sensation De'Von Achane hit Injured Reserve this week with a knee injury.
  Fact 1.0: Johnny Cardoso and his fresh legs are back just in time.
  Fact 1.1: Rookie sensation De'Von Achane hit Injured Reserve this week.
  Fact 1.2: De'Von Achane hit Injured Reserve with a knee injury.

Retrieved items:
<item 1>
USA squad for October friendlies 2023: USMNT roster for friendly matches vs Germany, Ghana (Sporting News, 2023-10-04T22:42:00+00:00)
With Berhalter having since confirmed the two have broken the ice and talked over the phone, they will now look to move forward without any more personal
conflict.

After the October roster release, Berhalter confirmed that the two spoke "weeks ago" on Zoom, with the U.S. head coach appreciative of Reyna's willingness to
move forward. He spoke that because Reyna has not yet played for his club this season, their goal will be to "get him minutes" so he can build his fitness and
return to Dortmund in the best position possible to begin his club campaign.

Berhalter stated that they see Reyna being able to play three different positions: central midfielder, attacking midfielder, and winger, with further
clarification being his skill-set fitting a "between the lines" style of wide play if he were deployed on the wing, as opposed to one glued to the touchline.
</item 1>

<item 2>
Canada vs Jamaica live score, updates, highlights, result from CONCACAF Nations League quarterfinal second leg (Sporting News, 2023-11-21T22:35:30+00:00)
Antonio was replaced by Salernitana striker Trivante Stewart, but he struggled and was himself withdrawn in the 81st minute. Demarai Gray could step in, but he
himself is struggling for fitness, and while he remains with the team he was not part of the first leg matchday squad. Another option is Colorado Switchbacks
forward Romario Williams, who scored 15 goals in 29 USL Championship matches this season.

Other injuries for Jamaica to contend with include Cory Burke and Amari'i Bell, who both miss out on this international window.

Jamaica projected starting lineup (4-3-3): Blake (GK) — Leigh, Bernard, Pinnock, Lembikisa — Latibeaudiere, D. Johnson, Bailey — Reid, Nicholson, R. Williams.
</item 2>

<item 3>
Gio Reyna looks good in start for USMNT vs. Germany, but those involved in the finish are punished in 3-1 defeat (Sporting News, 2023-10-14T23:41:07+00:00)
Sergino Dest allowed the ball to deflect off his foot and into dangerous territory at the top of the box, de la Torre became a spectator with a too-great view
and central defender Chris Richards missed two opportunities to shut down the threat. McKennie, central defender Tim Ream and right back Joe Scally all bore
some responsibility for the second.

“When you watch the goals back — I had the opportunity to do that in the coaches’ office — it’s really these fine moments, these split seconds where you need to
be well-positioned, not open up any gaps in your back line,” Berhalter said. “And that’s where it went wrong.”

With the attempt to transform Musah into some semblance of Adams, though, it almost was certain to go wrong. The USMNT conjured a more compelling attack because
of Reyna’s presence, but the ability to control the game went missing.
</item 3>

<item 4>
Gio Reyna looks good in start for USMNT vs. Germany, but those involved in the finish are punished in 3-1 defeat (Sporting News, 2023-10-14T23:41:07+00:00)
“I think he played a good game, showed his quality. I think everyone in the first half played well. So it was good to see him, good to have him on the field.”

It was not good to not have him on the field, because the collapse from a 1-1 halftime tie not only followed Reyna’s departure, but probably was connected. The
U.S. was able to threaten the opposing goal enough with Reyna on the field to cause the opposition to concern itself with defending more often than it
preferred. Replacement Luca de la Torre dulled the attack and did nothing to dissuade Germany’s rampage toward the American net.

There was plenty of blame to be shared on the goals by Niclas Fullkrug in the 58th minute and Jamal Musiala three minutes later.
</item 4>

<item 5>
Gio Reyna looks good in start for USMNT vs. Germany, but those involved in the finish are punished in 3-1 defeat (Sporting News, 2023-10-14T23:41:07+00:00)
The decision about how to replace Adams in the formation was complex — too much so to be resolved in one afternoon.

Reyna was exceptional throughout the first half: less influential than he might have been if given greater access to the ball but evincing a welcome taste for
covering ground to become helpful on defense or to retrieve the ball and launch an attack. Weston McKennie was his typical self, particularly when he could
connect with star winger Christian Pulisic along the left side.

Young Yunus Musah was deployed in the defensive midfield role, and it did not work. Whenever Germany were able to quell an American surge in the first half and
regain possession, they plowed with ease through through the center of the park. He is elite at maintaining possession when operating as a box-to-box
midfielder, and lately his club play, with Pulisic at AC Milan, has seen Musah in more of an attacking role.
</item 5>

<item 6>
Canada vs Jamaica live score, updates, highlights, result from CONCACAF Nations League quarterfinal second leg (Sporting News, 2023-11-21T22:35:30+00:00)
That would allow them to rotate Kamal Miller out, who picked up a yellow card in the first leg, keeping him safe for the semifinals.

In would step Alistair Johnston and Derek Cornelius, with Steven Vitoria retained alongside them.

The only injury absence for Canada at the moment is striker Lucas Cavallini, who withdrew from the squad prior to the two matches, replaced by Jacen
Russell-Rowe.

Canada projected starting lineup (3-5-2): Borjan (GK) — Cornelius, Vitoria, Johnston — Laryea, Eustaquio, Kone, Kaye, Davies — David, Larin.

Jamaica saw star striker Michail Antonio go down with a gutting ACL injury in the first leg, and he now begins a lengthy rehab process. West Ham are reportedly
furious about how it was handled, angered by both the conditions of the pitch in Jamaica and the fact he played another 10 minutes after sustaining the injury.
</item 6>

<item 7>
USA squad for October friendlies 2023: USMNT roster for friendly matches vs Germany, Ghana (Sporting News, 2023-10-04T22:42:00+00:00)
MORE: Breaking down Germany squad, tactics as Julian Nagelsmann debuts as manager

USA roster for October friendlies vs Germany, Ghana

The USMNT squad for the October friendlies was announced on Thursday, October 5. Gregg Berhalter has called in 23 players, which is one fewer than were brought
in for the friendlies against Uzbekistan and Oman last month.

Below is the official 23-man roster for the upcoming October friendlies against Germany and Ghana.

PSV midfielder Malik Tillman was forced to withdraw from the squad due to injury, with Alex Zendejas of Club America named to the roster in his place.

MORE: Tracking every goal scored by Christian Pulisic at new club AC Milan

Pos Name Club Caps Goals GK Ethan Horvath Nott. Forest (ENG) 9 0 GK Matt Turner Nott. Forest (ENG) 33 0 DEF Cameron Carter-Vickers Celtic (SCO) 12 0 DEF Sergino
Dest PSV Eindhoven
</item 7>

<item 8>
Gio Reyna looks good in start for USMNT vs. Germany, but those involved in the finish are punished in 3-1 defeat (Sporting News, 2023-10-14T23:41:07+00:00)
So many national team fans sided with Reyna during the controversy that developed as the team competed at last winter's World Cup. They got what they argued for
in this game, but not what they wished.
</item 8>

<item 9>
USA squad for October friendlies 2023: USMNT roster for friendly matches vs Germany, Ghana (Sporting News, 2023-10-04T22:42:00+00:00)
Also missing is Antonee Robinson, who Gregg Berhalter said is dealing with a sports hernia, and while he played through it last camp and recently for his club,
it's affecting him and therefore rest is the best option here.

Johnny Cardoso had to pull out of the September camp due to an ankle injury, but he has returned to action with Brazilian club Internacional, indicating a
return to fitness, and has been included on the squad.

With Adams injured, there has been some noise on social media calling for Heidenheim midfielder Lennard Maloney to be brought into the squad in October, and he
was given the call-up here. The 23-year-old was born in Berlin to an American father, appearing for the U.S. at the U-20 level in the past and stating his
desire to reach the USMNT senior squad.

MORE: All players in U.S.
</item 9>

<item 10>
USA squad for October friendlies 2023: USMNT roster for friendly matches vs Germany, Ghana (Sporting News, 2023-10-04T22:42:00+00:00)
MORE: All players in U.S. national team pool currently playing for a European club

Gio Reyna to make USMNT return under Gregg Berhalter

The ability of Berhalter and Reyna to work together moving forward has been a major talking point since the fallout of their saga after the 2022 World Cup.

Reyna's lack of playing time at the World Cup was exacerbated by the actions of his parents in the weeks that followed, which sparked a domestic violence
investigation of Berhalter before his eventual re-hire as national team head coach.

Now, having recovered from a leg fracture suffered in summer Nations League duty under an interim USMNT head coach, the 20-year-old Borussia Dortmund playmaker
returns to the international fold under Berhalter for the first time.

The USMNT head coach has been forthcoming about the situation, admitting upon his re-hire that the two have a "lot of work to do" but insisting time was needed
to broach the situation.
</item 10>

<item 11>
USA vs Germany player ratings: USMNT defensive errors blight positive performance by Pulisic, Weah (Sporting News, 2023-10-14T22:46:31+00:00)
CM: Gio Reyna — 7

In his first 45 minutes of international football under Gregg Berhalter since the World Cup fallout, Gio Reyna was his usual self, drawing the attention of
German defenders and proving visionary with his passing. He combined with Folarin Balogun on a number of occasions on the counter, which is a link-up that U.S.
fans have been begging for more of.

Reyna was only fit for 45 minutes of play here as he returns from a leg injury, but it's no coincidence that Germany truly secured midfield dominance when he
came off the field. A promising performance for the young playmaker who should head back to Borussia Dortmund after this international break with plenty of
confidence moving forward.

RW: Tim Weah — 8

Though Germany carved the U.S. defense to pieces, their attacking threat remained strong, and almost all the danger came from the wings.
</item 11>

<item 12>
USA squad for October friendlies 2023: USMNT roster for friendly matches vs Germany, Ghana (Sporting News, 2023-10-04T22:42:00+00:00)
0 MID Gio Reyna Borussia Dortmund (GER) 20 4 MID Malik Tillman PSV Eindhoven (NED) 6 0 FWD Brenden Aaronson Union Berlin (GER) 34 8 FWD Folarin Balogun Arsenal
(ENG) 4 2 FWD Ricardo Pepi PSV Eindhoven (NED) 18 9 FWD Christian Pulisic AC Milan (ITA) 62 26 FWD Tim Weah Juventus (ITA) 33 5 FWD Alex Zendejas Club America
(MEX) 7 1

Tyler Adams, Antonee Robinson injured for October camp

Gregg Berhalter largely has the entire first-choice player pool available for selection, with only a few injuries to work around.

There is one notable absence, as captain Tyler Adams remains sidelined with a frustrating long-term hamstring injury.
</item 12>

<item 13>
USA vs Germany player ratings: USMNT defensive errors blight positive performance by Pulisic, Weah (Sporting News, 2023-10-14T22:46:31+00:00)
The Celtic defender had 22 touches and gave away possession just two times, completing two dribbles and making two clearances. Carter-Vickers was not a target
for Germany to attack nearly as much as the other three defenders on the pitch, a strong nod to his abilities.

65th min: Brenden Aaronson — 7

Probably the the U.S. substitute to make the biggest impact, Brenden Aaronson was bright in the attack, dazzling with his dribbling skills and helping usher the
U.S. forward amidst a second-half onslaught from Germany. A good showing from the Union Berlin winger who needs confidence at the moment.

65th min: Ricardo Pepi — 5

It wasn't an ideal situation for Ricardo Pepi to enter in, coming on with the U.S. shell-shocked from the quickfire Germany double. Still, he provided very
little up front, with just eight touches in 24 minutes of action.
</item 13>

<item 14>
Gio Reyna looks good in start for USMNT vs. Germany, but those involved in the finish are punished in 3-1 defeat (Sporting News, 2023-10-14T23:41:07+00:00)
are likely to play between now and the 2026 World Cup here in North America, and so it was viewed by coach Gregg Berhalter as a significant opportunity to learn
about his team. And what was learned, above all, was there needs to be an alternative approach to the midfield for whenever Tyler Adams is unavailable to play.

Adams is the one proven defensive midfielder — a No. 6, as they call it in soccer jargon — on the USMNT roster. He has been absent from the national team since
injuring his hamstring in the spring, and it’s possible now he’ll need a second surgery to address the issue.

The decision about how to replace him in the lineup was rather simple. It always was the plan for Reyna to play roughly half this game, and the obvious choice
in that circumstance, given the sometimes ridiculous controversy that developed from Reyna’s reserve role at the World Cup, was to have him start.
</item 14>

<item 15>
USA vs Germany player ratings: USMNT defensive errors blight positive performance by Pulisic, Weah (Sporting News, 2023-10-14T22:46:31+00:00)
Dest was lively up front as he combined with Christian Pulisic and worked very well on the ball. He got Mats Hummels booked on a counter-attack and delivered a
few very quality crosses.

Dest was also guilty of some of the game's worst defensive errors. He was completely non-existent as Leroy Sane blazed by him on the first Germany goal, and he
nearly got his team in trouble again right before halftime by holding onto the ball too long after making an interception rather than just clearing it out of
danger. Filling in for Antonee Robinson at left-back rather than his natural right, Germany targeted him on multiple occasions, to great profit.

He had a few good moments guarding Leroy Sane in the second half, but he was a top culprit for Germany's second goal by playing Niklas Fullkrug onside. If the
U.S.
</item 15>

<item 16>
USA vs Germany player ratings: USMNT defensive errors blight positive performance by Pulisic, Weah (Sporting News, 2023-10-14T22:46:31+00:00)
He featured his usual pressing, and came close to scoring once on a messy chance that Germany just barely managed to clear, but otherwise he didn't quite cause
the opposing back line problems to open spaces up for teammates.

75th min: Johnny Cardoso — 6

Johnny Cardoso only had 15 minutes at the end to showcase his abilities, but it begged the question why he or Lennard Maloney weren't the ones introduced at
halftime instead of Luca de la Torre. Cardoso lost possession just once in his 10 touches and completed a tackle, although he was also dribbled past a few times
as well.

75th min: Kevin Paredes — 5

With 10 touches in his 15 minutes of action towards the end of a mucked up game of numerous substitutions, Kevin Paredes didn't stand out, losing possession
four times and completing four passes. Tough to evaluate his time on the field, but he didn't stand out.
</item 16>

<item 17>
USA squad for October friendlies 2023: USMNT roster for friendly matches vs Germany, Ghana (Sporting News, 2023-10-04T22:42:00+00:00)
He was initially injured while playing for Leeds United back in March, and while he had finally recovered to make his Bournemouth debut in the EFL Cup in late
September, he played just 20 minutes off the bench before suffering a setback, sidelined again for the foreseeable future.

Gio Reyna has returned from his leg fracture and is called in for the first time under Berhalter since their falling out at the 2022 World Cup. Also included is
defender Cameron Carter-Vickers who only just made his debut for Celtic this season thanks to a hamstring injury. On the heels of that return just before the
international break, he is called in.

Goalkeepers Zack Steffen and Sean Johnson remain out of action due to injuries. Additionally, Josh Sargent is sidelined with a long-term ankle injury, while
Daryl Dike is also out after tearing his Achilles.
</item 17>

<item 18>
USA vs Germany player ratings: USMNT defensive errors blight positive performance by Pulisic, Weah (Sporting News, 2023-10-14T22:46:31+00:00)
If the U.S. had profited more from his attacking presence the narrative around his performance might be different, but with the story of the match focused on
the defensive errors, Dest is a guilty party.

CM: Yunus Musah — 6

Another player who did well in possession but struggled defensively, Yunus Musah dazzled at times on the ball but was beaten on the first Germany goal by
whiffing with his slide.

Musah's nutmeg of Florian Wirtz was a highlight moment of the match, and he was active on both ends of the pitch. Yet a few bad giveaways or defensive moments
betrayed his usual stoutness in the middle of the pitch. Overall, the mistakes didn't take away from his performance quite as much as Dest's, but they were
still concerning.

CM: Weston McKennie — 4

Other than a pair of critical blocks right before halftime, Weston McKennie was a mess through the first 45 minutes.
</item 18>

<item 19>
Gio Reyna looks good in start for USMNT vs. Germany, but those involved in the finish are punished in 3-1 defeat (Sporting News, 2023-10-14T23:41:07+00:00)
Even if one can argue he has the skill set to succeed at the No. 6, he did not appear comfortable transitioning on short notice.

MORE: USA vs. Germany player ratings

“Tyler’s a great player. It’s not nice to talk about guys that aren't here, though,” Berhalter told reporters afterward. “I’d rather talk about the guys that
are on the field.”

He didn’t, though, relative to how this group filled Adams’ huge vacancy. He was pleased to talk about the pressure the USMNT applied in the first half,
including a ball played over the top of Germany’s defense to Pulisic early in the game that was stopped by an offside call both player and coach disputed, and
Pulisic’s brilliant goal in the 27th minute that involved him driving to his right and blasting a curving shot while four defenders converged and ace goalkeeper
Marc-Andre Ter Stegen watched it sail into the upper righthand corner.
</item 19>

<item 20>
USA squad for October friendlies 2023: USMNT roster for friendly matches vs Germany, Ghana (Sporting News, 2023-10-04T22:42:00+00:00)
Cameron Carter-Vickers Celtic (SCO) 12 0 DEF Sergino Dest PSV Eindhoven (NED) 28 2 DEF DeJuan Jones New England Revolution (USA) 7 0 DEF Kristoffer Lund Palermo
(ITA) 2 0 DEF Kevin Paredes Wolfsburg (GER) 1 0 DEF Tim Ream Fulham (ENG) 52 1 DEF Chris Richards Crystal Palace (ENG) 12 1 DEF Miles Robinson Atlanta United
(USA) 27 3 DEF Joe Scally Gladbach (GER) 6 0 MID Johnny Cardoso Internacional (BRA) 7 0 MID Luca de la Torre Celta Vigo (SPA) 17 0 MID Lennard Maloney
Heidenheim (GER) 0 0 MID Weston McKennie Juventus (ITA) 46 11 MID Yunus Musah AC Milan (ITA) 29 0 MID Gio Reyna Borussia Dortmund (GER) 20 4 MID Malik Tillman
</item 20>
=== END ITEM id=73127a0972b58bfd ===

=== ITEM id=7842a656d04689d2 ===
Gold sentences and facts:
Sentence 0: In the year since Altman catapulted ChatGPT to global fame, he has become Silicon Valley’s sought-after voice on the promise and potential dangers of artificial intelligence and his sudden and mostly unexplained exit brought uncertainty to the industry’s future.
  Fact 0.0: In the year since Altman catapulted ChatGPT to global fame, he has become Silicon Valley's sought-after voice on the promise and potential dangers of artificial intelligence.
  Fact 0.1: Altman had a sudden and mostly unexplained exit.
  Fact 0.2: Altman's exit brought uncertainty to the industry's future.
Sentence 1: This week, we cover Sam Altman backing a teen’s AI startup, Google’s hardware event (and first impressions of the Pixel 8 Pro), Flexport drama, and the ongoing FTX fallout.
  Fact 1.0: This week's coverage includes Sam Altman backing a teen's AI startup.
  Fact 1.1: This week's coverage includes Google's hardware event.
  Fact 1.2: This week's coverage includes first impressions of the Pixel 8 Pro.
  Fact 1.3: This week's coverage includes Flexport drama.
  Fact 1.4: This week's coverage includes the ongoing FTX fallout.
Sentence 2: Based on the board’s language and the way these giant tech companies work, this is the prevailing theory floating around right now.
  Fact 2.0: The prevailing theory floating around right now is based on the board's language.
  Fact 2.1: The prevailing theory is based on the way these giant tech companies work.

Retrieved items:
<item 1>
Sam Altman ousted as OpenAI’s CEO (TechCrunch, 2023-11-17T20:41:13+00:00)
Couldn't be a prouder brother. — Jack Altman (@jaltma) November 17, 2023

It’s unclear what missteps Altman might’ve made helming OpenAI… at present. But they evidently relate to his relationship with OpenAI’s rather unusual board
makeup and corporate governance structure — and perhaps OpenAI’s active talks to raise substantial new capital.

As a recent piece in VentureBeat explored, OpenAI’s for-profit subsidiary, OpenAI Global, LLC, is fully controlled by OpenAI’s nonprofit. While the for-profit
subsidiary is permitted to commercialize its tech, it’s subject to the nonprofit’s mission: attaining artificial general intelligence (AGI), or AI that can — as
OpenAI defines it — “outperform humans at most economically valuable work.”

The nonprofit OpenAI’s board has the power to both determine when the company has achieved AGI and exclude this AGI from IP licenses and other commercial terms,
including with Microsoft, one of OpenAI’s biggest investors and a steadfast integrator of OpenAI’s various technologies.
</item 1>

<item 2>
OpenAI's ex-chairman accuses board of going rogue in firing Altman: 'Sam and I are shocked and saddened by what the board did' (Fortune,
2023-11-18T15:33:09+00:00)
He’s sat with multiple heads of state to discuss AI’s potential and perils. Just Thursday, he took part in a CEO summit at the Asia-Pacific Economic Cooperation
conference in San Francisco, where OpenAI is based.

He predicted AI will prove to be “the greatest leap forward of any of the big technological revolutions we’ve had so far.” He also acknowledged the need for
guardrails, calling attention to the existential dangers future AI could pose.

Some computer scientists have criticized that focus on far-off risks as distracting from the real-world limitations and harms of current AI products. The U.S.
Federal Trade Commission has launched an investigation into whether OpenAI violated consumer protection laws by scraping public data and publishing false
information through its chatbot.
</item 2>

<item 3>
A timeline of Sam Altman’s firing from OpenAI — and the fallout (TechCrunch, 2023-11-18T20:16:46+00:00)
That’s according to Bloomberg, which in a brief — citing sources close to the matter — said that discussions are happening between Quora CEO Adam D’Angelo, one
current member of the OpenAI board, and Altman — and possibly other board members as well.

Board tensions boil over

The New York Times reports that, before his ousting, Sam Altman made a move to push out board member Helen Toner because he thought a paper she had co-written
was overly critical of OpenAI. That, among other issues, led to OpenAI’s current predicament. Speaking of, The Times indicates that negotiations to hire Altman
back continue — but that one major sticking point remaining is “guardrails” meant to improve Altman’s communication with the board.

—

November 20

Altman joins Microsoft

Sam Altman, Greg Brockman and colleagues announce that they’ll join Microsoft to lead a new AI research team.
</item 3>

<item 4>
A timeline of Sam Altman’s firing from OpenAI — and the fallout (TechCrunch, 2023-11-18T20:16:46+00:00)
one takeaway: go tell your friends how great you think they are. — Sam Altman (@sama) November 18, 2023

Investors pushing for Altman’s return

Investors — furious at the turn of events — are reportedly exerting pressure on OpenAI’s board to reinstate Altman, going so far as to recruit Microsoft.
Nadella is said to be sympathetic.

Board agrees to reverse course — in principle

The Verge reports that the board agreed in principle to resign and to allow Altman and Brockman to return. It waffled, however, missing a deadline yesterday by
which many OpenAI staffers were set to leave the company. Altman is said to be ambivalent about coming back and asking for “significant” governance changes.

—

November 17

Brockman demoted

Brockman says he got a text from Sutskever shortly after noon on Friday asking for a quick call.
</item 4>

<item 5>
WTF is going on at OpenAI? We have theories (TechCrunch, 2023-11-18T00:09:53+00:00)
Any significant snafu in that department would reflect on her, as well as Altman.

5. Perhaps a difference of AI ethics or philosophy?

Altman is a proud techno-optimist, and often speaks fondly of the possibilities of AGI, or artificial general intelligence, a theoretical software system that
achieves human-like intellect and versatility.

The board’s statement pointedly includes that “OpenAI was deliberately structured to advance our mission: to ensure that artificial general intelligence
benefits all humanity” and that new leadership was necessary. It’s possible that Sam’s zeal for AGI, even absent a secret project or agreement, led to a major
rift between him and the board.

It’s been obvious to all that Altman took the company in a much more corporate direction from its origins, changing its legal status and aggressively pursuing
enterprise and consumer applications. That doesn’t sound a lot like the “mission” the board wants to advance.
</item 5>

<item 6>
OpenAI's ex-chairman accuses board of going rogue in firing Altman: 'Sam and I are shocked and saddened by what the board did' (Fortune,
2023-11-18T15:33:09+00:00)
He was recruited in 2014 to take lead of the startup incubator YCombinator.

“Sam is one of the smartest people I know, and understands startups better than perhaps anyone I know, including myself,” read YCombinator co-founder Paul
Graham’s 2014 announcement that Altman would become its president. Graham said at the time that Altman was “one of those rare people who manage to be both
fearsomely effective and yet fundamentally benevolent.”

OpenAI started out as a nonprofit when it launched with financial backing from Tesla CEO Elon Musk and others. Its stated aims were to “advance digital
intelligence in the way that is most likely to benefit humanity as a whole, unconstrained by a need to generate financial return.”

That changed in 2018 when it incorporated a for-profit business Open AI LP, and shifted nearly all its staff into the business, not long after releasing its
first generation of the GPT large language model for mimicking human writing.
</item 6>

<item 7>
WTF is going on at OpenAI? We have theories (TechCrunch, 2023-11-18T00:09:53+00:00)
In perhaps the most unexpected tech news of the year, billionaire and AI evangelist Sam Altman has been ejected from his CEO role at OpenAI by the company’s
board after an apparent vote of no confidence. Its exact wording in a release issued this afternoon: Altman’s “departure follows a deliberative review process
by the board, which concluded that he was not consistently candid in his communications with the board, hindering its ability to exercise its responsibilities.”

What the hell is happening at the most hyped company in the world?! Here are some totally speculative theories that occurred to us and others around the web.

1. Did Altman circumvent the board in a major deal?

Based on the board’s language and the way these giant tech companies work, this is the prevailing theory floating around right now. “Not consistently candid” is
a very diplomatic way of saying Altman lied.
</item 7>

<item 8>
A timeline of Sam Altman’s firing from OpenAI — and the fallout (TechCrunch, 2023-11-18T20:16:46+00:00)
Later Monday, that number climbed to over 650.

Altman and Brockman considering return

As reported by The Verge, Altman’s move to Microsoft isn’t a done deal — and both Altman and Brockman are still open to returning to OpenAI. That is, if the
remaining board members who initially fired him step aside.

OpenAI board considers merger

OpenAI’s board of directors approached Dario Amodei, the co-founder and CEO of rival large-language model developer Anthropic, about a potential merger of the
two companies, The Information reports. The approach was part of an effort by OpenAI to persuade Amodei to replace Altman as CEO — but Amodei quickly turned
down the CEO offer.

—

November 19

Altman to meet at OpenAI HQ

According to The Information, Altman is expected to meet at OpenAI’s San Francisco headquarters as executives at OpenAI push to have him reinstated as CEO.
</item 8>

<item 9>
A timeline of Sam Altman’s firing from OpenAI — and the fallout (TechCrunch, 2023-11-18T20:16:46+00:00)
In a dramatic turn of events late Friday, ex-Y Combinator president Sam Altman was fired as CEO of AI startup OpenAI, the company behind viral AI hits like
ChatGPT, GPT-4 and DALL-E 3, by OpenAI’s board of directors. Then, the company’s longtime president and co-founder, Greg Brockman, resigned — as did three
senior OpenAI researchers. And the fallout continues.

Tip TechCrunch

Do you work at OpenAI and know more about Sam Altman’s departure?

Do you work at OpenAI and know more about Sam Altman’s departure? Get in touch with TechCrunch

It’s a fast-moving situation that we’re still trying to get to the bottom of. No doubt more will become clear as time goes on. To make it easier to follow all
that’s happened in the meantime, though, we’ve put together a timeline; we’ll do our best to keep it current.
</item 9>

<item 10>
A timeline of Sam Altman’s firing from OpenAI — and the fallout (TechCrunch, 2023-11-18T20:16:46+00:00)
—

November 18

“Not … in response to malfeasance”

In an internal memo obtained by Axios sent Saturday morning, OpenAI COO Brad Lightcap said yesterday’s announcement “took [the management team] by surprise” and
that management had had “multiple conversations with the board to try to better understand the reasons and process behind their decision.” Discussions were
ongoing as of Saturday morning, per the memo.

“We can say definitively that the board’s decision was not made in response to malfeasance or anything related to our financial, business, safety, or
security/privacy practices,” Lightcap added. “This was a breakdown in communication between Sam and the board … We still share your concerns about how the
process has been handled, are working to resolve the situation, and will provide updates as we’re able.”

OpenAI’s funding in jeopardy

The planned sale of OpenAI employee shares that would value the startup at about $86 billion could be in jeopardy.
</item 10>

<item 11>
OpenAI's ex-chairman accuses board of going rogue in firing Altman: 'Sam and I are shocked and saddened by what the board did' (Fortune,
2023-11-18T15:33:09+00:00)
Altman also co-founded Worldcoin, a biometric and cryptocurrency project that’s been scanning people’s eyeballs with the goal of creating a vast digital
identity and financial network.

___

Associated Press business writers Haleluya Hadero in New York, Kelvin Chan in London and Michael Liedtke and David Hamilton in San Francisco contributed to this
report.
</item 11>

<item 12>
WTF is going on at OpenAI? We have theories (TechCrunch, 2023-11-18T00:09:53+00:00)
“Not consistently candid” is a very diplomatic way of saying Altman lied.

It’s possible that Altman — and potentially OpenAI President Greg Brockman, who stepped down as chairman simultaneously, then resigned — wanted to make a bold
move that he knew the board would not like. It’s not uncommon for these deals to be hammered out quietly in smoke- (or vape-) filled rooms and then presented as
a fait accompli, but if it was controversial enough and the board found out about these maneuvers, it could be fuel for an ouster.

But what kind of deal would be big and dangerous enough for a summary dismissal of the CEO and famous face of the company? The man was onstage two weeks ago; I
just talked with him! What could have happened since then?

Few would be shocked if Microsoft, which is deeply, deeply embedded in OpenAI as an investor and customer, is a factor here. Could Altman have been working with
— or against — OpenAI’s patron in secret?
</item 12>

<item 13>
A timeline of Sam Altman’s firing from OpenAI — and the fallout (TechCrunch, 2023-11-18T20:16:46+00:00)
After sending a Google Meet link, Brockman was told that he was being removed from the board as chairman “but was vital to the company and would retain his
role” as president, and that Altman had been fired.

Altman’s firing publicly announced

OpenAI published a post on its blog announcing the executive shake-up. The company’s management team was aware shortly after.

i loved my time at openai. it was transformative for me personally, and hopefully the world a little bit. most of all i loved working with such talented people.
will have more to say about what’s next later. 🫡 — Sam Altman (@sama) November 17, 2023

All-hands meeting

OpenAI held an all-hands meeting Friday afternoon during which Sutskever defended Altman’s ouster. He dismissed suggestions that pushing Altman out amounted to
a “hostile takeover,” and claimed that it was necessary to protect OpenAI’s mission of “making AI beneficial to humanity.”
</item 13>

<item 14>
OpenAI's ex-chairman accuses board of going rogue in firing Altman: 'Sam and I are shocked and saddened by what the board did' (Fortune,
2023-11-18T15:33:09+00:00)
Altman posted Friday on X: “i loved my time at openai. it was transformative for me personally, and hopefully the world a little bit. most of all i loved
working with such talented people. will have more to say about what’s next later.”

The Associated Press and OpenAI have a licensing and technology agreement allowing OpenAI access to part of the AP’s text archives.

Altman helped start OpenAI as a nonprofit research laboratory in 2015. But it was ChatGPT’s explosion into public consciousness that thrust Altman into the
spotlight as a face of generative AI — technology that can produce novel imagery, passages of text and other media. On a world tour this year, he was mobbed by
a crowd of adoring fans at an event in London.

He’s sat with multiple heads of state to discuss AI’s potential and perils.
</item 14>

<item 15>
A timeline of Sam Altman’s firing from OpenAI — and the fallout (TechCrunch, 2023-11-18T20:16:46+00:00)
We are collaborating to figure out the details. Thank you so much for your patience through this. — OpenAI (@OpenAI) November 22, 2023

Altman also posted about the new deal, giving some insight into the roller coaster that we’ve all been riding since his firing was revealed Friday. In his
words, his decision to join Microsoft on Sunday “was the best path for me and the team.” Since then, the new board’s composition and Microsoft’s support appear
to have been enough to bring him back to the AI company he co-founded.

Altman and board in talks

OpenAI’s board of directors is reportedly in talks with Sam Altman, ex-Y Combinator president and an OpenAI co-founder, to return to OpenAI as CEO as soon as
this week.
</item 15>

<item 16>
WTF is going on at OpenAI? We have theories (TechCrunch, 2023-11-18T00:09:53+00:00)
Some CEOs get a pass on things like having kids with direct reports, but not all.

Altman also has three siblings, and his younger sister Annie has publicly accused him of abuse. We have no way of evaluating these allegations, which involve
private matters.

Our read on the board’s language in dismissing Altman, however, is that it was not a legal or personal problem that provoked the action, but a professional or
business one.

We probably won’t know the full truth on this for a long time, as the characters in the drama are likely to be NDA’ed up. Per various whispers and leaks, an
all-hands meeting about the situation this afternoon didn’t produce any revelations beyond banal reassurances that the company is fine and they’ll get a fresh
CEO soon. Expect to hear a lot of rumors before we hear the real thing.
</item 16>

<item 17>
Sam Altman ousted as OpenAI’s CEO (TechCrunch, 2023-11-17T20:41:13+00:00)
Altman has a long history steering OpenAI. After co-founding the company alongside Peter Thiel, LinkedIn cofounder Reid Hoffman and others, Altman, who formerly
headed startup accelerator Y Combinator and also holds a seat on the board at Worldcoin, the ambitious cryptocurrency project, initially served as a co-chair of
OpenAI alongside Elon Musk. Musk left in 2018 to avoid a conflict of interest with Tesla.

Altman over the past several months played an active role in attempting to shape regulators’ responses to AI, appearing at U.S. congressional hearings and
meeting personally with world leaders including President Joe Biden, German Chancellor Olaf Scholz and French President Emmanuel Macron, among others, on an
international policy tour.

More important than being one of the most brilliant and impactful people our industry has ever had Sam is one of the most generous and caring people I know.
I've never met someone who has supported and lifted up more people around them than him. Couldn't be a prouder brother.
</item 17>

<item 18>
Sam Altman ousted as OpenAI’s CEO (TechCrunch, 2023-11-17T20:41:13+00:00)
Sam Altman has been fired from OpenAI, Inc., the 501(c)(3) nonprofit that acts as the governing body for OpenAI, the AI startup behind ChatGPT, DALL-E 3, GPT-4
and other highly capable generative AI systems. He’ll both leave the company’s board of directors and step down as CEO.

In a post on OpenAI’s official blog, the company writes that Altman’s departure follows a “deliberative review process by the board” that concluded that Altman
“wasn’t consistently candid in his communications” with other board members, “hindering its ability to exercise its responsibilities.”

Tip TechCrunch

Do you work at OpenAI and know more about Sam Altman’s departure?

Do you work at OpenAI and know more about Sam Altman’s departure? Get in touch with TechCrunch

“The board no longer has confidence in [Altman’s] ability to continue leading OpenAI,” the blog post reads.
</item 18>

<item 19>
A timeline of Sam Altman’s firing from OpenAI — and the fallout (TechCrunch, 2023-11-18T20:16:46+00:00)
The Information, speaking to three sources formerly with the company, reports that they no longer expect the sale — led by Thrive Capital — to happen, or, if it
does, to come with a lesser valuation because of the recent turn of events.

Altman planning new venture

Altman has been telling investors that he’s planning to launch a new venture, according to The Information. Brockman is expected to join the effort — whatever
form it takes. (Possibly an AI chip startup.)

i love you all. today was a weird experience in many ways. but one unexpected one is that it has been sorta like reading your own eulogy while you’re still
alive. the outpouring of love is awesome. one takeaway: go tell your friends how great you think they are.
</item 19>

<item 20>
A timeline of Sam Altman’s firing from OpenAI — and the fallout (TechCrunch, 2023-11-18T20:16:46+00:00)
Timeline of Sam Altman’s firing from OpenAI

November 29

Microsoft gains a board observer

Microsoft will gain representation on the new initial board of directors in the form of a non-voting observer, OpenAI announced. It wasn’t immediately clear who
this observer might be — only that they won’t have an official vote in board business.

November 21

Sam Altman, OpenAI reach agreement on return as CEO and ‘initial’ new board

In a sudden late announcement, OpenAI revealed that it and Altman “have reached an agreement in principle” for him to return as the company’s CEO. In addition
to Altman’s return, its new “initial” board will include former Salesforce chief executive Bret Taylor, former US Secretary of the Treasury Larry Summers and
Quora founder Adam D’Angelo.

We have reached an agreement in principle for Sam Altman to return to OpenAI as CEO with a new initial board of Bret Taylor (Chair), Larry Summers, and Adam
D'Angelo.
</item 20>
=== END ITEM id=7842a656d04689d2 ===

=== ITEM id=cf1e1b0dada149b3 ===
Gold sentences and facts:
Sentence 0: Depending on which one you get, it might come with USB ports so you can charge multiple devices in one outlet.
  Fact 0.0: Depending on which one you get, it might come with USB ports.
  Fact 0.1: The USB ports let you charge multiple devices in one outlet.
Sentence 1: With a 10,000-mAh capacity, WIRED contributor Simon Hill says it charged his iPhone 14 Pro once and then delivered an additional 64 percent the second time before tapping out.
  Fact 1.0: The device has a 10,000-mAh capacity.
  Fact 1.1: WIRED contributor Simon Hill says the device charged his iPhone 14 Pro once.
  Fact 1.2: Simon Hill says the device then delivered an additional 64 percent to his iPhone 14 Pro the second time.
  Fact 1.3: Simon Hill says the device tapped out before the second charge was finished.

Retrieved items:
<item 1>
41 Best Prime Day Apple Deals (2023) (Wired, 2023-10-10T22:58:00+00:00)
It fluctuates in price very often, but it's still a good deal.

Satechi's fold-up wireless charging stand has a 10,000-mAh capacity (complete with LEDs to show you how much juice you have left) and can wirelessly charge both
your phone and a wireless earbuds case (as long as it's Qi-certified). There's also a USB-C port to power up a third device. The only gripe we have is that it's
slow—delivering 10 watts of wireless charging power for phones (7.5 watts for iPhones), 5 watts for earbuds, and 10 watts via the USB-C port.

Apple 35-Watt Dual USB-C Port Compact Power Adapter Photograph: Apple

Apple's dual USB-C port power adapter is an essential accessory if you have a tendency to forget to pack chargers for your devices on the go. With two USB-C
ports, you can use it to recharge both your iPhone and MacBook simultaneously.
</item 1>

<item 2>
Some of our favorite devices are on sale for Black Friday. (The Verge, 2023-11-24T15:42:21+00:00)
Universal travel adapter

Victoria Song, senior reviewer

Epicka universal travel adapter $ 20 $ 25 20 % off $ 20 $ 20 $ 25 20 % off Epicka’s universal travel adapter is an all-in-one adapter that includes four
different plugs that cover over 150 countries. $20 at Amazon

In my youth, I forgot to pack plug adapters for international trips one too many times. Buying them once you’ve landed in another country isn’t always easy,
either. And if you’re like me, your relatives in rural Korea don’t always have more than one plug for your American devices — in which case, you’ll have to
share with your six other cousins. Nope. Absolutely not. Which is why I never leave this country without a universal travel adapter.

Basically, it’s six plug adaptors in one. Depending on which one you get, it might come with USB ports so you can charge multiple devices in one outlet.
</item 2>

<item 3>
These Are Our Favorite Walmart Cyber Monday Deals (Wired, 2023-11-27T14:22:26+00:00)
Remember when people would break down the doors to get inside Walmart to score the best deals? Isn't civilization nice? Fill up your digital shopping cart
instead with these Walmart Cyber Monday deals on some of the best devices we've tested this year—no door-breaking necessary.

We test products year-round and handpicked these deals. The discounts we show are based on actual street prices at retailers in the past few months. Products
that are sold out or no longer discounted as of publishing will be crossed out. We'll update this guide periodically.

If you buy something using links in our stories, we may earn a commission. This helps support our journalism. Learn more.

Top Deals

Jump to a Topic: PC and Gaming Deals, Home and Kitchen Deals, TV Deals, Watch and Fitness Tracker Deals, Headphone and Earbud Deals, Smart Home Deals, Speaker
Deals

PC and Gaming Deals

Need a new laptop? We test dozens throughout the year.
</item 3>

<item 4>
78 Absolute Best October Prime Day Deals (2023) (Wired, 2023-10-10T14:23:00+00:00)
The performance is fast, the recharging is fast. It even looks fast. It has a bright, 120-Hz AMOLED screen, great speakers, and surprisingly good battery life
for all that. Our main gripe is the IP64 water- and dust-resistance rating, which is not nearly as good as other flagship phones.

Ugreen's charger is, as the name suggests, a 145-watt charger with a 25,000-mAh battery. It's surprisingly compact for the power it provides, although at 1.1
pounds, it's definitely not ultralight. There are two USB-C ports and one USB-A port. What sets the Ugreen apart is that you can actually draw 145 watts while
charging. That works out to one USB-C port at 100W and the other at 45W. Very few other batteries we've tested are capable of that feat.

Photograph: Anker

The Anker Nano is one of our favorite portable power banks, particularly for phones.
</item 4>

<item 5>
78 Absolute Best October Prime Day Deals (2023) (Wired, 2023-10-10T14:23:00+00:00)
If you want to stay powered up on your travels, you need a travel adapter, and this one from Epicka is the budget pick in our Best Travel Adapters guide. It
works in more than 150 countries and has four USB-A ports on the bottom, plus a 15-watt USB-C port on the side. It has dropped this low before, but not for a
while.

Get not one, not two, but four of our favorite smart bulbs for the best price we've seen all year. These smart bulbs are easy to use, beautifully vibrant, comes
with a variety of present colors, and work with Amazon Alexa and Google Assistant.

Photograph: Nanoleaf

We love this light kit for adding fun ambient lighting to any room. Each hexagon is its own light, so you can control and customize the hexagons into just about
any design of your choosing—both in color schemes. and onto your wall. Just grab a level to make sure you put them on straight!
</item 5>

<item 6>
Some of our favorite devices are on sale for Black Friday. (The Verge, 2023-11-24T15:42:21+00:00)
(Or, “That’s pretty cheap, but maybe I’ll wait until the price goes down some more...”)

So here are some of our staff’s most-liked gear and gadgets, much of which is still discounted from Cyber Monday.

Tech tools

Electronics repair kit

Alex Cranz, managing editor

Tekton Everybit Tech Rescue Kit $ 22 $ 29 24 % off $ 22 $ 22 $ 29 24 % off A 46-piece screwdriver kit that can help you repair practically any tech device out
there. $22 at Amazon

I own at least two of these little Tekton Everybit Tech Rescue Kits, and I frequently buy them for friends and family, too. For an average price of $35, you get
a screwdriver with nearly every bit you’d need for most gadgets (including the weird ones for Apple products), a plastic and a metal spudger, tweezers, and a
suction cup. I’ve replaced batteries in iPhones with this kit.
</item 6>

<item 7>
The best Black Friday deals you can get for under $25 (The Verge, 2023-11-23T17:58:09+00:00)
$16 at Amazon

Dbrand leather phone skins $ 24 $ 35 31 % off $ 24 $ 24 $ 35 31 % off Dbrand’s leather skins add a little grip to the back of a slick phone, but they’re mostly
a style choice. They’re high quality and non-damaging. $24 at Dbrand

EWA’s MagOne magnetic phone grip is on sale for $15.99 ($4 off) at Amazon. It’s a grip and kickstand in one, making bigger phones a little more wieldy without
getting in the way of MagSafe charging.

is on sale for $15.99 ($4 off) at Amazon. It’s a grip and kickstand in one, making bigger phones a little more wieldy without getting in the way of MagSafe
charging. You can buy Uni’s 10-foot USB-C to USB-C cable for $7.99 ($12 off) at Amazon.
</item 7>

<item 8>
54 Best Apple Black Friday Deals (2023): iPad, Apple Watch, AirPods (Wired, 2023-11-25T00:00:29+00:00)
Flip over to the back, and you'll find two USB-A ports, two USB-C ports, and three AC outlets. It's a compact and tidy charger, and our unit has been going
strong for more than two years. It fluctuates in price very often, and it's hit as low as $60 in the past, but we still think this is a good deal.

This power bank is one of our favorites if you're looking for an upgrade. It has an absolutely gargantuan 24,000-mAh capacity, and it can be charged up in an
hour. It can output up to 140W, making it powerful enough to charge laptops and tablets, as well as charging your phone super quick.

This is a great charger if you travel often with an iPhone and Apple Watch. Instead of packing two separate chargers, the Duo offers the luxury of both all
within a compact, foldable form factor.
</item 8>

<item 9>
54 Best Apple Black Friday Deals (2023): iPad, Apple Watch, AirPods (Wired, 2023-11-25T00:00:29+00:00)
There's also a USB-C port to power up a third device. The only gripe we have is that it's slow—delivering 10 watts of wireless charging power for phones (7.5
watts for iPhones), 5 watts for earbuds, and 10 watts via the USB-C port.

This travel kit from ESR works well if you want something that can prop your iPhone in portrait or landscape orientation. It can also display your Apple Watch
in Nightstand mode, so you can peek and see how long before you must get out of bed. The main body folds open with a MagSafe charging pad for your iPhone and a
slot behind for your AirPods. Unfortunately, it doesn't support fast charging and it's capped at 7.5 watts and 5 watts for the iPhone and Apple Watch,
respectively. But it's a good, affordable charger. It dipped lower to $40 back in July, but this is still a solid deal.
</item 9>

<item 10>
The best Black Friday deals you can get for under $25 (The Verge, 2023-11-23T17:58:09+00:00)
The lengthy USB-C cable is capable of delivering 100 watts of power to beefy laptops like the 16-inch MacBook Pro, but you can also use it to power up
smartphones, tablets, and other gadgets.

for $7.99 ($12 off) at Amazon. The lengthy USB-C cable is capable of delivering 100 watts of power to beefy laptops like the 16-inch MacBook Pro, but you can
also use it to power up smartphones, tablets, and other gadgets. PopSockets is taking 25 percent off various grips, mounts, cases, and other accessories,
including this PopGrip for MagSafe phone handle / stand (on sale for $22.50) which is compatible with iPhone 12 phones and newer.

phone handle / stand (on sale for $22.50) which is compatible with iPhone 12 phones and newer. The worst thing about AirTags is you need something to stick them
in. Here’s a pack of four AirTag keychain holders for $8.91.
</item 10>

<item 11>
Some of our favorite devices are on sale for Black Friday. (The Verge, 2023-11-24T15:42:21+00:00)
Granted, it’s bulkier than buying one or two specialized adapter plugs, but if you’ve got a multi-continent itinerary, it’s a game-changer. What I like about
this one from Epicka is that it comes with a spare fuse in case things go sideways with voltage.

The only caveat is that, although it says “universal,” it’s technically only the four most common types of plugs. That’ll get you by in most countries, but it’s
not a guarantee in places like Brazil, South Africa, or India. Even so, I’ll take this over price gouging at airport electronics shops or having to take time
out of my schedule to visit a local hardware store.

Lovely lights

Brandon Widder, senior commerce editor

I’m a big fan of ambient lighting, even when I’m 50 miles from the nearest outlet.
</item 11>

<item 12>
78 Absolute Best October Prime Day Deals (2023) (Wired, 2023-10-10T14:23:00+00:00)
It clicks right into the bottom of your device, and even comes in fun colors. This model has a 12-watt Lightning connector but there is a 22.5-watt USB-C
version for the iPhone 15 or Android phones for $22 ($10 off). Both connectors fold away when you aren't using them. If you use a particularly thick case, this
may not work. You'll likely be able to charge most phones fully once before needing to recharge the power bank itself via the included USB-C cable.

We are big fans of the Backbone One (8/10, WIRED recommends) and it appears in our Best Mobile Game Controllers guide. It slides open to cradle your phone and
is very responsive, with buttons and bumpers that feel nice and clicky. The USB-C version that is on sale will work with most Android phones and the new iPhone
15 range.

This 2-in-1 charger is featured in our guide to the Best MagSafe Accessories. The price matches the lowest we have tracked.
</item 12>

<item 13>
41 Best Prime Day Apple Deals (2023) (Wired, 2023-10-10T22:58:00+00:00)
We recommend Satechi's 108-watt wall charger if you own a MacBook Pro. It comes with three USB-C ports, each with the ability to charge at 100 watts if only one
device is connected (making it perfect for the 16-inch MacBook Pro). But if you want to charge three gadgets simultaneously, it'll charge them at 45, 30, and 30
watts, respectively, 58, 30, and 20 watts, or 65, 20, and 20 watts, depending on what you're charging. It's also on the larger side, but it's not as big as the
one from Apple, and the plug folds up too.

Satechi 165-Watt GaN Four-Port Charger Photograph: Satechi

This handy charger has enough power to juice up two MacBook Pro models simultaneously. But it's also great if you have a lot of devices around your desk that
constantly need to be charged.
</item 13>

<item 14>
54 Best Apple Black Friday Deals (2023): iPad, Apple Watch, AirPods (Wired, 2023-11-25T00:00:29+00:00)
You can use it to charge your iPhone (at the fastest 15-watt charging speed Apple allows), fast-charge the Apple Watch Series 7/8/9/Ultra, and use the base to
power up your AirPods Pro (or AirPods with a wireless charging case) simultaneously all from a single outlet. It dipped as low as $100 back in August, but this
is still a great deal.

We like Anker's chargers and its Prime range employs GaN technology to shrink the size down. This 67-watt charger has fold-out prongs and offers two USB-C ports
and one USB-A. It supports fast charging for iPhones (in addition to Pixel phones and Samsung Galaxy phones) but can also charge a MacBook.

Photograph: Satechi

This handy charger has enough power to juice up two MacBook Pro models simultaneously. But it's also great if you have a lot of devices around your desk that
constantly need to be charged.
</item 14>

<item 15>
41 Best Prime Day Apple Deals (2023) (Wired, 2023-10-10T22:58:00+00:00)
Unfortunately, it maxes out at 7.5 watts for charging, but you can remove the USB-C charging cable when the iPhone is fully charged. There's also a built-in fan
to help keep the temperature down when it's hot out.

Anker Nano USB-C Power Bank Photograph: Anker

This is a great power bank if you don't want to worry about carrying a cable around, since it simply slots into the bottom of your phone—thanks to the built-in
rotating USB-C or Lightning connector (MFi certified). It also offers 5,000 mAh of energy, and there's a USB-C port for recharging it, complete with a short
cable. However, it might not be a good pick if you have a thick case.

This 2-in-1 charger is featured in our guide to the Best MagSafe Accessories. The price matches the lowest we have tracked. It doesn't have a built-in Apple
Watch charger, but it can top off your compatible wireless earbuds whilst simultaneously charging your iPhone.
</item 15>

<item 16>
78 Absolute Best October Prime Day Deals (2023) (Wired, 2023-10-10T14:23:00+00:00)
How to find the best Prime Big Deal Days deals?

WIRED is the only publication posting about Amazon Prime Day. Wait, sorry, looks a handful of other sites are, too. But you should only read WIRED, because we
actually vet every deal and compare it to the actual street price instead of an absurd MSRP that you will never observe in the wild. Also, we review the
products we recommend and don't just send you Prime Day deals on junk that will break. It's only a deal if you need it and it's good!

Are other retailers running sales?

Amazon Prime Day has inspired many imitators and you'll find sales from competitors like Best Buy, Target, and the like. Some of these sales are great, and
we'll flag when they are, but others are worth holding off on until Black Friday and Cyber Monday, when WIRED will again be the Internet's only website with
deals posts.

Retailer Sale Pages
</item 16>

<item 17>
The 53 best Black Friday deals we could find at Amazon, Walmart, Target and more (Engadget, 2023-11-19T15:13:23+00:00)
In the consumer electronics space, we expect most companies to have some sort of Black Friday promotion, so that means you will likely be able to get big-ticket
items from brands like Apple, Samsung, Google, Sony, LG, Sonos and others for much cheaper prices.

Your Black Friday Shopping Guide: See all of Yahoo’s Black Friday coverage, here. Follow Engadget for Black Friday tech deals. Learn about Black Friday trends
on In The Know. Hear from Autoblog’s experts on the best Black Friday deals for your car, garage, and home, and find Black Friday sales to shop on AOL,
handpicked just for you.
</item 17>

<item 18>
54 Best Apple Black Friday Deals (2023): iPad, Apple Watch, AirPods (Wired, 2023-11-25T00:00:29+00:00)
You can use it to charge your AirPods (as long as it comes with a wireless charging case) and other Qi-certified devices too.

Photograph: Zendure

The Zendure SuperTank Pro is the largest portable battery you can get away with taking onto an airplane. Which is why it's our favorite for air travel. With a
26,800-mAh capacity, four USB ports, and up to 100 watts in or out, it's the only portable power bank you'll need for every device on even extremely long
flights. Plus it kinda looks like a mini carry-on suitcase, that's neat. Just be sure to click the coupon button to get the full deal.

This is a great power bank if you don't want to carry a cable around. It comes complete with a built-in rotating USB-C or Lightning connector (MFi certified)
that slots into the bottom of your phone.
</item 18>

<item 19>
The 30 best Amazon Prime Day deals for October 2023 (Engadget, 2023-10-10T17:41:37+00:00)
Engadget has been testing and reviewing consumer tech since 2004. Our stories may include affiliate links; if you buy something through a link, we may earn a
commission. Read more about how we evaluate products.

Here are the best Prime Day deals we could find gathered in one place.

Amazon's second Prime Day-like sale event for 2023 kicked off in the wee hours this morning, and it's turning out to be a great opportunity to get holiday
shopping done early. Or, if you missed out on any of the deals from Prime Day in July, this year's "Prime Big Deal Days" will let you make up for that. The sale
has brought back many of the same steep discounts we saw during the summer as well as new sales. We've also put together category roundups for deals on
headphones, tablets, TVs, video games and other gadgets.
</item 19>

<item 20>
Some of our favorite devices are on sale for Black Friday. (The Verge, 2023-11-24T15:42:21+00:00)
A car-friendly charger

Sean Hollister, senior editor

Maybe someday Apple will realize that it wasn’t the brightest idea to artificially limit the reach of its MagSafe charging ecosystem and you’ll be able to plop
your phone on a magical minimalist disc that charges it at high speeds. In the meanwhile, a standard Qi charger is about the best you’ll get — and the bulky but
practical iOttie Easy One Touch Qi does it with the satisfying snap of springs. When you push your phone into its waiting jaws, it depresses a button that
causes those jaws to firmly snap closed on either side of your device. When you want to remove it, you pinch a pair of levers with your finger and thumb to
release as you grab your slab. It’s wide enough to fit practically anything on the market, save an opened Samsung Z Fold. I’ve used one for years with Android
and Apple phones alike, including newer MagSafe handsets.
</item 20>

<item 21>
Some of our favorite devices are on sale for Black Friday. (The Verge, 2023-11-24T15:42:21+00:00)
And while I’ve long been a proponent of MPOWERD’s solar-powered Luci lanterns, I recently picked up the company’s like-minded string lights for car camping and
overnight jaunts in the backcountry when I don’t mind toting a little extra weight with me.

The 18-foot string is certainly not the brightest you can buy — it packs a series of 100-lumen LEDs, whereas your average headlamp might offer 400 — but it can
swap between six different colors and features a 2,000mAh battery for when your phone needs some emergency juice. Best of all, you can charge the lights via USB
or solar, meaning you can spend less time worrying about how to keep them going and more time taking in the vibes.

Memory card holder and reader

Becca Farsace, senior producer

As a video person who is constantly on the move, there is nothing better than the consolidation of gear — especially when it pertains to dongles.
</item 21>

<item 22>
Some of our favorite devices are on sale for Black Friday. (The Verge, 2023-11-24T15:42:21+00:00)
But that’s just life while traveling with a dog anyway.

An extension cord for awkward situations

Sarah Jeong, deputy features editor

Anker 321 Power Strip $ 15 $ 26 42 % off $ 15 $ 15 $ 26 42 % off This all-in-one 20W USB-C power cube boasts three AC outlets, two USB-A ports, and one USB-C
port. $15 at Amazon

Nobody wants to carry a power strip or an extension cord with them on their vacation. It’s probably unnecessary if you’re staying in relatively modern buildings
and definitely unnecessary if you’re camping. But sometimes you want to stay in a charming historical hotel or a lovely cabin in the woods, and it’s only when
you go to charge your devices at night that you realize that the only electrical socket in the bedroom is in the corner farthest away from the bed and there’s
already two lamps plugged into it.
</item 22>

<item 23>
The best Black Friday deals you can get for under $25 (The Verge, 2023-11-23T17:58:09+00:00)
$21 at The Verge Store

Verge Deals / Sign up for Verge Deals to get deals on products we've tested sent to your inbox daily. Email (required) Sign up By submitting your email, you
agree to our Terms and Privacy Notice . This site is protected by reCAPTCHA and the Google Privacy Policy and Terms of Service apply.
</item 23>

<item 24>
Some of our favorite devices are on sale for Black Friday. (The Verge, 2023-11-24T15:42:21+00:00)
It’s not the prettiest accessory one can attach to their desk, but it is highly functional.

Charging station

Jess Weatherbed, news writer

I got this as a birthday present from my partner after several months of arguing over our sleeping arrangements. Something about me trailing the charging cables
for my devices in the bed being “dangerous” and “extremely uncomfortable.” Anyway, after begrudgingly acknowledging my poor charging habits, I have to admit
that having this on my desk has had benefits outside of not garroting myself mid-slumber.

My iPhone, Apple Watch, and wireless earbuds rarely run out of juice, as I no longer fall asleep before plugging them in. I’ve also taken to using the charger’s
upright positioning for phones to my advantage — serving as a desk clock, a tiny display for Slack or Discord, and as a hub to remotely control the various
smart devices around my home. It helps me separate the device from being my phone and instead helps me build the habit of it being another tool to boost my
productivity.
</item 24>

<item 25>
78 Absolute Best October Prime Day Deals (2023) (Wired, 2023-10-10T14:23:00+00:00)
This 2.5-inch cube from Anker is a great compact charger. It comes with a MagSafe pad on top (that charges at up to 15 watts), a top section that hinges to a
60-degree angle to reveal a charging surface for your AirPods, and a shelf on the side that has a built-in Apple Watch charger (which can comfortably
accommodate any Apple Watch including the Ultra). You'll also get a 5-foot cable and a 30-watt charger in the box. It supports fast charging too.

Anker 737 Power Bank Photograph: Anker

We just added this power bank as our top upgrade pick in our guide to the Best Portable Chargers. It's pricey, but today's deal makes it more accessible. It
charges from zero to completely full in an hour and boasts a whopping 24,000-mAh capacity. And it's powerful enough to charge laptops and tablets as well as
phones and other gadgets. There's even a built-in display to monitor stats like temperature and remaining battery percentage.
</item 25>

<item 26>
The best Black Friday deals you can get for under $25 (The Verge, 2023-11-23T17:58:09+00:00)
Every Black Friday and Cyber Monday, there’s always a ton of hype about the discounts on big-ticket items — you know, the deals on the OLED TVs, game consoles,
and tablets of the world. But it never fails that many of the ultra-cheap items make for some of the most clutch deals, namely because spending very little on a
small accessory you end up using all the time can be the most gratifying.

This year, throughout the annual shopping weekend, we’re once again keeping an eye out for the best tech deals on those little gadgets and gizmos under $25.
We’ve also got roundups for those working with $50 or $100 budgets, but the items below are pretty much as cheap as you can get in the tech world without simply
buying absolute crap that’s destined to end up as e-waste.

Some of these helpful budget deals include gems like fully compatible MagSafe chargers for the latest iPhones, a colorful smart speaker, Apple’s handy AirTag
trackers, and some staff favorites.
</item 26>

<item 27>
Some of our favorite devices are on sale for Black Friday. (The Verge, 2023-11-24T15:42:21+00:00)
This is key because the loose sockets of older buildings reject the beautiful convenience of folding prongs. There are more than enough sockets for one person,
and with some finagling (and maybe an extra charging brick), it can accommodate two people’s devices.

If you’re traveling overseas, don’t forget to purchase a different plug type for the region you’re going to or pack an adapter.

A portable smart speaker

Brandon Widder, senior commerce editor

Sonos Roam $ 134 $ 180 26 % off $ 134 $ 134 $ 180 26 % off The Sonos Roam is a truly portable Sonos speaker with a rugged design that’s built to withstand the
elements. It also features wireless charging and supports AirPlay 2, Alexa, and Google Assistant. $134 at Best Buy$134 at Sonos

For the longest time, my go-to portable speaker for camping and backpacking was the Ultimate Ears Roll 2.
</item 27>

<item 28>
The best Black Friday deals you can get for under $25 (The Verge, 2023-11-23T17:58:09+00:00)
Come on in and peruse these affordable wares.

Phone accessories and smart trackers

Apple AirTag $ 24 $ 29 17 % off $ 24 $ 24 $ 29 17 % off Apple’s AirTag can help you find your lost items with its ultra wideband technology. You’ll get the best
compatibility with an iPhone, though Apple released an Android app that can detect an AirTag’s location and notify you if one seems to be following you. Read
our review. $24 at Amazon$80 at Amazon (four-pack)

Anker 511 Charger (30W) $ 16 $ 23 30 % off $ 16 $ 16 $ 23 30 % off Anker’s 30W 511 Charger juices phones quickly and is small enough to carry around in a purse.
</item 28>

<item 29>
The 53 best Black Friday deals we could find at Amazon, Walmart, Target and more (Engadget, 2023-11-19T15:13:23+00:00)
It also has a decent 1080p webcam. The chassis isn't especially light at 3.6 pounds, and we found the battery to last roughly seven hours in testing, which is
just OK. But for less than $400, this is a strong value for those who just want an affordable notebook for the basics.

FAQs

When is Black Friday 2023?

Black Friday 2023 lands on November 24 this year, with Cyber Monday 2023 following up a few days later on November 27. However, the early deals have already
begun. The entire month of November might as well be renamed Holiday Shopping Month as that’s when companies really start to get serious with their discounts.

What day is Black Friday 2023 at Walmart?

Walmart Black Friday deals will begin early this year on November 8. Walmart+ members will get a few hours of early access to Black Friday deals if they shop
online on November 8, with access for everyone else starting later that day.
</item 29>

<item 30>
Some of our favorite devices are on sale for Black Friday. (The Verge, 2023-11-24T15:42:21+00:00)
Older buildings especially suffer from what I can only describe as loose socket syndrome, where those very convenient modern boxy socket extenders with five
different USB and USB-C charging ports simply cannot stay in place and fall right out of the wall because they’re too heavy. After one (totally pleasant)
vacation where I had to charge my phone, watch, AirPods, and laptop in a weird corner of my room with the plug-in charging hub propped up on a strategically
balanced mountain of books and sham pillows, I bought this Anker combination extension cord / power strip. It’s not a full power strip — just a cube with a few
sockets along with USB and USB-C charging ports at the end of a five-foot cable. I’ve brought it on a few trips since then. It takes up extra space in my
suitcase but each time has left me feeling vindicated about the purchase.

The three prongs at the end of the cable are static, rather than folding flat for easy packing.
</item 30>

<item 31>
The best portable chargers for your gear (Polygon, 2023-11-17T13:10:12+00:00)
One of the chargers we recommend below can match the speed delivered by Apple’s 140 W wall charger that’s included with 16-inch MacBook Pro laptops. On the
Windows side of things, most laptops don’t go above 90 or 100 W at peak USB-C recharging speed, so you should be set with our picks.

It might be tough to justify spending $100 to $150 on a power bank just to charge a laptop. I get it. However, it’s easier to justify getting one if you
frequently travel with multiple power-guzzling devices that you’d like to recharge simultaneously, as these are but a few available that can manage
multitasking.

Update (Nov. 17): Checked prices and stock for accuracy.
</item 31>

<item 32>
41 Best Prime Day Apple Deals (2023) (Wired, 2023-10-10T22:58:00+00:00)
It charges at the maximum 15-watt rate, and you can turn your iPhone sideways for iOS 17’s new StandBy mode, converting it into a bedside alarm clock.

Anker 737 Power Bank Photograph: Anker

We just added this power bank as our top upgrade pick in our guide to the Best Portable Chargers. It's pricey, but today's deal makes it more accessible. It
charges from zero to completely full in an hour and boasts a whopping 24,000-mAh capacity. And it's powerful enough to charge laptops and tablets as well as
phones and other gadgets. There's even a built-in display to monitor stats like temperature and remaining battery percentage.

Nomad's wireless charger is one of our favorite MagSafe wireless chargers. It delivers the maximum 15-watt charging speed possible, but it's not meant to be
held up while you use the phone. It doesn't have a power adapter but includes a USB-C-to-USB-C cable. It'll sit securely on your desk and can charge an Apple
Watch, too.
</item 32>

<item 33>
78 Absolute Best October Prime Day Deals (2023) (Wired, 2023-10-10T14:23:00+00:00)
The price matches the lowest we have tracked. It doesn't have a built-in Apple Watch charger, but it can top off your compatible wireless earbuds whilst
simultaneously charging your iPhone. It charges at the maximum 15-watt rate, and you can turn your iPhone sideways for iOS 17’s new StandBy mode, converting it
into a bedside alarm clock.

Anker 3-in-1 MagSafe Wireless Charging Dock Photograph: Anker

Anker makes our favorite lists often, including with this super compact 3-in-1 wireless charger. A MagSafe pad charges iPhones up to 15 watts at a slanted
angle, and on the side is a standard Apple Watch puck (no fast-charging support). In the space inside the triangle is where you can place your AirPods Pro (or
any other wireless earbuds case) to top them up.

This 2.5-inch cube from Anker is a great compact charger.
</item 33>

<item 34>
The 26 best Amazon Prime Day deals you can still shop for today (Engadget, 2023-10-12T15:44:16+00:00)
It's been an Engadget favorite for a while thanks to its sequential read speeds of 560MB/s and a standard 2.5-inch design that should make it easy to swap in
for your desktop's or laptop's used-up drive.

Other great deals for after Prime Day

Anker 511 Charger

Anker 511 Charger (Nano 3) $20 $23 Save $3 See at Amazon

New Apple Watches don’t come with power adapters, so if you’re looking for one that can deliver a quick charge, we recommend Anker's 511 (Nano 3) charging
brick. It’s down to $20 which is a mild, $4 discount, but still a welcome price for a worthwhile smartwatch accessory.
</item 34>

<item 35>
54 Best Apple Black Friday Deals (2023): iPad, Apple Watch, AirPods (Wired, 2023-11-25T00:00:29+00:00)
It has a little stand to prop it up, a short cord that goes out to an outlet, and four USB-C ports that output 100 watts each. This is the lowest price we've
tracked.

Click the coupon button to see the discount at checkout. We recommend this Baseus charger if you're looking for a high-capacity option. With a 10,000-mAh
capacity, WIRED contributor Simon Hill says it charged his iPhone 14 Pro once and then delivered an additional 64 percent the second time before tapping out. It
comes with a USB-C cable too. The 6,000-mAh version is also on sale for $30 ($20 off). These deals are only applicable to Amazon Prime members.

Apple 35-Watt Dual USB-C Port Power Adapter Photograph: Apple

If you tend to forget to pack chargers for your devices, Apple's dual USB-C port power adapter is an essential accessory.
</item 35>

<item 36>
The best portable chargers for your gear (Polygon, 2023-11-17T13:10:12+00:00)
In terms of tablets, picking either of the two power bank options we recommend below should be a good fit for what you may have. For reference, all iPads
currently ship with 20 W power adapters, and so far, tablets from other manufacturers like Samsung and Amazon haven’t deviated far from that charging speed.

Whichever you choose, it never hurts to get some extra USB-C-to-USB-C cables. You can get this two-pack at Amazon for $15.99, so you can keep one at home and
then put the other in a bag you bring around during everyday travels.

Best power bank for your Switch

The Nintendo Switch console with detachable Joy-Cons can receive a maximum of 18 W through its USB-C charging port (the Switch Lite is slightly less demanding).
So, the very same compact power bank we recommend for phones and tablets can be used to charge your Switch.

Same note as before: We suggest buying more charging cables than you’ll immediately need (stow one permanently in the bag you frequently use).
</item 36>

<item 37>
The 53 best Black Friday deals we could find at Amazon, Walmart, Target and more (Engadget, 2023-11-19T15:13:23+00:00)
Engadget has been testing and reviewing consumer tech since 2004. Our stories may include affiliate links; if you buy something through a link, we may earn a
commission. Read more about how we evaluate products.

Black Friday is almost here, but we wouldn't blame you if you've felt like it's been here for a while. What was once a one-day event filled with limited-time
deals has grown into a month-long event with a plethora of sales to sift through each day. This year, Amazon, Target, Best Buy and Walmart and several other
retailers kicked off their Black Friday deals early, with some starting as soon as late October. If you're overwhelmed by the amount of discounts showing up in
your feeds, you're not alone.



To help, we at Engadget have collected all of the best Black Friday deals you can shop right now, primarily on tech we've tested and reviewed. These are the
best tech deals you can get this year, on the gadgets we know to be worth your money.
</item 37>

<item 38>
How to tackle Black Friday shopping like a boss (Polygon, 2023-11-12T15:00:00+00:00)
However, it likely won’t be the last opportunity for you to save. As someone who’s been covering deals for The Verge and Polygon for years, I’m shocked by how
often I see these low prices, often spoken of as if they’re a flash in the pan, extending through much of December. That said, it’s more likely that price cuts
on products you want can go by unnoticed unless you’re tracking deals yourself.

Track deals yourself

We love to tell you about great deals. But there’s a limit to the kind of products that we write about, and the ones that we see. If you’re not comfortable
leaving it up to the chance that we’re going to tell you about price cuts on the products you really want, you have the power to track deals with relative ease.

If you only shop on Amazon, CamelCamelCamel remains one of the best tools for tracking price changes and history for products.
</item 38>

<item 39>
78 Absolute Best October Prime Day Deals (2023) (Wired, 2023-10-10T14:23:00+00:00)
We have seen this dip a little lower, but this is still a solid deal.

The unusual design of Satechi's fold-up wireless charging stand allows it to charge both phones and a wireless earbuds case (if it supports Qi wireless
charging), plus a USB-C port to plug in a third device—not bad for a device that fits in your bag. It has a 10,000-mAh capacity with LEDs to show how much juice
is left. The downside is that it is slow, offering up to 10 watts of wireless charging power for phones (7.5 watts for iPhones), 5 watts for earbuds, and 10
watts from the USB-C port.

Best Prime Day Headphone Deals

Photograph: Sony

While its predecessor was one of the best pairs of wireless earbuds around, the WF-1000XM5 (7/10, WIRED Recommends) is still no slouch. They produce balanced
sound, are comfortable to wear, and have active noise cancellation.
</item 39>

<item 40>
How to tackle Black Friday shopping like a boss (Polygon, 2023-11-12T15:00:00+00:00)
The best deals of the year were once known to occur during Black Friday and Cyber Monday, which in 2023 take place on Friday, Nov. 24, and Monday, Nov. 27,
respectively. Not anymore!

In 2023, the best Black Friday deals kicked off on Nov. 17 at Amazon, Best Buy, Walmart, and other retailers. Instead of hosting all of the doorbuster deals on
one day right after Thanksgiving, retailers see the value in maximizing the amount of time you have to shop for products before Black Friday. And I’ve gotta
say, it’s quite nice to not feel rushed while shopping.

In addition to presenting you with the best board game deals and price drops on gaming monitors, plus deals on SSDs and microSD cards, there are some bits of
info and wisdom I’ve organized into sections below that might help you feel more informed as you begin to shop for the holidays.
</item 40>

<item 41>
The 53 best Black Friday deals we could find at Amazon, Walmart, Target and more (Engadget, 2023-11-19T15:13:23+00:00)
That's a $30 discount. Both the USB-C and Lightning models are on sale; the latter is compatible with Apple's iPhone 15 series as well as numerous Android
phones. We gave the Backbone a positive review last year, and we recommend it in our guide to the best gaming handhelds: It's more cramped than a traditional
gamepad, naturally, but it has all the requisite buttons to play modern games, it's extremely easy to set up and it works well for remote streaming from your
console or PC. If nothing else, it's a much better value than the PlayStation Portal.

PlayStation, Xbox and Switch video game sales

If you're all set on the hardware front, a truckload of PlayStation 5, Xbox Series X/S and Nintendo Switch games have dropped in price for Black Friday too.
Both Sony and Microsoft have rolled out holiday sales on their respective digital storefronts, while a number of physical games are down to new lows at Amazon,
Best Buy and others.
</item 41>

<item 42>
Some of our favorite devices are on sale for Black Friday. (The Verge, 2023-11-24T15:42:21+00:00)
Black Friday and Cyber Monday may be in the past, but some sales are still hanging around (although maybe not for long). For more leftover deals we recommend
across all categories, be sure to check out the rest of the Cyber Monday still available here.

Every month or so, we like to ask our staff about their favorite stuff — whether it’s pet toys, travel aids, kitchen gadgets, or straightforward tech. And the
results are usually very different, very interesting, and a lot of fun.

In celebration of the annual post-Thanksgiving sales, we looked through some of our recent “favorites” articles and found deals on a lot of the tech, kitchen
tools, travel aids, and pet toys we like. We thought we’d list a few in case you’ve read about them in the past and thought, “Well, that sort of sounds good,
but it’s a bit pricey.”
</item 42>

<item 43>
54 Best Apple Black Friday Deals (2023): iPad, Apple Watch, AirPods (Wired, 2023-11-25T00:00:29+00:00)
It's not a huge discount, but this is the lowest price we've seen.

This charger packs two USB-C ports, one of which can output 65 watts when used alone. If you plug two devices in, the top will deliver 45 watts and the bottom
can do 20 watts. It also folds up and looks super sleek.

Satechi Duo Wireless Charger Stand and Power Bank Photograph: Satechi

Click the coupon button to see the full discount at checkout. The Satechi Duo combines the convenience of wireless charging with the portability of a power
bank. It has a 10,000-mAh capacity (complete with LEDs to show you how much juice you have left) and can wirelessly charge both your phone and a wireless
earbuds case (as long as it's Qi-certified). There's also a USB-C port to power up a third device.
</item 43>

<item 44>
The best portable chargers for your gear (Polygon, 2023-11-17T13:10:12+00:00)
If you buy something from a Polygon link, Vox Media may earn a commission. See our ethics statement .

Your time — and money — is precious. Polygon’s What to Buy has rounded up the best in gaming hardware, entertainment, tabletop, merch, and toys, and how to get
them for the best value, so you spend less time shopping and more time enjoying.

A power bank, also commonly referred to as a portable charger, is an essential purchase for anyone who’s on the go with a phone, tablet, or a gaming handheld
like a Nintendo Switch or a Steam Deck. It lets you recharge your precious gadgets one or more times via a USB connection, so you can keep talking, texting, or
gaming throughout the day.

Power banks range in price depending on a few factors: how compact it is, what kind of ports it has, how much battery capacity it contains (typically measured
in mAh), and how quickly it can recharge your devices.
</item 44>

<item 45>
Some of our favorite devices are on sale for Black Friday. (The Verge, 2023-11-24T15:42:21+00:00)
That’s where my vacuum sealer comes in.

While I can’t speak to the quality of other vacuum sealers, the FoodSaver I have is awesome. Not only does the thing help keep raw meat and other food fresher
for longer in the fridge, but it also helps save space in the freezer (each package of meat becomes much flatter when all the air is sucked out of it). With
this little machine, I can load up whatever I want in one of the FoodSaver bags, insert the open end into the machine, which vacuums up all the air and then
closes the bag using its heat sealing feature in one fell swoop. It’s pretty neat!

For the traveler

Carry-on backpack

Kaitlin Hatton, audience manager

Earlier this year, I committed to traveling more, and so I took a look at the gear I had that could be replaced after years of trekking the globe. My ratty old
secondhand carry-on bag was the first thing to be replaced.
</item 45>

<item 46>
How to tackle Black Friday shopping like a boss (Polygon, 2023-11-12T15:00:00+00:00)
It’s all explained here.

Most Black Friday return policies extend through January 2024

According to Best Buy, certain products bought now (dating back to Oct. 27) until Dec. 30 may be returned before Jan. 13.

Target will accept most unopened items in new condition within 90 days of purchase.

Walmart has an extended holiday return policy in effect that allows items purchased between Oct. 1 and Dec. 31 to be returned on or before Jan. 31, 2024.

Amazon’s extended return policy is in effect for most items that are purchased between Nov. 1 and Dec. 31. Those items can be returned before Jan. 31, 2024.
However, Apple-brand products purchased in that time frame can only be returned on or before Jan. 15, 2024.

Black Friday likely won’t be your last opportunity to save

There will be amazing deals during Black Friday and Cyber Monday on everything, from tabletop and board games to toys, video games, and more.
</item 46>

<item 47>
78 Absolute Best October Prime Day Deals (2023) (Wired, 2023-10-10T14:23:00+00:00)
There's even a built-in display to monitor stats like temperature and remaining battery percentage.

This heavy metal brick is a good option for charging laptops and smaller gadgets. It packs 20,000 mAh and supports a wide variety of fast charging standards.
There is one USB-C PD port rated at 65 watts, two USB-A QC ports at 30 watts apiece, and a micro-USB input (though you are best using the USB-C to recharge it).
We haven't seen it go on sale very often.

This travel kit from ESR works well if you want something that can prop your iPhone in portrait or landscape orientation. It can also display your Apple Watch
in Nightstand mode, so you can peek and see how long before you must get out of bed. The main body folds open with a MagSafe charging pad for your iPhone and a
slot behind for your AirPods. We have seen this dip a little lower, but this is still a solid deal.
</item 47>

<item 48>
The best Apple Black Friday deals on iPads, AirPods, Apple Watches and more (Engadget, 2023-11-24T09:30:32+00:00)
Engadget has been testing and reviewing consumer tech since 2004. Our stories may include affiliate links; if you buy something through a link, we may earn a
commission. Read more about how we evaluate products.

There are still tons of Apple Black Friday deals you can shop for online; these are our favorites.

Most of Apple’s current-generation devices went on sale for Black Friday — and most of those deals look like they’ll carry over through the weekend and into
Cyber Monday. That means if you were otherwise occupied on Friday, you can still save on iPads, Apple Watches, MacBooks, AirPods and more. Some of the prices
are new lows and many others match the record prices we’ve seen previously. We’ve tested and reviewed nearly every device below, so check out the links to our
full reviews and guides if you need help deciding which deals to snag. These are the best Apple Black Friday deals you can still get today from retailers like
Amazon, Target, Best Buy and B&H Photo.
</item 48>

<item 49>
The 53 best Black Friday deals we could find at Amazon, Walmart, Target and more (Engadget, 2023-11-19T15:13:23+00:00)
One of the company's 67W USB-C chargers is 37 percent off, bringing it to $38 and matching its all-time low. Our top 65W fast wall charger, Anker's 735
GaNPrime, is down to a new low price after a 38 percent discount. And the Anker Nano Power Bank is seeing one of its first discounts since coming on the market
a couple months ago.

Meta Quest 2 VR headset

The Meta Quest 2 VR headset is $50 off and down to $249 at several retailers. That matches the best price we’ve seen. At GameStop and Target, you can get the
device with a $50 gift card. Despite the launch of the impressive Quest 3, we still consider the Quest 2 to be one of the best VR headsets available right now —
precisely because of its more affordable price.
</item 49>

<item 50>
The 53 best Black Friday deals we could find at Amazon, Walmart, Target and more (Engadget, 2023-11-19T15:13:23+00:00)
If you're a frequent Target shopper, note that the retailer is selling this same bundle with a $75 store gift card for $450.

Xbox Series S

A starter bundle that includes a 512GB capacity Xbox Series S and a three-month subscription to Game Pass Ultimate is currently $240 at Microsoft. The console
by itself has a list price of $300 and three months of access to Xbox's service will set you back $51 after this year's price hike (though the first month is
often just $1 for new members).

Anker accessories sale

A slew of Anker chargers, cables, power adapters and other accessories are on sale for Black Friday at Amazon, with discounts ranging up to 37 percent. The
deals extend to our favorite premium power bank, Anker's Prime 20,000 mAH battery with charging base, which is $130 for the set and its lowest price ever.
</item 50>
=== END ITEM id=cf1e1b0dada149b3 ===
