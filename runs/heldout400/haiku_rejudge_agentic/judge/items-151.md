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

=== ITEM id=78a462b9ab1caedf ===
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
Craft beer and cheese boxes: Swans tap into something ‘uniquely Sydney’ to draw AFLW crowds | Jack Snape (The Guardian, 2023-10-26T23:42:29+00:00)
But crowds are short of targets that could accelerate the extension of the regular season, which grows from 10 games this year to 12 games by 2025.

5,474 fans were at North Sydney Oval to watch the Swans’ first AFLW victory earlier in the season. Photograph: Jack Snape/The Guardian

The league enjoyed strong attendances in round one, which were partly attributed to the so-called Matildas effect, but some clubs have struggled to attract fans
in large numbers this season. Crowds of less than 1,500 have been recorded in both traditional markets such as Melbourne and Perth, as well as areas targeted
for growth like Blacktown and the Gold Coast.

The A-League Women launched this month with a first round highlighted by a crowd of more than 11,000 in Sydney.

Graph facts:
- Ground-breaking Gold Coast: Suns make AFLW finals for a second time -MENTIONS-> AFLW
- Gold Coast Suns -LOCATED_IN-> Gold Coast
- Ground-breaking Gold Coast: Suns make AFLW finals for a second time -MENTIONS-> Gold Coast
- Comeback kings Richmond take the AFLW points in Cairns -MENTIONS-> AFLW
- Sydney Swans -PART_OF-> AFLW

Entities: Gold Coast, Jack Snape, AFLW, Craft beer and cheese boxes: Swans tap into something ‘uniquely Sydney’ to draw AFLW crowds, Swans, North Sydney Oval,
Matildas, A-League Women, Blacktown, Swans’ first AFLW victory, A-League Women launch, Perth, Melbourne, Sydney, The Guardian
</item 1>

<item 2>
Comeback kings Richmond take the AFLW points in Cairns (The Age, 2023-10-20T23:51:34+00:00)
Molloy, the Swans’ star off-season recruit, turned on one of the best performances of her career to demolish their injury-hit hosts.

The former Collingwood ace was a class above as she kicked 3.4 from her 28 possessions and played a crucial role in setting up numerous other Sydney goals.
Loading “She [Molloy] was itching to get in there [in the midfield],” Swans coach Scott Gowans said. “We kept her forward deliberately. It was important Chloe
played a good block of time forward so we could have a look at it.” The Blues lost to Greater Western Sydney by two points on Saturday afternoon, denying them
the chance to leapfrog into the top eight, 8.1 (49) to 7.5 (47).

Narrm reclaim top spot Reigning premiers Narrm have reclaimed top spot on the ladder and their position as flag favourites after defeating third-placed North
Melbourne by 23 points at Ikon Park on Saturday night.

Graph facts:
- How ego and fear fuelled the rise of artificial intelligence -PUBLISHED_BY-> The Age
- ASX set to open higher as Wall Street rebounds; $A rises -PUBLISHED_BY-> The Age
- Is Google Search better than the rest? And is that fair? -PUBLISHED_BY-> The Age
- When pop culture and sport collide: A timeline of Taylor Swift’s NFL takeover -PUBLISHED_BY-> The Age
- At 37, Alexandra found herself broke, heartbroken and living with her parents -PUBLISHED_BY-> The Age

Entities: The Age, AFLW, Comeback kings Richmond take the AFLW points in Cairns, Scott Gowans, Blues lost to Greater Western Sydney, Narrm, Narrm defeated North
Melbourne, Richmond take the AFLW points in Cairns, Cairns, Richmond, Ikon Park, Greater Western Sydney, North Melbourne, Swans, Chloe Molloy
</item 2>

<item 3>
Comeback kings Richmond take the AFLW points in Cairns (The Age, 2023-10-20T23:51:34+00:00)
It was the Western Bulldogs’ eighth loss in a row, and their biggest-ever defeat, and they have just two rounds left to clinch a win this season. They next play
17th-placed West Coast and then top-four side North Melbourne. For the Swans, however, the victory has them challenging for a top-eight spot alongside the likes
of Gold Coast, Geelong, Collingwood, St Kilda and Carlton.

The Swans, who went winless in their debut season last year, were propelled to victory by a Chloe Molloy masterclass. After a five-goal second term, which
included a four-minute burst of three majors, Sydney had already brought up their largest score in an AFLW game by halftime. Swan Chloe Molloy celebrates a
goal. Credit: AFL Photos / Getty Images The Bulldogs held their own in the last quarter, but the damage had already been done. The total was also the biggest
the Dogs have conceded in their history, topping the 83 Melbourne kicked against them in round three.

Graph facts:
- Which M.L.B. Playoff Team Has the Most Daunting Home Field? We Ranked Them. -MENTIONS-> Getty Images
- Video games in 2023: Acquisitions, layoffs, unions -MENTIONS-> Getty Images
- From Justin Timberlake to Sam Asghari: A timeline of Britney Spears’ relationships -MENTIONS-> Getty Images
- West Indies v England: fifth and deciding men’s T20 cricket international – live -MENTIONS-> Getty Images
- Amazon sellers sound off on the FTC's 'long-overdue' antitrust case -MENTIONS-> Getty Images

Entities: Getty Images, The Age, West Coast, AFLW, Gold Coast, Geelong, Comeback kings Richmond take the AFLW points in Cairns, Western Bulldogs, Melbourne,
Cairns, Richmond, North Melbourne, Swans, Chloe Molloy, AFL Photos
</item 3>

<item 4>
Craft beer and cheese boxes: Swans tap into something ‘uniquely Sydney’ to draw AFLW crowds | Jack Snape (The Guardian, 2023-10-26T23:42:29+00:00)
“Even at Henson Park [the Swans’ other home ground in Sydney’s gentrified inner west], they’ve gone to microbreweries in the local area and invited them in, but
also promoted through their clientele to come to the football.”

The Swans forward Bella Smith said the players were proud of the club’s attendance record, and were grateful for the support. “We get there two hours before the
start of the game and the hill [at Henson Park] is bare. When you come out for the first team warm-up, the hill is slowly filling and then by the time the first
bounce gets around, it is full,” she said. “You can hear it, the atmosphere is amazing.”

The average attendance for the AFLW is a little over 2,500 a match this year, tracking approximately on par with previous seasons and with a finals uptick
likely to come.
</item 4>

<item 5>
Ground-breaking Gold Coast: Suns make AFLW finals for a second time (The Age, 2023-11-03T23:45:04+00:00)
WEEK ONE FINALS FIXTURE SATURDAY NOVEMBER 11 First Qualifying Final: Adelaide v Brisbane Lions, 2.35pm local (3.05pm AEDT), Norwood Oval First Elimination
Final: Gold Coast v Sydney, 6.15pm local (7.15pm AEDT), Heritage Bank Stadium SUNDAY NOVEMBER 12 Second Elimination Final: Geelong v Essendon, 1.05pm AEDT,
GMHBA Stadium Second Qualifying Final: Melbourne v North Melbourne, 3.05pm AEDT, Ikon Park The eight was not decided until the final game of the regular season,
when Sydney downed Fremantle by 22 points to leapfrog St Kilda into the final spot, after the Saints had kept themselves in the race with a 20-point win over
Carlton earlier in the day, 7.4 (46) to 3.8 (26).

Graph facts:
- How ego and fear fuelled the rise of artificial intelligence -PUBLISHED_BY-> The Age
- ASX set to open higher as Wall Street rebounds; $A rises -PUBLISHED_BY-> The Age
- Is Google Search better than the rest? And is that fair? -PUBLISHED_BY-> The Age
- When pop culture and sport collide: A timeline of Taylor Swift’s NFL takeover -PUBLISHED_BY-> The Age
- At 37, Alexandra found herself broke, heartbroken and living with her parents -PUBLISHED_BY-> The Age

Entities: The Age, Essendon, Geelong, Melbourne, Adelaide, Ground-breaking Gold Coast: Suns make AFLW finals for a second time, AFLW finals, Sydney, Norwood
Oval, Heritage Bank Stadium, GMHBA Stadium, Ikon Park, First Qualifying Final, First Elimination Final, Second Elimination Final
</item 5>

<item 6>
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

Graph facts:
- Ground-breaking Gold Coast: Suns make AFLW finals for a second time -MENTIONS-> Adelaide
- Comeback kings Richmond take the AFLW points in Cairns -MENTIONS-> Adelaide
- NBL Round 5: Harvey hits Hail Mary for Hawks to topple Taipans at the buzzer, Cheatham injured as Sixers upset Breakers -MENTIONS-> Adelaide
- Najwa Allen -WORKS_FOR-> Adelaide
- Chelsea Randall -WORKS_FOR-> Adelaide

Entities: Adelaide, Jack Snape, Sydney Swans, AFLW, Henson Park, Loretta Johns, Australian rules football, Harbour City, marquee pride match, Craft beer and
cheese boxes: Swans tap into something ‘uniquely Sydney’ to draw AFLW crowds, Collingwood, Sydney, The Guardian
</item 6>

<item 7>
Ground-breaking Gold Coast: Suns make AFLW finals for a second time (The Age, 2023-11-03T23:45:04+00:00)
With only 0.4 of a percentage point separating Sydney in ninth and St Kilda in eighth, the win assured the Swans of retaking eighth spot. If the Saints had
qualified, it would have been their maiden finals appearance. Saints goalkicker Jesse Wardlaw had a heavy collision with Carlton’s Keeley Sherar late in the
final quarter of their clash.

Georgia Patrikios showed her class in the do-or-die match, collecting 25 disposals and eight contested possessions in a powerhouse display, while Hannah Priest
was a brick wall in defence, with nine intercept possessions. Abbie McKay worked overtime to counter Patrikios in the midfield, but her side was outclassed
across the park. McKay racked up 28 disposals and 12 clearances alongside Mimi Hill (30 disposals), while defender Sherar (17) toiled against a potent Saints
attack. Meanwhile, Sydney prevailed in a fierce tug-of-war with the Dockers to seal their first AFLW finals appearance.

Graph facts:
- How ego and fear fuelled the rise of artificial intelligence -PUBLISHED_BY-> The Age
- ASX set to open higher as Wall Street rebounds; $A rises -PUBLISHED_BY-> The Age
- Is Google Search better than the rest? And is that fair? -PUBLISHED_BY-> The Age
- When pop culture and sport collide: A timeline of Taylor Swift’s NFL takeover -PUBLISHED_BY-> The Age
- At 37, Alexandra found herself broke, heartbroken and living with her parents -PUBLISHED_BY-> The Age

Entities: The Age, AFLW, Ground-breaking Gold Coast: Suns make AFLW finals for a second time, Sydney, Jesse Wardlaw, Keeley Sherar, Georgia Patrikios, Hannah
Priest, Abbie McKay, Mimi Hill, Dockers, Suns, Gold Coast, St Kilda, Carlton
</item 7>

<item 8>
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

Graph facts:
- Livingstone -SAID-> Women’s World Cup
- Livingstone -PARTICIPATED_IN-> West Indies v England: first T20 international
- West Indies v England: fifth and deciding men’s T20 cricket international – live -MENTIONS-> Livingstone
- West Indies v England: first T20 international – live -MENTIONS-> Livingstone
- Livingstone -PARTICIPATED_IN-> first T20 international

Entities: Livingstone, Western Bulldogs, Molloy, Jack Snape, AFLW, Craft beer and cheese boxes: Swans tap into something ‘uniquely Sydney’ to draw AFLW crowds,
Swans, AFLW competition, game time and scoring, AFL, The Guardian
</item 8>

<item 9>
Ground-breaking Gold Coast: Suns make AFLW finals for a second time (The Age, 2023-11-03T23:45:04+00:00)
Down by four points at half-time in the low-scoring game, a third-term goal from Rebecca Privitelli sparked the Swans to a 5.10 (40) to 2.6 (18) win on Sunday
at Fremantle Oval.

The second-year expansion side stormed home with three unanswered goals to secure eighth spot. It’s a dramatic improvement, with the Swans winless in their
first season. “Any club that’s able to go from winless one year to a finals berth, it’s incredible,” Sydney co-captain Chloe Molloy told Fox Footy after the
match. “You can see the emotion on the girls’ faces. Coming into the season, the odds were against us. Just to get into the eight, we’ve managed to do that.
We’re stoked. It’s pretty special for the club.”

Graph facts:
- How ego and fear fuelled the rise of artificial intelligence -PUBLISHED_BY-> The Age
- ASX set to open higher as Wall Street rebounds; $A rises -PUBLISHED_BY-> The Age
- Is Google Search better than the rest? And is that fair? -PUBLISHED_BY-> The Age
- When pop culture and sport collide: A timeline of Taylor Swift’s NFL takeover -PUBLISHED_BY-> The Age
- At 37, Alexandra found herself broke, heartbroken and living with her parents -PUBLISHED_BY-> The Age

Entities: The Age, Fremantle Oval, Fox Footy, Ground-breaking Gold Coast: Suns make AFLW finals for a second time, AFLW finals, Rebecca Privitelli, Suns, Gold
Coast, Swans, Chloe Molloy
</item 9>

<item 10>
Craft beer and cheese boxes: Swans tap into something ‘uniquely Sydney’ to draw AFLW crowds | Jack Snape (The Guardian, 2023-10-26T23:42:29+00:00)
Fans of Australian rules football in the Harbour City have a reputation for being fickle. After a winless campaign in 2022, the Swans – led by ex-Collingwood
player Chloe Molloy – have been competitive this season. The team won their first AFLW match in round one and are within a game of the top eight with two rounds
to play.

But general manager of women’s football at the AFL, Nicole Livingstone, said the Swans’ off-field offering had tapped into something “uniquely Sydney” and
helped make matches more attractive.

“To go to North Sydney Oval and to have quince paste and hand-picked grapes, cheddar cheese and biscuits in a little box that’s got this beautiful wrapping on
it – they’ve really thought about what they’re doing and made it an experience,” she said.

Graph facts:
- England choose survival over oblivion and a chance of summer trip to France -MENTIONS-> women’s football
- England thrash Scotland but cannot avoid Team GB Olympic heartbreak -MENTIONS-> women’s football
- Comeback kings Richmond take the AFLW points in Cairns -MENTIONS-> AFLW
- Ground-breaking Gold Coast: Suns make AFLW finals for a second time -MENTIONS-> AFLW
- Craft beer and cheese boxes: Swans tap into something ‘uniquely Sydney’ to draw AFLW crowds -MENTIONS-> Livingstone

Entities: women’s football, AFLW, Jack Snape, Harbour City, Craft beer and cheese boxes: Swans tap into something ‘uniquely Sydney’ to draw AFLW crowds, Swans,
Chloe Molloy, Nicole Livingstone, North Sydney Oval, first AFLW match, winless campaign, Collingwood, AFL, Sydney, The Guardian
</item 10>
=== END ITEM id=78a462b9ab1caedf ===

=== ITEM id=095971341490ee58 ===
Gold sentences and facts:
Sentence 0: For the past two weeks, coaches across college football have been riveted by the alleged Michigan sign-stealing scheme, but inside the Big Ten, the topic has been more than just a curiosity.
  Fact 0.0: For the past two weeks, coaches across college football have been riveted by the alleged Michigan sign-stealing scheme.
  Fact 0.1: Inside the Big Ten, the alleged Michigan sign-stealing scheme has been more than just a curiosity.
Sentence 1: While it is almost impossible to quantify how much the alleged in-person scouting and sign-stealing would have helped the Wolverines over the past two seasons, it will be a tough stain to erase knowing those violations are now on the Big Ten record.
  Fact 1.0: It is almost impossible to quantify how much the alleged in-person scouting and sign-stealing would have helped the Wolverines over the past two seasons.
  Fact 1.1: The in-person scouting and sign-stealing are alleged violations.
  Fact 1.2: The alleged violations are now on the Big Ten record.
  Fact 1.3: Knowing the alleged violations are on the Big Ten record will be a tough stain to erase.

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
=== END ITEM id=095971341490ee58 ===
