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

=== ITEM id=4939b91b4341455a ===
Gold sentences and facts:
Sentence 0: For the Swans, however, the victory has them challenging for a top-eight spot alongside the likes of Gold Coast, Geelong, Collingwood, St Kilda and Carlton.
  Fact 0.0: The victory has the Swans challenging for a top-eight spot.
  Fact 0.1: The Swans are challenging for a top-eight spot alongside Gold Coast, Geelong, Collingwood, St Kilda and Carlton.
Sentence 1: With only 0.4 of a percentage point separating Sydney in ninth and St Kilda in eighth, the win assured the Swans of retaking eighth spot.
  Fact 1.0: Only 0.4 of a percentage point separates Sydney in ninth from St Kilda in eighth.
  Fact 1.1: The win assured the Swans of retaking eighth spot.
Sentence 2: “Our team has put a lot of work into sourcing some wonderful food offerings, entertainment and activities for kids, while we’ve had a local focus in promoting our games,” she said.
  Fact 2.0: The speaker said her team has put a lot of work into sourcing some wonderful food offerings, entertainment and activities for kids.
  Fact 2.1: The speaker said the team has had a local focus in promoting its games.

Retrieved items:
<item 1>
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
Graph query row:
article: Comeback kings Richmond take the AFLW points in Cairns
published_on: 2023-10-20T23:51:34+00:00
entity: Emma Swanson
passage: Comeback kings Richmond take the AFLW points in Cairns (The Age, 2023-10-20T23:51:34+00:00)
Michael Prior guided West Coast to a gutsy win. Credit: Getty

Aisling McCarthy and Amy Franklin drew first blood for West Coast within the opening five minutes, while the hosts were kept goalless. The Bombers only added
three behinds in the second term and the Eagles failed to score as the blustery conditions took their toll on the game. West Coast’s Kellie Gibson broke the
goal drought with the only major of the third quarter, before the home side came to life with the wind at their backs in the final term, with skipper Bonnie
Toogood and Amber Clarke reducing the margin to four points. Toogood continued to pepper the scoreboard, but the Eagles held on to secure a rare victory. “We’ve
had a horror week back at home, and it’s been really bloody tough on all of us, and Mick as well. I’m just really proud of the girls,” West Coast captain Emma
Swanson said post-game on the broadcast. “I can name the whole team.
</item 3>

<item 4>
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
</item 4>

<item 5>
Craft beer and cheese boxes: Swans tap into something ‘uniquely Sydney’ to draw AFLW crowds | Jack Snape (The Guardian, 2023-10-26T23:42:29+00:00)
“Even at Henson Park [the Swans’ other home ground in Sydney’s gentrified inner west], they’ve gone to microbreweries in the local area and invited them in, but
also promoted through their clientele to come to the football.”

The Swans forward Bella Smith said the players were proud of the club’s attendance record, and were grateful for the support. “We get there two hours before the
start of the game and the hill [at Henson Park] is bare. When you come out for the first team warm-up, the hill is slowly filling and then by the time the first
bounce gets around, it is full,” she said. “You can hear it, the atmosphere is amazing.”

The average attendance for the AFLW is a little over 2,500 a match this year, tracking approximately on par with previous seasons and with a finals uptick
likely to come.
</item 5>

<item 6>
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
</item 6>

<item 7>
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
</item 7>

<item 8>
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
</item 10>
=== END ITEM id=4939b91b4341455a ===
