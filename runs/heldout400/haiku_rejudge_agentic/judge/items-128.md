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

=== ITEM id=37cff9fa118f7019 ===
Gold sentences and facts:
Sentence 0: “Hopefully we can go one better this year,” he added, recalling the 2-0 defeat by Canada in 2022.
  Fact 0.0: The speaker said, 'Hopefully we can go one better this year.'
  Fact 0.1: The speaker recalled the 2-0 defeat by Canada in 2022.
Sentence 1: After a winless campaign in 2022, the Swans – led by ex-Collingwood player Chloe Molloy – have been competitive this season.
  Fact 1.0: The Swans had a winless campaign in 2022.
  Fact 1.1: The Swans are led by ex-Collingwood player Chloe Molloy.
  Fact 1.2: The Swans have been competitive this season.

Retrieved items:
<item 1>
Graph query row:
article: Craft beer and cheese boxes: Swans tap into something ‘uniquely Sydney’ to draw AFLW crowds
published_on: 2023-10-26T23:42:29+00:00
connections: [{'entity': 'The Guardian', 'relationship': 'PUBLISHED_BY'}, {'entity': 'Livingstone', 'relationship': 'MENTIONS'}, {'entity': 'women’s football',
'relationship': 'MENTIONS'}, {'entity': 'AFLW', 'relationship': 'MENTIONS'}, {'entity': 'Western Bulldogs', 'relationship': 'MENTIONS'}, {'entity': 'Adelaide',
'relationship': 'MENTIONS'}, {'entity': 'Molloy', 'relationship': 'MENTIONS'}, {'entity': 'Gold Coast', 'relationship': 'MENTIONS'}, {'entity': 'Jack Snape',
'relationship': 'MENTIONS'}, {'entity': 'Sydney Swans', 'relationship': 'MENTIONS'}, {'entity': 'Henson Park', 'relationship': 'MENTIONS'}, {'entity': 'Loretta
Johns', 'relationship': 'MENTIONS'}, {'entity': 'Australian rules football', 'relationship': 'MENTIONS'}, {'entity': 'Harbour City', 'relationship':
'MENTIONS'}, {'entity': 'marquee pride match', 'relationship': 'MENTIONS'}, {'entity': 'Swans', 'relationship': 'MENTIONS'}, {'entity': 'Chloe Molloy',
'relationship': 'MENTIONS'}, {'entity': 'Nicole Livingstone', 'relationship': 'MENTIONS'}, {'entity': 'North Sydney Oval', 'relationship': 'MENTIONS'},
{'entity': 'first AFLW match', 'relationship': 'MENTIONS'}, {'entity': 'winless campaign', 'relationship': 'MENTIONS'}, {'entity': 'Matildas', 'relationship':
'MENTIONS'}, {'entity': 'A-League Women', 'relationship': 'MENTIONS'}, {'entity': 'Blacktown', 'relationship': 'MENTIONS'}, {'entity': 'Swans’ first AFLW
victory', 'relationship': 'MENTIONS'}, {'entity': 'A-League Women launch', 'relationship': 'MENTIONS'}, {'entity': 'domestic women’s sport', 'relationship':
'MENTIONS'}, {'entity': 'long-term growth', 'relationship': 'MENTIONS'}, {'entity': 'AFLW competition', 'relationship': 'MENTIONS'}, {'entity': 'game time and
scoring', 'relationship': 'MENTIONS'}, {'entity': 'Collingwood', 'relationship': 'MENTIONS'}, {'entity': 'Perth', 'relationship': 'MENTIONS'}, {'entity': 'AFL',
'relationship': 'MENTIONS'}, {'entity': 'Melbourne', 'relationship': 'MENTIONS'}, {'entity': 'Sydney', 'relationship': 'MENTIONS'}, {'entity': 'Women’s World
Cup', 'relationship': 'MENTIONS'}]
passages: ['Craft beer and cheese boxes: Swans tap into something ‘uniquely Sydney’ to draw AFLW crowds | Jack Snape (The Guardian,
2023-10-26T23:42:29+00:00)\nQuince paste and hand-picked grapes have played a part in helping the Sydney Swans launch themselves to the top of the AFLW
attendance charts, less than two years after their first match in the women’s league.\n\nThe Swans play their final home match of the regular season this
weekend with an average attendance of 4,365 from four games so far – 124 more than the next best club Adelaide – and a marquee pride match against Collingwood
at Henson Park to come on Sunday where a crowd in excess of 4,000 is expected.\n\nThe club’s focus has been to target local communities and make match days
compelling for fans on and off the field, according to executive GM of marketing at the Swans, Loretta Johns.\n\n“Our team has put a lot of work into sourcing
some wonderful food offerings, entertainment and activities for kids, while we’ve had a local focus in promoting our games,” she said.\n\nFans of Australian
rules football in the Harbour City have a reputation for being fickle.', 'Craft beer and cheese boxes: Swans tap into something ‘uniquely Sydney’ to draw AFLW
crowds | Jack Snape (The Guardian, 2023-10-26T23:42:29+00:00)\nFans of Australian rules football in the Harbour City have a reputation for being fickle. After a
winless campaign in 2022, the Swans – led by ex-Collingwood player Chloe Molloy – have been competitive this season. The team won their first AFLW match in
round one and are within a game of the top eight with two rounds to play.\n\nBut general manager of women’s football at the AFL, Nicole Livingstone, said the
Swans’ off-field offering had tapped into something “uniquely Sydney” and helped make matches more attractive.\n\n“To go to North Sydney Oval and to have quince
paste and hand-picked grapes, cheddar cheese and biscuits in a little box that’s got this beautiful wrapping on it – they’ve really thought about what they’re
doing and made it an experience,” she said.', 'Craft beer and cheese boxes: Swans tap into something ‘uniquely Sydney’ to draw AFLW crowds | Jack Snape (The
Guardian, 2023-10-26T23:42:29+00:00)\nBut crowds are short of targets that could accelerate the extension of the regular season, which grows from 10 games this
year to 12 games by 2025.\n\n5,474 fans were at North Sydney Oval to watch the Swans’ first AFLW victory earlier in the season. Photograph: Jack Snape/The
Guardian\n\nThe league enjoyed strong attendances in round one, which were partly attributed to the so-called Matildas effect, but some clubs have struggled to
attract fans in large numbers this season. Crowds of less than 1,500 have been recorded in both traditional markets such as Melbourne and Perth, as well as
areas targeted for growth like Blacktown and the Gold Coast.\n\nThe A-League Women launched this month with a first round highlighted by a crowd of more than
11,000 in Sydney.', 'Craft beer and cheese boxes: Swans tap into something ‘uniquely Sydney’ to draw AFLW crowds | Jack Snape (The Guardian,
2023-10-26T23:42:29+00:00)\nBut more modest attendances in subsequent games, including two sub-2,000 crowds in Perth, suggest the popularity of the Matildas’
run to the semi-finals has only had a limited impact on support for domestic women’s sport.\n\nskip past newsletter promotion Sign up to Australia Sport Free
newsletter Get a daily roundup of the latest sports news, features and comment from our Australian sports desk Privacy Notice: Newsletters may contain info
about charities, online ads, and content funded by outside parties. For more information see our Newsletters may contain info about charities, online ads, and
content funded by outside parties. For more information see our Privacy Policy . We use Google reCaptcha to protect our website and the Google Privacy Policy
and Terms of Service apply. after newsletter promotion\n\nLivingstone said the “green and gold fever” associated with the Women’s World Cup was a one-off and
the AFL has plans in place for long-term growth.', 'Craft beer and cheese boxes: Swans tap into something ‘uniquely Sydney’ to draw AFLW crowds | Jack Snape
(The Guardian, 2023-10-26T23:42:29+00:00)\n“We’ll continue to keep building the base and keep building the foundations and have it not just strong at year
seven, season eight, but strong for generations to come.”\n\nIn a bid to improve the spectacle of the AFLW, the league introduced longer quarters this year and
an interchange cap to help increase scoring. The effect, according to Livingstone, has been positive.\n\n“We’ve increased game time by eight minutes per match,
on average, and we’ve had record scores across the rounds,” she said. “Whereas leading into the season we hadn’t scored more than 100 goals in a round, we’ve
regularly been scoring over 100 goals in a round.”\n\nThe Swans recorded their highest total last round, in their 14-12-96 to 6-3-39 defeat of the Western
Bulldogs, the league’s winless bottom side. Molloy recorded 17 score involvements to set a new AFLW mark.\n\nBut Livingstone said the AFL was happy with the
level of competition.', 'Craft beer and cheese boxes: Swans tap into something ‘uniquely Sydney’ to draw AFLW crowds | Jack Snape (The Guardian,
2023-10-26T23:42:29+00:00)\nBut Livingstone said the AFL was happy with the level of competition. “Our ladder, from position five to 13 – there’s only a game
separating it. It just shows that this is a really competitive competition.”']
</item 1>

<item 2>
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
</item 2>

<item 3>
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
</item 3>

<item 4>
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
</item 4>

<item 5>
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
</item 5>

<item 6>
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
Craft beer and cheese boxes: Swans tap into something ‘uniquely Sydney’ to draw AFLW crowds | Jack Snape (The Guardian, 2023-10-26T23:42:29+00:00)
But more modest attendances in subsequent games, including two sub-2,000 crowds in Perth, suggest the popularity of the Matildas’ run to the semi-finals has
only had a limited impact on support for domestic women’s sport.

skip past newsletter promotion Sign up to Australia Sport Free newsletter Get a daily roundup of the latest sports news, features and comment from our
Australian sports desk Privacy Notice: Newsletters may contain info about charities, online ads, and content funded by outside parties. For more information see
our Newsletters may contain info about charities, online ads, and content funded by outside parties. For more information see our Privacy Policy . We use Google
reCaptcha to protect our website and the Google Privacy Policy and Terms of Service apply. after newsletter promotion

Livingstone said the “green and gold fever” associated with the Women’s World Cup was a one-off and the AFL has plans in place for long-term growth.

Graph facts:
- West Indies v England: first T20 international – live -MENTIONS-> Livingstone
- Livingstone -PARTICIPATED_IN-> West Indies v England: first T20 international
- Livingstone -SAID-> game time and scoring
- West Indies v England: fifth and deciding men’s T20 cricket international – live -MENTIONS-> Livingstone
- Livingstone -PARTICIPATED_IN-> first T20 international

Entities: Livingstone, Jack Snape, AFLW, Craft beer and cheese boxes: Swans tap into something ‘uniquely Sydney’ to draw AFLW crowds, Swans, Matildas, domestic
women’s sport, long-term growth, Perth, AFL, Sydney, The Guardian, Women’s World Cup
</item 8>

<item 9>
‘Biggest win of my career’: De Minaur, Popyrin power Australia into Davis Cup final (The Age, 2023-11-24T23:10:22+00:00)
The second of Sinner’s Djokovic victories came in Italy’s impressive Davis Cup semi-final triumph over Serbia, after saving three consecutive match points on
serve, before besting the world No.1 again in the doubles. World No.4 Sinner, coached by Australia’s Darren Cahill, is the first player to topple Djokovic twice
in the same season since Russia’s Daniil Medvedev in 2019. Italian teammates Lorenzo Sonego and Jannik Sinner will face Australia in the Davis Cup final.
Credit: Getty However, if Popyrin can get the job done in the No.2 singles then Australia will, at minimum, reach the doubles with a live chance, and a good
one, with 2022 Wimbledon champions Max Purcell and Matt Ebden to be unleashed.

Graph facts:
- People are turning to Snap Map for firsthand perspectives from Gaza -MENTIONS-> Russia
- 'Too big to fail' Egypt faces economic crisis but has an anti-Gaza refugee policy. Israel and the U.S. hope that will change -MENTIONS-> Russia
- Hitting the Books: We are the frogs in the boiling pot, it's time we started governing like it. -MENTIONS-> Russia
- Deepfake election risks trigger EU call for more generative AI safeguards -MENTIONS-> Russia
- Dissecting impact of Israel-Hamas conflict on markets -MENTIONS-> Russia

Entities: Russia, The Age, Australia, Jannik Sinner, ‘Biggest win of my career’: De Minaur, Popyrin power Australia into Davis Cup final, Darren Cahill, Daniil
Medvedev, Lorenzo Sonego, Max Purcell, Matt Ebden, Italy, Davis Cup semi-final triumph over Serbia, 2022 Wimbledon championships, Popyrin, Davis Cup final
</item 9>

<item 10>
‘Biggest win of my career’: De Minaur, Popyrin power Australia into Davis Cup final (The Age, 2023-11-24T23:10:22+00:00)
Credit: AP It had followed a superb win from teammate Popyrin, who defied jangling nerves to earn a 7-6 (7-5), 6-2 victory over Otto Virtanen. It meant the
Australians’ crack doubles team of Max Purcell and Matt Ebden weren’t even needed.

It will be Australia’s 49th appearance in the Davis Cup final, as they seek the trophy for the 29th t28-time champions Australia, in a 49th final, seek a first
win in the World Cup of men’s tennis in 20 years. “We’re a very solid squad, a lot of depth in every single position, and if we can get ourselves to a deciding
doubles, we’ve got an unbelievable pairing as well,” de Minaur said. “So we singles guys have just got do our best to try and get a win and let the doubles guys
do the job.

Graph facts:
- How ego and fear fuelled the rise of artificial intelligence -PUBLISHED_BY-> The Age
- ASX set to open higher as Wall Street rebounds; $A rises -PUBLISHED_BY-> The Age
- Is Google Search better than the rest? And is that fair? -PUBLISHED_BY-> The Age
- When pop culture and sport collide: A timeline of Taylor Swift’s NFL takeover -PUBLISHED_BY-> The Age
- At 37, Alexandra found herself broke, heartbroken and living with her parents -PUBLISHED_BY-> The Age

Entities: The Age, Australia, ‘Biggest win of my career’: De Minaur, Popyrin power Australia into Davis Cup final, Max Purcell, Matt Ebden, Popyrin, Davis Cup
final, De Minaur, Otto Virtanen
</item 10>
=== END ITEM id=37cff9fa118f7019 ===

=== ITEM id=a1241d74729b5242 ===
Gold sentences and facts:
Sentence 0: Kooker added later in the speech: “Music is a tremendous driver for AI technology, and AI technology presents a tremendous opportunity for the creative development of music.
  Fact 0.0: Kooker added later in the speech.
  Fact 0.1: Kooker said music is a tremendous driver for AI technology.
  Fact 0.2: Kooker said AI technology presents a tremendous opportunity for the creative development of music.
Sentence 1: Robert Kyncl, CEO of Warner Music Group – and an ex-high flyer at YouTube – commented: “With each major leap in technology, the music industry navigates a fresh set of challenges and opportunities.
  Fact 1.0: Robert Kyncl is the CEO of Warner Music Group.
  Fact 1.1: Robert Kyncl is an ex-high flyer at YouTube.
  Fact 1.2: Robert Kyncl commented that with each major leap in technology, the music industry navigates a fresh set of challenges and opportunities.

Retrieved items:
<item 1>
Graph query row:
music_business_worldwide_article: Sony Music’s artists aren’t involved in YouTube’s new voice-cloning AI experiment
robert_kyncl_passage: Sony Music’s artists aren’t involved in YouTube’s new voice-cloning AI experiment. Not unrelated: Google’s recent filing with the US
Copyright Office. (Music Business Worldwide, 2023-11-23T18:48:48+00:00)
Especially when SME has shown its own willingness to experiment with generative AI music platforms this week via this announcement about a project with David
Gilmour, The Orb, and AI company Vermillio?

Sources close to Sony Music’s HQ in New York have suggested to MBW that the company is taking an “artist-led” approach to its experiments with any AI platform
that can manipulate the work (or voice) of its roster.

In the case of YouTube’s ‘Dream Track‘, MBW understands, SME first offered the opportunity to participate to a number of its artists, but this group of acts did
not show enthusiasm about participating.

(If one of these artists was Bad Bunny, signed to Sony’s The Orchard, you can see why: The Puerto Rican superstar expressed fury the other week about a popular
track on TikTok that had cloned his vocal stylings via AI.)
dennis_kooker_article: Sony Music has issued nearly 10,000 deepfake takedowns… and other things we learned from Dennis Kooker’s speech about AI
dennis_kooker_passage: Sony Music has issued nearly 10,000 deepfake takedowns… and other things we learned from Dennis Kooker’s speech about AI (Music Business
Worldwide, 2023-11-30T21:28:03+00:00)
Senior political figures in Washington are becoming increasingly interested in the impact of AI on the law.

On Wednesday (November 29), US Senate Majority Leader Chuck Schumer held an Artificial Intelligence (AI) Insight Forum to discuss the importance and role of
creative copyright and intellectual property in the development of AI.

Dennis Kooker, Sony Music Entertainment‘s President of Global Digital Business & US Sales, delivered a speech at the Forum, outlining the major music company’s
position on AI and copyright law.

Kooker’s public statements on the topic arrive a week after MBW pointed out that there was a notable lack of endorsement from SME, and its artists, of YouTube‘s
new experimental AI project called ‘Dream Track’ that lets creators clone the vocals of successful musicians.

The initial cohort of stars involved in the project, as we noted last week, include recording artists signed to Warner Music Group and Universal Music Group –
or labels within each of those two majors, but no Sony Music artists.
connection: None
</item 1>

<item 2>
Sony Music has issued nearly 10,000 deepfake takedowns… and other things we learned from Dennis Kooker’s speech about AI (Music Business Worldwide,
2023-11-30T21:28:03+00:00)
(In our report, MBW suggested that YouTube parent Google‘s recent submission to The United States Copyright Office (USCO) – in response to a request for written
submissions as part of a study around copyright law and policy issues raised by AI systems – may have influenced Sony’s absence from Dream Track.)

Google was just one of various tech giants and AI companies that submitted responses to the USCO, amongst them, Anthropic and Stability AI.

Anthropic, in its submission to the USCO, also argued that training (large language models) LLM’s on copyrighted material is fair use.

Stability AI made a similar argument. Its position on the issue of “Fair Use” concerning AI and copyrighted content resulted in the resignation of prominent
generative AI executive Ed Newton-Rex from the company.

As reported by MBW last week, Universal Music Group submitted its response to the USCO.

Graph facts:
- ChatGPT: Everything you need to know about the AI-powered chatbot -MENTIONS-> AI systems
- Biden’s AI EO hailed as broad, but not deep without legislation to match -MENTIONS-> AI systems
- Biden’s AI EO -MENTIONS-> AI systems
- Sony Music’s artists aren’t involved in YouTube’s new voice-cloning AI experiment -MENTIONS-> AI systems
- Sam Altman ousted as OpenAI’s CEO -MENTIONS-> AI systems

Entities: AI systems, large language models, AI, Google, Universal Music Group, Anthropic, Music Business Worldwide, Dream Track, Sony Music has issued nearly
10,000 deepfake takedowns… and other things we learned from Dennis Kooker’s speech about AI, United States Copyright Office, copyright law and policy issues,
Stability AI, fair use, copyrighted material, Ed Newton-Rex
</item 2>

<item 3>
Sony Music has issued nearly 10,000 deepfake takedowns… and other things we learned from Dennis Kooker’s speech about AI (Music Business Worldwide,
2023-11-30T21:28:03+00:00)
He added that “platforms are quick to point to the loopholes in the law as an excuse to drag their feet or to not take the deepfakes down when requested.”

Kooker commended the No FAKES Act policy drafted by senators earlier this year, which he explained “would create a federal property right in one’s voice or
visual likeness and protect against unauthorized AI-generated replicas”.

2. Sony has ‘roughly 200 active conversations taking place’ with AI startups that “include potential equity investments”

Kooker explained that despite the negative impacts of AI, including unauthorized clones and low-quality AI music, there are still “many positive and optimistic
developments to highlight”.

He highlighted legitimate AI startups and established firms “with music ideas that want to partner with the industry”.

According to Kooker, Sony Music has “roughly 200 active conversations taking place with startups and established players” right now about building new
AI-related products and tools.

Graph facts:
- News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line -MENTIONS-> AI
- There’s something going on with AI startups in France -MENTIONS-> AI
- Founders, are events useful? -MENTIONS-> AI
- One year later, ChatGPT is still alive and kicking -MENTIONS-> AI
- Sam Altman backs teens’ startup, Google unveils the Pixel 8 and TikTok tests an ad-free tier -MENTIONS-> AI

Entities: AI, Music Business Worldwide, Sony Music has issued nearly 10,000 deepfake takedowns… and other things we learned from Dennis Kooker’s speech about
AI, Dennis Kooker, deepfakes, No FAKES Act, AI startups, AI-generated music, Sony Music
</item 3>

<item 4>
Sony Music has issued nearly 10,000 deepfake takedowns… and other things we learned from Dennis Kooker’s speech about AI (Music Business Worldwide,
2023-11-30T21:28:03+00:00)
Dennis Kooker, Sony Music

Kooker added later in the speech: “Music is a tremendous driver for AI technology, and AI technology presents a tremendous opportunity for the creative
development of music.

“But these opportunities must be grounded by the human creators’ vision with the machine assisting, not with the machine replacing the human creator.”

To achieve this, Kooker said that SME wants Congress to embrace the following principals:

Assure Consent, Compensation, and Credit. “New products and businesses built with music must be developed with the consent of the owner and appropriate
compensation and credit. It is essential to understand why the training of AI models is being done, what products will be developed as a result, and what the
business model is that will monetize the use of the artist’s work. Congress and the agencies should assure that creators’ rights are recognized and respected.”
Confirm That Copying Music to Train AI Models is Not Fair Use.

Graph facts:
- Deepfake election risks trigger EU call for more generative AI safeguards -MENTIONS-> AI models
- Mayo Clinic sees AI as 'transformative force' in health care, appoints Dr. Bhavik Patel as chief AI officer -MENTIONS-> AI models
- EU lawmakers bag late night deal on ‘global first’ AI rules -MENTIONS-> AI models
- Bhavik Patel -SAID-> AI models
- Sam Bankman-Fried didn’t ask where the $8 billion went -MENTIONS-> Congress

Entities: AI models, Congress, Music Business Worldwide, Sony Music has issued nearly 10,000 deepfake takedowns… and other things we learned from Dennis
Kooker’s speech about AI, fair use, Dennis Kooker, Sony Music, AI technology, Creators’ rights, music
</item 4>

<item 5>
Sony Music has issued nearly 10,000 deepfake takedowns… and other things we learned from Dennis Kooker’s speech about AI (Music Business Worldwide,
2023-11-30T21:28:03+00:00)
These products range from tools for “creative or marketing assistance”, to tools that “potentially give us the ability to better protect artist content or find
it when used in an unauthorized fashion” as well as “brand new products that have never been launched before”.

Kooker said that some of these conversations “also include potential equity investments which would accelerate the development of these companies”.

The Sony Music exec cited, as an example, a recent generative AI-powered project around a reissue and remix of an album. Although Kooker didn’t name it
specifically, SME did recently announce a generative AI project with The Orb and David Gilmour, which was a partnership between Sony Music Entertainment, Legacy
Recordings AI company Vermillio

“These artists are known for their cutting-edge experimentation in music,” said Kooker.

Graph facts:
- AfroFuture’s vision is to bring African art, cuisine, fashion, and music to a global audience. -PUBLISHED_BY-> Music Business Worldwide
- Latin artists have been paving the way for deals that offer more control, transparency, and better economics. -PUBLISHED_BY-> Music Business Worldwide
- Michelle Jubelirer on Ice Spice, The Beatles, and reinventing Capitol Music Group -PUBLISHED_BY-> Music Business Worldwide
- Michelle Jubelirer on Ice Spice, The Beatles, and reinventing Capitol Music Group: ‘I don’t think any label group has been able to turn things around as
quickly as my team and I have done in less than two years.’ -PUBLISHED_BY-> Music Business Worldwide
- ‘AfroFuture’s vision is to bring African art, cuisine, fashion, and music to a global audience.’ -PUBLISHED_BY-> Music Business Worldwide

Entities: Music Business Worldwide, Sony Music has issued nearly 10,000 deepfake takedowns… and other things we learned from Dennis Kooker’s speech about AI,
Dennis Kooker, Sony Music Entertainment, The Orb, David Gilmour, Legacy Recordings, Vermillio, generative AI project around a reissue and remix of an album,
potential equity investments, generative AI
</item 5>

<item 6>
Sony Music’s artists aren’t involved in YouTube’s new voice-cloning AI experiment. Not unrelated: Google’s recent filing with the US Copyright Office. (Music
Business Worldwide, 2023-11-23T18:48:48+00:00)
Yet something conspicuous is not on there: any endorsement whatsoever from the world’s second-biggest recording company, Sony Music Entertainment, or its
artists.

Why is Sony missing?

MBW understands that Sony Music (the recorded music arm of Sony Music Group) has, like Universal and Warner, been in discussions for months with YouTube over
potentially licensing the ‘Dream Track’ experiment.

During this period of discussion, YouTube has made a number of positive announcements regarding the biggest issue for any rightsholder regarding AI-driven voice
replication of artists: their ability to police it.

In August, YouTube and Universal Music Group jointly announced that the video platform was launching an ‘AI Music Incubator’ – a program by which new tools and
innovations will be developed at YouTube in close conjunction with artists and the music biz.

Graph facts:
- Google -MENTIONS-> AI Principles
- Sam Altman backs teens’ startup, Google unveils the Pixel 8 and TikTok tests an ad-free tier -MENTIONS-> Google
- Apple defends Google Search deal in court: ‘There wasn’t a valid alternative’ -MENTIONS-> Google
- More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo -MENTIONS-> Google
- Google -MENTIONS-> Apple

Entities: Google, Universal Music Group, US Copyright Office, Music Business Worldwide, Dream Track, Sony Music, Sony Music Group, Warner, AI Music Incubator,
AI-driven voice replication of artists, YouTube, Sony Music’s artists aren’t involved in YouTube’s new voice-cloning AI experiment.
</item 6>

<item 7>
Sony Music has issued nearly 10,000 deepfake takedowns… and other things we learned from Dennis Kooker’s speech about AI (Music Business Worldwide,
2023-11-30T21:28:03+00:00)
Senior political figures in Washington are becoming increasingly interested in the impact of AI on the law.

On Wednesday (November 29), US Senate Majority Leader Chuck Schumer held an Artificial Intelligence (AI) Insight Forum to discuss the importance and role of
creative copyright and intellectual property in the development of AI.

Dennis Kooker, Sony Music Entertainment‘s President of Global Digital Business & US Sales, delivered a speech at the Forum, outlining the major music company’s
position on AI and copyright law.

Kooker’s public statements on the topic arrive a week after MBW pointed out that there was a notable lack of endorsement from SME, and its artists, of YouTube‘s
new experimental AI project called ‘Dream Track’ that lets creators clone the vocals of successful musicians.

The initial cohort of stars involved in the project, as we noted last week, include recording artists signed to Warner Music Group and Universal Music Group –
or labels within each of those two majors, but no Sony Music artists.

Graph facts:
- How the conspiracy-fueled Epoch Times went mainstream and made millions -MENTIONS-> Washington
- La desigualdad del estrés por calor -MENTIONS-> Washington
- NFL roundup: Pats hit new low as Lions reach 9-3 for first time since 1962 -MENTIONS-> Washington
- 'Too big to fail' Egypt faces economic crisis but has an anti-Gaza refugee policy. Israel and the U.S. hope that will change -MENTIONS-> Washington
- ChatGPT: Everything you need to know about the AI-powered chatbot -MENTIONS-> Washington

Entities: Washington, AI, Warner Music Group, copyright law, Universal Music Group, US Senate, Chuck Schumer, Music Business Worldwide, Artificial Intelligence
(AI) Insight Forum, Dream Track, creative copyright, intellectual property, Sony Music has issued nearly 10,000 deepfake takedowns… and other things we learned
from Dennis Kooker’s speech about AI, Dennis Kooker, Sony Music Entertainment
</item 7>

<item 8>
Sony Music has issued nearly 10,000 deepfake takedowns… and other things we learned from Dennis Kooker’s speech about AI (Music Business Worldwide,
2023-11-30T21:28:03+00:00)
Kooker also openly discussed Sony Music’s efforts to get unauthorized AI content removed from platforms online – including the fact that the firm has already
issued close to 10,000 separate takedowns.

Kooker further discussed Sony Music’s optimism about AI’s potential for enhancing creativity, as well as key principles the company has adopted around
generative AI.

Here are three things that stood out from Kooker’s speech…

1. Sony Music has issued close to 10,000 takedowns for unauthorized deep fakes of artists

Kooker argued in his speech that, while “the music industry sees great potential with various forms of AI technology in this early phase of AI,” so far, “the
available generative AI products are not delivering on the expected promise of new products in creative industries”.

He clarified that point by explaining that some generative AI platforms “are not expanding the business model or enhancing human creativity”.

Graph facts:
- News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line -MENTIONS-> AI
- There’s something going on with AI startups in France -MENTIONS-> AI
- Founders, are events useful? -MENTIONS-> AI
- One year later, ChatGPT is still alive and kicking -MENTIONS-> AI
- Sam Altman backs teens’ startup, Google unveils the Pixel 8 and TikTok tests an ad-free tier -MENTIONS-> AI

Entities: AI, human creativity, Music Business Worldwide, Sony Music has issued nearly 10,000 deepfake takedowns… and other things we learned from Dennis
Kooker’s speech about AI, Dennis Kooker’s speech about AI, deep fakes, Dennis Kooker, Sony Music, music industry, generative AI
</item 8>

<item 9>
Warner Music Group boss Robert Kyncl on AI, why labels still exist, and 2 other things we learned from his Q&A at the Code Conference (Music Business Worldwide,
2023-09-28T23:37:22+00:00)
In the meantime, music emerged out of [digitization] better and stronger and more resilient. So, I would imagine the same will happen here, which is we’ll
likely be first.”

Asked when that sea change will occur, Kyncl said: “I would say within the next year you will see lots of evolution around AI… What you will likely see is
increasing quality at a very fast pace.”

Kyncl drew an analogy between the explosion of AI use among members of the public and the explosion of user-generated content (UGC) a decade or two ago. Just as
user-generated content often infringed on copyright, so too does AI threaten the rights of artists and music rights holders.

“I would say within the next year you will see lots of evolution around AI… What you will likely see is increasing quality at a very fast pace.”

Graph facts:
- How OpenAI's ChatGPT has changed the world in just a year -MENTIONS-> copyright
- NMPA: Generative AI is ‘the greatest risk to the human creative class that has ever existed’ -MENTIONS-> copyright
- 8 predictions for AI in 2024 -MENTIONS-> copyright
- Sony Music has issued nearly 10,000 deepfake takedowns… and other things we learned from Dennis Kooker’s speech about AI -MENTIONS-> copyright
- ChatGPT: Everything you need to know about the AI-powered chatbot -MENTIONS-> copyright

Entities: copyright, AI, Warner Music Group, Music Business Worldwide, Warner Music Group boss Robert Kyncl on AI, why labels still exist, and 2 other things we
learned from his Q&A at the Code Conference, Robert Kyncl, digitization, user-generated content (UGC), Code Conference
</item 9>

<item 10>
Sony Music has issued nearly 10,000 deepfake takedowns… and other things we learned from Dennis Kooker’s speech about AI (Music Business Worldwide,
2023-11-30T21:28:03+00:00)
“About the time that we started the discussion with this artist, we had begun informal talks with a generative AI start-up company whose business model focused
on working with IP rightsholders the ‘right way’.

“In other words, they respect intellectual property rights and want to work with rights holders in ways that enhance and protect the copyrighted works.”

3. SME has asked Congress to embrace a set of principles around generative AI

Kooker told the lawmakers during his speech on Wednesday that “if copyrights are protected and enforced appropriately, we are at the beginning of a multi-decade
marathon that will change the creative and commercial landscape for music”.

With that in mind, Kooker explained that Sony Music has established a set of principles to guide the company’s decision-making around generative AI.

“If copyrights are protected and enforced appropriately, we are at the beginning of a multi-decade marathon that will change the creative and commercial
landscape for music.”

Graph facts:
- Sen. Robert Menendez, facing bribery charges, stumps bank CEOs by asking how much their firms paid in settlements for allegedly ripping off consumers
-MENTIONS-> Congress
- Biden’s AI EO hailed as broad, but not deep without legislation to match -MENTIONS-> Congress
- How OpenAI's ChatGPT has changed the world in just a year -MENTIONS-> Congress
- Meta seeks legislation that would require parents to approve teens’ app downloads -MENTIONS-> Congress
- Sam Bankman-Fried didn’t ask where the $8 billion went -MENTIONS-> Congress

Entities: Congress, Music Business Worldwide, Sony Music has issued nearly 10,000 deepfake takedowns… and other things we learned from Dennis Kooker’s speech
about AI, Dennis Kooker’s speech about AI, Dennis Kooker, SME, generative AI start-up company, intellectual property rights, copyrights, deepfake takedowns,
creative and commercial landscape, Sony Music, generative AI, music
</item 10>
=== END ITEM id=a1241d74729b5242 ===

=== ITEM id=4d753189b61a387e ===
Gold sentences and facts:
Sentence 0: “It’s not the NHL that he was playing in the last few years,” Davidson said, “but the level of performance and how he could control a game and dominate a game in junior — it indicated he was going to be able to do that at some point in the NHL.
  Fact 0.0: Davidson said the player was not playing in the NHL in the last few years.
  Fact 0.1: Davidson said the level of performance showed the player could control a game and dominate a game in junior.
  Fact 0.2: Davidson said this performance indicated the player was going to be able to do that at some point in the NHL.
Sentence 1: There, he helped guide more than a few future NHL stalwarts — namely Mattias Ekholm, Ryan Ellis, Filip Forsberg, Calle Järnkrok, Viktor Arvidsson, and Kevin Fiala, among many others.
  Fact 1.0: The player helped guide more than a few future NHL stalwarts at that place.
  Fact 1.1: Mattias Ekholm, Ryan Ellis, Filip Forsberg, Calle Järnkrok, Viktor Arvidsson, and Kevin Fiala were among the future NHL stalwarts guided there.
  Fact 1.2: Many other future NHL stalwarts besides those named were also guided there.

Retrieved items:
<item 1>
He’s Hockey’s Brightest Young Star. This Is What Makes His Shot So Special. (The New York Times, 2023-11-17T21:47:19+00:00)
“So many players pull the puck in but can’t get the puck to release from under their body. His footwork is so underrated in that aspect. If you watch his front
leg, at times it’s literally in the air at release — most have their back leg in the air at release. When he transfers his weight, he actually clears space for
his hands to get tighter because he’s not afraid to actually lunge into the shot.”

A fourth shooting coach, who hasn’t worked with Bedard and also requested anonymity because he works for another NHL club, expressed a little more hesitancy.

Advertisement

“I’m in the minority I’m sure, but I am very curious to see how well his trademark shooting style works at this level,” the coach said. “He really likes to pull
pucks right into his feet.

Graph facts:
- Which M.L.B. Playoff Team Has the Most Daunting Home Field? We Ranked Them. -PUBLISHED_BY-> The New York Times
- How U.S. Women’s Soccer Team Lured Emma Hayes Away From Chelsea -PUBLISHED_BY-> The New York Times
- After a Stunning Snub, College Football Is Facing an Identity Crisis -PUBLISHED_BY-> The New York Times
- N.F.L. Executives on the Biggest Questions Facing Super Bowl Contenders -PUBLISHED_BY-> The New York Times
- Alcoholism, Ayahuasca and the Enlightenment of an N.F.L. Player -PUBLISHED_BY-> The New York Times

Entities: The New York Times, He’s Hockey’s Brightest Young Star. This Is What Makes His Shot So Special., fourth shooting coach, Bedard, NHL, Hockey
</item 1>

<item 2>
He’s Hockey’s Brightest Young Star. This Is What Makes His Shot So Special. (The New York Times, 2023-11-17T21:47:19+00:00)
“It’s a lot of different angles he can give you.”

Sabres first-round pick Zach Benson, who has played on teams with and against Bedard and skates with him in the summers, says it’s Bedard’s ability to set up
the shot with his agility pre-shot that stands out to him.

“He can move left to right like no one I’ve seen before,” Benson said.

“It’s just trying to get out, face him, and hope it hits ya,” said Scott Ratzlaff, Seattle Thunderbirds goalie and Buffalo Sabres prospect, with a smile. “He
can shoot from anywhere and he’s lethal from anywhere. You’ve just always got to be ready just in case he shoots it. And then he’s got a really good toe drag
release, so it’s watching for that and making sure you’re lined up.”

A number of Chicago Blackhawks over the last few decades liked to put in extra time shooting.

Graph facts:
- Which M.L.B. Playoff Team Has the Most Daunting Home Field? We Ranked Them. -PUBLISHED_BY-> The New York Times
- How U.S. Women’s Soccer Team Lured Emma Hayes Away From Chelsea -PUBLISHED_BY-> The New York Times
- After a Stunning Snub, College Football Is Facing an Identity Crisis -PUBLISHED_BY-> The New York Times
- N.F.L. Executives on the Biggest Questions Facing Super Bowl Contenders -PUBLISHED_BY-> The New York Times
- Alcoholism, Ayahuasca and the Enlightenment of an N.F.L. Player -PUBLISHED_BY-> The New York Times

Entities: The New York Times, He’s Hockey’s Brightest Young Star. This Is What Makes His Shot So Special., Zach Benson, Bedard, Scott Ratzlaff, Buffalo Sabres,
Seattle Thunderbirds, Chicago Blackhawks, Hockey
</item 2>

<item 3>
He’s Hockey’s Brightest Young Star. This Is What Makes His Shot So Special. (The New York Times, 2023-11-17T21:47:19+00:00)
What is it, really, about Connor Bedard’s shot — a shot that, though he’s just 18 years old, has for years been talked about as if he has patented it — that
makes it so dangerous and unique?

What’s the nerdy science of it, from his feet up to his knees, hips, hands and head? How does he prep, shape and let it go? What does it look like to goalies,
and to professional shooting coaches, those who’ve taught him — or, more accurately, watched him after he taught himself — and had to stop him? The Athletic
spoke with five NHL shooting and skills coaches, his teammates, past opponents and him to try to take apart, piece by piece, “The Bedard.”

Advertisement

Their intel and stories help frame the NHL’s brightest young star’s signature skill.

If there’s one thing Tim Turk knows, it’s shooting.

Graph facts:
- How Did the Houston Astros Cheat in 2017? Everything to Know About a Dark Moment in MLB History -MENTIONS-> The Athletic
- Michigan sign-stealing scandal winners and losers, from Jim Harbaugh to Tony Petitti -MENTIONS-> The Athletic
- Browns survive Bears Hail Mary on wild, near-catch by Darnell Mooney: 'Damn near s— myself' -MENTIONS-> The Athletic
- Which M.L.B. Playoff Team Has the Most Daunting Home Field? We Ranked Them. -MENTIONS-> The Athletic
- How U.S. Women’s Soccer Team Lured Emma Hayes Away From Chelsea -MENTIONS-> The Athletic

Entities: The Athletic, The New York Times, The Bedard, shooting, He’s Hockey’s Brightest Young Star. This Is What Makes His Shot So Special., Tim Turk, NHL,
Connor Bedard, Hockey
</item 3>

<item 4>
He’s Hockey’s Brightest Young Star. This Is What Makes His Shot So Special. (The New York Times, 2023-11-17T21:47:19+00:00)
Bedard wasn’t so reliant on teammates on those goals, either. They were more about the work he’s putting in on the other side of the puck.

“Even in practice, we’ve talked to him about maybe tracking harder and attacking pucks on the forecheck and showing him a couple clips,” Richardson said. “Ten,
12 games in, he’s really figuring things out and realizes — why sit back and let things come to him? Go get it.”

None of this is surprising to anyone involved. Bedard doesn’t show it if he impresses himself. And as grateful as the Blackhawks are to have drafted Bedard,
this is what they expected. This is why teams lined up to take losses last season.

“It’s not the NHL that he was playing in the last few years,” Davidson said, “but the level of performance and how he could control a game and dominate a game
in junior — it indicated he was going to be able to do that at some point in the NHL.

Graph facts:
- Which M.L.B. Playoff Team Has the Most Daunting Home Field? We Ranked Them. -PUBLISHED_BY-> The New York Times
- How U.S. Women’s Soccer Team Lured Emma Hayes Away From Chelsea -PUBLISHED_BY-> The New York Times
- After a Stunning Snub, College Football Is Facing an Identity Crisis -PUBLISHED_BY-> The New York Times
- N.F.L. Executives on the Biggest Questions Facing Super Bowl Contenders -PUBLISHED_BY-> The New York Times
- Alcoholism, Ayahuasca and the Enlightenment of an N.F.L. Player -PUBLISHED_BY-> The New York Times

Entities: The New York Times, He’s Hockey’s Brightest Young Star. This Is What Makes His Shot So Special., Bedard, Richardson, Davidson, NHL, Blackhawks, Hockey
</item 4>

<item 5>
He’s Hockey’s Brightest Young Star. This Is What Makes His Shot So Special. (The New York Times, 2023-11-17T21:47:19+00:00)
“To me, when he takes a shot, it looks like he just bends down, picks it up with his hand and places it where he wants to.

“He believes that ‘Hey, f— it, the puck is only two and a half inches wide, I can put it wherever I want.'”

Today, Turk would take Bedard’s shot against anyone’s.

Advertisement

“On a shot release basis only, if I had to bet on who could get the puck off the quickest, with the most deception, with optimal speed, power and accuracy based
on a starting point to a finishing point, I’m picking Connor Bedard over Auston Matthews,” Turk said. “And I’m not taking anything away from Auston. It’s just a
little bit different because one’s a righty, one’s a lefty and one’s got a little bit of a higher-angle pull-in change.”

If there’s one thing Nick Quinn wants people to know about “The Bedard,” it’s that it isn’t just some natural gift.

Graph facts:
- Which M.L.B. Playoff Team Has the Most Daunting Home Field? We Ranked Them. -PUBLISHED_BY-> The New York Times
- How U.S. Women’s Soccer Team Lured Emma Hayes Away From Chelsea -PUBLISHED_BY-> The New York Times
- After a Stunning Snub, College Football Is Facing an Identity Crisis -PUBLISHED_BY-> The New York Times
- N.F.L. Executives on the Biggest Questions Facing Super Bowl Contenders -PUBLISHED_BY-> The New York Times
- Alcoholism, Ayahuasca and the Enlightenment of an N.F.L. Player -PUBLISHED_BY-> The New York Times

Entities: The New York Times, He’s Hockey’s Brightest Young Star. This Is What Makes His Shot So Special., Turk, Auston Matthews, Nick Quinn, The Bedard, Connor
Bedard, Hockey
</item 5>

<item 6>
He’s Hockey’s Brightest Young Star. This Is What Makes His Shot So Special. (The New York Times, 2023-11-17T21:47:19+00:00)
9 pick) went first, and Turk could tell he was a talented and hard-working player with a good shot and real promise.

Then, on Connor’s turn, Turk placed him in a position and watched how he got the puck ready to shoot. Only Connor didn’t stickhandle and prepare for the shot,
he just took the puck and shot it right away.

After watching him run through the same number of shots as Nate did, Turk spoke for the first time.

“Timeout one second, just give me a second here, I’m just trying to see what you’re doing so I’m going to have you take an extra few shots so that I can analyze
where the puck’s starting and where it’s finishing,” he said.

Another round later, he knew that this Connor kid, whose last name he still didn’t know and who was then 16, was very different from even his high-end peers
simply by his proficiency in shooting from a “frontal position.”
</item 6>

<item 7>
He’s Hockey’s Brightest Young Star. This Is What Makes His Shot So Special. (The New York Times, 2023-11-17T21:47:19+00:00)
If there’s one thing Tim Turk knows, it’s shooting. A self-described “shooting and scoring coach,” Turk has more than two decades of experience working with NHL
stars, national federations and half a dozen NHL teams.

Turk had no idea who Bedard was when he fielded a call from agent Greg Landry of Newport Sports asking him what he was doing tomorrow.

“Tomorrow?” he said. “I’m booked up a year in advance.”

“We have these two players coming in from out west and we want you to see them,” Landry said.

“Listen, I’m booked up,” Turk said again.

“You have to see them,” Landry insisted.

After moving around his schedule, Turk found himself at Gary Roberts’ personal rink in Uxbridge, Ontario.

“Hey guys, I’m Tim Turk,” he said as he stepped into the small locker room.

“My name’s Nate,” said the redhead, extending a hand.

“My name’s Connor,” said the other, standing shorter and extending his.

Graph facts:
- Which M.L.B. Playoff Team Has the Most Daunting Home Field? We Ranked Them. -PUBLISHED_BY-> The New York Times
- How U.S. Women’s Soccer Team Lured Emma Hayes Away From Chelsea -PUBLISHED_BY-> The New York Times
- After a Stunning Snub, College Football Is Facing an Identity Crisis -PUBLISHED_BY-> The New York Times
- N.F.L. Executives on the Biggest Questions Facing Super Bowl Contenders -PUBLISHED_BY-> The New York Times
- Alcoholism, Ayahuasca and the Enlightenment of an N.F.L. Player -PUBLISHED_BY-> The New York Times

Entities: The New York Times, He’s Hockey’s Brightest Young Star. This Is What Makes His Shot So Special., Greg Landry, Newport Sports, Nate, Connor, Gary
Roberts, national federations, NHL teams, Gary Roberts’ personal rink, Uxbridge, Ontario, Tim Turk, Bedard, NHL
</item 7>

<item 8>
Graph query row:
article: Wild HC Dean Evason, hire John Hynes as replacement
publisher: Yardbarker
person: Dean Evason
published_on: 2023-11-27T23:45:08+00:00
page: Wild fire HC Dean Evason, hire John Hynes as replacement (Yardbarker, 2023-11-27T23:45:08+00:00)
passage: Wild fire HC Dean Evason, hire John Hynes as replacement (Yardbarker, 2023-11-27T23:45:08+00:00)
The Wild have relieved head coach Dean Evason and assistant coach Bob Woods of their duties effective immediately, the team said in a statement Monday evening.
The Athletic’s Michael Russo reports the Wild will be naming former Devils and Predators bench boss John Hynes as Evason’s successor.

Wild GM Bill Guerin gave the following comment:

"Dean did an excellent job during his tenure with the Minnesota Wild, especially as Head Coach of our team. I am very thankful for his hard work and dedication
to our organization. I would also like to thank Bob for his hard work during his time as an Assistant Coach with the Wild. I wish Dean, Bob and their families
all the best in the future."

Evason had been a member of the Wild organization since he was brought on as an assistant in 2018, while Woods joined the Wild bench one season prior.
</item 8>

<item 9>
He’s Hockey’s Brightest Young Star. This Is What Makes His Shot So Special. (The New York Times, 2023-11-17T21:47:19+00:00)
For one, he learned to utilize his teammates more. He accumulated plenty of shots on net early in the season, but they were often from distance and more
individualistic chances. Then, he scored his first NHL goal on a wraparound and his next three goals off passes from teammates.

Advertisement

“Look, I’m sure he’ll figure out the one-on-one and how to score that way or create chances that way,” Blackhawks general manager Kyle Davidson said. “There is
also an element of working off your teammates. You are in the NHL, so everyone is a pretty good player and you’re able to create using more than just yourself.
I think he’s figured that out.”

Bedard found more success on the Blackhawks’ last road trip. He left Chicago with five goals in 11 games and returned from Florida with nine goals in 13 games,
scoring twice against the Tampa Bay Lightning and again against the Panthers.

Bedard wasn’t so reliant on teammates on those goals, either.

Graph facts:
- Which M.L.B. Playoff Team Has the Most Daunting Home Field? We Ranked Them. -PUBLISHED_BY-> The New York Times
- How U.S. Women’s Soccer Team Lured Emma Hayes Away From Chelsea -PUBLISHED_BY-> The New York Times
- After a Stunning Snub, College Football Is Facing an Identity Crisis -PUBLISHED_BY-> The New York Times
- N.F.L. Executives on the Biggest Questions Facing Super Bowl Contenders -PUBLISHED_BY-> The New York Times
- Alcoholism, Ayahuasca and the Enlightenment of an N.F.L. Player -PUBLISHED_BY-> The New York Times

Entities: The New York Times, Chicago, He’s Hockey’s Brightest Young Star. This Is What Makes His Shot So Special., Bedard, Kyle Davidson, Tampa Bay Lightning,
NHL, Panthers, Blackhawks, Florida, Hockey
</item 9>

<item 10>
He’s Hockey’s Brightest Young Star. This Is What Makes His Shot So Special. (The New York Times, 2023-11-17T21:47:19+00:00)
“I can tell you firsthand it didn’t happen by accident. Connor’s worked on this since he was little,” said Quinn, an NHL skills coach who has worked with Bedard
each summer for several years.

If there’s another, it’s how hard Bedard’s shot is to defend because he doesn’t show you he’s preparing for it.

“As replicated in so many other areas of Connor’s game, it’s the deception and elite multi-tasking that catches opponents off guard,” Quinn said. “Connor’s
ability to create deception and change the shot angle at top speed is like very few I’ve ever seen. The multi-tasking involved with attempting these shots at
top speed is far beyond most player’s capabilities.”

Another shooting coach, who requested anonymity for this story because he works for another NHL club, pointed to Bedard’s hands and legs.

“What’s really interesting is how high and left he can get his top hand,” the coach said.

Graph facts:
- Which M.L.B. Playoff Team Has the Most Daunting Home Field? We Ranked Them. -PUBLISHED_BY-> The New York Times
- How U.S. Women’s Soccer Team Lured Emma Hayes Away From Chelsea -PUBLISHED_BY-> The New York Times
- After a Stunning Snub, College Football Is Facing an Identity Crisis -PUBLISHED_BY-> The New York Times
- N.F.L. Executives on the Biggest Questions Facing Super Bowl Contenders -PUBLISHED_BY-> The New York Times
- Alcoholism, Ayahuasca and the Enlightenment of an N.F.L. Player -PUBLISHED_BY-> The New York Times

Entities: The New York Times, He’s Hockey’s Brightest Young Star. This Is What Makes His Shot So Special., Quinn, anonymous shooting coach, NHL, Connor Bedard,
Hockey
</item 10>
=== END ITEM id=4d753189b61a387e ===
