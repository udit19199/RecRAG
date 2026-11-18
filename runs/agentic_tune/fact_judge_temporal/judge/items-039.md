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

=== ITEM id=e55eaa3b744008db ===
Gold sentences and facts:
Sentence 0: “But nah, it was awesome.” On Saturday, the competition also said goodbye to a range of players including Western Bulldogs forward Richelle Cranston, who has battled stage five chronic kidney disease while playing this season, and Hawthorn’s Akec Makur Chuot, who played 40 games for three clubs.
  Fact 0.0: Someone said 'But nah, it was awesome.'
  Fact 0.1: On Saturday, the competition also said goodbye to a range of players.
  Fact 0.2: The competition said goodbye to Richelle Cranston on Saturday.
  Fact 0.3: Richelle Cranston is a Western Bulldogs forward.
  Fact 0.4: Richelle Cranston has battled stage five chronic kidney disease while playing this season.
  Fact 0.5: The competition said goodbye to Akec Makur Chuot on Saturday.
  Fact 0.6: Akec Makur Chuot is a Hawthorn player.
  Fact 0.7: Akec Makur Chuot played 40 games for three clubs.
Sentence 1: The Dogs are Cranston’s third club after starting at Melbourne and having a stint with Geelong before heading to the Bulldogs last year.
  Fact 1.0: The Dogs are Richelle Cranston's third club.
  Fact 1.1: Richelle Cranston started at Melbourne.
  Fact 1.2: Richelle Cranston had a stint with Geelong.
  Fact 1.3: Richelle Cranston joined the Bulldogs last year.

Retrieved items:
<item 1>
Ground-breaking Gold Coast: Suns make AFLW finals for a second time (The Age, 2023-11-03T23:45:04+00:00)
“But I go out with on a happy note. I love these girls and I’ll be over at the pancake pocket [Pancake Parlour at Whitten Oval] watching them from now on and
have a few bevs on the sideline.” Roos’ lock in top-four spot; Dogs have issues to work through in off-season North Melbourne secured third spot on the ladder
and the double chance in finals after downing the Western Bulldogs by a whopping 46 points. They will head into finals with momentum but a glaring issue to work
on: kicking accuracy. Despite the comfortable win, they could have done a lot more damage after booting double the number of behinds to goals.

Star midfielder Ash Riddell, who had 36 touches and seven clearances, said she was excited to go up against Melbourne or Adelaide in the first weekend of finals
depending on other results this round, as they were the only two sides who beat them this year.

Graph facts:
- How ego and fear fuelled the rise of artificial intelligence -PUBLISHED_BY-> The Age
- ASX set to open higher as Wall Street rebounds; $A rises -PUBLISHED_BY-> The Age
- Is Google Search better than the rest? And is that fair? -PUBLISHED_BY-> The Age
- When pop culture and sport collide: A timeline of Taylor Swift’s NFL takeover -PUBLISHED_BY-> The Age
- At 37, Alexandra found herself broke, heartbroken and living with her parents -PUBLISHED_BY-> The Age

Entities: The Age, Western Bulldogs, Melbourne, Ash Riddell, Adelaide, Ground-breaking Gold Coast: Suns make AFLW finals for a second time, Suns, North
Melbourne, finals, Pancake Parlour, Whitten Oval, Gold Coast
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
person: Richelle Cranston
article: Comeback kings Richmond take the AFLW points in Cairns
published_on: 2023-10-20T23:51:34+00:00
passage: Comeback kings Richmond take the AFLW points in Cairns (The Age, 2023-10-20T23:51:34+00:00)
“Tackled above waist and below shoulders. No sling or double motion. Sheppard lands on her back. ′ ‘Girls’ play hard too,” Harford said on X, the social media
site formerly known as Twitter. This echoed Collingwood defender and Fox Footy commentator Ruby Schleicher’s thoughts at the time, who called it a “monster
tackle” but believed it was “fair”. “I don’t see any dirtiness in that,” Schleicher said on Fox Footy. “It was pretty heavy on her head, wasn’t it? I look at
that – I still see a fair tackle. There’s no doubt the match review [officer] will look at it because they have to I guess. The duty of care for the player.”

The decision a week comes after Adelaide premiership defender Najwa Allen was handed a three-ban suspension, the longest ban in AFLW history, for a high bump on
Western Bulldog Kirsten McLeod. The Crows elected not to challenge the decision.
article_connections: [{'entity': 'Saints', 'relationship': 'MENTIONS'}, {'entity': 'Getty Images', 'relationship': 'MENTIONS'}, {'entity': 'X', 'relationship':
'MENTIONS'}, {'entity': 'Harford', 'relationship': 'MENTIONS'}, {'entity': 'Ruby Schleicher', 'relationship': 'MENTIONS'}, {'entity': 'Fox Footy',
'relationship': 'MENTIONS'}, {'entity': 'tackle', 'relationship': 'MENTIONS'}, {'entity': 'Najwa Allen', 'relationship': 'MENTIONS'}, {'entity': 'Western
Bulldog', 'relationship': 'MENTIONS'}, {'entity': 'Kirsten McLeod', 'relationship': 'MENTIONS'}, {'entity': 'Michael Prior', 'relationship': 'MENTIONS'},
{'entity': 'Windy Hill', 'relationship': 'MENTIONS'}, {'entity': 'Kate Bartlett', 'relationship': 'MENTIONS'}, {'entity': 'Isabella Lewis', 'relationship':
'MENTIONS'}, {'entity': 'West Coast win over Essendon at Windy Hill', 'relationship': 'MENTIONS'}, {'entity': 'West Coast', 'relationship': 'MENTIONS'},
{'entity': 'Bombers', 'relationship': 'MENTIONS'}, {'entity': 'AFLW', 'relationship': 'MENTIONS'}, {'entity': 'Aisling McCarthy', 'relationship': 'MENTIONS'},
{'entity': 'Amy Franklin', 'relationship': 'MENTIONS'}, {'entity': 'Kellie Gibson', 'relationship': 'MENTIONS'}, {'entity': 'Bonnie Toogood', 'relationship':
'MENTIONS'}, {'entity': 'Amber Clarke', 'relationship': 'MENTIONS'}, {'entity': 'Emma Swanson', 'relationship': 'MENTIONS'}, {'entity': 'Swanson',
'relationship': 'MENTIONS'}, {'entity': 'Maddy Prespakis', 'relationship': 'MENTIONS'}, {'entity': 'Georgia Nanscawen', 'relationship': 'MENTIONS'}, {'entity':
'Steph Cain', 'relationship': 'MENTIONS'}, {'entity': 'Essendon', 'relationship': 'MENTIONS'}, {'entity': 'Gold Coast', 'relationship': 'MENTIONS'}, {'entity':
'AAP', 'relationship': 'MENTIONS'}, {'entity': 'round-eight match', 'relationship': 'MENTIONS'}, {'entity': 'Victoria Park', 'relationship': 'MENTIONS'},
{'entity': 'Jordyn Allen', 'relationship': 'MENTIONS'}, {'entity': 'Claudia Gunjaca', 'relationship': 'MENTIONS'}, {'entity': 'Getty', 'relationship':
'MENTIONS'}, {'entity': 'Sabrina Frederick', 'relationship': 'MENTIONS'}, {'entity': 'Magpies', 'relationship': 'MENTIONS'}, {'entity': 'Brianna Davey',
'relationship': 'MENTIONS'}, {'entity': 'Geelong', 'relationship': 'MENTIONS'}, {'entity': 'Daniel Lowther', 'relationship': 'MENTIONS'}, {'entity': 'J’Noemi
Anderson', 'relationship': 'MENTIONS'}, {'entity': 'Dana East', 'relationship': 'MENTIONS'}, {'entity': 'Orlagh Lally', 'relationship': 'MENTIONS'}, {'entity':
'Jamiee Lambert', 'relationship': 'MENTIONS'}, {'entity': 'Hayley Miller', 'relationship': 'MENTIONS'}, {'entity': 'Fremantle Oval', 'relationship':
'MENTIONS'}, {'entity': 'Fremantle-St Kilda game', 'relationship': 'MENTIONS'}, {'entity': 'Aine Tighe', 'relationship': 'MENTIONS'}, {'entity': 'Gabby
O’Sullivan', 'relationship': 'MENTIONS'}, {'entity': 'Angelique Stannett', 'relationship': 'MENTIONS'}, {'entity': 'Ella Friend', 'relationship': 'MENTIONS'},
{'entity': 'Tyanna Smith', 'relationship': 'MENTIONS'}, {'entity': 'Richelle Cranston', 'relationship': 'MENTIONS'}, {'entity': 'Western Bulldogs',
'relationship': 'MENTIONS'}, {'entity': 'stage five chronic kidney disease', 'relationship': 'MENTIONS'}, {'entity': 'resonium', 'relationship': 'MENTIONS'},
{'entity': "Bulldogs' 57-point loss to Sydney", 'relationship': 'MENTIONS'}, {'entity': 'Scott Gowans', 'relationship': 'MENTIONS'}, {'entity': 'Blues lost to
Greater Western Sydney', 'relationship': 'MENTIONS'}, {'entity': 'Narrm', 'relationship': 'MENTIONS'}, {'entity': 'Narrm defeated North Melbourne',
'relationship': 'MENTIONS'}, {'entity': 'Richmond take the AFLW points in Cairns', 'relationship': 'MENTIONS'}, {'entity': 'Kangaroos', 'relationship':
'MENTIONS'}, {'entity': 'Alice O’Loughlin', 'relationship': 'MENTIONS'}, {'entity': 'Melbourne', 'relationship': 'MENTIONS'}, {'entity': 'Maeve Chaplin',
'relationship': 'MENTIONS'}, {'entity': 'Tyla Hanks', 'relationship': 'MENTIONS'}, {'entity': 'Eden Zanker', 'relationship': 'MENTIONS'}, {'entity': 'Niamh
Martin', 'relationship': 'MENTIONS'}, {'entity': 'Tess Craven', 'relationship': 'MENTIONS'}, {'entity': 'Kate Hore', 'relationship': 'MENTIONS'}, {'entity':
'Hanks', 'relationship': 'MENTIONS'}, {'entity': 'Olivia Purcell', 'relationship': 'MENTIONS'}, {'entity': 'Ash Riddell', 'relationship': 'MENTIONS'},
{'entity': 'Emma Kearney', 'relationship': 'MENTIONS'}, {'entity': 'Alyssa Bannan', 'relationship': 'MENTIONS'}, {'entity': 'Lulu Pullar', 'relationship':
'MENTIONS'}, {'entity': 'Demons', 'relationship': 'MENTIONS'}, {'entity': 'Adelaide', 'relationship': 'MENTIONS'}, {'entity': 'Brisbane Lions and Adelaide
top-four thriller', 'relationship': 'MENTIONS'}, {'entity': 'Cairns', 'relationship': 'MENTIONS'}, {'entity': 'Brighton Homes Arena', 'relationship':
'MENTIONS'}, {'entity': 'Crows', 'relationship': 'MENTIONS'}, {'entity': 'Orla O’Dwyer', 'relationship': 'MENTIONS'}, {'entity': 'Bella Dawes', 'relationship':
'MENTIONS'}, {'entity': 'Lily Postlethwaite', 'relationship': 'MENTIONS'}, {'entity': 'Niamh Kelly', 'relationship': 'MENTIONS'}, {'entity': 'the game',
'relationship': 'MENTIONS'}, {'entity': 'Springfield', 'relationship': 'MENTIONS'}, {'entity': 'Richmond', 'relationship': 'MENTIONS'}, {'entity': 'Hawthorn',
'relationship': 'MENTIONS'}, {'entity': 'Emelia Yassir', 'relationship': 'MENTIONS'}, {'entity': 'Monique Conti', 'relationship': 'MENTIONS'}, {'entity': 'Grace
Egan', 'relationship': 'MENTIONS'}, {'entity': 'Emily Bates', 'relationship': 'MENTIONS'}, {'entity': 'Jasmine Fleming', 'relationship': 'MENTIONS'}, {'entity':
'Cazaly’s Stadium', 'relationship': 'MENTIONS'}, {'entity': 'Richmond-Hawthorn match', 'relationship': 'MENTIONS'}, {'entity': 'Bec Goddard', 'relationship':
'MENTIONS'}, {'entity': 'Aine McDonagh', 'relationship': 'MENTIONS'}, {'entity': 'Katie Brennan', 'relationship': 'MENTIONS'}, {'entity': 'Greta Bodey',
'relationship': 'MENTIONS'}, {'entity': 'Mattea Breed', 'relationship': 'MENTIONS'}, {'entity': 'Laura McClelland', 'relationship': 'MENTIONS'}, {'entity':
'Sarah Hosking', 'relationship': 'MENTIONS'}, {'entity': 'Sophie Locke', 'relationship': 'MENTIONS'}, {'entity': 'AFLW game', 'relationship': 'MENTIONS'},
{'entity': 'Bridie Hipwell', 'relationship': 'MENTIONS'}, {'entity': 'Courtney Jones', 'relationship': 'MENTIONS'}, {'entity': 'Yassir', 'relationship':
'MENTIONS'}, {'entity': 'Caitlin Greiser', 'relationship': 'MENTIONS'}, {'entity': 'Sydney', 'relationship': 'MENTIONS'}, {'entity': 'Ikon Park',
'relationship': 'MENTIONS'}, {'entity': 'Lions', 'relationship': 'MENTIONS'}, {'entity': 'Ally Anderson', 'relationship': 'MENTIONS'}, {'entity': 'game',
'relationship': 'MENTIONS'}, {'entity': 'Greater Western Sydney', 'relationship': 'MENTIONS'}, {'entity': 'North Melbourne', 'relationship': 'MENTIONS'},
{'entity': 'Whitten Oval', 'relationship': 'MENTIONS'}, {'entity': 'Swans', 'relationship': 'MENTIONS'}, {'entity': 'Chloe Molloy', 'relationship': 'MENTIONS'},
{'entity': 'AFL Photos', 'relationship': 'MENTIONS'}, {'entity': 'Collingwood', 'relationship': 'MENTIONS'}, {'entity': 'Brisbane Lions', 'relationship':
'MENTIONS'}, {'entity': 'St Kilda', 'relationship': 'MENTIONS'}, {'entity': 'Fremantle', 'relationship': 'MENTIONS'}, {'entity': 'Carlton', 'relationship':
'MENTIONS'}, {'entity': 'concussion', 'relationship': 'MENTIONS'}, {'entity': 'Instagram', 'relationship': 'MENTIONS'}, {'entity': 'Blues', 'relationship':
'MENTIONS'}]
person_connections: [{'entity': 'Comeback kings Richmond take the AFLW points in Cairns', 'relationship': 'MENTIONS'}, {'entity': 'Ground-breaking Gold Coast:
Suns make AFLW finals for a second time', 'relationship': 'MENTIONS'}, {'entity': 'stage five chronic kidney disease', 'relationship': 'SAID'}, {'entity':
'Geelong', 'relationship': 'WORKS_FOR'}, {'entity': 'Western Bulldogs', 'relationship': 'WORKS_FOR'}, {'entity': 'Melbourne', 'relationship': 'WORKS_FOR'},
{'entity': 'Bulldogs', 'relationship': 'WORKS_FOR'}, {'entity': 'Western Bulldogs', 'relationship': 'PARTICIPATED_IN'}, {'entity': 'Instagram', 'relationship':
'PARTICIPATED_IN'}]
</item 3>

<item 4>
Comeback kings Richmond take the AFLW points in Cairns (The Age, 2023-10-20T23:51:34+00:00)
If the Lions had lost, they were at risk of falling out of the top four with second-season side Essendon chasing their spot. Last season’s best and fairest Ally
Anderson said her side loved playing at its training ground in Springfield, which it has made a fortress, and work well in hot conditions . “It was really hot.
It was a grind… But yeah, it was a good finish,” Anderson told Fox Footy post-game. “Every week that we play here, every time that we play here, we get better
and better. So, it was just a matter of time before we finally got the first win and then it hopefully [will] open the floodgates a bit.”

The Lions next play St Kilda before taking on the reigning premiers Melbourne in the final round. After this weekend, they have staked their claim as being
premiership contenders again. Adelaide will play top-four side North Melbourne next weekend before 18th-placed Western Bulldogs.

Graph facts:
- How ego and fear fuelled the rise of artificial intelligence -PUBLISHED_BY-> The Age
- ASX set to open higher as Wall Street rebounds; $A rises -PUBLISHED_BY-> The Age
- Is Google Search better than the rest? And is that fair? -PUBLISHED_BY-> The Age
- When pop culture and sport collide: A timeline of Taylor Swift’s NFL takeover -PUBLISHED_BY-> The Age
- At 37, Alexandra found herself broke, heartbroken and living with her parents -PUBLISHED_BY-> The Age

Entities: The Age, AFLW, Essendon, Comeback kings Richmond take the AFLW points in Cairns, Western Bulldogs, Melbourne, Adelaide, Cairns, Springfield, Fox
Footy, Richmond, Lions, Ally Anderson, North Melbourne, St Kilda
</item 4>

<item 5>
Comeback kings Richmond take the AFLW points in Cairns (The Age, 2023-10-20T23:51:34+00:00)
“I can name the whole team. Everyone had their second in the last couple minutes.

“That was such a team effort, everyone had a moment there when they had to put their head over the footy and lay a tackle and not let it slip.” Swanson (25
disposals, eight clearances) stood out for the Eagles, while Bombers workhorse Maddy Prespakis (30 disposals, nine clearances) was typically prominent in the
midfield alongside Georgia Nanscawen (21, six). With two rounds left, West Coast have a chance to build their winning streak when they meet the bottom of the
table Western Bulldogs next round. Slipping one place to sixth, Essendon have must-win matches against Carlton and Gold Coast on the road to the finals. The
Bombers will be without Steph Cain, who was sidelined with a concussion after an elbow to the head from Swanson.

Graph facts:
- How ego and fear fuelled the rise of artificial intelligence -PUBLISHED_BY-> The Age
- ASX set to open higher as Wall Street rebounds; $A rises -PUBLISHED_BY-> The Age
- Is Google Search better than the rest? And is that fair? -PUBLISHED_BY-> The Age
- When pop culture and sport collide: A timeline of Taylor Swift’s NFL takeover -PUBLISHED_BY-> The Age
- At 37, Alexandra found herself broke, heartbroken and living with her parents -PUBLISHED_BY-> The Age

Entities: The Age, West Coast, AFLW, Swanson, Maddy Prespakis, Georgia Nanscawen, Steph Cain, Essendon, Gold Coast, Comeback kings Richmond take the AFLW points
in Cairns, Western Bulldogs, Cairns, Richmond, Carlton, concussion
</item 5>

<item 6>
Comeback kings Richmond take the AFLW points in Cairns (The Age, 2023-10-20T23:51:34+00:00)
Dogs forward reveals battle with kidney disease Western Bulldogs forward Richelle Cranston has revealed she’s been living, and playing, with stage five chronic
kidney disease. “I still manage to do the things I love,” the inaugural AFLW player posted to Instagram. “It’s just really hard sometimes.” Richelle Cranston of
the Bulldogs: “It’s just really hard sometimes.” Credit: Getty Images The 34-year-old posted the caption along with a video detailing her life with the disease
including weekly blood tests and daily blood pressure checks, tracking water and food intake, taking medications including resonium (to lower potassium levels),
all while training in the high-performance environment.

The Dogs are Cranston’s third club after starting at Melbourne and having a stint with Geelong before heading to the Bulldogs last year. Cranston lined up for
the Bulldogs in her side’s 57-point loss to Sydney on Saturday at Whitten Oval, 14.12 (96) to 6.3 (39).

Graph facts:
- Which M.L.B. Playoff Team Has the Most Daunting Home Field? We Ranked Them. -MENTIONS-> Getty Images
- Video games in 2023: Acquisitions, layoffs, unions -MENTIONS-> Getty Images
- From Justin Timberlake to Sam Asghari: A timeline of Britney Spears’ relationships -MENTIONS-> Getty Images
- West Indies v England: fifth and deciding men’s T20 cricket international – live -MENTIONS-> Getty Images
- Amazon sellers sound off on the FTC's 'long-overdue' antitrust case -MENTIONS-> Getty Images

Entities: Getty Images, The Age, Geelong, Comeback kings Richmond take the AFLW points in Cairns, Richelle Cranston, Western Bulldogs, stage five chronic kidney
disease, resonium, Bulldogs' 57-point loss to Sydney, Melbourne, Cairns, Richmond, Sydney, Whitten Oval, AFLW
</item 6>

<item 7>
Ground-breaking Gold Coast: Suns make AFLW finals for a second time (The Age, 2023-11-03T23:45:04+00:00)
“It was an awesome game and four quarters of Port Adelaide footy … It was awesome to finish up like that,” Phillips told Fox Footy after the 13.10 (88) to 4.6
(30) win over the Giants. Erin Phillips celebrates a goal during her final AFLW match. Credit: AFL Photos Earlier, she told the ABC that she wanted her legacy
to include doing everything she could to grow the game.

“Hopefully I can be remembered as someone who took their opportunities, loved the game and was willing to do whatever it took to ensure the next generation was
in a better position,” she said. Richelle Cranston of the Bulldogs is chaired off the ground by teammates Isabelle Pritchard and Ellie Blackburn after
announcing her retirement. Credit: Getty The 38-year-old kicked a momentous goal from a set shot in the final quarter, and she was smothered by her jubilant
teammates afterwards.

Graph facts:
- How ego and fear fuelled the rise of artificial intelligence -PUBLISHED_BY-> The Age
- ASX set to open higher as Wall Street rebounds; $A rises -PUBLISHED_BY-> The Age
- Is Google Search better than the rest? And is that fair? -PUBLISHED_BY-> The Age
- When pop culture and sport collide: A timeline of Taylor Swift’s NFL takeover -PUBLISHED_BY-> The Age
- At 37, Alexandra found herself broke, heartbroken and living with her parents -PUBLISHED_BY-> The Age

Entities: The Age, Fox Footy, Richelle Cranston, Ground-breaking Gold Coast: Suns make AFLW finals for a second time, Erin Phillips, Bulldogs, Isabelle
Pritchard, Ellie Blackburn, final AFLW match, Port Adelaide-Giants game, Gold Coast Suns, AFL Photos, Port Adelaide, Giants, ABC
</item 7>

<item 8>
Comeback kings Richmond take the AFLW points in Cairns (The Age, 2023-10-20T23:51:34+00:00)
Geelong coach Daniel Lowther said the strong winds made it hard for his side to clear the ball out of their defensive 50-metre zone and to move the ball with
any certainty around the ground.

Loading “At one point the rain hit our box, it was flowing,” Lowther said. “Both teams had to fight in the conditions, but it didn’t help the game. “Our marking
contest wasn’t too flash, our ground ball representation was a bit hit and miss, so our defence were under a bit more pressure than they should have been, but
they fought back really well,” Lowther said. Collingwood next play a Sydney line-up including former Magpies star Chloe Molloy, while Geelong will take on
Richmond at Ikon Park.

Dockers spoil Saints’ finals hopes Fremantle recorded a comfortable 24-point win yesterday to bring an abrupt end to St Kilda’s four-game streak and dash their
AFLW finals hopes. J’Noemi Anderson is tackled by a group of Dockers.

Graph facts:
- How ego and fear fuelled the rise of artificial intelligence -PUBLISHED_BY-> The Age
- ASX set to open higher as Wall Street rebounds; $A rises -PUBLISHED_BY-> The Age
- Is Google Search better than the rest? And is that fair? -PUBLISHED_BY-> The Age
- When pop culture and sport collide: A timeline of Taylor Swift’s NFL takeover -PUBLISHED_BY-> The Age
- At 37, Alexandra found herself broke, heartbroken and living with her parents -PUBLISHED_BY-> The Age

Entities: The Age, AFLW, Geelong, Daniel Lowther, J’Noemi Anderson, Comeback kings Richmond take the AFLW points in Cairns, Cairns, Richmond, Sydney, Ikon Park,
Chloe Molloy, Collingwood, St Kilda, Fremantle
</item 8>

<item 9>
Comeback kings Richmond take the AFLW points in Cairns (The Age, 2023-10-20T23:51:34+00:00)
“Richmond were just absolute beasts, especially their midfielders. They really worked hard all night,” Hawthorn coach Bec Goddard said after the game. “Our work
rate really dropped off and we just watched them play some really good footy. “If you’re a Hawks fan, you would’ve seen some really exciting moments of footy
tonight.

“Keep getting excited. We are coming, we’re growing, we’re getting better.” Hawthorn’s Irish ace Aine McDonagh kicked the opening goal before Tigers skipper
Katie Brennan slotted two unanswered majors to celebrate her 50th game in style. Fellow 50-gamer Greta Bodey put boot to ball to snatch her own goal before
quarter-time. A skirmish erupted in the second term after Brennan took exception to a tackle by Mattea Breed on Laura McClelland. McClelland was gifted a
50-metre penalty following a bump against Sarah Hosking by Sophie Locke and converted from inside the goal square while players were occupied by the tussle.

Graph facts:
- How ego and fear fuelled the rise of artificial intelligence -PUBLISHED_BY-> The Age
- ASX set to open higher as Wall Street rebounds; $A rises -PUBLISHED_BY-> The Age
- Is Google Search better than the rest? And is that fair? -PUBLISHED_BY-> The Age
- When pop culture and sport collide: A timeline of Taylor Swift’s NFL takeover -PUBLISHED_BY-> The Age
- At 37, Alexandra found herself broke, heartbroken and living with her parents -PUBLISHED_BY-> The Age

Entities: The Age, AFLW, Comeback kings Richmond take the AFLW points in Cairns, Cairns, Richmond, Hawthorn, Bec Goddard, Aine McDonagh, Katie Brennan, Greta
Bodey, Mattea Breed, Laura McClelland, Sarah Hosking, Sophie Locke, AFLW game
</item 9>

<item 10>
Ground-breaking Gold Coast: Suns make AFLW finals for a second time (The Age, 2023-11-03T23:45:04+00:00)
“I don’t think I’ve been as nervous to kick a goal in my life, to be honest,” she joked. “But nah, it was awesome.” On Saturday, the competition also said
goodbye to a range of players including Western Bulldogs forward Richelle Cranston, who has battled stage five chronic kidney disease while playing this season,
and Hawthorn’s Akec Makur Chuot, who played 40 games for three clubs.

Cranston, who was an inaugural AFLW player, will start kidney dialysis in January so had no option but to hang up the boots and focus on her treatment and
health. “[I’m feeling] sad. Emotions hit me when the siren went. It’s been a really emotional week, the girls have really gotten around me. But yeah, my health
has sort of deteriorated, it’s been a really hard year,” Cranston told Fox Footy after the Bulldogs lost to North Melbourne 7.13 (55) to 1.3 (9).

Graph facts:
- How ego and fear fuelled the rise of artificial intelligence -PUBLISHED_BY-> The Age
- ASX set to open higher as Wall Street rebounds; $A rises -PUBLISHED_BY-> The Age
- Is Google Search better than the rest? And is that fair? -PUBLISHED_BY-> The Age
- When pop culture and sport collide: A timeline of Taylor Swift’s NFL takeover -PUBLISHED_BY-> The Age
- At 37, Alexandra found herself broke, heartbroken and living with her parents -PUBLISHED_BY-> The Age

Entities: The Age, Richelle Cranston, Western Bulldogs, stage five chronic kidney disease, Fox Footy, Hawthorn, Ground-breaking Gold Coast: Suns make AFLW
finals for a second time, AFLW finals, Akec Makur Chuot, kidney dialysis, Western Bulldogs vs North Melbourne, North Melbourne, Gold Coast Suns, Gold Coast,
AFLW
</item 10>
=== END ITEM id=e55eaa3b744008db ===

=== ITEM id=bc97a4fa65b703c2 ===
Gold sentences and facts:
Sentence 0: The case, filed by Arkansas-based publisher Helena World Chronicle, argues that Google “siphons off” news publishers’ content, their readers and ad revenue through anticompetitive means.
  Fact 0.0: The case was filed by Helena World Chronicle.
  Fact 0.1: Helena World Chronicle is an Arkansas-based publisher.
  Fact 0.2: The case argues that Google 'siphons off' news publishers' content.
  Fact 0.3: The case argues that Google 'siphons off' news publishers' readers.
  Fact 0.4: The case argues that Google 'siphons off' news publishers' ad revenue.
  Fact 0.5: The case argues that this siphoning happens through anticompetitive means.
Sentence 1: Epic will also suggest that Google hid items from discovery by allowing execs and employees to have “off the record” chats, but Google will counter this as well, saying it has provided the court with thousands of chat logs and millions of documents.
  Fact 1.0: Epic will also suggest that Google hid items from discovery.
  Fact 1.1: Epic will suggest that Google hid items from discovery by allowing execs and employees to have 'off the record' chats.
  Fact 1.2: Google will counter this as well.
  Fact 1.3: Google will say it has provided the court with thousands of chat logs.
  Fact 1.4: Google will say it has provided the court with millions of documents.

Retrieved items:
<item 1>
What to know about Fortnite maker Epic Games’ antitrust battle with Google, starting today (TechCrunch, 2023-11-06T16:47:11+00:00)
To make its case, Epic plans to bring claims under Sections 1 and 2 of U.S. antitrust law, the Sherman Act, as well as California’s anticompetitive laws, the
Cartwright Act, and the Unfair Competition Law. It will attempt to argue that Google restrains competition within two separate markets, including the
distribution of apps to Android users and the market for payment processing solutions for content inside Android apps.

Google argues its commissions aren’t just tied to billing, but offers discount for third-party billing options

Today, Google requires apps to use its own first-party billing system, and charges app developers a 15% to 30% commission on the sales it processes.

However, Google will counter Epic’s arguments by pointing out that it has rolled out a new option for app developers called User Choice Billing, which Epic has
declined to use.

Graph facts:
- Alphabet -ACCUSED_OF-> Sherman Act
- Alphabet -ACCUSED_OF-> U.S. antitrust law
- News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line -MENTIONS-> U.S. antitrust law
- News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line -MENTIONS-> Sherman Act
- Epic v. Google, explained -MENTIONS-> User Choice Billing

Entities: U.S. antitrust law, Sherman Act, User Choice Billing, Google first-party billing system, Epic Games’ antitrust battle with Google, California’s
anticompetitive laws, Cartwright Act, Unfair Competition Law, distribution of apps to Android users, payment processing solutions for content inside Android
apps, Android, What to know about Fortnite maker Epic Games’ antitrust battle with Google, starting today, Epic Games, TechCrunch, Google
</item 1>

<item 2>
What to know about Fortnite maker Epic Games’ antitrust battle with Google, starting today (TechCrunch, 2023-11-06T16:47:11+00:00)
“Android’s choice and flexibility work well for consumers and developers of all sizes. We look forward to making our case in court as we fight to keep our users
safe from harm, partner with developers to grow their businesses, and keep the Android ecosystem thriving and healthy for everyone,” reads a Google blog post by
Wilson White, VP, Government Affairs & Public Policy.

Epic’s lawsuit originally involved Match and several U.S. attorneys general, but Google reached a settlement with the latter last month and with Match last
week. Epic will now fight the antitrust battle on its own. The lawsuit, held in the U.S. District Court in California’s Northern District, will include
testimony from a number of Google and Android execs, including CEO Sundar Pichai, and Epic Games CEO Tim Sweeney, as well as Apple, Netflix and other Android
developers.

Google is also involved in another antitrust lawsuit with the Department of Justice over its alleged search monopoly.

Graph facts:
- More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo -MENTIONS-> Sundar Pichai
- Apple defends Google Search deal in court: ‘There wasn’t a valid alternative’ -MENTIONS-> Sundar Pichai
- Tim Sweeney on Epic’s victory royale over Google -MENTIONS-> Sundar Pichai
- 5 things we learned so far about the Google antitrust case -MENTIONS-> Sundar Pichai
- The Epic v. Google trial may come down to simple v. complicated -MENTIONS-> Sundar Pichai

Entities: Sundar Pichai, Android, What to know about Fortnite maker Epic Games’ antitrust battle with Google, starting today, Epic Games, Wilson White, Match,
U.S. attorneys general, Epic's antitrust lawsuit, U.S. District Court in California’s Northern District, Department of Justice, Google's antitrust lawsuit with
the Department of Justice, search monopoly, antitrust, Netflix, Apple
</item 2>

<item 3>
What to know about Fortnite maker Epic Games’ antitrust battle with Google, starting today (TechCrunch, 2023-11-06T16:47:11+00:00)
Epic, meanwhile, lost its antitrust battle with Apple and is now asking the Supreme Court to weigh in.

Graph facts:
- What to know about Fortnite maker Epic Games’ antitrust battle with Google, starting today -MENTIONS-> Sundar Pichai
- What to know about Fortnite maker Epic Games’ antitrust battle with Google, starting today -MENTIONS-> U.S. antitrust law
- What to know about Fortnite maker Epic Games’ antitrust battle with Google, starting today -MENTIONS-> Sherman Act
- What to know about Fortnite maker Epic Games’ antitrust battle with Google, starting today -MENTIONS-> User Choice Billing
- What to know about Fortnite maker Epic Games’ antitrust battle with Google, starting today -MENTIONS-> Google first-party billing system

Entities: What to know about Fortnite maker Epic Games’ antitrust battle with Google, starting today, Epic Games, antitrust, antitrust battle with Google,
antitrust battle with Apple, Apple, Supreme Court, TechCrunch, Google, Fortnite
</item 3>

<item 4>
What to know about Fortnite maker Epic Games’ antitrust battle with Google, starting today (TechCrunch, 2023-11-06T16:47:11+00:00)
Epic Games will specifically reference one agreement it had made with the OEM OnePlus, to make Epic games available on its devices through the Epic Games app.
But OnePlus had to cancel the deal over concerns that the arrangement would have it bypassing the Google Play Store. The case will also examine a deal between
Google and Samsung that Epic says was designed to prevent the Galaxy Store from being a competitive threat and require Google Play Billing on apps distributed
by Samsung.

Epic will argue Google’s developer agreements are anti-competitive

A second set of agreements Epic will highlight are those between Google and Android app developers themselves. One such program, the “Project Hug” initiative, a
part of the Google Games Velocity Program, will be cited as an example of where Google paid competitors not to distribute their apps on Android outside of the
Play Store by targeting those that were most at risk of attrition from Play, then approaching them with a monetary offer.

Graph facts:
- More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo -MENTIONS-> Android
- ChatGPT’s mobile app hits record $4.58M in revenue in September -MENTIONS-> Android
- ChatGPT: Everything you need to know about the AI-powered chatbot -MENTIONS-> Android
- The best Black Friday deals you can get for under $25 -MENTIONS-> Android
- European football: Eintracht Frankfurt hit Bayern for five in famous thrashing -MENTIONS-> Android

Entities: Android, Galaxy Store, What to know about Fortnite maker Epic Games’ antitrust battle with Google, starting today, OnePlus, Epic Games app, Google
Play Store, Android app developers, Project Hug, Google Games Velocity Program, antitrust battle, agreement with OnePlus, deal between Google and Samsung,
Google developer agreements, anti-competitive practices, Epic Games
</item 4>

<item 5>
Graph query row:
article: News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line
publisher: ['TechCrunch']
mentioned_entities: [{'label': '__KGBuilder__', 'name': 'Alphabet'}, {'label': '__KGBuilder__', 'name': 'Helena World Chronicle'}, {'label': '__KGBuilder__',
'name': 'U.S. District Court in D.C.'}, {'label': '__KGBuilder__', 'name': 'Search Generative Experience'}, {'label': '__KGBuilder__', 'name': 'Bard AI
chatbot'}, {'label': '__KGBuilder__', 'name': 'Knowledge Graph'}, {'label': '__KGBuilder__', 'name': 'anticompetitive behavior'}, {'label': '__KGBuilder__',
'name': 'U.S. antitrust law'}, {'label': '__KGBuilder__', 'name': 'Sherman Act'}, {'label': '__KGBuilder__', 'name': 'Arkansas'}, {'label': '__KGBuilder__',
'name': 'question-and-answer technologies'}, {'label': '__KGBuilder__', 'name': 'Publishers'}, {'label': '__KGBuilder__', 'name': 'Knowledge Panel'}, {'label':
'__KGBuilder__', 'name': 'Featured Snippets'}, {'label': '__KGBuilder__', 'name': 'class action antitrust suit'}, {'label': '__KGBuilder__', 'name':
'misappropriation of Publishers’ content'}, {'label': '__KGBuilder__', 'name': 'The Wall Street Journal'}, {'label': '__KGBuilder__', 'name': 'report on
Thursday'}, {'label': '__KGBuilder__', 'name': 'deal to license news for AI model training'}, {'label': '__KGBuilder__', 'name': 'SGE'}, {'label':
'__KGBuilder__', 'name': 'Google Bard AI'}, {'label': '__KGBuilder__', 'name': 'GoogleBot'}, {'label': '__KGBuilder__', 'name': 'Google general search
service'}, {'label': '__KGBuilder__', 'name': 'AdSense'}, {'label': '__KGBuilder__', 'name': 'News Media Alliance'}, {'label': '__KGBuilder__', 'name':
'Washington Post'}, {'label': '__KGBuilder__', 'name': 'Epic Games lawsuit against Google over app store antitrust issues'}, {'label': '__KGBuilder__', 'name':
'app store antitrust issues'}, {'label': '__KGBuilder__', 'name': 'U.S. Justice Department'}, {'label': '__KGBuilder__', 'name': 'Hausfeld'}, {'label':
'__KGBuilder__', 'name': 'Helena World Chronicle, LLC'}, {'label': '__KGBuilder__', 'name': 'class action antitrust suit against Google'}, {'label':
'__KGBuilder__', 'name': 'U.S. Justice Department lawsuit against Google'}, {'label': '__KGBuilder__', 'name': '2020 Justice Department civil antitrust suit
over search and search advertising'}, {'label': '__KGBuilder__', 'name': 'digital ad technologies'}, {'label': '__KGBuilder__', 'name': 'Clayton Act'},
{'label': '__KGBuilder__', 'name': 'search and search advertising'}, {'label': '__KGBuilder__', 'name': 'digital news and reference publishing'}, {'label':
'__KGBuilder__', 'name': 'class-wide monetary and injunctive relief'}, {'label': '__KGBuilder__', 'name': 'free marketplace of ideas'}, {'label':
'__KGBuilder__', 'name': 'free press'}, {'label': '__KGBuilder__', 'name': 'Epic Games'}, {'label': '__KGBuilder__', 'name': 'antitrust'}, {'label':
'__KGBuilder__', 'name': 'The Atlantic'}, {'label': '__KGBuilder__', 'name': 'Axel Springer'}, {'label': '__KGBuilder__', 'name': 'Allen Institute for AI'},
{'label': '__KGBuilder__', 'name': 'class action lawsuit'}, {'label': '__KGBuilder__', 'name': 'consumers'}, {'label': '__KGBuilder__', 'name': 'AI'}, {'label':
'__KGBuilder__', 'name': 'AI technologies'}, {'label': '__KGBuilder__', 'name': 'Similarweb'}, {'label': '__KGBuilder__', 'name': 'Google'}, {'label':
'__KGBuilder__', 'name': 'artificial intelligence'}, {'label': '__KGBuilder__', 'name': 'D.C.'}, {'label': '__KGBuilder__', 'name': 'OpenAI'}, {'label':
'__KGBuilder__', 'name': 'labor'}]
connected_via_shared_entity: [{'article': 'What to know about Fortnite maker Epic Games’ antitrust battle with Google, starting today', 'connected_via': 'U.S.
antitrust law'}, {'article': 'What to know about Fortnite maker Epic Games’ antitrust battle with Google, starting today', 'connected_via': 'Sherman Act'},
{'article': 'What to know about Fortnite maker Epic Games’ antitrust battle with Google, starting today', 'connected_via': 'Epic Games'}, {'article': 'What to
know about Fortnite maker Epic Games’ antitrust battle with Google, starting today', 'connected_via': 'antitrust'}, {'article': 'What to know about Fortnite
maker Epic Games’ antitrust battle with Google, starting today', 'connected_via': 'Google'}, {'article': 'Tim Sweeney on Epic’s victory royale over Google',
'connected_via': 'Epic Games'}, {'article': 'Tim Sweeney on Epic’s victory royale over Google', 'connected_via': 'antitrust'}, {'article': 'Tim Sweeney on
Epic’s victory royale over Google', 'connected_via': 'Google'}, {'article': 'The Epic v. Google trial may come down to simple v. complicated', 'connected_via':
'Epic Games'}, {'article': 'The Epic v. Google trial may come down to simple v. complicated', 'connected_via': 'Google'}, {'article': 'Epic v. Google,
explained', 'connected_via': 'Epic Games'}, {'article': 'Epic v. Google, explained', 'connected_via': 'Google'}]
directly_mentioned_articles: []
</item 5>

<item 6>
What to know about Fortnite maker Epic Games’ antitrust battle with Google, starting today (TechCrunch, 2023-11-06T16:47:11+00:00)
Match may have settled its antitrust lawsuit with Google last week, but Fortnite maker Epic Games is still set to go to trial with the tech giant today,
November 6, in hopes of convincing a jury that Google engages in anticompetitive behavior with regard to its Android app store, Google Play, and its commission
structure. This case will differ slightly from Epic’s battle with Apple on the same topic because, this time, Epic can’t allege there’s no other way to load
apps onto Android devices, as it could with Apple, because Android does allow for sideloading apps. Though Epic will suggest Google makes that process
cumbersome, the bulk of its arguments will center around what it believes are anticompetitive agreements between Google and device manufacturers and developers.

The precedent Epic’s case will rely on is a Microsoft case where courts found Microsoft to have a monopoly over the operating system and were abusing it, making
it hard for users to download alternative browsers, like Netscape, on Windows devices.
</item 6>

<item 7>
Epic v. Google, explained (The Verge, 2023-11-05T11:00:00+00:00)
Google will say that every Samsung Galaxy phone has “a competing app store right there on the homescreen,” even though Google apparently tried to quash that
store, too, fearing it could cost the company billions.

We caught Google trying to play the China card in a pre-briefing with press, reminding us that Epic is 40 percent owned by Chinese company Tencent — even though
Google itself allegedly considered teaming up with Tencent to eliminate the Epic threat. While the parties can’t argue it in court, Google can attempt to subtly
play the China card: “Google may ask a witness once about Tencent’s domicile,” reads one of Judge Donato’s orders.

In the same press briefing, Google suggested to journalists that Epic should have joined the company’s “User Choice Billing” program if it wanted an alternative
payment system, instead of suing. But that program didn’t exist until 2022, long after the lawsuit, and it only knocks 4 percentage points off Google’s fee.

When should we expect a jury verdict?

Graph facts:
- What to know about Fortnite maker Epic Games’ antitrust battle with Google, starting today -MENTIONS-> User Choice Billing
- The fate of US venture capital in China teeters on uncertainty -MENTIONS-> Tencent
- A comprehensive list of 2023 tech layoffs -MENTIONS-> Tencent
- Video games in 2023: Acquisitions, layoffs, unions -MENTIONS-> Tencent
- European neobroker Scalable Capital raises $65M on a flat $1.4B valuation -MENTIONS-> Tencent

Entities: User Choice Billing, Tencent, Epic, China, Google, Epic v. Google, explained, Epic v. Google lawsuit, Judge Donato, Samsung Galaxy phone, The Verge
</item 7>

<item 8>
Epic v. Google, explained (The Verge, 2023-11-05T11:00:00+00:00)
But if you do want to see a vision of a future that Epic would accept, look no further than this letter it sent to state attorneys general — laying out a
settlement that would basically prohibit Google from doing the anticompetitive things that Epic claims it did.

There’s also this tweet from Sweeney:

Google wouldn’t tell The Verge if it’s offered Epic any kind of settlement yet.

Why if I just want the dirty laundry on both these companies? You promised dirt...

Oh, it could get good.

The allegations made before trial suggest Google was playing loads of dirty tricks with names like “Project Hug,” “Project Agave,” and “Project Banyan”, with
Epic claiming it’s seen evidence Google was paying off game developers and phone makers not to abandon its app store — to the tune of hundreds of millions of
dollars. Google allegedly even toyed with the idea of buying Epic to keep Fortnite from inspiring other game developers to abandon ship.

Plus, there’s evidence Google deleted potentially incriminating messages to keep them out of a court’s hands.

Graph facts:
- Tim Sweeney on Epic’s victory royale over Google -MENTIONS-> Project Hug
- What to know about Fortnite maker Epic Games’ antitrust battle with Google, starting today -MENTIONS-> Project Hug
- Tim Sweeney -SAID-> Project Hug
- Project Hug -PART_OF-> Google Games Velocity Program
- A comprehensive list of 2023 tech layoffs -MENTIONS-> Epic

Entities: Project Hug, Epic, Google, Epic v. Google, explained, Sweeney, state attorneys general, Project Agave, Project Banyan, game developers, phone makers,
Google's app store, settlement, anticompetitive practices, paying game developers and phone makers not to abandon Google's app store, deleting potentially
incriminating messages
</item 8>

<item 9>
Epic v. Google, explained (The Verge, 2023-11-05T11:00:00+00:00)
On Monday, Fortnite publisher Epic Games will drag a tech giant into court, alleging that its mobile app store is an illegal monopoly. You may be feeling some
déjà vu!

After all, didn’t Epic already go to trial with Apple, resulting in a ruling where Apple (mostly) won? Didn’t Epic fail to #FreeFortnite with its Llamacorn
legal gambit? Didn’t that all happen years ago? What are we doing here again?

The short answer: while Epic’s antitrust claims against Apple got their day in court, a similar lawsuit against Google never did. On November 6th, Epic v.
Google will finally go to trial... a mere 1,180 days after Epic originally sued.

Hi, I’m Sean, and I’ll be your guide to this whole delightful mess.

Update, December 15th, 2023: It’s basically over now. What happened? Read this, this, and this to get the whole picture.

Original story continues:

Llamacorn what now? What is happening?

Graph facts:
- News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line -MENTIONS-> Epic Games
- What to know about Fortnite maker Epic Games’ antitrust battle with Google, starting today -MENTIONS-> Epic Games
- A comprehensive list of 2023 tech layoffs -MENTIONS-> Epic Games
- Video games in 2023: Acquisitions, layoffs, unions -MENTIONS-> Epic Games
- More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo -MENTIONS-> Epic Games

Entities: Epic Games, Sean, Apple, Google, Epic v. Google trial, Epic v. Apple trial, #FreeFortnite, Llamacorn, antitrust claims, illegal monopoly, Epic v.
Google, explained, Fortnite, The Verge
</item 9>

<item 10>
What to know about Fortnite maker Epic Games’ antitrust battle with Google, starting today (TechCrunch, 2023-11-06T16:47:11+00:00)
It will note this process is streamlined and helps keep users safe. But Epic will claim that this process is made to be overly difficult for end users, with
multiple steps involved and “scare screens” that warn users of the security risks involved with sideloading.

Epic will also suggest that Google hid items from discovery by allowing execs and employees to have “off the record” chats, but Google will counter this as
well, saying it has provided the court with thousands of chat logs and millions of documents.

The tech giant will ultimately push the notion that this case isn’t about competition, but about money — that is, Epic Games wants to reach the Play Store’s 2.5
billion users without having to pay.

“Android’s choice and flexibility work well for consumers and developers of all sizes.

Graph facts:
- ChatGPT: Everything you need to know about the AI-powered chatbot -MENTIONS-> Android
- The best Black Friday deals you can get for under $25 -MENTIONS-> Android
- More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo -MENTIONS-> Android
- ChatGPT’s mobile app hits record $4.58M in revenue in September -MENTIONS-> Android
- European football: Eintracht Frankfurt hit Bayern for five in famous thrashing -MENTIONS-> Android

Entities: Epic Games’ antitrust battle with Google, Android, What to know about Fortnite maker Epic Games’ antitrust battle with Google, starting today, Epic
Games, Sideloading, Security risks, Off-the-record chats, Chat logs and documents provided to the court, Play Store access and payment, TechCrunch, Google,
competition, Play Store, Fortnite
</item 10>
=== END ITEM id=bc97a4fa65b703c2 ===

=== ITEM id=77dbca58857f04a7 ===
Gold sentences and facts:
Sentence 0: When he entered free agency, a dozen teams lined up, curious to see if they could meet his eye.
  Fact 0.0: A dozen teams lined up when the free agent entered free agency.
  Fact 0.1: The dozen teams were curious to see if they could meet the free agent's eye.
Sentence 1: As the player, I explore a system of the Imperium with a loyal crew of companions, making impactful choices and determining the fate of those around me.
  Fact 1.0: The player explores a system of the Imperium.
  Fact 1.1: The player explores with a loyal crew of companions.
  Fact 1.2: The player makes impactful choices.
  Fact 1.3: The player determines the fate of those around the player.

Retrieved items:
<item 1>
Rogue Trader’s role-playing embraces the brutality and freedom of Warhammer 40K (Polygon, 2023-12-06T13:00:00+00:00)
For 40K fans, this is a rare treat — a game that digs past the heroic facade of bolters and battles and taps into the grimdark dystopia that makes this
particular sci-fi setting so damn compelling.

Warhammer 40,000: Rogue Trader will be released on Dec. 7 on Windows PC. The game was reviewed using a pre-release download code provided by Owlcat Games. Vox
Media has affiliate partnerships. These do not influence editorial content, though Vox Media may earn commissions for products purchased via affiliate links.
You can find additional information about Polygon’s ethics policy here .

Graph facts:
- How to choose the best class for you in Baldur’s Gate 3 -PUBLISHED_BY-> Polygon
- The best video games of the year so far -MENTIONS-> Windows PC
- The best Xbox controller to buy right now -MENTIONS-> Windows PC
- The best video games of the year so far -MENTIONS-> Warhammer 40,000: Rogue Trader
- The best Warlock subclass, feats, and build in Baldur’s Gate 3 -PUBLISHED_BY-> Polygon

Entities: Warhammer 40,000: Rogue Trader, Windows PC, Polygon, Vox Media, Warhammer 40K, Rogue Trader’s role-playing embraces the brutality and freedom of
Warhammer 40K, Owlcat Games
</item 1>

<item 2>
Rogue Trader’s role-playing embraces the brutality and freedom of Warhammer 40K (Polygon, 2023-12-06T13:00:00+00:00)
“In the grim darkness of the far future, there is only war.” That’s the tagline of Warhammer 40,000, one of the most over-the-top and brutal sci-fi settings
around. But even in a merciless, brutal dystopia that grinds its people into dust, some get to enjoy being on the top of the food chain. Warhammer 40,000: Rogue
Trader is about the perils and pleasures of being atop that particular hierarchy.

Rogue Traders in 40K are freelance explorers, scouting the far frontiers of the Imperium of Man. It’s a risky career, but it comes with luxurious rewards:
agency, freedom, power, and a giant flagship. As the player, I explore a system of the Imperium with a loyal crew of companions, making impactful choices and
determining the fate of those around me.

Rogue Trader is a computer role-playing game in the vein of Baldur’s Gate 3 or Pillars of Eternity.

Graph facts:
- The best video games of the year so far -MENTIONS-> Warhammer 40,000: Rogue Trader
- How to choose the best class for you in Baldur’s Gate 3 -PUBLISHED_BY-> Polygon
- The best Warlock subclass, feats, and build in Baldur’s Gate 3 -PUBLISHED_BY-> Polygon
- 9 great movies to watch on Tubi for free right now -PUBLISHED_BY-> Polygon
- The future of DC movies in 2022 and beyond -PUBLISHED_BY-> Polygon

Entities: Warhammer 40,000: Rogue Trader, Polygon, Baldur’s Gate 3, Warhammer 40,000, Imperium of Man, Pillars of Eternity, Rogue Traders, Rogue Trader’s
role-playing embraces the brutality and freedom of Warhammer 40K
</item 2>

<item 3>
Rogue Trader’s role-playing embraces the brutality and freedom of Warhammer 40K (Polygon, 2023-12-06T13:00:00+00:00)
The victory tasted like ash in my mouth after all that frustration.

Voidship combat is another aspect that feels clunky and frustrating. Like the squad-based skirmishes, naval encounters are also turn-based, where positioning is
ultra important. Space naval battles should feel tense, but instead, I’m mostly annoyed at having to continually rotate my ship and set up my zones of attack. I
wish I could delegate these annoyances to my Senechal — to delegate the duties of character leveling and ship combat, the better to appreciate all of the things
Rogue Trader is doing so well.

Rogue Trader is a dense, vast game, and much of it has clearly been crafted with love for the expansive lore of the 40K canon. While there are small annoyances
and clunky features along the way, the political intrigue, cast of characters, and moral choices have me hooked.

Graph facts:
- The best Warlock subclass, feats, and build in Baldur’s Gate 3 -PUBLISHED_BY-> Polygon
- 9 great movies to watch on Tubi for free right now -PUBLISHED_BY-> Polygon
- The future of DC movies in 2022 and beyond -PUBLISHED_BY-> Polygon
- The best movies of 2023 so far -PUBLISHED_BY-> Polygon
- How to choose the best class for you in Baldur’s Gate 3 -PUBLISHED_BY-> Polygon

Entities: Polygon, Warhammer 40K, Rogue Trader’s role-playing embraces the brutality and freedom of Warhammer 40K, Rogue Trader
</item 3>

<item 4>
Rogue Trader’s role-playing embraces the brutality and freedom of Warhammer 40K (Polygon, 2023-12-06T13:00:00+00:00)
I create my custom protagonist, determining stats and bonuses based on my backstory, home world, and so on. I’m then thrust into a high-pressure job interview
as a potential heir to a Rogue Trader aboard her flagship. Little do we know that there’s a coup in the works, and traitors aboard the ship. After a deadly
struggle against heretics and demons, I ascend to the position of Rogue Trader. I’m the captain now, and I get to decide how to run my ship.

This is a vast game, with tons of features you’d expect from a CRPG — companions (each with their own narrative paths and conversation trees), top-down
strategic combat, and branching choices that impact the world around you. I get to make choices all day — in fact, that’s part of the gig of a Rogue Trader.
Some are more important than others; whenever I get to specific points in the story, I can choose from one of three major paths.

Graph facts:
- The best Warlock subclass, feats, and build in Baldur’s Gate 3 -PUBLISHED_BY-> Polygon
- 9 great movies to watch on Tubi for free right now -PUBLISHED_BY-> Polygon
- The future of DC movies in 2022 and beyond -PUBLISHED_BY-> Polygon
- The best movies of 2023 so far -PUBLISHED_BY-> Polygon
- How to choose the best class for you in Baldur’s Gate 3 -PUBLISHED_BY-> Polygon

Entities: Polygon, Warhammer 40K, CRPG, Rogue Trader’s role-playing embraces the brutality and freedom of Warhammer 40K, Rogue Trader
</item 4>

<item 5>
Rogue Trader’s role-playing embraces the brutality and freedom of Warhammer 40K (Polygon, 2023-12-06T13:00:00+00:00)
I can call on them to unlock doors or perform other environmental checks, but they come most in handy in combat.

When I meet opposition, it’s deeply satisfying to control my troops in a turn-based battle. Each fight takes place on a grid; it’s very similar to Baldur’s Gate
3 or even XCOM 2. Some positions provide cover, while others are out in the open. Friendly fire is also a very real concern. An arc of auto-fire from a bolter,
or a Navigator’s third eye opening, can harm friend as well as foe. My Rogue Trader is a sniper, and she would be lost without her Senechal taking the front
lines. There’s a lot of firepower at my disposal, and it’s mostly quite satisfying to use — even if I occasionally shred my poor Senechal with a devastating
AOE.

I can chat with the companions between battles, learning more about their pasts.

Graph facts:
- The best Warlock subclass, feats, and build in Baldur’s Gate 3 -PUBLISHED_BY-> Polygon
- 9 great movies to watch on Tubi for free right now -PUBLISHED_BY-> Polygon
- The future of DC movies in 2022 and beyond -PUBLISHED_BY-> Polygon
- The best movies of 2023 so far -PUBLISHED_BY-> Polygon
- How to choose the best class for you in Baldur’s Gate 3 -PUBLISHED_BY-> Polygon

Entities: Polygon, Baldur’s Gate 3, Warhammer 40K, XCOM 2, Rogue Trader, Senechal, Rogue Trader’s role-playing embraces the brutality and freedom of Warhammer
40K
</item 5>

<item 6>
Rogue Trader’s role-playing embraces the brutality and freedom of Warhammer 40K (Polygon, 2023-12-06T13:00:00+00:00)
Dogmatic choices exult the God-Emperor and loathe the mutant and the witch, the Heretical options pursue corruptive power, and the Ionoclast path is the closest
thing we have to modern-day “good guy” morality.

After the events of the tutorial, my beautiful voidship is run down and my staff is struggling. As the newly anointed Rogue Trader, I have to trek around the
various planets of the Koronus Expanse to get a new Navigator, fix my ship up, and avoid any major diplomatic incidents with the locals. My voidship is the size
of a modern city; I’m as much a governor as a captain, and I have to manage the ship, its cargo, and its many occupants.

In the process of getting back on my feet, I uncover a nefarious cult and a deep conspiracy. It’s not an easy job, but I’m blessed with a handful of companions
from the Imperium to help me out.
</item 6>

<item 7>
Rogue Trader’s role-playing embraces the brutality and freedom of Warhammer 40K (Polygon, 2023-12-06T13:00:00+00:00)
He’s cruel, depraved, and an absolute hoot. Yes, he may flay a few too many people for my tastes, but he’s the best companion to bring to a party.

In the grand scheme of things, this is one of the most complete and detailed explorations of the 40K universe you can find. The game is an homage to the
Warhammer 40K RPGs from Fantasy Flight Games, including Rogue Trader and Dark Heresy. I spent years as a teenager and young adult poring over these sourcebooks
(which Rogue Trader is inspired by), learning more about the tiny details of life in the Imperium. Developer Owlcat Games has paid the same attention to every
detail of the Koronus Expanse. I delve into ancient facilities staffed by tech-priests of the Machine God, the hostile xenos city of Commorragh, or massive
cities built to honor the God-Emperor of Mankind.

Graph facts:
- The best Warlock subclass, feats, and build in Baldur’s Gate 3 -PUBLISHED_BY-> Polygon
- 9 great movies to watch on Tubi for free right now -PUBLISHED_BY-> Polygon
- The future of DC movies in 2022 and beyond -PUBLISHED_BY-> Polygon
- The best movies of 2023 so far -PUBLISHED_BY-> Polygon
- How to choose the best class for you in Baldur’s Gate 3 -PUBLISHED_BY-> Polygon

Entities: Polygon, Rogue Trader’s role-playing embraces the brutality and freedom of Warhammer 40K, Rogue Trader, Warhammer 40K, Fantasy Flight Games, Dark
Heresy, Owlcat Games, Koronus Expanse, Machine God, Commorragh, God-Emperor of Mankind, Imperium
</item 7>

<item 8>
Rogue Trader’s role-playing embraces the brutality and freedom of Warhammer 40K (Polygon, 2023-12-06T13:00:00+00:00)
I can chat with the companions between battles, learning more about their pasts. Most of them have deep and dark secrets I can uncover with a little time or
patience, and they have fascinating stories to tell. Abelard, my Senechal, is a guy who sucks morally but will back me to the absolute hilt. I grew to love
hearing about his days in the Imperial Navy, and he was the one guy I felt like I could trust. Augusta, a Sister of Battle, starts as a one-note zealot, but
cracks form in that facade when I learn about her past and doubts. Cassia and Pasqal both represent two sub-factions in 40K, and they have lots to share about
the Navigators and Tech-Priests.

My absolute favorite companion is Marazhai Aezyrraesh, a dark space elf who feeds off the suffering of sentient beings. He’s cruel, depraved, and an absolute
hoot.

Graph facts:
- The best Warlock subclass, feats, and build in Baldur’s Gate 3 -PUBLISHED_BY-> Polygon
- 9 great movies to watch on Tubi for free right now -PUBLISHED_BY-> Polygon
- The future of DC movies in 2022 and beyond -PUBLISHED_BY-> Polygon
- The best movies of 2023 so far -PUBLISHED_BY-> Polygon
- How to choose the best class for you in Baldur’s Gate 3 -PUBLISHED_BY-> Polygon

Entities: Polygon, Warhammer 40K, Rogue Trader’s role-playing embraces the brutality and freedom of Warhammer 40K, Abelard, Imperial Navy, Augusta, Cassia,
Pasqal, Navigators, Tech-Priests, Marazhai Aezyrraesh, suffering of sentient beings, Rogue Trader
</item 8>

<item 9>
Graph query row:
Polygon_article: Rogue Trader’s role-playing embraces the brutality and freedom of Warhammer 40K
Polygon: Polygon
game: Warhammer 40,000: Rogue Trader
NYT_article: M.L.B. Superstar Shohei Ohtani to Sign $700 Million Deal With Dodgers
publisher: The New York Times
person: Shohei Ohtani
shared_connection: None
connection_labels: None
narrative_choice_passage: Rogue Trader’s role-playing embraces the brutality and freedom of Warhammer 40K (Polygon, 2023-12-06T13:00:00+00:00)
“In the grim darkness of the far future, there is only war.” That’s the tagline of Warhammer 40,000, one of the most over-the-top and brutal sci-fi settings
around. But even in a merciless, brutal dystopia that grinds its people into dust, some get to enjoy being on the top of the food chain. Warhammer 40,000: Rogue
Trader is about the perils and pleasures of being atop that particular hierarchy.

Rogue Traders in 40K are freelance explorers, scouting the far frontiers of the Imperium of Man. It’s a risky career, but it comes with luxurious rewards:
agency, freedom, power, and a giant flagship. As the player, I explore a system of the Imperium with a loyal crew of companions, making impactful choices and
determining the fate of those around me.

Rogue Trader is a computer role-playing game in the vein of Baldur’s Gate 3 or Pillars of Eternity.
</item 9>

<item 10>
Rogue Trader’s role-playing embraces the brutality and freedom of Warhammer 40K (Polygon, 2023-12-06T13:00:00+00:00)
I love the characters, the environments, the writing, the lore, and the flow of battles. But I have concerns with the game’s pacing. By the end of the first
chapter, I had leveled up 16 times. Each level offered marginal rewards, like being able to move slightly farther during the character’s turn in combat or
having a higher parry chance when being attacked. A slow drip-feed system means each level feels less important, and even though I’m growing stronger, I don’t
get that sense of long-term satisfaction.

It’s especially frustrating to hit a roadblock like the one at the end of Act 1, where an incredibly tough boss rolls out of nowhere and spanks my crew — and I
can’t leave to go grind experience somewhere else. I eventually found out a way to cheese the fight by focusing on my melee fighters’ positioning, but it took
far too long banging my head against the wall. The victory tasted like ash in my mouth after all that frustration.
</item 10>
=== END ITEM id=77dbca58857f04a7 ===
