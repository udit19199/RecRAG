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

=== ITEM id=62aaf07d8741b8e8 ===
Gold sentences and facts:
Sentence 0: “But nah, it was awesome.” On Saturday, the competition also said goodbye to a range of players including Western Bulldogs forward Richelle Cranston, who has battled stage five chronic kidney disease while playing this season, and Hawthorn’s Akec Makur Chuot, who played 40 games for three clubs.
  Fact 0.0: A speaker said 'But nah, it was awesome.'
  Fact 0.1: On Saturday, the competition said goodbye to a range of players.
  Fact 0.2: On Saturday, the competition said goodbye to Western Bulldogs forward Richelle Cranston.
  Fact 0.3: Richelle Cranston has battled stage five chronic kidney disease while playing this season.
  Fact 0.4: On Saturday, the competition said goodbye to Hawthorn's Akec Makur Chuot.
  Fact 0.5: Akec Makur Chuot is a Hawthorn player.
  Fact 0.6: Akec Makur Chuot played 40 games for three clubs.
Sentence 1: The Dogs are Cranston’s third club after starting at Melbourne and having a stint with Geelong before heading to the Bulldogs last year.
  Fact 1.0: The Dogs are Richelle Cranston's third club.
  Fact 1.1: Richelle Cranston started at Melbourne.
  Fact 1.2: Richelle Cranston had a stint with Geelong.
  Fact 1.3: Richelle Cranston joined the Bulldogs last year.

Retrieved items:
<item 1>
Who’s the next Gawn, Grundy or Dusty? AFL’s talent guru names top 40 draft prospects and compares them to current stars (The Roar | Sports Writers Blog,
2023-11-15T23:48:27+00:00)
???https://t.co/nf4CwUbK9A — AFL (@AFL) November 11, 2023

Archer Reid

State: Victoria Country

State League Club: Gippsland Power

Community Club: Inverloch-Kongwak

Date of Birth: 19/09/05

Height: 203cm

A mobile tall forward who is an exceptional overhead mark, good below his knees and has a natural goal sense. Continued to develop physically and improved his
forward craft as the season progressed. Averaged 13.5 disposals, five marks and kicked 16 goals in 11 matches in the Coates Talent League for the Gippsland
Power to finish runner-up in the club’s best and fairest award.
</item 1>

<item 2>
Comeback kings Richmond take the AFLW points in Cairns (The Age, 2023-10-20T23:51:34+00:00)
It was the Western Bulldogs’ eighth loss in a row, and their biggest-ever defeat, and they have just two rounds left to clinch a win this season. They next play
17th-placed West Coast and then top-four side North Melbourne. For the Swans, however, the victory has them challenging for a top-eight spot alongside the likes
of Gold Coast, Geelong, Collingwood, St Kilda and Carlton.

The Swans, who went winless in their debut season last year, were propelled to victory by a Chloe Molloy masterclass. After a five-goal second term, which
included a four-minute burst of three majors, Sydney had already brought up their largest score in an AFLW game by halftime. Swan Chloe Molloy celebrates a
goal. Credit: AFL Photos / Getty Images The Bulldogs held their own in the last quarter, but the damage had already been done. The total was also the biggest
the Dogs have conceded in their history, topping the 83 Melbourne kicked against them in round three.
</item 2>

<item 3>
Who’s the next Gawn, Grundy or Dusty? AFL’s talent guru names top 40 draft prospects and compares them to current stars (The Roar | Sports Writers Blog,
2023-11-15T23:48:27+00:00)
Didn’t test at the national Draft Combine due to injury.

Weapon: Neat kicking and ball-winning ability

AFL player comparison: Jack Macrae (Western Bulldogs)

Riley Hardeman

State: Western Australia

State League Club: Swan Districts

Community Club: Caversham

Date of Birth: 27/02/05

Height: 185cm

Left-footed medium defender who is strong overhead, has line-breaking speed and is a penetrating kick. These attributes were on show as captain of Western
Australia at the National Championships where he earned All Australian selection after averaging 18.5 disposals and 5.8 marks and displayed composure under
pressure. Made his senior debut for Swan Districts this year and impressed when representing Australia as part of the AFL Academy.
</item 3>

<item 4>
Ground-breaking Gold Coast: Suns make AFLW finals for a second time (The Age, 2023-11-03T23:45:04+00:00)
“We love going in those four walls. We train there three, four times a week. So having that familiarity of the ground is awesome ... So I would be super-excited
to make finals for one, but then play at home would be just that extra and having all our support behind us really drives us on the field.” Tears abound in
Phillips’ emotional farewell

There were emotional scenes at Alberton Oval as the siren sounded on AFLW great Erin Phillips’ last game, with Port Adelaide saluting her with a huge 58-point
win over Greater Western Sydney on Saturday. The crowd, which included Phillips’ father Greg and mother Julie (both in tears after the game), more family and
friends, and Adelaide captain Chelsea Randall, erupted as the curtain finally came down for the three-time premiership player after eight seasons and 67 games.
</item 4>

<item 5>
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
</item 5>

<item 6>
McRae reveals hidden message he carried under collar of his grand final shirt (The Sydney Morning Herald, 2023-09-30T22:45:39+00:00)
Nathan Murphy, who took no further part in the grand final after being concussed in a clash with Brisbane’s Linc McCarthy, spared a thought for all three
players amid the post-match chaos. “It was pretty emotional seeing them. You go from one high like myself to a low like theirs,” Murphy said. “You’ve got to put
yourself in their shoes, and see that, but they’re so good. They’re turning it into a positive, and they’re just happy for everyone else’s success, so it’s
awesome.” The trio’s grand final heartbreak sparked renewed debate about whether every player on the list should receive a premiership medallion, or at least
those who played a certain number of games. First-year Magpie Oleg Markov has experienced both sides of the ledger. He played in Saturday’s premiership, but
missed out on all three Richmond flags (2017, 2019 and 2020) despite playing senior games in each of those years.
</item 6>

<item 7>
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
</item 7>

<item 8>
Comeback kings Richmond take the AFLW points in Cairns (The Age, 2023-10-20T23:51:34+00:00)
Magpies ruck Sabrina Frederick, who starred with 39 hit-outs and used her height to bring the ball to ground at the advantage of her midfielders, said her side
had been practising for games like this.

“Even in the chaos, it’s quite comfortable because we train it. We train what scenarios to do in what moments, and even though it’s a bit crazy, it felt so
calm. We put players in place and we know what we’re doing,” said Frederick. “But close games are good ones, for everyone. We knew they were going to come out
like that [in the last quarter]. When they’re down, they’re just going to give us everything.” Collingwood’s Brianna Davey kicks her team forward against
Geelong. Credit: Getty Frederick admitted at times she “couldn’t see” and felt like she could be “blown away”.
</item 8>

<item 9>
Ground-breaking Gold Coast: Suns make AFLW finals for a second time (The Age, 2023-11-03T23:45:04+00:00)
AAP Cats secure elimination final home-ground advantage

Geelong have not only secured their spot in the top eight with a huge 40-point win over Hawthorn on Saturday at GMHBA Stadium but locked in home-ground
advantage for the first elimination final. The Cats finished sixth, with the 9.6 (60) to 3.2 (20) win over the Hawks boosting their percentage to a healthy
141.2, almost 40 ahead of Essendon, who sit one rung below them on equal points. Momentum is also in Geelong’s favour after two big wins in a row, having
demolished Richmond by 38 points last weekend. Chloe Scheer (left) and Julia Crockett-Grills of the Cats celebrate. Credit: Getty Against the Hawks, they
controlled ball movement from the get-go, kicking four goals straight to open the match and keeping their opposition goalless to half-time.
</item 9>

<item 10>
Craft beer and cheese boxes: Swans tap into something ‘uniquely Sydney’ to draw AFLW crowds | Jack Snape (The Guardian, 2023-10-26T23:42:29+00:00)
But Livingstone said the AFL was happy with the level of competition. “Our ladder, from position five to 13 – there’s only a game separating it. It just shows
that this is a really competitive competition.”
</item 10>

<item 11>
Ground-breaking Gold Coast: Suns make AFLW finals for a second time (The Age, 2023-11-03T23:45:04+00:00)
Chloe Scheer was once again the star in front of goals with three majors, while the Cats finished with seven individual goalkickers. Georgie Prespakis (35
touches, eight clearances) and Amy McDonald (23 touches, eight clearances) were a formidable midfield duo, while the retiring Renee Garing had a whopping 11
tackles. Overall, it was a strong team effort from a side that will be hungry for a finals win. Last year, the Cats had an impressive home-and-away run but
bowed out narrowly in an elimination final to North Melbourne. Geelong young defender Rebecca Webster told this masthead earlier in the week how important the
home crowd is for them come knockout finals. “Playing at GMHBA is like a massive advantage for us,” said Webster. “We love going in those four walls. We train
there three, four times a week.
</item 11>

<item 12>
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
</item 12>

<item 13>
Ground-breaking Gold Coast: Suns make AFLW finals for a second time (The Age, 2023-11-03T23:45:04+00:00)
Melbourne’s Lauren Peace fights for the ball againt the Lions. Credit: Getty Images The Lions were at risk of slipping from the top four after last week’s
21-point loss to St Kilda, but will now finish the home-and-away season fourth, avoiding a sudden-death elimination final and marking coach Craig Starcevich’s
75th game in style.

Meanwhile, Melbourne ceded the minor premiership to Adelaide and dropped to second spot. The Lions put on a show across the park, with powerhouse duo Ally
Anderson (22 disposals) and Jade Ellenger (22) going to work in the midfield while Dakota Davidson (two goals) led the charge in the forward line. Captain Bre
Koenen was superb in defence as she played on Melbourne’s leader Kate Hore, collecting a game-high 12 intercept possessions and limiting the prolific goalkicker
to just one goal. Lions duo Charlotte Mullins and Ruby Svarc celebrate.
</item 13>

<item 14>
Who’s the next Gawn, Grundy or Dusty? AFL’s talent guru names top 40 draft prospects and compares them to current stars (The Roar | Sports Writers Blog,
2023-11-15T23:48:27+00:00)
Also showed he can go forward and have an impact during his time with the Murray Bushrangers.

Weapon: Rebound ability and athleticism

AFL player comparison: Sam Taylor (GWS Giants)

Ethan Read

State: Queensland

State League Club: Gold Coast Suns Academy

Community Club: Palm Beach Currumbin

Date of Birth 07/07/05

Height: 202cm

Mobile and skillful ruckman whose work rate and ball use are outstanding features of his game, along with his aggressive work at stoppages. A member of the Gold
Coast Suns Academy, he was a dominant force for the Allies at the National Championships and earned All Australian honours. Had a massive impact on all four
games, averaging 21.5 disposals at 76 percent efficiency, 7.5 marks and 13 hitouts to play an important role in the Allies’ first title win.
</item 14>

<item 15>
‘Right people in right roles’: QRU breaks silence on RA’s reform hopes, backs review into Wallabies’ RWC flop (The Roar | Sports Writers Blog,
2023-10-10T23:06:29+00:00)
We need to maximize the development of the players. At the end of the day, that’s the key to success and alignment of your resources, for a small country.

“Because Australia, if you look at the history of rugby, when we were amateur, we were probably ahead of the game because we had the influence of rugby league.
We had the influence of AFL and our players were more professional than the rest of the world.”

He added: “I know what’s going on and we need to get a better system of developing players. We need to develop better players and we need to develop a larger
batch of better players and that only comes through changing the whole system.”

While all five Australian Super Rugby sides fundamentally are on board with regards to moving to a centralised high-performance program, The Roar added that
some states, including the ACT Brumbies as well as the QRU, don’t want to give up their commercial assets.
</item 15>

<item 16>
McRae reveals hidden message he carried under collar of his grand final shirt (The Sydney Morning Herald, 2023-09-30T22:45:39+00:00)
Loading “We just want to win. We come into the club every day and get better, and we just set ourselves up, so when we come to game day, we know we’ve done
everything, to get the result we want.” Maynard had a tumultuous start to September after his infamous collision with Melbourne’s Angus Brayshaw left the Demon
concussed and him defending himself at the tribunal. With a three-match ban hanging over his head, Maynard feared his dream to achieve premiership glory could
be lost before he beat the charge.

Collingwood’s triumph helped banish the pain of their grand final defeat to West Coast in 2018, when Maynard was unfortunate not to receive a free kick for a
block against Willie Rioli in the same passage that led to Dom Sheed’s match-winning goal. “Everyone knows that was a bizarre and crazy sort of week, mentally
[after the Brayshaw incident],” Maynard said. “I wasn’t going well because of the what-ifs, if I missed this.
</item 16>

<item 17>
Ground-breaking Gold Coast: Suns make AFLW finals for a second time (The Age, 2023-11-03T23:45:04+00:00)
But two goals from Amelia Radford and Georgia Gee kept the hosts close, while defender Ellyse Gamble (nine intercept possessions) proved a roadblock for the
Suns’ attackers. Single lit up the park with a clinical slice through the corridor, dodging two tackles to put the ball in the hands of Georgia Clayden, who
capped the centre clearance with the only major of the third term. With both sides goalless in the final quarter, the Suns were able to cruise home. With AAP
</item 17>

<item 18>
Craft beer and cheese boxes: Swans tap into something ‘uniquely Sydney’ to draw AFLW crowds | Jack Snape (The Guardian, 2023-10-26T23:42:29+00:00)
“Even at Henson Park [the Swans’ other home ground in Sydney’s gentrified inner west], they’ve gone to microbreweries in the local area and invited them in, but
also promoted through their clientele to come to the football.”

The Swans forward Bella Smith said the players were proud of the club’s attendance record, and were grateful for the support. “We get there two hours before the
start of the game and the hill [at Henson Park] is bare. When you come out for the first team warm-up, the hill is slowly filling and then by the time the first
bounce gets around, it is full,” she said. “You can hear it, the atmosphere is amazing.”

The average attendance for the AFLW is a little over 2,500 a match this year, tracking approximately on par with previous seasons and with a finals uptick
likely to come.
</item 18>

<item 19>
Comeback kings Richmond take the AFLW points in Cairns (The Age, 2023-10-20T23:51:34+00:00)
Geelong coach Daniel Lowther said the strong winds made it hard for his side to clear the ball out of their defensive 50-metre zone and to move the ball with
any certainty around the ground.

Loading “At one point the rain hit our box, it was flowing,” Lowther said. “Both teams had to fight in the conditions, but it didn’t help the game. “Our marking
contest wasn’t too flash, our ground ball representation was a bit hit and miss, so our defence were under a bit more pressure than they should have been, but
they fought back really well,” Lowther said. Collingwood next play a Sydney line-up including former Magpies star Chloe Molloy, while Geelong will take on
Richmond at Ikon Park.

Dockers spoil Saints’ finals hopes Fremantle recorded a comfortable 24-point win yesterday to bring an abrupt end to St Kilda’s four-game streak and dash their
AFLW finals hopes. J’Noemi Anderson is tackled by a group of Dockers.
</item 19>

<item 20>
Who’s the next Gawn, Grundy or Dusty? AFL’s talent guru names top 40 draft prospects and compares them to current stars (The Roar | Sports Writers Blog,
2023-11-15T23:48:27+00:00)
Shared the role of Victoria Country co-captain with Harley Reid and performed consistently. Had an exciting year with the Dandenong Stingrays averaging 19
disposals and 4.8 tackles to earn selection in the Coates Talent League Team of the Year. Recorded a top 10 finish in the 20m sprint at the national Draft
Combine with a slick 2.99 seconds. A dual sports athlete who was a member of Australia’s under-19 cricket program.

Weapon: Speed and endurance

AFL player comparison: Jordan Dawson (Adelaide Crows)

Zane Duursma

State: Victoria Country

State League Club: Gippsland Power

Community Club: Foster

Date of Birth: 28/08/05

Height: 189cm

A mercurial forward or midfielder who reads the play exceptionally well, is strong overhead and deadly accurate around the goals.
</item 20>

<item 21>
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
</item 21>

<item 22>
Who’s the next Gawn, Grundy or Dusty? AFL’s talent guru names top 40 draft prospects and compares them to current stars (The Roar | Sports Writers Blog,
2023-11-15T23:48:27+00:00)
He only fully focused on a potential AFL career last year after previously playing volleyball. Starred with the Calder Cannons and Victoria Metro this year
kicking 33 goals in the talent pathway. His 2km time of 6:05 at the national Draft Combine was a strong indicator of his elite endurance but also his
competitiveness for a player of his size. Is eligible to join the Western Bulldogs under the father-son rule as the son of former Bulldogs utility Matthew
Croft, who played 186 games for the club.

Weapon: Aerial ability

AFL player comparison: Max King (St Kilda)

Jordan Croft has nominated as a father-son prospect for the Western Bulldogs ????
</item 22>

<item 23>
Comeback kings Richmond take the AFLW points in Cairns (The Age, 2023-10-20T23:51:34+00:00)
The Kangaroos’ Alice O’Loughlin and Melbourne’s Maeve Chaplin battle for the ball. Credit: AFL Photos It was a hard-fought, low-scoring battle between the two
powerhouse sides that play a similar game style. Each set up well, are fast and like a good dose of contested marking. But Narrm controlled more of the game by
tightening around the contest and being cleaner going forward. They also defended extremely well by suffocating the opportunities North created and prevailed
5.7 (37) to 2.2 (14). While the Demons scored just one goal a quarter in the second half – Tyla Hanks in the third and Eden Zanker in the fourth – the Kangaroos
were goalless for three terms, having led by five points at the first change. Their opening goals were thanks to Niamh Martin and Tess Craven.

Then a crafty goal from Demons skipper Kate Hore to open the second term put her side up, and Melbourne never gave up the lead.
</item 23>

<item 24>
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
</item 24>

<item 25>
‘Right people in right roles’: QRU breaks silence on RA’s reform hopes, backs review into Wallabies’ RWC flop (The Roar | Sports Writers Blog,
2023-10-10T23:06:29+00:00)
The QRU also expressed its satisfaction that RA was willing to discuss the annual $1.7m being reinstated, which had been withheld because of the Covid pandemic
to keep the game afloat.

Meanwhile, The Roar understands the ACT Brumbies have sent RA confirmation of solvency for the next 12 months.

RA was forced to take out a $40m loan from Ares to keep the game afloat.

Advertisement

While RA has around another $15m left to spend from the loan, the governing body is looking to borrow up to another $60m so they’re in the strongest possible
position to capitalise on the two major money-making events – the 2025 British and Irish Lions series and home men’s World Cup in 2017 – and succeed.
</item 25>

<item 26>
McRae reveals hidden message he carried under collar of his grand final shirt (The Sydney Morning Herald, 2023-09-30T22:45:39+00:00)
Markov is “a big advocate” for all players being recognised.

“The best thing we can do is wrap our arms around them, and just let him feel what we sort of feel,” he said. “I’ve been on the other side of the fence, and it
is hard because you’re seeing all your friends succeed and sort of living the dream that you aspire to have, but I know it makes people hungrier. I know the
boys who missed out will be the ones who are training the house down, and pushing the boys who have seen the mountain top. They’re going to motivate the boys to
try and see it again.” ‘Incredibly sore’: Howe played out decider with rib injury Peter Ryan

This wasn’t the type of “car crash” injury Jeremy Howe has suffered before, but it was brutal nonetheless. The high-flying Magpie has revealed he played out the
grand final in pain after he was floored by a bone-crunching knock to the ribs from Lions forward Charlie Cameron during the last quarter.
</item 26>

<item 27>
Who’s the next Gawn, Grundy or Dusty? AFL’s talent guru names top 40 draft prospects and compares them to current stars (The Roar | Sports Writers Blog,
2023-11-15T23:48:27+00:00)
His ferocious attack on the ball and his opponent helped set the tone for the Allies to dominate the midfield battle throughout the National Championships and
he was rewarded with All Australian honours. Averaged 29.2 disposals, 11.2 contested possessions and a team-high 6.8 tackles across four matches for the Allies
and showed excellent repeat speed and courage as well as clean hands playing both on the inside and outside. His 6:06 effort in the 2km time trial at the
national Draft Combine showcased his elite endurance. He has been part of the Sydney Swans’ Academy since the under-10s.
</item 27>

<item 28>
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
</item 28>

<item 29>
Ground-breaking Gold Coast: Suns make AFLW finals for a second time (The Age, 2023-11-03T23:45:04+00:00)
“I’m excited like I think the group’s coming together really nicely and where really well-prepared compared to previous years,” Riddell told Fox Footy. “So, I
can’t wait I think it’s an exciting time for us.” The game was one-way traffic against the Dogs with the game played almost entirely in the Roos’ attacking
half. Riddell credited her side’s “team performance” after two losses to the Dees and Crows. North Melbourne have secured a top-four spot for finals. Credit:
Getty For the Western Bulldogs, it was another tough day out as they notched their ninth loss of the season. They won just one game this year, an eight-point
victory last weekend over 17th-placed West Coast.

Britney Gutknecht kicked the Dogs’ only goal against the Kangaroos, which came early in the final term to avoid a record for the lowest AFLW score in the
history of the competition.
</item 29>

<item 30>
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
</item 30>

<item 31>
‘Right people in right roles’: QRU breaks silence on RA’s reform hopes, backs review into Wallabies’ RWC flop (The Roar | Sports Writers Blog,
2023-10-10T23:06:29+00:00)
RA is also in the early stages of looking to renew its next broadcast deal, with the governing body hoping to at least double its current deal with Nine
Entertainment/Stan Sport, which is seeing them paid $29m per year until the end of 2025.
</item 31>

<item 32>
Ground-breaking Gold Coast: Suns make AFLW finals for a second time (The Age, 2023-11-03T23:45:04+00:00)
“I don’t think I’ve been as nervous to kick a goal in my life, to be honest,” she joked. “But nah, it was awesome.” On Saturday, the competition also said
goodbye to a range of players including Western Bulldogs forward Richelle Cranston, who has battled stage five chronic kidney disease while playing this season,
and Hawthorn’s Akec Makur Chuot, who played 40 games for three clubs.

Cranston, who was an inaugural AFLW player, will start kidney dialysis in January so had no option but to hang up the boots and focus on her treatment and
health. “[I’m feeling] sad. Emotions hit me when the siren went. It’s been a really emotional week, the girls have really gotten around me. But yeah, my health
has sort of deteriorated, it’s been a really hard year,” Cranston told Fox Footy after the Bulldogs lost to North Melbourne 7.13 (55) to 1.3 (9).
</item 32>

<item 33>
Ground-breaking Gold Coast: Suns make AFLW finals for a second time (The Age, 2023-11-03T23:45:04+00:00)
We’re stoked. It’s pretty special for the club.” Missing star ruck Ally Morphett to a season-ending wrist injury, Laura Gardiner stood the tallest in the
midfield, while Molloy (two goals), Montana Ham, Privitelli and Brooke Lochland hit the scoreboard. An off-season recruit from Geelong, 20-year-old Gardiner had
the ball on a string with a game-high 33 disposals and 18 contested possessions.

Demons denied the minor premiership The Brisbane Lions have toppled AFLW heavyweights Melbourne by 25 points to clinch a double chance in the finals and deny
the Demons the minor premiership. A year after Melbourne walked away with the 2022 premiership, the Lions avenged their upset grand final loss by defeating the
defending premiers 8.5 (53) to 4.4 (28) at Brighton Homes Arena on Saturday. Melbourne’s Lauren Peace fights for the ball againt the Lions.
</item 33>

<item 34>
Who’s the next Gawn, Grundy or Dusty? AFL’s talent guru names top 40 draft prospects and compares them to current stars (The Roar | Sports Writers Blog,
2023-11-15T23:48:27+00:00)
Weapon: Power and contested ball-winning ability

AFL player comparison: Dustin Martin (Richmond)

Archie Roberts

State: Victoria Metro

State League Club: Sandringham Dragons

Community Club: Hampton Rovers

Date of Birth: 18/11/05

Height: 184cm

A left-footed half-back who is a penetrating kick, smart interceptor and uses his anticipation and clean hands to have an impact. Showed good rebounding speed
and fine decision-making ability to set up play from defence for Victoria Metro during the National Championships, where he averaged 20.8 disposals and 3.8
marks to earn All Australian honours before becoming a back-to-back premiership player for the Sandringham Dragons in the Coates Talent League. His impressive
time of 6:08 for the 2km time trial at the national Draft Combine gave a glimpse of his elite endurance and competitiveness.
</item 34>

<item 35>
Ground-breaking Gold Coast: Suns make AFLW finals for a second time (The Age, 2023-11-03T23:45:04+00:00)
“It was an awesome game and four quarters of Port Adelaide footy … It was awesome to finish up like that,” Phillips told Fox Footy after the 13.10 (88) to 4.6
(30) win over the Giants. Erin Phillips celebrates a goal during her final AFLW match. Credit: AFL Photos Earlier, she told the ABC that she wanted her legacy
to include doing everything she could to grow the game.

“Hopefully I can be remembered as someone who took their opportunities, loved the game and was willing to do whatever it took to ensure the next generation was
in a better position,” she said. Richelle Cranston of the Bulldogs is chaired off the ground by teammates Isabelle Pritchard and Ellie Blackburn after
announcing her retirement. Credit: Getty The 38-year-old kicked a momentous goal from a set shot in the final quarter, and she was smothered by her jubilant
teammates afterwards.
</item 35>

<item 36>
Comeback kings Richmond take the AFLW points in Cairns (The Age, 2023-10-20T23:51:34+00:00)
Molloy, the Swans’ star off-season recruit, turned on one of the best performances of her career to demolish their injury-hit hosts.

The former Collingwood ace was a class above as she kicked 3.4 from her 28 possessions and played a crucial role in setting up numerous other Sydney goals.
Loading “She [Molloy] was itching to get in there [in the midfield],” Swans coach Scott Gowans said. “We kept her forward deliberately. It was important Chloe
played a good block of time forward so we could have a look at it.” The Blues lost to Greater Western Sydney by two points on Saturday afternoon, denying them
the chance to leapfrog into the top eight, 8.1 (49) to 7.5 (47).

Narrm reclaim top spot Reigning premiers Narrm have reclaimed top spot on the ladder and their position as flag favourites after defeating third-placed North
Melbourne by 23 points at Ikon Park on Saturday night.
</item 36>

<item 37>
Who’s the next Gawn, Grundy or Dusty? AFL’s talent guru names top 40 draft prospects and compares them to current stars (The Roar | Sports Writers Blog,
2023-11-15T23:48:27+00:00)
Was also named captain of the All Australian team after leading the Allies to their first title. Caught the eye with a strong performance for the Australia
under-18 team in its representative match against Port Adelaide’s SANFL side and backed it up in the second match against Carlton’s VFL side where he collected
24 disposals to be one of Australia’s best in victory. Finished the year as a back-to-back Coates Talent League premiership player with the Sandringham Dragons
where he averaged 31 disposals across nine matches.
</item 37>

<item 38>
Who’s the next Gawn, Grundy or Dusty? AFL’s talent guru names top 40 draft prospects and compares them to current stars (The Roar | Sports Writers Blog,
2023-11-15T23:48:27+00:00)
Played all four games for Victoria Metro at the National Championships and was his team’s best player in the loss to the Allies, while also shining with 22
disposals in the win over Western Australia. Across his 16 matches with the Eastern Ranges in the Coates Talent League he averaged 19.4 disposals to be one of
their most consistent players during their run to the grand final. Recorded a slick time of 2.92 seconds on the 20m sprint during the national Draft Combine
highlighting his dazzling acceleration and backed it up by recording 6:28 for the 2km time trial which demonstrated a combination of elite speed and endurance
which are essential to play on the wing.

Weapon: Elite speed and ball use

AFL player comparison: Nick Martin (Essendon)
</item 38>

<item 39>
Who’s the next Gawn, Grundy or Dusty? AFL’s talent guru names top 40 draft prospects and compares them to current stars (The Roar | Sports Writers Blog,
2023-11-15T23:48:27+00:00)
???



Watch the Vic Metro v Vic Country game: https://t.co/YWgc6dPtb1 pic.twitter.com/ajKnGaievS — AFL (@AFL) September 22, 2022

Darcy Wilson

State: Victoria Country

State League Club: Murray Bushrangers

Community Club: Wangaratta Rovers

Date of Birth: 18/08/05

Height: 186cm

An elusive and creative forward or wingman who covers the ground exceptionally well and has innate ability around goals. He is still developing physically but
had an outstanding 2023 season. Averaged 25.4 disposals and kicked 16 goals for the Murray Bushrangers to win the team’s best and fairest and earn selection in
the Coates Talent League Team of the Year.
</item 39>

<item 40>
Who’s the next Gawn, Grundy or Dusty? AFL’s talent guru names top 40 draft prospects and compares them to current stars (The Roar | Sports Writers Blog,
2023-11-15T23:48:27+00:00)
Uses his dazzling speed to break the lines and kicks the ball with pinpoint accuracy on his preferred left foot. Averaged 32.8 disposals and 5.8 clearances
across four matches for the Allies at the National Championships to earn All Australian honours. Was equally dominant for the Tasmania Devils in the Coates
Talent League having more than 30 disposals in seven of his nine matches and also kicked 12 goals. Showed he could match it against more experienced opponents
by having 20 disposals and three clearances in Australia’s win over Carlton’s VFL team.
</item 40>

<item 41>
Comeback kings Richmond take the AFLW points in Cairns (The Age, 2023-10-20T23:51:34+00:00)
Dogs forward reveals battle with kidney disease Western Bulldogs forward Richelle Cranston has revealed she’s been living, and playing, with stage five chronic
kidney disease. “I still manage to do the things I love,” the inaugural AFLW player posted to Instagram. “It’s just really hard sometimes.” Richelle Cranston of
the Bulldogs: “It’s just really hard sometimes.” Credit: Getty Images The 34-year-old posted the caption along with a video detailing her life with the disease
including weekly blood tests and daily blood pressure checks, tracking water and food intake, taking medications including resonium (to lower potassium levels),
all while training in the high-performance environment.

The Dogs are Cranston’s third club after starting at Melbourne and having a stint with Geelong before heading to the Bulldogs last year. Cranston lined up for
the Bulldogs in her side’s 57-point loss to Sydney on Saturday at Whitten Oval, 14.12 (96) to 6.3 (39).
</item 41>

<item 42>
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
</item 42>

<item 43>
Ground-breaking Gold Coast: Suns make AFLW finals for a second time (The Age, 2023-11-03T23:45:04+00:00)
Down by four points at half-time in the low-scoring game, a third-term goal from Rebecca Privitelli sparked the Swans to a 5.10 (40) to 2.6 (18) win on Sunday
at Fremantle Oval.

The second-year expansion side stormed home with three unanswered goals to secure eighth spot. It’s a dramatic improvement, with the Swans winless in their
first season. “Any club that’s able to go from winless one year to a finals berth, it’s incredible,” Sydney co-captain Chloe Molloy told Fox Footy after the
match. “You can see the emotion on the girls’ faces. Coming into the season, the odds were against us. Just to get into the eight, we’ve managed to do that.
We’re stoked. It’s pretty special for the club.”
</item 43>

<item 44>
Ground-breaking Gold Coast: Suns make AFLW finals for a second time (The Age, 2023-11-03T23:45:04+00:00)
Lions duo Charlotte Mullins and Ruby Svarc celebrate. Credit: Getty Images The Lions got the game on their terms early with five repeat entries into the
attacking 50m in the opening 10 minutes of the game. Davidson and Ellie Hampson drew first blood before Casey Sherriff kicked Melbourne’s opening goal.

Tayla Harris, Tyla Hanks and Hore slotted a goal each in the second term to put their side ahead. Crumbling under the Brisbane attack, the visitors conceded
four unanswered goals from Taylor Smith, Sophie Conway (two goals) and Ruby Svarc while remaining goalless in the third term. Midfield stars Olivia Purcell (14
contested possessions) and Tyla Hanks (11) showed their mettle as they attempted to drag their side over the line, but the Demons were outclassed and sorely
missing Paxy Paxman (hamstring). Melbourne could be without Maeve Chaplin in their finals campaign, after the defender sustained a concussion injury in the
third term.
</item 44>

<item 45>
McRae reveals hidden message he carried under collar of his grand final shirt (The Sydney Morning Herald, 2023-09-30T22:45:39+00:00)
Jeremy Howe was collected by Charlie Cameron late in the game Credit: Getty Images

Cameron crashed into Howe as he cleared the ball from Collingwood’s defence, leaving him gasping on the ground with badly bruised ribs. The Magpies defender is
accustomed to extreme pain, having broken his arm in six places after a sickening midair collision in the opening round of the season. This pain was tolerable
by comparison, and he got up and played out the match, helping Collingwood hold on to a four-point win in one of the great grand finals. “The incident late in
the game with Charlie ... He managed to hit me relatively late, it was probably in play, [but I] got a down-field free kick,” Howe told Channel Nine on Sunday.
Jeremy Howe’s rib injury did not prevent him celebrating Collingwood’s premiership. Credit: Getty

“The ribs are incredibly sore but I think if we lost they’d be sorer. “I’ve got six months to get them right.
</item 45>

<item 46>
Ground-breaking Gold Coast: Suns make AFLW finals for a second time (The Age, 2023-11-03T23:45:04+00:00)
WEEK ONE FINALS FIXTURE SATURDAY NOVEMBER 11 First Qualifying Final: Adelaide v Brisbane Lions, 2.35pm local (3.05pm AEDT), Norwood Oval First Elimination
Final: Gold Coast v Sydney, 6.15pm local (7.15pm AEDT), Heritage Bank Stadium SUNDAY NOVEMBER 12 Second Elimination Final: Geelong v Essendon, 1.05pm AEDT,
GMHBA Stadium Second Qualifying Final: Melbourne v North Melbourne, 3.05pm AEDT, Ikon Park The eight was not decided until the final game of the regular season,
when Sydney downed Fremantle by 22 points to leapfrog St Kilda into the final spot, after the Saints had kept themselves in the race with a 20-point win over
Carlton earlier in the day, 7.4 (46) to 3.8 (26).
</item 46>

<item 47>
Who’s the next Gawn, Grundy or Dusty? AFL’s talent guru names top 40 draft prospects and compares them to current stars (The Roar | Sports Writers Blog,
2023-11-15T23:48:27+00:00)
A fine intercept mark and displays fantastic speed away from the contest. Averaged 17.7 disposals in 10 matches for the Power to catch the attention of AFL
scouts. Is the brother of Hawthorn star Changkuoth Jiath and a member of the Hawks’ NGA program. Performed solidly during testing at the state Draft Combine in
Melbourne with his 3.03 seconds on the 20m sprint and 6:32 seconds for the 2km time trial both impressive results.
</item 47>

<item 48>
‘Right people in right roles’: QRU breaks silence on RA’s reform hopes, backs review into Wallabies’ RWC flop (The Roar | Sports Writers Blog,
2023-10-10T23:06:29+00:00)
“The very strength of our State unions is the expertise they bring in their own markets.

Advertisement

“We are open to work on commercial initiatives together but will always retain our independence for the benefit of Queensland Rugby, its members and
stakeholders.

“Queensland Rugby has built a sustainable business model through the elimination of legacy operating debt, delivered four-straight operating profits and worked
in partnership with Governments to successfully deliver $31 million of improvements to Ballymore through the construction of the National Rugby Training Centre,
the home of the Wallaroos and Queensland Reds.”

The letter also pointed to the 71,000 players across the state, including more than 16,000 of those female participants, 257 schools featuring rugby as part of
its program, as well as its Hospital Cup competition and regional club competitions, as evidence that the game was in good health in Queensland.
</item 48>

<item 49>
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
</item 49>

<item 50>
‘Right people in right roles’: QRU breaks silence on RA’s reform hopes, backs review into Wallabies’ RWC flop (The Roar | Sports Writers Blog,
2023-10-10T23:06:29+00:00)
In the open letter to their members on Wednesday morning, QRU chief executive David Hanham and chairman Brett Clark moved to update constituents on the reform
measures that RA is attempting to usher through the game.

Last week, The Roar revealed that the New South Wales Rugby Union was in the latter stages of handing over their license to RA.

By doing so, the NSW Waratahs, who were on the brink of insolvency, handed over their commercial and high-performance property.

Advertisement

The decision will see all male and female Waratahs players, including coaches and high-performance staff, as well as all commercial and operations staff, run
out of RA’s office.

NSW Waratahs chief executive Paul Doorn informed stakeholders of the decision last week.

RA is cognisant that not every Super Rugby franchise will follow suit immediately, but the governing body is hopeful that now one of the two biggest states has
agreed to hand over the keys, others will follow suit swiftly.
</item 50>

<item 51>
Who’s the next Gawn, Grundy or Dusty? AFL’s talent guru names top 40 draft prospects and compares them to current stars (The Roar | Sports Writers Blog,
2023-11-15T23:48:27+00:00)
Was an outstanding player for Victoria Country at the National Championships to earn All Australian honours, which included a match-winning performance against
Victoria Metro in the final game where he amassed 22 disposals, 10 marks and four goals. Was also dominant for the Gippsland Power to win their best and fairest
and earn selection in the Coates Talent League Team of the Year. Finished the year with 42 goals across 20 matches in all competitions. His natural leap was
highlighted at the national Draft Combine with his running vertical jump of 97cm ranked second overall. Hails from a talented football family, with his brother
Xavier now at Essendon and sister Yasmin playing for Port Adelaide.
</item 51>

<item 52>
Who’s the next Gawn, Grundy or Dusty? AFL’s talent guru names top 40 draft prospects and compares them to current stars (The Roar | Sports Writers Blog,
2023-11-15T23:48:27+00:00)
???



The 200cm forward is an elite runner, sharp kick and strong mark.#AFLDraft pic.twitter.com/f9twMnNtfO — AFL (@AFL) October 12, 2023

Daniel Curtin

State: Western Australia

State League Club: Claremont

Community Club: West Coast

Date of Birth: 08/03/05

Height: 197cm

A left-footed defender who experienced a frustrating run with injury during the 2022 season. Managed to get back to play in the AFL Futures game at the MCG on
AFL Grand Final day and excelled for Team Houli with 22 disposals and nine marks to claim best on the ground honours.
</item 52>

<item 53>
McRae reveals hidden message he carried under collar of his grand final shirt (The Sydney Morning Herald, 2023-09-30T22:45:39+00:00)
“I’m super grateful that I’ve got a little chant like that, and hopefully, it’s a little legacy I can continue, and I’m just happy the Magpie army are looking
after me. I feel like they’re always on my side, no matter what mistake I make ... if anything, they motivate me to continue to take the game on and just be my
true self.” Markov felt like he was cheating on Collingwood when he began training with the Blues, after finding out the Pies overlooked him for what was then
their final list spot.

That followed Gold Coast delisting the 27-year-old, who also played 23 games for Richmond. There was interest in him at the time, but he feared his career might
be over. “I went through the national draft and no one looked into me, so there were technically 17 clubs that didn’t really want me,” he said.
</item 53>

<item 54>
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
</item 54>

<item 55>
Ground-breaking Gold Coast: Suns make AFLW finals for a second time (The Age, 2023-11-03T23:45:04+00:00)
With only 0.4 of a percentage point separating Sydney in ninth and St Kilda in eighth, the win assured the Swans of retaking eighth spot. If the Saints had
qualified, it would have been their maiden finals appearance. Saints goalkicker Jesse Wardlaw had a heavy collision with Carlton’s Keeley Sherar late in the
final quarter of their clash.

Georgia Patrikios showed her class in the do-or-die match, collecting 25 disposals and eight contested possessions in a powerhouse display, while Hannah Priest
was a brick wall in defence, with nine intercept possessions. Abbie McKay worked overtime to counter Patrikios in the midfield, but her side was outclassed
across the park. McKay racked up 28 disposals and 12 clearances alongside Mimi Hill (30 disposals), while defender Sherar (17) toiled against a potent Saints
attack. Meanwhile, Sydney prevailed in a fierce tug-of-war with the Dockers to seal their first AFLW finals appearance.
</item 55>

<item 56>
Who’s the next Gawn, Grundy or Dusty? AFL’s talent guru names top 40 draft prospects and compares them to current stars (The Roar | Sports Writers Blog,
2023-11-15T23:48:27+00:00)
An excellent decision-maker and ball user by hand or foot, he averaged 19.5 disposals, 12.5 contested possessions and 6.5 clearances to finish third in the
Larke Medal voting and took out Victoria Country’s MVP. Further added to his very impressive record by earning selection in the All Australian team for the
second year in a row. Handled the step up to playing senior football with three impressive games in the VFL and was the best player for the Australia under-18
team in its match against Port Adelaide’s SANFL team.
</item 56>

<item 57>
McRae reveals hidden message he carried under collar of his grand final shirt (The Sydney Morning Herald, 2023-09-30T22:45:39+00:00)
Adams and McStay – who arrived from the Lions in the off-season as an unrestricted free agent – leaned on each other to cope with the difficult moment, and
needed some privacy to deal with their personal misfortune before they could share in the team’s raucous celebrations. McStay may need as many as eight weeks to
recover from the high-grade medial collateral ligament strain he suffered in the club’s preliminary final win. Collingwood’s injured recruit Dan McStay at
training before the grand final. Credit: Eddie Jim “We shed a few tears and a bit of emotion, then we brushed it off and said, ‘Let’s get on with it and go
celebrate with the boys’. It was a great night,” Adams said. John Noble was another sad story, playing every game in the home-and-away season, only to be
dropped for Collingwood’s qualifying final clash with Melbourne and never forcing his way back in.
</item 57>

<item 58>
Comeback kings Richmond take the AFLW points in Cairns (The Age, 2023-10-20T23:51:34+00:00)
If the Lions had lost, they were at risk of falling out of the top four with second-season side Essendon chasing their spot. Last season’s best and fairest Ally
Anderson said her side loved playing at its training ground in Springfield, which it has made a fortress, and work well in hot conditions . “It was really hot.
It was a grind… But yeah, it was a good finish,” Anderson told Fox Footy post-game. “Every week that we play here, every time that we play here, we get better
and better. So, it was just a matter of time before we finally got the first win and then it hopefully [will] open the floodgates a bit.”

The Lions next play St Kilda before taking on the reigning premiers Melbourne in the final round. After this weekend, they have staked their claim as being
premiership contenders again. Adelaide will play top-four side North Melbourne next weekend before 18th-placed Western Bulldogs.
</item 58>

<item 59>
Craft beer and cheese boxes: Swans tap into something ‘uniquely Sydney’ to draw AFLW crowds | Jack Snape (The Guardian, 2023-10-26T23:42:29+00:00)
But more modest attendances in subsequent games, including two sub-2,000 crowds in Perth, suggest the popularity of the Matildas’ run to the semi-finals has
only had a limited impact on support for domestic women’s sport.

skip past newsletter promotion Sign up to Australia Sport Free newsletter Get a daily roundup of the latest sports news, features and comment from our
Australian sports desk Privacy Notice: Newsletters may contain info about charities, online ads, and content funded by outside parties. For more information see
our Newsletters may contain info about charities, online ads, and content funded by outside parties. For more information see our Privacy Policy . We use Google
reCaptcha to protect our website and the Google Privacy Policy and Terms of Service apply. after newsletter promotion

Livingstone said the “green and gold fever” associated with the Women’s World Cup was a one-off and the AFL has plans in place for long-term growth.
</item 59>

<item 60>
Who’s the next Gawn, Grundy or Dusty? AFL’s talent guru names top 40 draft prospects and compares them to current stars (The Roar | Sports Writers Blog,
2023-11-15T23:48:27+00:00)
Progressed from dominating under-18 games at club level to playing seven matches at reserves level averaging 18.7 disposals. Featured among South Australia’s
best in the National Championship matches against the Allies and Victoria Metro and averaged 18 disposals, 3.8 clearances and 8.5 tackles across the tournament.
Showcased his power and speed at the national Draft Combine with his running vertical jump of 95cm ranking third and 2.93 seconds for the 20m spring ranking
fifth.

Weapon: Power and left-foot kick

Player Comparison: Jack Crisp (Collingwood)

Will McCabe

State: South Australia

State League Club: Central District

Community Club: Tanunda

Date of Birth: 29/09/05

Height: 197cm

A tall defender who has also gone forward at times this year and has continued to develop nicely. A competitive type who has a good natural leap and is very
strong overhead.
</item 60>

<item 61>
Who’s the next Gawn, Grundy or Dusty? AFL’s talent guru names top 40 draft prospects and compares them to current stars (The Roar | Sports Writers Blog,
2023-11-15T23:48:27+00:00)
Weapon: Dazzling speed and ball-winning ability

AFL player comparison: Zac Merrett (Essendon)

Ashton Moya

Pronunciation: Moy-ya

State: South Australia

State League Club: Glenelg

Community Club: Plympton

Date of Birth: 15/04/05

Height: 187cm

A medium forward who really caught the eye last year with his goal kicking and marking ability, which included a haul of four majors against Western Australia
at the National Championships. Was slightly hampered by a hip injury at various parts of the 2023 season but still managed to kick 23 goals in 17 matches in the
talent pathway and showed glimpses of his elite potential. Was fully fit at season’s end and excelled at the national Draft Combine with his time of 8.24
seconds on the agility test in the top bracket.
</item 61>

<item 62>
McRae reveals hidden message he carried under collar of his grand final shirt (The Sydney Morning Herald, 2023-09-30T22:45:39+00:00)
McRae revealed the legacy theme when he spoke to players, coaches, past players, club employees, and their friends and family at Collingwood’s official
post-premiership celebrations held at Centrepiece in Melbourne on Saturday night. Craig McRae with Brayden Maynard. Credit: AFL Photos / Getty Images He said
each player on the list did the same, inscribing a personal message under the collar of their own jumper with words describing the legacy they wanted to leave
as a person and a player in 2023. “All the players had inside their jumper something they want to be remembered for,” McRae said.

“It was an action. It was theirs ... when we created legacy, today players ideally lived theirs. In my shirt I had [written] 44 sons. I’ve got three beautiful
daughters now, I don’t have a son, but I consider all these guys in the squad my sons, so I wore that under my collar today.”
</item 62>

<item 63>
Craft beer and cheese boxes: Swans tap into something ‘uniquely Sydney’ to draw AFLW crowds | Jack Snape (The Guardian, 2023-10-26T23:42:29+00:00)
Fans of Australian rules football in the Harbour City have a reputation for being fickle. After a winless campaign in 2022, the Swans – led by ex-Collingwood
player Chloe Molloy – have been competitive this season. The team won their first AFLW match in round one and are within a game of the top eight with two rounds
to play.

But general manager of women’s football at the AFL, Nicole Livingstone, said the Swans’ off-field offering had tapped into something “uniquely Sydney” and
helped make matches more attractive.

“To go to North Sydney Oval and to have quince paste and hand-picked grapes, cheddar cheese and biscuits in a little box that’s got this beautiful wrapping on
it – they’ve really thought about what they’re doing and made it an experience,” she said.
</item 63>

<item 64>
Comeback kings Richmond take the AFLW points in Cairns (The Age, 2023-10-20T23:51:34+00:00)
“Richmond were just absolute beasts, especially their midfielders. They really worked hard all night,” Hawthorn coach Bec Goddard said after the game. “Our work
rate really dropped off and we just watched them play some really good footy. “If you’re a Hawks fan, you would’ve seen some really exciting moments of footy
tonight.

“Keep getting excited. We are coming, we’re growing, we’re getting better.” Hawthorn’s Irish ace Aine McDonagh kicked the opening goal before Tigers skipper
Katie Brennan slotted two unanswered majors to celebrate her 50th game in style. Fellow 50-gamer Greta Bodey put boot to ball to snatch her own goal before
quarter-time. A skirmish erupted in the second term after Brennan took exception to a tackle by Mattea Breed on Laura McClelland. McClelland was gifted a
50-metre penalty following a bump against Sarah Hosking by Sophie Locke and converted from inside the goal square while players were occupied by the tussle.
</item 64>

<item 65>
Who’s the next Gawn, Grundy or Dusty? AFL’s talent guru names top 40 draft prospects and compares them to current stars (The Roar | Sports Writers Blog,
2023-11-15T23:48:27+00:00)
Announced himself as an AFL prospect when he kicked six goals from 26 disposals in the GWV Rebels’ loss to the Dandenong Stingrays in Round 4. He holds his
width when playing on the wing which provides his team with an outlet option with its ball movement. Had a consistent year at club level averaging 19.6
disposals in 14 matches for the Rebels and finished third in the best and fairest while playing in all four matches for Victoria Country. A strong all-round
athlete, he ran a time of 2.98 seconds in the 20m sprint during pre-season testing and then completed the 2km time trial in 6:13 at the national Draft Combine.
</item 65>

<item 66>
Ground-breaking Gold Coast: Suns make AFLW finals for a second time (The Age, 2023-11-03T23:45:04+00:00)
The Dogs have plenty to work on for next season as they look to regain a competitive edge, which was lost after a finals berth last year. Along with being in a
rebuild, they were badly injury-hit this season and used 32 players, more than any other club. Suns secure finals spot AAP Unsung hero Lucy Single has helped
Gold Coast book their second AFLW finals appearance with a 14-point win over Essendon. Losing midfield powerhouse Charlie Rowbottom to an ankle injury proved no
obstacle in the Suns’ maiden clash with the Bombers, as they finished the regular season in fifth spot after a 6.3 (39) to 3.7 (25) victory to kick off pride
round at Great Barrier Reef Arena on Friday.

While their men’s side have yet to make a finals appearance, this is the second time Gold Coast’s women have clinched a play-off spot, having also done so in
the cancelled 2020 season.
</item 66>

<item 67>
McRae reveals hidden message he carried under collar of his grand final shirt (The Sydney Morning Herald, 2023-09-30T22:45:39+00:00)
The Magpies coach said on Saturday night he was yet to return to St Vincent’s Hospital where his wife Gabrielle gave birth to their daughter Maggie at 7.45am on
grand final morning. Gabrielle went to the hospital with her sister at 10pm on Friday night after going into labour, before ringing McRae at 6am to join them at
the hospital to be with them when the baby was born. Collingwood president Jeff Browne praised McRae and the players for their character and driving the change
the club undertook when the new coach was appointed. He said the win was “the ultimate satisfaction” for him, and he was rapt to be there after a health scare
put him in hospital briefly on the eve of the finals.

“[It was] a bit of pit stop, a little bit of late pre-season training and got myself back into it,” Browne said.
</item 67>

<item 68>
Comeback kings Richmond take the AFLW points in Cairns (The Age, 2023-10-20T23:51:34+00:00)
Comeback kings Richmond take the points in Cairns A late goal by Emelia Yassir has helped Richmond to their first win in four games, putting their AFLW finals
campaign back on track with an 11-point victory over Hawthorn. Tight tussle: Richmond’s Emelia Yassir is tackled. Credit: Getty Images Separated by one point at
half-time and down by nine points entering the last quarter, the Tigers pulled off a 6.5 (41) to 4.6 (30) victory to keep their finals hopes alive at Cazaly’s
Stadium in Cairns on Friday night.

Hawthorn were limited to just one point in the decisive final quarter. Midfield duo Monique Conti (35 disposals, seven clearances) and Grace Egan (28, 5) were
typically influential for Richmond, while Hawthorn were led by Emily Bates (18, four) and Jasmine Fleming (18, one). “Richmond were just absolute beasts,
especially their midfielders.
</item 68>

<item 69>
‘Right people in right roles’: QRU breaks silence on RA’s reform hopes, backs review into Wallabies’ RWC flop (The Roar | Sports Writers Blog,
2023-10-10T23:06:29+00:00)
The QRU confirmed that on Wednesday morning.

Advertisement

“The QRU is committed to working with Rugby Australia to lead reform of our game at a high-performance level, and help Rugby Australia develop an improved game
through high-performance centralisation,” the statement read.

“Centralisation” would include a national player-contracting model, and alignment of key high-performance staff, such as coaching, strength and conditioning,
and medical to operate as one national system.”

The letter pointed to RA’s involvement in Les Kiss’ appointment as Reds coach, with the former State of Origin winger turned coach, who has spent the best part
of two decades coaching in Ireland and England, replacing Brad Thorn.

As The Roar has previously reported, the QRU stopped short of wanting to hand over its commercial property.

“While the QRU is very supportive of high-performance alignment, it will not agree to any proposal on centralising commercial or corporate functions,” the
letter read.
</item 69>

<item 70>
Comeback kings Richmond take the AFLW points in Cairns (The Age, 2023-10-20T23:51:34+00:00)
“Tackled above waist and below shoulders. No sling or double motion. Sheppard lands on her back. ′ ‘Girls’ play hard too,” Harford said on X, the social media
site formerly known as Twitter. This echoed Collingwood defender and Fox Footy commentator Ruby Schleicher’s thoughts at the time, who called it a “monster
tackle” but believed it was “fair”. “I don’t see any dirtiness in that,” Schleicher said on Fox Footy. “It was pretty heavy on her head, wasn’t it? I look at
that – I still see a fair tackle. There’s no doubt the match review [officer] will look at it because they have to I guess. The duty of care for the player.”

The decision a week comes after Adelaide premiership defender Najwa Allen was handed a three-ban suspension, the longest ban in AFLW history, for a high bump on
Western Bulldog Kirsten McLeod. The Crows elected not to challenge the decision.
</item 70>

<item 71>
Comeback kings Richmond take the AFLW points in Cairns (The Age, 2023-10-20T23:51:34+00:00)
Michael Prior guided West Coast to a gutsy win. Credit: Getty

Aisling McCarthy and Amy Franklin drew first blood for West Coast within the opening five minutes, while the hosts were kept goalless. The Bombers only added
three behinds in the second term and the Eagles failed to score as the blustery conditions took their toll on the game. West Coast’s Kellie Gibson broke the
goal drought with the only major of the third quarter, before the home side came to life with the wind at their backs in the final term, with skipper Bonnie
Toogood and Amber Clarke reducing the margin to four points. Toogood continued to pepper the scoreboard, but the Eagles held on to secure a rare victory. “We’ve
had a horror week back at home, and it’s been really bloody tough on all of us, and Mick as well. I’m just really proud of the girls,” West Coast captain Emma
Swanson said post-game on the broadcast. “I can name the whole team.
</item 71>

<item 72>
Who’s the next Gawn, Grundy or Dusty? AFL’s talent guru names top 40 draft prospects and compares them to current stars (The Roar | Sports Writers Blog,
2023-11-15T23:48:27+00:00)
A competitive type who has a good natural leap and is very strong overhead. He had an excellent year at under-18 level for Central District where he averaged
21.8 disposals and 5.9 marks across nine matches. He was also one of South Australia’s better players at the National Championships. Is eligible to join
Hawthorn under the father-son rule as the son of Luke McCabe who played 138 games for the Hawks from 1995 to 2004.

Weapon: Overhead marking

AFL player comparison: Jordan Ridley (Essendon)

Advertisement

Colby McKercher

State: Tasmania

State League Club: Tasmania Devils

Community Club: Launceston

Date of Birth: 12/04/05

Height: 180cm

Had an outstanding year which included winning the Coates Talent League’s Morrish Medal and finishing runner-up in the Larke Medal at the National
Championships.
</item 72>

<item 73>
Who’s the next Gawn, Grundy or Dusty? AFL’s talent guru names top 40 draft prospects and compares them to current stars (The Roar | Sports Writers Blog,
2023-11-15T23:48:27+00:00)
Weapon: Composure and step through congestion

AFL player comparison: Adam Cerra (Carlton)

George Stevens

State: Victoria Country

State League Club: GWV Rebels

Community Club: South Warrnambool

Date of Birth: 14/04/05

Height: 189cm

Advertisement

A strong-bodied defender who also plays as an inside midfielder to great effect. He reads the cues to intercept well when playing across half back, is a good
decision-maker and has a penetrating kick. Had an outstanding year which included captaining the Australia under-18 team during the AFL Academy’s matches
against SANFL and VFL sides. Won All Australian honours at the National Championships when representing Victoria Country and was named captain of the Coates
Talent League Team of the Year after averaging 28.8 disposals and five tackles across 13 matches for the GWV Rebels, also winning the club’s best and fairest
award.
</item 73>

<item 74>
‘Right people in right roles’: QRU breaks silence on RA’s reform hopes, backs review into Wallabies’ RWC flop (The Roar | Sports Writers Blog,
2023-10-10T23:06:29+00:00)
In the wake of the Wallabies’ World Cup flop, the Queensland Rugby Union have reaffirmed their position that they are committed to moving to a centralised
high-performance model but won’t follow the NSW Waratahs by handing over the keys to Rugby Australia.

They also expressed their strong support for Rugby Australia to conduct a “fully independent and transparent top-to-bottom review of the Wallabies program and
how it can be improved” in light of the historic first World Cup pool exit.

They added: “It should also provide a clear view of what is needed in a leadership and governance framework to ensure we put the right people, in the right
roles, in the right system.”

Facebook Twitter Whatsapp Reddit Email Share

The QRU has also expressed their desire for Rugby Australia to reinstate the annual $1.7 million in funding to each of the Super Rugby franchises that was
withdrawn following the Covid pandemic to help keep the game afloat.
</item 74>

<item 75>
Craft beer and cheese boxes: Swans tap into something ‘uniquely Sydney’ to draw AFLW crowds | Jack Snape (The Guardian, 2023-10-26T23:42:29+00:00)
But crowds are short of targets that could accelerate the extension of the regular season, which grows from 10 games this year to 12 games by 2025.

5,474 fans were at North Sydney Oval to watch the Swans’ first AFLW victory earlier in the season. Photograph: Jack Snape/The Guardian

The league enjoyed strong attendances in round one, which were partly attributed to the so-called Matildas effect, but some clubs have struggled to attract fans
in large numbers this season. Crowds of less than 1,500 have been recorded in both traditional markets such as Melbourne and Perth, as well as areas targeted
for growth like Blacktown and the Gold Coast.

The A-League Women launched this month with a first round highlighted by a crowd of more than 11,000 in Sydney.
</item 75>

<item 76>
Comeback kings Richmond take the AFLW points in Cairns (The Age, 2023-10-20T23:51:34+00:00)
The Crows elected not to challenge the decision. ‘Horror week’: Bombers stunned by struggling, controversy-hit Eagles West Coast withstood a late-term surge
from Essendon to narrowly claim their second win of the AFLW season.

Fresh off a controversy caused by comments their coach Michael Prior made following last week’s 70-point loss to Melbourne, the Eagles prevailed 3.5 (23) to 2.7
(19) in a slugfest at the appropriately named Windy Hill on Sunday. West Coast’s Kate Bartlett (left) and Isabella Lewis celebrate Sunday’s surprise win over
Essendon flanked by their teammates. Credit: Getty Prior had blasted the decision to pit strugglers West Coast against defending premiers Melbourne, saying:
“How we play that team is beyond me. That is what you get when you get fixturing like that.” He later backtracked on the criticism, but his comments did put
added pressure and focus on his team this week. Michael Prior guided West Coast to a gutsy win.
</item 76>

<item 77>
Comeback kings Richmond take the AFLW points in Cairns (The Age, 2023-10-20T23:51:34+00:00)
Bridie Hipwell kicked the only goal of the third quarter to give Hawthorn a game-high nine-point lead. Loading Richmond rebounded in the last term, with
Courtney Jones ending Hawthorn’s three-goal run with her first touch of the night, cutting the margin to three points. Dogged by inaccuracy in front of goal,
Yassir reduced the deficit to a single point before finally breaking through on her third attempt. That put the Tigers ahead by six points with 11 minutes left,
before Caitlin Greiser sealed the win with a massive goal from 45 metres.
</item 77>

<item 78>
Comeback kings Richmond take the AFLW points in Cairns (The Age, 2023-10-20T23:51:34+00:00)
Conversely, the score was the most the Saints have conceded in any quarter this season.

Aine Tighe set the tone for the Dockers with a brilliant goal less than 30 seconds into the first quarter, followed by two majors from Gabby O’Sullivan and
another by Angelique Stannett. Ella Friend slotted a much-needed goal for the visitors to make it 4.2 (26) to 1.3 (9) at quarter-time. With less than a minute
till the long break, captain Hayley Miller (two goals) kicked the only major of the second quarter to give Fremantle a comfortable 26-point lead at half-time.
Despite two goals from Tyanna Smith and J’Noemi Anderson in the second half, the Saints struggled to eat into the deficit. Fremantle (4-4) face reigning
premiers Melbourne next, before finishing the season against 11th-placed Sydney.
</item 78>

<item 79>
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
</item 79>

<item 80>
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
</item 80>

<item 81>
Who’s the next Gawn, Grundy or Dusty? AFL’s talent guru names top 40 draft prospects and compares them to current stars (The Roar | Sports Writers Blog,
2023-11-15T23:48:27+00:00)
Holds his ground very well both under the ball in stoppages and marking contests and provides great opportunities for his teammates at ground level. Averaged 12
disposals and 24 hitouts when representing Victoria Metro at the National Championships to earn All Australian honours. Both his running vertical jump of 87cm
and 2km time of 6:34 seconds at the national Draft Combine were impressive indicators of his natural leap and endurance.

Weapon: Mobility and marking

AFL player comparison: Brodie Grundy (Sydney Swans)

Clay Hall

State: Western Australia

State League Club: Peel Thunder

Community Club: Harvey Brunswick Leschenault

Date of Birth: 12/05/05

Height: 189cm

Advertisement

Prolific disposal winner who uses the ball effectively on his preferred left foot and excels at stoppages using his clean hands and vision to create for his
team.
</item 81>

<item 82>
Ground-breaking Gold Coast: Suns make AFLW finals for a second time (The Age, 2023-11-03T23:45:04+00:00)
Single filled the massive Rowbottom-size gap, collecting 25 disposals and six clearances in her prolific performance across the field while also playing a
crucial tagging role on Maddy Prespakis. Coast with the most: Georgia Clayden of the Suns celebrates after kicking a goal. Credit: AFL Photos / Getty Images The
former rugby sevens player was well-supported by Claudia Whitford (25 possessions) and Alison Drennan (22 touches). Prespakis was held to eight contested
possessions, well below a season average of 14 per game, while Georgia Nanscawen proved a potent threat with 13 contested possessions and three score
involvements.

The Bombers broke through first with a goal from Amber Clarke but young talent Darcie Davies (two goals) was able to respond after being gifted 50 metres from a
protected area infringement by Mia Bush. Daisy D’Arcy, Jacqueline Dupuy, Jamie Stanton and Davies attempted to break away from the low-scoring tug-of-war in the
second with goals of their own.
</item 82>

<item 83>
Comeback kings Richmond take the AFLW points in Cairns (The Age, 2023-10-20T23:51:34+00:00)
Hanks was pivotal for the Demons and starred with 26 disposals, seven tackles and one goal, while her partner in the midfield Olivia Purcell had 28 touches,
five clearances and five tackles. For the Roos, Ash Riddell led the disposal count with 31, while skipper Emma Kearney was crucial in defence. Melbourne’s
Alyssa Bannan is tackled by Lulu Pullar of the Kangaroos. Credit: Getty Hanks said with the state of competition, they just had to take it week by week, but it
was good to get the win. “We knew we were up against the best tonight,” said Hanks. “It’s a nice feeling tonight.”

Lions roar in big battle Meanwhile, on Saturday afternoon, the Brisbane Lions put an end to Adelaide’s unbeaten run with a three-point victory at Brighton Homes
Arena in a top-four thriller, 5.12 (42) to 6.3 (39), in a spectacle worthy of being a final.
</item 83>

<item 84>
Comeback kings Richmond take the AFLW points in Cairns (The Age, 2023-10-20T23:51:34+00:00)
“I can name the whole team. Everyone had their second in the last couple minutes.

“That was such a team effort, everyone had a moment there when they had to put their head over the footy and lay a tackle and not let it slip.” Swanson (25
disposals, eight clearances) stood out for the Eagles, while Bombers workhorse Maddy Prespakis (30 disposals, nine clearances) was typically prominent in the
midfield alongside Georgia Nanscawen (21, six). With two rounds left, West Coast have a chance to build their winning streak when they meet the bottom of the
table Western Bulldogs next round. Slipping one place to sixth, Essendon have must-win matches against Carlton and Gold Coast on the road to the finals. The
Bombers will be without Steph Cain, who was sidelined with a concussion after an elbow to the head from Swanson.
</item 84>

<item 85>
McRae reveals hidden message he carried under collar of his grand final shirt (The Sydney Morning Herald, 2023-09-30T22:45:39+00:00)
Past presidents Mark Korda and Eddie McGuire, as well as previous coach and champion player Nathan Buckley, Anthony Rocca, Luke Ball and Sharrod Wellingham were
among the throng celebrating the club’s 16th premiership. Browne said the club wanted to recognise and acknowledge that any success was built on the efforts of
those who went before them. “We are one big club. We are Collingwood ... this club has got a rich history, and days like this celebrate that history and salute
the people that have gone before. We should never forget that,” Browne said.
</item 85>

<item 86>
McRae reveals hidden message he carried under collar of his grand final shirt (The Sydney Morning Herald, 2023-09-30T22:45:39+00:00)
“And, to be honest, maybe 18, after I didn’t get the last list spot at Collingwood, but I’m glad the stars aligned the way they did.” In a twist of fate,
defender Charlie Dean re-fractured his right foot at training, opening another list spot for Collingwood to snatch Markov back from Carlton – and the rest is
history. “Having trained at Collingwood for two months, I felt like I was almost doing the dirty on ‘Fly’ [Craig McRae], going behind his back – I was almost
like a cheater,” he said. “He was very respectful and understanding, and it probably allowed me to open myself up and be vulnerable to that football club. I’m
super grateful for them showing interest in me, in the first place.

“I know I’ve signed with [Collingwood] and on the dotted lines, I’ve agreed to hate Carlton, but there’s a bit of me that is really grateful and appreciative to
them.”
</item 86>

<item 87>
Comeback kings Richmond take the AFLW points in Cairns (The Age, 2023-10-20T23:51:34+00:00)
J’Noemi Anderson is tackled by a group of Dockers. Credit: AFL Photos The Dockers dominated the Saints from the start to win 6.9 (45) to 3.3 (21) at Fremantle
Oval. Having been on the verge of making their first finals appearance, St Kilda (4-4) could now miss the top eight even if they beat the fourth-placed Brisbane
Lions and the ninth-placed Carlton.

Dana East was everywhere for Fremantle, leading the game in disposals (19) and contested possessions (11), alongside Orlagh Lally (19 disposals, five
clearances). Jamiee Lambert was the shining light for the otherwise subdued Saints, collecting a game-high seven clearances and 13 contested possessions. Docker
Hayley Miller hunts a loose ball against St Kilda. Credit: Getty Fremantle flew out of the gates from the first siren, producing their highest-scoring quarter
of the season with 26 points in the opening term.
</item 87>

<item 88>
Who’s the next Gawn, Grundy or Dusty? AFL’s talent guru names top 40 draft prospects and compares them to current stars (The Roar | Sports Writers Blog,
2023-11-15T23:48:27+00:00)
Averaged 17.1 disposals at 76 percent efficiency and kicked seven goals across 12 matches in the Coates Talent League to announce himself as a Draft prospect.
His 6:24 in the 2km time trial at the state Draft Combine in Victoria was impressive for a player of his type.

Weapon: Rebound and ball use

AFL player comparison: Bailey Dale (Western Bulldogs)

Mitchell Edwards

State: Western Australia

State League Club: Peel Thunder

Community Club: South Mandurah

Date of Birth: 02/06/05

Height: 206cm

Advertisement

An exciting ruckman still growing into his body. Has the height and natural leap to give his team a distinct advantage at stoppages. Has continued to develop
this year and shown glimpses of his aerial ability at state under-18 level and took several exciting marks during the National Championships.
</item 88>

<item 89>
Who’s the next Gawn, Grundy or Dusty? AFL’s talent guru names top 40 draft prospects and compares them to current stars (The Roar | Sports Writers Blog,
2023-11-15T23:48:27+00:00)
Has already played nine matches at senior level with Peel Thunder averaging 18.9 disposals. Starred when he represented Western Australia at the National
Championships averaging 27.2 disposals, 11 contested possessions, six clearances and five tackles to win All Australian honours. Is the son of former West Coast
Eagles and Geelong Cats forward Derek Hall but is not eligible as a father-son to either club given Derek played only two games for the Eagles and 74 matches
for the Cats. Didn’t test at the national Draft Combine due to injury.
</item 89>

<item 90>
Ground-breaking Gold Coast: Suns make AFLW finals for a second time (The Age, 2023-11-03T23:45:04+00:00)
“But I go out with on a happy note. I love these girls and I’ll be over at the pancake pocket [Pancake Parlour at Whitten Oval] watching them from now on and
have a few bevs on the sideline.” Roos’ lock in top-four spot; Dogs have issues to work through in off-season North Melbourne secured third spot on the ladder
and the double chance in finals after downing the Western Bulldogs by a whopping 46 points. They will head into finals with momentum but a glaring issue to work
on: kicking accuracy. Despite the comfortable win, they could have done a lot more damage after booting double the number of behinds to goals.

Star midfielder Ash Riddell, who had 36 touches and seven clearances, said she was excited to go up against Melbourne or Adelaide in the first weekend of finals
depending on other results this round, as they were the only two sides who beat them this year.
</item 90>

<item 91>
Comeback kings Richmond take the AFLW points in Cairns (The Age, 2023-10-20T23:51:34+00:00)
The Lions’ Orla O’Dwyer speeds away against the Crows. Credit: Getty Images Adelaide started fast but the game evened up as the Lions had the better of the
second quarter with three goals to one. The Crows then took back control in the third with three goals to none, which set the stage for a tense final term. The
fourth quarter was a heated arm wrestle, and scores were tied with four minutes left on the clock. The Lions’ Bella Dawes kicked the only goal of the last term,
but it was a series of behinds in the final minutes that won the game for them, including one from Lily Postlethwaite that broke the stalemate.

Adelaide’s Niamh Kelly navigates her way through traffic against brisbane. Credit: Getty The two sides have had intense battles in the competition’s history,
including two grand finals, and this was no different.
</item 91>

<item 92>
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
</item 92>

<item 93>
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
</item 93>

<item 94>
Comeback kings Richmond take the AFLW points in Cairns (The Age, 2023-10-20T23:51:34+00:00)
AAP Pies hold on against comeback Cats

Sunday’s clash between Collingwood and Geelong at Victoria Park was less about who was the overall better side than it was decided by who could adapt best to
the blustery wind and rain. The answer: the Magpies. Collingwood survived a fourth quarter comeback from Geelong to finish with a six-point victory and stay
firmly in finals contention, whereas the Cats now face an uphill battle to finish in the eight in two weeks’ time. Jordyn Allen of the Magpies is tackled by
Geelong’s Claudia Gunjaca in the round-eight match at Victoria Park. Credit: Getty Geelong scored just one point in the first quarter and had just 1.3 on the
board at three-quarter-time, but kicked two goals to none in the final term to close in on Collingwood.
</item 94>

<item 95>
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
</item 95>

<item 96>
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
</item 96>

<item 97>
Who’s the next Gawn, Grundy or Dusty? AFL’s talent guru names top 40 draft prospects and compares them to current stars (The Roar | Sports Writers Blog,
2023-11-15T23:48:27+00:00)
Continued that form into this season and was outstanding for Western Australia at the National Championships winning his team’s MVP award and All-Australian
honours after averaging 23.5 disposals and 6.5 marks playing as a defender and midfielder. Played a handful of senior games for Claremont and ended the season
being one of his team’s best in a premiership win at under-18 level. His agility, speed for his size and precise kicking are all features of his game.
</item 97>

<item 98>
Who’s the next Gawn, Grundy or Dusty? AFL’s talent guru names top 40 draft prospects and compares them to current stars (The Roar | Sports Writers Blog,
2023-11-15T23:48:27+00:00)
AFL Talent Ambassador Kevin Sheehan has revealed his top 40 prospects ahead of this month’s AFL Draft.

Sheehan’s picks were announced in a press release by the AFL on Thursday. The 2023 draft will be held on November 20 and 21 with the rookie draft to follow a
day later.

Sheehan has profiled all 129 players who attended the 2023 AFL Draft Combine or 2023 AFL State Draft Combine last month. His full list can be found at the AFL
Draft Hub.

Facebook Twitter Whatsapp Reddit Email Share

The AFL Draft will be shown on Fox Footy and Kayo.
</item 98>

<item 99>
Who’s the next Gawn, Grundy or Dusty? AFL’s talent guru names top 40 draft prospects and compares them to current stars (The Roar | Sports Writers Blog,
2023-11-15T23:48:27+00:00)
Weapon: Clean hands and ball use

AFL player comparison: Daniel Rioli (Richmond)

Jake Rogers

State: Queensland

State League Club: Gold Coast Suns Academy

Community Club: Broadbeach

Date of Birth: 02/03/05

Height: 170cm

A tenacious livewire through the midfield, the Gold Coast Suns Academy member is a one-touch player with great vision and creativity. Showed all those traits at
the National Championship by starring for the Allies where he consistently set up his team through the middle of the ground. Quick and with elite agility, he
was always in the contest. Averaged 22.5 disposals at 77 percent efficiency, 5.5 clearances and 5.2 tackles to be a driving force for the title-winning team.
</item 99>

<item 100>
McRae reveals hidden message he carried under collar of his grand final shirt (The Sydney Morning Herald, 2023-09-30T22:45:39+00:00)
Brisbane Lions (2001-03), Geelong (2007, 2009, 2011), Hawthorn (2013-15) and Richmond (2017, 2019-20) all won three flags in short succession this century, with
the Magpies’ challenge now to back up their memorable season. Brayden Maynard enjoys the Pies’ win. Credit: Eddie Jim Maynard initially baulked when asked
whether Collingwood could win back-to-back flags, but said their bulletproof culture instilled belief they could “do anything”. “We’ve got to lap [this
premiership] up, and we’re going to have some serious fun, and we might burn down Melbourne,” a still-disbelieving Maynard said.

“I’m looking forward to celebrating with the team because it’s taken two years of getting after it and getting better every day – we’ve got better for this
moment. Loading “We just want to win.
</item 100>
=== END ITEM id=62aaf07d8741b8e8 ===

=== ITEM id=112ccb4e6d4fa4ae ===
Gold sentences and facts:
Sentence 0: The Buckeyes settled for a 52-yard field goal late in the first half, which Jayden Fielding missed wide left.
  Fact 0.0: The Buckeyes settled for a 52-yard field goal late in the first half.
  Fact 0.1: Jayden Fielding missed the Buckeyes' 52-yard field goal wide left.
Sentence 1: His 31-yard field goal is good, and the Buffs take the first lead of the game on their first drive.
  Fact 1.0: A 31-yard field goal is good.
  Fact 1.1: The Buffs take the first lead of the game on their first drive.

Retrieved items:

=== END ITEM id=112ccb4e6d4fa4ae ===

=== ITEM id=7dd88d448ba2d090 ===
Gold sentences and facts:
Sentence 0: Rangers also make a lot of choices during character creation (and leveling up) that determine how they play.
  Fact 0.0: Rangers make a lot of choices during character creation.
  Fact 0.1: Rangers make a lot of choices during leveling up.
  Fact 0.2: The choices Rangers make during character creation and leveling up determine how they play.
Sentence 1: During character creation, you must pick the otherworldy patron that you’ve decided to serve.
  Fact 1.0: During character creation, you must pick an otherworldly patron.
  Fact 1.1: The patron you must pick during character creation is the one you have decided to serve.

Retrieved items:
<item 1>
How is it still getting worse for Sam Bankman-Fried? (The Verge, 2023-10-12T23:08:14+00:00)
The recording was later sent to Drappi, who sent it to the government. Joseph Bankman, the defendant’s father and a senior advisor to FTX’s philanthropic arm,
wasn’t in the room for the recording. Barbara Fried looked unhappy, rubbing her left temple as though she had a tremendous headache.

And in a small miracle of pacing, the government got Zac Prince, the founder and former CEO of crypto lender BlockFi, on the stand just long enough to blame
FTX’s bankruptcy for BlockFi’s subsequent bankruptcy before we broke for the day.

I’ve been asked by some people why the reporting coming out of the trial seems so skewed toward the prosecution. There’s an answer for this: the prosecution has
put together a strong, comprehensive indictment of Bankman-Fried’s behavior at FTX and Alameda. The defense, so far, has managed to do almost nothing in
response.
</item 1>

<item 2>
Sam Bankman-Fried was a terrible boyfriend (The Verge, 2023-10-10T23:50:21+00:00)
FTX was a fraud “from the start,” the Securities and Exchange Commission alleges — with a “multi-billion-dollar deficiency caused by his own misappropriation of
customer funds.” Follow along for all the latest news and regular updates from the trial.

The day started off promisingly for the defense as it cross-examined Gary Wang, the chief technology officer of FTX and co-owner of both FTX and Alameda.
Christian Everdell, one of Bankman-Fried’s defense attorneys, couldn’t undo the damage of last week’s code review. But he managed to shake the rust off long
enough to make Wang sound less reliable, drowning the jury in confusing technicalities.

Last week, Wang testified that Alameda got access to a special credit line and an option to take its balance into the negative without triggering liquidation —
something he alleged other accounts at FTX didn’t get. Everdell tried to undermine this claim by talking about the spot margin program, which let users lend
each other assets for margin trading.
</item 2>

<item 3>
Sam Bankman-Fried didn’t ask where the $8 billion went (The Verge, 2023-10-31T19:52:07+00:00)
Oh, but he was testifying to Congress about keeping customer funds safe.

Sam Bankman-Fried, the founder of failed cryptocurrency exchange FTX, is on trial for seven counts of wire fraud and conspiracy. FTX was a fraud “from the
start,” the Securities and Exchange Commission alleges — with a “multi-billion-dollar deficiency caused by his own misappropriation of customer funds.” Follow
along for all the latest news and regular updates from the trial.

Who was making decisions to spend $8 billion of customer funds? Bankman-Fried couldn’t recall knowing anything about it. Were there rules or requirements for
how money borrowed from FTX would be returned? Were there rules for risk management? “I was concerned with overall risk management,” Bankman-Fried said.

But it was the testimony about June 2022 that resonated the most to me. Didn’t Bankman-Fried ask what “fiat@ftx” was? He did.
</item 3>

<item 4>
How is it still getting worse for Sam Bankman-Fried? (The Verge, 2023-10-12T23:08:14+00:00)
I have been waiting for a juicy cross-examination, as I live for chaos and drama. I am beginning to think I am not going to get one.

Ellison had given, in her direct testimony, fairly damning evidence tying FTX CEO Sam Bankman-Fried to the conspiracy to take FTX customer funds. There were
fake balance sheets, one of which was sent to crypto lender Genesis. After a Genesis representative received the balance sheet, he texted Ellison to tell her
he’d spoken to Bankman-Fried — strongly suggesting that Bankman-Fried was aware of the contents of the fake balance sheet. Not great!

Sam Bankman-Fried, the founder of failed cryptocurrency exchange FTX, is on trial for seven counts of wire fraud and conspiracy. FTX was a fraud “from the
start,” the Securities and Exchange Commission alleges — with a “multi-billion-dollar deficiency caused by his own misappropriation of customer funds.” Follow
along for all the latest news and regular updates from the trial.
</item 4>

<item 5>
Is Sam Bankman-Fried a bad ‘man’ or a good ‘boy’? Lawyers swap opening statements before first witnesses take the stand (Fortune, 2023-10-04T23:32:53+00:00)
“One year ago, it looked like the defendant was on the top of the world,” began Thane Rehn, a prosecutor for the government, in his opening statement. The
former CEO of FTX oversaw a supposedly thriving crypto exchange, jetted between international locales, and hobnobbed with celebrities like Tom Brady and Larry
David. He repeatedly emphasized to customers that their money was safe and secure.

But “all of that, all of it, was built on lies,” Rehn declared to the jury. “Behind the curtains, he was not what he appeared to be.” What followed was a
roughly 30-minute story that repeatedly emphasized how Bankman-Fried allegedly stole customer funds to facilitate his jet-setting lifestyle, donate millions to
political candidates, and finance risky bets.

The key to his alleged scheme? Alameda Research, a crypto hedge fund he also owned, argued Rehn.
</item 5>

<item 6>
The jury finally hears from Sam Bankman-Fried (The Verge, 2023-10-28T00:12:41+00:00)
When he found it, he was “very surprised!”

Besides the two obvious lies Bankman-Fried told on the stand — about Alameda Research’s name and about “allow_negative” — I have been struck by how little he
seems to know about his own companies. Apparently, Singh, Wang, and Ellison were out there just doing whatever their little hearts desired. Because
Bankman-Fried was a CEO, but definitely not the kind that pays any attention to money at his crypto trading firm and futures exchange.
</item 6>

<item 7>
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
</item 7>

<item 8>
Is Sam Bankman-Fried a bad ‘man’ or a good ‘boy’? Lawyers swap opening statements before first witnesses take the stand (Fortune, 2023-10-04T23:32:53+00:00)
Rather, he was willing to give up his personal wealth to make customers whole, Cohen argued.

“In the end, Sam started and built two billion-dollar businesses,” he concluded. “He didn’t steal any money.”

A Frenchman who lives in London testifies in New York

After lawyers from both sides depicted two very different Bankman-Frieds, the prosecution called its first two witnesses to the stand—and they weren’t
blockbuster names or former lieutenants-turned-government-cooperators, like Ellison.

The first was a victim: Marc-Antoine Julliard, a Paris-born cocoa trader who lives in London. In 2021, Julliard, who had coiffed hair and spoke with a strong
French accent, decided to invest in crypto and landed on FTX as his exchange of choice, where he traded cryptocurrencies like Bitcoin and Dogecoin.

On Nov. 8, in the crypto exchange’s final few days, he tried to pull out his cash and crypto.
</item 8>

<item 9>
How is it still getting worse for Sam Bankman-Fried? (The Verge, 2023-10-12T23:08:14+00:00)
The testimony set up an argument for the prosecution to bring in those tapes. The judge ruled in favor of the jury hearing the recordings and we briefly
recessed.

That was when Bankman-Fried’s mother approached the defense lawyers.

The late afternoon was short and snappy. Christian Drappi, a former Alameda software engineer who looked like a handsome funeral director in a black suit and
tie, testified briefly to set up the tape. When Changpeng Zhao, the CEO of Binance, announced on Twitter that he intended to acquire FTX, Ellison confessed the
theft of customer funds to him and a few other employees, Drappi said. The all-hands took place the following day, and was secretly recorded by an employee
who’d joined Alameda three days before.

In the recording, Ellison did indeed confess to stealing customer funds with Bankman-Fried’s approval. Drappi said he resigned less than 24 hours after the
meeting. The recording was later sent to Drappi, who sent it to the government.
</item 9>

<item 10>
The FTX trial is bigger than Sam Bankman-Fried (The Verge, 2023-09-28T12:00:00+00:00)
Bankman-Fried is accused of using “billions of dollars in misappropriated FTX customer deposits” to help buy more than $200 million of real estate for himself,
make billions of dollars of investments for his own interest, and repay Alameda’s lenders, according to the indictment, Additionally, Bankman-Fried allegedly
used more than $100 million of customer funds to make political contributions; prosecutors can show evidence of those contributions in this trial, even though
they aren’t part of the charges brought. A second trial is scheduled for March 2024, with additional charges.

Before his fall, Bankman-Fried made himself out to be the Good Boy of crypto — the trustworthy face of a sometimes-shady industry. He was also very interested
in publicity, sitting for many interviews both before and after the fall of FTX. The quick rise of FTX as an industry force was at least in part due to
Bankman-Fried’s appetite for attention.
</item 10>
=== END ITEM id=7dd88d448ba2d090 ===

=== ITEM id=759ed6842916571c ===
Gold sentences and facts:
Sentence 0: SBF, as he’s known, has pleaded not-guilty to some seven charges of fraud and conspiracy.
  Fact 0.0: SBF is known as SBF.
  Fact 0.1: SBF has pleaded not guilty to some seven charges.
  Fact 0.2: The charges are of fraud and conspiracy.
Sentence 1: Because FTX had grown so much, he could no longer run both companies, he said.
  Fact 1.0: FTX had grown so much.
  Fact 1.1: SBF said he could no longer run both companies because FTX had grown so much.
Sentence 2: The prosecution painted Bankman-Fried as someone who knowingly committed fraud to achieve great wealth, power and influence, while the defense countered that the FTX founder acted in good faith, never meant to commit fraud or steal and basically got in over his head.
  Fact 2.0: The prosecution painted Bankman-Fried as someone who knowingly committed fraud to achieve great wealth, power and influence.
  Fact 2.1: The defense countered that the FTX founder acted in good faith.
  Fact 2.2: The defense countered that the FTX founder never meant to commit fraud or steal.
  Fact 2.3: The defense countered that the FTX founder basically got in over his head.

Retrieved items:
<item 1>
The Steam Deck OLED arrives November 16 with an improved screen and longer battery life (Engadget, 2023-11-09T18:00:32+00:00)
The original Steam Deck had a 7-inch 800p LCD running at 60Hz. The new model squeezes a 7.4-inch OLED screen into the same space, significantly shrinking the
bezels around the display. The new panel runs at the same resolution, but with a refresh rate of up to 90Hz. It’s also HDR, covering 110 percent of the P3 color
space with a peak brightness of 1,000 nits — the old model maxed out at around 400 nits. Like all OLED screens, it has pure blacks and near-instantaneous
response times. As before, anti-glare etched glass is available, but only on the premium model.

The Steam Deck OLED has a die-shrunk version of the same custom APU found in the original Steam Deck. The new APU is produced on a 6nm process, rather than the
older 7nm process.
</item 1>

<item 2>
The Steam Deck OLED arrives November 16 with an improved screen and longer battery life (Engadget, 2023-11-09T18:00:32+00:00)
Manufacturers typically use die-shrinks to either improve performance or efficiency, and Valve has chosen the latter. The max frequency of both the GPU and CPU
are identical between the LCD and OLED models, so for developers tuning their games for Steam Decks, they only have a single performance target.

In concert with the more-efficient processor, there’s also a 25 percent larger battery with a 50Whr capacity, compared to the original’s 40Whr. According to
Valve, these changes combined result in a 30-50 percent increase in longevity between the two generations — the official battery life claim is "three to 12
hours." In our review, we measured a playtime increase of 62 percent when playing Hades on the OLED model versus our 18-month-old original. (Batteries degrade
over time, which explains why our gains were more pronounced.)

The new internals make the Steam Deck OLED marginally lighter, and also cooler to the touch when playing games, aided by a slightly larger fan.
</item 2>

<item 3>
Steam Deck OLED review: It's just better (Engadget, 2023-11-09T18:00:38+00:00)
Currently, more than 10,000 titles on Steam fall under the playable or verified labels, with more added every day.

When will the Steam Deck OLED be available?

The Steam Deck OLED will go on sale on November 16th at 1pm ET, and Valve has units ready to ship that same day. The company has lowered the price of its LCD
lineup and will sell through its backstock of 64GB and 512GB models until they’re all gone. It’ll continue selling the 256GB LCD model at a reduced price of
$400, while the 512GB OLED model will cost $550 and the 1TB OLED model will cost $650. There’s also a limited edition 1TB OLED version priced at $680 that comes
with a special carrying case, a translucent gray body and orange accents.

Valve

Wrap-up

According to the developers I spoke with at Valve, this is the definitive version of the Steam Deck.
</item 3>

<item 4>
The Steam Deck OLED arrives November 16 with an improved screen and longer battery life (Engadget, 2023-11-09T18:00:32+00:00)
The range still tops out at $649, but you’re getting a lot more for your money: All of the hardware improvements, 1TB of storage instead of 512GB and an
improved carry case that can pop out into a slimmer shell that’s smaller and easier to fit in a bag. As mentioned, it also has the same anti-glare etched glass
as the outgoing 512GB LCD model.

Finally, there’s also a Limited Edition version of the 1TB model, with a translucent shell and a customized carry case, which will be sold at $679. We imagine
stocks will not last very long, as it looks pretty rad:

Valve

While there are no concrete updates on future hardware developments, Valve engineer Pierre-Loup Griffais noted that, so long as interest in handheld gaming
remains high, the company has "plans for successive generations of handhelds." For now, Griffais explained, it's important to Valve that developers don't need
to "split their attention" between different performance targets.
</item 4>

<item 5>
The Steam Deck OLED arrives November 16 with an improved screen and longer battery life (Engadget, 2023-11-09T18:00:32+00:00)
Surprise! Valve has announced a new Steam Deck and — double surprise — we’ve already reviewed it. It might not look like a big change from the outside, but the
Steam Deck OLED is a thorough mid-cycle refresh of a machine that’s not even two years old. At the same time, Valve is also adjusting the configuration of its
entry-level model, making it a far more compelling option. Apologies to anyone who literally bought a Steam Deck yesterday.

At an event held for press prior to the device's launch, Greg Coomer, a product designer at Valve, called the new model "the definitive version of the Steam
Deck." He explained that the device contains many things the company wanted to include at the LCD model's launch but wasn't able to, along with some additions
based on community feedback.

As its name suggests, the most noticeable upgrade to the Steam Deck OLED is the display. The original Steam Deck had a 7-inch 800p LCD running at 60Hz.
</item 5>

<item 6>
Steam Deck OLED review: It's just better (Engadget, 2023-11-09T18:00:38+00:00)
The updated display is the highlight of the device, while things like faster charging, improved antennas and smoother haptics are welcome bonuses. More
importantly, this is Valve renewing its commitment to portable PC gaming, and that’s a relief to see. The Steam Deck is the granddaddy of handheld PCs and Valve
has the resources to continue innovating in this space. For now, the OLED model is a half-step toward the Steam Deck 2, which may be the final version of the
device — unless Valve finally learns how to count to three.
</item 6>

<item 7>
Steam Deck OLED review: It's just better (Engadget, 2023-11-09T18:00:38+00:00)
The HDR OLED model is what they would have released the first time around, if the technology had existed before 2022. The handheld market has evolved quickly —
in just over a year, the Steam Deck proved there’s widespread demand for handheld PCs, and its success helped drive chip makers and display manufacturers to
build hardware specifically for high-fidelity mobile play. This is great news for Valve and the Steam Deck, and also the other handhelds coming out of companies
like Ayaneo, ASUS, Ayn, Logitech, Retroid and Razer. Hell, maybe even Playdate will get an HDR OLED display one day. (Just kidding; Playdate is literally
perfect).

So, no, the HDR OLED refresh isn't the Steam Deck 2, but it is a welcome improvement on an already good device. The Steam Deck OLED is the facetuned,
photoshopped, spit-shined version of Valve’s handheld, and it features tangible, clever improvements.
</item 7>

<item 8>
Steam Deck OLED review: It's just better (Engadget, 2023-11-09T18:00:38+00:00)
The first thing to know about the Steam Deck OLED is that it’s not the Steam Deck 2. The Steam Deck OLED is a mid-cycle refresh from Valve, similar to the
Switch OLED upgrade, but there’s a lot more going on internally here. Its screen is better, its battery life is better, its thumbsticks are better, the haptics
are better, and its chip and thermals are better. One thing that hasn’t changed is its form factor — this is still one of the biggest babies on the handheld
block.

The new Steam Deck’s main upgrade is its OLED screen, which replaces the original LCD and, crucially, has native support for HDR and faster frame rates. The
OLED also has smaller bezels, making the display larger at 7.4 inches, compared with the original 7-inch screen.
</item 8>

<item 9>
The Steam Deck OLED arrives November 16 with an improved screen and longer battery life (Engadget, 2023-11-09T18:00:32+00:00)
Other improvements include support for both Wi-Fi 6E and Bluetooth 5.3, together with a new antenna array that will give a better experience when using both at
the same time. There’s also a reworked power supply (it has a logo now!) with a longer 2.5m cable that will charge the OLED model from 20 to 80 percent in “as
little as 45 minutes.” The thumbsticks have also been made 1mm taller, and the touch pads' haptics are slightly sharper.

Valve

The Steam Deck OLED starts at $549, and will replace all but the entry-level model in the range.
</item 9>

<item 10>
The Steam Deck OLED arrives November 16 with an improved screen and longer battery life (Engadget, 2023-11-09T18:00:32+00:00)
When asked about the potential for future hardware — like a return of the Steam Machine or an updated Steam Controller — there were no firm updates, but there
does seem to be something happening. Griffais agreed that enabling the Steam Deck to work as a hybrid living room console was a "side goal" of the team, and
noted that everything Valve does to improve SteamOS will benefit other devices. Coomer added that Valve is still working internally on "a bunch of controllers
and prototypes." If SteamOS is being used on larger displays, the team would want to augment that experience with better input options. "We don't have any
announcements on that front, but it's why we're building prototypes and thinking about the future there," Coomer said.

The Steam Deck OLED will be available to order on November 16th at 1PM ET. Valve expects units to begin shipping as soon as orders start to roll in. The updated
pricing on the existing LCD models is already in effect.
</item 10>
=== END ITEM id=759ed6842916571c ===
