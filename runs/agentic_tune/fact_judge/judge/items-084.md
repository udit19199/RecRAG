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

=== ITEM id=01ad21e37870a3ff ===
Gold sentences and facts:
Sentence 0: Under American Todd Boehly’s co-ownership, expanding Chelsea’s profile and reach in the U.S. would make sense, especially with USWNT internationals Catarina Macario and Mia Fishel playing their club football there — and CBS Sports holding WSL rights.
  Fact 0.0: Todd Boehly is American.
  Fact 0.1: Todd Boehly is a co-owner of Chelsea.
  Fact 0.2: Expanding Chelsea's profile and reach in the U.S. would make sense under Todd Boehly's co-ownership.
  Fact 0.3: Catarina Macario is a USWNT international.
  Fact 0.4: Mia Fishel is a USWNT international.
  Fact 0.5: Catarina Macario and Mia Fishel play their club football in the U.S.
  Fact 0.6: CBS Sports holds WSL rights.
Sentence 1: It was 24 September, Chelsea’s second home Premier League defeat on the spin after the one against Nottingham Forest and they were booed off again.
  Fact 1.0: The match was on 24 September.
  Fact 1.1: The match was Chelsea's second home Premier League defeat in a row.
  Fact 1.2: The defeat followed the one against Nottingham Forest.
  Fact 1.3: Chelsea were booed off again.
Sentence 2: Late checks will be made on the fitness of Reece James, Axel Disasi and Nicolas Jackson, while Benoit Badiashile could return from a long-term injury.
  Fact 2.0: Late checks will be made on the fitness of Reece James.
  Fact 2.1: Late checks will be made on the fitness of Axel Disasi.
  Fact 2.2: Late checks will be made on the fitness of Nicolas Jackson.
  Fact 2.3: Benoit Badiashile could return from a long-term injury.

Retrieved items:
<item 1>
Chelsea secure last-eight spot after Raheem Sterling sinks Blackburn (The Guardian, 2023-11-01T22:03:18+00:00)
It is no wonder Mauricio Pochettino spends most of his time watching Chelsea on the edge of the touchline with his hands clasped tightly in his pockets and a
tense expression on his face.

Even with his side 2-0 up and cruising against Blackburn thanks to goals from Benoît Badiashile and a rejuvenated Raheem Sterling, the Chelsea manager was a
bundle of nerves throughout what turned out to be a comfortable victory in the end.

But while the lack of cutting edge that has become a hallmark of the Todd Boehly-era in west London still looms large as he prepares for his first return to
table-topping Tottenham on Monday night, Pochettino could take heart from Cole Palmer’s brilliant performance as the former Manchester City forward helped to
dispatch Jon Dahl Tomasson’s inexperienced side.

Graph facts:
- Chelsea secure last-eight spot after Raheem Sterling sinks Blackburn -MENTIONS-> Anfield
- Chelsea secure last-eight spot after Raheem Sterling sinks Blackburn -MENTIONS-> Middlesbrough
- Chelsea secure last-eight spot after Raheem Sterling sinks Blackburn -MENTIONS-> Fulham
- Chelsea secure last-eight spot after Raheem Sterling sinks Blackburn -MENTIONS-> Port Vale
- Chelsea secure last-eight spot after Raheem Sterling sinks Blackburn -MENTIONS-> Premier League

Entities: Chelsea secure last-eight spot after Raheem Sterling sinks Blackburn, Blackburn, Benoît Badiashile, Raheem Sterling, Chelsea, Mauricio Pochettino,
Manchester City, The Guardian, Todd Boehly, Jon Dahl Tomasson, Cole Palmer, Tottenham
</item 1>

<item 2>
Chelsea secure last-eight spot after Raheem Sterling sinks Blackburn (The Guardian, 2023-11-01T22:03:18+00:00)
For more information see our Newsletters may contain info about charities, online ads, and content funded by outside parties. For more information see our
Privacy Policy . We use Google reCaptcha to protect our website and the Google Privacy Policy and Terms of Service apply. after newsletter promotion

But the Swedish goalkeeper was at fault when Chelsea finally found the goal they were looking for, palming Gallagher’s cross after he had combined with Palmer
straight to the feet of Badiashile and he calmly sidefooted home.

Blackburn’s supporters then felt aggrieved when a Callum Brittain cross struck Gallagher’s trailing arm but once again Robinson waved away strong appeals for a
penalty.

The visitors showed they had no intention of surrendering without a fight when Harry Leonard caught Axel Disasi napping and raced onto a long ball at the start
of the second half, only to curl his shot wide of the post. Not for the first time, Robert Sánchez was then far too casual as the Chelsea goalkeeper’s pass was
almost intercepted.

Graph facts:
- Chelsea secure last-eight spot after Raheem Sterling sinks Blackburn -MENTIONS-> Anfield
- Chelsea secure last-eight spot after Raheem Sterling sinks Blackburn -MENTIONS-> Middlesbrough
- Chelsea secure last-eight spot after Raheem Sterling sinks Blackburn -MENTIONS-> Fulham
- Chelsea secure last-eight spot after Raheem Sterling sinks Blackburn -MENTIONS-> Port Vale
- Chelsea secure last-eight spot after Raheem Sterling sinks Blackburn -MENTIONS-> Premier League

Entities: Chelsea secure last-eight spot after Raheem Sterling sinks Blackburn, Blackburn, Raheem Sterling, Chelsea vs Blackburn match, Gallagher, Palmer,
Badiashile, Swedish goalkeeper, Callum Brittain, Robinson, Harry Leonard, Axel Disasi, Robert Sánchez, Google Privacy Policy, Terms of Service
</item 2>

<item 3>
How U.S. Women’s Soccer Team Lured Emma Hayes Away From Chelsea (The New York Times, 2023-11-15T21:40:47+00:00)
Soccer’s planned combined headquarters and national training facility in Georgia. As of yet, there’s no targeted date set for the move.

Hayes, too, will benefit in ways other than compensation and prestige. She spoke to reporters about looking forward to spending more time with her five-year-old
son, Harry. She has never been to one of his sports days, picked him up from school or taken him to an after-school club and she wants to do that.

The main visible wrinkle in the process was Chelsea’s surprise move of announcing Hayes’ departure on Saturday, Nov. 4. With the contract not yet finalized and
U.S. Soccer board approval still needed, Chelsea issued their statement at 11 a.m. ET in the U.S. in which it noted she would leave at the end of the season “to
pursue a new opportunity outside of the WSL and club football.” The club feared that the news was starting to leak and wanted to share the news on its own.

Graph facts:
- He’s Hockey’s Brightest Young Star. This Is What Makes His Shot So Special. -PUBLISHED_BY-> The New York Times
- Which M.L.B. Playoff Team Has the Most Daunting Home Field? We Ranked Them. -PUBLISHED_BY-> The New York Times
- After a Stunning Snub, College Football Is Facing an Identity Crisis -PUBLISHED_BY-> The New York Times
- N.F.L. Executives on the Biggest Questions Facing Super Bowl Contenders -PUBLISHED_BY-> The New York Times
- Alcoholism, Ayahuasca and the Enlightenment of an N.F.L. Player -PUBLISHED_BY-> The New York Times

Entities: The New York Times, club football, U.S. Women’s Soccer Team, U.S. Soccer, Georgia, WSL, Chelsea announcement of Hayes’ departure, Emma Hayes, How U.S.
Women’s Soccer Team Lured Emma Hayes Away From Chelsea, Chelsea, the U.S., Harry, soccer
</item 3>

<item 4>
How U.S. Women’s Soccer Team Lured Emma Hayes Away From Chelsea (The New York Times, 2023-11-15T21:40:47+00:00)
England and Chelsea captain Millie Bright was “devastated,” and most players were understandably sad — many of whom owe their career progression to Hayes — but
know they still have a job to do this season.

“It’s been a bit of a whirlwind. As a player and a person, I was devastated. I’ve been here nine years under Emma and the things I’ve learned,” Bright said.
“She’s a mentor, a coach, a friend, a life coach. It’s more than just football playing under her.”

Bright credits Hayes for her on and off-field guidance. (Photo by Marc Atkins, Getty Images)

Hayes’ American assistant Denise Reddy, born in New Jersey, is likely to follow her across the pond. The former United States Under-20 coach has remained
faithful to her friend of 20 years and voluntarily quit her job as assistant at Chicago Red Stars in 2010 when Hayes was fired as head coach. Chelsea’s general
manager Paul Green will stay at the club.

Graph facts:
- He’s Hockey’s Brightest Young Star. This Is What Makes His Shot So Special. -PUBLISHED_BY-> The New York Times
- Which M.L.B. Playoff Team Has the Most Daunting Home Field? We Ranked Them. -PUBLISHED_BY-> The New York Times
- After a Stunning Snub, College Football Is Facing an Identity Crisis -PUBLISHED_BY-> The New York Times
- N.F.L. Executives on the Biggest Questions Facing Super Bowl Contenders -PUBLISHED_BY-> The New York Times
- Alcoholism, Ayahuasca and the Enlightenment of an N.F.L. Player -PUBLISHED_BY-> The New York Times

Entities: The New York Times, Getty Images, U.S. Women’s Soccer Team, Emma Hayes, Millie Bright, Denise Reddy, United States Under-20, Chicago Red Stars, Paul
Green, Marc Atkins, How U.S. Women’s Soccer Team Lured Emma Hayes Away From Chelsea, Chelsea, England, New Jersey, football
</item 4>

<item 5>
Chelsea and England star Cole Palmer: ‘The most annoying thing about moving down south? The traffic’ (The Independent - Sports, 2023-11-15T22:32:05+00:00)
“I was always breaking her pipes,” he recalled. He has come a long way since then; particularly in 2023. “Yeah, it’s been crazy,” he said. “The changes have
happened so fast. From winning the Euros to signing for Chelsea then getting signed up here.”

Palmer chats with Arsenal winger Bukayo Saka (The FA/Getty)

His part in England Under-21s’ European Championship win may have caught Gareth Southgate’s eye. His immediate decision was whether he had done enough to
impress Pep Guardiola. There had been interest in taking him on loan, from Brighton and Burnley and RB Leipzig. City decided that he could only leave on a
permanent deal and Chelsea forked out £42.5m. “Who knows what would have happened if I had stayed?” Palmer mused. “Maybe I would have played more, maybe not.
But I think the decision to go to Chelsea so far is paying off.”

That is an understatement.

Graph facts:
- Carabao Cup fixtures, draws, results, TV channel, and live streams for every round of EFL League Cup 2023/24 -MENTIONS-> Burnley
- Ten Hag demands both positivity and pragmatism from erratic Onana -MENTIONS-> Burnley
- Manchester divided as ten Hag pressure mounts after United Hammered, Ange-ball pays off again as Spurs sink Everton -MENTIONS-> Burnley
- Pochettino looks past James mistake and stakes future on his ‘Chelsea DNA’ -MENTIONS-> Burnley
- Liverpool enjoy home comforts but Dyche’s Everton offer derby danger | Andy Hunter -MENTIONS-> Burnley

Entities: Burnley, Chelsea, Pep Guardiola, England, RB Leipzig, Bukayo Saka, Gareth Southgate, City, Brighton, Arsenal, Cole Palmer, Chelsea and England star
Cole Palmer: ‘The most annoying thing about moving down south? The traffic’, England Under-21s, European Championship win, The FA
</item 5>

<item 6>
How U.S. Women’s Soccer Team Lured Emma Hayes Away From Chelsea (The New York Times, 2023-11-15T21:40:47+00:00)
There is the possibility of Hayes retaining a connection to the club via some sort of ambassadorial role, but it’s likely contingent upon a lack of conflict
with the USWNT role and responsibilities. Under American Todd Boehly’s co-ownership, expanding Chelsea’s profile and reach in the U.S. would make sense,
especially with USWNT internationals Catarina Macario and Mia Fishel playing their club football there — and CBS Sports holding WSL rights.

There is, for now, an immediate task for Hayes to focus on. Chelsea faces off against Real Madrid on Wednesday for their first match of the UWCL Champions
League group stage. Her full American arrival will not come for another half a year after that.

(Top photo: Julian Finney/Getty Images)

Graph facts:
- He’s Hockey’s Brightest Young Star. This Is What Makes His Shot So Special. -PUBLISHED_BY-> The New York Times
- Which M.L.B. Playoff Team Has the Most Daunting Home Field? We Ranked Them. -PUBLISHED_BY-> The New York Times
- After a Stunning Snub, College Football Is Facing an Identity Crisis -PUBLISHED_BY-> The New York Times
- N.F.L. Executives on the Biggest Questions Facing Super Bowl Contenders -PUBLISHED_BY-> The New York Times
- Alcoholism, Ayahuasca and the Enlightenment of an N.F.L. Player -PUBLISHED_BY-> The New York Times

Entities: The New York Times, CBS Sports, Getty Images, U.S. Women’s Soccer Team, WSL, Emma Hayes, How U.S. Women’s Soccer Team Lured Emma Hayes Away From
Chelsea, Catarina Macario, Mia Fishel, Chelsea vs Real Madrid UWCL Champions League group stage match, UWCL Champions League group stage, Julian Finney,
Chelsea, Real Madrid, Todd Boehly
</item 6>

<item 7>
Chelsea and England star Cole Palmer: ‘The most annoying thing about moving down south? The traffic’ (The Independent - Sports, 2023-11-15T22:32:05+00:00)
That is an understatement. And if there seemed a risk involved in swapping the stability of City for the seeming madhouse of Stamford Bridge, there was evidence
of more cool-headed thinking from a 21-year-old. Chelsea have been manic buyers but he said: “I did look at how many players were there but I don’t think people
realise how many players they actually got rid off, as well.”

The departures of, among others, Mason Mount, Kai Havertz, Christian Pulisic and Joao Felix, meant there could be opportunities in the attacking midfield
positions. “I looked at the squad and thought: ‘If I go there and know what I can do then I’ll have the chance of playing’,” he said. “And thankfully it’s
happened.” If there seemed plenty of traffic in the queue for places in Southgate’s attack, Palmer seems to have accelerated past several rivals. Now an England
debut could follow in the next week.

Graph facts:
- Everton players and fans will be fired up by points penalty, asserts Sean Dyche -MENTIONS-> Chelsea
- Carabao Cup fixtures, draws, results, TV channel, and live streams for every round of EFL League Cup 2023/24 -MENTIONS-> Chelsea
- Chelsea secure last-eight spot after Raheem Sterling sinks Blackburn -MENTIONS-> Chelsea
- How U.S. Women’s Soccer Team Lured Emma Hayes Away From Chelsea -MENTIONS-> Chelsea
- Leaky defence ‘isn’t good enough’ and ‘scars’ grow from throwing away leads – It’s all going wrong for Manchester United in Europe -MENTIONS-> Chelsea

Entities: Chelsea, Kai Havertz, Mason Mount, England, Stamford Bridge, Christian Pulisic, City, Cole Palmer, Chelsea and England star Cole Palmer: ‘The most
annoying thing about moving down south? The traffic’, Joao Felix, Southgate, The Independent
</item 7>

<item 8>
How U.S. Women’s Soccer Team Lured Emma Hayes Away From Chelsea (The New York Times, 2023-11-15T21:40:47+00:00)
Chelsea’s general manager Paul Green will stay at the club. It is unclear whether any other members of Chelsea’s technical staff are expected to depart.

GO DEEPER What next for Chelsea after Emma Hayes?

The relentless nature of managing a club takes its toll and after what will be 12 years at Chelsea come the end of the season, Hayes, a single parent whose
father died last month, decided that it was the right time for a change, professionally and personally.

Advertisement

“The biggest factors are my son, leaving at the top and giving the club enough time to be able to transition without there being too much disruption,” she said
in her press conference.

The club has received several applications regarding Hayes’ replacement but has not yet started an official recruitment process. She will meet with Chelsea’s
technical directors once a week to create a succession plan and will have a say in who takes the job after her.

Graph facts:
- He’s Hockey’s Brightest Young Star. This Is What Makes His Shot So Special. -PUBLISHED_BY-> The New York Times
- Which M.L.B. Playoff Team Has the Most Daunting Home Field? We Ranked Them. -PUBLISHED_BY-> The New York Times
- After a Stunning Snub, College Football Is Facing an Identity Crisis -PUBLISHED_BY-> The New York Times
- N.F.L. Executives on the Biggest Questions Facing Super Bowl Contenders -PUBLISHED_BY-> The New York Times
- Alcoholism, Ayahuasca and the Enlightenment of an N.F.L. Player -PUBLISHED_BY-> The New York Times

Entities: The New York Times, U.S. Women’s Soccer Team, Emma Hayes, Paul Green, How U.S. Women’s Soccer Team Lured Emma Hayes Away From Chelsea, Chelsea
</item 8>

<item 9>
Graph query row:
article: How U.S. Women’s Soccer Team Lured Emma Hayes Away From Chelsea
publisher: The New York Times
mentioned_entities: ['The Athletic', 'World Cup', 'Chicago', 'CBS Sports', 'Getty Images', 'English Football League', 'Chelsea Women', 'U.S. women’s national
team', 'Women’s Super League', 'club football', 'U.S. Women’s Soccer Team', 'Matt Crocker', 'Vlatko Andonovski', 'Cindy Parlow Cone', 'women’s soccer',
'Crocker', 'Cone', 'J.T. Batson', 'U.S. Soccer', 'September meeting', 'Gregg Berhalter', 'Tony Gustavsson', 'English FA', "U.S. Women's National Soccer Team",
'international window', 'Georgia', 'WSL', 'Chelsea announcement of Hayes’ departure', 'private conversations', 'Andonovski', 'Emma Hayes', 'U.S. Soccer
Federation', "U.S. Women's Soccer Team", 'Twila Kilgore', '2024 Olympics', '2024 CONCACAF Gold Cup', 'four friendlies', 'international windows', 'Twila',
'Olympic tournament', 'the federation', 'upcoming Olympic tournament', 'this past summer’s World Cup', "Spain's national team", "Spain's federation", "Chelsea's
away game at Aston Villa", 'Millie Bright', 'Denise Reddy', 'United States Under-20', 'Chicago Red Stars', 'Paul Green', 'Marc Atkins', 'Catarina Macario', 'Mia
Fishel', 'Chelsea vs Real Madrid UWCL Champions League group stage match', 'UWCL Champions League group stage', 'Julian Finney', 'Chelsea', 'Paris', 'Aston
Villa', 'Atlanta', 'Sarina Wiegman', 'England', 'Real Madrid', 'Berhalter', 'the U.S.', 'Australia', 'USMNT', 'Harry', 'Champions League final', 'Todd Boehly',
'2027 World Cup', 'soccer', 'New Jersey', 'London', 'Olympics', 'U.S.', 'football', 'USA']
connections: ['The Athletic -MENTIONS- How Did the Houston Astros Cheat in 2017? Everything to Know About a Dark Moment in MLB History', 'The Athletic
-MENTIONS- Which M.L.B. Playoff Team Has the Most Daunting Home Field? We Ranked Them.', 'The Athletic -MENTIONS- Michigan sign-stealing scandal winners and
losers, from Jim Harbaugh to Tony Petitti', "The Athletic -MENTIONS- Browns survive Bears Hail Mary on wild, near-catch by Darnell Mooney: 'Damn near s—
myself'", 'The Athletic -MENTIONS- He’s Hockey’s Brightest Young Star. This Is What Makes His Shot So Special.', 'The Athletic -MENTIONS- How U.S. Women’s
Soccer Team Lured Emma Hayes Away From Chelsea', 'The Athletic -MENTIONS- Top 25 MLB offseason trade candidates', 'The Athletic -MENTIONS- MLB Winter Meetings
tracker: Live updates on news, rumors, signings and trades for 2023 free agency', 'The Athletic -MENTIONS- Michigan sign-stealing scandal news: Latest updates
on possible Jim Harbaugh suspension from NCAA cheating investigation', "The Athletic -MENTIONS- Fantasy Injury Updates Week 10: News on Justin Jefferson,
Ja'Marr Chase, T.J. Hockenson & Curtis Samuel", "The Athletic -MENTIONS- Bills vs. Bengals live score, updates, highlights from NFL 'Sunday Night Football'
game", 'The Athletic -MENTIONS- Alcoholism, Ayahuasca and the Enlightenment of an N.F.L. Player', 'The Athletic -MENTIONS- Wild HC Dean Evason, hire John Hynes
as replacement', 'The Athletic -MENTIONS- Tennis News: Kyrgios escalates beef with Becker, change ‘warrior’ Novak had to make, WTA criticised for charisma
fail', 'The Athletic -MENTIONS- MLB free agency tracker: Live updates on news, rumors, signings and trades after 2023 Winter Meetings', 'The Athletic -MENTIONS-
Shohei Ohtani rumors tracker: Latest news, updates on Dodgers, Cubs & more as free agency short list emerges', "The Athletic -MENTIONS- Shohei Ohtani rumors
tracker: Latest news, updates on free agent's MLB future", 'The Athletic -MENTIONS- Michigan’s Rivals Fume About Sign-Stealing: ‘This Was Damn Near Espionage’',
'The Athletic -WORKS_FOR- Evan Drellich', 'The Athletic -WORKS_FOR- Ken Rosenthal', 'The Athletic -WORKS_FOR- Joe Buscaglia', 'The Athletic -WORKS_FOR- Andrew
Baggarly', 'World Cup -MENTIONS- THE GREATEST! Boks beat All Blacks in instant classic to claim record FOURTH RWC in controversial, dramatic final', 'World Cup
-MENTIONS- Chicago Fire vs Inter Miami live score, result, updates, highlights from Lionel Messi in MLS match', 'World Cup -MENTIONS- West Indies v England:
fifth and deciding men’s T20 cricket international – live', 'World Cup -MENTIONS- How U.S. Women’s Soccer Team Lured Emma Hayes Away From Chelsea', 'World Cup
-MENTIONS- UK View: ‘A bridge too far’ – Boycott, former greats tear into ‘dysfunctional’ Poms with World Cup defence all but dead', "World Cup -MENTIONS- The
story behind anguished Wallabies star's photo", 'World Cup -MENTIONS- Legacy vs ego: The Springboks are playing for South Africa, the All Blacks are playing for
themselves', 'World Cup -MENTIONS- England choose survival over oblivion and a chance of summer trip to France', 'World Cup -MENTIONS- Erik ten Hag could pay
the price for Manchester United’s incompetent Champions League campaign', 'World Cup -MENTIONS- Gio Reyna looks good in start for USMNT vs. Germany, but those
involved in the finish are punished in 3-1 defeat', 'World Cup -MENTIONS- Man United next manager: Assessing candidates to replace Erik ten Hag if Red Devils
sack Dutch boss', 'World Cup -MENTIONS- The Soccer Star on the Verge of Breaking an 89-Year-Old Record', 'World Cup -MENTIONS- Eddie Jones says he needs to
‘give myself an uppercut’ but ‘100 percent’ committed to coaching Wallabies', 'World Cup -MENTIONS- ‘Right people in right roles’: QRU breaks silence on RA’s
reform hopes, backs review into Wallabies’ RWC flop', 'World Cup -MENTIONS- RA set to announce local as high-performance director – and how it could help land
Schmidt as Wallabies coach', 'World Cup -MENTIONS- Argentina vs Paraguay live score, result, updates, highlights from CONMEBOL World Cup qualifying', "World Cup
-MENTIONS- Inter Miami friendly matches for 2024 preseason: Lionel Messi to play Newell's Old Boys, China tour cancelled", 'World Cup -MENTIONS- Thirty-seven
phases of agony define Ireland’s greatest heartbreak as curse continues', 'World Cup -MENTIONS- Who Are Mikaela Shiffrin’s Parents? All You Need to Know About
Skiing Legend’s Family', 'World Cup -MENTIONS- USA vs Germany player ratings: USMNT defensive errors blight positive performance by Pulisic, Weah', 'World Cup
-PARTICIPATED_IN- Springboks', 'World Cup -PARTICIPATED_IN- U.S. Women’s Soccer Team', 'World Cup -PARTICIPATED_IN- England team', 'World Cup -PARTICIPATED_IN-
England', 'World Cup -PARTICIPATED_IN- Wallabies', 'World Cup -PARTICIPATED_IN- Wales', 'World Cup -PARTICIPATED_IN- Dane Coles', 'World Cup -PARTICIPATED_IN-
Varane', 'World Cup -PARTICIPATED_IN- USMNT', 'World Cup -PARTICIPATED_IN- Eddie Jones', 'World Cup -PARTICIPATED_IN- All Blacks', 'World Cup -PARTICIPATED_IN-
Mikaela Shiffrin', 'World Cup -LOCATED_IN- France', 'World Cup -LOCATED_IN- South Africa', 'Chicago -MENTIONS- Chicago Fire vs Inter Miami live score, result,
updates, highlights from Lionel Messi in MLS match', 'Chicago -MENTIONS- NFL Fantasy Football Week 5 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for
every game', "Chicago -MENTIONS- Browns survive Bears Hail Mary on wild, near-catch by Darnell Mooney: 'Damn near s— myself'", 'Chicago -MENTIONS- Sam
Bankman-Fried didn’t ask where the $8 billion went', 'Chicago -MENTIONS- He’s Hockey’s Brightest Young Star. This Is What Makes His Shot So Special.', 'Chicago
-MENTIONS- How U.S. Women’s Soccer Team Lured Emma Hayes Away From Chelsea', 'Chicago -MENTIONS- The people who ruined the internet', 'Chicago -MENTIONS- NFL
power rankings: Lions, 49ers level up; Texans jump Bengals, Bills; Jets, Patriots sink to bottom for Week 11', "Chicago -MENTIONS- Bears vs. Vikings live score,
updates, highlights from NFL 'Monday Night Football' game", 'Chicago -MENTIONS- NFL Fantasy Football Week 6 Lineup Decisions: Starts, Sits, Sleepers, Busts to
know for every game', 'Chicago -MENTIONS- The best new TV series on every streaming service', "Chicago -MENTIONS- Taylor Swift's 1989: The stories behind her
biggest album", 'Chicago -MENTIONS- NFL Best Bets Week 9: Colts clobber Panthers, Dolphins drop Chiefs in Germany', 'Chicago -MENTIONS- Patrick Kane
destinations: Sabres, Red Wings among potential landing spots for free agent winger', "Chicago -MENTIONS- Shohei Ohtani rumors tracker: Latest news, updates on
free agent's MLB future", "Chicago -MENTIONS- Blackhawks vs. Penguins live score, updates, highlights from Connor Bedard's NHL debut", 'Chicago -MENTIONS-
NASCAR offseason news roundup: Short track testing in Phoenix, new spotters for stars, more moves for 2024', "Chicago -MENTIONS- Bears vs. Panthers live score,
updates, highlights from NFL 'Thursday Night Football' game", 'Chicago -MENTIONS- Bears', 'Chicago -MENTIONS- Fantasy Football RB Rankings Week 12: Who to
start, best sleepers at running back', 'Chicago -MENTIONS- Fantasy Football QB Rankings Week 14: Who to start, best sleepers at quarterback', 'Chicago
-MENTIONS- Fly with Delta frequently? Small business owners can score a solid welcome bonus and other perks with the Delta SkyMiles Gold Business Card',
'Chicago -LOCATED_IN- Blackhawks', 'Chicago -LOCATED_IN- Soldier Field', 'Chicago -LOCATED_IN- Brendan Hunt', 'Chicago -LOCATED_IN- Chicago Fire', 'Chicago
-LOCATED_IN- Psychic Near Me', "Chicago -LOCATED_IN- Chicago's largest Black law firm", 'Chicago -LOCATED_IN- Hard Femme', 'Chicago -LOCATED_IN- Blackhawks vs.
Penguins game', "Chicago -LOCATED_IN- Bears vs. Panthers 'Thursday Night Football' game", 'Chicago -LOCATED_IN- Bears', 'Chicago -LOCATED_IN- McKillip Animal
Hospital', 'CBS Sports -MENTIONS- Taylor Swift-Travis Kelce timeline: Everything to know about rumored romance between pop star, Chiefs TE', 'CBS Sports
-MENTIONS- How U.S. Women’s Soccer Team Lured Emma Hayes Away From Chelsea', 'CBS Sports -WORKS_FOR- Ian Eagle', "Getty Images -MENTIONS- Amazon sellers sound
off on the FTC's 'long-overdue' antitrust case", 'Getty Images -MENTIONS- Which M.L.B. Playoff Team Has the Most Daunting Home Field? We Ranked Them.', 'Getty
Images -MENTIONS- Video games in 2023: Acquisitions, layoffs, unions', 'Getty Images -MENTIONS- From Justin Timberlake to Sam Asghari: A timeline of Britney
Spears’ relationships', 'Getty Images -MENTIONS- West Indies v England: fifth and deciding men’s T20 cricket international – live', 'Getty Images -MENTIONS-
Should you introduce your partner to your family during the holidays? Relationship experts weigh in', 'Getty Images -MENTIONS- How the conspiracy-fueled Epoch
Times went mainstream and made millions', 'Getty Images -MENTIONS- From Ten Hag’s future to a leaking roof: Ratcliffe’s Manchester United in-tray', 'Getty
Images -MENTIONS- Everton players and fans will be fired up by points penalty, asserts Sean Dyche', "Getty Images -MENTIONS- How OpenAI's ChatGPT has changed
the world in just a year", 'Getty Images -MENTIONS- He’s Hockey’s Brightest Young Star. This Is What Makes His Shot So Special.', 'Getty Images -MENTIONS- How
U.S. Women’s Soccer Team Lured Emma Hayes Away From Chelsea', 'Getty Images -MENTIONS- After a Stunning Snub, College Football Is Facing an Identity Crisis',
'Getty Images -MENTIONS- Comeback kings Richmond take the AFLW points in Cairns', 'Getty Images -MENTIONS- Ground-breaking Gold Coast: Suns make AFLW finals for
a second time', 'Getty Images -MENTIONS- Microsoft Activision: What does deal mean for gamers?', 'Getty Images -MENTIONS- McRae reveals hidden message he
carried under collar of his grand final shirt', "Getty Images -MENTIONS- Sex, Signal messages, and sabotaging FTX's code: SBF execs and Bahamas roommates tell
all in court", 'Getty Images -MENTIONS- Top ten craziest fight endings in boxing history – Floyd Mayweather’s ruthless KO, Mike Tyson’s bite and a baffled
Lennox Lewis', 'Getty Images -MENTIONS- Ten Hag’s gamble pays off against Chelsea as McTominay steals show | Jonathan Liew', "Getty Images -MENTIONS- Taylor
Swift's 1989: The stories behind her biggest album", 'Getty Images -MENTIONS- Taylor Swift reveals secret start to her relationship with Travis Kelce', 'Getty
Images -MENTIONS- Sleepy tea, counting backwards and dense earplugs: Inside the lives of insomniacs', "Getty Images -MENTIONS- Sweeping White House AI executive
order takes aim at the technology's toughest challenges", 'Getty Images -MENTIONS- N.F.L. Executives on the Biggest Questions Facing Super Bowl Contenders',
'Getty Images -MENTIONS- Alcoholism, Ayahuasca and the Enlightenment of an N.F.L. Player', 'Getty Images -MENTIONS- From a Turkey trot to a movie marathon,
Friendsgiving celebrations that aren’t a sit-down dinner', 'Getty Images -MENTIONS- Ten Hag demands both positivity and pragmatism from erratic Onana', 'Getty
Images -MENTIONS- Pokémon is no longer just a game — it’s a lifestyle', 'Getty Images -MENTIONS- European roundup: Bellingham helps Real Madrid go top while
Bayern draw', 'Getty Images -MENTIONS- Britney Spears and Kevin Federline: A complete timeline of their relationship', 'Getty Images -MENTIONS- Jada Pinkett
Smith reveals why she and Will Smith never signed a prenup', 'Getty Images -MENTIONS- West Indies clinch T20 series after Shai Hope edges hosts past England
total', 'Getty Images -MENTIONS- Coinbase rallies more than 60% in same month that FTX and Binance founders brace for prison', 'Getty Images -MENTIONS- The
Soccer Star on the Verge of Breaking an 89-Year-Old Record', "Getty Images -MENTIONS- With hopes of victory fading, Ukraine's fight against Russia could get
even harder in 2024", 'Getty Images -MENTIONS- ‘Biggest win of my career’: De Minaur, Popyrin power Australia into Davis Cup final', 'Getty Images -MENTIONS-
The Most Interesting Articles', 'Getty Images -MENTIONS- We know far more about the deep ocean than the moon or Mars, says explorer Jon Copley', "Getty Images
-MENTIONS- 'They seemed primed to take over': How the Great Dying doomed the 'beast tooth' and set the stage for the dawn of the dinosaurs", 'Getty Images
-MENTIONS- “We Sell More F-150s to NASCAR Fans” – Ford Insider Reveals Insane Mustang Stat Amid Big Change', 'Getty Images -MENTIONS- Michigan’s Rivals Fume
About Sign-Stealing: ‘This Was Damn Near Espionage’', 'Getty Images -MENTIONS- M.L.B. Superstar Shohei Ohtani to Sign $700 Million Deal With Dodgers', "Getty
Images -MENTIONS- The inside story of Dave Clark's tumultuous last days at Flexport", 'Getty Images -MENTIONS- Thirty-seven phases of agony define Ireland’s
greatest heartbreak as curse continues', 'Getty Images -MENTIONS- Liverpool enjoy home comforts but Dyche’s Everton offer derby danger | Andy Hunter', "Getty
Images -MENTIONS- Dangerous 'superbugs' are a growing threat, and antibiotics can't stop their rise. What can?", 'Getty Images -MENTIONS- 7 of the best ski
holidays in Canada', 'Getty Images -MENTIONS- 7 of the best luxury ski resorts for a five-star experience', 'Getty Images -MENTIONS- He killed Jon Snow. Now
Owen Teale is coming to ruin your Christmas', 'Getty Images -PARTICIPATED_IN- Getty Images lawsuit against Stability AI', 'Getty Images -PART_OF- Marc Atkins',
'English Football League -MENTIONS- Carabao Cup fixtures, draws, results, TV channel, and live streams for every round of EFL League Cup 2023/24', 'English
Football League -MENTIONS- Chelsea secure last-eight spot after Raheem Sterling sinks Blackburn', 'English Football League -MENTIONS- How U.S. Women’s Soccer
Team Lured Emma Hayes Away From Chelsea', 'English Football League -PART_OF- Middlesbrough', 'English Football League -PART_OF- Port Vale', 'English Football
League -PART_OF- League One', 'English Football League -PART_OF- Championship', 'English Football League -PART_OF- League Two', 'Chelsea Women -MENTIONS- How
U.S. Women’s Soccer Team Lured Emma Hayes Away From Chelsea', 'Chelsea Women -WORKS_FOR- Emma Hayes', 'U.S. women’s national team -MENTIONS- How U.S. Women’s
Soccer Team Lured Emma Hayes Away From Chelsea', 'U.S. women’s national team -WORKS_FOR- Emma Hayes', 'Women’s Super League -MENTIONS- How U.S. Women’s Soccer
Team Lured Emma Hayes Away From Chelsea', 'club football -MENTIONS- How U.S. Women’s Soccer Team Lured Emma Hayes Away From Chelsea', 'U.S. Women’s Soccer Team
-MENTIONS- How U.S. Women’s Soccer Team Lured Emma Hayes Away From Chelsea', 'U.S. Women’s Soccer Team -SAID- Emma Hayes', 'U.S. Women’s Soccer Team -WORKS_FOR-
Vlatko Andonovski', 'U.S. Women’s Soccer Team -WORKS_FOR- Emma Hayes', 'U.S. Women’s Soccer Team -WORKS_FOR- Twila', 'U.S. Women’s Soccer Team -PARTICIPATED_IN-
World Cup', 'U.S. Women’s Soccer Team -PARTICIPATED_IN- this past summer’s World Cup', 'Matt Crocker -MENTIONS- How U.S. Women’s Soccer Team Lured Emma Hayes
Away From Chelsea', 'Vlatko Andonovski -MENTIONS- How U.S. Women’s Soccer Team Lured Emma Hayes Away From Chelsea', 'Vlatko Andonovski -WORKS_FOR- U.S. Women’s
Soccer Team', 'Cindy Parlow Cone -MENTIONS- How U.S. Women’s Soccer Team Lured Emma Hayes Away From Chelsea', 'Cindy Parlow Cone -SAID- Emma Hayes', 'Cindy
Parlow Cone -WORKS_FOR- U.S. Soccer', 'women’s soccer -MENTIONS- How U.S. Women’s Soccer Team Lured Emma Hayes Away From Chelsea', 'Crocker -MENTIONS- How U.S.
Women’s Soccer Team Lured Emma Hayes Away From Chelsea', 'Crocker -SAID- new head coach in place by December', 'Crocker -SAID- Twila Kilgore', 'Crocker
-WORKS_FOR- U.S. Soccer', 'Crocker -PARTICIPATED_IN- September meeting', 'Cone -MENTIONS- How U.S. Women’s Soccer Team Lured Emma Hayes Away From Chelsea',
'Cone -WORKS_FOR- U.S. Soccer', 'Cone -PARTICIPATED_IN- September meeting', 'J.T. Batson -MENTIONS- How U.S. Women’s Soccer Team Lured Emma Hayes Away From
Chelsea', 'J.T. Batson -WORKS_FOR- U.S. Soccer', 'J.T. Batson -PARTICIPATED_IN- September meeting', 'U.S. Soccer -MENTIONS- How U.S. Women’s Soccer Team Lured
Emma Hayes Away From Chelsea', 'U.S. Soccer -SAID- Emma Hayes', 'U.S. Soccer -WORKS_FOR- Cindy Parlow Cone', 'U.S. Soccer -WORKS_FOR- Crocker', 'U.S. Soccer
-WORKS_FOR- Cone', 'U.S. Soccer -WORKS_FOR- J.T. Batson', 'U.S. Soccer -WORKS_FOR- Emma Hayes', 'U.S. Soccer -WORKS_FOR- Twila Kilgore', 'U.S. Soccer
-LOCATED_IN- Georgia', 'September meeting -MENTIONS- How U.S. Women’s Soccer Team Lured Emma Hayes Away From Chelsea', 'September meeting -PARTICIPATED_IN-
Crocker', 'September meeting -PARTICIPATED_IN- Cone', 'September meeting -PARTICIPATED_IN- J.T. Batson', 'Gregg Berhalter -MENTIONS- How U.S. Women’s Soccer
Team Lured Emma Hayes Away From Chelsea', 'Gregg Berhalter -MENTIONS- USA squad for October friendlies 2023: USMNT roster for friendly matches vs Germany,
Ghana', 'Gregg Berhalter -MENTIONS- Gio Reyna looks good in start for USMNT vs. Germany, but those involved in the finish are punished in 3-1 defeat', 'Gregg
Berhalter -MENTIONS- What USMNT can expect from Julian Nagelsmann, Germany in first match for new manager', 'Gregg Berhalter -MENTIONS- USA vs Germany player
ratings: USMNT defensive errors blight positive performance by Pulisic, Weah', 'Gregg Berhalter -SAID- Antonee Robinson', 'Gregg Berhalter -WORKS_FOR- USMNT',
'Gregg Berhalter -PARTICIPATED_IN- USA vs Germany match', 'Tony Gustavsson -MENTIONS- How U.S. Women’s Soccer Team Lured Emma Hayes Away From Chelsea', 'Tony
Gustavsson -WORKS_FOR- Australia', 'English FA -MENTIONS- How U.S. Women’s Soccer Team Lured Emma Hayes Away From Chelsea', "U.S. Women's National Soccer Team
-MENTIONS- How U.S. Women’s Soccer Team Lured Emma Hayes Away From Chelsea", "U.S. Women's National Soccer Team -WORKS_FOR- Emma Hayes", 'international window
-MENTIONS- How U.S. Women’s Soccer Team Lured Emma Hayes Away From Chelsea', 'Georgia -MENTIONS- How the conspiracy-fueled Epoch Times went mainstream and made
millions', 'Georgia -MENTIONS- How U.S. Women’s Soccer Team Lured Emma Hayes Away From Chelsea', 'Georgia -MENTIONS- How shooting Beyoncé and Kamala Harris made
this photographer famous too', 'Georgia -MENTIONS- Eddie Jones says he needs to ‘give myself an uppercut’ but ‘100 percent’ committed to coaching Wallabies',
'Georgia -MENTIONS- RA set to announce local as high-performance director – and how it could help land Schmidt as Wallabies coach', 'Georgia -LOCATED_IN- U.S.
Soccer', 'WSL -MENTIONS- How U.S. Women’s Soccer Team Lured Emma Hayes Away From Chelsea', 'Chelsea announcement of Hayes’ departure -MENTIONS- How U.S. Women’s
Soccer Team Lured Emma Hayes Away From Chelsea', 'Chelsea announcement of Hayes’ departure -PARTICIPATED_IN- Chelsea', 'private conversations -MENTIONS- How
U.S. Women’s Soccer Team Lured Emma Hayes Away From Chelsea', 'private conversations -SAID- Emma Hayes', 'Andonovski -MENTIONS- How U.S. Women’s Soccer Team
Lured Emma Hayes Away From Chelsea', 'Emma Hayes -MENTIONS- How U.S. Women’s Soccer Team Lured Emma Hayes Away From Chelsea', 'Emma Hayes -SAID- U.S. Women’s
Soccer Team', 'Emma Hayes -SAID- private conversations', 'Emma Hayes -SAID- U.S. Soccer Federation', 'Emma Hayes -SAID- Chelsea', 'Emma Hayes -SAID- USA', 'Emma
Hayes -SAID- Cindy Parlow Cone', 'Emma Hayes -SAID- U.S. Soccer', 'Emma Hayes -WORKS_FOR- Chelsea Women', 'Emma Hayes -WORKS_FOR- U.S. women’s national team',
'Emma Hayes -WORKS_FOR- U.S. Women’s Soccer Team', 'Emma Hayes -WORKS_FOR- U.S. Soccer', "Emma Hayes -WORKS_FOR- U.S. Women's National Soccer Team", "Emma Hayes
-WORKS_FOR- U.S. Women's Soccer Team", 'Emma Hayes -WORKS_FOR- Chicago Red Stars', 'Emma Hayes -WORKS_FOR- Chelsea', 'U.S. Soccer Federation -MENTIONS- How U.S.
Women’s Soccer Team Lured Emma Hayes Away From Chelsea', 'U.S. Soccer Federation -SAID- Emma Hayes', "U.S. Women's Soccer Team -MENTIONS- How U.S. Women’s
Soccer Team Lured Emma Hayes Away From Chelsea", "U.S. Women's Soccer Team -WORKS_FOR- Emma Hayes", "U.S. Women's Soccer Team -WORKS_FOR- Twila Kilgore", 'Twila
Kilgore -MENTIONS- How U.S. Women’s Soccer Team Lured Emma Hayes Away From Chelsea', 'Twila Kilgore -SAID- Crocker', 'Twila Kilgore -WORKS_FOR- U.S. Soccer',
"Twila Kilgore -WORKS_FOR- U.S. Women's Soccer Team", '2024 Olympics -MENTIONS- How U.S. Women’s Soccer Team Lured Emma Hayes Away From Chelsea', '2024 CONCACAF
Gold Cup -MENTIONS- How U.S. Women’s Soccer Team Lured Emma Hayes Away From Chelsea', 'four friendlies -MENTIONS- How U.S. Women’s Soccer Team Lured Emma Hayes
Away From Chelsea', 'international windows -MENTIONS- How U.S. Women’s Soccer Team Lured Emma Hayes Away From Chelsea', 'Twila -MENTIONS- How U.S. Women’s
Soccer Team Lured Emma Hayes Away From Chelsea', 'Twila -WORKS_FOR- U.S. Women’s Soccer Team', 'Olympic tournament -MENTIONS- How U.S. Women’s Soccer Team Lured
Emma Hayes Away From Chelsea', 'the federation -MENTIONS- How U.S. Women’s Soccer Team Lured Emma Hayes Away From Chelsea', 'upcoming Olympic tournament
-MENTIONS- How U.S. Women’s Soccer Team Lured Emma Hayes Away From Chelsea', 'this past summer’s World Cup -MENTIONS- How U.S. Women’s Soccer Team Lured Emma
Hayes Away From Chelsea', 'this past summer’s World Cup -PARTICIPATED_IN- U.S. Women’s Soccer Team', "Spain's national team -MENTIONS- How U.S. Women’s Soccer
Team Lured Emma Hayes Away From Chelsea", "Spain's federation -MENTIONS- How U.S. Women’s Soccer Team Lured Emma Hayes Away From Chelsea", "Chelsea's away game
at Aston Villa -MENTIONS- How U.S. Women’s Soccer Team Lured Emma Hayes Away From Chelsea", "Chelsea's away game at Aston Villa -PARTICIPATED_IN- Chelsea",
"Chelsea's away game at Aston Villa -PARTICIPATED_IN- Aston Villa", 'Millie Bright -MENTIONS- How U.S. Women’s Soccer Team Lured Emma Hayes Away From Chelsea',
'Millie Bright -MENTIONS- Wullaert Fires Belgium to 3-2 Home Win over England In UEFA Women’s Nations League', 'Millie Bright -WORKS_FOR- Chelsea', 'Millie
Bright -WORKS_FOR- England', 'Millie Bright -PARTICIPATED_IN- Belgium 3-2 England', 'Denise Reddy -MENTIONS- How U.S. Women’s Soccer Team Lured Emma Hayes Away
From Chelsea', 'Denise Reddy -WORKS_FOR- Chicago Red Stars', 'Denise Reddy -LOCATED_IN- New Jersey', 'United States Under-20 -MENTIONS- How U.S. Women’s Soccer
Team Lured Emma Hayes Away From Chelsea', 'Chicago Red Stars -MENTIONS- How U.S. Women’s Soccer Team Lured Emma Hayes Away From Chelsea', 'Chicago Red Stars
-WORKS_FOR- Emma Hayes', 'Chicago Red Stars -WORKS_FOR- Denise Reddy', 'Paul Green -MENTIONS- How U.S. Women’s Soccer Team Lured Emma Hayes Away From Chelsea',
'Paul Green -WORKS_FOR- Chelsea', 'Marc Atkins -MENTIONS- How U.S. Women’s Soccer Team Lured Emma Hayes Away From Chelsea', 'Marc Atkins -PART_OF- Getty
Images', 'Catarina Macario -MENTIONS- How U.S. Women’s Soccer Team Lured Emma Hayes Away From Chelsea', 'Catarina Macario -WORKS_FOR- Chelsea', 'Mia Fishel
-MENTIONS- How U.S. Women’s Soccer Team Lured Emma Hayes Away From Chelsea', 'Mia Fishel -WORKS_FOR- Chelsea', 'Chelsea vs Real Madrid UWCL Champions League
group stage match -MENTIONS- How U.S. Women’s Soccer Team Lured Emma Hayes Away From Chelsea', 'Chelsea vs Real Madrid UWCL Champions League group stage match
-PARTICIPATED_IN- Chelsea', 'Chelsea vs Real Madrid UWCL Champions League group stage match -PARTICIPATED_IN- Real Madrid', 'Chelsea vs Real Madrid UWCL
Champions League group stage match -PART_OF- UWCL Champions League group stage', 'UWCL Champions League group stage -MENTIONS- How U.S. Women’s Soccer Team
Lured Emma Hayes Away From Chelsea', 'UWCL Champions League group stage -PART_OF- Chelsea vs Real Madrid UWCL Champions League group stage match', 'Julian
Finney -MENTIONS- How U.S. Women’s Soccer Team Lured Emma Hayes Away From Chelsea', 'Chelsea -MENTIONS- Leaky defence ‘isn’t good enough’ and ‘scars’ grow from
throwing away leads – It’s all going wrong for Manchester United in Europe', 'Chelsea -MENTIONS- Everton players and fans will be fired up by points penalty,
asserts Sean Dyche', 'Chelsea -MENTIONS- Carabao Cup fixtures, draws, results, TV channel, and live streams for every round of EFL League Cup 2023/24', 'Chelsea
-MENTIONS- Chelsea secure last-eight spot after Raheem Sterling sinks Blackburn', 'Chelsea -MENTIONS- How U.S. Women’s Soccer Team Lured Emma Hayes Away From
Chelsea', 'Chelsea -MENTIONS- Pochettino looks past James mistake and stakes future on his ‘Chelsea DNA’', 'Chelsea -MENTIONS- Arteta reveals Pochettino
influence on his career | Poch: He is family', 'Chelsea -MENTIONS- Man United vs Chelsea final score, result and highlights as Scott McTominay brace gives Ten
Hag vital win', 'Chelsea -MENTIONS- City old boy has final say for Chelsea in eight-goal thriller, supreme Salah leads Liverpool surge', 'Chelsea -MENTIONS-
Paul Mitchell signed Tottenham and Chelsea superstars, fell out with Daniel Levy and has tailor-made plan for Manchester United', 'Chelsea -MENTIONS- Man United
vs Chelsea head to head record, all-time results between Premier League giants', 'Chelsea -MENTIONS- Ten Hag’s gamble pays off against Chelsea as McTominay
steals show | Jonathan Liew', 'Chelsea -MENTIONS- Top goalscorers of 2023: Cristiano Ronaldo brings up 50 goals with Erling Haaland and Harry Kane also high on
list', 'Chelsea -MENTIONS- Liverpool vs Everton begins epic Saturday with World Cup semi-final as well as Chelsea vs Arsenal – and it could end with a bang',
'Chelsea -MENTIONS- Man United next manager: Assessing candidates to replace Erik ten Hag if Red Devils sack Dutch boss', 'Chelsea -MENTIONS- Inside the Man
United takeover and it’s anything but smooth sailing for Sir Jim Ratcliffe as he seeks board vote', 'Chelsea -MENTIONS- Breaking down each VAR intervention from
wild first half of Tottenham vs Chelsea officiated by Michael Oliver', 'Chelsea -MENTIONS- VAR apology list in Premier League: Updated list of PGMOL apologies
for referee mistakes in English football', 'Chelsea -MENTIONS- Chelsea and England star Cole Palmer: ‘The most annoying thing about moving down south? The
traffic’', 'Chelsea -MENTIONS- Tottenham’s Unbeaten Run Ends With 4-1 Home Loss to Chelsea', 'Chelsea -MENTIONS- Ange aghast as Spurs suffer ‘unbelievable’ West
Ham turnaround – with two of the worst goals you’ll see', 'Chelsea -SAID- Emma Hayes', 'Chelsea -SAID- Erik ten Hag', 'Chelsea -WORKS_FOR- Benoît Badiashile',
'Chelsea -WORKS_FOR- Emma Hayes', 'Chelsea -WORKS_FOR- Millie Bright', 'Chelsea -WORKS_FOR- Paul Green', 'Chelsea -WORKS_FOR- Catarina Macario', 'Chelsea
-WORKS_FOR- Mia Fishel', 'Chelsea -WORKS_FOR- Reece James', 'Chelsea -WORKS_FOR- Pochettino', 'Chelsea -WORKS_FOR- Mauricio Pochettino', 'Chelsea -WORKS_FOR-
Romelu Lukaku', 'Chelsea -WORKS_FOR- Armando Broja', 'Chelsea -WORKS_FOR- Graham Potter', 'Chelsea -WORKS_FOR- Jose Mourinho', 'Chelsea -WORKS_FOR- Cole
Palmer', 'Chelsea -WORKS_FOR- Sterling', 'Chelsea -WORKS_FOR- Fernandez', 'Chelsea -WORKS_FOR- Thiago Silva', 'Chelsea -WORKS_FOR- Moises Caicedo', 'Chelsea
-PARTICIPATED_IN- Middlesbrough vs. Chelsea semifinal first leg', 'Chelsea -PARTICIPATED_IN- Middlesbrough vs. Chelsea semifinal second leg', 'Chelsea
-PARTICIPATED_IN- Chelsea 1-1 Newcastle United', 'Chelsea -PARTICIPATED_IN- Chelsea vs Blackburn Rovers', 'Chelsea -PARTICIPATED_IN- Chelsea vs Brighton & Hove
Albion', 'Chelsea -PARTICIPATED_IN- Chelsea vs Blackburn match', 'Chelsea -PARTICIPATED_IN- Chelsea announcement of Hayes’ departure', "Chelsea
-PARTICIPATED_IN- Chelsea's away game at Aston Villa", 'Chelsea -PARTICIPATED_IN- Chelsea vs Real Madrid UWCL Champions League group stage match', 'Chelsea
-PARTICIPATED_IN- Chelsea vs Aston Villa', 'Chelsea -PARTICIPATED_IN- Opening weekend draw with Liverpool', 'Chelsea -PARTICIPATED_IN- Chelsea defeat against
Nottingham Forest', 'Chelsea -PARTICIPATED_IN- Chelsea visit to Fulham', 'Chelsea -PARTICIPATED_IN- Arsenal-Chelsea match', 'Chelsea -PARTICIPATED_IN-
Manchester United vs Chelsea match', 'Chelsea -PARTICIPATED_IN- Man United vs Chelsea', 'Chelsea -PARTICIPATED_IN- Man United vs Chelsea match', 'Chelsea
-PARTICIPATED_IN- Chelsea against Brighton', 'Chelsea -PARTICIPATED_IN- Chelsea-Manchester City 4-4 match', 'Chelsea -PARTICIPATED_IN- eight-goal thriller',
'Chelsea -PARTICIPATED_IN- Manchester United 4-1 home win over Chelsea last season', 'Chelsea -PARTICIPATED_IN- Chelsea 3-2 win over Brighton & Hove Albion',
'Chelsea -PARTICIPATED_IN- Manchester United vs Chelsea at Old Trafford', 'Chelsea -PARTICIPATED_IN- 2008 UEFA Champions League final', 'Chelsea
-PARTICIPATED_IN- Chelsea 1-0 win against Manchester United', 'Chelsea -PARTICIPATED_IN- Man United 4-1 Chelsea', 'Chelsea -PARTICIPATED_IN- Chelsea 1-1 Man
United', 'Chelsea -PARTICIPATED_IN- Man United 1-1 Chelsea', 'Chelsea -PARTICIPATED_IN- Chelsea 0-0 Man United', 'Chelsea -PARTICIPATED_IN- Man United 0-0
Chelsea', 'Chelsea -PARTICIPATED_IN- Man United 1-3 Chelsea', 'Chelsea -PARTICIPATED_IN- Chelsea 0-2 Man United', 'Chelsea -PARTICIPATED_IN- Chelsea 1-2 Man
United', 'Chelsea -PARTICIPATED_IN- Man United 4-0 Chelsea', 'Chelsea -PARTICIPATED_IN- game', 'Chelsea -PARTICIPATED_IN- Chelsea vs Arsenal match', 'Chelsea
-PARTICIPATED_IN- Chelsea vs Arsenal', 'Chelsea -PARTICIPATED_IN- London derby clash', 'Chelsea -PARTICIPATED_IN- Champions League final', 'Chelsea
-PARTICIPATED_IN- penalty', 'Chelsea -PARTICIPATED_IN- Tottenham vs Chelsea', 'Chelsea -PARTICIPATED_IN- Chelsea 2-1 West Ham', 'Chelsea -PARTICIPATED_IN-
Tottenham 1-4 Chelsea match', 'Chelsea -PARTICIPATED_IN- Tottenham-Chelsea match', 'Chelsea -PARTICIPATED_IN- Tottenham vs Chelsea match', 'Chelsea
-PARTICIPATED_IN- Tottenham’s 4-1 home loss to Chelsea', 'Chelsea -PARTICIPATED_IN- Tottenham’s Unbeaten Run Ends With 4-1 Home Loss to Chelsea', 'Chelsea
-PARTICIPATED_IN- Chelsea vs Manchester City', 'Chelsea -LOCATED_IN- Stamford Bridge', 'Chelsea -LOCATED_IN- London', 'Chelsea -COMPETES_WITH- Brentford',
'Chelsea -COMPETES_WITH- Blackburn', 'Chelsea -COMPETES_WITH- Manchester City', 'Chelsea -COMPETES_WITH- Manchester United', 'Chelsea -COMPETES_WITH-
Newcastle', 'Chelsea -COMPETES_WITH- United', 'Chelsea -COMPETES_WITH- Brighton', 'Chelsea -COMPETES_WITH- Arsenal', 'Chelsea -COMPETES_WITH- Tottenham',
'Chelsea -COMPETES_WITH- Middlesbrough', 'Chelsea -COMPETES_WITH- Man United', 'Chelsea -PART_OF- Premier League', 'Paris -MENTIONS- ‘It happens’: All Blacks
defend star centre after spray as Cheika heaps pressure on ‘favourites’ New Zealand', 'Paris -MENTIONS- Is Sam Bankman-Fried a bad ‘man’ or a good ‘boy’?
Lawyers swap opening statements before first witnesses take the stand', 'Paris -MENTIONS- There’s something going on with AI startups in France', 'Paris
-MENTIONS- The best movies of 2023 so far', 'Paris -MENTIONS- How U.S. Women’s Soccer Team Lured Emma Hayes Away From Chelsea', 'Paris -MENTIONS- Arteta reveals
Pochettino influence on his career | Poch: He is family', 'Paris -MENTIONS- Is Sam Bankman-Fried’s defense even trying to win?', 'Paris -MENTIONS- Where to
watch England vs South Africa: Live stream, TV channel, lineups, odds for 2023 Rugby World Cup semifinal', 'Paris -MENTIONS- England thrash Scotland but cannot
avoid Team GB Olympic heartbreak', 'Paris -MENTIONS- With Amo, the founder of Zenly wants to make social apps social again', 'Paris -MENTIONS- England choose
survival over oblivion and a chance of summer trip to France', 'Paris -MENTIONS- The Crown: What is fact and what is fiction on season 6?', 'Paris -MENTIONS-
Fans react to most ‘heartbreaking’ scene in The Crown', 'Paris -MENTIONS- Liverpool vs Everton begins epic Saturday with World Cup semi-final as well as Chelsea
vs Arsenal – and it could end with a bang', 'Paris -MENTIONS- Another cruel comeback sees Newcastle perish in group of death', "Paris -MENTIONS- Attorney
general going after Jeffrey Epstein's estate says she was fired for her dogged pursuit: 'My bar license, my integrity were more important to me'", 'Paris
-MENTIONS- Thirty-seven phases of agony define Ireland’s greatest heartbreak as curse continues', 'Paris -MENTIONS- Palestine’s growing tech industry has been
literally blown apart by Israel’s war on Hamas', 'Paris -LOCATED_IN- France', 'Paris -LOCATED_IN- Stade de France', 'Paris -LOCATED_IN- Princess Diana’s fatal
Paris car crash', 'Paris -LOCATED_IN- FAIR', 'Paris -LOCATED_IN- Poolside', 'Paris -LOCATED_IN- Mauricio Pochettino', 'Paris -LOCATED_IN- Mikel Arteta', 'Paris
-LOCATED_IN- 2023 Rugby World Cup semifinal', 'Paris -LOCATED_IN- Amo', 'Paris -LOCATED_IN- game', 'Paris -LOCATED_IN- Believe', 'Paris -LOCATED_IN- car crash',
'Paris -LOCATED_IN- Pont de l’Alma tunnel', 'Paris -LOCATED_IN- England vs Springboks', 'Paris -LOCATED_IN- Paris Olympics', 'Paris -LOCATED_IN- TwitchCon
Europe', 'Aston Villa -MENTIONS- Leaky defence ‘isn’t good enough’ and ‘scars’ grow from throwing away leads – It’s all going wrong for Manchester United in
Europe', 'Aston Villa -MENTIONS- Carabao Cup fixtures, draws, results, TV channel, and live streams for every round of EFL League Cup 2023/24', 'Aston Villa
-MENTIONS- How U.S. Women’s Soccer Team Lured Emma Hayes Away From Chelsea', 'Aston Villa -MENTIONS- Pochettino looks past James mistake and stakes future on
his ‘Chelsea DNA’', 'Aston Villa -MENTIONS- Man United vs Chelsea final score, result and highlights as Scott McTominay brace gives Ten Hag vital win', 'Aston
Villa -MENTIONS- Salah seals emphatic Europa League win for Liverpool against Toulouse', 'Aston Villa -MENTIONS- Jamaica vs Canada live score, result, updates,
highlights from CONCACAF Nations League quarterfinal first leg', 'Aston Villa -MENTIONS- Canada vs Jamaica live score, updates, highlights, result from CONCACAF
Nations League quarterfinal second leg', 'Aston Villa -MENTIONS- VAR apology list in Premier League: Updated list of PGMOL apologies for referee mistakes in
English football', 'Aston Villa -MENTIONS- Manchester divided as ten Hag pressure mounts after United Hammered, Ange-ball pays off again as Spurs sink Everton',
'Aston Villa -MENTIONS- Liverpool enjoy home comforts but Dyche’s Everton offer derby danger | Andy Hunter', 'Aston Villa -WORKS_FOR- Bailey', 'Aston Villa
-WORKS_FOR- Leon Bailey', 'Aston Villa -PARTICIPATED_IN- Aston Villa vs Everton', "Aston Villa -PARTICIPATED_IN- Chelsea's away game at Aston Villa", 'Aston
Villa -PARTICIPATED_IN- Chelsea vs Aston Villa', 'Aston Villa -PARTICIPATED_IN- Manchester City vs Aston Villa match', 'Aston Villa -PARTICIPATED_IN- Aston
Villa 4-1 win at AZ Alkmaar', 'Aston Villa -PARTICIPATED_IN- Aston Villa 3-1 Manchester United', 'Aston Villa -PARTICIPATED_IN- Fulham 3-0 Aston Villa', 'Aston
Villa -PARTICIPATED_IN- Crystal Palace 3-1 Aston Villa', 'Aston Villa -COMPETES_WITH- AZ Alkmaar', 'Aston Villa -COMPETES_WITH- Manchester City', 'Atlanta
-MENTIONS- A comprehensive list of 2023 tech layoffs', 'Atlanta -MENTIONS- Here’s how Rainforest, a budding Stripe rival, aims to win over software companies',
'Atlanta -MENTIONS- How U.S. Women’s Soccer Team Lured Emma Hayes Away From Chelsea', 'Atlanta -MENTIONS- The obsessive tormenter who made professors’ lives
miserable', 'Atlanta -MENTIONS- NFL Fantasy Football Week 6 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game', "Atlanta -MENTIONS- Can
Alabama make College Football Playoff? How Texas, Oregon losses could impact Tide's 2023 chances", "Atlanta -MENTIONS- NFL Week 13 injuries: Browns to start Joe
Flacco with Dorian Thompson-Robinson out, Saints' Olave questionable", 'Atlanta -MENTIONS- N.F.L. Executives on the Biggest Questions Facing Super Bowl
Contenders', "Atlanta -MENTIONS- Bryce Harper 'Atta Boy' controversy, explained: Why Orlando Arcia's comments led to backlash, Alanna Rizzo rant", 'Atlanta
-MENTIONS- MLB free agency tracker: Live updates on news, rumors, signings and trades after 2023 Winter Meetings', 'Atlanta -MENTIONS- College Bowl Odds,
Spreads, Money Lines and Totals - NCAAF 2023 Bowl Season', 'Atlanta -MENTIONS- Hottest seat on each NFC team: Buns burning for these four head coaches',
'Atlanta -MENTIONS- Spill toasts one year with a 2.7M seed extension, Kerry Washington and champagne', 'Atlanta -MENTIONS- Fantasy Football QB Rankings Week 14:
Who to start, best sleepers at quarterback', 'Atlanta -LOCATED_IN- Rainforest', 'Atlanta -LOCATED_IN- murder of six Asian women in three Atlanta area spas',
'Atlanta -LOCATED_IN- Atlanta Falcons', 'Atlanta -LOCATED_IN- Mercedes-Benz Stadium', 'Atlanta -LOCATED_IN- College Football Playoff National Championship',
'Atlanta -LOCATED_IN- Chick-fil-A Peach Bowl', 'Atlanta -LOCATED_IN- in-person meetups', 'Atlanta -LOCATED_IN- Convoy', 'Sarina Wiegman -MENTIONS- How U.S.
Women’s Soccer Team Lured Emma Hayes Away From Chelsea', 'Sarina Wiegman -MENTIONS- Wullaert Fires Belgium to 3-2 Home Win over England In UEFA Women’s Nations
League', 'Sarina Wiegman -MENTIONS- England thrash Scotland but cannot avoid Team GB Olympic heartbreak', 'Sarina Wiegman -MENTIONS- England choose survival
over oblivion and a chance of summer trip to France', 'Sarina Wiegman -SAID- England', 'Sarina Wiegman -WORKS_FOR- England', 'Sarina Wiegman -PARTICIPATED_IN-
Belgium 3-2 Home Win over England', 'England -MENTIONS- THE GREATEST! Boks beat All Blacks in instant classic to claim record FOURTH RWC in controversial,
dramatic final', 'England -MENTIONS- ‘It happens’: All Blacks defend star centre after spray as Cheika heaps pressure on ‘favourites’ New Zealand', 'England
-MENTIONS- West Indies v England: first T20 international – live', 'England -MENTIONS- West Indies v England: fifth and deciding men’s T20 cricket international
– live', 'England -MENTIONS- How U.S. Women’s Soccer Team Lured Emma Hayes Away From Chelsea', 'England -MENTIONS- Pochettino looks past James mistake and
stakes future on his ‘Chelsea DNA’', 'England -MENTIONS- Arteta reveals Pochettino influence on his career | Poch: He is family', 'England -MENTIONS- UK View:
‘A bridge too far’ – Boycott, former greats tear into ‘dysfunctional’ Poms with World Cup defence all but dead', 'England -MENTIONS- England', 'England
-MENTIONS- ICC World Cup 2023: India at Cricket World Cup semi-finals so far', 'England -MENTIONS- Wullaert Fires Belgium to 3-2 Home Win over England In UEFA
Women’s Nations League', 'England -MENTIONS- Where to watch England vs South Africa: Live stream, TV channel, lineups, odds for 2023 Rugby World Cup semifinal',
'England -MENTIONS- Legacy vs ego: The Springboks are playing for South Africa, the All Blacks are playing for themselves', 'England -MENTIONS- England thrash
Scotland but cannot avoid Team GB Olympic heartbreak', 'England -MENTIONS- England choose survival over oblivion and a chance of summer trip to France',
'England -MENTIONS- Top goalscorers of 2023: Cristiano Ronaldo brings up 50 goals with Erling Haaland and Harry Kane also high on list', 'England -MENTIONS-
Liverpool vs Everton begins epic Saturday with World Cup semi-final as well as Chelsea vs Arsenal – and it could end with a bang', 'England -MENTIONS- Man
United next manager: Assessing candidates to replace Erik ten Hag if Red Devils sack Dutch boss', 'England -MENTIONS- West Indies clinch T20 series after Shai
Hope edges hosts past England total', "England -MENTIONS- When was the first Cricket World Cup? History, winners' list and tournament details ahead of 2023
men's ODI showpiece", 'England -MENTIONS- ‘So used to seeing us winning’: Healy denies Aussie era ending after India loss, calls for more Tests', 'England
-MENTIONS- Chelsea and England star Cole Palmer: ‘The most annoying thing about moving down south? The traffic’', 'England -MENTIONS- Gabriel Jesus’s sublime
goal and assist gives Arsenal vital victory in Sevilla', 'England -MENTIONS- Theatre producer and Everton chairman Kenwright dies', 'England -SAID- Alex
Greenwood', 'England -SAID- Aatif Nawaz', 'England -SAID- Sir Geoff Boycott', 'England -SAID- Hussain', 'England -SAID- Atherton', 'England -SAID- Sarina
Wiegman', 'England -SAID- Steve Borthwick', 'England -WORKS_FOR- Millie Bright', 'England -WORKS_FOR- Jos Buttler', 'England -WORKS_FOR- Eoin Morgan', 'England
-WORKS_FOR- Buttler', 'England -WORKS_FOR- Sarina Wiegman', 'England -WORKS_FOR- Steve Borthwick', 'England -WORKS_FOR- Harry Kane', 'England -WORKS_FOR- Gareth
Southgate', 'England -WORKS_FOR- Eddie Jones', 'England -WORKS_FOR- Cole Palmer', 'England -PARTICIPATED_IN- World Cup', 'England -PARTICIPATED_IN- All
Blacks-England semi-final', 'England -PARTICIPATED_IN- West Indies v England: first T20 international', 'England -PARTICIPATED_IN- first T20 international',
'England -PARTICIPATED_IN- West Indies v England series', 'England -PARTICIPATED_IN- West Indies v England: fifth and deciding men’s T20 cricket international',
'England -PARTICIPATED_IN- West Indies v England fifth and deciding men’s T20 cricket international', 'England -PARTICIPATED_IN- fifth and deciding men’s T20
cricket international', 'England -PARTICIPATED_IN- T20 World Cup', 'England -PARTICIPATED_IN- India vs England semi-final', 'England -PARTICIPATED_IN-
India-England semi-final', 'England -PARTICIPATED_IN- 1987 India-England semi-final', 'England -PARTICIPATED_IN- Belgium 3-2 England', 'England
-PARTICIPATED_IN- Belgium 3-2 Home Win over England', 'England -PARTICIPATED_IN- England-Netherlands showdown', 'England -PARTICIPATED_IN- Wullaert Fires
Belgium to 3-2 Home Win over England In UEFA Women’s Nations League', 'England -PARTICIPATED_IN- England host Netherlands', 'England -PARTICIPATED_IN- 2019
Rugby World Cup final', 'England -PARTICIPATED_IN- 2019 Rugby World Cup', 'England -PARTICIPATED_IN- second semi-final', 'England -PARTICIPATED_IN- England vs
South Africa 2023 Rugby World Cup semifinal', 'England -PARTICIPATED_IN- 2023 Rugby World Cup semifinal', "England -PARTICIPATED_IN- South Africa's 27-13
victory at Twickenham", 'England -PARTICIPATED_IN- 2019 World Cup final', 'England -PARTICIPATED_IN- Nations League', 'England -PARTICIPATED_IN- Friday’s late
3-2 win against the Netherlands', 'England -PARTICIPATED_IN- England-Scotland showdown', 'England -PARTICIPATED_IN- Nations League campaign', 'England
-PARTICIPATED_IN- match', 'England -PARTICIPATED_IN- game', 'England -PARTICIPATED_IN- Rugby World Cup', 'England -PARTICIPATED_IN- England vs South Africa
Rugby World Cup semi-final', 'England -PARTICIPATED_IN- England vs South Africa ODI Cricket World Cup match', 'England -PARTICIPATED_IN- England vs Fiji Rugby
World Cup quarter-final', 'England -PARTICIPATED_IN- Cricket World Cup', 'England -PARTICIPATED_IN- England vs Springboks', 'England -PARTICIPATED_IN- England
vs Afghanistan', 'England -PARTICIPATED_IN- England vs New Zealand', 'England -PARTICIPATED_IN- T20 series', 'England -PARTICIPATED_IN- Inaugural ODI World
Cup', 'England -PARTICIPATED_IN- 1992 World Cup final', 'England -PARTICIPATED_IN- Ashes multi-format series', 'England -LOCATED_IN- Belgium', 'England
-COMPETES_WITH- West Indies', 'England -COMPETES_WITH- India', 'England -COMPETES_WITH- South Africa', 'England -COMPETES_WITH- Netherlands', 'England
-COMPETES_WITH- Dutch', 'England -COMPETES_WITH- Scotland', 'England -COMPETES_WITH- Springboks', 'England -COMPETES_WITH- Australia', 'England -COMPETES_WITH-
Pakistan', 'England -PART_OF- Tom Curry', 'Real Madrid -MENTIONS- European football: Eintracht Frankfurt hit Bayern for five in famous thrashing', 'Real Madrid
-MENTIONS- How U.S. Women’s Soccer Team Lured Emma Hayes Away From Chelsea', 'Real Madrid -MENTIONS- Paul Mitchell signed Tottenham and Chelsea superstars, fell
out with Daniel Levy and has tailor-made plan for Manchester United', 'Real Madrid -MENTIONS- Champions League permutations, scenarios: What teams need to
qualify from groups for 2023/24 Round of 16 stage', 'Real Madrid -MENTIONS- Top goalscorers of 2023: Cristiano Ronaldo brings up 50 goals with Erling Haaland
and Harry Kane also high on list', 'Real Madrid -MENTIONS- Man United next manager: Assessing candidates to replace Erik ten Hag if Red Devils sack Dutch boss',
'Real Madrid -MENTIONS- European roundup: Bellingham helps Real Madrid go top while Bayern draw', 'Real Madrid -MENTIONS- Another cruel comeback sees Newcastle
perish in group of death', 'Real Madrid -MENTIONS- French soccer players back African mobile video network StarNews Mobile in $3M round', 'Real Madrid
-WORKS_FOR- Zinedine Zidane', 'Real Madrid -WORKS_FOR- Lopetegui', 'Real Madrid -WORKS_FOR- Carlo Ancelotti', 'Real Madrid -WORKS_FOR- Bellingham', 'Real Madrid
-WORKS_FOR- Aurélien Tchouaméni', 'Real Madrid -PARTICIPATED_IN- Real Betis 1-1 Real Madrid draw', 'Real Madrid -PARTICIPATED_IN- Chelsea vs Real Madrid UWCL
Champions League group stage match', 'Real Madrid -PARTICIPATED_IN- group stage', 'Real Madrid -PARTICIPATED_IN- Real Madrid 3-2 win away to Union Berlin',
'Real Madrid -PARTICIPATED_IN- Real Madrid 1-0 Union Berlin', 'Real Madrid -PARTICIPATED_IN- Napoli 2-3 Real Madrid', 'Real Madrid -PARTICIPATED_IN- Braga 1-2
Real Madrid', 'Real Madrid -PARTICIPATED_IN- Real Madrid 3-0 Braga', 'Real Madrid -PARTICIPATED_IN- Real Madrid 4-2 Napoli', 'Real Madrid -PARTICIPATED_IN-
Union Berlin 2-3 Real Madrid', 'Real Madrid -COMPETES_WITH- Liverpool', 'Berhalter -MENTIONS- How U.S. Women’s Soccer Team Lured Emma Hayes Away From Chelsea',
'Berhalter -MENTIONS- USA squad for October friendlies 2023: USMNT roster for friendly matches vs Germany, Ghana', 'Berhalter -MENTIONS- Gio Reyna looks good in
start for USMNT vs. Germany, but those involved in the finish are punished in 3-1 defeat', 'Berhalter -SAID- Tyler Adams', 'Berhalter -SAID- Reyna', 'Berhalter
-SAID- USMNT vs. Germany', 'Berhalter -WORKS_FOR- USMNT', 'the U.S. -MENTIONS- How U.S. Women’s Soccer Team Lured Emma Hayes Away From Chelsea', "the U.S.
-MENTIONS- We could end the AIDS epidemic in less than a decade. Here's how.", 'the U.S. -LOCATED_IN- the U.S. government', 'Australia -MENTIONS- How U.S.
Women’s Soccer Team Lured Emma Hayes Away From Chelsea', 'Australia -MENTIONS- UK View: ‘A bridge too far’ – Boycott, former greats tear into ‘dysfunctional’
Poms with World Cup defence all but dead', 'Australia -MENTIONS- ICC World Cup 2023: India at Cricket World Cup semi-finals so far', 'Australia -MENTIONS- Who’s
the next Gawn, Grundy or Dusty? AFL’s talent guru names top 40 draft prospects and compares them to current stars', 'Australia -MENTIONS- IND vs AUS
Head-to-Head Record in ODI World Cup: Check India vs Australia match stats in World Cup Finals and ODI series', 'Australia -MENTIONS- Eddie Jones says he needs
to ‘give myself an uppercut’ but ‘100 percent’ committed to coaching Wallabies', "Australia -MENTIONS- When was the first Cricket World Cup? History, winners'
list and tournament details ahead of 2023 men's ODI showpiece", 'Australia -MENTIONS- ‘So used to seeing us winning’: Healy denies Aussie era ending after India
loss, calls for more Tests', 'Australia -MENTIONS- ‘Biggest win of my career’: De Minaur, Popyrin power Australia into Davis Cup final', 'Australia -MENTIONS-
ICC World Cup 2023 Most Runs and Wickets: Full List of top 10 batsmen and bowlers in 2023 edition of cricket World Cup, Know where Virat Kohli, Rohit Sharma,
Shami, Bumrah, rank', 'Australia -MENTIONS- Thirty-seven phases of agony define Ireland’s greatest heartbreak as curse continues', 'Australia -SAID- white-ball
cricket', 'Australia -SAID- De Minaur', 'Australia -WORKS_FOR- Tony Gustavsson', 'Australia -WORKS_FOR- Mark Waugh', 'Australia -WORKS_FOR- Damien Fleming',
'Australia -WORKS_FOR- Brett Lee', 'Australia -WORKS_FOR- Jason Gillespie', 'Australia -WORKS_FOR- Matthew Hayden', 'Australia -WORKS_FOR- Adam Gilchrist',
'Australia -WORKS_FOR- Ricky Ponting', 'Australia -WORKS_FOR- Damien Martyn', 'Australia -WORKS_FOR- Healy', 'Australia -WORKS_FOR- Lleyton Hewitt', 'Australia
-WORKS_FOR- Hewitt', 'Australia -WORKS_FOR- Mitch Marsh', 'Australia -PARTICIPATED_IN- South Africa vs Australia semi-final', 'Australia -PARTICIPATED_IN-
Australia vs South Africa Cricket World Cup 2023 Semi-Final Match', 'Australia -PARTICIPATED_IN- second semi-final of the 2015 World Cup', 'Australia
-PARTICIPATED_IN- second semi-final', "Australia -PARTICIPATED_IN- ICC Men's ODI World Cup final", "Australia -PARTICIPATED_IN- ICC Men's ODI World Cup 2023
final", 'Australia -PARTICIPATED_IN- India vs Australia 2023 World Cup final', 'Australia -PARTICIPATED_IN- 2023 World Cup league fixture', 'Australia
-PARTICIPATED_IN- 2003 final', 'Australia -PARTICIPATED_IN- 1983, Match 11, Prudential World Cup', 'Australia -PARTICIPATED_IN- 1983, Match 23, Prudential World
Cup', 'Australia -PARTICIPATED_IN- 1987, Match 3, Reliance World Cup', 'Australia -PARTICIPATED_IN- 1987, Match 15, Reliance World Cup', 'Australia
-PARTICIPATED_IN- 1992, Match 12', 'Australia -PARTICIPATED_IN- 1996, Match 19, Wills World Cup', 'Australia -PARTICIPATED_IN- 1999, 1st Super, ICC World Cup',
'Australia -PARTICIPATED_IN- India vs Australia match', 'Australia -PARTICIPATED_IN- 2003 Final, ICC World Cup', 'Australia -PARTICIPATED_IN- 2003 World Cup
final', 'Australia -PARTICIPATED_IN- 2011, 2nd Quarter-Final, ICC Cricket World Cup', 'Australia -PARTICIPATED_IN- 2015, 2nd Semi-Final', 'Australia
-PARTICIPATED_IN- 2023, 5th Match, ICC Cricket World Cup', 'Australia -PARTICIPATED_IN- India vs Australia match in the ODI World Cup', 'Australia
-PARTICIPATED_IN- Inaugural ODI World Cup', 'Australia -PARTICIPATED_IN- 2011 World Cup', 'Australia -PARTICIPATED_IN- Australia-India Test', 'Australia
-PARTICIPATED_IN- 50-over World Cup', 'Australia -PARTICIPATED_IN- Ashes multi-format series', 'Australia -PARTICIPATED_IN- one-off Test', 'Australia
-PARTICIPATED_IN- losses in England', 'Australia -PARTICIPATED_IN- Australia loss to India', 'Australia -PARTICIPATED_IN- multi-format 2021-22 series against
India', 'Australia -PARTICIPATED_IN- Davis Cup final', 'Australia -PARTICIPATED_IN- Davis Cup', 'Australia -PARTICIPATED_IN- Davis Cup semi-final', 'Australia
-PARTICIPATED_IN- quarter-final with the Czech Republic', 'Australia -PARTICIPATED_IN- 2-0 defeat by Canada', "Australia -PARTICIPATED_IN- final of the ICC
Men's ODI World Cup 2023", 'Australia -PARTICIPATED_IN- 1991 Australia victory', 'Australia -PARTICIPATED_IN- Test match', 'Australia -COMPETES_WITH- India',
'Australia -COMPETES_WITH- England', 'Australia -COMPETES_WITH- Pakistan', 'Australia -COMPETES_WITH- Finland', 'Australia -COMPETES_WITH- Serbian side',
'Australia -COMPETES_WITH- Italian team', 'Australia -COMPETES_WITH- Italy', 'Australia -PART_OF- Ricky Ponting', 'Australia -PART_OF- Popyrin', 'Australia
-PART_OF- De Minaur', 'Australia -PART_OF- Felix Auger-Aliassime', 'Australia -PART_OF- Denis Shapovalov', 'USMNT -MENTIONS- Chicago Fire vs Inter Miami live
score, result, updates, highlights from Lionel Messi in MLS match', 'USMNT -MENTIONS- How U.S. Women’s Soccer Team Lured Emma Hayes Away From Chelsea', 'USMNT
-MENTIONS- USA squad for October friendlies 2023: USMNT roster for friendly matches vs Germany, Ghana', 'USMNT -MENTIONS- Canada vs Jamaica live score, updates,
highlights, result from CONCACAF Nations League quarterfinal second leg', 'USMNT -MENTIONS- Gio Reyna looks good in start for USMNT vs. Germany, but those
involved in the finish are punished in 3-1 defeat', 'USMNT -MENTIONS- What USMNT can expect from Julian Nagelsmann, Germany in first match for new manager',
'USMNT -MENTIONS- USA vs Germany player ratings: USMNT defensive errors blight positive performance by Pulisic, Weah', 'USMNT -WORKS_FOR- Landon Donovan',
'USMNT -WORKS_FOR- Gregg Berhalter', 'USMNT -WORKS_FOR- Gio Reyna', 'USMNT -WORKS_FOR- Berhalter', 'USMNT -PARTICIPATED_IN- World Cup', 'USMNT -PARTICIPATED_IN-
USMNT vs. Germany 3-1 defeat', 'USMNT -PARTICIPATED_IN- USMNT vs. Germany', 'USMNT -PARTICIPATED_IN- USA friendly', 'USMNT -PARTICIPATED_IN- USA vs Germany',
'USMNT -PARTICIPATED_IN- USA vs Germany match', 'USMNT -COMPETES_WITH- Germany', 'Harry -MENTIONS- How U.S. Women’s Soccer Team Lured Emma Hayes Away From
Chelsea', 'Harry -MENTIONS- The Crown: What is fact and what is fiction on season 6?', 'Harry -MENTIONS- Fans react to most ‘heartbreaking’ scene in The Crown',
'Harry -MENTIONS- Thanksgiving is Spider-Man’s holiday', 'Harry -PARTICIPATED_IN- The last phone call between Harry, William and Diana', 'Harry -LOCATED_IN-
Balmoral', 'Champions League final -MENTIONS- How U.S. Women’s Soccer Team Lured Emma Hayes Away From Chelsea', 'Champions League final -MENTIONS- Man United vs
Chelsea head to head record, all-time results between Premier League giants', 'Champions League final -MENTIONS- Erik ten Hag could pay the price for Manchester
United’s incompetent Champions League campaign', 'Champions League final -MENTIONS- Ten Hag demands both positivity and pragmatism from erratic Onana',
'Champions League final -PARTICIPATED_IN- Chelsea', 'Champions League final -PARTICIPATED_IN- Manchester City', 'Champions League final -PARTICIPATED_IN- André
Onana', 'Champions League final -PARTICIPATED_IN- Internazionale', 'Champions League final -LOCATED_IN- Istanbul', 'Todd Boehly -MENTIONS- Chelsea secure
last-eight spot after Raheem Sterling sinks Blackburn', 'Todd Boehly -MENTIONS- How U.S. Women’s Soccer Team Lured Emma Hayes Away From Chelsea', 'Todd Boehly
-MENTIONS- Man United next manager: Assessing candidates to replace Erik ten Hag if Red Devils sack Dutch boss', 'Todd Boehly -MENTIONS- Inside the Man United
takeover and it’s anything but smooth sailing for Sir Jim Ratcliffe as he seeks board vote', 'Todd Boehly -PARTICIPATED_IN- Todd Boehly deal to invest in
Chelsea', '2027 World Cup -MENTIONS- How U.S. Women’s Soccer Team Lured Emma Hayes Away From Chelsea', '2027 World Cup -MENTIONS- Eddie Jones says he needs to
‘give myself an uppercut’ but ‘100 percent’ committed to coaching Wallabies', '2027 World Cup -SAID- Eddie Jones', 'soccer -MENTIONS- Moneyline Betting - What
is the Moneyline and How Does it Work?', 'soccer -MENTIONS- How U.S. Women’s Soccer Team Lured Emma Hayes Away From Chelsea', "soccer -MENTIONS- Young Taylor
Swift fans go viral for struggling to open the star's CD: 'Making me laugh'", 'soccer -MENTIONS- Point Spread Betting - What is a Point Spread Bet?', "soccer
-MENTIONS- Billie Jean King sees progress in her crusade for more investment women’s sports: 'We’re kind of at a tipping point'", 'soccer -MENTIONS- Twitch is
the best way to watch sports, when you can actually watch sports on Twitch', 'soccer -SAID- Billie Jean King', 'New Jersey -MENTIONS- Christmas Day preview:
49ers, Ravens square off in potential Super Bowl sneak peek', 'New Jersey -MENTIONS- Taylor Swift-Travis Kelce timeline: Everything to know about rumored
romance between pop star, Chiefs TE', 'New Jersey -MENTIONS- After losing her leg to the flu, Virginia woman urges people to get vaccinated: ‘Don’t waste
time’', 'New Jersey -MENTIONS- How the conspiracy-fueled Epoch Times went mainstream and made millions', 'New Jersey -MENTIONS- How U.S. Women’s Soccer Team
Lured Emma Hayes Away From Chelsea', 'New Jersey -MENTIONS- Cranberries: A Thanksgiving Staple With a Rich History and Unique Biology', "New Jersey -MENTIONS-
Chiefs vs. Jets live score, updates, highlights from NFL 'Sunday Night Football' game", 'New Jersey -MENTIONS- Best sportsbook bonus offers for NFL Monday Night
Football Eagles vs. Seahawks: Claim over $5,000 in bonuses from Bet365, BetMGM, BetRivers, Caesars Sportsbook, DraftKings and FanDuel', 'New Jersey -MENTIONS-
Christmas proves that love triumphs over hate — even amid war, suffering, tragedy and turmoil', 'New Jersey -MENTIONS- Best NHL prop bets tonight: Artemi
Panarin, Cale Makar highlight top parlay picks for Rangers-Red Wings, Devils-Avalanche', "New Jersey -MENTIONS- 5 myths about ketamine, the drug tied to Matthew
Perry's death, according to doctors", 'New Jersey -MENTIONS- Best sports betting promo codes for college basketball: Claim $5,000+ in bonuses from Caesars
Sportsbook, BetMGM, Bet365, FanDuel, DraftKings, BetRivers', "New Jersey -MENTIONS- Attorney general going after Jeffrey Epstein's estate says she was fired for
her dogged pursuit: 'My bar license, my integrity were more important to me'", 'New Jersey -MENTIONS- Travis Kelce admits the NFL is overdoing it with their
Taylor Swift coverage', 'New Jersey -LOCATED_IN- Cedar Grove', 'New Jersey -LOCATED_IN- CSL Seqirus', 'New Jersey -LOCATED_IN- Denise Reddy', 'New Jersey
-LOCATED_IN- cranberries', 'New Jersey -LOCATED_IN- Jets', 'New Jersey -LOCATED_IN- Chiefs vs. Jets Sunday Night Football game', 'New Jersey -LOCATED_IN- Initia
Nova Medical Solutions', 'New Jersey -LOCATED_IN- Erin Palinski-Wade', 'New Jersey -LOCATED_IN- Teterboro Airport', 'New Jersey -LOCATED_IN- Donna Kelce',
'London -MENTIONS- How ego and fear fuelled the rise of artificial intelligence', 'London -MENTIONS- Is Sam Bankman-Fried a bad ‘man’ or a good ‘boy’? Lawyers
swap opening statements before first witnesses take the stand', 'London -MENTIONS- There’s something going on with AI startups in France', 'London -MENTIONS-
West Indies v England: first T20 international – live', 'London -MENTIONS- Carabao Cup fixtures, draws, results, TV channel, and live streams for every round of
EFL League Cup 2023/24', 'London -MENTIONS- How U.S. Women’s Soccer Team Lured Emma Hayes Away From Chelsea', "London -MENTIONS- Gen Z just saw the worst war in
Israel in 50 years break out all over the internet, but they were raised on 'disturbing images'", 'London -MENTIONS- Patterns of Surface Warming Matter for
Climate Sensitivity', 'London -MENTIONS- Wullaert Fires Belgium to 3-2 Home Win over England In UEFA Women’s Nations League', 'London -MENTIONS- Is Sam
Bankman-Fried’s defense even trying to win?', 'London -MENTIONS- The Halloween Countdown: 31 days of horror to watch', 'London -MENTIONS- Japan’s prime minister
Kishida says: ‘reassess our economy’', 'London -MENTIONS- Michelle Jubelirer on Ice Spice, The Beatles, and reinventing Capitol Music Group: ‘I don’t think any
label group has been able to turn things around as quickly as my team and I have done in less than two years.’', 'London -MENTIONS- Microsoft Activision: What
does deal mean for gamers?', "London -MENTIONS- We could end the AIDS epidemic in less than a decade. Here's how.", 'London -MENTIONS- 8 authors and their 8
gift book picks for this holiday season', 'London -MENTIONS- Fans react to most ‘heartbreaking’ scene in The Crown', 'London -MENTIONS- Liverpool vs Everton
begins epic Saturday with World Cup semi-final as well as Chelsea vs Arsenal – and it could end with a bang', 'London -MENTIONS- Sleepy tea, counting backwards
and dense earplugs: Inside the lives of insomniacs', 'London -MENTIONS- Man United next manager: Assessing candidates to replace Erik ten Hag if Red Devils sack
Dutch boss', 'London -MENTIONS- Watch great noir this Noirvember', "London -MENTIONS- OpenAI's ex-chairman accuses board of going rogue in firing Altman: 'Sam
and I are shocked and saddened by what the board did'", 'London -MENTIONS- Breaking down each VAR intervention from wild first half of Tottenham vs Chelsea
officiated by Michael Oliver', "London -MENTIONS- Billie Jean King sees progress in her crusade for more investment women’s sports: 'We’re kind of at a tipping
point'", "London -MENTIONS- Israel's prince of public diplomacy", 'London -MENTIONS- Simply Homes nabs $22M, leverages AI to tackle affordable housing crisis',
'London -MENTIONS- Tottenham’s Unbeaten Run Ends With 4-1 Home Loss to Chelsea', 'London -MENTIONS- The airline parts scandal sparking panic and introspection
in aerospace', 'London -MENTIONS- Theatre producer and Everton chairman Kenwright dies', 'London -MENTIONS- Inside the $US2.3b venue that will transform the
concert industry', 'London -MENTIONS- Consistent policy is key to unlocking the climate tech boom, but the UK’s U-turn isn’t helping', 'London -LOCATED_IN-
England', 'London -LOCATED_IN- London derby', 'London -LOCATED_IN- Activision Blizzard', 'London -LOCATED_IN- DeepMind', 'London -LOCATED_IN- King', 'London
-LOCATED_IN- FTX', 'London -LOCATED_IN- Wembley Stadium', 'London -LOCATED_IN- Chelsea', 'London -LOCATED_IN- Awa Business School', 'London -LOCATED_IN-
Imperial College London', 'London -LOCATED_IN- Marc-Antoine Julliard', 'London -LOCATED_IN- series of conferences', 'London -LOCATED_IN- Whitehall', 'London
-LOCATED_IN- James Wise', 'London -LOCATED_IN- Crystal Palace', 'London -LOCATED_IN- event in London', 'London -LOCATED_IN- Tottenham Hotspur Stadium', 'London
-LOCATED_IN- Tottenham', 'London -LOCATED_IN- AOG', 'London -LOCATED_IN- Frank and Percy', 'London -LOCATED_IN- ABBA Voyage show', 'London -LOCATED_IN- ABBA’s
show in London', 'London -LOCATED_IN- ABBA Arena', 'Olympics -MENTIONS- How U.S. Women’s Soccer Team Lured Emma Hayes Away From Chelsea', 'Olympics -MENTIONS-
England choose survival over oblivion and a chance of summer trip to France', 'Olympics -MENTIONS- The Soccer Star on the Verge of Breaking an 89-Year-Old
Record', 'Olympics -MENTIONS- RA set to announce local as high-performance director – and how it could help land Schmidt as Wallabies coach', 'U.S. -MENTIONS-
SBF’s trial starts soon, but how did he — and FTX — get here?', 'U.S. -MENTIONS- Sam Altman backs teens’ startup, Google unveils the Pixel 8 and TikTok tests an
ad-free tier', 'U.S. -MENTIONS- More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo', "U.S. -MENTIONS- Chiefs vs.
Packers live score, updates, highlights from NFL 'Sunday Night Football' game", "U.S. -MENTIONS- Vikings vs. Broncos live score, updates, highlights from NFL
'Sunday Night Football' game", 'U.S. -MENTIONS- There’s something going on with AI startups in France', 'U.S. -MENTIONS- What to know about Fortnite maker Epic
Games’ antitrust battle with Google, starting today', 'U.S. -MENTIONS- One year later, ChatGPT is still alive and kicking', 'U.S. -MENTIONS- ChatGPT: Everything
you need to know about the AI-powered chatbot', 'U.S. -MENTIONS- Italy gives OpenAI to-do list for lifting ChatGPT suspension order', 'U.S. -MENTIONS- A
comprehensive list of 2023 tech layoffs', 'U.S. -MENTIONS- After losing her leg to the flu, Virginia woman urges people to get vaccinated: ‘Don’t waste time’',
'U.S. -MENTIONS- Ozempic, Wegovy may be linked to stomach paralysis and other digestive issues in large-scale study', 'U.S. -MENTIONS- 2023 Kentucky online
sports betting sites: Best legal sportsbooks, promos, bonuses, mobile apps, how to bet', 'U.S. -MENTIONS- How the conspiracy-fueled Epoch Times went mainstream
and made millions', 'U.S. -MENTIONS- Biden’s AI EO hailed as broad, but not deep without legislation to match', 'U.S. -MENTIONS- Meta has a moderation bias
problem, not just a ‘bug,’ that’s suppressing Palestinian voices', 'U.S. -MENTIONS- How U.S. Women’s Soccer Team Lured Emma Hayes Away From Chelsea', 'U.S.
-MENTIONS- Buckle up for more ‘turbulence’ in the housing market, BofA says—it’s a housing recession, 1980s style', 'U.S. -MENTIONS- Meet the American who made
cranberries a Turkey Day tradition, Marcus Urann, farmer with can-do spirit', 'U.S. -MENTIONS- Cranberries: A Thanksgiving Staple With a Rich History and Unique
Biology', "U.S. -MENTIONS- Israel's blockade of Gaza means that a region the size of Philadelphia faces a real risk of starvation, says Peace Studies scholar",
"U.S. -MENTIONS- Gen Z just saw the worst war in Israel in 50 years break out all over the internet, but they were raised on 'disturbing images'", 'U.S.
-MENTIONS- Meta seeks legislation that would require parents to approve teens’ app downloads', "U.S. -MENTIONS- Dolphins vs. Eagles live score, updates,
highlights from NFL 'Sunday Night Football' game", "U.S. -MENTIONS- Bears vs. Vikings live score, updates, highlights from NFL 'Monday Night Football' game",
"U.S. -MENTIONS- Mayo Clinic sees AI as 'transformative force' in health care, appoints Dr. Bhavik Patel as chief AI officer", 'U.S. -MENTIONS- Beyoncé tops box
office with $21 million debut for concert film ‘Renaissance’', 'U.S. -MENTIONS- The fate of US venture capital in China teeters on uncertainty', 'U.S.
-MENTIONS- Amazon brings conversational AI to kids with launch of ‘Explore with Alexa’', 'U.S. -MENTIONS- If your startup doesn’t seem impossible at first, it’s
not hard enough', "U.S. -MENTIONS- Chiefs vs. Jets live score, updates, highlights from NFL 'Sunday Night Football' game", 'U.S. -MENTIONS- Michelle Jubelirer
on Ice Spice, The Beatles, and reinventing Capitol Music Group: ‘I don’t think any label group has been able to turn things around as quickly as my team and I
have done in less than two years.’', 'U.S. -MENTIONS- Nike misses revenue expectations for the first time in two years, beats on earnings and gross margin',
"U.S. -MENTIONS- 'Too big to fail' Egypt faces economic crisis but has an anti-Gaza refugee policy. Israel and the U.S. hope that will change", "U.S. -MENTIONS-
Sex, Signal messages, and sabotaging FTX's code: SBF execs and Bahamas roommates tell all in court", 'U.S. -MENTIONS- Keep Labs’ smart storage redefines
medication management', 'U.S. -MENTIONS- Canada vs Jamaica live score, updates, highlights, result from CONCACAF Nations League quarterfinal second leg', 'U.S.
-MENTIONS- 5 things we learned so far about the Google antitrust case', "U.S. -MENTIONS- We could end the AIDS epidemic in less than a decade. Here's how.",
'U.S. -MENTIONS- WebSummit derailed by founder’s public fight with those supporting Israel in Hamas conflict', "U.S. -MENTIONS- Cowboys vs. 49ers live score,
updates, highlights from NFL 'Sunday Night Football' game", "U.S. -MENTIONS- Lions vs. Packers live score, updates, highlights from NFL 'Thursday Night
Football' game", "U.S. -MENTIONS- Seahawks vs. Giants live score, updates, highlights from NFL 'Monday Night Football' game", "U.S. -MENTIONS- Bills vs. Bengals
live score, updates, highlights from NFL 'Sunday Night Football' game", 'U.S. -MENTIONS- Meta turned a blind eye to kids on its platforms for years, unredacted
lawsuit alleges', 'U.S. -MENTIONS- The small and affordable EVs we lost in 2023', 'U.S. -MENTIONS- Northern Lights: Here are the best tips to help you spot the
stunning display in the US and abroad', 'U.S. -MENTIONS- Google fakes an AI demo, Grand Theft Auto VI goes viral and Spotify cuts jobs', 'U.S. -MENTIONS- The
Steam Deck OLED improves much more than its screen', 'U.S. -MENTIONS- 8 predictions for AI in 2024', "U.S. -MENTIONS- Check Point sees 'fantastic' year ahead
even as rival report disappointing billings", 'U.S. -MENTIONS- Binance to pay $4.3B in fines and CEO ‘CZ’ to step down, plead guilty to anti-money laundering
charges', 'U.S. -MENTIONS- Coinbase rallies more than 60% in same month that FTX and Binance founders brace for prison', 'U.S. -MENTIONS- Temu was the
most-downloaded iPhone app in the U.S. in 2023', 'U.S. -MENTIONS- U.S.-China relations are now more about crisis prevention', "U.S. -MENTIONS- At Christmas, Don
McLean sings the praises of 'Silent Night,' other holiday classics: 'My mother's favorite'", 'U.S. -MENTIONS- World Series 2023 live score: Rangers vs.
Diamondbacks updates, highlights, results from Game 4', "U.S. -MENTIONS- Billie Jean King sees progress in her crusade for more investment women’s sports:
'We’re kind of at a tipping point'", 'U.S. -MENTIONS- Twitch is the best way to watch sports, when you can actually watch sports on Twitch', "U.S. -MENTIONS-
Jets vs. Raiders live score, updates, highlights from NFL 'Sunday Night Football' game", "U.S. -MENTIONS- With hopes of victory fading, Ukraine's fight against
Russia could get even harder in 2024", 'U.S. -MENTIONS- Israel’s startup ecosystem: down but not out', 'U.S. -MENTIONS- Patriots vs. Broncos live score,
updates, highlights from NFL Sunday night football game', 'U.S. -MENTIONS- Adobe: Thanksgiving US online sales nudge up to $5.6B; Salesforce: $31.7B spent
globally', 'U.S. -MENTIONS- The housing winter arrives early as existing home sales crash to Great Recession levels and the long-feared ‘deep freeze’ sets in',
"U.S. -MENTIONS- Bears vs. Panthers live score, updates, highlights from NFL 'Thursday Night Football' game", 'U.S. -MENTIONS- World Series 2023 live score:
Rangers vs. Diamondbacks updates, highlights, results from Game 3', "U.S. -MENTIONS- 'In God We Trust' essential to America's future as national faith appears
to wane, says Washington pastor", 'U.S. -MENTIONS- Tesla ‘digs its own grave with the Cybertruck,’ Convoy collapses and Rivian scores a win at Rebelle', 'U.S.
-MENTIONS- Amazon’s Tye Brady discusses generative AI, humanoid robots and mobile manipulation', "U.S. -MENTIONS- The inside story of Dave Clark's tumultuous
last days at Flexport", 'U.S. -MENTIONS- Beeper reversed engineered iMessage to bring blue bubble texts to Android users', 'U.S. -MENTIONS- What’s up with
Tesla’s Cybertruck? Everything to know about much-hyped electric pickup', 'U.S. -MENTIONS- Fly with Delta frequently? Small business owners can score a solid
welcome bonus and other perks with the Delta SkyMiles Gold Business Card', 'U.S. -MENTIONS- The best horror movies you can watch right now', "U.S. -MENTIONS-
Dangerous 'superbugs' are a growing threat, and antibiotics can't stop their rise. What can?", 'U.S. -MENTIONS- Pittsburgh boy, 10, needs second liver
transplant to save his life: ‘Only possible through love’', "U.S. -MENTIONS- Israeli girl spends 9th birthday in 'tunnels of Gaza,' with 'no friends, 'no
light,' says distraught father", 'U.S. -MENTIONS- The best comedies you’ll find on streaming right now', 'U.S. -SAID- Sylvester', 'U.S. -LOCATED_IN- Chapter 11
bankruptcy', 'U.S. -LOCATED_IN- Copa America 2024', 'U.S. -LOCATED_IN- research institutions', 'U.S. -LOCATED_IN- California', 'U.S. -LOCATED_IN- autonomous
truck startup', 'U.S. -LOCATED_IN- Binance', 'U.S. -LOCATED_IN- crypto staking services', 'U.S. -LOCATED_IN- Texas', 'U.S. -LOCATED_IN- eight U.S. medical
centers', 'football -MENTIONS- Manchester United face Galatasaray with high hopes but bad memories', 'football -MENTIONS- When pop culture and sport collide: A
timeline of Taylor Swift’s NFL takeover', 'football -MENTIONS- Taylor Swift is Time’s Person of the Year', 'football -MENTIONS- NFL Fantasy Football Week 5
Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game', 'football -MENTIONS- Moneyline Betting - What is the Moneyline and How Does it Work?',
'football -MENTIONS- 2023 Kentucky online sports betting sites: Best legal sportsbooks, promos, bonuses, mobile apps, how to bet', 'football -MENTIONS- How U.S.
Women’s Soccer Team Lured Emma Hayes Away From Chelsea', 'football -MENTIONS- Pochettino looks past James mistake and stakes future on his ‘Chelsea DNA’',
'football -MENTIONS- Arteta reveals Pochettino influence on his career | Poch: He is family', 'football -MENTIONS- ‘Bleeds green and gold’: Ballsy career-saving
call that’s delivered Arnie a record-breaking moment', 'football -MENTIONS- Thursday Night Football FanDuel Picks: NFL DFS lineup advice for Week 4
Lions-Packers single-game tournaments', 'football -MENTIONS- Who’s the next Gawn, Grundy or Dusty? AFL’s talent guru names top 40 draft prospects and compares
them to current stars', "football -MENTIONS- Cowboys vs. 49ers live score, updates, highlights from NFL 'Sunday Night Football' game", 'football -MENTIONS- Are
NBA Christmas Day Games a Better Tradition Than NFL Thanksgiving Day Games?', "football -MENTIONS- Seahawks vs. Giants live score, updates, highlights from NFL
'Monday Night Football' game", "football -MENTIONS- Bills vs. Bengals live score, updates, highlights from NFL 'Sunday Night Football' game", 'football
-MENTIONS- Taylor Swift reveals secret start to her relationship with Travis Kelce', 'football -MENTIONS- N.F.L. Executives on the Biggest Questions Facing
Super Bowl Contenders', 'football -MENTIONS- Alcoholism, Ayahuasca and the Enlightenment of an N.F.L. Player', 'football -MENTIONS- Ten Hag demands both
positivity and pragmatism from erratic Onana', 'football -MENTIONS- Is the Coverage on Taylor Swift and Travis Kelce’s Romance Nothing but Hype or a Pivotal
Shift in NFL’s Century-Old Game Plan Towards Global Recognition?', 'football -MENTIONS- European roundup: Bellingham helps Real Madrid go top while Bayern
draw', "football -MENTIONS- Michigan didn't need Jim Harbaugh against Penn State, but Ohio State is a different game", 'football -MENTIONS- Point Spread Betting
- What is a Point Spread Bet?', 'football -MENTIONS- Twitch is the best way to watch sports, when you can actually watch sports on Twitch', 'football -MENTIONS-
Patriots vs. Broncos live score, updates, highlights from NFL Sunday night football game', "football -MENTIONS- Bears vs. Panthers live score, updates,
highlights from NFL 'Thursday Night Football' game", 'football -MENTIONS- Argentina vs Paraguay live score, result, updates, highlights from CONMEBOL World Cup
qualifying', 'football -MENTIONS- Ange aghast as Spurs suffer ‘unbelievable’ West Ham turnaround – with two of the worst goals you’ll see', 'football -MENTIONS-
Liverpool enjoy home comforts but Dyche’s Everton offer derby danger | Andy Hunter', 'football -MENTIONS- Totals Betting - What is an Over/Under Bet?',
'football -MENTIONS- Travis Kelce admits the NFL is overdoing it with their Taylor Swift coverage', 'football -SAID- Taylor Swift', 'football -PARTICIPATED_IN-
Travis Kelce', 'USA -MENTIONS- 2023 Kentucky online sports betting sites: Best legal sportsbooks, promos, bonuses, mobile apps, how to bet', "USA -MENTIONS-
Inter Miami vs FC Cincinnati live score, result, updates, highlights with Lionel Messi's MLS playoffs on line", 'USA -MENTIONS- Carabao Cup fixtures, draws,
results, TV channel, and live streams for every round of EFL League Cup 2023/24', 'USA -MENTIONS- Watch Argentina vs Paraguay live stream, TV channel, lineups,
betting odds for World Cup qualifying match', 'USA -MENTIONS- How U.S. Women’s Soccer Team Lured Emma Hayes Away From Chelsea', 'USA -MENTIONS- Man United vs
Chelsea final score, result and highlights as Scott McTominay brace gives Ten Hag vital win', 'USA -MENTIONS- Jamaica vs Canada live score, result, updates,
highlights from CONCACAF Nations League quarterfinal first leg', 'USA -MENTIONS- Where to watch England vs South Africa: Live stream, TV channel, lineups, odds
for 2023 Rugby World Cup semifinal', 'USA -MENTIONS- Major NASCAR Snubs as Autosport Releases the Top 50 Motorsport Drivers in 2023', 'USA -MENTIONS- Is the
Coverage on Taylor Swift and Travis Kelce’s Romance Nothing but Hype or a Pivotal Shift in NFL’s Century-Old Game Plan Towards Global Recognition?', 'USA
-MENTIONS- Europe’s artists and labels could lose $137m annually to US recorded music rightsholders if EU law isn’t changed, trade groups warn', 'USA -MENTIONS-
Breaking down each VAR intervention from wild first half of Tottenham vs Chelsea officiated by Michael Oliver', 'USA -MENTIONS- Is Kevin Durant the Most
Disrespected Superstar in the History of NBA?', 'USA -MENTIONS- What about the Mediterranean lifestyle reduces mortality? Scientists find out', 'USA -MENTIONS-
Argentina vs Paraguay live score, result, updates, highlights from CONMEBOL World Cup qualifying', "USA -MENTIONS- Inter Miami friendly matches for 2024
preseason: Lionel Messi to play Newell's Old Boys, China tour cancelled", 'USA -MENTIONS- Inside the $US2.3b venue that will transform the concert industry',
'USA -MENTIONS- 7 of the best luxury ski resorts for a five-star experience', 'USA -MENTIONS- USA vs Germany player ratings: USMNT defensive errors blight
positive performance by Pulisic, Weah', 'USA -SAID- Emma Hayes', 'USA -SAID- Helen Smith', 'USA -LOCATED_IN- Tennessee', 'USA -LOCATED_IN- Arizona', 'USA
-LOCATED_IN- Harvard Chan School of Public Health', 'USA -LOCATED_IN- Texas', 'USA -LOCATED_IN- Vail', 'USA -LOCATED_IN- Colorado']
</item 9>

<item 10>
Ten Hag’s gamble pays off against Chelsea as McTominay steals show | Jonathan Liew (The Guardian, 2023-12-06T23:45:56+00:00)
But he also pops up in the channels, on the wing, either supporting the man in possession or offering himself for the quick switch of play.

Great times for a great guy. But, of course, if you commit a midfielder to this role, then you leave gaps elsewhere. One of the reasons Chelsea found it so easy
to play through United was that, with the forwards pushed high and McTominay as the secret sauce in the final third, there were open spaces every time they
played through the press. Chelsea’s goal was a case in point: two simple vertical passes straight through the centre via Mykhailo Mudryk to Cole Palmer, passes
that McTominay could see coming but could do precious little to stop.

skip past newsletter promotion Sign up to Football Daily Free daily newsletter Kick off your evenings with the Guardian's take on the world of football Privacy
Notice: Newsletters may contain info about charities, online ads, and content funded by outside parties.

Graph facts:
- Everton players and fans will be fired up by points penalty, asserts Sean Dyche -MENTIONS-> Chelsea
- Carabao Cup fixtures, draws, results, TV channel, and live streams for every round of EFL League Cup 2023/24 -MENTIONS-> Chelsea
- Chelsea secure last-eight spot after Raheem Sterling sinks Blackburn -MENTIONS-> Chelsea
- How U.S. Women’s Soccer Team Lured Emma Hayes Away From Chelsea -MENTIONS-> Chelsea
- Leaky defence ‘isn’t good enough’ and ‘scars’ grow from throwing away leads – It’s all going wrong for Manchester United in Europe -MENTIONS-> Chelsea

Entities: Chelsea, Jonathan Liew, Ten Hag’s gamble pays off against Chelsea as McTominay steals show | Jonathan Liew, The Guardian, United, McTominay, Cole
Palmer, Mykhailo Mudryk, Football Daily, Ten Hag
</item 10>
=== END ITEM id=01ad21e37870a3ff ===
