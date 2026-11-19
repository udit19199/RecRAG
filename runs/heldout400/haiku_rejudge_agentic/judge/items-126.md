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

=== ITEM id=48a9cb4604432248 ===
Gold sentences and facts:
Sentence 0: It would seem that, as the Springboks play for their country, the All Blacks are playing for themselves as much as anyone – and the generation of players and leaders that will call time on their international careers at week’s end.
  Fact 0.0: It would seem that the Springboks are playing for their country.
  Fact 0.1: It would seem that the All Blacks are playing for themselves as much as anyone.
  Fact 0.2: A generation of All Blacks players and leaders will call time on their international careers at week's end.
Sentence 1: The All Blacks had their chances, but their usual safe handling, a feature throughout the World Cup after their first-up loss to France, evaded them against the Springboks’ blitz defence in the wet conditions.
  Fact 1.0: The All Blacks had their chances.
  Fact 1.1: The All Blacks' usual safe handling was a feature throughout the World Cup.
  Fact 1.2: The All Blacks lost their first-up match at the World Cup to France.
  Fact 1.3: The All Blacks' safe handling evaded them against the Springboks' blitz defence.
  Fact 1.4: The conditions were wet.
Sentence 2: It comes despite Argentina knocking over the All Blacks last year in Christchurch, as well as a maiden defeat in 2020 in Sydney.
  Fact 2.0: Argentina knocked over the All Blacks last year in Christchurch.
  Fact 2.1: This comes despite Argentina knocking over the All Blacks last year in Christchurch.
  Fact 2.2: The All Blacks suffered a maiden defeat in 2020 in Sydney.

Retrieved items:
<item 1>
Legacy vs ego: The Springboks are playing for South Africa, the All Blacks are playing for themselves (The Roar | Sports Writers Blog,
2023-10-26T22:19:21+00:00)
“There were a few tears and I think that’s awesome to see, players showing vulnerability, just opening up about how much this tour and this sort of game means.



“There were some (people) saying this is probably one of the most important games of All Blacks history. (The) All Blacks went through a bit of a hard path to
get here, media, friends, family even criticising us and all that, and we stuck together. We weren’t even said to be in the final and now look at us, we’re
here.

Advertisement

“Different players take it differently and some of the older players probably were hit a little bit harder and it was good to see just a bit of emotion and the
want from them. Seeing that gives all the other players an extra two, three per cent to think ‘OK, we are going to the well and we’ve got to give it our all’.”

The All Blacks will experience a changing of the guard next season.

Graph facts:
- THE GREATEST! Boks beat All Blacks in instant classic to claim record FOURTH RWC in controversial, dramatic final -MENTIONS-> Springboks
- Liverpool vs Everton begins epic Saturday with World Cup semi-final as well as Chelsea vs Arsenal – and it could end with a bang -MENTIONS-> Springboks
- Rassie Erasmus -WORKS_FOR-> Springboks
- Jacques Nienaber -WORKS_FOR-> Springboks
- Erasmus -WORKS_FOR-> Springboks

Entities: Springboks, South Africa, Legacy vs ego: The Springboks are playing for South Africa, the All Blacks are playing for themselves, All Blacks, The Roar
</item 1>

<item 2>
‘It happens’: All Blacks defend star centre after spray as Cheika heaps pressure on ‘favourites’ New Zealand (The Roar | Sports Writers Blog,
2023-10-18T20:55:55+00:00)
Advertisement

“In the game in 2020 there were things you could pinpoint. You can’t say that one game means we can do it as there are a lot of games where we haven’t.

“That moment has to be just a part of it. What we have done since then, all the success and failures [matters in our preparation]. The other team are heavy
favourites, everyone is expecting a New Zealand and South Africa final. We just have to focus on what is required.”

Cheika has maintained that his Los Pumas side was too tight earlier in the tournament but has begun to free themselves up since qualifying for the knockout
stages – and rewarding their fans by doing so.

On inspiring the country?

“It’s not easy,” he said.

“We want to inspire not just by our results but by our ambition, by our ability to get back up and overcome obstacles.

Graph facts:
- ‘It happens’: All Blacks defend star centre after spray as Cheika heaps pressure on ‘favourites’ New Zealand -MENTIONS-> Sam Cane
- ‘It happens’: All Blacks defend star centre after spray as Cheika heaps pressure on ‘favourites’ New Zealand -MENTIONS-> Sam Whitelock
- ‘It happens’: All Blacks defend star centre after spray as Cheika heaps pressure on ‘favourites’ New Zealand -MENTIONS-> Brodie Retallick
- ‘It happens’: All Blacks defend star centre after spray as Cheika heaps pressure on ‘favourites’ New Zealand -MENTIONS-> Mark Telea
- ‘It happens’: All Blacks defend star centre after spray as Cheika heaps pressure on ‘favourites’ New Zealand -MENTIONS-> Johnny Sexton

Entities: ‘It happens’: All Blacks defend star centre after spray as Cheika heaps pressure on ‘favourites’ New Zealand, Cheika, Los Pumas, South Africa, All
Blacks, New Zealand, The Roar
</item 2>

<item 3>
Legacy vs ego: The Springboks are playing for South Africa, the All Blacks are playing for themselves (The Roar | Sports Writers Blog,
2023-10-26T22:19:21+00:00)
The All Blacks will experience a changing of the guard next season.

Indeed, the All Blacks’ social media channels described the team selection for the World Cup final as the “Last Dance”.

2015 World Cup winner Dane Coles will call time on his career, while his tight-five teammates Brodie Retallick and Sam Whitelock, who is bidding to become the
first player to win the Webb Ellis Cup for a third time, will head overseas.

Fellow World Cup winners Aaron Smith and Beauden Barrett will also head to Japan.

Coles, who was labelled by Samisoni Taukei’aho as a “Grandad” type figure in the All Blacks, has even been denied the chance of a farewell send-off after
missing selection.

Advertisement

“Yeah, tough,” said coach Ian Foster of the conversation with the 36-year-old around leaving him out.

Graph facts:
- THE GREATEST! Boks beat All Blacks in instant classic to claim record FOURTH RWC in controversial, dramatic final -MENTIONS-> Springboks
- Liverpool vs Everton begins epic Saturday with World Cup semi-final as well as Chelsea vs Arsenal – and it could end with a bang -MENTIONS-> Springboks
- Rassie Erasmus -WORKS_FOR-> Springboks
- Jacques Nienaber -WORKS_FOR-> Springboks
- Erasmus -WORKS_FOR-> Springboks

Entities: Springboks, World Cup, Beauden Barrett, World Cup final, Sam Whitelock, Brodie Retallick, South Africa, Dane Coles, Samisoni Taukei’aho, Legacy vs
ego: The Springboks are playing for South Africa, the All Blacks are playing for themselves, Japan, All Blacks, Ian Foster, Aaron Smith, Webb Ellis Cup
</item 3>

<item 4>
Legacy vs ego: The Springboks are playing for South Africa, the All Blacks are playing for themselves (The Roar | Sports Writers Blog,
2023-10-26T22:19:21+00:00)
It’s been far from smooth, having overseen their historic defeats to Argentina as well as the 2-1 series loss to Ireland on home soil.

A famous 35-23 win over the Springboks at Ellis Park saw him save his career, with Scott Robertson being lined up on the side.

But it wasn’t enough for New Zealand Rugby to back him before this year’s Test season got underway, with Robertson announced as Foster’s successor.

“Am I at peace?” Foster repeated a question during his World Cup team announcement when asked about where his mental state was ahead of his last Test in charge
of the All Blacks.

“Yeah, I’m reasonably peaceful. I love the word but I’m not sure I’m peaceful right now.”

Advertisement

He added: “I’ve just started to get my back nice and straight from last year and now you’re trying to make me hunched over again!

“It’s the life of this business, it’s a tough game.

Graph facts:
- THE GREATEST! Boks beat All Blacks in instant classic to claim record FOURTH RWC in controversial, dramatic final -MENTIONS-> Springboks
- Liverpool vs Everton begins epic Saturday with World Cup semi-final as well as Chelsea vs Arsenal – and it could end with a bang -MENTIONS-> Springboks
- Rassie Erasmus -WORKS_FOR-> Springboks
- Jacques Nienaber -WORKS_FOR-> Springboks
- Erasmus -WORKS_FOR-> Springboks

Entities: Springboks, Scott Robertson, Argentina, Ireland, South Africa, Legacy vs ego: The Springboks are playing for South Africa, the All Blacks are playing
for themselves, historic defeats to Argentina, 2-1 series loss to Ireland, 35-23 win over the Springboks, Ellis Park, Foster, Test season, World Cup team
announcement, last Test, All Blacks
</item 4>

<item 5>
Legacy vs ego: The Springboks are playing for South Africa, the All Blacks are playing for themselves (The Roar | Sports Writers Blog,
2023-10-26T22:19:21+00:00)
The pain of the days and weeks after that dark July, as well as the bumps and bruises along the way, including as recently as August when they suffered their
biggest defeat to the Boks to lose the inaugural and prestigious Qatar Cup, still runs deep.

“It was quite special to hear some of the things those certain players said from the heart. You know, you think you know someone but once they open up and
really be vulnerable in front of a team, it was quite special to hear,” back-rower Dalton Papali’i said this week.



“It was truly something special when you hear those boys open up. You know those leaders have been through a lot in their careers, the highs and the lows, the
off-field and on-field stuff, and for them to open up, it sort of makes you feel welcomed and you want to play this game for them.

Graph facts:
- THE GREATEST! Boks beat All Blacks in instant classic to claim record FOURTH RWC in controversial, dramatic final -MENTIONS-> Springboks
- Liverpool vs Everton begins epic Saturday with World Cup semi-final as well as Chelsea vs Arsenal – and it could end with a bang -MENTIONS-> Springboks
- Rassie Erasmus -WORKS_FOR-> Springboks
- Jacques Nienaber -WORKS_FOR-> Springboks
- Erasmus -WORKS_FOR-> Springboks

Entities: Springboks, South Africa, Dalton Papali’i, Qatar Cup, Legacy vs ego: The Springboks are playing for South Africa, the All Blacks are playing for
themselves, All Blacks, The Roar | Sports Writers Blog
</item 5>

<item 6>
Legacy vs ego: The Springboks are playing for South Africa, the All Blacks are playing for themselves (The Roar | Sports Writers Blog,
2023-10-26T22:19:21+00:00)
Pressure? The South African laughs at the notion of what pressure is compared to the inequality and challenges the Rainbow Nation continues to experience.

Advertisement

The question is: how can New Zealand’s All Blacks withstand and overcome the hitherto unshakable wave of momentum and spirit the Springboks play with under
Erasmus in this weekend’s World Cup final?

It would seem that, as the Springboks play for their country, the All Blacks are playing for themselves as much as anyone – and the generation of players and
leaders that will call time on their international careers at week’s end.

The backlash and toxicity that followed their historic home series loss to Ireland was a humbling and public reminder of the expectation of being an All Black.

It’s why Saturday’s World Cup final is personal for the men wearing black.

All Blacks social media managers were forced to delete a thousand comments an hour following the humbling series loss in Wellington.

Graph facts:
- THE GREATEST! Boks beat All Blacks in instant classic to claim record FOURTH RWC in controversial, dramatic final -MENTIONS-> Springboks
- Liverpool vs Everton begins epic Saturday with World Cup semi-final as well as Chelsea vs Arsenal – and it could end with a bang -MENTIONS-> Springboks
- Rassie Erasmus -WORKS_FOR-> Springboks
- Jacques Nienaber -WORKS_FOR-> Springboks
- Springboks -PARTICIPATED_IN-> World Cup

Entities: Springboks, World Cup final, South Africa, Erasmus, Wellington, historic home series loss to Ireland, challenges, Legacy vs ego: The Springboks are
playing for South Africa, the All Blacks are playing for themselves, inequality, All Blacks, Ireland, New Zealand, The Roar
</item 6>

<item 7>
Legacy vs ego: The Springboks are playing for South Africa, the All Blacks are playing for themselves (The Roar | Sports Writers Blog,
2023-10-26T22:19:21+00:00)
“We explain to the whole squad why we are going a certain route. It is not about individual selection,” Springboks coach Jacques Nienaber said shortly after
announcing two changes to his starting side and five in total, after going in with a forwards-heavy 7-1 bench split.

Advertisement

“It is for South Africa. As selectors we know there will be pressure on us if we don’t deliver a result but at the back of our minds we know this selection is
for South Africa.

“The players take it like that. It is not about person or ego, not that they have egos, but you can’t think of that. You have to think about South Africa and
the Springboks.”

Seemingly the Springboks are playing for more than themselves. They’re playing for their country.

It’s a powerful force, one that saw them progress through the knockout stages to reach the final after consecutive one-point victories.

Pressure?

Graph facts:
- Erasmus -WORKS_FOR-> Springboks
- Liverpool vs Everton begins epic Saturday with World Cup semi-final as well as Chelsea vs Arsenal – and it could end with a bang -MENTIONS-> Springboks
- Rassie Erasmus -WORKS_FOR-> Springboks
- THE GREATEST! Boks beat All Blacks in instant classic to claim record FOURTH RWC in controversial, dramatic final -MENTIONS-> Springboks
- Springboks -PARTICIPATED_IN-> World Cup

Entities: Springboks, Jacques Nienaber, South Africa, final, Legacy vs ego: The Springboks are playing for South Africa, the All Blacks are playing for
themselves, All Blacks, The Roar | Sports Writers Blog
</item 7>

<item 8>
Legacy vs ego: The Springboks are playing for South Africa, the All Blacks are playing for themselves (The Roar | Sports Writers Blog,
2023-10-26T22:19:21+00:00)
Four years ago, Rassie Erasmus delivered one of the great rugby speeches ahead of the 2019 World Cup final.

“You don’t have the right today to worry about your mistakes,” Erasmus said before the Springboks sealed a stunning 32-12 victory over England to claim their
third William Webb Ellis Cup.

“If you worry about your mistakes today, you’ve got ego problems.

Facebook Twitter Whatsapp Reddit Email Share

“You are not representing yourself today. You are fighting for the things that happen in South Africa.

“You are tackling for the boy who didn’t get the same opportunities that you got eventually.”

Four years later and the “ego” Erasmus spoke about is what is driving the standards and expectations around the Springboks’ bid to go back-to-back.

“We explain to the whole squad why we are going a certain route.

Graph facts:
- Jacques Nienaber -WORKS_FOR-> Springboks
- THE GREATEST! Boks beat All Blacks in instant classic to claim record FOURTH RWC in controversial, dramatic final -MENTIONS-> Springboks
- THE GREATEST! Boks beat All Blacks in instant classic to claim record FOURTH RWC in controversial, dramatic final -MENTIONS-> William Webb Ellis Cup
- Liverpool vs Everton begins epic Saturday with World Cup semi-final as well as Chelsea vs Arsenal – and it could end with a bang -MENTIONS-> Springboks
- Erasmus -WORKS_FOR-> Springboks

Entities: William Webb Ellis Cup, Springboks, Rassie Erasmus, ego, South Africa, England, rugby, 2019 World Cup final, Legacy vs ego: The Springboks are playing
for South Africa, the All Blacks are playing for themselves, All Blacks, The Roar | Sports Writers Blog
</item 8>

<item 9>
Legacy vs ego: The Springboks are playing for South Africa, the All Blacks are playing for themselves (The Roar | Sports Writers Blog,
2023-10-26T22:19:21+00:00)
“We don’t want to talk too much about post-game stuff, but that was a tough one, probably the toughest I have had as a coach. He took it the way I expected him
to. He’s a champion.”

Papali’i said the All Blacks were intent on sending their leaders out on a high.

“There’s a few leaders in the team, it’s going to be their last game in the jersey, so it’s a big game for us and even the other players that are staying,
knowing that we want to send those boys out on a high,” he said.

“How much they put into the jersey. We are standing on their…what’s that quote? It’s standing on the shoulders of giants.”

The generational change isn’t just occurring on the field but off it, too.

Advertisement

Foster will depart the All Blacks set up after 12 years, the last four as head coach.

Graph facts:
- THE GREATEST! Boks beat All Blacks in instant classic to claim record FOURTH RWC in controversial, dramatic final -MENTIONS-> Springboks
- Liverpool vs Everton begins epic Saturday with World Cup semi-final as well as Chelsea vs Arsenal – and it could end with a bang -MENTIONS-> Springboks
- Rassie Erasmus -WORKS_FOR-> Springboks
- Jacques Nienaber -WORKS_FOR-> Springboks
- Erasmus -WORKS_FOR-> Springboks

Entities: Springboks, South Africa, Papali’i, generational change, Legacy vs ego: The Springboks are playing for South Africa, the All Blacks are playing for
themselves, Foster, All Blacks, The Roar
</item 9>

<item 10>
Legacy vs ego: The Springboks are playing for South Africa, the All Blacks are playing for themselves (The Roar | Sports Writers Blog,
2023-10-26T22:19:21+00:00)
“It’s the life of this business, it’s a tough game. It’s tough when you’re trying to get your performance right. It’s a tough game when people around you see
things differently, but we’ve learned a lot about ourselves.

“We take massive pride in the jersey and making sure we reach the levels we want to. The rest of that is probably a conversation for another day.

“It hasn’t altered our vision of where we want this team to be. I know we’ve surprised a few people but I don’t think we’ve surprised ourselves.”

Graph facts:
- THE GREATEST! Boks beat All Blacks in instant classic to claim record FOURTH RWC in controversial, dramatic final -MENTIONS-> Springboks
- Liverpool vs Everton begins epic Saturday with World Cup semi-final as well as Chelsea vs Arsenal – and it could end with a bang -MENTIONS-> Springboks
- Rassie Erasmus -WORKS_FOR-> Springboks
- Jacques Nienaber -WORKS_FOR-> Springboks
- Erasmus -WORKS_FOR-> Springboks

Entities: Springboks, South Africa, Legacy vs ego: The Springboks are playing for South Africa, the All Blacks are playing for themselves, All Blacks, The Roar
</item 10>
=== END ITEM id=48a9cb4604432248 ===

=== ITEM id=0cbacf0482e8e904 ===
Gold sentences and facts:
Sentence 0: The Cowboys opened Week 13 by rallying to beat the Seahawks on Thursday night, keeping the heat on the losing Eagles ahead of them before hosting that team in Dallas in Week 14.
  Fact 0.0: The Cowboys opened Week 13 by rallying to beat the Seahawks on Thursday night.
  Fact 0.1: The Cowboys' win kept the heat on the Eagles.
  Fact 0.2: The Eagles were a losing team.
  Fact 0.3: The Eagles were ahead of the Cowboys.
  Fact 0.4: The Cowboys will host the Eagles in Dallas in Week 14.
Sentence 1: Fueled by a stout defensive showing, a gritty run game and a bit of fairy dust from the likes of St. Brown, Sam LaPorta and Josh Reynolds, the Lions cruised to a 34-20 win over their hated NFC North rivals, Green Bay.
  Fact 1.0: The Lions' win was fueled by a stout defensive showing.
  Fact 1.1: The Lions' win was fueled by a gritty run game.
  Fact 1.2: The Lions' win was fueled by a bit of fairy dust from St. Brown.
  Fact 1.3: The Lions' win was fueled by a bit of fairy dust from Sam LaPorta.
  Fact 1.4: The Lions' win was fueled by a bit of fairy dust from Josh Reynolds.
  Fact 1.5: The Lions cruised to a 34-20 win.
  Fact 1.6: The Lions beat Green Bay 34-20.
  Fact 1.7: Green Bay is the Lions' hated NFC North rival.

Retrieved items:
<item 1>
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

Graph facts:
- Christmas Day preview: 49ers, Ravens square off in potential Super Bowl sneak peek -MENTIONS-> Philadelphia Eagles
- Travis Kelce says he tried to delete his ‘nonsense’ tweets before they went viral -MENTIONS-> Philadelphia Eagles
- Taylor Swift-Travis Kelce timeline: Everything to know about rumored romance between pop star, Chiefs TE -MENTIONS-> Philadelphia Eagles
- NFL standings: Updated AFC, NFC playoff picture for Week 13 of 2023 season -MENTIONS-> Philadelphia Eagles
- Dolphins vs. Eagles live score, updates, highlights from NFL 'Sunday Night Football' game -MENTIONS-> Philadelphia Eagles

Entities: Philadelphia Eagles, NFL, NFC, Bills, Bengals, Texans, Super Bowl 57 rematch, NFL power rankings: Lions, 49ers level up; Texans jump Bengals, Bills;
Jets, Patriots sink to bottom for Week 11, Jets, Chiefs, Lions, Sporting News, Patriots, 49ers
</item 1>

<item 2>
N.F.L. Executives on the Biggest Questions Facing Super Bowl Contenders (The New York Times, 2023-10-16T23:58:53+00:00)
But they also haven’t really beaten anyone. They were down 14 points to Carolina. Do you think they are coming back from 13 down against Philly? In my opinion,
no.”

Advertisement

• Dallas Cowboys: Is this team any better than previous Dallas teams that fell short?

What I think: It’s a tough time to evaluate the Cowboys fairly because they were so bad against San Francisco on a big stage in their most recent game. I just
don’t see where Dallas is better or going to get considerably better, especially on offense. And every time they enjoy success, they seem to enjoy it a little
too much. That is a reflection of their owner.

Exec comment: “Run defense and consistency on defense, and then offensively, what is their identity? They just are not very good. Their best games are when they
play great defense and get turnovers, but that is not going to happen enough, especially against the good teams.”

• Detroit Lions: Have they fixed their defense?

Graph facts:
- Which M.L.B. Playoff Team Has the Most Daunting Home Field? We Ranked Them. -PUBLISHED_BY-> The New York Times
- How U.S. Women’s Soccer Team Lured Emma Hayes Away From Chelsea -PUBLISHED_BY-> The New York Times
- After a Stunning Snub, College Football Is Facing an Identity Crisis -PUBLISHED_BY-> The New York Times
- He’s Hockey’s Brightest Young Star. This Is What Makes His Shot So Special. -PUBLISHED_BY-> The New York Times
- Alcoholism, Ayahuasca and the Enlightenment of an N.F.L. Player -PUBLISHED_BY-> The New York Times

Entities: The New York Times, Dallas Cowboys, Carolina, Super Bowl, Detroit Lions, N.F.L., Philly, N.F.L. Executives on the Biggest Questions Facing Super Bowl
Contenders, San Francisco
</item 2>

<item 3>
Prisco's Week 12 NFL picks: Jaguars beat Texans in AFC South battle, Saints get key divisional win vs. Falcons (CBSSports.com, 2023-11-23T21:30:38+00:00)
Dak Prescott plays well again. Cowboys roll.

Pick: Cowboys 32, Commanders 16 (Result: Cowboys 45-10)

Thursday, 8:20 p.m. ET (NBC, fubo)

This is a huge division game for two teams that have had a good rivalry over the past decade. The Seahawks are coming off a bad loss to the Rams, while the
49ers are coming off two dominant victories. Geno Smith is banged up, so his health is key. The 49ers have the look of a Super team again. They show it here.

Pick: 49ers 28, Seahawks 19 (Result: 49ers 31-13)

Friday, 3 p.m. ET (Amazon Prime Video)

This Friday game the day after Thanksgiving isn't what it was expected to be with the Jets being an offensive disaster.

Graph facts:
- Chiefs vs. Packers live score, updates, highlights from NFL 'Sunday Night Football' game -MENTIONS-> Fubo
- Vikings vs. Broncos live score, updates, highlights from NFL 'Sunday Night Football' game -MENTIONS-> Fubo
- Inter Miami vs FC Cincinnati live score, result, updates, highlights with Lionel Messi's MLS playoffs on line -MENTIONS-> Fubo
- Watch Argentina vs Paraguay live stream, TV channel, lineups, betting odds for World Cup qualifying match -MENTIONS-> Fubo
- Raiders vs. Lions live score, updates, highlights from NFL 'Monday Night Football' game -MENTIONS-> Fubo

Entities: Fubo, NFL, Jaguars, Saints, CBSSports.com, Texans, Amazon Prime Video, Jets, Dak Prescott, AFC South, Rams, Geno Smith, Seahawks, NBC, Commanders
</item 3>

<item 4>
Prisco's Week 12 NFL picks: Jaguars beat Texans in AFC South battle, Saints get key divisional win vs. Falcons (CBSSports.com, 2023-11-23T21:30:38+00:00)
Thursday, 12:30 p.m. ET (Fox, fubo)

This is a big game in the division for both teams, a survival test for Green Bay and a big one for the Lions if they want to try and get home-field advantage in
the conference. The Lions have had defensive issues as of late and Jordan Love has played well the past two weeks. That will keep the Packers in this game, but
the Lions win it.

Pick: Lions 31, Packers 27 (Result: Packers 29, Lions 22)

Thursday, 4:30 p.m. ET (CBS, Paramount+)

The Commanders are coming off a dud last week against the Giants, while Dallas is flying high. The Cowboys are playing as well as any team in the league.
Washington's offensive line is an issue, which will lead to the Cowboys having a field day with their defensive front. Dak Prescott plays well again. Cowboys
roll.

Graph facts:
- Chiefs vs. Packers live score, updates, highlights from NFL 'Sunday Night Football' game -MENTIONS-> Fubo
- Vikings vs. Broncos live score, updates, highlights from NFL 'Sunday Night Football' game -MENTIONS-> Fubo
- Inter Miami vs FC Cincinnati live score, result, updates, highlights with Lionel Messi's MLS playoffs on line -MENTIONS-> Fubo
- Watch Argentina vs Paraguay live stream, TV channel, lineups, betting odds for World Cup qualifying match -MENTIONS-> Fubo
- Raiders vs. Lions live score, updates, highlights from NFL 'Monday Night Football' game -MENTIONS-> Fubo

Entities: Fubo, Jordan Love, Jaguars, Saints, Packers, CBSSports.com, Texans, NFL, Lions, Paramount+, Dak Prescott, AFC South, Commanders, Prisco, CBS
</item 4>

<item 5>
Graph query row:
article: 200+ of the best deals from Amazon's Cyber Monday sale
matchup: Cyber Weekend
score: None
publisher: Mashable
page: None
passage: None
</item 5>

<item 6>
Cowboys vs. 49ers live score, updates, highlights from NFL 'Sunday Night Football' game (Sporting News, 2023-10-08T22:47:03+00:00)
The 49ers knocked the Cowboys out of the playoffs in both of the last two seasons, so a win for Dallas on Sunday could be helpful just from a psychological
standpoint.

Graph facts:
- Brock Purdy -MENTIONS-> NFL
- Raiders vs. Lions live score, updates, highlights from NFL 'Monday Night Football' game -MENTIONS-> NFL
- Fans spot Travis Kelce wearing Taylor Swift-themed friendship bracelet before she attended his game -MENTIONS-> NFL
- Chiefs vs. Packers live score, updates, highlights from NFL 'Sunday Night Football' game -MENTIONS-> NFL
- Christmas Day preview: 49ers, Ravens square off in potential Super Bowl sneak peek -MENTIONS-> NFL

Entities: NFL, Sporting News, Cowboys vs. 49ers live score, updates, highlights from NFL 'Sunday Night Football' game, playoffs, Cowboys, NFL 'Sunday Night
Football' game, 49ers
</item 6>

<item 7>
Cowboys vs. 49ers live score, updates, highlights from NFL 'Sunday Night Football' game (Sporting News, 2023-10-08T22:47:03+00:00)
Jerry Jones said earlier in the week that Sunday night's game would show him how the Cowboys stack up against the NFL's best.

The answer? Not very well.

Everything that could go wrong for Dallas on Sunday did go wrong, in a game that confirmed a 49ers team that ended the Cowboys' season in each of the last two
years remains a level above its NFC counterpart.

The 49ers carved up Dallas' defense on the opening drive, and that was a sign of what was to come. Brock Purdy threw four touchdowns in the 42-10 rout,
including three to George Kittle. Christian McCaffrey scored a touchdown for a 14th consecutive game as all of San Francisco's offensive weapons got involved.

MORE: Comparing stats for NFC stars Nick Bosa, Micah Parsons

Defensively, the 49ers were relentless. Nobody had a larger impact than Fred Warner, who finished the night with an interception, sack and forced fumble in a
swarming performance.

Graph facts:
- Christmas Day preview: 49ers, Ravens square off in potential Super Bowl sneak peek -MENTIONS-> Brock Purdy
- NFL standings: Updated AFC, NFC playoff picture for Week 13 of 2023 season -MENTIONS-> Brock Purdy
- Fantasy Football Week 13 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game -MENTIONS-> Brock Purdy
- NFL Fantasy Football Week 5 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game -MENTIONS-> Brock Purdy
- NFL power rankings: Lions, 49ers level up; Texans jump Bengals, Bills; Jets, Patriots sink to bottom for Week 11 -MENTIONS-> Brock Purdy

Entities: Brock Purdy, NFL, Jerry Jones, NFC, Sporting News, Cowboys vs. 49ers game, Nick Bosa, Cowboys vs. 49ers live score, updates, highlights from NFL
'Sunday Night Football' game, Micah Parsons, Cowboys, Christian McCaffrey, 49ers, Fred Warner, George Kittle
</item 7>

<item 8>
Lions vs. Packers live score, updates, highlights from NFL 'Thursday Night Football' game (Sporting News, 2023-09-28T23:54:07+00:00)
NFL WEEK 4 PICKS: Straight up | Against the spread | Projection model

The Sporting News tracked live updates from Thursday's clash between the Lions and the Packers. Scroll below to check out highlights and live results as two of
the NFC's oldest sides faced off for NFC North glory.

Lions vs. Packers score

1 2 3 4 F Lions 14 13 0 7 34 Packers 3 0 8 9 20

Lions vs. Packers live updates, highlights from Thursday Night Football

(All times Eastern)

End of fourth quarter: Lions 34, Packers 20

11:19 p.m.: Lions complete yet another fourth down, officially putting the game on ice.

11:12 p.m.: Onside kick spills out of bounds, and it's looking like that might be that. Packers have one timeout remaining but Montgomery's fiery run through
the tackles should make that obsolete.

11:11 p.m.: FIELD GOAL!

Graph facts:
- Brock Purdy -MENTIONS-> NFL
- Raiders vs. Lions live score, updates, highlights from NFL 'Monday Night Football' game -MENTIONS-> NFL
- Fans spot Travis Kelce wearing Taylor Swift-themed friendship bracelet before she attended his game -MENTIONS-> NFL
- Chiefs vs. Packers live score, updates, highlights from NFL 'Sunday Night Football' game -MENTIONS-> NFL
- Christmas Day preview: 49ers, Ravens square off in potential Super Bowl sneak peek -MENTIONS-> NFL

Entities: NFL, NFC North, Packers, NFC, Lions, Sporting News, Lions vs. Packers game, Montgomery, Lions vs. Packers live score, updates, highlights from NFL
'Thursday Night Football' game
</item 8>

<item 9>
Seahawks vs. Giants live score, updates, highlights from NFL 'Monday Night Football' game (Sporting News, 2023-10-02T22:00:04+00:00)
It was all Seahawks on “Monday Night Football” as the team turned in a dominant defensive effort en route to a 24-3 win over the Giants to wrap up Week 4 of the
NFL season.

In addition to sacking Daniel Jones 11 times, Seattle’s defense forced New York’s QB to commit three turnovers — a fumble and two interceptions, none more
notable than rookie Devon Witherspoon's 97-yard pick-6 that turned the game on its ear.

With the win, the Seahawks move to 3-1 on the season, including a perfect 2-0 record away from home. The Giants drop to 1-3 on the season and have been
outscored 64-3 in two primetime home losses.

The Sporting News tracked score updates and highlights from throughout Monday night's meeting between New York and Seattle.

Seahawks vs.

Graph facts:
- Brock Purdy -MENTIONS-> NFL
- Raiders vs. Lions live score, updates, highlights from NFL 'Monday Night Football' game -MENTIONS-> NFL
- Fans spot Travis Kelce wearing Taylor Swift-themed friendship bracelet before she attended his game -MENTIONS-> NFL
- Chiefs vs. Packers live score, updates, highlights from NFL 'Sunday Night Football' game -MENTIONS-> NFL
- Christmas Day preview: 49ers, Ravens square off in potential Super Bowl sneak peek -MENTIONS-> NFL

Entities: NFL, Monday Night Football game, Sporting News, Daniel Jones, Seahawks, Seahawks vs. Giants live score, updates, highlights from NFL 'Monday Night
Football' game, Devon Witherspoon, Giants
</item 9>

<item 10>
NFL power rankings: Lions, 49ers level up; Texans jump Bengals, Bills; Jets, Patriots sink to bottom for Week 11 (Sporting News, 2023-11-13T21:54:00+00:00)
Their playoff hopes look much better after their stunning comeback vs. the Ravens.

9. Jacksonville Jaguars 6-3 (4)

The Jaguars got an NFC wake-up call against the 49ers, who were much more prepared after a Week 9 bye than they were. Jacksonville needs to forget that debacle
and get back on track, stat. The team gets a division double vs. Tennessee and Houston before facing Cincinnati.

10. Dallas Cowboys 6-3 (11)

The Cowboys took advantage of their "no contest" vs. the Giants at home with Dak Prescott continuing to ball as a passer and runner. They need to keep the
defense and running game success, too, in bigger games going forward.

MORE: Inside the Cowboys historic offensive explosion vs. Giants

11.

Graph facts:
- Brock Purdy -MENTIONS-> NFL
- Raiders vs. Lions live score, updates, highlights from NFL 'Monday Night Football' game -MENTIONS-> NFL
- Fans spot Travis Kelce wearing Taylor Swift-themed friendship bracelet before she attended his game -MENTIONS-> NFL
- Chiefs vs. Packers live score, updates, highlights from NFL 'Sunday Night Football' game -MENTIONS-> NFL
- Christmas Day preview: 49ers, Ravens square off in potential Super Bowl sneak peek -MENTIONS-> NFL

Entities: NFL, Jacksonville Jaguars, Dallas Cowboys, Bills, Houston, Bengals, Cincinnati, playoff hopes, Texans, power rankings, NFC, Week 11, Week 9 bye,
historic offensive explosion, NFL power rankings: Lions, 49ers level up; Texans jump Bengals, Bills; Jets, Patriots sink to bottom for Week 11
</item 10>
=== END ITEM id=0cbacf0482e8e904 ===

=== ITEM id=fb777b4bf3d9d82c ===
Gold sentences and facts:
Sentence 0: 9:47 p.m. — The Bears answer right back and sack Dobbs, with Sweat and Brisker in there to take him down.
  Fact 0.0: At 9:47 p.m., the Bears answered right back.
  Fact 0.1: At 9:47 p.m., the Bears sacked Dobbs.
  Fact 0.2: Sweat and Brisker were in on the sack to take Dobbs down.
Sentence 1: The Ravens defense has allowed the fewest points per game this season (16.1) and second-fewest total yards (4,030), while San Francisco's unit ranks second in points (16.7) and 10th in yardage (310 per game).
  Fact 1.0: The Ravens defense has allowed the fewest points per game this season, 16.1.
  Fact 1.1: The Ravens defense has allowed the second-fewest total yards this season, 4,030.
  Fact 1.2: San Francisco's defense ranks second in points, at 16.7.
  Fact 1.3: San Francisco's defense ranks 10th in yardage, at 310 yards per game.

Retrieved items:
<item 1>
NFL power rankings: Lions, 49ers level up; Texans jump Bengals, Bills; Jets, Patriots sink to bottom for Week 11 (Sporting News, 2023-11-13T21:54:00+00:00)
They can win with offense or defense. Dan Campbell is coach of the year.

4. San Francisco 49ers 6-3 (8)

The 49ers had a surprise offensive get-well game in Jacksonville with Brock Purdy leading the way and all his key supporting weapons doing damage, even without
Christian McCaffrey scoring. The biggest development was the defense going back to dominating a good offense.

5. Baltimore Ravens 7-3 (2)

The Ravens' offense is just fine with the running and passing of Lamar Jackson, but he still can't afford to make big mistakes, which he did vs. the Browns. The
defense needs to pick up the pieces, too, stat, ahead of facing the Bengals on a short week.

6. Miami Dolphins 6-3 (6)

The Dolphins got a much-needed bye to figure out how to beat the better teams in the league.

Graph facts:
- Christmas Day preview: 49ers, Ravens square off in potential Super Bowl sneak peek -MENTIONS-> Brock Purdy
- NFL standings: Updated AFC, NFC playoff picture for Week 13 of 2023 season -MENTIONS-> Brock Purdy
- Fantasy Football Week 13 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game -MENTIONS-> Brock Purdy
- NFL Fantasy Football Week 5 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game -MENTIONS-> Brock Purdy
- Brock Purdy -MENTIONS-> NFL

Entities: Brock Purdy, Lamar Jackson, Bills, Bengals, Texans, Baltimore Ravens, NFL power rankings: Lions, 49ers level up; Texans jump Bengals, Bills; Jets,
Patriots sink to bottom for Week 11, Miami Dolphins, Week 11, NFL, Jets, Lions, Sporting News, Browns, Dan Campbell
</item 1>

<item 2>
Christmas Day preview: 49ers, Ravens square off in potential Super Bowl sneak peek (Yardbarker, 2023-12-24T23:34:39+00:00)
The Ravens defense has allowed the fewest points per game this season (16.1) and second-fewest total yards (4,030), while San Francisco's unit ranks second in
points (16.7) and 10th in yardage (310 per game).

The 49ers defense might be slightly inferior statistically, but it could have more of an impact in this matchup. San Francisco has racked up an NFL-leading 25
turnovers and has given up the second-fewest rushing yards (1,252), which should help slow down the team with the league's most effective running game.

Graph facts:
- Christmas Day preview: 49ers, Ravens square off in potential Super Bowl sneak peek -MENTIONS-> Las Vegas Raiders at Kansas City Chiefs
- Christmas Day preview: 49ers, Ravens square off in potential Super Bowl sneak peek -MENTIONS-> Aidan O’Connell
- Christmas Day preview: 49ers, Ravens square off in potential Super Bowl sneak peek -MENTIONS-> 63-21 win over the Chargers
- Christmas Day preview: 49ers, Ravens square off in potential Super Bowl sneak peek -MENTIONS-> pass defense
- Christmas Day preview: 49ers, Ravens square off in potential Super Bowl sneak peek -MENTIONS-> 49ers, Ravens square off in potential Super Bowl sneak peek

Entities: Christmas Day preview: 49ers, Ravens square off in potential Super Bowl sneak peek, NFL, Ravens, 49ers, Yardbarker
</item 2>

<item 3>
NFL power rankings: Lions, 49ers level up; Texans jump Bengals, Bills; Jets, Patriots sink to bottom for Week 11 (Sporting News, 2023-11-13T21:54:00+00:00)
32. Carolina Panthers 1-8 (31)

The Panthers found better defense and special teams to keep them in the game in Chicago, but Bryce Young still can't unlock a limited passing game and the
offensive line and running game hasn't help nearly enough.

Graph facts:
- NFL Fantasy Football Week 5 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game -MENTIONS-> Chicago
- Browns survive Bears Hail Mary on wild, near-catch by Darnell Mooney: 'Damn near s— myself' -MENTIONS-> Chicago
- Sam Bankman-Fried didn’t ask where the $8 billion went -MENTIONS-> Chicago
- He’s Hockey’s Brightest Young Star. This Is What Makes His Shot So Special. -MENTIONS-> Chicago
- Chicago Fire vs Inter Miami live score, result, updates, highlights from Lionel Messi in MLS match -MENTIONS-> Chicago

Entities: Chicago, Bills, Bengals, Texans, NFL power rankings: Lions, 49ers level up; Texans jump Bengals, Bills; Jets, Patriots sink to bottom for Week 11,
Carolina Panthers, Jets, Lions, Sporting News, Patriots, Bryce Young, 49ers
</item 3>

<item 4>
NFL Fantasy Football Week 5 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-04T21:56:25+00:00)
TOUCHDOWNS: Harris has zero. Somehow it gets worse: He has one carry inside the 15-yard line. The Steelers have run the second-fewest red-zone plays (18, one
more than New England) and second-fewest plays inside the 5-yard line (two, one more than New England). The Steelers have played four games! This is nuts.

RAVENS: Have yet to allow a rushing score to a running back this year. They've been otherwise solid against RBs, holding them to 4.0 yards per carry on the
season and roughly league-average in other metrics including missed tackles (15).
</item 4>

<item 5>
NFL power rankings: Lions, 49ers level up; Texans jump Bengals, Bills; Jets, Patriots sink to bottom for Week 11 (Sporting News, 2023-11-13T21:54:00+00:00)
But the reeling defense and pass protection continue to be un-winnable assets.

24. Atlanta Falcons 4-5 (20)

The Falcons were forced into another QB change because of injury. They found a few more answers offensively but that position still made it an overall mess. The
defense also doesn't seem to be well prepared to different styles of offenses.

25. Green Bay Packers 3-6 (25)

The Packers saw some improved work from Jordan Love and his young weapons as they spread the ball well in the passing game, but the run defense woes and key
holes in coverage have all but doomed their playoff hopes.

26. Los Angeles Rams 3-6 (26)

The Rams are hoping to find a few more answers during a bye, as Sean McVay has seen his good coaching job earlier in the season go by the wayside with key
offensive injuries.

Graph facts:
- Chiefs vs. Packers live score, updates, highlights from NFL 'Sunday Night Football' game -MENTIONS-> Jordan Love
- Fantasy Football Week 13 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game -MENTIONS-> Jordan Love
- NFL Fantasy Football Week 5 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game -MENTIONS-> Jordan Love
- FanDuel Picks Week 15: Best NFL DFS lineup advice for daily fantasy football tournaments -MENTIONS-> Jordan Love
- Thursday Night Football FanDuel Picks: NFL DFS lineup advice for Week 4 Lions-Packers single-game tournaments -MENTIONS-> Jordan Love

Entities: Jordan Love, Green Bay Packers, Bills, Bengals, Texans, NFL power rankings: Lions, 49ers level up; Texans jump Bengals, Bills; Jets, Patriots sink to
bottom for Week 11, Atlanta Falcons, Los Angeles Rams, Jets, Lions, Sporting News, Patriots, 49ers, Sean McVay
</item 5>

<item 6>
NFL power rankings: Lions, 49ers level up; Texans jump Bengals, Bills; Jets, Patriots sink to bottom for Week 11 (Sporting News, 2023-11-13T21:54:00+00:00)
the Patriots. At least Jonathan Taylor is doing his normal things and the defense is playing well again while Gardner Minshew looks meh.

18. Los Angeles Chargers 4-5 (17)

The Chargers got well offensively with Justin Herbert, Keenan Allen and Austin Ekeler against the Lions. But the defense went back to crumbling vs. big pass
plays and added wilting against the run. The pressure is right back on Brandon Staley with a trip to Green Bay next.

20. Tampa Bay Buccaneers 4-5 (23)

Baker Mayfield and the offense is back on track to move the ball well after a brief midseason slump. They're still not lighting it up, but the Bucs are back to
playing smart complementary football with their defense rebounding and still might have found a long-term QB answer.

21.

Graph facts:
- Brock Purdy -MENTIONS-> NFL
- Raiders vs. Lions live score, updates, highlights from NFL 'Monday Night Football' game -MENTIONS-> NFL
- Fans spot Travis Kelce wearing Taylor Swift-themed friendship bracelet before she attended his game -MENTIONS-> NFL
- Chiefs vs. Packers live score, updates, highlights from NFL 'Sunday Night Football' game -MENTIONS-> NFL
- Christmas Day preview: 49ers, Ravens square off in potential Super Bowl sneak peek -MENTIONS-> NFL

Entities: NFL, Bills, Bengals, Texans, NFL power rankings: Lions, 49ers level up; Texans jump Bengals, Bills; Jets, Patriots sink to bottom for Week 11, Green
Bay, Justin Herbert, Week 11, Jets, Lions, Sporting News, Austin Ekeler, Gardner Minshew, Los Angeles Chargers, Brandon Staley
</item 6>

<item 7>
NFL Fantasy Football Week 5 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-04T21:56:25+00:00)
Only Joshua Dobbs, who ran for 55 yards, has scored double-digit Fantasy points against the Niners (18 points). Not that the Cowboys have been tested much --
Daniel Jones, Zach Wilson and the Patriots QBs are the other guys they've man-handled.

O-LINE: San Francisco has allowed a pass rush pressure on 36.4% of their snaps, which sounds like a lot but it's just a shade worse than league average. That
number is weighted by the Giants' decision to blitz on 84.6% of Purdy's dropbacks in Week 2, creating an inflated pass rush pressure rate. Despite the Giants'
efforts, Purdy still threw for 310 yards and two touchdowns in a game he didn't have Brandon Aiyuk in.

COWBOYS: Have become one of the most aggressive defenses in the league, ranking third from the bottom in zone coverage snaps.

Graph facts:
- Vikings vs. Broncos live score, updates, highlights from NFL 'Sunday Night Football' game -MENTIONS-> Joshua Dobbs
- Bears vs. Vikings live score, updates, highlights from NFL 'Monday Night Football' game -MENTIONS-> Joshua Dobbs
- 2024 NFL Mock Draft: Bengals' Joe Burrow gets another playmaker; four QBs go in Round 1, but Pats opt for WR -MENTIONS-> Joshua Dobbs
- Joshua Dobbs -WORKS_FOR-> Vikings
- Joshua Dobbs -PARTICIPATED_IN-> Vikings vs. Broncos game

Entities: Joshua Dobbs, NFL Fantasy Football Week 5 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game, Niners, Week 2, CBSSports.com,
fantasy football, Zach Wilson, Purdy, Daniel Jones, Cowboys, Giants, Patriots, Brandon Aiyuk
</item 7>

<item 8>
Graph query row:
article: Christmas Day preview: 49ers, Ravens square off in potential Super Bowl sneak peek
publisher: Yardbarker
passage: Christmas Day preview: 49ers, Ravens square off in potential Super Bowl sneak peek (Yardbarker, 2023-12-24T23:34:39+00:00)
Christmas Day isn't just for the NBA, as the NFL is set to air an exciting triple-header featuring fierce division clashes and a highly anticipated matchup we
could see again on Super Bowl Sunday. Here are the top players and storylines to monitor for Monday's slate.

Las Vegas Raiders (6-8) at Kansas City Chiefs (9-5) | 1 p.m. ET

New chapter in Patrick Mahomes-Maxx Crosby rivalry: This hasn't been as much of a rivalry between the teams as of late, with Kansas City holding a 10-1 record
over the Raiders since Mahomes became its starting quarterback. Yet it has sparked an exciting feud between the two-time Super Bowl MVP and the two-time Pro
Bowler Crosby, who has five sacks in nine career games against the Chiefs.

Mahomes and Crosby might not like facing one another, though it's clear there's mutual respect between the two, which only makes their rivalry more intriguing
to watch.

Can Raiders QB Aidan O’Connell earn next season’s starting job?
</item 8>

<item 9>
N.F.L. Executives on the Biggest Questions Facing Super Bowl Contenders (The New York Times, 2023-10-16T23:58:53+00:00)
• Baltimore Ravens: Are the Ravens better off shifting to more of a pass-oriented offense?

What I think: Baltimore has never scored fewer points or amassed less offensive EPA through the first six games of a season with Lamar Jackson in the lineup.
The schedule and injuries have played roles, but with the Ravens shifting toward a more conventional pass offense, are the explosive quarterback runs mostly in
the past? Jackson has one rush longer than 20 yards (he’s had four or five by now previously). He’s gaining at least eight yards on 15 percent of carries, half
his previous rate. I’m not expecting a consistent precision passing game to suddenly materialize. Will this new offense be tougher to defend?

Advertisement

Exec comment: “The defense keeps them in it, but I don’t necessarily think this offensive change is going to make them any better when it counts. Lamar Jackson
does not appear as dynamic as a runner, but in this offense, he does not need to be.

Graph facts:
- NFL Fantasy Football Week 5 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game -MENTIONS-> Lamar Jackson
- 2023 Kentucky online sports betting sites: Best legal sportsbooks, promos, bonuses, mobile apps, how to bet -MENTIONS-> Lamar Jackson
- NFL power rankings: Lions, 49ers level up; Texans jump Bengals, Bills; Jets, Patriots sink to bottom for Week 11 -MENTIONS-> Lamar Jackson
- NFL Fantasy Football Week 6 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game -MENTIONS-> Lamar Jackson
- Christmas Day preview: 49ers, Ravens square off in potential Super Bowl sneak peek -MENTIONS-> Lamar Jackson

Entities: Lamar Jackson, The New York Times, Baltimore Ravens, Super Bowl, N.F.L., N.F.L. Executives on the Biggest Questions Facing Super Bowl Contenders
</item 9>

<item 10>
Bears vs. Vikings live score, updates, highlights from NFL 'Monday Night Football' game (Sporting News, 2023-11-27T23:32:04+00:00)
The honeymoon period is over for Joshua Dobbs in Minnesota.

The folk hero quarterback threw four interceptions in the Vikings' 12-10 "Monday Night Football" loss to the Bears, leading a sluggish offense that didn't find
its footing until the fourth quarter in a game that Minnesota wouldn't have needed many points to win.

The Bears' defense swarmed Dobbs throughout the night, sacking him only twice but bringing a great deal of pressure as the defensive backs made plays. Despite
the barrage of turnovers, Chicago largely failed to take advantage. The Bears scored only three points off of the four Dobbs interceptions.

MORE: How Joshua Dobbs got his 'Passtronaut' nickname

Meanwhile, Chicago's first turnover of the night — a deflating fumble by Fields when the Bears had a chance to extend their lead to two possessions — was
quickly turned into a touchdown by the Vikings.

Graph facts:
- Brock Purdy -MENTIONS-> NFL
- Raiders vs. Lions live score, updates, highlights from NFL 'Monday Night Football' game -MENTIONS-> NFL
- Fans spot Travis Kelce wearing Taylor Swift-themed friendship bracelet before she attended his game -MENTIONS-> NFL
- Chiefs vs. Packers live score, updates, highlights from NFL 'Sunday Night Football' game -MENTIONS-> NFL
- Christmas Day preview: 49ers, Ravens square off in potential Super Bowl sneak peek -MENTIONS-> NFL

Entities: NFL, Vikings, Joshua Dobbs, Chicago, Minnesota, Fields, Bears vs. Vikings 'Monday Night Football' game, Bears vs. Vikings live score, updates,
highlights from NFL 'Monday Night Football' game, Sporting News, Bears
</item 10>
=== END ITEM id=fb777b4bf3d9d82c ===
