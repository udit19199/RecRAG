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

This file has 11 items.

=== ITEM id=6dac9dfc93769920 ===
Gold sentences and facts:
Sentence 0: Jelena Janković (Serbian Cyrillic: Јелена Јанковић, ] , born 28 February 1985) is a Serbian professional tennis player.
  Fact 0.0: Jelena Janković was born on 28 February 1985.
  Fact 0.1: Jelena Janković's name is written in Serbian Cyrillic as Јелена Јанковић.
  Fact 0.2: Jelena Janković is a Serbian professional tennis player.
Sentence 1: Marat Mubinovich Safin (Russian: Мара́т Муби́нович Са́фин ; ] , Tatar: Марат Мөбин улы Сафин "Marat Möbin uğlı Safin" ; born 27 January 1980) is a Russian politician and retired professional tennis player.
  Fact 1.0: Marat Mubinovich Safin's name is written in Russian as Мара́т Муби́нович Са́фин.
  Fact 1.1: Marat Mubinovich Safin's name is written in Tatar as Марат Мөбин улы Сафин.
  Fact 1.2: Marat Mubinovich Safin's Tatar name is romanized as Marat Möbin uğlı Safin.
  Fact 1.3: Marat Mubinovich Safin was born on 27 January 1980.
  Fact 1.4: Marat Mubinovich Safin is a Russian politician.
  Fact 1.5: Marat Mubinovich Safin is a retired professional tennis player.

Retrieved items:
<item 1>
Jelena Janković
Jelena Janković (Serbian Cyrillic: Јелена Јанковић, ] , born 28 February 1985) is a Serbian professional tennis player.

 Janković is a former World No. 1 in singles, a ranking achieved preceding her finals appearance at the 2008 US Open.

 Janković's career highlights include winning the 2007 Wimbledon Mixed Doubles title with Jamie Murray, the 2010 Indian Wells Masters, the Internazionali BNL
d'Italia twice, in 2007 and 2008, and the aforementioned appearance in the finals of the 2008 US Open.

 She is coached by her brother Marko.
</item 1>

<item 2>
Ana Ivanovic career statistics
This is a list of the main career statistics of Serbian professional tennis player, Ana Ivanovic.

 Ivanovic won fifteen WTA singles titles including one grand slam singles title at the 2008 French Open and three WTA Tier I singles titles.

 She was also the runner-up at the 2007 French Open and 2008 Australian Open and a semi-finalist at the 2007 Wimbledon Championships and 2007 WTA Tour
Championships.

 On June 9, 2008 Ivanovic became the world No. 1 for the first time in her career.
</item 2>

<item 3>
Jorge Lozano
Jorge Lozano (born 17 May 1963 in San Luis Potosí) is a retired professional tennis player from Mexico.

 He reached his highest doubles ranking of World No. 4 in August 1988.

 His highest singles ranking was World No. 51, achieved the following month.

 During his career, he won two mixed doubles titles at the French Open: in 1990 with Arantxa Sánchez, and in 1988 with Lori McNeil.

 He reached the round of 16 in singles at the 1988 US Open and also the semifinal in doubles that same year.

 Qualified twice for the Doubles Masters at the Royal Albert Hall in London and reached the semifinals in 1988.

 In his career, he won nine doubles titles, but no singles titles.

 He turned professional in 1986, and in his career, he won $U.S.739,424 in prize money.

 He was the first player to be beaten by Pete Sampras in the main draw of a Grand Slam tournament, at the French Open in 1989.
</item 3>

<item 4>
Ivanovic–Janković rivalry
Serbian tennis players Ana Ivanovic and Jelena Janković are both former world No. 1 and have met 12 times so far.

 Their overall head-to-head series is 9–3, in Ivanovic's favor.

 Janković is two years older than Ivanovic, so their paths connect them to play each other.

 They are the first female players to achieve major successes for independent Serbia.
</item 4>

<item 5>
Jelena Janković career statistics
This is a list of the main career statistics of Serbian professional tennis player and former World No. 1, Jelena Janković.

 To date, Janković has won 15 WTA singles titles including four WTA Tier I singles titles, one WTA Premier Mandatory singles title and one WTA Premier 5 singles
title.

 Janković was also the runner-up in singles at the 2008 US Open, a semi-finalist at the year-ending WTA Tour Championships in 2008, 2009 and 2013 and a
quarterfinalist at the 2008 Beijing Olympics.
</item 5>

<item 6>
2014 Rogers Cup – Women's Doubles
Jelena Janković and Katarina Srebotnik were the defending champions, but they chose not to compete together.

 Janković played alongside Klára Koukalová, but lost in the second round to Michaëlla Krajicek and Barbora Záhlavová-Strýcová.

 Srebotnik teamed up with Květa Peschke, but lost in the second round to Chan Hao-ching and Chan Yung-jan.

<br>
</item 6>

<item 7>
Marat Safin
Marat Mubinovich Safin (Russian: Мара́т Муби́нович Са́фин ; ] , Tatar: Марат Мөбин улы Сафин "Marat Möbin uğlı Safin" ; born 27 January 1980) is a Russian
politician and retired professional tennis player.

 He achieved the Association of Tennis Professionals (ATP) world No. 1 singles ranking on November 20, 2000.

 Nobody taller than Safin has ever reached his world number-one ranking.

 Safin is the older brother of former world No. 1 WTA player, Dinara Safina.

 They are the first and only brother–sister tandem in tennis history who have both achieved No. 1 rankings.

 On court, Safin was famous for his emotional outbursts and sometimes fiery temper.
</item 7>

<item 8>
Paola Suárez
Paola Suárez (born 23 June 1976) is a retired professional tennis player from Argentina.

 She was one of the most prominent women's doubles players throughout the early and mid-2000s, winning eight Grand Slam titles, all of them with Virginia Ruano
Pascual, and holding the No. 1 doubles ranking for 87 non-consecutive weeks.

 She was also a singles semifinalist at the 2004 French Open.
</item 8>

<item 9>
Dinara Safina career statistics
This is a list of the main career statistics of retired, Russian professional tennis player, Dinara Safina.

 Throughout her career, Safina won twelve WTA singles titles including three Tier I singles titles at the 2008 Qatar Telecom German Open, Rogers Cup and Toray
Pan Pacific Open respectively; one Premier Mandatory singles title at the 2009 Mutua Madrileña Madrid Open and one Premier 5 singles title at the 2009
Internazionali BNL d'Italia.

 She was also the runner-up at the 2008 French Open and the 2009 Australian Open and French Open as well as a silver medalist in singles at the 2008 Beijing
Olympics.
</item 9>

<item 10>
Hicham Arazi
During his career, he notably gained victories over former World No. 1s and Grand Slam champions Roger Federer, Andre Agassi, Yevgeny Kafelnikov, Marat Safin,
Lleyton Hewitt, Juan Carlos Ferrero, Carlos Moyá and Jim Courier.
</item 10>
=== END ITEM id=6dac9dfc93769920 ===

=== ITEM id=4dee6233e28e0985 ===
Gold sentences and facts:
Sentence 0: Elizabeth Austin (born 1958) is an American writer living in Oak Park, Illinois.
  Fact 0.0: Elizabeth Austin was born in 1958.
  Fact 0.1: Elizabeth Austin is an American writer.
  Fact 0.2: Elizabeth Austin lives in Oak Park, Illinois.
Sentence 1:  She has also written for "The Washington Monthly, Ladies Home Journal, Good Housekeeping, Self", and "Shape".
  Fact 1.0: Elizabeth Austin has also written for The Washington Monthly.
  Fact 1.1: Elizabeth Austin has also written for Ladies Home Journal.
  Fact 1.2: Elizabeth Austin has also written for Good Housekeeping.
  Fact 1.3: Elizabeth Austin has also written for Self.
  Fact 1.4: Elizabeth Austin has also written for Shape.
Sentence 2: Shape is a monthly English language fitness magazine started by Weider Publications in 1981, founded by Christine MacIntyre (a pioneer in women's free weight fitness) and became the number one women's fitness magazine.
  Fact 2.0: Shape is a monthly English language fitness magazine.
  Fact 2.1: Shape was started by Weider Publications in 1981.
  Fact 2.2: Shape was founded by Christine MacIntyre.
  Fact 2.3: Christine MacIntyre is a pioneer in women's free weight fitness.
  Fact 2.4: Shape became the number one women's fitness magazine.

Retrieved items:
<item 1>
Wendy Austin
Dr Wendy Elizabeth Austin Hewitt, Wendy Austin is a BBC journalist and broadcaster in BBC Northern Ireland, and former presenter of BBC Radio Ulster’s current
affairs programme "Talkback".
</item 1>

<item 2>
Memphis Sport
Memphis Sport is a sports and fitness magazine featuring articles on local and regional teams, players and events.

 In addition, "Memphis Sport" focuses on health and fitness for an active lifestyle.

 The magazine debuted in Memphis, Tennessee in July 2006.

 Local franchises covered include the University of Memphis Tigers, Memphis Redbirds, Mississippi Riverkings and Tennessee Titans.
</item 2>

<item 3>
Anais Zanotti
Anais Zanotti (born January 8, 1986 in Saint-Tropez) is a French model and stunt woman of French and Italian ancestry who lives in Miami, Florida.

 She has been featured in magazines such as "Playboy", "FHM", "Maxim", "GQ", "Cosmopolitan", "Vogue", and "People", Men's Fitness magazine, Muscle and Fitness,
Women Fitness magazine and more.

 She has over 40 magazine covers.

 Zanotti is also a certified skydiver with over 1250 jumps.GS .
</item 3>

<item 4>
Football World
Football World, later renamed "Athletic World", was an American magazine devoted to the coverage of inter-collegiate sports.

 Its masthead described it as "A Magazine With a Mission to Serve the College Man," a publication "devoted to Inter-collegiate Athletics and sports of Amateur
standing only."

 It was founded in 1921 by J. D. Fetzer.

 The name of the magazine was later changed to "Athletic World" as the coverage extended to a broader range of sports, including women's swimming.

 Unlike other sports magazines of the era, which focused on promoting a healthy lifestyle, "Football World/Athletic World" celebrated the entertainment value of
sports with a special emphasis on the personalities of famous athletes.

 The magazine was renamed "Outing" in December 1924, reflecting a change in its focus.
</item 4>

<item 5>
Elle (magazine)
Elle is a worldwide lifestyle magazine of French origin that focuses on fashion, beauty, health, and entertainment.

 "Elle" is also the world's best-selling fashion magazine.

 It was founded by Pierre Lazareff and his wife Hélène Gordon in 1945.

 The title, in French, means "she" or "her".
</item 5>

<item 6>
Essence (magazine)
Essence is a monthly magazine for African American women between the ages of 18 and 49.

 It is the only magazine that focuses on reaching an audience of black women, revolves around the black woman experience, and has remained for a long period of
time.

 The magazine covers fashion, lifestyle and beauty, with an intimate girlfriend-to-girlfriend tone, and its slogan "Fierce, Fun, and Fabulous" suggests the
magazine's goal of empowering African-American women.

 The topics the magazine discusses range from celebrities, to fashion, to point-of-view pieces addressing current issues in the African-American community.

 A number of its readers engage closely and personally with the publication, and it claims to be the magazine "for and about Black women".
</item 6>

<item 7>
Shape (magazine)
Shape is a monthly English language fitness magazine started by Weider Publications in 1981, founded by Christine MacIntyre (a pioneer in women's free weight
fitness) and became the number one women's fitness magazine.

 At that time, Weider Enterprises consisted primarily of the bodybuilding magazine "Muscle & Fitness".

 Joe Weider and Christine MacIntyre had differing views of how to present "Shape", Weider endorsing a less journalistic and more commercial approach to
articles, MacIntyre endorsing a more academic, doctor-based magazine.

 Weider also endorsed a sexier approach to editorial while MacIntyre endorsed a healthier look for women, eschewing sexiness in the models and the copy.

 MacIntyre largely won that battle, editing a magazine that required that every byline have an advanced medical degree, that cover models should look healthy
rather than sexy, and that sexist language be avoided.

 Christine MacIntyre was the editor-in-chief until her death in 1988.

 Tara Kraft is the current editor-in-chief.

 "Shape" found a readership based on that formula.
</item 7>

<item 8>
Elizabeth Austin (writer)
Elizabeth Austin (born 1958) is an American writer living in Oak Park, Illinois.

 Austin has lived in the Midwestern United States all her life.

 She began her journalism career at the now-defunct City News Bureau of Chicago and has written articles for national news magazines such as "Newsweek" and
"Time", in addition to a wide range of other national and regional publications, such as "People", and the "Chicago Tribune".

 She has also written for "The Washington Monthly, Ladies Home Journal, Good Housekeeping, Self", and "Shape".

 She has often written about medicine and health care.

 Austin also has been heard on WBEZ, the National Public Radio station in Chicago, as well as the BBC in the UK and Scotland.

 She has appeared on the Today Show and on ITV in Europe.
</item 8>

<item 9>
Muscle &amp; Fitness (British magazine)
Muscle & Fitness is a British fitness magazine, which contains articles on strength and fitness training, diet plans and strategies for men and women, and tips
and advice.
</item 9>

<item 10>
Ingrid Romero
Ingrid Romero (born July 10, 1985) is an international fitness model, fitness competitor, and fitness coach.

 Training & Fitness Magazine called her "one of the most recognized faces in the fitness industry," and "one of the most accomplished female fitness stars
ever."
</item 10>
=== END ITEM id=4dee6233e28e0985 ===

=== ITEM id=c29a34a5eee9a48b ===
Gold sentences and facts:
Sentence 0:  After the Blackhawks jumped to an early 4–1 lead in the first game of the series, Mario Lemieux and the Penguins came back to win the game, sweep the series in four games, and win their second consecutive and second overall Stanley Cup.
  Fact 0.0: The Blackhawks jumped to an early 4–1 lead in the first game of the series.
  Fact 0.1: Mario Lemieux and the Penguins came back to win the first game of the series.
  Fact 0.2: Mario Lemieux and the Penguins swept the series in four games.
  Fact 0.3: The Penguins won their second consecutive Stanley Cup.
  Fact 0.4: The Penguins won their second overall Stanley Cup.
Sentence 1: Mario Lemieux, OC, CQ ( ; ] ; born October 5, 1965) is a Canadian former professional ice hockey player and current owner of the Pittsburgh Penguins team.
  Fact 1.0: Mario Lemieux was born on October 5, 1965.
  Fact 1.1: Mario Lemieux has the honorifics OC and CQ.
  Fact 1.2: Mario Lemieux is a Canadian.
  Fact 1.3: Mario Lemieux is a former professional ice hockey player.
  Fact 1.4: Mario Lemieux is the current owner of the Pittsburgh Penguins team.

Retrieved items:
<item 1>
History of the National Hockey League
The history of the National Hockey League begins with the end of its predecessor league, the National Hockey Association (NHA), in 1917.

 After unsuccessfully attempting to resolve disputes with Eddie Livingstone, owner of the Toronto Blueshirts, executives of the three other NHA franchises
suspended the NHA, and formed the National Hockey League (NHL), replacing the Livingstone team with a temporary team in Toronto, the Arenas.

 The NHL's first quarter-century saw the league compete against two rival major leagues—the Pacific Coast Hockey Association and Western Canada Hockey
League—for players and the Stanley Cup.

 The NHL first expanded into the United States in 1924 with the founding of the Boston Bruins, and by 1926 consisted of ten teams in Ontario, Quebec, the Great
Lakes region, and the Northeastern United States.

 At the same time, the NHL emerged as the only major league and the sole competitor for the Stanley Cup; in 1947, the NHL completed a deal with the Stanley Cup
trustees to gain full control of the Cup.

 The NHL's footprint spread across Canada as Foster Hewitt's radio broadcasts were heard coast-to-coast starting in 1933.
</item 1>

<item 2>
1992 Spengler Cup
The 1992 Spengler Cup was held in Davos, Switzerland, between December 26, 1992, and December 31, 1992.

 All matches were played at HC Davos's home arena, Eisstadion Davos.

 The final was won 6-5 by Team Canada over Färjestad BK.
</item 2>

<item 3>
1984 Stanley Cup Finals
The 1984 Stanley Cup Final was held between the Edmonton Oilers and the then-defending champion New York Islanders.

 The Islanders had swept the Oilers in four straight games to win the Cup.

 In 1984, the Islanders were seeking their fifth consecutive Stanley Cup championship, but the upstart Oilers would win the best-of-seven series four games to
one to win their first Stanley Cup, becoming the third post-1967 expansion team and first former World Hockey Association team to win the Cup, and also the
first team based west of Chicago to win the Cup since the WCHL's Victoria Cougars became the last non-NHL team to win it in .

 It was also the fifth straight Finals of teams that joined the NHL in 1967 or later and a rematch of the 1983 Finals—a Stanley Cup Finals rematch would not
happen again until the Finals.

 s of 2017 , the Islanders' four consecutive Cup wins (, , , 1983) and their appearance in the 1984 Cup Finals is an NHL record of 19 consecutive playoff series
wins that currently stands unbroken.
</item 3>

<item 4>
1983 Stanley Cup Finals
The 1983 Stanley Cup Finals was contested by the Edmonton Oilers in their first-ever Finals appearance and the defending champion New York Islanders, in their
fourth, and fourth consecutive, Finals appearance.

 The Islanders would win the best-of-seven series four games to none, to win their fourth-straight and fourth-overall Stanley Cup.

 It was also the fourth straight Finals of post-1967 expansion teams, and the first involving a former World Hockey Association (WHA) team.

 This is also the most recent time that a defending Stanley Cup champion has won the cup four years in a row, and also the first (and, to date, only) time a
North American professional sports team has won four consecutive titles in any league competition with more than twenty teams.

 Since 1983, no professional sports team on the continent has managed to win four straight championships and no NHL team has won more than two consecutive
championships (most recently the Pittsburgh Penguins in and ).
</item 4>

<item 5>
1998 Stanley Cup Finals
The 1998 Stanley Cup Finals was the championship series of the National Hockey League's (NHL) 1997–1998 season, and the culmination of the 1998 Stanley Cup
playoffs.

 It was contested by the Western Conference champion and defending Stanley Cup champion Detroit Red Wings and the Eastern Conference champion Washington
Capitals.

 It was the 105th year of the Stanley Cup being contested.

 The series was the Capitals' first appearance in a Stanley Cup Final since the franchise's inception in 1974.

 The Red Wings won the series for the second year in a row, four games to none.

 It was the Wings' ninth Stanley Cup, and the most recent time when a Finals concluded with a sweep (as of 2017).

 This was also the last time until that a Stanley Cup Finals ended after an NBA Finals in the same season had concluded.
</item 5>

<item 6>
1917–18 Toronto Hockey Club season
The 1917–18 Toronto Hockey Club season was the first season of the new "Toronto" franchise in the newly organized National Hockey League (NHL).

 The team was intended as a 'temporary' franchise, operating without an official club nickname (the press would dub them the "Blue Shirts" or "Torontos", and in
1948 the NHL would engrave "Toronto Arenas" on the Stanley Cup as the 1917–18 winner) and without a formal organization separate from the Toronto Arena Company
that managed the Arena Gardens.

 Despite this, the team came together to win the first NHL Championship, competing against existing teams that had transferred directly from the National Hockey
Association (NHA).

 Toronto would go on to win the Stanley Cup by defeating the Pacific Coast Hockey Association champion Vancouver Millionaires – the first Stanley Cup for an NHL
team and the second Cup for a Toronto team after the Toronto Blueshirts' victory in the 1913–14 season of the NHA.
</item 6>

<item 7>
1992 Stanley Cup Finals
The 1992 Stanley Cup Finals NHL championship series was contested by the Prince of Wales Conference and defending Stanley Cup champion Pittsburgh Penguins and
the Clarence Campbell Conference champion Chicago Blackhawks.

 The Blackhawks were appearing in their first Finals since .

 After the Blackhawks jumped to an early 4–1 lead in the first game of the series, Mario Lemieux and the Penguins came back to win the game, sweep the series in
four games, and win their second consecutive and second overall Stanley Cup.

 It was the 99th year of the Stanley Cup, and the first to extend into the month of June.

 It was the last final for Chicago Stadium as it closed in 1994.
</item 7>

<item 8>
1984 Stanley Cup Finals
This would be the second of eight consecutive Finals contested by a team from Alberta (the Oilers appeared in six, the Calgary Flames in two), and the first of
five consecutive Finals to end with the Cup presentation on Alberta ice (the Oilers won four times, the Montreal Canadiens one).
</item 8>

<item 9>
Mario Lemieux
Mario Lemieux, OC, CQ ( ; ] ; born October 5, 1965) is a Canadian former professional ice hockey player and current owner of the Pittsburgh Penguins team.

 He played parts of 17 seasons in the National Hockey League's (NHL) with the Pittsburgh Penguins, between 1984 and 2006.

 Dubbed "The Magnificent One" or "Le Magnifique" (as well as "Super Mario"), he is widely acknowledged to have been one of the best players of all time.

 He is the only player to score one goal in each of the five possible situations in a single NHL game, a feat he accomplished in 1988.

 A gifted playmaker and fast skater despite his large size, Lemieux often beat defencemen with fakes and dekes.
</item 9>

<item 10>
Graph query row:
person: Mario Lemieux
team_relationship: MEMBER_OF
team: Pittsburgh Penguins
event: 1992 Stanley Cup Finals
held_on: 1992
team_event_relationship: None
person_event_relationship: None
</item 10>
=== END ITEM id=c29a34a5eee9a48b ===

=== ITEM id=9f86a281a83bef6f ===
Gold sentences and facts:
Sentence 0:  Game included wild boar, and red, fallow and roe deer.
  Fact 0.0: Game included wild boar.
  Fact 0.1: Game included red deer.
  Fact 0.2: Game included fallow deer.
  Fact 0.3: Game included roe deer.
Sentence 1:  This common species is native to western Eurasia, but has been introduced to Antigua & Barbuda, Argentina, South Africa, Fernando Pó, São Tomé, Madagascar, Mauritius, Mayotte, Réunion, Seychelles, Comoro Islands, Morocco, Algeria, Tunisia, Cyprus, Israel, Cape Verde, Lebanon, Australia, New Zealand, Canada, the United States, the Falkland Islands, and Peru.
  Fact 1.0: The common species is native to western Eurasia.
  Fact 1.1: The common species has been introduced to Antigua & Barbuda, Argentina, South Africa, Fernando Pó, São Tomé, Madagascar, Mauritius, Mayotte, Réunion, Seychelles, Comoro Islands, Morocco, Algeria, Tunisia, Cyprus, Israel, Cape Verde, Lebanon, Australia, New Zealand, Canada, the United States, the Falkland Islands, and Peru.

Retrieved items:
<item 1>
Visayan spotted deer
The Visayan spotted deer ("Rusa alfredi"), also known as the Philippine spotted deer, is a nocturnal and endangered species of deer located primarily in the
rainforests of the Visayan islands of Panay and Negros though it once roamed other islands such as Cebu, Guimaras, Leyte, Masbate, and Samar.

 It is one of three endemic deer species in the Philippines, although it was not recognized as a separate species until 1983.

 An estimated 2,500 mature individuals survived worldwide as of 1996, according to the IUCN, although it is uncertain of how many of them still survive in the
wild.

 The diet of the deer, which consists of a variety of different types of grasses, leaves, and buds within the forest, is the primary indicator of its habitat.

 Since 1991 the range of the species has severely decreased and is now almost co-extensive with that of the Visayan warty pig.
</item 1>

<item 2>
Fallow deer
The fallow deer ("Dama dama") is a ruminant mammal belonging to the family Cervidae.

 This common species is native to western Eurasia, but has been introduced to Antigua & Barbuda, Argentina, South Africa, Fernando Pó, São Tomé, Madagascar,
Mauritius, Mayotte, Réunion, Seychelles, Comoro Islands, Morocco, Algeria, Tunisia, Cyprus, Israel, Cape Verde, Lebanon, Australia, New Zealand, Canada, the
United States, the Falkland Islands, and Peru.

 Some taxonomers include the rarer Persian fallow deer as a subspecies ("D. d. mesopotamica"), while others treat it as an entirely different species ("D.
mesopotamica").
</item 2>

<item 3>
Forests of Mara and Mondrem
The Forests of Mara and Mondrem were adjacent medieval forests in Cheshire, England, which in the 11th century extended to over 60 sqmi , stretching from the
Mersey in the north almost to Nantwich in the south, and from the Gowy in the west to the Weaver in the east.

 Mara and Mondrem were a hunting forest of the Norman Earls of Chester, established soon after 1071 by the first earl, Hugh d'Avranches.

 They might earlier have been an Anglo-Saxon hunting forest.

 Game included wild boar, and red, fallow and roe deer.
</item 3>

<item 4>
Indian muntjac
The Indian muntjac ("Muntiacus muntjak"), also called red muntjac and barking deer, is a common muntjac deer species in South and Southeast Asia.

 It is listed as Least Concern on the IUCN Red List.
</item 4>

<item 5>
Moose
The moose (North America) or elk (Eurasia), "Alces alces", is the largest extant species in the deer family.

 Moose are distinguished by the broad, flat (or "palmate") antlers of the males; other members of the family have antlers with a dendritic ("twig-like")
configuration.

 Moose typically inhabit boreal forests and temperate broadleaf and mixed forests of the Northern Hemisphere in temperate to subarctic climates. Hunting and
other human activities have caused a reduction in the size of the moose's range over time.

 Moose have been reintroduced to some of their former habitats.

 Currently, most moose are found in Canada, Alaska, New England, Fennoscandia, Baltic States, and Russia.

 Their diet consists of both terrestrial and aquatic vegetation.

 The most common moose predators are the gray wolf along with bears and humans. Unlike most other deer species, moose are solitary animals and do not form
herds.

 Although generally slow-moving and sedentary, moose can become aggressive and move quickly if angered or startled.

 Their mating season in the autumn features energetic fights between males competing for a female.
</item 5>

<item 6>
Lame Deer, Montana
Lame Deer (Meaveʼhoʼeno in Cheyenne) is a census-designated place (CDP) in Rosebud County, Montana.

 The community is named after Miniconjou Lakota chief Lame Deer, who was killed by the U.S. Army in 1877 under a flag of truce south of the town.

 It is the tribal and government agency headquarters of the Northern Cheyenne Indian Reservation.

 This is the location of the Chief Dull Knife College and the annual Northern Cheyenne Powwow.
</item 6>

<item 7>
White-bellied musk deer
The White-bellied musk deer or Himalayan musk deer ("Moschus leucogaster") is a musk deer species occurring in the Himalayas of Nepal, Bhutan, India, Pakistan
and China.

 It is listed as endangered by the IUCN because of overexploitation resulting in a probable serious population decline.
</item 7>

<item 8>
Red deer
The red deer ("Cervus elaphus") is one of the largest deer species.

 The red deer inhabits most of Europe, the Caucasus Mountains region, Asia Minor, Iran, parts of western Asia, and central Asia.

 It also inhabits the Atlas Mountains region between Morocco and Tunisia in northwestern Africa, being the only species of deer to inhabit Africa.

 Red deer have been introduced to other areas, including Australia, New Zealand, United States, Canada, Peru, Uruguay, Chile and Argentina.

 In many parts of the world, the meat (venison) from red deer is used as a food source.
</item 8>

<item 9>
Marsh deer
The marsh deer ("Blastocerus dichotomus") is the largest deer species from South America reaching a length of 2 m and a shoulder height of 1.2 m .

 It is found in Argentina, Bolivia, Peru, Brazil, Uruguay and Paraguay.

 Formerly found in much of tropical and subtropical South America, it ranged east of the Andes, south from the Amazon rainforest, west of the Brazilian Atlantic
rainforest and north of the Argentinian Pampa.

 Today it is largely reduced to isolated populations at marsh and lagoon zones in the Paraná, Paraguay, Araguaia and Guapore river basins.

 Small populations also occur in the southern Amazon, including Peru where protected in Bahuaja-Sonene National Park.

 It is listed as a vulnerable species by the IUCN and on CITES Appendix I.
</item 9>

<item 10>
Sri Lankan sambar deer
The Sri Lankan sambar deer ("Rusa unicolor unicolor") is a subspecies of sambar deer that lives in Sri Lanka.

 This subspecies is one of the largest sambar deer species with the largest antlers both in size and in body proportions.

 Large males weight up to 270–280 kg.

 Sri Lankan sambar live in lowland dry forests and mountain forests.

 Large herds of sambar deer roam the Horton Plains National Park, where it is the most common large mammal.
</item 10>
=== END ITEM id=9f86a281a83bef6f ===

=== ITEM id=df8c42feedc8a11c ===
Gold sentences and facts:
Sentence 0: Dorian Gray is the main character of the novel "The Picture of Dorian Gray" (1890) by Oscar Wilde.
  Fact 0.0: Dorian Gray is the main character of the novel The Picture of Dorian Gray.
  Fact 0.1: The Picture of Dorian Gray was published in 1890.
  Fact 0.2: The Picture of Dorian Gray was written by Oscar Wilde.
Sentence 1: The Picture of Dorian Gray is a philosophical novel by Oscar Wilde, first published complete in the July 1890 issue of "Lippincott's Monthly Magazine".
  Fact 1.0: The Picture of Dorian Gray is a philosophical novel.
  Fact 1.1: The Picture of Dorian Gray was written by Oscar Wilde.
  Fact 1.2: The Picture of Dorian Gray was first published complete.
  Fact 1.3: The Picture of Dorian Gray was first published complete in the July 1890 issue of Lippincott's Monthly Magazine.
Sentence 2:  The magazine's editor feared the story was indecent, and without Wilde's knowledge, deleted roughly five hundred words before publication.
  Fact 2.0: The editor of Lippincott's Monthly Magazine feared the story was indecent.
  Fact 2.1: The editor of Lippincott's Monthly Magazine, without Wilde's knowledge, deleted roughly five hundred words before publication.

Retrieved items:
<item 1>
The Picture of Dorian Gray (1945 film)
The Picture of Dorian Gray is a 1945 American horror-drama film based on Oscar Wilde's 1890 novel of the same name.

 Released in March 1945 by Metro-Goldwyn-Mayer, the film is directed by Albert Lewin and stars George Sanders as Lord Henry Wotton and Hurd Hatfield as Dorian
Gray.

 Shot primarily in black-and-white, the film features four inserts in 3-strip Technicolor of Dorian's portrait as a special effect (the first two of his
portrait's original state, and the second two after a major period of degeneration).
</item 1>

<item 2>
Dorian Gray (1970 film)
Dorian Gray (Italian: "Il dio chiamato Dorian") aka "The Sins of Dorian Gray" is a 1970 movie adaptation of Oscar Wilde's novel "The Picture of Dorian Gray"
starring Helmut Berger.

 The Italian title translates as "A God Called Dorian".
</item 2>

<item 3>
A Portrait of Dorian Gray
A Portrait of Dorian Gray (2005) is the fashion designer and photographer Karl Lagerfeld's rendition of Oscar Wilde's novel "The Picture of Dorian Gray" (1891)
in photography.

 The models, Larry Scott and Eva Herzigová, star as Mr. and Mrs. Dorian Gray.
</item 3>

<item 4>
The Picture of Dorian Gray (1917 film)
The Picture of Dorian Gray (German: Das Bildnis des Dorian Gray) is a 1917 German silent fantasy film directed by Richard Oswald and starring Bernd Aldor, Ernst
Pittschau, and Ernst Ludwig.

 The film is based on the novel "The Picture of Dorian Gray" by Oscar Wilde.
</item 4>

<item 5>
The Picture of Dorian Gray (1910 film)
The Picture of Dorian Gray (1910), a.k.a. Dorian Grays Portræt, is a Danish silent film based on the novel "The Picture of Dorian Gray" by Oscar Wilde.
</item 5>

<item 6>
The Picture of Dorian Gray
The Picture of Dorian Gray is a philosophical novel by Oscar Wilde, first published complete in the July 1890 issue of "Lippincott's Monthly Magazine".

 The magazine's editor feared the story was indecent, and without Wilde's knowledge, deleted roughly five hundred words before publication.

 Despite that censorship, "The Picture of Dorian Gray" offended the moral sensibilities of British book reviewers, some of whom said that Oscar Wilde merited
prosecution for violating the laws guarding the public morality.

 In response, Wilde aggressively defended his novel and art in correspondence with the British press, although he personally made excisions of some of the most
controversial material when revising and lengthening the story for book publication the following year.
</item 6>

<item 7>
Dorian Gray (disambiguation)
Dorian Gray is the main character of the novel "The Picture of Dorian Gray" (1890) by Oscar Wilde.
</item 7>

<item 8>
Dorian Gray (character)
Dorian Gray is the main character in Oscar Wilde's novel The Picture of Dorian Gray.

 Dorian Gray is a handsome, susceptible, wealthy young Victorian gentleman.

 He falls in love with and devotes himself to Sibyl Vane.

 Dorian has two friends, Basil and Henry.

 Basil is an artist who draws a portrait of Dorian that emphasizes his youthful, noble, and natural beauty.

 Henry is a friend who has a cynical nature and who shares various thoughts with Dorian.
</item 8>

<item 9>
El retrato de Dorian Gray
The Picture of Dorian Gray ("El retrato de Dorian Gray"), is a Mexican telenovela, based on "The Picture of Dorian Gray", the novel written by Oscar Wilde.

 The main character is the handsome young man called Dorian Gray (played by late Enrique Álvarez Félix).

 This telenovela was made in 1969.
</item 9>

<item 10>
Graph query row:
work: The Picture of Dorian Gray
kind: philosophical novel
published_on: July 1890
relationship: CREATED_BY
connected_entity: Oscar Wilde
</item 10>
=== END ITEM id=df8c42feedc8a11c ===

=== ITEM id=6daaee9e1ab20bb4 ===
Gold sentences and facts:
Sentence 0: The Night of the Arts (Finnish: "Taiteiden yö" , Swedish: "Konstens natt" ) is an annual event held in several major cities in Finland, in late August.
  Fact 0.0: The Night of the Arts is an annual event.
  Fact 0.1: The Night of the Arts is known in Finnish as 'Taiteiden yö'.
  Fact 0.2: The Night of the Arts is known in Swedish as 'Konstens natt'.
  Fact 0.3: The Night of the Arts is held in several major cities in Finland.
  Fact 0.4: The Night of the Arts is held in late August.
Sentence 1:  It is one of many White Night festivals held worldwide.
  Fact 1.0: The Night of the Arts is one of many White Night festivals.
  Fact 1.1: White Night festivals are held worldwide.
Sentence 2: The White Nights are a kind of all-night arts festival held in many cities in the summer.
  Fact 2.0: The White Nights are a kind of all-night arts festival.
  Fact 2.1: The White Nights festival is held in many cities.
  Fact 2.2: The White Nights festival is held in the summer.
Sentence 3:  The "white nights" is the name given in areas of high latitude to the weeks around the summer solstice in June during which sunsets are late, sunrises are early and darkness is never complete.
  Fact 3.0: 'White nights' is the name given in areas of high latitude to the weeks around the summer solstice in June.
  Fact 3.1: During white nights, sunsets are late.
  Fact 3.2: During white nights, sunrises are early.
  Fact 3.3: During white nights, darkness is never complete.

Retrieved items:
<item 1>
Night of the Arts
The Night of the Arts (Finnish: "Taiteiden yö" , Swedish: "Konstens natt" ) is an annual event held in several major cities in Finland, in late August.

 It is one of many White Night festivals held worldwide.
</item 1>

<item 2>
Green Man Festival
Green Man is an independent music and arts festival held annually in mid-August in the Brecon Beacons, Wales since 2003.

 It has evolved into a 20,000 capacity 4-day event, showcasing predominantly live music (in particular alternative, indie, rock, folk, dance and americana),
with additional events showcasing literature, film, comedy, theatre and poetry.

 2015 saw 1,500 multi-arts acts perform across 17 stages.

 The festival site is divided into 10 areas, each offering a unique festival experience.

 Ceilidhs, all-night bonfires and secret gigs all add to the festival's unique identity.
</item 2>

<item 3>
Tanglewood Music and Arts Festival
Tanglewood Music and Arts Festival is an annual two-day music, comedy and arts event held within the grounds of Narrow Water Castle, County Down, Northern
Ireland.

 The festival, founded in 2011, has attracted significant attention both within the local area and around Northern Ireland with buses coming from as far as
Belfast to the event.

 For the 2011 and 2012 festivals Tanglewood was held on one day; however for 2013 it has expanded into a two-day camping festival.
</item 3>

<item 4>
Kerala School Kalolsavam
School Arts Festival of Kerala (Malayalam: കേരള സ്കൂൾ കലോത്സവം) is an annual event conducted by the state government of Kerala, featuring several art
competitions for high school and higher secondary school students of Kerala.

 The festival was started in 1956, and till 2008, it was called as "Kerala State School Youth Festival".

 The participants are students from classes 8th to 12th.

 Winners from different revenue districts for a particular event will be competing in state level competition.

 The event is usually conducted in December–January months of a year and is considered to be the biggest cultural event of Asia.

 57th edition of Kerala School Kalolsavam 2017 is scheduled for January 16 to 22 at Kannur.
</item 4>

<item 5>
Flip Animation Festival
FLIP is an animation festival primarily hosted by the Light House Media Centre in Wolverhampton, UK.

 It is one of two festivals hosted by Light House, the other of which is Deaffest.

 Official literature for the festival says that FLIP occurs annually at the beginning of November and attracts submissions from more than 30 countries world
wide.

 As well as screening the selected open submissions, FLIP also consists of special screenings, talks from professionals within the animation world, workshops,
industry panels, portfolio reviews and competitions.

 The festival was set up, managed and programmed by Peter McLuskie between 2004 and 2011.

 It grew out of the 'Animation Forum', also based at Light House and which was later rebranded as Animation Forum West Midlands and found a home at Birmingham
City University.

 In 2009 the festival was awarded a Black Country Tourism Award for Event of the Year.
</item 5>

<item 6>
White Night festivals
The White Nights are a kind of all-night arts festival held in many cities in the summer.

 The original festival is the White Nights Festival held in Saint Petersburg, Russia.

 The "white nights" is the name given in areas of high latitude to the weeks around the summer solstice in June during which sunsets are late, sunrises are
early and darkness is never complete.

 In Saint Petersburg, the Sun does not set until after 10 p.m., and the twilight lasts almost all night.
</item 6>

<item 7>
Joulupöytä
Joulupöytä (translated "Yule table") is the traditional assortment of foods served at Christmas in Finland, similar to the Swedish julbord.

 It contains many different dishes, most of them typical for the season.

 The main dish is usually a large Christmas ham, which is eaten with mustard or bread along with the other dishes.

 Fish is also served (often lutefisk and gravlax), and the ham is served with "laatikko"s, casseroles made with swede, potato and carrot, occasionally liver.

 The traditional Christmas beverage is either alcoholic or non-alcoholic mulled wine ("glögi" in Finnish).
</item 7>

<item 8>
Great Northern Arts Festival
The Great Northern Arts Festival is held each year in Inuvik, Northwest Territories.

 Lasting approximately ten days, it is an annual event held each summer.

 It was established in 1989 by Sharlene Alexander and Sue Rose.

 While the festival's mandate is to serve Northern artists and craftspeople, artists from around the world also attend.

 The festival is designed as a venue for artists to see each other's work and to experiment with new applications.

 It includes arts of all types, such as fiber art and Music of the Northwest Territories.

 Artists partake in demonstrations, masterclasses, seminars, and workshops.

 The festival's gallery offers pieces for sale.

 There are cultural presentations each evening which include concerts, dance, fashion, music, and storytelling.

 Media coverage brings festival highlights to viewing audiences.
</item 8>

<item 9>
Satumaa
"Satumaa" (roughly, in English "The Fabled Land" or "The Fairytale Land") is the quintessential Finnish tango.

 It was written by Unto Mononen, and published in 1955.

 The most famous recording is probably the one made by Reijo Taipale in 1962.

 The lyrics tell a story of a distant land beyond the sea — a happy paradise — however, the narrator can only reach it in his thoughts.

 The song has been recorded countless times, mainly by male Finnish tango singers.
</item 9>

<item 10>
Artscape (festival)
Artscape is an annual art festival held in the Mount Royal neighborhood of Baltimore, Maryland in July.

 Since its first annual event in 1982, it has become the largest free arts festival in America.

 It has boasted acts such as Ray Charles, Aretha Franklin, and Matisyahu in the past, attracting over 350,000 people from the city, and surrounding areas.

 Film programming during Artscape is provided by Maryland Film Festival.

 There are artists in a variety of visual and performing media.

 Events are free and open to the public.
</item 10>
=== END ITEM id=6daaee9e1ab20bb4 ===

=== ITEM id=c58565d56d9bfbee ===
Gold sentences and facts:
Sentence 0: Essence is a monthly magazine for African American women between the ages of 18 and 49.
  Fact 0.0: Essence is a monthly magazine.
  Fact 0.1: Essence is for African American women.
  Fact 0.2: Essence is aimed at women between the ages of 18 and 49.
Sentence 1: ALT for Damerne (meaning "All for the Ladies" in English) is a Danish language weekly women's magazine published in Copenhagen, Denmark.
  Fact 1.0: ALT for Damerne is a women's magazine.
  Fact 1.1: ALT for Damerne means 'All for the Ladies' in English.
  Fact 1.2: ALT for Damerne is written in the Danish language.
  Fact 1.3: ALT for Damerne is a weekly magazine.
  Fact 1.4: ALT for Damerne is published in Copenhagen, Denmark.

Retrieved items:
<item 1>
Hänt Extra
Hänt Extra is a Swedish language weekly celebrity and youth magazine published by Aller Media in Stockholm, Sweden.
</item 1>

<item 2>
Elle (magazine)
Elle is a worldwide lifestyle magazine of French origin that focuses on fashion, beauty, health, and entertainment.

 "Elle" is also the world's best-selling fashion magazine.

 It was founded by Pierre Lazareff and his wife Hélène Gordon in 1945.

 The title, in French, means "she" or "her".
</item 2>

<item 3>
Esquire (magazine)
Esquire is an American men's magazine, published by the Hearst Corporation in the United States.

 Founded in 1933, it flourished during the Great Depression under the guidance of founders Arnold Gingrich, David A. Smart and Henry L. Jackson.
</item 3>

<item 4>
Exclaim!
Exclaim!

 is a monthly Canadian music magazine that features in-depth coverage of new music across all genres with a special focus on Canadian and cutting-edge artists.

 Content is based on the monthly print publication, which publishes 9 issues per year, distributing over 103,000 copies to over 2,600 locations across Canada.

 The magazine has an average of 361,200 monthly readers.

 Their website, exclaim.ca, has an average of 675,000 unique visitors a month.
</item 4>

<item 5>
Essence Music Festival
The Essence Festival, known as "the party with a purpose", is an annual music festival which started in 1995 as a one-time event to celebrate the 25th
anniversary of "Essence", a magazine aimed primarily towards African-American women.

 It is the largest event celebrating African-American culture and music in the United States.

 Locally referred to as the Essence Fest, it has been held in New Orleans, Louisiana every year since 1994 except for 2006, when it was held in Houston, Texas
due to Hurricane Katrina's effect on New Orleans.

 It also held in Durban, KwaZulu-Natal in 2016.

 It features artists simultaneously performing on a main stage as well as four standing-room only superlounge stages.
</item 5>

<item 6>
State (magazine)
State.ie (formerly State Magazine) is an Irish website and formerly a monthly music magazine, which launched in April 2008 and ceased to print in January 2009
having published a total of ten issues.

 The magazine continues online and was voted Best Music Site in 2008 and Best Web Publication in 2010 in the Irish Web Awards.

 Originally the concept of the magazine involved a hard copy of which there was a charge to purchase, however after six issues it was decided to make the
magazine's content free both online and in print.

 The first issue, April 2008, appeared on Irish shelves on 6 March 2008 and featured Michael Stipe of R.E.M. on the cover.

 This immediately garnered comparisons between the new magazine's similarities with "Hot Press" who featured Stipe on their cover at the same time, a move
widely thought to be an attempt by Hot Press to stifle State's status as a serious 'alternative' to the more established local magazine.

 At a price of €5.50, "State" charged €2 more than "Hot Press".
</item 6>

<item 7>
Diva (magazine)
DIVA is a leading lesbian magazine in the United Kingdom, published monthly.

 It was launched in March 1994 by Millivres Ltd, under the editorship of Frances Williams and is now published by Twin Media Group, which is owned by Silke
Bader and Linda Riley.

 Silke Bader is also the publisher of Curve magazine in the USA and LOTL in Australia, and Linda Riley runs the European Diversity Awards and is the former
publisher of g3 magazine in the UK.
</item 7>

<item 8>
Essence (magazine)
Essence is a monthly magazine for African American women between the ages of 18 and 49.

 It is the only magazine that focuses on reaching an audience of black women, revolves around the black woman experience, and has remained for a long period of
time.

 The magazine covers fashion, lifestyle and beauty, with an intimate girlfriend-to-girlfriend tone, and its slogan "Fierce, Fun, and Fabulous" suggests the
magazine's goal of empowering African-American women.

 The topics the magazine discusses range from celebrities, to fashion, to point-of-view pieces addressing current issues in the African-American community.

 A number of its readers engage closely and personally with the publication, and it claims to be the magazine "for and about Black women".
</item 8>

<item 9>
SexIs
SexIs Magazine is a quarterly print publication and daily webzine devoted to sex and sexual culture, founded in 2008 by Web Merchants, parent company of sex toy
e-tailer EdenFantasys.com.

 The first print issue debuted in November 2009, distributed nationwide as an insert in "BUST magazine".

 The website publishes articles, columns, video presentations and news items daily.
</item 9>

<item 10>
Alt for Damerne
ALT for Damerne (meaning "All for the Ladies" in English) is a Danish language weekly women's magazine published in Copenhagen, Denmark.
</item 10>
=== END ITEM id=c58565d56d9bfbee ===

=== ITEM id=924df357d2867c79 ===
Gold sentences and facts:
Sentence 0: Corn Ranch is a spaceport in the West Texas town of Van Horn, Texas, where flight tests of the New Shepard are carried out by Blue Origin.
  Fact 0.0: Corn Ranch is a spaceport.
  Fact 0.1: Corn Ranch is located in the West Texas town of Van Horn, Texas.
  Fact 0.2: Flight tests of the New Shepard are carried out at Corn Ranch.
  Fact 0.3: Flight tests of the New Shepard are carried out by Blue Origin.
Sentence 1: Blue Origin is an American privately funded aerospace manufacturer and spaceflight services company set up by Amazon.com founder Jeff Bezos with its headquarters in Kent, Washington.
  Fact 1.0: Blue Origin is an American privately funded aerospace manufacturer.
  Fact 1.1: Blue Origin is a spaceflight services company.
  Fact 1.2: Blue Origin was set up by Amazon.com founder Jeff Bezos.
  Fact 1.3: Blue Origin has its headquarters in Kent, Washington.

Retrieved items:
<item 1>
Corn Ranch
Corn Ranch is a spaceport in the West Texas town of Van Horn, Texas, where flight tests of the New Shepard are carried out by Blue Origin.

 The 165,000-acre (670 km²) land parcel was purchased by Internet billionaire Jeff Bezos.

 The first flight test took place on November 13, 2006 with the goal of providing commercial tourist flights.
</item 1>

<item 2>
Texas Air &amp; Space Museum
The Texas Air & Space Museum is an aviation museum located near Rick Husband Amarillo International Airport in Amarillo, Texas.

 The museum displays civilian and military aircraft as well as a wide range of air and space artifacts.
</item 2>

<item 3>
H-T Ranch
H-T Ranch is a historic ranch complex 10 mi west of Amidon, North Dakota.

 The ranch originally consisted of ten buildings; however, only the ranch house and log barn survive.

 The ranch served as the headquarters of the Little Missouri Cattle Company, which was run by Arthur Clark Hidekoper.

 Hidekober established the ranch in the 1880s, and by the end of the decade, it had become the "most notable" ranch in the area.

 The ranch house, called Shackford, was built in 1896; its uncommon style resembles a bungalow but also borrows from other styles such as the Stick style.

 By 1906, the "Fargo Forum" described the ranch as "the biggest and most important [ranch] in the state" and "one of the largest horse raising outlets in the
world".

 In the same year, Hidekoper sold the 70,000 acre ranch; the sale was the largest land deal in North Dakota history.

 After the sale, a land company reduced the ranch to 5000 acre ; it was later used as a dude ranch in the 1920s.
</item 3>

<item 4>
SyberJet Aircraft
SyberJet Aircraft is the manufacturer of the SJ30 light business jet.

 The company's headquarters is in Cedar City, Utah adjacent to the Cedar City Regional Airport with additional engineering offices and manufacturing, service,
repair and fatigue test facilities near and on the San Antonio International Airport in San Antonio, Texas.
</item 4>

<item 5>
Northern California TRACON
Northern California TRACON (NCT) (Terminal Radar Approach Control), or NorCal TRACON for short, is an air traffic control facility that provides safety alerts,
separation, and sequencing of air traffic arriving, departing, and transiting the airspace and airports in Northern California.

 Located in Rancho Cordova near Sacramento, NCT controls airspace over 19000 square miles, and serves Reno International Airport, Sacramento International
Airport, San Jose International Airport, Oakland International Airport, and San Francisco International Airport, plus 19 other smaller airports with air traffic
control towers.

 NCT is the 3rd busiest TRACON in America.

 NorCal TRACON is the step between local control (in an airport's control tower) and Air Route Traffic Control Center (ARTCC), in this case, Oakland Center
(ICAO code: ZOA).

 San Francisco International Airport is the 2nd largest airport in California and the largest airport serving Northern California.
</item 5>

<item 6>
Graph query row:
organization: Blue Origin
headquarters: Kent, Washington
</item 6>

<item 7>
Blue Origin
Blue Origin is an American privately funded aerospace manufacturer and spaceflight services company set up by Amazon.com founder Jeff Bezos with its
headquarters in Kent, Washington.

 The company is developing technologies to enable private human access to space with the goal to dramatically lower costs and increase reliability.

 Blue Origin is employing an incremental approach from suborbital to orbital flight, with each developmental step building on its prior work.

 The company motto is ""Gradatim Ferociter"", Latin for "Step by Step, Ferociously".

 Blue Origin is developing a variety of technologies, with a focus on rocket-powered Vertical Takeoff and Vertical Landing (VTVL) vehicles for access to
suborbital and orbital space.

 The company's name refers to the blue planet, Earth, as the point of origin.
</item 7>

<item 8>
Grand Sky
Grand Sky is an UAS (unmanned aerial systems)-specific business and aviation park located at Grand Forks Air Force Base near Grand Forks, North Dakota.

 A groundbreaking for the development was held in September 2015, making it the first UAS business and aviation park in the U.S.

 The park consists of 217 acres and includes access to the Grand Forks Air Force Base runway for large UAS flights.
</item 8>

<item 9>
Rick Husband Amarillo International Airport
Rick Husband Amarillo International Airport (IATA: AMA, ICAO: KAMA, FAA LID: AMA) is a public airport six miles (10 km) east of downtown Amarillo, in Potter and
Randall Counties, Texas, United States.

 The airport was renamed in 2003 after NASA astronaut and Amarillo native Rick Husband, who died in the Space Shuttle "Columbia" disaster in February of that
year.
</item 9>

<item 10>
M7 Aerospace
M7 Aerospace LP is an aerospace company with its headquarters on the property of San Antonio International Airport in Uptown San Antonio, Texas, United States.
</item 10>
=== END ITEM id=924df357d2867c79 ===

=== ITEM id=a1b8872f34fe9433 ===
Gold sentences and facts:
Sentence 0:  He is the main vocalist of the boy group Highlight.
  Fact 0.0: The person described is the main vocalist of the boy group Highlight.
Sentence 1:  She had been a trainee at S.M. Entertainment's Starlight Academy during her middle school years before debuting as a member of the agency's girl group, Girls' Generation, in 2007.
  Fact 1.0: The person described was a trainee at S.M. Entertainment's Starlight Academy during her middle school years.
  Fact 1.1: The person described debuted as a member of Girls' Generation, the girl group of S.M. Entertainment, in 2007.
  Fact 1.2: The person described debuted in Girls' Generation after her trainee years.

Retrieved items:
<item 1>
Around Us Entertainment
Around Us Entertainment is a South Korean independent entertainment company established by Yoon Doo-joon, Yang Yo-seob, Yong Jun-hyung, Lee Gi-kwang, Son
Dong-woon from boy group Highlight, formerly known as BEAST, after leaving their former label Cube Entertainment.
</item 1>

<item 2>
Park So-yeon (singer)
Park So-yeon (born October 5, 1987), better known mononymously as Soyeon, is a South Korean singer and actress.

 She is best known as the former main vocalist and member of South Korean girl group T-ara.
</item 2>

<item 3>
Kim Tae-yeon discography
Apart from soundtrack recordings, Taeyeon has also recorded duets with other artists, most notably "Like a Star" with The One and "Different" with Kim Bum-soo,
which peaked at numbers one and two on the Gaon Digital Chart, respectively.
</item 3>

<item 4>
Kim Tae-yeon
Kim Tae-yeon (born March 9, 1989), referred to as Taeyeon, is a South Korean singer.

 She had been a trainee at S.M. Entertainment's Starlight Academy during her middle school years before debuting as a member of the agency's girl group, Girls'
Generation, in 2007.

 Since then, she has risen to prominence due to the group's success on the Asian music scene and further participated in the agency's projects Girls'
Generation-TTS and SM the Ballad.

 Aside from group activities, she has also recorded songs for various television dramas and movies.
</item 4>

<item 5>
Yang Yo-seob
Yang Yo-seob (born January 5, 1990), more commonly known as Yoseob, is a South Korean singer and musical actor.

 He is the main vocalist of the boy group Highlight.
</item 5>

<item 6>
Highlight (band)
Highlight (Korean: 하이라이트 ) is a South Korean boy band formerly known as Beast (Korean: 비스트).

 The band consists of five members: Yoon Doo-joon, Yong Jun-hyung, Yang Yo-seob, Lee Gi-kwang, and Son Dong-woon.

 Original member Jang Hyun-seung officially left the group in April 2016.

 Later that year, the five remaining members moved labels from Cube Entertainment to Around Us Entertainment and subsequently changed their name to Highlight in
2017.
</item 6>

<item 7>
Kim Tae-yeon discography
Kim Tae-yeon, better known by the mononym Taeyeon, is a South Korean singer.

 Her discography consists of one studio album, two extended plays (EPs), twenty singles (including four as featured artist), and three promotional singles.

 She debuted as a member of South Korean girl group Girls' Generation in August 2007 and initially gained some popularity as a singer upon recording soundtrack
songs "If" for "Hong Gil Dong" and "Can You Hear Me" for "Beethoven Virus" (2008).

 She subsequently established herself as one of the most renowned vocalist on the South Korean music scene with further soundtrack recordings, notably "I Love
You" for "" (2010), "Missing You like Crazy" for "The King 2 Hearts", "Closer" for "To the Beautiful You" (2012), and "And One" for "That Winter, the Wind
Blows" (2013), all of which managed to enter the top ten of South Korea's Gaon Digital Chart.
</item 7>

<item 8>
The First Collage
The First Collage is the debut and first EP by Yang Yo-seob, released on November 26, 2012.
</item 8>

<item 9>
Yang Yo-seob discography
The discography of South Korean singer-songwriter, musical actor Yang Yo-seob consists of one extended play and two singles.
</item 9>

<item 10>
List of songs written by Super Junior
Super Junior (Korean: 슈퍼주니어 ; "Syupeo Junieo") is a South Korean boy band.

 Formed in 2005 by producer Lee Soo-man of S.M. Entertainment, the group comprised a total of thirteen members at its peak.

 Super Junior originally debuted with twelve members, consisting of leader Leeteuk, Heechul, Hangeng, Yesung, Kangin, Shindong, Sungmin, Eunhyuk, Siwon,
Donghae, Ryeowook and Kibum.

 Kyuhyun joined the group in 2006.
</item 10>
=== END ITEM id=a1b8872f34fe9433 ===

=== ITEM id=40d56e4270196bb1 ===
Gold sentences and facts:
Sentence 0: The Night of the Arts (Finnish: "Taiteiden yö" , Swedish: "Konstens natt" ) is an annual event held in several major cities in Finland, in late August.
  Fact 0.0: The Night of the Arts is an annual event.
  Fact 0.1: The Night of the Arts is known in Finnish as 'Taiteiden yö'.
  Fact 0.2: The Night of the Arts is known in Swedish as 'Konstens natt'.
  Fact 0.3: The Night of the Arts is held in several major cities in Finland.
  Fact 0.4: The Night of the Arts is held in late August.
Sentence 1:  It is one of many White Night festivals held worldwide.
  Fact 1.0: The Night of the Arts is one of many White Night festivals.
  Fact 1.1: White Night festivals are held worldwide.
Sentence 2: The White Nights are a kind of all-night arts festival held in many cities in the summer.
  Fact 2.0: The White Nights are a kind of all-night arts festival.
  Fact 2.1: The White Nights festival is held in many cities.
  Fact 2.2: The White Nights festival is held in the summer.
Sentence 3:  The "white nights" is the name given in areas of high latitude to the weeks around the summer solstice in June during which sunsets are late, sunrises are early and darkness is never complete.
  Fact 3.0: 'White nights' is the name given in areas of high latitude to the weeks around the summer solstice in June.
  Fact 3.1: During white nights, sunsets are late.
  Fact 3.2: During white nights, sunrises are early.
  Fact 3.3: During white nights, darkness is never complete.

Retrieved items:
<item 1>
White Night festivals
The White Nights are a kind of all-night arts festival held in many cities in the summer.

 The original festival is the White Nights Festival held in Saint Petersburg, Russia.

 The "white nights" is the name given in areas of high latitude to the weeks around the summer solstice in June during which sunsets are late, sunrises are
early and darkness is never complete.

 In Saint Petersburg, the Sun does not set until after 10 p.m., and the twilight lasts almost all night.
</item 1>

<item 2>
Joulupöytä
Joulupöytä (translated "Yule table") is the traditional assortment of foods served at Christmas in Finland, similar to the Swedish julbord.

 It contains many different dishes, most of them typical for the season.

 The main dish is usually a large Christmas ham, which is eaten with mustard or bread along with the other dishes.

 Fish is also served (often lutefisk and gravlax), and the ham is served with "laatikko"s, casseroles made with swede, potato and carrot, occasionally liver.

 The traditional Christmas beverage is either alcoholic or non-alcoholic mulled wine ("glögi" in Finnish).
</item 2>

<item 3>
Flip Animation Festival
FLIP is an animation festival primarily hosted by the Light House Media Centre in Wolverhampton, UK.

 It is one of two festivals hosted by Light House, the other of which is Deaffest.

 Official literature for the festival says that FLIP occurs annually at the beginning of November and attracts submissions from more than 30 countries world
wide.

 As well as screening the selected open submissions, FLIP also consists of special screenings, talks from professionals within the animation world, workshops,
industry panels, portfolio reviews and competitions.

 The festival was set up, managed and programmed by Peter McLuskie between 2004 and 2011.

 It grew out of the 'Animation Forum', also based at Light House and which was later rebranded as Animation Forum West Midlands and found a home at Birmingham
City University.

 In 2009 the festival was awarded a Black Country Tourism Award for Event of the Year.
</item 3>

<item 4>
Heikki Savolainen (gymnast)
Heikki Ilmari Savolainen (28 September 1907 – 29 November 1997) was a Finnish artistic gymnast.

 He competed in five consecutive Olympics from 1928 to 1952 and won at least one medal in each of them.

 In 1928 he won a bronze on pommel horse, which was the first-ever medal in gymnastics for Finland.

 Winning his last medal at the 1952 Summer Olympics in Helsinki, he became the oldest gymnastics medalist, at 44 years old; he delivered the Olympic Oath in the
opening ceremony of those games.

 In 1932 Savolainen and his teammate Einari Teräsvirta had the same score on horizontal bar, but the Finnish team voted to give the silver medal to Savolainen.

 In 1948 he again had the same score as teammates Veikko Huhtanen and Paavo Aaltonen on pommel horse, and the gold medal was shared between the three.
</item 4>

<item 5>
Women's Nordic Football Championship
Women's Nordic Football Championship was an international football competition contested by the women's national football teams of the Nordic countries.

 The tournament was held annually between 1974 and 1982.

 Finland, Denmark and Sweden competed from the start, Norway joined the tournament in 1978.

 Iceland and Faroe Islands did not take part at the competition.
</item 5>

<item 6>
Night of the Arts
The Night of the Arts (Finnish: "Taiteiden yö" , Swedish: "Konstens natt" ) is an annual event held in several major cities in Finland, in late August.

 It is one of many White Night festivals held worldwide.
</item 6>

<item 7>
Green Man Festival
Green Man is an independent music and arts festival held annually in mid-August in the Brecon Beacons, Wales since 2003.

 It has evolved into a 20,000 capacity 4-day event, showcasing predominantly live music (in particular alternative, indie, rock, folk, dance and americana),
with additional events showcasing literature, film, comedy, theatre and poetry.

 2015 saw 1,500 multi-arts acts perform across 17 stages.

 The festival site is divided into 10 areas, each offering a unique festival experience.

 Ceilidhs, all-night bonfires and secret gigs all add to the festival's unique identity.
</item 7>

<item 8>
Scandinavian folklore
Scandinavian folklore or Nordic folklore is the folklore of Norway, Sweden, Finland, Denmark, Iceland and the Faroe Islands.
</item 8>

<item 9>
Cardiant
Cardiant is a Finnish heavy metal band.

 It was established in city of Hämeenlinna in 2000, but most of band members have been replaced and the current assembly has been working since 2008.

 In the early years, the band recorded several demos until the it signed a contract with the record company Underground Symphony.

 The first album, Midday Moon, was released in 2005, but the record company did not released it in Europe.

 However, other record company, Marquee/Avalon, released it in Japan.
</item 9>

<item 10>
Great Northern Arts Festival
The Great Northern Arts Festival is held each year in Inuvik, Northwest Territories.

 Lasting approximately ten days, it is an annual event held each summer.

 It was established in 1989 by Sharlene Alexander and Sue Rose.

 While the festival's mandate is to serve Northern artists and craftspeople, artists from around the world also attend.

 The festival is designed as a venue for artists to see each other's work and to experiment with new applications.

 It includes arts of all types, such as fiber art and Music of the Northwest Territories.

 Artists partake in demonstrations, masterclasses, seminars, and workshops.

 The festival's gallery offers pieces for sale.

 There are cultural presentations each evening which include concerts, dance, fashion, music, and storytelling.

 Media coverage brings festival highlights to viewing audiences.
</item 10>
=== END ITEM id=40d56e4270196bb1 ===

=== ITEM id=df0ece99d98ed4d1 ===
Gold sentences and facts:
Sentence 0: Mirrors is a 2008 supernatural horror film directed by Alexandre Aja, starring Kiefer Sutherland, Paula Patton, and Amy Smart.
  Fact 0.0: Mirrors is a 2008 supernatural horror film.
  Fact 0.1: Mirrors was directed by Alexandre Aja.
  Fact 0.2: Kiefer Sutherland starred in Mirrors.
  Fact 0.3: Paula Patton starred in Mirrors.
  Fact 0.4: Amy Smart starred in Mirrors.
Sentence 1: Paula Maxine Patton (born December 5, 1975) is an American actress.
  Fact 1.0: Paula Maxine Patton was born on December 5, 1975.
  Fact 1.1: Paula Maxine Patton is an American actress.

Retrieved items:
<item 1>
Mirrors (film)
Mirrors is a 2008 supernatural horror film directed by Alexandre Aja, starring Kiefer Sutherland, Paula Patton, and Amy Smart.

 The film was first titled "Into the Mirror", but the name was later changed to "Mirrors".

 Filming began on May 1, 2007, and it was released in American theaters on August 15, 2008.
</item 1>

<item 2>
Illusions &amp; Mirrors
Illusions & Mirrors is a 2013 short film directed by Iranian-born American artist and filmmaker Shirin Neshat.

 The film, shot in black-and-white, stars Tarek Aylouch, Michael Markiewicz, and the Israeli-born American actress Natalie Portman.

 The film was commissioned by Dior, for which Portman is a spokesmodel, for the Miss Dior Expo at the Grand Palais, Paris.

 Shirin Neshat said that the film is a tribute to the black-and-white silent films made by such surrealist filmmakers as Man Ray, Jean Cocteau, Luis Buñuel, and
Maya Deren.
</item 2>

<item 3>
Zooey Deschanel filmography
American actress Zooey Deschanel made her film debut in the 1999 comedy feature "Mumford".

 She went on to gained public attention by co-starred in the comedy-drama "Almost Famous" (2000), the independent drama "Manic" (2001), opposite Joseph
Gordon-Levitt, and the comedy-drama "The Good Girl" (2002).

 She landed her first major role as a 18-year-old virgin in the romantic drama "All the Real Girls" (2003), for which she was nominated for an Independent
Spirit Award for Best Female Lead.

 Deschanel's biggest commercial success, as of 2017, came with the Christmas-fantasy film "Elf" (also in 2003), which grossed over $220 million worldwide.
</item 3>

<item 4>
License to Wed
License to Wed is a 2007 American romantic comedy film starring Robin Williams, Mandy Moore and John Krasinski, and directed by Ken Kwapis.

 The film was released in theaters on July 3, 2007.
</item 4>

<item 5>
These Old Broads
These Old Broads is a 2001 television film written by Carrie Fisher and starring her mother Debbie Reynolds, as well as Shirley MacLaine, Joan Collins, and
Elizabeth Taylor in her final film role.

 In a 2001 BBC "Omnibus" documentary about Elizabeth Taylor, Shirley MacLaine says that Julie Andrews and Lauren Bacall were originally planned to be in the
movie.

 The role of Miriam Hodges was originally offered to June Allyson.
</item 5>

<item 6>
Jack &amp; Diane (film)
Jack & Diane is a 2012 American romantic horror film written and directed by Bradley Rust Gray, and starring Riley Keough and Juno Temple.

 Olivia Thirlby and Ellen Page were originally cast for the lead roles in 2008 before the project was postponed.
</item 6>

<item 7>
Mira Nair
Mira Nair (born 15 October 1957) is an Indian American filmmaker based in New York City.

 Her production company, Mirabai Films, specializes in films for international audiences on Indian society, whether in the economic, social or cultural spheres.

 Among her best known films are "Mississippi Masala", "The Namesake", the Golden Lion-winning "Monsoon Wedding" and "Salaam Bombay!

", which was nominated for the Academy Award for Best Foreign Language Film.
</item 7>

<item 8>
Paula Patton
Paula Maxine Patton (born December 5, 1975) is an American actress.

 Patton made her film debut in the 2005 comedy film "Hitch" and starred in the epic fantasy film "Warcraft" (2016), based on the game series of the same name.

 Patton has also been the female lead in "Déjà Vu" (2006), "" (2011) and "2 Guns" (2013), and appeared in the critically acclaimed "Precious" (2009).
</item 8>

<item 9>
Hitch (film)
Hitch is a 2005 American romantic comedy film directed by Andy Tennant and starring Will Smith.

 The film, which was written by Kevin Bisch, co-stars Eva Mendes, Kevin James, and Amber Valletta.

 Smith plays the main fictional character of the film, Alex "Hitch" Hitchens, who is a professional dating consultant who makes a living teaching men how to woo
women.

 The film was released on February 11, 2005 by Columbia Pictures.
</item 9>

<item 10>
Heather Burns
Heather Burns (born April 7, 1975) is an American actress, known for her role as Miss Rhode Island in the 2000 film "Miss Congeniality".

 Her other film appearances include "You've Got Mail" (1998), "Two Weeks Notice" (2002) and "Bewitched" (2005).
</item 10>
=== END ITEM id=df0ece99d98ed4d1 ===
